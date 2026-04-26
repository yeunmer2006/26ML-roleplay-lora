#!/usr/bin/env python3
"""
LoRA 训练脚本 - 角色扮演对话生成
基于 Qwen2.5-3B-Instruct 微调
"""

import os
import sys
import torch
import yaml
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loader import format_conversation, tokenize_function, load_local_dataset


class Config:
    """训练配置类"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = project_root / "configs" / "lora_config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 模型配置
        self.model_name = config["model"]["name"]
        self.max_seq_length = config["model"]["max_seq_length"]
        self.use_4bit_quantization = config["model"].get("use_4bit_quantization", False)

        # LoRA 配置
        self.lora_r = config["lora"]["r"]
        self.lora_alpha = config["lora"]["lora_alpha"]
        self.lora_target_modules = config["lora"]["target_modules"]
        self.lora_dropout = config["lora"]["lora_dropout"]
        self.lora_bias = config["lora"]["bias"]
        self.lora_task_type = config["lora"]["task_type"]

        # 训练配置
        self.output_dir = config["training"]["output_dir"]
        self.num_train_epochs = config["training"]["num_train_epochs"]
        self.per_device_train_batch_size = config["training"]["per_device_train_batch_size"]
        self.per_device_eval_batch_size = config["training"]["per_device_eval_batch_size"]
        self.gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.gradient_checkpointing = config["training"].get("gradient_checkpointing", False)
        self.learning_rate = float(config["training"]["learning_rate"])
        self.weight_decay = float(config["training"]["weight_decay"])
        self.warmup_ratio = float(config["training"]["warmup_ratio"])
        self.lr_scheduler_type = config["training"]["lr_scheduler_type"]
        self.logging_steps = config["training"]["logging_steps"]
        self.save_steps = config["training"]["save_steps"]
        self.eval_steps = config["training"]["eval_steps"]
        self.evaluation_strategy = config["training"]["evaluation_strategy"]
        self.save_total_limit = config["training"]["save_total_limit"]
        self.fp16 = config["training"]["fp16"]
        self.optim = config["training"]["optim"]
        self.load_best_model_at_end = config["training"]["load_best_model_at_end"]
        self.metric_for_best_model = config["training"]["metric_for_best_model"]
        self.greater_is_better = config["training"]["greater_is_better"]

        # 数据配置
        self.dataset_name = config["data"]["dataset_name"]
        self.test_size = config["data"]["test_size"]
        self.seed = config["data"]["seed"]


def load_model_and_tokenizer(config: Config):
    """
    加载模型和分词器

    Args:
        config: 配置对象

    Returns:
        model: 模型
        tokenizer: 分词器
    """
    print(f"\n=== 加载模型: {config.model_name} ===")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("已设置 pad_token = eos_token")

    # 加载模型
    if config.use_4bit_quantization:
        print("使用 4-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    print(f"模型加载完成，参数量: {model.num_parameters() / 1e6:.1f}M")

    return model, tokenizer


def apply_lora(model, config: Config):
    """
    应用 LoRA 配置到模型

    Args:
        model: 基座模型
        config: 配置对象

    Returns:
        添加了 LoRA 的模型
    """
    print(f"\n=== 应用 LoRA (r={config.lora_r}, alpha={config.lora_alpha}) ===")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def prepare_dataset(config: Config, tokenizer):
    """
    准备数据集

    Args:
        config: 配置对象
        tokenizer: 分词器

    Returns:
        train_dataset: 训练集
        val_dataset: 验证集
    """
    print("\n=== 准备数据集 ===")

    # 尝试从本地加载，否则从 HuggingFace 加载
    processed_dir = project_root / "processed"

    if (processed_dir / "train.jsonl").exists():
        train_data, val_data = load_local_dataset(str(processed_dir))
        from datasets import Dataset

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
    else:
        # 从 HuggingFace 加载
        print(f"从 HuggingFace 加载数据集: {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, split="train")

        # 过滤有 system prompt 的数据
        dataset = dataset.filter(
            lambda x: any(m["from"] == "system" for m in x["conversations"])
        )

        # 划分数据集
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(dataset)),
            test_size=config.test_size,
            random_state=config.seed
        )

        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")

    # 数据预处理
    print("\n=== 数据预处理 ===")

    # 格式化对话
    train_dataset = train_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="格式化训练集"
    )

    val_dataset = val_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=val_dataset.column_names,
        desc="格式化验证集"
    )

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        remove_columns=["text"],
        desc="Tokenize 训练集"
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        remove_columns=["text"],
        desc="Tokenize 验证集"
    )

    return train_dataset, val_dataset


def create_trainer(model, tokenizer, train_dataset, val_dataset, config: Config):
    """
    创建训练器

    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练集
        val_dataset: 验证集
        config: 配置对象

    Returns:
        trainer: 训练器
    """
    print("\n=== 创建训练器 ===")

    # 输出目录
    output_dir = project_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy=config.evaluation_strategy,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        optim="adamw_bnb_8bit",
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        report_to="none",
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型不使用 MLM
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    return trainer


def train(config_path: str = None, quick_test: bool = False):
    """
    主训练函数

    Args:
        config_path: 配置文件路径
        quick_test: 是否快速测试（只训练少量数据）
    """
    print("=" * 60)
    print("LoRA 训练 - 角色扮演对话生成")
    print("=" * 60)

    # 加载配置
    config = Config(config_path)

    # 验证 GPU
    print("\n=== 验证环境 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config)

    # 应用 LoRA
    model = apply_lora(model, config)

    # 准备数据集
    train_dataset, val_dataset = prepare_dataset(config, tokenizer)

    if quick_test:
        print("\n=== 快速测试模式 ===")
        train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(5, len(val_dataset))))
        config.num_train_epochs = 1
        config.logging_steps = 1
        config.save_steps = 50
        config.eval_steps = 50

    # 创建训练器
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, config)

    # 开始训练
    print("\n=== 开始训练 ===")
    trainer.train()

    # 保存模型
    print("\n=== 保存模型 ===")
    final_dir = project_root / config.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"模型已保存到: {final_dir}")

    print("\n=== 训练完成 ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA 训练脚本")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式")
    args = parser.parse_args()

    train(config_path=args.config, quick_test=args.quick_test)
