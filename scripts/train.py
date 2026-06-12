#!/usr/bin/env python3
"""Qwen2.5 QLoRA 角色扮演训练入口。"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_loader import encode_conversation, load_local_dataset


class Config:
    """读取 YAML，同时为旧配置提供合理默认值。"""

    def __init__(self, config_path=None):
        path = Path(config_path) if config_path else PROJECT_ROOT / "configs/train_4060.yaml"
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        self.path = path
        self.raw = raw

        model = raw["model"]
        lora = raw["lora"]
        training = raw["training"]
        data = raw["data"]

        self.model_name = model["name"]
        self.max_seq_length = int(model.get("max_seq_length", 512))
        self.use_4bit_quantization = bool(model.get("use_4bit_quantization", True))
        self.bnb_compute_dtype = model.get("bnb_compute_dtype", "float16")

        self.lora_r = int(lora.get("r", 8))
        self.lora_alpha = int(lora.get("lora_alpha", 16))
        self.lora_target_modules = lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.lora_dropout = float(lora.get("lora_dropout", 0.05))
        self.lora_bias = lora.get("bias", "none")

        self.output_dir = training.get("output_dir", "output/experiments/default")
        self.num_train_epochs = float(training.get("num_train_epochs", 1))
        self.max_steps = int(training.get("max_steps", -1))
        self.train_batch_size = int(training.get("per_device_train_batch_size", 1))
        self.eval_batch_size = int(training.get("per_device_eval_batch_size", 1))
        self.per_device_train_batch_size = self.train_batch_size
        self.per_device_eval_batch_size = self.eval_batch_size
        self.gradient_accumulation_steps = int(
            training.get("gradient_accumulation_steps", 8)
        )
        self.gradient_checkpointing = bool(
            training.get("gradient_checkpointing", True)
        )
        self.learning_rate = float(training.get("learning_rate", 2e-4))
        self.weight_decay = float(training.get("weight_decay", 0.01))
        self.warmup_ratio = float(training.get("warmup_ratio", 0.03))
        self.lr_scheduler_type = training.get("lr_scheduler_type", "cosine")
        self.logging_steps = int(training.get("logging_steps", 10))
        self.save_steps = int(training.get("save_steps", 250))
        self.save_strategy = training.get("save_strategy", "steps")
        self.eval_strategy = training.get("eval_strategy", "epoch")
        self.save_total_limit = int(training.get("save_total_limit", 2))
        self.fp16 = bool(training.get("fp16", True))
        self.bf16 = bool(training.get("bf16", False))
        self.optim = training.get("optim", "paged_adamw_8bit")
        self.dataloader_num_workers = int(training.get("dataloader_num_workers", 2))
        self.seed = int(data.get("seed", 42))
        self.dataset_name = data.get(
            "dataset_name",
            "KaraKaraWitch/PIPPA-ShareGPT-formatted",
        )
        self.data_dir = data.get("data_dir", "processed")
        self.max_train_samples = data.get("max_train_samples", 4000)
        self.max_eval_samples = data.get("max_eval_samples", 200)
        # best-model 选择相关的三个字段：让 yaml 的设置真正生效
        self.load_best_model_at_end = bool(training.get("load_best_model_at_end", False))
        self.metric_for_best_model = training.get("metric_for_best_model", None)
        self.greater_is_better = training.get("greater_is_better", None)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_model_source(value):
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_path = PROJECT_ROOT / path
    return str(project_path) if project_path.exists() else value


def load_model_and_tokenizer(config, model_path=None):
    source = resolve_model_source(model_path) if model_path else config.model_name
    print(f"\n=== 加载模型: {source} ===")
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.name_or_path = config.model_name
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.bnb_compute_dtype == "bfloat16" else torch.float16
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": 0},
    }
    if config.use_4bit_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    model.name_or_path = config.model_name
    model.config._name_or_path = config.model_name
    if config.use_4bit_quantization:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, config):
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


def _select(dataset, limit, seed):
    if limit is None or int(limit) <= 0 or len(dataset) <= int(limit):
        return dataset
    return dataset.shuffle(seed=seed).select(range(int(limit)))


def prepare_dataset(config, tokenizer, data_dir=None, train_limit=None, eval_limit=None):
    data_path = resolve_path(data_dir or config.data_dir)
    if (data_path / "train.jsonl").exists():
        train_data, val_data, _ = load_local_dataset(str(data_path))
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(val_data)
    else:
        print(f"本地数据不存在，从 Hugging Face 加载: {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, split="train")
        split = dataset.train_test_split(test_size=0.1, seed=config.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    train_dataset = _select(
        train_dataset,
        train_limit if train_limit is not None else config.max_train_samples,
        config.seed,
    )
    eval_dataset = _select(
        eval_dataset,
        eval_limit if eval_limit is not None else config.max_eval_samples,
        config.seed,
    )

    def encode(example):
        return encode_conversation(example, tokenizer, config.max_seq_length)

    train_dataset = train_dataset.map(
        encode,
        remove_columns=train_dataset.column_names,
        desc="编码训练集",
    ).filter(lambda example: any(label != -100 for label in example["labels"]))
    eval_dataset = eval_dataset.map(
        encode,
        remove_columns=eval_dataset.column_names,
        desc="编码验证集",
    ).filter(lambda example: any(label != -100 for label in example["labels"]))

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise RuntimeError("编码后训练集或验证集为空，请检查数据和聊天模板")
    print(f"有效训练/验证样本: {len(train_dataset)}/{len(eval_dataset)}")
    return train_dataset, eval_dataset


def create_trainer(model, tokenizer, train_dataset, eval_dataset, config, output_dir,
                   max_steps=None, benchmark=False):
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps if max_steps is None else max_steps,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=1 if benchmark else config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="no" if benchmark else config.save_strategy,
        eval_strategy="no" if benchmark else config.eval_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        bf16=config.bf16,
        optim=config.optim,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        include_num_input_tokens_seen=True,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir, config, data_dir, mode, extra=None):
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    data_path = resolve_path(data_dir)
    data_files = {}
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        path = data_path / name
        if path.exists():
            data_files[name] = _file_sha256(path)

    manifest = {
        "mode": mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_id": config.model_name,
        "config": config.raw,
        "data_sha256": data_files,
    }
    if extra:
        manifest.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；QLoRA 训练必须在 NVIDIA GPU 环境运行")

    config = Config(args.config)
    output_dir = resolve_path(args.output_dir or config.output_dir)
    data_dir = args.data_dir or config.data_dir
    train_limit = args.max_train_samples
    eval_limit = args.max_eval_samples

    model, tokenizer = load_model_and_tokenizer(config, args.model_path)
    model = apply_lora(model, config)
    train_dataset, eval_dataset = prepare_dataset(
        config,
        tokenizer,
        data_dir=data_dir,
        train_limit=train_limit,
        eval_limit=eval_limit,
    )
    write_manifest(
        output_dir,
        config,
        data_dir,
        "benchmark" if args.benchmark_steps else "train",
        {"train_samples": len(train_dataset), "eval_samples": len(eval_dataset)},
    )

    if args.benchmark_steps:
        trainer = create_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            config,
            output_dir,
            max_steps=args.benchmark_steps,
            benchmark=True,
        )
        started = time.monotonic()
        trainer.train()
        elapsed = time.monotonic() - started
        seconds_per_step = elapsed / args.benchmark_steps
        optimizer_steps = math.ceil(
            len(train_dataset)
            / (config.train_batch_size * config.gradient_accumulation_steps)
        )
        estimated_seconds = seconds_per_step * optimizer_steps
        result = {
            "benchmark_steps": args.benchmark_steps,
            "seconds_per_step": seconds_per_step,
            "estimated_optimizer_steps": optimizer_steps,
            "estimated_train_seconds": estimated_seconds,
            "estimated_train_minutes": estimated_seconds / 60,
            "limit_minutes": args.max_runtime_minutes,
            "passed": estimated_seconds <= args.max_runtime_minutes * 60,
        }
        (output_dir / "benchmark.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 3

    trainer = create_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config,
        output_dir,
        max_steps=args.max_steps,
    )
    started = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.monotonic() - started

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    write_manifest(
        output_dir,
        config,
        data_dir,
        "train",
        {
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "elapsed_seconds": elapsed,
            "train_metrics": train_result.metrics,
            "final_model": "final_model",
        },
    )
    print(f"训练完成，Adapter 已保存到: {final_dir}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 QLoRA 训练")
    parser.add_argument("--config", default="configs/train_4060.yaml")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--model_path",
        default=os.getenv("MODEL_DIR"),
        help="基座模型目录或模型 ID；默认读取 MODEL_DIR",
    )
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--benchmark_steps", type=int, default=0)
    parser.add_argument("--max_runtime_minutes", type=float, default=110)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM：请降低序列长度或 batch size。", file=sys.stderr)
        sys.exit(2)
