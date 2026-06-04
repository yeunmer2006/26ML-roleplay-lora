#!/usr/bin/env python3
"""
数据加载与清洗脚本
从 HuggingFace 加载 PIPPA 数据集并进行清洗
"""

import os
import json
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_and_clean_dataset(
    dataset_name: str = "KaraKaraWitch/PIPPA-ShareGPT-formatted",
    val_size: float = 0.1,
    test_size: float = 0.1,
    max_conversation_length: int = 8000,
    seed: int = 42
):
    """
    加载并清洗数据集，支持训练集/验证集/测试集划分

    Args:
        dataset_name: HuggingFace 数据集名称
        val_size: 验证集比例
        test_size: 测试集比例
        max_conversation_length: 单条对话最大长度
        seed: 随机种子
    """
    print(f"=== 加载数据集: {dataset_name} ===")
    dataset = load_dataset(dataset_name, split="train")
    print(f"原始数据: {len(dataset)} 条")
    print(f"字段: {dataset.column_names}")

    # 数据清洗
    print("\n=== 数据清洗 ===")
    cleaned_data = []

    for i, item in enumerate(dataset):
        # 跳过空对话
        if not item.get("conversations"):
            continue

        conversations = item["conversations"]

        # 确保有 system 角色设定
        if not any(m.get("from") == "system" for m in conversations):
            continue

        # 跳过对话轮次不足的（至少 1 轮 human-gpt）
        msgs = [m for m in conversations if m.get("from") in ["human", "gpt"]]
        if len(msgs) < 2:
            continue

        # 跳过超长内容
        total_len = sum(len(m.get("value", "")) for m in conversations)
        if total_len > max_conversation_length:
            continue

        cleaned_data.append(item)

        if (i + 1) % 2000 == 0:
            print(f"已处理 {i + 1} 条...")

    print(f"清洗后数据: {len(cleaned_data)} 条")

    # 数据集划分：8/1/1（训练/验证/测试）
    print("\n=== 数据集划分 (8/1/1) ===")

    # 先划分出测试集（10%）
    train_val_data, test_data = train_test_split(
        cleaned_data,
        test_size=test_size,
        random_state=seed
    )

    # 再从剩余数据中划分验证集（目标 10% / 剩余 90% ≈ 11.11%）
    val_ratio_in_remains = val_size / (1 - test_size)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio_in_remains,
        random_state=seed
    )

    print(f"训练集: {len(train_data)} 条 ({len(train_data)/len(cleaned_data)*100:.1f}%)")
    print(f"验证集: {len(val_data)} 条 ({len(val_data)/len(cleaned_data)*100:.1f}%)")
    print(f"测试集: {len(test_data)} 条 ({len(test_data)/len(cleaned_data)*100:.1f}%)")

    return train_data, val_data, test_data


def save_dataset(train_data, val_data, test_data=None, output_dir: str = "./processed"):
    """保存清洗后的数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存为 JSONL 格式
    print(f"\n=== 保存数据到 {output_dir} ===")

    splits = [("train", train_data), ("val", val_data)]
    if test_data is not None:
        splits.append(("test", test_data))

    for split_name, data in splits:
        file_path = output_path / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"已保存: {file_path} ({len(data)} 条)")


def load_local_dataset(data_dir: str = "./processed"):
    """从本地加载已清洗的数据集"""
    print(f"\n=== 从本地加载数据: {data_dir} ===")

    train_data = []
    val_data = []
    test_data = []

    train_path = Path(data_dir) / "train.jsonl"
    val_path = Path(data_dir) / "val.jsonl"
    test_path = Path(data_dir) / "test.jsonl"

    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        print(f"训练集: {len(train_data)} 条")

    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                val_data.append(json.loads(line.strip()))
        print(f"验证集: {len(val_data)} 条")

    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        print(f"测试集: {len(test_data)} 条")

    return train_data, val_data, test_data


def format_conversation(example, tokenizer):
    """
    将 PIPPA 格式转换为模型输入格式

    Args:
        example: 单条数据示例
        tokenizer: 分词器

    Returns:
        格式化后的字典
    """
    conversations = example["conversations"]

    messages = []
    for msg in conversations:
        msg_from = msg.get("from", "")
        msg_value = msg.get("value", "")

        if msg_from == "system":
            messages.append({"role": "system", "content": msg_value})
        elif msg_from == "human":
            messages.append({"role": "user", "content": msg_value})
        elif msg_from == "gpt":
            messages.append({"role": "assistant", "content": msg_value})

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return {"text": text}


def tokenize_function(examples, tokenizer, max_length: int = 2048):
    """
    Tokenize 文本

    Args:
        examples: 批量数据
        tokenizer: 分词器
        max_length: 最大序列长度

    Returns:
        tokenize 后的字典
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )

    # 设置 labels 用于语言模型训练
    result["labels"] = result["input_ids"].copy()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据加载与清洗")
    parser.add_argument("--dataset", type=str, default="KaraKaraWitch/PIPPA-ShareGPT-formatted",
                        help="HuggingFace 数据集名称")
    parser.add_argument("--output_dir", type=str, default="./processed",
                        help="输出目录")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="测试集比例")
    args = parser.parse_args()

    # 加载并清洗数据（8/1/1 划分）
    train_data, val_data, test_data = load_and_clean_dataset(
        dataset_name=args.dataset,
        val_size=args.val_size,
        test_size=args.test_size
    )

    # 保存到本地
    save_dataset(train_data, val_data, test_data, args.output_dir)

    print("\n=== 数据加载完成 ===")
