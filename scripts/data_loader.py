#!/usr/bin/env python3
"""PIPPA 数据下载、清洗和训练样本编码。"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit


ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


def load_and_clean_dataset(
    dataset_name: str = "KaraKaraWitch/PIPPA-ShareGPT-formatted",
    val_size: float = 0.1,
    test_size: float = 0.1,
    max_conversation_length: int = 8000,
    seed: int = 42,
):
    """下载、清洗并按 8/1/1 划分 PIPPA 数据。"""
    print(f"=== 加载数据集: {dataset_name} ===")
    dataset = load_dataset(dataset_name, split="train")
    cleaned_data = []
    seen = set()

    for item in dataset:
        conversations = item.get("conversations") or []
        bot = item.get("bot") or {}
        bot_description = bot.get("description", "").strip()
        system_messages = [
            message.get("value", "").strip()
            for message in conversations
            if message.get("from") == "system"
        ]
        dialogue = [
            message
            for message in conversations
            if message.get("from") in {"human", "gpt"}
            and message.get("value", "").strip()
        ]

        if not (bot_description or any(system_messages)) or len(dialogue) < 2:
            continue
        if not any(message.get("from") == "human" for message in dialogue):
            continue
        if not any(message.get("from") == "gpt" for message in dialogue):
            continue

        total_length = sum(len(message.get("value", "")) for message in conversations)
        if total_length > max_conversation_length:
            continue
        fingerprint = hashlib.sha256(
            json.dumps(conversations, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        cleaned_data.append(item)

    groups = [
        (item.get("bot") or {}).get("name") or item.get("id") or str(index)
        for index, item in enumerate(cleaned_data)
    ]
    holdout_size = val_size + test_size
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_size,
        random_state=seed,
    )
    train_indices, holdout_indices = next(
        outer_split.split(cleaned_data, groups=groups)
    )
    train_data = [cleaned_data[index] for index in train_indices]
    holdout_data = [cleaned_data[index] for index in holdout_indices]
    holdout_groups = [groups[index] for index in holdout_indices]

    relative_test_size = test_size / holdout_size
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=seed,
    )
    val_indices, test_indices = next(
        inner_split.split(holdout_data, groups=holdout_groups)
    )
    val_data = [holdout_data[index] for index in val_indices]
    test_data = [holdout_data[index] for index in test_indices]

    print(f"原始数据: {len(dataset)} 条")
    print(f"清洗后数据: {len(cleaned_data)} 条")
    print(f"训练/验证/测试: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    return train_data, val_data, test_data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset(train_data, val_data, test_data=None, output_dir: str = "./processed"):
    """将数据保存为 JSONL，并生成可复现性清单。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    splits = [("train", train_data), ("val", val_data)]
    if test_data is not None:
        splits.append(("test", test_data))

    manifest = {"format": "PIPPA-ShareGPT", "splits": {}}
    for split_name, data in splits:
        file_path = output_path / f"{split_name}.jsonl"
        with file_path.open("w", encoding="utf-8") as file:
            for item in data:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest["splits"][split_name] = {
            "rows": len(data),
            "sha256": _sha256(file_path),
        }
        print(f"已保存: {file_path} ({len(data)} 条)")

    manifest_path = output_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_local_dataset(data_dir: str = "./processed"):
    """从本地目录加载 train/val/test JSONL。"""
    data_path = Path(data_dir)

    def load_split(name):
        path = data_path / f"{name}.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    print(f"从 {data_path} 加载训练/验证/测试: "
          f"{len(train_data)}/{len(val_data)}/{len(test_data)}")
    return train_data, val_data, test_data


def build_messages(example):
    """将 PIPPA 角色转换为 Hugging Face chat messages。"""
    conversations = example.get("conversations", [])
    bot = example.get("bot") or {}
    bot_name = (bot.get("name") or "").strip()
    bot_description = (bot.get("description") or "").strip()
    original_system = next(
        (
            message.get("value", "").strip()
            for message in conversations
            if message.get("from") == "system" and message.get("value", "").strip()
        ),
        "",
    )

    if bot_description:
        system_content = (
            f"Character name: {bot_name}\n{bot_description}"
            if bot_name
            else bot_description
        )
    else:
        system_content = original_system

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    for message in conversations:
        role = ROLE_MAP.get(message.get("from"))
        content = message.get("value", "").strip()
        if role and role != "system" and content:
            messages.append({"role": role, "content": content})
    return messages


def format_conversation(example, tokenizer):
    """保留旧接口，用聊天模板将对话格式化为文本。"""
    text = tokenizer.apply_chat_template(
        build_messages(example),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def _template_ids(tokenizer, messages, add_generation_prompt=False):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(encoded, "get"):
        return encoded["input_ids"]
    return encoded


def truncate_messages(messages, tokenizer, max_length):
    """保留 system 和最近的完整 user-assistant 后缀。

    max_length 是软预算。如果 system 与最后一组 user-assistant 消息已经超出
    预算，仍完整保留该组，避免产生缺少用户输入的孤立 assistant 回复。
    """
    if not messages:
        return []

    system = messages[:1] if messages[0]["role"] == "system" else []
    dialogue = messages[len(system):]
    last_assistant = next(
        (
            index
            for index in range(len(dialogue) - 1, -1, -1)
            if dialogue[index]["role"] == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return system

    dialogue = dialogue[:last_assistant + 1]
    user_starts = [
        index
        for index, message in enumerate(dialogue)
        if message["role"] == "user"
    ]
    for start in user_starts:
        candidate = system + dialogue[start:]
        if len(_template_ids(tokenizer, candidate)) <= max_length:
            return candidate

    if user_starts:
        return system + dialogue[user_starts[-1]:]
    return system


def encode_conversation(example, tokenizer, max_length: int = 512):
    """编码对话，并只对 assistant token 计算训练损失。

    对超长样本保留完整角色卡，并按消息边界删除最旧对话。
    """
    messages = truncate_messages(build_messages(example), tokenizer, max_length)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": [], "length": 0}

    input_ids = _template_ids(tokenizer, messages)
    labels = [-100] * len(input_ids)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if index == 0:
            end = len(_template_ids(tokenizer, messages[:1]))
            labels[:end] = input_ids[:end]
            continue
        prefix_ids = _template_ids(
            tokenizer,
            messages[:index],
            add_generation_prompt=True,
        )
        through_assistant_ids = _template_ids(tokenizer, messages[: index + 1])
        start = len(prefix_ids)
        end = min(len(through_assistant_ids), len(labels))
        if through_assistant_ids[:start] != prefix_ids:
            raise ValueError("聊天模板前缀不稳定，无法可靠生成 assistant labels")
        labels[start:end] = input_ids[start:end]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "length": len(input_ids),
    }


def tokenize_function(examples, tokenizer, max_length: int = 512):
    """兼容旧调用；新训练流程应使用 encode_conversation。"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    parser = argparse.ArgumentParser(description="下载并清洗 PIPPA 数据")
    parser.add_argument(
        "--dataset",
        default="KaraKaraWitch/PIPPA-ShareGPT-formatted",
    )
    parser.add_argument("--output_dir", default="./processed")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    train_data, val_data, test_data = load_and_clean_dataset(
        dataset_name=args.dataset,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    save_dataset(train_data, val_data, test_data, args.output_dir)


if __name__ == "__main__":
    main()
