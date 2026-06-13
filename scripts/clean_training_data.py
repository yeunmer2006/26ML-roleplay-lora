#!/usr/bin/env python3
"""清洗现有 train/val 数据，并保持 test 集不变。"""

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def words(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def repeated_sentence(text, repeat_limit):
    sentences = [
        " ".join(words(sentence))
        for sentence in re.split(r"(?:[.!?。！？]+\s*|\n+)", text)
    ]
    counts = Counter(sentence for sentence in sentences if len(sentence.split()) >= 3)
    return any(count >= repeat_limit for count in counts.values())


def ngram_repetition(text, ngram_size, min_ngrams):
    tokens = words(text)
    ngrams = [
        tuple(tokens[index:index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    ]
    if len(ngrams) < min_ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def has_user_assistant_pair(conversations):
    seen_user = False
    for message in conversations:
        role = message.get("from")
        content = (message.get("value") or "").strip()
        if not content:
            continue
        if role == "human":
            seen_user = True
        elif role == "gpt" and seen_user:
            return True
    return False


def rejection_reason(item, threshold, ngram_size, min_ngrams, repeat_limit):
    conversations = item.get("conversations") or []
    if not has_user_assistant_pair(conversations):
        return "missing_user_assistant_pair"

    assistant_texts = [
        (message.get("value") or "").strip()
        for message in conversations
        if message.get("from") == "gpt" and (message.get("value") or "").strip()
    ]
    for text in assistant_texts:
        if repeated_sentence(text, repeat_limit):
            return "repeated_sentence"
        if ngram_repetition(text, ngram_size, min_ngrams) >= threshold:
            return "high_ngram_repetition"
    return None


def clean_split(records, args):
    kept = []
    reasons = Counter()
    for item in records:
        reason = rejection_reason(
            item,
            args.max_repetition,
            args.ngram_size,
            args.min_ngrams,
            args.sentence_repeat_limit,
        )
        if reason:
            reasons[reason] += 1
        else:
            kept.append(item)
    return kept, reasons


def main():
    parser = argparse.ArgumentParser(description="清洗 processed 训练/验证数据")
    parser.add_argument("--input_dir", default="processed")
    parser.add_argument("--output_dir", default="processed_clean")
    parser.add_argument("--ngram_size", type=int, default=4)
    parser.add_argument("--min_ngrams", type=int, default=20)
    parser.add_argument("--max_repetition", type=float, default=0.35)
    parser.add_argument("--sentence_repeat_limit", type=int, default=3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(input_dir),
        "rules": {
            "ngram_size": args.ngram_size,
            "min_ngrams": args.min_ngrams,
            "max_repetition": args.max_repetition,
            "sentence_repeat_limit": args.sentence_repeat_limit,
            "require_user_before_assistant": True,
        },
        "splits": {},
    }

    for split in ("train", "val"):
        source = input_dir / f"{split}.jsonl"
        records = load_jsonl(source)
        cleaned, reasons = clean_split(records, args)
        target = output_dir / f"{split}.jsonl"
        write_jsonl(target, cleaned)
        manifest["splits"][split] = {
            "source_rows": len(records),
            "kept_rows": len(cleaned),
            "removed_rows": len(records) - len(cleaned),
            "removed_by_reason": dict(sorted(reasons.items())),
            "sha256": sha256(target),
        }

    source_test = input_dir / "test.jsonl"
    target_test = output_dir / "test.jsonl"
    shutil.copyfile(source_test, target_test)
    manifest["splits"]["test"] = {
        "rows": sum(1 for line in target_test.open(encoding="utf-8") if line.strip()),
        "copied_without_changes": True,
        "sha256": sha256(target_test),
    }

    manifest_path = output_dir / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
