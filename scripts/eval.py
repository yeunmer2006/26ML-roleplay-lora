#!/usr/bin/env python3
"""Three-way ablation evaluation for role-play LoRA models."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_loader import build_messages, load_local_dataset, truncate_messages


SYSTEMS = ("base_no_card", "base_with_card", "lora_with_card")
SYSTEM_LABELS = {
    "base_no_card": "Base, no card",
    "base_with_card": "Base + card",
    "lora_with_card": "LoRA + card",
}
BASELINE_SYSTEMS = ("base_no_card", "base_with_card")
PERFORMANCE_FIELDS = (
    "latency_seconds",
    "tokens_per_second",
    "peak_gpu_memory_mb",
)
SINGLE_WEIGHTS = {
    "role_identity": 0.35,
    "style": 0.20,
    "relevance": 0.20,
    "naturalness": 0.15,
    "immersion": 0.10,
}
MULTI_WEIGHTS = {
    "role_identity": 0.25,
    "memory": 0.25,
    "coherence": 0.20,
    "style": 0.15,
    "immersion": 0.15,
}
REFUSAL_PATTERNS = (
    "as an ai",
    "as a language model",
    "i cannot",
    "i can't comply",
    "作为一个ai",
    "作为人工智能",
    "我不能满足",
    "我无法满足",
)
MULTI_TURN_PROMPTS = (
    "请以这个角色的身份介绍你自己和你的背景。",
    "如果朋友要求你为了集体利益隐瞒一个无伤大雅的真相，你会怎么处理？请按你的价值观回答。",
    "请记住这条信息：我最喜欢的颜色是蓝色，我的代号是北星。先正常回应我。",
    "我刚才告诉你的颜色和代号分别是什么？回答后请继续保持角色，不要改成普通AI助手。",
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_model_source(value):
    candidate = resolve_path(value)
    return str(candidate) if candidate.exists() else value


def stable_int(value, seed=42):
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return {}
    records = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                records[record["sample_id"]] = record
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        values = records.values() if isinstance(records, dict) else records
        for record in sorted(values, key=lambda item: item["sample_id"]):
            file.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
    temporary.replace(path)


def model_name(value):
    return Path(str(value).rstrip("/")).name.casefold()


def adapter_base_model(adapter_path):
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    base_model = load_json(config_path).get("base_model_name_or_path")
    if not base_model:
        raise RuntimeError(f"Missing base_model_name_or_path in {config_path}")
    return base_model


def comparable_record(record):
    return {
        key: value
        for key, value in record.items()
        if key not in {"systems", "judge", "judge_consistency"}
    }


def validate_reuse_manifest(source, args, adapter_path):
    expected = {
        "dataset": args.dataset,
        "safety_rules": args.safety_rules,
        "seed": args.seed,
        "single_samples": args.single_samples,
        "multi_samples": args.multi_samples,
        "max_new_tokens": args.max_new_tokens,
        "generation": {"do_sample": False},
    }
    differences = [
        f"{key}: source={source.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if source.get(key) != value
    ]
    model_names = {
        model_name(source.get("base_model", "")),
        model_name(args.base_model),
        model_name(adapter_base_model(adapter_path)),
    }
    if "" in model_names or len(model_names) != 1:
        differences.append(
            "base_model: "
            f"source={source.get('base_model')!r}, current={args.base_model!r}, "
            f"adapter={adapter_base_model(adapter_path)!r}"
        )
    if differences:
        raise RuntimeError(
            "Baseline evaluation is incompatible:\n- " + "\n- ".join(differences)
        )


def import_baseline_records(current, source, kind):
    if set(current) != set(source):
        missing = sorted(set(current) - set(source))
        extra = sorted(set(source) - set(current))
        raise RuntimeError(
            f"Baseline {kind} sample IDs differ: missing={missing[:5]}, extra={extra[:5]}"
        )

    for sample_id, record in current.items():
        source_record = source[sample_id]
        if comparable_record(record) != comparable_record(source_record):
            raise RuntimeError(
                f"Baseline {kind} sample content differs: {sample_id}"
            )
        for system in BASELINE_SYSTEMS:
            result = source_record.get("systems", {}).get(system)
            if not result or result.get("error"):
                raise RuntimeError(
                    f"Baseline {kind} result is missing or failed: "
                    f"{sample_id} / {system}"
                )
            imported = copy.deepcopy(result)
            for field in PERFORMANCE_FIELDS:
                imported.pop(field, None)
            record["systems"][system] = imported


def reuse_baseline(args, adapter_path, single_records, multi_records):
    source_dir = resolve_path(args.reuse_baseline)
    output_dir = resolve_path(args.output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise RuntimeError("--reuse_baseline must differ from --output_dir")

    manifest_path = source_dir / "manifest.json"
    single_path = source_dir / "single_turn_samples.jsonl"
    multi_path = source_dir / "multi_turn_samples.jsonl"
    for path in (manifest_path, single_path, multi_path):
        if not path.exists():
            raise FileNotFoundError(f"Baseline evaluation file not found: {path}")

    validate_reuse_manifest(load_json(manifest_path), args, adapter_path)
    import_baseline_records(
        single_records, load_jsonl(single_path), "single-turn"
    )
    import_baseline_records(
        multi_records, load_jsonl(multi_path), "multi-turn"
    )
    return {
        "source": args.reuse_baseline,
        "systems": list(BASELINE_SYSTEMS),
        "performance_fields_excluded": list(PERFORMANCE_FIELDS),
    }


def append_failure(path, record):
    path = Path(path)
    failures = load_jsonl(path)
    failures[record["sample_id"]] = record
    write_jsonl(path, failures)


def remove_failure(path, sample_id):
    failures = load_jsonl(path)
    if sample_id in failures:
        del failures[sample_id]
        write_jsonl(path, failures)


def load_safety_rules(path):
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return {
        category: [term.casefold() for term in terms]
        for category, terms in raw["categories"].items()
    }


def safety_matches(item, rules):
    bot = item.get("bot") or {}
    parts = [bot.get("name", ""), bot.get("description", "")]
    parts.extend(message.get("value", "") for message in item.get("conversations", []))
    text = "\n".join(parts).casefold()
    return {
        category: [term for term in terms if term in text]
        for category, terms in rules.items()
        if any(term in text for term in terms)
    }


def character_id(item):
    bot = item.get("bot") or {}
    name = (bot.get("name") or "").strip()
    description = (bot.get("description") or "").strip()
    identity = name or hashlib.sha256(description.encode("utf-8")).hexdigest()[:12]
    return identity.casefold()


def extract_single_turn(item):
    messages = build_messages(item)
    target_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if messages[index]["role"] == "assistant"),
        None,
    )
    if target_index is None:
        return None
    context = messages[:target_index]
    if not context or context[-1]["role"] != "user":
        return None
    system = next(
        (message["content"] for message in messages if message["role"] == "system"),
        "",
    )
    return {
        "context_with_card": context,
        "context_no_card": [
            message for message in context if message["role"] != "system"
        ],
        "character_card": system,
        "reference": messages[target_index]["content"],
    }


def select_safe_samples(items, rules, single_count, multi_count, seed):
    eligible = []
    excluded = []
    seen_characters = set()
    ordered = sorted(
        items,
        key=lambda item: stable_int(item.get("id", character_id(item)), seed),
    )

    for item in ordered:
        sample_id = str(item.get("id") or stable_int(json.dumps(item, sort_keys=True), seed))
        matches = safety_matches(item, rules)
        extracted = extract_single_turn(item)
        char_id = character_id(item)
        reason = None
        if matches:
            reason = "safety_filter"
        elif extracted is None:
            reason = "invalid_dialogue"
        elif not extracted["character_card"].strip():
            reason = "missing_character_card"
        elif char_id in seen_characters:
            reason = "duplicate_character"

        if reason:
            excluded.append({
                "sample_id": sample_id,
                "character_id": char_id,
                "reason": reason,
                "matches": matches,
            })
            continue

        seen_characters.add(char_id)
        eligible.append((item, extracted))
        if len(eligible) >= max(single_count, multi_count):
            break

    if len(eligible) < max(single_count, multi_count):
        raise RuntimeError(
            f"安全且角色唯一的样本不足：需要 {max(single_count, multi_count)}，"
            f"只有 {len(eligible)}"
        )

    single = []
    for item, extracted in eligible[:single_count]:
        bot = item.get("bot") or {}
        single.append({
            "sample_id": str(item.get("id") or stable_int(json.dumps(item, sort_keys=True), seed)),
            "character_id": character_id(item),
            "character_name": bot.get("name", ""),
            **extracted,
            "systems": {},
            "judge": None,
            "judge_consistency": None,
        })

    multi = []
    for item, extracted in eligible[:multi_count]:
        bot = item.get("bot") or {}
        multi.append({
            "sample_id": "multi-" + str(
                item.get("id") or stable_int(json.dumps(item, sort_keys=True), seed)
            ),
            "character_id": character_id(item),
            "character_name": bot.get("name", ""),
            "character_card": extracted["character_card"],
            "prompts": list(MULTI_TURN_PROMPTS),
            "systems": {},
            "judge": None,
            "judge_consistency": None,
        })
    return single, multi, excluded


def _template_ids(tokenizer, messages, add_generation_prompt=False):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return encoded["input_ids"] if hasattr(encoded, "get") else encoded


def assistant_perplexity(model, tokenizer, context, reference, max_length=1024):
    full_messages = context + [{"role": "assistant", "content": reference}]
    full_messages = truncate_messages(full_messages, tokenizer, max_length)
    if not full_messages or full_messages[-1]["role"] != "assistant":
        return None

    prompt_messages = full_messages[:-1]
    prompt_ids = (
        _template_ids(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        if prompt_messages
        else []
    )
    full_ids = _template_ids(tokenizer, full_messages)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prefix changed while scoring reference")

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    supervised = sum(label != -100 for label in labels)
    if supervised == 0:
        return None

    device = next(model.parameters()).device
    input_tensor = torch.tensor([full_ids], device=device)
    label_tensor = torch.tensor([labels], device=device)
    attention_mask = torch.ones_like(input_tensor)
    with torch.inference_mode():
        output = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            labels=label_tensor,
        )
    loss = float(output.loss)
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "supervised_tokens": supervised,
    }


def tokenize_words(text):
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9']+", text.casefold())


def ngrams(tokens, n):
    return [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]


def response_metrics(text):
    tokens = tokenize_words(text)
    unigram = ngrams(tokens, 1)
    bigram = ngrams(tokens, 2)
    repeated_bigrams = len(bigram) - len(set(bigram))
    lower = text.casefold().strip()
    refusal = not lower or any(pattern in lower for pattern in REFUSAL_PATTERNS)
    return {
        "response_chars": len(text),
        "response_tokens": len(tokens),
        "distinct_1": len(set(unigram)) / len(unigram) if unigram else 0.0,
        "distinct_2": len(set(bigram)) / len(bigram) if bigram else 0.0,
        "repetition_rate": repeated_bigrams / len(bigram) if bigram else 0.0,
        "empty_or_refusal": refusal,
    }


def generate_response(model, tokenizer, messages, max_new_tokens=256):
    input_ids = _template_ids(tokenizer, messages, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(inputs)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated_ids = output[0, len(input_ids):]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    token_count = int(generated_ids.numel())
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "response": text,
        "generated_tokens": token_count,
        "latency_seconds": elapsed,
        "tokens_per_second": token_count / elapsed if elapsed else 0.0,
        "peak_gpu_memory_mb": peak_memory,
        "automatic": response_metrics(text),
    }


def progress_line(stage, completed, total, started):
    elapsed = time.monotonic() - started
    rate = elapsed / completed if completed else 0.0
    remaining = rate * max(total - completed, 0)
    print(
        f"[eval] {stage}: {completed}/{total}, "
        f"elapsed={elapsed / 60:.1f}m, eta={remaining / 60:.1f}m",
        flush=True,
    )


def run_single_system(model, tokenizer, records, system, output_path, max_new_tokens):
    pending = [
        sample_id for sample_id in sorted(records)
        if system not in records[sample_id]["systems"]
    ]
    total = len(pending)
    if not total:
        return
    started = time.monotonic()
    print(f"[eval] Starting single-turn {SYSTEM_LABELS[system]}: {total} samples", flush=True)
    for completed, sample_id in enumerate(pending, 1):
        record = records[sample_id]
        context = (
            record["context_no_card"]
            if system == "base_no_card"
            else record["context_with_card"]
        )
        try:
            result = generate_response(
                model, tokenizer, context, max_new_tokens=max_new_tokens
            )
            result["assistant_perplexity"] = assistant_perplexity(
                model,
                tokenizer,
                context,
                record["reference"],
            )
            result["error"] = None
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        record["systems"][system] = result
        write_jsonl(output_path, records)
        progress_line(
            f"single-turn {SYSTEM_LABELS[system]}",
            completed,
            total,
            started,
        )


def run_multi_system(model, tokenizer, records, system, output_path, max_new_tokens):
    pending = [
        sample_id for sample_id in sorted(records)
        if system not in records[sample_id]["systems"]
    ]
    total = len(pending)
    if not total:
        return
    started = time.monotonic()
    print(f"[eval] Starting multi-turn {SYSTEM_LABELS[system]}: {total} characters", flush=True)
    for completed, sample_id in enumerate(pending, 1):
        record = records[sample_id]
        history = []
        if system != "base_no_card":
            history.append({"role": "system", "content": record["character_card"]})
        turns = []
        error_message = None
        total_latency = 0.0
        total_tokens = 0
        peak_memory = 0.0
        for prompt in record["prompts"]:
            history.append({"role": "user", "content": prompt})
            try:
                generated = generate_response(
                    model, tokenizer, history, max_new_tokens=max_new_tokens
                )
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
                break
            response = generated["response"]
            turns.append({"user": prompt, "assistant": response})
            history.append({"role": "assistant", "content": response})
            total_latency += generated["latency_seconds"]
            total_tokens += generated["generated_tokens"]
            peak_memory = max(peak_memory, generated["peak_gpu_memory_mb"])
        record["systems"][system] = {
            "turns": turns,
            "latency_seconds": total_latency,
            "generated_tokens": total_tokens,
            "tokens_per_second": total_tokens / total_latency if total_latency else 0.0,
            "peak_gpu_memory_mb": peak_memory,
            "error": error_message,
        }
        write_jsonl(output_path, records)
        progress_line(
            f"multi-turn {SYSTEM_LABELS[system]}",
            completed,
            total,
            started,
        )


def load_base_model(model_path):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    return model, tokenizer


def anonymize_systems(sample_id, systems, seed, alternate=False):
    labels = ["A", "B", "C"]
    shuffled = list(SYSTEMS)
    local_seed = stable_int(sample_id, seed) + (1 if alternate else 0)
    random.Random(local_seed).shuffle(shuffled)
    mapping = dict(zip(labels, shuffled))
    answers = {}
    for label, system in mapping.items():
        system_result = systems[system]
        answers[label] = (
            system_result.get("response")
            if "response" in system_result
            else system_result.get("turns", [])
        )
    return mapping, answers


def judge_schema_prompt(kind, character_card, context, answers):
    dimensions = SINGLE_WEIGHTS if kind == "single" else MULTI_WEIGHTS
    dimension_text = ", ".join(dimensions)
    payload = {
        "character_card": character_card,
        "conversation_context": context,
        "candidate_answers": answers,
    }
    return (
        "You are an impartial evaluator of role-play dialogue. The candidate labels "
        "are randomized. Do not infer model identity. Evaluate only the supplied text. "
        f"Score each candidate from 1 to 5 on: {dimension_text}. "
        "Return strict JSON with this shape: "
        '{"scores":{"A":{"dimension":1},"B":{},"C":{}},'
        '"ranking":["A","B","C"],"reason":"short explanation"}. '
        "Use every candidate exactly once in ranking. No markdown.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def parse_json_response(text):
    cleaned = text.strip()
    # MiniMax-M3 默认开启 adaptive thinking，会在内容前注入
    # <think>...</think> 块；JSON parser 看到非 JSON 前缀会直接抛错。
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def validate_judgment(value, dimensions):
    if sorted(value.get("ranking", [])) != ["A", "B", "C"]:
        raise ValueError("ranking must contain A, B, C exactly once")
    scores = value.get("scores", {})
    for label in ("A", "B", "C"):
        if label not in scores:
            raise ValueError(f"missing scores for {label}")
        for dimension in dimensions:
            score = scores[label].get(dimension)
            if not isinstance(score, (int, float)) or not 1 <= score <= 5:
                raise ValueError(f"invalid {dimension} score for {label}")
    return value


class JudgeClient:
    def __init__(self, base_url, api_key, model, timeout=120, retries=3,
                 max_completion_tokens=4096, disable_thinking=True):
        # 兼容 base url 的几种常见写法：
        #   https://api.minimaxi.com
        #   https://api.minimaxi.com/v1
        #   https://api.minimaxi.com/v1/chat/completions
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            self.url = base_url
        elif base_url.endswith("/v1"):
            self.url = base_url + "/chat/completions"
        else:
            self.url = base_url + "/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_completion_tokens = max_completion_tokens
        self.disable_thinking = disable_thinking

    def evaluate(self, prompt, dimensions):
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": self.max_completion_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # M3 默认开启 adaptive thinking，会在 content 里塞 <think>…</think>；
        # 显式关掉，judge 任务用不到思考，且避免污染 JSON 输出。
        # M2.x 关闭无效，会回退到 parse_json_response 里 strip 一次。
        if self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
            # reasoning_split 即使不开 thinking 也无害：把任何残余
            # thinking 内容拆到独立字段，content 保持干净。
            payload["reasoning_split"] = True
        last_error = None
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                message = body["choices"][0]["message"]
                # reasoning_split 启用时 thinking 走 reasoning_content；
                # 未启用或模型不支持时，content 里可能还带 <think>…</think>。
                content = message.get("content") or message.get("reasoning_content") or ""
                if not content:
                    raise ValueError(f"empty assistant message: {message!r}")
                parsed = validate_judgment(parse_json_response(content), dimensions)
                return {"parsed": parsed, "raw": content}
            except Exception as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge API failed after {self.retries} attempts: {last_error}")


def map_judgment_to_systems(judgment, mapping, weights):
    parsed = judgment["parsed"]
    system_scores = {}
    for label, system in mapping.items():
        dimensions = parsed["scores"][label]
        system_scores[system] = {
            **dimensions,
            "weighted_total": sum(
                dimensions[dimension] * weight
                for dimension, weight in weights.items()
            ),
        }
    ranking = [mapping[label] for label in parsed["ranking"]]
    return {
        "scores": system_scores,
        "ranking": ranking,
        "reason": parsed.get("reason", ""),
        "raw": judgment["raw"],
        "anonymous_mapping": mapping,
    }


def judge_records(records, kind, client, output_path, failures_path, seed):
    weights = SINGLE_WEIGHTS if kind == "single" else MULTI_WEIGHTS
    pending = [
        sample_id for sample_id in sorted(records)
        if not records[sample_id].get("judge")
        and not any(
            records[sample_id]["systems"].get(system, {}).get("error")
            for system in SYSTEMS
        )
    ]
    total = len(pending)
    if not total:
        return
    started = time.monotonic()
    print(f"[eval] Starting {kind} judge: {total} samples", flush=True)
    for completed, sample_id in enumerate(pending, 1):
        record = records[sample_id]
        mapping, answers = anonymize_systems(sample_id, record["systems"], seed)
        context = (
            record.get("context_with_card", [])
            if kind == "single"
            else record["prompts"]
        )
        prompt = judge_schema_prompt(
            kind, record["character_card"], context, answers
        )
        try:
            judgment = client.evaluate(prompt, weights)
            record["judge"] = map_judgment_to_systems(judgment, mapping, weights)
            if stable_int(sample_id, seed) % 10 == 0:
                second_mapping, second_answers = anonymize_systems(
                    sample_id, record["systems"], seed, alternate=True
                )
                second_prompt = judge_schema_prompt(
                    kind, record["character_card"], context, second_answers
                )
                second = map_judgment_to_systems(
                    client.evaluate(second_prompt, weights),
                    second_mapping,
                    weights,
                )
                record["judge_consistency"] = {
                    "ranking": second["ranking"],
                    "top_match": second["ranking"][0] == record["judge"]["ranking"][0],
                    "raw": second["raw"],
                    "anonymous_mapping": second["anonymous_mapping"],
                }
            remove_failure(failures_path, sample_id)
            write_jsonl(output_path, records)
        except Exception as error:
            append_failure(failures_path, {
                "sample_id": sample_id,
                "kind": kind,
                "error": f"{type(error).__name__}: {error}",
            })
        progress_line(f"{kind} judge", completed, total, started)


def metric_stats(values, seed=42):
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "std": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(clean, size=len(clean), replace=True)))
        for _ in range(1000)
    ]
    return {
        "count": len(clean),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std": float(np.std(clean)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def paired_difference(records, left, right, metric_getter, seed):
    differences = []
    for record in records.values():
        left_value = metric_getter(record, left)
        right_value = metric_getter(record, right)
        if left_value is not None and right_value is not None:
            differences.append(float(left_value) - float(right_value))
    return metric_stats(differences, seed)


def judge_metric(record, system, dimension):
    judge = record.get("judge") or {}
    return (judge.get("scores") or {}).get(system, {}).get(dimension)


def auto_metric(record, system, metric):
    result = record.get("systems", {}).get(system, {})
    if metric == "assistant_perplexity":
        return (result.get("assistant_perplexity") or {}).get("perplexity")
    return (result.get("automatic") or {}).get(metric, result.get(metric))


def pairwise_win_rates(records):
    pairs = (
        ("lora_with_card", "base_with_card"),
        ("base_with_card", "base_no_card"),
        ("lora_with_card", "base_no_card"),
    )
    output = {}
    for left, right in pairs:
        counts = Counter()
        for record in records.values():
            ranking = (record.get("judge") or {}).get("ranking")
            if not ranking:
                continue
            counts["left_win" if ranking.index(left) < ranking.index(right) else "right_win"] += 1
        total = counts["left_win"] + counts["right_win"]
        output[f"{left}_vs_{right}"] = {
            "left_win_rate": counts["left_win"] / total if total else None,
            "tie_rate": 0.0 if total else None,
            "right_win_rate": counts["right_win"] / total if total else None,
            "count": total,
        }
    return output


def summarize(single_records, multi_records, excluded_count, failure_count, seed):
    summary = {
        "counts": {
            "single_turn": len(single_records),
            "multi_turn": len(multi_records),
            "excluded": excluded_count,
            "judge_failures": failure_count,
        },
        "systems": {},
        "paired_differences": {},
        "single_turn_win_rates": pairwise_win_rates(single_records),
        "multi_turn_win_rates": pairwise_win_rates(multi_records),
    }
    for system in SYSTEMS:
        summary["systems"][system] = {
            "automatic": {
                metric: metric_stats(
                    [auto_metric(record, system, metric) for record in single_records.values()],
                    seed,
                )
                for metric in (
                    "assistant_perplexity",
                    "distinct_1",
                    "distinct_2",
                    "repetition_rate",
                    "response_chars",
                    "latency_seconds",
                    "tokens_per_second",
                    "peak_gpu_memory_mb",
                )
            },
            "single_judge": {
                dimension: metric_stats(
                    [judge_metric(record, system, dimension) for record in single_records.values()],
                    seed,
                )
                for dimension in (*SINGLE_WEIGHTS, "weighted_total")
            },
            "multi_judge": {
                dimension: metric_stats(
                    [judge_metric(record, system, dimension) for record in multi_records.values()],
                    seed,
                )
                for dimension in (*MULTI_WEIGHTS, "weighted_total")
            },
        }
        refusal_values = [
            auto_metric(record, system, "empty_or_refusal")
            for record in single_records.values()
            if auto_metric(record, system, "empty_or_refusal") is not None
        ]
        summary["systems"][system]["automatic"]["empty_or_refusal_rate"] = (
            sum(bool(value) for value in refusal_values) / len(refusal_values)
            if refusal_values else None
        )

    for left, right in (
        ("lora_with_card", "base_with_card"),
        ("base_with_card", "base_no_card"),
        ("lora_with_card", "base_no_card"),
    ):
        key = f"{left}_minus_{right}"
        summary["paired_differences"][key] = {
            "single_weighted_total": paired_difference(
                single_records, left, right,
                lambda record, system: judge_metric(record, system, "weighted_total"),
                seed,
            ),
            "multi_weighted_total": paired_difference(
                multi_records, left, right,
                lambda record, system: judge_metric(record, system, "weighted_total"),
                seed,
            ),
            "assistant_perplexity": paired_difference(
                single_records, left, right,
                lambda record, system: auto_metric(record, system, "assistant_perplexity"),
                seed,
            ),
        }

    consistency = [
        record["judge_consistency"]["top_match"]
        for records in (single_records, multi_records)
        for record in records.values()
        if record.get("judge_consistency")
    ]
    summary["judge_consistency_rate"] = (
        sum(consistency) / len(consistency) if consistency else None
    )
    judged = sum(
        bool(record.get("judge"))
        for records in (single_records, multi_records)
        for record in records.values()
    )
    total = len(single_records) + len(multi_records)
    summary["judge_success_rate"] = judged / total if total else None
    return summary


def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_report(summary, manifest, single_records):
    rows = []
    single_win = summary["single_turn_win_rates"]
    for system in SYSTEMS:
        data = summary["systems"][system]
        automatic = data["automatic"]
        judge = data["single_judge"]
        multi = data["multi_judge"]
        if system == "lora_with_card":
            win = single_win.get("lora_with_card_vs_base_with_card", {}).get("left_win_rate")
        elif system == "base_with_card":
            win = single_win.get("base_with_card_vs_base_no_card", {}).get("left_win_rate")
        else:
            win = None
        rows.append(
            f"| {SYSTEM_LABELS[system]} | "
            f"{fmt(automatic['assistant_perplexity']['mean'])} | "
            f"{fmt(judge['role_identity']['mean'])} | "
            f"{fmt(judge['style']['mean'])} | "
            f"{fmt(multi['memory']['mean'])} | {fmt(win)} |"
        )

    examples = []
    judged_records = [
        record for record in single_records.values() if record.get("judge")
    ]
    judged_records.sort(
        key=lambda record: (
            judge_metric(record, "lora_with_card", "weighted_total") or 0
        ) - (
            judge_metric(record, "base_with_card", "weighted_total") or 0
        ),
        reverse=True,
    )
    for title, candidates in (
        ("Representative success", judged_records[:1]),
        ("Representative failure", judged_records[-1:]),
    ):
        if candidates:
            record = candidates[0]
            examples.append(
                f"### {title}\n\n"
                f"- Character: {record['character_name']}\n"
                f"- User: {record['context_with_card'][-1]['content'][:300]}\n"
                f"- Base + card: {record['systems']['base_with_card']['response'][:500]}\n"
                f"- LoRA + card: {record['systems']['lora_with_card']['response'][:500]}\n"
                f"- Judge: {record['judge']['reason']}\n"
            )

    pair_rows = []
    for pair, values in single_win.items():
        pair_rows.append(
            f"| {pair} | {fmt(values['left_win_rate'])} | "
            f"{fmt(values['tie_rate'])} | "
            f"{fmt(values['right_win_rate'])} | {values['count']} |"
        )

    automatic_rows = []
    single_judge_rows = []
    multi_judge_rows = []
    for system in SYSTEMS:
        data = summary["systems"][system]
        automatic = data["automatic"]
        single_judge = data["single_judge"]
        multi_judge = data["multi_judge"]
        automatic_rows.append(
            f"| {SYSTEM_LABELS[system]} | "
            f"{fmt(automatic['assistant_perplexity']['mean'])} | "
            f"{fmt(automatic['distinct_1']['mean'])} | "
            f"{fmt(automatic['distinct_2']['mean'])} | "
            f"{fmt(automatic['repetition_rate']['mean'])} | "
            f"{fmt(automatic['empty_or_refusal_rate'])} | "
            f"{fmt(automatic['tokens_per_second']['mean'])} |"
        )
        single_judge_rows.append(
            f"| {SYSTEM_LABELS[system]} | "
            f"{fmt(single_judge['role_identity']['mean'])} | "
            f"{fmt(single_judge['style']['mean'])} | "
            f"{fmt(single_judge['relevance']['mean'])} | "
            f"{fmt(single_judge['naturalness']['mean'])} | "
            f"{fmt(single_judge['immersion']['mean'])} | "
            f"{fmt(single_judge['weighted_total']['mean'])} |"
        )
        multi_judge_rows.append(
            f"| {SYSTEM_LABELS[system]} | "
            f"{fmt(multi_judge['role_identity']['mean'])} | "
            f"{fmt(multi_judge['memory']['mean'])} | "
            f"{fmt(multi_judge['coherence']['mean'])} | "
            f"{fmt(multi_judge['style']['mean'])} | "
            f"{fmt(multi_judge['immersion']['mean'])} | "
            f"{fmt(multi_judge['weighted_total']['mean'])} |"
        )

    reuse_note = ""
    if manifest.get("reuse_baseline"):
        source = manifest["reuse_baseline"]["source"]
        reuse_note = (
            f"- Reused baseline: `{source}` "
            "(baseline latency, throughput and GPU memory excluded)\n"
        )

    return f"""# Role-Play LoRA Evaluation Report

## Experiment

- Base model: `{manifest['base_model']}`
- Adapter: `{manifest['adapter']}`
{reuse_note}- Seed: {manifest['seed']}
- Single-turn samples: {summary['counts']['single_turn']}
- Multi-turn samples: {summary['counts']['multi_turn']}
- Excluded samples: {summary['counts']['excluded']}
- Judge model: `{manifest.get('judge_model') or 'disabled'}`
- Judge success rate: {fmt(summary['judge_success_rate'])}
- Judge order consistency: {fmt(summary['judge_consistency_rate'])}

## Core Results

| System | PPL ↓ | Fidelity ↑ | Style ↑ | Memory ↑ | Win Rate ↑ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Automatic Metrics

| System | PPL ↓ | Distinct-1 ↑ | Distinct-2 ↑ | Repetition ↓ | Refusal ↓ | Tokens/s ↑ |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(automatic_rows)}

## Single-Turn Judge Scores

| System | Identity | Style | Relevance | Naturalness | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(single_judge_rows)}

## Multi-Turn Challenge

| System | Identity | Memory | Coherence | Style | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(multi_judge_rows)}

## Pairwise Single-Turn Ranking

| Comparison | Left Win Rate | Tie Rate | Right Win Rate | Samples |
|---|---:|---:|---:|---:|
{chr(10).join(pair_rows)}

## Interpretation

The primary comparison is `LoRA + card` versus `Base + card`. The
`Base + card` versus `Base, no card` comparison estimates the contribution of
the character prompt alone. Confidence intervals and paired differences are
available in `summary.json`.

## Cases

{chr(10).join(examples) if examples else "No judged examples are available."}

## Filtering And Failures

- Safety and quality exclusions: {summary['counts']['excluded']}
- Judge API failures: {summary['counts']['judge_failures']}
- Full excluded records: `excluded_samples.jsonl`
- Full failed judge calls: `judge_failures.jsonl`
"""


def prepare_output(args, adapter_path):
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "single": output_dir / "single_turn_samples.jsonl",
        "multi": output_dir / "multi_turn_samples.jsonl",
        "excluded": output_dir / "excluded_samples.jsonl",
        "failures": output_dir / "judge_failures.jsonl",
        "manifest": output_dir / "manifest.json",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
    }
    if args.resume and paths["single"].exists() and paths["multi"].exists():
        existing_manifest = load_json(paths["manifest"])
        return (
            paths,
            load_jsonl(paths["single"]),
            load_jsonl(paths["multi"]),
            existing_manifest.get("reuse_baseline"),
        )

    _, _, test_data = load_local_dataset(str(resolve_path(args.dataset)))
    if not test_data:
        raise RuntimeError("Test dataset is empty")
    rules = load_safety_rules(resolve_path(args.safety_rules))
    single, multi, excluded = select_safe_samples(
        test_data,
        rules,
        args.single_samples,
        args.multi_samples,
        args.seed,
    )
    single_records = {record["sample_id"]: record for record in single}
    multi_records = {record["sample_id"]: record for record in multi}
    reuse_info = None
    if getattr(args, "reuse_baseline", None):
        reuse_info = reuse_baseline(
            args, adapter_path, single_records, multi_records
        )
    write_jsonl(paths["single"], single_records)
    write_jsonl(paths["multi"], multi_records)
    write_jsonl(paths["excluded"], excluded)
    return paths, single_records, multi_records, reuse_info


def compare(args):
    if not args.base_model:
        raise RuntimeError(
            "Set --base_model or MODEL_DIR to the base model directory or model ID"
        )
    if not args.adapter:
        raise RuntimeError("Set --adapter or ADAPTER_DIR to the LoRA adapter directory")
    adapter_source = resolve_path(args.adapter)
    if not adapter_source.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_source}")
    paths, single_records, multi_records, reuse_info = prepare_output(
        args, adapter_source
    )
    base_source = resolve_model_source(args.base_model)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "safety_rules": args.safety_rules,
        "seed": args.seed,
        "single_samples": args.single_samples,
        "multi_samples": args.multi_samples,
        "max_new_tokens": args.max_new_tokens,
        "generation": {"do_sample": False},
        "judge_model": os.getenv("JUDGE_MODEL") if not args.skip_judge else None,
        "systems": SYSTEM_LABELS,
    }
    if reuse_info:
        manifest["reuse_baseline"] = reuse_info
    atomic_write_json(paths["manifest"], manifest)

    missing_systems = {
        system
        for system in SYSTEMS
        if any(system not in record["systems"] for record in single_records.values())
        or any(system not in record["systems"] for record in multi_records.values())
    }
    if missing_systems:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to generate missing evaluation answers")
        base_model, tokenizer = load_base_model(base_source)
        for system in BASELINE_SYSTEMS:
            if system not in missing_systems:
                continue
            run_single_system(
                base_model, tokenizer, single_records, system,
                paths["single"], args.max_new_tokens,
            )
            run_multi_system(
                base_model, tokenizer, multi_records, system,
                paths["multi"], args.max_new_tokens,
            )

        if "lora_with_card" in missing_systems:
            lora_model = PeftModel.from_pretrained(base_model, str(adapter_source))
            lora_model.eval()
            run_single_system(
                lora_model, tokenizer, single_records, "lora_with_card",
                paths["single"], args.max_new_tokens,
            )
            run_multi_system(
                lora_model, tokenizer, multi_records, "lora_with_card",
                paths["multi"], args.max_new_tokens,
            )

    if not args.skip_judge:
        api_key = os.getenv("JUDGE_API_KEY")
        base_url = os.getenv("JUDGE_BASE_URL")
        judge_model = os.getenv("JUDGE_MODEL")
        if not all((api_key, base_url, judge_model)):
            raise RuntimeError(
                "Set JUDGE_API_KEY, JUDGE_BASE_URL and JUDGE_MODEL, "
                "or pass --skip_judge"
            )
        # 历史教训：环境变量里被塞进了 'echo\rexport JUDGE_API_KEY' 这种
        # shell 残片，会让 Authorization header 出现非法 \r 字符。这里
        # 显式 strip 一次并校验前缀，避免 120 次调用全部失败。
        api_key = api_key.replace("\r", "").replace("\n", "").strip()
        if not api_key.startswith("ey") and not api_key.startswith("sk-"):
            raise RuntimeError(
                "JUDGE_API_KEY 看起来不对（既不以 'sk-' 也不以 'ey' 开头）："
                f" {api_key[:20]!r}…"
            )
        client = JudgeClient(base_url, api_key, judge_model)
        judge_records(
            single_records, "single", client, paths["single"],
            paths["failures"], args.seed,
        )
        judge_records(
            multi_records, "multi", client, paths["multi"],
            paths["failures"], args.seed,
        )

    failure_count = len(load_jsonl(paths["failures"]))
    excluded_count = len(load_jsonl(paths["excluded"]))
    summary = summarize(
        single_records, multi_records, excluded_count, failure_count, args.seed
    )
    atomic_write_json(paths["summary"], summary)
    paths["report"].write_text(
        render_report(summary, manifest, single_records),
        encoding="utf-8",
    )
    print(f"Evaluation outputs written to: {resolve_path(args.output_dir)}")


def build_parser():
    parser = argparse.ArgumentParser(description="Role-play LoRA evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare", help="Run three-way ablation")
    compare_parser.add_argument(
        "--base_model",
        default=os.getenv("MODEL_DIR"),
        help="Base model directory or model ID; defaults to MODEL_DIR",
    )
    compare_parser.add_argument(
        "--adapter",
        default=os.getenv("ADAPTER_DIR"),
        help="LoRA adapter directory; defaults to ADAPTER_DIR",
    )
    compare_parser.add_argument("--dataset", default="processed")
    compare_parser.add_argument("--output_dir", required=True)
    compare_parser.add_argument(
        "--safety_rules",
        default="configs/eval_safety_terms.json",
    )
    compare_parser.add_argument("--single_samples", type=int, default=100)
    compare_parser.add_argument("--multi_samples", type=int, default=20)
    compare_parser.add_argument("--max_new_tokens", type=int, default=256)
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument(
        "--reuse_baseline",
        help="Reuse compatible base outputs from an evaluation directory",
    )
    compare_parser.add_argument("--resume", action="store_true")
    compare_parser.add_argument("--skip_judge", action="store_true")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    if parsed_args.command == "compare":
        compare(parsed_args)
