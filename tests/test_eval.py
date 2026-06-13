import json
from types import SimpleNamespace

import pytest

from scripts.eval import (
    JudgeClient,
    assistant_perplexity,
    anonymize_systems,
    import_baseline_records,
    compare,
    map_judgment_to_systems,
    metric_stats,
    render_report,
    run_multi_system,
    safety_matches,
    select_safe_samples,
    summarize,
    validate_reuse_manifest,
    validate_judgment,
    write_jsonl,
    build_parser,
)


def sample_item(identifier, name="Character", description="A calm scholar."):
    return {
        "id": identifier,
        "bot": {"name": name, "description": description},
        "conversations": [
            {"from": "system", "value": "Example dialogue"},
            {"from": "human", "value": "Who are you?"},
            {"from": "gpt", "value": "I am a scholar."},
        ],
    }


def complete_system(response, ppl=2.0):
    return {
        "response": response,
        "assistant_perplexity": {
            "loss": 0.69,
            "perplexity": ppl,
            "supervised_tokens": 4,
        },
        "latency_seconds": 1.0,
        "tokens_per_second": 10.0,
        "peak_gpu_memory_mb": 100.0,
        "automatic": {
            "distinct_1": 1.0,
            "distinct_2": 1.0,
            "repetition_rate": 0.0,
            "response_chars": len(response),
            "empty_or_refusal": False,
        },
        "error": None,
    }


def judged_record(sample_id="one"):
    systems = {
        "base_no_card": complete_system("plain", 4.0),
        "base_with_card": complete_system("card", 3.0),
        "lora_with_card": complete_system("lora", 2.0),
    }
    return {
        "sample_id": sample_id,
        "character_name": "Scholar",
        "character_card": "A calm scholar.",
        "context_with_card": [
            {"role": "system", "content": "A calm scholar."},
            {"role": "user", "content": "Who are you?"},
        ],
        "systems": systems,
        "judge": {
            "scores": {
                "base_no_card": {
                    "role_identity": 2, "style": 2, "relevance": 3,
                    "naturalness": 3, "immersion": 2, "weighted_total": 2.3,
                },
                "base_with_card": {
                    "role_identity": 3, "style": 3, "relevance": 4,
                    "naturalness": 4, "immersion": 3, "weighted_total": 3.35,
                },
                "lora_with_card": {
                    "role_identity": 5, "style": 4, "relevance": 4,
                    "naturalness": 4, "immersion": 5, "weighted_total": 4.45,
                },
            },
            "ranking": ["lora_with_card", "base_with_card", "base_no_card"],
            "reason": "LoRA follows the role best.",
        },
        "judge_consistency": {"top_match": True},
    }


def test_safety_filter_blocks_sensitive_content():
    rules = {"sexual_content": ["sex"], "violence": ["torture"]}
    assert safety_matches(sample_item("x", description="Explicit sex role"), rules)
    assert not safety_matches(sample_item("y"), rules)


def test_sample_selection_is_deterministic_and_unique_by_character():
    items = [
        sample_item("1", "A"),
        sample_item("2", "A"),
        sample_item("3", "B"),
        sample_item("4", "C"),
    ]
    first = select_safe_samples(items, {}, 2, 1, 42)
    second = select_safe_samples(items, {}, 2, 1, 42)
    assert [record["sample_id"] for record in first[0]] == [
        record["sample_id"] for record in second[0]
    ]
    assert len({record["character_id"] for record in first[0]}) == 2


def test_anonymization_is_stable_and_reversible():
    systems = {
        "base_no_card": {"response": "one"},
        "base_with_card": {"response": "two"},
        "lora_with_card": {"response": "three"},
    }
    mapping, answers = anonymize_systems("sample", systems, 42)
    mapping_again, answers_again = anonymize_systems("sample", systems, 42)
    assert mapping == mapping_again
    assert answers == answers_again
    assert set(mapping.values()) == set(systems)


def test_judgment_validation_and_mapping():
    value = {
        "scores": {
            label: {
                "role_identity": score,
                "style": score,
                "relevance": score,
                "naturalness": score,
                "immersion": score,
            }
            for label, score in zip(("A", "B", "C"), (1, 3, 5))
        },
        "ranking": ["C", "B", "A"],
        "reason": "C is best.",
    }
    validated = validate_judgment(
        value,
        {
            "role_identity": 0.35, "style": 0.2, "relevance": 0.2,
            "naturalness": 0.15, "immersion": 0.1,
        },
    )
    mapped = map_judgment_to_systems(
        {"parsed": validated, "raw": json.dumps(value)},
        {
            "A": "base_no_card",
            "B": "base_with_card",
            "C": "lora_with_card",
        },
        {
            "role_identity": 0.35, "style": 0.2, "relevance": 0.2,
            "naturalness": 0.15, "immersion": 0.1,
        },
    )
    assert mapped["ranking"][0] == "lora_with_card"
    assert mapped["scores"]["lora_with_card"]["weighted_total"] == 5


def test_judge_client_retries_and_parses(monkeypatch):
    calls = {"count": 0}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            value = {
                "scores": {
                    label: {"role_identity": 4}
                    for label in ("A", "B", "C")
                },
                "ranking": ["A", "B", "C"],
                "reason": "ok",
            }
            return {"choices": [{"message": {"content": json.dumps(value)}}]}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary")
        return Response()

    monkeypatch.setattr("scripts.eval.requests.post", fake_post)
    monkeypatch.setattr("scripts.eval.time.sleep", lambda _: None)
    client = JudgeClient("https://judge.example/v1", "secret", "judge", retries=3)
    result = client.evaluate("prompt", {"role_identity": 1.0})
    assert result["parsed"]["ranking"] == ["A", "B", "C"]
    assert calls["count"] == 2


def test_summary_and_report_use_same_values():
    single = {"one": judged_record()}
    multi_record = judged_record("multi-one")
    multi_record["judge"]["scores"] = {
        system: {
            "role_identity": score["role_identity"],
            "memory": 4,
            "coherence": 4,
            "style": score["style"],
            "immersion": score["immersion"],
            "weighted_total": 4,
        }
        for system, score in multi_record["judge"]["scores"].items()
    }
    multi = {"multi-one": multi_record}
    summary = summarize(single, multi, 2, 0, 42)
    report = render_report(
        summary,
        {
            "base_model": "base",
            "adapter": "adapter",
            "seed": 42,
            "judge_model": "judge",
        },
        single,
    )
    assert summary["systems"]["lora_with_card"]["automatic"][
        "assistant_perplexity"
    ]["mean"] == 2.0
    assert "2.000" in report
    assert "LoRA + card" in report


def test_metric_stats_has_bootstrap_interval():
    result = metric_stats([1, 2, 3], seed=42)
    assert result["count"] == 3
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]


def test_perplexity_masks_prompt_tokens():
    import torch

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("add_generation_prompt"):
                return [1, 2, 3]
            return [1, 2, 3, 4, 5]

    class Output:
        loss = torch.tensor(0.5)

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.labels = None

        def parameters(self):
            return iter([self.weight])

        def __call__(self, **kwargs):
            self.labels = kwargs["labels"].tolist()[0]
            return Output()

    model = Model()
    result = assistant_perplexity(
        model,
        Tokenizer(),
        [{"role": "user", "content": "hello"}],
        "answer",
    )
    assert model.labels == [-100, -100, -100, 4, 5]
    assert result["supervised_tokens"] == 2


def test_perplexity_truncation_preserves_card_and_reference():
    import torch

    class Tokenizer:
        def __init__(self):
            self.seen = []

        def apply_chat_template(
            self,
            messages,
            tokenize=True,
            add_generation_prompt=False,
        ):
            self.seen.append([dict(message) for message in messages])
            ids = []
            for message in messages:
                ids.append({"system": 1, "user": 2, "assistant": 3}[message["role"]])
                ids.extend(ord(char) for char in message["content"])
            if add_generation_prompt:
                ids.append(3)
            return ids

    class Output:
        loss = torch.tensor(0.5)

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.labels = None

        def parameters(self):
            return iter([self.weight])

        def __call__(self, **kwargs):
            self.labels = kwargs["labels"].tolist()[0]
            return Output()

    tokenizer = Tokenizer()
    model = Model()
    context = [
        {"role": "system", "content": "CARD"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old-answer"},
        {"role": "user", "content": "new"},
    ]
    result = assistant_perplexity(
        model,
        tokenizer,
        context,
        "REFERENCE",
        max_length=16,
    )

    scored_messages = tokenizer.seen[-1]
    assert scored_messages[0] == {"role": "system", "content": "CARD"}
    assert scored_messages[-1] == {
        "role": "assistant",
        "content": "REFERENCE",
    }
    assert {"role": "user", "content": "old"} not in scored_messages
    assert result["supervised_tokens"] == len("REFERENCE")


def test_perplexity_supports_reference_without_prompt():
    import torch

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages
            return [1, 2, 3]

    class Output:
        loss = torch.tensor(0.5)

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.labels = None

        def parameters(self):
            return iter([self.weight])

        def __call__(self, **kwargs):
            self.labels = kwargs["labels"].tolist()[0]
            return Output()

    model = Model()
    result = assistant_perplexity(
        model,
        Tokenizer(),
        [],
        "reference",
        max_length=1,
    )

    assert model.labels == [1, 2, 3]
    assert result["supervised_tokens"] == 3


def test_multi_turn_passes_complete_history(monkeypatch, tmp_path):
    observed = []

    def fake_generate(model, tokenizer, messages, max_new_tokens):
        observed.append([dict(message) for message in messages])
        return {
            "response": f"answer-{len(observed)}",
            "generated_tokens": 1,
            "latency_seconds": 1.0,
            "tokens_per_second": 1.0,
            "peak_gpu_memory_mb": 0.0,
            "automatic": {},
        }

    monkeypatch.setattr("scripts.eval.generate_response", fake_generate)
    records = {
        "multi-one": {
            "sample_id": "multi-one",
            "character_card": "A scholar.",
            "prompts": ["one", "two", "three", "four"],
            "systems": {},
        }
    }
    run_multi_system(
        object(), object(), records, "base_with_card",
        tmp_path / "multi.jsonl", 32,
    )
    assert len(observed) == 4
    assert observed[0][0]["role"] == "system"
    assert observed[1][-2:] == [
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "two"},
    ]


def test_resume_with_complete_generations_does_not_load_model(monkeypatch, tmp_path):
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    single = judged_record()
    multi = judged_record("multi-one")
    write_jsonl(output_dir / "single_turn_samples.jsonl", {"one": single})
    write_jsonl(output_dir / "multi_turn_samples.jsonl", {"multi-one": multi})
    write_jsonl(output_dir / "excluded_samples.jsonl", {})

    monkeypatch.setattr(
        "scripts.eval.load_base_model",
        lambda _: (_ for _ in ()).throw(AssertionError("model should not load")),
    )
    monkeypatch.setattr(
        "scripts.eval.reuse_baseline",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("baseline should not be imported again")
        ),
    )
    args = SimpleNamespace(
        output_dir=str(output_dir),
        resume=True,
        dataset="unused",
        safety_rules="configs/eval_safety_terms.json",
        single_samples=1,
        multi_samples=1,
        seed=42,
        base_model="remote/model-id",
        adapter=str(adapter),
        max_new_tokens=16,
        skip_judge=True,
        reuse_baseline="missing-source",
    )
    compare(args)
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()


def test_import_baseline_keeps_quality_metrics_and_removes_performance():
    current = {"one": judged_record()}
    current["one"]["systems"] = {}
    current["one"]["judge"] = None
    current["one"]["judge_consistency"] = None
    source = {"one": judged_record()}

    import_baseline_records(current, source, "single-turn")

    assert set(current["one"]["systems"]) == {
        "base_no_card",
        "base_with_card",
    }
    result = current["one"]["systems"]["base_with_card"]
    assert result["response"] == "card"
    assert result["assistant_perplexity"]["perplexity"] == 3.0
    assert "latency_seconds" not in result
    assert "tokens_per_second" not in result
    assert "peak_gpu_memory_mb" not in result
    assert current["one"]["judge"] is None


def test_import_baseline_rejects_changed_sample_content():
    current = {"one": judged_record()}
    current["one"]["systems"] = {}
    source = {"one": judged_record()}
    source["one"]["character_card"] = "A different character."

    with pytest.raises(RuntimeError, match="sample content differs"):
        import_baseline_records(current, source, "single-turn")


def test_reuse_manifest_accepts_same_model_name_across_machines(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
    }))
    args = SimpleNamespace(
        base_model="/root/models/Qwen2.5-3B-Instruct",
        dataset="processed",
        safety_rules="configs/eval_safety_terms.json",
        seed=42,
        single_samples=100,
        multi_samples=20,
        max_new_tokens=256,
    )
    source = {
        "base_model": "/home/user/models/Qwen2.5-3B-Instruct",
        "dataset": "processed",
        "safety_rules": "configs/eval_safety_terms.json",
        "seed": 42,
        "single_samples": 100,
        "multi_samples": 20,
        "max_new_tokens": 256,
        "generation": {"do_sample": False},
    }

    validate_reuse_manifest(source, args, adapter)


def test_reuse_manifest_rejects_generation_mismatch(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
    }))
    args = SimpleNamespace(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset="processed",
        safety_rules="configs/eval_safety_terms.json",
        seed=42,
        single_samples=100,
        multi_samples=20,
        max_new_tokens=128,
    )
    source = {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "dataset": "processed",
        "safety_rules": "configs/eval_safety_terms.json",
        "seed": 42,
        "single_samples": 100,
        "multi_samples": 20,
        "max_new_tokens": 256,
        "generation": {"do_sample": False},
    }

    with pytest.raises(RuntimeError, match="max_new_tokens"):
        validate_reuse_manifest(source, args, adapter)


def test_eval_model_paths_default_to_environment(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "/models/qwen")
    monkeypatch.setenv("ADAPTER_DIR", "/models/adapter")

    args = build_parser().parse_args([
        "compare",
        "--output_dir",
        "output/evaluations/test",
    ])

    assert args.base_model == "/models/qwen"
    assert args.adapter == "/models/adapter"


def test_eval_explicit_model_paths_override_environment(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "/models/from-env")
    monkeypatch.setenv("ADAPTER_DIR", "/adapter/from-env")

    args = build_parser().parse_args([
        "compare",
        "--base_model",
        "/models/from-cli",
        "--adapter",
        "/adapter/from-cli",
        "--output_dir",
        "output/evaluations/test",
    ])

    assert args.base_model == "/models/from-cli"
    assert args.adapter == "/adapter/from-cli"


def test_eval_parses_reuse_baseline_with_resume():
    args = build_parser().parse_args([
        "compare",
        "--base_model",
        "Qwen/Qwen2.5-3B-Instruct",
        "--adapter",
        "output/experiments/train_3/final_model",
        "--output_dir",
        "output/evaluations/train_3",
        "--reuse_baseline",
        "output/evaluations/train_1",
        "--resume",
    ])

    assert args.reuse_baseline == "output/evaluations/train_1"
    assert args.resume is True
