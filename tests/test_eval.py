import json
from types import SimpleNamespace

import pytest

from scripts.eval import (
    JudgeClient,
    assistant_perplexity,
    anonymize_systems,
    compare,
    map_judgment_to_systems,
    metric_stats,
    render_report,
    run_multi_system,
    safety_matches,
    select_safe_samples,
    summarize,
    validate_judgment,
    write_jsonl,
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
    )
    compare(args)
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
