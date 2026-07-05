import json

from scripts.clean_training_data import clean_split, main


class Args:
    max_repetition = 0.35
    ngram_size = 4
    min_ngrams = 20
    sentence_repeat_limit = 3


def sample(reply):
    return {
        "id": reply[:8],
        "bot": {"name": "Role", "description": "Character"},
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": reply},
        ],
    }


def test_clean_split_removes_repeated_sentences():
    records = [
        sample("A normal response that moves the scene forward."),
        sample("I will obey you. I will obey you. I will obey you."),
    ]

    kept, reasons = clean_split(records, Args())

    assert kept == records[:1]
    assert reasons["repeated_sentence"] == 1


def test_cleaner_copies_test_split_without_changes(tmp_path, monkeypatch):
    source = tmp_path / "processed"
    target = tmp_path / "processed_clean"
    source.mkdir()
    valid = sample("A normal response.")
    for split in ("train", "val", "test"):
        (source / f"{split}.jsonl").write_text(
            json.dumps(valid, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    original_test = (source / "test.jsonl").read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "clean_training_data.py",
            "--input_dir",
            str(source),
            "--output_dir",
            str(target),
        ],
    )

    main()

    assert (target / "test.jsonl").read_bytes() == original_test
    manifest = json.loads((target / "cleaning_manifest.json").read_text())
    assert manifest["splits"]["test"]["copied_without_changes"] is True
