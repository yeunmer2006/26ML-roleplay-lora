import json
from pathlib import Path

from scripts.resource_manager import (
    find_complete_model,
    prepare_data,
    resolve_model,
    validate_data_dir,
    validate_model_dir,
)


def make_model(path, complete=True):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    if not complete:
        return path
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    index = {"weight_map": {"model.layer": "model-00001-of-00001.safetensors"}}
    (path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return path


def make_data(path):
    path.mkdir(parents=True)
    row = {
        "id": "sample",
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi"},
        ],
    }
    for split in ("train", "val", "test"):
        (path / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )
    return path


def test_model_validation_requires_tokenizer_and_all_weight_shards(tmp_path):
    model = make_model(tmp_path / "model")
    valid, _ = validate_model_dir(model)
    assert valid

    (model / "model-00001-of-00001.safetensors").unlink()
    valid, reason = validate_model_dir(model)
    assert not valid
    assert "缺少权重分片" in reason


def test_incomplete_explicit_model_falls_back_to_project_cache(tmp_path):
    project = tmp_path / "project"
    incomplete = make_model(tmp_path / "incomplete", complete=False)
    cached = make_model(
        project / ".cache" / "modelscope" / "Qwen2.5-3B-Instruct"
    )

    resolved, checked = find_complete_model(
        project,
        explicit=str(incomplete),
        home=tmp_path / "home",
    )

    assert resolved == cached.resolve()
    assert checked[0][0] == incomplete.resolve()
    assert not checked[0][1]


def test_complete_huggingface_snapshot_used_after_project_candidates(tmp_path):
    project = tmp_path / "project"
    make_model(
        project / ".cache" / "huggingface" / "Qwen2.5-3B-Instruct",
        complete=False,
    )
    home = tmp_path / "home"
    repo = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen2.5-3B-Instruct"
    )
    revision = "revision-1"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    snapshot = make_model(repo / "snapshots" / revision)

    resolved, _ = find_complete_model(project, home=home)

    assert resolved == snapshot.resolve()


def test_model_download_falls_back_to_modelscope(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()

    def fail_huggingface(model_id, target):
        raise RuntimeError("simulated Hugging Face failure")

    def complete_modelscope(model_id, target):
        make_model(target)

    monkeypatch.setattr(
        "scripts.resource_manager._download_huggingface",
        fail_huggingface,
    )
    monkeypatch.setattr(
        "scripts.resource_manager._download_modelscope",
        complete_modelscope,
    )

    resolved = resolve_model(project, download=True, home=tmp_path / "home")

    assert resolved == (
        project / ".cache" / "modelscope" / "Qwen2.5-3B-Instruct"
    ).resolve()


def test_valid_local_data_is_reused_without_download(tmp_path, monkeypatch):
    project = tmp_path / "project"
    data = make_data(project / "processed")

    def fail_run(*args, **kwargs):
        raise AssertionError("本地数据完整时不应调用下载")

    monkeypatch.setattr("scripts.resource_manager.subprocess.run", fail_run)
    resolved = prepare_data(project, "processed", download=True)

    assert resolved == data.resolve()
    assert (data / "dataset_manifest.json").is_file()


def test_missing_data_is_downloaded_to_staging_then_installed(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)

    def fake_run(command, cwd, check):
        output_dir = Path(command[command.index("--output_dir") + 1])
        make_data(output_dir)

    monkeypatch.setattr("scripts.resource_manager.subprocess.run", fake_run)

    resolved = prepare_data(project, "processed", download=True)

    assert resolved == (project / "processed").resolve()
    valid, _, rows = validate_data_dir(resolved)
    assert valid
    assert rows == {"train": 1, "val": 1, "test": 1}


def test_invalid_jsonl_is_rejected(tmp_path):
    data = make_data(tmp_path / "processed")
    (data / "val.jsonl").write_text("{invalid\n", encoding="utf-8")

    valid, reason, _ = validate_data_dir(data)

    assert not valid
    assert "val.jsonl" in reason
