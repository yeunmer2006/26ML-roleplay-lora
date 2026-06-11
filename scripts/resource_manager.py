#!/usr/bin/env python3
"""Resolve and validate local training model and dataset resources."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATASET_ID = "KaraKaraWitch/PIPPA-ShareGPT-formatted"
REQUIRED_DATA_SPLITS = ("train", "val", "test")


def log(message):
    print(f"[resources] {message}", file=sys.stderr)


def resolve_project_path(value, project_root=PROJECT_ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _tokenizer_files_present(path):
    if not (path / "tokenizer_config.json").is_file():
        return False
    return (path / "tokenizer.json").is_file() or (
        (path / "vocab.json").is_file() and (path / "merges.txt").is_file()
    )


def validate_model_dir(model_dir):
    """Return (valid, reason) after checking config, tokenizer and all weights."""
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        return False, "目录不存在"
    if not (path / "config.json").is_file():
        return False, "缺少 config.json"
    if not _tokenizer_files_present(path):
        return False, "缺少 tokenizer_config.json 或 tokenizer 文件"

    index_files = (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    )
    for index_path in index_files:
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = sorted(set(index.get("weight_map", {}).values()))
        except (OSError, json.JSONDecodeError) as error:
            return False, f"权重索引不可读: {error}"
        if not weight_files:
            return False, f"{index_path.name} 未列出权重分片"
        missing = [
            name
            for name in weight_files
            if not (path / name).is_file() or (path / name).stat().st_size == 0
        ]
        if missing:
            preview = ", ".join(missing[:3])
            return False, f"缺少权重分片: {preview}"
        return True, f"完整分片模型，共 {len(weight_files)} 个权重文件"

    single_weights = (
        path / "model.safetensors",
        path / "pytorch_model.bin",
    )
    if any(item.is_file() and item.stat().st_size > 0 for item in single_weights):
        return True, "完整单文件模型"
    return False, "缺少模型权重或权重索引"


def _deduplicate_paths(paths):
    seen = set()
    result = []
    for path in paths:
        try:
            normalized = Path(path).expanduser().resolve()
        except OSError:
            continue
        key = str(normalized)
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _project_model_candidates(project_root, model_name):
    candidates = [
        project_root / "models" / model_name,
        project_root / "models" / "Qwen" / model_name,
        project_root / ".cache" / "huggingface" / model_name,
        project_root / ".cache" / "modelscope" / model_name,
    ]
    for root in (project_root / "models", project_root / ".cache"):
        if root.is_dir():
            candidates.extend(
                config.parent
                for config in root.rglob("config.json")
                if model_name.lower() in str(config.parent).lower()
            )
    return candidates


def _huggingface_cache_candidates(home, model_id):
    repo_dir = home / ".cache" / "huggingface" / "hub" / (
        "models--" + model_id.replace("/", "--")
    )
    candidates = []
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(repo_dir / "snapshots" / revision)
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        candidates.extend(
            sorted(
                (item for item in snapshots.iterdir() if item.is_dir()),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        )
    return candidates


def _modelscope_cache_candidates(home, model_id, model_name):
    roots = [
        home / ".cache" / "modelscope",
        home / ".cache" / "modelscope" / "hub",
        home / ".cache" / "modelscope" / "hub" / "models",
    ]
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates.extend(
            config.parent
            for config in root.rglob("config.json")
            if model_name.lower() in str(config.parent).lower()
            or model_id.lower() in str(config.parent).lower()
        )
    return candidates


def model_candidates(project_root, model_id=DEFAULT_MODEL_ID, explicit=None, home=None):
    model_name = model_id.rsplit("/", 1)[-1]
    user_home = Path(home).expanduser() if home else Path.home()
    candidates = []
    if explicit:
        candidates.append(resolve_project_path(explicit, project_root))
    candidates.extend(_project_model_candidates(project_root, model_name))
    candidates.extend(_huggingface_cache_candidates(user_home, model_id))
    candidates.extend(
        _modelscope_cache_candidates(user_home, model_id, model_name)
    )
    return _deduplicate_paths(candidates)


def find_complete_model(project_root, model_id=DEFAULT_MODEL_ID, explicit=None, home=None):
    checked = []
    for candidate in model_candidates(project_root, model_id, explicit, home):
        valid, reason = validate_model_dir(candidate)
        checked.append((candidate, valid, reason))
        if valid:
            return candidate, checked
    return None, checked


def _download_huggingface(model_id, target):
    from huggingface_hub import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(target))


def _download_modelscope(model_id, target):
    from modelscope import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id,
        cache_dir=str(target.parent),
        local_dir=str(target),
    )


def resolve_model(
    project_root,
    model_id=DEFAULT_MODEL_ID,
    explicit=None,
    download=False,
    home=None,
):
    model_path, checked = find_complete_model(
        project_root,
        model_id=model_id,
        explicit=explicit,
        home=home,
    )
    for candidate, valid, reason in checked:
        status = "可用" if valid else "跳过"
        log(f"{status}: {candidate} ({reason})")
    if model_path:
        log(f"复用本地模型: {model_path}")
        return model_path
    if not download:
        raise RuntimeError("没有找到完整模型，请先运行 scripts/prepare_training.sh")

    model_name = model_id.rsplit("/", 1)[-1]
    hf_target = project_root / ".cache" / "huggingface" / model_name
    log(f"本地无完整模型，从 Hugging Face 下载: {model_id}")
    try:
        _download_huggingface(model_id, hf_target)
        valid, reason = validate_model_dir(hf_target)
        if not valid:
            raise RuntimeError(f"下载后模型不完整: {reason}")
        log(f"Hugging Face 下载完成: {hf_target}")
        return hf_target.resolve()
    except Exception as error:
        log(f"Hugging Face 下载失败，将回退 ModelScope: {error}")

    modelscope_target = project_root / ".cache" / "modelscope" / model_name
    _download_modelscope(model_id, modelscope_target)
    valid, reason = validate_model_dir(modelscope_target)
    if not valid:
        raise RuntimeError(f"ModelScope 下载后模型不完整: {reason}")
    log(f"ModelScope 下载完成: {modelscope_target}")
    return modelscope_target.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data_dir(data_dir):
    """Parse all required JSONL files and return (valid, reason, split rows)."""
    path = Path(data_dir).expanduser()
    rows = {}
    if not path.is_dir():
        return False, "目录不存在", rows
    for split in REQUIRED_DATA_SPLITS:
        split_path = path / f"{split}.jsonl"
        if not split_path.is_file():
            return False, f"缺少 {split_path.name}", rows
        count = 0
        line_number = 0
        try:
            with split_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("记录不是 JSON 对象")
                    if not isinstance(item.get("conversations"), list):
                        raise ValueError("缺少 conversations 列表")
                    count += 1
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            return False, f"{split_path.name} 第 {line_number} 行无效: {error}", rows
        if count == 0:
            return False, f"{split_path.name} 为空", rows
        rows[split] = count
    return True, "三个 split 均完整且可解析", rows


def write_dataset_manifest(data_dir, rows):
    data_path = Path(data_dir)
    manifest = {"format": "PIPPA-ShareGPT", "splits": {}}
    for split in REQUIRED_DATA_SPLITS:
        split_path = data_path / f"{split}.jsonl"
        manifest["splits"][split] = {
            "rows": rows[split],
            "sha256": _sha256(split_path),
        }
    (data_path / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_data(
    project_root,
    data_dir,
    dataset_id=DEFAULT_DATASET_ID,
    download=False,
):
    target = resolve_project_path(data_dir, project_root)
    valid, reason, rows = validate_data_dir(target)
    if valid:
        if not (target / "dataset_manifest.json").is_file():
            write_dataset_manifest(target, rows)
        log(f"复用本地数据: {target} ({reason}; {rows})")
        return target
    log(f"本地数据不可用: {target} ({reason})")
    if not download:
        raise RuntimeError("没有找到完整数据，请先运行 scripts/prepare_training.sh")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}-download-",
        dir=target.parent,
    ) as temporary:
        staging = Path(temporary) / target.name
        command = [
            sys.executable,
            str(project_root / "scripts" / "data_loader.py"),
            "--dataset",
            dataset_id,
            "--output_dir",
            str(staging),
        ]
        log(f"从 Hugging Face 下载并清洗数据: {dataset_id}")
        subprocess.run(command, cwd=project_root, check=True)
        valid, reason, rows = validate_data_dir(staging)
        if not valid:
            raise RuntimeError(f"下载后的数据无效: {reason}")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staging), str(target))
    log(f"数据准备完成: {target} ({rows})")
    return target


def write_resource_manifest(project_root, model_path, data_path, model_id, dataset_id):
    cache_dir = project_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def portable(path):
        try:
            return str(Path(path).resolve().relative_to(project_root))
        except ValueError:
            return str(Path(path).resolve())

    manifest = {
        "model_id": model_id,
        "model_path": portable(model_path),
        "dataset_id": dataset_id,
        "data_path": portable(data_path),
    }
    (cache_dir / "training_resources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_parser = subparsers.add_parser("resolve-model")
    model_parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    model_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    model_parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR"))
    model_parser.add_argument("--download", action="store_true")

    data_parser = subparsers.add_parser("prepare-data")
    data_parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    data_parser.add_argument("--data-dir", default="processed")
    data_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    data_parser.add_argument("--download", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    prepare_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    prepare_parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR"))
    prepare_parser.add_argument("--data-dir", default="processed")
    prepare_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    try:
        if args.command == "resolve-model":
            path = resolve_model(
                project_root,
                model_id=args.model_id,
                explicit=args.model_dir,
                download=args.download,
            )
        elif args.command == "prepare-data":
            path = prepare_data(
                project_root,
                args.data_dir,
                dataset_id=args.dataset_id,
                download=args.download,
            )
        else:
            model_path = resolve_model(
                project_root,
                model_id=args.model_id,
                explicit=args.model_dir,
                download=True,
            )
            data_path = prepare_data(
                project_root,
                args.data_dir,
                dataset_id=args.dataset_id,
                download=True,
            )
            write_resource_manifest(
                project_root,
                model_path,
                data_path,
                args.model_id,
                args.dataset_id,
            )
            print(json.dumps({
                "model_path": str(model_path),
                "data_path": str(data_path),
            }))
            return 0
        print(path)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[resources] 错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
