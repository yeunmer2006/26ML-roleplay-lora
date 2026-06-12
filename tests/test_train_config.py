#!/usr/bin/env python3
"""
Config 对 load_best_model_at_end / metric_for_best_model / greater_is_better 的解析测试。

针对 scripts.train.Config 在解析 best-model 三个字段时的行为做单元测试，
确保 yaml 里的设置能被显式读取并透传给 TrainingArguments。
"""

import sys
from pathlib import Path

import pytest
import yaml
from transformers import TrainingArguments

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train import Config


def _build_base_config(tmp_path, eval_strategy="no", save_strategy="no",
                       load_best_model_at_end=None,
                       metric_for_best_model=None,
                       greater_is_better=None):
    """构造一个最小可用的训练配置字典。"""
    cfg = {
        "model": {
            "name": "Qwen/Qwen2.5-3B-Instruct",
            "max_seq_length": 256,
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        },
        "training": {
            "output_dir": str(tmp_path / "out"),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "eval_strategy": eval_strategy,
            "save_strategy": save_strategy,
        },
        "data": {"dataset_name": "dummy", "seed": 42},
    }
    training = cfg["training"]
    if load_best_model_at_end is not None:
        training["load_best_model_at_end"] = load_best_model_at_end
    if metric_for_best_model is not None:
        training["metric_for_best_model"] = metric_for_best_model
    if greater_is_better is not None:
        training["greater_is_better"] = greater_is_better
    return cfg


@pytest.fixture
def minimal_yaml(tmp_path):
    """不含 best-model 三个字段的最小合法配置。"""
    cfg = _build_base_config(tmp_path)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def explicit_yaml(tmp_path):
    """显式设置三个 best-model 字段，且 eval/save 策略一致以满足 Trainer 校验。"""
    cfg = _build_base_config(
        tmp_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


class TestBestModelConfig:
    """Config 类对 best-model 三个字段的解析行为。"""

    def test_defaults_when_yaml_omits_fields(self, minimal_yaml):
        """yaml 不含这三个字段时，应使用与 TrainingArguments 一致的默认（False / None / None）。"""
        config = Config(str(minimal_yaml))

        assert config.load_best_model_at_end is False
        assert config.metric_for_best_model is None
        assert config.greater_is_better is None

    def test_explicit_values_are_read(self, explicit_yaml):
        """yaml 显式设置后，三个字段应被原样读取。"""
        config = Config(str(explicit_yaml))

        assert config.load_best_model_at_end is True
        assert config.metric_for_best_model == "eval_loss"
        assert config.greater_is_better is False

    def test_lora_config_yaml_has_three_fields_set(self):
        """现有 lora_config.yaml 里的三个字段应被正确读取（这是仓库内唯一显式带这三个字段的配置）。"""
        config = Config(str(project_root / "configs" / "lora_config.yaml"))

        assert config.load_best_model_at_end is True
        assert config.metric_for_best_model == "eval_loss"
        assert config.greater_is_better is False

    def test_train_smoke_yaml_uses_defaults(self):
        """smoke 配置没有这三个字段，应走默认值。"""
        config = Config(str(project_root / "configs" / "train_smoke.yaml"))

        assert config.load_best_model_at_end is False
        assert config.metric_for_best_model is None
        assert config.greater_is_better is None

    def test_values_pass_through_to_training_arguments(self, explicit_yaml, tmp_path):
        """Config 字段应能透传给 TrainingArguments。"""
        config = Config(str(explicit_yaml))
        args = TrainingArguments(
            output_dir=str(tmp_path / "ta_out"),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,
        )

        assert args.load_best_model_at_end is True
        assert args.metric_for_best_model == "eval_loss"
        assert args.greater_is_better is False

    def test_field_types(self, explicit_yaml):
        """Config 字段类型应与 TrainingArguments 期望一致。"""
        config = Config(str(explicit_yaml))

        assert isinstance(config.load_best_model_at_end, bool)
        assert isinstance(config.greater_is_better, bool)
        assert isinstance(config.metric_for_best_model, (str, type(None)))

    def test_python_bool_passthrough(self, tmp_path):
        """用 Python bool 写入 yaml 会被 yaml.safe_load 还原为 bool。"""
        cfg = _build_base_config(
            tmp_path,
            eval_strategy="epoch",
            save_strategy="epoch",
        )
        cfg["training"]["load_best_model_at_end"] = True
        cfg["training"]["metric_for_best_model"] = "eval_loss"
        cfg["training"]["greater_is_better"] = False
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        config = Config(str(path))

        assert config.load_best_model_at_end is True
        assert config.greater_is_better is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
