#!/usr/bin/env python3
"""
模型加载与推理单元测试
"""

import sys
import torch
import pytest
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train import Config


class TestModelConfig:
    """模型配置测试类"""

    def test_config_load_local(self):
        """测试加载本地配置文件"""
        config = Config(str(project_root / "configs" / "train_4060.yaml"))

        assert config.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert config.lora_r == 8
        assert config.max_seq_length == 512
        assert config.max_train_samples == 4000

    def test_config_load_smoke(self):
        """测试加载冒烟配置文件"""
        config = Config(str(project_root / "configs" / "train_smoke.yaml"))

        assert config.lora_r == 8
        assert config.max_seq_length == 256
        assert config.max_steps == 10

    def test_lora_config_values(self):
        """测试 LoRA 配置值是否合理"""
        config = Config(str(project_root / "configs" / "train_4060.yaml"))

        # LoRA rank 应为正数
        assert config.lora_r > 0, "LoRA rank 应大于 0"

        # LoRA alpha 通常是 rank 的 1-2 倍
        assert config.lora_alpha >= config.lora_r, "LoRA alpha 应 >= rank"

        # 目标模块应包含核心 Attention 层
        expected_modules = ["q_proj", "k_proj", "v_proj"]
        for module in expected_modules:
            assert module in config.lora_target_modules, \
                f"目标模块应包含 {module}"

    def test_training_config_values(self):
        """测试训练配置值是否合理"""
        config = Config(str(project_root / "configs" / "train_4060.yaml"))

        # 学习率应为正数
        assert config.learning_rate > 0, "学习率应为正数"

        # Batch size 应为正数
        assert config.per_device_train_batch_size > 0, "batch size 应为正数"

        # 训练轮次应为正数
        assert config.num_train_epochs > 0, "训练轮次应为正数"

    def test_inference_config_accessible(self):
        """测试推理配置可访问"""
        config = Config(str(project_root / "configs" / "train_4060.yaml"))

        # 检查 Config 类是否正确读取 inference 配置
        # 由于 train.py 的 Config 类主要服务于训练，
        # 推理配置通过 InferenceConfig 读取，这里验证配置文件本身
        import yaml
        with open(project_root / "configs" / "lora_config.yaml", "r") as f:
            full_config = yaml.safe_load(f)

        assert "inference" in full_config
        assert "max_new_tokens" in full_config["inference"]
        assert "temperature" in full_config["inference"]


class TestModelAvailability:
    """模型可用性测试（需要网络连接）"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_cuda_available(self):
        """测试 CUDA 是否可用"""
        assert torch.cuda.is_available()
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def test_pytorch_version(self):
        """测试 PyTorch 版本"""
        assert torch.__version__ >= "2.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
