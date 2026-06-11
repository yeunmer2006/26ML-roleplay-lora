#!/usr/bin/env python3
"""
推理脚本单元测试
"""

import sys
import json
import pytest
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.inference import (
    load_character_card,
    chat,
    InferenceConfig
)


class TestInference:
    """推理功能测试类"""

    @pytest.fixture
    def character_card_path(self):
        """角色卡路径 fixture"""
        return str(project_root / "configs" / "character_cards" / "alina.json")

    @pytest.fixture
    def character_card_path_luoji(self):
        """罗辑角色卡路径 fixture"""
        return str(project_root / "configs" / "character_cards" / "luoji.json")

    def test_load_character_card(self, character_card_path):
        """测试加载角色卡"""
        desc = load_character_card(character_card_path)

        assert isinstance(desc, str), "角色描述应为字符串"
        assert len(desc) > 0, "角色描述不应为空"
        # 检查是否包含关键字段
        assert "姓名" in desc or "name" in desc.lower() or "人设" in desc

    def test_load_character_card_luoji(self, character_card_path_luoji):
        """测试加载罗辑角色卡"""
        desc = load_character_card(character_card_path_luoji)

        assert isinstance(desc, str), "角色描述应为字符串"
        assert len(desc) > 0, "角色描述不应为空"
        assert "罗辑" in desc or "逻辑" in desc, "应包含角色名称"

    def test_load_nonexistent_card(self):
        """测试加载不存在的角色卡"""
        with pytest.raises(FileNotFoundError):
            load_character_card("nonexistent/path/card.json")

    def test_inference_config_load(self):
        """测试推理配置加载"""
        config = InferenceConfig(str(project_root / "configs" / "lora_config.yaml"))

        assert hasattr(config, "model_name")
        assert config.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert hasattr(config, "max_new_tokens")
        assert hasattr(config, "temperature")

    def test_character_cards_exist(self):
        """测试角色卡目录是否存在并包含文件"""
        cards_dir = project_root / "configs" / "character_cards"
        assert cards_dir.exists(), "角色卡目录应存在"

        card_files = list(cards_dir.glob("*.json"))
        assert len(card_files) > 0, "应至少有一个角色卡"


class TestCharacterCardFormat:
    """角色卡格式测试"""

    def test_alina_card_format(self):
        """测试 alina 角色卡格式"""
        card_path = project_root / "configs" / "character_cards" / "alina.json"
        with open(card_path, "r", encoding="utf-8") as f:
            card = json.load(f)

        # 检查必要字段
        required_fields = ["name", "persona"]
        for field in required_fields:
            assert field in card, f"角色卡应包含 {field} 字段"

    def test_luoji_card_format(self):
        """测试罗辑角色卡格式"""
        card_path = project_root / "configs" / "character_cards" / "luoji.json"
        with open(card_path, "r", encoding="utf-8") as f:
            card = json.load(f)

        required_fields = ["name", "persona"]
        for field in required_fields:
            assert field in card, f"角色卡应包含 {field} 字段"

        # 检查《三体》相关内容
        assert "三体" in card.get("persona", "") or "面壁者" in card.get("background", ""), \
            "罗辑角色卡应包含《三体》相关描述"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
