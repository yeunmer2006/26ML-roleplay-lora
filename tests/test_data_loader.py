#!/usr/bin/env python3
"""
数据加载器单元测试
"""

import sys
import json
import pytest
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loader import (
    encode_conversation,
    load_local_dataset,
    format_conversation,
    tokenize_function
)


class TestDataLoader:
    """数据加载器测试类"""

    @pytest.fixture
    def sample_data(self):
        """测试用样本数据"""
        return {
            "id": "test-001",
            "bot": {
                "name": "Test Character",
                "description": "A test character for unit testing"
            },
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": "Hello, how are you?"},
                {"from": "gpt", "value": "I'm doing great, thank you!"}
            ]
        }

    @pytest.fixture
    def mock_tokenizer(self):
        """模拟分词器"""
        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "<eos>"
                self.bos_token = "<bos>"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                result = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    result += f"<|{role}|>\n{content}\n"
                if add_generation_prompt:
                    result += "<|assistant|>\n"
                return result

            def __call__(self, text, truncation=False, max_length=None, padding=None):
                return {
                    "input_ids": list(range(min(len(text), max_length or 100))),
                    "attention_mask": [1] * min(len(text), max_length or 100)
                }

        return MockTokenizer()

    def test_load_local_dataset(self):
        """测试从本地加载数据集"""
        train_data, val_data, test_data = load_local_dataset("./processed")

        assert isinstance(train_data, list), "训练集应为列表"
        assert isinstance(val_data, list), "验证集应为列表"
        assert isinstance(test_data, list), "测试集应为列表"
        assert len(train_data) > 0, "训练集不应为空"
        assert len(val_data) > 0, "验证集不应为空"

        # 检查数据格式
        for item in train_data[:3]:
            assert "id" in item, "数据应包含 id 字段"
            assert "bot" in item, "数据应包含 bot 字段"
            assert "conversations" in item, "数据应包含 conversations 字段"

    def test_format_conversation(self, sample_data, mock_tokenizer):
        """测试对话格式化"""
        result = format_conversation(sample_data, mock_tokenizer)

        assert "text" in result, "格式化结果应包含 text 字段"
        assert isinstance(result["text"], str), "text 字段应为字符串"
        assert "<|system|>" in result["text"], "应包含 system 角色标记"
        assert "<|user|>" in result["text"], "应包含 user 角色标记"
        assert "<|assistant|>" in result["text"], "应包含 assistant 角色标记"

    def test_format_conversation_with_empty_conversations(self, mock_tokenizer):
        """测试空对话处理"""
        empty_data = {
            "id": "test-002",
            "conversations": []
        }
        result = format_conversation(empty_data, mock_tokenizer)
        assert "text" in result

    def test_tokenize_function(self, mock_tokenizer):
        """测试 tokenize 功能"""
        sample = {
            "text": "This is a test message for tokenization."
        }
        result = tokenize_function(sample, mock_tokenizer, max_length=50)

        assert "input_ids" in result, "结果应包含 input_ids"
        assert "attention_mask" in result, "结果应包含 attention_mask"
        assert "labels" in result, "结果应包含 labels"
        assert len(result["input_ids"]) == len(result["labels"]), \
            "input_ids 和 labels 长度应一致"

    def test_encode_conversation_masks_non_assistant_tokens(
        self, sample_data, mock_tokenizer
    ):
        """训练标签只监督 assistant 内容。"""
        result = encode_conversation(sample_data, mock_tokenizer, max_length=200)

        assert result["length"] == len(result["input_ids"])
        assert any(label == -100 for label in result["labels"])
        assert any(label != -100 for label in result["labels"])

    def test_data_integrity(self):
        """测试数据完整性"""
        train_data, val_data, _ = load_local_dataset("./processed")

        # 检查 system 角色存在
        for item in train_data[:10]:
            has_system = any(
                msg.get("from") == "system"
                for msg in item.get("conversations", [])
            )
            assert has_system, f"数据 {item.get('id')} 缺少 system 角色"

        # 检查对话轮次
        for item in train_data[:10]:
            msgs = item.get("conversations", [])
            human_count = sum(1 for m in msgs if m.get("from") == "human")
            gpt_count = sum(1 for m in msgs if m.get("from") == "gpt")
            assert human_count > 0, "应至少包含一条 human 消息"
            assert gpt_count > 0, "应至少包含一条 gpt 消息"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
