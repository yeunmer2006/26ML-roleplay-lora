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
    encode_conversation_windows,
    load_local_dataset,
    format_conversation,
    tokenize_function,
    truncate_messages,
)


class TrackingTokenizer:
    """为消息裁剪测试提供稳定、可还原的字符级聊天模板。"""

    def __init__(self):
        self._tokens = {}
        self._texts = {}
        self._next_token = 1

    def _id(self, value):
        if value not in self._tokens:
            token = self._next_token
            self._next_token += 1
            self._tokens[value] = token
            self._texts[token] = value
        return self._tokens[value]

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        parts = []
        for message in messages:
            parts.append(f"<{message['role']}>")
            parts.extend(message["content"])
        if add_generation_prompt:
            parts.append("<assistant>")
        return [self._id(part) for part in parts] if tokenize else "".join(parts)

    def token_text(self, token_ids):
        return "".join(
            self._texts[token]
            for token in token_ids
            if not self._texts[token].startswith("<")
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

    def test_truncation_preserves_system_and_recent_messages(self):
        tokenizer = TrackingTokenizer()
        messages = [
            {"role": "system", "content": "CARD"},
            {"role": "assistant", "content": "old-answer"},
            {"role": "user", "content": "recent-question"},
            {"role": "assistant", "content": "recent-answer"},
        ]
        expected = messages[:1] + messages[2:]
        max_length = len(tokenizer.apply_chat_template(expected, tokenize=True))

        result = truncate_messages(messages, tokenizer, max_length)

        assert result == expected
        assert result[0]["content"] == "CARD"
        assert result[-1]["content"] == "recent-answer"

    def test_truncation_drops_unsupervised_trailing_messages(self):
        tokenizer = TrackingTokenizer()
        messages = [
            {"role": "system", "content": "CARD"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer-one"},
            {"role": "assistant", "content": "answer-two"},
            {"role": "user", "content": "trailing-user"},
        ]

        result = truncate_messages(messages, tokenizer, max_length=10_000)

        assert result == messages[:-1]

    def test_encode_drops_orphan_assistant_opening(self):
        tokenizer = TrackingTokenizer()
        example = {
            "bot": {"name": "Unknown", "description": ""},
            "conversations": [
                {"from": "gpt", "value": "opening"},
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "answer"},
            ],
        }

        result = encode_conversation(example, tokenizer, max_length=100)

        assert any(label != -100 for label in result["labels"])
        assert tokenizer.token_text([
            token
            for token, label in zip(result["input_ids"], result["labels"])
            if label != -100
        ]) == "answer"

    def test_truncation_uses_soft_limit_for_card_and_last_pair(self):
        tokenizer = TrackingTokenizer()
        messages = [
            {"role": "system", "content": "VERY-LONG-CARD"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "VERY-LONG-ANSWER"},
        ]

        result = truncate_messages(messages, tokenizer, max_length=1)

        assert result == messages
        assert len(tokenizer.apply_chat_template(result, tokenize=True)) > 1

    def test_truncation_returns_no_orphan_assistant_without_user(self):
        tokenizer = TrackingTokenizer()
        messages = [
            {"role": "system", "content": "CARD"},
            {"role": "assistant", "content": "orphan"},
        ]

        result = truncate_messages(messages, tokenizer, max_length=1)

        assert result == messages[:1]

    def test_encode_after_truncation_supervises_all_kept_assistants(self):
        tokenizer = TrackingTokenizer()
        example = {
            "bot": {"name": "Role", "description": "CARD"},
            "conversations": [
                {"from": "gpt", "value": "old"},
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "answer-one"},
                {"from": "gpt", "value": "answer-two"},
                {"from": "human", "value": "unused-tail"},
            ],
        }
        kept_messages = [
            {"role": "system", "content": "Character name: Role\nCARD"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer-one"},
            {"role": "assistant", "content": "answer-two"},
        ]
        max_length = len(tokenizer.apply_chat_template(
            kept_messages,
            tokenize=True,
        ))

        result = encode_conversation(example, tokenizer, max_length=max_length)
        supervised = [
            token
            for token, label in zip(result["input_ids"], result["labels"])
            if label != -100
        ]

        assert tokenizer.token_text(supervised) == "answer-oneanswer-two"

    def test_window_encoding_spreads_turns_and_labels_target_only(self):
        tokenizer = TrackingTokenizer()
        example = {
            "bot": {"name": "Role", "description": "CARD"},
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
                {"from": "human", "value": "q3"},
                {"from": "gpt", "value": "a3"},
            ],
        }

        windows = encode_conversation_windows(
            example,
            tokenizer,
            max_length=1000,
            max_windows=2,
        )
        supervised = [
            tokenizer.token_text([
                token
                for token, label in zip(window["input_ids"], window["labels"])
                if label != -100
            ])
            for window in windows
        ]

        assert supervised == ["a1", "a3"]

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
