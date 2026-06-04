#!/usr/bin/env python3
"""
LoRA 评估脚本 - 角色扮演对话生成
评估模型的角色一致性、对话质量等指标
"""

import os
import sys
import json
import torch
import yaml
import argparse
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loader import load_local_dataset


class EvaluationMetrics:
    """评估指标计算类"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def calculate_perplexity(self, texts: list, max_length: int = 512) -> float:
        """
        计算困惑度

        Args:
            texts: 文本列表
            max_length: 最大长度

        Returns:
            平均困惑度
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    labels=inputs["input_ids"]
                )
                loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        return perplexity

    def calculate_roleplay_fidelity(self, character_desc: str, responses: list) -> float:
        """
        计算角色一致性分数

        通过检测回复中是否出现角色特有的词汇和模式来评估

        Args:
            character_desc: 角色描述
            responses: 角色回复列表

        Returns:
            角色一致性分数 (0-1)
        """
        # 提取角色特征关键词
        char_keywords = self._extract_keywords(character_desc)

        if not char_keywords:
            return 0.5  # 无特征词时返回默认分数

        scores = []
        for response in responses:
            response_lower = response.lower()
            matches = sum(1 for kw in char_keywords if kw.lower() in response_lower)
            score = min(matches / len(char_keywords), 1.0)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    def _extract_keywords(self, text: str, top_k: int = 20) -> list:
        """提取关键词（简单实现：高频名词/形容词）"""
        # 简单分词
        words = text.replace("\n", " ").split()
        # 过滤停用词和短词
        stopwords = {"的", "是", "在", "了", "和", "与", "或", "及", "等", "the", "a", "an", "is", "are", "and", "or"}
        words = [w for w in words if len(w) >= 2 and w not in stopwords]
        # 统计词频
        word_freq = Counter(words)
        return [w for w, _ in word_freq.most_common(top_k)]

    def calculate_response_length_stats(self, responses: list) -> dict:
        """计算回复长度统计"""
        lengths = [len(r) for r in responses]
        return {
            "mean_length": np.mean(lengths),
            "median_length": np.median(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "std_length": np.std(lengths)
        }

    def calculate_diversity(self, responses: list) -> float:
        """
        计算回复多样性（基于 unique n-gram 比例）

        Args:
            responses: 回复列表

        Returns:
            多样性分数 (0-1)
        """
        def get_ngrams(text, n=2):
            words = text.split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

        all_ngrams = []
        for response in responses:
            all_ngrams.extend(get_ngrams(response[:500]))  # 限制长度

        if not all_ngrams:
            return 0.0

        unique_ratio = len(set(all_ngrams)) / len(all_ngrams)
        return unique_ratio


def load_model_and_tokenizer(base_model_name: str, adapter_path: str = None):
    """加载模型和分词器"""
    print("=== 加载模型 ===")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path, dtype=torch.float16)
    else:
        print("未找到 LoRA adapter，使用基座模型")
        model = base_model

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, character_card: str, user_input: str,
                     max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """生成单个回复"""
    messages = [
        {"role": "system", "content": character_card},
        {"role": "user", "content": user_input}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in response:
        return response.split("assistant")[-1].strip()
    return response[len(input_text):].strip()


def evaluate_model(model, tokenizer, val_data: list, device, config: dict) -> dict:
    """
    评估模型

    Args:
        model: 模型
        tokenizer: 分词器
        val_data: 验证数据
        device: 设备
        config: 配置

    Returns:
        评估结果字典
    """
    print("\n=== 开始评估 ===")

    metrics_calc = EvaluationMetrics(model, tokenizer, device)

    # 收集回复
    generated_responses = []
    reference_responses = []
    character_cards = []

    # 限制评估样本数
    max_eval_samples = min(config.get("max_eval_samples", 100), len(val_data))
    val_data = val_data[:max_eval_samples]

    print(f"评估样本数: {max_eval_samples}")

    for i, item in enumerate(val_data):
        if (i + 1) % 20 == 0:
            print(f"已处理 {i + 1}/{max_eval_samples}")

        conversations = item.get("conversations", [])
        bot = item.get("bot", {})
        char_desc = bot.get("description", "")

        # 提取对话对 (human -> gpt)
        human_msgs = [c for c in conversations if c.get("from") == "human"]
        gpt_msgs = [c for c in conversations if c.get("from") == "gpt"]

        if not human_msgs or not gpt_msgs:
            continue

        # 取最后一轮对话
        user_input = human_msgs[-1].get("value", "")
        reference_resp = gpt_msgs[-1].get("value", "")

        if not user_input or not reference_resp:
            continue

        character_cards.append(char_desc)

        # 生成回复
        try:
            gen_resp = generate_response(
                model, tokenizer, char_desc, user_input,
                max_new_tokens=config.get("max_new_tokens", 512),
                temperature=config.get("temperature", 0.7)
            )
            generated_responses.append(gen_resp)
            reference_responses.append(reference_resp)
        except Exception as e:
            print(f"生成失败: {e}")
            continue

    # 计算指标
    print("\n=== 计算指标 ===")

    results = {}

    # 1. 困惑度
    if generated_responses:
        perplexity = metrics_calc.calculate_perplexity(reference_responses)
        results["perplexity"] = perplexity
        print(f"困惑度: {perplexity:.4f}")

    # 2. 角色一致性
    if generated_responses and character_cards:
        fidelity_scores = []
        for char_desc, gen_resp in zip(character_cards, generated_responses):
            score = metrics_calc.calculate_roleplay_fidelity(char_desc, [gen_resp])
            fidelity_scores.append(score)
        results["roleplay_fidelity"] = np.mean(fidelity_scores)
        print(f"角色一致性: {results['roleplay_fidelity']:.4f}")

    # 3. 回复长度统计
    if generated_responses:
        length_stats = metrics_calc.calculate_response_length_stats(generated_responses)
        results["response_length_stats"] = length_stats
        print(f"回复平均长度: {length_stats['mean_length']:.1f} 字符")

    # 4. 多样性
    if generated_responses:
        diversity = metrics_calc.calculate_diversity(generated_responses)
        results["diversity"] = diversity
        print(f"回复多样性: {diversity:.4f}")

    # 5. 样本示例
    results["samples"] = []
    num_samples = min(3, len(generated_responses))
    for i in range(num_samples):
        results["samples"].append({
            "character": character_cards[i][:100] + "..." if len(character_cards[i]) > 100 else character_cards[i],
            "user_input": val_data[i]["conversations"][-2].get("value", "")[:100] if len(val_data[i]["conversations"]) >= 2 else "",
            "reference": reference_responses[i][:200] + "..." if len(reference_responses[i]) > 200 else reference_responses[i],
            "generated": generated_responses[i][:200] + "..." if len(generated_responses[i]) > 200 else generated_responses[i],
        })

    return results


def save_results(results: dict, output_path: str):
    """保存评估结果"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 评估脚本")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型名称")
    parser.add_argument("--adapter", type=str, default=None,
                        help="LoRA adapter 路径")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--dataset", type=str, default="./processed",
                        help="数据集目录")
    parser.add_argument("--output", type=str, default="./outputs/eval_results.json",
                        help="输出结果路径")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="最大评估样本数")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        base_model_name = args.base_model or config["model"]["name"]
        inference_config = config.get("inference", {})
    else:
        base_model_name = args.base_model or "Qwen/Qwen2.5-3B-Instruct"
        inference_config = {}

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(base_model_name, args.adapter)

    # 加载数据
    train_data, val_data, test_data = load_local_dataset(args.dataset)
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")

    # 评估配置
    eval_config = {
        "max_new_tokens": inference_config.get("max_new_tokens", 512),
        "temperature": inference_config.get("temperature", 0.7),
        "max_eval_samples": args.max_samples,
    }

    # 评估
    results = evaluate_model(model, tokenizer, val_data, device, eval_config)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    if "perplexity" in results:
        print(f"困惑度 (Perplexity): {results['perplexity']:.4f}")
    if "roleplay_fidelity" in results:
        print(f"角色一致性 (Roleplay Fidelity): {results['roleplay_fidelity']:.4f}")
    if "diversity" in results:
        print(f"回复多样性 (Diversity): {results['diversity']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()