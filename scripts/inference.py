#!/usr/bin/env python3
"""
LoRA 推理脚本 - 角色扮演对话
加载微调后的模型进行交互式对话
"""

import os
import sys
import json
import torch
import yaml
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def resolve_model_source(value: str) -> str:
    """Resolve an existing local model path relative to the project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_path = project_root / path
    return str(project_path) if project_path.exists() else value


def resolve_local_path(value: str = None) -> str:
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else project_root / path)


class InferenceConfig:
    """推理配置类"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = project_root / "configs" / "lora_config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 模型配置
        self.model_name = config["model"]["name"]

        # 推理配置
        self.max_new_tokens = config["inference"]["max_new_tokens"]
        self.temperature = config["inference"]["temperature"]
        self.top_p = config["inference"]["top_p"]
        self.repetition_penalty = config["inference"]["repetition_penalty"]


def load_model_and_tokenizer(base_model_name: str, adapter_path: str = None):
    """
    加载模型和分词器

    Args:
        base_model_name: 基座模型名称
        adapter_path: LoRA adapter 路径

    Returns:
        model: 加载了 adapter 的模型
        tokenizer: 分词器
    """
    base_model_name = resolve_model_source(base_model_name)
    adapter_path = resolve_local_path(adapter_path)
    print(f"=== 加载模型: {base_model_name} ===")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )

    # 加载 LoRA adapter
    if adapter_path and Path(adapter_path).exists():
        print(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            dtype=torch.float16,
        )
    else:
        print("未找到 LoRA adapter，使用基座模型")
        model = base_model

    model.eval()
    print("模型加载完成")

    return model, tokenizer


def load_character_card(card_path: str) -> str:
    """
    加载角色卡

    Args:
        card_path: 角色卡文件路径

    Returns:
        角色描述文本
    """
    with open(card_path, "r", encoding="utf-8") as f:
        card = json.load(f)

    # 构建角色描述
    character_desc = []

    if "name" in card:
        character_desc.append(f"姓名：{card['name']}")

    if "persona" in card:
        character_desc.append(f"人设：{card['persona']}")

    if "background" in card:
        character_desc.append(f"背景：{card['background']}")

    if "speech_style" in card:
        character_desc.append(f"说话风格：{card['speech_style']}")

    if "appearance" in card:
        character_desc.append(f"外貌：{card['appearance']}")

    return "\n".join(character_desc)


def chat(
    model,
    tokenizer,
    character_card: str,
    user_input: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> str:
    """
    角色扮演对话

    Args:
        model: 模型
        tokenizer: 分词器
        character_card: 角色描述
        user_input: 用户输入
        max_new_tokens: 最大生成长度
        temperature: 采样温度
        top_p: nucleus 采样阈值
        repetition_penalty: 重复惩罚

    Returns:
        角色回复
    """
    # 构建消息
    messages = [
        {"role": "system", "content": character_card},
        {"role": "user", "content": user_input}
    ]

    # 应用聊天模板
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取 assistant 回复
    if "assistant" in response:
        assistant_response = response.split("assistant")[-1].strip()
    else:
        assistant_response = response[len(input_text):].strip()

    return assistant_response


def interactive_chat(
    model,
    tokenizer,
    character_card: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
):
    """
    交互式对话

    Args:
        model: 模型
        tokenizer: 分词器
        character_card: 角色描述
        max_new_tokens: 最大生成长度
        temperature: 采样温度
        top_p: nucleus 采样阈值
        repetition_penalty: 重复惩罚
    """
    print("\n" + "=" * 60)
    print("角色扮演对话系统")
    print("=" * 60)
    print("输入你的对话，输入 exit/quit/q 退出")
    print("-" * 60)

    # 显示角色信息
    print(f"\n【角色设定】\n{character_card}\n")
    print("-" * 60)

    # 对话历史
    conversation_history = [
        {"role": "system", "content": character_card}
    ]

    while True:
        try:
            user_input = input("\n【用户】: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("对话结束")
                break

            if not user_input:
                continue

            # 添加到历史
            conversation_history.append({"role": "user", "content": user_input})

            # 生成回复
            response = chat(
                model,
                tokenizer,
                character_card,
                user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

            print(f"\n【角色】: {response}")

            # 添加到历史
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n对话结束")
            break


def batch_inference(
    model,
    tokenizer,
    character_card: str,
    test_inputs: list,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
):
    """
    批量推理测试

    Args:
        model: 模型
        tokenizer: 分词器
        character_card: 角色描述
        test_inputs: 测试输入列表
        max_new_tokens: 最大生成长度
        temperature: 采样温度
        top_p: nucleus 采样阈值
        repetition_penalty: 重复惩罚
    """
    print("\n" + "=" * 60)
    print("批量推理测试")
    print("=" * 60)
    print(f"角色设定: {character_card[:100]}...")
    print("=" * 60)

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n【测试 {i}】")
        print(f"【用户】: {user_input}")

        response = chat(
            model,
            tokenizer,
            character_card,
            user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        print(f"【角色】: {response}")
        print("-" * 50)


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA 推理脚本")
    parser.add_argument(
        "--base_model",
        type=str,
        default=os.getenv("MODEL_DIR"),
        help="基座模型目录或模型 ID；默认读取 MODEL_DIR",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=os.getenv("ADAPTER_DIR"),
        help="LoRA adapter 路径；默认读取 ADAPTER_DIR",
    )
    parser.add_argument("--character", type=str,
                        default="configs/character_cards/alina.json",
                        help="角色卡路径")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--batch", action="store_true",
                        help="批量推理模式")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=None,
                        help="采样温度")
    return parser


def main():
    args = build_parser().parse_args()

    # 加载配置
    if args.config:
        config = InferenceConfig(args.config)
    else:
        config = InferenceConfig()

    # 模型名称
    base_model = args.base_model or config.model_name

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(base_model, args.adapter)

    # 加载角色卡
    character_card = load_character_card(args.character)

    # 参数
    max_new_tokens = args.max_new_tokens or config.max_new_tokens
    temperature = args.temperature or config.temperature
    top_p = config.top_p
    repetition_penalty = config.repetition_penalty

    # 运行模式
    if args.batch:
        # 批量测试
        test_inputs = [
            "你好，请问你是谁？",
            "你能告诉我关于这个世界的事情吗？",
            "你有什么特殊的能力吗？",
        ]
        batch_inference(
            model, tokenizer, character_card, test_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    else:
        # 交互式对话
        interactive_chat(
            model, tokenizer, character_card,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )


if __name__ == "__main__":
    main()
