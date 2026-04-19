# LoRA 角色扮演对话生成

基于 LoRA 微调的小型 LLM（1.5B–4B 参数）角色扮演对话生成项目。

## 项目成员

| 姓名 | 学号 | 分工 |
|------|------|------|
| | | |
| | | |
| | | |

## 技术栈

- **模型**: Qwen2.5-1.5B-Instruct / Llama-3.2-3B-Instruct
- **微调方法**: LoRA (PEFT)
- **框架**: PyTorch, Transformers, Axolotl
- **环境**: Google Colab (免费 GPU)

## 快速开始

### 1. 环境准备

```bash
pip install torch transformers peft datasets accelerate
```

### 2. 数据格式

使用 PIPPA-ShareGPT 格式：

```json
{
  "conversations": [
    {"from": "system", "value": "角色描述..."},
    {"from": "human", "value": "用户输入"},
    {"from": "gpt", "value": "角色回复"}
  ]
}
```

### 3. 训练

```bash
python scripts/train.py --config configs/lora_config.yaml
```

### 4. 推理

```bash
python scripts/inference.py --model models/checkpoint --character character_card.json
```

## 项目结构

```
├── data/                  # 原始数据集
├── processed/             # 清洗后的数据
├── models/                # 微调模型 checkpoints
├── scripts/               # 训练和推理脚本
├── configs/               # LoRA / 训练配置
└── outputs/               # 训练日志和输出
```

## 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| PIPPA-ShareGPT | ~16,000 条 | 通用角色扮演 |
| hieunguyenminh/roleplay | ~5,000 条 | 多样化角色 |

## 参考资料

- [PEFT 文档](https://huggingface.co/docs/peft)
- [Axolotl 框架](https://github.com/OpenAccess-AI-Collective/axolotl)
- [PIPPA 数据集](https://huggingface.co/datasets/PIPPA)
