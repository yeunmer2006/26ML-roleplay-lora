# LoRA 角色扮演对话生成

基于 LoRA 微调的小型 LLM（1.5B–4B 参数）角色扮演对话生成项目。

## 项目成员

| 姓名 | 学号 | 分工 |
|------|------|------|
| 刘易函 | | |
| 廖绪丞 | | |
| 龙泓潭 | | |

## 技术栈

- **基座模型**: Qwen2.5-3B-Instruct
- **微调方法**: LoRA (PEFT)
- **框架**: PyTorch, Transformers, PEFT, BitsAndBytes
- **环境**: 本地 (RTX 4060 8GB) / Google Colab (T4 GPU)

---

## 环境准备

### 安装依赖

```bash
pip install -r requirements.txt
```

### 验证环境

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 快速开始

### 1. 下载数据集

```bash
python scripts/data_loader.py --output_dir ./processed
```

### 2. 开始训练

```bash
# 本地训练（RTX 4060 8GB）
python scripts/train.py --config configs/lora_config.yaml

# Colab 训练（T4 GPU 高配置）
python scripts/train.py --config configs/lora_config_colab.yaml

# 快速测试（少量数据，1 epoch）
python scripts/train.py --config configs/lora_config.yaml --quick_test
```

### 3. 推理测试

```bash
# 交互式对话
python scripts/inference.py --adapter output/lora_roleplay/final_model

# 指定角色卡
python scripts/inference.py --adapter output/lora_roleplay/final_model --character configs/character_cards/harry_potter.json
```

---

## 训练指南

### 重新训练

从零开始训练新模型：

```bash
# 1. 清理旧的输出（可选）
rm -rf output/

# 2. 下载并清洗数据
python scripts/data_loader.py --output_dir ./processed

# 3. 开始训练
python scripts/train.py --config configs/lora_config.yaml  # 本地
python scripts/train.py --config configs/lora_config_colab.yaml  # Colab
```

### 快速训练

用于验证流程是否正常工作：

```bash
python scripts/train.py --config configs/lora_config.yaml --quick_test
```

### 指定角色卡推理

使用特定角色卡进行对话：

```bash
python scripts/inference.py \
    --adapter output/lora_roleplay/final_model \
    --character configs/character_cards/harry_potter.json
```

可用角色卡：
- `configs/character_cards/luoji.json` - 罗辑（《三体》）
- `configs/character_cards/alina.json` - 阿丽娜
- `configs/character_cards/harry_potter.json` - 哈利波特
- `configs/character_cards/hermione.json` - 赫敏·格兰杰
- `configs/character_cards/gandalf.json` - 甘道夫（《魔戒》）

### 新增角色卡

创建新的角色卡 JSON 文件：

```json
{
  "name": "角色名称",
  "persona": "角色人设描述...",
  "appearance": "外貌特征描述（可选）",
  "background": "背景故事...",
  "speech_style": "说话风格描述，如：温柔、活泼、严谨...",
  "example_dialogues": {
    "日常": "对话示例",
    "特殊场景": "对话示例"
  }
}
```

然后使用该角色卡推理：

```bash
python scripts/inference.py \
    --adapter output/lora_roleplay/final_model \
    --character configs/character_cards/你的新角色.json
```

---

## 配置文件说明

| 配置文件 | 适用场景 | 主要参数 |
|----------|----------|----------|
| `configs/lora_config.yaml` | 本地 RTX 4060 8GB | r=16, batch=1, seq=1024 |
| `configs/lora_config_colab.yaml` | Colab T4 | r=32, batch=4, seq=2048 |

---

## 项目结构

```
project/
├── configs/
│   ├── lora_config.yaml              # 本地训练配置
│   ├── lora_config_colab.yaml        # Colab 训练配置
│   └── character_cards/
│       ├── template.json             # 角色卡模板
│       ├── luoji.json                # 罗辑（《三体》）
│       ├── alina.json                # 阿丽娜
│       ├── harry_potter.json         # 哈利波特
│       ├── hermione.json             # 赫敏·格兰杰
│       └── gandalf.json              # 甘道夫（《魔戒》）
├── scripts/
│   ├── data_loader.py                # 数据下载与清洗
│   ├── train.py                      # 训练脚本
│   └── inference.py                  # 推理脚本
├── processed/                        # 清洗后的数据
├── output/lora_roleplay/             # 训练输出
│   └── final_model/                  # LoRA 权重
├── data/                             # 原始数据（可选）
├── models/                           # 模型存储（可选）
└── requirements.txt                  # Python 依赖
```

---

## 数据集

目前使用 PIPPA-ShareGPT-formatted 数据集进行训练。

| 数据集 | HuggingFace ID | 规模 |
|--------|----------------|------|
| PIPPA-ShareGPT-formatted | KaraKaraWitch/PIPPA-ShareGPT-formatted | ~16,000 条 |

---

## 角色卡格式

```json
{
  "name": "角色名称",
  "persona": "人设描述",
  "background": "背景故事",
  "speech_style": "说话风格",
  "appearance": "外貌特征（可选）",
  "example_dialogues": [
    {
      "user": "用户输入示例",
      "assistant": "角色回复示例"
    }
  ]
}
```

---

## 训练输出

训练完成后，模型保存在：

```
output/lora_roleplay/final_model/
├── adapter_config.json       # LoRA 配置
├── adapter_model.safetensors  # LoRA 权重
└── tokenizer/                # 分词器
```

---

## Google Colab 使用指南

### 1. 挂载 Google Drive

将模型和大文件存到 Google Drive，避免 Colab 断开后丢失：

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone 代码

```python
!git clone -b yeunmer git@github.com:yeunmer2006/26ML-roleplay-lora.git
%cd 26ML-roleplay-lora
```

### 3. 安装依赖

```python
!pip install -r requirements.txt
```

### 4. 开始训练

```python
!python scripts/train.py --config configs/lora_config_colab.yaml
```

### 5. 训练产物存储

Colab `/content` 目录掉线会清空，建议存到 Google Drive：

```
/content/drive/MyDrive/ML_Project/
├── models/          # 基座模型
├── processed/       # 清洗后数据
└── outputs/         # 训练输出
```

---

## 常见问题

### 显存不足 (OOM)

```bash
# 方案1：启用梯度检查点
gradient_checkpointing: true

# 方案2：降低 batch size
per_device_train_batch_size: 1

# 方案3：使用更小的模型
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
```

### 训练不稳定

```bash
# 降低学习率
learning_rate: 1e-4

# 增加 warmup
warmup_ratio: 0.2
```

---

## 参考资料

- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Qwen2.5 模型](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [BitsAndBytes 量化](https://github.com/bitsandbytes-foundation/bitsandbytes)