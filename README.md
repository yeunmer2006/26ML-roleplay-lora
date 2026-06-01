# 基于 LoRA 微调的角色扮演对话生成

使用 LoRA (Low-Rank Adaptation) 技术对小型 LLM（Qwen2.5-3B-Instruct）进行高效微调，实现角色扮演对话生成。

## 项目成员

| 姓名 | 学号 | 分工 |
|------|------|------|
| 刘易函 | | |
| 廖绪丞 | | |
| 龙泓潭 | | |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证环境

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 3. 训练模型

```bash
# 快速测试（推荐先跑，约 5-10 分钟）
python scripts/train.py --config configs/lora_config_test.yaml

# 本地训练（RTX 4060，约 30-60 分钟）
python scripts/train.py --config configs/lora_config_local.yaml

# Google Colab 训练（T4 GPU，约 30 分钟）
python scripts/train.py --config configs/lora_config_colab.yaml
```

### 4. 推理测试

```bash
# 交互式对话
python scripts/inference.py --adapter output/lora_roleplay/final_model

# 指定角色卡
python scripts/inference.py --adapter output/lora_roleplay/final_model \
    --character configs/character_cards/alina.json

# 批量测试
python scripts/inference.py --adapter output/lora_roleplay/final_model --batch
```

### 5. 模型评估

```bash
python scripts/eval.py --adapter output/lora_roleplay/final_model --max_samples 50
```

## 项目结构

```
project/
├── configs/                      # 配置文件
│   ├── lora_config.yaml          # 默认配置
│   ├── lora_config_local.yaml   # RTX 4060
│   ├── lora_config_colab.yaml   # Google Colab
│   ├── lora_config_test.yaml    # 快速测试
│   └── character_cards/         # 角色卡
├── scripts/                      # Python 脚本
│   ├── data_loader.py           # 数据下载与清洗
│   ├── train.py                 # 训练脚本
│   ├── inference.py             # 推理脚本
│   └── eval.py                  # 评估脚本
├── tests/                       # 单元测试
├── notebooks/                   # Colab 笔记本
├── processed/                   # 清洗后的数据
├── output/lora_roleplay/        # 训练输出
│   └── final_model/             # LoRA 权重
└── requirements.txt
```

## 配置文件说明

| 配置 | 硬件 | 主要参数 | 预计时间 |
|------|------|----------|----------|
| `lora_config_test.yaml` | 快速测试 | r=8, seq=256, batch=2 | ~5-10 分钟 |
| `lora_config_local.yaml` | RTX 4060 | r=16, seq=1024, batch=1 | ~30-60 分钟 |
| `lora_config_colab.yaml` | Colab T4 | r=32, seq=2048, batch=4 | ~30 分钟 |

## 角色卡格式

```json
{
  "name": "角色名称",
  "persona": "人设描述",
  "background": "背景故事",
  "speech_style": "说话风格",
  "appearance": "外貌特征（可选）"
}
```

**可用角色卡：**
- `alina.json` - 默认角色
- `luoji.json` - 罗辑（《三体》）
- `harry_potter.json` - 哈利波特
- `hermione.json` - 赫敏·格兰杰
- `gandalf.json` - 甘道夫

## 数据集

| 数据集 | HuggingFace ID | 规模 |
|--------|----------------|-----:|
| PIPPA-ShareGPT-formatted | KaraKaraWitch/PIPPA-ShareGPT-formatted | ~16,000 |

## 常见问题

### 显存不足 (OOM)

```yaml
# 启用梯度检查点
gradient_checkpointing: true

# 降低 batch size
per_device_train_batch_size: 1

# 降低序列长度
max_seq_length: 256
```

### 训练不稳定

```yaml
# 降低学习率
learning_rate: 1e-4

# 增加 warmup
warmup_ratio: 0.2
```

## 参考资料

- [PEFT 文档](https://huggingface.co/docs/peft)
- [Qwen2.5 模型](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [BitsAndBytes 量化](https://github.com/bitsandbytes-foundation/bitsandbytes)
