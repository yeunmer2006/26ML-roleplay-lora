# 基于 LoRA 微调的角色扮演对话生成

使用 LoRA (Low-Rank Adaptation) 技术对小型 LLM（Qwen2.5-3B-Instruct）进行高效微调，实现角色扮演对话生成。

## 项目成员

| 姓名 | 学号 | 分工 |
|------|------|------|
| 刘易函 | | |
| 廖绪丞 | | |
| 龙泓潭 | | |

## 快速开始

### 1. 克隆代码仓库

```bash
git clone git@github.com:yeunmer2006/26ML-roleplay-lora.git
cd 26ML-roleplay-lora
```

### 2. 创建 conda 环境并安装依赖

```bash
conda create -n ml_roleplay python=3.10 -y
conda activate ml_roleplay
pip install -r requirements.txt
```

### 3. 验证环境

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 4. 下载基座模型（如无网络问题可跳过）

模型约 6GB，首次训练会自动下载。如遇到网络问题，可使用镜像站加速：

```bash
# 设置 HuggingFace 镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到缓存目录
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct
```

### 5. 训练模型

```bash
python scripts/train.py --config configs/lora_config_test.yaml

python scripts/train.py --config configs/lora_config_local.yaml

python scripts/train.py --config configs/lora_config_colab.yaml
```

### 6. 推理测试

```bash
# 交互式对话
python scripts/inference.py --adapter output/lora_roleplay/final_model

# 指定角色卡
python scripts/inference.py --adapter output/lora_roleplay/final_model \
    --character configs/character_cards/alina.json

# 批量测试
python scripts/inference.py --adapter output/lora_roleplay/final_model --batch
```

### 7. 模型评估

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
