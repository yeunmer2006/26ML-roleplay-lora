# LoRA 角色扮演对话生成

基于 LoRA 微调的小型 LLM（1.5B–4B 参数）角色扮演对话生成项目。

## 项目成员

刘易函 廖绪丞 龙泓潭
| 姓名 | 学号 | 分工 |
|------|------|------|
| | | |
| | | |
| | | |

## 技术栈

- **基座模型**: Qwen2.5-3B-Instruct (已选择)
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

## 可用命令

### 1. 数据下载与清洗

```bash
# 下载 PIPPA 数据集并清洗
python scripts/data_loader.py --output_dir ./processed

# 指定数据集（默认：KaraKaraWitch/PIPPA-ShareGPT-formatted）
python scripts/data_loader.py --dataset <dataset_name> --output_dir ./processed
```

### 2. 模型训练

```bash
# 本地训练（RTX 4060 8GB ）
python scripts/train.py --config configs/lora_config_local.yaml

# Colab 训练（T4 GPU 高配置）
python scripts/train.py --config configs/lora_config_colab.yaml

# 快速测试（少量数据，1 epoch）
python scripts/train.py --config configs/lora_config_local.yaml --quick_test
```

### 3. 推理测试

```bash
# 批量测试（使用 LoRA adapter）
python scripts/inference.py --batch --adapter output/lora_roleplay/final_model

# 交互式对话
python scripts/inference.py --adapter output/lora_roleplay/final_model

# 指定角色卡
python scripts/inference.py --adapter output/lora_roleplay/final_model --character configs/character_cards/alina.json

# 使用基座模型（无 LoRA）
python scripts/inference.py
```

---

## 配置文件说明

| 配置文件 | 适用场景 | 主要参数 |
|----------|----------|----------|
| `lora_config_local.yaml` | RTX 4060 8GB | r=16, batch=1, seq=1024 |
| `lora_config_colab.yaml` | Colab T4 | r=32, batch=4, seq=2048 |

---

## 项目结构

```
project/
├── configs/
│   ├── lora_config.yaml              # LoRA 配置（通用）
│   ├── lora_config_local.yaml        # 本地训练配置
│   ├── lora_config_colab.yaml       # Colab 训练配置
│   └── character_cards/
│       ├── template.json             # 角色卡模板
│       └── alina.json                 # 示例角色卡
├── scripts/
│   ├── __init__.py
│   ├── data_loader.py                # 数据下载与清洗
│   ├── train.py                      # 训练脚本
│   └── inference.py                   # 推理脚本
├── processed/                        # 清洗后的数据
│   ├── train.jsonl                   # 训练集
│   └── val.jsonl                     # 验证集
├── output/lora_roleplay/             # 训练输出
│   └── final_model/                   # LoRA 权重
├── data/                             # 原始数据（可选）
├── models/                           # 模型存储（可选）
└── requirements.txt                  # Python 依赖
```

---

## 数据集

> 目前暂且使用 PIPPA-ShareGPT-formatted 数据集进行训练，后续计划扩展更多角色扮演对话数据集。

| 数据集 | HuggingFace ID | 规模 | 说明 |
|--------|----------------|------|------|
| PIPPA-ShareGPT-formatted | KaraKaraWitch/PIPPA-ShareGPT-formatted | ~16,000 条 | 

---

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

---

## 训练输出

训练完成后，模型保存在：
```
output/lora_roleplay/final_model/
├── adapter_config.json      # LoRA 配置
├── adapter_model.safetensors # LoRA 权重
└── tokenizer/               # 分词器
```

---

## Google Colab 使用指南

### 1. 挂载 Google Drive

建议将模型和大文件存到 Google Drive，避免 Colab 断开后丢失：

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

Colab `/content` 目录掉线会清空，模型和输出建议存到 Google Drive：

```
/content/drive/MyDrive/ML_Project/
├── models/          # 基座模型
├── processed/       # 清洗后数据
└── outputs/         # 训练输出
```

修改 `configs/lora_config_colab.yaml` 中的路径配置即可。

### 6. 查看代码

- 文件浏览器：左侧面板 > 文件
- 快速查看：`!cat scripts/train.py`
- GitHub：https://github.com/yeunmer2006/26ML-roleplay-lora/tree/yeunmer

---

## 常见问题

### 显存不足 (OOM)
```bash
# 方案1：启用梯度检查点
# 修改 configs/lora_config_local.yaml:
gradient_checkpointing: true

# 方案2：降低 batch size
per_device_train_batch_size: 1

# 方案3：使用更小的模型
# 使用 Qwen2.5-1.5B-Instruct
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