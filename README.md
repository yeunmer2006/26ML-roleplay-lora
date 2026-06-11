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

### 4. 准备基座模型

模型约 6GB。准备脚本会依次检查用户指定目录、项目内 `models/` 和
`.cache/`、Hugging Face 标准缓存、ModelScope 标准缓存。检查会验证
tokenizer、权重索引和全部权重分片，找到完整模型后不会重复下载。

本地没有完整模型时，会优先从 Hugging Face 下载，失败后自动回退到
ModelScope。需要 Hugging Face 镜像时可提前设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 5. 训练模型

```bash
# 首次准备：安装依赖，检查模型和 processed/ 数据，缺失时自动下载
bash scripts/prepare_training.sh

# 先运行约 10 steps 的冒烟测试
bash scripts/run_training.sh smoke

# RTX 4060 正式训练（先执行 50-step 时间预检）
bash scripts/run_training.sh train

# 从 checkpoint 继续
bash scripts/run_training.sh train \
    --resume output/experiments/run_001/checkpoint-250
```

训练脚本会自动激活 `ml_roleplay` Conda 环境。PyTorch 不由项目依赖文件
重装，以保留与本机 CUDA 匹配的版本。

模型和数据目录可显式覆盖，路径相对于项目根目录解析：

```bash
MODEL_DIR=models/Qwen2.5-3B-Instruct DATA_DIR=processed \
  bash scripts/prepare_training.sh
```

本地和 AutoDL 使用同一流程。若 `processed/train.jsonl`、`val.jsonl`、
`test.jsonl` 完整可解析，脚本会直接复用；否则从
`KaraKaraWitch/PIPPA-ShareGPT-formatted` 下载、清洗并写入 `processed/`，
不需要手动制作或上传压缩包。

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
export JUDGE_BASE_URL="https://your-provider.example/v1"
export JUDGE_MODEL="your-judge-model"
read -rsp "Judge API Key: " JUDGE_API_KEY
echo
export JUDGE_API_KEY

BASE_MODEL="$(python scripts/resource_manager.py resolve-model \
  --project-root .)"

python scripts/eval.py compare \
  --base_model "${BASE_MODEL}" \
  --adapter output/experiments/run_001/final_model \
  --dataset processed \
  --output_dir output/evaluations/run_001

unset JUDGE_API_KEY JUDGE_BASE_URL JUDGE_MODEL
```

评估会比较无角色卡基座、角色卡基座和角色卡 LoRA 三组系统，默认使用
100 条单轮样本和 20 个四轮挑战。结果包含逐样本 JSONL、汇总 JSON 和可直接
用于报告的 Markdown。添加 `--resume` 可继续中断的生成或裁判任务；添加
`--skip_judge` 可仅运行本地自动指标。

API Key 不直接写入脚本或命令参数。若密钥曾以明文形式保存或提交，应立即
在服务商控制台撤销并重新生成。

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
