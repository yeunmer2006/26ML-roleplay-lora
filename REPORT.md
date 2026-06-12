# 当前项目架构报告

## 1. 报告目的

本文档面向项目组内部，说明当前仓库已经实现的训练、推理、评测与资源管理架构，重点回答以下问题：

1. 当前 LoRA 微调是否采用单一 adapter 架构
2. 角色设定是在训练阶段固化，还是在推理阶段通过角色卡注入
3. 当前系统的数据流、模块边界和主要限制是什么

本报告基于当前仓库代码实现整理，而不是基于最初设想或课程提案。

---

## 2. 总体结论

### 2.1 架构结论

当前项目采用的是：

- 一个基础模型：`Qwen/Qwen2.5-3B-Instruct`
- 一次训练产出一个 LoRA adapter 目录
- 推理时加载“基础模型 + 单个 LoRA adapter”
- 同时可叠加一个角色卡，作为 system prompt 注入

因此，**当前实现是单一 adapter 架构，不是多 adapter 路由，也不是 adapter fusion，更不是一个进程内同时管理多个角色 adapter 的架构。**

### 2.2 角色能力来源

当前角色扮演能力来自两个层次：

1. **LoRA adapter**
   负责把基础模型整体调整到更适合角色扮演对话的分布。
2. **角色卡**
   在推理时以 system prompt 形式注入具体 persona、背景、说话风格等设定。

换句话说，当前系统不是“每个角色一个独立 LoRA adapter”的方案，而是更接近：

**一个通用角色扮演 LoRA + 多个可切换角色卡**

---

## 3. 架构总览

### 3.1 逻辑分层

当前代码可以分成五层：

1. 资源准备层
   - 负责检查本地模型和数据是否完整
   - 必要时自动下载模型和数据

2. 数据处理层
   - 清洗 PIPPA 数据集
   - 划分 train/val/test
   - 转成 Hugging Face chat template 可训练格式

3. 训练层
   - 加载 Qwen2.5-3B-Instruct
   - 使用 4-bit 量化和 LoRA 进行 QLoRA 训练
   - 输出 adapter 权重和训练 manifest

4. 推理层
   - 加载基础模型
   - 可选加载单个 LoRA adapter
   - 加载角色卡并拼成 system prompt
   - 进行交互式或批量推理

5. 评测层
   - 比较三种系统：
     - Base, no card
     - Base + card
     - LoRA + card
   - 输出自动指标、LLM judge 指标和 Markdown 报告

### 3.2 目录与职责

| 路径 | 作用 |
|---|---|
| `scripts/resource_manager.py` | 模型与数据资源解析、校验、下载 |
| `scripts/data_loader.py` | 数据清洗、切分、编码 |
| `scripts/train.py` | QLoRA 训练主入口 |
| `scripts/inference.py` | 推理主入口 |
| `scripts/eval.py` | 三路对比评测 |
| `scripts/prepare_training.sh` | 训练前环境与资源准备 |
| `scripts/run_training.sh` | 冒烟训练与正式训练封装 |
| `configs/train_4060.yaml` | 当前正式训练配置 |
| `configs/train_smoke.yaml` | 冒烟配置 |
| `configs/lora_config.yaml` | 推理默认配置 |
| `configs/character_cards/*.json` | 角色卡 |

---

## 4. 训练架构

### 4.1 基础模型

训练脚本 `scripts/train.py` 默认加载：

- `Qwen/Qwen2.5-3B-Instruct`

模型路径可以来自：

- 命令行 `--model_path`
- 环境变量 `MODEL_DIR`
- Hugging Face 模型 ID

资源管理器会优先复用本地目录，不完整时再触发下载。

### 4.2 QLoRA 方案

当前训练采用标准 QLoRA 思路：

- 4-bit 量化加载基础模型
- 基础模型参数冻结
- 通过 PEFT 在指定 attention 投影层上插入 LoRA

当前默认 target modules 为：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

`configs/train_4060.yaml` 中的正式训练 LoRA 参数为：

- `r: 8`
- `lora_alpha: 16`
- `lora_dropout: 0.05`
- `bias: none`

### 4.3 单 adapter 训练产物

训练完成后，脚本会把结果保存到：

- `output/experiments/<run_name>/final_model`

这里保存的是：

- LoRA adapter 权重
- tokenizer 配置

不是完整重训后的全量模型副本。

这说明当前训练产物的组织方式是：

- **一次 run 对应一个 adapter 目录**
- **一次推理最多加载一个 adapter 目录**

没有看到以下能力：

- 同时加载多个 adapter
- 在运行时切换多个已注册 adapter
- adapter fusion / merge
- 基于角色 ID 自动匹配对应 adapter

因此，从训练产物设计上也可以确认：**当前是单 adapter 架构。**

### 4.4 训练配置与资源约束

正式训练配置面向 RTX 4060 8GB，主要特征是：

- `max_seq_length: 512`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 8`
- `gradient_checkpointing: true`
- `optim: paged_adamw_8bit`

这说明当前系统设计目标明确偏向：

- 单卡消费级 GPU 可训练
- 优先保证显存可用性
- 牺牲吞吐，换取训练可落地

### 4.5 训练数据监督方式

`scripts/data_loader.py` 中的 `encode_conversation()` 只对 assistant token 计算 loss：

- system 和 user 内容作为上下文输入
- assistant 回复作为监督目标

这属于标准的监督微调 SFT 形式，适用于多轮对话建模。

---

## 5. 数据架构

### 5.1 数据来源

当前默认数据集是：

- `KaraKaraWitch/PIPPA-ShareGPT-formatted`

### 5.2 数据清洗逻辑

数据加载脚本会过滤掉以下样本：

- 没有 bot 描述且没有 system 信息
- 没有完整 user/assistant 对话
- 总长度过长
- 重复样本

### 5.3 数据划分方式

数据切分不是简单随机，而是基于角色分组的 `GroupShuffleSplit`：

- 先按角色维度做 train / holdout
- 再把 holdout 切成 val / test

这样做的意义是：

- 尽量避免同一角色同时出现在训练集和测试集
- 让评测更接近“泛化到新角色”的场景

这也是当前架构里非常关键的一点。因为系统目标并不是记住某一个固定角色，而是学习“角色扮演对话模式”。

### 5.4 本地数据形态

处理后的数据写入：

- `processed/train.jsonl`
- `processed/val.jsonl`
- `processed/test.jsonl`

并附带：

- `processed/dataset_manifest.json`

manifest 用于记录 split 行数和哈希，便于实验复现。

---

## 6. 推理架构

### 6.1 推理加载路径

`scripts/inference.py` 的核心流程是：

1. 加载基础模型
2. 如果提供 adapter 目录，则调用 `PeftModel.from_pretrained()` 加载该 adapter
3. 读取角色卡 JSON
4. 把角色卡转成文本描述
5. 作为 `system` 消息，与用户输入一起送入 chat template

因此，当前推理依赖两类输入：

- 模型侧输入：base model，外加可选 LoRA adapter
- 提示侧输入：character card

### 6.2 当前不是“每角色一个 adapter”

这部分是本项目最容易被误解的地方。

虽然项目里有多个角色卡文件，例如：

- `alina.json`
- `gandalf.json`
- `harry_potter.json`
- `hermione.json`
- `luoji.json`

但这些角色卡只在推理时作为 prompt 使用，并不对应独立 adapter 文件。

也就是说，当前仓库并没有实现这种结构：

- `adapter_alina`
- `adapter_gandalf`
- `adapter_harry`

然后根据用户选择切换不同 adapter。

当前实际结构是：

- 一个 LoRA adapter 学“如何更像角色扮演模型”
- 多个角色卡定义“当前扮演谁”

### 6.3 对话状态管理现状

`interactive_chat()` 中虽然维护了 `conversation_history`，但 `chat()` 实际每次只接收：

- 当前角色卡
- 当前单轮 user_input

没有真正把完整多轮历史重新送回生成函数。

这意味着当前交互式推理在实现上更接近：

- **单轮角色扮演问答**

而不是严格意义上的：

- **带长期上下文记忆的多轮会话系统**

这点和评测脚本中的 multi-turn challenge 是有差异的。评测脚本会显式维护 `history`，而普通交互推理脚本当前没有完整复用历史。

### 6.4 推理生成策略

默认推理参数在 `configs/lora_config.yaml` 中，包括：

- `max_new_tokens`
- `temperature`
- `top_p`
- `repetition_penalty`

这部分是典型的采样式生成配置，没有引入检索、工具调用、函数调用或外部记忆模块。

---

## 7. 评测架构

### 7.1 三路对比设计

`scripts/eval.py compare` 明确比较三种系统：

1. `base_no_card`
   - 只有基础模型
   - 不加角色卡

2. `base_with_card`
   - 基础模型
   - 加角色卡

3. `lora_with_card`
   - 基础模型
   - 加 LoRA adapter
   - 加角色卡

这个设计非常重要，因为它把能力增益拆成两部分：

- 角色卡 prompt 本身带来的增益
- LoRA 微调在角色卡基础上继续带来的增益

### 7.2 自动指标

当前评测实现了以下自动指标：

- assistant perplexity
- Distinct-1 / Distinct-2
- repetition rate
- 响应长度
- tokens per second
- peak GPU memory
- empty/refusal rate

### 7.3 LLM Judge 评估

如果提供：

- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`
- `JUDGE_MODEL`

评测脚本还会对匿名化候选答案进行打分，分别衡量：

- 单轮：role identity、style、relevance、naturalness、immersion
- 多轮：role identity、memory、coherence、style、immersion

这说明当前评测架构已经不是简单看 loss，而是显式围绕“角色一致性”展开。

---

## 8. 资源管理与可复现性设计

### 8.1 资源解析

`scripts/resource_manager.py` 负责：

- 检查本地模型目录是否完整
- 检查 tokenizer、config 和所有权重分片
- 检查 `processed/` 下 train/val/test 是否完整
- 不完整时自动下载和重建

### 8.2 manifest 机制

项目中已经有两类 manifest：

1. 数据 manifest
   - 记录数据 split 和 sha256

2. 训练 run manifest
   - 记录 git commit
   - Python / Torch / CUDA
   - GPU 信息
   - 配置内容
   - 数据文件 sha256

这说明当前系统已经具备基础的实验可追踪能力。

---

## 9. 当前架构的优点

### 9.1 工程上简单清晰

当前架构是“单 base model + 单 adapter + 多角色卡”，优点是：

- 训练、推理、评测路径统一
- 不需要维护大量角色专属权重
- adapter 文件体积小，便于保存和迁移
- 角色切换成本低，只需要替换角色卡

### 9.2 硬件成本低

通过 4-bit + LoRA，项目可以在 RTX 4060 8GB 上完成训练，这对课程项目和原型验证非常重要。

### 9.3 评测设计较完整

三路对比架构让我们能够回答更具体的问题：

- 角色卡是否已经足够
- LoRA 是否真的带来额外收益
- 多轮一致性是否有提升

---

## 10. 当前架构的限制

### 10.1 不是多 adapter 架构

当前系统不能：

- 为不同角色分别训练和加载不同 adapter
- 在运行时动态切换多个 adapter
- 把多个 adapter 组合或融合

因此，如果未来目标变成“高保真复刻多个具体 IP 角色”，当前架构可能不够细粒度。

### 10.2 角色卡与 LoRA 职责耦合不够清晰

当前 LoRA 是在多角色角色扮演数据上训练出来的，角色卡又在推理阶段再次定义 persona。这样虽然实用，但也带来一个问题：

- LoRA 学到的是通用角色扮演风格增强
- 具体角色身份主要仍然由 prompt 决定

因此它更像“role-play capability adapter”，而不是“character identity adapter”。

### 10.3 交互式推理没有完整多轮记忆

当前 `scripts/inference.py` 的交互模式没有真正把完整历史传给生成函数，所以用户体验上可能会误以为系统支持连续多轮记忆，但实现上并不完整。

### 10.4 配置文件存在历史遗留

仓库中有一部分文档和配置还保留较早期设想，例如：

- `configs/lora_config.yaml` 中仍带有较旧的训练字段
- 现有 README / 旧报告对目录与配置的描述并不完全等同于当前实现

因此，项目对外说明时应优先以代码入口为准。

---

## 11. 回答核心问题：LoRA 微调是否是单一 adapter

答案是：**是，当前实现是单一 adapter 架构。**

证据有三层：

1. 训练脚本 `scripts/train.py`
   - 一次训练只构造一个 `LoraConfig`
   - 一次训练只输出一个 `final_model` adapter 目录

2. 推理脚本 `scripts/inference.py`
   - 只接收一个 `--adapter`
   - 只调用一次 `PeftModel.from_pretrained()`

3. 评测脚本 `scripts/eval.py`
   - 也只比较一个 adapter 路径对应的 `lora_with_card`

因此，当前系统不是：

- 多 adapter 并行
- 角色专属 adapter 集合
- adapter router
- mixture-of-adapters

而是：

**单一 LoRA adapter + 可切换角色卡**

---

## 12. 如果后续要升级，建议的演进方向

### 12.1 方向一：保持单 adapter，但强化推理层

适合当前项目阶段，成本最低。可以优先做：

- 修复交互式推理中的多轮历史传递
- 统一训练与推理的 chat template 格式
- 增强角色卡模板字段与示例对话利用率

这条路线下，LoRA 仍然作为通用角色扮演增强器存在。

### 12.2 方向二：扩展为多角色 adapter 架构

如果未来要面向“少量角色，但要求高拟合度”，可以改成：

- 每个角色单独训练一个 adapter
- 推理时根据角色选择对应 adapter

这种架构的优点是角色保真度更高，缺点是：

- 训练与存储成本上升
- 角色数量增加后管理复杂
- 评测矩阵会迅速变大

### 12.3 方向三：通用 adapter + 角色检索/记忆

如果未来目标是“大量角色、动态切换”，可以考虑：

- 继续保留单一通用 LoRA
- 外挂角色知识库或检索模块
- 把角色卡扩展成结构化 persona memory

这会比“每个角色一个 adapter”更容易扩展。

---

## 13. 最终结论

当前仓库已经形成一条完整、可运行的原型链路：

- 资源准备
- 数据清洗与切分
- QLoRA 训练
- 单 adapter 推理
- 三路对比评测

从代码事实看，项目当前的核心架构应准确描述为：

**基于 Qwen2.5-3B-Instruct 的单 LoRA adapter 角色扮演系统，使用多角色数据进行通用 role-play 能力微调，并在推理阶段通过角色卡注入具体 persona。**

所以，对“LoRA 微调是否是单一 adapter”的回答应当明确写成：

**是。当前是单一 adapter，不是多 adapter。角色切换主要依赖角色卡，而不是切换不同的 LoRA 权重。**
