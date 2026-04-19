# 机器学习项目提案：基于 LoRA 微调的小型 LLM 角色扮演对话生成

## 1. 选题背景
大语言模型（LLM）在开放域对话中表现出色，但在角色扮演（Roleplay）场景下——要求模型持续扮演一个具有特定人设、背景故事、说话风格的虚拟角色——通用模型往往缺乏角色一致性，容易“出戏”。

本项目探索如何通过 **LoRA (Low-Rank Adaptation)** 对小型开源 LLM（1.5B - 4B 参数）进行高效微调，使其具备稳定的角色扮演能力，适用于游戏 NPC 对话、互动叙事、虚拟陪伴等场景。

LoRA 通过冻结原始模型权重、仅训练低秩分解矩阵来实现高效微调，显存占用极低，Google Colab 免费 GPU 即可完成全部训练流程。

## 2. 任务定义
给定一个角色描述（包括姓名、人设、背景故事、说话风格等）作为 `system prompt`，模型需要在多轮对话中始终保持该角色的身份，生成符合角色性格和语言风格的回复。

### 应用场景
| 设定 | 描述 | 应用场景 |
| :--- | :--- | :--- |
| **通用角色扮演** | 模型能根据任意角色卡（Character Card）进入角色并持续对话 | 社交 AI、虚拟角色 |
| **游戏 NPC 对话** | 聚焦于 RPG 游戏中的 NPC，需结合游戏世界观和任务系统生成对话 | 游戏开发、互动叙事 |
| **多角色场景对话** | 在同一场景中扮演多个不同角色，保持各自性格差异 | 剧本生成、互动小说 |

## 3. 推荐基座模型
| 模型 | 参数量 | 来源 |
| :--- | :--- | :--- |
| **Qwen2.5-1.5B-Instruct** | 1.5B | HuggingFace |
| **Qwen2.5-3B-Instruct** | 3B | HuggingFace |
| **Llama-3.2-3B-Instruct** | 3B | HuggingFace |
| **Phi-3.5-mini-instruct** | 3.8B | HuggingFace |

## 4. 推荐数据集
### 4.1 通用角色扮演对话
*   **PIPPA**: 约 16,000 条对话。最知名的角色扮演数据集，含角色描述和多轮对话。
*   **PIPPA-ShareGPT**: PIPPA 的 ShareGPT 格式版本，可直接用于 Axolotl 等微调框架。
*   **hieunguyenminh/roleplay**: 5,000+ 条。由 Gemini 生成的多样化原创角色对话。
*   **Gryphe-Aesir-RPG**: RPG 角色卡对话数据集，适合奇幻/RPG 风格微调。

### 4.2 游戏 NPC 对话
*   **amaydle/npc-dialogue**: 1,915 条。NPC 角色传记 + 对话，含角色情绪标注。
*   **dprashar/npc_dialogue_rpg_quests**: RPG 任务对话数据集。
*   **video-game-text-corpora**: 含《星球大战：旧共和国》、《上古卷轴》系列等经典游戏的 NPC 对话。

### 4.3 多角色场景对话
*   **agentlans/multi-character-dialogue**: 10,000+ 条。含场景设定和角色描述，涵盖奇幻、科幻、日常等多种场景。
