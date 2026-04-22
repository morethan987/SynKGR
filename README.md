# SynKGR: Synthetic Knowledge Graph Reasoning with Discrimination

SynKGR 是一个基于 **蒙特卡洛树搜索（MCTS）+ 大语言模型判别器** 的知识图谱增强与推理框架。其核心思路是：

1. 识别知识图谱中的**稀疏实体**（连接数极少的实体）
2. 利用 **MCTS** 在候选实体空间中搜索，结合 **KGE 模型评分**、**图结构特征**和 **LLM 语义判别**三种策略逐步过滤候选
3. 使用微调后的 LLM（Alpaca-7B + KoPA 前缀 + LoRA）作为**三元组判别器**，筛选出高置信度的辅助三元组
4. 将辅助三元组加入原始知识图谱，通过 **CompGCN + ConvE** 模型（带自适应消息聚合和动态损失约束）训练最终的知识图谱嵌入
5. 在链接预测任务上评估 MRR、Hits@1/3/10 等指标

---

## 目录

- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [环境配置](#环境配置)
- [数据集说明](#数据集说明)
- [外部模型与数据获取](#外部模型与数据获取)
- [完整运行流程](#完整运行流程)
  - [Step 0: 编译 OpenKE C++ 扩展](#step-0-编译-openke-c-扩展)
  - [Step 1: 训练 KGE 基础模型 (RotatE)](#step-1-训练-kge-基础模型-rotate)
  - [Step 2: 生成实体嵌入向量 (entity2embedding)](#step-2-生成实体嵌入向量-entity2embedding)
  - [Step 3: 微调 LLM 判别器](#step-3-微调-llm-判别器)
  - [Step 4: MCTS 预处理](#step-4-mcts-预处理)
  - [Step 5: 运行 MCTS 搜索](#step-5-运行-mcts-搜索)
  - [Step 6: 准备辅助三元组](#step-6-准备辅助三元组)
  - [Step 7: 训练最终 KGE 模型 (CompGCN + ConvE)](#step-7-训练最终-kge-模型-compgcn--conve)
- [测试与评估](#测试与评估)
- [辅助工具](#辅助工具)
- [消融实验](#消融实验)
- [常见问题](#常见问题)

---

## 项目结构

```
SynKGR/
├── data/                          # 数据集目录
│   ├── FB15k-237N/                # FB15k-237N 数据集（已内置）
│   │   ├── train.txt              # 训练集三元组（87,282 条）
│   │   ├── valid.txt              # 验证集（7,041 条）
│   │   ├── test.txt               # 测试集（8,226 条）
│   │   ├── entity2id.txt          # 实体→ID 映射（14,541 个实体）
│   │   ├── relation2id.txt        # 关系→ID 映射（237 个关系）
│   │   ├── entity2name.txt        # 实体→名称映射
│   │   ├── entity2des.txt         # 实体→描述文本
│   │   ├── 1-1.txt / 1-n.txt / n-1.txt / n-n.txt  # 按关系类型分类的测试三元组
│   │   └── type_constrain.txt     # 类型约束
│   ├── CoDEx-S/                   # CoDEx-S 数据集（已内置）
│   │   ├── train.txt              # 训练集三元组（32,888 条）
│   │   ├── valid.txt              # 验证集（1,827 条）
│   │   ├── test.txt               # 测试集（1,828 条）
│   │   ├── entity2id.txt          # 实体→ID 映射（2,034 个实体）
│   │   ├── relation2id.txt        # 关系→ID 映射（42 个关系）
│   │   ├── entity2name.txt        # 实体→名称映射
│   │   └── entity2des.txt         # 实体→描述文本
│   ├── run_embedding.py           # 实体嵌入向量生成脚本
│   └── data_preview.py            # 数据预览与分析工具
│
├── openke/                        # OpenKE 知识图谱嵌入框架
│   ├── base/                      # C++ 底层实现（需编译）
│   ├── module/
│   │   ├── model/                 # KGE 模型（RotatE, TransE, ComplEx 等）
│   │   ├── loss/                  # 损失函数
│   │   └── strategy/              # 训练策略（负采样等）
│   ├── data/                      # 数据加载器
│   ├── config/                    # 训练器与测试器
│   └── make.sh                    # C++ 编译脚本
│
├── LLM_Discriminator/             # LLM 判别器模块
│   ├── finetune.py                # LLM + KoPA + LoRA 微调脚本
│   ├── kopa.py                    # KoPA（Knowledge Prefix Adapter）实现
│   ├── discriminator.py           # 三元组判别器（用于 MCTS 节点评估）
│   ├── process_kge.py             # 加载预训练 KGE 嵌入
│   ├── test_finetuned_llm.py      # 测试微调后 LLM 的准确率
│   ├── test_with_discriminator.py # 使用判别器进行批量测试
│   ├── inference.py               # 推理脚本
│   └── data/                      # 微调数据目录（需下载）
│
├── MCTS/                          # 蒙特卡洛树搜索模块
│   ├── run_mcts.py                # MCTS 主入口（分布式运行）
│   ├── preprocess.py              # MCTS 预处理（提取稀疏实体及关系）
│   ├── mcts_tree.py               # MCTS 树搜索算法实现
│   ├── node.py                    # 搜索节点定义（根节点/KGE节点/图节点/LLM节点）
│   ├── kg_enhancer.py             # 知识图谱增强器（协调 MCTS 搜索流程）
│   ├── kg_data_loader.py          # 知识图谱数据加载器
│   ├── model_calls.py             # OpenKE 模型调用客户端 & LocalLLM 调用
│   ├── rollout_policy.py          # Rollout 策略（UCB1/LinUCB/MomentumReward）
│   ├── prompts.py                 # LLM Prompt 模板
│   ├── logits_processor.py        # 二值输出处理器
│   ├── utils.py                   # 工具函数（设备检测、分布式、稀疏实体检测）
│   └── setup_logger.py            # 日志配置
│
├── loss_restraint_KGE_model/      # 最终 KGE 模型（带损失约束的 CompGCN）
│   ├── run.py                     # 主训练/测试入口
│   ├── models.py                  # CompGCN_TransE / CompGCN_DistMult / CompGCN_ConvE
│   ├── compgcn_conv.py            # CompGCN 图卷积层
│   ├── compgcn_conv_adapt.py      # 带自适应聚合的 CompGCN 卷积层
│   ├── data_loader.py             # 数据加载器
│   ├── helper.py                  # 辅助函数
│   └── message_passing.py         # 消息传递模块
│
├── scripts/                       # 运行脚本
│   ├── run_embedding.sh           # Step 2: 生成 entity2embedding
│   ├── train_kge.sh               # Step 1: 训练 RotatE 基础模型
│   ├── run_finetune_fb15k237n.sh  # Step 3: 微调 LLM（FB15k-237N）
│   ├── run_finetune_codex.sh      # Step 3: 微调 LLM（CoDEx-S）
│   ├── preprocess_mcts.sh         # Step 4: MCTS 预处理
│   ├── run_mcts.sh                # Step 5: 运行 MCTS 搜索
│   ├── train_loss_restrain_kge.sh # Step 7: 训练最终 KGE 模型
│   ├── test_finetuned_llm.sh      # 测试微调后 LLM
│   ├── test_relation_type.sh      # 按关系类型评估
│   ├── test_entity_degree.sh      # 按实体度数评估
│   └── test_case_study.sh         # 案例分析
│
├── pyproject.toml                 # Python 项目配置（uv 包管理）
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

---

## 环境要求

### 硬件要求

| 阶段 | 最低 GPU 要求 | 推荐 GPU | 说明 |
|------|-------------|---------|------|
| Step 1: 训练 RotatE | 1× GPU (≥16GB) | 1× GPU | CPU 亦可，但较慢 |
| Step 2: 生成嵌入 | 1× GPU | 1× GPU | SentenceTransformer，单卡即可 |
| Step 3: 微调 LLM | 8× GPU (≥16GB each) | 8× A100 | Alpaca-7B + LoRA，需分布式 |
| Step 4-5: MCTS | 1× GPU (≥24GB) | 1× A100 | 加载 7B 模型推理 |
| Step 7: 最终 KGE | 1× GPU (≥16GB) | 1× GPU | CompGCN + ConvE |

> 本项目同时支持 NVIDIA GPU（CUDA）和华为昇腾 NPU，代码会自动检测可用设备。

### 软件要求

- **操作系统**: Linux（推荐 Ubuntu 20.04+）
- **Python**: 3.10（严格版本，由 `.python-version` 指定）
- **CUDA**: 12.4（如使用 GPU）
- **C++ 编译器**: g++（用于编译 OpenKE 的 C++ 扩展）
- **包管理器**: [uv](https://docs.astral.sh/uv/)（推荐）或 pip

---

## 环境配置

### 方式一：使用 uv（推荐）

本项目使用 `uv` 作为包管理器，已配置好 `pyproject.toml` 和 `uv.lock`。

```bash
# 1. 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/morethan987/SynKGR.git
cd SynKGR

# 3. uv 会根据 .python-version (3.10) 自动创建虚拟环境并安装依赖
uv sync
```

### 方式二：使用 pip + venv

```bash
# 1. 确保 Python 3.10 已安装
python --version  # 应输出 Python 3.10.x

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装 PyTorch（CUDA 12.4 版本）
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 torch_scatter（需匹配 PyTorch 和 CUDA 版本）
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# 5. 安装其余依赖
pip install -r requirements.txt
```

### 主要 Python 依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.4.0 | 深度学习框架 |
| `transformers` | 4.28.0 | HuggingFace Transformers（LLM 加载与推理） |
| `peft` | 0.3.0 | LoRA 参数高效微调 |
| `datasets` | 2.18.0 | 数据集加载 |
| `accelerate` | 0.28.0 | 分布式训练 |
| `sentencepiece` | 0.2.0 | Tokenizer |
| `torch_geometric` | 2.6.1 | 图神经网络 |
| `torch_scatter` | 2.1.2 | 图散射操作 |
| `scikit_learn` | 1.4.1 | 机器学习工具 |
| `fire` | 0.6.0 | 命令行参数解析 |
| `ordered_set` | 4.1.0 | 有序集合 |

此外，运行 `data/run_embedding.py` 还需要 `sentence-transformers` 库（用于加载 `all-MiniLM-L6-v2` 模型），请手动安装：

```bash
pip install sentence-transformers
```

---

## 数据集说明

本项目使用两个标准知识图谱数据集，均已内置在 `data/` 目录中：

### FB15k-237N

FB15k-237N 是 FB15k-237 的一个扩展版本，增加了实体名称和描述信息。

| 统计项 | 数量 |
|--------|------|
| 实体数 | 14,541 |
| 关系数 | 237 |
| 训练三元组 | 87,282 |
| 验证三元组 | 7,041 |
| 测试三元组 | 8,226 |

**文件说明**：

| 文件 | 格式 | 说明 |
|------|------|------|
| `train.txt` | `head\trelation\tail` | 训练集 |
| `valid.txt` | `head\trelation\tail` | 验证集 |
| `test.txt` | `head\trelation\tail` | 测试集 |
| `entity2id.txt` | `entity\tid` | 实体到整数 ID 的映射 |
| `relation2id.txt` | `relation\tid` | 关系到整数 ID 的映射 |
| `entity2name.txt` | `entity\tname` | 实体到可读名称的映射 |
| `entity2des.txt` | `entity\tdescription` | 实体到自然语言描述的映射 |
| `1-1.txt` ~ `n-n.txt` | `head\trelation\tail` | 按关系映射类型分类的测试集子集 |
| `type_constrain.txt` | 自定义格式 | 关系的类型约束 |

### CoDEx-S

CoDEx-S 是一个小规模知识图谱数据集，适合快速验证。

| 统计项 | 数量 |
|--------|------|
| 实体数 | 2,034 |
| 关系数 | 42 |
| 训练三元组 | 32,888 |
| 验证三元组 | 1,827 |
| 测试三元组 | 1,828 |

---

## 外部模型与数据获取

在运行完整流程之前，需要额外获取以下外部资源：

### 1. Alpaca-7B 基座模型

微调 LLM 判别器需要 **Alpaca-7B** 模型。可通过 HuggingFace 获取：

```bash
# 方式一：通过 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download wxjiao/alpaca-7b --local-dir /path/to/Alpaca-7B

# 方式二：在 Python 中使用 transformers 自动下载
# 首次运行时模型会自动缓存到 ~/.cache/huggingface/
```

下载后，将模型放置到本地路径（例如 `/path/to/Alpaca-7B`），后续脚本中的 `MODEL_PATH` 需要指向该路径。

> **注意**：脚本中默认使用的模型路径为 `wxjiao/alpaca-7b`（HuggingFace 自动下载）或本地路径如 `/home/ma-user/work/model/Alpaca-7B`。请根据实际情况修改脚本中的 `MODEL_PATH` 变量。

### 2. LLM 微调数据

微调 LLM 判别器需要额外的大规模 JSON 数据文件，**未包含在仓库中**。请从以下链接下载：

- **下载地址**：[Google Drive - data.zip](https://drive.google.com/file/d/1J1Ioi23jTMaBkBDYzfIy2MAZYMUIjFWW/view)

下载并解压后，将文件放置到 `LLM_Discriminator/data/` 目录下。解压后应包含：

```
LLM_Discriminator/data/
├── FB15k-237N-test.json        # FB15k-237N 微调/测试数据
├── CoDeX-S-test.json           # CoDEx-S 微调/测试数据
├── FB15k-237N-rotate.pth       # FB15k-237N 的 RotatE 预训练模型（用于 KoPA 初始化）
└── CoDeX-S-rotate.pth          # CoDEx-S 的 RotatE 预训练模型（用于 KoPA 初始化）
```

> **重要**：这些 `.pth` 文件中的 RotatE 模型与 Step 1 训练的模型可能不同。`LLM_Discriminator/data/` 中的模型是用于 KoPA 前缀嵌入初始化的，而 Step 1 训练的模型是用于 MCTS 中的 OpenKE 评分的。如果你的 Step 1 产出与下载文件中的模型参数（维度等）一致，可以复用。

### 3. SentenceTransformer 模型

生成实体嵌入需要 `all-MiniLM-L6-v2` 模型。首次运行 `run_embedding.sh` 时会自动从 HuggingFace 下载（约 80MB）。如果运行环境无法联网，请提前下载：

```bash
# 在有网络的环境中
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# 模型会缓存到 ~/.cache/huggingface/ 或 ~/.cache/torch/sentence_transformers/
```

---

## 完整运行流程

以下按顺序说明从零开始到产出最终结果的完整流程。

### 整体流水线概览

```
Step 0: 编译 OpenKE C++ 扩展
  ↓
Step 1: 训练 RotatE 基础模型 ──→ 产出: LLM_Discriminator/data/{DATASET}-rotate.pth
  ↓
Step 2: 生成实体嵌入向量 ──→ 产出: data/{DATASET}/entity2embedding.pth
  ↓
Step 3: 微调 LLM 判别器 ──→ 产出: LLM_Discriminator/output/alpaca7b_{DATASET}/
  ↓
Step 4: MCTS 预处理 ──→ 产出: MCTS/output/{DATASET}/processed_data.pth
  ↓
Step 5: 运行 MCTS 搜索 ──→ 产出: MCTS/output/{DATASET}/discovered_triplets.txt
  ↓
Step 6: 准备辅助三元组 ──→ 产出: data/{DATASET}/auxiliary_triples.txt
  ↓
Step 7: 训练最终 KGE 模型 ──→ 产出: loss_restraint_KGE_model/output/{DATASET}/
```

> **注意**：以下步骤中，以 `fb15k-237n` 为例。如需使用 `CoDEx-S` 数据集，请修改脚本中的相关变量（`DATA_PATH`、`DATA_SET` 等），各脚本中已用注释标注了 CoDEx-S 的配置。

---

### Step 0: 编译 OpenKE C++ 扩展

OpenKE 框架依赖 C++ 扩展进行高效的数据处理，需要先编译：

```bash
cd openke
bash make.sh
cd ..
```

**产出**：`openke/release/Base.so`（C++ 共享库）

**验证**：确认 `openke/release/Base.so` 文件已生成。

---

### Step 1: 训练 KGE 基础模型 (RotatE)

此步骤训练一个 RotatE 模型，其用途有二：
1. 提供 KGE 嵌入用于 KoPA 初始化（Step 3）
2. 在 MCTS 搜索中作为评分模型（Step 5）

**运行命令**：

```bash
bash scripts/train_kge.sh
```

**脚本配置**（`scripts/train_kge.sh`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `MODEL_NAME` | RotatE | 模型类型 |
| `DIMENSION` | 512 | 嵌入维度 |
| `NEGATIVE_SAMPLES` | 32 | 负采样数量 |
| `MARGIN` | 6.0 | Margin 损失的间隔参数 |
| `BATCH_SIZE` | 2048 | 批大小 |
| `LEARNING_RATE` | 2e-5 | 学习率 |
| `EPOCHS` | 3000 | 训练轮数 |

**产出**：

```
LLM_Discriminator/train_kge_output/
└── {TIMESTAMP}/
    └── checkpoint.pth          # RotatE 模型检查点
```

训练完成后，需将模型文件复制/重命名为 `LLM_Discriminator/data/{DATASET}-rotate.pth`。

> **或者**：如果从 Google Drive 下载的 `data.zip` 中已包含 `{DATASET}-rotate.pth`，可跳过此步骤。但需确保模型维度与脚本参数一致（`rel_embeddings.weight` 维度为 512）。

**日志**：`LLM_Discriminator/logs/train_kge_RotatE_{TIMESTAMP}.log`

---

### Step 2: 生成实体嵌入向量 (entity2embedding)

使用 SentenceTransformer 模型（`all-MiniLM-L6-v2`）将每个实体的描述文本编码为向量，用于 MCTS 搜索中 LLM 节点的语义过滤。

**运行命令**：

```bash
bash scripts/run_embedding.sh
```

**脚本配置**（`scripts/run_embedding.sh`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `MODEL_PATH` | `all-MiniLM-L6-v2` | SentenceTransformer 模型名 |
| `DATA_PATH` | `data/CoDEx-S` | 数据集路径（按需修改） |
| `--device` | `cuda:2` | 计算设备 |
| `--batch_size` | 16 | 编码批大小 |

> **注意**：默认配置为 CoDEx-S。若处理 FB15k-237N，需修改脚本中的 `DATA_PATH` 为 `data/FB15k-237N`。

**产出**：

```
data/{DATASET}/entity2embedding.pth
```

这是一个 Python 字典文件，格式为 `{entity_id: Tensor}`，其中每个 Tensor 是 384 维的向量（MiniLM-L6-v2 的输出维度）。

**日志**：`data/logs/get_embedding_{TIMESTAMP}.log`

---

### Step 3: 微调 LLM 判别器

使用 LoRA + KoPA（Knowledge Prefix Adapter）对 Alpaca-7B 进行微调，使其成为一个知识图谱三元组判别器，能够判断给定三元组是否正确。

**前置条件**：
- Alpaca-7B 基座模型已下载
- `LLM_Discriminator/data/` 中已有微调 JSON 数据和 RotatE 模型文件

**运行命令**：

```bash
# FB15k-237N
bash scripts/run_finetune_fb15k237n.sh

# CoDEx-S
bash scripts/run_finetune_codex.sh
```

**脚本配置**（以 `run_finetune_fb15k237n.sh` 为例）：

| 参数 | 值 | 说明 |
|------|----|------|
| `MODEL_PATH` | 本地 Alpaca-7B 路径 | 基座 LLM |
| `DATA_PATH` | `data/FB15k-237N-test.json` | 微调数据 |
| `OUTPUT_DIR` | `output/alpaka_7b_fb` | 输出目录 |
| `KGE_MODEL` | `data/FB15k-237N-rotate.pth` | RotatE 模型（KoPA 初始化） |
| `num_epochs` | 3 | 训练轮数 |
| `lora_r` | 64 | LoRA 秩 |
| `learning_rate` | 3e-4 | 学习率 |
| `batch_size` | 8 | 等效批大小 |
| `micro_batch_size` | 8 | 单卡批大小 |
| `num_prefix` | 1 | KoPA 前缀数量 |
| `lora_target_modules` | q,k,v,o_proj | LoRA 目标模块 |

此脚本使用 `torchrun` 进行**多卡分布式训练**，默认使用 8 张 GPU。请根据实际 GPU 数量修改 `CUDA_VISIBLE_DEVICES` 和 `WORLD_SIZE`。

**产出**：

```
LLM_Discriminator/output/alpaka_7b_fb/     # FB15k-237N
├── adapter_model.bin                       # LoRA 权重
├── adapter_config.json                     # LoRA 配置
├── embeddings.pth                          # KoPA 嵌入（关键文件，Step 5 需要）
└── ...

LLM_Discriminator/output/alpaca7b_CoDeX-S/  # CoDEx-S
├── adapter_model.bin
├── adapter_config.json
├── embeddings.pth
└── ...
```

> `embeddings.pth` 包含训练后的 `PretrainKGEmbedding` 模块（含 adapter 线性层和冻结的 KGE 嵌入），是 MCTS 中判别器的核心组件。

**日志**：`logs/finetune_{TIMESTAMP}.log`

---

### Step 4: MCTS 预处理

从训练集中识别**稀疏实体**（出现频率低于阈值的实体），并提取每个稀疏实体参与的（位置, 关系）对，为 MCTS 搜索做准备。

**运行命令**：

```bash
bash scripts/preprocess_mcts.sh
```

**脚本配置**（`scripts/preprocess_mcts.sh`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `--data_folder` | `data/CoDEx-S` | 数据集路径 |
| `--output_path` | `MCTS/output/codex-s` | 输出路径 |
| `--threshold` | `5e-4` | 稀疏实体的频率阈值 |

> **阈值说明**：
> - **FB15k-237N** 使用 `9e-5`
> - **CoDEx-S** 使用 `5e-4`
>
> 阈值越低，筛选出的稀疏实体越少（只保留出现频率极低的实体）。

**产出**：

```
MCTS/output/{DATASET}/processed_data.pth
```

这是一个 Python 字典文件，格式为：

```python
{
    entity_id_1: [(position_1, relation_1), (position_2, relation_2), ...],
    entity_id_2: [(position_1, relation_1), ...],
    ...
}
```

其中 `position` 为 `'head'` 或 `'tail'`，表示该稀疏实体在三元组中的位置。

**示例**：
- 若训练集中存在三元组 `(sparse_ent_A, relation_X, entity_B)`，则 `entity_id_A` 对应列表中有 `('head', 'relation_X')`
- 若训练集中存在三元组 `(entity_C, relation_Y, sparse_ent_A)`，则 `entity_id_A` 对应列表中有 `('tail', 'relation_Y')`

---

### Step 5: 运行 MCTS 搜索

对每个稀疏实体的每个（位置, 关系）组合，执行蒙特卡洛树搜索，发现新的辅助三元组。

**运行命令**：

```bash
bash scripts/run_mcts.sh
```

**脚本配置**（`scripts/run_mcts.sh`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `MODEL_PATH` | `wxjiao/alpaca-7b` | Alpaca-7B 模型路径 |
| `DATA_PATH` | `data/CoDEx-S` | 数据集路径 |
| `OUTPUT_DIR` | `MCTS/output/codex-s` | 输出路径 |
| `LORA_PATH` | `LLM_Discriminator/output/alpaca7b_CoDeX-S` | LoRA 权重路径 |
| `EMBEDDING_PATH` | `{LORA_PATH}/embeddings.pth` | KoPA 嵌入路径 |
| `ENTITY2EMBEDDING_PATH` | `data/CoDEx-S/entity2embedding.pth` | SentenceTransformer 嵌入 |
| `KGE_MODEL` | `LLM_Discriminator/data/CoDeX-S-rotate.pth` | RotatE 模型路径 |
| `--exploration_weight` | 1.0 | UCT 探索权重 $c$ |
| `--leaf_threshold` | 32 | 叶节点候选实体阈值 |
| `--mcts_iterations` | 10 | 每个实体的 MCTS 迭代次数 |
| `--budget_per_entity` | 200 | 每个实体的判别器调用预算 |
| `--checkpoint_interval` | 1 | 每处理 N 个实体保存一次检查点 |
| `--without_llm` | flag | 是否禁用 LLM 判别器（消融实验用） |

**MCTS 搜索流程**：

1. 对每个稀疏实体，将所有候选实体（除自身外）作为初始候选集
2. 构建搜索树根节点，扩展出 3 种过滤策略子节点：
   - **GraphNode**：基于图结构（Jaccard 相似度、关系热度）过滤，保留 Top 50%
   - **KGENode**：基于 RotatE 模型评分过滤，保留 Top 30%
   - **LLMNode**：基于 SentenceTransformer 语义嵌入相似度过滤，保留 Top 30%
3. 使用 UCT 策略进行 Selection，Rollout 阶段使用 `MomentumRewardPolicy` 策略选择过滤路径
4. 到达叶节点（候选数 ≤ `leaf_threshold`）后，使用微调后的 LLM 判别器逐个判断三元组正确性
5. 收集所有被判别为正确的三元组

**产出**：

```
MCTS/output/{DATASET}/
├── discovered_triplets.txt      # 发现的辅助三元组（核心产出）
└── checkpoints/
    ├── checkpoint_rank_0.json   # 各进程的检查点
    └── ...
```

`discovered_triplets.txt` 格式为每行一个三元组：

```
head_entity\trelation\tail_entity
```

**检查点机制**：MCTS 支持断点续跑。如果程序中断，重新运行相同的脚本即可从检查点恢复已处理的实体。检查点还保存了 Rollout 策略的状态。

**日志**：`MCTS/logs/codex_{TIMESTAMP}.log` 或 `MCTS/logs/fb15k_{TIMESTAMP}.log`

> **运行时间**：此步骤非常耗时（可能需要数小时至数天，取决于稀疏实体数量和预算设置）。

---

### Step 6: 准备辅助三元组

MCTS 搜索完成后，需要将产出的辅助三元组移动到数据集目录中：

```bash
# 以 FB15k-237N 为例
cp MCTS/output/fb15k-237n/discovered_triplets.txt data/FB15k-237N/auxiliary_triples.txt

# 以 CoDEx-S 为例
cp MCTS/output/codex-s/discovered_triplets.txt data/CoDEx-S/auxiliary_triples.txt
```

**产出**：

```
data/{DATASET}/auxiliary_triples.txt
```

该文件格式与 `train.txt` 相同（`head\trelation\tail`），在 Step 7 中会被自动加载并与原始训练集合并。

---

### Step 7: 训练最终 KGE 模型 (CompGCN + ConvE)

使用 CompGCN（Composition-based Graph Convolutional Network）结合 ConvE 打分函数，在增强后的知识图谱上训练最终的 KGE 模型。模型具有两个关键特性：

1. **自适应消息聚合**：通过注意力机制为原始边和辅助边分配不同权重
2. **动态损失约束**：随着训练进行，逐步丢弃高损失的辅助三元组（减少噪声影响）

**运行命令**：

```bash
bash scripts/train_loss_restrain_kge.sh
```

**脚本配置**（`scripts/train_loss_restrain_kge.sh`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `DATA_SET` | `CoDEx-S` | 数据集名称 |
| `NAME` | `codex_train` | 实验名称 |
| `--mode` | `train` | 运行模式 |
| `--score_func` | `conve` | 打分函数（ConvE） |
| `--opn` | `corr` | CompGCN 组合操作（Corr - 相关性） |
| `--adapt_aggr` | `1` | 启用自适应消息聚合 |
| `--loss_delta` | `0.002` | 损失约束增长速率 $\eta$ |
| `--keep_aux` | `True` | 保留辅助三元组 |
| `--batch` | `256` | 批大小 |
| `--lr` | `5e-4` | 学习率 |
| `--epoch` | `500` | 最大训练轮数 |
| `--gpu` | `1` | GPU 编号 |

**关键参数说明**：

- **`loss_delta`**：控制辅助三元组丢弃速率。每 30 个 epoch，`clean_rate` 会减少 `loss_delta`。当某条辅助三元组的损失排名超过 `clean_rate` 比例时，该三元组会被丢弃。
  - `0.002`：正常训练速度
  - `-1`：禁用辅助三元组（不加载 `auxiliary_triples.txt`，等同于基线）
- **`adapt_aggr`**：
  - `1`：启用自适应消息聚合
  - `-1`：禁用（标准 CompGCN）
- **`keep_aux`**：
  - `True`：保留辅助三元组
  - `False`：用于消融实验（不增强知识图谱）

**产出**：

```
loss_restraint_KGE_model/output/{DATASET}/
├── {NAME}_{TIMESTAMP}.pth       # 模型检查点（含模型权重、优化器状态、最佳验证结果）
├── entity2id.txt                # 实体→ID 映射
└── relation2id.txt              # 关系→ID 映射
```

**训练日志**（在控制台和日志文件中）会输出：
- 每 100 步的训练损失和最佳验证 MRR
- 每 30 个 epoch 的测试集评估结果：MRR、MR、Hits@1/3/10
- 最终测试集结果

**日志**：`loss_restraint_KGE_model/logs/codex_{TIMESTAMP}.log`

**训练策略**：
- 每 30 个 epoch 更新 `clean_rate = clean_rate - loss_delta`，加速噪声辅助三元组的淘汰
- 验证集 MRR 连续 25 个 epoch 未提升时触发 Early Stopping
- 验证集 MRR 连续 10 个 epoch 未提升时，衰减 gamma（Margin）

---

## 测试与评估

训练完成后，可以使用以下脚本进行多种评估：

### 整体测试

使用已保存的最佳模型在测试集上进行评估：

```bash
# 修改 scripts/train_loss_restrain_kge.sh 中的参数：
# --mode overall
# --restore
# --name {训练时的实验名称}
```

### 按关系类型评估

评估模型在不同关系映射类型（1-1、1-n、n-1、n-n）上的表现：

```bash
bash scripts/test_relation_type.sh
```

需要修改脚本中的 `--name` 为训练时的实验名称。

### 按实体度数评估

评估模型在不同度数区间实体上的表现（分析对稀疏实体的改善效果）：

```bash
bash scripts/test_entity_degree.sh
```

实体按训练集度数分为 8 个区间：`[0,10), [10,15), [15,20), [20,50), [50,100), [100,200), [200,350), >=350`。

### 案例分析

对具体三元组进行预测，展示模型的推理能力：

```bash
bash scripts/test_case_study.sh
```

输出每个案例的 Raw Rank、Filtered Rank 和 Top-5 预测结果。

### 测试微调后 LLM

评估微调后 LLM 判别器在测试数据上的准确率：

```bash
bash scripts/test_finetuned_llm.sh
```

---

## 辅助工具

### 数据预览与分析（`data/data_preview.py`）

提供多种数据分析和可视化工具：

```python
from data.data_preview import *

# 检查 entity2embedding.pth 文件
check_enity2embedding("data/FB15k-237N/entity2embedding.pth")

# 计算两个 KG 文件的 Jaccard 相似度
kg_similarity("data/FB15k-237N/test.txt", "data/FB15k-237N/auxiliary_triples.txt")

# 查看实体度数
get_degree("data/FB15k-237N/train.txt", "/m/0m0bj", "data/FB15k-237N/auxiliary_triples.txt")

# 检查辅助三元组在测试集中的命中情况
check_in_test(load_data("auxiliary_triples.txt"), load_data("test.txt"))

# 统计实体度数分布
get_entity_degree_distribution('data/FB15k-237N')
```

---

## 消融实验

通过修改脚本参数可以进行以下消融实验：

### 1. 禁用 LLM 判别器

在 `scripts/run_mcts.sh` 中保留 `--without_llm` 参数。此时 MCTS 叶节点评估将使用随机判断（50% 概率标记为正确），替代 LLM 判别器。

### 2. 禁用辅助三元组

在 `scripts/train_loss_restrain_kge.sh` 中设置 `--loss_delta -1`。此时训练不会加载 `auxiliary_triples.txt`。

### 3. 禁用自适应消息聚合

在 `scripts/train_loss_restrain_kge.sh` 中设置 `--adapt_aggr -1`。

### 4. 丢弃辅助三元组但保留损失约束

设置 `--keep_aux False` 来测试不增强知识图谱的效果。

---

## 常见问题

### Q1: OpenKE 编译失败

确保系统安装了 `g++` 和相关的 C++ 开发库。编译命令：

```bash
cd openke && bash make.sh && cd ..
```

如果出现编译错误，检查 `g++ --version` 是否 ≥ 7.0。

### Q2: torch_scatter 安装失败

`torch_scatter` 需要匹配 PyTorch 和 CUDA 版本。本项目使用：
- PyTorch 2.4.0 + CUDA 12.4
- torch_scatter 2.1.2

安装命令：
```bash
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

### Q3: GPU 显存不足

- **Step 3（微调 LLM）**：减少 `micro_batch_size`，或增加 GPU 数量。最低配置为单卡，`micro_batch_size=1`，`gradient_accumulation_steps=64`
- **Step 5（MCTS）**：使用 `--dtype fp16` 减少模型显存占用

### Q4: MCTS 中途崩溃如何恢复

MCTS 支持检查点恢复。确保 `MCTS/output/{DATASET}/checkpoints/` 目录下有检查点文件，重新运行相同脚本即可自动恢复。

### Q5: 如何切换数据集

所有脚本中都包含 FB15k-237N 和 CoDEx-S 两种配置（另一数据集的配置已注释）。取消注释对应的配置并注释掉当前配置即可。

### Q6: 如何在 NPU 上运行

代码已内置 NPU 支持。设置环境变量 `ASCEND_VISIBLE_DEVICES` 并确保安装了 `torch_npu`。脚本会自动检测设备类型。

### Q7: 关于 sentence-transformers 依赖

`requirements.txt` 中未列出 `sentence-transformers`，但 `data/run_embedding.py` 依赖它。请手动安装：

```bash
pip install sentence-transformers
```
