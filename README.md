# LLM Knowledge Base

LLM 训练与推理技术知识库，由 Claude Code Agent 维护。

## 规模

- **99** 篇原始论文（`raw/`）
- **102** 个 Wiki 页面（`wiki/`）

## 结构

```
raw/  和  wiki/  目录镜像
├── 01_theory/                  # 理论研究
│   ├── 01_models/              # 模型架构 + 模型家族 (DeepSeek/Kimi/GLM)
│   ├── 02_pretraining/         # 预训练：优化器、低精度、初始化
│   ├── 03_sft/                 # SFT + 低参微调 (LoRA/PEFT)
│   ├── 04_posttraining/        # 后训练对齐：RLHF、DPO、GRPO
│   └── 05_inference/           # 推理技术：CoT、RAG、Agent
└── 02_engineering/             # 工程实现
    ├── 01_ai_frameworks/       # AI框架：PyTorch compile、CUDA Graphs
    ├── 02_train_frameworks/    # 训练框架：Megatron-LM、分布式
    ├── 03_infer_frameworks/    # 推理框架：vLLM、TRT-LLM
    └── 04_posttrain_frameworks/ # 后训练框架 (预留)
```

## 使用

知识页面之间通过 `[[wiki link]]` 交叉引用，支持 Obsidian 打开浏览。

```
cd llm-knowledge
claude       # 启动后直接提问，Agent 自动检索 wiki
```

## 维护

由 Claude Code Agent 按 [CLAUDE.md](CLAUDE.md) 中定义的 Ingest Workflow 自动维护：

1. 新论文放入 `raw/` → Agent 自动读取并生成 wiki 页面
2. 更新领域 `index.md` 索引
3. 追加 `changelog.md` 记录
