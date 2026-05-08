# LLM Knowledge Base

LLM 训练与推理技术知识库，由 Claude Code Agent 维护。

## 规模

- **99** 篇原始论文（`raw/`）
- **102** 个 Wiki 页面（`wiki/`）

## 结构

```
raw/                          # 原始论文（只读）
├── 01_architecture/          # Transformer、Scaling Laws
├── 02_training/              # 优化器、训练技巧
├── 03_alignment/             # RLHF、DPO、GRPO
├── 04_reasoning_and_retrieval/ # CoT、RAG
├── 05_model_families/        # 各厂商模型技术报告
├── 06_moe_and_distributed/   # MoE、分布式训练
├── 07_multimodal/            # 视觉-语言、音频-语言
├── 08_agents/                # Agent、工具使用
└── 09_pytorch/               # PyTorch 源码分析

wiki/                         # Agent 生成的知识页面
├── index.md                  # 总索引
├── changelog.md              # 收录日志
├── llm/                      # LLM 训练与优化
│   ├── 01_architecture/
│   ├── 02_training/
│   ├── 03_alignment/
│   ├── 04_reasoning_and_retrieval/
│   ├── 05_model_families/    # DeepSeek / Kimi / GLM
│   ├── 06_infra/             # Megatron-LM 等
│   ├── 07_multimodal/
│   └── 08_agents/
└── torch_compile/            # PyTorch 编译栈
    ├── cudagraphs/           # CUDA/NPU Graphs
    └── inductor/             # TorchInductor
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
