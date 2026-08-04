# LLM Knowledge Base

LLM 训练与推理技术知识库，由 Claude Code Agent 维护。

## 结构

```
raw/            # 原始源材料（论文 PDF / 文章 / 图表源），只读；与 wiki/ 目录不要求镜像对齐
wiki/           # 生成的分析页（Obsidian vault），功能分类树是唯一内容权威
├── 01_theory/        # 理论：模型家族 / 预训练 / SFT / 后训练对齐 / 推理技术 / 分布式并行
├── 02_engineering/   # 工程：AI框架 / 训练框架 / 推理框架 / 后训练框架 / GPU Kernel / 自动并行 / 训练可靠性
├── courses/          # 纯导读层：torch_compile_end_to_end / posttraining_frontier 两条学习路线（只含阅读顺序+链接+一句话导读，不承载正文）
├── index.md          # 总索引（域级表格）
└── changelog.md      # 当季变更日志；历史条目按季度归档于 wiki/changelog/
docs/           # 流程文档（specs / plans / research）
tools/          # 维护工具：check_links.py（链接健康检查）、figs/ + html2md/（图表源与再生脚本）、labs_torch_compile/（demo）；详见 tools/README.md
```

各域页面清单见 [wiki/index.md](wiki/index.md)。

## 使用

页面间用 `[[wiki link]]` 交叉引用，Obsidian 打开 `wiki/` 浏览；或 `cd llm-knowledge && claude` 直接提问。

## 维护

按 [CLAUDE.md](CLAUDE.md) 定义的 Workflow 由 Agent 维护。当前结构整改：
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`。
