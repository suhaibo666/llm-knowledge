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

## 本地文档站点

需要 Node.js ≥ 22、npm ≥ 10.9.2 和 Git ≥ 2。在仓库根目录运行：

```bash
npm run docs
```

命令会在浏览器打开 `http://127.0.0.1:8080`，并监听 Markdown 变化。首次运行需要联网下载已锁定版本的 Quartz、社区插件和 Mermaid；成功安装后，日常启动与构建均复用 `.cache/llm-knowledge-docs/` 下的仓库私有运行时，不再访问包仓库或 CDN。

常用命令：

```bash
npm run docs -- --port 8088  # HTTP 使用 8088，热更新 WebSocket 使用 8089
npm run docs -- --no-open    # 启动但不自动打开浏览器
npm run docs:build           # 仅生成静态站点
npm run docs:test            # 单元测试 + 本地浏览器端到端验收
npm run docs:repair          # 显式重建损坏或版本漂移的私有运行时
```

HTTP 和热更新端口都只绑定 `127.0.0.1`，不向局域网暴露。站点直接读取且只展示 `wiki/`；它不会复制、格式化或改写任何 Markdown，Obsidian wikilink、callout、Mermaid 与公式兼容均由站点层处理。

也可以继续用 Obsidian 打开 `wiki/` 浏览，或 `cd llm-knowledge && claude` 直接提问。

## 维护

按 [CLAUDE.md](CLAUDE.md) 定义的 Workflow 由 Agent 维护。当前结构整改：
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`。
