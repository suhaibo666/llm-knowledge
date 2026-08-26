# LLM Knowledge Wiki — 基本法

本文件只定义这个知识库**是什么**：它的分层、内容权威、溯源要求和质量门禁。
**具体的文档操作不写在这里**——那些落在 [`skills/`](skills/README.md)，按需加载。

任何 agent 在动手前读完本文件即可；要真正写页面时，再按下表打开对应技能。

## 技能索引（按需加载，不默认加载）

| 你要做的事 | 读 |
|---|---|
| 往 `wiki/` 新增/改写/重命名/合并页面，维护 index 与 changelog | [`maintaining-llm-knowledge`](skills/maintaining-llm-knowledge/SKILL.md) |
| 写或改 LaTeX 公式 | [`writing-obsidian-math`](skills/writing-obsidian-math/SKILL.md) |
| 画或改 Mermaid 图 | [`writing-mermaid-diagrams`](skills/writing-mermaid-diagrams/SKILL.md) |
| 把一份新的源拆解成 wiki 页 | [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md) |

技能只有这一份公共副本，`.claude/` 与 `.codex/` 都指向它；索引与维护约定见
[`skills/README.md`](skills/README.md)。

## Architecture

This wiki has three layers:

1. **Raw sources** (`raw/`) — 文档类源材料的**索引层**：论文的 arXiv/官方链接说明页、文章、图表源。只读，agent 从中取来源，不改写。
2. **Wiki** (`wiki/`) — LLM 生成的 markdown。这一层完全由 agent 拥有：创建页面、更新、维护交叉引用、保持一致。
3. **基本法**（本文件）— 定义结构与权威；**怎么做**在 `skills/`。

`raw/` 和 `wiki/` 是**互相独立的两棵树**——不要求目录对目录镜像。多数 `wiki/` 域的源是外部代码仓库（以旁置 checkout 引用，不复制进 `raw/`），而不是 `raw/` 里的某个子树。什么该进 `raw/` 见下方 Provenance Policy。

## Directory Layout

- 功能分类树（`wiki/01_theory/`、`wiki/02_engineering/`）是**唯一的内容权威**。目录结构以文件系统与各级 `index.md` 为准；**不要**在 `CLAUDE.md` 或 `README.md` 里维护一份平行的深层 ASCII 树——它会立刻和实际目录漂移。
- 每个目录的 `index.md` **只维护本目录的条目表**（页面列表 + 一句话 + 状态/段位），不越权描述子孙目录的内部细节，也不需要画出多级嵌套树。
- `wiki/index.md`（总索引）**只保留域级表格**（入口 + 一句话 + 页面数 + 状态），不重复各级 `index.md` 已有的内容，不画深层目录树。页面数按 `find`/`rglob` 现算，随内容漂移及时校正，不长期沿用一次性统计的旧数字。
- `README.md` 面向仓库外的读者，描述目录职责与层次关系；精确统计交给 `wiki/index.md` 的域表。

## Courses（学习路线导读层）

`wiki/courses/` 是功能树之上的**纯导读层**，服务于"我该按什么顺序读"这类跨目录学习路径需求（例如 torch.compile 端到端、后训练前沿）。

- **学习域 / 系列课只能以 `wiki/courses/` 下的一个导读页创建**，不得新建独立的纵向目录（例如历史上的 `19_torch_compile_end_to_end/`、`03_posttraining/` 都是这种反模式，已在 kb-reorg 中解散/并入功能树）。
- 导读页**只含**：阅读顺序、到功能树各页的 `[[wiki links]]`、每篇一句话导读。**禁止承载正文**——不得复述功能树页面已有的机制内容，不得成为"第二份真相来源"。
- 需要新增实质内容时，写入功能树对应模块的页面，再从课程页补一条链接 + 一句话导读；发现课程页内容与功能树矛盾或过时，去修功能树页面，课程页只更新链接/顺序/导读语，不在原地展开正文。
- 这条规则的目的是堵住"新学习域重新讲一遍旧内容"这一复发根因（详见 `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §1 复发根因诊断）。

## Provenance Policy（溯源政策）

- **`raw/` 只收文档类源材料**：论文来源说明页、文章、图表源（`.eddx`/`.html` 等）。**不要**把外部代码仓库复制进 `raw/`——代码类源以"旁置 checkout + 钉死 commit"的方式引用，`raw/` 与 `wiki/` 目录结构不要求镜像对齐。
- **代码分析页必须在页头钉住"仓库 + commit 基线"**：分析某个代码仓库（PyTorch、Megatron-LM、vLLM、torch_npu 等）产出的页面，页头必须写明所分析的仓库标识与具体 commit/tag（例如 "基线 `main @ 8a694930`"），后续同仓库有新分析时对照同一基线或显式声明新基线，不含糊带过"最新代码"。
- 不要求"每条断言都能追溯到 `raw/` 里的一个文件"这种全称表述——多数工程域页面的源是代码仓库而非 `raw/` 文档，只要满足"文档类源在 `raw/` 里可查、代码类源的仓库 + commit 基线在页头钉住"即可视为源忠实。核心原则仍是**源忠实**：断言要有可核验的定位符（`file:line`、`§/Table/Eq`、`raw/` 文件名等），冲突时以源为准（方法见 [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md)）。

## Quality Gates

改动落库前必须过的门禁（细则与逐条修法在对应技能里）：

```bash
python tools/check_links.py --strict     # wikilink 健康：broken/ambiguous/bare_index/orphans 必须为 0
python tools/check_math.py --changed --strict   # 本次改动的公式：error 与 warning 都必须为 0
python -m pytest tools/                  # 维护工具自身的单测
npm run docs:test                        # 本地文档站点单测 + 端到端验收
```

全库基线：截至 2026-08-26，`check_links` 与 `check_math --strict` 在全部 409 页上均为 0 error / 0 warning。**新出现的告警一定是本次改动引入的**，不要当作历史遗留放过。
