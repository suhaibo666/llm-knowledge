# skills/ — 公共 agent 技能

这里是本仓库**唯一**的一份技能定义，供所有 agent 共用。此前 `.claude/skills/` 与
`.agents/skills/` 各存一份副本，两边已经开始漂移（`source-faithful-analysis` 的宿主
说明一处写 `CLAUDE.md`、一处写 `AGENTS.md`），并且要靠一条单测强行比对来维持同步。
现在只保留这一份。

## 加载约定

`CLAUDE.md` 是本知识库的**基本法**：它只定义知识库*是什么*——三层结构、功能树的唯一
权威地位、溯源政策、质量门禁。**具体怎么写文档不属于基本法**，落在这里的技能中，
**按需加载，不默认加载**。

判断该读哪一篇：

| 你要做的事 | 读 |
|---|---|
| 往 `wiki/` 里新增/改写/重命名/合并页面，维护 index 与 changelog | [`maintaining-llm-knowledge`](maintaining-llm-knowledge/SKILL.md) |
| 写或改任何 LaTeX 公式 | [`writing-obsidian-math`](writing-obsidian-math/SKILL.md) |
| 画或改任何 Mermaid 图 | [`writing-mermaid-diagrams`](writing-mermaid-diagrams/SKILL.md) |
| 把一份新的源（论文/代码仓库/规格/数据集）拆解成 wiki 页 | [`source-faithful-analysis`](source-faithful-analysis/SKILL.md) |

一次任务通常要读不止一篇：例如"把某论文写成一篇分析页"= `source-faithful-analysis`
（怎么拆）+ `maintaining-llm-knowledge`（放哪、怎么命名、怎么挂链接）+ 公式/图技能
（按页面里实际出现的元素）。

## 各家 agent 怎么用

- **Claude Code** — `CLAUDE.md` 在会话开始时加载，其中列出本目录；需要时用 Read 打开对应
  `SKILL.md`。技能不再放在 `.claude/skills/`，因此不会被当作内置技能自动发现，这是刻意的：
  基本法之外的内容一律按需读取。
- **Codex** — `AGENTS.md` 指向 `CLAUDE.md`，同样按上表按需读取本目录。

只保留这两家。历史上的 `.agents/skills/`（重复副本）与 `opencode.json` 已删除。

## 改技能时

技能是**规范**，不是随笔：

- 规则要能被 `tools/` 下的检查器验证，或者明确写清"这条只能人工核对"。
- 改了规则就同步改检查器与其单测（`tools/check_math.py`、`tools/check_links.py`、
  `tools/test_*.py`），不要让文档与门禁各说各话。
- 不要把同一条规则同时写进 `CLAUDE.md` 和技能——基本法只留"是什么"，技能只留"怎么做"。
