# LLM Knowledge Wiki — Schema & Maintenance Rules

This document governs how the LLM agent maintains this knowledge wiki. Read it before any wiki operation.

## Architecture

This wiki has three layers:

1. **Raw sources** (`raw/`) — Immutable source documents (PDFs, papers, articles). The LLM reads from them but NEVER modifies them.
2. **Wiki** (`wiki/`) — LLM-generated markdown files. The LLM owns this layer entirely. It creates pages, updates them, maintains cross-references, and keeps everything consistent.
3. **This schema** (`CLAUDE.md`) — Tells the LLM how the wiki is structured, what conventions to follow, and what workflows to execute.
4. **Analysis methodology** (`.claude/skills/source-faithful-analysis/`) — A **repo-bundled Claude Code skill** that tells the LLM *how to decompose and analyze a source* into those wiki pages (Claude Code auto-loads it when this repo is open). This schema governs *structure*; the skill governs *process*.

## Analysis & Decomposition Methodology

把一个源（论文 / 代码 / 规格 / 数据集 / 任意材料）拆解、深挖、落成 wiki 页的**标准分解流程**封装为**仓库自带技能** [`.claude/skills/source-faithful-analysis/`](.claude/skills/source-faithful-analysis/SKILL.md)（Claude Code 打开本仓库即自动加载，无需另装）：两条原则（**源忠实**——每条断言带已核验定位符、冲突时以源为准；**抓本质**——主线先行 + 动机/机制/证据/为什么不选替代），Phase 0–5 工作流，并行 writer-agent 契约；按来源类型再读该技能的 `references/{codebase,paper,general}.md` 包。下面的 Ingest / Query Workflow 是该技能在本库的落地实例。

## Directory Layout

完整的 raw/ 和 wiki/ 目录结构分别由各层级的 `index.md` 维护，详见 [[index]]（总索引）和各领域入口。

## Page Types

| Type | Suffix | Purpose | Example |
|------|--------|---------|---------|
| Index | `index.md` | Domain entry point, directory contents & link map | `01_theory/index.md` |
| Entity | `*_analysis.md` | Deep analysis of a specific paper/technology | `muon_analysis.md` |
| Guide | `*_guide.md` | How-to or implementation walkthrough | `npu_lowering_guide.md` |
| Comparison | `comparison.md` | Side-by-side comparison of approaches | `npugraphs/comparison.md` |
| Changelog | `changelog.md` | Chronological log of all ingest operations | `wiki/changelog.md` |

## Naming Conventions

- File names use `snake_case`
- Analysis pages end with `_analysis`
- Guide pages end with `_guide`
- Index pages are always named `index.md`（每目录一个，作为入口）
- One concept per page; prefer splitting over merging

## Ingest Workflow

When a new source is added to `raw/`, follow this sequence:

1. **Read** the source document thoroughly
2. **Discuss** key takeaways with the user before writing
3. **Create** a new wiki page (or update an existing one if the topic is already covered)
4. **Update** the domain `index.md` to include the new page
5. **Cross-reference**: Add `[[wiki links]]` to and from all related existing pages
6. **Append** an entry to `wiki/changelog.md` documenting what was added/updated
7. **Flag contradictions**: If new information contradicts existing wiki content, preserve both claims and add a `> [!contradiction]` callout

## Cross-Reference Rules

- Every page MUST contain a `## Related Pages` section at the bottom with `[[wiki links]]`
- Every new page MUST link to at least one existing page
- Use Obsidian `[[wiki link]]` syntax for internal links
- When updating a page, check if other pages should gain a backlink

## Update Principles

- **Merge over coexist**: 主题重叠时合并到权威页，被并页在修复全部入链后删除，并在 changelog 记录并入目标(2026-07 结构整改起生效;完整规则见 docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md §6)
- Mark outdated claims with `> [!deprecated] Updated by [[page_name]]`
- Mark contradictions with `> [!contradiction] See also [[page_name]]`
- Record the date of each significant update in the page header
- When a page grows too large (>500 lines), propose a split to the user

## Quality Standards

- Write in the same language as the source material (Chinese for Chinese sources, English for English sources)
- Use Mermaid diagrams for architecture, data flow, and sequence visualizations（务必遵守下方「Mermaid 规范与生成后校验」）
- Use LaTeX for mathematical formulas
- Include code references with file paths and line numbers when analyzing source code
- Every claim should trace back to a source in `raw/`

## Mermaid 规范与生成后校验

Mermaid 渲染失败源于**节点/连线文本里混入了 mermaid 的语法定界符**(`[] () {} |` 和换行)。原理:mermaid 用 `[]`/`()`/`{}` 界定节点形状、用 `|` 界定连线标签,文本里再出现这些字符可能让解析器判断不出边界。**按严重度分两档**(本库实测沉淀):

> **① 必崩(零容忍)—— 形状内嵌套定界符**:`X[(无 [N,V] 激活)]`(圆柱 `[(...)]` 里又有 `[`)、`{判断[i]}`(菱形里有 `[`)这类**特殊形状内再出现 `[]`/`()`** 一定解析失败。修法:特殊形状(圆柱 `[(...)]`、子程序 `[[...]]`、菱形 `{...}`)内**只放最简纯文本**,绝不嵌套定界符 → `X["无 N×V 激活"]`。
>
> **② 渲染器相关(求稳一律避免)**:`A["logits[N,V]"]`(带引号矩形标签里的 `[]`/`()`)、`A -. "文字" .-> B`(虚线内联引号标签)——这些在多数 mermaid 版本能渲、个别版本崩,**不稳定**。为可移植求稳:张量形状写 `N×d`/`B·S·V` 不写 `[N,V]`;带文字连线统一用管道标签 `A -->|文字| B` / `A -.->|文字| B`(文字内不放引号、括号、`|`)。已能渲的旧图不必为此大改,但新图按此写。

**其它常见坑 → 修法:**
- 管道标签里带引号/括号:`A -->|"文字(x)"| B` → `A -->|文字 x| B`。
- 标签里直接敲回车换行 → 用 `<br/>`。
- 中文/符号当节点 id → id 用英文数字(`H`、`NA`),中文/特殊字符放进 `["..."]` 标签里。
- 子图标题 `subgraph id["标题"]`:标题可含 `( )`、`：`、`/`(已验证可渲),但**不可含 `[ ]` 或 `|`**;每个 subgraph 必须用**单独一行** `end` 闭合。
- 代码块**首行**必须是图类型声明(`flowchart TB|LR`、`sequenceDiagram`、`graph TD` 等)。

**生成后校验(必做,不可跳):**
1. 写完一个 mermaid 块**立即重读**它,对照上面逐条扫:标签里有没有裸 `[] ()`、特殊形状有没有嵌套定界符、连线文字有没有引号/括号/`|`、有没有 `<br/>` 之外的换行。
2. 提交前对本次改动的每个文件 `grep -n mermaid <file>` 定位所有图,**逐块**再过一遍清单。
3. 有条件就实渲确认(mermaid-cli `mmdc`,或在线 live editor 粘一遍);不能渲就严格按清单人工核对。
4. 发现问题就地改;**绝不**把"可能能渲"的块留到 commit。

## MCP Tools

Two MCP servers are configured in `.mcp.json`:

1. **filesystem** (`@modelcontextprotocol/server-filesystem`) — Read, write, search, and manage files in `wiki/` and `raw/`. Use for all file operations.
2. **qmd** (`qmd mcp`) — Search engine over wiki pages. Currently operating in **BM25 keyword mode only** (no embedding model). Use `search` for keyword matching; `vsearch` and `query` require embeddings (not yet available on this system).

When to use which:
- **filesystem** for: reading/writing files, listing directories, exact grep
- **qmd** for: keyword search across all wiki pages, finding pages that mention a specific term, cross-reference discovery

## Query Workflow

When the user asks a question:

1. **Search wiki first**: Use `qmd search` to find wiki pages matching relevant keywords
2. **Navigate the graph**: Check the relevant `index.md` to understand the domain landscape and follow `[[wiki links]]`
3. **Read the pages**: Use filesystem `read_file` to read the full content of the most relevant pages
4. **Synthesize**: If the answer requires synthesizing multiple pages, do so and note which pages contributed
5. **Check raw sources on gap**: If the answer is NOT in the wiki, do NOT just say "not found". Instead:
   - Scan `raw/` for relevant source documents (check filenames for topic keywords)
   - If a relevant source exists in `raw/`, **automatically ingest it** following the Ingest Workflow, then answer the question from the newly created wiki content
   - If no relevant source exists in `raw/`, say so and offer to create a stub wiki page to track this knowledge gap
6. **Grow the wiki**: Every query that reveals a gap should result in either a new wiki page or a note in the relevant index.md under Knowledge Gaps

## Maintenance Workflow

Periodically (or when the user requests):

1. **Consistency check**: Verify all `[[wiki links]]` point to existing pages
2. **Orphan check**: Find pages not linked from any `index.md` and integrate them
3. **Contradiction review**: Scan for `> [!contradiction]` callouts and propose resolutions
4. **Staleness review**: Flag pages that haven't been updated in >30 days for review
