---
name: maintaining-llm-knowledge
description: Use when creating, updating, renaming, merging, indexing or auditing pages in the llm-knowledge wiki - page types and the NN_ segment numbering, naming rules, the ingest workflow, cross-reference rules, merge-over-coexist, and the query/maintenance routines. Load it before writing into wiki/, not by default.
---

# Maintaining the LLM Knowledge Wiki

`CLAUDE.md` is the constitution: it fixes what the wiki *is* (three layers, the 功能树 as the only content authority, provenance, quality gates). This skill carries the *operations* - how to actually add, name, link, merge and audit a page. Load it on demand when you are about to write into `wiki/`.

Formula and diagram conventions live in their own skills: [`writing-obsidian-math`](../writing-obsidian-math/SKILL.md) and [`writing-mermaid-diagrams`](../writing-mermaid-diagrams/SKILL.md). The decomposition method for turning a source into pages is [`source-faithful-analysis`](../source-faithful-analysis/SKILL.md).
### Page Types

| Type | Suffix | Purpose | Example |
|------|--------|---------|---------|
| Index | `index.md` | Domain entry point, directory contents & link map | `01_theory/index.md` |
| Entity | `*_analysis.md` | Deep analysis of a specific paper/technology | `11_muon_analysis.md` |
| Guide | `*_guide.md` | How-to or implementation walkthrough | `20_npu_lowering_guide.md` |
| Quickstart | `*_quickstart.md` | Minimal path to a working example before the deep dive | `01_autograd_engine_quickstart.md` |
| Deepdive | `*_deepdive.md`（无下划线分隔，非 `_deep_dive`） | Focused deep dive on one subtopic within a module | `22_npu_fusion_passes_deepdive.md` |
| Comparison | `comparison.md` | Side-by-side comparison of approaches | `npu/30_comparison.md` |
| Changelog | `changelog.md` | Chronological log of all ingest operations | `wiki/changelog.md`（历史条目按季度归档于 `wiki/changelog/`） |

**合法后缀白名单只有这 6 种**：`_analysis`、`_guide`、`_quickstart`、`_deepdive`、`comparison`、`index`。不使用 `_deep_dive`（下划线版）、`_report`、`_methodology`、`_overview`、`_map`、`_model`、`_concepts`、`_details`、`_diagrams`、`_v2` 等历史遗留后缀。

**禁止在 `wiki/` 内创建 `README.md`**——每个目录的入口一律是 `index.md`；`README.md` 是仓库根层面给人类浏览者看的文件类型，不是 wiki 页面类型。

**段位编号约定**：内容页文件名加两位数字前缀 `NN_`（`index.md` 不编号），十位数字表示"段"、决定阅读顺序，从文件名即可看出由浅入深的位置：

| 段 | 编号区间 | 含义 |
|---|---|---|
| 段 0 | `01`–`09` | 入门 / 导览（quickstart、knowledge map、overview 类） |
| 段 1 | `10`–`19` | 核心机制主线（按流水线/学习顺序排列的 `_analysis` 页） |
| 段 2 | `20`–`29` | 深潜 / 专题（`_deepdive`、专项分析、边角机制） |
| 段 3 | `30`–`39` | 方法论 / 对照 / 工程实践（开发 guide、`comparison`、排查实践） |

某段页面超出容量时占用相邻空段，并在该目录 `index.md` 的段位表里注明；硬件子目录（`npu/`、`cuda/` 等）页面较多时同一规则递归适用（子目录内独立编号，不占用父目录的号段）。少于 4 篇内容页的小目录不强制编号。

### Naming Conventions

- File names use `snake_case`（非 snake_case 一律视为需修正：无大写字母、无驼峰、无点号，如 `kimi_k2.5` 需写作 `kimi_k2_5`）
- Index pages are always named `index.md`（每目录一个，作为入口，不编号）
- One concept per page; prefer splitting over merging — 但主题**重叠**（不是同一概念的自然拆分）时适用下方 Update Principles 的 Merge over coexist

### Ingest Workflow

When a new source is added to `raw/`, follow this sequence:

1. **Read** the source document thoroughly
2. **Discuss** key takeaways with the user before writing
3. **Create** a new wiki page in the correct 功能树 module (or update an existing one if the topic is already covered) — follow Page Types/Naming Conventions/段位编号 above
4. **Update** the domain `index.md` to include the new page（条目表 + 段位，不画深层树）
5. **Cross-reference**: Add `[[wiki links]]` to and from all related existing pages（遵守下方 Cross-Reference Rules）
6. **Append** an entry to `wiki/changelog.md` documenting what was added/updated（示例性质的 `[[...]]` 用反引号转义，见 Cross-Reference Rules）
7. **Update the radar baseline**: 如果这次分析把某个代码仓库的基线推进了，同步改 `docs/radar/watchlist.yaml` 里对应条目的 `kb_baseline`。漏了这一步，`tools/radar.py` 会每周继续报同一批已经处理过的陈旧漂移，很快就没人看这份周报了。
8. **Flag contradictions**: If new information contradicts existing wiki content, preserve both claims and add a `> [!contradiction]` callout

### Cross-Reference Rules

- Every page MUST contain a `## Related Pages` section at the bottom（`index.md` 豁免——它本身就是链接地图）。挑选 **3–7 条精选链接**，每条后面跟**一句话**说明关联是什么（不是链接堆砌）；多于 7 条说明该页需要收缩,不是塞进更多链接。
- **裸基名默认合法**：内容页文件名在全库唯一，`[[page_name]]` 可以直接用，不强制加显示名——除非该链接直接充当句子的主语/宾语且文件名本身无法让读者看懂在指什么（判断有争议时倾向于加显示名，成本很低）。
- **`index` 链接必须路径限定 + 显示名**：`index.md` 这个文件名在全库不唯一（每个目录一个），裸 `[[index]]` 一定歧义。永远写成 `[[<相对 wiki 根的路径>/index|<该目录的语义显示名>]]`，例如 `[[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM]]`。链接目标是某个**目录**本身（而非目录内某一篇具体页面）时，同样改写为指向该目录的 `index`，不要用裸目录名。
- **禁止使用 `../` 相对路径**：链接路径一律从 `wiki/` 根开始写完整相对路径（如 `01_theory/06_distributed_parallelism/index`），不要用 `../`/`../../` 这类相对上跳——它们在目录迁移后极易失效且难以静态检查。
- **示例链接必须转义**：如果 `[[...]]` 出现在正文中只是作为**语法示例**被讨论（而不是想让它被解析成真实链接），例如 changelog 里记录"把 `RL_PPO_Loss_and_GRPO_Analysis` 改名为 `rl_ppo_loss_and_grpo_analysis`"这类历史操作说明，一律用反引号包裹（`` `[[index]]` ``），不要让示例文本被解析成真链接。
- Every new page MUST link to at least one existing page
- When updating a page, check if other pages should gain a backlink
- 验收基线：`python tools/check_links.py --strict` 必须 broken=0、ambiguous=0、bare_index=0（详见 `tools/check_links.py` 内的检查项说明）

### Update Principles

- **Merge over coexist**（合并优于并存）：发现主题重叠时必须合并到权威页——不是"两份都保留、互相链接"。合并前先判定谁是权威版本（更全/基线更新/粒度更好），把被并页的独有增量吸收进权威页；**删除前必须先修复全部入链**指向权威页；删除后在 `wiki/changelog.md` 记录并注明并入目标。
- Mark outdated claims with `> [!deprecated] Updated by [[page_name]]`
- Mark contradictions with `> [!contradiction] See also [[page_name]]`
- Record the date of each significant update in the page header
- When a page grows too large (>500 lines), propose a split to the user

### Quality Standards

- Write in the same language as the source material (Chinese for Chinese sources, English for English sources)
- Use Mermaid diagrams for architecture, data flow, and sequence visualizations（务必遵守 [`writing-mermaid-diagrams`](../writing-mermaid-diagrams/SKILL.md)）
- Use LaTeX for mathematical formulas（必须遵守 [`writing-obsidian-math`](../writing-obsidian-math/SKILL.md)）
- Include code references with file paths and line numbers when analyzing source code（配合 `CLAUDE.md` 的 Provenance Policy：仓库 + commit 基线）

### Baseline Header Convention

代码分析页的基线页头是 `tools/check_locators.py` 的解析输入。**新页一律用规范式**（每仓一行）：

```
> **源码基线**：`owner/repo@<完整或 ≥12 位 hex>`（`branch`，YYYY-MM-DD）
```

- 一页分析多个仓时，每个仓都写一行（或在其小节内用同格式钉出）；**引用了某仓的 `path:line`
  就必须钉过该仓的 commit**——checker 对没钉仓的引用报 `missing_file` error。
- 历史写法（`verl main @ 254a23ed`、`名称 vX@hex` 等）checker 宽容解析，但不再新增。
- 该仓需在 `docs/radar/watchlist.yaml` 有条目及 `checkout:` 本地检出，否则引用只能记
  `unresolved/unverifiable` warning 而无法验证。
- 验收：`python tools/check_locators.py`（missing_file 必须为 0；见 CLAUDE.md 质量门禁）。

### MCP Tools

Two MCP servers are configured in `.mcp.json`:

1. **filesystem** (`@modelcontextprotocol/server-filesystem`) — Read, write, search, and manage files in `wiki/` and `raw/`. Use for all file operations.
2. **qmd** (`qmd mcp`) — Search engine over wiki pages. Currently operating in **BM25 keyword mode only** (no embedding model). Use `search` for keyword matching; `vsearch` and `query` require embeddings (not yet available on this system).

When to use which:
- **filesystem** for: reading/writing files, listing directories, exact grep
- **qmd** for: keyword search across all wiki pages, finding pages that mention a specific term, cross-reference discovery

### Query Workflow

When the user asks a question:

1. **Search wiki first**: Use `qmd search` to find wiki pages matching relevant keywords
2. **Navigate the graph**: Check the relevant `index.md` to understand the domain landscape and follow `[[wiki links]]`
3. **Read the pages**: Use filesystem `read_file` to read the full content of the most relevant pages
4. **Synthesize**: If the answer requires synthesizing multiple pages, do so and note which pages contributed
5. **Check raw sources on gap**: If the answer is NOT in the wiki, do NOT just say "not found". Instead:
   - Scan `raw/` for relevant source documents (check filenames for topic keywords); if the topic is code-based, check whether a relevant sibling repo checkout is already referenced elsewhere in the wiki
   - If a relevant source exists, **automatically ingest it** following the Ingest Workflow, then answer the question from the newly created wiki content
   - If no relevant source exists, say so and offer to create a stub wiki page (in the correct 功能树 module, not a new standalone learning-track directory — see Courses) to track this knowledge gap
6. **Grow the wiki**: Every query that reveals a gap should result in either a new wiki page or a note in the relevant index.md under Knowledge Gaps

### Maintenance Workflow

Periodically (or when the user requests):

1. **Consistency check**: Run `python tools/check_links.py --strict`（broken/ambiguous/裸 index 必须为 0）
2. **Math check**: Run `python tools/check_math.py --changed --strict`（本次变更中的公式错误和警告必须为 0）
3. **Orphan check**: `check_links.py` 的 orphans 项已覆盖"无入链且未被任何 index.md 提及"的孤儿页；发现即整合入相应 index
4. **Contradiction review**: Scan for `> [!contradiction]` callouts and propose resolutions
5. **Staleness review**: Flag pages that haven't been updated in >30 days for review
6. **Duplication review**: 发现同一主题在多个页面复述（而不是自然的"实现差异"）时，适用 Update Principles 的 Merge over coexist
