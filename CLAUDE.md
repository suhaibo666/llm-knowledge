# LLM Knowledge Wiki — Schema & Maintenance Rules

This document governs how the LLM agent maintains this knowledge wiki. Read it before any wiki operation.

## Architecture

This wiki has three layers:

1. **Raw sources** (`raw/`) — Immutable source documents (PDFs, papers, articles). The LLM reads from them but NEVER modifies them.
2. **Wiki** (`wiki/`) — LLM-generated markdown files. The LLM owns this layer entirely. It creates pages, updates them, maintains cross-references, and keeps everything consistent.
3. **This schema** (`CLAUDE.md`) — Tells the LLM how the wiki is structured, what conventions to follow, and what workflows to execute.
4. **Analysis methodology** (`.claude/skills/source-faithful-analysis/`) — A **repo-bundled Claude Code skill** that tells the LLM *how to decompose and analyze a source* into those wiki pages (Claude Code auto-loads it when this repo is open). This schema governs *structure*; the skill governs *process*.

`raw/` and `wiki/` are **independent trees** — there is no requirement that they mirror each other directory-for-directory. Most `wiki/` domains are sourced from external code repositories (checked out as sibling directories, not copied into `raw/`), not from a matching `raw/` subtree. See **Provenance Policy** below for what does and doesn't belong in `raw/`.

## Analysis & Decomposition Methodology

把一个源（论文 / 代码 / 规格 / 数据集 / 任意材料）拆解、深挖、落成 wiki 页的**标准分解流程**封装为**仓库自带技能** [`.claude/skills/source-faithful-analysis/`](.claude/skills/source-faithful-analysis/SKILL.md)（Claude Code 打开本仓库即自动加载，无需另装）：两条原则（**源忠实**——每条断言带已核验定位符、冲突时以源为准；**抓本质**——主线先行 + 动机/机制/证据/为什么不选替代），Phase 0–5 工作流，并行 writer-agent 契约；按来源类型再读该技能的 `references/{codebase,paper,general}.md` 包。下面的 Ingest / Query Workflow 是该技能在本库的落地实例。

## Directory Layout

- 功能分类树（`wiki/01_theory/`、`wiki/02_engineering/`）是唯一的内容权威。目录结构以文件系统与各级 `index.md` 为准；**不要**在 `CLAUDE.md` 或 `README.md` 里维护一份平行的深层 ASCII 树——它会立刻和实际目录漂移。
- 每个目录的 `index.md` **只维护本目录的条目表**（页面列表 + 一句话 + 状态/段位），不越权描述子孙目录的内部细节，也不需要画出多级嵌套树。
- `wiki/index.md`（总索引）**只保留域级表格**（入口 + 一句话 + 页面数 + 状态），不重复各级 `index.md` 已有的内容，不画深层目录树。页面数按 `find`/`rglob` 现算，随内容漂移及时校正，不长期沿用一次性统计的旧数字。
- `README.md` 描述结构时**不写精确页面数**（页数变化快，写死的数字很快就会失真、误导读者）；只描述目录职责与层次关系，精确统计交给 `wiki/index.md` 的域表。

## Courses（学习路线导读层）

`wiki/courses/` 是功能树之上的**纯导读层**，服务于"我该按什么顺序读"这类跨目录学习路径需求（例如 torch.compile 端到端、后训练前沿）。

- **学习域 / 系列课只能以 `wiki/courses/` 下的一个导读页创建**，不得新建独立的纵向目录（例如历史上的 `19_torch_compile_end_to_end/`、`03_posttraining/` 都是这种反模式，已在 kb-reorg 中解散/并入功能树）。
- 导读页**只含**：阅读顺序、到功能树各页的 `[[wiki links]]`、每篇一句话导读。**禁止承载正文**——不得复述功能树页面已有的机制内容，不得成为"第二份真相来源"。
- 需要新增实质内容时，写入功能树对应模块的页面，再从课程页补一条链接 + 一句话导读；发现课程页内容与功能树矛盾或过时，去修功能树页面，课程页只更新链接/顺序/导读语，不在原地展开正文。
- 这条规则的目的是堵住"新学习域重新讲一遍旧内容"这一复发根因（详见 `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §1 复发根因诊断）。

## Page Types

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

## Naming Conventions

- File names use `snake_case`（非 snake_case 一律视为需修正：无大写字母、无驼峰、无点号，如 `kimi_k2.5` 需写作 `kimi_k2_5`）
- Index pages are always named `index.md`（每目录一个，作为入口，不编号）
- One concept per page; prefer splitting over merging — 但主题**重叠**（不是同一概念的自然拆分）时适用下方 Update Principles 的 Merge over coexist

## Ingest Workflow

When a new source is added to `raw/`, follow this sequence:

1. **Read** the source document thoroughly
2. **Discuss** key takeaways with the user before writing
3. **Create** a new wiki page in the correct 功能树 module (or update an existing one if the topic is already covered) — follow Page Types/Naming Conventions/段位编号 above
4. **Update** the domain `index.md` to include the new page（条目表 + 段位，不画深层树）
5. **Cross-reference**: Add `[[wiki links]]` to and from all related existing pages（遵守下方 Cross-Reference Rules）
6. **Append** an entry to `wiki/changelog.md` documenting what was added/updated（示例性质的 `[[...]]` 用反引号转义，见 Cross-Reference Rules）
7. **Flag contradictions**: If new information contradicts existing wiki content, preserve both claims and add a `> [!contradiction]` callout

## Cross-Reference Rules

- Every page MUST contain a `## Related Pages` section at the bottom（`index.md` 豁免——它本身就是链接地图）。挑选 **3–7 条精选链接**，每条后面跟**一句话**说明关联是什么（不是链接堆砌）；多于 7 条说明该页需要收缩,不是塞进更多链接。
- **裸基名默认合法**：内容页文件名在全库唯一，`[[page_name]]` 可以直接用，不强制加显示名——除非该链接直接充当句子的主语/宾语且文件名本身无法让读者看懂在指什么（判断有争议时倾向于加显示名，成本很低）。
- **`index` 链接必须路径限定 + 显示名**：`index.md` 这个文件名在全库不唯一（每个目录一个），裸 `[[index]]` 一定歧义。永远写成 `[[<相对 wiki 根的路径>/index|<该目录的语义显示名>]]`，例如 `[[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM]]`。链接目标是某个**目录**本身（而非目录内某一篇具体页面）时，同样改写为指向该目录的 `index`，不要用裸目录名。
- **禁止使用 `../` 相对路径**：链接路径一律从 `wiki/` 根开始写完整相对路径（如 `01_theory/06_distributed_parallelism/index`），不要用 `../`/`../../` 这类相对上跳——它们在目录迁移后极易失效且难以静态检查。
- **示例链接必须转义**：如果 `[[...]]` 出现在正文中只是作为**语法示例**被讨论（而不是想让它被解析成真实链接），例如 changelog 里记录"把 `RL_PPO_Loss_and_GRPO_Analysis` 改名为 `rl_ppo_loss_and_grpo_analysis`"这类历史操作说明，一律用反引号包裹（`` `[[index]]` ``），不要让示例文本被解析成真链接。
- Every new page MUST link to at least one existing page
- When updating a page, check if other pages should gain a backlink
- 验收基线：`python tools/check_links.py --strict` 必须 broken=0、ambiguous=0、bare_index=0（详见 `tools/check_links.py` 内的检查项说明）

## Update Principles

- **Merge over coexist**（合并优于并存）：发现主题重叠时必须合并到权威页——不是"两份都保留、互相链接"。合并前先判定谁是权威版本（更全/基线更新/粒度更好），把被并页的独有增量吸收进权威页；**删除前必须先修复全部入链**指向权威页；删除后在 `wiki/changelog.md` 记录并注明并入目标。（完整判重/迁移规程见 `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3、§6）
- Mark outdated claims with `> [!deprecated] Updated by [[page_name]]`
- Mark contradictions with `> [!contradiction] See also [[page_name]]`
- Record the date of each significant update in the page header
- When a page grows too large (>500 lines), propose a split to the user

## Provenance Policy（溯源政策）

- **`raw/` 只收文档类源材料**：论文 PDF、文章、图表源（`.eddx`/`.html` 等）。**不要**把外部代码仓库复制进 `raw/`——代码类源以"旁置 checkout + 钉死 commit"的方式引用（见下条），`raw/` 与 `wiki/` 目录结构不要求镜像对齐。
- **代码分析页必须在页头钉住"仓库 + commit 基线"**：分析某个代码仓库（PyTorch、Megatron-LM、vLLM、torch_npu 等）产出的页面，页头必须写明所分析的仓库标识与具体 commit/tag（例如 "基线 `main @ 8a694930`"），后续同仓库有新分析时对照同一基线或显式声明新基线，不含糊带过"最新代码"。
- 不再要求"每条断言都能追溯到 `raw/` 里的一个文件"这种全称表述——多数工程域页面的源是代码仓库而非 `raw/` 文档，只要满足"文档类源在 `raw/` 里可查、代码类源的仓库+commit 基线在页头钉住"即可视为源忠实。核心原则仍是**源忠实**：断言要有可核验的定位符（`file:line`、`§/Table/Eq`、`raw/` 文件名等），冲突时以源为准（详见 `.claude/skills/source-faithful-analysis/SKILL.md`）。

## Quality Standards

- Write in the same language as the source material (Chinese for Chinese sources, English for English sources)
- Use Mermaid diagrams for architecture, data flow, and sequence visualizations（务必遵守下方「Mermaid 规范与生成后校验」）
- Use LaTeX for mathematical formulas（必须遵守下方「Obsidian 公式规范与生成后校验」，并读取 `.claude/skills/writing-obsidian-math/SKILL.md`）
- Include code references with file paths and line numbers when analyzing source code（配合上方 Provenance Policy 的仓库 + commit 基线）

## Obsidian 公式规范与生成后校验

本库以 Obsidian 为主要阅读端，公式统一使用其文档化的 dollar 定界符：行内公式写成 `$...$`，块级公式的起止 `$$` 必须各自独占一行。新增或修改的 Markdown **禁止**使用 `\(...\)`、`\[...\]`，也禁止把块级公式写成 `$$x+y$$`。

公式除了“能渲染”，还必须表达正确的数学语义：

- 多字母语义下标用 `\mathrm{...}` 或 `\text{...}`，例如 `$C_{\mathrm{low}}$`；不要把 `low` 写成三个相乘的斜体变量。
- 条件概率用 `\mid`，绝对值用 `\lvert...\rvert`；Markdown 表格中的裸 `|` 会先被解释为分栏符。
- 不把 API 标识符直接写成 `accept\_rate` 这类数学变量；优先定义 `$r_{\mathrm{accept}}$`，必须展示字面 API 名时使用 `\texttt{accept\_rate}`。
- 多行推导使用 `\begin{aligned}...\end{aligned}`、`&` 对齐点和显式 `\\` 换行；源码换行本身不会让公式换行。
- 表格单元格只放短行内公式，块级公式移到表格外。

**生成后校验（必做，不可跳）：**

1. 每次新增、改写或审查含公式的 Markdown，必须应用 `writing-obsidian-math` skill，并逐式检查定界符、花括号、`\left`/`\right`、语义下标、竖线含义和多行对齐。
2. 编辑完成后对目标文件或目录运行 `python tools/check_math.py --strict <path>`，错误和警告都必须处理。
3. 提交前运行 `python tools/check_math.py --changed --strict`；不能为了消除启发式警告而改变数学含义或把正文藏进行内代码。
4. 自动检查不能证明代数正确或忠实于论文/源代码，仍需人工确认符号定义、分子分母、条件分布和等价变形没有被排版整改改变。

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
   - Scan `raw/` for relevant source documents (check filenames for topic keywords); if the topic is code-based, check whether a relevant sibling repo checkout is already referenced elsewhere in the wiki
   - If a relevant source exists, **automatically ingest it** following the Ingest Workflow, then answer the question from the newly created wiki content
   - If no relevant source exists, say so and offer to create a stub wiki page (in the correct 功能树 module, not a new standalone learning-track directory — see Courses) to track this knowledge gap
6. **Grow the wiki**: Every query that reveals a gap should result in either a new wiki page or a note in the relevant index.md under Knowledge Gaps

## Maintenance Workflow

Periodically (or when the user requests):

1. **Consistency check**: Run `python tools/check_links.py --strict`（broken/ambiguous/裸 index 必须为 0）
2. **Math check**: Run `python tools/check_math.py --changed --strict`（本次变更中的公式错误和警告必须为 0）
3. **Orphan check**: `check_links.py` 的 orphans 项已覆盖"无入链且未被任何 index.md 提及"的孤儿页；发现即整合入相应 index
4. **Contradiction review**: Scan for `> [!contradiction]` callouts and propose resolutions
5. **Staleness review**: Flag pages that haven't been updated in >30 days for review
6. **Duplication review**: 发现同一主题在多个页面复述（而不是自然的"实现差异"）时，适用 Update Principles 的 Merge over coexist
