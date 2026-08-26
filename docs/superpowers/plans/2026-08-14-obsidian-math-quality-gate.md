# Obsidian Math Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 统一当前 slime/vime Markdown 的 Obsidian 公式写法，并提供可自动执行的公式写作 skill 和质量检查器。

**Architecture:** `tools/check_math.py` 负责忽略代码区域后提取 Markdown 数学片段，并输出稳定的规则编号诊断；`tools/test_check_math.py` 锁定结构错误、风格警告与误报豁免。`writing-obsidian-math` skill 保存人工判断规范，`CLAUDE.md` 把严格自检接入日常写作流程。

**Tech Stack:** Python 3 标准库、pytest、Markdown、MathJax/Obsidian 数学语法、Git CLI。

## Global Constraints

- 只整改 `wiki/02_engineering/04_posttrain_frameworks/slime/` 下当前生成的 slime/vime 文档，不批量改写其他历史页面。
- 检查器不得新增第三方运行时依赖。
- 默认模式只让错误导致非零退出；`--strict` 让错误和警告都导致非零退出。
- 检查器必须忽略代码围栏和行内代码，并避免把 `$5.5M` 误判成公式。
- `.agents` 与 `.claude` 中的 skill 必须保持内容一致。
- 保留用户已有和本任务无关的工作区修改。

---

### Task 1: 以测试定义公式检查行为

**Files:**
- Create: `tools/test_check_math.py`

**Interfaces:**
- Consumes: Python 导入路径中的 `tools.check_math`。
- Produces: 对 `check_text(text: str, path: str) -> list[Diagnostic]` 和 CLI 退出语义的行为约束。

- [ ] **Step 1: 写入当前必然失败的导入与行为测试**

  测试分别断言 `MATH001`、`MATH002`、`MATH003`、`MATH005`、`MATH102`，并加入标准公式、代码围栏、行内代码、货币文本和 `\texttt{...}` 的反例。

- [ ] **Step 2: 运行 RED 测试**

  Run: `python -m pytest tools/test_check_math.py -q`

  Expected: FAIL during collection with `ModuleNotFoundError: No module named 'tools.check_math'`。

- [ ] **Step 3: 保留失败输出作为 TDD 基线并进入实现**

  只接受“待实现模块不存在”这一预期失败；如果是测试语法错误，先修复测试并重新确认 RED。

### Task 2: 实现零依赖 Markdown 数学检查器

**Files:**
- Create: `tools/check_math.py`
- Modify: `tools/README.md`
- Test: `tools/test_check_math.py`

**Interfaces:**
- Consumes: UTF-8 Markdown 文本、显式文件/目录参数、Git 变更文件列表。
- Produces: `Diagnostic(severity, code, path, line, message)`、`check_text`、`check_file`、`collect_changed_markdown` 和命令行退出码。

- [ ] **Step 1: 实现诊断模型与代码区域屏蔽**

  使用标准库 `dataclasses`、`pathlib`、`re` 和 `subprocess`；逐行跟踪反引号或波浪线代码围栏，并在非围栏行中屏蔽行内代码。

- [ ] **Step 2: 实现结构错误规则**

  检测非标准定界符、未闭合 `$$`、排除货币后的未成对 `$`、公式花括号和表格公式；输出 `path:line [severity CODE] message`。

- [ ] **Step 3: 实现风格警告规则**

  在允许的文本命令参数被屏蔽后检查裸 `\_`，识别语义下标、裸竖线和过长的未对齐块公式。

- [ ] **Step 4: 实现文件收集和 CLI**

  显式路径可递归收集 `.md`；`--changed` 合并 `git diff`、`git diff --cached` 与 `git ls-files --others`，去重后只保留存在的 Markdown 文件。

- [ ] **Step 5: 运行 GREEN 测试**

  Run: `python -m pytest tools/test_check_math.py -q`

  Expected: all tests PASS。

### Task 3: 整改当前 slime/vime 公式

**Files:**
- Modify: `wiki/02_engineering/04_posttrain_frameworks/slime/*.md` 中实际命中规则的页面

**Interfaces:**
- Consumes: Task 2 的严格检查器与设计规范。
- Produces: 只使用 `$...$` 和独占行 `$$`、语义明确且可在 Obsidian 稳定渲染的公式。

- [ ] **Step 1: 统一定界符**

  把代码区域外的 `\(...\)` 转为 `$...$`，把 `\[...\]` 转为起止独占行的 `$$`。

- [ ] **Step 2: 修复已识别的语义排版问题**

  将 `accept\_rate`/`accept\_length` 改为解释过的数学符号，将 `TP\_size` 改为 `$N_{\mathrm{TP}}$` 语义，将 `p_{train-old}`、`C_{low}`、`\lambda_{opd}` 等改成直立语义下标，并把条件概率裸 `|` 改成 `\mid`。

- [ ] **Step 3: 整理多行公式**

  对多行等式使用 `aligned`，确保 `$$` 独占一行，保持公式数学结论不变。

- [ ] **Step 4: 对目标目录运行严格检查**

  Run: `python tools/check_math.py --strict wiki/02_engineering/04_posttrain_frameworks/slime`

  Expected: exit 0 and `Checked ... Markdown file(s): 0 error(s), 0 warning(s).`

### Task 4: 新增公式写作 skill 与仓库规则

**Files:**
- Create: `.agents/skills/writing-obsidian-math/SKILL.md`
- Create: `.agents/skills/writing-obsidian-math/evals/evals.json`
- Create: `.claude/skills/writing-obsidian-math/SKILL.md`
- Create: `.claude/skills/writing-obsidian-math/evals/evals.json`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: 设计规范和 `tools/check_math.py` CLI。
- Produces: 在含公式 Markdown 写作场景触发的 skill、三条代表性评估提示和仓库级强制自检入口。

- [ ] **Step 1: 编写 `.agents` skill**

  frontmatter 的 description 只描述触发条件；正文包含定界符、符号语义、表格/多行公式、坏例到好例、人工核对与两种自检命令。

- [ ] **Step 2: 添加评估提示**

  `evals/evals.json` 包含“从零写带公式的分析”“修复混合定界符页面”“审查表格和概率公式”三类任务，并明确期望输出通过严格检查。

- [ ] **Step 3: 同步 `.claude` 镜像**

  复制相同字节内容，并用 SHA-256 比较两组文件。

- [ ] **Step 4: 更新 `CLAUDE.md`**

  在公式规范处链接 skill，列出 `$...$`、独占行 `$$` 和禁止非标准定界符，并要求公式编辑后执行 `python tools/check_math.py --strict <files>`。

### Task 5: 完成全量验收

**Files:**
- Modify: `wiki/changelog.md`

**Interfaces:**
- Consumes: Tasks 1-4 的代码、文档和 skill。
- Produces: 可复核的测试结果与变更记录。

- [ ] **Step 1: 更新 changelog**

  记录 slime/vime 公式整改、公式 skill 和自动检查器，不混入无关内容。

- [ ] **Step 2: 运行公式检查器测试**

  Run: `python -m pytest tools/test_check_math.py -q`

  Expected: all tests PASS。

- [ ] **Step 3: 运行目标页面严格公式检查**

  Run: `python tools/check_math.py --strict wiki/02_engineering/04_posttrain_frameworks/slime`

  Expected: exit 0，零错误、零警告。

- [ ] **Step 4: 运行仓库链接检查**

  Run: `python tools/check_links.py --strict`

  Expected: exit 0。

- [ ] **Step 5: 检查补丁完整性与 skill 镜像**

  Run: `git diff --check`

  Expected: exit 0；随后比较 `.agents` 与 `.claude` 两份 `SKILL.md` 和 `evals.json` 的 SHA-256，值分别相同。

## Self-Review

- Spec coverage: 五个任务分别覆盖测试先行、检查器、slime/vime 整改、skill/仓库集成和最终验收，没有遗漏设计范围。
- Placeholder scan: 计划不含占位关键词、模糊的延期描述或未给出验收命令的步骤。
- Type consistency: Task 1 依赖的 `Diagnostic` 与 `check_text` 名称和 Task 2 的产出一致；Tasks 3-5 使用统一的 `tools/check_math.py --strict` 接口。
