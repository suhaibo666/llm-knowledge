# Qwen3.8 Knowledge Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Qwen3.8-Max 官方发布文章、开源模型卡与许可证中的可核验内容摄入 LLM Knowledge Wiki。

**Architecture:** 以官方博客 2026-08-03 快照和 Hugging Face `Qwen/Qwen3.8-2.4T-A95B@207bd685` 为双基线。正文集中在一篇 Qwen3.8 权威分析页，家族目录只做导航，主模型索引、相关页反链与 changelog 负责接入全库图谱。

**Tech Stack:** Markdown、Obsidian wikilinks、Mermaid、PowerShell、仓库 `tools/check_links.py`。

## Global Constraints

- 重要事实必须带官方博客行号、模型卡行号、许可证行号或固定 Hugging Face revision。
- 区分托管版 `Qwen3.8-Max` 与开放权重 `Qwen3.8-2.4T-A95B`，不得混写模态、thinking 与上下文能力。
- 所有新内容页使用合法 `_analysis` 后缀，并在末尾保留 3–7 条带说明的 `## Related Pages`。
- 只复现能够支撑论点的评测数字，并连同 harness、重复次数、内部基准等条件一起说明。
- 新建或修改的 Mermaid 块必须通过仓库语法规范复核。

---

### Task 1: 固定来源快照

**Files:**
- Create: `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Max_blog_2026-08-03.txt`
- Create: `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_2_4T_A95B_model_card_207bd685.md`
- Create: `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Max_LICENSE_207bd685.txt`

**Interfaces:**
- Consumes: Qwen 官方博客与 Hugging Face revision `207bd685a7e3696cfaff12ded7c6a7ea0f88c996`。
- Produces: 后续正文可按 `文件:行号` 引用的不可变快照。

- [x] **Step 1:** 保存官方博客渲染后正文并写入来源 URL、发布日期与抓取日期。
- [x] **Step 2:** 保存固定 revision 的模型卡与许可证原文。
- [x] **Step 3:** 用 `rg -n` 核对架构、真实工作 RL、评测脚注与许可证限制均可定位。

### Task 2: 写 Qwen3.8 权威分析页

**Files:**
- Create: `wiki/01_theory/01_models/alibaba_qwen/10_qwen3_8_analysis.md`

**Interfaces:**
- Consumes: Task 1 的三份快照与固定模型配置。
- Produces: Qwen3.8 的单一权威正文页。

- [x] **Step 1:** 写来源基线、开放权重/托管版边界与中央论点。
- [x] **Step 2:** 写 92 层混合注意力 + 512 专家 MoE 的完整结构图和精确参数表。
- [x] **Step 3:** 按动机→机制→证据→替代方案边界分析真实工作 RL 的三件套。
- [x] **Step 4:** 用少量评测切片刻画能力，并单列 harness、内部基准和 vendor-demo 限制。
- [x] **Step 5:** 写部署语义、自定义许可证和未披露知识缺口。

### Task 3: 接入家族与全局索引

**Files:**
- Create: `wiki/01_theory/01_models/alibaba_qwen/index.md`
- Modify: `wiki/01_theory/01_models/index.md`
- Modify: `wiki/index.md`

**Interfaces:**
- Consumes: Task 2 的权威页名 `10_qwen3_8_analysis`。
- Produces: 从总索引到 Qwen 家族再到正文的稳定导航路径。

- [x] **Step 1:** 创建只含条目表、阶段位与知识缺口的 Qwen 家族索引。
- [x] **Step 2:** 在模型总索引增加 Qwen 家族入口与 Qwen3.8 条目，刷新日期。
- [x] **Step 3:** 递归重算模型域与 Qwen 子域页面数，更新 `wiki/index.md`。

### Task 4: 补反链与变更记录

**Files:**
- Modify: `wiki/01_theory/04_posttraining/24_agentic_rl_algorithm_analysis.md`
- Modify: `wiki/changelog.md`

**Interfaces:**
- Consumes: Qwen3.8 权威页。
- Produces: 2T+ 开放权重模型横向对照入口与可审计摄入记录。

- [x] **Step 1:** 在 Agentic RL 权威页的 Related Pages 增加 Qwen3.8 工业配方反链。
- [x] **Step 2:** 在 changelog 顶部记录来源基线、核心结论、边界与修改文件。

### Task 5: 验证

**Files:**
- Verify: 所有本次新增和修改文件。

**Interfaces:**
- Consumes: Tasks 1–4 的完整改动。
- Produces: 无断链、无歧义索引、无 Mermaid 语法风险、无无源关键断言的可交付变更。

- [x] **Step 1:** 运行 `python tools/check_links.py --strict`，要求 `broken=0`、`ambiguous=0`、`bare_index=0`。
- [x] **Step 2:** 运行仓库测试；若无专门文档测试，至少运行 `pytest -q`。
- [x] **Step 3:** 用 `rg -n mermaid` 逐块复核 Mermaid，并搜索所有新增 wikilink 目标。
- [x] **Step 4:** 检查 `git diff --check`、`git diff --stat` 与最终 diff，确保没有覆盖用户已有改动。
