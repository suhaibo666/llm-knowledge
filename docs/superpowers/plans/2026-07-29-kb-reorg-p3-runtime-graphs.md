# 知识库整改 P3:runtime_graphs(现 06_graphs)目录内去重 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 `wiki/02_engineering/01_ai_frameworks/06_graphs/` 内 4 组自重复(spec §3.3):删 2 个 README、时序图页并入 Guide、npu Graph Tree 双写合并、comparison 收缩为差异表。14 页 → 10 页,净删 ≈3000 行。

**Architecture:** 纯编辑类合并,每个合并对独立 commit(可单独回退)。目录本身不移动(P4 才改名为 03_runtime_graphs),所有操作在当前路径进行。

**Tech Stack:** 手工编辑 + `tools/check_links.py`(P0 产物)验收;mermaid 块按 CLAUDE.md「Mermaid 规范与生成后校验」逐块过检查单。

**Spec:** `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.3

**编辑总规则(每个任务都适用):**
1. **只搬运/压缩既有文字,不新造技术断言**;被并方独有的事实必须全部落到权威页,拿不准是否重复的先保留。
2. 两页说法冲突 → 都保留 + `> [!contradiction]` callout(CLAUDE.md 约定)。
3. 删页前先 `python tools/check_links.py --json` 找出全部入链并改指新目标;删后 broken 数不得高于任务开始前。
4. 每个任务结束:checker + `git add -A && git commit`。

**前置:** P0–P2 已合入 main;`git checkout -b reorg/p3`;记录起始 `python tools/check_links.py` 计数。

---

### Task 1: 删 `cuda/README.md`(与 index/Guide 职能重复)

**Files:**
- Delete: `wiki/.../06_graphs/cuda/README.md`(288 行,H1 与 Guide 一字不差)
- Modify: `wiki/.../06_graphs/cuda/index.md`(26 行)

- [ ] **Step 1: 对照三个文件的结构**

```bash
cd /e/97-codes/torch_parallel/llm-knowledge/wiki/02_engineering/01_ai_frameworks/06_graphs/cuda
grep -n "^#" README.md
grep -n "^#" index.md
grep -n "^#" PyTorch_CUDA_Graphs_Complete_Guide.md | head -20
```

- [ ] **Step 2: 把 README 中 index/Guide 都没有的导航性内容并入 `index.md`**(预期主要是「四种使用方式概览」对照表——若 Guide 已有同表则只在 index 放链接不复制表)。index.md 保持纯导航:条目 + 一句话 + 链接,不放机制正文。

- [ ] **Step 3: 改入链后删除**

```bash
cd /e/97-codes/torch_parallel/llm-knowledge
python tools/check_links.py --json | grep -i "cuda/README"   # 找入链(README 基名有歧义,注意甄别指向本目录的)
git rm wiki/02_engineering/01_ai_frameworks/06_graphs/cuda/README.md
python tools/check_links.py    # broken 不升
```

- [ ] **Step 4: Commit** — `docs(graphs): fold cuda/README into index, drop duplicate overview`

---

### Task 2: `CUDA_Graphs_Timing_Diagrams.md` 时序图内联进 Guide 后删除

**Files:**
- Modify: `wiki/.../cuda/PyTorch_CUDA_Graphs_Complete_Guide.md`(1564 行)
- Delete: `wiki/.../cuda/CUDA_Graphs_Timing_Diagrams.md`(627 行)

- [ ] **Step 1: 建立四段映射**(两文件同为「方式1 backend=cudagraphs / 方式2 reduce-overhead / 方式3 torch.cuda.graph() / 方式4 make_graphed_callables」四段结构):

```bash
grep -n "方式\|^#" CUDA_Graphs_Timing_Diagrams.md
grep -n "方式\|^#" PyTorch_CUDA_Graphs_Complete_Guide.md
```

- [ ] **Step 2: 逐段搬运**:每段的时序图(mermaid/代码块)插到 Guide 对应小节的文字讲解之后;图旁只保留 Timing 页独有的解说句,两页重复的文字丢弃。**每搬一个 mermaid 块立即按 CLAUDE.md 检查单过一遍**(形状内嵌套定界符/裸 `[]()`/管道标签引号/换行)。

- [ ] **Step 3: 改入链 → 删除 → 验证**:checker 找 `Timing_Diagrams` 入链改指 Guide 对应小节(`[[PyTorch_CUDA_Graphs_Complete_Guide#方式N ...]]`);`git rm` 删页;checker broken 不升;Guide 预期净增 ≤300 行(只收图和独有解说)。

- [ ] **Step 4: Commit** — `docs(graphs): inline timing diagrams into CUDA graphs guide`

---

### Task 3: 删 `npu/README.md`(三写 overview 归一)

**Files:**
- Delete: `wiki/.../06_graphs/npu/README.md`(125 行)
- Modify: `wiki/.../06_graphs/npu/aclgraph.md`(269 行,承接 overview 职能)、`wiki/.../06_graphs/npu/index.md`(31 行,纯导航)

- [ ] **Step 1:** `grep -n "^#" npu/README.md npu/aclgraph.md npu/index.md` 对照;README 中 aclgraph.md 没有的段(预期:「与 CUDA Graphs 的对应关系」小节)并入 aclgraph.md;与 [[comparison]] 重复的对照内容不并,改为链接(Task 5 会收缩 comparison)。
- [ ] **Step 2:** 改入链 → `git rm npu/README.md` → checker 不升。
- [ ] **Step 3: Commit** — `docs(graphs): fold npu/README into aclgraph overview`

---

### Task 4: npu Graph Tree 双写合并(本阶段最大编辑,4095 行 → 单权威页)

**Files:**
- Modify: `wiki/.../npu/torch_compile_npugraphs_deep_dive.md`(2397 行,主干保留)
- Delete: `wiki/.../npu/npugraphs_memory_reuse_analysis.md`(1698 行,吸收后删除)

- [ ] **Step 1: 画双方章节树**

```bash
grep -n "^#" torch_compile_npugraphs_deep_dive.md
grep -n "^#" npugraphs_memory_reuse_analysis.md
```

重叠区(spec 数据):deep_dive §三「NPU Graph Tree 核心机制」(~589 行)↔ memory_reuse 的「Graph Tree 机制」+「内存复用策略」+「@torch.compile 场景案例」(~800 行)。注意 memory_reuse 内有「关键代码解析(合并自 memory_management)」节——上次没合干净的痕迹,本次一并处理。

- [ ] **Step 2: 逐节对比搬运**:以 deep_dive §三 为骨架;memory_reuse 独有的内容(预期:内存池 checkpoint 细节、复用策略的代码级解析、@torch.compile 场景案例)搬入 §三 对应子节或新增子节;两页共有的内容以 deep_dive 表述为准丢弃 memory_reuse 版本;源码行号引用一律保留出处更细的那份。

- [ ] **Step 3: mermaid 逐块校验**(搬运过的每块过检查单)。

- [ ] **Step 4: 改入链 → 删页 → 验证**:checker 找 `npugraphs_memory_reuse_analysis` 全部入链(含各 index)改指 deep_dive;`git rm`;checker 不升。体量核对:deep_dive 净增 ≤500 行(2397→≤2900),两页合计净删 ≥1200 行。

- [ ] **Step 5: Commit** — `docs(graphs): merge Graph Tree duplicate into npugraphs deep dive`

---

### Task 5: deep_dive 内两处收缩(§四 与 附录 A)

**Files:**
- Modify: `wiki/.../npu/torch_compile_npugraphs_deep_dive.md`
- Modify(可能): `wiki/.../npu/npugraphs_make_graphed_callables_deep_dive.md`(669 行)、`wiki/.../npu/aclgraph_deep_analysis.md`(570 行)

- [ ] **Step 1: §四「与 make_graphed_callables 的对比」收缩**:该节(原 1471–1561 行附近)只留结论对比表 + `[[npugraphs_make_graphed_callables_deep_dive]]` 链接;节内超出 make_graphed_callables 页的细节先搬去该页再删。
- [ ] **Step 2: 附录 A「reduce-overhead 完整编译流程」收缩**(spec 决策:捕获路径的权威页 = `aclgraph_deep_analysis`):附录 A 独有内容搬入 `aclgraph_deep_analysis.md`,附录本体替换为一段摘要 + 链接。
- [ ] **Step 3:** checker 不升;commit — `docs(graphs): dedupe make_graphed_callables and reduce-overhead sections`

---

### Task 6: `comparison.md` 收缩为差异表(632 行 → ≤300 行)

**Files:**
- Modify: `wiki/.../npu/comparison.md`

- [ ] **Step 1:** `grep -n "^#" comparison.md`,识别哪些节是**复述**两侧各自机制(API 讲解、实现原理讲解、代码示例)、哪些是**真对比**(差异表、行为差异说明)。
- [ ] **Step 2:** 复述节替换为一句话 + 指向权威页链接(cuda 侧 → Guide,npu 侧 → aclgraph/deep_dive);对比表和差异结论全保留。
- [ ] **Step 3:** checker 不升;commit — `docs(graphs): reduce comparison page to genuine diff tables`

---

### Task 7: 目录 index 刷新 + 阶段门 + 合回 main

**Files:**
- Modify: `wiki/.../06_graphs/index.md`(22 行)、`cuda/index.md`、`npu/index.md`

- [ ] **Step 1:** 三个 index 更新为删并后的真实页面清单(cuda:Guide + index;npu:aclgraph、aclgraph_deep_analysis、aclgraph_multistream_rng_analysis、torch_compile_npugraphs_deep_dive、npugraphs_make_graphed_callables_deep_dive、comparison + index),每条带一句话定位。
- [ ] **Step 2: 阶段门**

```bash
python -m pytest tools/test_check_links.py -q
python tools/check_links.py        # broken ≤ P3 起始值;06_graphs 下 md 数 = 10
ls wiki/02_engineering/01_ai_frameworks/06_graphs -R
git status --short
```

- [ ] **Step 3:** `wiki/changelog.md` 追加 P3 条目(删了哪 4 页、并入哪里);merge 回 main(`git merge --no-ff reorg/p3`),删分支。
- [ ] **Step 4:** 完成后回到路线图 `2026-07-29-kb-reorg-00-roadmap.md`,把 P3 标 ✅,并按 spec §2 映射表 + §3.1 编写 P4 计划(`kb-reorg-p4-*.md`)。
