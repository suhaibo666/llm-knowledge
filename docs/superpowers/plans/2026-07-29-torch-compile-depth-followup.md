# Torch.compile 深度复检修订 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复第二轮复检指出的 CommonMark/Mermaid 缺陷，并让 A01–A05 与六篇调用链重点页都具备可机械识别、源码忠实的“源码跟读”主链。

**Architecture:** 先用课程结构测试锁定有序列表、Mermaid 管道标签和目标页源码跟读契约，再以 `E:/97-codes/torch_parallel/p@e8f97c1a` 为唯一事实源逐页定位调用链。A01–A05 新增独立主链；B04、B07、D01、D02、D06、F01 复核现有主链并只补真实缺口，避免重复章节和定位灌水。

**Tech Stack:** Markdown、Python `unittest`、PyTorch 固定源码 checkout、课程 claim ledger。

## Global Constraints

- 不修改 `E:/97-codes/torch_parallel/pytorch` 的脏工作区。
- 所有新增源码定位必须先打开 `p@e8f97c1a` 对应范围。
- 正式 `[S]` evidence span 不超过 30 行；正文定位不超过 100 行。
- Mermaid 连线文字使用 `A -->|文字| B`，标签内不放引号、括号或 `|`。
- 不删除既有正文，只扩展或修正。
- 本轮不提交、不推送。

---

### Task 1: 锁定复检结构缺陷

**Files:**
- Modify: `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/test_volume_demo_contract.py`
- Modify: `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/f03_ddp_compile_boundaries_and_optimizer_analysis.md`
- Modify: `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs.md`

**Interfaces:**
- Consumes: 63 篇课程 Markdown。
- Produces: 对代码围栏外无空格有序列表、Mermaid 管道标签引号以及目标页源码跟读的结构门禁。

- [ ] **Step 1: 写失败测试**

新增三个行为检查：代码围栏外 `^\s*\d+\.(?!\s)` 必须为 0；管道标签内容不得以引号包裹；A01–A05、B04、B07、D01、D02、D06、F01 每篇至少有一个 `## ...源码跟读` 章节且至少包含五个源码定位。

- [ ] **Step 2: 运行测试确认失败**

运行目标 `unittest`。预期：有序列表在 F03 失败、Mermaid 在 C09 失败、源码跟读契约仅 A01–A05 失败；六篇重点页应证明当前工作区已经存在对应章节。

- [ ] **Step 3: 修复机械缺陷**

给 F03 四条有序列表补空格；删除 C09 两条管道标签中的引号。

- [ ] **Step 4: 运行机械门禁**

预期有序列表和 Mermaid 测试通过，A 卷源码跟读测试仍失败。

### Task 2: A01–A05 源码跟读

**Files:**
- Modify: `a01_tensor_storage_layout_and_views_analysis.md`
- Modify: `a02_operator_schema_dispatch_and_autograd_analysis.md`
- Modify: `a03_python_frames_code_objects_and_bytecode_analysis.md`
- Modify: `a04_dispatch_modes_proxy_tensor_and_fake_tensor_analysis.md`
- Modify: `a05_eager_capture_compile_and_replay_cost_model_analysis.md`

**Interfaces:**
- Consumes: 固定 PyTorch 源码基线及每页既有概念主线。
- Produces: 五条端到端源码路径，分别覆盖 Tensor/Storage/View、schema/dispatcher/autograd、frame/eval-frame/bytecode、mode/proxy/fake、compile first-call/cache/runtime。

- [ ] **Step 1: 定位并打开每条调用链**

用 `rg` 找入口、关键状态对象、分支与出口；逐个读取不超过 30 行的关键源码片段。

- [ ] **Step 2: 写 A01–A05 主链**

每篇新增“入口 → 状态/所有权 → 关键分支 → 出口/consumer → why-not/失败边界”，并使用已核验定位。

- [ ] **Step 3: 运行源码跟读契约**

预期十一篇目标页全部满足章节和最小定位密度要求。

### Task 3: 六篇重点页复核与证据重建

**Files:**
- Review/Modify only if needed: B04、B07、D01、D02、D06、F01
- Modify: `docs/audits/torch_compile_end_to_end/2026-07-28/course_claim_decisions/*.jsonl`
- Modify: `course_claim_ledger.jsonl`
- Modify: `course_claim_summary.md`
- Modify: `review_remediation_report_2026-07-29.md`

**Interfaces:**
- Consumes: 十一篇源码跟读和固定基线。
- Produces: 无伪造定位、无证据降级、无推理环的最新总账与报告。

- [ ] **Step 1: 复核六篇现有主链**

确认每篇已有显式 `源码跟读`、真实调用链、状态/所有权、失败边界和精确定位；只补缺失维度。

- [ ] **Step 2: 重建 decisions 与 ledger**

运行课程决策生成器和 ledger validator，要求 claim/decision 一一对应且错误为 0。

- [ ] **Step 3: 全量验证**

运行 99 项审计工具测试、66 项 Labs 测试、全部源码定位与 Markdown/链接/Mermaid 门禁、`git diff --check`。

- [ ] **Step 4: 更新最终报告**

记录 A 卷新增主链、六篇现有主链的复核事实、机械缺陷修复和最终数字；明确没有提交或推送。
