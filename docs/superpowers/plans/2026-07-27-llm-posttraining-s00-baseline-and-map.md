# LLM 后训练前沿 S00 当前快照与学习入口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立统一的 `wiki/03_posttraining/` 入口，完成可追溯的 2026-07-27 前沿快照 D01 和可顺序阅读的学习入口 D00。

**Architecture:** S00 只建立研究坐标、来源基线和学习导航，不提前代写 S01–S04 的机制深挖。D01 负责回答“当前有哪些主线、证据来自哪里、后续为什么要深挖”，D00 负责回答“按什么顺序学习、每一步学到什么程度”，`index.md` 负责把两者接入知识库并连接旧理论/工程页面。

**Tech Stack:** Markdown、Obsidian Wiki Links、Git、PowerShell、GitHub REST API、arXiv 与项目官方文档

## Global Constraints

- 所有新研究文档统一位于 `wiki/03_posttraining/`，不得把 D00–D11 分散回旧理论和工程目录。
- 旧目录 `wiki/01_theory/04_posttraining/` 与 `wiki/02_engineering/04_posttrain_frameworks/` 不迁移、不复制，只作为关联来源。
- 页面标题和文件名前缀必须使用两位数字；D00 为学习入口，D01 为当前前沿地图。
- 论文结论绑定 arXiv ID 与版本；框架结论绑定仓库、分支、commit 和核验日期。
- 关键技术事实只采用论文、官方源码、官方文档、release note、issue/PR 或作者材料。
- 无法从公开一手来源验证的工业说法标记为“公开信息不足”或“研究判断”。
- 当前工作区已有未提交修改；只暂存本研究文件，不覆盖或提交用户已有变更。
- `wiki/changelog.md` 已存在用户修改，若增加 S00 条目，必须只暂存本次新增 hunk。
- 不更新、重置或清理现有源码仓库；S00 通过只读远端查询固定基线。

---

## File Map

| 文件 | 责任 |
|---|---|
| `wiki/03_posttraining/index.md` | 统一领域入口、范围边界、阅读顺序、旧知识链接和当前完成状态 |
| `wiki/03_posttraining/00_posttraining_source_reading_guide.md` | D00 学习路线、阶段能力门槛、上一页/下一页导航和源码阅读方法 |
| `wiki/03_posttraining/01_posttraining_frontier_map_analysis.md` | D01 当前前沿地图、来源快照、算法—Infra—框架—硬件坐标和深挖队列 |
| `wiki/index.md` | 增加 `03_posttraining` 顶级纵向领域和快速导航入口 |
| `wiki/changelog.md` | 记录 S00 新领域、D00 和 D01 的建立；仅暂存新增条目 |

## Task 1: 固定 S00 来源基线并完成 D01 前沿地图

**Files:**
- Create: `wiki/03_posttraining/01_posttraining_frontier_map_analysis.md`
- Read: `wiki/01_theory/04_posttraining/index.md`
- Read: `wiki/02_engineering/04_posttrain_frameworks/index.md`
- Read: `wiki/02_engineering/04_posttrain_frameworks/verl/index.md`

**Interfaces:**
- Consumes: 现有理论与工程页面、四个框架官方仓库、当前日期 2026-07-27
- Produces: D01 的固定分类、框架基线表、现有覆盖/知识空白表和 D02–D11 的研究入口

- [ ] **Step 1: 盘点现有后训练知识覆盖**

Run:

```powershell
rg --files wiki/01_theory/04_posttraining wiki/02_engineering/04_posttrain_frameworks |
  Sort-Object
```

Expected:

- 输出既有算法页、Infra 页和 verl 源码分析页；
- 不存在 `wiki/03_posttraining/` 中的同名复制；
- 将既有页面分为“可直接复用、需要复核、当前缺失”三类。

- [ ] **Step 2: 只读固定四个框架远端基线**

Run once for each of `verl-project/verl`、`THUDM/slime`、`inclusionAI/AReaL`、`alibaba/ROLL`:

```powershell
$headers = @{'User-Agent'='Codex'}
$repo = 'verl-project/verl'
$meta = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$repo"
$head = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$repo/commits?per_page=1"
[PSCustomObject]@{
  repo = $repo
  branch = $meta.default_branch
  pushed_at = $meta.pushed_at
  commit = $head[0].sha
  commit_date = $head[0].commit.committer.date
  message = $head[0].commit.message
}
```

Expected:

- 每个项目得到仓库、默认分支、40 位 commit、commit 日期与最近提交标题；
- 查询不执行 `git fetch`，不改变本地源码仓库；
- D01 页面头部记录统一核验日期 `2026-07-27`。

- [ ] **Step 3: 核验前沿来源**

优先打开并核验：

- 四个框架的官方 README、release、文档和架构页；
- `DeepSeek-R1`、GRPO、DAPO、GSPO 及现有 TIM 页面引用的一手论文；
- Agentic RL、fully-async、train-inference mismatch、rollout 长尾和 environment/sandbox 的官方论文或项目页；
- 2025-01-01 至 2026-07-27 发布、且能明确说明新机制或新系统约束的候选工作。

每个候选项记录：

| 字段 | 内容 |
|---|---|
| Track | Reasoning RL、Agentic RL、Infra、Framework、Hardware |
| Problem | 解决的失败模式或瓶颈 |
| Mechanism | 一句话机制，不复述摘要 |
| Primary source | 论文版本或官方源码/文档 |
| Public implementation | 官方实现及固定 commit；没有则写“未发现官方实现” |
| Evidence level | 论文、代码、示例、CI/测试、实测 |
| Deep-dive target | D02–D11 中对应页面 |

Expected:

- 二手文章只用于发现线索，不作为表格的唯一来源；
- 将来源事实、机制推导和研究判断分开书写；
- 不写无硬件与配置上下文的性能倍数。

- [ ] **Step 4: 写 D01**

`01_posttraining_frontier_map_analysis.md` 使用以下固定结构：

1. 页面头部：`D01`、`S00`、核验日期、来源基线；
2. 一条主线：当前后训练前沿由算法目标、在线数据生成和训练/推理系统共同决定；
3. 五层地图：必要基础、Reasoning RL、Agentic RL、Infra、框架与硬件；
4. 四框架定位表：verl、slime、AReaL、ROLL；
5. 现有知识复用表：链接旧算法、verl、sandbox、TIM 页面；
6. 知识空白与 D02–D11 深挖队列；
7. 证据强度与 staleness 说明；
8. 上一页 `[[00_posttraining_source_reading_guide]]`、下一页 `[[02_reasoning_rl_algorithm_evolution_analysis]]`，下一页标为“计划于 S01 建立”；
9. `## Related Pages`。

- [ ] **Step 5: 验证 D01**

Run:

```powershell
rg -n '\b(TBD|TODO)\b|待补|待定|据说|业内认为' wiki/03_posttraining/01_posttraining_frontier_map_analysis.md
rg -n 'https?://|arXiv|commit|核验日期|来源事实|研究判断' wiki/03_posttraining/01_posttraining_frontier_map_analysis.md
```

Expected:

- 第一条命令无输出；
- 第二条命令能定位来源、版本和事实/判断边界；
- D01 本身不伪装成 D02–D11 的完整机制分析。

## Task 2: 建立 D00 学习路线与统一领域索引

**Files:**
- Create: `wiki/03_posttraining/00_posttraining_source_reading_guide.md`
- Create: `wiki/03_posttraining/index.md`
- Read: `docs/superpowers/specs/2026-07-27-llm-posttraining-frontier-research-design.md`

**Interfaces:**
- Consumes: D01、设计中的 S00–S05 和 D00–D11 编号
- Produces: 单一学习入口、顺序导航、能力检查点和旧知识接入关系

- [ ] **Step 1: 写 D00 学习路线**

`00_posttraining_source_reading_guide.md` 必须包含：

- 页面头部：`D00`、所属阶段 `S00/S05`、核验日期；
- `D00 → D11` 的完整顺序表；
- 每篇文档的“学习问题、前置知识、读完后应能完成的任务”；
- 六级能力门槛：数学对象、训练迭代、verl 主链路、异步对照、权重同步与资源布局、新框架/NPU 适配评估；
- 论文阅读方法：问题 → 机制 → 证据 → 替代方案 → 限制；
- 源码阅读方法：入口 → owner → 状态/数据结构 → 调用链 → 通信 → 失败路径；
- 首页导航：上一页为 `[[index]]`，下一页为 `[[01_posttraining_frontier_map_analysis]]`；
- `## Related Pages` 链接旧理论与工程入口。

- [ ] **Step 2: 写统一领域 index**

`wiki/03_posttraining/index.md` 必须包含：

- 领域目标和“不再按理论/工程割裂”的设计理由；
- `S00–S05` 阶段表和 `D00–D11` 阅读表；
- 当前完成状态：S00 的 D00、D01；
- 现有知识入口：旧算法目录、旧 Infra 目录、verl 目录、TIM、sandbox、RL Infra；
- 维护规则：新内容写入本目录，旧内容只链接不复制；
- `## Related Pages`。

- [ ] **Step 3: 验证顺序与导航**

Run:

```powershell
$files = Get-ChildItem wiki/03_posttraining -File |
  Where-Object { $_.Name -match '^\d{2}_' } |
  Sort-Object Name |
  Select-Object -ExpandProperty Name
$files
```

Expected:

```text
00_posttraining_source_reading_guide.md
01_posttraining_frontier_map_analysis.md
```

Run:

```powershell
rg -n 'D0[0-9]|D1[01]|S0[0-5]|上一页|下一页|Related Pages' wiki/03_posttraining
```

Expected:

- D00/D01 页面包含编号、阶段、导航和 Related Pages；
- index 同时列出完整规划与当前完成状态；
- 不出现旧目录中的新文件路径。

## Task 3: 接入总索引、记录变更并完成机械验证

**Files:**
- Modify: `wiki/index.md`
- Modify: `wiki/changelog.md`
- Verify: `wiki/03_posttraining/index.md`
- Verify: `wiki/03_posttraining/00_posttraining_source_reading_guide.md`
- Verify: `wiki/03_posttraining/01_posttraining_frontier_map_analysis.md`

**Interfaces:**
- Consumes: Task 1 和 Task 2 的三个新页面
- Produces: 可从知识库首页进入的纵向领域、可审计变更记录和只包含 S00 文件的提交

- [ ] **Step 1: 更新总索引**

在 `wiki/index.md` 中：

- 将目录树增加 `03_posttraining/`；
- 在“领域总览”增加 `03 后训练纵向学习域`，入口为 `[[03_posttraining/index]]`，页面数为 3；
- 在“按主题查找”增加“LLM 后训练前沿学习路线”，链接 `[[03_posttraining/index]]`、`[[00_posttraining_source_reading_guide]]`、`[[01_posttraining_frontier_map_analysis]]`；
- 更新首页日期为 `2026-07-27`。

- [ ] **Step 2: 更新 changelog 且保护用户修改**

在 `wiki/changelog.md` 顶部既有格式下增加 2026-07-27 条目，内容只描述：

- 新增统一领域 `wiki/03_posttraining/`；
- 新增 D00 学习路线；
- 新增 D01 前沿地图；
- 旧理论/工程页面保持原位并通过链接复用。

暂存前比较 `HEAD`、工作区与本次新增 hunk，确保不把先前用户修改放入暂存区。

- [ ] **Step 3: 检查 Wiki 链接目标**

Run:

```powershell
$root = Resolve-Path wiki
$newFiles = Get-ChildItem wiki/03_posttraining -Filter *.md
$missing = @()
foreach ($file in $newFiles) {
  $text = Get-Content -Raw -Encoding UTF8 $file.FullName
  foreach ($m in [regex]::Matches($text, '\[\[([^\]|#]+)')) {
    $target = $m.Groups[1].Value
    $name = [IO.Path]::GetFileName($target)
    $found = rg --files wiki | Where-Object {
      [IO.Path]::GetFileNameWithoutExtension($_) -eq $name -or
      $_ -replace '\\','/' -eq "wiki/$target.md"
    }
    if (-not $found -and $target -ne '02_reasoning_rl_algorithm_evolution_analysis') {
      $missing += "$($file.Name) -> $target"
    }
  }
}
$missing
```

Expected:

- 无输出；
- 唯一允许的前向规划页 D02 在 D01 中明确标为 S01 计划，不作为已完成链接目标。

- [ ] **Step 4: 检查格式与占位符**

Run:

```powershell
rg -n '\b(TBD|TODO)\b|待补|待定|<<<<<<<|=======|>>>>>>>' wiki/03_posttraining
git diff --check -- wiki/03_posttraining wiki/index.md wiki/changelog.md
```

Expected:

- 两条命令均无错误；
- 页面没有未完成占位符、冲突标记或尾随空白。

- [ ] **Step 5: 暂存并审计范围**

Stage:

```powershell
git add -- wiki/03_posttraining wiki/index.md
```

对 `wiki/changelog.md` 只暂存 S00 新增 hunk，然后运行：

```powershell
git diff --cached --name-only
git diff --cached --check
git diff --cached --stat
```

Expected staged files:

```text
wiki/03_posttraining/00_posttraining_source_reading_guide.md
wiki/03_posttraining/01_posttraining_frontier_map_analysis.md
wiki/03_posttraining/index.md
wiki/changelog.md
wiki/index.md
```

不得暂存旧理论页、旧工程页、源码仓库或其他用户修改。

- [ ] **Step 6: 提交 S00**

Run:

```powershell
git commit -m "docs(posttraining): establish S00 learning map"
```

Expected:

- commit 成功；
- 提交只包含 S00 五个文件；
- 工作区中用户既有修改仍然保留。
