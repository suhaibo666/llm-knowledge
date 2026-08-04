# 知识库整改 — 总路线图(计划索引)

> **For agentic workers:** 本文件是路线图,不是可执行计划。执行请打开各阶段计划文档,并用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务执行。

**Spec:** `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`(已批准,含全部数据锚点与决策)

**Goal:** 把 400 页知识库整改为"功能树唯一权威 + courses 薄导读"结构:消除 13 组高重叠、坏链清零、ai_frameworks 重组为 5 个架构层两级目录、仓库瘦身约 60MB。

## 计划文档索引(滚动编写)

| 阶段 | 计划文档 | 状态 |
|---|---|---|
| P0 工具与基线 + P1 快速止血 + P2 图源入库 | `2026-07-29-kb-reorg-p0-p2-foundation.md` | ✅ 已执行完(merge 77fd28a,broken 138→0) |
| P3 runtime_graphs(原 06_graphs)去重 | `2026-07-29-kb-reorg-p3-runtime-graphs.md` | ✅ 已执行完(merge 65a8fe2,14→10 页,净删 ~3300 行) |
| P4 ai_frameworks 两级重组 + 19 号解散 | `2026-07-30-kb-reorg-p4-ai-frameworks.md` | ✅ 已执行完(merge c5c084e,57 commits:5 架构层两级重组、19 号 63 篇解散、8 个旧大文判重处置、118 页分段编号、课程页落地;wiki 398→375 页,broken=0 全程) |
| P5 后训练三域整合 + courses/posttraining | `2026-07-31-kb-reorg-p5-posttraining.md` | ✅ 已执行完(merge e9e22f7:03_posttraining 域解散、GRPO 三写归一、verl 双基线整合含 use_v1 翻转发现、36 页分段编号;wiki 375→374,broken=0 全程) |
| P6 横向页收缩 + P7 命名/索引/CLAUDE.md 收尾 | `2026-07-31-kb-reorg-p6-p7-finale.md` | ✅ 已执行完(P6 merge 44d642d:13 组高重叠全清;P7 merge 2c33414:全库编号+spec §3.4 补执行+链接四项全零+CLAUDE.md/README 定稿) |

**整改全程完成(2026-08-04)**:wiki 402→372 页;高重叠 13 组清零;检查器 broken/ambiguous/bare_index/orphans 全零;后续维护按修订版 CLAUDE.md 执行。

后续计划的**决策已全部锁定在 spec §2/§3**(含旧→新目录映射表、五大重复组逐组处置、C 卷分发规则);阶段计划只需把 spec 决策展开为逐文件任务,不得改变决策。

## 阶段门(每阶段结束必须全部满足才进入下一阶段)

1. `python tools/check_links.py` 的 broken 计数 ≤ 上一阶段基线(P1 起应持续下降,P7 结束 = 0)
2. 本阶段删除/移动的每个文件,入链已全部修复(check_links 无新增 broken)
3. 涉及 mermaid 图移动的,按 CLAUDE.md「Mermaid 规范与生成后校验」逐块过检查单
4. `git status` 干净,阶段分支合回 main

## 分支策略(重要)

用户有多个并行会话向 main 提交内容(如 `5bfd529`)。**禁止开一条长命整改分支**——每阶段开短命分支 `reorg/p<N>`,阶段门通过后立即合回 main 并删分支;开工前先 `git pull`/确认 main 最新。若阶段执行中 main 有新提交,合回前 rebase 并重跑 check_links。

## 风险与回退

- 所有删除均有 git 历史兜底,唯一例外:`docs/reports/*.docx` 是 untracked,删除不可恢复(P1 任务里有内容已被 wiki 覆盖的前置核验)。
- 编辑类合并(P3–P6)每个合并对单独 commit,发现合错单对回退,不影响其他。
- P4 的目录移动(git mv)与内容合并分开提交:先纯移动+修链一个 commit,再逐对合并。
