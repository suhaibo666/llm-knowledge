# 知识库分解方法论 — 源忠实分析（Source-Faithful Analysis）

> 本目录是 llm-knowledge 知识库的**标准分解要求**：把任何一个源（论文 / 代码 / 规格 / 数据集 / 任意材料）拆解、深挖、落成 wiki 页时，**都应遵循这套流程**。它与根目录 `CLAUDE.md` 互补——`CLAUDE.md` 管 wiki 的*结构与约定*（页面类型、命名、交叉引用、changelog），本方法论管*分析与分解的过程*（怎么忠实于源、怎么抓本质、怎么拆页、怎么并行落地）。

镜像自全局技能 `~/.claude/skills/source-faithful-analysis`，签入仓库便于无技能环境（他人 / CI / 子代理）也能照此执行。

---

## 两条原则（贯穿全部）

1. **源忠实（Be faithful to the source）**：每条非平凡断言都带一个**已核验的定位符**（`file:line` / `§·Table·Fig·Eq` / clause / column·row / timestamp），引用前先打开源读到那一处；钉死基线（commit / arXiv 版本 / 快照）；区分"源陈述 X"与"我推断 Y"；当源与坊间说法/营销/先验冲突，**以源为准并显式标出**。
2. **抓本质（Capture the essence / 抓住重点）**：先给**一条主线**（thesis）；每个单元讲**为什么**而非只讲是什么——动机 → 机制 → 证据 → 为什么不选替代方案；复现**带 baseline 列**的关键证据；点出取舍/代价/边界。不要罗列函数签名或复述摘要。

## 工作流（Phase 0–5，详见核心文档）

| 阶段 | 做什么 |
|---|---|
| 0 锚定 | 拿到源 + 可引用形态（代码 clone+grep；论文 PDF→页码 dump；…）、基线、参考风格、粒度（概要 / 概要+深挖系列） |
| 1 建图 | 先看骨架（目录 / 章节图表 / TOC+schema），分 live-vs-legacy / 新-vs-背景，定入口，**按 subsystem×depth / theme / aspect 拆页并先定页名** |
| 2 定位-读-引 | locate → 定点读 → 引用已核验定位符 + 用自己的话讲机制与为什么（保真循环） |
| 3 并行 fan-out | 大且可分时，一页一个 writer-agent，走严格契约；**先写一页校准页 + 建图工具链**，再铺开 |
| 4 整合校验 | 写概要/索引、更新仓库主脊（`index.md` / `changelog` / `[[links]]`）、渲染图、订正旧估算、**机械核验 0 悬空链接** |
| 5 按需生长 | 后续提问暴露的缺口 → 回到源重读 → 折回成新页/小节，绝不凭记忆作答 |

## 按来源类型选包

确定分析对象后，读对应包拿到**具体定位符格式、摄入配方、本质清单、专属红旗**：

| 源 | 读 | 定位符 | 基线 |
|---|---|---|---|
| 代码 / 框架 / 库 | [`references/codebase.md`](references/codebase.md) | `file:line` | commit / branch / date |
| 论文（arXiv/PDF） | [`references/paper.md`](references/paper.md) | `§ / Table / Fig. / Eq.` | arXiv id + **版本** |
| 其它（规格/RFC、数据集、API、在跑系统/日志、报告、产品） | [`references/general.md`](references/general.md) | clause / column·row / endpoint / timestamp | 版本 / 快照 / 时间窗 |

并行写作的契约模板：[`references/parallel-agent-contract.md`](references/parallel-agent-contract.md)。
完整方法论核心：[`source-faithful-analysis.md`](source-faithful-analysis.md)。

> 跨源是常态、不是例外——一篇模型论文要**同时**对照它 released 的 `config.json`/repo（论文给"为什么"、权重给"是什么"），两个包都读。

## 本库的范例落地

GLM-5（arXiv 2602.15763v2）系列即按此方法论产出：概要 [[glm_5_analysis]] + 7 篇逐章深挖（架构/数据/训练Infra/后训练/Agentic RL/稳定性/低精度）+ 完整模型结构图（取自 released config，抓到"论文 80 层 vs 权重 78 层"contradiction）+ 掩盖/缓存时间线图。流程图工具链（SVG→PNG）在 gitignored `.html2md/`（见记忆 `html-to-md-tooling`），渲染产物 PNG 提交到各页 `assets/`。
