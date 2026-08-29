---
name: drawing-wiki-figures
description: Use before adding any figure to a wiki page - whether the content deserves a figure at all, which medium to pick (mermaid / generated SVG / ASCII / table), the house visual spec and the verified generate-to-external-SVG pipeline, and the write-the-spec-first workflow. Load it whenever a page needs a picture; for mermaid's own parser traps read writing-mermaid-diagrams as well.
---

# Drawing Wiki Figures

本库 371 个内容页里，**152 页（41%）既没有 mermaid 也没有渲染图**，全靠 ASCII 框线撑着；而
`15_megatron_pp_schedulers_analysis.md` 一页 1024 行、456 行是框线字符、零张图——流水线时序本质
是一张二维网格（时间 × stage），用等宽字符画是所有媒介里最差的一种。

这篇技能管三件事：**该不该画、用什么画、画成什么样**。

## 1. 该不该画 —— 先砍，再画

### 只画三样

1. **机制流程** —— 输入 → 各模块 → 输出，这条路到底怎么走
2. **核心机制的内部构造** —— 最需要解释的那一处，拆开看
3. **最有说服力的那一个结果 / 代价** —— 不是全部实验

### 不画这四样

- ❌ 背景、相关工作、术语介绍 —— 那是正文 beat 1 的事
- ❌ 抽象的"设计哲学"示意 —— 没有信息量的图等于噪声
- ❌ **一句话能讲清的东西** —— 一句话讲得完就别画
- ❌ 第 N 个消融的柱状图

### 两条验收线

> **A. 陌生读者线**：一个没读过源码 / 没读过论文的人，只看这张图 + 图上的标注，能不能理解这个机制？
> 不能 → 拆成两张，或者补标注。
>
> **B. 非复刻线**：这张图跟上游的 Figure 1 / 官方架构图长得一样吗？
> 一样 → 那你画它没有意义。**重画的价值在于重新组织信息**——按你正文的五拍顺序重组，标出源图没标的
> 维度、代价、被否掉的路径。

B 线是本库的旧伤：`8f6f2f4`（重画 ACT 深潜页三张配图）返工的根因就是图直接沿用了上游示意、与正文
立论脱节，上游一改就整体失效。**图必须服务本页的 thesis，不是给上游图配个中文标题。**

## 2. 用什么画 —— 媒介选型

| 内容形态 | 媒介 | 为什么 |
|---|---|---|
| 调用链、pass pipeline、状态机、模块依赖、决策分支 | **mermaid** `flowchart` / `stateDiagram` | 纯拓扑关系，mermaid 原生表达，后续改得动 |
| 多方交互、通信握手、请求生命周期 | **mermaid** `sequenceDiagram` | 同上 |
| **二维网格**：流水线时序（时间×stage）、张量/专家分片布局、内存与显存布局、attention 掩码、通信拓扑几何 | **生成脚本 → 外部 `.svg`**（见 §3.5） | mermaid 无法做精确二维定位，ASCII 更不行。这是本库图示欠债最集中的一类 |
| 甘特型：计算/通信重叠、掩盖、双流 | **生成脚本 → 外部 `.svg`**（见 §3.5） | 需要按时间轴按比例定位 |
| 目录树、两三层的简单层级 | **ASCII** | 本来就合适，保留 |
| 定量对比、超参、消融 | **表格** | 图画不过表格 |

**ASCII 只保留给目录树。** 其他场景里它出现，就是"本该画图但没画"的信号。

不要用文生图模型出位图：它会写错标签、糊掉维度和公式，产物不可 diff、不能改一个数字就重渲染，
图上的数字**无法溯源**——与本库第一原则直接冲突。渲染图一律走生成脚本产 `.svg`（§3.5），文字是真文字。

## 3. 画成什么样 —— 渲染图的视觉规范

设计 token 定义在 `tools/figs/figstyle.css`，分两层：底层是原有的扁平浅色系，上层是
**paper-figure layer**（视觉规范借鉴 zsyggg/paper-craft-skills 的 `styles/paper-figure.md`）。
下面的 class 名是这套 token 的**命名约定**：HTML 图直接用；SVG 生成脚本在自己的 `<style>`
里沿用同名 class 与同一组色值（见 `tools/figs/svg/pp_schedule.mjs`），两边不要各起一套。

### 一条主线：默认中性，强调色只标重点

老的 `.blue/.purple/.green/.amber/.slate/.rose` 是**按类别上色**：六色等权，结果是什么都被强调、
于是什么都没被强调。新图按下面来：

- **多数节点用 `.neutral`**（白底灰边）；背景、对照组、被否掉的路径用 `.ghost`（更淡）
- **每张图最多两个强调色**：`.acc1`（蓝，本页核心贡献 / 关键路径）、`.acc2`（橙，代价 / 瓶颈 / 冲突点）
- 分类语义交给**分组框和图注**，不要交给颜色

### 路径分级

主路径 `.arrow.main`（粗、着色），辅助路径 `.arrow.aux`（细、淡）。读者要能一眼看出先看哪、后看哪。

### 可复用图元

| 图元 | class | 用在 |
|---|---|---|
| 矩阵小格（含热力三档 `.h1/.h2/.h3`、未用 `.x`） | `.cells` / `.cell` | attention 掩码、张量分片、KV cache 占用、专家路由 |
| token 序列（选中 `.on` / 被拒 `.rej`） | `.seq` / `.tok` | 序列并行、投机解码、chunked prefill |
| 局部放大框 | `.zoom` | 画面偏空时补这个，**不要留白** |
| 维度 / dtype 短签 | `.dim` | 贴在图元旁，`[B,S,H] bf16` 这种 |
| 图注 | `.cap` | 像论文 figure caption：短标签，不写整句解释 |

### 密度与文字

- 主体图解占画面 **75%–85%**。偏空就补局部放大框、维度标注、对比路径——不是拉大字号。
- 标注短到像论文图注：模块名、变量名、关键操作、至多一行解释。整句解释放正文。
- 内容超出一页可读范围就**拆成两张**，别硬塞。

### 禁止

蓝底白线工程图纸风、3D 渲染、大段文字解释、复刻上游图布局、做成 PPT 封面或营销海报。

## 3.5 渲染图的产出方式：生成脚本 → 外部 `.svg` → 标准图片语法

**结论（2026-08-29 实测得出，别再重试其它写法）：**

```
数据/仿真 → Node 生成脚本 → wiki/<域>/assets/<name>.svg → ![说明](assets/<name>.svg)
```

参考实现：[`tools/figs/svg/pp_schedule.mjs`](../../tools/figs/svg/pp_schedule.mjs)。

### 为什么是这条

| | 内联 `<svg>` 进 md | **外部 `.svg` + `![]()`** | HTML→PNG（旧法） |
|---|---|---|---|
| Obsidian 编辑态看得见图 | ❌ 几十 KB 文本墙 | ✅ 同普通图片 | ✅ |
| docs-site | ❌ 被 Quartz 打坏（下方三条） | ✅ 渲染为 `<img>`，资产 MD5 一致 | ✅ |
| 文字可 grep / diff | ✅ | ✅ | ❌ 二进制 |
| 缩放 | 矢量 | 矢量 | 固定 2x，放大糊 |
| 依赖 | 无 | 无 | Edge + puppeteer |

外链 `.svg` 由浏览器当**独立 XML 文档**解析，因此 `<defs>` / `<pattern>` / 驼峰属性 /
`<style>` 全部可用，且 `<style>` 天然隔离在该文档内，不会泄漏成全站 CSS。

### 已确认打不通的写法：内联 SVG

Quartz v5 的 `remark-rehype → hast-util-to-jsx-runtime` 会把内联 SVG 打坏三处（原始
HTML 本身是放行的，`<div>` 探针通过——坏的只是 SVG 专属解析）：

1. `<svg …>` 被立刻自闭合，其余元素全部漏到 svg 外面；
2. 驼峰属性被小写化（`patternTransform` → `patterntransform`），而 SVG 属性**大小写敏感** → 失效；
3. 自闭合标签被当成嵌套（`<rect/><rect/>` → `<rect><rect></rect></rect>`）。

### 生成脚本的硬要求：图必须由数据算出来

`.html2md/gen_pp_fig.mjs` 立下的规矩，新脚本一律照办：

> emits a grid whose cells come directly from a discrete-event sim, **so the drawing cannot
> disagree with the schedule**.

即**图上的每个数字都是算出来的，不是手写的**。上面那张 PP 图里 `makespan=22`、在途激活
`8 份 / 4 份`，都来自仿真；改 `node pp_schedule.mjs 8 16` 结论自动重算。这是 Principle 1
（源真实性）在图上的落法——手写的图会和正文一起腐烂，算出来的不会。

拿不到数据源的示意图（架构、概念关系）不适用本条，但那类通常该用 mermaid。

## 4. 先写 spec，再动手画

**不要直接写 HTML。** 先用自然语言把这张图描述到"箭头从哪到哪、方框里写什么字"，再实现——尤其在
派并行 agent 画图时，spec 是防止跑偏的唯一手段。

```
图 2：1F1B 稳态为什么峰值显存是 O(pp) 而不是 O(m)

要讲清楚：同样的气泡率下，GPipe 攒了 m 份激活、1F1B 只在途 ~pp 份。

布局（上下两条时间轴对照）：
- 上半 GPipe：4 行（stage 0-3），横轴时间；先一整片 F 格子，再一整片 B 格子
  右侧 .zoom 框标出"峰值在此：m=8 份激活全部在存"
- 下半 1F1B：同样 4 行；warmup 阶梯 → 稳态 F/B 交替 → cooldown
  右侧 .zoom 框标出"峰值在此：在途 4 份"
- 两条轴的总长度画成一样 —— 这是本图要证的"气泡率相同"

标注：
- 格子：F 用深档、B 用同族浅档、气泡用中性斜纹（`fill:url(#hatch)`）
- 每行左侧标 "stage 0".."stage 3"
- 右侧两个标注框：GPipe 那个用 acc2（代价色），1F1B 那个用 acc1（收益色）
- 两条轴下方各画一条 makespan 尺规——等长就是本图要证的结论
- .cap: "P=4, m=8。气泡率同为 (P-1)/(m+P-1)=3/11；峰值激活 GPipe ∝m、1F1B ∝P"
```

一次要画多张时：**先把第一张画到位当校准页**，再照着它铺开剩下的。图返工成本高，不要 N 张一起错。

## 5. 提交前校验

- [ ] 这张图过了**陌生读者线**和**非复刻线**？
- [ ] 媒介选对了？（二维网格 / 甘特没有用 ASCII 顶替）
- [ ] 强调色 ≤ 2 个，且只标了核心贡献与代价？多数节点是 `.neutral` / `.ghost`？
- [ ] 主/辅路径分级了？
- [ ] 主体占画面 75%–85%，没有大块留白？
- [ ] 图上每个数字、维度、`file:line` 都能在正文或源码里对上？（图和正文同源，不是各写各的）
- [ ] 图注是短标签，不是整句解释？
- [ ] mermaid 图另外按 [`writing-mermaid-diagrams`](../writing-mermaid-diagrams/SKILL.md) 的清单过一遍
- [ ] 渲染图是**外部 `.svg`** + `![](assets/x.svg)`，不是内联 `<svg>`（内联会被 docs-site 打坏）
- [ ] 图上的数字是脚本算出来的，不是手写进 SVG 的？
- [ ] 渲染结果眼过一遍：有没有溢出、有没有 `[[…]]` 漏进标注
