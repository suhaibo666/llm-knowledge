# `torch.compile` A–F 图学习系列审查修订最终报告

> 修订日期：2026-07-29（含第二轮复检闭环）
> 唯一源码事实基线：`E:/97-codes/torch_parallel/p`
> 固定 commit：`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> 范围：原理解读、源码定位、结构渲染、证据账本与既有 Demo 证据标识
> 本轮未执行：提交、推送、覆盖用户的 `E:/97-codes/torch_parallel/pytorch` checkout

## 1. 最终结论

两轮审查意见均已逐项落实。课程现有 63 篇、20,504 行。第一轮完成 B04、B07、D01、
D02、D06、F01 与 E04、E06、E07 的机制级源码跟读；第二轮补齐 A01–A05，使卷 A
从“概念解释 + 阅读路线”升级为从入口、状态所有者、关键分支到产物的完整源码调用链。
C 卷的超宽定位、架构图、CommonMark 与 Mermaid 排版问题也已同步修复。

当前总账包含 7,313 个 claim decision，7,313/7,313 均有决策，验证错误为 0：

| 证据类型 | 数量 |
|---|---:|
| `[S]` 固定源码证据 | 1,396 |
| `[R]` 运行时证据 | 367 |
| `[I]` 有父结论的机制推理 | 4,347 |
| `[M]` mock/codegen 边界证据 | 19 |
| `[B]` 环境能力阻断 | 43 |
| 非断言/不适用 | 1,141 |

`[R]` 从复核前的 366 变为 367，不是新增实验：C19 中原本因 `-普通reuse`
渲染错误而合并的一个 claim，在修成两个合法列表项后拆成两个原子 claim；两个子 claim
分别继承同一份原始 runtime receipt。迁移器要求“完整规范化文本相等、同页、连续且唯一”
才允许这种一对多迁移，避免把运行证据按相似文本猜配。

## 2. 按审查问题逐项修复

### 2.1 深度断层

第二轮为 A01–A05 全部增加机制级源码跟读：

| 页面 | 新增调用链 |
|---|---|
| A01 | view kernel → `as_view` → `DifferentiableViewMeta` → shared version → inplace/rebase → functionalization |
| A02 | `OpOverload` → Dispatcher → Autograd wrapper → redispatch → Node/Edge → Engine |
| A03 | `torch.compile` wrapper → eval-frame → ConvertFrame → InstructionTranslator → GuardedCode |
| A04 | `make_fx` → mode stack → Proxy/Fake dispatch → FX Node/abstract output → output-tree binding |
| A05 | wrapper → code-object `ExtraState` lookup → Dynamo/backend → GuardedCode → cache entry/replay |

A 卷由 277 行/篇、11.6 个定位/篇提升为 358 行/篇、26.0 个定位/篇；五篇均有独立
源码跟读和架构/数据流图，不再是系列中源码密度最低的一卷。

六篇优先页新增机制级源码跟读：

| 页面 | 新增主线 |
|---|---|
| B04 | `InstructionTranslator` 初始化、指令分派、栈与 block state、graph break/continuation |
| B07 | guard 生成、cache lookup、miss、recompile 与 cache limit 的控制链 |
| D01 | `compile_fx` 从入口、配置与 AOT 分流到 inner compile/artifact 的编排 |
| D02 | AOT runtime wrapper、输入输出 ABI、compiled function 装配与 lazy backward |
| D06 | CUDAGraph Trees 的 warmup、record、path/cache、checkpoint 与 replay |
| F01 | Compiled Autograd 捕获 eager autograd engine、累积状态、图闭包与编译出口 |

第二轮复检声称上述六篇“未增加源码跟读”，但当前工作区逐篇机械核查的结果是：六篇均存在
独立 `## ...源码跟读...` 章节，单节 89–101 行、10–17 个源码定位。它们的调用链、状态
所有者、失败边界与设计原因均已覆盖，因此本轮没有重复添加同义内容。

为避免 E 卷继续停留在工程清单层，还补强：

- E04：Dynamo/backend、AOT partition、Inductor lowering/codegen/runtime 的故障边界；
- E06：compiled/eager correctness helper、输入克隆、梯度与 alias/mutation 语义；
- E07：cold/warm/steady-state 计时、cache 控制、同步与设备测量边界。

当前“源码跟读”章节分布为 A 5、B 2、C 16、D 3、E 3、F 1。未机械地给每篇添加同名
标题；其余页面继续通过原有逐机制源码段落承载内容。

### 2.2 Markdown 与公式渲染

- 以代码围栏感知的方式修复 30 篇中的 673 处 `-条目`，统一为合法 CommonMark
  `- 条目`；
- 修复 B07 成本公式中丢失的加号，使
  `T_total = T_lookup + T_capture + T_backend + T_codegen + T_run` 完整；
- 修复 F03 四处 `3.条目` 到 `6.条目`，使其成为合法 CommonMark 有序列表；
- 清理 C09、C17、B01 共 7 条 Mermaid 文字连线中的引号或非标准虚线写法；不是只修
  复检点名的两条，而是把同类违规在整个课程范围内归零；
- 新增回归测试，禁止课程正文在代码围栏外重新出现同类列表标记错误。

最终扫描：非法列表标记 0，Markdown link、wikilink、围栏、Mermaid 与
`Related Pages` 结构错误均为 0。

### 2.3 固定源码 checkout

课程入口、领域入口和 `source_map.md` 现在都把
`E:/97-codes/torch_parallel/p` 声明为本地跳转的唯一基线。该 checkout 当前：

- HEAD：`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`；
- working tree：clean；
- 1,688/1,688 个课程源码定位均可解析，覆盖 130 个源码文件。

用户原有的 `E:/97-codes/torch_parallel/pytorch` 保持在
`ea5655fcebf726ec4cf1a859de75d2d0e6425805`，且仍有 27 个工作区条目；本轮没有
checkout、清理或覆盖它。这样既恢复文档跳转一致性，也不破坏用户未提交内容。

### 2.4 超宽定位与正式 `[S]` 证据

- 原有 56 条超过 100 行的定位已全部收窄到具体函数、方法或条件分支；
- 课程当前 1,688 条正文定位中，超过 100 行为 0，最大跨度 99 行；
- 仍有 248 条 31–99 行的区域导航定位，它们只承担阅读入口，不直接作为单段正式证据；
- 决策生成器会把宽导航范围切为不超过 30 行的正式 `[S]` 证据片段；
- 当前账本共有 5,013 个 `[S]` evidence span，最大 30 行，超过 30 行为 0。

因此“正文区域导航”和“punctual evidence”已经分层：前者允许覆盖完整小函数或相邻分支，
后者严格满足设计规定的 30 行上限。

### 2.5 C 卷图示

C 卷保留原 ASCII 图，同时增加 Mermaid：

- C09：joint graph 到 forward/backward ABI；
- C13：candidate 索引、递归匹配与 apply；
- C14：FX/Scheduler DCE、effect/order；
- C17：FX → GraphLowering → Inductor IR/Scheduler；
- C20：Scheduler 依赖、融合与 codegen pipeline。

C01–C21 正文现有 7 张 Mermaid；计入 C00 支持索引共 8 张，解决了“只有 1 张 Mermaid”
与库内架构/数据流表达约定不一致的问题。

### 2.6 被取代的 FAIL 证据

原始失败 artifact 保留，不删除历史。以下目录新增 `SUPERSEDED.md`：

- `acceptance/e/`：`minifier_repro` 由 `e-minifier-fixed/` 替代，卷 E 最终为
  7 PASS / 2 BLOCKED / 0 FAIL；
- `acceptance/f/`：由完整重跑 `f-fixed/` 替代，卷 F 最终为
  4 PASS / 4 BLOCKED / 0 FAIL。

根 `.gitignore` 只放行这两个说明文件，其他本地生成 artifact 仍保持忽略。
`demo_delivery_report_2026-07-29.md` 也记录了同一替代关系。

## 3. 证据账本重建的关键修复

文档排版、行号和定位变化会改变 claim ID。若直接按新 ID 重新生成，旧 C 卷人工审计过的
`[R]/[M]/[B]` 可能被静默降级。为此补充了以下机制：

1. ID 与文本均未变化时直接复用；
2. 同页同文本唯一时重新绑定新 ID；
3. 同页存在重复文本时，只有新旧出现次数相等才按源码顺序一一对应；
4. 对本轮唯一允许的 `-item` → `- item` 机械修复做规范化匹配；
5. 若列表修复导致一个旧 claim 拆成连续多个新 claim，只在完整文本、连续性与唯一性同时
   成立时一对多迁移；
6. 父 claim ID 和 evidence 中的父引用同步重写；
7. 迁移后的 `[I]` 不能被误当成直接源码父锚点，防止形成 inference cycle；
8. 无法证明等价的旧决策不猜配，由当前规则重新审计。

对应回归测试覆盖了 changed/unchanged claim、stale parent、重复文本顺序、一对一排版修复、
一对多 claim 拆分、runtime receipt、父链迁移和 `[S]` 30 行切片。

## 4. 本轮补充的新知识

除了修排版和定位，本轮实际增加的原理内容包括：

- differentiable view 如何把 base lineage、共享 version counter、inplace legality、
  `rebase_history` 与 functionalization 串成一条语义链；
- Autograd wrapper 为什么是 Dispatcher kernel，以及 redispatch、Autograd Edge 与
  FX data edge 为什么不能混为一谈；
- `torch.compile(fn)` 创建 wrapper 与首次 frame 捕获为什么是两个时刻，GuardedCode
  如何把 transformed code 与成立前提绑定；
- `make_fx` 为什么需要 Proxy 与 Fake 两套状态：前者记录图关系，后者执行抽象语义，
  `track_tensor_tree` 再把两类产物逐叶绑定；
- code-object `ExtraState`、backend bucket、guard manager 与 LRU entry 如何共同定义
  Dynamo cache hit，为什么一次 hit 不能代表整条编译栈都被跳过；
- Python bytecode symbolic execution 为什么需要显式维护 value stack、block stack、
  instruction pointer、symbolic locals 与 side-effect state；
- guard 并非 cache 的附属字符串，而是 specialization 可复用性的运行时判定；
- `compile_fx` 如何同时承担模式分流、AOT 协作、inner compile 与 artifact 装配；
- lazy backward 为什么把 backward 编译时机推迟到第一次真实反向执行；
- CUDAGraph Trees 为什么不仅按输入 shape 缓存，还必须建模地址、path、liveness 与
  checkpoint；
- Compiled Autograd 与 AOTAutograd 的捕获时机、图边界和状态所有权差异；
- 调试时如何沿 Dynamo → AOTAutograd → Inductor → native compile → wrapper/runtime
  定位责任边界；
- correctness 验收为何必须覆盖 gradient、alias、mutation、dtype/device 和输入克隆；
- 性能测量为何必须分离 cold compile、warm cache 与 steady state，并显式同步设备。

## 5. 最终验证记录

| 验证项 | 结果 |
|---|---:|
| 课程 Markdown | 63 篇 / 20,504 行 |
| 源码定位 | 1,688/1,688 有效，130 个源码文件 |
| 正文定位超过 100 行 | 0，最大 99 行 |
| 正式 `[S]` evidence span | 5,013 个，最大 30 行 |
| 非法 CommonMark 无序/有序列表标记 | 0 |
| Mermaid | 46 张；带引号文字连线违规 0 |
| 结构、链接、围栏、Mermaid、Related Pages | 0 错误 |
| claim decisions | 7,313/7,313 |
| claim-ledger validation errors | 0 |
| 审计工具测试 | 99/99，`OK` |
| Labs/Demo contract 测试 | 69/69，`OK` |
| 固定源码 checkout | exact SHA，clean |
| `git diff --check` | exit 0 |

11 个 CUDA/Linux/native/multi-rank/AOTI 相关用例继续保持 `BLOCKED`，没有改写成
`PASS`。本轮没有新增或伪造目标硬件实验结论。

## 6. 工作区边界

本轮只修改审查范围内的课程、审计生成器、账本、报告、回归测试以及两个
`SUPERSEDED.md`。以下已有用户工作保持原样，不属于本轮交付：

- `wiki/01_theory/04_posttraining/index.md` 的既有修改；
- `.superpowers/`、`docs/reports/`、`raw/_ingest/` 与其他既有未跟踪材料；
- `E:/97-codes/torch_parallel/pytorch` 的脏工作区。

截至本报告生成时，本轮变更尚未提交或推送。
