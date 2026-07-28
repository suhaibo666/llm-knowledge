# `torch.compile` 端到端 A→F 学习系列最终交付报告

> 交付日期：2026-07-28  
> 固定源码 checkout：`E:/97-codes/torch_parallel/p`  
> 固定源码 commit：`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 交付策略：原理与源码优先；不新增 demo；设计文档不提交，课程正文按用户后续授权提交

## 1. 交付结论

本轮已按批准的设计完成 `torch.compile` 端到端 A→F 课程。课程不再以零散问题组织，
而是从 eager/Python 执行模型一路推进到 Dynamo、图编译、artifact/runtime、调试验收以及
训练、分布式、扩展和部署。

最终课程入口是：

- `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/index.md`
- `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/00_torch_compile_end_to_end_index.md`

`19_torch_compile_end_to_end` 目录现统一容纳 63 个 Markdown：

- 1 个领域入口 `index.md`；
- 1 个课程总索引 `00`；
- 39 篇 A/B/D/E/F 正文；
- 1 个卷 C 支持索引和 21 篇 C 正文；
- 卷 C 的 Labs 与证据资产也已迁入本目录，旧目录
  `16_graph_compiler_foundations` 已删除。

统一课程包含 61 个编号入口/正文和卷 C 的 `C00` 支持索引；若再计入端到端
领域入口 `index.md`，共 63 个 Markdown、18,330 行，现均位于同一目录。篇幅没有被人为限制；信息完整性和
机制解释优先于单篇长度。

## 2. 编号与内容清单

| 卷 | 数量 | 范围 | 角色 |
|---|---:|---|---|
| 00 | 1 | 总索引 | 六卷地图、前置依赖、三条阅读路径 |
| A | 5 | A01–A05 | Tensor/storage、operator/dispatcher/autograd、Python frame、dispatch mode、成本模型 |
| B | 10 | B01–B10 | `torch.compile` API、eval-frame、bytecode、VariableTracker、OutputGraph、guards、break、dynamic shape、backend |
| C | 21 | C01–C21 | Graph Compiler Foundations：FX/AOT/Inductor 图编译核心 |
| D | 7 | D01–D07 | `compile_fx`、AOT runtime wrapper、async compile、cache、wrapper memory、CUDAGraph Trees、artifact |
| E | 9 | E01–E09 | 日志、explain、recompile、分阶段定位、minifier、正确性、性能与生产 rollout |
| F | 8 | F01–F08 | Compiled Autograd、checkpoint/recompute、DDP、FSDP/DTensor、custom op/backend、AOTI、freezing/CUDA Graph |

机器可读的 62 项统一清单见 `course_manifest.json`。其中编号课程
61 = `00 + A5 + B10 + C21 + D7 + E9 + F8`，另包含卷 C 的 `C00` 支持索引。
卷 C 同时保留自己的独立 manifest 和 claim ledger，以便单独审计和被其他知识域引用。

## 3. 对批准设计的符合性

| 设计要求 | 落地结果 |
|---|---|
| 先解释为什么设计 | 每篇从问题背景、需要区分的对象或错误直觉进入机制，不以 API 罗列起笔 |
| 概念后必须连接源码 | 39 篇正文均包含固定 checkout 的 repository-relative `file:line` 定位 |
| 解释对象、owner、读写和 consumer | 贯穿 Dynamo frame state、FX use-def、AOT fw/bw ABI、Inductor artifact/runtime 等章节 |
| 解释调用链和状态机 | 用文字、表格及 Mermaid 表达 capture、compile、cache、load、execute、recompile/fallback |
| 解释 why-not 与失败边界 | 明确相邻设计为什么不能替代，并区分 graph break、guard miss、compile/load/runtime failure |
| 包含不变量和复杂度 | 每卷均覆盖身份、alias/mutation/effect、shape/address/lifetime 等不变量及参数化成本 |
| 编号可顺序学习 | 总索引逐篇列出 A01→F08；卷 C 显示编号 C01–C21 与原 01–21 一一映射 |
| 页面进入父索引并回链 | 总索引包含全部 60 篇正文；39 篇新正文直接回 `00`，C01–C21 经 C00 回到端到端课程 |
| 无损并入既有卷 C | C00–C21、Labs 与证据资产实体迁入目录 19；正文原顺序与独立 ledger 保留 |
| 原理解读阶段不加 demo | 没有新增课程演示脚本或伪造运行结果 |
| 提交边界 | 设计文档不纳入提交；课程正文、审计结果和必要导航按用户后续明确授权提交 |

## 4. 本轮实际修改

### 4.1 新增

- `19_torch_compile_end_to_end/` 领域入口、总索引和 A/B/D/E/F 39 篇正文；
- `course_manifest.json`：包含 61 个统一编号页面和卷 C 的 `C00` 支持索引；
- `source_map.md`：A–F 的 load-bearing 源码入口；
- `build_course_decisions.py`：可重复生成 claim decisions；
- `course_claim_decisions/*.jsonl`：按 00/A/B/D/E/F 拆分的事实单元决定；
- `course_claim_ledger.jsonl`、`course_claim_summary.md`、`course_claim_errors.json`；
- 本最终交付报告。

### 4.2 更新

- `01_ai_frameworks/index.md`：加入编号 19 的端到端学习域；
- `19_torch_compile_end_to_end/00_pytorch_graph_series_index.md`：声明其为卷 C 并建立回链；
- Dynamo、AOTAutograd、Inductor、runtime memory、FX/export/extensibility、
  distributed primitives、compile cache 七个领域索引：增加端到端课程反向入口；
- `wiki/changelog.md`：记录本次课程、证据数量、限制和关键纠正。

卷 C 后续按用户要求完成实体归并：`00_pytorch_graph_series_index.md`、`01–21`、
`labs/` 及其 artifacts 已迁入 `19_torch_compile_end_to_end/`；全库链接、manifest、
claim decisions 和运行时证据路径同步更新，旧目录 16 已删除。

### 4.3 卷 C 深度复核

卷 C 不是范围占位符。C01–C21 共 8,928 行、603 个正文源码定位：

- 21/21 篇包含参数化复杂度；
- 21/21 篇以最终非空 `Related Pages` 收尾；
- 16/21 篇设置独立“源码跟读”主链；
- C17–C21 虽未使用统一标题，但把 GraphLowering、IR、memory、Scheduler 和 codegen
  源码直接编织进逐机制章节。

重点页的深度包括：

| 页面 | 机制深度 |
|---|---|
| C02 | Graph/Node/GraphModule、侵入式链表、args/users 双向同步、ownership、lint/recompile |
| C09 | metadata analysis、joint inputs、required closure、fresh Graph 提取、fw/bw ABI 与 lazy backward |
| C10 | saved value 分类、min-cut、memory budget、recompute 节点复制、bw reorder 与 runtime wrapper |
| C13 | PatternExpr 子类动机、DAG sharing、MatchContext、candidate index、逆序 apply、entry 类型与复杂度 |
| C14 | FX/Scheduler DCE 区分、dead 定义、stable topo、effect order、cleanup 顺序 |
| C20–C21 | dependency 构造、fusion candidate/legality/profitability、kernel/wrapper 映射、autotune/cache/provenance |

卷 C 独立账本现有 3,134/3,134 个 claim decisions、0 error；上一轮添加课程桥接造成的
113 个旧 ID 与 116 个缺失决定已通过文本哈希 reconciliation 修复，并为 3 个新增导航
单元补上非断言决定。

## 5. 新增的系统知识

### 5.1 从 Python 到 guarded graph

课程把 `torch.compile()` 的 wrapper 创建、第一次 frame 命中、eval-frame callback、
bytecode symbolic execution、VariableTracker/Source、OutputGraph 提交、guard 安装和后续
cache lookup 分成独立时间点。由此可解释：

- 为什么调用 `torch.compile()` 本身不等于已经编译；
- graph break 怎样生成 resume function 并恢复 Python 状态；
- guard failure 为什么选择另一 cache entry 或触发新 specialization；
- dynamic shape 的 generalization、recompile 和 fallback 为什么是不同机制。

### 5.2 从 FX/AOT 到 Inductor runtime

课程在既有卷 C 之后补上了容易缺失的“后半程”：

- `compile_fx` 如何编排 inference、forward、backward 与 inner compile；
- backward 为什么可以 lazy compile；
- async worker、native compiler、module load 与 callable lifetime；
- Dynamo cache、AOTAutograd cache、FX graph cache、code cache、autotune cache 的 key/value
  与失效边界；
- generated wrapper 如何处理 allocation、reuse、stream、call ABI 与输出；
- CUDAGraph Trees 为什么需要 warmup、record、replay、path 和 liveness；
- artifact 为什么会经历 created、serialized、loaded、executed、invalidated 状态。

### 5.3 调试和验收

卷 E 不再把所有问题归为“compile 失败”，而是建立阶段化证据树：

1. capture/graph break；
2. guards/recompile；
3. AOTAutograd/partition；
4. Inductor lowering；
5. scheduler/codegen/native compile；
6. artifact load；
7. wrapper/runtime；
8. numerical/gradient/alias/mutation correctness；
9. cold compile、warm cache、steady-state performance。

同时补齐 minifier、repro、compiler bisector、日志/counter/artifact 证据能力边界，以及
shadow/canary/fallback/rollback 的 production 策略。

### 5.4 训练、分布式和扩展

卷 F 新增以下机制链：

- Compiled Autograd 捕获 eager autograd engine 执行，与 AOTAutograd 从 forward
  提前生成 backward 的设计边界；
- user activation checkpoint 与 AOT partitioner recompute 的叠加和双重重算风险；
- DDP optimizer、bucket/reducer、compiled-autograd 和 compile region 的关系；
- FSDP 参数 materialization、DTensor placement、collective effects 与 rank-dependent state；
- custom op 的 schema、fake/meta、autograd、functionalization、decomposition、lowering/
  fallback 完整契约；
- Dynamo backend、Inductor device backend、dispatcher device kernel 三种扩展层；
- AOTInductor 的 `ExportedProgram → compile_fx_aot → files → PT2 → loader → C ABI runner`
  部署链；
- training/inference、grad mode、freezing、CUDA Graph 四条正交轴及组合条件。

## 6. 纠正和收紧的关键口径

1. **fw/bw 不是一张带跨图边的 FX Graph**：它们是独立 Graph；saved tensors 通过
   forward outputs、runtime context 和 backward placeholders 建立 ABI。
2. **recompute 不是 backward 指回 forward Node**：partitioner 在 backward fresh graph
   中重新创建计算节点。
3. **PatternMatcher 不是每个 pattern 对整图盲扫**：注册 root candidate 索引先缩小候选，
   再做局部递归匹配；复杂度必须写成候选桶、pattern 大小和 pass/fixpoint 的参数。
4. **DCE 的 dead 不是“在逆图里”**：核心判断来自 use-def/users 与副作用；反向遍历是
   扫描顺序，不会创建另一张反图。
5. **topological order 不是任意重排**：数据依赖、effect/mutation 和稳定性共同限制顺序；
   rewrite 后是否需要 lint、DCE、稳定拓扑或 recompile 取决于 pass contract。
6. **JIT cache 不是部署 ABI**：AOTInductor 需要 export constraints、PT2 archive、call spec、
   target binary、constant ownership 和 C ABI loader。
7. **freezing 不是 `eval()`**：它会把安全参数内联并 constant-fold；可选参数擦除会使原
   eager module 明确不可执行。
8. **CUDA Graph static input 首先是地址契约**：普通输入可复制进静态 buffer；dynamic
   shape 通常按 distinct size 记录新图；partitioned forward 的 saved activation 不能全部
   当作 backward static input。
9. **两处来源地图旧路径已纠正**：minifier 使用
   `torch/_functorch/fx_minifier.py`；AOTInductor 公开打包入口位于
   `torch/_inductor/__init__.py`，package 实现在 `torch/_inductor/package/package.py`。
10. **11 个卷 C 草案链接已改为正式文件名**：链接内容未丢失，最终导航范围无断链。

## 7. 证据闭合

课程 claim parser 对段落、列表项、表格数据行和非空代码块生成原子事实单元。最终统计：

| 项目 | 数量 |
|---|---:|
| claim units | 6,312 |
| `[S]` 固定源码事实单元 | 1,287 |
| `[R]` 正式 runtime evidence | 366 |
| `[I]` 绑定已验证父结论的机制推论 | 3,552 |
| `[M]` codegen/mock 边界证据 | 19 |
| 非断言导航、问题或操作说明 | 1,045 |
| `[B]` 当前环境限制 | 43 |
| claim-ledger validation errors | 0 |

统一账本对 A/B/D/E/F 新正文采用同节/同页父结论绑定；对卷 C 则直接复用其人工审计后的
`[S]/[R]/[I]/[M]/[B]` decisions，不用自动推论覆盖原证据。ledger 继续校验父结论存在、
已验证、支持理由完整且无推论环。

完整结果：

- `course_claim_ledger.jsonl`
- `course_claim_summary.md`
- `course_claim_errors.json`
- `course_claim_decisions/`

## 8. 最终验证结果

| 门禁 | 结果 |
|---|---|
| manifest 数量 | 62（61 个编号入口/正文 + C00 支持索引） |
| 统一课程 Markdown | 63（再计入端到端领域入口） |
| 父索引收录 | 60/60 篇正文 |
| `Related Pages` 最终且非空 | 62/62 |
| H1 与文件显示编号 | 60/60 |
| 导航检查范围 | 71 页 |
| dangling wikilink | 0 |
| Mermaid/Markdown fences | 0 个不平衡或空 Mermaid；结构检查通过 |
| 课程正文 + source map 源码定位 | 1,303 个，文件路径和起止行边界有效 |
| 正式 `[S]` 证据范围 | 由 claim ledger 验证，单段均不超过 30 行 |
| claim decisions | 6,312/6,312 |
| claim-ledger errors | 0 |
| 审计工具测试 | 90/90，`OK` |
| 固定源码 checkout | exact SHA，clean |
| Git index | empty |
| tracked `raw/` diff | empty |
| `git diff --check` | exit 0 |

## 9. 明确保留的能力边界

当前环境观察为 PyTorch `2.9.1+cpu`，没有可用的 MSVC `cl`、CUDA 或 Triton。因此：

- C++ native kernel compile/execute 未在本轮新增验证；
- CUDA/Triton kernel、autotune、CUDAGraph capture/replay 未实测；
- 分布式多 rank、目标 accelerator backend 和 AOTI 目标机 ABI 未进行本轮 native 验收；
- 性能数字不能从源码、generated source 或 CPU-only 观察外推。

这不是原理课程的未完成项，而是后续“演示与目标硬件实验卷”的执行前提。用户已明确当前
阶段先不做 demo，所以本轮没有用占位脚本制造表面覆盖。

## 10. 后续可选工作

源代码级 A→F 主线已经完整。若进入下一阶段，建议按以下顺序扩展：

1. 为每卷设计最小可执行 observability demo，而不是重复正文；
2. 在具备 MSVC/CUDA/Triton 的目标环境补 native receipts；
3. 增加 Transformer 训练和推理两条端到端实验；
4. 增加 DDP/FSDP/DTensor 多 rank 实验；
5. 增加 AOTInductor 构建机→运行机 compatibility matrix；
6. PyTorch 基线升级时重跑 source locator、claim hash 和 semantic diff。

这些后续工作均不影响当前原理与固定源码课程的交付完整性。
