# PyTorch 图编译原理学习系列最终交付报告

> 交付日期：2026-07-28  
> 设计依据：`docs/superpowers/specs/2026-07-23-pytorch-graph-learning-series-design.md`  
> 固定源码基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 主课程：`wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/`  
> 本轮范围：完善原理解读与源码机制；暂停新增演示 demo  
> Git 处置：原始交付时不创建 commit；2026-07-28 按用户后续授权随 A→F 课程提交

## 1. 交付结论

图编译学习材料已经由散点问题重构为一条编号化、可按顺序阅读的课程：

```text
00  总索引、阅读路径与源码跟读方法
01–06  图的概念、数据结构与语义基础
07–11  捕获、规范化、AOT 正反向构图与 runtime ABI
12–16  FX 改图、PatternMatcher、DCE、保序、pass 与合法性
17–21  GraphLowering、Inductor IR、内存、Scheduler、Codegen 与 Autotune
```

本轮重点修复了“概念说明较强、源码结合较弱”的问题。正文不再只给出类名或函数名，
而是按以下闭环组织源码解读：

1. 先说明该对象要解决的图语义问题；
2. 给出 driver 到核心实现的真实调用链；
3. 区分函数读取、写回和新建的状态；
4. 标出不变量、合法性检查与失败边界；
5. 说明后续 consumer 如何使用这些状态；
6. 给出设计原因和参数化复杂度；
7. 明确哪些结论来自固定源码、当前运行、推论、generated-only 或环境阻塞。

最终符合性结论分为三层：

| 范围 | 结论 |
|---|---|
| 00–21 课程结构、原理与固定源码机制 | PASS |
| 历史 28 页的保留、去向、纠错与隔离 | PASS（含 quarantine，不等于旧结论全部正确） |
| 原生 C++ kernel、CUDA/Triton 实测 | BLOCKED（当前环境缺少对应能力） |

## 2. 文档体系改动

### 2.1 编号与导航

- 使用 `00` 总索引和 `01–21` 正文编号。
- 文件名、H1、总索引顺序和前后导航使用同一逻辑顺序。
- 框架总索引、AOTAutograd、Inductor、FX/export、runtime memory 和 compile cache
  入口回链到课程总索引。
- 总索引新增源码阅读方法和各阶段入口表，读者可以从 FX、AOT、PatternMatcher 或
  Inductor 任一阶段进入，再回到完整主线。

### 2.2 单篇写作结构

21 篇正文均保留“概念先行”，并补强以下内容：

- 关键类型不是只列继承关系，而是说明它们分别表达的图语义；
- 关键函数不是孤立 locator，而是放回 caller/callee 和数据流；
- 代码路径明确区分“创建对象”“写 use-def”“执行节点”“生成新图”“运行新图”；
- 对象 identity、metadata、runtime value、signature 和 provenance 分开讨论；
- 复杂度使用图规模、边数、候选数、窗口、重复次数等参数表达；
- 对源码能够证明和不能证明的边界作显式说明。

## 3. 各部分完成内容

### 3.1 Part I：图、FX 与横切语义

`01–06` 建立了整套图对象坐标系：

- 区分 eager autograd graph、FX Graph、AOT fresh graph、Inductor IR 与 Scheduler graph；
- 解释它们为何不能共享一个“万能 GraphNode”数据结构；
- 追踪
  `symbolic_trace → Tracer.trace → create_proxy → Graph.create_node → GraphModule`；
- 追踪
  `replace_all_uses_with → _update_args_kwargs → erase_node → lint → recompile`；
- 区分 FX `Node` identity、`node.meta["val"]`、Interpreter `env[Node]` 和 graph signature；
- 说明 ShapeEnv、SymNode、guard/deferred assert 与图复用的关系；
- 说明 alias、mutation、effect token、input mutation writeback 和 nested GraphModule ownership。

### 3.2 Part II：捕获与 AOT 正反向构图

`07–11` 将 frontend、joint graph、fw/bw 和 runtime ABI 串成一条完整链：

- 比较 symbolic FX、make_fx、Dynamo 和 export 的拦截层与产物；
- 追踪 decomposition、functionalization、duplicate input、synthetic base 与 capture checks；
- 从 `create_joint()` 和 `autograd.grad()` 追到 partitioner 的 fresh graph extraction；
- 明确 forward Graph 与 backward Graph 之间没有跨 Graph 的 `Node` 边；
- 将跨图依赖拆成 fw 额外输出、runtime saved slots、bw placeholders 和调用参数组装；
- 解释 min-cut partitioner 如何选择 save/recompute；
- 明确 recompute 是把所需计算复制进 backward fresh graph，不是 backward Node 回连 forward；
- 说明跨阶段关系依赖 lineage/provenance，而不是 Python 对象 identity。

### 3.3 Part III：安全改图、PatternMatcher、DCE 与保序

`12–16` 将改图解释为带不变量和失败边界的状态转换：

- FX 改图覆盖插入、重连、擦除、复制、graph setter、lint 与 recompile；
- `PatternExpr` 被解释为“待匹配子图的递归约束语言”，不是运行图中的 Node；
- 说明 `Arg`、`KeywordArg`、`Ignored` 的捕获语义以及
  `CallFunction`、`MultiOutputPattern` 的适用图形；
- 追踪 `PatternMatcherPass` 的 root 注册、候选索引、逆序 snapshot、匹配、handler 与清理；
- 纠正“每个 pattern 都从头扫描整图”的过度概括；
- 区分 FX DCE、nested graph DCE、stable topo 和 Scheduler DCE；
- 说明 dead node 的定义依赖用户集合和副作用判定，而不是“反向图中的节点”；
- 追踪 pre-grad、joint、post-grad pass driver、bounded repeat、late reinplace 与 fixpoint；
- 将 metadata、alias/mutation、effect、layout、stage contract、atomic failure 放入统一合法性矩阵。

### 3.4 Part IV：Lowering、IR、内存、Scheduler 与 Codegen

`17–21` 补全从 FX 到可调用产物的后半程：

- 追踪
  `GraphLowering.run → Interpreter.run → run_node → call_function → lowering/fallback`；
- 区分 Interpreter `env`、lazy IR value 和 GraphLowering 已注册 operation/buffer；
- 解释 `register_lowering` wrapper 承担 broadcast、type promotion 和 IR validation 的原因；
- 区分 realization 的决策位置与 `StorageBox.realize()` 执行位置；
- 将 Inductor IR 拆成计算、存储和调度关系；
- 从 `LoopBody → extract_read_writes → MemoryDep` 解释 scheduler dependency 的来源；
- 追踪 alias user closure、mutation rename、last use 与 wrapper reuse key；
- 参数化 Scheduler topo、fusion candidate、compile 和 autotune 复杂度；
- 追踪
  `Scheduler group → backend scheduling → kernel/call + wrapper → PyCodeCache module`；
- 区分 FX graph cache、Python/C++ code cache、algorithm selection cache 和 Triton config cache；
- 区分 algorithm-level autotune 与 generated-kernel autotune，避免把两者合并成一个阶段。

## 4. 修正的关键知识

| 旧的易错理解 | 修正后的机制 |
|---|---|
| fw 与 bw 通过图边直接相连 | 两者是独立 FX Graph；依赖由 saved-value ABI 和 runtime 参数传递表达 |
| saved tensor 自己“连接”两张图 | partitioner 决定 fw outputs/bw placeholders，runtime 按槽位保存并组装参数 |
| recompute 是 bw 回到 fw 执行 | partitioner 将需要重算的 joint 子图复制进 backward fresh graph |
| reverse graph node 与 forward node 有特殊边 | “reverse traversal”是同一图节点顺序；AOT backward Graph 是另一张图 |
| pattern 是一个待执行的 FX graph | pattern 是描述候选子图结构、参数捕获和约束的 `PatternExpr` 语法树 |
| 每个 pattern 都逐节点扫描整图 | matcher 按注册 root 建候选索引，并对候选 snapshot 执行匹配 |
| dead node 只看是否有用户 | 还必须考虑 side effect、output、mutation/effect 等不可删除语义 |
| 改图后总有统一全图重排 | 各 pass 按需要执行 lint、DCE、stable topo、recompile；没有自动万能收尾 |
| realization 都发生在一个固定阶段 | 多个 consumer 决定何时需要实体化，`realize()`负责执行状态转换 |
| aggressive fusion 必然是二次复杂度 | 候选生成受 group 和窗口约束，应写成参数化上界 |
| autotune 只有一个缓存 | algorithm selection 与 generated-kernel config 是不同层次和缓存键 |

## 5. 新增的系统知识

本轮在原问题之外补充了以下连接点：

1. **图不是唯一对象**：同一编译链中存在多种 graph/IR，每一种数据结构都服务于不同
   不变量和 consumer。
2. **阶段边界会重建 identity**：FX Node、AOT fresh Node、IR value 和 scheduler node
   不能按对象相等跨阶段追踪，必须使用 signature、origin、trace 或显式映射。
3. **运行时 ABI 是 fw/bw 的真实桥梁**：saved tensor、symbolic scalar、tangent、
   mutation output 等都需要稳定的顺序和槽位协议。
4. **匹配与重写是两阶段事务**：pattern 先产生 captures/Match，handler 再改图；候选
   snapshot、rollback 和失败原子性用来隔离搜索与 mutation。
5. **DCE 与拓扑排序解决不同问题**：前者判断是否可删除，后者构造满足依赖的稳定顺序；
   两者都必须尊重 effect/mutation。
6. **Lowering 后仍有多套关系**：计算表达式、存储实体和调度依赖不能混为一个节点层次。
7. **内存复用依赖语义闭包**：last use 不是只看单个 buffer，必须结合 alias、mutation、
   layout、device 和 wrapper 状态。
8. **Codegen 产物包含 kernel/call 与 wrapper**：生成源码不等于已编译执行，provenance
   也不等于 runtime PC 映射。
9. **复杂度必须分阶段**：capture、match、rewrite、DCE、topo、fusion、codegen 和
   autotune 的主导参数不同，不能只用一个 `O(N)` 或 `O(N²)` 总括。

## 6. 证据与历史审计

### 6.1 新课程 claim gate

| 项目 | 数量 |
|---|---:|
| claim candidates / decisions | 3134 / 3134 |
| validation errors | 0 |
| 固定源码 `[S]` | 916 |
| 当前环境运行 `[R]` | 366 |
| 有已验证父结论的推论 `[I]` | 1284 |
| generated-only/mock `[M]` | 19 |
| blocked `[B]` | 41 |
| nonassertive/not-applicable | 508 |

正式文件：

- `course_claim_ledger.jsonl`
- `course_claim_summary.md`
- `course_claim_errors.json`
- `course_claim_decisions/`

所有 `[S]` 决定使用不超过 30 行的固定源码范围。`[R]/[M]/[B]` 决定绑定 producer
脚本哈希、命令、退出码、artifact 哈希和 selector。

### 6.2 历史材料

| 项目 | 数量 |
|---|---:|
| 冻结旧页 | 28 |
| historical claims / decisions | 2190 / 2190 |
| corrected | 91 |
| unresolved and retain-quarantined | 2099 |
| structural decisions | 1602 |
| correction dispositions | 94 / 94 |
| destination validation errors | 0 |
| unresolved claims imported by new course | 0 |

这里的“审计闭合”表示每个历史 claim 和结构单元都有保留、纠正或隔离决定；不表示
2099 个 unresolved 结论已经被证实。

## 7. 验证结果

| 验证 | 结果 |
|---|---|
| 固定 PyTorch checkout HEAD | `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` |
| 固定 PyTorch checkout 工作树 | clean |
| 课程 claim gate | 3134 claims、3134 decisions、0 errors |
| 结构 delivery gate | 22 course files、28 legacy files、0 errors |
| 审计工具测试 | 90 tests，OK |
| 课程合同测试 | 42 tests，OK |
| 既有 runtime producers | 21/21 成功 |
| Git index | 无暂存改动 |

本轮没有新增 demo。既有 Lab 只用于回归现有证据和防止原理解读与运行事实脱节。

## 8. 仍然存在的能力边界

当前 Windows/CPU 环境缺少 MSVC `cl`、CUDA 和 Triton，因此以下内容保持
`BLOCKED`，没有通过推论伪装为已实测：

- generated C++ pointwise/reduction kernel 的本机编译、加载和数值对照；
- CUDA/Triton kernel 与真实 autotune candidates/timings/winner/cache；
- GPU allocator snapshot/trace；
- native fusion/reorder 性能对照；
- runtime PC/profiler sample 到 generated source 的映射。

这些边界不影响固定源码机制的讲解，但限制了 native runtime 结论的证据等级。

## 9. 推荐阅读顺序

首次系统学习按 `00 → 01 → … → 21` 阅读。已有 FX 基础但重点关心本次问题时，可按：

```text
02 FX 数据结构
  → 09 AOT joint/fw/bw
  → 10 saved tensor 与 recompute
  → 12 FX 改图
  → 13 PatternExpr 与 matcher
  → 14 DCE 与 topo
  → 15 pass driver
  → 17–21 Inductor 后半程
```

遇到源码版本差异时，先回到 `00` 的源码阅读方法，再核对固定 commit，不应直接套用
运行环境中的行号或从类名猜测完整调用路径。
