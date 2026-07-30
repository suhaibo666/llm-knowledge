# 15 · Graph Pass 流水线、顺序与 Fixed Point

> 前置：[[15_graph_normalization_decomposition_and_functionalization_analysis]]、[[23_dead_code_topology_and_effect_order_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. Pass位置由可用不变量决定

| stage | 常见可见形态 | 适合 |
|---|---|---|
| Dynamo | Python捕获、高层ops、guards | graph break/specialization附近优化 |
| pre-grad | 高层FX、未显式backward | module/高层canonicalization |
| AOT decomposition | tracing operator set | composite→primitive |
| joint | forward+gradient同图 | save/recompute前联合优化 |
| post-grad | 独立functional ATen fw/bw | 低层pattern/device rewrite |
| lowering | target→Inductor IR | implementation/layout/fallback |
| Scheduler | realized buffer deps | fusion/reorder/liveness |
| codegen | backend kernel/wrapper | tiling/indexing/launch |

“能在某stage写出来”不等于“应放在那里”。

## 2. 三个Inductor FX driver不同

### pre-grad

可运行built-in/custom passes、configured matcher counter、stable sort、quant lifting、lint、
recompile（`torch/_inductor/fx_passes/pre_grad.py:336-433`）。

### joint

有ordered joint pattern lists、constant folding等，并根据match/change条件做stage cleanup
（`torch/_inductor/fx_passes/joint_graph.py:700-772`）。

### post-grad

有多轮pattern、device/communication/mutation-tail相关顺序与inference-aware hooks
（`torch/_inductor/fx_passes/post_grad.py:165-180`;
`torch/_inductor/fx_passes/post_grad.py:227-303`;
`torch/_inductor/fx_passes/post_grad.py:385-474`）。最后一段才覆盖 late
reinplace/alias 修正与 final lint/recompile；只读 main-pattern 范围不足以证明 mutation tail。

不能用一个“所有pass都pattern match→DCE→sort”图替代。

## 3. 同一语义规则不能原样跨 stage 搬运

以同一个代数变换为例：

```text
add(matmul(x, weight), bias) → addmm(bias, x, weight)
```

规则的数学含义没有变，但可匹配的 target spelling 由输入图契约决定：

| Lab 图契约 | 捕获方式 | 改写前 target | 改写后 target |
|---|---|---|---|
| front-end-like 高层 FX | `symbolic_trace` | `torch.matmul`、`torch.add` | `torch.addmm` |
| functional-ATen-like FX | `make_fx` | `aten.mm.default`、`aten.add.Tensor` | `aten.addmm.default` |

`symbolic_trace`通过 `Tracer.trace`记录执行并组装 `GraphModule`
（`torch/fx/_symbolic_trace.py:1361-1421`）；`make_fx`则返回一个用真实、Fake 或 symbolic
Tensor 执行后产生 FX 图的包装器
（`torch/fx/experimental/proxy_tensor.py:3312-3383`）。两种捕获入口本身已经足以产生不同
target vocabulary。

这正是 stage placement 必须显式声明输入不变量的原因。固定源码还直接警告 pre-grad IR
尚未 functionalize/normalize，规则必须自行处理 alias、mutation 与不同参数 schema
（`torch/_inductor/fx_passes/pre_grad.py:336-351`）；post-grad driver 接收的 IR 才声明为
normalized and functionalized
（`torch/_inductor/fx_passes/post_grad.py:165-171`）。

### 本页 Lab 的证据边界

`part3_passes.py`对上述两个 FX 图应用**同一语义、两份 stage-specific target contract**：

1. 用错误 stage 的 contract 时必须命中 `0` 次；
2. 用正确 contract 时各命中 `1` 次；
3. 两张改写图都与 eager 结果数值一致；
4. 第二次应用各命中 `0` 次，证明该规则在这两个受限输入上幂等。

这是可执行的 **stage-contract 模拟实验**，不是实际 `torch.compile` hook：

- 没有调用 `torch.compile`；
- 没有把规则注册进 `pre_grad_passes`或 `post_grad_passes`；
- 没有经过 AOTAutograd、functionalization 或 Inductor 的真实 pass ordering。

实验 matcher 刻意限制为无 kwargs 的二元 `matmul`和二维数值样例，用于隔离 target
vocabulary 问题。运行结果覆盖“pattern spelling 与输入图契约绑定”和“错误 contract
不命中”。

该模拟实验没有覆盖 broadcast、dynamic shape、alias 或 mutation，不能单独证明规则已经
安全接入真实 pre-grad/post-grad 流水线。真实 hook 集成还必须另外验证注册顺序、dynamic
shape、alias/mutation 与阶段 cleanup。

上述边界只描述 `part3_passes.py`。本页后面的
`part3_real_stage_hooks.py`已经通过 `torch._inductor.config` 的真实 custom-pass 入口补上
placement 证据。

该真实 hook Lab 仍未覆盖 shape、alias、mutation 等更广 legality 维度；这些边界留给第
16 篇的受限合同与后续目标设备验证。

## 4. Registration order

同root bucket中rules按registration order尝试。若先应用宽pattern，可能erase root，使后续
更专用rule无机会。优先级需要：

- 更具体rule先；
- canonicalization先于依赖canonical form的fusion；
- destructive rewrite后重新建立必要metadata；
- mutation/effect tail pass最后。

## 5. Single round、bounded repeat、fixed point

### single round

PatternMatcherPass一次snapshot candidates并逆序遍历；new roots不重访
（`torch/_inductor/pattern_matcher.py:2627-2645`）。

### bounded repeat

driver显式运行N次，例如pre-grad counter。

### fixed point

重复直到所有pass `modified=False`，但通常还有max steps防止不收敛。
FX `PassManager`默认 `steps=1`；只在configured steps内根据modified提前停止
（`torch/fx/passes/infra/pass_manager.py:154-195`;
`torch/fx/passes/infra/pass_manager.py:254-317`）。

`PassResult.modified=True`本身不触发无限迭代。

## 6. 幂等性

理想canonicalization：

```text
P(P(graph)) == P(graph)
```

fusion/replacement也应避免匹配自己的输出，除非driver有明确fixed-point语义。

非幂等风险：

- A→B与另一个pass B→A；
- 每轮插入新cast/view；
- name/meta变化让同语义Node重复生成；
- replacement root仍匹配search pattern。

## 7. Candidate lifecycle

一次Pattern apply：

1. 按registered root keys查询Graph side table；
2. snapshot与逆序sort；
3. 遍历candidate bucket；
4. match、safety、extra_check；
5. apply；
6. 不加入新candidates。

这使单轮复杂度和行为可控。需要新结果继续优化时，driver必须再调用一轮。

## 8. Pass冲突

常见冲突：

- decomposition暴露pattern，但也摧毁专用op；
- CSE合并sharing，改变 `_users`约束；
- DCE先删diagnostic/effect被误判Node；
- layout pass插copy使fusion pattern断开；
- partition前rewrite改变save cost；
- post-grad reinplace恢复mutation，后续rules需effect-aware。

解决方式是明确输入/输出invariant与ordering dependency，而不是只靠import顺序。

## 9. Recursion

stage driver可能递归处理HOP/subgraphs；单个PatternMatcherPass不自动代表整个nested tree。
每个child需独立Graph ownership、cleanup、lint/recompile。

## 10. Observer、counter与debug

可靠pass应记录：

- attempted/matched/applied；
- rejection原因；
- before/after Node数；
- stage与graph id；
- pattern name；
- generated artifact；
- compile time。

counter不是generic `changed` bit；例如joint path的count可能仅统计某类match，不能据此推断
其他custom pass是否修改图。

### `GraphTransformObserver`：三个 driver 共用的包裹机制

上面三个 driver 的源码跟读都提到"经 observer 包裹执行"，机制在
`GraphTransformObserver`（`torch/fx/passes/graph_transform_observer.py:22`）：

- **计时**：`apply_gm_pass`/`apply_graph_pass`（同文件 79/93）内部用
  `dynamo_timed(f"pass.{subsystem}.{passname}"` — 若无 `subsystem` 则退化为
  `f"pass.{passname}"`（85-88、98-101）——所以同一个 pass 名在不同 driver 下的计时
  标签不同（例如 joint 阶段是 `pass.joint_graph_passes.<name>`），不能跨 driver 直接
  比较原始 `passname` 聚合出的统计。
- **可禁用**：调用前先过 `_check_disable_pass()`（105-116）：按名——`passname.upper()`
  是否出现在 `config.disabled_passes.upper()`（108）；按子系统——若构造时传了
  `subsystem`，还会问 `CompilerBisector.disable_subsystem("inductor", subsystem, ...)`
  （114-116）。两条判据任一命中就整个跳过该次 `pass_fn` 调用，返回 `None`——这是
  "怀疑某 pass 引入 bug/劣化"时一键排查的抓手：先按名关，定位不到再按子系统整段关。
- **构造时机决定它管不管这个 pass**：`GraphTransformObserver` 不是全局单例，是每个
  driver 在调用点手工 `functools.partial(...)` 或直接 `GraphTransformObserver(gm, name,
  subsystem=...)` 构造出来的——**没有被这样包裹调用的 pass，天然不受计时、按名/按
  子系统禁用约束**，上面 post-grad 小节里 `b2b_gemm`/`micro_pipeline_tp` 就是具体反例。

## 11. 选择stage的决策树

1. 是否需要Python/module语义？→ Dynamo/pre-grad。
2. 是否需要forward与backward同时可见？→ joint。
3. 是否要求functional ATen且独立fw/bw？→ post-grad。
4. 是否产生lazy loop/layout/extern implementation？→ lowering。
5. 是否依赖buffer reads/writes/fusion group？→ Scheduler。
6. 是否只与tile/index/mask/launch有关？→ codegen。

若多个stage都可实现，选择invariant最强、规则最局部、验证最容易的一层。

### 选对 stage 之后，还有六个必须回答的"为什么"

"放在哪个 stage"只是第一问；一个可合入、可维护的 pass 设计至少还要写清另外六件事——
下面每条给出它在本系列的落点，不是本页新开一套独立标准：

1. **收益为什么存在**：减少 launch、HBM 往返、同步、重排，还是暴露更强 kernel？这是
   下注依据，profiling 先行，不是拍脑袋。
2. **为什么是这个阶段**：即上面的决策树——依赖信息何时首次出现，为什么相邻阶段不合适。
3. **等价为什么成立**：dtype、shape、stride/layout、alias/mutation、随机数、异常和数值
   误差的前提分别是什么，详见 [[25_graph_rewrite_legality_validation_and_complexity_analysis]]
   §1–§7。
4. **动态形状为什么安全**：符号恒等、运行时 guard，还是必须拒绝？"看到 SymInt 就跳过"
   只是临时保守策略，见同页 §7 Fake/meta checks 与 runtime guards。
5. **收益为什么能兑现**：替换后的 op 是否有 lowering/kernel；scheduler/codegen 是否真的
   把它融合或发射成目标实现？对应 [[25_graph_rewrite_legality_validation_and_complexity_analysis]]
   §10 性能值得性与 §8 差异测试矩阵的"收益"行。
6. **为什么可运维**：是否可开关、可计数、可 dump、可 bisect；缓存 key 是否包含影响生成
   结果的 pass 配置/源码——对应本页上方"GraphTransformObserver"小节的计时/禁用机制。

## 源码跟读：三个 driver 怎样把 stage contract 写进 pass 顺序

### 1. Pre-grad 源码首先警告“尚未 functional/normalized”

`pre_grad_passes` 的 docstring 明确要求规则自行处理 alias、mutation 和所有参数 schema，并
建议优先考虑 functionalization/normalization 后的 joint/post-grad
（`torch/_inductor/fx_passes/pre_grad.py:336-351`）。

整段 pass 处理还有一层前置门控：`pre_grad_passes()` 先查 `config.pattern_matcher`（默认
`True`，`torch/_inductor/config.py:290`），关闭时整个 pre-grad pattern 体系（含下面的
`lazy_init`）都不会跑（`pre_grad.py:353`）。门控通过后先调用 `lazy_init()`
（`pre_grad.py:354`，其定义见下方），再按 `config.is_predispatch` 分岔成两条互斥路径
（`pre_grad.py:360-361`）：predispatch 路径转去 `_run_pre_dispatch_passes()`
（`pre_grad.py:200-205`），执行一份显式的 `default_pass_list`（`pre_grad.py:220-224` 起，
"order matters" 注释标明这是有序列表，而非任意顺序的规则集合）；非 predispatch（OSS）
路径才是下面这条更常读到的顺序。

实际顺序中，非 pre-dispatch 路径先做 numpy compatibility、`fuse_fx`，再强制
normalization pattern 优先；之后是 group batch fusion 与 configured pattern passes。
每个配置项的 `counter` 决定同一 matcher pass 运行次数
（`torch/_inductor/fx_passes/pre_grad.py:353-383`;
`torch/_inductor/fx_passes/pre_grad.py:384-402`）。

最后 custom passes 才运行，随后统一 stable topological sort、quant lift、lint、recompile
（`torch/_inductor/fx_passes/pre_grad.py:421-433`）。

`lazy_init()` 本身只做一次性注册：装饰器 `@init_once_fakemode` 保证幂等，函数体 `import`
`apply_gumbel_max_trick`、`efficient_conv_bn_eval`、`split_cat` 三个子模块以触发它们的
pattern 注册副作用，`fbcode` 环境下还多 `import fb`（`pre_grad.py:174-182`）。这是
"lazy"的准确含义：不是延迟到真正需要时才注册，而是延迟到第一次调用、此后不再重复。

这给出两个实现事实：

- counter 是显式 bounded repeat，不是 matcher 自己 fixed point；
- pre-grad cleanup 是 driver 尾声，不是每个 PatternEntry 自动执行。

> [!note] `binary_folding` 不属于 `pre_grad_passes()`
> `binary_folding.py` 的 pattern 经 `register_binary_folding_pattern` 注册进
> **freezing**（`torch/_inductor/fx_passes/freezing_patterns.py:98-101,115`），由
> `config.enable_linear_binary_folding` 门控（`config.py:1670`），`pre_grad_passes()` 本身
> 不引用它。freezing 是推理场景下的独立预处理阶段，不是 pre-grad 流水线的一部分——
> 名字相邻不代表属于同一 driver，必须以谁调用谁为准。

### 2. Joint driver 把“必须先 canonicalize”和“有 change 才 cleanup”写进控制流

joint driver 一进入就调用 `canonicalize_aten_ir_passes`，注释标记“must occur before other
passes”；随后依次是 custom pre、noop removal、可选 constant folding、early patterns
（`torch/_inductor/fx_passes/joint_graph.py:700-733`）。

再往后是 AutoChunker、ordered `pass_patterns`、RNG replacement 与 custom post
（`torch/_inductor/fx_passes/joint_graph.py:735-766`）。只有 `count` 非零，才 stable sort、
lint、recompile（`torch/_inductor/fx_passes/joint_graph.py:768-772`）。

这里的 `count` 是 driver 自己累积的变化信号。新增 custom pass 若无条件 `count += 1`，
即使它内部没改图也会触发 cleanup；反过来，若某修改路径未正确计数，可能跳过预期尾声。
所以不能把任意 counter 当成精确“实际修改 Node 数”。

`joint_graph_passes()` 的 `lazy_init()`（`@init_once_fakemode`，joint_graph.py:81-89）与
pre-grad 那份是两个不同函数，各自注册各自阶段的 pattern：它 `import` 并调用
`_pad_mm_init`/`_sfdp_init`/`_misc_patterns_init` 三个子模块的初始化函数，这才是 SDPA/
pad_mm 等 pattern 真正登记进 `pass_patterns` 桶的位置。AutoChunker 必须在 pad_mm 之前跑，
源码注释直接写明原因——"Make sure AutoChunker happens before pad_mm so we don't need to
handle padding when searching for chunking patterns"（joint_graph.py:733-734，即§4.2 的
`config.auto_chunker.enable` 检查处）：chunk 搜索本身不想再处理 padding 过的形状。

`joint_graph_passes()` 入口还把 `GraphTransformObserver` 偏特化成
`subsystem="joint_graph_passes"` 的局部别名（`functools.partial(...)`，joint_graph.py:706-709），
custom-pre/remove_noop/constant-fold/`pass_patterns`/custom-post 全部经它包裹执行——
这是 §7/§10 讨论的 observer 机制在 joint 阶段的具体落点，见下方"GraphTransformObserver"
小节。

> [!note] `decompose_mem_bound_mm.py` 不是 joint pass
> `joint_graph.py` 只 `import` 了它的 `check_device` 辅助函数用于 pad_mm 相关判断
> （joint_graph.py:40,989），并未把 `decompose_mem_bound_mm` 本身注册进 joint 的任何
> `pass_patterns` 桶——那是 post-grad 侧的 pass（见下方 post-grad 小节）。

### 3. Post-grad 的输入契约最强，但尾部又主动重新引入 mutation

`post_grad_passes` 明确声明会分别在 fw、bw 调用，输入 IR 已 normalized/functionalized。
开头可做 DCE/locality，随后执行 custom pre
（`torch/_inductor/fx_passes/post_grad.py:165-199`）。

主 pattern 区先 group fusion、noop/assert removal、ordered pass lists，再执行配置化
post-grad patterns；custom post 在主 patterns 之后
（`torch/_inductor/fx_passes/post_grad.py:227-267`;
`torch/_inductor/fx_passes/post_grad.py:281-289`）。

之后 driver 仍有 random ordering、stable sort、constructor movement、FakeTensor metadata
增量更新等步骤
（`torch/_inductor/fx_passes/post_grad.py:292-303`）。更晚的 distributed/overlap 变换完成
后，源码明确要求 reinplace 等 mutation-introducing passes 保持最后，之后做 alias 修正与
HOP decomposition，最终 recompile + lint
（`torch/_inductor/fx_passes/post_grad.py:449-474`）。

所以“post-grad 是 functional graph”是该 driver **入口/大部分 pass 的 invariant**，不是
最终 graph 永远无 mutation。late reinplace 依赖前面的分析结果，在结束前受控恢复更高效
的 in-place 形式——具体是 `reinplace_inplaceable_ops`，源码用它自己的函数名当依据，明确
要求排在收尾附近（当前基线调用点 post_grad.py:451-452）。

post-grad 的三轮 `pass_patterns`（`PatternMatcherPass()` 列表，post_grad.py:85-88）不是
临时分组：`register_lowering_pattern` 默认落 `pass_number=1`，桶 `[0]` 在 OSS 默认路径
下经常是空的，靠 freezing/早期量化一类前置阶段去填充；三桶严格按 `[0]→[1]→[2]` 顺序
执行（§3.1「桶内有序」的源码依据）。

`post_grad_passes()` 入口同样把 `GraphTransformObserver` 偏特化为局部别名
（post_grad.py:172-173），DCE/locality/custom-pre/noop/pattern 桶/`b2b_gemm`/通信
bucketing/custom-post/stable-sort/`reinplace_inplaceable_ops` 等几乎每一步都经它包裹
执行并计时——但**不是全部**：

> [!note] 两个 pass 没有走 `GraphTransformObserver`
> `config.b2b_gemm_pass` 门控的 `B2B_GEMM_PASS.apply(gm.graph)` 与 `config._micro_pipeline_tp`
> 门控的 `micro_pipeline_tp_pass(gm.graph)` 都是直接函数调用，不经过
> `GraphTransformObserver(...).apply_graph_pass(...)`（post_grad.py:266-270，对照同文件其余
> 几十处一律用 observer 包裹的写法）。这意味着这两个 pass **不会**出现在 observer 驱动的
> 计时、按名/按子系统禁用（见下方"GraphTransformObserver"小节）或 dump 链路里；排查它们
> 只能用各自的 `config` 开关，不能假设"所有 post-grad pass 都能被 bisect 关掉"。

> [!note] `fused_int_mm_mul` 已是孤儿函数，`config.decompose_mem_bound_mm` 名不副实
> `check_shape_cuda_and_fused_int_mm_mul_enabled`（post_grad.py:2048，读
> `config.force_fuse_int_mm_with_mul`）在当前基线**没有任何 `register_*` 调用引用它**——
> 全仓库 `grep` 只命中它自己的定义，是死代码。另外，`config.decompose_mem_bound_mm`
> （`config.py:1636`，默认 `False`）看名字像是门控 `decompose_mem_bound_mm.py` 这个 pass，
> 但实际门控是 `lazy_init()` 里的 `torch._C._has_mkldnn` 判断（post_grad.py:833-838，仅决定
> 是否 `import` 该模块）与 `post_grad_fusion_options` 字典里是否包含对应 key
> （post_grad.py:244）——`config.decompose_mem_bound_mm` 这个 bool 本身在这条路径上不起
> 门控作用。两条都是"配置名字暗示的语义"与"源码实际行为"不一致的例子，说明**只信名字、
> 不读调用点会得出错误结论**，这条经验对任何一个 pass 都成立，不只是这两个。

### 4. 为什么同一个规则跨 stage 不能只改注册位置

从三个 driver 可直接推出输入条件：

```text
pre-grad:
  高层/未规范，alias+mutation+schema 由规则承担

joint:
  forward+backward 同图，canonical ATen，partition 尚未发生

post-grad:
  fw 或 bw 独立，functional/normalized 入口，late tail 可 reinplace
```

同一个 `add(matmul(...), bias)`：

- pre-grad 可能是 Python function、module 或不同 schema；
- joint/post-grad 通常是 ATen overload，但 joint 还含 forward/backward sharing 与
  partition metadata；
- post-grad 已切成一侧，规则不能再查看跨 boundary save/recompute。

因此迁移规则至少要重审 target vocabulary、users 边界、metadata、alias/effect invariant、
是否影响 partition cost，以及 cleanup/observer 归属。

### 5. `PassManager` 的 fixed-point 语义是“整组 passes 有界重复”

FX `PassManager.__call__` 先解决 ordering constraints并检查 invariants，然后最多执行
`self.steps` 轮。每轮按顺序调用所有 passes，OR 累积 `res.modified`；GraphModule 结果会
recompile，可选地每个 pass 后检查。整轮没有修改才提前停止
（`torch/fx/passes/infra/pass_manager.py:254-302`;
`torch/fx/passes/infra/pass_manager.py:312-317`）。

所以：

```text
PatternMatcherPass.apply 单轮
  candidate snapshot，一次扫描

PassManager 一轮
  pass1.apply → pass2.apply → ... → passN.apply

PassManager fixed point
  重复“整组 passes”，最多 steps 轮
```

若 pass1 产生 pass2 可消费的形态，同轮即可继续；若 pass2 产生 pass1 的形态，要下一轮。
若 A→B 与 B→A 循环且都报告 modified，只有 `steps` 上限阻止无限振荡。

### 6. 设计一个可证明收敛的 pipeline

源码只提供 bounded execution 机制，不替用户证明收敛。规则设计应给出单调 measure：

| 规则类型 | 可用 measure |
|---|---|
| canonical spelling | 非 canonical Node 数严格下降 |
| decomposition | operator level/rank 单向下降 |
| fusion | 可融合 root 数或 operation 数下降 |
| cleanup | live pure/no-user Node 数下降 |
| bounded search | driver counter/steps 显式上限 |

若没有单调 measure，就必须将其定义为 single round/bounded heuristic，并验证最终形态，不应
宣称 fixed point。

### 7. Observer 的位置也是语义

三个 driver 都用 `GraphTransformObserver` 包裹 pass，但 observer 记录的是传给它的那次
Graph/GM 变化。它不会自动把独立 capture、AOT partition 或后端 lowering 串成连续对象链。
调试 artifact 要同时记录 stage、graph identity/run id、pass name、before/after 与 cleanup，
才能解释“这次命中发生在哪张图”。

### 8. `GroupBatchFusionBase`：与 `PatternExpr` 并存的另一套 pass 结构

group_batch/split_cat 家族（pre-grad 与 post-grad 都有）不是 `PatternExpr` 声明式匹配的
实例，而是另一套独立的类层级机制（`torch/_inductor/fx_passes/group_batch_fusion.py`）：

- `GroupBatchFusionBase`（101）只定义 `match`/`fuse` 两个抽象方法；两个语义子类
  `GroupFusion`（153，"任意形状，用 fbgemm 一类 op 做组融合"）与 `BatchFusion`
  （159，"同形状，用 batched op 做批融合"）。
- 注册用 `@register_fusion(name, pre_grad=...)`（118）装饰子类，按 `pre_grad` 填进
  `PRE_GRAD_FUSIONS` 或 `POST_GRAD_FUSIONS`（114-115）两张独立表，而不是 `PatternExpr`
  用的 `(op, target)` 候选桶。
- 驱动链路：`group_batch_fusion_passes`（1677）→ `generate_fusion_from_config`（1664，
  只挑 `options` 里已注册的名）→ 对每条规则 `apply_group_batch_fusion`（1615）：逆序遍历
  节点，收集候选，`find_independent_subset_greedy`（1488，受 `min_fuse_set_size` 约束）
  贪心找互不依赖的子集，再调用 `rule.fuse`。

这是"目录里 batch_\*/group_\* 那一排"背后的统一机制：与 §2.1 起讨论的 PatternExpr
声明式匹配是两条并行的改图基础设施，都能被同一个 driver（pre_grad_passes/
post_grad_passes）调用，但注册表、候选选取和改写落地方式完全不同——读到
`register_fusion`/`GroupBatchFusionBase` 时不要套用 `PatternEntry` 的心智模型。

### 源码边界

上述顺序绑定当前 SHA 和配置分支；实际启用哪些 pass 由 config、device、inference/training、
distributed 选项决定。文档可以陈述 driver 中的相对顺序与 invariant，不能无条件声称每次
编译都会执行列表中的所有可选步骤。

## 12. 复杂度

不同 stage 的图规模不同，应写为：

```text
Σgraph g Σpass q (
  candidate(q,g) + match(q,g) + rewrite(q,g) + cleanup(q,g)
)
```

- fixed `k`轮对相应 stage 乘 `k`；
- until-stable `r`轮乘实际 `r`，且 `r`必须有上界或单调 measure；
- pattern candidate snapshot降低单轮重入，但重复driver仍会重新查询/sort；
- custom handler、extra check、FakeTensor trace 与 backend compile 是外生成本。

没有 pass 命中率、graph size 与 convergence rounds 分布时，期望复杂度未定义。

证明收敛常用单调measure：

- 某类Node数严格下降；
- expression canonical rank下降；
- rewrite只从高level到低level；
- bounded counter。

## 13. 已验证 Lab

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part3_passes.py
python -B tools\labs_torch_compile\part3_end_to_end_pass.py `
  --output-dir tools\labs_torch_compile\artifacts\part3
python -B tools\labs_torch_compile\part3_real_stage_hooks.py `
  --output-dir tools\labs_torch_compile\artifacts\part3_real_stage_hooks
python -B tools\labs_torch_compile\series_artifact_bundle.py `
  --output-dir tools\labs_torch_compile\artifacts\end_to_end
```

`part3_passes.py`构造：

- 一个single-round pass；
- PassManager `steps=1`与 `steps=4`；
- 同一 `add(matmul)`融合在 front-end-like 与 functional-ATen-like 两种 target contract
  下的正例、错 stage 反例和二次运行幂等性检查；
- 两个互逆passes形成oscillation并由max steps截断。

贯穿 rewrite 作为正例必须第二次 apply 返回 `False`且 code 不变；oscillation pair 作为错误/
边界例每轮做 `add→sub→add`，无法收敛，只能被 `steps=4`截断。stage-contract 实验把
同一语义规则分别写成高层 FX 与 functional ATen target contract；交换 contract 时均为
零命中。

这些结果说明：把 rule 平移到另一个 stage 前，必须重写 pattern/legality contract。

实测：

```text
pass_manager_steps_1_count=1
pass_manager_steps_4_count=3
front_end_like_before_targets=matmul,add
functional_aten_like_before_targets=aten.mm.default,aten.add.Tensor
front_end_like_after_targets=addmm
functional_aten_like_after_targets=aten.addmm.default
stage_target_spelling_differs=True
stage_correct_contract_rewrites=True
stage_wrong_contract_rejected=True
stage_rewrite_idempotent=True
stage_outputs_match=True
stage_contract_kind=simulated_frontend_like_vs_functional_aten
actual_torch_compile_stage_hook_executed=False
oscillation_bounded_at_four=True
oscillation_final_target_is_add=True
second_run_modified=False
second_run_code_unchanged=True
```

`CountToThree`在第三次返回 stable 后提前停止，没有机械跑满4次；oscillation pair 则跑到
显式上限。新增 stage-contract 项由 `part3_passes.py`自身做 assertions；自动合同
`EditingAndPassManagerContractTest`与 `EndToEndPassContractTest`继续覆盖既有编辑、
PassManager 与贯穿 rewrite 合同。Part III artifact 位于 `tools/labs_torch_compile/artifacts/part3/`；环境、
命令与 stage artifact 见 [`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

真实 hook Lab 另行安装 scoped config patch：

- pre-grad 输入是 Python/operator target，`operator.matmul + operator.add`命中一次并改成
  `torch.addmm`；把 functional-ATen contract 放到此阶段时命中 0；
- post-grad 输入是 functional ATen target，`aten.mm.default + aten.add.Tensor`命中一次
  并改成 `aten.addmm.default`；把 Python/operator contract 放到此阶段时命中 0；
- 两个 hook 都由真实 `torch.compile(..., backend="inductor", fullgraph=True)` driver 调用；
- 每个 hook 在同一张已改写 Graph 上立即再运行一次相同 rewrite，第二轮命中 0 且
  `graph.python_code()`不变；这才是 pass 幂等证据。随后 compiled callable 的第二次调用
  未重新编译是独立的 cache 观察，不能替代 pass 幂等；
- forward、输入梯度和权重梯度都与 eager 一致，config 在退出后恢复。

固定源码入口见 `torch/_inductor/fx_passes/pre_grad.py:421-426`、
`torch/_inductor/fx_passes/post_grad.py:190-198`、
`torch/_inductor/fx_passes/post_grad.py:281-289`。

2026-07-27 实测 pre/post correct hits 均为 1、wrong-stage hits 均为 0、second rewrite
hits 均为 0；该路径执行真实 Inductor extern addmm/mm。

该 Lab 没有生成、编译或执行 Inductor 生成的 native C++ kernel，因而不提供这一级证据。

## 14. 跨框架方法论对照（历史保留，基线独立于本页）

> [!note] 与本页其余内容的基线关系
> 本节归纳自四篇独立分析——upstream 本身即上表 Pattern 引擎的固定基线；torch_npu
> `b3c8a815b`（v2.7.1）、vLLM `97a98006b0`、SGLang `d6ef68881e`（均约 2026-07-20 前后核验）
> ——这三个下游基线**不随本页 PyTorch upstream 基线同步更新**，具体机制细节以各自专题页
> （下方链接）为准；本节只保留跨框架都成立的方法论结论。

四家对 "pass 放在哪、怎么声明匹配、命中后怎么落地" 给出了不同答案，但都能投影到同一组
问题上：

| 维度 | upstream Inductor | torch_npu | vLLM | SGLang |
|---|---|---|---|---|
| **主引擎** | 声明式 `PatternMatcherPass`（自建，§2.1） | 官方钩子 + 手工遍历（自建 `ascend_custom_passes` 注册表） | **复用** torch 的 `pattern_matcher` + `VllmInductorPass` 包装 | **fork vLLM 骨架，但抽空**融合层 |
| **落点** | pre/joint/post_grad + lowering | post_grad/pre_grad **custom 钩子**（仅推理为主） | `post_grad_custom_post_pass` 钩子 + pre_grad IR functionalization | `post_grad_custom_post_pass` 钩子（但 pass 为 no-op） |
| **融合朝向** | codegen（Triton/C++ 模板） | **厂商手工库 ACLNN** + Cube 模板/DVM | **厂商/手写 kernel**（FlashInfer/cutlass/symm_mem/AITER/`_C.*`） | 预融合 kernel + inductor 原生 `combo_kernels` |
| **代表页** | 本系列 C13/C15/C16 | [[22_npu_fusion_passes_deepdive]] | [[vllm_ir_and_fusion_passes_analysis]] | [[sglang_compilation_passes_analysis]] |

**一句话读法**：upstream 造引擎、定义在哪三阶段落地；三家下游都从 `post_grad_custom_post_pass`
这个官方钩子接进去（§2.6 已给出该钩子在本页固定基线下的确切 config 位置），然后各自决定
"融合朝向什么"——这个选择直接决定了要不要写融合 pass（vLLM 写十几个、SGLang 一个不写，
不是能力差异，而是"把融合放在编译层还是 kernel 层"的路线选择）。

命中后怎么落地，除本页 C13 已详述的三种 upstream 落地形态（graph-pattern/lowering-pattern/
replacement）外，下游还有两种本页未覆盖、因为它们不是 upstream 机制的形态：**rewrite-
existing-op**（让已有 kernel 多干一步，例如 vLLM 把 quant 塞进 attention kernel 的
epilogue）与 **fallback/换手工算子**（torch_npu 把 attention/通信 fallback 到 ACLNN；vLLM
换 FlashInfer/AITER）。选择哪种朝向不是 upstream 单方面能决定的问题，取决于目标硬件是否
已有极致手工 kernel。

## 学习顺序

- 上一篇：[[23_dead_code_topology_and_effect_order_analysis]]
- 下一篇：[[25_graph_rewrite_legality_validation_and_complexity_analysis]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[15_graph_normalization_decomposition_and_functionalization_analysis]]
- [[22_pattern_expression_and_matcher_engine_analysis]]
- [[23_dead_code_topology_and_effect_order_analysis]]
- [[25_graph_rewrite_legality_validation_and_complexity_analysis]]
- [[22_npu_fusion_passes_deepdive]] · [[vllm_ir_and_fusion_passes_analysis]] · [[sglang_compilation_passes_analysis]] — §14 跨框架对照的三个下游代表页
