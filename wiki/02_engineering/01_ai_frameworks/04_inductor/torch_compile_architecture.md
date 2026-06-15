# Inductor / torch.compile 概览(overview)

> 层次:overview · 纯 upstream(不展开 NPU 细节) · 最后更新 2026-06-15
>
> 这是 04_inductor 模块「由浅入深」的第一层。读完本页你应能回答:Inductor 解决什么问题、在 `torch.compile` 里处于什么位置、内部分哪几个阶段、有哪些核心概念。需要细节时,顺着每节末尾的 deepdive 链接往下走。

## 1. Inductor 是什么,解决什么问题

**TorchInductor 是 `torch.compile` 的默认后端编译器**:把一张已经捕获、已经处理过自动微分的 FX 计算图,**降级(lowering)成基于循环的中间表示,做激进的算子融合(fusion),最后生成 Triton(GPU)或 C++/OpenMP(CPU)kernel 代码**并编译执行。

它针对的核心矛盾是**内存带宽 vs. 算力**。现代加速器的浮点算力远超访存带宽,深度学习里大量逐元素算子(`add`/`mul`/`sin`/激活/归一化…)本身计算极轻,瓶颈在于反复把张量从显存读进来、算一下、再写回去。Eager 模式下每个算子一个 kernel,中间结果都要落地到全局内存。Inductor 的根本手段就是**把多个算子融合进同一个 kernel**,让中间结果留在寄存器/共享内存里,从而消除冗余的内存往返;再叠加自动调优(autotuning)、内存复用、布局优化等,逼近手写 kernel 的性能,同时完全不需要用户改模型代码。

直观对比 `relu(x + y)`:

```
Eager:   add → 写回 t;  relu → 读 t、写回 out      (2 个 kernel,3 次全局访存往返)
Inductor: 融合为 1 个 kernel: load x,y → t=x+y → max(t,0) → store out  (中间 t 不落地)
```

补一句术语:**Triton** 是 Inductor 在 GPU 上生成 kernel 所用的语言/编译器——一种用 Python 写、能编出高效 GPU kernel 的 DSL。Inductor 自己不直接发 PTX/CUDA,而是**生成 Triton 源码**,再交给 Triton 编译。这就是为什么后文反复出现「生成 Triton kernel」。

源码位于 `torch/_inductor/`(核验基准:`E:\97-codes\pytorch\pytorch`)。

## 2. 在 torch.compile 中的位置

`torch.compile` 是一条三段式流水线,Inductor 是其中的**后端**:

```
Eager Python 代码
        │
        ▼
┌──────────────────────────────┐
│ Dynamo (torch._dynamo)       │  PEP 523 拦截 Python 帧,符号执行字节码
│  → FX Graph(仅前向)+ Guards │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ AOT Autograd (_functorch)    │  追踪联合图、函数化、min-cut 分区
│  → Forward Graph + Backward  │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ ★ Inductor (torch._inductor) │  ← 本模块主题
│  Decomp → FX Passes →        │
│  Lowering → Scheduler →CodeGen│
└──────────────────────────────┘
        │
        ▼
  Triton / C++ Kernel + Wrapper → 编译执行
```

- **Dynamo**:前端,把动态的 Python 拆成「静态 FX 子图 + 守卫条件」,遇到不可捕获处 graph break 回退 Python。
- **AOT Autograd**:中段,提前(ahead-of-time)把前向图追踪出反向图,切成 Forward/Backward 两张图分别交给后端。
- **Inductor**:后端,把这些 ATen 级 FX 图变成真正能跑的高性能 kernel。

> 前两段的细节属于上游模块,见 [[02_dynamo/index]] 与 [[03_aot_autograd/index]];Inductor 的入口是 `torch._inductor.compile_fx.compile_fx`,被 AOT Autograd 作为 `fw_compiler`/`bw_compiler` 调用。

**边界辨析**(避免常见混淆):

- Inductor **不负责**捕获图、不处理 Python 动态性——那是 Dynamo 的事;它拿到的已经是规整的 ATen 级 FX 图。
- Inductor **不直接**求导——反向图是 AOT Autograd 提前生成好的,Inductor 只是把前向图、反向图各自当作普通图来编译。
- 编译产物会被**缓存**(进程内 + 落盘):`FxGraphCache` 缓存编译结果、`PyCodeCache` 缓存生成的 kernel 源码,因此首次编译慢、后续命中缓存可跳过大部分工作。这也是「torch.compile 第一次调用卡顿、之后变快」的原因。

## 3. Inductor 阶段一览

Inductor 内部是一条子流水线。从 ATen FX 图到落盘 kernel,大致经过五个阶段;整条管线由 `compile_fx`(入口)与 `GraphLowering`(`torch._inductor.graph`,统管 FX→IR 转换与最终 codegen)串起来:

| 阶段 | 一句话职责 | deepdive |
|------|-----------|----------|
| **Decomposition** | 把复杂/复合 ATen 算子拆解为一小撮原语算子,收敛 IR 规模,让后续 lowering 只需处理有限算子集 | 见 [[inductor_compiler_pipeline_analysis]] 分解小节 |
| **FX Graph Passes** | 在 FX 图层面做图级优化,分三段:pre-grad(高层重写)、joint-graph(常量折叠/模式匹配)、post-grad(底层融合/设备相关重写) | [[pre_grad_passes_guide]] / [[joint_graph_passes_guide]] / [[post_grad_passes_guide]] |
| **Lowering** | 把 ATen 算子逐一翻译为 **Inductor IR**(`lowerings[target]` 注册表),用 `Pointwise`/`Reduction` 等循环原语表达计算 | [[lowering_analysis]] |
| **Scheduler** | 对 IR 节点做依赖分析,决定融合(水平/垂直)、计算顺序、内存规划与缓冲区复用 | [[scheduler_analysis]] |
| **CodeGen** | 把调度后的 IR 翻译成具体后端代码:Triton kernel(GPU)或 C++/OpenMP(CPU),并生成驱动 kernel 的 Python/C++ wrapper;期间做 autotuning 选最优实现 | [[inductor_codegen_analysis]] |

横切关注点:**动态形状(dynamic shapes)** 贯穿上述每个阶段(符号化 size、guard、`ShapeEnv`、XBLOCK 选择等),单列一条全链路 deepdive,见 [[dynamic_shapes_full_analysis]]。

**为什么分这么多阶段?** 关键在于「在哪一层做哪种优化最自然」:图级重写(算子替换、常量折叠)在 FX 图上做最直接;而融合必须先把算子拆成统一的循环表示才能跨算子合并,所以需要一层独立于 ATen、独立于具体硬件的 **Inductor IR**——它向上承接任意前端算子,向下对接 Triton/C++ 等多种 codegen 后端。分层让「优化逻辑」与「目标硬件」解耦:同一套 lowering/scheduler 逻辑,换个 codegen 后端就能支持新设备(这也是 NPU 等后端的接入方式)。

## 4. 核心概念速览

理解 Inductor 主要是理解它的 **IR** 和 **调度** 两套抽象。最关键的一次认知跳变是:Inductor 用的不是 FX 那种「算子节点图」,而是一套**循环级 IR**——它不记「调用了 add」,而记「输出每个位置的值 = 一个关于输入索引的纯函数」。正因为算子被还原成了可组合的索引计算,跨算子融合才得以成立。

### Inductor IR(`torch._inductor.ir`)

不同于 FX 图的「算子节点」,Inductor IR 是**面向循环(loop-level)**的表示,核心是用 lambda/索引函数描述「某个输出位置如何由输入计算得来」,这正是融合的基础。关键类:

- **`Loops`(`ir.py`)**:循环型计算的基类,持有循环区间 `ranges` 与「给定索引返回值」的内部函数。
- **`Pointwise`**:逐元素计算(`add`/`sin`/激活…)。可被自由融合进任意循环,是融合的主力。
- **`Reduction`**:归约计算(`sum`/`mean`/`max`…)。带规约维度,通常是融合的边界。
- **`Buffer` / `ComputedBuffer`**:需要物化(落地到内存)的张量及其计算体。
- **`TensorBox` / `StorageBox`**:包裹张量与底层存储,管理布局(layout)与视图,使 view/reshape 等不必真正拷贝。

### SchedulerNode 与融合(`torch._inductor.scheduler`)

Lowering 产出的 IR 进入调度器后,被包成调度节点参与融合决策:

- **`BaseSchedulerNode` / `SchedulerNode`**:调度的基本单元,封装一个 IR 缓冲区及其读写依赖。
- **`FusedSchedulerNode`**:多个节点融合后的复合节点——融合成功意味着它们最终落进**同一个 kernel**。
- **`Scheduler`**:总调度器,基于依赖关系判断哪些节点可融合(逐元素+逐元素、逐元素接归约等)、安排执行顺序、规划内存与复用,把「能省的内存往返」尽量省掉。

### CodeGen 后端(Triton / C++ wrapper)

- **Triton 后端**:GPU 路径,把融合后的循环 IR 拼成 Triton kernel 的 Python 源码,经 autotuning 选 block size 等参数。
- **C++/OpenMP 后端**:CPU 路径,生成 C++ kernel。
- **Wrapper CodeGen**:无论哪个 kernel 后端,都要生成一段「调度 kernel、管理输入输出与内存」的驱动代码——默认是 Python wrapper,AOTInductor 场景下是 C++ wrapper(产出 `.so`)。后端通过 `register_backend_for_device` 按设备注册(`codegen/common.py`),这也是新硬件接入 Inductor 的扩展点。

由此引出 Inductor 的**两种使用形态**,二者共用上面整条 lowering/scheduler/codegen 管线,只在最外层的产物形态上不同:

- **JIT(`torch.compile`)**:运行时按帧惰性编译、生成 Python wrapper,适合训练与交互式开发。
- **AOTInductor(`torch.export` + AOT 编译)**:提前编译为自包含 `.so`,生成 C++ wrapper,不依赖 Python 运行时,面向部署/Serving。

## 5. 一个最小例子的旅程

把上面几节串起来,看 `out = torch.relu(x @ w + b)` 在 Inductor 里走过的路:

1. **入口**:AOT Autograd 把前向(及反向)FX 图交给 `compile_fx`。图里是 ATen 算子:`aten.mm`、`aten.add`、`aten.relu`。
2. **Decomposition / FX Passes**:复合算子被拆为原语;图级 pass 做常量折叠、模式匹配重写(例如把可融合的偏置加法标准化)。
3. **Lowering**:逐算子查 `lowerings[target]`——`mm` 走专门的矩阵乘模板,`add`/`relu` 各产出一个 `Pointwise` IR(用索引函数表达 `lambda i: max(acc[i] + b[i], 0)`)。
4. **Scheduler**:依赖分析发现 `add` 与 `relu` 都是逐元素且形状一致,可**垂直融合**;`mm` 作为计算密集算子通常单独成 kernel(或走 epilogue 融合),其输出缓冲区按需物化。
5. **CodeGen**:融合后的逐元素部分生成一个 Triton kernel(GPU)或 C++ 循环(CPU),autotuning 选 block 参数;再生成 wrapper,负责分配缓冲区、按序调用 `mm` kernel 与融合 kernel、返回 `out`。

最终落盘的是一段可缓存、可直接执行的 kernel + wrapper 代码。想逐行看,用 `TORCH_LOGS=output_code` / `TORCH_COMPILE_DEBUG=1`(见 [[inductor_quickstart]] 与 [[Pytorch_Compile_Debug_Analysis]])。

## 6. 由浅入深导航

本模块按「层次」组织,建议路径:

1. **overview(本页)** — 建立全局心智模型。
2. **上手** → [[inductor_quickstart]]:最小前向+反向示例、`torch.compile` 参数与 `torch._inductor.config` 速查、怎么看生成代码。
3. **deepdive** —
   - 端到端全景:[[inductor_compiler_pipeline_analysis]]、[[PyTorch_Inductor_Technical_Analysis]]
   - 各阶段:[[lowering_analysis]] · [[scheduler_analysis]] · [[inductor_codegen_analysis]] · FX passes([[pre_grad_passes_guide]] / [[joint_graph_passes_guide]] / [[post_grad_passes_guide]])
   - 横切专题:[[dynamic_shapes_full_analysis]]、[[unbacked_symint_analysis]]、[[flex_attention_analysis]]、调试 [[Pytorch_Compile_Debug_Analysis]]
4. **NPU 后端**(Ascend 适配,非 upstream)→ 见 [[04_inductor/npu/index]]。

读完本页,带走三句话即可:

- Inductor 是 `torch.compile` 的后端,核心使命是**用融合消除内存往返**。
- 它把 ATen FX 图**降级成循环级 IR**,在 IR 上做调度/融合,再 **codegen** 成 Triton/C++。
- 整套 lowering→scheduler→codegen 逻辑与硬件解耦,**换 codegen 后端即可支持新设备**。

---

## Related Pages

- [[01_ai_frameworks/index]] — 本域总索引
- [[02_engineering/01_ai_frameworks/index]]
- [[inductor_quickstart]] — 上手:最小示例与参数速查
- [[inductor_compiler_pipeline_analysis]] — deepdive:端到端编译管线全景
- [[04_inductor/npu/index]] — NPU Inductor 后端(硬件子目录)
