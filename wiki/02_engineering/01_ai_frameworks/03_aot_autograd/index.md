# 03 · AOTAutograd — 目录索引

> 介于 Dynamo 与 Inductor 之间:functionalization、前/反向联合图(joint graph)生成与 min-cut partition。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-06-13

---

## 模块概述

AOTAutograd(ahead-of-time autograd)在编译期就把一段可微计算的**前向与反向拆成两张独立的 FX 图**,而不像 eager 那样在运行时由 autograd 引擎边执行边构建反向。这样做的意义在于:反向计算从此成为一张**显式的图**,可以和前向图一起接受图级优化(算子融合、重计算换显存、decomposition 等);否则反向只是运行时一连串 Tensor 调用,编译器无从插手。它解决的核心问题,就是「让梯度计算也能被编译优化」。

在 `torch.compile` 三段管线里它处于中段:**Dynamo** 从 Python 字节码捕获出带 guards 的前向 FX 图;**AOTAutograd** 接过这张前向图,ahead-of-time 地 trace 出前向 + 反向并加以切分;**Inductor**(或其他后端)再对切分结果做 lowering 与 codegen。一句话概括各自产出:Dynamo 产「捕获到的前向图」,AOTAutograd 产「切分好的前向图 + 反向图」,Inductor 产「可执行 kernel」。

为此 AOTAutograd 承担三大职责:**functionalization**——把原地写(in-place / mutation)、view 等改写成纯函数式 ATen 算子,消除别名与副作用,后端才好优化;**joint graph**——ahead-of-time 同时 trace 前向与反向,合成一张「前 + 反向联合图」;**partition**——用 partitioner 把联合图切成独立的 forward / backward 两张图,并决定哪些中间张量 save 给反向、哪些在反向里重计算(min-cut vs default,详见 quick start)。

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[aot_autograd_quickstart]] | **quick start** | 看前/反向图:`backend="aot_eager"` + `TORCH_LOGS=aot_graphs`;看联合图 `aot_joint_graph`;partitioner(min-cut vs default)与重计算;`aot_function` 最小用法;`AOT_PARTITIONER_DEBUG`/activation_memory_budget |
| [[aotautograd_analysis]] | deep dive | aot_function/aot_module、joint graph 构建、partitioner、functionalization、runtime wrappers |

> joint graph 上的优化 pass 见 [[joint_graph_passes_guide]](实现于 Inductor `fx_passes/joint_graph.py`)。

---

## 关联域

- [[10_eager_autograd/index]] — **eager 对应物**:运行时动态磁带 + C++ 引擎(`.backward()`);本编译期模块的 eager 侧根源,两者对照见该页表格
- [[02_dynamo/index]] — 上游:图捕获
- [[04_inductor/index]] — 下游:lowering 与 codegen
- [[01_ai_frameworks/index]] — 本域总索引
