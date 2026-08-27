---
title: "通信控制面：从 Process Group 到可掩盖的 collective"
---

# 通信控制面：从 Process Group 到可掩盖的 collective

> **论点式副标题**：TorchTitan 并不存在一个统一的“通信重叠后端”；它把通信问题拆成 process-group 控制、symmetric-memory 资源注册、collective/GEMM 所有权、显式预取以及 host-sync 可捕获性五层，并让编译器、FSDP module 与模型模块分别拥有不同的排程边界。
>
> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页回答**：通信控制面如何从 `CommConfig` 落到 backend、mesh 与 timeout；FSDP symmetric memory、编译期 Async TP 和模块级 dist-GEMM 为什么是三条独立路径；FSDP/EP 在哪里显式改变通信时序；CUDA Graph 为什么把 host sync 当成组合边界。
>
> **Sibling 边界**：本页不重复 TP/SP 的布局证明、EP dispatcher 的完整算法、FSDP 参数生命周期、CUDA Graph capture/replay、DistMuon/FlexShard 优化器通信或 GraphTrainer pass 内部；这些分别由 12、15、20/21、23、26、27 页拥有。

---

## 1. Overview

### 1.1 背景：优化通信不是“把 async 打开”

一个训练 step 中至少有四类不同的等待：参数 all-gather/reduce-scatter、TP 激活重分布、EP token all-to-all，以及 host 为了获知动态 split/capacity 而发生的 device-to-host 同步。它们使用的 mesh、可移动的边界和正确性依赖不同；把它们统称为“异步通信”，会掩盖真正决定能否重叠的所有权问题。TorchTitan 的 mesh 本身就同时建立 dense storage、dense forward/backward、sparse EP 等视图，并明确承认当前 `DeviceMesh` 会为维度重建重复 process group（`torchtitan/distributed/parallel_dims.py:147-185`、`torchtitan/distributed/parallel_dims.py:220-289`）。

### 1.2 Thesis：控制面分层，而不是统一调度器

TorchTitan 当前的核心选择是：**先用 `CommConfig` 和 `ParallelDims` 决定通信域与失效策略，再把某一段通信的排程权交给最了解相邻计算的一层**。因此：

- 编译期 Async TP 把“哪些 TP collective 与哪些 GEMM 匹配”交给 Inductor；TorchTitan 只注册准确的 dense TP group 并打开 pass（`torchtitan/distributed/compile.py:75-96`）。
- dist-GEMM 把 all-gather/GEMM 或 GEMM/reduce-scatter直接封进模型投影模块，不要求 compiler pass（`torchtitan/distributed/linear.py:47-68`、`torchtitan/distributed/linear.py:356-383`）。
- FSDP symmetric memory 只替换每个 `FSDPModule` 的通信实现；通信何时预取、forward 后是否 reshard 仍由独立调度策略决定（`torchtitan/distributed/fsdp.py:102-132`、`torchtitan/distributed/fsdp.py:384-424`）。
- EP backend 首先改变 token 传输的同步与容量语义；“无 CPU 同步”“custom op 同步返回”“能与计算重叠”是三件不同的事（`torchtitan/models/common/token_dispatcher.py:275-308`、`torchtitan/distributed/minimal_async_ep/api.py:298-365`）。

### 1.3 概念表：五层分别回答什么

| 层 | 所有者 | 决策问题 | 当前代表机制 |
|---|---|---|---|
| 进程通信控制 | `CommConfig` + distributed init | backend、初始化窗口、超时后证据 | NCCL/device default、CPU Gloo、Flight Recorder、两阶段 timeout |
| 通信域 | `ParallelDims` / `DeviceMesh` | collective 应落在哪个 rank group | dense TP、FSDP storage、sparse EP、loss mesh |
| 资源与协议 | symmetric memory / FSDP API | rank 间可寻址 workspace 如何注册 | FSDP symm-mem、Async-TP group registration、dist-GEMM symm-mem op |
| 排程与所有权 | compiler、module、FSDP hooks | 谁能看见 collective 两侧的计算 | Inductor micro-pipeline、dist-GEMM、explicit prefetch |
| capture admissibility | Trainer config validator | 路径是否需要 host sync / 动态 Python 决策 | CUDA Graph 的 PP/EP guards |

`spmd_types` 与 `partial_dtensor` 不属于“重叠算法”：它们决定布局表达和 collective 接口。当前只接受这两个 backend，默认是 `spmd_types`（`torchtitan/config/configs.py:174-180`、`torchtitan/config/configs.py:261-266`）。

### 1.4 关键图：一次通信优化如何落地

```text
CommConfig ──> init_process_group ──> ParallelDims / named meshes
                                             │
                   ┌─────────────────────────┼─────────────────────────┐
                   │                         │                         │
          FSDPModule comm API         dense TP process group      sparse EP group
          + reshard/prefetch          │                         dispatch/experts/combine
                   │             ┌─────┴─────┐                       │
             parameter comm      compiler   model module        host-sync policy
                                Async TP    dist-GEMM           + CUDA Graph guard
```

这张图只表达所有权，不声称三条分支共享同一 stream 或 scheduler。当前模型集成顺序也印证这一点：先安装 SPMD/TP sharding，再 AC，再逐 block compile，最后 FSDP；FSDP symm-mem 是 `fully_shard` 之后的 module 设置（`torchtitan/models/llama3/parallelize.py:40-78`、`torchtitan/distributed/fsdp.py:355-374`）。

### 1.5 Quick Start：先选所有者，再选开关

| 目标 | 首选路径 | 必要判据 | 不应误选 |
|---|---|---|---|
| 已经 compile model，希望编译器自动找 TP collective/GEMM 邻接 | Async TP | compile 开、components 含 model、TP>1 才有实际 group | 把它当 eager `async_op` |
| eager Trainer 也要把 TP collective 折入 GEMM | dist-GEMM | `spmd_types`、Sequence Parallel、支持的模型投影 | 同时保留 block-boundary redistribution |
| 改善 FSDP2 module 通信实现 | FSDP symmetric memory | CUDA；NVIDIA capability 至少 9.0 | 期待它自动改变 prefetch/reshard |
| EP + CUDA Graph 避免 CPU sync | MinimalAsyncEP 或 non-blocking HybridEP | 固定容量/重计算等各自约束 | 将“可 capture”写成“必有通信重叠” |

---

## 2. Process-group 控制面：先保证通信域与失败语义一致

### 2.1 背景 / 问题

在任何 overlap 发生前，所有 rank 必须对 backend、group membership、timeout 和故障证据达成一致。初始化阶段还会包含 lazy init 与首次编译，若直接采用训练期的短 timeout，慢 rank 可能尚未完成准备，快 rank 已用新窗口发起 collective，形成伪故障。`CommConfig` 因而把 init timeout 与首步后的 train timeout 分开，默认分别为 300 秒和 100 秒（`torchtitan/config/configs.py:318-345`）。

### 2.2 为什么这样设计

选中的路线是“**一次初始化全局 PG，再由 named mesh 切出专用通信域，并在首个真实 step 后一致缩短 timeout**”。明显替代是每个优化模块自行创建 group、设置 timeout；那会让 TP/FSDP/EP 的 group 生命期与错误策略分裂。判据是：membership 与 fault policy 属于训练进程全局控制面，算子只应消费准确 group。

另一个替代是让 watchdog 立即 abort work；Flight Recorder 为了能在 timeout 时留下 dump，初始化代码强制把 NCCL async error handling 设成 skip-cleanup 模式 3，并配置 ring buffer、dump-on-timeout 和文件前缀（`torchtitan/distributed/utils.py:508-528`）。这里优先的是“超时后仍可诊断”，不是最低退出延迟。

### 2.3 当前实现 / 调用链

`Trainer.init_distributed()` 把 `config.comm`、CPU offload 需要的 CPU backend 和 dump 路径交给 `init_distributed()`，随后才由 world size 构造 `ParallelDims`（`torchtitan/trainer.py:628-637`）。默认 backend 从当前 device type 的默认映射取得；打开 CPU offload 时组合成 device backend 加 `cpu:gloo`（`torchtitan/distributed/utils.py:498-506`）。

真实链路是：

```text
Trainer.init_distributed
  -> dist_utils.init_distributed(CommConfig)
     -> Flight Recorder / NCCL error env
     -> init_process_group(init_timeout)
  -> ParallelDims.from_config(world_size)
  -> ParallelDims.build_mesh()
  -> first successful relative train step
  -> set_pg_timeouts(train_timeout, every 1-D mesh group + world group)
```

首个相对 step 结束后（恢复训练也从本进程的第一步计数），Trainer 调 `set_pg_timeouts()`（`torchtitan/trainer.py:988-997`）。该函数先 barrier，再 device synchronize，然后遍历所有一维 mesh group 和 world group 设置短 timeout；源码明确说明同步是为避免快慢 rank 切换 timeout 的竞态（`torchtitan/distributed/utils.py:539-567`）。

### 2.4 约束 / 失败边界

- 若 `torch.distributed` 已初始化，函数直接返回，并警告传入的 `comm_config` 和其他设置不会生效（`torchtitan/distributed/utils.py:452-458`）。嵌入式调用者不能假设 TorchTitan 会覆盖宿主 PG。
- `fake_backend` 是无 GPU 的配置 dry run：它要求合法的 `NGPU`/`RANK` 并初始化 fake mode，不进入真实通信优化路径（`torchtitan/distributed/utils.py:465-489`）。
- 初始化还关闭 autograd multithreading，因为 AC backward 的线程需要看到 thread-local DeviceMesh/SPMD context；这是让 backward collective 找到正确 PG 的正确性约束，不是性能调参（`torchtitan/distributed/utils.py:460-463`）。
- `CommConfig` 不选择 NCCL algorithm、chunk size 或 collective fusion；把这些能力归给它会越过当前源码边界。

### 2.5 锚点趋势

当前 `ParallelDims.build_mesh()` 的 TODO 明确指出 `DeviceMesh` 会为相同维度重建 process group，期望未来由 DeviceMesh 共享 group；在此之前 TorchTitan 接受冗余创建，而不引入 fake mesh 来绕开（`torchtitan/distributed/parallel_dims.py:180-185`）。这是源码锚定的资源收敛方向，不代表现状已有 group 去重。

---

## 3. Symmetric memory 是共享底座，不是统一功能开关

### 3.1 背景 / 问题

FSDP 参数通信、编译期 Async TP 和 dist-GEMM 都会触及 symmetric memory，但三者需要的通信域和调用协议不同：FSDP 面向每个 `FSDPModule`，Async TP 面向准确的 dense TP process group，dist-GEMM 则在 autograd Function 内直接调用 fused symmetric-memory op。相同底座不意味着它们共享开关或执行流。

### 3.2 为什么这样设计

选中的路线是“**资源按实际消费者注册，排程权仍留在消费者**”。明显替代是一个全局 `enable_symmetric_memory` 同时改写所有 collective；它无法表达 FSDP module API、compiler pass 和模型投影各自不同的布局契约。判据不是“是否用了相同 allocator”，而是“谁拥有 collective 前后的计算与输出布局”。

FSDP 接线的历史动机有明确锚点：提交 `58b034444b78886cea3d57fad3d292e8c3809ed2` 只把 symmetric-memory 通信接入默认 Trainer 的 FSDP2 路径，并在正文中明确排除 GraphTrainer 和 Inductor Async-TP。当前代码仍保持这条 ownership 边界。

### 3.3 当前实现 / 状态

FSDP 路径在所有 block 和 root model `fully_shard()` 之后遍历 `FSDPModule`，逐个启用 force-sum reduction 和 `set_symm_mem_for_comm()`（`torchtitan/distributed/fsdp.py:102-109`、`torchtitan/distributed/fsdp.py:355-374`）。这说明开关改变的是 module 的通信实现，而非把 FSDP 图交给编译器。

Async TP 路径从 backend-aware 的 dense TP mesh 取得唯一 group name，显式 `enable_symm_mem_for_group()` 后再设置 Inductor `_micro_pipeline_tp`（`torchtitan/distributed/compile.py:75-96`）。dist-GEMM 则延迟导入 CUDA-only symmetric-memory 模块，并直接调用 `fused_all_gather_matmul` / `fused_matmul_reduce_scatter`（`torchtitan/distributed/linear.py:35-44`、`torchtitan/distributed/linear.py:118-141`、`torchtitan/distributed/linear.py:356-383`）。

### 3.4 约束 / 成本 / 失败边界

- FSDP symm-mem 默认关闭；配置层拒绝无 CUDA 的环境，对 NVIDIA 要求 compute capability 至少 9.0，HIP 不走 NVIDIA capability 判断（`torchtitan/config/configs.py:162-166`、`torchtitan/config/configs.py:272-281`）。源码没有检查 NVLink 拓扑，因此不能把“必须 NVLink”写成已验证 guard。
- dist-GEMM 当前假定同一 group 上只有一个 symmetric-memory op in flight。每个 op 从共享 workspace offset 0 分配，跨 stream 并发会发生别名；顺序 module forward/autograd backward 才是当前安全执行方式（`torchtitan/distributed/linear.py:19-25`）。
- Async TP 的 group 注册存在上游 TODO：PyTorch 自动为 pass 所用 PG 注册 symmetric memory 后，这个显式调用可删除（`torchtitan/distributed/compile.py:83-93`）。
- 三条路径可以组合，但“都使用 symmetric memory”本身不证明吞吐必增；资源占用、额外 barrier 与拓扑收益属于具体硬件/上游实现。

### 3.5 锚点趋势

编译路径的注册 TODO 是明确的上游收敛点；dist-GEMM 的当前文件则明确说若要 deliberate cross-stream overlap，需要不同 workspace offsets，而不是再加 barrier（`torchtitan/distributed/linear.py:19-25`）。因此可能的演进是减少 TorchTitan 手工注册、增强 workspace 隔离；这两点都不能写成现有能力。

---

## 4. TP overlap 的两种所有权：编译器发现 vs 模块显式封装

### 4.1 背景 / 问题

默认 TP 把 all-gather/reduce-scatter 当作模块边界的 redistribution，把 GEMM 当普通算子。要掩盖通信，必须有一层同时看见 collective 与相邻 GEMM；仅把 collective 发成 async 而不知道在何处 wait，通常只是把等待搬了位置。

### 4.2 为什么有两条路线

**编译期 Async TP** 选择让 compiler 在已编译 block 中匹配和微流水 collective/GEMM。替代方案是 TorchTitan 在每个模型 forward 手写异步 op 与 wait；那会复制模型逻辑、固定 schedule，并难以跨模型维护。判据是：已经使用 model compile，且希望编译器从图中统一发现邻接关系。

**dist-GEMM** 选择让投影模块直接拥有 fused collective+GEMM。替代方案仍是要求用户打开 compiler pass；这会把并行策略绑定到编译路径。提交 `9a711521ac2973fe230a3f38efc6aedfc7d1f9c6` 的正文正是以“eager Trainer 也能使用”为动机，并把 backend 做成 model config 属性。判据是：模型有受支持的 QKV/FFN 投影、希望 eager 可达，并愿意接受更窄的 backend/layout 约束。

### 4.3 当前实现 / 真实调用链

编译路径要求 `compile.enable=True` 且 components 含 model，否则配置构造立即失败（`torchtitan/config/configs.py:295-315`）。`apply_compile()` 在逐 TransformerBlock `fullgraph=True` compile 前，用 `get_dense_tp_mesh()` 配置 Async TP（`torchtitan/distributed/compile.py:39-72`）。若开关关闭或 TP mesh 不存在则直接返回；因此 TP=1 是合法 no-op，不是配置错误（`torchtitan/distributed/compile.py:75-81`）。单测固定了“注册准确 group + 打开 Inductor flag”的 TorchTitan 责任边界（`tests/unit_tests/gpu/test_compile_moe.py:64-91`）。

dist-GEMM 的链路不同：`tp_gemm_backend="dist_gemm"` 在模型 config 构造期选择专用投影；forward 时从当前 `spmd_types` context 解析 TP group，因为 `__init__`/parallelize 阶段尚无 mesh context（`torchtitan/models/common/config_utils.py:62-68`、`torchtitan/models/common/dist_gemm.py:70-85`）。column-parallel forward 把 sequence shard all-gather 与 GEMM 合并；row-parallel forward 把 GEMM 与 reduce-scatter 合并（`torchtitan/distributed/linear.py:47-68`、`torchtitan/distributed/linear.py:356-383`）。

关键不只是换 kernel：dist-GEMM 同时改写 sharding contract。attention/FFN 不再声明边界 all-gather，row-parallel 输出也不再先产生 Partial 再 redistribute；否则 framework 会重复发 collective（`torchtitan/models/common/decoder_sharding.py:226-252`、`torchtitan/models/common/decoder_sharding.py:329-350`）。这正是“模块拥有通信边界”相对“每算子外部手写 collective”的本质差异。

### 4.4 约束 / fallback / 失败分支

- dist-GEMM 只支持 `spmd_types` 且必须启用 Sequence Parallel；前者因为模块消费/返回 plain local tensor，后者因为 fused GEMM 本身替代的就是 SP all-gather/reduce-scatter（`torchtitan/models/common/dist_gemm.py:88-110`）。CPU 单测固定了 DTensor backend 与 SP-off 的失败分支（`tests/unit_tests/cpu/test_dist_gemm.py:119-140`）。
- 选择 dist-GEMM 但 TP=1 时会 warning 一次并回退 stock projection；这是可运行 fallback，不应误报为融合成功（`torchtitan/models/common/dist_gemm.py:53-67`）。
- Async TP 的 collective 匹配、chunking 和 wait 放置属于所依赖 PyTorch Inductor；TorchTitan 当前源码只证明 group 注册与 pass flag，不能据此编造固定内核 schedule。
- dist-GEMM 的 fused QKV 路线要求可被单一 GEMM 消费的 fused QKV；当前测试明确拒绝 unfused Q/K/V 配置（`tests/unit_tests/cpu/test_dist_gemm.py:102-117`）。
- 当前多 GPU 集成定义覆盖 FSDP2+TP2 dist-GEMM，说明该组合有测试入口；它不是对所有模型、精度和拓扑的性能保证（`tests/integration_tests/h100.py:64-71`）。

### 4.5 锚点趋势

提交 `737594746fda65a6d94dc9482ef07863a80c8588` 把 Async TP 从 parallelism 配置迁到 compile，并修复为选择准确 dense TP PG；旧心智模型“在 `tensor_parallel.py` 打开 eager Async TP”已经失效。dist-GEMM 提交正文把 sharding 两分支称为 transitional，预期 redistribution collective 普遍进入模块后可收敛；这是历史作者意图，不代表当前两分支已统一。

---

## 5. FSDP overlap：通信实现、reshard 和 prefetch 是三条旋钮

### 5.1 背景 / 问题

FSDP 参数 all-gather 能否被当前层计算掩盖，主要取决于“下一模块何时预取”和“forward 后参数是否立即 reshard”。symmetric memory 只能改变通信实现，不能自动选择这两个时机。EP 的动态 token count 还会在 host 侧插入同步，破坏 FSDP 隐式 prefetch 原本依赖的 CPU ahead-of-GPU 节奏。

### 5.2 为什么显式覆盖默认调度

普通 dense 模型可让 FSDP 使用隐式 prefetch；EP 路径则选择显式把下一个 module 写入 forward/backward prefetch 链。明显替代是继续依赖调用顺序自动推断，但 EP 的 D2H sync 会让 host 无法及时走到下一 module。判据是是否存在 EP host synchronization。源码注释直接给出这一理由（`torchtitan/distributed/fsdp.py:384-389`）；历史提交 `a54725cfab21c82f5189cfdfff93c2c9347ac025` 也以 profiler 中恢复前后向预取为验证锚点。

PP 下的默认选择则是 forward 后不 reshard，避免每个 microbatch 重复、昂贵且不可掩盖的 all-gather。明显替代是始终 reshard 以省参数驻留内存；判据是 PP microbatch 复用带来的通信放大是否比内存更昂贵（`torchtitan/distributed/fsdp.py:112-132`）。

### 5.3 当前实现 / 状态链

`apply_fsdp_to_decoder()` 先对各 block 与 root model `fully_shard()`，再应用 symmetric-memory module 设置，最后在 EP>1 时建立显式链（`torchtitan/distributed/fsdp.py:355-387`）：

- forward：embedding 预取首 block；每个 block 预取下一 block；最后一 block 预取 norm 与 lm_head（`torchtitan/distributed/fsdp.py:389-406`）。
- backward：lm_head 预取反向首 block；每个反向 block 预取前一 block；最后回到 embedding（`torchtitan/distributed/fsdp.py:408-424`）。

这条链只决定参数通信的发起相对顺序。FSDP/HSDP 的 unshard、backward overlap 和内存峰值由 sibling 20/21 页解释，本页不重复其 module-state 生命周期。

### 5.4 约束 / 成本 / 不做什么

- 显式链只在 `ep_degree > 1` 安装；dense 路径不会无条件覆盖 FSDP 默认 prefetch（`torchtitan/distributed/fsdp.py:384-390`）。
- `fsdp_reshard_after_forward` 的 `always`/`never`/`default` 是内存与通信频率的策略；默认只在 PP 下变为不 reshard（`torchtitan/config/configs.py:150-160`、`torchtitan/distributed/fsdp.py:124-136`）。
- 不 reshard 会延长完整参数驻留时间；显式预取会提前占用下一 module 参数。两者用内存换通信窗口，不能仅以“更多 overlap”评价。
- `enable_fsdp_symm_mem` 不会安装 prefetch 链，也不会覆写 reshard policy；把三者合成一个“FSDP overlap 开关”是错误心智模型。

### 5.5 锚点趋势

当前源码没有 TODO 表示要把 EP 显式 prefetch 删除；因此不能推断隐式调度即将取代它。可确认的演进只有：该逻辑已从模型私有实现合并到公共 `distributed/fsdp.py`，现状由 decoder 共享路径统一接线。

---

## 6. EP 调度：先识别 host-sync，再讨论 overlap

### 6.1 背景 / 问题

EP dispatch 的输出大小由路由动态决定。标准可变长 all-to-all 要先交换 token counts，再把 split materialize 成 Python list；这条 host 依赖既阻断 CUDA Graph capture，也可能打断 FSDP prefetch。专用 backend 的首要差异因此不是名字中是否有 “Async”，而是它如何处理动态容量、何处等待、返回值何时可读。

### 6.2 为什么统一 dispatcher、保留不同同步语义

TorchTitan 选择统一 `dispatch -> local experts -> combine` 计算边界，而不强迫所有 backend 暴露相同 async handle。明显替代是一个通用 async dispatcher API，让上层统一决定 wait；但 DeepEP handle、HybridEP fixed capacity、MinimalAsyncEP symmetric buffer 的生命周期不同，抽象会隐藏 token drop、host sync 与 backward 约束。判据是专家计算只依赖“dispatch 返回可消费的 routed rows”，而通信 backend 保留自己的容量与同步协议（`torchtitan/models/common/moe.py:137-169`）。

### 6.3 当前实现 / 四种同步轮廓

| 路径 | 当前同步边界 | 对控制面的含义 |
|---|---|---|
| standard | count exchange 显式 `wait_tensor()`；remote output splits blocking D2H，再转 Python list（`torchtitan/models/common/token_dispatcher.py:275-314`） | 动态精确容量，但 host-sync；随后才发可变长 token all-to-all |
| DeepEP v2 training | compact layout 需要 host sync 得到精确 per-expert counts（`torchtitan/distributed/deepep/deepep.py:140-194`） | 可反向、节省静态展开空间，但训练路径不可 capture |
| HybridEP blocking | 默认取得精确 size；non-blocking 用预估 fixed capacity 免 CPU sync（`torchtitan/models/common/token_dispatcher.py:888-920`） | 以容量/潜在 drop 换 graphability |
| MinimalAsyncEP | symmetric-memory 直写后在 custom op 内 barrier，返回时 buffer 可读（`torchtitan/distributed/minimal_async_ep/api.py:298-365`） | 无 host sync，但明确不提供 microbatch comm overlap |

DeepEP 底层 combine 使用 async-with-compute-stream 并返回 event，但当前 dispatcher 随即 `sync_combine()` 后才返回（`torchtitan/distributed/deepep/deepep.py:250-327`、`torchtitan/models/common/token_dispatcher.py:865-878`）。上层又在整个 routed experts 返回后才计算 shared experts，所以当前调用链没有“DeepEP combine 与 shared experts overlap”（`torchtitan/models/common/moe.py:440-452`）。

HybridEP 的 non-blocking capacity factor 为 1.0 时按 worst case 分配、避免 drop 但占用最大；小于 1 时节省内存，overflow token 可能静默丢弃，而检查 GPU overflow flag 本身会重新引入 `.item()` 同步（`torchtitan/distributed/deepep/hybridep.py:80-105`、`torchtitan/distributed/deepep/hybridep.py:176-183`）。

### 6.4 约束 / fallback / 组合边界

- standard 路径在 compile/non-strict tracing 或非 `spmd_types` 时用 functional `all_to_all_single`，默认 eager `spmd_types` 用 `spmd.all_to_all`；换接口没有消除 count wait 与 D2H split（`torchtitan/models/common/token_dispatcher.py:245-273`、`torchtitan/models/common/token_dispatcher.py:316-370`）。
- DeepEP 的 no-grad inference 可在 `cudagraphable=True` 时走静态 expand、无 host sync；autograd 开启会安全回落 compact training 路径，而非让 inference-only layout 参与 backward（`torchtitan/distributed/deepep/deepep.py:490-507`）。
- MinimalAsyncEP 要求 EP>1、expert 数整除 EP、存在 training runtime config，并使用 full recompute；它不再要求 `TP=CP=PP=1`（`torchtitan/distributed/minimal_async_ep/api.py:84-140`）。当前 H100 集成定义实际覆盖 FSDP+CP+TP+MinimalAsyncEP（`tests/integration_tests/h100.py:57-62`）。
- MinimalAsyncEP buffer 是进程内单例配置；再次初始化为不同 group/shape/dtype/device 会失败，且 backend 必须是 CUDA symmetric memory（`torchtitan/distributed/minimal_async_ep/api.py:174-225`）。

### 6.5 锚点趋势

MinimalAsyncEP 当前 barrier 注释记录了从逐 peer Python signal/wait kernel 循环迁移为单个 fused barrier 的演进，目标是消除 CUDA Graph/compiled step 暴露出的 launch critical path（`torchtitan/distributed/minimal_async_ep/api.py:345-365`）。这证明它在缩短同步边界，不证明同步边界已消失。

---

## 7. CUDA Graph guard：可捕获性是控制面约束，不是 overlap 证明

### 7.1 背景 / 问题

CUDA Graph replay 要求捕获区间的形状、资源与 host 控制流稳定。通信路径若把 device 值拷回 CPU 再决定 split，capture 时便无法把这段 Python 决策变成可重放图。另一方面，一个 op 可以完全在 GPU 上同步等待，虽没有 host sync、可以 capture，却仍没有与计算重叠。

### 7.2 为什么在配置构造期拒绝

TorchTitan 选择在 Trainer config validator 中按 dispatcher 类型和模式拒绝不兼容组合。明显替代是直到首次 capture 才让 CUDA runtime 报错；那会把语义错误推迟到昂贵的模型构建、warmup 之后。判据是“当前路径是否需要 CPU synchronization”，而不是 backend 名称里是否有 async（`torchtitan/trainer.py:165-195`）。

提交 `f84224af0995debb4b32bb1a0050796ab9135c49` 的正文给出了设计理由：PP 需要不同 capture boundary；任何 captured path 的 CPU sync 都不支持；标准 blocking EP 因此应改用 non-blocking HybridEP 或 MinimalAsyncEP。

### 7.3 当前 guard / 调用链

当 CUDA Graph 未显式关闭时，配置验证先无条件拒绝 `pipeline_parallel_degree > 1`。EP=1 直接通过；EP>1 时遍历模型里的 dispatcher config，只接受 MinimalAsyncEP，或设置了 `non_blocking_capacity_factor` 的 HybridEP，其他 dispatcher 都报错并建议关闭 graphs（`torchtitan/trainer.py:165-195`）。

这条 guard 与 backend 内部事实一致：standard materialize CPU split；HybridEP 只有 fixed-capacity non-blocking 分支免 host sync；MinimalAsyncEP 在 GPU custom op 内等待。CUDA Graph 的捕获、copy-in、replay 和 shape guards 由 23 页拥有，本页只负责解释通信为什么被准入或拒绝。

### 7.4 约束 / 常见误读

- **MinimalAsyncEP 被准入不等于它有 overlap**：其 custom op 明确 wait 后才返回，并声明不提供 microbatch communication overlap（`torchtitan/distributed/minimal_async_ep/api.py:309-315`）。
- **non-blocking HybridEP 被准入不等于无代价**：固定容量为 1.0 可能显著增内存，小于 1 可能丢 token（`torchtitan/models/common/token_dispatcher.py:896-919`）。
- **PP+任意 dispatcher 都不是当前 core Trainer CUDA Graph 组合**：validator 在检查 EP backend 之前就拒绝 PP；关闭 CUDA Graph 后，PP/EP 是否可组合仍需回到 mesh 与各 backend 约束判断（`torchtitan/trainer.py:165-176`）。
- 旧提交曾列出 FSDP `reshard_after_forward=always` 的捕获限制，但当前 validator 已无此 guard；不能把历史限制继续写成 HEAD 现状。

### 7.5 锚点趋势

PP 的错误信息仍写明 “not support ... yet”，历史提交也指出需要把每个 stage 的 forward/backward action 与 Python schedule/P2P handoff分开捕获。这是有锚点的未来边界调整；当前不能推断完成时间或已存在 fallback capture。

---

## 8. 组合判据与旧心智模型纠正

### 8.1 背景 / 问题

单独可用的通信机制，组合后可能争用 workspace、重复 collective、引入 host sync 或跨越不兼容的 capture boundary。组合审计应沿“group 是否准确—布局契约是否唯一—谁放 wait—是否 host sync—内存代价”逐层检查，而不是把所有优化开关同时设为 true。

### 8.2 为什么采用分层判据

明显替代是维护一张静态的开关兼容矩阵；但 compiler、model config、dispatcher 与 PyTorch runtime 会独立演进，矩阵很快失真。选中的判据直接对应当前 guard 和调用链，可以解释失败原因：

1. collective 是否落在正确 named mesh/process group；
2. 边界 redistribution 是否已由 fused module 接管；
3. wait 由 compiler、module 还是 dispatcher 拥有；
4. 是否把 device 动态值 materialize 到 host；
5. 固定容量、完整参数驻留或 symmetric workspace 的内存成本是否可接受。

### 8.3 当前决策表

| 需求 | 路径 | 当前前提 | 明确不保证 |
|---|---|---|---|
| compiled TP micro-pipeline | Async TP | compile model + TP mesh；准确 group 注册 | TP=1 有效果；固定 Inductor kernel schedule |
| eager/compile 都可达的 TP fused GEMM | dist-GEMM | 支持模型、`spmd_types`、SP；TP=1 warning fallback | DTensor backend；跨 stream 并发 workspace 安全 |
| FSDP communication backend | FSDP symm-mem | CUDA / NVIDIA SM90+ guard | 自动 prefetch、自动更改 reshard |
| EP 精确动态容量 | standard / blocking HybridEP / DeepEP compact | 接受 host sync | core Trainer CUDA Graph capture |
| EP 无 host sync capture | MinimalAsyncEP / non-blocking HybridEP | 固定容量与 backend-specific guards | 自动计算-通信 overlap |

当前测试配方分别覆盖 TP2 Async-TP compile、TP2 dist-GEMM 和 FSDP symm-mem，且另有 FSDP2+TP2+PP2+Float8+Async-TP 组合但显式关闭 CUDA Graph（`torchtitan_recipes/tests/h100.py:21-51`）。这些是集成接线证据，不是跨硬件性能结论。

### 8.4 已失效 / 被替代的旧断言

> [!deprecated] **`full_dtensor` 是现行通信优化 backend**
> 已失效。当前只接受 `partial_dtensor` 与默认 `spmd_types`；backend 选择是布局/协议层，不等于 overlap 机制（`torchtitan/config/configs.py:174-180`、`torchtitan/config/configs.py:261-266`）。

> [!deprecated] **Async TP 从 `distributed/tensor_parallel.py` 的 parallelism 开关进入 eager 路径**
> 已被提交 `737594746...` 替代。现行入口是 `CompileConfig.enable_async_tensor_parallel` 与 `distributed/compile.py`，并要求 model compile（`torchtitan/config/configs.py:295-315`、`torchtitan/distributed/compile.py:39-52`）。

> [!deprecated] **dist-GEMM 只是 Async TP 的另一个名字**
> 错。前者是模型 config 选择的显式 fused module，可在 eager Trainer 运行；后者是编译器 pass。二者对 sharding boundary 的 ownership 不同（`torchtitan/models/common/decoder_sharding.py:226-252`、`torchtitan/distributed/compile.py:75-96`）。

> [!deprecated] **DeepEP combine 当前与 shared experts 重叠**
> 错。dispatcher 立即同步 combine，上层随后才计算 shared experts（`torchtitan/models/common/token_dispatcher.py:865-878`、`torchtitan/models/common/moe.py:440-452`）。

> [!deprecated] **MinimalAsyncEP 当前要求 `TP=CP=PP=1`，且 “Async” 意味 microbatch overlap**
> 两者都错。当前验证没有该 degree 限制；实现又明确在 custom-op 内 wait 并否认 microbatch overlap（`torchtitan/distributed/minimal_async_ep/api.py:84-140`、`torchtitan/distributed/minimal_async_ep/api.py:298-315`）。

### 8.5 锚点趋势 / 审慎推断

当前可锚定的方向只有三项：Async TP 的手工 symmetric-memory group 注册等待上游自动化；dist-GEMM 的模块内 collective 使外部 sharding 分支有机会收敛；PP CUDA Graph 需要新的 stage capture boundary。除此之外，源码没有承诺把 FSDP、TP、EP 合并成统一通信 scheduler，故不作此推断。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|ParallelDims 与 Mesh]] —— named mesh、dense/sparse rank 域及 backend-aware group 的来源
- [[02_engineering/02_train_frameworks/torchtitan/12_torchtitan_tp_analysis|Tensor Parallelism]] —— TP/SP 布局、redistribute 调用链与 Async-TP/dist-GEMM 的模型语义
- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|Expert Parallelism]] —— 四类 dispatcher 的布局、路由与完整前后向协议
- [[02_engineering/02_train_frameworks/torchtitan/20_torchtitan_fsdp_prefetch_overlap_memory_analysis|FSDP 预取、重叠与内存]] —— 参数状态、unshard/reshard、显式预取的内部生命周期
- [[02_engineering/02_train_frameworks/torchtitan/21_torchtitan_hsdp_backward_overlap_analysis|HSDP 反向重叠]] —— replicate/shard 两级梯度通信与 backward 排程
- [[02_engineering/02_train_frameworks/torchtitan/23_torchtitan_compute_memory_optimizations_analysis|计算与内存优化]] —— compile component ordering、CUDA Graph capture/replay 与精度/内存组合
- [[02_engineering/02_train_frameworks/torchtitan/26_torchtitan_flex_shard_dist_muon_analysis|FlexShard 与 DistMuon]] —— optimizer step 内 storage/compute layout 重分布及其 all-to-all 流水
