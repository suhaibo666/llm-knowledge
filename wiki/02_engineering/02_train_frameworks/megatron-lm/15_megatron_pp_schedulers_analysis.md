---
title: "Megatron-LM 流水线并行(Pipeline Parallelism)调度器深度解析"
---

# Megatron-LM 流水线并行(Pipeline Parallelism)调度器深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，未改动 `pipeline_parallel/` 下任何调度器源文件，本页落在改动文件（`transformer_config.py`）上的引用已在新基线下逐条打开重核，均为纯行号漂移。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/pipeline_parallel/schedules.py`(2653 行)、`megatron/core/pipeline_parallel/combined_1f1b.py`、`megatron/core/models/common/model_chunk_schedule_plan.py`
> 适用读者:已了解 transformer 训练、TP/DP/PP 基本概念,想吃透 Megatron PP 调度实现的工程师。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **首次阅读边界**：Dense 主线首读 §1–§5 和 §11.1–§11.3，掌握 microbatch、标准 1F1B、VPP 和 bubble 即可；§6–§10、§11.4 与 §12 是 P2P overlap、MoE combined-1F1B、多模块与研究边界的进阶路径。两条路径共用同一张 PP 调度表，因此不拆页。
> **所有权边界**：本页拥有 PP P2P 和 combined-1F1B 的本地触发/等待机制；多个并行轴同时开启后的全局时间线与竞争归 [[20_megatron_comm_overlap_analysis]]。
> **最近更新**：2026-09-03。标明首读/进阶边界，并接收 combined-1F1B 的 `delay_wgrad_compute` 与显存顺序开关 owner。

---

## 1. 背景：模型切成 pp 段之后,难的是"什么时候算、什么时候收发"

### 1.1 要解决的问题

PP 把模型按层切成 `pp` 段,每段一个 stage。这一步只解决"权重和优化器态放不下",**没有**解决"卡怎么才不空转":一个 stage 算完自己那段前向就必须等下游,朴素的"先跑完所有 microbatch 的前向、再统一反向"还会让每个 stage 同时持有所有在飞 microbatch 的激活 —— 显存与空转同时爆掉。

Megatron 的答案是把"何时算前向、何时算反向、何时收发"编成一张**静态时间表**,而不是靠运行时调度。整个 PP 子系统对外只有一个入口 `get_forward_backward_func`(`megatron/core/pipeline_parallel/schedules.py:48`),它挑一张表**返回一个函数**;而所有"重叠"能力(P2P 重叠、MoE A2A 重叠)都不是新表,而是这几张表**内部的分支**(见下方层级图)。本页要回答的就是:这几张表分别长什么样、气泡与显存各值多少、以及为什么重叠是分支而不是并列的调度器(§2)。

### 1.2 调度器分发逻辑

所有 PP 调度的统一入口是 `get_forward_backward_func`(`megatron/core/pipeline_parallel/schedules.py:48`):

```python
if pp_size > 1:
    if vp_size is not None:
        forward_backward_func = forward_backward_pipelining_with_interleaving      # 交错 1F1B (VPP)
    else:
        forward_backward_func = forward_backward_pipelining_without_interleaving   # 标准 1F1B
else:
    forward_backward_func = forward_backward_no_pipelining                         # 无流水线
```

注意:它只在**顶层**三选一。`overlap_p2p_comm`(P2P 重叠)和 `overlap_moe_expert_parallel_comm`(combined-1F1B)不是独立的顶层函数,而是寄生在上面三个之内的**子分支 / 子调度**。所以本文 5 个"调度器"的真实层级关系是:

```
get_forward_backward_func
├── forward_backward_no_pipelining                 ← 调度器①
│      └── overlap_moe_expert_parallel_comm 分支 ── combined_1f1b_schedule_for_no_pipelining   ← 调度器⑤ 的一种形态
├── forward_backward_pipelining_without_interleaving ← 调度器②(标准 1F1B)
└── forward_backward_pipelining_with_interleaving   ← 调度器③(VPP)
       ├── config.overlap_p2p_comm 分支             ← 调度器④(P2P-overlap 1F1B)
       └── overlap_moe_expert_parallel_comm 分支 ── combined_1f1b_schedule_for_interleaved_pipelining ← 调度器⑤ 的另一种形态
```

> [!contradiction] 基线 `71092579` 下顶层分发已不止三选一：`get_forward_backward_func` 最前面新增一条 `isinstance(schedule_pg_collection, MultiModuleProcessGroupCollection)` 早退分支，直接返回 `forward_backward_pipelining_without_interleaving`（`megatron/core/pipeline_parallel/schedules.py:149-150`，#5373 引入）。上面的代码块与树图对应旧基线 `ee3f1ffa…`；新基线里多模块 / 跨网格路径（§8.2）不再经过 `pp_size`/`vp_size` 判断，而是被这条分支直接接管。

| # | 调度器 | 代码位置 | 触发条件 | 本质 |
|---|--------|---------|---------|------|
| ① | 无流水线 | `megatron/core/pipeline_parallel/schedules.py:674` | PP=1 | 基线,无 stage 间通信 |
| ② | 标准 1F1B | `megatron/core/pipeline_parallel/schedules.py:2279` | PP>1, VP=None | 主力调度,峰值显存 O(pp) |
| ③ | 交错 1F1B(VPP) | `megatron/core/pipeline_parallel/schedules.py:1098` | PP>1, VP≠None | 气泡再除以 vp |
| ④ | P2P-overlap 1F1B | `megatron/core/pipeline_parallel/schedules.py:1849` 起的 `overlap_p2p_comm` 分支 | VPP + `overlap_p2p_comm` | 把 P2P 通信移出关键路径 |
| ⑤ | combined-1F1B(MoE A2A 重叠) | `megatron/core/pipeline_parallel/combined_1f1b.py` + `megatron/core/models/common/model_chunk_schedule_plan.py` | `overlap_moe_expert_parallel_comm` | 用 F/B 互相掩盖 MoE all-to-all |

### 1.3 公共记号

| 符号 | 含义 |
|------|------|
| `pp` | 流水线并行度(stage 数 = `pipeline_model_parallel_size`) |
| `r` | 当前 stage 的 rank,`0 ≤ r ≤ pp-1` |
| `m` | 每个 global batch 的 microbatch 数(`num_microbatches`) |
| `vp` | 虚拟流水线度 / 模型块数(`virtual_pipeline_model_parallel_size`,即 `len(model)`) |
| `N` | `microbatch_group_size_per_vp_stage`,默认 `=pp` |
| `t_f` / `t_b` | 单个 microbatch 在**单个完整 stage(L/pp 层)** 上的前向 / 反向耗时 |
| `A` | 单个 microbatch 在单个完整 stage 上的激活显存 |
| `L` | 模型总层数;`Ψ` 模型总参数量 |

### 1.4 公共基础设施(所有调度器共用)

**两个计算原语**
- `forward_step`(`megatron/core/pipeline_parallel/schedules.py:399`):喂入 `input_tensor` → 产出 `output_tensor`;仅末 stage 调 `loss_func`。损失缩放见 `megatron/core/pipeline_parallel/schedules.py:323-342`(除以 token 数、除以 `num_microbatches`,保证 loss 与切分无关)。
- `backward_step`(`megatron/core/pipeline_parallel/schedules.py:534`):对 `output_tensor` 反向,返回对 `input_tensor` 的梯度。

**两个显存优化原语**(理解 PP 显存的关键,`megatron/core/pipeline_parallel/schedules.py:166-228`)
- `deallocate_output_tensor`:激活 `send_forward` 给下游后,其数值在本 stage 已无用,只剩 `.grad_fn` 还要参与反向。于是把 `.data` 替换成标量空张量,**立即释放真实激活显存,同时保留计算图**。
  ```python
  out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
  ```
- `custom_backward`:`torch.autograd.backward` 会校验 output 与 grad 形状一致,而 output 已被缩成标量会报错。于是直接调 C++ 引擎 `Variable._execution_engine.run_backward`(不做形状校验)。

**P2P 通信原语**(`megatron/core/pipeline_parallel/p2p_communication.py`,封装在 `P2PCommunicator`):`recv_forward` / `send_forward` / `recv_backward` / `send_backward`,以及**融合算子** `send_forward_recv_backward`、`send_backward_recv_forward`、`send_forward_recv_forward`、`send_forward_backward_recv_forward_backward`。融合算子让一个 stage **同时**收发两个方向,是 1F1B 高效流水的关键。

**通信张量形状**(`get_tensor_shapes`,`megatron/core/pipeline_parallel/schedules.py:2217`):
```
shape = [ seq_len / (cp_size · tp_size_if_sequence_parallel),  micro_batch_size,  hidden_size ]
```

**P2P 通信的两种底层实现**(`megatron/core/pipeline_parallel/p2p_communication.py`,决定调度器④能否开启):

| 实现 | 触发条件 | 机制 |
|------|---------|------|
| `_batched_p2p_ops`(`:17`) | `batch_p2p_comm=True`(默认) | 把本次要发的方向打包成一组 `P2POp`,一次 `torch.distributed.batch_isend_irecv` 发出;简单规整,但**不能与 `overlap_p2p_comm` 共存** |
| `_p2p_ops`(`:55`) | `batch_p2p_comm=False` | 逐个 `isend`/`irecv`,按 **even/odd rank 错开收发顺序**避免死锁(若所有 rank 同时先发会互等对方 recv);返回带名字的 `reqs` 字典供延迟 `wait()` —— **这是调度器④ `overlap_p2p_comm` 的必需路径** |

细节:`group.size()==2` 且非 UCC 后端时,借全局 `WORLD` group 跑其中一个方向 —— 单个 NCCL 通信器内两个 P2P 会串行化,借 WORLD group 让两个方向真正并行。

`P2PCommunicator`(`:140`)对外暴露的完整方法表:

| 方法 | 方向 | 用途 |
|------|------|------|
| `recv_forward` / `send_forward` | prev↔current / current↔next | 单向激活收发,首/末 stage 自动跳过 |
| `recv_backward` / `send_backward` | next↔current / current↔prev | 单向梯度收发 |
| `send_forward_recv_backward` / `send_backward_recv_forward` | 双向 | 稳态 1F1B 核心融合算子 |
| `send_forward_recv_forward` / `send_backward_recv_backward` | 双向 | warmup/cooldown 用 |
| `send_forward_backward_recv_forward_backward` | 四向 | VPP overlap 优化(调度器④) |

**变长序列:先换形状再换张量**:`config.variable_seq_lengths=True` 时,每个 microbatch 序列长度可能不同,`get_tensor_shapes` 无法静态算出形状。`_communicate_shapes`(`megatron/core/pipeline_parallel/p2p_communication.py:186`)先用一轮 3 元素 int64 tensor 交换形状元数据,再按收到的形状分配 buffer 做真正的张量 P2P;固定长度时跳过这步。

**Hyper Connections 对通信形状的影响**:`config.enable_hyper_connections=True` 时,中间 stage 的 P2P tensor shape 从 `[S,B,H]` 扩为 `[S,B,H×num_residual_streams]`(hyper connection 模块扩展了残差流数量),但只在中间 stage 生效,首/末 stage 仍是标准 `H`(`megatron/core/pipeline_parallel/schedules.py:2217-2276`;结构见 `10_megatron_model_structure_analysis.md` §`HyperConnectionTransformerLayer`)。

**与激活换出的关系**:PP/VPP 显存不够时,除了减小 `pp`/`vp`,还可以用**细粒度激活换出**(`megatron/core/pipeline_parallel/fine_grained_activation_offload.py` 的 `PipelineOffloadManager`,D2H/H2D 异步双流,与重计算正交可叠加)把部分层的激活挪到 CPU pinned memory;`megatron/core/pipeline_parallel/schedules.py` 每步收尾调 `off_interface.reset()`,与 PP/VPP 调度天然兼容。参数与机制见 `22_megatron_memory_optimization_analysis.md` §3.3(现行权威页;含 `saved_tensors_hooks` 挂钩、`OffloadTensorGroup` 双 event、`post_warmup_callback` 自适应调优等机制细节,及 2026-06-16 `max_inflight_offloads` 节流更新)。

### 1.5 PP 进程组与拓扑结构

**进程组创建**(`megatron/core/parallel_state.py:1094`,`initialize_model_parallel` 内):遍历 `decoder_rank_generator.get_ranks('pp')` 生成的 rank 列表逐组 `create_group`,支持三种后端:

| 后端 | 触发 | 说明 |
|------|------|------|
| `None`(默认) | 不设置 | PyTorch 默认 NCCL 后端 |
| `nccl` | 显式指定 | 启用 NCCL 选项优化(`get_nccl_options("pp", ...)`) |
| `ucc` | `pipeline_model_parallel_comm_backend="ucc"` | UCC 统一通信层,需 `CUDA_DEVICE_MAX_CONNECTIONS>1`,设置 `UCX_RNDV_THRESH=0`、`UCC_CL_BASIC_TLS=^sharp,nccl` 等 UCX/UCC 环境变量,利用 UCC 的通用通信层实现跨平台优化 |

**Embedding / Position Embedding 组**(`megatron/core/parallel_state.py:1123`):创建 PP 组的同时,额外建 `EMBEDDING_GROUP`(`get_embedding_ranks`)和 `POSITION_EMBEDDING_GROUP`(`get_position_embedding_ranks`)两个辅助组,用于在 PP **首尾 stage**(权重共享 embedding 与末层 output 常合并成同一份参数)之间同步梯度。

**Defer Embedding WGrad**:embedding 权重梯度是大 GEMM(`vocab_size×hidden_size`,vocab 常 128K),计算耗时但不需要立即完成。`config.defer_embedding_wgrad_compute=True` 时,`drain_embedding_wgrad_compute`(定义在 `megatron/core/utils.py:1167`，调用点 `megatron/core/pipeline_parallel/schedules.py:873` 的 `finish_embedding_wgrad_compute`)把这部分 GEMM 推迟到**pipeline flush 阶段**才执行 —— 利用 cooldown 的 bubble 隐藏这段计算延迟,与"用调度隐藏通信/计算"是同一套设计哲学。

**PP 与其他并行维度的关系**:

| 并行维度 | 通信组 | 物理链路 | 与 PP 的交互 |
|----------|--------|---------|--------------|
| TP | tp_group | NVLink | TP 在 PP stage 内切分 head;PP 在 stage 边界通信 |
| DP | dp_group | IB / NVLink | 梯度同步在 PP 的 backward flush 后进行(见 §②.2 Grad Sync) |
| CP | cp_group | NVLink / IB | CP 切分 seq 后,PP 的 P2P 通信量降低 CP 倍(见 §1.4 通信张量形状) |
| EP | ep_group | IB All-to-All | MoE 层 EP A2A 可与 PP 1F1B 重叠(调度器⑤ combined-1F1B) |
| PP | pp_group | NVLink / IB P2P | 层间切分,stage 间 send/recv 激活值 |

---

## 2. 为什么这么设计：为省显存绕开 PyTorch 的公开反向 API,并把"重叠"做成表内分支

PP 实现里有三处不那么显然的选择:**反向不走 `torch.autograd.backward`**、**P2P 有三条底层实现而"重叠"只能走其中一条**、**VPP 的 microbatch 分组是个可调数而不是写死的深度优先**。前两条源码把理由直接写进了 docstring,第三条只留了取值示例;"为什么重叠是分支而不是新调度器"源码全程沉默。

**① 反向直接调 C++ autograd 引擎,是为了让"把激活缩成标量"这个省显存手段能成立。**
`custom_backward` 的 docstring 把理由和被否掉的替代写在一起:「To make the 'deallocate_output_tensor' (above) optimization work, the C++ autograd engine must be called directly, **bypassing Pytorch's torch.autograd.backward**. Pytorch's 'backward' checks that the output and grad have the same shape, while C++'s 'backward' does not.」(`megatron/core/pipeline_parallel/schedules.py:199-206`)。
链条是这样闭合的:`deallocate_output_tensor` 的 docstring 先说明激活发给下游之后「is only useful for its '.grad_fn' field, and not its '.data'」(`:167-171`),于是把 `.data` 换成 `torch.empty((1,))`(`:196`);形状一旦变成 `[1]`,PyTorch 公开 API 的形状校验必然失败,所以只能直接调 `Variable._execution_engine.run_backward`(`:220-228`,注释直接指向 `torch/csrc/autograd/python_engine.cpp`,`:219`)。`custom_backward` 再用 `assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"`(`:208`)把这个前提反向钉死 —— 两个函数互为对方的前提。
→ 判据是**显存优先于 API 稳定性**:被否掉的替代(用公开 `torch.autograd.backward`)不是做不到,而是做到了就必须放弃 `deallocate_output_tensor`。代价见 §10 第 4 条。

**② P2P 有三条实现,"重叠"只能走逐个 `isend`/`irecv` 那条 —— 另两条一条互斥、一条要求自编译 PyTorch。**
`overlap_p2p_comm` 与 `batch_p2p_comm` 在配置里互相声明互斥:「Must be False if batch_p2p_comm is true.」(`megatron/core/model_parallel_config.py:380-383`)与「Must be False if overlap_p2p_comm is True.」(`:385-388`),运行期在交错调度里再拦一次(`megatron/core/pipeline_parallel/schedules.py:1180-1181`,`ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")`)。
**第三条路是被留下但默认关闭的替代**:`use_ring_exchange_p2p`「Use custom ring_exchange kernel instead of torch.distributed.batch_isend_irecv(). **Requires custom built torch with torch.distributed.ring_exchange.**」(`megatron/core/model_parallel_config.py:395-398`),调用点至今仍在(`megatron/core/pipeline_parallel/p2p_communication.py:226-227`、`:367-373`、`:384`)。判据由 docstring 自陈:它要求一个自己 patch 过的 PyTorch,所以进不了默认路径。另一处同样"留在原地当兼容层"的是 `batch_p2p_sync`:「When using batch_isend_irecv, do a cuda.device.synchronize afterward **to work around a bug in older version of PyTorch**」(`:390-393`)。
这两条路的时间线也能读出来:`overlap_p2p_comm` 由提交 `f9283c5a8`(2023-05-31,commit message 即「Add option to overlap p2p communication.」)引入;§1.4 里那段"`group.size()==2` 时借全局 WORLD 组跑其中一个方向"的写法来自 `3ee489d9e`(2024-05-31,「Enable virtual pipelining and P2P communication overlap at PP=2」)—— 标题本身就说明它是为了让 PP=2 这个退化拓扑也能重叠。

**③ VPP 的 microbatch 分组做成可调数,而不是写死深度优先。**
`microbatch_group_size_per_vp_stage` 由提交 `4295be1a6`(2024-11-01,commit message 即「ADLR/megatron-lm!2117 - tunable schedule with overlapping」)引入 —— 标题里的 "tunable" 指的就是它。默认值在 `__post_init__` 里被设成 `pipeline_model_parallel_size`,docstring 明写该默认「specifies a depth-first schedule」(`megatron/core/model_parallel_config.py:426-435`、`:600-601`),并给出取 2 与取 3 两组 rank 时间线示范排布差异(`:436-444`)。**被否掉的替代就是写死深度优先**;但源码只给了时间线,没写这个旋钮换的是什么 —— 见下方推断。

> [!note] 推断
> ①② 的理由都由 docstring 自陈,可以直接引用。**下面两条是本页的重建,不是源码的自陈:**
> (a) **`microbatch_group_size_per_vp_stage` 的判据。** 源码只给了取值范围校验(必须落在 `[pp, m]`,否则 `ValueError`,`megatron/core/pipeline_parallel/schedules.py:1255-1262`;余数不为 0 且小于 `pp` 时 `RuntimeError`,理由原话「it introduces dependency bubbles in the pipeline and reduces throughput」,`:1266-1274`)与两组时间线,**没有**说"调大这个数是拿显存换气泡"。本页据时间线重建:一组内同时在飞的 microbatch 越多,warmup 期堆积的激活越多、气泡填得越满。要引用这条判断,请回到 `megatron/core/model_parallel_config.py:436-444` 与 `megatron/core/pipeline_parallel/schedules.py:1255-1274`,不要引用本段。
> (b) **为什么"重叠"是表内分支,而不是与三张表并列的第四、第五张表。** 源码只呈现结构:`get_forward_backward_func` 返回的是函数而不是调度器对象(`megatron/core/pipeline_parallel/schedules.py:48`),`overlap_p2p_comm` 与 `overlap_moe_expert_parallel_comm` 都在被返回的函数体内部分支(§1.2 的层级图),**没有任何注释解释这个取舍**。本页的重建是:重叠改的只是"同一张表里通信何时被 `wait()`",时间表本身不变,拆成独立函数会把 warmup / steady / cooldown 三段逻辑整体复制一遍。这是推断,不要当作作者的自陈。

---

## 3. 调度器① — 无流水线 `forward_backward_no_pipelining`

### ①.1 动机与解决的问题

**动机**:`PP=1` 时不存在流水线,但 Megatron 仍需要一个统一签名的 `forward_backward_func`,让上层训练循环无需区分是否开 PP。它也是其他调度器的**正确性基线**和 CUDA Graph 捕获的兜底路径。

**解决的问题**:在单个 PP stage 内对 `m` 个 microbatch 做**梯度累加**(gradient accumulation),把显存撑不下的大 global batch 拆成多次小前向/反向。

### ①.2 源码与流程

`megatron/core/pipeline_parallel/schedules.py:674`。核心是一个朴素循环 + `no_sync` 上下文:

```python
with no_sync_func():                          # 前 m-1 个 microbatch:只累加梯度,不触发 DP 通信
    for i in range(num_microbatches - 1):
        output_tensor, num_tokens = forward_step(...)
        if not forward_only:
            backward_step(input_tensor, output_tensor, output_tensor_grad, config)
# 最后一个 microbatch 放在 no_sync 之外 —— 触发 DP 梯度 all-reduce / reduce-scatter
output_tensor, num_tokens = forward_step(...)
if not forward_only:
    backward_step(...)
```

若开启 `overlap_moe_expert_parallel_comm`,则改走 `combined_1f1b_schedule_for_no_pipelining`(见调度器⑤)。

**流程图**

```
┌─────────────────────────────────────────────┐
│ for i in 0 .. m-2:   (no_sync 上下文内)        │
│    ┌──────────┐   ┌───────────┐               │
│    │ forward  │ → │ backward  │ → 梯度累加      │
│    └──────────┘   └───────────┘               │
├─────────────────────────────────────────────┤
│ i = m-1:             (no_sync 之外)            │
│    forward → backward → 触发 DP 梯度同步        │
└─────────────────────────────────────────────┘
```

### ①.3 流水线模拟图

无流水线,单设备时间轴串行(`F`=前向, `B`=反向):

```
Device 0 | F0 B0  F1 B1  F2 B2  F3 B3 |  → DP all-reduce
时间 →     └ mb0 ┘└ mb1 ┘└ mb2 ┘└ mb3 ┘
```

### ①.4 开销分析

| 维度 | 取值 | 说明 |
|------|------|------|
| **峰值激活显存** | `1 × A`(完整模型) | F 完立刻 B,任意时刻只有 1 个 microbatch 的激活在世 |
| **模型态显存/卡** | `18 · Ψ`(Adam 混合精度,**不分摊**) | PP=1,无层切分 |
| **PP 通信量** | `0` | 无 stage 间通信 |
| **PP 气泡** | `0` | 无流水线就无气泡 |

模型态 18 bytes/param 拆解:bf16 权重 2 + fp32 梯度 4 + fp32 master 4 + Adam 动量 4 + Adam 方差 4(bf16 训练梯度强制 fp32 累加,详见 ddp_optimizer_analysis)。

> [!update] 2026-09-02 · 系数更正
> 本节表格与 §②.4 的公式原写 `16 · Ψ` / `16·N_params`，与紧邻的拆解算式（2+4+4+4+4 = **18**）自相矛盾，现已统一为 `18`。
> `16` 来自 ZeRO 论文的 fp16 假设（梯度按 2 字节算）；Megatron 在 bf16 下**强制 fp32 梯度累加**，梯度是 4 字节——
> `megatron/training/arguments.py:1331-1333` 在未显式关闭时把 `accumulate_allreduce_grads_in_fp32` 置真，
> 并打印 "accumulate and all-reduce gradients in fp32 for bfloat16 data type."。
> [[26_megatron_optimizer_step_internals_deepdive]] §4.2 按 18 立论（该机制在 2026-09-03 从 16 拆出），本页此前未同步。

### ①.5 适用场景及原因

- **模型 + 激活单卡(或单 TP/DP 组)能放下**:此时引入任何流水线只会带来气泡,纯亏吞吐。
- **小模型调试 / 单元测试 / CUDA Graph 捕获兜底**:行为最简单、最易复现。
- **不推荐**用于装不下的大模型 —— 那是 PP 存在的理由。

---

## 4. 调度器② — 标准 1F1B `forward_backward_pipelining_without_interleaving`

### ②.1 动机与解决的问题

**要解决的根本问题**:模型大到单卡放不下,必须按层(深度)切成 `pp` 段。但朴素切分有两个致命缺陷:

1. **朴素顺序流水**:任意时刻只有 1 个 stage 在算,利用率仅 `1/pp`。
2. **GPipe 式调度**(先全前向、再全反向):利用率上去了,但**峰值激活显存 ∝ m** —— 必须同时囤 `m` 份激活等反向,`m` 一大就 OOM。

**1F1B 的动机**:在稳态让每个设备"做一次前向、立刻做一次反向"。反向一旦完成就能释放对应 microbatch 的激活,使**在世激活数从 `m` 压到 `pp`**。1F1B 的气泡率与 GPipe **完全相同**,它换来的纯粹是显存。

> GPipe vs 1F1B 对照(GPipe 不在本仓库,仅作动机说明):
> ```
> GPipe   : 在世激活 = m 份  →  峰值显存 ∝ m   (m≫pp 时 OOM)
> 1F1B    : 在世激活 = pp 份  →  峰值显存 ∝ pp   (与 m 无关)  ✅
> ```

### ②.2 源码与流程

`megatron/core/pipeline_parallel/schedules.py:2279`。三阶段:**warmup(纯前向)→ 稳态 1F1B → cooldown(纯反向)**。

**① 热身数量**(`megatron/core/pipeline_parallel/schedules.py:2417`):
```python
num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)   # = min(pp - r - 1, m)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches
```
即 `warmup = pp - r - 1`:**越靠前的 stage(rank 小)热身越多**,因为它的反向梯度要等最久才能从末端回流。

**② Warmup —— 只前向**(`megatron/core/pipeline_parallel/schedules.py:2478`):
```python
for i in range(num_warmup_microbatches):
    input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, ...)
    output_tensor, num_tokens = forward_step(...)
    p2p_communicator.send_forward(output_tensor, ...)
    if not forward_only:
        input_tensors.append(input_tensor)     # 入队,留给后面的反向
        output_tensors.append(output_tensor)
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

> **Partial Activation Checkpointing(仅影响 warmup 段的显存-计算权衡)**:当 `config.num_microbatches_with_partial_activation_checkpoints` 不为空时(`megatron/core/pipeline_parallel/schedules.py:2421`),`max_outstanding_backprops = num_warmup_microbatches + 1`;`forward_step` 据此决定每个 microbatch 是否 checkpoint:
> ```python
> checkpoint_activations_microbatch = (
>     i % max_outstanding_backprops >= config.num_microbatches_with_partial_activation_checkpoints
> )
> ```
> **原理**:warmup 阶段 outstanding backprop 数量逐渐增加(最多 `num_warmup_microbatches+1`),越早进入的 microbatch 在世时间越长、显存压力越大。前 `num_microbatches_with_partial_activation_checkpoints` 个 microbatch 落在 window 内不做 checkpoint(省重算开销),其余 microbatch 做 full checkpointing —— 这是"重计算 × microbatch 维度"的自适应窗口策略(配套阅读:`18_megatron_recompute_analysis.md`)。

**③ 稳态 1F1B**(`megatron/core/pipeline_parallel/schedules.py:2523`),核心是用**融合通信**把"发激活"和"收梯度"合并:
```python
for i in range(num_microbatches_remaining):
    output_tensor, num_tokens = forward_step(...)                                  # 一次前向
    output_tensor_grad = p2p_communicator.send_forward_recv_backward(output_tensor, ...)  # 发激活 + 收梯度

    input_tensors.append(input_tensor);  output_tensors.append(output_tensor)
    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    input_tensor  = input_tensors.pop(0)        # FIFO:取出最早的 microbatch
    output_tensor = output_tensors.pop(0)
    input_tensor_grad = backward_func(input_tensor, output_tensor, output_tensor_grad, config)  # 一次反向

    if last_iteration:
        p2p_communicator.send_backward(input_tensor_grad, ...)
    else:
        input_tensor = p2p_communicator.send_backward_recv_forward(input_tensor_grad, ...)  # 发梯度 + 收激活
```
> `input_tensors` / `output_tensors` 是 FIFO 队列:稳态每轮 `append` 一个、`pop(0)` 一个,**队列长度恒为 `warmup + 1`** —— 这就是 1F1B 峰值显存为 `O(pp)` 的代码证据。

**④ Cooldown —— 只反向**(`megatron/core/pipeline_parallel/schedules.py:2596`):把队列里剩下的 `warmup` 个 microbatch 清空。

最后 `finalize_model_grads_func`(`megatron/core/pipeline_parallel/schedules.py:2637`)做 DP 梯度规约、SP 的 layernorm all-reduce、PP 首尾 stage 共享 embedding 的 all-reduce。

**Grad Sync 与 Bubble 的重叠**:稳态 1F1B 中,DP 梯度 AllReduce 默认通过 `no_sync_func` 禁用,直到最后一个 microbatch 的 backward 完成才由 `enable_grad_sync()`(`megatron/core/pipeline_parallel/schedules.py:2598`)启用。关键在**非首 stage**:它们在 cooldown 阶段还有若干 backward 要做,梯度 AllReduce 被安排在这些 cooldown microbatch 的 bubble 期间**异步执行**,把 DP 通信开销藏进 idle time;首 stage 因为最早耗尽 warmup、最早进入 flush,梯度同步只能在 flush 时完成,没有 bubble 可借。

**流程图**

```
                  ┌──────────── warmup (pp-r-1 次) ────────────┐
recv_forward ─► forward_step ─► send_forward ─► 入队 input/output_tensors
                  └────────────────────────────────────────┘
                                  │
                  ┌───────────── 稳态 1F1B (m-warmup 次) ──────────────┐
        forward_step ─► send_forward_recv_backward(发激活+收梯度)
              │                                            │
        入队 ──┘                          pop(0) 出队 ──► backward_step
                                                           │
                              send_backward_recv_forward(发梯度+收激活)
                  └──────────────────────────────────────────────────┘
                                  │
                  ┌──────────── cooldown (pp-r-1 次) ───────────┐
        pop(0) ─► recv_backward ─► backward_step ─► send_backward
                  └────────────────────────────────────────┘
                                  │
                       finalize_model_grads (DP/SP/embedding 规约)
```

### ②.3 流水线模拟图

`pp=4, m=8`,设 `t_f = t_b = 1`。每个设备的**精确执行序列**(直接来自源码,`[F B]` 表示一对稳态 1F1B):

```
Dev0 : F0 F1 F2 [F3 B0][F4 B1][F5 B2][F6 B3][F7 B4] B5 B6 B7      warmup=3, cooldown=3
Dev1 : F0 F1 [F2 B0][F3 B1][F4 B2][F5 B3][F6 B4][F7 B5] B6 B7     warmup=2, cooldown=2
Dev2 : F0 [F1 B0][F2 B1][F3 B2][F4 B3][F5 B4][F6 B5][F7 B6] B7    warmup=1, cooldown=1
Dev3 : [F0 B0][F1 B1][F2 B2][F3 B3][F4 B4][F5 B5][F6 B6][F7 B7]   warmup=0, cooldown=0
```

完整空间-时间图(由上面的精确序列 + 数据依赖逐 op 解算得到,`t_f=t_b=1`,槽 1–22,`··`=气泡空泡)。
**关键:一个 microbatch 的前向、反向各自沿对角线逐级传播,绝不会整列同时算同一个 microbatch** ——
反向 `Bj` 必须在末端 Dev3 算完后,才依次回传 Dev2→Dev1→Dev0:

```
slot  1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22
Dev0  F0  F1  F2  F3  ··  ··  ··  B0  F4  B1  F5  B2  F6  B3  F7  B4  ··  B5  ··  B6  ··  B7
Dev1  ··  F0  F1  F2  ··  ··  B0  F3  B1  F4  B2  F5  B3  F6  B4  F7  B5  ··  B6  ··  B7  ··
Dev2  ··  ··  F0  F1  ··  B0  F2  B1  F3  B2  F4  B3  F5  B4  F6  B5  F7  B6  ··  B7  ··  ··
Dev3  ··  ··  ··  F0  B0  F1  B1  F2  B2  F3  B3  F4  B4  F5  B5  F6  B6  F7  B7  ··  ··  ··
```

读图:
- **前向对角线**:`F0` 走 Dev0@1 → Dev1@2 → Dev2@3 → Dev3@4(激活沿设备下行)。
- **反向对角线**:`B0` 走 Dev3@5 → Dev2@6 → Dev1@7 → Dev0@8(梯度沿设备上行)—— 这正是"不可能同列"的根因:`B0@Dev_d` 必须等 `B0@Dev_{d+1}` 完成。
- 槽 1–7 = **填充**(各设备错位启动 warmup 前向);槽 8–16 = **稳态 1F1B 密排**(4 卡全忙、无空泡);槽 17–22 = **排空**(cooldown 反向)。
- 每个设备恰好 16 个计算 op(8 F + 8 B)+ **6 个空泡** = 22 槽。空泡数 `= (pp-1)(t_f+t_b) = 3×2 = 6`,各设备相等,只是分布位置不同:靠前设备(Dev0)空泡偏中后段(槽 5–7、17、19、21),靠后设备(Dev3)空泡偏开头(槽 1–3)与结尾。
- makespan `= (m+pp-1)(t_f+t_b) = (8+3)×2 = 22`。

### ②.4 开销分析

**峰值激活显存**:rank `r` 上 FIFO 队列峰值长度 `= warmup + 1 = pp - r`。
$$
M_{\text{act}}^{\text{1F1B}}(r) = (pp-r)\cdot A \le pp\cdot A \quad(\text{rank 0 最坏,且与 }m\text{ 无关})
$$

**模型态显存/卡**:PP 把模型按层切 `pp` 段,每卡只持有 `1/pp`:
$$
M_{\text{model}}^{\text{per-device}} = \frac{18\cdot N_{\text{params}}}{pp}\ \text{bytes} \quad(\textbf{PP 的根本收益})
$$

**气泡推导**(从末端设备空闲入手):
- 第 1 个 microbatch 前向需先穿过前 `pp-1` 个 stage 才到末端 → 开头空闲 `(pp-1)t_f`。
- 末端做完最后一个反向后,梯度还要回穿 `pp-1` 个 stage → 结尾空闲 `(pp-1)t_b`。
- 故每设备气泡 `t_bubble = (pp-1)(t_f+t_b)`,有效计算 `t_id = m(t_f+t_b)`。

$$
\boxed{
\begin{aligned}
\text{Bubble}_{\text{1F1B}}
&= \frac{t_{\text{bubble}}}{t_{\text{id}}}
= \frac{(pp-1)(t_f+t_b)}{m(t_f+t_b)}
= \frac{pp-1}{m}
\end{aligned}
}
$$

**示例**:`pp=4, m=8` → Bubble `= 3/8 = 37.5%`;`pp=4, m=32` → Bubble `= 3/32 = 9.4%` —— 印证"`m` 越大气泡越小"。

**通信量**:每个 microbatch 前向穿 `pp-1` 个边界、反向穿 `pp-1` 个边界,每次传一个 `[s/(cp·tp), b, h]` 张量:
$$
\text{P2P 传输次数} = 2m(pp-1),\quad \text{单次量} = \frac{s\cdot b\cdot h}{cp\cdot tp}\ \text{元素}
$$

| 维度 | 1F1B |
|------|------|
| 峰值激活显存 | `(pp-r)·A` ≤ `pp·A` |
| 模型态显存/卡 | `18·Ψ / pp` |
| 气泡率 | `(pp-1)/m` |
| P2P 通信量 | `2m(pp-1)` 次点对点 |

### ②.5 适用场景及原因

- **模型态单卡放不下、需跨节点扩展**:PP 只传边界激活(点对点、量小),与 TP 的逐层 all-reduce 相比通信便宜得多 → **PP 适合跨机,TP 适合机内 NVLink**。
- **`m ≫ pp`**(典型 `m ≥ 8p`):此时气泡率 `(pp-1)/m` 已足够小,不必引入 VPP 的额外通信。
- **显存是首要约束**:1F1B 峰值激活 `O(pp)` 是所有 1F1B 系调度里最省的。
- **跨节点带宽紧张**:不能用 VPP(VPP 通信 ×vp)时,标准 1F1B 是最稳的选择。
- **典型工业配置**:`TP(机内 NVLink)× PP(跨机 IB)× DP(最外层)`。

---

## 5. 调度器③ — 交错 1F1B / 虚拟流水线 VPP `forward_backward_pipelining_with_interleaving`

### ③.1 动机与解决的问题

**问题**:标准 1F1B 气泡率 `(pp-1)/m`。当 `pp` 大而 `m` 不够大(`m/pp` 小,例如 `pp=16, m=32` → 气泡 47%),气泡成为吞吐瓶颈;而无限增大 `m` 会撑爆激活显存,也受 global batch size 上限约束。

**VPP 的动机**:把每张卡负责的连续层段再拆成 `vp` 个**不连续的小模型块(model chunk)**。设备 0 不再只管 `[0, L/pp)` 层,而是同时管 chunk0 `[0, L/(pv))`、chunk1、chunk2……分散在模型不同深度。一个 microbatch 在单个"设备-块"上的前向耗时降为 `t_f/vp`,流水线"填充/排空"时间随之缩短 `vp` 倍 → **气泡率除以 `vp`**。

代价:每个 microbatch 要在设备间往返 `vp` 趟,**P2P 通信量 ×vp**;激活显存略增。

### ③.2 源码与流程

`megatron/core/pipeline_parallel/schedules.py:1098`。`model` 参数变成一个 module 列表(`len(model) = vp`)。VPP 把每张卡负责的连续层段拆成 `vp` 个**不连续的 model chunk**,分散在模型不同深度,而不是整段连续层。例如 `pp=4, vp=2, 16` 层:

| 物理 GPU | VPP Stage 0 | VPP Stage 1 |
|----------|-------------|-------------|
| GPU 0 | layers 1-2 | layers 9-10 |
| GPU 1 | layers 3-4 | layers 11-12 |
| GPU 2 | layers 5-6 | layers 13-14 |
| GPU 3 | layers 7-8 | layers 15-16 |

一般规则:GPU `r`(0-indexed)持有的 `vp` 个 chunk 里,第 `c` 个(0-indexed)覆盖层区间 `[c·pp·(L/pp/vp) + r·(L/pp/vp), c·pp·(L/pp/vp) + (r+1)·(L/pp/vp))`(`L` 为总层数,每个 chunk 厚度 `L/(pp·vp)` 层);这种交错放置让每个物理 stage 内部有多个独立的 forward/backward 流可以交替执行,从而把 pipeline bubble 打得更碎(即 §③.1 的"气泡率除以 `vp`")。

**① 热身数量**(`get_pp_rank_microbatches`,`megatron/core/pipeline_parallel/schedules.py:913`):
```python
num_warmup_microbatches  = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
# 即 warmup = 2(pp - r - 1) + (vp - 1)·N
total_num_microbatches = num_microbatches * num_model_chunks   # = m·vp
```

**② 调度表(schedule table)—— VPP 的灵魂**(`get_schedule_table`,`megatron/core/pipeline_parallel/schedules.py:938`):把"虚拟 microbatch 序号"映射到 `(microbatch_id, model_chunk_id)`。源码注释里的例子(PP2、`m=5`、`vp=2`、`N=3`):
```
virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
microbatch_id         | 0 1 2 0 1 2 3 4 3 4
model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
```
`get_model_chunk_id`(`megatron/core/pipeline_parallel/schedules.py:1346`)在反向时把 chunk 顺序反转(`vp - id - 1`),因为反向要从深层 chunk 往浅层走。

**③ 三阶段主循环**:warmup(`megatron/core/pipeline_parallel/schedules.py:1675`)/ 稳态 1F1B(`megatron/core/pipeline_parallel/schedules.py:1835`)/ cooldown(`megatron/core/pipeline_parallel/schedules.py:2073`),结构与标准 1F1B 同构,但:
- `input_tensors` / `output_tensors` 变成**每个 chunk 一个队列的二维列表**(`megatron/core/pipeline_parallel/schedules.py:1240`)。
- 每步通过 `forward_step_helper` / `backward_step_helper`(`megatron/core/pipeline_parallel/schedules.py:1487` / `1560`)按调度表跳到正确的 `model[chunk_id]` 与 `data_iterator[chunk_id]`。
- `num_released_microbatches`(`megatron/core/pipeline_parallel/schedules.py:1359`)处理首 rank 缓冲多个输入时的索引偏移。

**④ 约束**(`megatron/core/pipeline_parallel/schedules.py:1254-1274`):`N ∈ [pp, m]`;`m % N` 必须为 0 或 ≥ `pp`(否则末尾 microbatch 组不满 → 额外气泡,直接 `raise`);`num_layers` 须能被 `pp·vp` 整除。

**流程图**

```
build schedule_table  →  microbatch_id_table, model_chunk_id_table
        │
        ▼
warmup: for k in 0..warmup-1
        chunk = get_model_chunk_id(k, forward=True)
        forward_step_helper(k) on model[chunk]
        send_forward_recv_forward → input_tensors[next_chunk].append(...)
        │
        ▼
steady: for k in 0..remaining-1            forward_k = k + warmup
        fwd: forward_step_helper(forward_k) on model[chunk_f]
        bwd: backward_step_helper(k)        on model[chunk_b = vp-1-chunk]
        send_forward_backward_recv_forward_backward (一次融合四向通信)
        │
        ▼
cooldown: for k in remaining..m·vp-1
        backward_step_helper(k);  send_backward_recv_backward
        │
        ▼
finalize_model_grads (逐 chunk 规约未同步的梯度)
```

### ③.3 流水线模拟图

取 `pp=4, vp=2, m=8, N=4`(`N = microbatch_group_size_per_vp_stage` 取默认值 `pp`;受约束"`m % N` 为 0 或 ≥ pp",`m` 须为 4 的倍数,`m=8` 是非退化的最小值)。关键直觉:**当某设备等待时,它总能切到另一个 chunk 干活,从而填掉原本的气泡。**

记号:`f`/`F` = 前向 chunk0 / chunk1,`b`/`B` = 反向 chunk0 / chunk1,数字 = microbatch 编号。chunk0 在浅层、chunk1 在深层;一个 microbatch 前向路径 `f →(过PP)→ F`,反向路径 `B →(过PP)→ b`(反向从深层 chunk 开始)。

调度表(`get_schedule_table(m=8, vp=2, N=4)`,virtual_microbatch_id → microbatch_id / model_chunk_id):
```
virtual_microbatch_id | 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
microbatch_id         | 0  1  2  3  0  1  2  3  4  5  6  7  4  5  6  7
model_chunk_id        | 0  0  0  0  1  1  1  1  0  0  0  0  1  1  1  1
```
各 rank warmup `= 2(pp-r-1)+(vp-1)N`:rank0=10, rank1=8, rank2=6, rank3=4。

![标准 1F1B 与 VPP 的同轴对照：两个面板共用一条按真实耗时刻度的时间轴，VPP 每格只有 t_f/vp，makespan 从 22 t_f 降到 19 t_f，空泡/计算从 (pp−1)/m=3/8 降到 (pp−1)/(m·vp)=3/16；蓝框标出 mb0 的前向要往返 vp=2 趟，这既是收益来源也是 P2P ×vp 的代价来源。](assets/megatron_pp_vpp_vs_1f1b.svg)

*图 ③-1 交错 1F1B(VPP)：气泡率 ÷ vp，代价是 P2P ×vp。格子由 `tools/figs/svg/lib/megatron_pp_sim.mjs` 的离散事件仿真解算(算法照本页基线的 `megatron/core/pipeline_parallel/schedules.py`)，与下方 ASCII 表同源，故两者不可能对不上。*

下表是同一份解算结果的精确文本形式(便于 grep 与逐格核对)。完整空间-时间图(槽 1–38,一格 = 一个 chunk-op,耗时 `t_chunk = t_f/vp`):
```
slot  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38
Dev0  f0 f1 f2 f3 F0 F1 F2 F3 f4 f5 f6 B0 f7 B1 .. .. F4 B2 F5 B3 F6 b0 F7 b1 b2 b3 .. B4 .. B5 .. B6 .. B7 b4 b5 b6 b7
Dev1  .. f0 f1 f2 f3 F0 F1 F2 F3 f4 B0 f5 B1 f6 B2 f7 B3 F4 b0 F5 b1 F6 b2 F7 b3 .. B4 .. B5 .. B6 .. B7 b4 b5 b6 b7 ..
Dev2  .. .. f0 f1 f2 f3 F0 F1 F2 B0 F3 B1 f4 B2 f5 B3 f6 b0 f7 b1 F4 b2 F5 b3 F6 B4 F7 B5 .. B6 .. B7 b4 b5 b6 b7 .. ..
Dev3  .. .. .. f0 f1 f2 f3 F0 B0 F1 B1 F2 B2 F3 B3 f4 b0 f5 b1 f6 b2 f7 b3 F4 B4 F5 B5 F6 B6 F7 B7 b4 b5 b6 b7 .. .. ..
```

读图:
- **前向对角线**:`f0` 走 Dev0@1→Dev1@2→Dev2@3→Dev3@4;chunk0 在末端算完后,`F0`(chunk1)才从 Dev0@5 起步,再 →Dev1@6→Dev2@7→Dev3@8。
- **反向对角线**:`B0`(chunk1 反向)走 Dev3@9→Dev2@10→Dev1@11→Dev0@12;chunk1 反向在首端算完后,`b0`(chunk0 反向)才从 Dev3@17 起步上行。对角线在设备繁忙时会被**拉长**(如 `b0` 到 Dev0 已是槽 22),但**绝不会压成同列**。
- 槽 1–10 各设备错位灌入 warmup 前向(rank0 灌 10 个);中段 1F1B 密排;尾段 cooldown 排空。
- 每设备 32 个 chunk-op + **6 个空泡** = 38 槽。空泡 `= (pp-1)(t_f+t_b)/vp = 3×2/2 = 3` 个 stage-op `= 6` 个 chunk-op,气泡率 `6/32 = 3/16 = (pp-1)/(mv)`。Dev3 空泡集中在首尾(槽 1–3、36–38),靠前设备的空泡更分散。
- 对比标准 1F1B(`pp=4, m=8`,§②.3):makespan 22 个 stage-op;VPP makespan 38 个 chunk-op `= 19` 个 stage-op —— **更快**,气泡率从 `(pp-1)/m = 3/8` 降到 `(pp-1)/(mv) = 3/16`。

### ③.4 开销分析

**气泡推导**:一个 microbatch 在单个"设备-块"上前向只需 `t_f/vp`。重复 1F1B 的推导,填充/排空时间整体除以 `vp`:
$$
t_{\text{bubble}}^{\text{VPP}} = \frac{(pp-1)(t_f+t_b)}{vp}
$$
$$
\boxed{\text{Bubble}_{\text{VPP}} = \frac{1}{vp}\cdot\frac{pp-1}{m} = \frac{pp-1}{m\cdot vp}}
$$

**示例**:`pp=4, vp=2, m=8` → Bubble `= 3/(8×2) = 18.75%`(同 `pp,m` 下比标准 1F1B 的 `37.5%` 减半)。

**峰值激活显存**:rank 0 热身数 `warmup = 2(pp-1)+(vp-1)N`(取 `N=pp`),每个囤积单元是一个 chunk(`A/vp`):
$$
\begin{aligned}
M_{\text{act}}^{\text{VPP}}
&\approx \big[2(pp-1)+(vp-1)pp+1\big]\cdot\frac{A}{vp} \approx \Big(1+\frac{1}{vp}\Big)\,pp\,A
\end{aligned}
$$
即约为标准 1F1B 的 `(vp+1)/vp` 倍(`vp=2` → 1.5×),**略高于** 1F1B。

**通信量**:每个 microbatch 在设备间往返 `vp` 趟:
$$
\text{P2P 传输次数} \approx 2m\cdot vp\cdot(pp-1) = vp\times\text{标准 1F1B}
$$

| 维度 | 标准 1F1B | VPP(vp 块) |
|------|-----------|-------------|
| 气泡率 | `(pp-1)/m` | `(pp-1)/(m·vp)` ✅ |
| 峰值激活显存 | `pp·A` | `≈(1+1/vp)·pp·A` |
| 模型态显存/卡 | `18·Ψ/pp` | `18·Ψ/pp`(相同) |
| P2P 通信量 | `2m(pp-1)` | `≈2m·vp·(pp-1)`(×vp) |

### ③.5 适用场景及原因

- **`pp` 大、`m/pp` 偏小**:气泡率 `(pp-1)/m` 高,VPP 直接把它除以 `vp`,收益最大。
- **网络带宽足以吸收 `vp×` 通信**:通常机内 NVLink 充裕;跨机时建议同时开 `overlap_p2p_comm`(调度器④)把通信藏起来。
- **`num_layers` 能被 `pp·vp` 整除**:这是硬约束。
- **不推荐**:跨机带宽极紧、或 `m ≫ pp` 已使气泡可忽略时 —— 多付的 `vp×` 通信不划算。

---

## 6. 调度器④ — P2P-overlap 1F1B(`overlap_p2p_comm` 分支)

> 它不是独立函数,而是 `forward_backward_pipelining_with_interleaving` 内 `config.overlap_p2p_comm == True` 时走的另一套通信路径(`megatron/core/pipeline_parallel/schedules.py:1849` 起)。要求 `batch_p2p_comm = False`(`megatron/core/model_parallel_config.py:385` 定义，`megatron/core/pipeline_parallel/schedules.py:1180-1181` 断言；旧基线页面误归到 `megatron/core/transformer/transformer_config.py`，已更正)。

### ④.1 动机与解决的问题

**问题**:VPP(调度器③)默认用 `send_forward_backward_recv_forward_backward` —— 这是一个**同步**融合算子(`megatron/core/pipeline_parallel/schedules.py:2036`)。它把"算完一步 → 同步收发张量 → 再算下一步"串成串。当 `vp×` 通信量遇上跨机 IB 带宽时,这段 P2P 通信会**暴露在关键路径上**,稳态每一步都要等通信完成。

**`overlap_p2p_comm` 的动机**:把 P2P 通信改成**异步 `isend`/`irecv`**,只拿到 wait handle 不立即等待;在下一次计算**之后**才 `wait()`。这样通信与计算在 GPU 上并行,P2P 延迟被计算掩盖,移出关键路径。`overlap_p2p_comm_warmup_flush`(`megatron/core/model_parallel_config.py:417`)进一步把这种重叠扩展到 warmup / cooldown 阶段。

### ④.2 源码与流程

稳态循环(`megatron/core/pipeline_parallel/schedules.py:1849`)用 4 个回调把通信拆成"预取-计算-收尾":

```python
def pp_pre_forward(...):    # 计算前:等上一轮预取的 recv 完成
    recv_prev_wait_handle = recv_prev_wait_handles.pop(0);  recv_prev_wait_handle.wait()
    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

def pp_post_forward(output_tensor, ...):   # 计算后:异步发激活 + 异步预取下一个激活
    fwd_recv_buffer[...], fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
        output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, overlap_p2p_comm=True)
    if send_next_wait_handle is not None:  send_next_wait_handle.wait()
    send_next_wait_handle = fwd_wait_handles.pop("send_next")          # 句柄留到下一轮再等
    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

def pp_pre_backward(...):   # 与 pp_pre_forward 对称(recv_next)
def pp_post_backward(input_tensor_grad, ...):   # 与 pp_post_forward 对称(send_backward_recv_backward)

output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
    f_virtual_microbatch_id=forward_k, b_virtual_microbatch_id=backward_k,
    pre_forward=pp_pre_forward, post_forward=pp_post_forward,
    pre_backward=pp_pre_backward, post_backward=pp_post_backward)
```

关键机制:
- **专用收发缓冲** `fwd_recv_buffer` / `bwd_recv_buffer`(`megatron/core/pipeline_parallel/schedules.py:1668`),首/末 rank 缓冲大小 `= N - pp + 1`,用于预取多个张量。
- **延迟等待**:`send_next_wait_handle`、`recv_prev_wait_handles` 等句柄跨迭代保存,在真正需要数据时才 `wait()`。
- **释放安全点**(`megatron/core/pipeline_parallel/schedules.py:1788`):`isend` 是异步拷贝,释放源缓冲前必须确认拷贝完成,否则下游收到脏数据 —— 当 `deallocate_pipeline_outputs` 为真时强制 `wait` 一次。

**流程图(稳态单步)**

```
┌─ pp_pre_forward ──────────────┐
│  wait(上一轮预取的 recv_prev)  │
└───────────────┬───────────────┘
                ▼
       forward_step_helper(forward_k)         ← 计算
                │
┌─ pp_post_forward ─────────────┐
│  isend 激活(异步,存 send_next 句柄)         │
│  irecv 下一激活(异步,存 recv_prev 句柄)     │  ← 通信与下面的反向计算并行
└───────────────┬───────────────┘
                ▼
       backward_step_helper(backward_k)       ← 计算(掩盖上面的 P2P)
                │
┌─ pp_post_backward ────────────┐
│  isend 梯度 / irecv 下一梯度(异步)           │
└───────────────────────────────┘
```

### ④.3 流水线模拟图

![overlap_p2p_comm 前后对照：同步融合 P2P 时计算流每步干等一段通信(橙色斜纹)，改成异步 isend/irecv + 延迟 wait 后计算流连续、P2P 被下一次计算掩盖，壁钟从 11.0 t 降到 8.0 t。](assets/megatron_pp_p2p_overlap.svg)

*图 ④-1 `overlap_p2p_comm`：把 vp× 的 P2P 通信移出关键路径。**序列**(谁先谁后、四个回调挂在哪一步)逐条取自 `megatron/core/pipeline_parallel/schedules.py:1849` 的稳态循环，可核对；**时间比例**是作图参数 `t_f=t_b=4t`、`t_p2p=1.5t`，属于示意，本库没有该场景的实测带宽数据。省下的是**暴露的通信**，不是气泡——气泡率仍为 `(pp-1)/(m·vp)`，与 VPP 相同。*

### ④.4 开销分析

| 维度 | 取值 | 说明 |
|------|------|------|
| **气泡率** | `(pp-1)/(m·vp)`(与 VPP **相同**) | overlap 不改变气泡*结构*,只改变每步的*壁钟* |
| **壁钟时间** | `T ≈ (m·vp+pp-1)(t_f+t_b)`,不再叠加 `N_comm·t_p2p` | P2P 延迟被计算掩盖,移出关键路径 |
| **峰值激活显存** | VPP 基础上 **+** `fwd/bwd_recv_buffer`(`N-pp+1` 份 chunk 激活) | 预取缓冲的额外开销 |
| **通信量** | 与 VPP 相同(`≈2m·vp(pp-1)`) | 量不变,只是被隐藏 |

> 直觉:VPP 把气泡*比例*降到 `1/vp`,但通信 ×vp 可能把省下的时间又吐回去;`overlap_p2p_comm` 负责把这 `vp×` 通信藏到计算背后,让 VPP 的理论收益真正落地。二者通常**配套使用**。

### ④.5 适用场景及原因

- **VPP + 跨节点**:跨机 P2P 延迟高,不重叠就会吃掉 VPP 的收益 —— 此时 `overlap_p2p_comm` 几乎是必开项。
- **`overlap_p2p_comm_warmup_flush`**:`pp` 很大、warmup/cooldown 阶段占比可观时,把重叠也覆盖到这两段。
- **约束**:必须 `batch_p2p_comm = False`(二者互斥);需要额外的 recv buffer 显存。
- **不推荐**:`PP=1` 或纯机内小规模 —— 通信本就不在瓶颈上,徒增代码路径复杂度与缓冲显存。

---

## 7. 调度器⑤ — combined-1F1B(MoE All-to-All 重叠)`overlap_moe_expert_parallel_comm`

> 代码:`megatron/core/pipeline_parallel/combined_1f1b.py` + `megatron/core/models/common/model_chunk_schedule_plan.py`。通过 `overlap_moe_expert_parallel_comm = True` 开启。
>
> **它不是顶层独立调度,而是寄生在宿主调度内的"修饰器",且只有两种宿主形态**(`megatron/core/pipeline_parallel/schedules.py` 源码实证):
> - **PP=1 宿主**:`forward_backward_no_pipelining` 在 `megatron/core/pipeline_parallel/schedules.py:745` 调 `combined_1f1b_schedule_for_no_pipelining`。
> - **VPP 宿主**:`forward_backward_pipelining_with_interleaving` 经 `forward_backward_helper_wrapper` 在 `megatron/core/pipeline_parallel/schedules.py:1587` 调 `combined_1f1b_schedule_for_interleaved_pipelining`。
> - **没有"标准非交错 1F1B"宿主**:`forward_backward_pipelining_without_interleaving`(调度器②)全函数无 `overlap_moe_expert_parallel_comm` 分支。
>
> 因此:**combined-1F1B 有 VPP 版本,但没有非交错 1F1B 版本**。要画 4-stage 流水图,只能用它的 VPP 宿主形态(见 §⑤.3)。

### ⑤.1 动机与解决的问题

**问题**:MoE 模型用专家并行(EP),每个 MoE 层要做两次 **all-to-all**(token `dispatch` 把 token 发到对应专家所在卡、`combine` 收回结果)。消费 dispatched token 的专家计算与消费 combined output 的残差路径都对 A2A 结果有数据依赖；若不重排，等待就会暴露在关键路径。combined-1F1B 将 dispatch/combine 放到 communication stream，再用另一微批中不依赖该结果的计算掩盖这段窗口（`megatron/core/models/common/model_chunk_schedule_plan.py:330-334`、`:505-514`；`megatron/core/pipeline_parallel/utils.py:292-314`）。

**combined-1F1B 的动机**:在**层粒度**上,把 microbatch `i+1` 的**前向**和 microbatch `i` 的**反向**配对、共同调度。利用一个简单事实 —— **前向的 A2A 通信** 可以和 **反向的计算** 在 GPU 上并行,反之亦然。于是:
> 前向的 dispatch/combine A2A 被反向的 attention/MLP 计算掩盖;反向的 A2A 被前向计算掩盖。

它还配合 `delay_wgrad_compute`(`megatron/core/transformer/transformer_config.py:3712`):把反向拆成 **dgrad**(算输入梯度,在关键路径上)和 **wgrad**(算权重梯度,可延后)。`mlp_bwd_dw` 被排在 `dispatch_bwd` / `combine_fwd` 的 A2A 调度链里，主要用来填 A2A 通信缝隙；只有最后处理的 `attn_dw` 另被延后去配合 P2P（`megatron/core/models/common/model_chunk_schedule_plan.py:559-572`、`:593-596`）。这种拆分借用了 "zero-bubble" 系工作的核心思想，但本页 §9 会说明它并不是完整的 Zero-Bubble 调度。

### ⑤.2 源码与流程

**双 CUDA 流设计**:`set_streams`(`megatron/core/pipeline_parallel/combined_1f1b.py:67`)创建并保存 `comm_stream`；`get_comp_stream()` 则直接返回调用处的当前 CUDA stream（`megatron/core/pipeline_parallel/utils.py:350-367`）。MoE 的 `dispatch`/`combine` 节点跑在前者，`attn`/`mlp` 跑在当前计算流（`megatron/core/models/common/model_chunk_schedule_plan.py:330-337`）。

**三层调度结构**:

**第一层 —— 顶层调度**(`combined_1f1b_schedule_for_no_pipelining`,`megatron/core/pipeline_parallel/combined_1f1b.py:35`,以 PP=1、`m=4` 为例):
```
Phase 0: mb0 前向(单独跑,无重叠伙伴)
Phase 1: mb0 反向 ‖ mb1 前向     ← combined_forward_backward_step
Phase 2: mb1 反向 ‖ mb2 前向
Phase 3: mb2 反向 ‖ mb3 前向
Phase 4: mb3 反向(单独跑)
```

**第二层 —— `combined_forward_backward_step`**(`megatron/core/pipeline_parallel/combined_1f1b.py:281`):合并一个 F 和一个 B 的 pre/compute/post。前向不再直接产出张量,而是产出一个 **schedule plan**(`forward_step_func(..., return_schedule_plan=True)`),再交给:

**第三层 —— `TransformerModelChunkSchedulePlan.run`**(`megatron/core/models/common/model_chunk_schedule_plan.py:873`):模型块级的 1F1B,逐层把前向 layer `i` 和反向 layer `L-1-i` 配对(反向逆序,用 `pop_layer()` FILO 取):
```
Phase 0: p2p 同步 → 前向 preprocess → 反向 postprocess
Phase 1..L: 第 i 层  forward_layer[i]  ‖  backward_layer[L-1-i]
Phase L+1: send_forward_recv_backward → send_backward_recv_forward
Phase L+2: 第一层的 attn wgrad → 前向 postprocess → 反向 preprocess
```

**最内层 —— `TransformerLayerSchedulePlan.run`**(`megatron/core/models/common/model_chunk_schedule_plan.py:505`):单层内把 F/B 的子模块在两条流上交错,这是 A2A 真正被掩盖的地方:
```
comm_stream:  combine_bwd │ dispatch_fwd ─► dispatch_bwd │ combine_fwd
comp_stream:  attn_fwd    │ mlp_bwd ─► mlp_bwd_dw ─► mlp_fwd │ attn_bwd
              ↑ 前向 attn 计算掩盖反向 combine 的 A2A;反向 mlp 计算掩盖前向 dispatch 的 A2A
```
`mlp_bwd_dw`（`backward_dw`）就是 `delay_wgrad_compute` 拆出的权重梯度计算，它被刻意安排在两次 A2A 之间，用于填补空隙。最后一层的 `attn.backward_dw()` 还会通过 `is_last_layer_in_bwd` 延后执行，以便与 P2P 通信重叠。

**流程图**

```
combined_1f1b_schedule_for_{no_pipelining | interleaved_pipelining}
        │  (按 microbatch 配对 F[i+1] 与 B[i])
        ▼
combined_forward_backward_step
        │  forward → build_schedule_plan(f_schedule_plan)
        │  backward → 取出 b_schedule_plan
        ▼
TransformerModelChunkSchedulePlan.run(f_plan, b_plan)
        │  for i in layers:  配对 f_layer[i] 与 b_layer[L-1-i]
        ▼
TransformerLayerSchedulePlan.run(f_layer, b_layer)
        comp_stream ┊ attn_f  mlp_b  mlp_b_dw  mlp_f  attn_b
        comm_stream ┊ combine_b   dispatch_f  dispatch_b   combine_f
                      └── A2A 通信与对侧计算在 GPU 上真并行 ──┘
```

#### ⑤.2.1 主流程接入 + 正反向配对/分类是怎么保证的

上文讲了"分层执行"的结构,这里补"它**怎么挂进主流程**,以及**怎么保证 forward 与 backward 不混算、且配对正确**"(本小节描述的机制自 `dev@232c478d4` 起，行号已重核至基线 `71092579`)。

**(0) 主流程接入 —— 一个 `if` 接管 steady 段**。两个宿主函数开头加分支即可:`forward_backward_no_pipelining` 在 `megatron/core/pipeline_parallel/schedules.py:745`、`forward_backward_pipelining_with_interleaving` 在 `megatron/core/pipeline_parallel/schedules.py:1587`,命中 `config.overlap_moe_expert_parallel_comm and not forward_only` 就 delegate 给 `combined_1f1b_schedule_for_{no_pipelining|interleaved_pipelining}`。warmup/steady/cooldown 仍由 Layer-2 的 `get_pp_rank_microbatches` 生成,只是该模式**多调度一个 warmup microbatch**(`megatron/core/pipeline_parallel/schedules.py:915-919`,`num_warmup += 1`)—— 保证 steady 段里**永远有一个待反向的 microbatch** 能和当前 forward 配对。

四重机制保证"正反向分类执行":

**① plan 跟着 output tensor 走(配对的"账本",关键)**。combined 模式下模型不直接前向,而是 `forward_step_func(..., return_schedule_plan=True)` → `GPTModel.build_schedule_plan(...)`(`megatron/core/models/gpt/gpt_model.py:879`)产出 `f_schedule_plan`(`megatron/core/pipeline_parallel/combined_1f1b.py:390`)。前向收尾把它**挂到自己的输出上**:`output_tensor.schedule_plan = f_schedule_plan`(`megatron/core/pipeline_parallel/combined_1f1b.py:503`)。等这个 microbatch 轮到反向,从它**自己的** output tensor 取回:`b_schedule_plan = b_output_tensor[0].schedule_plan`(`megatron/core/pipeline_parallel/combined_1f1b.py:444`)。
→ f-plan 与 b-plan 来自**两个不同 microbatch、两个不同来源**,plan 随激活在流水线里传递 —— 这就是把"谁的前向、谁的反向"钉死的账本,天然不会错配。

**② 全程独立实参,不合并成单一 step**。`combined_forward_backward_step`(`megatron/core/pipeline_parallel/combined_1f1b.py:281`)里 forward 只碰 `f_model/f_schedule_plan`、backward 只碰 `b_model/b_output_tensor→b_schedule_plan`(retain_grad + loss 反向先单独做,`:450-457`),最后用**两个独立参数**进合并计算:`type(f_schedule_plan or b_schedule_plan).run(f_schedule_plan, b_schedule_plan, b_grad=b_grad, ...)`(`megatron/core/pipeline_parallel/combined_1f1b.py:471`)。

**③ 升序/降序层配对的正确性**。`TransformerModelChunkSchedulePlan.run`(`megatron/core/models/common/model_chunk_schedule_plan.py:873`)中:`f_layer = f_schedule_plan.get_layer(i)`(升序 0→N,`:937`)、`b_layer = b_schedule_plan.pop_layer()`(降序 N→0,FILO,`:938`)。**forward 第 i 层 配 backward 第 (N−1−i) 层**,正吻合 1F1B 中"做反向的 microbatch 领先,所以它的高层先算完"的依赖。`f_input`(前向激活)与 `b_grad`(反向梯度)是**两条独立数据流**各传各的(`:941`),绝不交叉;不配对的层再各跑 backward-only(`:953`)/ forward-only(`:964`)。

**④ 角色化节点 + 双流 + event 同步**(承 §⑤.2):一层拆成 `attn / moe_dispatch / mlp / moe_combine` 四个 `ScheduleNode`,A2A 节点绑 `comm_stream`、计算节点绑 `comp_stream`;跨流靠 CUDA event(`record_current_stream`/`wait_current_stream`,`megatron/core/models/common/model_chunk_schedule_plan.py:827-835`)保证依赖正确。于是 forward 层的 A2A 与配对 backward 层的计算在不同 stream 上真并行,而 autograd 正确性不受影响(forward 仍建图、backward 仍真反向)。

> 不变量:不支持 `checkpoint_activations_microbatch`(`megatron/core/pipeline_parallel/combined_1f1b.py:343` assert);VPP>1 + Megatron-FSDP 显式不支持;FSDP `optim_grads_params` 下因绕过 `TransformerLayer.forward` 的 hook,要给每层显式挂 reshard 回调(`megatron/core/pipeline_parallel/combined_1f1b.py:416-423`,见 [[16_megatron_distributed_optimizer_analysis]] / [[20_megatron_comm_overlap_analysis]] §3.3)。FP8 不是只支持 `delayed` recipe：delayed scaling 需要整个 combined pass 的 outer FP8 context，其他 recipe 则在逐层/逐节点循环内进入 context，以便细粒度决定 FP8 或 BF16（`megatron/core/pipeline_parallel/combined_1f1b.py:459-465`；`megatron/core/models/common/model_chunk_schedule_plan.py:549-583`）。只有 **full recompute + delayed-scaling FP8** 这项组合被 §⑤.4 所述配置守卫拒绝。

> [!update] combined-1F1B 的三处增量(MTP 排序 / 显存释放 / 死代码清理)——该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
>
> 自 `ee3f1ff` 以来,combined-1F1B 路径有三处源码变化,均**不改变**上文的双流重叠骨架:
>
> **1. MTP `mtp_post_process` 前向后移**(#4695,`megatron/core/models/common/model_chunk_schedule_plan.py:581-583`)。`TransformerLayerSchedulePlan.run` 里,带 MTP(多 token 预测)的层原先在 `moe_combine.forward` 之后**立刻**跑 `mtp_post_process.forward`;现在把它**挪到反向 `attn.backward` 之后**。docstring 明确(`:520-521`):「mtp_post_process_fwd 在 comp_stream 上排在 combine_fwd 之后,mtp_post_process_bwd 排在 combine_bwd 之前」。目的是让 MTP 的 output_layer/loss 计算与反向 attention 更好地错峰,避免它挤占 combine A2A 的掩盖窗口。非 MTP 模型的 `mtp_post_process` 是 `NoopScheduleNode`(`megatron/core/models/common/model_chunk_schedule_plan.py:375`),此改动无影响。
> （注:同段落里的 `ep_overlap_early_attn_memory_release` 配置决定 `attn.backward` 排在 dispatch 之后(早释放 attn 显存)还是 combine 之后,见 `:563-579`。)
>
> **2. loss 节点输入显存的及时释放**(#4908 / #4909,`megatron/core/pipeline_parallel/combined_1f1b.py:24`、`456-480`)。新增 `_release_tensor_storage`:在最后一个 stage,combined 反向跑完 loss 节点后,立刻 `loss_node._release_state()`(`:457`)并把 `loss_node.inputs` 的 CUDA 存储 `untyped_storage().resize_(0)` 抹零(先 `record_stream(current_stream)` 保证跨流安全)。这缓解了 §⑤.4 提到的「F 与 B 两个 microbatch 激活同时在世」带来的峰值,属于纯显存优化,不改语义。
>
> **3. 删除死代码 `manual_release_grads`**(#4511,`megatron/core/pipeline_parallel/utils.py`、`megatron/core/models/gpt/fine_grained_callables.py`)。`ScheduleNode` 上原有的 `manual_release_grads` / `delay_grads_release` 两个标志位及其释放分支**从未被置真**,已整段删除;dgrad/wgrad 的显存释放现在统一交给上面 (2) 的 `_release_tensor_storage` 与正常 GC。这不影响 `delay_wgrad_compute` 的 F/B/W 三段拆分语义(见 §⑤.1)。

### ⑤.3 流水线模拟图

combined-1F1B 是宿主调度的修饰器,**不改变 PP 级流水线的阶梯形状**。分三个粒度看。

**(a) PP 级 —— combined-1F1B 跑在 VPP 宿主上(`pp=4, vp=2, m=8, N=4`)**

阶梯与 §③.3 的纯 VPP **几乎相同**,唯一结构差异是 warmup **+1**(`megatron/core/pipeline_parallel/schedules.py:915-919`:多调度一个前向,确保每个 1F1B 步里 F 与 B 相互独立)。记号同 §③.3(`f`/`F`=前向 chunk0/1,`b`/`B`=反向 chunk0/1):

```
slot  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38
Dev0  f0 f1 f2 f3 F0 F1 F2 F3 f4 f5 f6 f7 B0 .. .. F4 B1 F5 B2 F6 B3 F7 b0 b1 b2 .. b3 .. B4 .. B5 .. B6 B7 b4 b5 b6 b7
Dev1  .. f0 f1 f2 f3 F0 F1 F2 F3 f4 f5 B0 f6 B1 f7 B2 F4 B3 F5 b0 F6 b1 F7 b2 .. b3 .. B4 .. B5 .. B6 B7 b4 b5 b6 b7 ..
Dev2  .. .. f0 f1 f2 f3 F0 F1 F2 F3 B0 f4 B1 f5 B2 f6 B3 f7 b0 F4 b1 F5 b2 F6 b3 F7 B4 .. B5 .. B6 B7 b4 b5 b6 b7 .. ..
Dev3  .. .. .. f0 f1 f2 f3 F0 F1 B0 F2 B1 F3 B2 f4 B3 f5 b0 f6 b1 f7 b2 F4 b3 F5 B4 F6 B5 F7 B6 B7 b4 b5 b6 b7 .. .. ..
```

要点:
- warmup `= 2(pp-r-1)+(vp-1)N+1`:rank0=11(纯 VPP 是 10),其余各 +1。阶梯整体右移一格,makespan 与空泡数和纯 VPP 一致(每设备 32 op + 6 空泡 = 38)。
- 稳态区**相邻的一对 `(F, B)` 属于同一个 `combined_forward_backward_step`**(如 Dev0 槽 16–17 的 `F4 B1`)—— 它把前向 mb 与反向 mb 融合成一个调度单元,内部 F/B 子模块在 comp/comm 双流上交错(下方 (c))。
- 关键:上图是 **compute-only 视角**。纯 VPP 若不重叠,每个 op 的真实壁钟 = 计算 + A2A 通信;combined-1F1B 把 A2A 藏进对侧计算,使真实壁钟逼近这张理想图。它换的是**单 op 的有效耗时**,不是阶梯形状或气泡数。

**(b) PP=1 宿主 —— 无阶梯,单设备内 F/B 重叠**

PP=1 无 stage 间流水(只有 1 个 stage,画不成 4-stage 图)。combined-1F1B 在单设备内把 mb `i` 的反向与 mb `i+1` 的前向配对(`megatron/core/pipeline_parallel/combined_1f1b.py:35`,以 m=4 为例):

```
Device 0 | [ f0 ]  [ b0 ‖ f1 ]  [ b1 ‖ f2 ]  [ b2 ‖ f3 ]  [ b3 ]
            phase0    phase1       phase2       phase3      phase4
            首个mb    ──── 融合步:F 与 B 的 A2A 互相掩盖 ────    末mb
```
首 microbatch 的前向、末 microbatch 的反向各自独跑(无配对伙伴),中间 m-1 步两两融合。

**(c) 单元内部 —— 层级双流重叠(combined-1F1B 的核心)**

每个融合步内,`TransformerLayerSchedulePlan.run`(`megatron/core/models/common/model_chunk_schedule_plan.py:505`)把 F、B 的子模块在两条 CUDA 流上交错:

```
未做跨微批重排(A2A 结果依赖暴露):
comp |attn_f|       |mlp_f|         |attn_b|       |mlp_b|
comm        |disp_f|       |comb_f|        |comb_b|      |disp_b|
            └等待┘         └等待┘          └等待┘        └等待┘
            ↑ 消费 A2A 结果的节点在依赖满足前不能继续

开 overlap_moe_expert_parallel_comm:
comp |attn_f|mlp_b|mlp_b_dw|mlp_f|attn_b|     ← 提供连续计算候选区间
comm |comb_b|  disp_f | disp_b | comb_f |     ← A2A 可被对侧无依赖计算部分掩盖
```

这张图只表示合法的排队顺序与并发窗口，不按真实 duration 缩放；若独立计算短于 A2A，通信头尾仍会暴露，不能从图形重合推导“全程掩盖”。

### ⑤.4 开销分析

| 维度 | 取值 / 影响 |
|------|------------|
| **PP 气泡** | **不改变**,继承宿主调度:PP=1 → 0;VPP → `(pp-1)/(m·vp)`。注意 VPP 形态下 warmup **+1**(`megatron/core/pipeline_parallel/schedules.py:918`) |
| **有效 `t_f` / `t_b`** | 下降幅度取决于有多少 exposed A2A 能被另一微批的独立计算覆盖；源码没有给固定百分比，必须由目标作业 trace 测量 |
| **峰值激活显存** | **升高**:F 与 B 两个 microbatch 的激活同时在世;额外持有 `schedule_plan` 对象;`delay_wgrad_compute` 需多存 wgrad 所需输入 |
| **A2A 通信量** | 不变,只是被隐藏;`high_priority_a2a_comm_stream`(`megatron/core/transformer/transformer_config.py:766`)可把通信流设为 CUDA 高优先级，资源旋钮详见 [[14_megatron_ep_analysis]] §8.4 |
| **PP 通信量** | 与宿主调度相同 |

**约束**(`megatron/core/transformer/transformer_config.py:3583-3740`):需 torch ≥ 2.6.0;仅支持 EP + `alltoall`/`flex` dispatcher;仅 bf16/fp16;`delay_wgrad_compute` 必须与本特性同开;VPP 形态下不支持 Megatron-FSDP。**full recompute 已支持但不是无条件支持**：必须关闭 `distribute_saved_activations`，MTP + uniform 时 `recompute_num_layers==1`，attention/hidden dropout 都为 0，并禁止 delayed-scaling FP8（`:3602-3647`）；非 full 路径要求 `recompute_method` / `recompute_num_layers` 为 `None`，且所有路径都禁止 `recompute_modules` 包含 `moe`（`:3648-3658`）。

### ⑤.5 适用场景及原因

- **MoE 大模型 + 专家并行(EP)**:这是本特性的唯一目标场景 —— 非 MoE 模型没有 dispatch/combine A2A,无可掩盖。
- **A2A 占比高**:专家数多、EP 度大、跨机 EP 时 A2A 开销大,收益最显著。
- **显存有余量**:换取吞吐的代价是更高的激活峰值,显存吃紧时需配合减小 microbatch 或层数。
- **可与 PP=1 或 VPP 叠加**:小规模 MoE 用 `PP=1 + combined_1f1b`;大规模 MoE 用 `VPP + combined_1f1b + overlap_p2p_comm` 全家桶。
- **不推荐**:稠密(非 MoE)模型;需要 full recomputation 的极限显存场景;torch 版本过低。

### ⑤.6 与 PP P2P 的资源竞争

MoE 层里,PP 的 P2P(激活/梯度收发)与 EP 的 All-to-All(dispatch/combine)可能同时发生,争抢同一条 NVLink/IB 带宽:

| 场景 | PP 通信 | EP 通信 | 竞争分析 |
|------|---------|---------|---------|
| 标准 1F1B(无 combined) | P2P send/recv 激活值 | A2A dispatch/combine | 直接竞争带宽 |
| combined-1F1B | P2P 与计算交错 | A2A 被反向计算掩盖 | 理想情况下 A2A 不占用带宽窗口,PP P2P 独享 |
| 首/末 stage | 单向 P2P(缺 backward 或 forward 一侧) | 全量 A2A | 首 stage 无 backward P2P,可更专注 EP 通信 |

缓解手段:PP P2P 走独立 CUDA stream(`overlap_p2p_comm`,调度器④),与 EP A2A 的 `comm_stream`(§⑤.2)分离;combined-1F1B 本身通过 layer 级交错,让 EP A2A 尽量发生在计算密集阶段而非 PP P2P 密集阶段。对非 MoE 层(纯 attention),PP P2P 是唯一的跨 rank 通信,不存在此竞争。

---

## 8. 目录周边设施:混合 CP 动态调度与多模块流水线

`pipeline_parallel/` 目录里,除 5 个调度器 + P2P/进程组基础设施(§1)外,还有两块相对独立的周边机制,借调度器②的 `is_multimodule` 分支 / `backward_step_multimodule` 挂钩接入主流程。

### 8.1 变长序列的混合 CP 动态调度(`megatron/core/pipeline_parallel/hybrid_cp_schedule.py`)

**动机**:固定 CP 度(`13_megatron_cp_analysis.md`)假设所有样本切同一个 `cp`,但**变长序列训练**(SFT、文档级语料、多分辨率多模态)样本长度差异巨大 —— 短样本用大 CP 是浪费,长样本用小 CP 又放不下,且 attention `O(S²)` 负载在 DP×CP 组间严重不均。

`BalancedCPScheduler`(`megatron/core/pipeline_parallel/hybrid_cp_schedule.py:14`)按样本长度动态决定 CP 度,并把样本打包成各组工作量均衡的批:

```python
def gpus_needed(self, seq_len):              # 该样本需要几张卡做 CP
    return max(1, 2 ** ceil(log2(seq_len / self.max_seq_len_per_rank)))

def get_total_workload(self, seq_length, cp_size):
    return (seq_length ** 2) / cp_size        # attention O(S²) 负载估计,除以 CP 摊分
```

`make_buckets_equal`(`:55`)/ `next_hdp_group`(`:104`)用 `seq²/cp` 做工作量估计,贪心打包成若干"hdp"(hybrid data parallel)组,使各组总工作量大致相等(`delta` 控制 5% 松弛);`hybrid_context_parallel_forward_backward`(`:477`)是配套的前向反向调度入口。

这是 CP 在**变长数据**下的负载均衡机制,与其他轴的均衡手段并列:

| 并行轴 | 负载不均衡的来源 | 均衡手段 |
|--------|----------------|---------|
| PP | microbatch 填充/排空 | VPP(本文调度器③) |
| CP | 因果掩码 | zigzag 切分(`13_megatron_cp_analysis.md` 转引的理论页 §4) |
| **CP(变长)** | **样本长度差异** | **本节:动态 CP 度 + 工作量均衡分桶** |
| EP | 路由不均 | aux_loss / 容量因子(`14_megatron_ep_analysis.md` §7) |

适用:变长序列训练 —— SFT、长文档预训练、多分辨率多模态。固定长度的标准预训练用不到。

> **与 `29_megatron_packed_dataset_dynamic_cp_analysis.md` 的关系**:`megatron/core/datasets/data_schedule.py` 的 `DefaultDynamicCPScheduler`(打包调度器的 `is_dynamic_cp=True` 子类,用 `dcp_*` 函数)才是动态 CP 真正的**集成入口** —— 序列打包与动态 CP 在那里被缝进同一条 `run()` 九步流水线。本节的 `BalancedCPScheduler` 是同一套均衡逻辑的**类形态兄弟**(独立实现,算法一致),两者不是"谁包含谁",是并行存在的两套实现。完整集成流程见 29 号页;13 号页讲固定 CP 的通用机制。

### 8.2 多模块/多模态流水线(`megatron/core/pipeline_parallel/bridge_communicator.py` + `megatron/core/pipeline_parallel/multimodule_communicator.py`)

**动机**:标准 `P2PCommunicator`(§1.4)假设上下游 PP stage 共享同一并行网格,TP/DP/CP 一致、激活形状对得上。但**多模态模型**的视觉编码器、LLM 主干、生成头是不同子模型,各自可能用完全不同并行配置(如编码器 TP=2/PP=1,LLM TP=8/PP=4),甚至张量维数不同(视觉编码器常出 2D `[b·s,h]`,LLM 要 3D `[s,b,h]`)。

`BridgeCommunicator`(`megatron/core/pipeline_parallel/bridge_communicator.py:39`)连接一对 `HyperCommGrid`(源→目标网格,见 `17_megatron_parallelism_orchestration_analysis.md` §5):`build_comm_map` 算出源网格哪些 rank 该发给目标网格哪些 rank,fan-in/fan-out 处理两侧 batch/并行度不同(用缓存的 broadcast 进程组);`dim_mapping`/`tensor_ndim` 处理 2D/3D 差异;对外暴露与 `P2PCommunicator` 同名接口(`send_forward`/`recv_forward`/`send_forward_recv_backward`/...),上层调度无感切换。当前限制:CP 暂不支持(两侧须 CP=1)。

`MultiModulePipelineCommunicator`(`megatron/core/pipeline_parallel/multimodule_communicator.py:110`)把多个子模块组织成一张 DAG(如 `image_encoder/audio_encoder → llm → generator`),为每条边建一个 `BridgeCommunicator`,张量以 `Dict[str, Tensor]` 按模块名组织传递 —— 使调度器②的非交错 1F1B 可以无感驱动一条跨异构子模型的流水线。

> [!update] 跨网格 P2P 改走专用进程组(#5234)——该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
>
> `ee3f1ff` 时跨网格 `dist.send/recv`/`P2POp` 不带 `group=` 参数,隐式走全局 `WORLD` 组,与其它集合通信共用 NCCL 通信器,有串行化/标签冲突风险。#5234 为每条桥接边新建专用进程组 `bridge_pg`(`megatron/core/pipeline_parallel/bridge_communicator.py:167-168`,取源/目标网格 TP leader 并集排序后 `dist.new_group`),类级缓存 `_bridge_pg_cache` 避免重复建组;此后所有桥接 `send_forward`/`recv_forward`/`send_backward`/`recv_backward` 及融合 `P2POp` 均显式传 `group=self.bridge_pg`(12 处站点)。意义:把跨网格 leader↔leader 的 P2P 从全局组隔离到独立通信器,避免与子网格内部 TP/PP/DP 集合通信争用 NCCL 资源。配套:`HyperCommGrid` 本身也在 #5148 获得 named views,为异构子模型提供几何基础(详见 [[17_megatron_parallelism_orchestration_analysis]] §5③)。

**适用场景**:VLM、音频-语言、encoder-decoder 等多模态/异构模型;纯单模型(GPT)走标准 `P2PCommunicator`,用不到这两个类。

---

## 9. 概念补充:DualPipe / DualPipeV / Zero-Bubble(本仓库**未实现**)

> 全仓库 case-insensitive 搜索 `dualpipe`、`zero bubble`、`zbh1`、`zbv` 均无命中。本节为**纯原理介绍**,不对应本仓库源码,供横向理解。

### 9.1 Zero-Bubble(ZB-H1 / ZB-H2 / ZBV)

- **来源**:Qi et al., *Zero Bubble Pipeline Parallelism*, ICLR 2024(中科大 / Sea AI Lab)。
- **核心思想**:把反向 `B` 拆成两半 —— **`B_dgrad`(算输入梯度,有数据依赖,必须在关键路径上)** 和 **`B_wgrad`(算权重梯度,只依赖本层激活,可任意延后)**。`W` 没有跨 stage 依赖,于是可以拿它去**填满气泡**。
  - **ZB-H1**:在 1F1B 基础上重排 `W`,气泡缩小到约 `1/3`。
  - **ZB-H2**:配合优化器侧的处理,理论气泡 → **0**(代价是峰值显存比 1F1B 高)。
  - **ZBV(V-shape)**:把 chunk 排成 V 形(类似 DualPipeV),在与 1F1B 同等显存下逼近零气泡。
- **与 Megatron 的关系**:Megatron 当前 `dev` 分支**没有** ZB 调度,但调度器⑤的 `delay_wgrad_compute`(F/B/W 三段拆分)正是 ZB 那套 `B = B_dgrad + B_wgrad` 拆分思想的局部应用 —— 只是用于 A2A 重叠,而非系统性消除气泡。

### 9.2 DualPipe / DualPipeV

- **来源**:DeepSeek-V3 技术报告(2024)。
- **核心思想**:**双向流水** —— 同时从流水线两端注入 microbatch,一批从 stage 0→pp-1,另一批从 pp-1→0,两个方向的前向/反向在中间相遇并对冲掩盖。配合把计算拆成 attention / all-to-all / MLP / PP-comm 等细粒度块两两重叠,几乎完全隐藏通信。
- **DualPipeV**:DualPipe 的 V 形单向变体,用一份模型副本(而非 DualPipe 的两份)实现近似效果,显存友好。
- **代价**:DualPipe 需在每个设备上**保留两份模型参数**(双向各一套),`参数显存 ≈ 2×`;实现复杂度高;最适合 DeepSeek-V3 那种超大 MoE + 跨节点 EP 的场景。
- **与 Megatron 的关系**:截至本 commit,Megatron-LM 主线**未合入** DualPipe/DualPipeV;若需可关注上游 PR 或 DeepSeek 开源的 `DualPipe` 仓库。

### 9.3 三类思路定位对比

| 方案 | 气泡 | 峰值显存 | 是否在本仓库 |
|------|------|---------|------------|
| 标准 1F1B | `(pp-1)/m` | `pp·A` | ✅ 调度器② |
| 交错 VPP | `(pp-1)/(m·vp)` | `≈(1+1/vp)pp·A` | ✅ 调度器③ |
| ZB-H1 | `≈(pp-1)/(3m)` | `≈pp·A` | ❌ 仅 `delay_wgrad` 拆分思想被借用 |
| ZB-H2 / ZBV | `≈0` | 高于 1F1B | ❌ |
| DualPipe | `≈0` | `≈2×` 参数 | ❌ |
| DualPipeV | `≈0` | 接近 1F1B | ❌ |

---

## 10. 约束

每个调度器的 ".4 开销分析" 回答的是"这张表贵在哪"。这一节回答另一个问题:**这套 PP 实现在什么前提下才成立,哪些组合被明令禁止,哪些只是默认没开**。每条都能落到一个 `file:line`。

| # | 前提 / 不变量 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | **`deallocate_pipeline_outputs` 默认关闭** | 字段默认 `False`(`megatron/core/model_parallel_config.py:400-403`);`deallocate_output_tensor` 首行即 `if (out is None) or (not deallocate_pipeline_outputs): return`(`megatron/core/pipeline_parallel/schedules.py:178-179`) | §1.4 那条"把 `.data` 换成标量"的省显存手段不显式打开就完全不生效 —— 按默认配置算 PP 显存会算错 |
| 2 | 被 pseudo-free 的张量不能是别人的视图 | `assert out._base is None, "counter-productive to free a view of another tensor."`(`megatron/core/pipeline_parallel/schedules.py:195`) | 视图释放不掉底层存储,直接 assert 失败 |
| 3 | `custom_backward` 要求 output 已被缩成标量 | `assert output.numel() == 1`(`megatron/core/pipeline_parallel/schedules.py:208`) | §2① 的两个函数互为前提,不能只开一半 |
| 4 | 反向钉在 PyTorch 私有入口上 | `Variable._execution_engine.run_backward`(`megatron/core/pipeline_parallel/schedules.py:220-228`),注释指向 `torch/csrc/autograd/python_engine.cpp`(`:219`) | 这是 §2① 那笔交易的账单:PyTorch 动这个内部 API,PP 反向就要跟着改 |
| 5 | `overlap_p2p_comm` 与 `batch_p2p_comm` 互斥 | 配置期互相声明(`megatron/core/model_parallel_config.py:380-388`)+ 运行期 `ValueError`(`megatron/core/pipeline_parallel/schedules.py:1180-1181`) | 开 overlap 必须同时把 `batch_p2p_comm` 关掉 |
| 6 | **非交错 1F1B 根本不支持 P2P 重叠** | `ValueError("Non-interleaved pipeline parallelism does not support overlapping p2p communication")`(`megatron/core/pipeline_parallel/schedules.py:2313-2315`) | 调度器④ 只能寄生在 VPP 上;不开 VPP 就没有这条路 |
| 7 | `overlap_p2p_comm_warmup_flush` 要求 `overlap_p2p_comm=True` 且 `batch_p2p_comm=False` | `megatron/core/model_parallel_config.py:603-607` | 三个 P2P 旋钮是一条链,不能各调各的 |
| 8 | `microbatch_group_size_per_vp_stage` 须落在 `[pp, m]`,且 `m % N` 要么为 0 要么 ≥ `pp` | `ValueError`(`megatron/core/pipeline_parallel/schedules.py:1255-1262`)/ `RuntimeError`(`:1266-1274`) | 后者的理由源码原话:「it introduces dependency bubbles in the pipeline and reduces throughput」 |
| 9 | 无流水线调度不接受模型分块,也不支持 `adjust_tensor_shapes_fn` | `assert`(`megatron/core/pipeline_parallel/schedules.py:717`、`:720-722`、`:724-726`) | PP=1 路径与 VPP 的 chunk 列表语义不通用 |
| 10 | 交错调度要求 `model` 与 `data_iterator` 都是 list,且同样不支持 `adjust_tensor_shapes_fn` | `assert`(`megatron/core/pipeline_parallel/schedules.py:1168-1172`、`:1173-1175`) | 传单个 module 进 VPP 直接 assert |
| 11 | `use_ring_exchange_p2p` 需要自编译的 PyTorch | docstring 明写「Requires custom built torch with torch.distributed.ring_exchange.」(`megatron/core/model_parallel_config.py:395-398`);调用点 `megatron/core/pipeline_parallel/p2p_communication.py:226-227`、`:367-373` | 默认 `False`;打开而 torch 没有该符号,会在通信时才炸 |
| 12 | **loss 缩放假设 CP 在流水线内是静态的** | 源码原话「TODO: This path assumes static CP across outstanding pipeline microbatches. Hybrid/dynamic CP currently requires per-token loss and no PP」(`megatron/core/pipeline_parallel/schedules.py:387-390`) | 动态 CP 与 PP 目前不能一起开(另见 §8.1) |
| 13 | 跨网格桥接不支持 CP | 源/目标网格的 CP 必须为 1,`assert`(`megatron/core/pipeline_parallel/bridge_communicator.py:105-110`) | §8.2 的多模态流水线在 CP>1 下建不起来 |

**代价**:§2① 换来的显存收益,代价是 PP 的反向永远绑在 PyTorch 的私有 C++ 入口与"输出必须先被 pseudo-free"这条约定上(上表 2–4 条)。§2② 换来的通信隐藏,代价是三个 P2P 旋钮从此必须成组设置(第 5、7 条),且只对 VPP 生效(第 6 条)。

**故意不做的事**:**本仓库没有 Zero-Bubble / DualPipe 调度**——§9 已记录全仓库 case-insensitive 搜索 `dualpipe`/`zero bubble`/`zbh1`/`zbv` 均无命中;调度器⑤ 的 `delay_wgrad_compute` 只借用了 `B = B_dgrad + B_wgrad` 的拆分思想,用途是掩盖 MoE A2A,而**不是**系统性消除气泡。另一处刻意留白是**动态 CP 与 PP 不同时支持**(上表第 12 条),源码把它写成 TODO 而不是绕开。

---

## 11. 横向对比与选型决策

### 11.1 总对比表

| 调度器 | 气泡率 | 峰值激活显存 | PP 通信量 | 关键收益 | 主要代价 |
|--------|--------|-------------|----------|---------|---------|
| ① 无流水线 | 0 | `1·A`(全模型) | 0 | 最简单 | 模型态不分摊,装不下大模型 |
| ② 标准 1F1B | `(pp-1)/m` | `pp·A` | `2m(pp-1)` | 模型态 `/pp`,显存最省 | 气泡随 `pp/m` 上升 |
| ③ 交错 VPP | `(pp-1)/(m·vp)` | `≈(1+1/vp)pp·A` | `≈2mv(pp-1)`(×vp) | 气泡 `/vp` | 通信 ×vp,显存略增 |
| ④ P2P-overlap | `(pp-1)/(m·vp)` | VPP + recv buffer | 同 VPP(被隐藏) | 通信移出关键路径 | 额外缓冲;须 `batch_p2p_comm=False` |
| ⑤ combined-1F1B | 继承宿主 | 宿主 + 1 个 microbatch + plan | 同宿主 | 掩盖可被另一微批独立计算覆盖的 MoE A2A | 激活峰值升高;仅限 MoE+EP |

### 11.2 选型决策树

```
模型 + 激活单卡(或单 TP 组)能放下?
├─ 是 ──► 调度器① 无流水线(PP=1),用 DP 扩 batch
└─ 否 ──► 必须切 PP
          │
          ├─ 是 MoE 模型且用了 EP?
          │   └─ 是 ──► 叠加 调度器⑤ combined-1F1B(掩盖 A2A)
          │
          ├─ m / pp 是否足够大(气泡 (pp-1)/m 可接受,如 ≥8)?
          │   ├─ 是 ──► 调度器② 标准 1F1B(通信最省、显存最省)
          │   └─ 否 ──► 调度器③ 交错 VPP(气泡 /vp)
          │              │
          │              └─ 跨节点 / 通信吃紧?
          │                  └─ 是 ──► 叠加 调度器④ overlap_p2p_comm
          │
          └─ 典型大模型工业配置:
             TP(机内 NVLink) × PP(跨机 IB,VPP+overlap_p2p_comm) × DP(最外层)
             若为 MoE:再叠加 EP + combined-1F1B
```

### 11.3 一句话总结

- **①→②**:用相同气泡,把模型态显存除以 `pp`,并让 PP 能廉价跨节点扩展。
- **②→③**:用 `vp×` 通信 + 略增显存,把气泡率除以 `vp`。
- **③→④**:不改气泡,把那 `vp×` 通信藏到计算背后,让 VPP 的理论收益真正落地。
- **⑤**:正交于上面四个 —— 专门为 MoE 把 all-to-all 通信用 F/B 互相掩盖掉。
- **DualPipe / Zero-Bubble**:把气泡推向 0 的下一代思路,本仓库尚未实现,但 `delay_wgrad_compute` 已借用其 `B=B_dgrad+B_wgrad` 拆分。

### 11.4 关键配置速查表

| 配置项 | 推荐值 | 对应调度器/机制 |
|--------|--------|-----------------|
| `pipeline_model_parallel_size` | 按模型层数/显存确定,常 4~16 | 调度器②③④⑤ 共用 |
| `virtual_pipeline_model_parallel_size` | 2~4(大模型气泡高时) | 调度器③ VPP |
| `num_microbatches` | `≥4×pp`,保证气泡率可接受 | 调度器②③ 气泡公式(§②.4/§③.4) |
| `microbatch_group_size_per_vp_stage` | 默认 `=pp`(深度优先) | 调度器③,受 §③.2④约束 |
| `overlap_p2p_comm` | `True`(跨节点必开) | 调度器④,须配合下一行 |
| `batch_p2p_comm` | `False`(若开 overlap) | 与 `overlap_p2p_comm` 互斥,见 §1.4 |
| `deallocate_pipeline_outputs` | `True` | §1.4 显存优化原语 |
| `pipeline_dtype` | 与模型一致 | `pp>1` 时必填 |
| `defer_embedding_wgrad_compute` | 大 vocab 时 `True` | §1.5 |
| `overlap_moe_expert_parallel_comm` | `True`(MoE 模型) | 调度器⑤ combined-1F1B |
| `delay_wgrad_compute` | 与上行同开 | 将 wgrad 延后为可填充的计算节点；单独开启会被配置校验拒绝（`megatron/core/transformer/transformer_config.py:3712-3721`） |
| `ep_overlap_early_attn_memory_release` | 显存峰值过高时试验 | 将 attention backward 提前以早释放激活，代价是部分 MoE 通信重新暴露（`megatron/core/model_parallel_config.py:355-366`）；要求上行的 combined overlap（`megatron/core/transformer/transformer_config.py:3736-3740`） |
| `num_microbatches_with_partial_activation_checkpoints` | 按显存压力设窗口 | §②.2 Partial Checkpointing |
| `--offload-modules` / fine-grained offloading | 显存墙严重时开 | 见 §1.4 与 `22_megatron_memory_optimization_analysis.md` §3.3 |

---

## 12. 发展趋势

> [!note] 推断:锚点是基线 `71092579` 下的源码事实(TODO、新增早退分支、形参校验)与 §1.2 / §8.2 已逐条核过的 PR 编号,方向判断由本页承担,不是源码的自陈计划。

**一、顶层分发正在从"按 `pp`/`vp` 三选一"变成"按传进来的进程组集合选"。** §1.2 已核过的 `[!contradiction]` 记录了 #5373 在 `get_forward_backward_func` 最前面加的早退分支(`megatron/core/pipeline_parallel/schedules.py:149-150`):多模块 / 跨网格路径不再经过 `pp_size`/`vp_size` 判断。同一方向上,三个调度函数都已接受 `p2p_communicator` / `pg_collection` 形参并逐个 `hasattr` 校验(`:713-714`、`:1156-1158`、`:2368-2369`),只传一半时直接 `ValueError("Provide both p2p_communicator and pg_collection, or neither")`(`:2380`)。**由此可推断**:§1.2 那张"三选一"的树图还会继续长分支,判据从并行度数值转向"调用方传了哪种进程组集合";再读 PP 入口时不要假设它只看 `pp_size`/`vp_size`。

**二、跨网格 / 多模态流水线是当前扩张最快的一块,但边界仍多。** §8.2 的 `[!update]` 已核过 #5234(每条桥接边独立进程组 `bridge_pg`)与 #5148(`HyperCommGrid` named views)。源码里还挂着三条未决:`megatron/core/pipeline_parallel/bridge_communicator.py:105` 的「TODO (ykarnati, pthombre) - CP support will be added in follow up PR.」(对应 §10 第 13 条那个 CP=1 断言)、`megatron/core/pipeline_parallel/multimodule_communicator.py:257` 的「TODO: ykarnati - improve this logic.」,以及 `megatron/core/pipeline_parallel/schedules.py:2257` 的「TODO: make this more robust, including flexible VPP layout」(hyper connections 下的 P2P 形状推导)。**由此可推断**:多模块流水线目前只在"CP=1 + 非交错 1F1B"这一格里成立 —— §1.2 那条早退分支返回的正是非交错调度;要在其上叠 VPP 或 CP,须先回源码确认这几条 TODO 是否已关闭。

**三、"一条流水线里 CP 是静态的"这个前提正被源码自己写成待办。** `forward_step` 的 loss 缩放分支里写着「TODO: This path assumes static CP across outstanding pipeline microbatches. Hybrid/dynamic CP currently requires per-token loss and no PP; if that changes, carry the scale per autograd context instead of via a process-wide scaler hook.」(`megatron/core/pipeline_parallel/schedules.py:387-390`);§8.1 的混合 CP 调度器里同时挂着「TODO[pmannan]: PP not yet supported. Add PP scheduling.」(`megatron/core/pipeline_parallel/hybrid_cp_schedule.py:190`)。**由此可推断**:动态 CP × PP 是一条正在被打通的边,而且打通方式源码已经预告了 —— 把缩放系数挂到每个 autograd context 上,替掉进程级的 scaler hook;在此之前 §10 第 12 条那条互斥必须当硬约束读。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准,后续版本可能漂移；2026-08-28 由 `ee3f1ff` 重定基线。2026-08-01 由 `20_megatron_pp_parallelism_analysis.md`(740 行)、`26_megatron_pp_supplements_analysis.md`(229 行)吸收增量后合并而成 —— spec §3.4(kb-reorg P6 阶段遗漏,P7 收尾期补执行),详见 `wiki/changelog.md`。*

## Related Pages

- [[12_megatron_tp_analysis]] — 说明每个 pipeline stage 内部的 TP/SP 执行与通信。
- [[13_megatron_cp_analysis]] — 说明 CP attention 通信怎样进入 microbatch 时间线。
- [[14_megatron_ep_analysis]] — 提供 MoE A2A 与 delayed wgrad 的轴内机制。
- [[17_megatron_parallelism_orchestration_analysis]] — 展开 PP/VPP rank、stage 与 ProcessGroup 构造。
- [[20_megatron_comm_overlap_analysis]] — 把 PP schedule 与 DP/TP/CP/EP overlap 合并到同一时间线。
- [[29_megatron_packed_dataset_dynamic_cp_analysis]] — 查看 packed microbatch 与 Dynamic CP 的调度入口。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] — 返回全部 35 篇内容页的主题索引。
