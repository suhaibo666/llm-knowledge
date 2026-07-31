# 通信优化与计算-通信掩盖总览 —— 对称内存 · Async-TP 微流水 · MinimalAsyncEP · 重叠矩阵(源码级)

> **代码基准**:torchtitan `main` @ `61c010fcb` · PyTorch `2.9.1+cpu`(个别更新版 API 已标注)
> **最后更新**:2026-06-16 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本篇聚焦**通信**侧性能手段;**算力/显存**侧(低精度、算子融合、编译、ChunkedCE)见 [[torchtitan_compute_memory_optimizations_analysis]]。
> 行号约定:torchtitan 以 `torchtitan/torchtitan/` 为根;PyTorch 2.9.1 以 `[pt]` 前缀。

---

## 0. 全景:三层通信优化

```
① 编排重叠(已覆盖)   多 stream + 预取/反向逆序:FSDP AG/RS/AR、CP ring P2P、PP P2P、EP a2a ── 见 02–07
② 算子级融合重叠      Async-TP 微流水:all-gather+matmul / matmul+reduce-scatter 拆块流水(编译期 FX pass)
③ 通信基座优化        对称内存:NVLink 零拷贝 / multimem 一发集合,降 all-gather/reduce-scatter 延迟
                     compute-comm reorder:编译期把集合通信往前挪,塞进计算影子
旁支:MinimalAsyncEP(symm-mem + Triton 的 EP dispatch);full_dtensor(整网 DTensor,单发分片)
```

定位:**①已在 02–07 逐维度讲透**(本篇 §1 给汇总矩阵 + 指针);②③是本篇新增,机制都在 **PyTorch**,torchtitan 只是开关:Async-TP = 一个 inductor flag(`tensor_parallel.py:113`),FSDP 对称内存 = 两个 FSDP2 方法(`fsdp.py:50-51`)。

---

## 1. 跨维度计算-通信掩盖矩阵(汇总)

| 维度 | 被掩盖的通信 | 藏在谁的影子里 | 机制 | 详见 |
|---|---|---|---|---|
| **FSDP** 前向 | all-gather(N+1) | compute(N) | 独立 AG 流 + CPU 跑在 GPU 前 + 预取 | [[torchtitan_fsdp_analysis]] §4、[[torchtitan_fsdp_prefetch_overlap_memory_analysis]] §3 |
| **FSDP** 前向 | copy-in(N+1) | all-gather(N) | 独立 copy_in 流 + AllGatherState 双缓冲 | [[torchtitan_fsdp_prefetch_overlap_memory_analysis]] §5.3 |
| **FSDP** 反向 | reduce-scatter(N) | compute(N-1) | 独立 RS 流 + 反向逆序预取 | [[torchtitan_fsdp_analysis]] §6 |
| **HSDP** 反向 | all-reduce(N) | compute(N-1) + RS(N-1) | 第 5 条 all_reduce 流(不同网络资源) | [[torchtitan_hsdp_backward_overlap_analysis]] §5 |
| **TP** | all-gather / reduce-scatter | 同层 matmul | `async_op` 机会主义;**Async-TP 真重叠**(§3) | [[torchtitan_tp_analysis]] §5 |
| **CP** | 下一步 K/V 的 ring P2P | 当前步 SDPA | functional collective + ring 轮转 | [[torchtitan_cp_analysis]] §6 |
| **PP** | stage 间激活/梯度 P2P | 相邻 microbatch 的 F/B | action-based runtime + isend/irecv | [[torchtitan_pp_analysis]] §5 |
| **EP** | combine all-to-all | `shared_experts(x)` | AsyncCollectiveTensor 延迟 wait / DeepEP `sync_combine` | [[torchtitan_ep_analysis]] §5 |

**共性**:全靠"独立 stream / functional collective + 把通信发起提前到计算之前 + 用 event/ACT 界定 wait 时机"。本篇 §3–§5 补的是**比上表更激进**的两类:把通信**拆进算子**(Async-TP)、把集合通信换成**更快的基座**(对称内存),以及**编译期重排**(reorder pass)。

---

## 2. Async Tensor Parallel:把 all-gather / reduce-scatter 拆进 matmul

[[torchtitan_tp_analysis]] §5.2 已点出 Async-TP;这里补**机制全貌**。入口 `maybe_enable_async_tp`(`distributed/tensor_parallel.py:102-115`)只做两件事:

```python
if not parallelism.enable_async_tensor_parallel: return        # :105
if not (compile_config.enable and "model" in compile_config.components):
    raise RuntimeError("Async TP requires 'model' in --compile.components and --compile.enable")  # :108-111
torch._inductor.config._micro_pipeline_tp = True               # :113
```

- **必须 compile**:它只是给一个 inductor FX pass 上膛,真正的拆分在编译时发生。
- pass 本体在 PyTorch:`[pt]_inductor/fx_passes/post_grad.py:168-169` 读 flag → `micro_pipeline_tp_pass`(`[pt]_inductor/fx_passes/micro_pipeline_tp.py:1052-1079`)。
- **拆什么**(`micro_pipeline_tp.py`):
  - `all_gather + matmul` → `torch.ops.symm_mem.fused_all_gather_matmul`(fp8 走 `fused_all_gather_scaled_matmul`)(`:584-605`)。docstring(`:611-625`):`A=all_gather(A_shard); C_i=A@B_i` collapse 成一个融合算子。
  - `matmul + reduce_scatter` → `torch.ops.symm_mem.fused_matmul_reduce_scatter`(`:807`)。
- **怎么重叠**:融合算子内部**按 rank 把 gather/scatter 切块**,把每块的 P2P 拷贝与上一块已到达数据的 matmul 流水起来(`[pt]_symmetric_memory/__init__.py:148-218` 的两 stream 策略)。于是阻塞式"集合通信 + GEMM"变成**通信藏在计算下**——这才是 "async" 的含义。
- **被掩盖的通信**:TP/SP 的 `all_gather_into_tensor`(权重/激活)与 `reduce_scatter_tensor`(输出)。
- 与 `reorder_for_compute_comm_overlap` 协同:若简单重叠已能掩盖,pass **故意排除**这些集合,避免拆分开销(`:1056-1068`)。

---

## 3. 对称内存 Symmetric Memory:FSDP 与 Async-TP 共用的基座

对称内存 = 进程组内每 rank 一块 **P2P 可寻址**的工作区(`enable_symm_mem_for_group` / `get_symm_mem_workspace`,`[pt]_symmetric_memory/__init__.py:24-123`),让节点内 NVLink 直接 P2P 拷贝,并提供 multicast/multimem 一发 all-gather 快路径(`:775-813`)。它是 Async-TP 与 FSDP 对称内存共同的底座。

### 3.1 FSDP 对称内存

`enable_fsdp_symm_mem`(`distributed/fsdp.py:44-51`)对每个 FSDP 模块翻两个开关:

```python
module.set_force_sum_reduction_for_comms(True)   # :50
module.set_symm_mem_for_comm()                   # :51
```

- 配置 `parallelism.enable_fsdp_symm_mem=False`(`configs.py:124-128`);`__post_init__` 限 **Hopper(SM9.0)+**(`configs.py:233-243`,NVLink 对称内存支持;ROCm 豁免)。在 `apply_fsdp_to_decoder` 顶层 `fully_shard` 后调用(`fsdp.py:282-283`)。
- `set_force_sum_reduction_for_comms(True)`(本机 torch 有:`[pt]_fully_shard.py:546-565`):把规约强制为纯 `SUM`(必要时拆 pre/post 缩放),"because NCCL currently supports zero-copy transfers only for this kind of collectives"——这是让 reduce-scatter 走**零拷贝/对称内存友好路径**的前提。
- ⚠️ **版本差**:`set_symm_mem_for_comm()` 在**本机 torch 2.9.1 不存在**(只有 `set_custom_all_gather`/`set_custom_reduce_scatter`/`set_force_sum_reduction_for_comms`),torchtitan 在此追踪更新版 torch;在 2.9.1 上会 `AttributeError`。按命名与同族 `set_custom_*` 推断,它把 FSDP2 的 AG/RS 换成 SymmetricMemory(一发/multimem NVLink)实现——但本机无源码可引,故只标"2.9.1 未提供"。

### 3.2 Async-TP 也基于对称内存

Async-TP 的融合算子要求该集合的进程组**已注册对称内存**,否则跳过融合:`micro_pipeline_tp.py:632-646`(`is_symm_mem_enabled_for_group` 不满足直接 return),reduce-scatter 同(`:859-862`)。核心 torchtitan 依赖 torch 自动注册;只有 `experiments/graph_trainer/passes.py:82-102` 仍显式 `enable_symm_mem_for_group`(注释:上游自动注册尚未落地)。

> 结论:**Async-TP 与 FSDP 对称内存共享 SymmetricMemory 基座**,都要 Hopper 级 NVLink;Async-TP 还要求模型区域被 compile。

### 3.3 编译期 compute-comm 重排

`reorder_for_compute_comm_overlap`(`#3020`)是另一类 inductor 手段:把集合通信节点在图里**往前挪**,塞进前面的计算影子。它与 Async-TP 微流水互补(§2 里 pass 会避开已能被它掩盖的集合),共同构成编译期的通信掩盖。

> [!note] 补充(2026-07-31 · 由 [[30_comm_compute_overlap_analysis]] 收缩合并) `experiments/graph_trainer/fsdp_passes.py` 是这类重排 pass 针对 **FSDP** 的专用实现:在 Inductor 图级别把 FSDP 的 AG/RS 重新排布,并额外开一条 NCCL process group 实现与主计算流的并发——与 §3.2 提到的 `experiments/graph_trainer/passes.py` 显式 symm-mem 注册同属 graph_trainer 实验分支,尚未进入 torchtitan 出货路径。

---

## 4. MinimalAsyncEP:symm-mem + Triton 的 EP dispatch(新后端)

[[torchtitan_ep_analysis]] 讲了 DeepEP/HybridEP;`#3561` 新增的 **MinimalAsyncEP** 是第三个 EP token-dispatch 后端,纯 PyTorch(symmetric memory + Triton),无 `deep_ep` 外部依赖。

- **做什么**:用对称内存的 all-to-allv 把路由 token **直接写进对端 rank 的接收缓冲、且已是最终 expert-major 布局**,跳过标准路径的"rank-major 物化 + `_permute()`"(`minimal_async_ep/api.py:529-531`)。
- **8 个自写 Triton kernel**(`minimal_async_ep/kernels.py`):count 交换、dispatch/combine 元数据、核心的 permute+scatter+P2P-put(`_copy_rows_to_peer_ptrs_kernel`)、top-k combine 规约(fp32 累加)及其反向。这是**全仓唯一的自写 Triton**。
- ⚠️ **重要纠正:MinimalAsyncEP 不做 comm/compute 重叠**。代码明说:`api.py:291-297` "this backend does **not** provide microbatch communication overlap";`token_dispatcher.py:823-828` "async combine overlap are intentionally out of scope"。它的性能 lever 不是流重叠,而是:① 对称内存一发 P2P put + **融合 barrier**(一个 kernel 同时 signal/poll 所有 peer,替掉 `2·(ep-1)` 次 per-peer signal,`api.py:336-348`);② dispatch/combine/backward 都是 `@torch.library.custom_op`,**靠 CUDA graph / compile 去掉 CPU launch 开销**。"Async" 指异步对称内存 put + barrier,**不是**通信与计算重叠。
- **对比**:DeepEP 的 combine **是**异步、延迟 `sync_combine()` 与 `shared_experts` 重叠(`deepep.py:191-285`、`moe.py:482-496`);MinimalAsyncEP 同步、不重叠。HybridEP 把不透明 handle 穿过编译图。
- **选择与约束**:设 `moe.experts.token_dispatcher=MinimalAsyncEPTokenDispatcher.Config`;约束严格(`api.py:98-136`):EP=数据并行组、**TP=CP=PP=1**、`num_experts%ep==0`、**不支持 full_dtensor**、**要求全量 AC**(`activation-checkpoint:full`)。

---

## 5. full_dtensor / spmd_types:整网 DTensor 的 SPMD 后端

`parallelism.spmd_backend: Literal["default","full_dtensor","spmd_types"]="default"`(`configs.py:142-149`)。[[torchtitan_parallel_dims_analysis]] §8 提过 `full_dtensor`,这里补定位与动机。

- **default(默认/legacy)**:把 `dp_shard×cp` 折成一个 `fsdp` 轴,**只有 TP/EP 表达成 DTensor**,DP/CP "out of band"(在 DTensor 程序外用 FSDP/CP 命令式处理)(`parallel_dims.py:322-329/536-539`)。
- **full_dtensor**:**整张 SPMD mesh(dp_replicate/dp_shard/cp/tp)都表达成 DTensor**,参数/buffer/输入全是 DTensor(`full_dtensor.py:7-16`)。dense mesh 不折叠 dp_shard/cp(`parallel_dims.py:286-296`);`fully_shard` 拿 `DataParallelMeshDims` 自己在初始化时折叠 dp_shard+cp(`fsdp.py:123-130`)。输入也经 `parallelize_inputs` 包成 DTensor(`full_dtensor.py:128-167`)。
- **spmd_types**:用外部 `spmd_types` 库 + `SpmdLayout`(`parallel_dims.py:55-115`,临时类,带 `TODO` 待上游 API)做**静态 SPMD 类型检查**(`debug.spmd_typechecking`)。
- **动机(性能/正确性)**:把整网分片**一次性、声明式**表达成单一 DTensor/SPMD 程序,便于 compile / autoparallel,取代 DP/CP 命令式 out-of-band 应用;`fully_shard` 从显式 `DataParallelMeshDims` 做单发分片折叠。
- **状态**:**实验性、过渡中**,`default` 才是出货路径(多处 `TODO ... after full_dtensor backend is removed` / `once migration to spmd_types is complete`)。MinimalAsyncEP 明确拒绝 full_dtensor(§4)。

> 相关:graph_trainer 实验的 **SimpleFSDP**(把整个数据并行表达成可追踪的 DTensor 集合通信、交编译器分桶重叠)是同一「整网 DTensor + 编译器」方向的近亲,详见专题 [[torchtitan_simple_fsdp_analysis]]。

---

## 6. 三类手段的硬件/编译前提速查

| 手段 | 必须 compile | 硬件 | torch 版本 | 配置开关 |
|---|---|---|---|---|
| Async-TP 微流水 | **是** | Hopper+ NVLink(对称内存) | 2.9.1 可用 | `enable_async_tensor_parallel` |
| FSDP 对称内存 | 否 | Hopper+(SM9.0)NVLink | `set_symm_mem_for_comm` 需 >2.9.1 | `enable_fsdp_symm_mem` |
| compute-comm reorder | 是(inductor) | 通用 | 2.9.1 可用 | inductor `reorder_for_compute_comm_overlap` |
| MinimalAsyncEP | 否(但要 AC full) | CUDA 对称内存后端 | 2.9.1 可用 | `token_dispatcher=MinimalAsyncEP...` |
| full_dtensor | 否 | 通用 | 2.9.1 可用 | `spmd_backend="full_dtensor"`(实验) |

---

## 7. 源码复核小结

| 断言 | 位置 | 结果 |
|---|---|---|
| Async-TP = 一个 inductor flag,必须 compile | `distributed/tensor_parallel.py:108-113` | OK |
| 微流水拆 AG+matmul / matmul+RS 成 symm_mem 融合算子 | `[pt]_inductor/fx_passes/micro_pipeline_tp.py:584-807` | OK |
| 融合算子内部按 rank 分块、P2P 与 matmul 流水 | `[pt]_symmetric_memory/__init__.py:148-218` | OK |
| FSDP 对称内存 = force_sum_reduction + symm_mem_for_comm | `distributed/fsdp.py:50-51` | OK |
| `set_force_sum_reduction_for_comms` 走 NCCL 零拷贝 | `[pt]_fully_shard.py:546-565` | OK |
| **`set_symm_mem_for_comm` 在 torch 2.9.1 不存在(追踪更新版)** | grep `[pt]_fully_shard` 无匹配 | OK |
| Async-TP 也要 symm-mem 注册,否则跳过融合 | `[pt]...micro_pipeline_tp.py:632-646` | OK |
| **MinimalAsyncEP 不做 comm/compute 重叠** | `minimal_async_ep/api.py:291-297` | OK |
| MinimalAsyncEP 直写 E-major,跳过 rank-major+permute | `api.py:529-531` | OK |
| MinimalAsyncEP 约束 TP=CP=PP=1、要 AC full、拒 full_dtensor | `api.py:98-136` | OK |
| full_dtensor 整网 DTensor,fully_shard 拿 DataParallelMeshDims | `full_dtensor.py:7-16`、`fsdp.py:123-130` | OK |
| default 仅 TP/EP 入 band,DP/CP out-of-band | `parallel_dims.py:536-550` | OK |

---

## 8. 小结

- **跨维度重叠(§1)**:02–07 已逐维度覆盖;本质都是"独立 stream/functional collective + 提前发起 + event/ACT 定 wait"。
- **Async-TP(§2)**:一个 inductor flag(必须 compile)把 `all-gather+matmul` / `matmul+reduce-scatter` 拆成 `symm_mem.fused_*` 融合算子,内部按 rank 分块、P2P 与 GEMM 流水——TP/SP 集合通信真正藏进 matmul。
- **对称内存(§3)**:NVLink 零拷贝 / multimem 一发集合的基座,Async-TP 与 FSDP 对称内存共用;FSDP 侧 `force_sum_reduction` + `set_symm_mem_for_comm`(后者需 >2.9.1),限 Hopper+。
- **MinimalAsyncEP(§4)**:新 EP 后端,symm-mem 直写 E-major + 自写 Triton + 融合 barrier;**不重叠通信计算**,lever 是 CUDA graph/compile 去 launch 开销——与 DeepEP(combine 重叠 shared_experts)路线不同。
- **full_dtensor(§5)**:把整网表达成单一 DTensor/SPMD 程序的实验后端,为 compile/autoparallel 铺路;默认仍是 DP/CP out-of-band 的 legacy 路径。
- 算力/显存侧手段见 [[torchtitan_compute_memory_optimizations_analysis]]。

---

## Related Pages

- [[torchtitan_compute_memory_optimizations_analysis]] —— 算力/显存侧性能手段,与本篇互补
- [[torchtitan_tp_analysis]] —— TP 通信与 Async-TP 入口
- [[torchtitan_ep_analysis]] —— EP token all-to-all(DeepEP/HybridEP),MinimalAsyncEP 的对照
- [[torchtitan_hsdp_backward_overlap_analysis]] —— HSDP 反向双流掩盖
- [[torchtitan_simple_fsdp_analysis]] —— 编译器友好 FSDP(整网 DTensor 近亲)
- [[21_async_collective_tensor_deepdive]] —— AsyncCollectiveTensor 源码追踪(异步通信底座)
- [[30_comm_compute_overlap_analysis]] —— 跨框架计算-通信掩盖对比
- [[torchtitan/index]] —— torchtitan 多维并行知识地图
