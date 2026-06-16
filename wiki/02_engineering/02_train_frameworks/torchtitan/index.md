# torchtitan 多维并行体系 — 知识地图

> **代码基准**:torchtitan `main` @ `61c010fcb` · PyTorch `2.9.1`(FSDP2/DTensor/pipelining 内核)
> **最后更新**:2026-06-16(新增 HSDP 反向、计算/显存优化、通信优化、SimpleFSDP 共 4 篇)
> 一套 12 篇 torchtitan 多维混合并行(DP/TP/CP/EP/PP)+ 训练机制(AC)+ 性能手段(低精度/算子融合/编译/对称内存/Async-TP)源码级分析。以 `fully_shard` 为标杆粒度——**参数怎么切、切完怎么取回、哪些通信能掩盖、异步怎么实现**——逐维度展开。torchtitan 是薄封装,真实机制深入到 PyTorch FSDP2 / DTensor / pipelining 内核。

---

## 设计哲学:一组 GPU,多重视图

torchtitan 的并行体系不是"5 套独立机制",而是**一套统一的 `DeviceMesh` 抽象** + **PyTorch 原生分布式原语**。核心一句话:

> **6 个并行度只是把同一组物理 GPU,按不同的逻辑方式重新分组,每个维度对应 `DeviceMesh` 的一个轴。**

所有逻辑集中在 `torchtitan/distributed/`,每个维度一个文件。这直接体现 torchtitan 的第一原则——**PyTorch-native**:核心并行代码不依赖任何非 PyTorch 库。

## 文档系列(7 篇)

| 页面 | 维度 | 核心机制 |
|------|------|---------|
| [[torchtitan_parallel_dims_analysis]] | **基座** | `ParallelDims` + 三张 `DeviceMesh`(dataloading / dense / sparse);维度约束、`fake` backend |
| [[torchtitan_fsdp_analysis]] | **DP** | FSDP2 `fully_shard`:逐参数切分、all-gather 预取、五条 CUDA stream 异步编排(**标杆篇**) |
| [[torchtitan_tp_analysis]] | **TP** | DTensor `redistribute` 通信选择、列并行→行并行配对、Sequence Parallel、Async TP 微流水、Loss Parallel |
| [[torchtitan_cp_analysis]] | **CP** | 序列维切分 + 负载均衡、Ring Attention K/V 环形轮转、`AsyncCollectiveTensor` 重叠 |
| [[torchtitan_pp_analysis]] | **PP** | 模型按层切 stage、P2P send/recv、调度气泡(GPipe/1F1B/Interleaved/ZBV/DualPipeV)、Zero Bubble |
| [[torchtitan_ep_analysis]] | **EP** | 专家权重 `Shard(0)`、token all-to-all dispatch/combine、`AsyncCollectiveTensor` 延迟 wait、DeepEP |

### 深挖伴篇(3 篇,带 SVG→PNG 机制图)

| 页面 | 主题 | 核心机制 |
|------|------|---------|
| [[torchtitan_fsdp_prefetch_overlap_memory_analysis]] | **DP 深挖** | 预取/掩盖时序对照图、copy-in 三步与唯一跨流同步点 `wait_event`、flat 双缓冲(ping-pong)、完整参数 ≤2 份的代码时序证明 |
| [[torchtitan_hsdp_backward_overlap_analysis]] | **HSDP 反向深挖** | reduce-scatter 与 all-reduce 双流掩盖、`foreach_reduce` 跨流编排、host 端算子下发顺序、AR∥RS 并发的正确性、reduce 路径 fp32 暂存与显存峰值(3 张机制图) |
| [[torchtitan_ac_analysis]] | **AC** | `checkpoint_wrapper` 接口链路、票据机制(发票/重算绑票/兑票)、SAC 缓存回放 + attention 端到端走查、显存预估三法、横跨 autograd×dispatch 两核心 |

### 性能手段与编译器路线(3 篇,2026-06-16 新增)

并行之外的吞吐/显存手段,横跨所有维度:

| 页面 | 主题 | 核心机制 |
|------|------|---------|
| [[torchtitan_compute_memory_optimizations_analysis]] | **算力/显存** | Float8(rowwise)/MXFP8 低精度 GEMM、FusedSwiGLU/MoE grouped GEMM/FusedQKV 融合算子、逐 block 编译即融合、融合 Adam、ChunkedCELoss、CPU offload。**纠正:rowwise 无 fp8 all-gather;核心循环不在 microbatch 间延迟 FSDP 规约** |
| [[torchtitan_comm_optimizations_overlap_analysis]] | **通信** | 跨维度计算-通信掩盖矩阵、Async-TP 微流水(`symm_mem.fused_all_gather_matmul`)、对称内存、MinimalAsyncEP(**不重叠通信计算**)、full_dtensor SPMD 后端 |
| [[torchtitan_simple_fsdp_analysis]] | **编译器 FSDP** | SimpleFSDP(graph_trainer 实验):分片表达成 DTensor `redistribute` 进图、通信由编译器 pass 分桶重叠;与 FSDP2 eager 多流编排逐项对比 |

## 六个维度速览

| 维度 | 切什么 | PyTorch 原语 | 通信模式 | 解决的问题 |
|------|--------|-------------|---------|-----------|
| **DP/FSDP** | 参数·梯度·优化器状态 | `fully_shard`(FSDP2) | all-gather + reduce-scatter | 显存放不下完整训练状态 |
| **TP** | 单个算子(matmul) | DTensor + `ShardingConfig` | 每层 all-reduce / all-gather | 单层太大、matmul 太慢 |
| **CP** | 序列维 activation | DTensor `Shard` + ring attention | K/V all-gather / ring P2P | 长序列 attention 放不下 |
| **PP** | 模型深度(层) | `torch.distributed.pipelining` | stage 间 P2P send/recv | 层数极多、节点间带宽差 |
| **EP** | MoE 专家 | DTensor `Shard(0)` + token all-to-all | token all-to-all | MoE 专家数过多 |

## 三张 DeviceMesh

`ParallelDims.build_mesh()` 把 1D `world` mesh `_unflatten` 成**三张相互重叠的逻辑 mesh**(详见 [[torchtitan_parallel_dims_analysis]]):

```
              init_device_mesh(world_size,)  →  1D "world" mesh
                            │
      ┌─────────────────────┼──────────────────────┐
      ▼                     ▼                      ▼
 dataloading_mesh      dense_mesh             sparse_mesh
 [pp,batch,cp,tp]      [pp,dp_replicate,      [pp,dp_replicate,
      │                 fsdp,tp]               efsdp,ep]
      │ flatten          fsdp=dp_shard×cp      efsdp=dp_shard×cp×tp/ep
      ▼ (batch,cp)
 loss_mesh

 数据加载、loss 规约    稠密层参数切分          MoE 专家参数切分
```

约束:`dp_replicate × dp_shard × cp × tp × pp == world_size`,**`ep` 例外**——EP 是对 `dp_shard×cp×tp` 子网格的重新切分。三张 mesh 都覆盖全部 GPU,只是切法不同。

## 并行施加管线

`parallelize_llama` / `parallelize_deepseekv3` 由外到内施加:

```
模型(meta device)
  │ [若 PP] pipeline_llm 先切 stage chunks,每个 chunk 独立走下面 ↓
  ▼
1. apply_cp_to_forward   CP:包裹 attention forward
2. model.parallelize()   TP:按 ShardingConfig 切权重 + 包裹 forward
3. apply_moe_ep_tp()     EP:专家权重 Shard(0) + 配置 token_dispatcher(仅 MoE)
4. apply_ac()            激活重计算(见 [[torchtitan_ac_analysis]])
5. apply_compile()       torch.compile(逐 TransformerBlock)
6. apply_fsdp()          DP:fully_shard,最外层 SPMD 包装
```

嵌套结构(外→内):`PP 切分` → 每 stage:`FSDP` → `compile` → `AC` → `TP/EP` → `CP`。

## 组合建议(网络层级映射)

多维并行的核心是把**通信量大的维度放在快的网络层级**:

```
节点内 NVLink(最快)  → TP(每层 all-reduce)、EP all-to-all
节点内/少量节点        → FSDP 分片(all-gather 流量大)
跨节点(较慢)         → HSDP 复制、CP
最慢的互联            → PP(stage 间只传激活,最省带宽)
```

## Cross-Domain Links

- [[tp_analysis]] / [[cp_analysis]] / [[ep_analysis]] / [[pp_schedulers_analysis]] / [[ddp_optimizer_analysis]] —— Megatron-LM 同维度源码级分析,可与本系列对照(CUDA/Megatron 生态 vs PyTorch-native)
- [[parallelism_orchestration_analysis]] —— Megatron-LM 进程组编排,与 `ParallelDims` 同类
- [[async_collective_tensor_deep_dive]] —— `AsyncCollectiveTensor` 源码追踪,是 TP/CP/EP 异步通信的共同底座
- [[comm_compute_overlap_analysis]] —— 计算通信掩盖对比(含 torchtitan 源码)
- [[comm_compute_fusion_guide]] —— 通算融合路线图
- [[distributed_optimizer_deep_dive]] —— FSDP2 / ZeRO / MindSpeed 三方对比
- [[llm_parallelism_analysis]] —— LLM 并行通信依赖 DAG

## Related Pages

- [[torchtitan_parallel_dims_analysis]] · [[torchtitan_fsdp_analysis]] · [[torchtitan_tp_analysis]] · [[torchtitan_cp_analysis]] · [[torchtitan_pp_analysis]] · [[torchtitan_ep_analysis]]
- [[torchtitan_fsdp_prefetch_overlap_memory_analysis]] · [[torchtitan_hsdp_backward_overlap_analysis]] · [[torchtitan_ac_analysis]] —— 深挖伴篇(FSDP 预取/掩盖/显存、HSDP 反向、激活重计算)
- [[torchtitan_compute_memory_optimizations_analysis]] · [[torchtitan_comm_optimizations_overlap_analysis]] · [[torchtitan_simple_fsdp_analysis]] —— 性能手段与编译器路线(低精度/融合/编译、对称内存/Async-TP、SimpleFSDP)
- [[megatron-lm/index]] —— Megatron-LM 知识地图(姊妹训练框架)
- [[02_engineering/02_train_frameworks/index]] —— 训练框架目录索引
