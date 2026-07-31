# 计算通信掩盖机制跨框架对比矩阵

*Megatron-LM combined_1f1b · torchtitan ZBV/DualPipe · MindSpeed fb-overlap/DualPipeV · TP/DP/PP/EP 掩盖维度全览*

> 本页是 **Megatron-LM / torchtitan / MindSpeed** 三个分布式训练框架计算通信掩盖机制的**横向对比矩阵页**：逐维度机制细节（含源码 file:line）已下沉到各框架的权威机制页——[[20_megatron_comm_overlap_analysis]]、[[24_torchtitan_comm_optimizations_overlap_analysis]]、[[11_mindspeed_comm_overlap_analysis]]——本页只保留三页都没有的**合成视图**：跨框架维度矩阵、combined_1f1b vs ZBV/DualPipeV 的架构差异分析、可达性矩阵。查具体机制请跳转子页；查"哪个框架用什么手段掩盖哪种通信"，本页即终点。
>
> **与"通算融合"的边界**：本页讲的是**掩盖**——通信与计算独立发生、靠调度/多 stream 重叠隐藏延迟（源码层仍是两次算子调用）；把通信编进**同一个 kernel** 的**融合**手段（WaveEP、DeepEP、MC2）见 [[31_comm_compute_fusion_guide]]。

**目录**

-   掩盖机制的概念分层
-   跨框架掩盖维度矩阵
-   combined_1f1b 与 ZBV/DualPipeV：两种"打通"哲学
-   Sub-Layer 级掩盖为何无法被泛化
-   框架差异总结

## 一 掩盖机制的概念分层

计算通信掩盖（Comm-Compute Overlap）在分布式训练中存在于多个粒度层次。理解这些层次，是理解三个框架差异的前提。

![图 1：掩盖机制的三层金字塔](assets/comm_compute_overlap_analysis_fig1.png)

*图 1：掩盖机制的三层金字塔*

三个框架实现这些层次的**载体**不同，[[11_mindspeed_comm_overlap_analysis]] §1 给出的三分类同样适用于跨框架理解：**融合**（把通信编进单一 kernel，如 Megatron/MindSpeed 的 MC2、DeepEP `FusedDispatch`——参见 [[31_comm_compute_fusion_guide]]）、**软件流水**（chunk 切分 + 双 stream 异步 handle，如 Megatron TP Bulk/Pipelined、MindSpeed CoC、torchtitan `AsyncCollectiveTensor`）、**换调度**（PP/EP 调度层面重排，如 combined_1f1b、ZBV/DualPipeV、MindSpeed fb-overlap+DualPipeV）。下面按维度逐一对照。

## 二 跨框架掩盖维度矩阵

| 维度 | Megatron-LM | torchtitan | MindSpeed |
| --- | --- | --- | --- |
| **TP** | Bulk（`ub_bulk_wgrad`/`ub_bulk_dgrad`：dgrad/wgrad 与 AG/RS 无依赖侧并行）+ Pipelined（chunk 流水，`tp_comm_overlap`，依赖 TE User Buffer）→ [[20_megatron_comm_overlap_analysis]] §二 | Async-TP 微流水（inductor pass，拆 `all-gather+matmul`/`matmul+reduce-scatter` 为 `symm_mem.fused_*` 融合算子，需 `enable_async_tensor_parallel`+compile+Hopper 对称内存）→ [[24_torchtitan_comm_optimizations_overlap_analysis]] §2 | MC2（单一融合大算子，`npu_all_gather_base_mm`/`npu_mm_reduce_scatter_base`）或 CoC（PyTorch 层 chunk 软流水/lcal 融合核，粒度可调）→ [[11_mindspeed_comm_overlap_analysis]] §2-3 |
| **DP（梯度/参数）** | Bucket 异步 ReduceScatter（`overlap_grad_reduce`）+ Prefetch AllGather（`overlap_param_gather`），forward pre-hook 驱动 → [[20_megatron_comm_overlap_analysis]] §三 | FSDP 独立 AG/RS stream + 反向逆序预取；HSDP 反向另开 all-reduce stream → [[11_torchtitan_fsdp_analysis]] §4/§6、[[21_torchtitan_hsdp_backward_overlap_analysis]] §5 | 梯度/参数 AG-RS 沿用 Megatron distributed optimizer（未在本域单列）；MindSpeed 自身新增 **async-log-allreduce**——掩盖的不是梯度而是 **loss 日志 all-reduce**，延迟到取值才 wait → [[11_mindspeed_comm_overlap_analysis]] §6 |
| **PP** | 异步 P2P（`overlap_p2p_comm`，even/odd rank 分组实现真并发；⚠️ 默认关闭）→ [[20_megatron_comm_overlap_analysis]] §四 | Action-based runtime：`RECV` 早发起、用前才 wait，`SEND` 发起不等 → [[14_torchtitan_pp_analysis]] §8 | optimize-p2p-comm（关 `batch_p2p_comm` 拆独立 isend/irecv）/ optimize-send-recv-comm（专用 P2P stream + 进程组，细粒度重叠）→ [[11_mindspeed_comm_overlap_analysis]] §5.3 |
| **EP（同 microbatch 内）** | Shared Expert 独立 stream 状态机（`moe_shared_expert_overlap`，与 EP+PP 交叉掩盖是**独立**开关）→ [[20_megatron_comm_overlap_analysis]] §5.8 | `AsyncCollectiveTensor` 延迟 wait：`combine()` 把 `shared_experts(x)` 塞进 a2a enqueue 到结果首用之间的窗口 → [[15_torchtitan_ep_analysis]] §4-5 | alltoall-overlap/allgather-overlap：异步 handle 藏进同微批 `op_dx`/`op_dw` GroupedMatMul → [[11_mindspeed_comm_overlap_analysis]] §4.1-4.2 |
| **EP（跨 microbatch）+ PP 打通** | **combined_1f1b**：Layer 拆 5 子节点，双 stream 交错 + delay-wgrad，硬编码模型内部结构 → 本页§三、[[20_megatron_comm_overlap_analysis]] §5.2-5.4/5.8 | ✗ 无——PP 调度（ZBV/DualPipe）不感知 EP 内部结构，EP 掩盖停留在 mb 内层次 | **fb-overlap**：微批 i 前向 ∥ 微批 i-1 反向 + `WeightGradStore` 延迟 dw；DualPipeV 的 overlap 稳态段直接调用 fb-overlap 函数，二者深度协同 → 本页§三、[[11_mindspeed_comm_overlap_analysis]] §4.3/§5.1 |
| **PP 气泡消除** | 传统 1F1B + P2P overlap（无 ZB/DualPipe 内置支持） | **ZBV/DualPipeV**：I/W 拆分（按 autograd 目标而非模型结构）+ `OVERLAP_F_B`（F/B 计算本身重叠发起）→ 本页§三、[[14_torchtitan_pp_analysis]] §7 | **DualPipeV**（双向切半，气泡占比 $O(P)/O(m)$）+ **RiPipe**（气泡内做重计算，近乎零代价）→ [[11_mindspeed_comm_overlap_analysis]] §5.1-5.2 |
| **A2A 专用融合内核** | DeepEP/HybridEP：两级通信去冗余（跨节点走 RDMA、节点内走 NVLink），"加速通信"而非"掩盖通信" → [[20_megatron_comm_overlap_analysis]] §5.6 | MinimalAsyncEP：symm-mem + 自写 Triton，**不做**通算重叠（明确声明），lever 是融合 barrier + CUDA graph/compile 去 launch 开销 → [[24_torchtitan_comm_optimizations_overlap_analysis]] §4 | alltoall-MC2：`npu_alltoallv_gmm`/`npu_gmm_alltoallv` 把 a2a-v 与专家 GEMM 编进单 kernel，与所有软流水 MoE-overlap 互斥 → [[11_mindspeed_comm_overlap_analysis]] §4.4 |

> **Megatron EP 行补注**：`overlap_dispatch_backward_with_experts_wgrad`（`moe_layer.py:441`、`519`，专用 `_delayed_wgrad_stream` 把专家 wgrad 与 dispatch 反向 A2A 重叠）见 [[14_megatron_ep_analysis]] §5.3。它与"EP（跨 microbatch）+ PP 打通"行的 combined_1f1b delay-wgrad（`delay_wgrad_compute`）是**两个相互独立、互斥**的机制（`transformer_config.py:2713`）：前者**不耦合 PP**，纯 EP 内 wgrad-vs-dispatch-A2A 重叠；后者是 combined_1f1b 内**跨 microbatch**的 wgrad 延迟，二选一。

## 三 combined_1f1b 与 ZBV/DualPipeV：两种"打通"哲学

Megatron combined_1f1b 与 torchtitan ZBV/DualPipeV、MindSpeed fb-overlap+DualPipeV 都在解决"跨 microbatch 打通掩盖"，但路线不同——完整源码级机制见各自权威页（[[20_megatron_comm_overlap_analysis]] §5、[[14_torchtitan_pp_analysis]] §4/§7、[[11_mindspeed_comm_overlap_analysis]] §4.3/§5.1），本节只讲架构差异。

### 3.1 Megatron combined_1f1b：硬编码模型结构换 sub-layer 级掩盖

`fine_grained_callables.py` 把每个 TransformerLayer 手工拆解为 5 个可调度子节点（attn_fwd/mlp_fwd/mlp_bwd/mlp_bwd_dw/attn_bwd），`TransformerLayerSchedulePlan.run()` 在两个 CUDA stream 上交错一个 f_layer 与一个 b_layer，`TransformerModelChunkSchedulePlan.run()` 再按镜像位置配对 forward/backward 层。这要求框架**知道** TransformerLayer 内部执行顺序（先 attn → 再 dispatch → 再 mlp → 再 combine）——即硬编码模型结构，换来的是 **sub-layer 级**的跨 microbatch EP+PP 打通（§四的可达性矩阵详述其代价）。

### 3.2 ZBV/DualPipeV：autograd 级拆分换模型无感知

PyTorch 的 backward 拆分完全在 **autograd 图级别**：`stage_backward_input` 只算到 dInput（`inputs=` 限定、`retain_graph=True`），`stage_backward_weight` 从缓存的中间梯度算到 dWeight。这不是按模型结构（attn vs mlp vs moe_dispatch）拆分，而是按**autograd 的计算目标**（dInput vs dWeight）拆分——一个 stage 可能包含多层 Transformer，但 I 统一是所有层的 dInput，W 统一是所有层的 dWeight。DualPipeV 在此基础上加 `OVERLAP_F_B`：把一个 stage 的 forward 与另一个 stage 的 backward 打包，计算本身重叠发起；但**这不是真正的并发**——两个子 action 在单 rank 上仍串行执行，"overlap" 来自跨 rank 的 P2P 异步通信（rank 0 做 B(stage1) 时 rank 1 正在做 F(stage0)，通信可重叠）。这条路线是**stage 级**的跨 microbatch 打通，换来的是模型零感知（PyTorch 上游实现，模型无需改造）。

### 3.3 MindSpeed fb-overlap+DualPipeV：layer 级跨微批 + 调度协同复用

MindSpeed 的 fb-overlap 让一个微批的前向层与另一个微批的反向层同时执行，`WeightGradStore` 把 dw 解耦延后填气泡，共享专家再叠一层独立 stream；DualPipeV 的 overlap 稳态段**直接调用** fb-overlap 的函数（`dualpipev_schedules.py:921,1098,1204`），二者不是并列关系而是调度深度协同。掩盖粒度是**layer 级**（前向层 ∥ 反向层），没有 Megatron 式的 layer 内 5 子节点拆解，因此不下探到 sub-layer；但因为跨微批互填 + dw 延迟，稳态每层暴露可压到 $\max(T_{\text{comp}},T_{a2a})$，是 MindSpeed 四条 EP 掩盖路线里暴露最低的一档。

> **三者定位**：Megatron combined_1f1b（sub-layer 级，模型强感知）＞ MindSpeed fb-overlap+DualPipeV（layer 级，patch 式模型感知）＞ torchtitan ZBV/DualPipeV（stage 级，模型零感知）——掩盖粒度与模型感知程度成正比，这不是巧合：更细的掩盖粒度必然需要框架知道更多模型内部结构（§四展开这一根本原因）。

## 四 Sub-Layer 级掩盖为何无法被泛化

### 4.1 根本原因：计算-通信边界是模型架构决定的

要在不同 micro-batch 之间交错 sub-layer 操作，框架**必须知道**：

1.  哪些是**纯计算**（attention GEMM、MLP GEMM、layernorm）
2.  哪些是**纯通信**（MoE dispatch AlltoAll、combine AlltoAll、TP AG/RS）
3.  这些操作在**单个 layer 内的执行顺序**（先 attn → 再 dispatch → 再 mlp → 再 combine）
4.  哪些中间 tensor 需要跨 stream 传递（residual、shared_expert_output）

这些信息**不属于任何通用的 IR**——既不在 PyTorch autograd graph 里（autograd 只知道 tensor 依赖），也不在 PipeSchedule 的 stage 定义里（stage 只是一个 nn.Module 切片）。

### 4.2 Megatron 的代价：硬编码模型结构

![图 7：combined_1f1b 需要的硬编码知识](assets/comm_compute_overlap_analysis_fig7.png)

*图 7：combined_1f1b 需要的硬编码知识*

### 4.3 三层掩盖能力的可达性矩阵

|  | Megatron combined_1f1b | torchtitan ZBV/DualPipe | torchtitan TokenDispatcher | MindSpeed fb-overlap+DualPipeV |
| --- | --- | --- | --- | --- |
| 同一 mb 内 EP 掩盖 | ✓ shared expert side stream | ✗ PP 调度不涉及 | ✓ AsyncCollectiveTensor | ✓ alltoall/allgather-overlap（同微批） |
| 跨 mb 的 stage/layer 级交错 | ✓ combined_1f1b | ✓ ZBV/DualPipe (F/I/W) | ✗ | ✓ fb-overlap（layer 级） |
| 跨 mb + sub-layer 级交错 | ✓ dispatch vs mlp 混排 | ✗ F 对 stage 是原子操作 | ✗ | ✗ 无 layer 内 5 子节点拆解 |
| 跨 mb + EP+PP 打通 | ✓ 独有（sub-layer 粒度） | ✗ | ✗ | ✓（layer 粒度，DualPipeV overlap 段直调 fb-overlap） |

> **结论**：torchtitan 的掩盖停留在两个独立层次——TokenDispatcher 的 mb 内掩盖 + ZBV/DualPipe 的 stage 级跨 mb 掩盖，两者没有打通。MindSpeed 的 fb-overlap+DualPipeV 打通了 EP+PP，但停在 layer 粒度（前向层 ∥ 反向层），未像 Megatron 那样拆到 layer 内 sub-node。Megatron 通过硬编码模型结构，将 EP 与 PP 两层打通并下探到 sub-layer 粒度，实现了跨 mb 的最细粒度并发——粒度越细，需要框架"知道"的模型内部结构越多，这是三个框架路线选择的核心权衡。

## 五 框架差异总结

### 5.1 架构哲学对比

| 维度 | Megatron-LM | torchtitan | MindSpeed |
| --- | --- | --- | --- |
| Stream 管理 | 手动 CUDA stream + Event 同步 | 依赖 `AsyncCollectiveTensor` + NCCL stream | 手动 stream（MC2/CoC 融合核或软件 chunk 流水），HCCL 底座 |
| 调度粒度 | Layer 内 sub-node 级（5 节点） | Stage 级（F/I/W） | Layer 级（fb-overlap 前向层∥反向层）；TP 侧算子 tile 级（MC2）/chunk 级（CoC） |
| 模型感知 | 硬编码 layer 内部结构 | 零感知（autograd graph 级） | Patch 式感知（`MindSpeedFeature` 猴补丁整类替换 Linear/MoELayer/调度函数） |
| EP+PP 交叉 | ✓ combined_1f1b（sub-layer） | ✗ | ✓ fb-overlap+DualPipeV（layer 级协同） |
| PP bubble 消除 | 传统 1F1B + P2P overlap | ✓ ZBV / DualPipe（更先进） | ✓ DualPipeV（双向切半）+ RiPipe（气泡填重计算） |
| 模型支持 | GPTModel（含 MTP） | llama3/4, qwen3, deepseek_v3, gpt_oss, flux | Megatron-patch 生态（Ascend/NPU 训练场景） |
| 代码复杂度 | 高（~1200 行调度 + 模型改造） | 低（调度在上游，模型无改造） | 高（大量 `MindSpeedFeature` 互斥矩阵，见 [[11_mindspeed_comm_overlap_analysis]] §7） |
| 与 compile 兼容 | 部分冲突（CUDA Graph 禁用） | ✓（co-design，Async-TP 甚至要求 compile） | 面向 NPU 亲和路线，非 torch.compile 生态 |

### 5.2 适用场景

> **Megatron combined_1f1b 适合**：千卡以上 MoE 训练，EP 通信延迟是瓶颈。需要将 EP A2A 通信与 PP 1F1B 打通才能最大化 MFU。典型场景：DeepSeek-V3 风格 MoE 在 GB200/H100 集群上训练。

> **torchtitan ZBV/DualPipe 适合**：中等规模训练，PP bubble 是主要瓶颈。利用上游 PyTorch 的先进调度（ZB/DualPipe），代码简洁，模型零侵入。典型场景：Dense 模型 PP 训练，或 MoE 模型在 EP 通信不构成主要瓶颈时。

> **MindSpeed fb-overlap+DualPipeV 适合**：Ascend/NPU 上的 MoE 大模型训练，在无法直接复用 CUDA 生态（TE、DeepEP）时，用 NPU 原生融合算子（MC2/alltoall-MC2）+ 软件流水/换调度组合逼近 Megatron combined_1f1b 的打通效果，但约束更重（各特性两两互斥矩阵庞大，需按稠密/MoE 场景组合选型）。

### 5.3 一句话总结

> **Megatron 做的是 sub-layer 级跨 micro-batch 的 EP+PP 打通，torchtitan 做的是 stage 级跨 micro-batch 的 PP 调度优化，MindSpeed 做的是 layer 级跨 micro-batch的 EP+PP 打通（NPU 原生融合算子 + patch 式调度协同）。**三者共同印证同一条规律：掩盖粒度每下探一级，需要框架"知道"的模型内部结构就多一分——通用的 PP 调度框架做不到 sub-layer 级掩盖，唯有硬编码或深度 patch 模型知识才能实现。

## Related Pages

- [[20_megatron_comm_overlap_analysis]] —— Megatron-LM 通信掩盖权威机制页（TP/DP/PP/EP/CP 全维度，源码 file:line）
- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— torchtitan 通信优化权威机制页（跨维度矩阵 + Async-TP/对称内存/MinimalAsyncEP）
- [[15_torchtitan_ep_analysis]] —— torchtitan EP token all-to-all 与 `AsyncCollectiveTensor` 掩盖机制
- [[14_torchtitan_pp_analysis]] —— torchtitan PP 调度、ZBV/DualPipeV 的 I/W 拆分与 `OVERLAP_F_B`
- [[11_mindspeed_comm_overlap_analysis]] —— MindSpeed 通信掩盖权威机制页（MC2/CoC/fb-overlap/DualPipeV/RiPipe，源码 file:line）
- [[31_comm_compute_fusion_guide]] —— 通算融合（把通信编进单一 kernel）：与本页"掩盖"手段的边界声明见页首
- [[23_tilelang_analysis]] —— WaveEP 的 tile-level IR 实现机制
- [[12_deepseek_v3_analysis]] —— DualPipe PP 通算重叠设计
- [[13_deepseek_v4_analysis]] —— WaveEP 细粒度 EP 重叠（wave-based expert scheduling）
