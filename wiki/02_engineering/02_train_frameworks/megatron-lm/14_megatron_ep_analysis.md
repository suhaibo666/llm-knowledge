---
title: "Megatron-LM 专家并行(Expert Parallelism)深度解析"
---

# Megatron-LM 专家并行(Expert Parallelism)深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **源码基线**：`deepseek-ai/DeepEP@af9a0403188392824fc3057452822235873e0612`（`main`，2026-06-15）——只覆盖 DeepEP v1 `Buffer` 与 v2 `ElasticBuffer`；HybridEP 分支与 Transformer Engine 不在任何冻结检出内，见 §2.5 的依赖边界表
> **核心源码**：`megatron/core/transformer/moe/{moe_layer.py,token_dispatcher.py,router.py,experts.py,moe_utils.py,fused_a2a.py,shared_experts.py,paged_stash.py,token_dispatcher_inference.py}`；`megatron/core/transformer/{transformer_config.py,cuda_graph_config.py}`；`megatron/core/tensor_parallel/mappings.py`
> **中心结论**：EP 不把一个 token 永久切给一张卡，而是把 router 的“token→专家”关系短暂改写为“token 副本→持有该专家的 rank”。路由权重由本地 expert 路径消费，combine 只回送并累计已经加权的 expert 输出。`MoELayer` 将这一闭环固定为 route/preprocess/dispatch/local-expert/combine/postprocess；dispatcher 只替换中间搬运与布局恢复，不替换模型语义。
> **适用范围**：训练态 MCore MoE 的 expert ownership、六条训练 token dispatcher 数据面（三种一级取值，其中 flex 再分四个二级 backend）与其 forward/backward 边界，以及只为满足本页目标而必需的配套机制（shared expert、两条 overlap、whole-MoE CUDA graph 的静态形状出口）。模型装配归 [[10_megatron_model_structure_analysis]]，进程组构造归 [[17_megatron_parallelism_orchestration_analysis]]，PP 的 combined-1F1B 调度归 [[15_megatron_pp_schedulers_analysis]]，跨轴资源竞争与工程选型归 [[20_megatron_comm_overlap_analysis]]、[[39_megatron_moe_training_optimization_analysis]]。
> **最近更新**：2026-09-05。按房子形状重排为「特性概览 → 详细方案 → 代码实现分析 → 配套机制 → 约束/适用/趋势 → 配置契约」；补 §1 的收益代价表与符号表、§3.1 的所有权类图与 §3.2 的反向调用树、§4 的配套机制小节；六条数据面统一按「压力/上限 → 本地计算 → 上线数据 → 重构 → 反向差异 → 增量成本」重放同一算例，并新增依赖边界一览表。

---

## 1. 特性概览

### 1.1 问题背景

MoE 用很多专家扩大参数量，但每个 token 只计算 top-$k$ 个专家，因此参数量与每 token 的 FLOPs 解耦；代价是当专家总数 $E$ 大到单卡放不下时，DP 会在每个副本保存全部专家、TP 只能把每个专家自己的矩阵切细（GEMM 越切越小、算术强度下降）、PP 只能按层切而一层里就有 $E$ 个专家——三条都没有沿“专家编号”这一维切开的能力。EP 正是这条缺失的轴：把 $E$ 个专家按编号分给 $e$ 个 rank，每 rank 只保存 $E/e$ 个专家。但专家的所有者由 router 在**每次 forward 动态决定**，而 token 的所有者由数据切分静态决定，两者不重合，于是每一层都必须把 token 送到专家所有者、再把结果送回 token 的原所有者——EP 的全部工程问题都来自这次往返。

### 1.2 解决方法

Megatron 把这次往返压进一个可替换的组件 `MoETokenDispatcher`，并把 MoE 层本身固定成六个阶段：`route → preprocess → dispatch → local-expert compute → combine → postprocess`（`MoELayer.forward` 的 `custom_forward`）。router 产出的 `probs`/`routing_map` 是**本次 forward 的路线表**而不是 token 的永久属性；dispatcher 负责把这张表变成通信元数据、把 token 搬到专家所有者、并缓存足以反置换的信息。因此换搬运后端不改变模型语义：`allgather` 用全域复制换掉逐目的地的变长 split，`alltoall` 用变长 split 换更少的 token 副本，`flex` 把同一接口交给四个可选的融合通信后端。路由权重不在 combine 里补乘，而是随 edge 一起送到本地 expert，由 `TEGroupedMLP`/`SequentialMLP` 在自己的激活/FC2 路径上消费——这一条决定了整条反向链的形状。

### 1.3 收益、开销和约束

| 维度 | 直接收益 | 必付成本或边界 |
|---|---|---|
| 参数与优化器状态 | 路由专家按编号分摊，每 rank 只持有 $E/e$ 个 expert 的权重、梯度与优化器状态 | 只覆盖**路由专家**；attention、shared expert、embedding 不在这条轴上，不能把结论推广成“全部参数除以 $e$” |
| token 激活 | 专家计算在到达本 rank 的连续段上进行，不需要物化全局 expert 输出 | EP 本身不做 activation sharding：dispatch buffer、置换 metadata、（allgather 时）全局 token 副本都是新增激活 |
| 通信 | `alltoall`/`flex` 只把 token 送向真正的专家所有者，避免全域复制 | dispatch 与 combine 都在层的关键路径上，一层两次；变长 split 还带 CPU 可见的元数据与同步点 |
| 计算 | 本地 experts 可按 `tokens_per_expert` 分段批量执行（grouped GEMM） | 最满的 expert 决定该层完成时间；空段/小段降低 grouped work 效率，router 的均衡策略只能缓解不能保证 |
| 形状与同步 | 容量策略可把动态形状钉成静态，换来 CUDA graph 可捕获 | 固定形状用额外 compute/memory 买：pad 出来的 slot 也要参与 GEMM 与搬运；不设容量则必然有一次 DtoH |
| 后端可替换性 | `MoELayer` 与 expert MLP 对四个 Flex backend 完全不变 | 每个 backend 各有安装、版本、shape、capacity 与 CUDA-graph 限制；三个 backend 的内部行为不在任何冻结检出内（§2.5） |

### 1.4 符号约定

| 符号 | 含义 |
|---|---|
| $E$、$e$ | 专家总数 `num_moe_experts`、EP 度 `expert_model_parallel_size`；每 rank 持有 $E/e$ 个本地专家 |
| $k$ | `moe_router_topk`，每个 token 选中的专家数 |
| $T_{\mathrm{local}}$、$T_{\mathrm{global}}$ | 本 rank 实际送进 router 的扁平 token 数、EP 域内的逻辑全局 token 数 |
| $H$ | hidden size（启用 `moe_latent_size` 时 dispatcher 看到的是 latent 维） |
| $t_i$、$y_i$ | 第 $i$ 个 hidden token、它的 MoE 层输出 |
| $p_{i,e}$、$f_e(\cdot)$ | token $i$ 到专家 $e$ 的路由权重、专家 $e$ 的 MLP |
| $K$、$K_{\mathrm{remote}}$ | 全局 route edge 总数、其中跨 rank 的条数 |
| $C$、$f_{\mathrm{cap}}$ | per-expert 容量、`moe_expert_capacity_factor` |
| EP / ETP / EDP | expert parallel / expert tensor parallel / expert data parallel |
| AG / RS / A2A | all-gather、reduce-scatter、all-to-all |

---

## 2. EP 详细方案

### 2.1 最小原语：一次 token 所有权改写

令 EP 域的逻辑全局 token 数为 $T_{\mathrm{global}}=4$，每个 rank 实际输入 router 的本地 token 数为 $T_{\mathrm{local}}=2$；总专家数 $E=4$、top-$k=2$、EP 度 $e=2$，并固定 TP=1 以免把 ETP 混进来。具名 hidden token $t_i\in\mathbb{R}^{H}$；EP rank 0 的实际 buffer 为 $[t_0,t_1]\in\mathbb{R}^{2\times H}$、拥有 experts ${0,1}$，rank 1 的实际 buffer 为 $[t_2,t_3]\in\mathbb{R}^{2\times H}$、拥有 experts ${2,3}$。两 rank 的逻辑并集是 $[4,H]$，但在通信前并不存在一张物化的“全局 router 表”：每 rank 的实际 `probs/routing_map` 是 $[T_{\mathrm{local}},E]=[2,4]$，概念并集才是 $[T_{\mathrm{global}},E]=[4,4]$。后文所有 dispatcher/backend 始终复演这同一组路线。

<!-- megatron-ep-figure-contract: {"tokensGlobal":4,"tokensLocalPerRank":2,"experts":4,"topk":2,"ep":2,"tp":1,"edges":8,"remoteEdges":4,"remoteUniqueRankCopies":3,"capacityAtHalf":1,"droppedEdgesAtHalf":0,"capacityAtOnePointFive":2,"slotsPerOwnedExpertAfterA2A":4,"realSlotsPerOwnedExpert":2,"zeroSlotsPerOwnedExpert":2,"backends":["DeepEP","DeepEPv2","HybridEP","NCCL-EP"],"inferenceSibling":{"field":"inference_moe_token_dispatcher_type","values":["nccl","nvls"],"classes":["NCCLAllGatherDispatcher","NVLSAllGatherVDispatcher"],"selector":"InferenceMode.is_active","owners":["30_megatron_rl_posttraining_consistency_analysis","31_megatron_inference_engine_analysis"]}} -->

| 原 token | 初始 owner | 选中专家 | 非零路由权重 | 计算期间的所有者 |
|---|---|---|---|---|
| $t_0$ | rank 0 | 0、3 | $p_{0,0}$、$p_{0,3}$ | rank 0 的 expert 0；rank 1 的 expert 3 |
| $t_1$ | rank 0 | 1、2 | $p_{1,1}$、$p_{1,2}$ | rank 0 的 expert 1；rank 1 的 expert 2 |
| $t_2$ | rank 1 | 2、3 | $p_{2,2}$、$p_{2,3}$ | rank 1 的两个本地专家 |
| $t_3$ | rank 1 | 0、1 | $p_{3,0}$、$p_{3,1}$ | rank 0 的两个本地专家 |

`TopKRouter.forward` 先做 gating，再由 `TopKRouter.routing` 产生 `probs` 与布尔 `routing_map`。二者都以本 rank 的扁平 token 为第一维、以全局专家为第二维，即实际形状 $[T_{\mathrm{local}},E]=[2,4]$；容量处理前，每行至多有 $k$ 条非零有效边，pad-to-capacity 后则可能多出概率为零的 shape slot。它不是把 expert id 塞进 token 的永久属性：它是本次 forward 的路线表，dispatcher 必须缓存足以反置换的 metadata。

所有 dispatcher 的 preprocess 都先把本 rank 的 `[...,H]` 视作 $[T_{\mathrm{local}},H]=[2,H]$ 并保存路线 metadata，但第一次排列的位置不同：AllToAll 在发送前展开 edge，AllGather 在 gather 后筛本地列，DeepEP/DeepEPv2 在收件后由 MCore 展开，HybridEP/NCCL-EP 则让依赖直接返回 expert-major。无论在哪一步实现，每条有效边 $(t_i,e_j)$ 最终都进入 expert owner 的连续段与 `tokens_per_expert`；`MoELayer.routed_experts_compute` 将相应 `permuted_probs` 一并传给本地 experts，故离开 `TEGroupedMLP`/`SequentialMLP` 的 edge 输出已经带权，combine 只逆转布局、跨 rank 回送并累计：

$$
y_i=\sum_{e\in\operatorname{TopK}(t_i)}p_{i,e}f_e(t_i).
$$

![EP 将四个 token 展开为八条路由边、在专家所有者上计算并回到原位置的前反向闭环](assets/megatron_ep_route_compute_combine.svg)

这张图里的数可以逐项结算。全局 route edge 总数为 $K=T_{\mathrm{global}}\cdot k=4\cdot2=8$，每个 origin rank 先产生 $T_{\mathrm{local}}k=4$ 条；四个专家各收到两条：$e_0:[t_0,t_3]$、$e_1:[t_1,t_3]$、$e_2:[t_1,t_2]$、$e_3:[t_0,t_2]$。其中 $t_0\to e_3$、$t_1\to e_2$、$t_3\to e_0$、$t_3\to e_1$ 共 $K_{\mathrm{remote}}=4$ 条跨 rank edge，其余四条留在初始 owner。前向因此不是一句“做 A2A”，而是以下四步：

1. rank 0/1 的路线表各定义四条带权逻辑 edge，并缓存 origin 与反置换信息；AllGather 或 rank-dedup manager 不一定在发送前物化四份 hidden copy。
2. 每条 copy 到 expert owner 后按全局 expert id 变成连续段；四个本地 MLP 消费各 edge 的 `permuted_probs`，形成 $p_{i,e}f_e(t_i)$。
3. 已加权 expert 输出沿相反所有权方向返回；origin 只累计同一 token 的两条边，恢复 rank 0 的 $[y_0,y_1]$ 与 rank 1 的 $[y_2,y_3]$，combine 不再重复乘权重。
4. backward 从 $g_i=\partial L/\partial y_i$ 开始。combine 的逆映射先把同一个 $g_i$ 发给产生两条已加权 edge 的 expert owner；本地 expert backward 穿过权重乘法，得到 $\partial L/\partial f_e(t_i)=p_{i,e}g_i$ 与 $\partial L/\partial p_{i,e}=\langle g_i,f_e(t_i)\rangle$，再产生参数梯度和 route input gradient；dispatch 的逆映射最终把同一 $t_i$ 的两份 input gradient 累加回初始 owner。

因此“回到原 rank”不是优化提示，而是正确性要求：后续 residual、loss 和上一层反向仍消费原 token 布局。若容量策略丢弃某条边，或 padding 为其补零，这个等式对应的有效边集合随之改变；不能把固定 shape 误说成不丢 token。

**权重在哪里乘，是一个可被推翻的默认。** 默认路径把 `permuted_probs` 交给本地 expert：`TEGroupedMLP.forward` 在 activation/FC2 路径上按 slot 乘，`SequentialMLP.forward` 按 `tokens_per_expert` 切成 `probs_list` 后作为各 local MLP 的 `per_token_scale`。只有 `moe_apply_probs_on_input=True` 时两者才改为在 expert 输入侧乘一次、再把 `permuted_probs` 重置为全 1，且该分支硬性 `assert moe_router_topk == 1`（该字段的 owner 是 [[21_megatron_fusion_operators_analysis]]）。无论走哪条，combine 都收到已加权结果——这是本页全部 backward 结论的前提。

### 2.2 EP、ETP、EDP 的本页语义

| 轴 | 本页消费的所有权 | 本页可见的数据面 | 不在本页展开的责任 |
|---|---|---|---|
| EP | 将 $E$ 个专家按连续编号切为每 rank $E/e$ 个本地专家 | dispatch/combine 在专家所有者之间搬 token | `ProcessGroup` 的生成与全局坐标 |
| ETP | 一个本地 expert 内的张量并行副本/切分 | `alltoall` 路径在 expert 计算前 AG、计算后 RS 恢复所需布局 | expert linear 的完整 TP 算法 |
| EDP | 相同 expert 参数副本之间的数据并行 | local expert backward 之后的梯度消费者 | 分布式优化器、状态分片和 step |

`BaseMoELayer.__init__` 从注入的 `pg_collection.ep` 取 EP rank，以 `num_moe_experts / ep_size` 建立连续 `local_expert_indices`；不可整除直接 `assert`。`pg_collection.expt_tp`、`ep`、`tp_ep`、`expt_dp` 的实际建立属于并行编排页。这里重要的是局部/全局身份的分界：router 一律用全局 expert 编号，expert module 只拥有 `local_expert_indices`，而 dispatcher 负责两种身份之间的 token 路线。

MoE parallel folding 让 expert 三轴不必逐项等于 attention 三轴，但同一 PP stage 的 rank 数必须守恒：$\mathrm{ETP}\cdot\mathrm{EP}\cdot\mathrm{EDP}=\mathrm{TP}\cdot\mathrm{CP}\cdot\mathrm{DP}$。例如 attention 的 TP=4、CP=2、DP=8 可折为 expert 的 ETP=1、EP=64、EDP=1；这解释了为何 ETP=1 不等于全模型 TP=1。该式与进程组坐标的构造、合法分解和 rank 排布由 [[17_megatron_parallelism_orchestration_analysis]] 负责；本页只消费注入后的 groups。

### 2.3 从原语放入 MoELayer：谁持有什么、选择轴在哪

先把训练的两级选择与正交推理 sibling 轴钉死。它们不是一个“任意 backend”字符串，而是三处不同的 live selection 点：

| 选择层级 | 配置字段与源码允许值 | live selection |
|---|---|---|
| 一级 dispatcher | `moe_token_dispatcher_type ∈ {allgather, alltoall, flex}` | `MoELayer.__init__` 构造 AllGather、AllToAll 或 Flex dispatcher，其他值 `raise ValueError` |
| Flex 二级 backend | `moe_flex_dispatcher_backend ∈ {deepep, deepepv2, hybridep, ncclep}` | `MoEFlexTokenDispatcher.__init__` 分别构造 `_DeepepManager`、`_DeepepV2Manager`、`_HybridEPManager`、`_NCCLEPManager` |
| 推理 sibling（正交轴） | `inference_moe_token_dispatcher_type ∈ {nccl, nvls}` | `MoELayer._setup_inference_mode` 构造 NCCL/NVLS 推理 dispatcher；`MoELayer.forward` 由 `InferenceMode.is_active()` 在保留的训练实例与推理实例间选择 |

枚举依据是源码自己的选择点：前两行取自 `TransformerConfig` 上两个 `Literal` 的取值集合，并与 `MoELayer.__init__` / `MoEFlexTokenDispatcher.__init__` 的分支逐一对上；第三行取自 `_setup_inference_mode` 与 `forward` 开头的实例切换。后文规范名称依次写作 **DeepEP、DeepEPv2、HybridEP、NCCL-EP**。一级 `flex` 只统一 MoELayer 接口；它没有消除四个 manager 的不同收件布局、同步点、workspace 和依赖上限。

上面是训练 dispatcher 主轴；源码还有一个正交的**推理 sibling 轴**，不能把它误并入 `{allgather, alltoall, flex}`。当 `transformer_impl="inference_optimized"` 且 EP>1 时，`inference_moe_token_dispatcher_type ∈ {nccl, nvls}`：`MoELayer._setup_inference_mode` 分别构造 `NCCLAllGatherDispatcher` 与 `NVLSAllGatherVDispatcher`（两者都继承自 `MoEAllGatherTokenDispatcher`，因此它们是 AllGather 的推理特化而不是第四、第五种训练数据面），同时保留训练 dispatcher；`MoELayer.forward` 再以 `InferenceMode.is_active()` 选择当前实例。`nccl` 是 NCCL AllGather/ReduceScatter（非 graph 的不等长输入会先 pad/gather/compact），`nvls` 是基于对称内存的变长 NVLS AllGather-V/ReduceScatter-V。这里仅登记选择边界：RL/post-training 中训练与生成/打分路径如何保持一致，归 [[30_megatron_rl_posttraining_consistency_analysis]]；推理引擎何时设置/清除 `InferenceMode`、请求批处理与 CUDA graph 生命周期明确归 [[31_megatron_inference_engine_analysis]]。

再看这一层里谁持有什么、为什么这样分：

| 组件 | 责任与跨界状态 | 选择它而非什么 | 完成/代价边界 |
|---|---|---|---|
| `MoELayer` | 持有 router、dispatcher、routed experts、可选 shared expert；串接六个阶段 | 不把通信埋入 router 或 MLP，因而同一 MoE 语义可换搬运后端 | 返回原 `hidden_shape` 的 output；只返回层输出，不产生 loss |
| `TopKRouter` | `hidden_states` → `probs`,`routing_map` | 不把 top-$k$ 决定留成隐式排序索引；布尔 map 可被 dispatcher、容量逻辑和置换共同消费 | 路由/均衡损失附着在计算图；偏斜仍由后续 `tokens_per_expert` 暴露 |
| `MoETokenDispatcher` | 保留原形状、路由 map、概率、split 或反置换信息；实现 dispatch/combine 两对接口 | 不为每个 backend 重写 expert MLP 或 MoELayer | metadata 是跨 forward/backward 的显存与同步负担 |
| routed experts | 按 `tokens_per_expert` 在本 rank 运行 expert MLP | 不在所有 rank 复制所有 expert 参数 | 输入到达本地、每段计数可用才可完成；最满 expert 决定计算尾部 |
| `SharedExpertMLP` | 每个 token 都过的固定 MLP，与 router 无关 | 不把它做成第 $E+1$ 个可路由专家，因而不占 EP 的编号空间、不进 dispatch | 非重叠时在 `postprocess` 相加；重叠时另有 stream/event 代价（§4.1） |

`MoELayer.__init__` 随后用 `num_local_experts` 建立 expert 模块，所以 dispatcher 不能改变“某 rank 有哪些专家”的参数语义，只能改变 token 如何到那里。同一构造器里 `MoELayer.routed_experts_compute` 把 `dispatch_postprocess` 返回的 `(dispatched_input, tokens_per_expert, permuted_probs)` 三元组原样交给 `experts`，再把结果交给 `combine_preprocess`——这三元组就是六条训练数据面必须共同产出的契约，也是它们唯一被允许不同的地方之外的全部共性。

### 2.4 三种一级 dispatcher：同一闭环，不同数据面

`megatron/core/transformer/moe/README.md` 给的是适用性建议，而非性能保证：allgather 面向 TP-only、小 EP 或大 top-$k$；alltoall 面向标准 EP 大于 1；flex 的 DeepEP/HybridEP 描述面向跨节点或特定 NVLink 拓扑。实际选择仍是 `TransformerConfig.moe_token_dispatcher_type` 的显式值，默认值是 `allgather`。同一份 README 里“EP All-to-All 未优化时可占训练时间 30–40%”是一句**没有给出模型、形状、拓扑与测量方法**的通用陈述，本页把它记作 README 主张而不作为结论引用——下文所有开销结算只写源码能证明的元素数、同步点与资源上限。

![同一路由算例在 AllGather、AllToAll 与 Flex 训练 dispatcher 中的对照，以及 nccl/nvls 正交推理 sibling 选择边界](assets/megatron_ep_dispatcher_variants.svg)

图中前三行的蓝色数据依赖和橙色成本标记都沿用 §2.1 的八条 edge，最终必须得到相同的 $[y_0,y_1]$ / $[y_2,y_3]$；差别只在中间 buffer 里复制的是原 token 还是 route copy、谁维护 split，以及 backward 用哪个对偶 collective。第四个 panel 只登记推理 sibling 的实例选择与 owner，不把 `nccl`/`nvls` 伪装成训练 dispatcher 的第四、第五种数据面。

#### 2.4.1 AllGather：用复制取消逐目的地 split

**它答的压力**：如何在没有 CPU 可见的变长 split 的前提下让每个 rank 拿到本地专家需要的全部 token。**封顶它的资源**：TP×EP 域的聚合带宽与每 rank 的激活显存——收件量与本地专家实际需要多少 token 无关。

`MoEAllGatherTokenDispatcher.dispatch_preprocess` 展平 hidden state 并保存 `routing_map`。`token_dispatch` 在 TP×EP 组上对 map、probs 和 token 做 first-dimension all-gather；所有参与 rank 随后都看见全局 token 副本。`dispatch_postprocess` 截取本地 expert 列，`permute` 出本地 expert 的 token 段（`num_out_tokens=tokens_per_expert.sum().item()` 是这条路上唯一的 host-visible 取值），并把本地列的 probs 转置后按 map `masked_select` 成每 edge 一个标量；本地计算后，`combine_preprocess` 先按缓存 mapping unpermute，`token_combine` 用 TP×EP reduce-scatter 将各 rank 的贡献聚回 token 原位置，最后 reshape。

代回本例：两个 rank 的实际输入与 route 表分别是 $[T_{\mathrm{local}},H]=[2,H]$、$[T_{\mathrm{local}},E]=[2,4]$；AG 后每个 rank 才物化 $[t_0,t_1,t_2,t_3]$ 以及聚合后的 $[T_{\mathrm{global}},E]=[4,4]$ map/probs。rank 0 只筛出 $e_0:[t_0,t_3]$、$e_1:[t_1,t_3]$ 四条本地 expert edge；rank 1 同理筛出 $e_2:[t_1,t_2]$、$e_3:[t_0,t_2]$。MLP 先消费每条 edge 的权重，本地 unpermute 与 RS 再累计并切回两个 origin：rank 0 得 $[y_0,y_1]$，rank 1 得 $[y_2,y_3]$。**上线的是原 token 而不是 route copy**：两个 rank 各发 2 行、各收 4 行，全组物化 $eT_{\mathrm{global}}=8$ 行，另加同样被 AG 的 $[4,4]$ map 与 probs。**反向**正好把这两步转置：combine RS 的反向是 AG，dispatch AG 的反向是 RS；本地 `permute`/`unpermute` 与 expert 内权重乘法按保存的 mapping/probs 回收多条 top-$k$ 边。

它拒绝的替代是“为每个接收 rank 计算可变 split 再做定向交换”：没有 `input_splits`/`output_splits` 的 CPU metadata 与相应同步，**增量成本**是把 token、完整 `probs` 和 map 复制到整个 TP×EP 域。`TransformerConfig.__post_init__` 明确拒绝 allgather 与 `variable_seq_lengths` 的组合；sequence packing 也只允许 alltoall 或 flex。因而它适合 README 列出的规模/拓扑包络，而不是所有 EP。

#### 2.4.2 AllToAll：只把 token 副本送向 expert 所有者

**它答的压力**：EP 大于 1 且路由稀疏时，AllGather 会把大量与本地专家无关的 token 也搬过来。**封顶它的资源**：变长 `output_splits` 的最大项——偏斜大的一方同时决定收件 buffer 与 expert compute 的尾部。

`MoEAlltoAllTokenDispatcher.preprocess` 从 `routing_map.sum(dim=0)` 计算各 expert 的 token 数。它把计数 reshape 成 EP rank 的 `input_splits`，在 `tp_ep_group` gather 计数，导出当前 rank 的 `output_splits`、`output_splits_tp` 和 `num_global_tokens_per_local_expert`。这些是本次路线的控制面；非 drop-and-pad 的 capacity/drop 或量化 padding 会让输出大小动态，代码须在分配 buffer 前安排 CUDA DtoH/sync，EP A2A 前也必须让 split 值可用。源码把这件事显式建模成一条**同步点阶梯** `before_permutation_1 < before_ep_alltoall < before_permutation_2 < before_finish < no_sync`，`preprocess` 按当前配置取其中最早的一档，并断言 DtoH 点不晚于 sync 点。

本例每个 origin 有四条 route copy，恰好两条给 rank 0、两条给 rank 1，所以两端 `input_splits=[2,2]`，接收后也各有四条 edge。这个平衡只属于本例；一般情况下 splits 由当前 `routing_map` 决定。四条跨 rank edge 真正越过 EP 边界，另四条在 collective 的本地 chunk 中完成所有权映射。数据面随后是：

1. `dispatch_preprocess` 以 `permute` 把本 rank 的 $[T_{\mathrm{local}},H]=[2,H]$ 展开为四条 route copy，按目标 expert 排列并保存反置换 mapping 与原 shape；两 rank 合计八条。
2. `token_dispatch` 用 `all_to_all(ep_group, ..., output_splits, input_splits)` 分别搬 token 与概率；token 到达即是本地 expert rank 的收件 buffer。**上线的是 route copy**：本例每 rank 发 4 行、收 4 行，其中 2 行真正跨 rank。
3. `dispatch_postprocess` 在 `tp_size>1` 时先在 TP 组 AG；多个 local expert 时用 `sort_chunks_by_idxs` 做第二次按 expert 分段的排列。它的反向工作在 `combine_preprocess`：解除第二排列，必要时 RS 回 TP 布局。
4. 本地 MLP 先用随 edge 到达的概率加权；`token_combine` 用反向 split 再做一次 EP A2A，`combine_postprocess` 以保存的 mapping `unpermute` 累计已加权 edge，并 view 回原 shape。

**反向**由 `tensor_parallel/mappings.py::_AllToAll.backward` 承担：它把保存的 `input_split_sizes` 与 `output_split_sizes` **对调**后再调一次 `_AllToAll`，因此不需要第二套 metadata；TP 辅助映射同样成对（forward RS ↔ backward AG）。**增量成本**相对 AllGather 是变长 metadata、两次本地重排，以及至少一个“split 已 materialize”的同步边界；它没有保证均衡，`output_splits` 很大的一方仍成为网络与 expert compute 的尾部。

#### 2.4.3 Flex：同一个接口，四条并未被统一的数据面

`MoEFlexTokenDispatcher._initialize_metadata` 将每 rank 的 $[T_{\mathrm{local}},E]=[2,4]$ 整理为覆盖 TP×EP 域的 $[T_{\mathrm{local}},\mathrm{world},E_{\mathrm{local}}]=[2,2,2]$ 视图（TP 维是 `expand` 出来的复制，因为同一 TP 组内每个 rank 要收到相同 token）；此后四个 manager 都必须返回 expert 可消费的连续段，并在 combine 后恢复 $[T_{\mathrm{local}},H]$。但“同一接口”到这里就结束了：DeepEP/DeepEPv2 收到的是按目标 rank 去重的 token，还需 MCore 第二次 local permute；HybridEP/NCCL-EP 则从依赖直接取得 expert-major buffer。四个 manager 之间连构造前提都不一致——DeepEP、DeepEPv2、NCCL-EP 各有一条 `assert tp_size * ep_size > 1`，HybridEP 没有。

### 2.5 Flex 的四条数据面

**先立证据等级，因为四条 lane 的可证范围并不相同。** 这条边界不是修辞：三个依赖里只有一个在冻结检出内。

| lane | 依赖与是否在冻结检出内 | MCore 冻结源能证明什么 | 只能作为依赖的公开契约引用 |
|---|---|---|---|
| DeepEP | `deep_ep.Buffer`（`DeepEP@af9a040` **在检出内**：Python 层、docstring 与部分 CUDA 源可读） | wrapper 的 autograd 形状：`FusedDispatch.forward` 传 `topk_weights`、`FusedCombine.forward` 不传、二者互为对偶 | kernel 内部的通道调度与 RDMA/NVLink 编排（未读 `.cu` 全文） |
| DeepEPv2 | `deep_ep.ElasticBuffer`（**在检出内**） | `_DeepepV2Manager` 每次 dispatch 前按当前 shape 取 buffer；wrapper 的 dispatch↔combine 反向对偶 | ElasticBuffer 内部的 SM/QP 自动推导与扩容策略 |
| HybridEP | `deep_ep.HybridEPBuffer`（**不在检出内**：冻结 `DeepEP@af9a040` 的 `main` 里 `git grep HybridEPBuffer` 无命中，它属于 `hybrid-ep` 分支） | wrapper 的 I/O、64 对齐、overflow flag 的读取与累计、rerun 协议、autograd 对偶，以及一次 `inspect.signature` 版本探测 | `dispatch_with_permute` / `combine_with_unpermute` 内部的两级 permute 与通信本身 |
| NCCL-EP | `transformer_engine.pytorch.ep`（**不在检出内**，本机也无 TE） | MCore 只调用模块函数 `ep_dispatch` / `ep_combine`，**自己不写 `torch.autograd.Function`** | 连 backward 都在 TE 内：MCore 侧只有 “autograd-aware” 的 docstring 断言，形状之外无可证内容 |

因此下面四段里，凡属右两列的内容一律标为依赖契约或 docstring 描述，不作为已验证执行叙述。

![Flex 的 DeepEP、DeepEPv2、HybridEP、NCCL-EP 四条独立 lane，以及各自的 MCore 依赖边界、布局、反向和资源成本](assets/megatron_ep_flex_backends.svg)

#### 2.5.1 DeepEP：rank 去重收件，MCore 再展开 expert edge

**它答的压力**：一个 token 选中同一 rank 上的多个专家时，AllToAll 会把它发送多次。**封顶它的资源**：NVLink/RDMA buffer 与固定的 SM 预算（v1 在 MCore 侧默认 20 个 SM）。

`_DeepepManager.setup_metadata` 将 router 输出中每 token 的两条真值边压成 top-k indices/weights；`dispatch` 跨过 **MCore ↔ DeepEP@af9a040 依赖边界**，调用 v1 `Buffer.get_dispatch_layout` / `Buffer.dispatch`。冻结 DeepEP 的 `is_token_in_rank`（docstring 明写为 `[num_tokens, num_ranks]` 的“该 token 是否要发给该 rank”）让一个 token 对同一目标 rank 只发送一份，收件仍带 local expert indices/weights。于是本例 rank 0 的 non-expanded 收件是 $[t_0,t_1,t_3]$，rank 1 是 $[t_0,t_1,t_2]$；$t_3$ 虽同时去 rank 0 的 experts 0/1，跨 rank 只是一份 token。三份跨 owner rank-copy 是对四条 remote expert-edge 的去重，不是少算了一条 expert edge。

回到 MCore，`get_permuted_hidden_states_by_experts` 把收件 indices 还原成 local multihot map，再 `permute` 为 rank 0 的 `e0:[t0,t3] | e1:[t1,t3]`、rank 1 的 `e2:[t1,t2] | e3:[t0,t2]`，即每 rank 四条 expert-major edge，并把 `permuted_probs` 交给 `MoELayer.routed_experts_compute → TEGroupedMLP.forward/SequentialMLP.forward`。权重就在该 local expert 路径消费；MLP 输出已经是 $p_{i,e}f_e(t_i)$，MCore local `unpermute` 先累计同 rank 的已加权 edge。随后 v1 forward 的 `FusedCombine.forward` 调 `Buffer.combine(x, handle=handle, ...)`，**不传 `topk_weights`**，只按 handle 回送/跨 rank 累计到 origin $[2,H]$。

DeepEP v1 **backward** 也必须按这个位置读：`FusedCombine.backward` 先用同一 handle 调 `Buffer.dispatch`，把 origin 的 $g_i$ 送回对应 expert copy；本地 expert autograd 穿过权重乘法，产生 $p_{i,e}g_i$、route-weight gradient 与 expert 参数梯度；最后 `FusedDispatch.backward` 才调用 `Buffer.combine` 回收 `grad_x`，并通过其可选 `topk_weights` 槽运输/累计 `grad_token_probs`。这次传的是**权重梯度数据**，不是在 forward combine 重新应用 route weights。

**增量成本**是 layout/handle、NVLink/RDMA buffer、float32 route weights（probs 不是 fp32 时 wrapper 直接 `.float()` 并 warning），以及精确收件数用于 MCore `num_out_tokens` 的 host-visible 同步；v1 默认占 20 个 SM，且冻结 `Buffer.get_dispatch_config/get_combine_config` 是一张写死的 `config_map`，只为 `{2,4,8,16,24,32,48,64,96,128,144,160}` 这些 group size 给配置、**最大 160**，其余直接 `assert num_ranks in config_map`。跨节点 kernel 的 source metadata 还能在同一目标 RDMA 域内先发一份、再 NVLink fan-out。**源码事实**止于这些协议；是否比 AllToAll 快必须在目标 topology/shape 实测。适合已经部署该 v1/NVSHMEM 路径、group size 落入支持集合，且能接受动态 layout/CUDA-graph 限制的训练。

#### 2.5.2 DeepEPv2：ElasticBuffer 搬运，MCore 仍做第二次 permute

**它答的压力**：v1 的 buffer 与 SM/QP 预算要按最坏 shape 静态给定，通信域也受 `config_map` 的 group size 限制。**封顶它的资源**：`WorkspaceLayout` 的三条编译期上限与每次扩容后的 buffer 显存。

`_DeepepV2Manager` 不调用 v1 构造器（源码注释写明 v2-only 镜像可能不带 v1 `Buffer` API）；它从同一 router 输出取得 top-k indices/weights，通过 `get_elastic_buffer` 按**当前 token 数、$H$、top-k** 取得可缓存、必要时放大的 V2 buffer，再调 `ElasticBuffer.dispatch`。冻结 DeepEP 的默认 `do_expand=False` 仍返回 non-expanded token + local expert indices/weights，因此本例的 rank 收件身份与 DeepEP lane 相同（各收 3 token），随后也由 MCore 展开为四条 expert-major edge、在本地 expert MLP 内消费 weights、unpermute，再以未重复加权的 `ElasticBuffer.combine` 回到 origin。V2 wrapper 的 **backward** 明示：dispatch backward 调 combine，combine backward 调 dispatch，并复用 forward handle——与 v1 同形，只是换了 buffer 对象。

区别在资源面：`num_sms=0` 让 ElasticBuffer 按 expert/top-k 自动推导 SM/QP（`get_theoretical_num_sms` / `get_theoretical_num_qps`，其内部推导属依赖侧）；MCore 的全局 buffer cache 会随当前 token 上界、$H$、top-k 扩容。未传 cached handle 时，冻结 V2 `dispatch` 的 `do_cpu_sync` 解析为 `value_or(do_cpu_sync, True)`，即默认 `do_cpu_sync=True`，MCore 又消费 CPU 侧 `num_recv_tokens_per_expert_list`，所以“elastic”不等于没有控制面同步。冻结 `WorkspaceLayout` 还硬断言 ranks≤1024、experts≤2048、experts/rank≤256（`kNumMaxRanks/kNumMaxExperts/kNumMaxExpertsPerRank`），并要求 `num_experts % num_ranks == 0`；README 的“up to EP2048”不能覆盖这些代码上限。README 给出的 Hopper/SM90、CUDA≥12.3、PyTorch≥2.10、NCCL≥2.30.4 是依赖环境包络；“更少 SM/更高吞吐”只是在指定机器与 shape 的发布测量，不是 MCore 或本例保证。**增量成本**相对 v1 是每次 dispatch 一次 buffer 查询/可能的扩容，换来自动 SM-QP 与更大通信域。

#### 2.5.3 HybridEP：依赖直接返回 expert-major

**它答的压力**：DeepEP lane 里 MCore 还要做第二次 local permute，且动态收件数必然带一次 DtoH。**封顶它的资源**：`num_permuted_tokens` 这个**整 rank** 的收件上界——超出它的 token 在依赖内被丢弃。

`_HybridEPManager.setup_metadata` 保留 router 输出的 routing map/probs；`hybrid_ep_dispatch` 调 `HybridEPBuffer.dispatch_with_permute`，将通信与两级 permute 融合，故依赖直接返回本例每 rank 四条 expert-major edge；`_HybridEPManager.get_permuted_hidden_states_by_experts` 只原样交出 hidden/probs，**不调用 MCore 二次 local permute**。这点与 NCCL-EP 相同，区别是本 lane 把通信和两级 permute 融进 HybridEP dependency。本地 expert MLP 同样先消费随 slot 返回的 weights；已加权输出再由 `combine_with_unpermute` 直接恢复 $[t_0,t_1]$ / $[t_2,t_3]$ 的 origin 顺序。**backward** 由 MCore 自己的两个 autograd Function 给出且互为对偶：`HybridEPCombine.backward` 调 `dispatch_with_permute` 复用同一 handle 把 $g_i$ 送回 expert slot，`HybridEPDispatch.backward` 调 `combine_with_unpermute` 把 hidden 梯度送回 origin，并**同时返回 `combined_probs` 作为 route-weight 梯度**——与 DeepEP 用 `topk_weights` 槽运输权重梯度是同一分工的另一种写法。

动态形状下，精确 `tokens_per_expert.sum()` 会形成 DtoH 点；设置 `moe_expert_rank_capacity_factor` 后，MCore 以本 rank 的 padded local-token 数×top-k×factor 给出对齐后的 `num_permuted_tokens` 上界，并据此把 `non_blocking=True` 传给依赖。超过该 rank budget 的 token 在 HybridEP 依赖内被 drop，handle 的 overflow flag 被 `_HybridEPManager.dispatch` 累计进 `over_budget`；这不是当场 hard trap。训练入口在设置 rank capacity 时以 `PagedStashRunner` 包装整步：一次 forward/backward 后跨 rank 汇总 `over_budget`（与 paged-stash overflow、host spill 一起 stack 成三项做**一次** `all_reduce`），常规情况下清梯度、清 rank capacity、禁用 paged stash，并用同一批 microbatches **整步 dropless rerun**——整个循环最多两次尝试（`assert num_tries < 2`），第二次已切到动态 dropless 路径。只有训练中的 Transformer Engine **whole-MoE CUDA graph 已经完成 capture** 时，静态 buffer overflow 不允许动态 fallback，`PagedStashRunner._raise_if_te_whole_moe_graph_overflow` 才抛 `RuntimeError`。`moe_hybridep_pad_variable_tokens` 还会 all-reduce 组内最大 local-token 数并按 64（`HYBRIDEP_TOKEN_ALIGNMENT`）pad，combine 后再截回原长度。

这里必须画清 **MCore ↔ HybridEP dependency 边界**：冻结 DeepEP@af9a040 的 `main` 只导出 `Buffer/ElasticBuffer`，不含 MCore 所 import 的 `HybridEPBuffer`；因此本页只证实 MCore wrapper 的 I/O、overflow flag、64 对齐、rerun 协议和 backward 对偶，不声称依赖内部 hop。这条边界在源码里还有一处**自陈**：`HybridEPDispatch.forward` 在请求 fused permute 或自定义 block 数时先 `inspect.signature(HybridEPBuffer.dispatch_with_permute)` 探测有没有 `fuse_permute_dispatch` 形参，没有就 warning 并静默降级——MCore 自己也不假设装的是哪个版本。另一处需要更正旧说法：FP8 dispatch 不是被“拒绝”，而是 `init_hybrid_ep_buffer` 的调用点把 `fp8_dispatch` **硬编码为 `False`** 并注释“Currently, we do not support fp8 dispatch”，配置期不会报错。**增量成本**是全进程共享的 `_hybrid_ep_buffer` 单例（首次 dispatch 按当时 shape 建立，测试需 `reset_hybrid_ep_buffer()` 重置）、64 对齐 slack，以及溢出时的整步重算。适合安装了匹配 HybridEP 分支、希望 fused rearrangement 且能给出容量/shape 预算，并能接受溢出整步重算的环境。

#### 2.5.4 NCCL-EP：TE EpBuffer 的 expert-major capacity view

**它答的压力**：在 TE-native 栈上取得静态收件形状，把动态 shape 的等待彻底去掉。**封顶它的资源**：`recv_capacity_per_rank`——它是硬上界，超出即失败而不是降级。

`_NCCLEPManager.setup_metadata` 将 router 的 probs 转成 top-k indices/weights，再跨过 **MCore ↔ Transformer Engine 依赖边界**。MCore 先构造 `transformer_engine.pytorch.ep.EpBuffer`，再调用模块函数 `transformer_engine.pytorch.ep.ep_dispatch(buffer, tokens, topk_idx, topk_weights)`；`EpBuffer` 不是拥有 `ep_dispatch` 的对象。collective bootstrap/barrier 后，TE 依赖直接给出 `[recv_capacity,H]` 的 expert-major buffer、ragged `tokens_per_expert` 与 per-slot weights；`_NCCLEPManager.get_permuted_hidden_states_by_experts` 只做 static no-op 或按有效计数 narrow，**不调用 MCore 二次 local permute**。dynamic 模式得到本 rank 四条有效 edge（全组八条），每 expert 两条；本地 MLP 消费 per-slot weights，输出已加权后由 `get_restored_hidden_states_by_experts` 再零填充回 capacity 行。combine 调用另一个模块函数 `transformer_engine.pytorch.ep.ep_combine(buffer, expert_out, num_local_tokens=...)`，不接收 route weights，最终恢复每 origin 的 $[2,H]$；static 模式则始终保留固定 capacity view。

**backward 是这条 lane 与另外三条最大的证据差异**：DeepEP/DeepEPv2/HybridEP 的反向都由 MCore 自己的 `torch.autograd.Function` 写出，可以逐个 hop 读；NCCL-EP 的 `nccl_ep_dispatch`/`nccl_ep_combine` 是普通函数，直接调 TE 的模块函数，MCore 里**没有对应的 autograd Function**。因此本页只能断言 TE docstring 自陈其 “autograd-aware”，以及闭环要求“combine gradient 必须回到 expert、dispatch gradient 必须回到 origin”，不能杜撰内部 transpose。

`.item()` 在 dynamic 模式是显式 DtoH，同一 config 在请求 combined-1F1B 的 `overlap_moe_expert_parallel_comm` 时会警告它序列化该路径；static 避开动态 shape wait，却要求 SM100+、TE fused grouped-MLP/op-fuser、`NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`（三条都是 `_NCCLEPManager.__init__` 的 `ValueError`）。bootstrap 先把 local-token ceiling 向 64 的 HT chunk 对齐，再以 top-k×rank factor 计算 receive budget，并进一步按 expert alignment 对齐；这些 slack 都会进入固定 buffer。**增量成本**：NCCL-EP 总是要求 `moe_expert_rank_capacity_factor`（否则构造期 `ValueError`），overflow 是硬错误；每次 dispatch 还新建一个 TE `EpBuffer`，`moe_ncclep_use_symm_mem=True` 当前直接 `NotImplementedError`。适合已经锁定匹配 TE/NCCL 栈并能预算 rank capacity 的路径；要以 static 消除同步还必须满足 Blackwell/fused-op 组合。

### 2.6 训练闭环：从层输出到梯度 ready

前向的 `postprocess` 返回 MoE MLP 输出后，Transformer layer 将它接入 residual，模型再走自己的 logits/loss 路径；这些装配与 objective 由 [[10_megatron_model_structure_analysis]]、[[24_megatron_linear_cross_entropy_analysis]] 负责。对本页而言，loss 的标量梯度首次回到 `MoELayer` output，才是 EP backward 的入口。

三类反向的共同形状是“combine grad 回 expert → expert backward 穿过 $p_{i,e}$ → dispatch grad 回 token owner”，但对偶算子各不相同：AllGather 是 RS↔AG 互换；AllToAll 由 `_AllToAll.backward` 对调 split 再走一次同一算子；四个 Flex backend 里有三个由 MCore 的 autograd Function 显式写出 dispatch↔combine 互调，NCCL-EP 则整段在 TE 内（§2.5）。特别是 DeepEP v1，forward `Buffer.combine` 没有权重实参；权重梯度来自 local expert 路径，随后才由 dispatch backward 的 handle 路线回收。

完成信号分三层，不能互相顶替：

1. **层 forward 完成**：`combine_postprocess` 之后 output 已回到原 `hidden_shape`，可交给 transformer/loss 路径。任何一个 dispatcher helper 的返回都不是层完成——`token_dispatch` 只表示 token buffer 可被本 rank expert 消费，`routed_experts_compute` 只表示本地专家输出已恢复为 combine 可消费的顺序。开启 `moe_layer_recompute` 时 `custom_forward` 会被重放，完成边界不变。
2. **层 backward 完成**：每个 local expert 的参数 `.grad` 与输入 hidden gradient 已就绪。`tests/unit_tests/transformer/moe/test_token_dispatcher.py::MoEModelTestContainer.dispatcher_dropless_test` 对置换/反置换后的 forward 值和 input gradient 做 close 检查，`test_a2a_token_dispatcher.py::TestAlltoAllDispatcher.test_forward_backward` 覆盖多组 TP/EP 形状。
3. **训练 step 完成**：EDP 的梯度同步、optimizer state 和参数更新随后才发生；这是下游所有者，不能从本地 expert backward 推断它已执行。

### 2.7 容量、丢弃与开销结算

容量路径必须区分层级，而且 `apply_router_token_dropping` 看到的是**每 rank 的本地 router 表**（`TopKRouter.routing` 在 `[T_{\mathrm{local}},E]` 上调用它）。其每 expert 容量由 `get_capacity` 给出：

$$
C=\left\lceil\frac{T_{\mathrm{local}}k}{E}f_{\mathrm{cap}}\right\rceil.
$$

`probs` policy 留本地每列最高权重，`position` policy 按本地 routing-map 位置取容量，超出的 map/prob 置零；当 $C$ 已经大于本地 token 数时源码直接跳过整段丢弃。本例每 rank 都有 $T_{\mathrm{local}}=2$，且四个 expert 各恰有一条来自该 rank 的真 edge：

- $f_{\mathrm{cap}}=0.5$：$C=\left\lceil(2\cdot2/4)\cdot0.5\right\rceil=1$。本地 router 表的每个全局 expert 列本来就只有一条真 edge，`topk(probs, k=1, dim=0)` 选中的正是它，因此两 rank 合计的 8 条真 edge **一条也不丢**；不能拿 $T_{\mathrm{global}}=4$ 代入后再假定“每 expert 丢一条”。
- $f_{\mathrm{cap}}=1.5$ 且 `moe_pad_expert_input_to_capacity=True`：$C=\left\lceil(2\cdot2/4)\cdot1.5\right\rceil=2$。每个发送 rank 为每个全局 expert 提供 2 个 slot（1 真 + 1 零）；AllToAll 汇聚两个 origin rank 后，**每个本地 expert** 因而固定为 $C\cdot e=4$ 个 slots，即 **2 真 + 2 零**（`MoEAlltoAllTokenDispatcher.preprocess` 的 drop_and_pad 分支直接写作 `capacity * tp_size * ep_size`）。这会把全组 8 条真 edge 扩成 16 个计算/layout slots；固定 shape 是用额外 compute/memory 买来的。

HybridEP 的 drop_and_pad 分支给出同形的每本地 expert `capacity * group.size()`（permuted 总上界写作 `capacity * group.size() * num_local_experts`）；DeepEP/DeepEPv2 则在配置期拒绝 pad-to-capacity。要与之区分的是 HybridEP/NCCL-EP 的 **rank capacity**：它是整个 manager 收件 buffer 的上限，不是 per-expert router capacity——HybridEP 的超额路线先 drop 并记 `over_budget`、由 `PagedStashRunner` 常规整步 dropless rerun，NCCL-EP 的依赖 overflow 才是 hard trap。

**六条训练数据面的统一开销账。** 下表只列源码能证明的项；性能排序不在其中。

| 方案 | 本例 dispatch→expert-major→combine | 上线的元素（本例） | 同步与必付资源（源码事实） | 硬约束/上限（源码事实） |
|---|---|---|---|---|
| allgather | 每 rank AG 成 $[4,H]$，筛为 4 local edge；MLP 后 RS 回 $[2,H]$ | 原 token：全组物化 $eT_{\mathrm{global}}=8$ 行 + $[4,4]$ map/probs | `permute` 的 `num_out_tokens` 一次 host 取值；无变长 split | 不支持 variable sequence length / sequence packing |
| alltoall | 8 edge 按 `input_splits=[2,2]` 交换；收件后第二排列；reverse A2A + unpermute | route copy：每 rank 发 4 收 4，其中 2 行跨 rank | split 的 DtoH/sync 阶梯（最早 `before_permutation_1`）、两次本地重排 | 容量策略改变 split/shape；偏斜直接放大收件 buffer |
| Flex/DeepEP | 每目标 rank 去重收 3 token，MCore 展开 4 edge；local expert 消费 $p$；unpermute + 无权重 v1 combine | rank-copy：跨 owner 共 3 份 hidden + indices/weights | v1 layout/handle、NVL/RDMA buffer、默认 20 SM、host-visible 收件计数 | probs 强制 float32；`config_map` 只支持列出的 group size，**最大 160**；不支持 pad-to-capacity |
| Flex/DeepEPv2 | Elastic non-expanded 收件（同为 3 token）→ MCore 展开 4 edge → unpermute + elastic combine | 同上 | 每次 dispatch 一次 buffer 查询/扩容、默认 `do_cpu_sync=True`；SM/QP 自动或显式 | ranks≤1024、experts≤2048、experts/rank≤256 且 $E$ 整除 rank 数；不支持 pad-to-capacity |
| Flex/HybridEP | fused dispatch-with-permute 直接给 4 edge；local expert 加权；fused combine-with-unpermute | expert-major slot：4 个/rank（+64 对齐 slack） | 无 rank capacity 时 `tokens_per_expert.sum()` 的 DtoH；全进程共享一个 `_hybrid_ep_buffer` | 超 rank budget 在依赖内 drop 并置 `over_budget`；captured TE whole-MoE graph 下 `RuntimeError`；FP8 dispatch 被硬编码关闭 |
| Flex/NCCL-EP | TE `ep_dispatch` 直接给 capacity/expert-major；local expert 加权；`ep_combine` 回 origin | capacity slot：`recv_capacity_per_rank` 行，其中 4 行有效 | dynamic 的 `.item()` DtoH；static 的固定 slack；每次 dispatch 新建 `EpBuffer`；bootstrap/barrier | 必须给 rank capacity，overflow 硬失败；static 需 SM100+/fused op/env；symm-mem 未实现 |
| 推理 sibling（对照） | 不参与训练闭环，仅登记选择边界（§2.3） | —— | —— | 只在 `transformer_impl="inference_optimized"` 下构造 |

若以 $b_h$ 表示 hidden 标量字节、$b_p$ 表示 route weight 字节，则本例 AllToAll dispatch 的跨 rank**逻辑 payload**为 $K_{\mathrm{remote}}(Hb_h+b_p)=4(Hb_h+b_p)$，combine 为 $4Hb_h$；DeepEP/DeepEPv2 的 non-expanded rank layout 可把 dispatch 的远端 hidden rank-copy 核算为 $3Hb_h$，但 indices/weights、header 和真实链路 hop 另计。AllGather 让每 rank 从 $T_{\mathrm{local}}=2$ 行扩大为 $T_{\mathrm{global}}=4$ 行，全组共物化 $eT_{\mathrm{global}}=8$ 行并携带 map/probs。以上是**分析元素账**，不是 NCCL/DeepEP 的物理链路字节或时延。

---

## 3. 代码实现分析

### 3.1 类与所有权

空心三角表示真实的 Python 继承，其余连线表示构造、持有或调用。`DeepEPBuffer` / `DeepEPElasticBuffer` / `HybridEPBuffer` / `TEEpBuffer` 是图中对 `deep_ep.Buffer`、`deep_ep.ElasticBuffer`、`deep_ep.HybridEPBuffer`、`transformer_engine.pytorch.ep.EpBuffer` 的可读化名称；指向它们的四条边就是 §2.5 那张表里的依赖边界，越过之后 Megatron 源码只能证明传进去的参数与传回来的形状。

```mermaid
classDiagram
direction TB
class BaseMoELayer
class MoELayer
class TopKRouter
class ProcessGroupCollection
class SharedExpertMLP
class TEGroupedMLP
class SequentialMLP
class MoETokenDispatcher
class MoEAllGatherTokenDispatcher
class MoEAlltoAllTokenDispatcher
class MoEFlexTokenDispatcher
class InferenceAllGatherDispatcherBase
class NCCLAllGatherDispatcher
class NVLSAllGatherVDispatcher
class _DispatchManager
class _DeepepManager
class _DeepepV2Manager
class _HybridEPManager
class _NCCLEPManager
class PagedStashRunner
class DeepEPBuffer
class DeepEPElasticBuffer
class HybridEPBuffer
class TEEpBuffer

BaseMoELayer <|-- MoELayer
MoETokenDispatcher <|-- MoEAllGatherTokenDispatcher
MoETokenDispatcher <|-- MoEAlltoAllTokenDispatcher
MoETokenDispatcher <|-- MoEFlexTokenDispatcher
MoEAllGatherTokenDispatcher <|-- InferenceAllGatherDispatcherBase
InferenceAllGatherDispatcherBase <|-- NCCLAllGatherDispatcher
InferenceAllGatherDispatcherBase <|-- NVLSAllGatherVDispatcher
_DispatchManager <|-- _DeepepManager
_DeepepManager <|-- _DeepepV2Manager
_DispatchManager <|-- _HybridEPManager
_DispatchManager <|-- _NCCLEPManager

BaseMoELayer --> ProcessGroupCollection : reads ep expt_tp tp_ep
MoELayer *-- TopKRouter : router
MoELayer *-- MoETokenDispatcher : active dispatcher
MoELayer *-- SharedExpertMLP : optional shared expert
MoELayer --> TEGroupedMLP : grouped experts
MoELayer --> SequentialMLP : per expert fallback
MoELayer --> NCCLAllGatherDispatcher : inference sibling
MoELayer --> NVLSAllGatherVDispatcher : inference sibling
MoEFlexTokenDispatcher *-- _DispatchManager : exactly one live manager
MoEAlltoAllTokenDispatcher --> SharedExpertMLP : overlap stream hooks
_DeepepManager --> DeepEPBuffer : dispatch and combine
_DeepepV2Manager --> DeepEPElasticBuffer : elastic dispatch and combine
_HybridEPManager --> HybridEPBuffer : dispatch with permute
_NCCLEPManager --> TEEpBuffer : ep dispatch and ep combine
PagedStashRunner --> MoEFlexTokenDispatcher : check and reset over budget
```

| 层次 | 责任 | 不负责什么 |
|---|---|---|
| `BaseMoELayer` | 从注入的 `pg_collection` 算出 `num_local_experts` 与连续 `local_expert_indices`，是“哪些专家归我”的唯一裁决点 | 不做任何通信，也不知道用的是哪个 dispatcher |
| `MoELayer` | 构造并选中 router/dispatcher/experts/shared expert；串接六阶段；持有 delayed-wgrad 的 event 与 stream | 不实现搬运算法，不产生 loss，不管 microbatch 调度 |
| `TopKRouter` | gating、top-$k$、group-limited 路由、容量丢弃、aux loss 附着与 expert bias 更新 | 不知道专家在哪个 rank；不做任何 EP collective |
| `MoETokenDispatcher` 三个子类 | 拥有本次 forward 的路线 metadata（`routing_map`、probs、splits、反置换 mapping、`hidden_shape`），并给出 dispatch/combine 两对接口 | 不改变“某 rank 有哪些专家”，不消费路由权重 |
| `_DispatchManager` 四个实现 | 把统一的 $[T_{\mathrm{local}},\mathrm{world},E_{\mathrm{local}}]$ 视图翻译成各依赖的调用，并负责收件布局是否还需 MCore 二次 permute | 不定义 MoELayer 的阶段划分；不拥有 expert 参数 |
| `TEGroupedMLP` / `SequentialMLP` | 在 `tokens_per_expert` 分段上跑本地 expert，并**消费 `permuted_probs`** | 不知道 token 从哪来，也不做反置换 |
| `SharedExpertMLP` | 固定 MLP 与它的 side-stream 状态机（`pre_forward_comm` → FC1 → FC2 → `post_forward_comm` → `get_output`） | 不参与 routing，不占 EP 编号空间 |
| `PagedStashRunner` | 整步包装：汇总 overflow 三项、决定 dropless rerun 或 fail-fast | 不知道 HybridEP 内部为什么溢出，只读 handle flag |

### 3.2 调用流程

**构造与选路阶段。** `TransformerConfig.__post_init__` 先把 deprecated 入口归一（`moe_enable_deepep` → `moe_flex_dispatcher_backend="deepep"`、`moe_router_topk_limited_devices` → `moe_router_group_topk`、`moe_{deepep,hybridep}_num_sms` → `moe_flex_dispatcher_num_sms`），再跑 §5.1 那批交叉校验。建模时 `MoELayer.__init__` 依 `moe_token_dispatcher_type` 三选一构造 dispatcher，Flex 再依 `moe_flex_dispatcher_backend` 四选一构造 manager；`num_local_experts` 与 `local_expert_indices` 已由 `BaseMoELayer.__init__` 定好，两者互不影响。

**一次训练前向。** 下面省略 CUDA graph 分段提前返回、latent MoE 投影与推理 dispatcher 替换；它保留训练 forward 中改变数据所有权或定义完成的调用：

```text
MoELayer.forward
|
+-- shared_experts_compute                         [可选；不重叠时]
+-- route -> TopKRouter.forward/routing             hidden -> probs + routing_map
+-- preprocess -> dispatcher.dispatch_preprocess    [..., H] -> [T_local, H] + metadata
+-- dispatch -> dispatcher.token_dispatch           原 token rank -> expert rank
+-- routed_experts_compute
|   +-- dispatcher.dispatch_postprocess             收件布局 -> 每本地 expert 连续段
|   +-- experts.forward(dispatched, tokens_per_expert, probs)
|   `-- dispatcher.combine_preprocess               解除本地 expert 分段
+-- combine -> dispatcher.token_combine             expert rank -> 原 token rank
`-- postprocess -> dispatcher.combine_postprocess   反置换、reshape、可选加 shared output
```

**同一前向在四个 Flex manager 上的分歧点。** 把上面第 5 行展开，可以看到“统一接口”实际只统一了两处：

```text
dispatcher.token_dispatch  ->  _comm_manager.dispatch(hidden)
|
+-- [deepep]    FusedDispatch.apply -> Buffer.get_dispatch_layout -> Buffer.dispatch
|                 -> non-expanded 收件；dispatch_postprocess 里 MCore permute 展开 expert edge
+-- [deepepv2]  DeepepV2Dispatch.apply -> get_elastic_buffer -> ElasticBuffer.dispatch
|                 -> 同上，do_expand=False；MCore permute 展开
+-- [hybridep]  HybridEPDispatch.apply -> HybridEPBuffer.dispatch_with_permute
|                 -> 依赖直接给 expert-major；dispatch_postprocess 原样转交
`-- [ncclep]    new_nccl_ep_buffer -> te_ep.ep_dispatch(buffer, ...)
                  -> 依赖直接给 capacity 视图；dispatch_postprocess 只 narrow 或 no-op
```

**一次训练反向。** 通信的逆映射不是口头的“再搬一次”，autograd 里每一步都有确定的对偶算子；下面以 AllToAll 主路径为例，右侧标出 Flex 各 lane 在同一位置的替换：

```text
dL/d y_i  (来自 residual / loss 路径)
|
`-- combine_postprocess.backward            permute 的逆：把 y 的梯度散回 edge
    `-- token_combine.backward
        |   [alltoall]  _AllToAll.backward  对调 input/output splits 再走一次 A2A
        |   [allgather] RS 的对偶 AG
        |   [deepep]    FusedCombine.backward -> Buffer.dispatch(handle)
        |   [hybridep]  HybridEPCombine.backward -> dispatch_with_permute(handle)
        |   [ncclep]    TE 内部（MCore 无 autograd Function）
        `-- combine_preprocess.backward     恢复本地 expert 分段
            `-- experts.backward            穿过 p_{i,e}：得到 p*g 与 <g, f_e(t_i)>
                |   [overlap_dispatch_backward_with_experts_wgrad]
                |       _RecordExpertDgradCompletion.backward 记 event
                `-- dispatch.backward
                    |   [alltoall]  _AllToAll.backward 送回 origin
                    |   [deepep]    FusedDispatch.backward -> Buffer.combine(topk_weights=grad)
                    |   [hybridep]  HybridEPDispatch.backward -> combine_with_unpermute
                    `-- dispatch_preprocess.backward   累加同一 t_i 的两份 input gradient
```

`overlap_dispatch_backward_with_experts_wgrad` 只在上图标注处插入 event/stream 协作，不改变任何一条边（机制见 §4.2）。

### 3.3 源码阅读路线

1. 所有权、live 选择与六阶段：`megatron/core/transformer/moe/moe_layer.py::BaseMoELayer.__init__`、`MoELayer.__init__`、`MoELayer.forward`。
2. route/capacity：`megatron/core/transformer/moe/router.py::TopKRouter.routing`；`moe_utils.py::apply_router_token_dropping`、`get_capacity`、`permute`、`unpermute`。
3. 全量复制路线：`megatron/core/transformer/moe/token_dispatcher.py::MoEAllGatherTokenDispatcher.token_dispatch`、`dispatch_postprocess`、`token_combine`。
4. 定向交换路线：`megatron/core/transformer/moe/token_dispatcher.py::MoEAlltoAllTokenDispatcher.preprocess`、`dispatch_postprocess`、`combine_preprocess`、`combine_postprocess`、`_maybe_update_cuda_sync_point`。
5. 四个 Flex manager：`megatron/core/transformer/moe/token_dispatcher.py::MoEFlexTokenDispatcher.__init__`、`_initialize_metadata`、`_DeepepManager`、`_DeepepV2Manager`、`_HybridEPManager`、`_NCCLEPManager`。
6. 权重消费与依赖 autograd：`megatron/core/transformer/moe/moe_layer.py::MoELayer.routed_experts_compute`；`experts.py::TEGroupedMLP.forward`、`SequentialMLP.forward`；`fused_a2a.py::FusedDispatch`、`FusedCombine`、`DeepepV2Dispatch`、`DeepepV2Combine`、`HybridEPDispatch`、`HybridEPCombine`。
7. TE NCCL-EP 模块 API：`megatron/core/transformer/moe/fused_a2a.py::new_nccl_ep_buffer`、`nccl_ep_dispatch`、`nccl_ep_combine`，其边界调用 `transformer_engine.pytorch.ep.ep_dispatch` / `ep_combine`。
8. HybridEP overflow/rerun：`megatron/core/transformer/moe/token_dispatcher.py::_HybridEPManager.dispatch`；`paged_stash.py::PagedStashRunner.check_moe_overflow`、`_raise_if_te_whole_moe_graph_overflow`、`prepare_for_rerun`、`__call__`。
9. shared expert 与 overlap：`megatron/core/transformer/moe/shared_experts.py::SharedExpertMLP.pre_forward_comm`、`linear_fc1_forward_and_act`、`linear_fc2_forward`、`post_forward_comm`、`get_output`；`moe_layer.py::_RecordExpertDgradCompletion`、`_RegisterDelayedWgradForExperts`。
10. 推理 sibling：`megatron/core/transformer/moe/moe_layer.py::MoELayer._setup_inference_mode`、`MoELayer.forward`；`token_dispatcher_inference.py::InferenceAllGatherDispatcherBase`、`NCCLAllGatherDispatcher`、`NVLSAllGatherVDispatcher`；`inference/utils.py::InferenceMode`。
11. 配置与 whole-MoE graph：`megatron/core/transformer/transformer_config.py::TransformerConfig.__post_init__`；`cuda_graph_config.py::validate_moe_cuda_graph_support`。
12. DeepEP 冻结内部：`deep_ep/buffers/legacy.py::Buffer.get_dispatch_layout`、`Buffer.dispatch`、`Buffer.combine`、`Buffer.get_dispatch_config`；`deep_ep/buffers/elastic.py::ElasticBuffer.dispatch`、`ElasticBuffer.combine`；`deep_ep/include/deep_ep/common/layout.cuh::WorkspaceLayout`。
13. 数值边界：`tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestFlexDispatcher.test_forward_backward`、`test_capacity_forward_backward`、`MoEModelTestContainer.dispatcher_capacity_test`、`dispatcher_drop_and_pad_test`；`test_grouped_tensor_dispatcher_numerics.py::TestGroupedTensorDispatcherNumerics`。

---

## 4. 配套机制

### 4.1 Shared expert：从相加到 side-stream 状态机

shared expert 不是被 router 选择的 EP expert：`moe_shared_expert_intermediate_size` 使 `MoELayer` 额外建立 shared MLP，非 overlap 路径在 `shared_experts_compute` 得到其输出，`postprocess` 才相加。它因此**不占 EP 的专家编号空间、不进 dispatch**，但每个 token 都要过它一次。

`moe_shared_expert_overlap=True` 则把它交给 alltoall/flex dispatcher 的 stream 协作；配置验证明确拒绝 allgather。源码里的 overlap 不是“免费并行”标签，而是 `SharedExpertMLP` 状态机：`pre_forward_comm` 先在 side stream 做所需 all-gather/copy；routed-token dispatch 发起后穿插 `linear_fc1_forward_and_act`；combine 发起后穿插 `linear_fc2_forward`，`post_forward_comm`/`get_output` 做 reduce-scatter/reduce，并在主流真正消费 shared output 前等待 event。因此它不改变八条 routed edge，却会新增 shared activation、stream/event，ETP>1 时还可能新增 shared collective；能否遮住 A2A 是 profile 结果。它与 `overlap_moe_expert_parallel_comm` 互斥（`__post_init__` 的 `assert`），因为两者争的是同一段可插入窗口。跨 microbatch/多轴的排队归 [[20_megatron_comm_overlap_analysis]]，combined-1F1B 主时间线归 [[15_megatron_pp_schedulers_analysis]]。

### 4.2 MoE 层内的两条 overlap，以及它们为什么互斥

`overlap_dispatch_backward_with_experts_wgrad` 是 MoELayer 内的 CUDA event/stream 协作，不是另一个 backward 算法。`_RecordExpertDgradCompletion` 被插在 expert 计算之前的前向图里，反向时在 expert dgrad 就绪处记录 event；`_RegisterDelayedWgradForExperts` 插在 dispatch 边界，反向时让专用 delayed-wgrad stream 等待该 dgrad event 后执行 `backward_dw`，主流在参数 grad hook 前再等待 wgrad。于是它试图让“route dX 回 origin”与“local expert dW”并行，完成边界仍是两者都 ready。配置要求 TE≥2.3，并与 `overlap_moe_expert_parallel_comm`、`delay_wgrad_compute` 三方互斥——三条都是 `__post_init__` 的 `assert`，理由是它们都想接管同一份 expert wgrad 的排程时机。

另一条 `overlap_moe_expert_parallel_comm` 才是 combined-1F1B 宿主：要求 torch≥2.6、EP>1、dispatcher 为 alltoall/flex、bf16/fp16；PP>1 时还要求 VPP，且与 shared-expert overlap 互斥。它改变 microbatch 间 work 的排队而不减少 dispatch/combine 元素数；完整 F/B 配对与 `delay_wgrad_compute` 归 [[15_megatron_pp_schedulers_analysis]]。`high_priority_a2a_comm_stream` 只把该路径的 communication stream 建成 CUDA 高优先级，不改变消息量或依赖。NCCL-EP dynamic shape 虽结果正确，却因前述 DtoH 同步失去这条 overlap 收益；这是配置 warning，不是 silent speedup。

### 4.3 whole-MoE CUDA graph 的两个静态形状出口

要把整个 MoE 层捕获进 CUDA graph，必须先消掉动态 shape。`validate_moe_cuda_graph_support` 把这件事写成一个**二选一**：要么 `moe_expert_capacity_factor` 配 `moe_pad_expert_input_to_capacity`（per-expert drop+pad，形状由 $C$ 决定，直接 `return` 放行），要么整组条件同时成立——`cuda_graph_impl="transformer_engine"`、`moe_token_dispatcher_type="flex"`、`moe_flex_dispatcher_backend="hybridep"`、`moe_expert_rank_capacity_factor` 非空、`moe_paged_stash`、`use_transformer_engine_op_fuser`——即 sync-free HybridEP 那条路。半配置一律 `assert` 失败。

两个出口的语义并不相同：drop+pad 在 **router 层**按每个全局 expert 的容量删边/补零（§2.7），sync-free HybridEP 在 **manager 层**按整个 rank 的收件预算截断（§2.5.3）。前者的形状在配置期就能算出，后者要靠 `PagedStashRunner` 在运行期兜底；也正因为图一旦 capture 就绑死了静态 buffer 地址，捕获后的溢出才只能 `RuntimeError` 而不能动态 fallback。`moe_paged_stash` 的底层显存页机制归 [[22_megatron_memory_optimization_analysis]]。

### 4.4 只是相邻、不由本页展开的机制

- **MoE latent projection**（`moe_latent_size`）在 `preprocess` 里用 `fc1_latent_proj` 把 hidden 降到 latent 维再 dispatch，`postprocess` 里用 `fc2_latent_proj` 升回；它改变 dispatcher 与 EP buffer 看到的宽度（`_NCCLEPManager` 显式用 `config.moe_latent_size or config.hidden_size` 定 buffer），但不改变路线语义。
- **Routing replay**（`moe_enable_routing_replay`）让 router 复用/回放路由决定；它改变 `probs`/`routing_map` 的来源，不改变 dispatcher 必须恢复 token ownership 的契约。
- **Grouped GEMM 与 GroupedTensor**（`moe_grouped_gemm`、`moe_use_grouped_tensor`、`moe_single_grouped_weight/bias`、`dense_grouped_gemm`）决定本地 expert 段怎么被执行与存储，owner 是 [[21_megatron_fusion_operators_analysis]]；本页只用到“它们不改变 route 语义”这一条。
- **Permute fusion**（`moe_permute_fusion`、`moe_permute_fusion_into_hybridep`）把置换算子融进 TE 或 HybridEP；前者要求 TE≥2.1，后者要求依赖侧有 `fuse_permute_dispatch` 形参否则 warning 降级。
- **偏斜可观测性**（`log_moe_overload_factor` 走 `MoELayer._maybe_record_overload_factor`，且只在 `self.training` 时记录）归 [[28_megatron_training_stability_observability_analysis]]。

---

## 5. 约束、适用场景与趋势

### 5.1 硬约束与失败边界

| 前提 | 源码边界 | 破坏后的行为 |
|---|---|---|
| EP>1 时必须有专家集合 | `TransformerConfig.__post_init__`：`expert_model_parallel_size > 1 and num_moe_experts is None` | `ValueError`；没有 expert 集合就没有 EP ownership |
| $E$ 能被 EP 度整除 | `BaseMoELayer.__init__` 的 `assert num_moe_experts % ep_size == 0` | `assert` 失败；本页这一路径要求各 rank 连续且等数的 local experts |
| MoE + attention TP 训练必须开 SP | `MoELayer.forward` 开头的 `if self.training and attn_tp_group.size() > 1 and not sequence_parallel` | 直接 `ValueError`；当前实现显式拒绝该性能退化组合（推理不受此限） |
| allgather + variable sequence length | `TransformerConfig.__post_init__` | `ValueError`；allgather 的全域形状假设不支持该组合 |
| sequence packing + MoE | 同上，仅接受 `alltoall`/`flex` | `assert` 失败 |
| flex + DeepEP/DeepEPv2 + pad-to-capacity | `__post_init__` 的 flex 分支 | `ValueError`；这两个 backend 不支持该 shape 策略（HybridEP 支持，`_HybridEPManager.setup_metadata` 有 drop_and_pad 分支） |
| `moe_pad_expert_input_to_capacity` 必须配容量因子 | `__post_init__` | `ValueError` |
| 容量丢弃只与部分均衡策略兼容 | `__post_init__`：`moe_expert_capacity_factor` 要求 load balancing 属于 `aux_loss`/`seq_aux_loss`/`global_aux_loss`/`none` | `ValueError`；`sinkhorn` 与容量路径不可同用 |
| shared-expert overlap + allgather | `__post_init__` 的 shared expert 分支 | `ValueError`；overlap 实现仅接到 alltoall/flex dispatcher |
| DeepEP/DeepEPv2/NCCL-EP 且 TP×EP=1 | `MoEFlexTokenDispatcher.__init__` 三条 `assert tp_size * ep_size > 1` | `assert` 失败；HybridEP 没有同一断言 |
| `moe_expert_rank_capacity_factor` 用于非 HybridEP/NCCL-EP | `__post_init__` | `ValueError`；rank buffer 上限是这两个 manager 的专属契约。HybridEP 用它还额外要求 `use_transformer_engine_op_fuser` 或 `moe_use_grouped_tensor` |
| HybridEP 超出 rank capacity（常规 runner） | `_HybridEPManager.dispatch` 读 handle flag；`PagedStashRunner.__call__` | 依赖内 drop、flag 累计为 `over_budget`；整步结束后清梯度/容量/stash 并 dropless rerun，最多两次尝试（第三次进循环即 `assert num_tries < 2` 失败） |
| HybridEP 超出 rank capacity（已捕获 TE whole-MoE graph） | `PagedStashRunner._raise_if_te_whole_moe_graph_overflow` | `RuntimeError`，不做动态 fallback；已捕获图引用静态 buffer，不能在原图下切换动态地址/形状 |
| NCCL-EP 未设置 rank capacity | `_NCCLEPManager.__init__` | `ValueError`；TE EpBuffer 必须先有接收上界，overflow 硬失败 |
| NCCL-EP static 未满足 SM100+/fused op/env | `_NCCLEPManager.__init__` 三条 `ValueError` | 拒绝构造；固定 ragged expert view 依赖该 kernel 组合 |
| NCCL-EP + GroupedTensor 但无 TE op fuser | `__post_init__` 的 ncclep 分支 | `ValueError` |
| `moe_ncclep_use_symm_mem=True` | `_NCCLEPManager.__init__` | `NotImplementedError`；这是预留项而非可用开关 |
| dispatch-backward overlap 与 combined-1F1B/delayed-wgrad 同开 | `__post_init__` 三条 `assert` | 三方互斥；前者另需 TE≥2.3 |
| whole-MoE CUDA graph 且未走 per-expert drop+pad | `validate_moe_cuda_graph_support` | 按整组条件 `assert` 拒绝半配置（§4.3） |
| `moe_input_jitter_eps` + graphed MoE recompute | `__post_init__` | 明确 unsupported |

两条**需要更正的旧说法**记在这里以免再被引用。其一，`moe_token_dropping` 并没有任何守卫：它的 docstring 写着“currently unsupported so should remain False”，`arguments.py` 把它列进“no CLI argument exists for these”，而 `megatron/core` 里除 dataclass 声明外**没有任何读取点**——设成 `True` 不会报错，只是完全无效；live 的容量路径由 `moe_expert_capacity_factor` / `moe_pad_expert_input_to_capacity` 表达。其二，HybridEP 的 FP8 dispatch 不是被配置校验“拒绝”，而是 `HybridEPDispatch.forward` 调 `init_hybrid_ep_buffer` 时把 `fp8_dispatch` 硬编码为 `False`。

**单测覆盖到哪为止。** `TestFlexDispatcher.test_forward_backward` 把四个 backend 都纳入 dropless 参数化，但 `test_capacity_forward_backward` 只参数化 `deepep`/`deepepv2`/`hybridep`——**NCCL-EP 没有容量路径的单测**；Flex 侧也**没有** drop-and-pad 的参数化（`dispatcher_drop_and_pad_test` 只在 AllToAll 侧被调用）。故“有单测”只证明被实例化的组合，不可推广成任意 backend/shape 的吞吐或数值保证。

### 5.2 何时使用 EP，以及怎么选 dispatcher

| 场景 | 建议 | 原因 |
|---|---|---|
| dense 模型或专家数少到单卡放得下 | 不用 EP | 每层多一次 dispatch/combine 往返是净亏 |
| 专家参数/优化器状态放不下，但激活放得下 | 用 EP，取能放下的最小 $e$ | EP 只沿专家编号切，$e$ 越大跨 rank edge 比例越高 |
| 激活放不下 | EP 救不了 | EP 不做 activation sharding，该找 CP/recompute/offload |
| 长尾偏斜严重 | 先调 router 均衡与 `moe_router_bias_update_rate`，再考虑容量 | 最满 expert 决定该层完成时间；容量是用丢边换尾部 |
| 需要 CUDA graph 捕获整个 MoE 层 | 只有 §4.3 的两条出口 | 其余组合被 `validate_moe_cuda_graph_support` 拒绝 |

选型决策树。它回答的是“给定拓扑与已安装依赖，两个 dispatcher 字段该填什么”，所有性能判断都是**分析推导、未测量**：

```text
EP == 1 且只有 TP ?
|
+-- 是 --> allgather                (默认值；无变长 split、无 DtoH，但会全域复制)
|
`-- 否 --> 需要变长序列 / sequence packing ?
    |
    +-- 是 --> 排除 allgather (配置期直接 ValueError)
    |
    `-- 继续按依赖可用性判定:
        |
        +-- 没装 DeepEP / HybridEP / TE NCCL-EP ?
        |   `-- alltoall            (唯一纯 MCore + NCCL 的定向交换基线)
        |
        +-- 装了 TE NCCL-EP，且能预算 rank capacity ?
        |   `-- flex + ncclep       (TE-native；要消除同步还需 SM100+ 与 fused op，
        |                            否则 dynamic 的 DtoH 会抵消 combined-1F1B 收益)
        |
        +-- 装了 HybridEP 分支，且能接受偶发整步重算 ?
        |   `-- flex + hybridep     (唯一能进 sync-free whole-MoE CUDA graph 的一条)
        |
        +-- 装了 DeepEP v2，或 TP×EP 组大小不在 v1 的 config_map 集合内 ?
        |   `-- flex + deepepv2     (v1 只支持写死的那组 group size，最大 160；
        |                            v2 的上限改由 WorkspaceLayout 的三条断言给)
        |
        `-- 其余 --> flex + deepep  (已部署 v1/NVSHMEM 且 rank 去重有价值时)

三条与本页边界有关的提醒:
  - 前两个判断问的是配置合法性 (会被 __post_init__ 直接拒),
    后四个分支问的是依赖可用性 (装没装、版本对不对), 两者不同源。
  - 每个分支括号里的推荐全部是分析推导: 源码只给约束与资源占用, 不给性能保证。
  - 换 backend 不改变 §2.1 的八条 edge, 只改变谁去重、谁维护 split、谁做第二次 permute。
```

### 5.3 当前演进方向

- **一级 dispatcher 的取值集合在收缩，二级 backend 在扩张。** 冻结 `Literal` 只剩三个一级值，历史的 `alltoall_seq` 在 `9e3adb533` 被删除后只在不可由该 `Literal` 选中的 guard 里留下字符串；同一时期 Flex 的二级取值从一个长到四个，并把 `moe_enable_deepep`、`moe_{deepep,hybridep}_num_sms` 都降级成向新字段转发的兼容入口。由此可推断：读到“Megatron 支持 N 种 dispatcher”的说法时，要先分清它数的是哪一级。
- **通信后端正在从“MCore 写 autograd”滑向“依赖自带 autograd”。** DeepEP/DeepEPv2/HybridEP 三条 lane 的反向都还是 MCore 的 `torch.autograd.Function`，NCCL-EP 已经整段交给 TE 的模块函数。由此可推断：本页这类“逐 hop 可证”的分析，在后续版本上的可证范围只会更小，结论必须带上证据等级才不会过期。
- **静态形状正在从 router 层下移到 manager 层。** per-expert 的 `moe_expert_capacity_factor` 是老出口，per-rank 的 `moe_expert_rank_capacity_factor` 是新出口，并且只有后者配 paged stash 才能进 sync-free whole-MoE CUDA graph。由此可推断：“MoE 是否丢 token”这个问题在新路径上要问两次——router 丢不丢，manager 的 rank budget 够不够。
- **溢出处理从“配置期拒绝”变成“运行期协议”。** `PagedStashRunner` 用一次 all-reduce 汇总三项 overflow、并允许整步 dropless rerun，是本基线里少见的把正确性兜底放在训练循环而不是校验函数里的设计。由此可推断：启用 rank capacity 的训练，其单步耗时分布本身就是双峰的，做性能归因时要先排除重跑步。

---

## 6. 配置契约

### `ModelParallelConfig`

| 字段 | 类型 | 默认 | 契约 |
|---|---|---|---|
| `expert_model_parallel_size` | `int` | `1` | EP 度 $e$，决定每 rank 的 expert ownership；`BaseMoELayer.__init__` 要求 $E$ 被它整除，大于 1 时 `num_moe_experts` 不得为 `None` |
| `expert_tensor_parallel_size` | `Optional[int]` | `None` | ETP 度；决定 expert 内布局恢复的需求（alltoall 路径的 TP AG/RS），不改变 router 使用的全局 expert id |
| `overlap_dispatch_backward_with_experts_wgrad` | `bool` | `False` | 在 MoELayer 内用 event/stream 把 expert wgrad 推迟到 dispatch 反向之后；要求 TE≥2.3，与 `overlap_moe_expert_parallel_comm`、`delay_wgrad_compute` 三方互斥。它不是 PP 调度开关 |

> `ModelParallelConfig` 共 74 个字段，本表列出由本页 owner 的 3 项；其余字段的唯一 owner 见 `docs/coverage/megatron-lm.yaml`。

### `TransformerConfig`

| 字段 | 类型 | 默认 | 契约 |
|---|---|---|---|
| `num_moe_experts` | `Optional[int]` | `None` | 非空时以 MoE 替换相应 MLP，并给 router/EP 定义 $E$ |
| `moe_router_topk` | `int` | `2` | 每 token 的有效 route 边数 $k$，直接决定 token 副本数、权重合并项数与负载 |
| `moe_router_load_balancing_type` | `Union[str, List[str]]` | `"aux_loss"` | 取值 `aux_loss`/`seq_aux_loss`/`global_aux_loss`/`sinkhorn`/`none`，可给列表组合多种 aux loss（此时 `moe_aux_loss_coeff` 必须等长）。只尝试改善 `tokens_per_expert`，不是均衡保证；`sinkhorn` 与容量路径互斥 |
| `moe_aux_loss_coeff` | `Union[float, List[float]]` | `0.0` | 上一项的权重；默认 0 意味着“选了 aux_loss 也可能没有实际梯度贡献”，必须显式给值 |
| `moe_router_score_function` | `Literal['softmax','sigmoid','sqrtsoftplus']` | `"softmax"` | 打分函数；改变 $p_{i,e}$ 的数值路径，不改变 expert owner |
| `moe_router_pre_softmax` | `bool` | `False` | 归一化放在 top-$k$ 之前；关闭时先选后归一 |
| `moe_router_topk_scaling_factor` | `Optional[float]` | `None` | 对路由分数的缩放，**只在 `moe_router_pre_softmax` 打开时生效** |
| `moe_router_dtype` | `Optional[Literal['fp32','fp64']]` | `None` | routing 与加权平均的高精度 dtype；专家数很多时用于稳定性。DeepEP 系 backend 无论如何都会把 probs 转 float32 并 warning |
| `moe_router_group_topk` | `Optional[int]` | `None` | group-limited routing 选中的组数，配合 router groups 使用 |
| `moe_router_topk_limited_devices` | `Optional[int]` | `None` | 已 deprecated；`__post_init__` 把它转写进 `moe_router_group_topk` 并 warning |
| `moe_router_bias_update_rate` | `float` | `1e-3` | expert bias 的更新步长（DeepSeek-V3 同值）；只能缓解偏斜 |
| `moe_input_jitter_eps` | `Optional[float]` | `None` | 在 router 输入加 jitter；与 graphed MoE recomputation 是 unsupported 组合 |
| `moe_enable_routing_replay` | `bool` | `False` | 复用/回放路由决定；改变 `probs`/`routing_map` 的来源，不改变 dispatcher 必须恢复 token ownership 的契约 |
| `moe_token_dispatcher_type` | `Literal['allgather','alltoall','flex']` | `"allgather"` | 一级 live dispatcher 选择开关；其他值在 `MoELayer.__init__` `raise ValueError` |
| `moe_flex_dispatcher_backend` | `Literal['deepep','deepepv2','hybridep','ncclep']` | `"deepep"` | 仅在训练态 flex 主轴选择 manager，受各 backend 的构造期 guard 约束；它不是推理 sibling dispatcher |
| `moe_enable_deepep` | `bool` | `False` | deprecated 兼容入口：dispatcher 必须已是 `flex` 且 backend 不是 `deepepv2`/`hybridep`，验证阶段才转写为 `deepep` 并 warning，否则 `ValueError` |
| `moe_expert_capacity_factor` | `Optional[float]` | `None` | per-expert 容量因子 $f_{\mathrm{cap}}$，在 **router 层**按本 rank 的路由表删边（§2.7）；负值被归一成 `None`；要求 load balancing 不是 `sinkhorn` |
| `moe_pad_expert_input_to_capacity` | `bool` | `False` | 把每个 expert 补齐到 $C$，换取固定 shape；必须先设容量因子，且 DeepEP/DeepEPv2 backend 下被拒 |
| `moe_token_drop_policy` | `Literal['probs','position']` | `"probs"` | 容量内保留哪些边：按概率取每列 top-$C$，或按 routing-map 位置取前 $C$ |
| `moe_token_dropping` | `bool` | `False` | 历史兼容字段：docstring 标注 unsupported，无 CLI flag，`megatron/core` 内**没有读取点也没有断言**——设 `True` 无效而非报错 |
| `moe_pad_experts_for_cuda_graph_inference` | `bool` | `False` | decode 期切到固定 drop/pad 形状以规避 D2H shape 同步，容量取推理期可能的最大值故不真丢 token；消费方在 inference 控制器 |
| `moe_shared_expert_intermediate_size` | `Optional[int]` | `None` | 非空时建立 shared expert，其值是 `num_shared_experts × 每个的 ffn_size`；非正值 `ValueError` |
| `moe_shared_expert_overlap` | `bool` | `False` | 把 shared expert 交给 alltoall/flex dispatcher 的 side stream（§4.1）；allgather 与 `overlap_moe_expert_parallel_comm` 下均被拒 |
| `moe_shared_expert_gate` | `bool` | `False` | 给已启用的 shared expert 加一个标量 gate |
| `use_grouped_gemm_for_shared_expert` | `bool` | `False` | 让 shared expert 走 `GroupedLinear(num_groups=1)` 以触发 TE grouped SwiGLU 融合；只在 shared expert 已启用时生效 |
| `moe_shared_expert_glu_interleave_size` | `Optional[int]` | `None` | shared expert 的 GLU 用 block-interleaved 布局；**只在上一项开启时生效** |
| `moe_permute_fusion` | `bool` | `False` | 通用 token rearrangement 融合；`__post_init__` 检查 TE≥2.1 提供的五个 fused 置换算子，缺一即 `ValueError` |
| `moe_permute_fusion_into_hybridep` | `bool` | `False` | 把置换融进 HybridEP 的 dispatch/combine；依赖侧无 `fuse_permute_dispatch` 形参时 warning 并降级为不融合 |
| `moe_hybridep_pad_variable_tokens` | `bool` | `False` | dispatch 前把不等长的 local token 数 all-reduce 取 max 再按 64 向上对齐，combine 后截回原长度 |
| `moe_latent_size` | `Optional[int]` | `None` | MoE latent projection 维度；非空时 dispatcher 与 EP buffer 都按 latent 维而非 `hidden_size` 计算宽度 |
| `moe_flex_dispatcher_num_sms` | `Optional[int]` | `None` | 统一控制四个 flex backend 的 dispatch/combine SM 预算；`None` 时 DeepEP v1 取 20、DeepEPv2 取 0（自动推导） |
| `moe_deepep_num_sms` | `Optional[int]` | `None` | 已 deprecated，`__post_init__` 向上一项转发；与 `moe_hybridep_num_sms` 取值冲突时 `ValueError` |
| `moe_hybridep_num_sms` | `Optional[int]` | `None` | 同上，已 deprecated 并向 `moe_flex_dispatcher_num_sms` 转发 |
| `moe_hybridep_num_blocks_permute` | `Optional[int]` | `None` | HybridEP permute 部分的 CUDA block 数；融合开启时等价于 SM 数（每 SM 一个 block） |
| `moe_hybridep_num_blocks_unpermute` | `Optional[int]` | `None` | 同上，对应 unpermute 部分 |
| `moe_hybridep_num_sms_preprocessing` | `int` | `108` | HybridEP metadata scan kernel 的 SM 预算；是资源旋钮而非收益保证 |
| `moe_ncclep_static_shape` | `bool` | `False` | 请求固定 receive/expert view 以避开动态 shape wait；需 SM100+、TE fused grouped-MLP 或 `moe_grouped_gemm`、以及 `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` |
| `moe_ncclep_use_symm_mem` | `bool` | `False` | symmetric-memory 零拷贝 payload 的预留项；设 `True` 直接 `NotImplementedError` |
| `high_priority_a2a_comm_stream` | `bool` | `False` | 把 combined-1F1B 的 A2A communication stream 建成 CUDA 高优先级；不改变消息量或依赖 |
| `dense_grouped_gemm` | `bool` | `False` | 给 dense MLP 选择 `GroupedLinear(num_groups=1)` 以触发 SM100+ MXFP8 融合；是相邻 compute-path 开关，不改变任何 EP collective |

> `TransformerConfig` 共 265 个字段，本表列出由本页 owner 的 40 项；其余字段的唯一 owner 见 `docs/coverage/megatron-lm.yaml`。本页正文另外解释了三个由别页 owner 的字段并链回其 owner：`moe_expert_rank_capacity_factor`（[[39_megatron_moe_training_optimization_analysis]]）、`moe_paged_stash`（[[22_megatron_memory_optimization_analysis]]）、`moe_apply_probs_on_input`（[[21_megatron_fusion_operators_analysis]]）。

三张 SVG 均由 `tools/figs/svg/megatron_ep_figures.mjs` 从同一组算例参数生成；`tools/figs/svg/lib/megatron_ep_figures.test.mjs` 同时读取 Markdown、生成器与 SVG，锁定 $T_{\mathrm{global}}=4$、$T_{\mathrm{local}}=2$/rank、8 条 route edge、4 条 remote expert-edge、3 份 remote rank-copy、两组 capacity 结果、一级三 dispatcher、Flex 四 backend、推理 sibling 选择边界与 backward 对偶。

## Related Pages

- [[10_megatron_model_structure_analysis]] — MoE 如何被装配为 transformer 的 MLP 位置，以及 loss 之前/之后的模型边界。
- [[12_megatron_tp_analysis]] — 对照 ETP 所需的 TP 布局与 collective 语义，不把专家内线性机制重复到本页。
- [[15_megatron_pp_schedulers_analysis]] — EP A2A 与 expert wgrad overlap 何时进入 combined-1F1B 的调度。
- [[17_megatron_parallelism_orchestration_analysis]] — EP/ETP/EDP 与组合通信组的构造、rank 坐标及注入。
- [[20_megatron_comm_overlap_analysis]] — 多轴 collective 与 compute overlap 的全局竞争时间线。
- [[39_megatron_moe_training_optimization_analysis]] — 将 dispatcher、容量、均衡与硬件条件放入工程选型总纲。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] — 返回本域全部页面的主题索引。
