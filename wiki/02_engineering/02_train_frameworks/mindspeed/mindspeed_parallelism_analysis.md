# MindSpeed 并行特性 — 划分结构源码级分析

> **代码基线**:MindSpeed core `master` @ `1432cb09`(猴补丁 Megatron `core_r0.17.0`)· MindSpeed-LLM `master` @ `0c16322d` · 2026-06-23
> 本页只讲 MindSpeed 在各并行维度上**怎么切、切到哪个 rank、谁替换了 Megatron 的哪段划分逻辑**(TP/PP/MoE-EP/DP/分层解耦)。每条非平凡结论都带 `file:line`,行号均经实际打开核对(路径相对各自仓库根:`mindspeed/...` 属 MindSpeed core,`mindspeed_llm/...` 属 MindSpeed-LLM)。
> **范围边界**:CP 上下文并行已独立成页 [[mindspeed_context_parallel_analysis]](见 §2);这些并行结构上的**通算掩盖**(send-recv overlap、MC2、CoC、MoE fb-overlap、DualPipeV/RiPipe 调度)归 [[mindspeed_comm_overlap_analysis]];KV-cache/重计算/Swap 等省显存手段归 [[mindspeed_memory_optimization_analysis]];融合算子归 [[mindspeed_ascend_affinity_analysis]]。本页只在交界处交叉引用。属 [[mindspeed/index]] 系列。

---

## 1. 总览

MindSpeed 不新造并行框架,而是把 Megatron 原生 5D 并行(TP-CP-EP-DP-PP)里"切分点"的实现按需替换成昇腾亲和版本。每个并行特性都是一个 `MindSpeedFeature`:`register_patches` 把 Megatron 的某个划分函数/类换成 `mindspeed/core/<domain>/` 下的实现,真正的机制在 core 里。

| 维度 | 解决的瓶颈 | MindSpeed 特性(feature) | 替换的 Megatron 划分点 |
|------|-----------|------------------------------|----------------------|
| **CP** 上下文并行 | 长序列 attention 激活/算力随 $s$ 线性涨 | ContextParallel / Ulysses / Adaptive / KvCache | `DotProductAttention`、`get_batch_on_this_cp_rank`、CP 通信组 |
| **TP** 张量并行 | 单卡放不下大权重;TP=2 的幂的对齐约束 | TP-2D / UnalignedLinear / VocabParallel(ReplaceIndexPut) | `ColumnParallelLinear`/`RowParallelLinear`、`VocabParallelEmbedding.forward` |
| **PP** 流水划分 | 各 stage 层数不均 → 气泡;embed/loss/MTP 非 transformer 层放置 | Noop / Unaligned / PPLayout / MultiParameter / VariableSeq / num-layer-list | `get_num_layers_to_build`、`get_transformer_layer_offset` |
| **MoE-EP** 专家并行 | 专家算力分散;TP 切专家低效;专家负载倾斜 | GMM / TpExtendEp / ExpertsPlacement / BalancedMoE / SharedExpert / FixRouter | `GroupedMLP`、`TopKRouter.routing`、`MoELayer` |
| **DP & 分片** | 优化器/梯度/参数冗余;Ascend 桶对齐 | LayerZeRO / CustomFSDP / TorchFSDP2 / BufferPad / ResetBucketOrder | `setup_model_and_optimizer`、`_ParamAndGradBuffer`、DDP |
| **分层解耦** | 边-云异构集群,边端 TP/DP 小于云端 | U-shaped-split / VDP / VTP / mamba-CP(MindSpeed-LLM) | `forward_backward_pipelining_*`、`initialize_model_parallel`、p2p |

```mermaid
flowchart LR
    subgraph FM["features_manager/*(薄:register_args + register_patches)"]
      F2[tensor_parallel/*]; F3[pipeline_parallel/*]
      F4[moe/*]; F5[distributed/*]; F6["LLM: layerwise_disaggregated/*"]
    end
    subgraph CORE["mindspeed/core/<domain>(厚:真正的切分机制)"]
      C2["tensor_parallel/<br/>tp_2d / unaligned_layers / vocab_parallel"]
      C3["pipeline_parallel/<br/>noop / unaligned / layout"]
      C4["transformer/moe/<br/>gmm / tp_extend_ep / expert_placement"]
      C5["distributed/<br/>layerzero / custom_fsdp / buffer_pad"]
    end
    F2-->C2; F3-->C3; F4-->C4; F5-->C5
    CORE -->|register_patch| MEGA["Megatron core_r0.17.0(宿主)"]
```

> **怎么读这些特性**:`features_manager/<domain>/*.py` 都很薄,只做两件事——`register_args` 注册 CLI 开关、`register_patches(pm, args)` 把 Megatron 的某个全限定名换成 core 实现。读懂一个并行特性 = 顺着它 `register_patch('megatron....X', mindspeed....Y)` 这一行,跳到 `mindspeed/core/<domain>/` 里看 `Y` 的真正机制。本页所有结论落在被替换的 `Y` 上,而非 feature 壳子。

### 记号约定

| 符号 | 含义 |
|------|------|
| $N$ / $tp$ | TP 度;TP-2D 下 $N=x\cdot y$($x=$tp_x、$y=$tp_y) |
| $s$ / $b$ / $h$ / $E$ | 序列长 / micro-batch / hidden / linear 输出宽 |
| $cp$ / $ep$ / $pp$ / $dp$ | CP / EP / PP / DP 度 |
| $V$ | SP 后单卡激活体量 $\frac{b\,s\,h}{N\cdot cp}$(TP-2D 通信代数用) |
| `pp_rank` / `vpp_rank` | 流水 stage 号 / 虚拟流水 chunk 号 |
| edge / cloud | 分层解耦的边端(少卡弱)/ 云端(多卡强) |

---

## 2. CP — 上下文并行(已独立成页)

CP 是 MindSpeed 改动最厚的并行维度(Ulysses / Ring 双环 / Hybrid / Adaptive / KV-cache 五套搬运策略 + 2·cp 因果配对切分 + RoPE 切分不变量 + 通信量代数与选型),已**独立成页深挖**,见 **[[mindspeed_context_parallel_analysis]]**。一句话定位:它把序列维 $s$ 切到 CP 组,让每卡只算 $s/cp$ 的 attention;难点是 attention 全局算子需跨卡交换 Q/K/V,MindSpeed 用 `--context-parallel-algo` 在五套自研内核(全在 PyTorch+`npu_fusion_attention` 层,不走 TE)间运行期分派。本页不再展开。

---

## 3. TP — 张量并行

### 3.1 TP-2D:权重的二维网格切分

**命题**:Megatron 一维 TP 把每个权重沿一个维度切 $N=$tp 份,$N$ 增大时 all-reduce 的通信组也涨到 $N$,"通信税" $\frac{N-1}{N}$ 趋近 1、且整组都要参与一次大集合通信(易跨节点、不可重叠)。TP-2D 把 TP 组重排成 $x\times y$ 网格($\text{tp\_x}\times\text{tp\_y}$),**同一个权重沿两个维度同时切**,把单次大 all-reduce 拆成 X 向 AllGather + Y 向 ReduceScatter 两组**更小**的集合通信。约束 `tp = tp_x·tp_y`、不支持 MoE(`tp_2d.py:39-42`)。

```mermaid
flowchart LR
    subgraph MESH["TP 组重排成 tp_x × tp_y 网格(以 2×4 为例,tp=8)"]
      direction TB
      R0["x0y0"]; R1["x0y1"]; R2["x0y2"]; R3["x0y3"]
      R4["x1y0"]; R5["x1y1"]; R6["x1y2"]; R7["x1y3"]
    end
    R0 -.->|"X 轴 AllGather(组大小 x)"| R4
    R0 -.->|"Y 轴 ReduceScatter(组大小 y)"| R3
```

切分形状由 `Linear2DSplitAlongFirstDim.forward` 的注释直给(`linear_2d_split_along_first_dim.py:64-65`):激活 `[s/(x·cp), b, h/y]`、权重 `[h/y, E/x]`——**隐藏维 $h$ 按 $y$ 切、输出维 $E$ 按 $x$ 切**。前向一条主路(非 overlap/coc 分支)就是 "AG → matmul → RS"(`:128-137`):

```python
# linear_2d_split_along_first_dim.py:129-137
# [s/(x*cp), b, H/y] -> [s/cp, b, H/y]            ← X 向 AllGather 把序列补回
total_input = sync_gather_along_first_dim(activation_input, ag_comm_intf, ...)
# [s/cp, b, H/y] @ [H/y, e/x] -> [s/cp, b, e/x]   ← 本地 matmul
matmul_res = torch.matmul(total_input, weight.t())
# [s/cp, b, E/x] -> [s/(y*cp), b, E/x]            ← Y 向 ReduceScatter 把部分和聚 + 重切序列
matmul_res = sync_reduce_scatter_along_first_dim(matmul_res, rs_comm_intf)
```

`ParallelLinear2D`(`parallel_linear_2d.py:24-85`)封装该 autograd,持 `ag_comm_intf`(X 向 AllGather 接口)与 `rs_comm_intf`(Y 向 ReduceScatter 接口)两套通信域(`:76-79`)。X/Y 及 ND1/ND2 通信组在 `parallel_state_2d.py` 构造(`_TP_X_PARALLEL_RING_RANKS`/`_TP_Y_...` 与 `_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1/ND2_*`,`:15-28`)。Feature 把 MLP、SelfAttention、Embedding、norm、PP 张量形状全换成 2D 版本(`tp_2d.py:44-94`),并与 CP 联动出 `TensorParallelYUnionCP`(tp_y 与 CP 合并成一个域,见 [[mindspeed_context_parallel_analysis]] §1)。

```
1D TP(组大小 N):        [一次 all-reduce over N 张卡]  通信税 (N-1)/N,组跨节点、不可叠 matmul
TP-2D(x·y=N):
   X 轴 AllGather  ──┐  组大小 x
                      ├─ matmul([s/cp,b,h/y]·[h/y,E/x])
   Y 轴 ReduceScatter─┘  组大小 y
   每个集合只在 √N 量级的小组内,通信税降到 (x-1)/x 或 (y-1)/y,且 AG 可与 matmul 重叠
```

**算法代数**:设 SP 后单卡激活体量 $V=\frac{b\,s\,h}{N\cdot cp}$。1D TP+SP 的一条 linear 前向 AllGather + 反向 ReduceScatter 各搬 $\approx\frac{N-1}{N}\cdot N V=(N-1)V$,通信组大小 $N$。TP-2D 把同一逻辑流量拆到两个正交小组:

$$
V_{\text{AG}_x}\approx\frac{x-1}{x}\cdot xV=(x-1)V,\qquad
V_{\text{RS}_y}\approx\frac{y-1}{y}\cdot yV=(y-1)V
$$

总量 $(x-1)+(y-1)=x+y-2$,对照 1D 的 $N-1=xy-1$:**当 $x=y=\sqrt N$ 时从 $\sim N$ 降到 $\sim 2\sqrt N$**,且每个集合的"爆炸半径"从 $N$ 卡缩到 $\sqrt N$ 卡(可压进节点内 HCCS 域),外加 `enable_overlap_ag_with_matmul`/`enable_overlap_matmul_with_rs` 把 AG/RS 叠进 matmul(重叠细节归 [[mindspeed_comm_overlap_analysis]])。这是 TP-2D 在大 TP 下赢 1D 的根因。

> **worked shape walk**(fc1,$s{=}4096,b{=}1,h{=}8192,E{=}32768,x{=}2,y{=}4,cp{=}1$):本卡激活 `[s/x, b, h/y]=[2048,1,2048]` → **AG(x=2)** 沿序列补回 `[4096,1,2048]` → matmul 权重 `[h/y,E/x]=[2048,16384]` 得 `[4096,1,16384]` → **RS(y=4)** 沿序列聚部分和 `[1024,1,16384]`。两个集合分别只在 2 卡、4 卡的小组内,无任何 8 卡大 all-reduce。

**ND1/ND2 — 两条 linear 用正交的主切轴**:TP-2D 的两层 linear(列并行 fc1、行并行 fc2)不能都按同一维切,否则退化回 1D。组构造 `initialize_ndmm_parallel_group(nd1_dim1_size=tp_x, nd2_dim1_size=tp_y)`(`parallel_state_2d.py:94-100`)给出两套独立通信域:**ND1**(第一条 linear)主切轴是 $x$、**ND2**(第二条)主切轴是 $y$,各自再有 dim1/dim2 子组(`_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1/DIM2`、`ND2_*`,`:18-21`)。于是 fc1 的输出切轴正是 fc2 的输入切轴,链式相消、无需在两层之间 all-gather 回全量——这是 "2D" 真正省通信的结构前提。开 `tp_2d` 时还顺带建 `TensorParallelYUnionCP`(tp_y ∪ cp 联合域,`:101-116`),供 CP 换组用(见 [[mindspeed_context_parallel_analysis]])。

**反向也是双集合流水**:`backward`(`linear_2d_split_along_first_dim.py:142-150` 的时序注释)把反向拆成四步流水,刻意把两路 AllGather、MM、ReduceScatter 在时间轴上错开以求重叠:

```
time ──────────────────────────────────────────────────────────►
| AG(grad_o, Y|X)
|            AG(activation_input, X|Y)
|            part_grad_act = MM(tot_grad_o, weight)
|                                          RS(part_grad_act, X|Y)   → grad_input
|                                          MM(tot_grad_o^T, tot_act_input) → grad_weight
```

先 AG(grad_output, 沿 Y) + AG(activation_input, 沿 X) 两路异步取全量,再 `MM(tot_grad_o, weight)` 算 partial grad_input,随即 RS(沿 X)聚成完整 grad_input,同时 `MM(tot_grad_o^T, tot_act_input)` 算 grad_weight(`:199-231`,grad-accum-fusion 走 `wgrad_gemm_accum_fp32/fp16`)。即前向是 AG→MM→RS、反向是 (AG‖AG)→MM→(RS‖MM),两个方向都把集合通信与 matmul 交错。

**CoC 融合核 — AG+MM+RS 合一**:开 `coc_fused_kernel` 时,前向三步不再分立,而是一次 `coc_ops.all_gather_matmul_reduce_scatter(activation_input, weight, ...)` 把 AllGather、matmul、ReduceScatter 融进单个昇腾算子(`linear_2d_split_along_first_dim.py:121-127`),省掉中间张量的两次 HBM 往返。物理 rank 摆放由 `get_comm_domain_rank(devid, ag_size, rs_size)`(`:28-38`)按 TFTF/FTFT 两种布局把 device id 映射到 `(comm_domain, coc_rank)`——例如 RS=8 时 `[0..7],[8..15]` 连续成 RS 组、RS=2 时 `[0,8],[1,9]...` 跨步成 RS 组。这决定 X/Y 两轴各落在哪些物理卡上,目标是把高频的 AG 轴压进同一 HCCS 域。

### 3.2 非对齐 Linear(UnalignedLinear)

**命题**:Megatron 要求 `output_size`、注意力头数能被 TP 整除;面对非 2 的幂头数 / 非整除维度只能 padding 浪费算力。UnalignedLinear 让每个 rank 持**不等长**切片。`unaligned_linear_feature.register_patches` 把 `ColumnParallelLinear`/`RowParallelLinear` 换成 `Unaligned*Adaptor`,并把 `megatron.core.utils.divide` 换成 `divide_adaptor`(放开整除断言),以适配 MHA/GQA 头的非均匀分布(`unaligned_linear_feature.py:36-42`)。

核心切法在 `UnalignedColumnParallelLinear.__init__`:本 rank 的输出宽度由 `unaligned_divide(output_size, world_size, rank)` 算出(`unaligned_column_parallel_linear.py:73`),其逻辑是把不能整除的**余数 +1 分摊到前 `numerator % world_size` 个 rank**(`unaligned_utils.py:7-11`):

```python
# unaligned_utils.py:8-11
res = numerator // world_size
if rank < numerator % world_size:   # 前 r 个 rank 各多拿 1
    res += 1
```

> **worked example**:`output_size=70, TP=8` → `70//8=8`、`70%8=6` → rank0–5 各 9、rank6–7 各 8,合计 $6\times9+2\times8=70$,零 padding。GQA 头不整除时同法走 `num_query_groups`(`:70-71`)。与 MC2、TP-2D、MoE 互斥(`unaligned_linear_feature.py:20-24`)。

### 3.3 Vocab 并行 ReplaceIndexPut

**命题**:词表并行 embedding 的 Megatron 实现对 mask 外 token 用 `index_put_` 置零,该算子在 NPU 上非确定且精度欠佳。`ReplaceIndexPutFeature` 把 `VocabParallelEmbedding.forward` 换成 `vocab_parallel_embedding_forward_impl`(`vocab_parallel.py:22-56`),改用纯张量算子:

```python
# vocab_parallel.py:29-47
input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)  # 越界 mask
masked_input = input_.clone() - self.vocab_start_index;  masked_input *= ~input_mask  # 平移到本 rank 词表区间并清零
output_parallel = self.weight[masked_input] if self.deterministic_mode \
                  else F.embedding(masked_input, self.weight)   # 确定性取行 / 否则 F.embedding(bf16 累加更准)
output_parallel *= ~input_mask[..., None]   # 屏蔽越界行
```

最后按是否序列并行选 `reduce_scatter` 或 `all_reduce` 汇聚跨 TP 的 embedding(`:49-55`)。整条路径无 `index_put_`,在 NPU 上确定可复现。两个分支各有取舍:

| 分支 | 触发 | 取数算子 | 取舍 |
|------|------|---------|------|
| `deterministic_mode` | 开确定性 | `weight[masked_input]`(`:39`) | 可复现,反向确定 |
| 否则 | 默认 | `F.embedding`(`:43`) | bf16 累加精度更高,但反向非确定 |
| `reduce_scatter_embeddings` | 开 SP | `reduce_scatter`(`:52`) | 输出已沿序列切,省一次 scatter |
| 否则 | 无 SP | `all_reduce`(`:55`) | 每卡全量 embedding |

---

## 4. PP 划分

> 这里只覆盖**层如何分到 stage**;DualPipeV/RiPipe/optimize-p2p 等**调度与气泡掩盖**归 [[mindspeed_comm_overlap_analysis]]。

**命题**:Megatron 默认每 stage 等分层数,但真实模型 stage 不等重——首尾要带 embedding/loss/MTP、MoE 层更贵,等分必致气泡。MindSpeed 给出一组"重新定义每 stage 装什么层"的特性,从最简单的逐 stage 层数列表到字符串布局 DSL。

### 4.1 noop 占位层 — 用零算力假层拉平负载

`--noop-layers 3,7` 在指定位置插入 `NoopTransformerLayer`(`noop_layers/adaptor.py:32-83`):它继承 `MegatronModule`、**无参数**,forward 只 `return hidden_states.clone(), context`(`:83`)——不含算力、不占权重。作用是把真实层"挤"到更均衡的 stage 分布(例如让某个本来超载的 stage 名义上多几层"空层",真实层因此后移)。同时要修正 FLOPs 统计与 MoE 指标的层号映射(`mindspeed_track_moe_metrics` 透传 `noop_layers`,`:143-159`)。

```
PP=2、24 层,首 stage 还要扛 embedding → 实际更重:
  朴素等分:  stage0 [L0..L11]+embed(重)   stage1 [L12..L23]       ← stage0 撑爆
  noop 3,7:  stage0 [L0,L1,noop,noop,L2..L9]+embed  stage1 [L10..L23]
             ↑ 两个空层占名额,真实层后移到 stage1,首 stage 真实算力↓、更均衡
```

### 4.2 层数自定义:从列表到二维到 DSL

| 特性 | 粒度 | 机制(已核对) |
|------|------|------|
| **num-layer-list**(LLM) | 逐 PP-stage 一个数 | `--num-layer-list 4,4,4,4`;`pre_validate` 把 `num_layers` 临时设为列表长度骗过 Megatron 校验(`num_layer_list.py:22`),`post_validate` 还原(`:40-43`);patch `get_num_layers_to_build`+`_get_layer_offset`(`:54-57`) |
| **Unaligned** | **PP×VPP 二维**嵌套 | `get_num_layers_to_build_unaligned` 直接返回 `layers[pp_rank][vpp_rank]`(`unaligned_pipeline.py:4-10`);offset 用**列优先前缀和**(VPP 内层先连续,`:23-35`) |

`get_layer_offset_pp_vp_unaligned`(`unaligned_pipeline.py:23-35`)算 offset 时**先按 VPP 列、再按 PP 行**展平(`for j in col: for i in row`)做前缀和——因为交错 VPP 调度里执行顺序是"所有 stage 的 chunk0,再所有 stage 的 chunk1",同一 model-chunk 的层在全局编号上必须连续。

> **worked example**:`layers=[[2,1],[3,2]]`(2 PP × 2 VPP,row=pp、col=vpp)。列优先展平 = `[2,3,1,2]`(先 vpp0 的两个 stage、再 vpp1),前缀和 `[0,2,5,6,8]` reshape 回 `[[0,5],[2,6]]`:stage0 的 chunk0 起始层 0、chunk1 起始层 5;stage1 的 chunk0 起始 2、chunk1 起始 6。
| **PPLayout** | 字符串 DSL | 用 `"E\|(t\|)*3,m\|m\|\|L"` 描述整张布局(E=embed、t=transformer、m=MTP、L=loss、`\|` 分 stage、`*` 重复,`pipeline_model_parallel_layout_feature.py:30-31`),`PipelineParallelLayerLayout` 解析 num_stages 并反推 VPP(`:64-69`);patch `get_num_layers_to_build`+`get_transformer_layer_offset`(`:204-211`);与 dualpipev/noop 互斥(`:103-110`) |

### 4.3 跨 stage 张量:多参数与变长

- **MultiParameter**:让 stage 间能传**多个张量**(Megatron 默认只传一个 hidden)。`pipeline_tensor_shapes` 描述跨界张量形状列表,缺省按精度推出一个默认项(`multi_parameter.py:43-52`):

  ```python
  # multi_parameter.py:43-52
  tensor_shape = (int(args.seq_length / args.context_parallel_size), args.micro_batch_size, args.hidden_size)
  dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
  args.pipeline_tensor_shapes = [{"shape": tensor_shape, "dtype": dtype}]
  ```
  随后 patch 全套 `send/recv_forward/backward`、`get_tensor_shapes`、`forward/backward_step`(`:74-118`),使流水线 P2P 按形状列表**逐个**收发(而非单一 hidden)。与 dualpipev、`moe_fb_overlap` 互斥(`:36-42`)。
- **VariableSeq**:支持 microbatch 间**变长序列**,patch `_communicate`/`_communicate_shapes` 在 P2P 前先同步动态 shape(`variable_seq_length.py:58-65`)——收方无法预知发方序列长度,故每次 P2P 先传一个 shape 头、再传数据。仅 MoE 场景启用(无 `num_moe_experts` 时 `pre_validate` 强制关闭,`:40-41`),因为非变长场景下这层 shape 同步纯属额外开销。

---

## 5. MoE-EP 结构

**命题**:专家并行把 `num_experts` 个专家分到 EP 组各 rank,token 经 router 后 all-to-all 到目标专家所在 rank 计算(EP 总览与三种 dispatcher 对照见 [[megatron_ep_analysis]])。MindSpeed 在"专家怎么算(GMM)、EP 怎么扩(tp-extend-ep)、负载怎么均(placement/balanced)"三处做文章。

```mermaid
flowchart LR
    T["tokens [s,b,h]"] --> R["Router topk routing_map"]
    R --> A2A["all-to-all-v(ep 或 tp×ep 域)"]
    A2A --> P["permute 按专家分组"]
    P --> G["GMM 分组矩阵乘 fc1→act→fc2"]
    G --> UP["unpermute + all-to-all 回原 rank"]
    R -.->|tp_extend_ep:跳过 tp-gather| A2A
    G -.->|placement/balanced:热专家重排/复制| G
```

### 5.1 GMM:EP 的计算原语

一张卡上有多个本地专家、每个是独立小 GEMM,逐个算利用率低。`MoEGmmFeature` 把 `GroupedMLP` 换成 `MindSpeedGmmExperts`(`gmm.py:25-27`),用**分组矩阵乘**一次算完所有本地专家。`GmmExpertsImpl.forward` 把本地专家权重 reshape 成 `[num_local_experts, h, -1]`(`gmm/experts.py:84-85`)后,fc1/fc2 各调一次 `gg.ops.gmm(..., tokens_per_expert, ...)`(`:95`、`:116`),按 `tokens_per_expert` 把变长分组喂进**单个** grouped-matmul 内核:

```python
# gmm/experts.py:95,116
fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, ...)  # :95
...
fc2_output = gg.ops.gmm(intermediate_parallel,        w2, tokens_per_expert, trans_b=False, ...)  # :116
```

```
逐专家 loop(慢):  [E0 GEMM][E1 GEMM][E2 GEMM][E3 GEMM]   4 个 kernel,launch 开销 ×4
GMM(快):          tokens_per_expert=[t0,t1,t2,t3]
                   ┌────────── 单个 grouped-matmul kernel ──────────┐
                   │ [t0×h]·W0 ‖ [t1×h]·W1 ‖ [t2×h]·W2 ‖ [t3×h]·W3 │  变长分组,一次发射
                   └───────────────────────────────────────────────┘
```

关键设计:专家权重**不再按 tp_size 切**(`gmm/experts.py:21-22` 注释 "avoid splitting by tp_size")——每个专家跑完整 GEMM(矩阵大、算得满),为 §5.2 的 tp-extend-ep 铺路。

两条工程旁路:① **空 rank 守卫**——负载倾斜下某 rank 可能一个 token 都没分到,`tokens_per_expert` 全零时 grouped-gemm 不能跑,代码 assert 后退化成一次普通 `torch.matmul`(`:124-151`),避免空分组崩内核;② **激活重算**——`should_recompute_activation` 命中且非通算重叠路径时,fc1 后的激活用 `CheckpointWithoutOutput.checkpoint` 算、`discard_output()` 丢存储、反向 hook 重算(`:103-109`、`:153-161`),省 MLP 中间激活(与 [[mindspeed_memory_optimization_analysis]] §2 同源)。

### 5.2 tp-extend-ep:用 TP 组扩 EP

**命题**:一维 TP 会把每个专家的权重再切到 TP 组——专家本就细粒度,再切既低效又多一跳通信。tp-extend-ep 反过来:**不切专家权重,而把 TP 组的 rank 直接并入 EP**,有效专家并行度变成 $tp\times ep$。`All2AllSeqTp2epDispatcherImpl` 的类注释点破本质(`tp_extend_ep/token_dispatcher.py:14-18`):

```
original logic is alltoall in tp region, then alltoallv in ep region   # 原:两跳 a2a(tp 域 + ep 域)
if use tp_extend_ep, just alltoallv in tp*ep region                    # 现:一跳 a2a-v(tp×ep 合并域)
```

配套 `routing_tp_extend_ep` 跳过 router 的 tp-gather、`MoELayer`→`MindSpeedAlltoAllSEQTptoEpMoELayer`(`tp_extend_ep.py:33-40`)。约束:`num_experts % (tp·ep) == 0`、需 `--moe-permutation-async-comm`+`--moe-grouped-gemm`、dispatcher 为 `alltoall_seq`(`tp_extend_ep.py:22-26`)。

```
原(tp=2, ep=2,专家权重被 tp 再切成半):
   token ─a2a(tp 域,size2)─→ ─a2a-v(ep 域,size2)─→ 专家(半权重 GEMM,算不满)
tp-extend-ep(有效 EP = tp·ep = 4,专家整权重):
   token ───────a2a-v(tp×ep 域,size4)──────→ 专家(完整权重 GEMM)
```

**代数**:dispatch 从 "tp 域 a2a(size $tp$)→ ep 域 a2a-v(size $ep$)" 两跳合成一跳 "tp×ep 域 a2a-v(size $tp\cdot ep$)",少一次集合通信启动 + 一次 token 物化,且专家 GEMM 维度不被 $tp$ 切碎(从 $E/(ep\cdot tp)$ 列恢复到 $E/ep$ 列,矩阵更大、利用率更高)。这正是 §5.1 GMM "不按 tp_size 切权重" 的配套前提。

### 5.3 专家放置与负载均衡

- **ExpertsPlacement**(动态再放置):router 负载天然倾斜,某些"热专家"被打爆。`expert_placement_init` 维护 `expert_mapping`(专家→物理位置置换)(`planner.py:9-23`);`predict_expert_load` 用 **EMA**(`ema_weight=0.9`)滚动预测每专家负载(`:26-39`);每 `--expert-placement-freq`(默认 50,`experts_placement.py:16`)步触发一次 `ExpertDynamicplacement.expert_placement_greedy`——**按预测负载降序贪心**把每个专家放到当前最空、且未超 `num_experts/num_devices` 配额的设备(`planner.py:111-150`),产出新的专家→设备映射 $Q$,把热专家分散到 EP 组各 rank。patch 挂在 fb-overlap 的 MoE 层上(`experts_placement.py:46-48`),需分布式优化器(`:28`)。

  ```python
  # planner.py:38(EMA)+ :127-142(贪心放置,精简)
  load = 0.9*load + 0.1*tokens                       # 每步滚动预测
  for i in sorted(experts, key=cost, reverse=True):  # 按预测负载降序
      q = argmin_device(samples_each_device[d] for d in devices
                        if experts_each_device[d] < E/D)   # 选最空且未超配额 ⌊E/D⌋ 的设备
      P[i] = q;  samples_each_device[q] += cost[i];  experts_each_device[q] += 1
  ```

  > **worked example**:4 卡、8 专家、预测负载 `[100,90,80,70,60,50,40,30]`(专家 0..7),配额 ⌊8/4⌋=2。降序贪心,每步把当前专家放到"`samples` 最小且未满配额"的卡:E0→卡0(=100)、E1→卡1(90)、E2→卡2(80)、E3→卡3(70)→ 四卡各 1 专家;E4(60)→当前最空卡3(70)→130;E5(50)→卡2(80)→130;E6(40)→卡1(90)→130;E7(30)→卡0(100)→130。**四卡各 130,完美摊平**。把原本随机映射可能造成的 (100+90) 挤一卡拉成均衡,代价是周期性 a2a 搬专家权重。`--enable-fine-grained-expert-placement` 进一步在阈值 `fine_grained_expert_placement_thre`(默认 0.08)以上才触发更细粒度重排(`experts_placement.py:18-23,36-41`)。

- **BalancedMoE**(冗余热专家):另一思路是**复制**热专家。`--balanced-moe-hot-expert-num`(默认 3)指定复制几个热专家、`--trans-hot-expert-group-num` 控传输分组(`balanced_moe.py:18-20`),把热专家副本铺到多个 EP rank 摊负载;校验热专家数 ≤ 本地专家数(`:36`)、EP≥16/32 才划算(`:66-72`),仅 alltoall 分发器(`:76-77`),换 `MoELayer`→`BalancedMoELayer` 并改 EP 组构造(`:81-87`)。

  ```
  placement(再放置):热专家整体搬到更空的卡 —— 总副本数不变,周期性搬权重
  balanced (复制)  :热专家 E_hot 在多卡各留一份副本 —— token 分摊到副本,省搬运但费显存
            卡0[E_hot E1]  卡1[E_hot E2]  卡2[E_hot E3] ... ← E_hot 多份,负载/份
  ```
- **SharedExpert**:`--n-shared-experts`(已废弃名)折算成 Megatron 的 `moe_shared_expert_intermediate_size`,即每 token 都过的常驻共享专家、不参与路由——本质是把废弃 CLI 名翻译成 Megatron 原生参数(`shared_expert.py:22-24`):

  ```python
  # shared_expert.py:22-24
  if args.n_shared_experts and args.moe_shared_expert_intermediate_size is None:
      args.moe_shared_expert_intermediate_size = args.n_shared_experts * (
          args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size)
  ```
- **FixRouter**:把 `megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity` 替成 MindSpeed 版,修正 router 在 EP>1 时的 capacity/topk 行为(`moe_fix_router.py:25`),硬约束 `expert_model_parallel_size > 1`(`:19-20`)。

> MoE 的通算重叠(fb-overlap、alltoall-overlap)与 zero-memory 重算分属 [[mindspeed_comm_overlap_analysis]] / [[mindspeed_memory_optimization_analysis]],本页只讲结构。

---

## 6. DP 与分布式分片

**命题**:纯 DP 下每卡冗余持有全量参数/梯度/优化器态。Megatron 的分布式优化器只把**优化器态**沿 DP 切(ZeRO-1)。MindSpeed 提供从轻到重的分片选项,最重的 LayerZeRO 把**参数+梯度+优化器态**都切(ZeRO-3 式),并能把 TP 维也纳入分片域。

| 特性 | 分片对象 | 实现要点 | file:line |
|------|---------|---------|-----------|
| **LayerZeRO** | 参数+梯度+优化器态(ZeRO-3 式) | 自研 `LayerZeRO3` 包住模型,持 `(zero3_process_group, zero1_process_group)` 两级组 + `tp_zero_process_group`;auto-wrap 递归把这三套组下发给子模块(`fsdp.py:75,107-108`)。反向 reduce-scatter 与前向 all-gather 重叠 | `layerzero/zero3/fsdp.py:75-141`;feature `layerzero.py:49-54` |
| **CustomFSDP** | 参数+梯度(按桶) | 复用 Megatron `custom_fsdp` buffer,patch `gradient_reduce_preprocessing`(选 AVG/SUM reduce-op)与 `GradReducePipeline.mark_bucket_ready`(桶组就绪才发 reduce-scatter) | `custom_fsdp_feature.py:13-16` |
| **TorchFSDP2** | 参数+梯度(DTensor) | 直接接 PyTorch FSDP2,patch `TorchFullyShardedDataParallel.__init__`、torch_dcp 存取与 meta-init 修正 | `torch_fully_sharded_data_parallel.py:32-49` |
| **BufferPad** | —(对齐) | 把 param/grad buffer 各桶起始地址 pad 到 512 字节对齐(Ascend 需要),patch `_ParamAndGradBuffer.__init__` | `buffer_pad.py:14-25` |
| **ResetBucketOrder** | —(顺序) | 让参数 all-gather 桶序对齐前向计算序,提升 `overlap_param_gather` 收益(必须先开 `overlap_param_gather`,`:21-22`) | `reset_bucket_group_order_feature.py:30-43` |

**LayerZeRO vs Megatron 分布式优化器**:Megatron 分布式优化器是 ZeRO-1——只把优化器态(fp32 master + 动量)沿 DP 切,参数/梯度仍每卡全量。LayerZeRO 是 ZeRO-3 式——`zero3_process_group` 域内把**参数、梯度、优化器态全切**(用时 all-gather 参数、算完 reduce-scatter 梯度),`zero1_process_group` 域只切优化器态作混合分片,再叠一层 `tp_zero_process_group` 把 TP 维也卷进分片(`fsdp.py:107-108`),分片粒度比 Megatron 更细、省显存更多,代价是前向多一轮参数 all-gather。Custom-FSDP 介于两者间——沿用 Megatron 桶式 buffer,只把 reduce 预处理与桶就绪逻辑换成昇腾版。

```
Megatron 分布式优化器(ZeRO-1):  P 全量 | G 全量 | O 沿 DP 切
LayerZeRO(ZeRO-3 式):           P 沿 zero3 切 | G 沿 zero3 切 | O 沿 zero3+zero1+tp_zero 切
                                  前向 all-gather(P)→算→reduce-scatter(G),用完即弃分片
```

LayerZeRO 的递归装配靠 auto-wrap:`LayerZeRO3.__init__` 把 `(zero3_process_group, zero1_process_group)` 与 `tp_zero_process_group` 打进 `root_kwargs`,`_auto_wrap` 按 `auto_wrap_policy` 逐子模块下发同一套组(`fsdp.py:105-126`),每个被包模块都成为一个独立的"分片单元",前向用前 all-gather 自己那份参数、算完即释放、反向 reduce-scatter 自己那份梯度——这是与 Megatron 分布式优化器(只切 O、参数始终全量驻留)最本质的区别。

**Custom-FSDP 的两处昇腾改写**(`custom_fsdp_feature.py:13-16`):① `gradient_reduce_preprocessing` 按 `average_in_collective`/`gradient_reduce_div_fusion`/dtype 选 reduce-op,bf16 下避开 div-fusion、改 `grad.mul_(scaling)` 后走 SUM(`param_and_grad_buffer.py:8-24`);② `mark_bucket_ready` 实现**桶组级**触发——一个 bucket 就绪不立刻发 reduce-scatter,而是等同组所有 bucket 的全部 grad-ready 参数齐了才发(`:27-47`,任一未齐即 `return False` 延后),把零散小桶合并成一次更大的集合通信。

**两个 Ascend 亲和的桶旋钮**:① **buffer-pad** 把每个 param/grad bucket 的**起始地址 pad 到 N 字节对齐**(Ascend 推荐 512),patch `_ParamAndGradBuffer.__init__`(`buffer_pad.py:14-25`;impl `param_and_grad_buffer_init_pad`)——HBM/HCCS 集合通信对对齐地址吞吐更高,代价是少量 padding 浪费。② **reset-bucket-order** 把参数 all-gather 的**桶序重排成前向计算序**(patch DDP 的 `_make_forward_pre_hook` 与 config/init,`reset_bucket_group_order_feature.py:30-43`),让"先用到的参数先 all-gather",提升 `overlap_param_gather` 的预取命中(故硬依赖 `--overlap-param-gather`,`:21-22`)。两者都不改分片语义,只调对齐与顺序。

**Torch-FSDP2** 则不走 Megatron 桶式 buffer,直接接 PyTorch 原生 FSDP2(DTensor 分片):patch `TorchFullyShardedDataParallel.__init__`(`:32-33`)、torch_dcp 格式的 checkpoint 存取(`load_checkpoint`/`generate_state_dict`,`:41-45`)、以及 Megatron meta-init 的 `get_model` 修正与 2D device-mesh 下的 `get_data_parallel_group_if_dtensor`(`:48-57`)。生态更标准(直接吃 PyTorch FSDP2 生态),代价是与 Megatron 5D 并行的耦合要一串兼容补丁。

> 三者的 all-gather/reduce-scatter **重叠**与省显存量化归 [[mindspeed_comm_overlap_analysis]] 与 [[mindspeed_memory_optimization_analysis]];本页只交代分片**结构**差异。

---

## 7. 分层解耦训练(MindSpeed-LLM)

**命题**:面向**边-云异构**集群——边端设备少/算力弱、云端多/强。把模型按层解耦成"边段 + 云段"的跨域流水,只让中间激活跨边-云传输。三个特性协同(均挂在 `--layerwise-disaggregated-training` 总开关下,`u_shaped_split_feature.py:15`)。

### 7.1 U-shaped split — 首尾层都钉在边端

把**首层与末层(embedding 输入 + 输出 head/loss)都放在 PP 第一个 stage(边端)**,中间层放云端,形成 U 形——数据与标签留在边端,只有中间隐藏态在边-云间往返。`schedules.py:253-257` 的 `is_end_stage` 文档原话:"In U-shaped split scenarios, the first and last layers deploy on the first pipeline stage"。loss 因此不在 `is_pipeline_last_stage` 算,而在**首 stage 的 U 形末端**算(`schedules.py:300-303`):

```python
# schedules.py:300-303
if not config.layerwise_disaggregated_training:
    should_compute_loss = parallel_state.is_pipeline_last_stage()
else:
    should_compute_loss = parallel_state.is_pipeline_first_stage() and is_end_stage  # U 形:首 stage 且末端
```

U-shaped 换掉 `forward_backward_pipelining_without_interleaving`、`initialize_model_parallel`、p2p `_communicate`/`send_forward`/`send_backward` 全套(`u_shaped_split_feature.py:49-60`)。重写后的调度有两处结构性改动:

- **边端持两个 model-chunk**:`forward_backward_pipelining_without_interleaving` 断言 `model` 必须是 chunk 列表(`schedules.py:830-831` "cloud-edge pipeline parallelism expected model chunking")——边端 stage 同时持有"首层 chunk"和"末层 chunk",数据迭代器也按 chunk 给(`:838-839`);非首 stage 的 `data_iterator` 置空(`:835-836`)。
- **末↔首专用通信组闭合 U**:云端最后一个 stage 算完要把激活送回**边端持末层的那个 chunk**,于是新建 `group_first_to_last`/`group_last_to_first` 一对跨域组(`schedules.py:1009-1019`),PP 数为奇数时还给首/末 stage 各加一条额外通信流——这正是把线性流水"首→…→末"弯成 U 形、让首尾在物理上同处边端的关键接线。

```
线性 PP:  stage0 → stage1 → stage2 → stage3        (末 stage 算 loss)
U 形:     边端 stage0[首层chunk] → 云1 → 云2 → 云端 stage_k
                ↑__末层chunk______________________________↓
            边端同时持首层+末层 chunk,末 stage 经 last→first 组把激活送回边端算 head+loss
```

```mermaid
flowchart LR
    subgraph EDGE["边端(PP stage0,TP/DP 小)"]
      E1[Embedding 首层]; E2[输出 head / loss 末层]
    end
    subgraph CLOUD["云端(中间 PP stages,TP/DP 大)"]
      M1[transformer 中间层 ...]
    end
    E1 -->|"激活 cross edge-cloud"| M1 -->|"激活回传"| E2
    E2 -.->|"VDP 跨 DP 平均梯度"| E1
```

### 7.2 VDP — 边/云不同数据并行度

边段与云段可有**不同 DP 度**。`_init_vdp_state` 按本地拓扑推边端 DP:`edge_dp_size = LOCAL_WORLD_SIZE / (cp · edge_tp)`,与配置的 `vdp_size` 不等即启用(`parallel_state.py:851-877`,核心 `:862-864`):

```python
# parallel_state.py:862-864
if int(os.environ['LOCAL_WORLD_SIZE']) % (context_parallel_size * edge_tp_size) == 0:
    edge_dp_size = int(os.environ['LOCAL_WORLD_SIZE']) // (context_parallel_size * edge_tp_size)
    _VDP_ENABLED = edge_dp_size != vdp_size   # 边端 DP ≠ 云端 vdp → 启用虚拟 DP
```

VDP 新建**两类跨域组**——`VDP_CROSS_CLOUD_TP`(跨云端 TP)与 `VDP_CROSS_EDGE_CLOUD`(跨边-云,`parallel_state.py:62-63`,构造 `:1182-1206`)——并 patch `finish_grad_sync`/`register_grad_ready`/`get_grad_norm_fp32`/`create_group`(`vdp_feature.py:30-47`),让边端在反向时跨自己的 DP 域平均梯度、且与云端 DP 度解耦。因为边端多出的 DP 副本在 Megatron 看来"凭空多了卡",VDP 把逻辑 world-size 撑成 `real_world_size + (vdp_size-1)·cp·tp`(`parallel_state.py:957`、`:1018`),让 Megatron 的组划分照常跑通。

### 7.3 VTP — 边/云不同张量并行度

不同 PP stage 可有**不同 TP 度**(边端 GPU 少 → TP 小,云端 TP 大)。`--tensor-model-parallel-size` 取全局**最大** TP,真实各 stage TP 由节点拓扑在分布式初始化后自动探测(feature docstring `vtp_feature.py:9-15`)。难点是 Megatron 的 `world_size % (tp·pp·cp) == 0` 校验会在异构 TP 下失败,VTP 因此在 `pre_validate_args` 把 `world_size` 临时膨胀到 `tp·pp·cp`(DP=1 最小合法值)骗过校验、`post_validate_args` 还原(`vtp_feature.py:20-46`),并 patch `torch.distributed.all_gather_into_tensor`(`:59`)处理 VTP 下全局 all_gather 的兼容(异构 TP 时跳过 timer-stats 的全局聚合,`utils.py:98-110`):

```python
# vtp_feature.py:36-39
if world_size % (tp * pp * cp) == 0:
    return                              # 已合法,不膨胀
args._vtp_orig_world_size = world_size
args.world_size = tp * pp * cp          # DP=1 最小合法值,过 Megatron 校验后还原
```

- **mamba-CP**:为 Mamba/SSM 块追加 `mamba_cp_algo` 上下文并行选项(同属 LLM 分层解耦域)。

三者协同:U-shaped 决定**层往哪放**(首尾钉边端),VDP 决定**边/云各自的 DP 度**,VTP 决定**边/云各自的 TP 度**——把一个本来要求"全集群同构 TP×DP"的训练,拆成边端低并行 + 云端高并行的异构流水,只让中间激活过边-云链路。

---

## 8. 特性约束与互斥(从 validate_args 读出)

这些并行特性多数会**接管 Megatron 的同一段划分逻辑**(同一个线性类、同一套并行组、同一个调度器),因此存在硬约束。下表全部来自各 feature 的 `validate_args`,是配置时的实测红线:

| 特性 | 关键约束 / 互斥 | file:line |
|---|---|---|
| **tp-2d** | `tp = tp_x·tp_y`;⊥ sequence_parallel / fused_rmsnorm / nanopipe / ascend_coc / **MoE**;CP 仅 megatron/ulysses | `tp_2d.py:29-42` |
| **unaligned-linear** | ⊥ MC2 / tp-2d / **MoE** | `unaligned_linear_feature.py:20-24` |
| **num-layer-list** | ⊥ VPP / dualpipev;`len(list)==pp`;开 noop 时自动失效 | `num_layer_list.py:27-38` |
| **pipeline-layout** | ⊥ dualpipev / noop / pipeline-num-transformer-layers / recompute-in-bubble·advance | `pipeline_model_parallel_layout_feature.py:103-115` |
| **multi-parameter** | ⊥ moe_fb_overlap / dualpipev | `multi_parameter.py:36-42` |
| **variable-seq** | 仅 MoE(无 num_moe_experts 自动关) | `variable_seq_length.py:40-41` |
| **tp-extend-ep** | `num_experts % (tp·ep)==0`;**需** permutation-async-comm + grouped-gemm + `alltoall_seq`;⊥ capacity-factor | `tp_extend_ep.py:22-28` |
| **balanced-moe** | 热专家数 ≤ 本地专家;仅 alltoall;**需** fb_overlap + grouped_gemm;⊥ capacity-factor | `balanced_moe.py:36,74-78` |
| **expert-placement** | **需**分布式优化器;EP>1 | `experts_placement.py:28-31` |
| **fix-router** | `expert_model_parallel_size > 1` | `moe_fix_router.py:19-20` |
| **reset-bucket-order** | **需** `--overlap-param-gather` | `reset_bucket_group_order_feature.py:21-22` |
| **swap-optimizer / reuse-fp32** | 与本页分片正交但共用 master/态布局,见 [[mindspeed_memory_optimization_analysis]] §9 | — |

**可叠加的典型组合**:tp-2d(打 TP)+ CP(经 `TensorParallelYUnionCP` 联动)+ PP noop/num-layer-list(打 stage 负载)+ LayerZeRO/分布式优化器(打 P/G/O 冗余),四维互不接管对方的划分点。**MoE 是最多互斥点**:tp-2d、unaligned-linear 都明确不支持 MoE,MoE 自己的 tp-extend-ep/balanced 又对分发器和 grouped-gemm 有硬要求。

---

## 9. 选型与一句话小结

### 9.1 维度内选型(约束已逐条核对)

```
TP:  权重大、TP 也大(≥8)且想压通信占比 → tp-2d(x·y 网格,两组小集合,可叠 matmul 重叠)
     头数/维度非整除、不想 padding        → unaligned-linear(余数摊前 r 个 rank;⊥ tp-2d/mc2/moe)
     词表并行要 NPU 确定性                 → vocab ReplaceIndexPut(纯张量 mask,无 index_put_)
PP:  只想逐 stage 配层数               → num-layer-list(LLM,最简)
     PP×VPP 二维不均                    → unaligned(嵌套列表,列优先 offset)
     要精确摆 embed/loss/MTP            → pipeline-model-parallel-layout(字符串 DSL)
     纯负载微调、不想改层数             → noop-layers(零算力占位)
MoE: 专家计算原语                      → grouped-gemm(GMM,空 rank 退普通 matmul)
     专家细、TP 切碎了                  → tp-extend-ep(有效 EP=tp·ep,一跳 a2a-v)
     热专家倾斜                        → expert-placement(EMA+贪心重放置)或 balanced-moe(复制热专家)
DP:  只切优化器态(默认)               → Megatron 分布式优化器(ZeRO-1)
     要切参数+梯度+优化器态            → LayerZeRO(ZeRO-3 式,叠 tp_zero)
     接 PyTorch 生态                   → torch-fsdp2;Ascend 桶对齐 → buffer-pad(512B)
边云异构: U-shaped-split + VDP + VTP(首尾钉边端、边/云各自 DP/TP)
```

### 9.2 一句话小结

- **TP**:TP-2D 用 $x\times y$ 网格把一次大 all-reduce 拆成 X-AllGather + Y-ReduceScatter 两组 $\sqrt N$ 量级小集合(ND1/ND2 正交主切轴),通信税从 $\frac{N-1}{N}$ 降到 $\frac{\sqrt N-1}{\sqrt N}$ 且可重叠;unaligned 放开整除约束、余数摊前若干 rank;vocab 用纯张量 mask 换掉 NPU 不确定的 `index_put_`。
- **PP**:noop 零算力假层 + num-layer-list/unaligned/layout 三档"自定义每 stage 装什么层",核心都是 patch `get_num_layers_to_build`+layer-offset,把非均匀负载摊平。
- **MoE-EP**:GMM 一次算完本地全专家(权重不按 tp 切),tp-extend-ep 把 TP 并入 EP(有效 EP=$tp\cdot ep$、一跳 a2a-v),placement/balanced 用 EMA 预测 + 贪心重放置/复制压热专家倾斜。
- **DP**:从 ZeRO-1(Megatron 默认,只切 O)到 LayerZeRO 的 ZeRO-3 式(切 P+G+O 并卷入 TP 维),Custom-FSDP/Torch-FSDP2 居中,buffer-pad/reset-bucket 是 Ascend 亲和的对齐与桶序微调。
- **分层解耦**:U-shaped(首尾钉边端)+ VDP(边/云不同 DP)+ VTP(边/云不同 TP),把异构边-云集群拆成低并行边段 + 高并行云段的跨域流水。

---

## Related Pages

- [[mindspeed/index]] —— MindSpeed×MindSpeed-LLM 特性总罗盘(四大类入口)
- [[mindspeed_context_parallel_analysis]] —— CP 上下文并行独立深挖(分派脊柱、双环前向/反向、通信量代数、TND 变长裁剪、RoPE 不变量、选型)
- [[mindspeed_comm_overlap_analysis]] —— 本页所有并行结构上的**通算掩盖**(CP send-recv overlap、MC2/CoC、TP-2D AG/RS 重叠、MoE fb-overlap、PP DualPipeV/RiPipe 调度)
- [[mindspeed_memory_optimization_analysis]] —— 重计算/Swap/MoE-zero-memory 等省显存手段(与 LayerZeRO/CustomFSDP 分片互补)
- [[mindspeed_ascend_affinity_analysis]] —— GMM/FlashAttention/swiglu 等昇腾融合算子(TP/MoE 的计算原语)
- [[megatron_ep_analysis]] —— Megatron 原生 MoE 专家并行(三种 dispatcher、Parallel Folding),与本页 §5 跨框架对照
- [[megatron-lm/index]] —— 被打补丁的宿主框架;对照阅读原生 5D 并行
