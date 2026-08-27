---
title: "vLLM 分布式推理：按状态切分维度建立通信合同"
---

# vLLM 分布式推理：按状态切分维度建立通信合同

> **源码基线**：`vllm-project/vllm@d66300a1baa7779c68c7dfa4e51eee2502b48017`
> **中心命题**：TP、PP、DP、EP、PCP/DCP 不是几种等价的“多卡开关”，它们分别切分参数、层、请求、专家或序列上下文。vLLM 先用 rank layout 建立正交 process group，再由 layer/runner 在正确 group 上执行 collective；拓扑正确性比 executor 进程如何启动更基础。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-27。按五拍重排章节顺序，并补齐发展趋势；机制正文与既有引用未改。

## 一、背景：先问“什么状态被切分”

| 维度 | 被切分/复制的对象 | 典型通信 | 主要瓶颈 |
|---|---|---|---|
| TP | 单层权重与 activation channel | all-reduce/all-gather/reduce-scatter | 高频小/中型 collective |
| PP | Transformer layers | stage 间 send/recv activation | pipeline bubble 与 stage 不均衡 |
| DP | 完整模型副本与请求集合 | 路由；MoE/同步场景另有 collective | 负载与 KV 容量偏斜 |
| EP | MoE experts | token dispatch/combine all-to-all | expert imbalance 与网络带宽 |
| PCP/DCP | prefill/decode 的 sequence/context | KV/attention 数据交换 | 长上下文通信与 backend 支持 |

混用这些概念会导致错误配置。例如 DP 增大吞吐但不会让单个过大模型装进一张卡；TP 能分权重，却让每层都支付通信；EP 只分 experts，dense 层通常仍按 TP/复制规则执行。

## 二、rank layout 是全系统的坐标系

`ParallelConfig` 保存 PP、TP、DP、EP、DCP、executor backend 与 DBO 等配置，并在 post-init 中校验组合；`vllm/config/parallel.py:119-342,475-559`。模型并行初始化把 world ranks reshape 为：

`ExternalDP × DP × PP × PCP × TP`

源码明确记录这一布局及 DP 内各 rank 必须同步 generate 的约束；`vllm/distributed/parallel_state.py:1817-1832`。随后通过 transpose/reshape 为每个维度生成 group：

- TP group：相邻 TP ranks；`vllm/distributed/parallel_state.py:1834-1849`；
- DCP/PCP group：按 context 维度重新排列；`vllm/distributed/parallel_state.py:1851-1886`；
- PP group：同一 tensor shard 在不同 pipeline stage 的 ranks；`vllm/distributed/parallel_state.py:1888-1904`；
- DP group：同一模型位置的不同 replicas；`vllm/distributed/parallel_state.py:1906-1921`；
- EP group：跨 DP×PCP×TP 组织 experts；`vllm/distributed/parallel_state.py:1923-1955`。

这个坐标系的不变量是：

> 任一 collective 的所有参与 rank 必须从同一布局推导、按同一顺序进入，并对 tensor shape/语义达成一致。

错误 group 往往不是数值误差，而是 deadlock；错误 shape 则可能在某个 rank 提前抛错，让其他 rank 永久等待。

## 三、`GroupCoordinator` 为什么不仅包一层 ProcessGroup

PyTorch ProcessGroup 固定一个 backend，但推理还需要 CPU metadata、device tensor、shared-memory broadcast、custom all-reduce 和 graph capture 兼容。`GroupCoordinator` 同时持有 CPU group、device group、device communicator 与可选 message queue；`vllm/distributed/parallel_state.py:380-518`。

对 layer 暴露的 `all_reduce()`、`all_gather()` 等先处理 world-size-one fast path，再转到平台 communicator；`vllm/distributed/parallel_state.py:662-748`。上层因此调用语义操作，而不是硬编码 NCCL/Gloo/custom kernel。

这也是 compile 层能识别通信 op 的前提：顶层 `all_reduce`/`all_gather` 有真实实现与 fake implementation；`vllm/distributed/parallel_state.py:152-199`。通信既是运行时副作用，也是图中的 shape 变换。

## 四、TP：layer ABI 决定 collective 放在哪里

column-parallel linear 沿输出维切权重，各 rank 产生不同 output shard；必要时 all-gather。row-parallel linear 沿输入维切权重，各 rank 产生 partial sum；必要时 all-reduce。实现不是 runner 在模型末尾统一通信，而是 layer 自己在 `apply()` 后按 `gather_output` 或 `reduce_results` 决定 collective；`vllm/model_executor/layers/linear.py:577-587,1652-1662`。

因此 TP 的正确性合同包括：checkpoint shard axis、local parameter shape、activation shard axis、bias 只加一次、collective group 与输出布局。量化 post-load 若替换 parameter，也必须恢复 TP ownership。

TP 的收益是降低单 rank 权重/计算，代价是几乎每层通信。优先把高带宽互联留给 TP；跨低速节点盲目增加 TP，常使 TPOT 恶化。

## 五、PP：切层并把一次 step 变成 stage 协议

PP rank 只加载自己的 layer range，前一 stage 发送 intermediate tensor，后一 stage 接收并继续。PP 能跨节点减少单 rank 权重，却引入 bubble、stage load imbalance 和更复杂的同步失败域。

在线推理的 Scheduler 仍形成同一个 logical step，所以所有 stage 必须对 scheduled batch 和 tensor shape 一致。若某 stage 因 OOM、compile 或 connector 分支没有进入 send/recv，其他 stage 会阻塞。

PP 不应只按 layer 数平均：embedding、LM head、MoE layer、multimodal encoder 和不同量化 kernel 的成本不同。真实划分需要 profile stage time 与 memory peak。

## 六、DP 与 Serving 路由：复制模型，不复制单请求

DP engine 各自拥有完整或同构模型副本、Scheduler 与 KV pool，请求通常只进入一个 DP rank。Serving client 可依据 running/waiting/KV usage 做内部负载均衡；`vllm/v1/engine/core_client.py:1435-1507`。

内部 coordinated DP 与 external DP 不同：源码布局注释指出，纳入模型的 DP group 需要各 rank 同步调用 generate，否则 collective 路径可能 deadlock；外部 DP replicas 则可独立生成；`vllm/distributed/parallel_state.py:1817-1823`。

所以“DP size”同时影响服务拓扑与模型通信，必须结合 external/internal/hybrid LB 模式解释，不能只用训练语境的 replica 概念。

## 七、EP：参数容量问题变成 token 路由问题

MoE 中每 token 只访问少数 experts。EP 把 experts 分到多个 rank，先按 router 结果 dispatch token，再在远端执行 expert，最后 combine。其通信量与 token/expert 分布有关，不像 TP 固定按 dense activation 通信。

EP group 由 DP×PCP×TP ranks 共同构成，并可启用 all-to-all communicator；`vllm/distributed/parallel_state.py:1923-1955`。EPLB 另建独立 process group，使负载统计/重排通信不与 forward collective 交错而死锁；`vllm/distributed/parallel_state.py:1957-1963`。

EP 的关键不变量是 token 的 source position、expert assignment、dispatch offset 与 combine offset 成对。expert 热点会同时造成计算和通信倾斜，单看平均 tokens/expert 不足以定位尾延迟。

## 八、Context Parallel：长序列不等于模型并行

PCP/DCP 沿 sequence/context 维切 attention 工作或 KV，而不是切模型 channel。DCP group 优先跨 PCP 再跨 TP 组织；`vllm/distributed/parallel_state.py:1851-1867`。它要求 attention backend 能消费相应分布式 metadata，并保证 softmax/statistics 的跨 rank 合并正确。

这类并行可降低单 rank 长上下文 KV/attention 压力，但增加每层 context 通信。短 decode 或 backend fallback 时，通信固定成本可能超过收益。

## 九、DBO：重叠通信与计算，而非新的模型切分

Dual Batch Overlap 把同一执行 batch 分为两个 microbatches，在一个 microbatch 通信时推进另一个计算。`ParallelConfig` 的 DBO 开关决定默认两个 ubatches；`vllm/config/parallel.py:211-231,551-559`。ubatch context 暴露 compute/comm stream 切换与 yield hook；`vllm/v1/worker/ubatching.py:15-186`。

它增加的状态包括每 ubatch 独立 workspace、attention metadata、collective handle 和 CUDA event。其不变量是：前一个 ubatch 仍在使用的通信 buffer/workspace 不能被下一个复用。某些 speculative、cascade attention、elastic EP 组合会禁用或限制 DBO，这说明 overlap 是能力矩阵，不是无条件开关。

## 十、Executor 只负责实现拓扑，不定义并行语义

`ParallelConfig` 根据 world size、平台和部署方式选择 `uni`、multiprocessing、Ray、external launcher 等 executor backend；`vllm/config/parallel.py:835-966`。Executor 负责创建 workers、RPC 与结果汇聚；真正的 TP/PP/EP tensor 语义仍在 process groups、layers 和 runners。

把性能问题简单归因于 Ray 或 multiprocessing 容易误判：多数稳态成本来自 collective、rank placement、kernel shape 与负载不均；executor 对启动、故障域和控制面延迟影响更直接。

## 十一、约束、选择与失败边界验证

| 目标 | 优先考虑 | 警惕 |
|---|---|---|
| 模型装不下单卡 | TP，必要时 PP | 跨节点 TP 带宽 |
| 提高独立请求吞吐 | DP | KV/请求路由偏斜 |
| 大 MoE 容量 | EP + 合适 TP/DP | all-to-all 与 expert hotspot |
| 超长 context | PCP/DCP | backend 支持与每层通信 |
| 隐藏 MoE/collective 延迟 | DBO | workspace 生命周期和兼容矩阵 |

验证顺序：

1. 打印每个 rank 的 DP/PP/PCP/TP/EP 坐标和 local device；
2. 核对 checkpoint shard、layer partition 与 process group；
3. 用 world-size-one 结果做数值基线；
4. 分别测 compute、collective、bubble、dispatch/combine；
5. 检查所有 rank 是否进入相同 collective 序列和 shape；
6. 再调整 placement、group size、microbatch 与 backend。

最小源码阅读顺序：`vllm/config/parallel.py:119-342,475-559,835-966` → `vllm/distributed/parallel_state.py:380-748,1751-1963` → `vllm/model_executor/layers/linear.py:577-587,1652-1662` → 目标 PP/EP/DCP runner → executor。

## 十二、发展趋势

> [!note]
> 本节离开“源码此刻是什么”，只收录源码自陈的在途改动；每条给出锚点，属于外推的部分单独标注。

1. **通信算子的自定义补丁带着上游退出条件。** `patched_fused_scaled_matmul_reduce_scatter` 上方写着 `# TODO: Remove this once the pytorch fix (https://github.com/pytorch/pytorch/pull/165086) gets released, in either 2.9.1 or 2.10`，见 `vllm/distributed/parallel_state.py:368-373`。即：TP+SP 这条融合通信路径当前的形态由 PyTorch 版本决定，会随上游发布回收。

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis|vLLM Engine 架构]] — executor 与 EngineCore 的控制边界。
- [[02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis|vLLM 模型与权重 ABI]] — layer partition、TP 权重与模型能力。
- [[02_engineering/03_infer_frameworks/vllm/16_vllm_serving_control_plane_analysis|vLLM Serving 控制面]] — DP frontend、内部/外部路由与进程生命周期。
- [[02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis|vLLM 量化设计]] — 量化 packed layout 与 TP shard 的组合约束。
- [[02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis|vLLM 融合算子与 Kernel]] — collective、MoE all-to-all 与设备 kernel。
- [[02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis|vLLM 分离式 KV Serving]] — 与模型并行正交的 prefill/decode 服务切分。
