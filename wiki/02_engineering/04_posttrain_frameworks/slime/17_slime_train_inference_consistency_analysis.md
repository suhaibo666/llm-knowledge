---
title: "slime 训推一致性分析：相同权重只是诊断起点"
---

# slime 训推一致性分析：相同权重只是诊断起点

> **源码基线**：`THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`（`main`，2026-08-12）
> **主题**：沿权重、输入、采样、路由、数值和并行六层检查训推一致性，并解释 routing replay、对齐算子与 TIS/MIS 校正。
> **适用范围**：固定基线的 Megatron/SGLang 一致性；权重传输归权重同步页，loss 缩放归 loss 页。
> **最近更新**：2026-09-10。依据冻结源码复核并补全本页机制。

训推一致性不是“权重同步成功”这一个布尔条件，而是一条逐层收紧的证据链。即使训练侧与 rollout 侧的参数逐元素相等，二者仍可能使用不同的输入 token、采样候选集、MoE 专家及其顺序、kernel/精度，以及不同的批次和并行规约路径。slime 因而同时提供行为策略元数据、输入回放、路由重放、对齐钩子、逐层数据导出和分级 CI；代价是更多元数据、额外前向计算与磁盘写入、受限的 kernel/拓扑，以及只在特定 GLM-5 软件栈上成立的严格门禁。

本文把三类结论分开：**源码事实**和**项目文档事实**都带 fixed-commit 定位符；标为“分析判断”的内容是根据实现约束和失败路径作出的推断，不代表项目作者原话。

## 1. 背景：为什么权重相等仍不能证明行为相等

对已生成 token $y_t$，最直接的比较量是

$$
\delta_t=
\left\lvert
\log p_{\mathrm{train}}(y_t\mid h_t)
-
\log p_{\mathrm{rollout}}(y_t\mid h_t)
\right\rvert.
$$

这里的历史 $h_t$、条件分布、专家路径和浮点执行路径必须同时有可比语义；只比较参数张量并没有固定其中任何一项。源码确实提供 `--check-weight-update-equal`：rollout 初始化时先 snapshot/reset，首次 push 后再 compare，但这个检查只回答“训练权重是否正确到达推理引擎”。[`slime/ray/placement_group.py:246-248`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/placement_group.py#L246-L248) [`train.py:26-30`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train.py#L26-L30)

### 1.1 六层诊断模型

```mermaid
flowchart TB
    W["L0 权重快照<br/>参数与版本"] --> I["L1 输入轨迹<br/>token span 与 mask"]
    I --> S["L2 采样分布<br/>温度与支持集"]
    S --> R["L3 MoE 路由<br/>专家集合与顺序"]
    R --> K["L4 数值路径<br/>kernel 与精度"]
    K --> P["L5 执行形态<br/>batch 与并行规约"]
    P --> O["观测结果<br/>逐 token 与逐层差异"]
```

| 层 | 要守住的不变量 | 主要比较或重放钩子 | 常见症状 | 为什么上一层通过仍不够 |
|---|---|---|---|---|
| L0 权重快照 | 一次比较使用同一完整参数版本 | 权重比较、引擎版本、`weight_versions` | 乱码、突变、整段大偏差 | 相同参数仍可能使用不同 token |
| L1 输入轨迹 | prompt/response token、位置和 mask 一一对应 | 按 Sample 对齐 rollout/训练数据导出结果 | 首个错位点后全段偏差 | token 相同不代表采样候选集相同 |
| L2 采样分布 | temperature 与 token 支持集一致 | selected-token logprob、top-p ids/offsets | 偏差随截断或温度系统漂移 | 同支持集仍可能选不同专家 |
| L3 MoE 路由 | 每 token、每层的 expert id 与 top-k 顺序一致 | routed-expert metadata、`--use-rollout-routing-replay`、ordered-top-k capture | 只在 MoE 层起跳或稀疏爆点 | 同专家不代表 expert 内数值路径相同 |
| L4 数值路径 | attention/GEMM/norm/KV/量化路径语义一致 | alignment hooks、layer/module dump | 小误差逐层放大 | 同算子族仍会被 batch/分片改变规约树 |
| L5 执行形态 | batch shape、TP/PP/CP/EP 下结果满足目标不变性 | debug replay、parallel check、端到端 gate | 改 batch 或并行度才出现漂移 | 这是完整执行图层，不能由局部 kernel 证明 |

> **分析判断**：这是一条“先排除离散错误，再定位连续数值误差”的顺序。token、支持集和 expert id 一旦不同，后续 hidden state 已不再是在比较同一计算；此时先调 deterministic kernel，只会让两条不同轨迹各自稳定地重复。

## 2. 为什么这么设计：一致性被做成可分层开关的证据链，而不是一个 bitwise 总开关

先把取舍摆在机制前面。“让训推一致”有几个更直观的做法，固定基线一个都没有当作默认路径，判据只有一条：**要对齐的不是“更准的数值”，而是另一台引擎的同一条数值路径；而这条路径只有被拆成可以单独打开、单独失败的层，才可能被定位。**

| 直觉方案 | 为什么看似可行 | 固定基线为什么不这么走 |
|---|---|---|
| **让训练侧算得更“准”**：把 LM head 之类抬到 FP32 | 训练侧本就有更高精度路径，改一个 dtype 即可 | 官方文档在列举 GLM-5 支持项时把判据直接写进括号：MoE router 用 fp32，但 LM head 在训练与 rollout 两侧都保持 bf16——带来对齐的是**精度匹配**，不是精度更高。[`docs/zh/advanced/reproducibility.md:61`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/reproducibility.md#L61) |
| **把整个训练栈改成 bitwise 确定** | 一次覆盖所有层，不必分层排查 | 对齐被实现成**可选装的替换层**，而不是新的默认训练路径：DeepGEMM 对齐模块的 docstring 自称 “an opt-in numerical-alignment hook”，只替换被选中的 Transformer Engine linear；而且只对齐 forward（SGLang 式 block-FP8），backward 仍走显式 BF16 GEMM 与解析式 norm 梯度。[`slime/backends/megatron_utils/alignment/deepgemm_forward.py:1-8`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L1-L8) |
| **一上来就在目标拓扑上全开** | 省掉逐层定界 | 第一版实现反而主动**缩小**范围以排除混杂变量：它要求 tensor parallel size 为 1，使每个目标是一整块矩阵，对上 SGLang 的 dense-TP1 执行，从而把 row-parallel 部分和舍入排除在外（源码写作 “as a confounder”）。[`slime/backends/megatron_utils/alignment/deepgemm_forward.py:10-12`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L10-L12) 同样的窄化出现在 router 覆写上：只覆盖由 DeepEP alignment bridge 注册的、非 grouped 的 router，其余训练路径保留 Megatron 原语义。[`slime/utils/routing_replay.py:56-68`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/routing_replay.py#L56-L68) |
| **直接用重要性校正把差异抹平** | 已经有 TIS / rollout logprob，不必先证明一致 | 参数校验直接禁止把两种“替换 old policy”的做法叠在一起：`use_rollout_logprobs` 与 `use_tis` 不能同时开启。校正因此是最后一步，而不是替代诊断的开关（见第 11.1 节）。[`slime/utils/arguments.py:1849-1850`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1849-L1850) |
| **把每步完整词表 logits 存下来做精确重放** | 重放最彻底，什么都能复原 | `Sample` 只保存选中 token 的 `rollout_log_probs`、ragged 的 top-p nucleus ids/offsets 和 routed experts；字段注释写明第 $i$ 个 response token 的候选集是 `rollout_top_p_token_ids[offsets[i]:offsets[i + 1]]`。[`slime/utils/types.py:121-126`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/types.py#L121-L126) 它保存的是**重建 behavior 分布所必需的最小集合**，代价是这套元数据只覆盖 top-p、不覆盖 top-k（见第 5 节末的警告框）。 |

源码还明确划出一条职责边界：共享的对齐环境变量被集中到一个模块，而集群相关的连通性设置（`PYTHONPATH`、`MASTER_ADDR`、网卡名、代理、IBGDA handler）被写明 “intentionally not here”，交给 launcher 负责。[`slime/backends/megatron_utils/alignment/env.py:1-7`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/env.py#L1-L7) 也就是说“可复现”被切成两半：数值语义归框架，部署环境归调用方——这解释了为什么第 9 节要把三种确定性分开讲。

> [!note] 推断
> 上面每一条取舍都能在源码或官方文档里找到原话，**但源码没有在任何一处写下“因此一致性应组织成 L0→L5 六层”这句总结**。第 1.1 节的六层模型，以及本页此后按层推进的顺序，是据实现形态与失败路径重建的组织方式，不代表项目作者原话。它之所以站得住，是因为每一层恰好对应源码中一个可以**单独开关、单独失败**的机制：`--check-weight-update-equal`（L0）、rollout/train dump 与 `--load-debug-rollout-data`（L1）、top-p 元数据与训练侧重算（L2）、`--use-rollout-routing-replay` 与 ordered-top-k capture（L3）、alignment hooks 与 layerwise dumper（L4）、`parallel_check` 与端到端 gate（L5）。若其中某一层将来被合并进另一层，这条分层就需要重画。

## 3. L0 权重快照：必要，但只证明参数提交

**问题背景与不变量。** rollout 不能读到缺 bucket 或跨版本的参数快照。disk 更新路径会 pause generation、flush cache、reload，并在 CI 模式逐 engine 比较加载后的 `weight_version`，成功后才恢复生成。[`slime/ray/actor_group.py:227-266`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L227-L266) Sample 又把 SGLang 返回的 `weight_version` 追加到列表，使 partial 或多段 response 能暴露跨版本边界。[`slime/utils/types.py:397-416`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/types.py#L397-L416)

**证据钩子与症状。** 先执行 weight compare，再检查 engine version 和每条 Sample 的 `weight_versions`。失败通常表现为生成突然失真、所有 token 的 logprob 都大幅偏离，或一次轨迹出现多个版本。

**为什么仍不够。** 参数相等只固定函数的参数，没有固定函数输入。聊天模板、tokenizer、partial 前缀、位置或 response span 任何一项不同，后续比较都不是同一个 $h_t$。

> **边界**：本页只把权重相等当作 L0 门槛；pause/flush、拓扑转换和 transport 的完整提交协议由 [[16_slime_weight_sync_analysis]] 负责。

## 4. L1 输入轨迹：先证明两侧在算同一个 token

**问题背景与不变量。** rollout 是文本、工具观察和异步请求的世界，训练侧则按 response span、mask 和并行 schedule 重组 tensor。`Sample.append_response_tokens` 要求新 token 与 logprob 等长；可训练 token 缺 logprob、不可训练 token 携带 logprob 都会报错，工具 token 只填 0 占位并以 `loss_mask=0` 排除训练。[`slime/utils/types.py:253-302`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/types.py#L253-L302) 每次追加还会校验 mask、rollout logprob 与 top-p offsets 的长度关系。[`slime/utils/types.py:418-443`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/types.py#L418-L443)

**证据钩子与症状。** rollout dump 保存 `Sample.to_dict()` 后的完整 samples；train dump 的 version-2 payload 用 `rollout_position` 优先、`sample_index` 次之恢复全局顺序，并保留 DP/micro-batch layout。[`slime/ray/rollout.py:703-720`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L703-L720) [`slime/backends/megatron_utils/train_dump_utils.py:112-188`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/train_dump_utils.py#L112-L188) 比较时应先 join sample，再核对 tokens、response length、loss mask 和位置；典型症状是从首个 token 错位处开始整段差异，而不是孤立的小数误差。

**为什么 L0 通过仍不够。** 同一权重对不同 token 序列给出不同 logits 是正常行为；因此 weight compare 不能替代输入 dump 对账。

### 4.1 调试回放固定的是训练输入，不是生成时的随机过程

`--load-debug-rollout-data` 直接反序列化已保存 Sample，并跳过新的 rollout；可选 subsample 也只是从 dump 取首尾子集。[`slime/ray/rollout.py:671-684`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L671-L684) 官方 debug 文档把它的用途明确写成“固定训练部分输入，去除 rollout 随机性”，并区分 rollout-only 与 train-only。[`docs/zh/developer_guide/debug.md:26-49`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/developer_guide/debug.md#L26-L49)

> **分析判断**：这类回放适合回答“同一批 Sample 更换并行度或 kernel 后是否仍得到相同训练行为”，但不能回答“重新采样能否产生同一 response”；后者还需要固定采样随机种子，并使用确定性的推理 kernel。

## 5. L2 采样分布：相同 token 也可能来自不同支持集

**问题背景与不变量。** rollout 对 logits 使用 temperature、top-p 和 top-k。若第 $t$ 步保留集合为 $S_t$，behavior distribution 可写成

$$
q_t(v)=
\frac{
\exp\!\left(z_{t,v}/T\right)\mathbf{1}[v\in S_t]
}{
\sum_{u\in S_t}\exp\!\left(z_{t,u}/T\right)
}.
$$

训练侧即使对同一已采样 token 重算 full-softmax，也没有在重建这个 $q_t$。SGLang 请求传入 temperature/top-p/top-k 并要求返回 selected-token logprob；top-p 非 1 时还请求每 token 的 nucleus ids。[`slime/rollout/sglang_rollout.py:94-107`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L94-L107) [`slime/rollout/sglang_rollout.py:175-182`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L175-L182)

**源码事实：行为策略元数据是完成重放所需的最小数据。** `Sample` 保存选中 token 的 `rollout_log_probs`、不等长的 top-p ids/offsets 和路由专家，而不是保存每一步的完整词表 logits。[`slime/utils/types.py:114-128`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/types.py#L114-L128) 转换器只在字段存在或功能已开启时，才把这些条件字段送入训练器；启用 top-p 却缺少相应数据时，会在转换或 loss 入口报错，不会静默退回完整 softmax。[`slime/ray/rollout.py:828-852`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L828-L852) [`slime/backends/megatron_utils/loss.py:83-94`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L83-L94)

训练重算先按 rollout temperature 缩放 logits，再依据 ragged nucleus 为 response row 建 keep mask；该 mask 覆盖 CP 本地、CP all-gather 与 TP vocab shard 情形。[`slime/backends/megatron_utils/loss.py:349-429`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L349-L429) [`slime/backends/megatron_utils/loss.py:513-589`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L513-L589)

**证据钩子与症状。** 先核对 sampling params，再比较逐 token rollout/train logprob；训练 loss 会报告 `train_rollout_logprob_abs_diff`。[`slime/backends/megatron_utils/loss.py:1136-1151`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/loss.py#L1136-L1151) 若偏差只在 temperature 或 top-p 打开后系统出现，应先查支持集 replay，而不是直接归因于权重或 kernel。

**为什么 L1 通过仍不够。** 同一 token 可以同时属于 full-softmax 和 nucleus，但它在两个归一化域中的概率不同；token ids 相等不等于 behavior distribution 相等。

> [!warning] 文档/接口能力与实现闭环不同
> CLI 和 SGLang 请求都支持 `rollout_top_k`，但固定基线的通用 `Sample` 字段与训练重放只实现 top-p nucleus ids/offsets，没有 top-k 支持集 payload。[`slime/utils/arguments.py:343-353`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L343-L353) [`slime/utils/types.py:121-126`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/types.py#L121-L126) 因此“能用 top-k 采样”不能写成“训练侧已能 exact replay top-k distribution”。

## 6. L3 MoE 路由：专家集合相同还不够，顺序也会进入数值语义

**问题背景与不变量。** MoE 的离散路径要求每个 response token、每个 MoE layer 的 top-k expert id 对齐。SGLang 可返回 `[token, layer, topk]` 路由；Sample 在 partial append 时按 `routed_experts_start_len` 拼接，并检查 token 行数、层数与 router top-k。[`slime/utils/types.py:352-395`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/types.py#L352-L395) RolloutManager 还会拒绝维度错误、空 capture，以及 MoE 层全零的可疑 PP capture，避免把缺失数据误当成 expert 0。[`slime/ray/rollout.py:107-140`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L107-L140)

**源码事实：`--use-rollout-routing-replay` 强制重放离散路由。** actor 把 rollout 路由按 PP/VP 本地层写入各 `RoutingReplay`，再在 logprob forward、训练 forward 与重算过程中消费；cursor 与消费阶段的具体边界见下一节。[`slime/backends/megatron_utils/actor.py:307-354`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L307-L354) [`slime/utils/routing_replay.py:78-140`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/routing_replay.py#L78-L140)

**源码事实：严格 GLM-5 对齐不依赖 `--use-rollout-routing-replay`。** alignment bridge 只对已注册、非 grouped router 使用与 SGLang 一致的 `torch.topk(sorted=False)`；注释明确指出，即使 expert set 相同，top-k column 顺序也会改变 DeepEP owner 的 BF16 累加顺序。[`slime/utils/routing_replay.py:49-75`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/routing_replay.py#L49-L75) 维护的 GLM-5 e2e gate 显式断言没有启用 rollout routing replay，以验证真实 router 与 experts 都参与训练。[`tests/test_glm52_6layer_deterministic_e2e.py:417-434`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_glm52_6layer_deterministic_e2e.py#L417-L434)

**证据钩子与症状。** 先比较 expert set，再比较 top-k 顺序，最后看第一处分叉是否落在 MoE 层。典型症状是 dense 层一致、进入首个 MoE 层后出现稀疏大偏差，或 expert id 相同但 combine 输出已有细小误差。

**为什么 L2 通过仍不够。** 相同采样支持集只约束 LM head 的输出分布定义；隐藏层中的 router 仍可能因微小数值差异跨过 top-k 边界，选择另一条计算图。

> **分析判断**：`--use-rollout-routing-replay` 是强诊断/校正手段，不是“原生训推路由相等”的证据。它能把离散 expert id 固定下来，却也会掩盖 router 本身为何分叉；因此应同时保留“自然路由 gate”和“强制 replay 定界”两种测试。

### 6.1 两个开关与四个 replay stage

`--use-routing-replay` 在训练侧记录 router 自然选择，用于训练 forward/backward 保持离散路径；`--use-rollout-routing-replay` 改为从 Sample 的 rollout expert ids 填充记录，并在参数校验中强制开启前一个开关。两者不能都简称成同一个功能。环境变量 `ENABLE_ROUTING_REPLAY=1` 启用 router wrapper，而 `ROUTING_REPLAY_STAGE` 指定此次调用如何消费记录：

| stage | router 行为 | actor 中的使用时机 |
|---|---|---|
| `fallthrough` | 自然计算 top-k，不新增 replay 记录 | ref/teacher forward |
| `record` | 自然计算 top-k，并将 ids 记到该 router 的队列 | 仅训练侧 replay 的 old/current logprob forward |
| `replay_forward` | 按 forward cursor 取 ids，从当前 scores gather 概率 | rollout 路由重放的 old/current logprob forward；以及训练 closure 的普通 forward |
| `replay_backward` | 按 backward cursor 取 ids，从当前 scores gather 概率 | actor 进入 train 前设定的环境；训练 forward 返回后恢复，供 backward 重算时的 router 调用读取 |

`model.train_one_step` 的 forward closure 会临时将 stage 设成 `replay_forward`，模型 forward 返回后恢复原 stage（actor 进入 train 前设为 `replay_backward`）；因此不能把“进入 train 时设置 backward”理解成训练 forward 也消费 backward cursor。每次取值校验 token 数和 top-k shape，超出已记录列表会索引失败。

`RoutingReplay.assert_all_consumed` 提供两个 cursor 都等于记录数的诊断，但本仓冻结基线没有调用它，不能宣称默认训练结束会自动验证未消费的记录。rollout 路由的独立 logprob forward 后清 forward cursor，使训练 forward 能从头消费；训练结束清所有记录及 lazy resources。强制 ids 仍从当前 scores 取概率，因而没有冻结 router probability 的数值计算或梯度。

源码阅读：`slime/utils/arguments.py::slime_validate_args`；`slime/utils/routing_replay.py::get_routing_replay_compute_topk`、`RoutingReplay`；`slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.fill_routing_replay`、`train_actor`；`slime/backends/megatron_utils/model.py::train_one_step`。

## 7. L4 kernel 与精度：路由相同后，数值仍可能逐层偏离

**问题背景与不变量。** dense/MoE GEMM、attention、RMSNorm、LM head、KV cache 和量化链都可能改变舍入与规约顺序。官方文档把严格 train/rollout logprob alignment 限定为 **GLM-5 结构**，要求 deterministic SGLang、batch-invariant DeepGEMM/DeepEP 和专用 Megatron patch；支持范围包括 DSA、block-FP8 forward、BF16 backward、FP32 router、匹配精度的 BF16 LM head 和 BF16/FP8 KV cache。[`docs/zh/advanced/reproducibility.md:53-69`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/reproducibility.md#L53-L69)

**源码事实：alignment utilities 是显式替换层。** shared env 固定 CUBLAS/NCCL/Transformer Engine 行为，开启 DeepGEMM batch invariance、DeepEP/DSA 配置，并指示 Megatron 借用 SGLang 的 RMS、router、RoPE 与 sparse MLA 路径。[`slime/backends/megatron_utils/alignment/env.py:19-56`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/env.py#L19-L56) combined hook 再安装 global batch-invariant ops、各 RMSNorm、dense/MoE forward、router GEMM、DeepEP bridge 与可选 layerwise dump。[`slime/backends/megatron_utils/alignment/deepgemm_forward.py:1110-1147`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L1110-L1147)

DeepGEMM 对齐 forward 还复制 SGLang 的 block-FP8 量化路径；Blackwell 分支必须复现 quantize→requantize 的有损链，单次量化不会 bit-match rollout。[`slime/backends/megatron_utils/alignment/deepgemm_forward.py:460-502`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L460-L502)

**证据钩子与症状。** layerwise dumper 记录 input ids、packed sequence offsets 和选定 decoder/module 输出；缺 input 或缺选定层会立即失败。[`slime/backends/megatron_utils/alignment/layerwise_alignment.py:41-113`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/layerwise_alignment.py#L41-L113) 第一处分叉若稳定落在 attention、norm、GEMM 或 KV 边界，才有资格继续做 kernel/precision 二分。

**为什么 L3 通过仍不够。** route ids 只决定调用哪些 experts，不决定 expert GEMM 的量化、输入 dtype、累加次序或 combine 精度；相同专家仍可产生不同 hidden states。

### 7.1 DeepGEMM MoE：替换整个专家 MLP 的舍入路径

该对齐模块面对的是“同一专家，但乘法、量化与 router probability 的先后顺序不同”。`deepgemm_moe_forward.py` 将整块 `TEGroupedMLP` 的 forward 替换为 block-FP8 grouped fc1 → SwiGLU → block-FP8 grouped fc2 → FP32 router probability multiply，再以自定义 autograd 实现 BF16 dgrad/wgrad 与解析 SwiGLU/probability 梯度。将 probability 移到 fc2 之后是对齐点，单独替换两个 linear 不能表达这条边界。

<!-- Figure spec: TB paired forward/backward paths for one expert row. Forward quantized grouped fc1, SwiGLU, quantized fc2 then router probability FP32. Backward saved BF16 inputs/weights and incoming gradient drive BF16 dgrad/wgrad and analytic activation/probability derivatives, not differentiating integer quantization. Reader can locate probability multiplication and distinct backward precision. -->
```mermaid
flowchart TB
    X["一个 expert 的 1 条有效 row<br/>BF16 输入与参数"] --> PAD["pad 到 128 rows<br/>127 条仅为计算对齐"]
    PAD --> F1["block-FP8 grouped fc1"]
    F1 --> A["SwiGLU 激活"]
    A --> F2["block-FP8 grouped fc2"]
    F2 --> P["最后以 FP32 乘 router probability<br/>仅还原有效 row，padding 不成新 token"]
    P --> D["反向输入梯度"]
    X -.-> SAV["保存 BF16 输入与参数<br/>及激活、probability 中间量"]
    D --> BW["解析 probability / SwiGLU 梯度<br/>BF16 dgrad 与 wgrad GEMM"]
    SAV --> BW
    BW --> OUT["输入、专家参数、router probability 梯度<br/>不把 FP8 forward 当作全程 FP8 训练"]
```

图中 forward 的两次 block 量化和 fc2 后的 probability multiply 是需要复现的数值顺序；backward 用另一组公式和精度传递训练梯度。route ids 相同只固定了 expert rows 的归属，不能证明这些运算相同。

启用方式是把 `slime.backends.megatron_utils.alignment.deepgemm_forward.enable_deepgemm_all_forward` 配置到 `--custom-megatron-before-log-prob-hook-path`，并把同模块的 `enable_deepgemm_all_forward_before_train_step` 配置到 `--custom-megatron-before-train-step-hook-path`，并通过 `--megatron-deepgemm-moe-forward-layers` 选择全局零起始 decoder layers；可用 `--megatron-deepgemm-moe-forward-modules` 改模块后缀，默认 `mlp.experts`。层号使用 `layer_number-1`，因此 PP 各 rank 的局部 layers 从 0 起编号不会误选层。`SGLANG_DEEPGEMM_BATCH_INVARIANT=1` 通过 SGLang wrapper 或 DeepGEMM setter 设置 batch-invariant 模式，找不到可用 API 或设置后读回不生效会报错。

额外的 ordered DeepEP bridge 要求 `deterministic_mode=True`、`moe_enable_deepep=True` 且选择非空 MoE layers；`MEGATRON_USE_SGLANG_ROUTER_GEMM=1` 则使组合 hook 加装 router GEMM。配置开关本身不能证明外部 kernel 一致：slime 只验证调用、布局与可用 API，DeepGEMM/DeepEP/SGLang 内部实现及兼容 patch 仍是依赖边界。

wrapper 明确要求 TP=1、expert TP=1、bias-free SiLU gated `TEGroupedMLP`、BF16 参数和 hidden/FFN 维度为 128 的倍数；拒绝与 Megatron FP8、QAT、SwiGLU clamp 或已单独包装的 expert linears 叠加。forward 将各 expert 的 rows 按 128 对齐，意味着额外 padding 与重排存储；`SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP` 控制 forward 分组，`SLIME_DEEPGEMM_MOE_GROUPED_BF16_BACKWARD` 控制 grouped backward，另外两个 BF16 backward group/padded-byte 环境项限制临时分配。这些是同一对齐计算内的资源控制，不是通用低精度训练模式。

源码阅读：`slime/backends/megatron_utils/alignment/deepgemm_forward.py::_enable_deepgemm_all_forward`；`slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py::enable_deepgemm_moe_forward`、`install_deepgemm_moe_forward`、`_validate_parallelism`、`_validate_te_grouped_mlp`、`_deepgemm_grouped_moe_forward`、`enable_sglang_deepep_moe_alignment`。

### 7.2 Deterministic route kernels：固定写者和累加顺序

这组 Triton kernels 不重新选择专家，而是把 token→route 的复制与反向梯度还原做成确定性操作。输入是 BF16 `tokens×hidden`、整数 `tokens×topk` 的 `output_index`，其中负值表示无效 slot；scatter 将 token row 写到唯一 expert-major route row。反向按每 token 的 top-k 列顺序累加，每次加法都经过 BF16 舍入，再转回 FP32 accumulator，复现参考实现逐次 BF16 in-place add 的可见边界，不能换成一次 FP32 总和后再 cast。

```mermaid
flowchart LR
    T["token A、B<br/>A slots 映射 2、0<br/>B slots 映射 1、无效"] --> S["scatter forward<br/>route rows 为 A、B、A"]
    S --> G["route 梯度 g0、g1、g2"]
    G --> A["A 按 slot 顺序累加<br/>先 g2 后 g0<br/>每次加后 BF16 舍入"]
    G --> B["B 仅取 g1<br/>无效 slot 贡献零"]
    A --> O["每个 token 梯度独占写者<br/>无 atomic 冲突"]
    B --> O
```

`ordered_route_grad` 将 token 梯度乘 FP32 top-k 权重再写回唯一 route row；`compact_route_positions` 用有效 slot 的前缀和生成紧凑的 `(token,slot)` 索引，并以异步断言检查实际 route 数与 metadata handle 的预期数量。避免 `nonzero` 的数据相关输出大小回传 CPU，是该 compaction 调用点的具体目的。

这些 kernels **没有独立 CLI 开关**：已进入对齐 bridge 后，CUDA 张量直接选择它们；CPU 使用索引循环参考路径。公共校验要求 CUDA、连续 BF16 二维值和连续整数二维 mapping；ordered grad 另外要求 FP32 权重。每个输出元素唯一写者是上游 route mapping 的不变量，底层公共 validator 不扫描检查 destination 是否重复。

源码阅读：`slime/backends/megatron_utils/alignment/deterministic_route_kernels.py::scatter_routes_forward`、`scatter_routes_backward`、`ordered_route_grad`、`compact_route_positions`；`slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py::_DeepEPScatterWithDeterministicBackward`、`_ordered_route_backward`、`_scatter_deepep_routes_with_padding`。

## 8. L5 批次与并行执行：同名 kernel 也可能采用不同的规约顺序

**问题背景与不变量。** rollout 的 continuous batching、SGLang TP/DP/EP 与训练的 dynamic micro-batch、TP/PP/CP/EP 会改变 shape、分片和 collective 顺序。strict alignment 的 global batch-invariant hook 之所以不仅设置 DeepGEMM，是因为 Megatron 与 SGLang 分属不同进程，RMS reduction、BMM、FP32 matmul 和 log-softmax 仍可能走普通 batch-shaped kernel。[`slime/backends/megatron_utils/alignment/deepgemm_forward.py:690-711`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L690-L711)

pipeline 也不是透明维度：每个 PP stage 的首个本地 layer 没有本地前序 residual sum，因此对齐 hook 必须在每个 PP 边界替换 standalone RMSNorm，而不能只处理 global layer 0。[`slime/backends/megatron_utils/alignment/deepgemm_forward.py:736-752`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L736-L752) 最初的 dense DeepGEMM probe 甚至明确限制 TP=1，以排除 row-parallel partial-sum rounding 这个混杂变量。[`slime/backends/megatron_utils/alignment/deepgemm_forward.py:1-12`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L1-L12) [`slime/backends/megatron_utils/alignment/deepgemm_forward.py:580-588`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/alignment/deepgemm_forward.py#L580-L588)

**证据钩子与症状。** 固定 rollout dump，改变训练并行配置与 dynamic batching，再比较 grad norm 或逐层输出。当前 `parallel_check` 覆盖 DP、TP2/PP2/CP2 组合以及 TP4、PP4、CP4，并复用同一 rollout dump。[`tests/test_qwen3_0.6B_parallel_check.py:12-21`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_qwen3_0.6B_parallel_check.py#L12-L21) [`tests/test_qwen3_0.6B_parallel_check.py:104-142`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_qwen3_0.6B_parallel_check.py#L104-L142) 失败若只随 batch size、packing 或某一并行维度出现，应优先查 shape-dependent kernel 和 collective，而不是重做权重同步。

**为什么 L4 通过仍不够。** “两侧都调用某种 GEMM/attention”没有固定输入分块和 reduction tree。除非算子本身满足 batch invariance，或端到端测试覆盖目标拓扑，否则局部 kernel 对齐不能推出完整并行执行对齐。

## 9. 确定性推理能保证到什么范围

**项目文档事实。** reproducibility 文档把 SGLang deterministic inference 与 Megatron deterministic mode 组合成 bitwise experiment reproduction recipe，要求使用 FlashInfer、卸载 FlashAttention 3，并设置 NCCL、Transformer Engine 与 CUBLAS 环境变量。[`docs/zh/advanced/reproducibility.md:3-26`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/reproducibility.md#L3-L26)

**源码事实。** rollout 开启 deterministic inference 后，为同一 prompt group 中第 $i$ 条 sample 设置 `rollout_seed+i`；训练端 deterministic mode 则固定 cudnn 选择并要求 PyTorch 使用确定性算法，缺确定性实现时不是 warn-only。[`slime/rollout/sglang_rollout.py:109-112`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L109-L112) [`slime/rollout/sglang_rollout.py:317-327`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L317-L327) [`slime/backends/megatron_utils/initialize.py:88-93`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/initialize.py#L88-L93)

> [!important] 不能把三种确定性混成一个开关
> 1. **rollout 自身可复现**：固定请求、seed 与 SGLang deterministic stack，重复生成稳定；
> 2. **training 自身可复现**：固定训练输入与 Megatron deterministic stack，重复执行稳定；
> 3. **train/rollout cross-engine alignment**：两套引擎对同一 token 的 logprob 或 hidden state 对齐。
>
> 前两项各自成立不能自动推出第三项。固定基线把第三项的严格支持明确限定在 GLM-5 专用 alignment stack，而不是所有模型/backend 的普遍保证。[`docs/zh/advanced/reproducibility.md:53-78`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/reproducibility.md#L53-L78)

## 10. CI 如何把“看起来一致”变成分级证据

固定 workflow 的 CPU matrix 注册了 Sample、rollout validation、train dump、layerwise comparison 等契约测试；GPU matrix则单独注册并行检查和 GLM-5 两级 gate。[`.github/workflows/pr-test.yml.j2`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/.github/workflows/pr-test.yml.j2#L65-L112)

| 门禁 | 固定了什么 | 断言 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| CPU 接口测试 | Sample/top-p/routing 数据形状与比较工具 | 异常输入会失败、比较逻辑正确 | 元数据约定与诊断工具不会静默失真 | 真实 GPU kernel 一致 |
| parallel check | 同一 rollout dump，不同 DP/TP/PP/CP | grad norm 近似相等 | 目标 Qwen 配置下并行训练结果不过度漂移 | 逐 token、逐位训推对齐 |
| GLM-5 e2e | 真实 weight update、DSA、FP8 DeepGEMM、DeepEP EP8 | mean train/rollout logprob abs diff 小于 `1e-6` | 该固定 recipe 的最终 behavior gate | 其他模型、拓扑、backend |
| GLM-5 layerwise | 同一短序列，decoder layer 0–5 | 所有匹配 hidden element 最大误差为 0 | 首六层边界 bitwise 对齐 | 长生成最终分布与训练更新 |

GLM-5 e2e 的 fixture 是单机 EP8、3 dense + 3 MoE layer，配置明确开启 deterministic SGLang、FP8 KV、DeepEP 和 alignment hooks。[`tests/test_glm52_6layer_deterministic_e2e.py:178-257`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_glm52_6layer_deterministic_e2e.py#L178-L257) 测试在 actor 内执行阈值断言，layerwise 变体再用 `max-hidden-diff 0` 比较 0–5 层。[`tests/test_glm52_6layer_deterministic_e2e.py:305-387`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/tests/test_glm52_6layer_deterministic_e2e.py#L305-L387)

> [!note] CI 文档描述与实际门禁粒度
> CI 文档把 `run-ci-precision` 概括为“不同并行设置下的数值一致性”；固定 workflow 中该 job 实际只注册 `test_qwen3_0.6B_parallel_check.py`。[`docs/zh/developer_guide/ci.md:84-94`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/developer_guide/ci.md#L84-L94) [`.github/workflows/pr-test.yml.j2`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/.github/workflows/pr-test.yml.j2#L48-L53) 该测试最终比较 grad norm，容差为 `rel_tol=abs_tol=0.01`，所以它是并行训练近似一致门禁，不是 bitwise logprob gate。[`slime/backends/megatron_utils/model.py:917-936`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/model.py#L917-L936)

## 11. 约束：更强确定性的成本与退化边界

下表中带定位符的是实现明确行为；其余成本是由 payload 形状、执行次数和受限配置直接推导的**分析判断**。

| 强化手段 | 得到的诊断能力 | 代价或限制 |
|---|---|---|
| 保存 rollout/train dump | 固定输入并逐 sample 对账 | CPU/磁盘 I/O、存储与敏感数据治理成本 |
| 行为策略元数据 | 重建选中 token 的概率、top-p 候选集与路由 | 每个 token 都要传输元数据；top-p/routing 数据量随序列长度、层数和 top-k 增长 |
| mismatch 重算 | 直接观测 train/rollout logprob 差 | 某些配置需额外训练侧 forward；实现会明确记录这一点 [`slime/utils/arguments.py:1849-1860`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1849-L1860) |
| `--use-rollout-routing-replay` | 隔离 MoE 离散路由差异 | 保存/搬运 per-token、per-layer expert ids，并不能证明自然 router 对齐 |
| 确定性算法 | 多次运行结果稳定，无法确定执行时立即暴露错误 | 禁止没有确定性实现的算子，限制可选 kernel；当前示例还要求使用 FlashInfer/移除 FA3 |
| GLM-5 alignment stack | layerwise zero 与低于 `1e-6` 的 e2e gate | 模型结构、patch、DeepGEMM/DeepEP、KV dtype 和部分拓扑受限；dense probe 起步为 TP1 |

> **分析判断**：确定性越强，吞吐优化器可自由选择的算法、batch shape 与通信路径越少；但固定基线没有给出一项可泛化到所有模型的统一性能税，因此不应在这里编造百分比。工程上更稳妥的是分级启用：日常记录轻量 behavior 指标，异常时先 replay 固定输入，再在可复现的小模型/短序列上打开 layerwise 与严格 kernel stack。

### 11.1 无法严格对齐时，校正不是证明

slime 可以用 rollout logprob 作为 behavior old policy，或重算 train old logprob 后做 mismatch/TIS；参数校验禁止 `use_rollout_logprobs` 与 `use_tis` 同时开启。[`slime/utils/arguments.py:1050-1082`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1050-L1082) [`slime/utils/arguments.py:1849-1859`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L1849-L1859)

> **分析判断**：importance correction 能降低已测量的 behavior mismatch 对目标函数的偏差，却不能修复错 token、缺 metadata、混权重版本或错误 expert capture。下一节在这些可比前提下解释 TIS/MIS；其外层并行缩放仍由 [[15_slime_loss_parallelism_analysis]] 负责。

### 11.2 TIS/MIS 校正机制

TIS 要修正的是训练侧旧策略与实际 rollout 行为策略之间的采样分布差。它与 PPO 当前/旧策略 ratio 是两个量：设训练旧策略、rollout 策略对同一已生成 token 的 logprob 分别为 `train_log_probs` 与 `rollout_log_probs`，则校正权重是两者差的指数；PPO 的 ratio 仍描述当前训练策略相对旧策略的更新。

`vanilla_tis_function` 逐 token 计算权重并 clamp 到 `[tis_clip_low,tis_clip]`，乘入 pg loss，原 loss masks 原样返回。例如原权重为 `[0.5,4]`、边界为 `[0,2]` 时变成 `[0.5,2]`，两 token 都保留。默认路径没有 rejection，也不采用直觉上的“超过阈值一律丢弃”。`icepop_function` 则把区间外权重置零，是另一可加载函数，不能与 vanilla clamp 混写。

下面图例指定 `tis_level=token`、`tis_batch_normalize=False`、`use_rs=False`，基础 mask 为 `[1,1]`；不沿用后文 YAML 的 batch-normalize 默认值。示例 `compute_mis_weights` 把加权与拒绝分开：先按 token、sequence 或 geometric 计算 log-ratio（分别为逐 token 值、完整 mask 下的和、完整 mask 下的均值），将它限制到 `[-20,20]` 再取 exp；`truncate` 只 cap 上界，`clip` 限制两端，`mask` 保留 IS 权重但令区间外 mask 归零。可再做独立 RS 和逐 token 极低 ratio 的整序列 veto。`use_tis=False` 会在计算 mismatch metrics 后提前返回，因此本实现的 `use_rs=True` 不能单独启动拒绝流程。

```mermaid
flowchart TB
    I["同一轨迹的 train-old / rollout 比<br/>两个 token 权重 0.5、4"] --> V["vanilla clamp 到 0 至 2<br/>权重 0.5、2；mask 1、1"]
    I --> M["MIS mask 区间 0.5 至 2<br/>权重 0.5、4；mask 1、0"]
    V --> L["主 loss 使用校正权重与返回 mask<br/>基础分母仍来自原 mask"]
    M --> L
    I -.-> Q["mismatch 与拒绝率指标<br/>保留 pre-RS 统计范围"]
```

MIS 的 CP wrapper 先 all-gather 每条 response 的两套 logprob，按完整 mask 计算上述规则，再切回本 rank 的 weights 和 metrics；返回的 modified masks 仍是完整 response 形态，交由通用 reducer 再取本地片段。可选 `tis_batch_normalize` 只在该次函数收到的 sequences 上计算权重均值，没有 DP all-reduce；它是 micro-batch 作用域，不能写成全局 optimizer batch 的统一归一化。该选项只实现 token/sequence 分支，与 geometric 同用会抛 `ValueError`。

核心 parser 的 TIS 参数只有以下两项数值阈值；相邻开关决定选择与观测入口：

| 核心 CLI | 默认 | 固定基线的作用 |
|---|---|---|
| `--use-tis` | false | policy loss 开启校正 hook |
| `--tis-clip` | 2.0 | vanilla/ICEPOP 的上界 |
| `--tis-clip-low` | 0 | vanilla/ICEPOP 的下界 |
| `--custom-tis-function-path` | None | 替换 vanilla，例如 `examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp` |
| `--get-mismatch-metrics` | false | 也进入 hook，校验强制要求 custom path；与 rollout logprobs 同用时额外重算训练 logprob |
| `--use-rollout-logprobs` | false | 直接用 rollout 作为 PPO old logprob；核心校验禁止与 `use_tis` 同开 |

MIS 示例 README 用 `--tis-*` 描述以下选项，但冻结核心 parser 未注册这些 CLI flags。真实接入是 `--custom-config-path examples/train_infer_mismatch_helper/mis.yaml`，在 args 上设置同名下划线属性；下表数值是示例 YAML，不是 parser 默认值。

| MIS 属性／README 名称 | 示例值 | 规则 |
|---|---|---|
| `tis_level`／`--tis-level` | token | token / sequence / geometric |
| `tis_mode`／`--tis-mode` | truncate | truncate / clip / mask；README 列举少了 mask，以实现为准 |
| `tis_lower_bound`／`--tis-lower-bound` | 0.5 | None 时取 `1/tis_upper_bound`；truncate 不使用下界 |
| `tis_upper_bound`／`--tis-upper-bound` | 2.0 | IS 上界；clip/mask 断言下界小于上界 |
| `tis_batch_normalize`／`--tis-batch-normalize` | true | 当前函数 batch 的权重均值归一；仅 token/sequence |
| `use_rs`、`rs_level` | true、token | IS 之后附加拒绝，聚合层级可独立选择 |
| `rs_lower_bound`、`rs_upper_bound` | null、null | 回退至 IS 对应边界 |
| `rs_veto_threshold` | 0.0001 | 任意有效 token ratio 太低时拒绝整条 sequence |

YAML 在部分核心断言之后才应用，且可覆盖已有属性；不要把“核心 parser 禁止某组合”当作自定义 YAML 写回后再次校验的保证。源码没有在加载 custom config 后重跑整套校验。

| 日志键 | 实际统计内容 |
|---|---|
| `train/tis`、`train/tis_abs`、`train/tis_clipfrac` | vanilla 的原始 ratio、距 1 的绝对差、被 clamp 的比例 |
| `train/mis_tis_weight_before_bound`、`train/mis_tis_weight_after_bound` | MIS 限界前后 IS 权重 |
| `train/mis_tis_truncate_fraction` | MIS truncate 超上界比例；clip/mask 模式分别产生 `mis_tis_clip_fraction_low/high`、`mis_tis_mask_fraction_low/high` |
| `train/mis_rs_mask_fraction_low/high`、`train/mis_is_ratio_mean_final` | RS 拒绝比例、最终权重均值 |
| `train/mis_kl`、`train/mis_k3_kl`、`train/mis_training_ppl`、`train/mis_rollout_ppl` | MIS hook 的概率差与两端 perplexity；不要求 use_tis |
| `train/train_rollout_logprob_abs_diff` | 主 policy 路径的 logprob 绝对差，使用重建后的主 reducer；与 hook 的 pre-RS metrics 范围不同 |

hook 产出的指标保留 pre-RS reducer，主 loss 与其他报告项使用返回 mask 重建 reducer，但保留原始完整 rollout 分母；外层 token normalizer 也不随 rejection 重算。这样一个被拒绝的 token 不贡献梯度，却仍计入拒绝率的基准。权重校正与严格对齐的区别仍成立：IS 的截断/拒绝改变估计量，不能反向证明跨引擎 bitwise 相等。

源码阅读：`slime/backends/megatron_utils/loss.py::vanilla_tis_function`、`icepop_function`、`policy_loss_function`；`examples/train_infer_mismatch_helper/mis.py::compute_mis_weights`、`compute_mis_weights_with_cp`、`add_ppl_metrics`、`truncate`、`clip`、`mask`；`examples/train_infer_mismatch_helper/mis.yaml`；`slime/utils/arguments.py::get_slime_extra_args_provider`、`slime_validate_args`。

## 12. 最小排查流程

```mermaid
flowchart TD
    A["发现训推不一致"] --> W{"权重快照与版本一致吗"}
    W -->|否| W0["停在 L0<br/>检查更新结果与 serving version"]
    W -->|是| I{"token、response span 与 mask 一致吗"}
    I -->|否| I0["检查 Sample 转换、join 键与 CP 切分"]
    I -->|是| S{"采样参数与支持集一致吗"}
    S -->|否| S0["检查 temperature、top-p、top-k 与回放元数据"]
    S -->|是| R{"MoE expert set 与顺序一致吗"}
    R -->|否| R0["用 `--use-rollout-routing-replay` 固定路由<br/>隔离 router 差异"]
    R -->|是| K{"首个 layerwise 分叉在哪里"}
    K --> K0["二分 attention、norm、GEMM、KV 与量化"]
    K0 --> P["改变 batch、packing 与并行拓扑复验"]
    P --> C["已确定来源与规模后<br/>再选择 TIS、rejection 或 rollout logprob"]
```

这是一条逐层排除链：上层身份或版本没有通过时，不应直接下钻 kernel，更不能先用 importance correction 掩盖输入契约错误。

1. **权重**：先跑 `check_weight_update_equal`，再查 engine version 与 Sample `weight_versions`；失败就停在 L0。
2. **输入**：保存 rollout/train dump，按 `rollout_position` 或 `sample_index` join，逐项比较 tokens、response span 和 mask。
3. **采样**：核对 temperature、top-p/top-k；top-p 非 1 时确认 ids/offsets 完整，再看逐 token logprob 差。
4. **路由**：MoE 先比 expert set，再比 top-k order；必要时用 `--use-rollout-routing-replay` 固定路由来判断差异是否来自 router。
5. **kernel/精度**：从第一处 layerwise 分叉二分到 attention、norm、GEMM、KV 或量化边界。
6. **并行执行**：用同一 dump 改变 batch/packing/TP/PP/CP/EP，只在目标拓扑上声明通过。
7. **校正**：前六层已确定 mismatch 来源与规模后，才选择 rollout logprob、TIS 或 rejection；不要用算法修正掩盖契约错误。

官方 debug 指南同样建议从首步 rollout/ref logprob、推一训一 KL 和固定输入的 rollout-only/train-only 开始，而不是直接把异常归因于 RL loss。[`docs/zh/developer_guide/debug.md:3-24`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/developer_guide/debug.md#L3-L24) [`docs/zh/developer_guide/debug.md:26-55`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/developer_guide/debug.md#L26-L55)

## 13. 发展趋势

本节离开“固定基线是什么”，因此只写有源码注释可锚定的在途改动，整节标为推断。

> [!note] 推断：锚点是源码注释原文，方向判断是本页的重建
> **“权重相等就该 KL 为零”这条直觉，在固定基线自己的 CI 里留着两处未定性的残差。** 其一，rollout 侧首轮 `log_probs` 与 `ref_log_probs` 的严格相等断言被注释掉、改成 `< 1e-8` 的容差，并挂着 `# TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.`。[`slime/backends/megatron_utils/data.py:361-363`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/data.py#L361-L363) 其二，PPO 配合 KL clipping 时首步 `ppo_kl` 同样不严格为零，断言上方写着 `# TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.`。[`slime/backends/megatron_utils/model.py:903-904`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/model.py#L903-L904)
>
> 值得对照的是：同样两处代码里另有一条**已经**被解释清楚的不相等——`--use-rollout-routing-replay` 为 actor 前向重放 rollout 路由，而 reference 前向有意走自然路由，因此初始 actor/ref KL 本就不应逐位为零；源码把这条原因单独写成注释，并据此放宽断言。[`slime/backends/megatron_utils/data.py:350-353`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/data.py#L350-L353) [`slime/backends/megatron_utils/model.py:905-907`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/model.py#L905-L907)
>
> **由此可推断**：两条 TODO 的存在方式（注释掉严格断言、保留容差、写明“不知道是否符合预期”）说明项目自己仍把“零残差”当作尚未达成、而非已经放弃的目标；后续要么它们被归因并收紧断言，要么被改写成显式的容差契约。在此之前，把这里的 `1e-8` 读成“本应为零的舍入噪声”会误判：本页第 1 节那句“权重相等不能证明行为相等”在这里不是方法论主张，而是项目 CI 中两条尚未定性的实测残差。

## Related Pages

- [[12_slime_sample_datasource_analysis]] — 行为策略元数据、token mask、中断续写与调试数据导出所依赖的 Sample 语义边界。
- [[15_slime_loss_parallelism_analysis]] — 校正 hook 如何进入 loss reducer，以及 DP/CP 下的目标函数缩放。
- [[16_slime_weight_sync_analysis]] — L0 权重快照背后的 pause/flush、拓扑转换与提交协议。
- [[31_slime_posttraining_stability_analysis]] — 训推 mismatch 如何与版本陈旧、数值异常和训练失稳共同进入稳定性诊断。
- [[22_slime_low_precision_training_rollout_analysis]] — 训练权重、rollout 权重与 KV cache 三条精度轴的完整实现边界。
- [[07_training_reliability/10_determinism_and_numerical_reliability_analysis|确定性与数值可靠性]] — batch invariance、浮点规约与跨引擎数值可靠性的通用背景。
