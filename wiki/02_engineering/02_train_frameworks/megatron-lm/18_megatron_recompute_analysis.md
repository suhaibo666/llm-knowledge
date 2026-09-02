---
title: "Megatron-LM 激活重计算(Activation Recomputation / Checkpointing)深度解析"
---

# Megatron-LM 激活重计算(Activation Recomputation / Checkpointing)深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；落在本轮改动文件（`recompute.py`、`transformer_config.py`、`multi_token_prediction.py`、`hybrid_block.py`、`cuda_graphs.py`）上的 39 处 `path:line` 引用已在新基线下逐条打开重核，行号漂移已就地更正；本轮新增内容见 §3.3 的 2026-09-01 [!update]。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/transformer/transformer_block.py`(`_checkpointed_forward`)、`megatron/core/recompute.py`(`checkpointed_forward`,整层重计算的共享实现)、`megatron/core/tensor_parallel/random.py`(`checkpoint`)、`megatron/core/transformer/transformer_config.py`(`recompute_*` 配置)
> 配套阅读:`22_megatron_memory_optimization_analysis.md` §3.3(激活换出 offloading)、五份并行文档
> 定位:"第二层补遗"第①份。激活重计算是与并行轴正交的**省显存**手段。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-09-01。推进至基线 `85902ef59`，重核全部受影响引用；§3.3 新增「层类型二分改为能力标志」一节。

---

## 1. 背景：激活显存墙——并行轴切不到的最后一块

### 1.1 重计算是什么

反向传播需要前向算出的**中间激活**。默认情况下,前向把所有中间激活都留在显存里等反向用 —— 这是训练显存的最大头。**激活重计算(recomputation / checkpointing)**:前向时**只保留少量"检查点"张量、丢弃其余激活**;反向需要某段激活时,**从最近的检查点重新跑一遍前向**把它算出来。

**用算力换显存**:多付一遍(部分)前向计算,换回大量激活显存。

### 1.2 与"激活换出"的关系

| 手段 | 换什么 | 文档 |
|------|--------|------|
| **重计算 recompute** | 用**算力**换显存(反向多跑一遍前向) | 本文 |
| **换出 offload** | 用 **PCIe 带宽**换显存(激活搬 CPU) | `22_megatron_memory_optimization_analysis.md` §3.3 |

二者正交,可叠加。算力有余量、显存紧 → recompute;PCIe 有余量 → offload。

### 1.3 两个维度:粒度 × 方法

```
recompute_granularity ─┬─ "full"      整层重计算 ──┬─ recompute_method "uniform"
                       │                           └─ recompute_method "block"
                       └─ "selective" 按子模块重计算 ── recompute_modules=[core_attn, moe_act, ...]
```

### 1.4 激活显存墙的量级与代价

一个 transformer 层前向产生大量中间激活:QKV、attention 分数矩阵 `[s,s]`、softmax 输出、FFN 中间 `[s,b,H]`……总量随**层数 `L`** 线性、随**序列 `s`** 线性(attention 分数甚至 `O(s²)`)。

并行轴只能部分缓解:TP 切 `1/t`(配 SP)、CP 切 `1/cp`、PP 用 1F1B 把在世激活压到 `O(p)`(见 `15_megatron_pp_schedulers_analysis.md`)。但当这些都用上仍然 OOM,就需要**直接丢弃激活、反向重算** —— 这是与并行正交的最后一道省显存手段。

**代价量化**:反向计算量 ≈ 2× 前向。不重计算时一步 = 前向 1 + 反向 2 = 3 个单位;**整层重计算**时反向里要重跑一遍前向 → 1 + (2+1) = 4 个单位 ≈ **+33% 计算**。所以重计算不是免费的,要按需用。

---

## 2. 为什么这么设计：先挑"显存大、重算便宜"的那一段

省激活显存有三条明面上的路：**并行切分**（TP/CP/PP，见五份并行文档）、**换出到 CPU**（offload，§1.2）、**丢掉再重算**（recompute）。前两条各有硬上限——并行切分受限于设备数与通信预算，offload 受限于 PCIe 带宽。重计算是唯一一条"只花本地算力"的路，因此被留作最后一道闸门。

真正的设计决断不在"要不要重计算"，而在**重计算谁**。源码把判据写在了配置字段的 docstring 里，不需要推断：

| 决断 | 选中的路线 | 被否掉的替代 | 源码给出的判据 |
|---|---|---|---|
| 重算粒度 | `selective`（按子模块） | `full`（整层） | `megatron/core/transformer/transformer_config.py:604-610`：默认目标 `core_attn` 是 "the memory intensive part of attention"，而 "These memory intensive activations are also **less compute intensive**, which makes activation checkpointing more efficient for LLMs (20B+)"，并直接引用 *Reducing Activation Recomputation in Large Transformer Models*（https://arxiv.org/abs/2205.05198） |
| 子模块用哪种 checkpoint | 输出也丢（output-discarding） | 只丢内部激活（标准 checkpointing） | `megatron/core/tensor_parallel/random.py:985-989`：被 checkpoint 的函数其**输出会在重算时重新生成**，所以 "the output store is not technically needed"；哪些模块走哪种由 `megatron/core/transformer/transformer_config.py:651-653` 明确列出 |
| 层维度怎么切 | `block`（只重算装不下的前 N 层） | `uniform`（全层均分重算） | `megatron/core/transformer/transformer_block.py` 的 `block` 分支注释 "fully use the device memory removing redundant re-computation"（见 §3.2）——判据是**避免多余重算**，而不是省得最多 |
| checkpoint 由谁实现 | Megatron / TE 自带 | PyTorch 的 `torch.utils.checkpoint` | fp8/fp4 要接管量化上下文（`megatron/core/transformer/transformer_block.py:677-689`），非 fp8 要接管 TP 的 RNG tracker（`megatron/core/tensor_parallel/random.py:718`）——见 §4.4 |

**判据本身是一句话**：选"显存占用 / 重算代价"比值最大的那一段。`core_attn` 的 `O(s²)` 注意力激活正好落在这个比值的顶点上——显存最大、重算只是 softmax/dropout。整层 `full` 之所以不是首选，正因为它把 GEMM 这种"显存中等、重算昂贵"的部分也一并重算了。

> [!note] 推断
> 以下三点是本页依据源码行为与提交历史重建的权衡，**源码本身并未陈述**，不要当作作者自陈的意图引用：
> - **为什么 recompute 与 offload 并存而不是二选一**：源码只在 `megatron/core/transformer/transformer_config.py:2293-2296`（`cpu_offloading` 与 recompute 互斥）和 `:2541-2551`（整层 `moe` 重算与 MoE 内子模块 offload 互斥）里写了**互斥关系**，从未说明"算力换显存 vs 带宽换显存"这层分工。§1.2 的那张对照表是本页的归纳。
> - **为什么 `uniform` 仍然保留**：`block` 在判据上严格更优（不做多余重算），源码没有解释为何不废弃 `uniform`。合理的重建是 `uniform` 只需一个 `recompute_num_layers` 就能覆盖任意层数、不必按 stage 调参，但这句在源码里查无出处。
> - **为什么默认值是 `core_attn` 而不是空**：`megatron/core/transformer/transformer_config.py:2337-2338` 在 `recompute_modules is None` 时无条件填 `["core_attn"]`，但没有注释说明这是"性价比默认"还是历史兼容。注意它与 `:2396-2402` 的警告（TE fused attention 下 `core_attn` 重计算多半是白费）存在张力，见 §7。

---

## 3. 全量重计算(`recompute_granularity='full'`)

整个 transformer 层作为重计算单元:前向只存层的**输入** `[s,b,h]`,层内所有激活全丢;反向重跑整层前向。由 `_checkpointed_forward`(`megatron/core/transformer/transformer_block.py:581`)实现,`recompute_method` 选两种切法。

### 3.1 `recompute_method='uniform'`

把本 PP stage 的层均匀分成若干**块**,每块 `recompute_num_layers` 层,对每块的**输入**做一个检查点:

```python
# megatron/core/transformer/transformer_block.py:702
while layer_idx < self.num_layers_per_pipeline_rank:
    chunk_end = min(layer_idx + recompute_num_layers, num_layers_per_pipeline_rank)
    hidden_states, context = checkpoint_handler(custom(layer_idx, chunk_end))   # 整块作一个检查点
    layer_idx += recompute_num_layers
```

特点:**所有层都重计算**,检查点数 = `层数 / recompute_num_layers`。`recompute_num_layers` 越大,检查点越少、省得越多,但反向重算的粒度越粗。极端 `recompute_num_layers = 全部层` → 整个 stage 只留一个输入检查点,省到极致。

### 3.2 `recompute_method='block'`

只对**前 `recompute_num_layers` 层**做检查点(重计算),其余层正常跑(留全部激活):

```python
# megatron/core/transformer/transformer_block.py:724
for layer_idx in range(num_layers_per_pipeline_rank):
    if recompute_skip_num_layers <= layer_idx < recompute_num_layers + recompute_skip_num_layers:
        hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))  # 这些层重计算
    else:
        hidden_states, context = custom(layer_idx, layer_idx + 1)(...)                 # 这些层不重计算
```

特点:**部分层重计算**。动机 —— 配合 1F1B,越靠前的 PP stage 在世 microbatch 越多(`15_megatron_pp_schedulers_analysis.md` §②.4),激活压力越大;`block` 让你**只把恰好装不下的那几层重计算**,其余层省下重算开销。"fully use the device memory removing redundant re-computation"(源码注释)。

### 3.3 `uniform` vs `block`

| | uniform | block |
|--|---------|-------|
| 重计算范围 | 所有层 | 前 `recompute_num_layers` 层 |
| 省显存 | 多 | 可调(按需) |
| 多余重算 | 可能有(全算) | 少(只算装不下的) |
| 适合 | 显存极紧、要全省 | 显存差一点点、精打细算 |

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `85902ef59`。
> **全量重计算逻辑抽成共享实现，并支持 HybridModel（#4496，118933a85）**
> - 新增 `megatron/core/recompute.py`，把整层重计算的 `checkpointed_forward`（含 `custom` / `chunk_runner` / `uniform`+`block` 两种切法）抽成**可复用函数**（`megatron/core/recompute.py:21`；`custom` 在 `:54`、`chunk_runner` 在 `:124`）；Hybrid（Mamba/GDN + attention 混合）模型的 `megatron/core/models/hybrid/hybrid_block.py` 现也走同一套：`recompute_granularity == 'full' and self.training` 时调 `checkpointed_forward(...)`（`megatron/core/models/hybrid/hybrid_block.py:1121-1122`，import 在 `:28`）。此前 full 重计算只在 GPT `transformer_block` 内可用，现 HybridModel 同样支持。
> - **GPT 路径行号已在新基线下重核**（机制仍有效）：`megatron/core/transformer/transformer_block.py` 的 `_checkpointed_forward` 在 `:581`、`checkpoint_handler` 在 `:674`、`uniform` 分支在 `:702`、`block` 分支在 `:724`（旧基线 `ee3f1ffa…` 依次为 `:464` / `:542` / `:570` / `:592`）。GPT 的 `transformer_block._checkpointed_forward` 仍保留自有实现（与 `megatron/core/recompute.py` 的共享版并存），故 §3 的描述对 GPT 依然成立。

> [!update] 2026-09-01（基线 `85902ef59`）
> **共享 `checkpointed_forward` 的层类型二分改为"能力标志"三分（#6704，`cd0e795f0`）**
>
> - **原来的问题**：上一条 [!update] 里那个共享实现，对"这一层接受哪些 kwarg"只做了二分——`isinstance(layer, TransformerLayer)` 为真就传全套 kwarg 并按 `(hidden_states, context)` 二元组接收返回值，否则**一律按裸 `MambaLayer` 处理**：把 `context` / `context_mask` / `attention_bias` / `padding_mask` / `input_ids` 五个 kwarg 全部 pop 掉，并把返回值当单张量。Hybrid 栈里包了一层 mHC 的 `HyperConnectionHybridLayer`（`megatron/core/models/hybrid/hybrid_block.py:75`）恰好落进这个 else 分支，而它的接口两头都对不上：它包着的可能是 MoE 层，**需要**路由元数据（`padding_mask`、hash MoE 用的 `input_ids`），却**不接受** TransformerLayer 接口里的三个 cross-attention kwarg，返回值又是二元组。于是"hybrid + 整层重计算"这一组合在旧基线上是坏的。
> - **选中的路线 vs 被否掉的替代**：显而易见的修法是在 `recompute.py` 里 `import hybrid_block` 再加一个 `isinstance(layer, HyperConnectionHybridLayer)` 分支。源码注释直接否掉了它——**会造成循环导入**（`megatron/core/recompute.py:79-81`：kwarg 集合的构造注释写明 hybrid mHC 包装层"expose an explicit capability flag so this module does not need to import hybrid_block (which would create a circular import)"）。`hybrid_block` 反过来要 `from megatron.core.recompute import checkpointed_forward`（`megatron/core/models/hybrid/hybrid_block.py:28`），共享模块再回指它就成环。取而代之的判据是**在包装类上挂一个类属性作能力声明**：`supports_hybrid_recompute_kwargs = True`（`megatron/core/models/hybrid/hybrid_block.py:121`），共享侧只用 `getattr(layer, "supports_hybrid_recompute_kwargs", False)` 探测（`megatron/core/recompute.py:97`）。**方向被反转了**：不再由重计算模块去认识每一种层，而是由层自己声明"我吃哪一套 kwarg"。
> - **落地形态**：`custom_forward` 内的分支变成三路（`megatron/core/recompute.py:95-115`）——`TransformerLayer` 走全套 kwarg；带能力标志的层只 pop `context` / `context_mask` / `attention_bias` 三个（`:102-103`），保留 `padding_mask` / `input_ids`，并按二元组解包；其余裸层仍 pop 五个、按单张量解包并把 `context` 置 `None`。
> - **边界（本页推断，源码未陈述）**：这是**约定而非契约**——任何层只要挂上这个属性就会被当作"接受路由元数据、返回二元组"，共享侧不做进一步校验，属性名拼错则静默退回裸层分支、错误延迟到 kwarg 不匹配才暴露。注释里"This also covers a wrapper around a MambaLayer; the wrapper narrows kwargs for its inner layer."（`megatron/core/recompute.py:100-101`）说明责任已经下推给包装层自己。
> - **同一 PR 的配套修复（与 §4.2 的 `layernorm` 输出丢弃重计算直接相关）**：mHC 的 fast path 绕过了 `TransformerLayer._forward_post_mlp()`，而后者才是"在 MLP 反向之前丢弃 pre-MLP layernorm 检查点"的地方；现在 fast path 里补上了这一步（`megatron/core/models/hybrid/hybrid_block.py:549-554`，条件为 `layer.recompute_pre_mlp_layernorm` 或 mHC 检查点管理器命中）。缺了它，selective `layernorm` 在 hybrid mHC 路径上省不到显存——这正是 §7.3 那条"输出丢弃 checkpointing 有个静默前提"的一个真实翻车案例：不报错，只是优化失效。

---

## 4. 选择性重计算(`recompute_granularity='selective'`)

### 4.1 动机:不是所有激活都"值得"重计算

各类激活的"显存占用 / 重算代价"比差异巨大:
- attention 分数矩阵 `[s,s]` —— 显存巨大(`O(s²)`),但 softmax/dropout 重算很**便宜**。
- 大矩阵乘的输入 —— 显存中等,但重算要再做一次 **GEMM**,贵。

**选择性重计算**只挑"显存大、重算便宜"的子模块下手 —— 这是 Megatron 2022 论文(Korthikanti et al.)的核心洞察:**选择性重计算注意力,能省掉大部分激活显存,而计算开销仅 ~1-2%**(远低于全量重计算的 +33%)。

### 4.2 `recompute_modules`

`recompute_modules` 列表选要重计算的子模块(字段定义 `megatron/core/transformer/transformer_config.py:631-635`;合法取值集合 `megatron/core/transformer/transformer_config.py:2342-2353`),可选:

| 子模块 | 说明 |
|--------|------|
| `core_attn` | 注意力核(softmax/dropout)—— **默认目标**,显存最重、重算最便宜 |
| `moe_act` | GroupedMLP 的激活函数 |
| `layernorm` | input_layernorm / pre_mlp_layernorm |
| `mla_up_proj` | MLA 上投影 + RoPE |
| `mlp` | dense MLP 子模块 |
| `moe` | 整个 MoE 层 |
| `shared_experts` | 共享专家 |
| `mhc` | hyper connections |
| `gdn` | 整个 GatedDeltaNet 模块（**新增**，见下方 [!update]） |
| `gdn_norm_out` | GDN 输出 norm 的输出丢弃重计算 —— 在新基线下**已恢复**且与 `gdn` 互斥，见下方 [!contradiction] |

> [!warning] 读序提示：下面这条 [!update] 的**最后一条「历史更正」已被推翻**——它说 `gdn_norm_out` 全仓已不存在，而自基线 `71092579` 起该模块已被 #6088 加回、与 `gdn` 并存且互斥。**先读紧随其后的 [!contradiction]**，再把这条 [!update] 当作旧基线的历史记录读。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `85902ef59`。
> **新增 `gdn`：GatedDeltaNet 整模块 selective 重计算（#5296，8dc6e6676）**
> 新基线的合法 `recompute_modules` 集合（`megatron/core/transformer/transformer_config.py:2342-2353`）含 `"gdn"`：core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts, mhc, **gdn**, gdn_norm_out。
> - 语义：`gdn` 在 `recompute_granularity='selective'` 下把**整个 GatedDeltaNet**（in_proj → conv1d → gated delta rule → gated norm → out_proj）包进一次 checkpoint 整体重算，用**标准 checkpointing**（非输出丢弃）。实现：开关字段在 `megatron/core/ssm/gated_delta_net/common.py:304-307`（`self.recompute_gdn`），重算包装在 `megatron/core/ssm/gated_delta_net/gdn.py:251-267`（`tensor_parallel.checkpoint(_checkpointed_compute, False, hidden_states)`，核心计算抽到 `_forward_compute`）；KDA 变体共用同一开关（`megatron/core/ssm/gated_delta_net/kda.py:321`）。
> - 约束：仅当模型为 hybrid、或 `experimental_attention_variant` 属 GDN 家族时可用（`megatron/core/transformer/transformer_config.py:2381-2389`）。
> - **历史更正**：先前 #4715（ff5264c33）曾引入过更细的 `gdn_norm_out`（只对 GDN 输出 norm + HP→CP all-to-all 做**输出丢弃** checkpointing），在 `dev@232c478d4` 时点它确实已被 #5296 的整模块 `gdn` 取代、全仓查无此项。

> [!contradiction] 上一条「`gdn_norm_out` 已不存在」自基线 `71092579` 起**不再成立**（在 `85902ef59` 下依然如此）：`3549dc62a`（#6088「Refactor: extract and split common logic between GDN & GDN2」，cherry-pick #5843）把 `gdn_norm_out` **重新加了回来**。新基线上二者**并存且互斥**——合法集合同时含 `"gdn"` 与 `"gdn_norm_out"`（`megatron/core/transformer/transformer_config.py:2342-2353`），并显式禁止同时指定（`megatron/core/transformer/transformer_config.py:2391-2395`，报错文案为 “'gdn' recomputes the full GDN-family layer, including gated norm.”）；`gdn_norm_out` 的开关字段 `self.recompute_norm_out` 也回到了 `megatron/core/ssm/gated_delta_net/common.py:302-306`。同一次重构还把原来的单文件 `megatron/core/ssm/gated_delta_net.py` 拆成了包 `megatron/core/ssm/gated_delta_net/`（`common.py` / `gdn.py` / `kda.py`），故本节原先指向 `gated_delta_net.py:267-269`、`:374-387` 的两处 locator 已按新布局改写。

### 4.3 标准 checkpointing vs 输出丢弃 checkpointing

selective 下有**两种**底层机制(README MoE §Fine-grained Recomputation 区分):

- **标准 checkpointing**(`core_attn`、`mlp`、`moe`):torch 风格 —— 存子模块**输入**,丢内部激活,反向重跑子模块前向。
- **输出丢弃 checkpointing**(`moe_act`、`layernorm`、`mla_up_proj`):更激进 —— 连子模块的**输出**也在前向丢掉,反向时重算。对这类"输出大、可由输入廉价重算"的子模块,比标准 checkpointing 省得更多。

### 4.4 `te_checkpoint` vs `tensor_parallel.checkpoint`

`checkpoint_handler`(`megatron/core/transformer/transformer_block.py:674`)按精度选实现:
- `fp8` / `fp4` → `te_checkpoint`(TransformerEngine 的版本,正确处理 fp8 量化上下文与 RNG;分支在 `megatron/core/transformer/transformer_block.py:677-689`)。
- 否则 → `tensor_parallel.checkpoint`(`megatron/core/tensor_parallel/random.py:718`,Megatron 自带,正确处理 TP 的 RNG tracker —— 保证重算的 dropout 与原前向用同一随机种子)。

> 重计算必须保证 dropout 等随机算子**重算与原前向结果一致**,否则梯度错。这就是为什么 checkpoint 要接管 CUDA RNG tracker。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `85902ef59`。
> **两个 checkpoint 健壮性修复**（注：上文 `checkpoint_handler` 在新基线为 `megatron/core/transformer/transformer_block.py:674`）
> - **重计算 × 训练 CUDA graph（#3919，ada8dfe6f）**：`tensor_parallel.checkpoint`（`megatron/core/tensor_parallel/random.py:718`）在函数入口新增——`if is_graph_warmup() or is_graph_capturing(): return function(*args)`（`megatron/core/tensor_parallel/random.py:728-729`）。**CG warmup/capture 期间直接跳过 checkpoint 包装**，因为被捕获的 graph 已记录全部算子、反向无法在已捕获 graph 内再跑一遍重算（与 `CheckpointWithoutOutput` 行为一致）。否则 recompute + 训练 CG 会冲突。
> - **MTP + packed sequence 重计算崩溃（#4593，2b77d32b1）**：`MultiTokenPredictionLayer` 新增自己的 `_checkpointed_forward`（`megatron/core/transformer/multi_token_prediction.py:2364`），仿照 `transformer_block`：把 `attention_mask` / `rotary_pos_emb` / `packed_seq_params` 等 **kwarg 张量经 `custom_forward` 闭包捕获**（`tensor_parallel.checkpoint` 只 stash 位置参数、`te_checkpoint` reentrant 实现也只追踪位置张量输入，kwarg 张量不进重算反向路径），并按精度路由 fp8/fp4 → `te_checkpoint`、否则 `tensor_parallel.checkpoint`。修复 packed-sequence 下 MTP 重计算 crash。

---

## 5. 与其他机制的配合

### 5.1 PP 的部分激活检查点

`num_microbatches_with_partial_activation_checkpoints`(见 `15_megatron_pp_schedulers_analysis.md` §②.2 `max_outstanding_backprops`):1F1B 里**早期 microbatch 多、激活压力大**,可只对前若干个 microbatch 做(部分)重计算,后面的不做。这是"重计算 × microbatch 维度"的精细调度。

### 5.2 与并行轴的关系

- 重计算与 TP/PP/CP/EP/DP **完全正交**,叠加使用。
- selective `moe` 重计算与 MoE 的 `moe_act`/`expert_fc1`(新基线另加 `fused_group_mlp`)**换出互斥**(`megatron/core/transformer/transformer_config.py:2541-2551`:整层重算了就不能再换出层内子模块)。
- `fp8` delayed scaling 不支持 `moe_act`/`layernorm` 重计算(`megatron/core/transformer/transformer_config.py:2413-2419`)。
- 整层 `full` 重计算与 `cuda_graph_impl` 有约束(`megatron/core/transformer/transformer_config.py:3460-3464`,非 selective 时需 `full_iteration`)。

### 5.3 与 offload 叠加

重计算(换算力)+ offload(换带宽)可同时开,各管一部分激活 —— 当算力和 PCIe 都还有余量时,二者分摊省显存压力。

---

## 6. 开销分析与选型

### 6.1 开销总表

| 方案 | 省激活显存 | 计算开销 | 适合 |
|------|-----------|---------|------|
| 不重计算 | —— | 基准 | 显存够 |
| selective(`core_attn`) | 大(去掉 `O(s²)` 的注意力激活) | **~1-2%** | **首选**:几乎免费,优先开 |
| full `block`(部分层) | 可调 | 按重算层数,`<33%` | selective 还不够、差一点 |
| full `uniform`(全层) | 最大 | **~+33%** | 显存极度紧张 |

### 6.2 选型决策

```
激活显存 OOM?
├─ 否 ──► 不重计算(最快)
└─ 是 ──► 先开 selective recompute(--recompute-granularity selective
          --recompute-modules core_attn)—— ~1-2% 开销换大块显存,几乎必开
          │
          ├─ 还差一点 ──► full + block,只重算装不下的那几层
          │
          └─ 还不够 ──► full + uniform(全层重算,+33%);
                        或叠加激活 offload(22_megatron_memory_optimization_analysis §3.3)
```

### 6.3 一句话总结

- **重计算 = 用算力换激活显存**:前向丢激活,反向从检查点重跑前向补回。
- **粒度**:`full`(整层,uniform 全算 / block 部分算)vs `selective`(按子模块)。
- **关键洞察**:selective 重计算 `core_attn` —— 去掉 `O(s²)` 的注意力激活,开销仅 ~1-2%,是性价比最高的一档,优先开;全量 `uniform` 才到 +33%。
- **正交性**:与所有并行轴正交;与 offload 互补(算力 vs 带宽)。

---

## 7. 约束

重计算不是"多花一点算力"这么干净。以下每条都能落到源码的 guard 上（全部按基线 `85902ef59` 核对）。

### 7.1 前提：每个 selective 子模块都带自己的准入条件

| 子模块 | 额外前提 | locator |
|---|---|---|
| `moe_act` | 必须开 `moe_grouped_gemm` | `megatron/core/transformer/transformer_config.py:2360-2363` |
| `mla_up_proj` | 必须是 `multi_latent_attention` | `megatron/core/transformer/transformer_config.py:2365-2369` |
| `gdn_norm_out` | 必须是 hybrid 模型或 GDN 家族 attention 变体 | `megatron/core/transformer/transformer_config.py:2371-2379` |
| `gdn` | 同上；且**不能与 `gdn_norm_out` 同时指定** | `megatron/core/transformer/transformer_config.py:2381-2389` / `:2391-2395` |
| `mhc` | 必须 `enable_hyper_connections=True`，且**不能与 `mlp` 同用** | `megatron/core/transformer/transformer_config.py:2440-2447`（docstring `:643-645`） |
| `shared_experts` | 与 `--moe-shared-expert-overlap` 互斥 | `megatron/core/transformer/transformer_config.py:2404-2411` |

### 7.2 不变量：`full` 与 `selective` 的参数是两套，不能混填

- `full` 必须同时给出 `recompute_method`（`uniform`/`block`）与 `recompute_num_layers`，否则构造期报错（`megatron/core/transformer/transformer_config.py:2305-2322`）。
- `selective` 必须**不给** `recompute_num_layers`（`megatron/core/transformer/transformer_config.py:2323-2329`）——因为 "'selective' always uses all layers."（docstring `:611`）。**层维度的取舍只存在于 `full` 的 `block` 方法里**，selective 侧刻意不提供"只对部分层做"的旋钮。
- `distribute_saved_activations` 与 `sequence_parallel` 互斥（`megatron/core/transformer/transformer_config.py:2331-2335`）。

### 7.3 代价与失效条件

- **`core_attn` 重计算可能是白花的**。`megatron/core/transformer/transformer_config.py:2396-2402` 在选了 `core_attn` 时无条件 `warnings.warn`：用 transformer_engine 实现时 core attention 很可能已是**融合版本**（激活本就不落显存），"For fused attention, you have no need to set 'core_attn' to recompute."。这是对 §6.1 那句"几乎免费、优先开"的重要限定——在 TE fused attention 下它可能既不省显存、又白付重算。
- **输出丢弃 checkpointing 有个静默前提**：调用方必须保证被丢弃的输出**确实被后继模块直接保存用于反向**，否则丢了也省不到（`megatron/core/tensor_parallel/random.py:991-992`：“to save memory with this method, the caller should make sure that the discarded output tensors are directly saved in the following modules for backward computation”）。这一条不会报错，只会让优化失效。
- **fp8 × selective 的两道门**：`delayed` scaling 不支持 `moe_act`/`layernorm` 重计算；即便换 recipe，这两个模块的 fp8 重计算也要求 `transformer-engine>=2.6.0dev0`（`megatron/core/transformer/transformer_config.py:2413-2425`）。
- **与 offload 的互斥面**：`cpu_offloading` 与**任何**重计算粒度互斥（`megatron/core/transformer/transformer_config.py:2293-2296`）；整层 `moe` 重算与 `moe_act`/`expert_fc1`/`fused_group_mlp` 的细粒度 offload 互斥（`:2541-2551`，报错文案写明"整层已经包进 checkpoint，再 offload 层内激活是冗余且会出错"）。§5.3"重计算与 offload 可同时开"成立的范围，仅限**互不重叠的模块集合**。
- **与 CUDA Graph 的门槛**：非 selective（即 `full`）重计算只在 `cuda_graph_impl == "full_iteration"` 下被允许（`megatron/core/transformer/transformer_config.py:3460-3464`）；且 CG warmup/capture 期间 `tensor_parallel.checkpoint` 会**直接跳过** checkpoint 包装（`megatron/core/tensor_parallel/random.py:728-729`，见 §4.4 的 [!update]）。
- **MTP 是残缺支持**：`recompute_method='block'` 在 MultiTokenPredictionLayer 上**未实现** —— 只 warn 然后**整段跳过重计算**（`megatron/core/transformer/multi_token_prediction.py:2507-2511`）；`uniform` 路径则硬性要求 `recompute_num_layers == 1`（`megatron/core/transformer/multi_token_prediction.py:2502-2504`）。开了 MTP 又配 `block`，显存不会按预期下降，而且不会报错。

### 7.4 故意不做的事

- 不提供"selective + 只重算部分层"的组合（见 §7.2）。
- 不为重计算提供自动选型：所有阈值（重算哪些模块、`block` 重算几层）都由用户给，源码只做合法性校验，不做代价估算。
- `--moe-layer-recompute` 这条老旋钮不再新增能力，只做兼容转写（`megatron/core/transformer/transformer_config.py:2427-2438`，见 §8）。

---

## 8. 发展趋势

以下每条都锚定基线 `85902ef59` 上实读到的 in-flight 痕迹（PR 记录、`TODO`、弃用告警）。**"方向"一栏是本页的推断，不是源码自陈的规划。**

| 锚点（实读） | locator | 推断的方向 |
|---|---|---|
| `gdn_norm_out` 被 #6088 加回来，与整模块 `gdn` **并存且显式互斥** | `megatron/core/transformer/transformer_config.py:2342-2353` / `:2391-2395`（详见 §4.2 的 [!contradiction]） | GDN 家族的重算粒度仍在细分——先合并成整模块、又拆回"只重算 gated norm + 布局还原"，说明粒度的最优点尚未定 |
| 整层重计算被抽成共享实现 `megatron/core/recompute.py`，HybridModel 接入，但 GPT 的 `transformer_block._checkpointed_forward` **仍保留自有副本** | `megatron/core/recompute.py:21`、`megatron/core/models/hybrid/hybrid_block.py:1121-1122`；GPT 侧 `megatron/core/transformer/transformer_block.py:581`（详见 §3.3 的 [!update]） | 两份实现并存是**未收敛**状态，收敛方向是 GPT 侧也改调共享函数 |
| 共享 `checkpointed_forward` 的层类型判别改用**层自报的能力标志** `supports_hybrid_recompute_kwargs`，理由是避免循环导入 | `megatron/core/recompute.py:79-81` / `:97`、`megatron/core/models/hybrid/hybrid_block.py:121`（详见 §3.3 的 2026-09-01 [!update]） | 层与重计算入口之间正在从"入口认识所有层类型"转向"层声明自己的 kwarg 契约"。当前只有一个布尔标志、只区分三档，若 hybrid 侧继续增加包装层，这里预计要长成一组更正式的能力位或协议 |
| `--moe-layer-recompute` 已 deprecated，告警要求改用 `--recompute-modules moe_layer` | `megatron/core/transformer/transformer_config.py:2427-2431` | 该告警**与实现不一致**：合法模块集合里根本没有 `moe_layer`（`:2342-2353`），代码自动追加的是 `"moe"`（`:2437-2438`）。迁移文案预计要修，或补一个 `moe_layer` 别名 |
| MTP 的 `block` 重计算未实现，`# TODO: implement block-based recompute for MTP` | `megatron/core/transformer/multi_token_prediction.py:2508` | MTP 侧的重计算会补齐到与主 block 对齐（当前只有 `uniform` 且强制 `recompute_num_layers==1`） |
| mHC 重计算的相位划分未完成：`TODO: partition checkpoints across phases so recompute_until replays only what each barrier needs`，并要补 `BEFORE_MLP_BWD` barrier | `megatron/core/transformer/mhc_recompute.py:62-65` | 目前所有 checkpoint 都挂在 `BEFORE_COMBINE_BWD` 一个相位上，重算粒度粗于调度粒度；方向是按反向 barrier 切分重算集合 |
| 重计算 × CUDA Graph 的已知代价：`TODO: (jiemingz) [interaction with recompute]` —— 重算会在反向里重跑前向，capture 时挂的 buffer metadata 因此丢失，导致**额外拷贝** | `megatron/core/transformer/cuda_graphs.py:1067-1072` | §7.3 那条"full 重计算只允许 full_iteration CG"的限制正是这条张力的外化；方向是让 metadata 在重算路径上存活，从而放宽组合 |

> [!note] 推断
> 上表右栏与本节的归纳均为本页依据锚点做的外推。源码只陈述了 `TODO`/告警本身，没有陈述路线图；引用时请回到左栏的 locator，不要引用右栏。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef59`(2026-09-01 由 `71092579` 重定基线;更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准。本文是"第二层补遗"3 份之①(激活重计算),后续:② 优化器内部、③ FP8 精度 + CUDA Graph + 算子融合。*

## Related Pages

- [[23_megatron_precision_cudagraph_fusion_analysis]] · [[15_megatron_pp_schedulers_analysis]]
- [[22_megatron_memory_optimization_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
