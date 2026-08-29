---
title: "Megatron-LM RL 后训练适配与训推一致性深度解析"
---

# Megatron-LM RL 后训练适配与训推一致性深度解析

> **源码基线**:`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`(`dev`,2026-08-27)
> **重定基线**:2026-08-28 由 `ee3f1ffa…`(2026-05-19)推进,跨 578 个提交;本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心:`megatron/core/resharding/`(refit)、`inference/`(推理引擎)、`inference/quantization/`(MXFP8)、`post_training/modelopt/`、`megatron/core/transformer/transformer_config.py`(`transformer_impl='inference_optimized'`)
> 配套阅读:`27_megatron_tp_fsdp_resharding_supplements_analysis.md` §5(refit 基础)、`17_megatron_parallelism_orchestration_analysis.md`、`14_megatron_ep_analysis.md`
> 定位:系统性专题。前面文档讲预训练;本文讲 **RL 后训练**(RLHF / GRPO / PPO)对 Megatron 提出的特殊需求,以及核心难题 **训推一致性(train-inference consistency)**。
> **三方分工**:本文是 Megatron 训练侧的 refit / 训推一致性实现(逐项收敛 gap);三平面机制视角(weight publish 协议、跨框架不变量)见 [[01_posttraining_infra_mechanism_analysis]] 第 6 节;verl 在 Megatron+vLLM 场景下的具体同步调用链见 [[33_megatron_vllm_weight_sync_analysis]]。
> **叙事顺序**:本页按五拍组织——背景 → 为什么这么设计(含被否掉的替代)→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**:2026-08-28。按五拍重排章节顺序;机制正文与既有引用未改。

---

## 0. 总览:Megatron Core 在 RL 里的角色

先澄清边界:**完整的 RL 训练环(GRPO/PPO 的 advantage 计算、loss、KL 约束)不在 `megatron/core` 里** —— 那是 NeMo-RL 等上层 RL 框架的事。Megatron Core 提供的是 RL 后训练的**底层积木**:

| 积木 | 代码 | 作用 |
|------|------|------|
| 训练模型 | 标准 Megatron 模型 | 跑 policy 梯度更新 |
| 推理引擎 | `inference/`(`dynamic_engine` 等) | rollout 阶段生成样本 |
| `inference_optimized` 实现 | `transformer_impl='inference_optimized'` | 推理专用的高效前向路径 |
| 推理量化 | `inference/quantization/`(MXFP8) | 推理模型低精度加速 |
| **Refit / Resharding** | `resharding/` | **每个 RL 迭代把训练模型权重搬进推理模型** |

本文重点是最后一项(refit)和它要保证的**训推一致性**。

---

## 1. 背景与设计取舍

### 1.1 背景:RL 后训练的结构与难点

一个 RL 后训练迭代(以 GRPO/PPO 为例)分两相:

```
┌─ ① Rollout(推理相)──────────────────────────────────┐
│  推理引擎用当前 policy 生成一批 (prompt → response)     │
│  顺带产出每个 token 的 logprob(rollout policy μ 下)    │
└────────────────────────┬──────────────────────────────┘
                         │
┌─ ② Train(训练相)──────▼──────────────────────────────┐
│  训练引擎对这批样本算 logprob(training policy π 下)、   │
│  advantage、importance ratio,做 policy 梯度更新         │
└────────────────────────┬──────────────────────────────┘
                         │
        ──── Refit:把更新后的权重搬回推理模型 ────► 下一迭代
```

**两相用的是两套引擎**:rollout 要快(KV cache、连续批处理、低精度),训练要能反向(标准 kernel、BF16、可微)。于是出现 RL 后训练的头号系统难题 ——

> **训推一致性**:同一个 (prompt, token),推理引擎算出的 logprob 和训练引擎算出的 logprob **不相等**。

这个不一致直接污染 PPO/GRPO 的 **importance ratio** `r = π_train(a|s) / μ_rollout(a|s)`:策略没变时 `r` 本应恒为 1,但引擎差异让它偏离 1。偏差逐 token、逐步累积 → 梯度有偏 → RL 训练发散或塌缩。

不一致的来源:

| 来源 | 说明 |
|------|------|
| **权重陈旧** | 推理模型的权重落后于训练模型 |
| **并行布局不同** | 训练 TP8×PP4,推理 TP2×PP1 —— 参数切分方式完全不同 |
| **精度不同** | 推理 MXFP8,训练 BF16 |
| **kernel 不同** | `inference_optimized` 的融合/KV-cache attention vs 训练 kernel |
| **批处理数值路径** | 连续批处理改变了 softmax/matmul 的规约顺序 |

Megatron 的解法不是"消灭全部差异"(做不到),而是**把差异逐项收敛到可控、可量化**,残差交给上层 RL 框架的 importance sampling 修正。下面逐项看。


### 1.2 为什么这么设计:四条取舍与它们各自否掉的替代

"把差异逐项收敛、残差交给 IS"只是结论。下面逐条给出源码陈述的理由与被否掉的替代;源码沉默处整段标为推断。

**① 权重同步走"在线重分片搬运",而不是"落盘 checkpoint 再由推理侧加载"。**
`resharding/` 的 README 开篇即定位:「Transfer model weights between different parallelism configurations (TP, PP, EP, DP) with optional format conversion (e.g. BF16 to MXFP8). **Used primarily in RL loops** to move weights from a training model to an inference model that may use a **different parallelism layout**」(`megatron/core/resharding/README.md:1-6`)。
真正把"重载"这条路堵死的是一条硬约束:MXFP8 变换必须**直接写进持久 buffer**——「The transform writes directly into persistent MXFP8Tensor buffers (via `.copy_()`) **so that CUDA-graph device-pointer captures remain valid across refits**」(`:115-117`),对应的调用契约是「Call `prepare_swap_model_weights` **once during initialization while the target model's parameters are still in BF16**」(`:42-46`)。
→ 判据(本页重建):任何"重建/重载推理模型"的方案都会换掉设备指针,让 [[31_megatron_inference_engine_analysis]] §7 里为 decode 捕获的一整套 CUDA Graph 全部失效——而 RL 主循环每迭代都要做一次。就地写回是唯一能与 CUDA Graph 共存的搬法。

**② plan 在 rank 0 集中构建一次并缓存,而不是每 rank 各自推导、也不是每次 refit 重算。**
README 把流程写成五步:各 rank 抽元数据 → `dist.gather_object()` 汇到 rank 0 → rank 0 **按名字**为每个目标参数找到匹配的源参数并路由到维度特定 planner(标准 TP 用 LCM tiling,Mamba `in_proj` 这类分区参数用 block-interleaved)→ 产出带全局唯一 `task_id` 的 `TransferOp` 对并 scatter 回去 → 「The plan is **cached** so repeated refits **skip steps 1-4**」(`megatron/core/resharding/README.md:87-97`)。
**被否掉的替代写在历史里**:`_PlanCacheKey` 原先只用 `(rank, src_config, dst_config, num_experts)`;#4762 补上 `src_rank_offset` / `dst_rank_offset`,注释即写明为什么——「Rank offsets **distinguish non-collocated configurations that would otherwise share the same (rank, sizes, num_experts) tuple** but route to different global ranks」(`megatron/core/resharding/refit.py:47-51`)。同一轮还把后端判定从 duck-typing 改成 `isinstance(refit_method, CopyService)`(§7 的 `[!update]`)。
→ 集中式 plan + 缓存这条路本身没被推翻,被推翻的是**缓存键选窄了**:两套非 collocated 配置会静默命中同一份 plan。

**③ `inference_optimized` 强制 fp32 router,源码给的理由是 decode 期的转换开销——不是精度。**
校验的报错信息原话:「`--transformer-impl='inference_optimized'` requires `--moe-router-dtype=fp32` **to avoid costly dtype conversions during decode**」(`megatron/core/transformer/transformer_config.py:2030-2034`)。
→ 值得留意口径:§4 正文把这条读成"与训练侧推荐一致、路由精度对齐"。那确实是它的**副作用**(训推两侧 router dtype 因此一致),但源码陈述的动机是**性能**。引用这条时请以报错信息为准。

**④ "现在是不是在推理"从 `eval()`/`train()` 模式改成进程级全局标志。**
这是本页范围内最清楚的一次替代被否:`MoELayer.train()` 重写(靠 `nn.Module` 的 mode 切 dispatcher)被 #4617 整段删除,改由 `InferenceMode.is_active()` 决定。根本原因与全部 locator 见 §4 的 `[!deprecated]`——`self.training` / `torch.is_grad_enabled()` / `inference_context is not None` 都分不清"引擎在做 rollout"与"训练相在用同一模型重算 RL logprob"(§5),而后者恰恰是本页的核心场景。

**⑤ 不追求 gap 归零,而是把 gap 变成一个被测量、被上报的量。**
§6 说残差交给上层 RL 框架做 IS 修正。基线源码里 Megatron 自己已经把这个比值算出来并上报:`update_inference_logprobs_group_stats` 用 `ratios = (old_logprobs - inference_logprobs).exp()` 直接算训练侧与推理侧的概率比,产出 `min/max/mean_piold_to_inf_prob` 以及 `min/max/mean_inf_train_prob_abs_diff`(`megatron/rl/rl_utils.py:452-483`,核心在 `:470-478`)。
→ 判据(本页重建):一个"做小但不为零"的差异只有在**可观测**时才是工程上可接受的;这组指标就是 §1 那张"不一致来源"表在运行期的量表。

> [!note] 推断
> 源码陈述的是**事实**:refit 的定位与"持久 buffer 保住 CUDA Graph 指针"这条约束、集中式 plan 的五步与缓存、`_PlanCacheKey` 补 rank offset 的注释、fp32 router 的报错理由、`InferenceMode` 取代 `train()` 重写、以及 π/μ 比值指标的存在。**把它们读成"因此排除了重载推理模型"、"因此 gap 必须可观测"这两条判据,是本页的重建**,源码从未这样自陈。要引用这几条判断,请回到 `megatron/core/resharding/README.md:1-6`、`:42-46`、`:88-97`、`:115-117`、`megatron/core/resharding/refit.py:47-51`、`megatron/core/transformer/transformer_config.py:2030-2034`、`megatron/rl/rl_utils.py:452-483` 这几个 locator,不要引用本段推断。

---

## 2. 解法①:Refit 消除"权重陈旧"

`resharding/`(详见 `27_megatron_tp_fsdp_resharding_supplements_analysis.md` §5)。每个 RL 迭代结束,`swap_model_weights(train_model, infer_model)` 把训练模型的**最新权重**搬进推理模型。

```python
from megatron.core.resharding import prepare_swap_model_weights, swap_model_weights
prepare_swap_model_weights(src_model=train_model, target_model=infer_model)  # 初始化时一次
# RL 循环里反复:
swap_model_weights(train_model, infer_model, refit_method="nccl")
```

意义:rollout 用的永远是**刚更新过的** policy,不存在"用三步前的旧权重采样"的陈旧偏差。`r` 偏离 1 的第一大来源被直接消除。

---

## 3. 解法②:Refit 的布局/格式保真

光"搬过去"不够,**搬得要精确** —— 否则布局/格式转换本身又引入差异。`resharding/` 在两处下了功夫。

### 3.1 并行布局重映射(`megatron/core/resharding/planner.py`)

训练和推理模型的并行度通常不同(训练求吞吐、推理求延迟)。refit 的 plan 构建:
- 每个 rank 抽取参数元数据(shape、TP/EP/PP 切分、所在组)。
- rank 0 集中构建 `ReshardPlan`:对每个目标参数,**按名字找到对应的源参数切片**,用维度特定的 planner 计算精确映射 —— 标准 TP 用 **LCM tiling**(最小公倍数分块,保证源/目标两种切分都能整除对齐),Mamba `in_proj` 这类分区参数用 block-interleaved。
- 产出 `TransferOp`(谁把哪段发给谁),scatter 给各 rank。

结果:推理模型每个分片拿到的,**就是训练权重对应位置的精确切片**,布局转换零误差。

### 3.2 MXFP8 格式转换(`megatron/core/resharding/transforms.py` `MXFP8ReshardTransform`)

推理模型常用 MXFP8(`fp8_recipe='mxfp8'` + `inference_optimized`)。refit 把 BF16 训练权重转成 MXFP8 时,难点在 **scale 布局**:

- **2D scale**:每行 scale 对应一行数据,切片独立 → 收到 BF16 切片立即逐片量化。
- **1D scale**(FlashInfer swizzled 布局):scale 跨整个权重张量交织编码,**部分更新会破坏布局** → 必须把所有 BF16 切片**累积进一个完整 buffer,等齐了一次性量化**。

而且转换直接 `copy_()` 写进**持久 MXFP8Tensor buffer**(其设备指针被 CUDA Graph 捕获),保证多次 refit 后 CUDA Graph 仍有效。

意义:推理模型拿到的是训练权重**忠实的 MXFP8 量化版本**,而不是"重新量化的近似" —— 把"精度不同"这一项差异收敛成单纯的"BF16 vs MXFP8 数值精度差",干净、可量化。

---

## 4. 解法③:`inference_optimized` —— 显式、独立、受控的推理路径

`transformer_impl` 有三选(`megatron/core/transformer/transformer_config.py:1436`):`local` / `transformer_engine` / **`inference_optimized`**。

`inference_optimized` 是 RL rollout 用的专用路径:
- `use_inference_optimized_layers`:推理优化的线性层(`megatron/core/tensor_parallel/inference_layers.py`,如推理专用的 all-gather)。
- `inference_grouped_gemm_backend`:MoE 推理的 grouped GEMM 后端(`flashinfer` / `torch` / `vllm`)。
- `inference_moe_token_dispatcher_type`:推理专用 MoE dispatcher(`nccl` / `nvls`)—— `megatron/core/transformer/moe/moe_layer.py` 的 `train()` 重写(`:421`)在 eval 模式自动切到推理 dispatcher、train 模式切回(见 `14_megatron_ep_analysis.md`)。
- 强制 `--moe-router-dtype=fp32`(`:2030`)—— 源码给的理由是 **decode 期的 dtype 转换开销**(报错原文 "to avoid costly dtype conversions during decode");训推两侧 router dtype 因此一致是**副作用**,不是动机。

> [!deprecated] `megatron/core/transformer/moe/moe_layer.py` 的 `train()` 重写自 `dev@232c478d4`(2026-06-16)起已被**移除**(#4617);基线 `71092579` 下 `git grep "def train("` 在该文件仍为 0 命中,本条依然成立。上一段正文中的 `:421` 是**旧基线 `ee3f1ff`** 的行号,新基线已无对应代码。推理 / 训练 dispatcher 的切换**不再依赖 `eval()`/`train()` 模式**,改由一个**进程级全局开关** `InferenceMode`(`megatron/core/inference/utils.py:20`)决定:`MoELayer.forward` 在入口处读 `InferenceMode.is_active()`,active → 推理 dispatcher、否则 → 训练 dispatcher(`megatron/core/transformer/moe/moe_layer.py:612`,另见 `:703`)。引擎进入推理时调用 `InferenceMode.set_active()`(`megatron/core/inference/engines/dynamic_engine.py:296`、`:857`、`megatron/core/inference/engines/static_engine.py:133`),退出时 `unset_active()`(`megatron/core/inference/engines/dynamic_engine.py:806`)。**根本原因**:`self.training` / `torch.is_grad_enabled()` / `inference_context is not None` 都无法可靠区分"引擎正在用模型做 rollout"与"训练相正在用同一模型重算 RL logprob"(二者都可能处于 `eval()`+`no_grad`)。改用单一进程级标志后,全代码库(attention、router、experts、mamba、`gpt_model` 等,见 `grep InferenceMode.is_active`)统一据此分流——这条**正是本节"显式、独立、受控的推理路径"取向的延续**:把"是否走推理路径"收敛成一个可审计的全局真值,而非散落各处的隐式 `self.training` 判断。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。§4 引用的行号已漂移——`transformer_impl` 三选枚举现位于 `megatron/core/transformer/transformer_config.py:1436`(`ee3f1ff` 为 `:1199`,`232c478d4` 为 `:1257`);`inference_optimized` 强制 `--moe-router-dtype=fp32` 的校验现位于 `megatron/core/transformer/transformer_config.py:2030`(`ee3f1ff` 为 `:1467`,`232c478d4` 为 `:1615`),且新增了 `inference_optimized` 对 expert-TP>1(`:2019`)、`fp8-recipe='mxfp8'` 需 `--fp8-param-gather`(`:2042`)的配套约束,`71092579` 下又多出 dropless-only(`:2023`)、不支持 GLU(`:2036`)两条。

设计取向:把推理路径做成**一个显式、独立命名的实现**,而不是偷偷改训练路径。好处是 —— 训推差异**集中在这一条路径里、可被审计**,你清楚知道每个数值差异来自哪。这是"工程上把不一致变可控"的关键设计。

---

## 5. 解法④:重算 logprob —— 用训练路径对齐

推理引擎能直接产出 logprob(`megatron/core/inference/sampling_params.py` 的 `return_log_probs` / `skip_prompt_log_probs` / `top_n_logprobs`),快,但**走的是推理 kernel**。

RL 训练里的标准做法:**不信任 rollout 的 logprob,在训练相用训练前向重新算一遍**。Megatron 的训练前向 + `fused_linear_cross_entropy`(`fusions/linear_cross_entropy/`)能高效产出每 token 的 logprob —— 与 policy 梯度**完全同一条 kernel 路径**。

于是:
- **policy 梯度用的 `π_train` logprob**:训练路径重算,自洽。
- **rollout 用的 `μ_rollout` logprob**:推理路径产出。
- 二者的差,就是纯粹的"推理 kernel/精度 vs 训练 kernel/精度"之差 —— **已经被解法①②③收敛到最小且定义清晰**。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。`megatron/core/inference/engines/dynamic_engine.py:1306`(#5167)修了一个 **logprob 切片越界**的 bug——会污染上层 IS 修正用的 rollout logprob。当请求 `num_tokens_to_generate == 0`(典型是 `echo + logprobs`,只要 prompt 的 logprob、不生成)时,prefill 步产出的 `request_log_probs` 布局是 `[<整段 prompt 的 logprob...>, <采样出的那个 token 的 logprob>]`;旧代码用 `request_log_probs[:keep]`(此时 `keep==0`)**从头部裁**,会把整段 prompt 的 logprob 全部丢掉。修复:改成**只裁尾部多余的** `request_log_probs[:-num_dropped]`(decode 步里全是生成 token,尾裁=头裁,等价)。同时把 `is_first_token` 事件与 TPOT 统计加了 `and tokens` 守卫(`:1336`/`:1349`),0 token 步不再污染指标。**意义**:rollout 侧 `μ` 的 per-token logprob 现在对 echo 类请求也对齐,IS 比值 `π/μ` 的分母不再缺失 prompt 段。

---

## 6. 残差:importance sampling 修正(RL 框架层)

即便做完①~④,`μ_rollout`(MXFP8 推理路径)与 `π_train`(BF16 训练路径)仍有**不可消除的小差异**。Megatron 不假装它为零,而是:

- 把这个 gap 收敛成**小、稳定、可量化**的量(靠①②③④)。
- 真正的数学修正交给上层 RL 框架:把 rollout 策略 `μ` 与训练策略 `π` 当作**确实不同**的两个分布,用 **importance sampling 比值** `π/μ`(及截断 IS / TIS 等变体)让 policy 梯度在 `μ ≠ π` 下仍**无偏**。

也就是说,Megatron 的职责是"**把不一致做小、做白盒**",让 RL 框架的 IS 修正能在一个良性的小 gap 上工作 —— 而不是在一个失控的大 gap 上硬修。

---

## 7. 部署形态:collocated vs 分离

`resharding/` 支持两种 RL 部署(README):

```
collocated(同卡同时持训练+推理模型):
  swap_model_weights(train_model, infer_model, "nccl")        ← 同 rank,short-circuit 成 tensor.copy_()

非 collocated(训练、推理在不相交的 rank 上):
  源 rank:  swap_model_weights(train_model, None, "nccl", src_rank_offset=0, dst_rank_offset=src_world)
  目标 rank:swap_model_weights(None, infer_model, "nccl", ...)
  空闲 rank:swap_model_weights(None, None, "nccl", ...)        ← 仍须参与集合通信
```

传输后端三选:`nccl`(GPU P2P,机内首选)/ `gloo`(CPU 中转,跨集群)/ `nvshmem`(流水线式 GPU↔GPU,高吞吐)。`_plan_cache` / `_service_cache` 缓存计划与传输器,反复 refit 不重建。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。#4762 对 `resharding/` 做了一轮**清理与重构**(本文 §2/§3/§7 的高层结论——LCM tiling、MXFP8 2D/1D scale 累积、`_plan_cache`/`_service_cache`、三传输后端、collocated 退化为本地 copy——经核对**仍然成立**)。落地的结构变化:
> - **统一 `CopyService` 抽象基类**(`megatron/core/resharding/copy_services/base.py`):nccl/gloo/nvshmem 三后端现共享 `submit_send`/`submit_recv`/`run`/`close()` 接口;`swap_model_weights(refit_method=...)` 既收字符串后端名、也收 `CopyService` 实例(`megatron/core/resharding/refit.py:351` 用 `isinstance(refit_method, CopyService)` 判定,替换原 duck-typing)。
> - **collocated 同 rank 短路**改由 **`task_id` 配对**实现(`megatron/core/resharding/copy_services/base.py:66` 的 `match_local_ops_by_task_id`):同一笔传输的 send/recv 共享全局唯一 `task_id`,本地直接 `copy_()`,并显式校验 send/recv 数量与 `task_id` 不重复。
> - **plan 缓存键加入 `src_rank_offset`/`dst_rank_offset`**(`megatron/core/resharding/refit.py:50-51`,`_PlanCacheKey` 在 `:37`):修复了"两个非 collocated 配置因 (rank, sizes, num_experts) 相同而错误共享同一 plan"的隐患。
> - **热路径提速**:`_get_config_tuple` 把 (TP,PP,EP,DP,expt_tp) 元组**记忆在 core 对象上**(每次 refit 查键 2~3 次);`_harmonize_buffer_dtypes` 的 `all_gather_object` 现**只做一次**、结果 `buffer_dtypes` 缓存在 plan 上,后续 refit 仅做本地 dtype 替换(替换后 `invalidate_refit_tensor_cache`)。
> - **缓存管理 API 扩充**:除 `clear_service_cache()`(现统一调 `service.close()` 释放 NVSHMEM GPU buffer)外,新增 `clear_plan_cache()` / `clear_all_caches()`(`megatron/core/resharding/refit.py:144`/`:152`,新基线下行号未变)。
> - `megatron/core/resharding/transforms.py` 的 `_ensure_sendable` 改用 `megatron.core.fp8_utils.is_mxfp8tensor`/`dequantize_fp8_tensor`,不再 try-import TransformerEngine 的 MXFP8Tensor。

---

## 8. 其他 RL 后训练适配

- **`post_training/modelopt/`**:NVIDIA ModelOpt 集成 —— 后训练量化 / 剪枝的 model spec 与 state-dict hook(GPT / Mamba / hybrid)。
- **推理引擎**(`inference/engines/`):`static_engine`(定长批)、`dynamic_engine`(连续/动态批处理,RL rollout 主力)、`mcore_engine`。配 KV cache、`dynamic_context`、调度器。
- **推理量化**(`inference/quantization/mxfp8_*`):把推理模型权重量化到 MXFP8。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**MRL(Megatron RL)rollout 服务端的 output parsers 接线**(#4768,`megatron/rl/inference/megatron.py:125`)。`MegatronLocal` 起动态文本生成服务时,`start_text_gen_server(..., parsers=...)` 之前**硬编码为空 `[]`**,导致命令行 `--rl-inference-parsers` 形同虚设;现改为透传 `args.rl_inference_parsers`。该参数定义在 `megatron/training/arguments.py:3640`(`nargs='*'`,默认 `[]`,如 `--rl-inference-parsers deepseek-r1-reasoning qwen3-coder-tool`),用于在 RL rollout 时**解析模型输出**(R1 reasoning 思维链、Qwen3-Coder 工具调用等结构化片段)。意义:RL 框架现能让 rollout 服务直接产出已解析的结构化 response,而非只回原始 token 流。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**rollout 的 policy-epoch / KV-cache-epoch 元数据下沉到 message 对象**(#4533,`megatron/rl/inference/megatron.py:83`、`megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py:735-737`)。chat completions 端点现把 `policy_epoch`、`kv_cache_epoch`、`num_evictions`(本次生成被驱逐的 KV 块数)写进每条 `message` 而非外层 `choice`;`MegatronLocal` 相应从 `choice.message.policy_epoch` 读取。**与训推一致性的关系**:`policy_epoch` 标记"这条样本是由第几代 policy 权重生成的"——配合 §2 的每迭代 Refit,上层 RL 框架据此判定 rollout 样本的**权重陈旧程度**,对跨 epoch 的 off-policy 样本施加(或拒绝)IS 修正;`kv_cache_epoch` 则用于 prefix cache 跨 Refit 失效的追踪。

---

## 9. 约束:refit 与 `inference_optimized` 各自要求什么、不做什么

前八节讲的是"怎么把 gap 收敛"。这一节列前提、代价与失效条件,每条带 locator。

**① refit 是一次集合通信,连不参与的 rank 都必须进来。**
非 collocated 部署下,既不持训练模型也不持推理模型的 rank 仍要调 `swap_model_weights(None, None, "nccl", ...)`——README 示例的注释即「Idle ranks (**must still participate in collectives**)」(`megatron/core/resharding/README.md:70`,§7 已给出三条调用形态)。漏掉一个 idle rank 就是挂死。

**② 两侧模型都必须带 `pg_collection`,且按并行度提供指定的组。**
「The source and destination models **must each have a `pg_collection` attribute**」,`tp` 必需、`dp` 必需(源侧缺失时可从 `parallel_state` 自动补)、`pp` 在 PP>1 时必需、`ep` 在 MoE 时必需、`expt_tp` 在 expert TP 时必需(`megatron/core/resharding/README.md:129-140`)。

**③ MXFP8 路径必须在初始化期、目标参数仍是 BF16 时先跑一次 `prepare_swap_model_weights`。**
README 把时序写死:「Call `prepare_swap_model_weights` once during initialization **while the target model's parameters are still in BF16**」(`megatron/core/resharding/README.md:42-46`)。错过这个窗口,§3.2 的持久 MXFP8 buffer 与 CUDA Graph 指针契约就建立不起来。

**④ 传输路径是 dtype 严格的,dtype 不一致会静默损坏数据。**
`_harmonize_buffer_dtypes` 的 docstring 说明了这个坑:MoE router 的 `expert_bias` 会在 trainer 首次前向时被 `_maintain_float32_expert_bias` 升到 fp32,而新建的推理模型仍是 `Float16Module` 包出来的 bf16;「The reshard send/recv path is **dtype-strict** — sending fp32 bytes into a bf16 receive buffer **corrupts the data** — so dst's buffer must match src's dtype before the transfer」(`megatron/core/resharding/refit.py:376-389`)。这条对齐是 refit 自己做的,但它意味着**持久 buffer 的 dtype 是 refit 契约的一部分**,不能在两侧各自改。

**⑤ 缓存必须在销毁进程组之前显式清理。**
「Call `clear_all_caches()` **before destroying distributed process groups** to avoid stale references. This also finalizes NVSHMEM resources.」(`megatron/core/resharding/README.md:126-127`;API 位置见 §7 的 `[!update]`)。

**⑥ `inference_optimized` 的 MoE 路径有六条硬拒绝,全部是构造期 `ValueError`。**(`megatron/core/transformer/transformer_config.py:2018-2047`)
- expert TP > 1 → 「Inference-optimized MoE layers does not support expert tensor parallelism.」(`:2019-2021`)
- 设了 `moe_expert_capacity_factor` → 「only support **dropless** MoE」(`:2023-2024`)
- `moe_router_padding_for_quantization` → 「do not support padded routing map for quantization.」(`:2025-2029`)
- router dtype ≠ fp32 → 见 §1.2 ③(`:2030-2034`)
- **`gated_linear_unit`(SwiGLU/GeGLU)→「does not yet support gated linear units」**(`:2036-2039`)——这条限制面很大,主流 MoE 模型基本都用 SwiGLU。
- `fp8 == "mxfp8"` 但未开 `fp8_param` → 要求 `--fp8-param-gather`(`:2042-2047`)。

**⑦ 边界的精确说法:RL 环不在 `megatron/core`,但同仓 `megatron/rl/` 里有。**
§0 的边界声明针对的是 `megatron/core`,基线下仍然成立。需要补一句的是:同一个仓库的 `megatron/rl/` 已经带了 GRPO 侧的实现(如 `calculate_grpo_advantages`,`megatron/rl/rl_utils.py:853`),但它自陈**尚不可外部使用**:「it is **not yet usable by external users** because not all required code has been released. The available code and examples **may change** as development progresses」(`megatron/rl/README.md:4`),并且「It is **not** intended as an enterprise framework」,企业级能力仍指向 NeMo-RL(`:17`)。

**⑧ Megatron-RL 反向依赖 core 的推理改动。** README 明说「**Significant modifications have been made to the Megatron Core inference code**」(`megatron/rl/README.md:10`)——即 §8 那些 `megatron/rl/*` 的改动与 [[31_megatron_inference_engine_analysis]] 的引擎是耦合演进的,不能只升一侧。

---

## 10. 发展趋势

每条都锚在基线源码或提交历史上;**方向解读是本页的推断,不是作者自陈**。

**① 一致性 gap 正在从"交给上层修正"变成"在仓内被量化上报"。**
`megatron/rl/rl_utils.py:452-483` 已经把 π_old/μ_inference 的比值与绝对概率差算成一组 group 统计量(`min/max/mean_piold_to_inf_prob`、`min/max/mean_inf_train_prob_abs_diff`,`:470-478`)。→ 推断:§6 "残差交给上层 RL 框架"的分工没变,但**判定 gap 是否良性的量表**已经进了 Megatron 自己。

**② refit 正在往"一份权重发给多个推理池"走,但这一步在当前 `dev` 基线上不存在。**
`ae2efd53a`(#5187,2026-06-22,"Disag MR2: **Refit into multiple destination pools** and tied-embedding + UVM fixes")给 `swap_model_weights` 加了 `num_dst_pools` / `dst_pool_index`,docstring 写明用途:「refit into `num_dst_pools` **disjoint destination pools** (e.g. **disaggregated prefill/decode instances** on separate rank windows), one collective pass per pool」,并给 `_PlanCacheKey` 加了 `pool_index` 以免 source-only rank 缓存命中后跳过集合通信而**死锁**。
**但基线 `71092579` 下 `git grep num_dst_pools` 为 0 命中**:该提交进了 `origin/main`(合并 `c2a9a6016` 的第二父 `fc4597c0c` 侧有 3 处命中),而 2026-06-22 的 `main → main2dev` 合并把 `megatron/core/resharding/refit.py` 解成了 dev 侧(第一父 `3346fa8a2`,0 处命中)。→ 源码没有说明这是有意回退还是合并丢失;**本页只陈述"基线下不存在",不判断意图**。读 `dev` 的人不要按 #5187 的接口写代码。

**③ 推理侧计划把 refit 包成一等公民 API。**
`megatron/core/inference/README.md:67` 的 roadmap 明列「**Weight update APIs.** `suspend_for_refit()`, `update_weights_from_collective()`, `resume_after_refit()` **wrapping the existing resharding/refit primitives** for RL workflows where weights swap between rollout steps.」——即 §2 的 `swap_model_weights` 与 §7 的 suspend/resume 会被收进引擎门面(详见 [[31_megatron_inference_engine_analysis]] §11)。

**④ rollout 输出解析还没接完。** §8 的 `--rl-inference-parsers` 已接通,但消费侧仍挂着「TODO: Handle **tool calls and reasoning** in `LLMChatMessage`」(`megatron/rl/inference/megatron.py:76`)。→ 推断:结构化 rollout(思维链、工具调用)目前只解析到端点层,还没进 RL 侧的消息对象。

**⑤ Megatron-RL 自陈仍在开发中,并给出了公开 roadmap。** 「Megatron-RL is **actively under development** … For a current roadmap of planned Megatron-RL features please see #1776」(`megatron/rl/README.md:4`)。→ 本页 §8 的两条 `[!update]` 都属于这条线上的增量,后续还会变。

---

## 11. 小结

- **Megatron Core 的 RL 角色**:提供积木(训练模型 + 推理引擎 + `inference_optimized` 路径 + MXFP8 量化 + **Refit**),不含完整 RL 环(在 NeMo-RL 等上层)。
- **训推一致性问题**:rollout 引擎与训练引擎对同一 token 算出的 logprob 不等 → PPO/GRPO 的 importance ratio 偏离 1 → 梯度有偏。来源:权重陈旧、布局不同、精度不同、kernel 不同、批处理数值路径。
- **Megatron 的解法是"逐项收敛、把 gap 做小做白盒"**:
  - ① **Refit 每迭代**消除权重陈旧;
  - ② Refit 的 **LCM tiling + MXFP8Transform** 保证布局/格式转换零误差、忠实量化;
  - ③ **`inference_optimized`** 把推理路径做成显式独立、可审计的实现;
  - ④ 训练相**重算 logprob**,让 policy 梯度走自洽的训练 kernel。
- **残差**(MXFP8 vs BF16 等不可消除的小差)交给上层 RL 框架的 **importance sampling 修正**(`π/μ` 比值 / 截断 IS)—— Megatron 负责让这个 gap 小到 IS 修正能良性工作。
- **部署**:collocated(同卡,refit 退化为 `copy_()`)/ 非 collocated(异卡,3 种传输后端)。

---

*生成依据:`Megatron-LM` `dev` 分支 `71092579`(2026-08-27;由 `ee3f1ff` 重定基线而来)。源码行号以该 commit 为准。完整 RL 训练环(GRPO/PPO loss、advantage、KL)位于上层 RL 框架(如 NeMo-RL),不在 `megatron/core`。配套文档:`27_megatron_tp_fsdp_resharding_supplements_analysis.md`、`17_megatron_parallelism_orchestration_analysis.md`、`14_megatron_ep_analysis.md`。*

## Related Pages

- [[27_megatron_tp_fsdp_resharding_supplements_analysis]] · [[31_megatron_inference_engine_analysis]] · [[14_megatron_ep_analysis]] · [[17_megatron_parallelism_orchestration_analysis]]
- [[33_megatron_vllm_weight_sync_analysis]] — verl 在 Megatron+vLLM 场景下的权重同步实现(本文的下游消费者之一)
- [[01_posttraining_infra_mechanism_analysis]] — 第 6 节「Weight publish 协议」,三平面机制视角(框架无关)
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
