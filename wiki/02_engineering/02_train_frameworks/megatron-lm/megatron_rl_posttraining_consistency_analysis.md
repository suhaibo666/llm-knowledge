# Megatron-LM RL 后训练适配与训推一致性深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心:`megatron/core/resharding/`(refit)、`inference/`(推理引擎)、`inference/quantization/`(MXFP8)、`post_training/modelopt/`、`transformer_config.py`(`transformer_impl='inference_optimized'`)
> 配套阅读:`megatron_tp_fsdp_resharding_supplements_analysis.md` §3(refit 基础)、`megatron_parallelism_orchestration_analysis.md`、`megatron_ep_analysis.md`
> 定位:系统性专题。前面文档讲预训练;本文讲 **RL 后训练**(RLHF / GRPO / PPO)对 Megatron 提出的特殊需求,以及核心难题 **训推一致性(train-inference consistency)**。
> **三方分工**:本文是 Megatron 训练侧的 refit / 训推一致性实现(逐项收敛 gap);三平面机制视角(weight publish 协议、跨框架不变量)见 [[01_posttraining_infra_mechanism_analysis]] 第 6 节;verl 在 Megatron+vLLM 场景下的具体同步调用链见 [[megatron_vllm_weight_sync_analysis]]。

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

## 1. RL 后训练的结构与难点

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

---

## 2. 解法①:Refit 消除"权重陈旧"

`resharding/`(详见 `megatron_tp_fsdp_resharding_supplements_analysis.md` §3)。每个 RL 迭代结束,`swap_model_weights(train_model, infer_model)` 把训练模型的**最新权重**搬进推理模型。

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

### 3.1 并行布局重映射(`planner.py`)

训练和推理模型的并行度通常不同(训练求吞吐、推理求延迟)。refit 的 plan 构建:
- 每个 rank 抽取参数元数据(shape、TP/EP/PP 切分、所在组)。
- rank 0 集中构建 `ReshardPlan`:对每个目标参数,**按名字找到对应的源参数切片**,用维度特定的 planner 计算精确映射 —— 标准 TP 用 **LCM tiling**(最小公倍数分块,保证源/目标两种切分都能整除对齐),Mamba `in_proj` 这类分区参数用 block-interleaved。
- 产出 `TransferOp`(谁把哪段发给谁),scatter 给各 rank。

结果:推理模型每个分片拿到的,**就是训练权重对应位置的精确切片**,布局转换零误差。

### 3.2 MXFP8 格式转换(`transforms.py` `MXFP8ReshardTransform`)

推理模型常用 MXFP8(`fp8_recipe='mxfp8'` + `inference_optimized`)。refit 把 BF16 训练权重转成 MXFP8 时,难点在 **scale 布局**:

- **2D scale**:每行 scale 对应一行数据,切片独立 → 收到 BF16 切片立即逐片量化。
- **1D scale**(FlashInfer swizzled 布局):scale 跨整个权重张量交织编码,**部分更新会破坏布局** → 必须把所有 BF16 切片**累积进一个完整 buffer,等齐了一次性量化**。

而且转换直接 `copy_()` 写进**持久 MXFP8Tensor buffer**(其设备指针被 CUDA Graph 捕获),保证多次 refit 后 CUDA Graph 仍有效。

意义:推理模型拿到的是训练权重**忠实的 MXFP8 量化版本**,而不是"重新量化的近似" —— 把"精度不同"这一项差异收敛成单纯的"BF16 vs MXFP8 数值精度差",干净、可量化。

---

## 4. 解法③:`inference_optimized` —— 显式、独立、受控的推理路径

`transformer_impl` 有三选(`transformer_config.py:1199`):`local` / `transformer_engine` / **`inference_optimized`**。

`inference_optimized` 是 RL rollout 用的专用路径:
- `use_inference_optimized_layers`:推理优化的线性层(`tensor_parallel/inference_layers.py`,如推理专用的 all-gather)。
- `inference_grouped_gemm_backend`:MoE 推理的 grouped GEMM 后端(`flashinfer` / `torch` / `vllm`)。
- `inference_moe_token_dispatcher_type`:推理专用 MoE dispatcher(`nccl` / `nvls`)—— `moe_layer.py` 的 `train()` 重写(`:421`)在 eval 模式自动切到推理 dispatcher、train 模式切回(见 `megatron_ep_analysis.md`)。
- 强制 `--moe-router-dtype=fp32`(`:1467`)—— 与训练侧推荐一致,**路由精度对齐**。

> [!deprecated] 2026-06-16:`moe_layer.py` 的 `train()` 重写已被**移除**(#4617,`moe_layer.py`)。推理 / 训练 dispatcher 的切换**不再依赖 `eval()`/`train()` 模式**,改由一个**进程级全局开关** `InferenceMode`(`megatron/core/inference/utils.py:20`)决定:`MoELayer.forward` 在入口处读 `InferenceMode.is_active()`,active → 推理 dispatcher、否则 → 训练 dispatcher(`megatron/core/transformer/moe/moe_layer.py:605`)。引擎进入推理时调用 `InferenceMode.set_active()`(`dynamic_engine.py:292`、`static_engine.py:133`),退出时 `unset_active()`(`dynamic_engine.py:787`)。**根本原因**:`self.training` / `torch.is_grad_enabled()` / `inference_context is not None` 都无法可靠区分"引擎正在用模型做 rollout"与"训练相正在用同一模型重算 RL logprob"(二者都可能处于 `eval()`+`no_grad`)。改用单一进程级标志后,全代码库(attention、router、experts、mamba、`gpt_model` 等,见 `grep InferenceMode.is_active`)统一据此分流——这条**正是本节"显式、独立、受控的推理路径"取向的延续**:把"是否走推理路径"收敛成一个可审计的全局真值,而非散落各处的隐式 `self.training` 判断。

> [!update] 2026-06-16 · dev@232c478d4:§4 引用的行号已漂移——`transformer_impl` 三选枚举现位于 `transformer_config.py:1257`(原 `:1199`);`inference_optimized` 强制 `--moe-router-dtype=fp32` 的校验现位于 `transformer_config.py:1615`(原 `:1467`),且新增了 `inference_optimized` 对 EP>1、`fp8-recipe='mxfp8'` 的配套约束(`:1603`、`:1631`)。

设计取向:把推理路径做成**一个显式、独立命名的实现**,而不是偷偷改训练路径。好处是 —— 训推差异**集中在这一条路径里、可被审计**,你清楚知道每个数值差异来自哪。这是"工程上把不一致变可控"的关键设计。

---

## 5. 解法④:重算 logprob —— 用训练路径对齐

推理引擎能直接产出 logprob(`sampling_params.py` 的 `return_log_probs` / `skip_prompt_log_probs` / `top_n_logprobs`),快,但**走的是推理 kernel**。

RL 训练里的标准做法:**不信任 rollout 的 logprob,在训练相用训练前向重新算一遍**。Megatron 的训练前向 + `fused_linear_cross_entropy`(`fusions/linear_cross_entropy/`)能高效产出每 token 的 logprob —— 与 policy 梯度**完全同一条 kernel 路径**。

于是:
- **policy 梯度用的 `π_train` logprob**:训练路径重算,自洽。
- **rollout 用的 `μ_rollout` logprob**:推理路径产出。
- 二者的差,就是纯粹的"推理 kernel/精度 vs 训练 kernel/精度"之差 —— **已经被解法①②③收敛到最小且定义清晰**。

> [!update] 2026-06-16 · dev@232c478d4:`dynamic_engine.py:1218`(#5167)修了一个 **logprob 切片越界**的 bug——会污染上层 IS 修正用的 rollout logprob。当请求 `num_tokens_to_generate == 0`(典型是 `echo + logprobs`,只要 prompt 的 logprob、不生成)时,prefill 步产出的 `request_log_probs` 布局是 `[<整段 prompt 的 logprob...>, <采样出的那个 token 的 logprob>]`;旧代码用 `request_log_probs[:keep]`(此时 `keep==0`)**从头部裁**,会把整段 prompt 的 logprob 全部丢掉。修复:改成**只裁尾部多余的** `request_log_probs[:-num_dropped]`(decode 步里全是生成 token,尾裁=头裁,等价)。同时把 `is_first_token` 事件与 TPOT 统计加了 `and tokens` 守卫(`:1260`/`:1276`),0 token 步不再污染指标。**意义**:rollout 侧 `μ` 的 per-token logprob 现在对 echo 类请求也对齐,IS 比值 `π/μ` 的分母不再缺失 prompt 段。

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

> [!update] 2026-06-16 · dev@232c478d4:#4762 对 `resharding/` 做了一轮**清理与重构**(本文 §2/§3/§7 的高层结论——LCM tiling、MXFP8 2D/1D scale 累积、`_plan_cache`/`_service_cache`、三传输后端、collocated 退化为本地 copy——经核对**仍然成立**)。落地的结构变化:
> - **统一 `CopyService` 抽象基类**(`copy_services/base.py`):nccl/gloo/nvshmem 三后端现共享 `submit_send`/`submit_recv`/`run`/`close()` 接口;`swap_model_weights(refit_method=...)` 既收字符串后端名、也收 `CopyService` 实例(`refit.py:349` 用 `isinstance(refit_method, CopyService)` 判定,替换原 duck-typing)。
> - **collocated 同 rank 短路**改由 **`task_id` 配对**实现(`base.py` 的 `match_local_ops_by_task_id`):同一笔传输的 send/recv 共享全局唯一 `task_id`,本地直接 `copy_()`,并显式校验 send/recv 数量与 `task_id` 不重复。
> - **plan 缓存键加入 `src_rank_offset`/`dst_rank_offset`**(`refit.py:44`):修复了"两个非 collocated 配置因 (rank, sizes, num_experts) 相同而错误共享同一 plan"的隐患。
> - **热路径提速**:`_get_config_tuple` 把 (TP,PP,EP,DP,expt_tp) 元组**记忆在 core 对象上**(每次 refit 查键 2~3 次);`_harmonize_buffer_dtypes` 的 `all_gather_object` 现**只做一次**、结果 `buffer_dtypes` 缓存在 plan 上,后续 refit 仅做本地 dtype 替换(替换后 `invalidate_refit_tensor_cache`)。
> - **缓存管理 API 扩充**:除 `clear_service_cache()`(现统一调 `service.close()` 释放 NVSHMEM GPU buffer)外,新增 `clear_plan_cache()` / `clear_all_caches()`(`refit.py:144`/`:152`)。
> - `transforms.py` 的 `_ensure_sendable` 改用 `megatron.core.fp8_utils.is_mxfp8tensor`/`dequantize_fp8_tensor`,不再 try-import TransformerEngine 的 MXFP8Tensor。

---

## 8. 其他 RL 后训练适配

- **`post_training/modelopt/`**:NVIDIA ModelOpt 集成 —— 后训练量化 / 剪枝的 model spec 与 state-dict hook(GPT / Mamba / hybrid)。
- **推理引擎**(`inference/engines/`):`static_engine`(定长批)、`dynamic_engine`(连续/动态批处理,RL rollout 主力)、`mcore_engine`。配 KV cache、`dynamic_context`、调度器。
- **推理量化**(`inference/quantization/mxfp8_*`):把推理模型权重量化到 MXFP8。

> [!update] 2026-06-16 · dev@232c478d4:**MRL(Megatron RL)rollout 服务端的 output parsers 接线**(#4768,`megatron/rl/inference/megatron.py:120`)。`MegatronLocal` 起动态文本生成服务时,`start_text_gen_server(..., parsers=...)` 之前**硬编码为空 `[]`**,导致命令行 `--rl-inference-parsers` 形同虚设;现改为透传 `args.rl_inference_parsers`。该参数定义在 `arguments.py:3472`(`nargs='*'`,默认 `[]`,如 `--rl-inference-parsers deepseek-r1-reasoning qwen3-coder-tool`),用于在 RL rollout 时**解析模型输出**(R1 reasoning 思维链、Qwen3-Coder 工具调用等结构化片段)。意义:RL 框架现能让 rollout 服务直接产出已解析的结构化 response,而非只回原始 token 流。

> [!update] 2026-06-16 · dev@232c478d4:**rollout 的 policy-epoch / KV-cache-epoch 元数据下沉到 message 对象**(#4533,`megatron/rl/inference/megatron.py:80`、`.../endpoints/chat_completions.py:732`)。chat completions 端点现把 `policy_epoch`、`kv_cache_epoch`、`num_evictions`(本次生成被驱逐的 KV 块数)写进每条 `message` 而非外层 `choice`;`MegatronLocal` 相应从 `choice.message.policy_epoch` 读取。**与训推一致性的关系**:`policy_epoch` 标记"这条样本是由第几代 policy 权重生成的"——配合 §2 的每迭代 Refit,上层 RL 框架据此判定 rollout 样本的**权重陈旧程度**,对跨 epoch 的 off-policy 样本施加(或拒绝)IS 修正;`kv_cache_epoch` 则用于 prefix cache 跨 Refit 失效的追踪。

---

## 9. 小结

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

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。完整 RL 训练环(GRPO/PPO loss、advantage、KL)位于上层 RL 框架(如 NeMo-RL),不在 `megatron/core`。配套文档:`megatron_tp_fsdp_resharding_supplements_analysis.md`、`megatron_parallelism_orchestration_analysis.md`、`megatron_ep_analysis.md`。*

## Related Pages

- [[megatron_tp_fsdp_resharding_supplements_analysis]] · [[megatron_inference_engine_analysis]] · [[megatron_ep_analysis]] · [[megatron_parallelism_orchestration_analysis]]
- [[megatron_vllm_weight_sync_analysis]] — verl 在 Megatron+vLLM 场景下的权重同步实现(本文的下游消费者之一)
- [[01_posttraining_infra_mechanism_analysis]] — 第 6 节「Weight publish 协议」,三平面机制视角(框架无关)
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
