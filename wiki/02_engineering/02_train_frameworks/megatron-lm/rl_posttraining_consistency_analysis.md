# Megatron-LM RL 后训练适配与训推一致性深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心:`megatron/core/resharding/`(refit)、`inference/`(推理引擎)、`inference/quantization/`(MXFP8)、`post_training/modelopt/`、`transformer_config.py`(`transformer_impl='inference_optimized'`)
> 配套阅读:`tp_fsdp_resharding_supplements_analysis.md` §3(refit 基础)、`parallelism_orchestration_analysis.md`、`ep_analysis.md`
> 定位:系统性专题。前面文档讲预训练;本文讲 **RL 后训练**(RLHF / GRPO / PPO)对 Megatron 提出的特殊需求,以及核心难题 **训推一致性(train-inference consistency)**。

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

`resharding/`(详见 `tp_fsdp_resharding_supplements_analysis.md` §3)。每个 RL 迭代结束,`swap_model_weights(train_model, infer_model)` 把训练模型的**最新权重**搬进推理模型。

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
- `inference_moe_token_dispatcher_type`:推理专用 MoE dispatcher(`nccl` / `nvls`)—— `moe_layer.py` 的 `train()` 重写(`:421`)在 eval 模式自动切到推理 dispatcher、train 模式切回(见 `ep_analysis.md`)。
- 强制 `--moe-router-dtype=fp32`(`:1467`)—— 与训练侧推荐一致,**路由精度对齐**。

设计取向:把推理路径做成**一个显式、独立命名的实现**,而不是偷偷改训练路径。好处是 —— 训推差异**集中在这一条路径里、可被审计**,你清楚知道每个数值差异来自哪。这是"工程上把不一致变可控"的关键设计。

---

## 5. 解法④:重算 logprob —— 用训练路径对齐

推理引擎能直接产出 logprob(`sampling_params.py` 的 `return_log_probs` / `skip_prompt_log_probs` / `top_n_logprobs`),快,但**走的是推理 kernel**。

RL 训练里的标准做法:**不信任 rollout 的 logprob,在训练相用训练前向重新算一遍**。Megatron 的训练前向 + `fused_linear_cross_entropy`(`fusions/linear_cross_entropy/`)能高效产出每 token 的 logprob —— 与 policy 梯度**完全同一条 kernel 路径**。

于是:
- **policy 梯度用的 `π_train` logprob**:训练路径重算,自洽。
- **rollout 用的 `μ_rollout` logprob**:推理路径产出。
- 二者的差,就是纯粹的"推理 kernel/精度 vs 训练 kernel/精度"之差 —— **已经被解法①②③收敛到最小且定义清晰**。

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

---

## 8. 其他 RL 后训练适配

- **`post_training/modelopt/`**:NVIDIA ModelOpt 集成 —— 后训练量化 / 剪枝的 model spec 与 state-dict hook(GPT / Mamba / hybrid)。
- **推理引擎**(`inference/engines/`):`static_engine`(定长批)、`dynamic_engine`(连续/动态批处理,RL rollout 主力)、`mcore_engine`。配 KV cache、`dynamic_context`、调度器。
- **推理量化**(`inference/quantization/mxfp8_*`):把推理模型权重量化到 MXFP8。

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

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。完整 RL 训练环(GRPO/PPO loss、advantage、KL)位于上层 RL 框架(如 NeMo-RL),不在 `megatron/core`。配套文档:`tp_fsdp_resharding_supplements_analysis.md`、`parallelism_orchestration_analysis.md`、`ep_analysis.md`。*

## Related Pages

- [[tp_fsdp_resharding_supplements_analysis]] · [[inference_engine_analysis]] · [[ep_analysis]] · [[parallelism_orchestration_analysis]]
- [[Megatron_vLLM_Weight_Sync_Analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
