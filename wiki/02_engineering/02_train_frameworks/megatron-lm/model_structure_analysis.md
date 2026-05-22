# Megatron-LM 模型结构深度解析(Model Structure)

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`transformer/spec_utils.py`、`transformer_layer.py`、`transformer_block.py`、`attention.py`、`multi_latent_attention.py`、`mlp.py`、`moe/router.py`、`multi_token_prediction.py`、`ssm/`、`models/`
> 配套阅读:`ep_analysis.md`(MoE dispatcher)、`cp_analysis.md`、`tp_analysis.md`、`recompute_analysis.md`
> 定位:之前 17 份文档都讲"**怎么把模型大规模训起来/服务起来**"(并行、显存、稳定性、推理、数据);本文讲"**模型本身长什么样**" —— 一个 transformer 模型由什么构成。

---

## 0. 总览:模型不是写死的,是"拼"出来的

Megatron 的模型**不是一个固定的 `nn.Module` 类**,而是由一套 **Spec(规格)系统**在运行时**组装**出来的。同一个 `TransformerLayer` 类,喂不同的 spec,就能变成:
- 用 TransformerEngine 算子 / 用 Megatron 自带算子 / 用 inference-optimized 算子;
- dense FFN 层 / MoE 层 / Mamba 层;
- 带 / 不带 hyper connections;
- MHA / GQA / MLA 注意力。

所以理解 Megatron 模型结构,要先理解 **Spec 系统**(§1),再看它拼出的**层结构**(§2)和各**组件**(§3–7)。

```
模型 = GPTModel(embedding + 一摞 TransformerLayer + output layer)
            │
TransformerLayer = 按 TransformerLayerSubmodules 这张"插槽表"组装
            │
每个插槽(self_attention / mlp / layernorm …) = 一个 ModuleSpec,运行时 build 成真实模块
```

---

## 1. Spec 系统:`ModuleSpec` 与模型组装

### 1.1 `ModuleSpec`(`spec_utils.py:13`)

```python
@dataclass
class ModuleSpec:
    module: Union[Tuple, type]   # 模块类本身,或 (路径, 类名) 二元组(动态 import)
    params: dict = {}            # 初始化该模块要传的 kwargs
    submodules: object = None    # 嵌套的子模块 spec(递归)
    metainfo: dict = {}
```

`build_module(spec, *args, **kwargs)`(`:74`)按 spec **递归实例化**:先 build 出 `submodules` 里的每个子 spec,再用它们 + `params` 构造 `module`。`spec()` 直接 `__call__` 等价于 `build_module`。

### 1.2 `TransformerLayerSubmodules` —— 一层的"插槽表"(`transformer_layer.py:217`)

一个 transformer 层有哪些可填的槽:

```python
@dataclass
class TransformerLayerSubmodules:
    input_layernorm:    = IdentityOp        # 注意力前的归一化
    self_attention:     = IdentityOp        # 自注意力
    self_attn_bda:      = IdentityFuncOp    # bias-dropout-add(注意力后)
    pre_cross_attn_layernorm / cross_attention / cross_attn_bda   # 交叉注意力(encoder-decoder)
    pre_mlp_layernorm:  = IdentityOp        # MLP 前的归一化
    mlp:                = IdentityOp        # FFN(dense MLP 或 MoELayer)
    mlp_bda:            = IdentityFuncOp    # bias-dropout-add(MLP 后)
    *_hyper_connection: = IdentityOp        # hyper connections 槽
```

默认全是 `IdentityOp`(空操作)—— **spec 不填的槽就是恒等映射**。`gpt_layer_specs.py` 等提供工厂函数,产出填好的 spec(TE 版 / local 版 / dense / MoE)。

### 1.3 为什么这样设计

- **可换实现**:同一层,spec 指向 TE 算子或 Megatron 自带算子或 `inference_optimized` 算子 —— 不改模型代码(呼应 `rl_posttraining_consistency_analysis.md` §4 的训推路径切换)。
- **可裁剪**:不需要的组件用 `IdentityOp` 占位。
- **可混搭**:dense 层与 MoE 层用不同 spec,按 `moe_layer_freq` 间隔排布;hybrid 模型把 Mamba 层 / attention 层 / MLP 层混排。

---

## 2. Transformer 层结构

### 2.1 `TransformerLayer`(`transformer_layer.py:279`)

`forward`(`:710`)= **注意力子层 + MLP 子层**,标准 pre-norm 残差结构:

```python
def forward(self, *args, **kwargs):
    hidden_states, context = self._forward_attention(*args, **kwargs)
    output = self._forward_mlp(hidden_states, ...)
    return output, context
```

展开(pre-norm + bias-dropout-add):

```
        x ───────────────────────┐ residual
        │                        │
   input_layernorm                │
        │                        │
   self_attention                 │
        │                        │
   self_attn_bda  ◄───────────────┘   bias + dropout + 加回 residual
        │
        ├───────────────────────┐ residual
        │                        │
   pre_mlp_layernorm              │
        │                        │
   mlp(dense FFN 或 MoELayer)     │
        │                        │
   mlp_bda  ◄──────────────────────┘
        │
       output
```

`bda` = **bias-dropout-add**,融合算子(见 `precision_cudagraph_fusion_analysis.md` §3 `fused_bias_dropout`)。

### 2.2 子类

- **`MoETransformerLayer`**(`:1983`):`mlp` 槽是 `MoELayer`(见 `ep_analysis.md`);处理 MoE 特有的 recompute、padding_mask 等。
- **`HyperConnectionTransformerLayer`**(`:1488`):hyper connections(mHC)—— 用**多条残差流**代替单条残差,层间连接更丰富;`*_hyper_connection` 槽生效,PP 通信传 n-stream 张量(见 `pp_schedulers_analysis.md` 里 `enable_hyper_connections` 的形状处理)。

### 2.3 `TransformerBlock`(`transformer_block.py`)

一摞 `TransformerLayer` 的容器。负责:层的构建与编号、`_checkpointed_forward`(激活重计算,见 `recompute_analysis.md`)、PP 下本 stage 只放 `num_layers_per_pipeline_rank` 层、final layernorm。

---

## 3. 注意力家族

`attention.py`:`Attention`(ABC)→ `SelfAttention` / `CrossAttention`。

### 3.1 MHA / GQA / MQA

`SelfAttentionSubmodules`:`linear_qkv`(QKV 投影)、`core_attention`(缩放点积注意力,TE fused 或 local)、`linear_proj`(输出投影)、`q_layernorm` / `k_layernorm`(QK-Norm,稳定性)。

按 **`num_query_groups`** 区分三档:
- **MHA**:`num_query_groups = num_attention_heads` —— 每个 Q 头有自己的 K/V 头。
- **GQA(分组查询注意力)**:`num_query_groups < num_attention_heads` —— 多个 Q 头**共享**一组 K/V 头。KV 头变少 → **KV cache 变小、推理 decode 带宽压力降**(`inference_engine_analysis.md` §2 的瓶颈)。
- **MQA**:`num_query_groups = 1` —— 所有 Q 头共享 1 组 K/V,GQA 的极端。

TP 下 QKV 投影是 ColumnParallel(按头切)、输出投影 RowParallel(见 `tp_analysis.md` §3.2)。

### 3.2 MLA —— 多头潜在注意力(`multi_latent_attention.py`)

`MultiLatentAttention` → `MLASelfAttention` / `FusedMLASelfAttention`。DeepSeek-V2/V3 的核心。

**动机**:GQA 减 KV 头,但 KV cache 仍 `∝ 头数 × head_dim`。MLA 更激进 —— 把 K/V **压成一个低秩潜在向量**:

```
标准 MHA/GQA:KV cache 存每个头的完整 K、V          → cache ∝ n_kv_heads · head_dim
MLA:         K/V 投影到一个低维 latent c_KV(+ 一个解耦的 RoPE key)
             KV cache 只存 c_KV → 远小于完整 K/V    → cache 砍到 ~1/10(DeepSeek-V3)
             compute 时再把 c_KV 上投影回每头的 K、V
```

- `MLASelfAttentionSubmodules` 含 Q / KV 的压缩(`q_lora` / `kv_lora` 风格低秩投影)与上投影。
- **`cache_mla_latents`**:推理时 KV cache 存 latent 而非完整 K/V;decode 时做 **"吸收(absorption)"** —— 把上投影矩阵数学上吸收进 Q 投影 / 输出投影,**永不物化完整 K/V**,`FusedMLASelfAttention`(`:1212`)即此路径。
- 与 CP 的配合见 `cp_analysis.md`。

### 3.3 实验性注意力变体

`transformer/experimental_attention_variant/`:DeepSeek 的 DSA、v4 hybrid attention、CSA、absorbed MLA —— 都是 MLA / 注意力的研究性变体,以独立模块 spec 形式接入,不展开。

---

## 4. MoE Router 算法(补 `ep_analysis.md` 的空白)

`ep_analysis.md` 讲了 token 怎么**分发**(dispatcher),但没讲 token 怎么**被路由**。补在这里。`moe/router.py`:`Router`(ABC)→ `TopKRouter`(`:138`)。

`TopKRouter.forward` 产出 `(probs, routing_map)`:

1. **打分**:轻量线性层把 hidden 投成 `[s, E]` logits(建议 fp32,见 `ep_analysis.md` §2.3)。
2. **score function**:`softmax` 或 `sigmoid` 把 logits 变概率。
3. **选 top-k**,可叠加:
   - **group-limited routing**(`moe_router_num_groups` / `group_topk`):先选 top-k 个**专家组**,再在组内选专家 —— 限制一个 token 跨越的节点数(DeepSeek-V3,利于 EP 跨节点通信)。
   - **`topk_routing_with_score_function`** —— 统一的 topk + score 实现。
4. **负载均衡**(决定路由怎么"被纠偏",对应 `ep_analysis.md` §4):
   - **`aux_loss`** —— 辅助损失惩罚不均衡(微批 / 序列 / 全局级)。
   - **`sinkhorn`** —— 最优传输式均衡(与 aux_loss 互斥)。
   - **aux-loss-free**(`moe_router_enable_expert_bias`)—— 给每个专家一个**动态偏置** `expert_bias`(`_maintain_float32_expert_bias` 保持 fp32),过载专家偏置降、冷门专家偏置升,**不引入辅助损失**(DeepSeek-V3 用法)。

产出的 `routing_map`(`[s,E]` 多热掩码)和 `probs` 就交给 `ep_analysis.md` 的 dispatcher。

---

## 5. MLP / 激活 / 归一化 / 位置编码

- **MLP**(`mlp.py`):dense FFN —— `linear_fc1`(ColumnParallel,`h→H`)→ 激活 → `linear_fc2`(RowParallel,`H→h`)。激活常用 **SwiGLU**(门控,`fc1` 输出对半切,一半过 SiLU 门控另一半;有 `fused_bias_swiglu` 融合 kernel)。
- **归一化**:`LayerNorm` / `RMSNorm`(现代模型多用 RMSNorm,省一个均值);有 `fused_layer_norm`。可选 **QK-Norm**(`q_layernorm`/`k_layernorm`,稳定 attention logit,呼应 `training_stability_observability_analysis.md` §1.5)。
- **位置编码**:**RoPE**(旋转位置编码)是主流;长上下文用 **YaRN**(`fused_mla_yarn_rope_apply`)。代码在 `models/common/embeddings/`。
- 这些组件都以 `ModuleSpec` 填进 §1.2 的层插槽。

---

## 6. MTP —— Multi-Token Prediction(`multi_token_prediction.py`)

标准 LM 每步只预测**下一个** token。**MTP**(DeepSeek-V3)让模型额外预测**再后面若干个** token —— 增加训练信号密度、并为推理的投机解码铺路。

实现:在主干之后接 `mtp_num_layers` 个 MTP 模块,每个预测一个更远的 token。`MTPLossAutoScaler` 给 MTP loss 缩放(`pp_schedulers_analysis.md` §0.3 见过)。MTP 与 PP 配合时可单独占 VPP stage(`mtp_standalone`)。

---

## 7. SSM / Mamba 与混合模型(`ssm/`)

注意力是 `O(s²)`。**状态空间模型(SSM)/ Mamba** 用一个**固定大小的循环状态**线性处理序列(`O(s)`),长序列友好。

- **`MambaMixer`**(`ssm/mamba_mixer.py`):Mamba 的核心 —— 选择性扫描(selective scan),输入依赖的状态转移。`MambaMixerSubmodules`。
- **`MambaLayer`**(`ssm/mamba_layer.py`):一个 Mamba 层(`MambaLayerSubmodules`)。
- **`GatedDeltaNet`**(`ssm/gated_delta_net.py`):门控 delta net,另一种线性注意力 / SSM 变体(Qwen3-Next)。
- **`MambaContextParallel`**:Mamba 的 CP 支持。
- **混合模型**:把 **Mamba 层 + attention 层 + MLP 层**按某种 pattern 交错(`models/hybrid/`、`hybrid_layer_allocation`)—— 兼顾 SSM 的长序列效率与 attention 的精确检索能力。`ssm/mlp_layer.py` 的 `MLPLayer` 就是 `TransformerLayer` 的纯 MLP 子类,供混合模型排布。

Mamba 的"KV"是固定大小循环状态,推理时由 `MambaSlotAllocator` 分配(见 `inference_engine_analysis.md` §5.2);打包用 `PackedSeqParams.seq_idx`(见 `dataset_analysis.md` §3.2)。

---

## 8. 具体模型装配(`models/`)

`models/` 把上述组件装成完整模型:

```
GPTModel = embedding(词嵌入 + 位置编码)
         + TransformerBlock(一摞 TransformerLayer,按 spec 组装,dense/MoE/Mamba 混排)
         + final layernorm
         + output layer(语言模型头,VocabParallelEmbedding 或独立投影)
```

`models/` 下:`gpt/`(GPT 系,含 layer specs、MoE module specs)、`bert/`、`T5/`、`mamba`、`hybrid/`(SSM-attention 混合)、`common/`(共享的 embedding、language_module 等)、多模态。`gpt_layer_specs.py` 是 spec 工厂的集中地。

PP 下,`GPTModel` 按 PP rank 只实例化本 stage 的层(首 stage 带 embedding、末 stage 带 output layer,见 `parallelism_orchestration_analysis.md` 的 `embd` 组)。

---

## 9. 小结

- **Spec 系统是地基**:模型不写死,由 `ModuleSpec` + `TransformerLayerSubmodules` 插槽表运行时**组装**;同一层类换 spec 就换实现(TE/local/inference)、换组件(dense/MoE/Mamba)、开关特性 —— 这也是训推路径切换、混合模型的实现基础。
- **TransformerLayer**:pre-norm 残差结构,注意力子层 + MLP 子层,各夹 layernorm 与 bias-dropout-add;子类 `MoETransformerLayer` / `HyperConnectionTransformerLayer`。
- **注意力家族**:MHA → GQA(共享 KV 头,省 KV cache)→ MLA(K/V 压成低秩 latent,KV cache 砍到 ~1/10,decode 用吸收);QK-Norm 稳定。
- **MoE Router**:打分 → score function(softmax/sigmoid)→ top-k(可 group-limited)→ 负载均衡(aux_loss / sinkhorn / aux-loss-free 动态偏置);产出的 routing_map 交给 `ep_analysis.md` 的 dispatcher。
- **MLP/归一化/位置编码**:SwiGLU FFN、RMSNorm、RoPE/YaRN —— 都是填进层插槽的 spec。
- **MTP**:额外预测更远的 token,增信号密度。
- **SSM/Mamba**:固定状态、`O(s)` 处理长序列;混合模型把 Mamba/attention/MLP 层交错。
- **`models/`** 把这些装成 `GPTModel` 等;PP 下按 stage 切层。

至此,Megatron 的"**模型本身**"与之前 17 份"**系统层**"文档合起来,构成完整图景。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。实验性注意力变体、各具体模型细节未逐一展开。配套文档:`ep_analysis.md`、`cp_analysis.md`、`tp_analysis.md`、`recompute_analysis.md`、`inference_engine_analysis.md`。*

## Related Pages

- [[ep_analysis]] · [[cp_analysis]] · [[tp_analysis]] · [[recompute_analysis]] · [[inference_engine_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
