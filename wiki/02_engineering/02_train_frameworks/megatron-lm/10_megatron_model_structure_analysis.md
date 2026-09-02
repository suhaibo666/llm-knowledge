---
title: "Megatron-LM 模型结构深度解析(Model Structure)"
---

# Megatron-LM 模型结构深度解析(Model Structure)

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；本页落在本轮改动文件上的引用已按 difflib 逐行对齐重定位（含裸续引 `:NNN`），指向历史基线（`ee3f1ff` / `232c478d4`）的引用按原样冻结、未参与重定位。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。原「正文以 `ee3f1ff` 为准、`[!update]` 段以 `232c478d4` 为准」的两套行号口径就此合一——全页只剩 `71092579` 一套。
> 核心文件:`megatron/core/transformer/spec_utils.py`、`megatron/core/transformer/transformer_layer.py`、`megatron/core/transformer/transformer_block.py`、`megatron/core/transformer/attention.py`、`megatron/core/transformer/multi_latent_attention.py`、`megatron/core/transformer/mlp.py`、`megatron/core/transformer/moe/router.py`、`megatron/core/transformer/multi_token_prediction.py`、`ssm/`、`models/`
> 配套阅读:`14_megatron_ep_analysis.md`(MoE dispatcher)、`13_megatron_cp_analysis.md`、`12_megatron_tp_analysis.md`、`18_megatron_recompute_analysis.md`
> 定位:之前 17 份文档都讲"**怎么把模型大规模训起来/服务起来**"(并行、显存、稳定性、推理、数据);本文讲"**模型本身长什么样**" —— 一个 transformer 模型由什么构成。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。

---

## 1. 背景:模型不是写死的,是"拼"出来的

Megatron 的模型**不是一个固定的 `nn.Module` 类**,而是由一套 **Spec(规格)系统**在运行时**组装**出来的。同一个 `TransformerLayer` 类,喂不同的 spec,就能变成:
- 用 TransformerEngine 算子 / 用 Megatron 自带算子 / 用 inference-optimized 算子;
- dense FFN 层 / MoE 层 / Mamba 层;
- 带 / 不带 hyper connections;
- MHA / GQA / MLA 注意力。

所以理解 Megatron 模型结构,要先理解 **Spec 系统**(§3),再看它拼出的**层结构**(§4)和各**组件**(§5–9)。

```
模型 = GPTModel(embedding + 一摞 TransformerLayer + output layer)
            │
TransformerLayer = 按 TransformerLayerSubmodules 这张"插槽表"组装
            │
每个插槽(self_attention / mlp / layernorm …) = 一个 ModuleSpec,运行时 build 成真实模块
```

---

## 2. 为什么这么设计：把「后端差异」和「结构差异」一起压进 spec，而不是压进模型代码

引入 spec 系统的提交只留下一行标题 —— “Modular transformer layer via spec based customization (try 2)”（`cff83981f`，2023-08-16），没有正文、没有关联设计文档。所以“为什么是 spec，而不是别的写法”只能从代码自己留下的约束里读。下面三条判据都是源码写着的；由它们推出的“因此不选另一条路”是本文的推理，见 §2.2 末尾的推断标注。

**判据一：后端之间差的不只是算子实现，还有“几个槽合成一个模块”。** `BackendSpecProvider` 这个 Protocol（`megatron/core/models/backends.py:53-54`，“A protocol for providing the submodules used in Spec building”）要求每个后端回答 `fuse_layernorm_and_linear()` —— “Does the backend support a single module for layernorm and linear”（`:67-68`）；`LocalSpecProvider`（`:99`）与 `InferenceSpecProvider`（`:149`）给出的答案不同。这类差异改变的是**层的插槽拓扑**，不是某个算子的实现。

**判据二：TE / apex 是可选依赖，模型定义不能在 import 期就绑死它们。** `megatron/core/models/gpt/gpt_layer_specs.py:49-57` 在 `HAVE_TE` 为假时把 `TEFusedMLP` / `TEFusedMLPWithGroupedLinear` / `TENorm` / `TESpecProvider` 一律置成 `None`；`:65-79` 在 apex 缺失时把 `LNImpl` 从 `FusedLayerNorm` 换成 `WrappedTorchNorm` 并 warn。`ModuleSpec.module` 允许写成 `(路径, 类名)` 二元组、由 `import_module` 在 build 时才动态 import（`megatron/core/transformer/spec_utils.py:44-56`）——“选哪个实现”被从 import 期推迟到了组装期。

**判据三：一个 spec 要能代表整栈的层。** `TransformerBlock` 拿到一个 `ModuleSpec` 时，靠 `issubclass(spec.module, BaseTransformerLayer)` 判断能否把它复制成 `num_layers` 份（`megatron/core/transformer/transformer_block.py:262-272`）；`BaseTransformerLayer` 的 docstring 自陈它是个 “dummy class”，存在的唯一目的就是让这个 fan-out 判定成立（`megatron/core/transformer/transformer_layer.py:298-308`）。

### 2.1 spec 换来的三种自由

- **可换实现**:同一层,spec 指向 TE 算子或 Megatron 自带算子或 `inference_optimized` 算子 —— 不改模型代码(呼应 `30_megatron_rl_posttraining_consistency_analysis.md` §4 的训推路径切换)。
- **可裁剪**:不需要的组件用 `IdentityOp` 占位。
- **可混搭**:dense 层与 MoE 层用不同 spec,按 `moe_layer_freq` 间隔排布;hybrid 模型把 Mamba 层 / attention 层 / MLP 层混排。

### 2.2 被否掉的替代

| 更直观的做法 | 看似省下什么 | 在基线 `71092579` 下会失去什么 |
|---|---|---|
| 每种后端各写一份 `TransformerLayer` 子类 | 结构一眼可见，不必绕 spec | 后端差异里含“layernorm 与 linear 是否合并成一个模块”这种拓扑差异（`megatron/core/models/backends.py:67-68`），子类数按 后端 × 结构 组合膨胀 |
| 在 `forward` 里用 `if HAVE_TE` 分支选实现 | 少一层间接 | 缺 TE / 缺 apex 时相关符号根本是 `None` 或被整体换掉（`megatron/core/models/gpt/gpt_layer_specs.py:49-57`、`:65-79`），`if` 分支要求这些符号在 import 期就存在 |
| 不需要的槽直接从层里删掉 | 少几次空调用 | 槽的默认值是 `IdentityOp` / `IdentityFuncOp`（`megatron/core/transformer/transformer_layer.py:279-292`），`forward` 因此不必对每个槽判空；删槽会把判空成本转嫁回 `forward` |
| 取消 `BaseTransformerLayer` 这个空基类 | 少一个“什么都不做”的类 | `_get_block_submodules` 正是靠 `issubclass(..., BaseTransformerLayer)` 才能把一个 spec 展开成整栈，否则落进 `raise Exception(f"specialize for {spec.module.__name__}.")`（`megatron/core/transformer/transformer_block.py:265-276`） |

> [!note] 推断
> 上表“看似省下什么 / 会失去什么”两列里，**每条 locator 指向的事实都是源码写着的**（Protocol 要回答什么、缺依赖时符号被置 `None`、槽的默认值、fan-out 的 `issubclass` 判定与 `raise Exception` 分支）；但“NVIDIA 曾考虑过这些替代并否决”**没有任何源码或提交信息支持** —— 引入提交 `cff83981f` 只有标题、无正文，`spec_utils.py` 与 `backends.py` 里也没有对比性的设计注释。把它们并列成“被否掉的替代”是本文的重构，不是作者原话。

---

## 3. Spec 系统:`ModuleSpec` 与模型组装

### 3.1 `ModuleSpec`(`megatron/core/transformer/spec_utils.py:13`)

```python
@dataclass
class ModuleSpec:
    module: Union[Tuple, type]   # 模块类本身,或 (路径, 类名) 二元组(动态 import)
    params: dict = {}            # 初始化该模块要传的 kwargs
    submodules: object = None    # 嵌套的子模块 spec(递归)
    metainfo: dict = {}
```

`build_module(spec, *args, **kwargs)`(`:74`)按 spec **递归实例化**:先 build 出 `submodules` 里的每个子 spec,再用它们 + `params` 构造 `module`。`spec()` 直接 `__call__` 等价于 `build_module`。

### 3.2 `TransformerLayerSubmodules` —— 一层的"插槽表"(`megatron/core/transformer/transformer_layer.py:252`)

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

默认全是 `IdentityOp`(空操作)—— **spec 不填的槽就是恒等映射**。`megatron/core/models/gpt/gpt_layer_specs.py` 等提供工厂函数,产出填好的 spec(TE 版 / local 版 / dense / MoE)。

---

## 4. Transformer 层结构

### 4.1 `TransformerLayer`(`megatron/core/transformer/transformer_layer.py:314`)

`forward`(`:861`)= **注意力子层 + MLP 子层**,标准 pre-norm 残差结构:

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

`bda` = **bias-dropout-add**,融合算子(见 `23_megatron_precision_cudagraph_fusion_analysis.md` §5 `fused_bias_dropout`)。

### 4.2 子类

- **`MoETransformerLayer`**(`:2868`):`mlp` 槽是 `MoELayer`(见 `14_megatron_ep_analysis.md`);处理 MoE 特有的 recompute、padding_mask 等。
- **`HyperConnectionTransformerLayer`**(`:1910`):hyper connections(mHC)—— 用**多条残差流**代替单条残差,层间连接更丰富;`*_hyper_connection` 槽生效,PP 通信传 n-stream 张量(见 `15_megatron_pp_schedulers_analysis.md` 里 `enable_hyper_connections` 的形状处理)。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> - **行号漂移**(全页已统一到 `71092579`,本条仅记录漂移轨迹):`TransformerLayer` `megatron/core/transformer/transformer_layer.py` `ee3f1ff:279` → `232c478d4:313` → `71092579:314`;`forward` `:710` → `:841` → `:842`;`TransformerLayerSubmodules` `:217` → `:251` → `:252`;`HyperConnectionTransformerLayer` `:1488` → `:1715` → `:1881`;`MoETransformerLayer` `:1983` → `:2213` → `:2844`。结构未变,仅因 DSv4/mHC 等新增而下移。
> - **mHC 现已支持 HybridModel**(#4949):`HyperConnectionHybridLayer`(`megatron/core/models/hybrid/hybrid_block.py:75`)作为**包装器**驱动被包的 `TransformerLayer`(经 `_called_from_hybrid_mhc_wrapper` 旁路直接调用其 `forward`,见 `megatron/core/models/hybrid/hybrid_block.py:470`,绕过"请用 HyperConnectionTransformerLayer"的断言),让 Mamba/GDN/attention 混合栈也能用多残差流;n-stream BDA 负责残差合并。更快的 mHC 融合 kernel 见 #4624。

### 4.3 `TransformerBlock`(`megatron/core/transformer/transformer_block.py`)

一摞 `TransformerLayer` 的容器。负责:层的构建与编号、`_checkpointed_forward`(激活重计算,见 `18_megatron_recompute_analysis.md`)、PP 下本 stage 只放 `num_layers_per_pipeline_rank` 层、final layernorm。

---

## 5. 注意力家族

`megatron/core/transformer/attention.py`:`Attention`(ABC)→ `SelfAttention` / `CrossAttention`。

### 5.1 MHA / GQA / MQA

`SelfAttentionSubmodules`:`linear_qkv`(QKV 投影)、`core_attention`(缩放点积注意力,TE fused 或 local)、`linear_proj`(输出投影)、`q_layernorm` / `k_layernorm`(QK-Norm,稳定性)。

按 **`num_query_groups`** 区分三档:
- **MHA**:`num_query_groups = num_attention_heads` —— 每个 Q 头有自己的 K/V 头。
- **GQA(分组查询注意力)**:`num_query_groups < num_attention_heads` —— 多个 Q 头**共享**一组 K/V 头。KV 头变少 → **KV cache 变小、推理 decode 带宽压力降**(`31_megatron_inference_engine_analysis.md` §1.2 的瓶颈)。
- **MQA**:`num_query_groups = 1` —— 所有 Q 头共享 1 组 K/V,GQA 的极端。

TP 下 QKV 投影是 ColumnParallel(按头切)、输出投影 RowParallel(见 `12_megatron_tp_analysis.md` §6.2)。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。注意力输出门控。`attention_output_gate`(整 `head_dim` 门)与新增的 **`head_wise_attn_gate`**(每头一个标量 sigmoid 门,Step-3.5-Flash,#4841,`megatron/core/transformer/attention.py:1448`)二选一、不可同开。门权重并入 `linear_qkv` 一并产出,对 `core_attn_out` **逐头乘 `sigmoid(gate)`**;head-wise 门仅自注意力(`attention_type != "cross"`),且要求 `num_attention_heads` / `num_query_groups` 满足整除约束(`megatron/core/transformer/transformer_config.py:1594`,具体两条整除断言在 `:1603`/`:1611`)。

### 5.2 MLA —— 多头潜在注意力(`megatron/core/transformer/multi_latent_attention.py`)

`MultiLatentAttention` → `MLASelfAttention` / `FusedMLASelfAttention`。DeepSeek-V2/V3 的核心。

**动机**:GQA 减 KV 头,但 KV cache 仍 `∝ 头数 × head_dim`。MLA 更激进 —— 把 K/V **压成一个低秩潜在向量**:

```
标准 MHA/GQA:KV cache 存每个头的完整 K、V          → cache ∝ n_kv_heads · head_dim
MLA:         K/V 投影到一个低维 latent c_KV(+ 一个解耦的 RoPE key)
             KV cache 只存 c_KV → 远小于完整 K/V    → cache 砍到 ~1/10(DeepSeek-V3)
             compute 时再把 c_KV 上投影回每头的 K、V
```

- `MLASelfAttentionSubmodules` 含 Q / KV 的压缩(`q_lora` / `kv_lora` 风格低秩投影)与上投影。
- **`cache_mla_latents`**:推理时 KV cache 存 latent 而非完整 K/V;decode 时做 **"吸收(absorption)"** —— 把上投影矩阵数学上吸收进 Q 投影 / 输出投影,**永不物化完整 K/V**,`FusedMLASelfAttention`(`:1359`)即此路径。
- 与 CP 的配合见 `13_megatron_cp_analysis.md`。

### 5.3 实验性注意力变体

`transformer/experimental_attention_variant/`:DeepSeek 的 DSA、v4 hybrid attention、CSA、absorbed MLA —— 都是 MLA / 注意力的研究性变体,以独立模块 spec 形式接入。下面 §5.4 把其中最重要的 **DeepSeek-V4** 路径展开。

### 5.4 DeepSeek-V4:DSA 稀疏注意力 + 混合压缩注意力(NEW)

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> DeepSeek-V4 是 ee3f1ff 之后**最大的模型新增**(#5042 "Enable Deepseek-v4 hybrid_model Part 1/N")。它把 §5.3 一笔带过的"实验性变体"落地为可训练的完整路径,是 §5.2 MLA 的**稀疏化 + 压缩化**延伸。代码在 `transformer/experimental_attention_variant/`(`megatron/core/transformer/experimental_attention_variant/dsa.py`、`megatron/core/transformer/experimental_attention_variant/csa.py`、`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py`、`megatron/core/transformer/experimental_attention_variant/dsa_kernels.py`)。

DeepSeek 稀疏注意力由 `config.experimental_attention_variant` 选档,有两条路径:

| variant | 上层注意力类 | core attention | 索引器 | 压缩 | 对应模型 |
|---|---|---|---|---|---|
| `dsa` | `MLASelfAttention` | `DSAttention`(`megatron/core/transformer/experimental_attention_variant/dsa.py:1928`) | `DSAIndexer`(`megatron/core/transformer/experimental_attention_variant/dsa.py:1402`) | 无(对未压缩 KV 选 top-k) | DeepSeek-V3.2-Exp |
| `dsv4_hybrid` | `DSv4HybridSelfAttention`(`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:483`,父类 `DSv4HybridAttention:59`) | `CompressedSparseAttention`(`megatron/core/transformer/experimental_attention_variant/csa.py:1703`) | `CSAIndexer`(`megatron/core/transformer/experimental_attention_variant/csa.py:1435`) | `Compressor`(`megatron/core/transformer/experimental_attention_variant/csa.py:1001`)4× / 128× | DeepSeek-V4 |

两条都建在 MLA(§5.2)上 —— Q/KV 仍走低秩压缩投影,稀疏化加在 **core attention** 这一层。spec 工厂在 `megatron/core/models/gpt/experimental_attention_variant_module_specs.py`(`get_dsa_module_spec_for_backend:124`、`get_dsv4_hybrid_module_spec_for_backend:176`)。

**(1) DSA = DeepSeek Sparse Attention —— 学出来的 top-k 检索**

标准因果注意力让每个 query attend 前面所有 token(每 query `O(s)`)。DSA 加一个轻量 **indexer**:对每个 query 用小型打分网络算出对所有(压缩)KV 位置的 `index_scores`,只保留 **top-k**(`dsa_indexer_topk`,recipe 里 512)个最相关位置参与真正的注意力 —— 把每 query 的注意力开销从 `O(s)` 降到 `O(k)`。

- indexer 需要**被训练**:`compute_dsa_indexer_loss`(`megatron/core/transformer/experimental_attention_variant/dsa.py:465`)用 KL 散度让 indexer 预测的 top-k 分布逼近真实注意力分布(系数 `dsa_indexer_loss_coeff`;`dsa_indexer_use_sparse_loss` 选稀疏版)。
- 这条损失经 `DSAIndexerLossAutoScaler`(`megatron/core/transformer/experimental_attention_variant/dsa.py:1149`)**单独缩放、单独反传**,不影响主注意力前向(类似 §8 MTP loss 的旁路 autoscaler 套路)。
- 混合布局下某些 PP rank 没有 indexer 层,跨 rank/层规约与日志由 `DSAIndexerLossLoggingHelper`(`megatron/core/transformer/experimental_attention_variant/dsa.py:292`)统一处理(无 indexer 的 rank 要补零参与集合通信,否则 hang)。

**(2) DSv4 的 CSA / HCA —— 在 DSA 之上再加 KV 压缩**

DeepSeek-V4 在 indexer 之外引入 **Compressor**(`megatron/core/transformer/experimental_attention_variant/csa.py:1001`):用一组**学习的门控权重 + per-position embedding**把每 `compress_ratio` 个 token 池化成一个压缩 KV token,attend 压缩后的短序列。`CompressedSparseAttention`(`megatron/core/transformer/experimental_attention_variant/csa.py:1703`)按 **per-layer `compress_ratio`** 三态构建,复用同一套代码:

| ratio | 含义 | Compressor | Indexer | 层符号 |
|---|---|---|---|---|
| 0 | 仅滑动窗口(`csa_window_size`,默认 128) | 不建 | 不建 | `W` |
| 4 | 窗口 + 4× 压缩 KV + 学习索引器(**重叠**压缩 `coff=2`) | 建 | 建 | `C` |
| 128 | 窗口 + 128× 压缩 KV、attend 全部(**非重叠** `coff=1`) | 建 | 不建 | `H` |

每层把 (a) 滑动窗口内的原始 KV 与 (b) 压缩 KV 拼起来(`_build_kv_full`),再加一个**可学习的 per-head attention sink**(`attn_sink`)。只有 `ratio==4` 的层拥有 indexer(并贡献 indexer loss);`ratio==128` 直接 attend 全部压缩 KV,无需 top-k;`ratio==0` 退化为纯滑窗注意力(整段 CSA/HCA 代码被复用)。`apply_dsa_kernel_fusion` 走 `megatron/core/transformer/experimental_attention_variant/dsa_kernels.py` 的融合 kernel(需 SM100+ Blackwell + FlashMLA/cuDNN)。

**(3) 混合注意力布局 —— D / C / H / W 层符号**

HybridModel 的层 pattern 字符串(见 §9)新增四个 MLA 系注意力符号(`megatron/core/models/hybrid/hybrid_layer_allocation.py` `Symbols`):`D`=DSA、`C`=CSA(ratio 4)、`H`=HCA(ratio 128)、`W`=窗口(ratio 0)。四者可互相混排,但 **`MLA_ATTENTION={D,C,H,W}` 与标准 `*` 注意力互斥**(同一模型不能既有 `*` 又有 MLA 系)。`hybrid_dsv4_stack_spec`(`megatron/core/models/hybrid/hybrid_layer_specs.py`)让 HybridModel 的 `D`/`C`/`H`/`W` 层复用 GPT 的 `get_dsv4_hybrid_module_spec_for_backend`,与 GPT `dsv4_hybrid` 路径**数值等价**。GPT 路径则用 `csa_compress_ratios` 数组**逐层**指定 ratio(DSv4-Flash recipe:`[0,0,4]+[128,4]*20+[0]`,即首两层窗口、第 3 层 CSA、之后 20 组 HCA/CSA 交替、末层窗口;数组长度须 ≥ `num_layers + mtp_num_layers`,#5042 把 `==` 放宽为 `>=` 以容纳 MTP 深度展开成多层)。

**(4) DSv4 的几何与分组输出投影**

- DSv4 hybrid 强制从 `v_head_dim`、`qk_pos_emb_head_dim` **派生** `qk_head_dim = kv_lora_rank = v_head_dim − qk_pos_emb_head_dim`(`megatron/core/transformer/transformer_config.py:4013`,派生赋值在 `:4023-4025`);recipe:`v_head_dim=512`、`qk_pos_emb_head_dim=64` → 派生 448。
- 输出投影是**分组低秩**:`o_groups`(默认 8)× `o_lora_rank`(默认 1024)的 `linear_o_group_proj` 再接 `linear_proj`(`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:179-202`)—— 比单一 `wo` 省参省算。
- RoPE:压缩层(ratio>1)用 **YaRN**(base `csa_compress_rotary_base`=40000);window-only 层(ratio==0)用**标准 RoPE**(#5018 修正了 window 层错用 YaRN 的 bug,并修了 `csa_dense_mode` 下的 dense loss)。
- 约束:DSv4 Hybrid **只支持 TP=1**,不支持 checkpoint core attention / offload qkv linear,**不支持推理**(TP=1/checkpoint/offload 三条断言在 `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:90-99`;推理断言另在 `forward` 的 `:253-254`)。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。DSv4-Flash 的 **Q-up FLOPs 统计修正** —— 用 `args.v_head_dim` 替代 `args.num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)`(#5142)。**locator 更正**:原写的 `megatron/training/training.py:516-529` 在 `232c478d4` 上是 GDN 层的 FLOPs 函数,并非本条;MLA 的 `q_term` 当时在 `232c478d4:657-667`。基线 `71092579` 下该式已收敛成 DSv4 专用的一行 `q_term = q_lora_rank * (hidden_size + num_attention_heads * v_head_dim + 1)`(`megatron/training/training.py:543`,`kv_term`/`o_term` 在 `:544`/`:545`)。另 #3026 修了 `dsa` 路径的 rope 与 spec 多个 bug。完整可跑配置见 DeepSeek-V4-Flash recipe(#5266,`examples/moe_recipes/deepseek_v4_flash/gb200/`——基线 `71092579` 下并无 `examples/moe_recipes/gb200/`,recipe 按模型分目录):`num_layers=43, hidden=4096, heads=64, num_experts=256, moe_topk=6, mtp_num_layers=1`。

---

## 6. MoE Router 算法(补 `14_megatron_ep_analysis.md` 的空白)

`14_megatron_ep_analysis.md` 讲了 token 怎么**分发**(dispatcher),但没讲 token 怎么**被路由**。补在这里。`megatron/core/transformer/moe/router.py`:`Router`(ABC)→ `TopKRouter`(`:157`)。

`TopKRouter.forward` 产出 `(probs, routing_map)`:

1. **打分**:轻量线性层把 hidden 投成 `[s, E]` logits(建议 fp32,见 `14_megatron_ep_analysis.md` §3.3)。
2. **score function**:`softmax` 或 `sigmoid` 把 logits 变概率。
3. **选 top-k**,可叠加:
   - **group-limited routing**(`moe_router_num_groups` / `group_topk`):先选 top-k 个**专家组**,再在组内选专家 —— 限制一个 token 跨越的节点数(DeepSeek-V3,利于 EP 跨节点通信)。
   - **`topk_routing_with_score_function`** —— 统一的 topk + score 实现。
4. **负载均衡**(决定路由怎么"被纠偏",对应 `14_megatron_ep_analysis.md` §7):
   - **`aux_loss`** —— 辅助损失惩罚不均衡(微批 / 序列 / 全局级)。
   - **`sinkhorn`** —— 最优传输式均衡(与 aux_loss 互斥)。
   - **aux-loss-free**(`moe_router_enable_expert_bias`)—— 给每个专家一个**动态偏置** `expert_bias`(`_maintain_float32_expert_bias` 保持 fp32),过载专家偏置降、冷门专家偏置升,**不引入辅助损失**(DeepSeek-V3 用法)。

产出的 `routing_map`(`[s,E]` 多热掩码)和 `probs` 就交给 `14_megatron_ep_analysis.md` 的 dispatcher。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。DeepSeek-V4 引入第三类"路由"——**hash routing(哈希路由)** 与**强制均衡**开关。
> - **哈希路由**(`megatron/core/transformer/moe/router.py:735 _hash_routing`,`is_hash_layer` 在 `:214` 判定,`tid2eid` 构表在 `:226-229`,#5042):前 `moe_n_hash_layers`(recipe=3)层的 MoE **不靠打分选专家**,而是用一张预先算好的 `tid2eid` 查找表把 **token id 直接映射到专家 id**(`tid2eid = (token_id + k) % num_experts`,取 topk 个;DSv4-Pro 的推理 checkpoint 直接带训练好的表)。需要 `actual_vocab_size`。哈希路由把"哪个 token 走哪个专家"固定下来,天然均衡且可缓存,常用于浅层。
> - **强制均衡**(调试/早期训练):`moe_router_force_load_balancing`(用 `apply_random_logits` 随机化 logits)、`moe_router_force_biased`(用 `apply_biased_logits` 施加确定性偏置)。在 hash 层这两者会**覆盖 `tid2eid` 的结果**,改用对(随机/偏置后)logits 取 top-k(`megatron/core/transformer/moe/router.py:764`,#5130)。#5130 同时把 **ClampedSwiGLU** 加进 MoE 的 `mlp_op_fuser`(融合算子角度由别处文档负责)。
> - **aux_loss / z_loss 接受 `padding_mask`**(`megatron/core/transformer/moe/router.py:641/707/774`,Qwen3.5,#4776):打包(THD)下 padding token 不计入负载均衡与 z-loss 统计;#4776 的 follow-up 仅触及多模态 example。

---

> [!update] 2026-09-01（基线 `85902ef59`，#6704）：**哈希路由的"层阈值"现在可以由调用方显式传入。** `MoELayer.__init__` 新增 `hash_moe_layer_threshold: Optional[int]`（`megatron/core/transformer/moe/moe_layer.py:239`），仅在非 None 时才放进 `router_kwargs` 透传（`:280-281`）；`Router.__init__` / `TopKRouter.__init__` 同步新增该参数（`megatron/core/transformer/moe/router.py:55`、`:187`），缺省回落到 `config.moe_n_hash_layers`（`:73-76`，字段定义在 `megatron/core/transformer/transformer_config.py:907`）。`RouterBuilder` 协议的签名也随之扩展。
>
> **为什么需要这个参数**：docstring 写明了动因——"Hybrid models use this to translate a **MoE-position count**"（`megatron/core/transformer/moe/router.py:195-196`）。`moe_n_hash_layers` 数的是"前几个 **MoE 层**"，而 router 手里只有 `layer_number`，即"第几**层**"。在 dense/MoE 交替或含 Mamba 段的 hybrid 模型里这两个计数不重合，直接拿 `layer_number` 去比 `moe_n_hash_layers` 会选错层。把翻译后的阈值从外部传进来，比让 router 反过来理解整个模型的层布局要窄得多——这与 §5.4 里 hybrid 层分配的责任划分是同一取向。

## 7. MLP / 激活 / 归一化 / 位置编码

- **MLP**(`megatron/core/transformer/mlp.py`):dense FFN —— `linear_fc1`(ColumnParallel,`h→H`)→ 激活 → `linear_fc2`(RowParallel,`H→h`)。激活常用 **SwiGLU**(门控,`fc1` 输出对半切,一半过 SiLU 门控另一半;有 `fused_bias_swiglu` 融合 kernel)。
- **归一化**:`LayerNorm` / `RMSNorm`(现代模型多用 RMSNorm,省一个均值);有 `fused_layer_norm`。可选 **QK-Norm**(`q_layernorm`/`k_layernorm`,稳定 attention logit,呼应 `28_megatron_training_stability_observability_analysis.md` §1.5)。
- **位置编码**:**RoPE**(旋转位置编码)是主流;长上下文用 **YaRN**(`fused_mla_yarn_rope_apply`)。代码在 `models/common/embeddings/`。
- 这些组件都以 `ModuleSpec` 填进 §3.2 的层插槽。

---

## 8. MTP —— Multi-Token Prediction(`megatron/core/transformer/multi_token_prediction.py`)

标准 LM 每步只预测**下一个** token。**MTP**(DeepSeek-V3)让模型额外预测**再后面若干个** token —— 增加训练信号密度、并为推理的投机解码铺路。

实现:在主干之后接 `mtp_num_layers` 个 MTP 模块,每个预测一个更远的 token。`MTPLossAutoScaler` 给 MTP loss 缩放(`15_megatron_pp_schedulers_analysis.md` §1.4 见过)。MTP 与 PP 配合时可单独占 VPP stage(`mtp_standalone`)。

> [!update] 2026-09-01（基线 `85902ef59`，#6583）：**hybrid 模型下 MTP 的部分 CUDA Graph 捕获改为跨层分组。** attention-only 层可以与紧随其后的 partial-MoE-capture 层合并成一张图：判定谓词 `_can_group_te_cuda_graph_with(next_layer)` 要求本层 attention-only 且下一层 `_inner_is_partial_moe_capture()`；成组后 `_set_te_cuda_graph_group_tail` 记住 tail，重放时 tail 用 `_resume_partial_moe_cuda_graph(out)` 回调 `inner_layer.resume_moe_experts_after_partial_cudagraph(out)` 续算被留在图外的专家段（均在 `megatron/core/models/hybrid/hybrid_block.py`）。`parameters()` 一并重写以纳入 group tail，否则优化器会漏掉 tail 层参数。
>
> **动因**：部分捕获本来就是为了绕开 MoE 的动态形状——把专家段留在图外。代价是每层多一次图边界。把 attention-only 的前驱并进同一张图，是用"分组"摊薄这些边界，而不是去把专家段本身变静态。图侧全貌见 [[23_megatron_precision_cudagraph_fusion_analysis]] §8.6。

---

## 9. SSM / Mamba 与混合模型(`ssm/`)

注意力是 `O(s²)`。**状态空间模型(SSM)/ Mamba** 用一个**固定大小的循环状态**线性处理序列(`O(s)`),长序列友好。

- **`MambaMixer`**(`megatron/core/ssm/mamba_mixer.py`):Mamba 的核心 —— 选择性扫描(selective scan),输入依赖的状态转移。`MambaMixerSubmodules`。
- **`MambaLayer`**(`megatron/core/ssm/mamba_layer.py`):一个 Mamba 层(`MambaLayerSubmodules`)。
- **`GatedDeltaNet`**(`megatron/core/ssm/gated_delta_net.py`):门控 delta net,另一种线性注意力 / SSM 变体(Qwen3-Next)。
- **`MambaContextParallel`**:Mamba 的 CP 支持。
- **混合模型**:把 **Mamba 层 + attention 层 + MLP 层**按某种 pattern 交错(`models/hybrid/`、`hybrid_layer_allocation`)—— 兼顾 SSM 的长序列效率与 attention 的精确检索能力。`megatron/core/ssm/mlp_layer.py` 的 `MLPLayer` 就是 `TransformerLayer` 的纯 MLP 子类,供混合模型排布。

Mamba 的"KV"是固定大小循环状态,推理时由 `MambaSlotAllocator` 分配(见 `31_megatron_inference_engine_analysis.md` §5.2);打包用 `PackedSeqParams.seq_idx`(见 `11_megatron_dataset_analysis.md` §5.2)。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。Mamba / GDN 增量
>
> > [!deprecated] **文件已不存在**:`megatron/core/ssm/gated_delta_net.py` 在基线 `71092579` 下已被删除(`git ls-tree` 零命中)。PR #6088(`1c44a5709`,cherry-pick #5843「Refactor: extract and split common logic between GDN & GDN2」)把它拆成一个包 `megatron/core/ssm/gated_delta_net/`,含 `common.py` / `gdn.py` / `kda.py`。下条的 THD 打包逻辑现位于 `megatron/core/ssm/gated_delta_net/gdn.py:189`(另有 `:333`、`common.py:872`/`:909` 的同类分支)。
>
> - **GDN 支持序列打包(THD)**(#2645,旧路径 `megatron/core/ssm/gated_delta_net.py:340+`,现 `megatron/core/ssm/gated_delta_net/gdn.py:189`):`packed_seq_params.qkv_format=='thd'` 时按 `cu_seqlens` 把打包 buffer 拆成各条子序列、**逐条**做 CP↔HP 的 all-to-all 再 chunk 扫描(要求 `batch==1`、非 deterministic 模式;后续 #4913 把逐序列 all-to-all 融成统一一次)。MTP 路径也透传 `packed_seq_params`。GDN 另有"整模块 `gdn` 选择性重计算"(#5296)与 `norm_out` 选择性重计算(#4715)。
> - **Mamba conv 参数直挂 mixer**(#4899,`megatron/core/ssm/mamba_mixer.py:310`):原 `self.conv1d`(`nn.Conv1d`)拆成直接的 `conv1d_weight` / `conv1d_bias` 两个 `nn.Parameter`(保留原 TP `partition_dim` / `partition_sizes` 元数据与初始化序列),便于 FSDP / 弹性(flextron)切分。
> - **混合层符号扩展**:`Symbols` 在 `M`(Mamba)/`G`(GDN)/`*`(attention)/`D`(DSA)/`-`(MLP)/`E`(MoE)之外新增 DeepSeek-V4 的 `C`/`H`/`W`(见 §5.4);`MLA_ATTENTION={D,C,H,W}` 与标准 `*` 互斥。

---

## 10. 具体模型装配(`models/`)

`models/` 把上述组件装成完整模型:

```
GPTModel = embedding(词嵌入 + 位置编码)
         + TransformerBlock(一摞 TransformerLayer,按 spec 组装,dense/MoE/Mamba 混排)
         + final layernorm
         + output layer(语言模型头,VocabParallelEmbedding 或独立投影)
```

`models/` 下:`gpt/`(GPT 系,含 layer specs、MoE module specs)、`bert/`、`T5/`、`mamba`、`hybrid/`(SSM-attention 混合)、`common/`(共享的 embedding、language_module 等)、多模态。`megatron/core/models/gpt/gpt_layer_specs.py` 是 spec 工厂的集中地。

PP 下,`GPTModel` 按 PP rank 只实例化本 stage 的层(首 stage 带 embedding、末 stage 带 output layer,见 `17_megatron_parallelism_orchestration_analysis.md` 的 `embd` 组)。

---

## 11. 约束：spec 组装不是免费的

这一页前面讲的都是“能拼出什么”。下面是拼装这条路的前提、代价、不变量，以及它明确不做的事 —— 每条都能在基线 `71092579` 上打开验证。

| 类别 | 约束 | locator |
|---|---|---|
| 前提 | spec 的 fan-out 只对两类模块成立：`TransformerBlock` 只把 `TransformerBlock` 或 `BaseTransformerLayer` 的子类 spec 展开成整栈，其余一律 `raise Exception(f"specialize for {spec.module.__name__}.")` | `megatron/core/transformer/transformer_block.py:262-276` |
| 代价 | 动态 import 的副作用未受控：`import_module` 顶上留着 “TODO: make this importer module more robust, at least make sure there are no side effects of using this as is” | `megatron/core/transformer/spec_utils.py:47-48` |
| 不变量 | 不填的槽不是“没有”，是 `IdentityOp` / `IdentityFuncOp`；`forward` 正是依赖这一点才不必对每个槽判空 | `megatron/core/transformer/transformer_layer.py:279-292` |
| 故意不做 | mHC 不是可自由插拔的槽：`enable_hyper_connections=True` 时直接调 `TransformerLayer.forward()` 抛 `RuntimeError`，必须经 `HyperConnectionTransformerLayer` 或 `HyperConnectionHybridLayer` | `megatron/core/transformer/transformer_layer.py:871-878` |
| 故意不做 | `HyperConnectionTransformerLayer` 要求 self-attention / MLP 两个 hyper-connection 槽都非 `IdentityOp`，并**明确不支持 cross-attention hyper connections**（`ValueError`） | `megatron/core/transformer/transformer_layer.py:1944-1957` |
| 故意不做 | mHC 的 hybrid 包装路径下不支持 EP overlap，docstring 明写 “EP-overlap is not supported on this path”，并配一条断言 | `megatron/core/transformer/transformer_layer.py:1543`、`:1545-1548` |
| 故意不做 | 同一模型的层 pattern 里，标准 `*` 与 MLA 系 `{+,D,C,H,W}` 不允许共存 | `megatron/core/models/hybrid/hybrid_layer_allocation.py:32`、`:302-307` |
| 失效条件 | DSv4 hybrid：TP=1、不支持 checkpoint core attention / offload qkv linear、不支持推理（详见 §5.4 的第 (4) 条） | `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:90-99`、`:253-254` |
| 失效条件 | GDN 不支持推理：`inference_context is not None` 时先断言必须 static batching，再 `raise NotImplementedError("GDN does not support inference for now.")` | `megatron/core/ssm/gated_delta_net/gdn.py:164-171` |
| 失效条件 | GDN 的 headwise CP 只接受 zigzag 布局，否则 `ValueError`；THD 打包分支另要求 `batch == 1` 且非 deterministic 模式 | `megatron/core/ssm/gated_delta_net/gdn.py:173-187`、`:189-193` |

一句话：spec 系统把“换实现/换结构”的成本压到了组装期，代价是**这些约束不再由类型系统保证，而是散在 build 期的断言与 pattern 校验里** —— 错配的 spec 不会在 import 时报错，会在 `build_module` / `_get_block_submodules` / pattern 校验那一刻才炸。

---

## 12. 发展趋势

> [!note] 本节是推断
> 下面每条都锚定基线 `71092579` 源码里**自陈**的 TODO 或弃用警告（locator 逐条打开核过）；“接下来会往哪走”是本文的推断，不是 NVIDIA 的承诺。

- **spec 的动态 import 仍是已知薄弱点**：`import_module` 自陈要更 robust、要保证“没有副作用”（`megatron/core/transformer/spec_utils.py:47-48`）。§11 的第二条代价就出在这里。
- **spec 工厂在收敛**：`_get_mlp_module_spec` 已标为 “on a deprecation track. Please switch to `get_mlp_module_spec`”（`megatron/core/models/gpt/gpt_layer_specs.py:513-515`）；三处 `fp8` 参数也已弃用（`:226-227`、`:429-430`、`:533-534`）。§3 的“spec 工厂集中在 `gpt_layer_specs.py`”这一格局不变，但入口函数在减少。
- **混合模型的配置口径从“比例”迁向“pattern 串”**：`hybrid_override_pattern`、`hybrid_attention_ratio`、`hybrid_mlp_ratio` 全部弃用，统一到 `hybrid_layer_pattern`（`megatron/core/models/hybrid/hybrid_model.py:198-227`）；PP>1 时不带 `|` 分段的 pattern 也已标 DEPRECATION（`megatron/core/models/hybrid/hybrid_layer_allocation.py:404-411`）。§9 的层符号表因此会越来越吃重。
- **层不再假定同构**：`sharded_state_dict` 里显式设 `non_homogeneous_layers=False` 会被警告并强制改回 True（`megatron/core/transformer/transformer_block.py:1106-1110`）。这与 §5.4 的逐层 `csa_compress_ratios`、§9 的混合层符号是同一个方向。
- **GDN 的推理路径是明写的缺口**：`# TODO: support inference` 就压在 `raise NotImplementedError` 上一行（`megatron/core/ssm/gated_delta_net/gdn.py:170-171`）—— §9 讲的 GDN 目前只是训练侧能力。
- **MoE router 的 per-layer 日志对 MTP 仍不正确**：`TODO (zijiey): fix the per_layer_logging for MTP`，注释同时说明它“does not affect the correctness of the calculation results”（`megatron/core/transformer/moe/router.py:589-592`）。

---

## 13. 小结

- **Spec 系统是地基**:模型不写死,由 `ModuleSpec` + `TransformerLayerSubmodules` 插槽表运行时**组装**;同一层类换 spec 就换实现(TE/local/inference)、换组件(dense/MoE/Mamba)、开关特性 —— 这也是训推路径切换、混合模型的实现基础。
- **TransformerLayer**:pre-norm 残差结构,注意力子层 + MLP 子层,各夹 layernorm 与 bias-dropout-add;子类 `MoETransformerLayer` / `HyperConnectionTransformerLayer`。
- **注意力家族**:MHA → GQA(共享 KV 头,省 KV cache)→ MLA(K/V 压成低秩 latent,KV cache 砍到 ~1/10,decode 用吸收);QK-Norm 稳定。**DeepSeek-V4**(§5.4)再往前一步:DSA 用学习的 indexer 做 top-k 稀疏检索,CSA/HCA 用 Compressor 压缩 KV(4×/128×),按 D/C/H/W 层符号混排,配分组低秩输出投影与 MoE 哈希路由。
- **MoE Router**:打分 → score function(softmax/sigmoid)→ top-k(可 group-limited)→ 负载均衡(aux_loss / sinkhorn / aux-loss-free 动态偏置);产出的 routing_map 交给 `14_megatron_ep_analysis.md` 的 dispatcher。
- **MLP/归一化/位置编码**:SwiGLU FFN、RMSNorm、RoPE/YaRN —— 都是填进层插槽的 spec。
- **MTP**:额外预测更远的 token,增信号密度。
- **SSM/Mamba**:固定状态、`O(s)` 处理长序列;混合模型把 Mamba/attention/MLP 层交错。
- **`models/`** 把这些装成 `GPTModel` 等;PP 下按 stage 切层。

至此,Megatron 的"**模型本身**"与之前 17 份"**系统层**"文档合起来,构成完整图景。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准。实验性注意力变体、各具体模型细节未逐一展开。配套文档:`14_megatron_ep_analysis.md`、`13_megatron_cp_analysis.md`、`12_megatron_tp_analysis.md`、`18_megatron_recompute_analysis.md`、`31_megatron_inference_engine_analysis.md`。*

---

## 配置契约：模型结构相关字段

本页正文按**结构**组织（Spec 系统、注意力家族、MoE Router、MTP、SSM）。本节给这些结构的**配置面**——`TransformerConfig` 里所有决定「模型长什么样」的字段。

**下表的类型、默认值与说明直接取自 `megatron/core/transformer/transformer_config.py` 的类体**（行号为该文件内行号），与 [[41_megatron_config_surface_analysis]] §2 的 `ArgumentGroupFactory` 生成 CLI 所用的是同一份声明，因此不会与实际 flag 漂移。

字段量大，按源码里的分段读最快：**注意力形状与变体**（`num_attention_heads` 家族、`window_size`、`qk_*`、`softmax_*`）、**DSA 索引器**（`dsa_indexer_*` 十项，对应 §5.4）、**linear attention / GDN**（`linear_*`、`kda_*`，对应 §9）、**Mamba**（`mamba_*`，同 §9）、**Hyper-Connection**（`mhc_*`、`num_residual_streams`）、**归一化与初始化**（`normalization`、`layernorm_*`、`init_method_std` 一组）、**MTP**（`mtp_*`）。


### `ModelParallelConfig`（`megatron/core/model_parallel_config.py`，2 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `perform_initialization` | `bool` | `field(default=True, metadata={'argpar…` | Controls weights initialization. This option can be useful when you know you are going to load values from a checkpoint. | `:149` |
| `use_cpu_initialization` | `bool` | `field(default=False, metadata={'argpa…` | When set to False, we initialize the weights directly on the GPU. CPU initialization is the same regardless of tensor model parallelism, but GPU initializati… | `:156` |

> 该类共 74 个字段，本表收 2 项；其余 72 项已在别处归属：主要归 [[15_megatron_pp_schedulers_analysis]] 16 项、[[12_megatron_tp_analysis]] 10 项、[[20_megatron_comm_overlap_analysis]] 10 项、[[22_megatron_memory_optimization_analysis]] 6 项，另散见 14 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。



### `TransformerConfig`（`megatron/core/transformer/transformer_config.py`，68 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `mtp_loss_scaling_factor` | `Optional[float]` | `0.1` | Weighting factor of Multi-Token Prediction (MTP) loss. We compute the average of the MTP losses across all depths, and multiply it the scaling factor to obta… | `:81` |
| `mtp_use_repeated_layer` | `bool` | `False` | Use a single MTP layer repeatedly instead of multiple separate layers. | `:88` |
| `mtp_hybrid_override_pattern` | `Optional[str]` | `None` | DEPRECATED: Use unified hybrid_layer_pattern instead. Legacy argument for loading old checkpoints. Force a specific hybrid layer pattern for MTP layers. | `:96` |
| `num_layers_in_first_pipeline_stage` | `Optional[int]` | `None` | Number of transformer layers on first pipeline stage. None implies equal layer division across PP ranks. | `:102` |
| `num_layers_in_last_pipeline_stage` | `Optional[int]` | `None` | Number of transformer layers on last pipeline stage. None implies equal layer division across PP ranks. | `:106` |
| `account_for_embedding_in_pipeline_split` | `bool` | `False` | If set, the embedding layer will be treated as a standard transformer layer in the context of partition and placement for pipeline parallelism. | `:136` |
| `account_for_loss_in_pipeline_split` | `bool` | `False` | If set, the loss layer will be treated as a standard transformer layer in the context of partition and placement for pipeline parallelism. | `:140` |
| `attention_backend` | `AttnBackend` | `AttnBackend.auto` | Attention backend to run. By default we let transformer engine decide the best backend to run (except in the case of local). If attention backend is local we… | `:150` |
| `softmax_scale` | `Optional[float]` | `None` | Softmax scale for attention scaling. | `:156` |
| `softmax_type` | `Literal['vanilla', 'off-by-one', 'learnable']` | `'vanilla'` | Applies modified softmax from https://www.evanmiller.org/attention-is-off-by-one.html. Supports both TE FusedAttention and local unfused attention. Supports … | `:159` |
| `kv_channels` | `Optional[int]` | `None` | Projection weights dimension in multi-head attention. This is set to hidden_size // num_attention_heads if not provided. | `:173` |
| `hidden_dropout` | `float` | `0.1` | Dropout probability for transformer hidden state. | `:177` |
| `fp32_residual_connection` | `bool` | `False` | If true, move residual connections to fp32. | `:183` |
| `apply_residual_connection_post_layernorm` | `bool` | `False` | If True, uses the original BERT residule connection ordering. | `:187` |
| `layernorm_epsilon` | `float` | `field(default=1e-05, metadata={'argpa…` | Epsilon value for any LayerNorm/RMSNorm operations. | `:190` |
| `layernorm_zero_centered_gamma` | `bool` | `field(default=False, metadata={'argpa…` | If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves numerical stability. | `:195` |
| `add_bias_linear` | `bool` | `field(default=True, metadata={'argpar…` | Include/exclude a bias term in all linear layers (QKV projections, after core attention, and two in MLP layer). | `:201` |
| `add_qkv_bias` | `bool` | `False` | Add a bias term only for QKV projections. | `:207` |
| `activation_func_fp8_input_store` | `bool` | `False` | Store the input of MLP activation function in FP8 for backprop to save memory. The stored input is casted back to the original precision before backprop comp… | `:216` |
| `window_size` | `Optional[Tuple[int, int]]` | `None` | If not None, then will use sliding window attention. The size of the window is specified by the numbers inside the tuple; -1 is special value meaning "infini… | `:236` |
| `window_attn_skip_freq` | `Optional[Union[int, List[int]]]` | `None` | Frequency of full attention layers among sliding window attention layers. Accepts either: - An integer N: Represents a (N-1):1 ratio, one full attention laye… | `:240` |
| `normalization` | `Literal['LayerNorm', 'RMSNorm']` | `'LayerNorm'` | Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`. | `:245` |
| `qk_layernorm` | `bool` | `False` | Whether to apply `normalization` type of normalization to the query and key embeddings. | `:248` |
| `qk_l2_norm` | `bool` | `False` | Whether to apply llama 4-style qk L2 norm. | `:251` |
| `qk_clip_alpha` | `float` | `0.5` | The balancing alpha for qk-clip. Q = Q * (eta ** alpha) | `:257` |
| `qk_clip_threshold` | `float` | `100` | The balancing threshold for qk-clip. eta = min(threshold / max_attention_logits, 1.0) | `:260` |
| `gated_attention_proj_granularity` | `Literal['elementwise', 'headwise']` | `'elementwise'` | Projection granularity for `attention_output_gate`. `elementwise` projects one gate per attention output element. `headwise` projects one scalar gate per att… | `:275` |
| `rotary_base_per_layer` | `Optional[List[float]]` | `None` | Per-layer RoPE theta values. Length must equal num_layers. When set, each SelfAttention layer creates its own RotaryEmbedding with the corresponding base; th… | `:281` |
| `no_rope_freq` | `Optional[Union[int, List[int]]]` | `None` | Controls which layers perform Rotary Position Embedding (RoPE). Accepts either: An integer N: Creates a pattern where RoPE is skipped every N-1 layers. For e… | `:307` |
| `experimental_attention_variant_loss_scale_func` | `Optional[Callable[[torch.Tensor], None]]` | `None` | Optional hook for experimental attention variants to receive the main loss scale. | `:334` |
| `dsa_indexer_n_heads` | `Optional[int]` | `None` | Number of DSA indexer heads. | `:340` |
| `dsa_indexer_head_dim` | `Optional[int]` | `None` | Dimension per DSA indexer head. | `:343` |
| `dsa_indexer_skip_topk_offset` | `int` | `0` | Layer offset for DSA cross-layer top-k sharing. | `:353` |
| `dsa_indexer_rope_interleaved` | `bool` | `False` | Whether DSA indexer RoPE should use MLA-style interleaving. | `:366` |
| `dsa_indexer_rotate_activation` | `bool` | `True` | Whether DSA indexer should apply Hadamard rotation before scoring. | `:369` |
| `dsa_indexer_scoring_relu` | `bool` | `True` | Whether DSA indexer should apply ReLU to q@k scores before weighting. | `:372` |
| `dsa_indexer_k_norm_epsilon` | `Optional[float]` | `None` | Optional epsilon override for the DSA indexer key LayerNorm. | `:375` |
| `dsa_indexer_k_norm_fp32` | `bool` | `False` | Whether DSA indexer key LayerNorm should run on fp32 inputs. | `:378` |
| `dsa_indexer_weights_proj_use_quantization` | `bool` | `True` | Whether `DSAIndexer` weights projection follows the enclosing FP8/FP4 quantization context. Disable this to keep the projection parameter outside FP8/FP4; `d… | `:381` |
| `dsa_indexer_weights_proj_output_dtype` | `Literal['bf16', 'fp32']` | `'bf16'` | Output dtype of the `DSAIndexer` weights projection. BF16 preserves the existing path. FP32 uses a true FP32-output projection and is not compatible with the… | `:387` |
| `linear_attention_type` | `Optional[str]` | `None` | Type of linear attention to use. Deprecated. Use experimental_attention_variant instead. | `:417` |
| `linear_attention_freq` | `Optional[Union[int, List[int]]]` | `None` | Frequency between LA (linear attention) layers and SDPA (scaled dot-product attention) layers. Accepts either: - An integer N: Represents a (N-1):N ratio, me… | `:420` |
| `linear_conv_kernel_dim` | `Optional[int]` | `4` | Conv kernel dimension for the gated delta net. | `:427` |
| `linear_key_head_dim` | `Optional[int]` | `128` | Query and key head dimension for the gated delta net. | `:430` |
| `linear_value_head_dim` | `Optional[int]` | `128` | Value and gate head dimension for the gated delta net. | `:433` |
| `linear_num_key_heads` | `Optional[int]` | `16` | Number of query and key heads for the gated delta net. | `:436` |
| `linear_num_value_heads` | `Optional[int]` | `32` | Number of value and gate heads for the gated delta net. | `:439` |
| `kda_safe_gate` | `bool` | `False` | Whether the KDA kernel should use bounded gate values. | `:442` |
| `kda_lower_bound` | `Optional[float]` | `None` | Optional lower bound for KDA's bounded gate values. | `:445` |
| `init_method_std` | `float` | `0.02` | Standard deviation of the zero mean normal for the default initialization method, not used if init_method and output_layer_init_method are provided. | `:471` |
| `embedding_init_method` | `Optional[Callable]` | `None` | Method to initialize weights of the embedding layer. If None, will be set as described in init_method above. | `:475` |
| `embedding_init_method_std` | `Optional[float]` | `None` | Standard deviation of the zero mean normal for the default initialization method for the embedding layer. If None, will be set to init_method_std. Setting th… | `:481` |
| `num_residual_streams` | `int` | `4` | Number of residual streams (n in paper). | `:1257` |
| `mhc_sinkhorn_iterations` | `int` | `20` | Number of Sinkhorn-Knopp iterations for doubly stochastic projection. | `:1260` |
| `mhc_init_gating_factor` | `float` | `0.01` | Initial value of Gating Factor (alpha in paper). | `:1263` |
| `mhc_recompute_layer_num` | `Optional[int]` | `None` | Number of layers in each mHC recompute group. Layers are grouped in their local transformer-block order. The last layer in each group leaves its final MLP BD… | `:1280` |
| `mhc_recompute_attn_cuda_graph_split` | `bool` | `False` | Opt into the attention-only Transformer Engine CUDA Graph split for mHC recompute. Off by default, in which case an mHC layer captures the same range as any … | `:1296` |
| `use_te_activation_func` | `bool` | `False` | Whether to use ffn activation functions implemented by TransformerEngine | `:1340` |
| `mrope_section` | `Optional[List[int]]` | `None` | Multimodal rope section is for channel dimension of temporal, height and width in rope calculation. | `:1393` |
| `mrope_interleaved` | `bool` | `False` | When True, use the interleaved T/H/W MRoPE layout (Qwen3.5-VL style) where H freqs occupy stride-3 positions {1,4,7,...} and W freqs occupy {2,5,8,...}. When… | `:1397` |
| `mamba_state_dim` | `int` | `128` | The dimensionality of the state representation in Mamba layers. | `:1406` |
| `mamba_head_dim` | `int` | `64` | The dimensionality of the heads in the Mamba layers. | `:1409` |
| `mamba_num_groups` | `int` | `8` | The number of groups used in Mamba layers. | `:1412` |
| `mamba_num_heads` | `Optional[int]` | `None` | The number of heads used in Mamba layers. If None, the number of heads will be hidden_size * expand // mamba_head_dim. | `:1415` |
| `mamba_training_ssm_states_dtype` | `Optional[torch.dtype]` | `None` | dtype of the materialized inter-chunk SSM states in Mamba training forwards and backwards. None causes the states to follow the activation dtype. | `:1419` |
| `use_mamba_mem_eff_path` | `bool` | `field(default=True, metadata={'argpar…` | Controls usage of the memory efficient path for Mamba layers. | `:1423` |
| `mlp_chunks_for_training` | `int` | `1` | The number of chunks along the sequence dimension to use for MLP computation during training. | `:1431` |
| `heterogeneous_block_specs` | `bool` | `False` | Whether to use heterogeneous block specs (nemotron-nas architecture). | `:1435` |

> 该类共 266 个字段，本表收 68 项；其余 198 项已在别处归属：主要归 [[14_megatron_ep_analysis]] 38 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 38 项、[[21_megatron_fusion_operators_analysis]] 26 项、本页他处 24 项，另散见 20 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。

> **一处编号提醒**：`num_layers_in_first_pipeline_stage` / `num_layers_in_last_pipeline_stage` 与`account_for_*_in_pipeline_split` 虽在 `# model architecture` 段里，但语义属流水线切分，机制见 [[15_megatron_pp_schedulers_analysis]]；本页只登记它们的契约。

## Related Pages

- [[14_megatron_ep_analysis]] · [[13_megatron_cp_analysis]] · [[12_megatron_tp_analysis]] · [[18_megatron_recompute_analysis]] · [[31_megatron_inference_engine_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]


