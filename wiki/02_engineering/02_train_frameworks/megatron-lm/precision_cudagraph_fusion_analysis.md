# Megatron-LM FP8 精度 · CUDA Graph · 算子融合 深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/fp8_utils.py`、`fp4_utils.py`、`enums.py`、`transformer/cuda_graphs.py`、`full_cuda_graph.py`、`fusions/`、`num_microbatches_calculator.py`
> 配套阅读:五份并行文档 + `recompute_analysis.md` + `optimizer_internals_analysis.md`
> 定位:"第二层补遗"第③份。这三块是与并行轴正交的**性能基建** —— 不改变并行策略,而是在精度、内核调度、内核形态三个层面榨吞吐与显存。

---

## 0. 总览

| 主题 | 解决的瓶颈 | 一句话 |
|------|-----------|--------|
| **FP8/FP4 低精度** | 显存 + 算力 + 通信 | 用 8/4 bit 做 GEMM 与通信,三重收益 |
| **CUDA Graph** | CPU 内核启动开销 | 把一串 kernel 录成一张图,一次重放 |
| **算子融合** | 内核启动 + HBM 读写 | 多个小算子合成一个 kernel |

README(MoE)把 MoE 训练的瓶颈归为三堵墙:**显存墙、通信墙、计算效率墙**。本文这三块分别针对它们。

---

## 1. FP8 / FP4 低精度训练

### 1.1 动机

bf16 已是主流,但 Hopper/Blackwell 的 Tensor Core 对 **FP8** 还能再快一档,且 8 bit 比 16 bit 再省一半。MoE 大模型尤其吃这个 —— GEMM 多、激活大、EP 通信重。FP4 更激进(Blackwell)。

注意:**实际的 FP8 GEMM 内核在 TransformerEngine 里**,Megatron 侧 `fp8_utils.py` 负责**选 recipe、建量化上下文、管 FP8 张量**。

### 1.2 四种 FP8 recipe(`enums.py:12` `Fp8Recipe`)

```python
class Fp8Recipe(str, Enum):
    delayed   = "delayed"      # 延迟缩放:用 amax 历史窗口定 scale
    tensorwise= "tensorwise"   # 整张量一个 scale(per-tensor)
    blockwise = "blockwise"    # 分块缩放:激活 1×128、权重 128×128
    mxfp8     = "mxfp8"        # 微缩放:1×32 一组,E8M0 scale
    # 还有 custom
```

| recipe | 缩放粒度 | 平台 | 定位 |
|--------|---------|------|------|
| delayed | per-tensor + amax 历史 | Hopper | 早期方案,需维护 amax history;`recompute`/A2A-overlap 有兼容限制 |
| tensorwise | 整张量 | Hopper, Blackwell | 保守,初期试验 |
| **blockwise** | 1×128 / 128×128 | Hopper | **生产首选**,DeepSeek-V3 级别已验证 |
| mxfp8 | 1×32 微块 | Blackwell | GB200 原生硬件支持 |

缩放粒度越细(blockwise/mxfp8),越能容忍张量内数值动态范围差异,精度越稳。`get_fp8_recipe`(`fp8_utils.py:536`)按配置返回 recipe,`get_fp8_context` 把一段 GEMM 包进对应的量化上下文。`get_fp8_align_size`(`:168`)给出 mxfp8 需要的对齐尺寸(故有 `--moe-router-padding-for-fp8`)。

### 1.3 FP8 的三重收益

FP8 同时砸向三堵墙:

| 墙 | FP8 收益 | 机制 |
|----|---------|------|
| **显存** | 激活省 ~50% | 线性层输入存 FP8 而非 bf16;FP8 primary weight 免 bf16 拷贝 |
| **算力** | GEMM 更快 | Hopper/Blackwell 的 FP8 Tensor Core 比 bf16 快 |
| **通信** | EP dispatch 省 50% | token 以 FP8 做 all-to-all(`ep_analysis.md`);参数 all-gather FP8(`--fp8-param-gather`) |

### 1.4 与并行轴的交织

FP8 不是孤立特性,它**贯穿前面所有文档**:
- **TP**:`ColumnParallel`/`RowParallel` 的 GEMM 走 FP8;`fp8_utils.py` 有 `is_column_parallel_linear`/`is_row_parallel_linear` 判定。
- **EP**:dispatch 的 A2A 用 FP8,通信量砍半(`ep_analysis.md`);`combined_1f1b` 的 fp8 上下文(`pp_schedulers_analysis.md` 调度器⑤)。
- **DP/ZeRO**:`--fp8-param-gather` 让参数 all-gather 走 FP8(`quantize_param_shard`、`post_all_gather_processing`,`fp8_utils.py:484/499`)。
- **重计算**:fp8 下用 `te_checkpoint`(`recompute_analysis.md` §3.4);delayed scaling 与某些 selective 重计算互斥。
- **首尾层**:`is_first_last_bf16_layer`(`:513`)—— 首尾层常保留 bf16(对精度最敏感)。

FP4(`fp4_utils.py`、`Fp4Recipe`)同理,更激进,Blackwell 专属。

---

## 2. CUDA Graph

### 2.1 动机:CPU 内核启动开销

GPU 执行每个 kernel 前,CPU 要先"启动"它(launch)。当模型由**大量小 kernel**组成 —— 尤其细粒度 MoE(几百个小专家 GEMM、路由、置换)—— CPU 来不及把 kernel 一个个塞给 GPU,**GPU 在 kernel 之间出现空隙、干等 CPU**。Nsight 时间线上表现为 kernel 间的缝隙。

**CUDA Graph**:把一段固定的 kernel 序列**录制(capture)成一张图**,之后整张图**一次性重放(replay)**,绕过逐 kernel 的 CPU 启动 —— 消除启动开销、消除 CPU 抖动(配合 `--manual-gc` 更稳)。

### 2.2 三种实现(`--cuda-graph-impl`)

| impl | 粒度 | 实现 |
|------|------|------|
| `local` | 每层一张图 | MCore 自带的图管理器(`cuda_graphs.py`) |
| `transformer_engine` | 每层一张图 | 用 TE 的 `make_graphed_callables()` |
| `full_iteration` | 整个前向+反向一张图 | 整步录成单图,消除最多 |

`cuda_graphs.py` 的机制:`_CudaGraphRunner` 包住一个可图化的模块;`_CudagraphGlobalRecord`(`:320`)记录所有 runner 的创建顺序;`create_cudagraphs`(`:478`)在**第一个训练步**真正录图(被 `pp_schedulers_analysis.md` 的 `schedules.py` 在步末调用 —— 前五份文档里 `schedules.py` 出现的 `create_cudagraphs()` 就是它)。`TensorReusePool`(`:161`)在图之间复用张量缓冲。

### 2.3 约束:动态形状

CUDA Graph 要求**每次重放的张量形状、地址固定**。问题:
- dropless MoE 的每个专家收到的 token 数**随路由动态变化** → MoE 层形状不定 → **无法图化**。
- 解法:① 设 `--moe-expert-capacity-factor` + `--moe-pad-expert-input-to-capacity` 让 MoE 形状静态(代价:丢/填 token);② 或只图化 attention(`--cuda-graph-modules attn`),MoE 层不动。

`cuda_graphs.py` 还要处理 RNG 状态(`_ensure_generator_state_is_cudagraph_safe`,`:293`)—— dropout 的随机数生成器在图重放下必须可控。与 VPP 配合时还要判定层属于哪个 VP chunk(`_determine_if_first_last_layer_of_this_vp_chunk`,`:249`)。

推理另有 `--inference-cuda-graph-scope=layer|block`。

---

## 3. 算子融合(`fusions/`)

### 3.1 动机:内核启动 + HBM 往返

两个相邻算子(如 `bias add` 后接 `GeLU`)若各是一个 kernel:① 两次启动开销;② 中间结果要**写回 HBM 再读出**(显存带宽是稀缺资源)。**融合**把它们写成**一个 kernel**:中间结果只在寄存器/共享内存里流转,不落 HBM,启动也只一次。

收益:更少 kernel 启动 + 更少 HBM 流量 → 直接提速,尤其对 memory-bound 的逐元素算子。

### 3.2 `fusions/` 清单

| 融合 kernel | 融合了什么 |
|------------|-----------|
| `fused_bias_gelu` / `fused_bias_geglu` / `fused_bias_swiglu` | bias 加法 + 激活函数(GeLU/GeGLU/SwiGLU) |
| `fused_bias_dropout` | bias + dropout(+ 残差) |
| `fused_softmax` | scale + mask + softmax(attention) |
| `fused_layer_norm` | LayerNorm / RMSNorm |
| `fused_cross_entropy` / `fused_linear_cross_entropy` | 交叉熵(后者连输出投影一起融) |
| `fused_pad_routing_map` | MoE 路由图的 FP8 对齐填充 |
| `fused_indices_converter` | MoE 路由索引转换 |
| `fused_mla_yarn_rope_apply` | MLA 的 YaRN RoPE 应用 |
| `fused_mhc_kernels` | hyper connections |
| `fused_weighted_squared_relu` | 加权 squared-ReLU 激活 |

### 3.3 MoE 专用融合(README)

MoE 是"小算子最多"的地方,有三个关键融合开关:
- `--moe-grouped-gemm`:把 `E/e` 个专家的 GEMM 批成**一次 grouped GEMM**(`ep_analysis.md` §2.4)。
- `--moe-router-fusion`:路由投影 + top-k + softmax + aux loss 融成少数 kernel。
- `--moe-permute-fusion`:token 置换/反置换融合。

它们直接对应 README"计算效率墙"的解法。

---

## 4. 附:`num_microbatches_calculator`

`num_microbatches_calculator.py`:由 `global_batch_size`、`micro_batch_size`、`data_parallel_size` 算出每步的 microbatch 数:

```
num_microbatches = global_batch_size / (micro_batch_size · data_parallel_size)
```

这个 `num_microbatches` 正是 `pp_schedulers_analysis.md` 里反复出现的 `m`(梯度累加步数 / 流水线 microbatch 数)。它还支持 **batch size ramp-up**(训练初期用小 global batch、逐步增大)。是连接"数据并行配置"与"PP 调度"的小齿轮。

---

## 5. 小结

- **FP8/FP4**:用低精度做 GEMM 与通信,**三重收益**(显存 ~50%、算力更快、EP/DP 通信砍半);4 种 recipe(delayed/tensorwise/blockwise/mxfp8),生产首选 **blockwise**(Hopper)/ **mxfp8**(Blackwell);实际内核在 TE,Megatron 管 recipe 与上下文;**贯穿所有并行轴**。
- **CUDA Graph**:把 kernel 序列录成图、一次重放,消除 CPU 启动开销与抖动;3 种粒度(local / transformer_engine / full_iteration);**动态形状是死穴** —— dropless MoE 需固定容量或只图化 attention。
- **算子融合**:多算子合成一 kernel,省启动 + 省 HBM 往返;`fusions/` 一堆逐元素/归一化/交叉熵融合;MoE 三件套 `grouped-gemm` / `router-fusion` / `permute-fusion`。
- 三者都与并行轴**正交**,是 kernel/精度层面的提速,叠加在并行策略之上。

至此"第二层补遗"3 份文档全部完成:① 激活重计算、② 优化器内部、③ FP8 精度 + CUDA Graph + 算子融合。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。FP8/FP4 的 GEMM 内核位于 TransformerEngine。配套文档:五份并行分析 + `recompute_analysis.md` + `optimizer_internals_analysis.md`。*

## Related Pages

- [[ep_analysis]] · [[recompute_analysis]] · [[optimizer_internals_analysis]]
- [[megatron_fusion_operators_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
