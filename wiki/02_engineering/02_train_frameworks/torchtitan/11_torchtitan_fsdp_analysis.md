---
title: "TorchTitan FSDP：参数存储平面、分组策略与全局 token 归一化"
---

# TorchTitan FSDP：参数存储平面、分组策略与全局 token 归一化

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：TorchTitan 的 FSDP 接线已经不是“DP degree 大于 1 才包模型”的条件分支，而是模型并行化末端的**参数存储适配层**：它把逻辑 `dp/cp/tp` 值布局投影到 dense/sparse storage mesh，按 TransformerBlock 建立 FSDP 单元，对 tied weights 与 MoE expert 参数改写分组/mesh，并关闭 FSDP 自带的 world-size 梯度除法，把归一化交给全局有效 token 数。degree=1 时这层仍负责 mixed precision；PP、EP、CP 则分别改变 reshard、prefetch 和 storage shard 规则。
>
> 本页聚焦 TorchTitan 如何配置和组合 PyTorch FSDP2。FSDP2 内部 all-gather/reduce-scatter、stream 与内存生命周期见 [[20_torchtitan_fsdp_prefetch_overlap_memory_analysis]]；HSDP 反向双层规约见 [[21_torchtitan_hsdp_backward_overlap_analysis]]；编译器内联 collective 的替代实现见 [[25_torchtitan_simple_fsdp_analysis]]。

---

## 1. Overview：FSDP 是 storage adapter，不只是一个 DP 开关

“FSDP 只在参数要跨 DP rank 切分时启用”是当前 TorchTitan 最危险的旧心智模型。Llama3 的并行化顺序是 Module layout → AC → compile → FSDP，而且共享入口被无条件调用；源码注释明确写明 shard degree=1 时 all-gather 是 no-op，但 `MixedPrecisionPolicy` 仍要安装（`torchtitan/models/llama3/parallelize.py:40`、`torchtitan/models/llama3/parallelize.py:46`、`torchtitan/models/llama3/parallelize.py:49`、`torchtitan/models/llama3/parallelize.py:57`）。

当前公共入口 `apply_fsdp_to_decoder()` 同时服务 dense 与 MoE decoder；dense 路径只是 `ep_degree=1`、无 `moe_enabled` block 的特例（`torchtitan/distributed/fsdp.py:168`、`torchtitan/distributed/fsdp.py:183`、`torchtitan/distributed/fsdp.py:185`）。2026-06-08 的提交 `d92336fee` 把此前分散在 Llama3/Llama4 的两份接线合并，提交正文给出的理由是 MoE 版本是 dense 版本的严格超集，继续复制会使新模型和修复发生分叉。

```text
model config / Module ShardingConfig
                |
                v
      model.parallelize(parallel_dims)
                |
          AC -> compile
                |
                v
   resolve dense/sparse storage mesh
                |
                v
 apply_fsdp_to_decoder(model, policies...)
      |          |             |
      |          |             +-- EP: per-param mesh + explicit prefetch
      |          +-- PP: reshard policy
      +-- block/root units, MP, offload, grad semantics
```

### Quick Start：从哪里进入源码

先从具体模型的 `parallelize.py` 看顺序，再进入公共 FSDP adapter。Dense Llama 解析一张 dense storage mesh（`torchtitan/models/llama3/parallelize.py:59`、`torchtitan/models/llama3/parallelize.py:68`）；Qwen3 MoE 额外解析 sparse storage mesh并传入 EP degree（`torchtitan/models/qwen3/parallelize.py:66`、`torchtitan/models/qwen3/parallelize.py:85`、`torchtitan/models/qwen3/parallelize.py:93`）。公共 adapter 的关键阶段如下：

| 阶段 | 当前入口 | 负责的选择 |
|---|---|---|
| storage mesh | `resolve_fsdp_mesh()` / `resolve_sparse_fsdp_mesh()` | dense 的 `dp_shard+cp`、sparse 的 `efsdp`、可选 replicate |
| policy | `apply_fsdp_to_decoder()` 开头 | mixed precision、CPU offload、reshard |
| unit | embedding/output、每个 block、root | all-gather 粒度与 tied weight 所有权 |
| MoE | `shard_placement_fn` | expert/non-expert 的 placement 与 mesh |
| post wiring | symm-mem、disable grad divide、prefetch | 通信实现、token 归一化、EP 时序 |

---

## 2. Storage mesh：为什么值布局和参数存储不能共用一张解释

### ① 背景/问题

在 `spmd_types` 路线中，前后向的 dense 参数逻辑上是 R@DP、R@CP，并可沿 TP 切某个 tensor dim；但长期参数存储又希望沿 `dp_shard` 和 CP 分散。如果把前向逻辑 layout 直接当成 FSDP storage layout，CP rank 会永久复制参数；如果只给 FSDP 一个旧式名为 `fsdp` 的一维 mesh，又会丢失 HSDP replicate 轴、CP shard 轴和 TP 轴在完整 mesh 中的相对关系。

### ② 为什么这么设计

**选中的路线**是给 `fully_shard()` 完整 storage mesh，再用 `DataParallelMeshDims` 明确指出其中哪些轴属于 shard/replicate；**明显替代方案**是按轴名猜 DP，或预先 flatten 成一张孤立 1D mesh。决定性标准是新轴加入后不能被 FSDP 静默误分类。公共入口的参数说明明确说命名契约不足以安全推断，所以显式传 `dp_mesh_dims`（`torchtitan/distributed/fsdp.py:210`、`torchtitan/distributed/fsdp.py:213`、`torchtitan/distributed/fsdp.py:215`）。

这不是纯理论选择。提交 `57cfb2745` 记录了 Kimi 2.7 在 `spmd_types` 下请求 legacy `fsdp` 轴而初始化失败；修复就是显式告诉 FSDP dense shard 轴是 `dp_shard`、sparse shard 轴是 `efsdp`。

### ③ 实现思路与细节

Dense storage 候选轴固定为 `[dp_replicate, dp_shard, cp, tp]`（`torchtitan/distributed/fsdp.py:28`）。resolver 先取得启用轴构成的 storage mesh，再把 `dp_shard` 设为基础 shard 轴；CP 开启时把 `cp` 追加到同一个 shard tuple，HSDP 开启时把 `dp_replicate` 设为 replicate 轴（`torchtitan/distributed/fsdp.py:32`、`torchtitan/distributed/fsdp.py:44`、`torchtitan/distributed/fsdp.py:54`、`torchtitan/distributed/fsdp.py:60`）。TP 留在完整 storage mesh 中但不被声明为 data-parallel axis，交给参数自身的 TP layout 处理。

Sparse storage 则使用 `[dp_replicate, efsdp, ep]`；只有 EP 开启才返回 mesh，并固定 `efsdp` 为 FSDP shard 轴（`torchtitan/distributed/fsdp.py:29`、`torchtitan/distributed/fsdp.py:65`、`torchtitan/distributed/fsdp.py:77`、`torchtitan/distributed/fsdp.py:82`）。因此 dense 与 routed expert 参数可在同一 block 中拥有不同 storage group，而 world rank 数不变。

`partial_dtensor` 保留旧兼容路径：模型 adapter 直接取 `fsdp` 或 `[dp_replicate, fsdp]` 子 mesh，并把 `dp_mesh_dims=None`（`torchtitan/models/llama3/parallelize.py:61`、`torchtitan/models/llama3/parallelize.py:65`）。这说明 `spmd_types` 是默认而非唯一后端，不能把 storage resolver 写成所有后端共同调用。

### ④ 约束/代价/失败边界

- `resolve_fsdp_mesh()` / sparse sibling 断言当前后端必须是 `spmd_types`（`torchtitan/distributed/fsdp.py:41`、`torchtitan/distributed/fsdp.py:74`）；兼容后端必须走模型 adapter 的旧 mesh 查询。
- dense mesh 即使 size=1 仍保留；此时参数没有有效 SPMD annotation，resolver 刻意不传 `DataParallelMeshDims`，避免让 FSDP把不存在的注解翻译成 DTensor（`torchtitan/distributed/fsdp.py:47`、`torchtitan/distributed/fsdp.py:52`）。
- CP 被纳入 storage shard 不等于参数在 forward 的 CP 逻辑上是 shard。前者是长期状态所有权，后者仍是 R@CP；两平面的对应关系见 [[13_torchtitan_cp_analysis]]。
- sparse resolver 只证明 rank/mesh 可用，不证明 expert 数、grouped GEMM layout 或 dispatcher backend 合法；这些 guard 属于模型/EP 配置。

---

## 3. FSDP unit：为什么按 block 分组，还要单独处理 tied weights 与尾层

### ① 背景/问题

整模型一个 FSDP unit 会让一次 forward 需要同时 all-gather 全部参数；每个 Linear 一个 unit 又会产生大量小 collective，且把模型结构细节泄漏给框架。decoder 还存在两个破坏简单“逐层包裹”的例外：embedding 与 lm_head 可能共享同一 Parameter；最后的 norm/lm_head 在反向开始时会立即再次使用。

### ② 为什么这么设计

**选中的路线**是以 TransformerBlock 为主单元，embedding/output/root 作为边界单元；**替代方案**是整模型或逐算子切分。决定性标准是用稳定的模型结构边界控制完整参数驻留窗口，同时保留 FSDP root 管理未被子单元认领的参数。当前公共 adapter 先处理 embedding/output，再逐 block，最后对 root 调用 `fully_shard()`（`torchtitan/distributed/fsdp.py:238`、`torchtitan/distributed/fsdp.py:267`、`torchtitan/distributed/fsdp.py:368`）。

### ③ 实现思路与细节

无权重绑定时，embedding 是独立 unit；norm 与 lm_head 被合在一个 unit（`torchtitan/distributed/fsdp.py:251`、`torchtitan/distributed/fsdp.py:260`）。每个 dense TransformerBlock 再单独 `fully_shard()`（`torchtitan/distributed/fsdp.py:361`）。最后 root wrapper 收拢剩余状态并建立统一 FSDP 根。

有权重绑定时，embedding、norm、lm_head 被放进同一个 module list 交给一次 `fully_shard()`；源码直接给出的原因是避免 shared parameter 被重复 all-gather（`torchtitan/distributed/fsdp.py:238`、`torchtitan/distributed/fsdp.py:240`、`torchtitan/distributed/fsdp.py:246`）。这不是一般性能启发式，而是共享参数所有权约束。

视觉编码器走另一种粒度：作为一个整体 FSDP unit，并要求先于 decoder 包裹。源码给出的判据是 vision encoder 相对 decoder 较小，一次 all-gather 比逐层切更高效（`torchtitan/distributed/fsdp.py:139`、`torchtitan/distributed/fsdp.py:149`、`torchtitan/distributed/fsdp.py:151`）。所以“一个 block 一个 unit”只适用于 decoder 主体，不是全仓库铁律。

### ④ 约束/代价/失败边界

- `model.layers` 必须提供稳定的 block 集合；非 decoder 架构应有自己的 adapter，不能强行调用公共 decoder helper。
- tied parameter 必须在同一个 FSDP unit 中被识别；复制两个 tensor 值不等于参数共享，也不会触发这里的保护。
- 最后 `[norm, lm_head]` 默认不 reshard，因为 FSDP 反向会立即预取；显式 policy=`always` 才覆盖该优化（`torchtitan/distributed/fsdp.py:258`、`torchtitan/distributed/fsdp.py:264`）。这会增加驻留但减少一次紧邻通信。
- block 粒度是当前 decoder adapter 的工程折中，不等于对任何模型/硬件都最优；源码没有自动搜索 unit 边界。

---

## 4. Mixed precision、CPU offload 与 reshard：为什么 degree=1 仍走同一入口

### ① 背景/问题

若 mixed precision、offload 与 reshard 分散成独立 wrapper，degree=1、FSDP、HSDP、CP-only 和 PP 会拥有不同构造顺序，组合测试呈指数增长。尤其 PP 的多个 microbatch 会反复进入同一 stage；默认在每个 forward 后释放完整参数会导致每个 microbatch 重新 all-gather。

### ② 为什么这么设计

**选中的路线**是由 FSDP adapter 统一安装 `MixedPrecisionPolicy`、可选 CPU offload 和 reshard policy；**替代方案**是只有发生真实参数分片时才调用 FSDP，其他情况走 autocast/独立 offload。决定性标准是保持模型并行化顺序和参数 dtype 语义一致。Llama 的注释明确说 degree=1 仍调用 FSDP 是为了 mixed precision（`torchtitan/models/llama3/parallelize.py:57`、`torchtitan/models/llama3/parallelize.py:58`）。

### ③ 实现思路与细节

共享入口建立 `MixedPrecisionPolicy(param_dtype, reduce_dtype, cast_forward_inputs=False)`（`torchtitan/distributed/fsdp.py:223`、`torchtitan/distributed/fsdp.py:226`）。训练配置把基础 `dtype` 与 FSDP mixed-precision param/reduce dtype 分开：基础 dtype 可以是 bf16/fp32，mixed param 默认为 bf16，reduce 只允许 fp32（`torchtitan/config/configs.py:89`、`torchtitan/config/configs.py:96`、`torchtitan/config/configs.py:104`）。CPU offload 开启时向同一 `fully_shard` config 注入 `CPUOffloadPolicy`（`torchtitan/distributed/fsdp.py:228`、`torchtitan/distributed/fsdp.py:231`）。

reshard 有 `always/never/default` 三值。默认在 PP 关闭时为 true，在 PP 开启时为 false；源码直接说明 PP 默认保留完整参数是为了避免每 microbatch 昂贵且难以重叠的 all-gather（`torchtitan/distributed/fsdp.py:112`、`torchtitan/distributed/fsdp.py:124`、`torchtitan/distributed/fsdp.py:129`）。配置层把它明确描述成显存与通信的权衡（`torchtitan/config/configs.py:147`、`torchtitan/config/configs.py:149`）。

### ④ 约束/代价/失败边界

- PP 默认 `reshard_after_forward=False` 以显存换通信；长 stage 或高并发 microbatch 下可能超出显存，用户可用 `always` 改回释放优先。
- `never` 会把完整参数保持更久；它不是“更快”的无条件开关，需结合 unit 大小和显存预算判断。
- CPU offload 同时涉及参数、梯度和 optimizer state，配置注释明确属于 FSDP 功能（`torchtitan/config/configs.py:72`、`torchtitan/config/configs.py:74`）；它引入 host/device 传输，不能只按显存收益评估。
- `mixed_precision_reduce` 当前 schema 只允许 fp32；文档若列 bf16 reduce 为现行配置会误导用户（`torchtitan/config/configs.py:104`）。

---

## 5. MoE 每参数 storage：为什么一个 block 内要同时使用 dense 与 sparse mesh

### ① 背景/问题

MoE block 同时含 dense attention/router/shared 参数和 routed expert 参数。若整个 block 全部沿 dense FSDP mesh 切分，expert 参数无法复用 `efsdp × ep` 的 sparse rank 重切；若整个 block都沿 sparse mesh，dense 参数又被错误地绑定到 EP 所有权。另一个问题是 expert weight 常含 expert 维与 hidden 维：当 FSDP×EP rank 多于 expert 数时沿 expert 维切会产生 padding/空片。

### ② 为什么这么设计

**选中的路线**是仍保留一个 block-level FSDP unit，但用 `shard_placement_fn` 按 Parameter 身份返回不同 placement 和 mesh info；**替代方案**是把 routed experts 拆成完全独立 wrapper，或让整个 block只能选一张 mesh。决定性标准是既保持 block 生命周期边界，又让参数存储所有权按 dense/sparse 角色分流。提交 `d92336fee` 的正文把 MoE helper 定义为 dense helper 的超集，正是这条统一路线。

### ③ 实现思路与细节

adapter 从 `routed_experts.inner_experts` 收集 expert parameter identity（`torchtitan/distributed/fsdp.py:274`、`torchtitan/distributed/fsdp.py:278`）。它比较 `efsdp × ep`（无 EP 时使用 FSDP mesh size）与 expert 数：rank 区域大于 expert 数时改沿 dim-1 shard，否则沿 dim-0 shard（`torchtitan/distributed/fsdp.py:282`、`torchtitan/distributed/fsdp.py:288`）。

三条执行分支是：

1. EP=1 且默认 dim-0 合法：普通 block-level `fully_shard()`（`torchtitan/distributed/fsdp.py:293`、`torchtitan/distributed/fsdp.py:296`）。
2. EP=1 但需避免 expert-dim padding：只为 expert param 返回 `Shard(1)`（`torchtitan/distributed/fsdp.py:301`、`torchtitan/distributed/fsdp.py:307`）。
3. EP>1：先分别从 sparse/dense mesh 构建 FSDP mesh info，再为 expert param 返回 sparse mesh + 选定 placement，其他 param 返回 dense mesh + `Shard(0)`（`torchtitan/distributed/fsdp.py:317`、`torchtitan/distributed/fsdp.py:329`、`torchtitan/distributed/fsdp.py:339`、`torchtitan/distributed/fsdp.py:346`）。

GPU 单测固定了三个当前分支：FSDP size 8 > experts 4 时 dim-1；等于 experts 8 时 dim-0；EP2 + eFSDP4 > experts4 时仍为 dim-1（`tests/unit_tests/gpu/test_fsdp_moe_sharding.py:79`、`tests/unit_tests/gpu/test_fsdp_moe_sharding.py:87`、`tests/unit_tests/gpu/test_fsdp_moe_sharding.py:104`、`tests/unit_tests/gpu/test_fsdp_moe_sharding.py:121`）。

### ④ 约束/代价/失败边界

- EP>1 时 `edp_mesh` 必须存在，代码用 assert 而非 fallback；非法 sparse mesh 不会退化成 dense FSDP（`torchtitan/distributed/fsdp.py:282`、`torchtitan/distributed/fsdp.py:327`）。
- 当前 per-param mesh 使用 PyTorch FSDP2 内部 `_get_mesh_info`/`FSDPMeshInfo` API（`torchtitan/distributed/fsdp.py:319`、`torchtitan/distributed/fsdp.py:323`）。这是版本耦合面，升级 PyTorch 时需要专门回归。
- 分支用 Parameter identity 判断 expert 集合；重参数化或 wrapper 若生成新 Parameter，必须在 `fully_shard()` 之前完成。
- dim-1 选择只减少 expert 维不足时的 padding，并不自动证明 grouped-GEMM kernel 在任意 shape 上高效。

---

## 6. 梯度语义与 EP prefetch：为什么不能照搬 FSDP 的默认平均

### ① 背景/问题

普通 data-parallel 训练常按 rank 数平均梯度，但 TorchTitan 的 batch 可以经过 packing、ignore labels、CP、TP 和 PP；不同 microbatch 的有效 token 数不一定等于固定 batch-size 乘序列长。若 loss 已按全局有效 token 数归一化，FSDP 再按 group size 除一次，梯度尺度就会错误。EP 还会出现 device-to-host split/count 同步，CPU 驱动的隐式下一层预取可能因此迟到。

### ② 为什么这么设计

**选中的路线**是把 loss/梯度唯一归一化基准设为 `global_valid_tokens`，并关闭所有 FSDP module 的自动 gradient divide；EP 打开时再显式串起相邻 FSDP units 的 forward/backward prefetch。**替代方案**是沿用 FSDP world-size 平均并假设每 rank token 数相同，同时依赖隐式执行顺序预取。决定性标准是对 token 所有权和 host 同步都显式建模。

### ③ 实现思路与细节

Trainer 在 batch mesh 上规约 local valid-token 数（`torchtitan/trainer.py:800`、`torchtitan/trainer.py:802`）。loss 使用 sum reduction，并在 forward 内除以 `global_valid_tokens`（`torchtitan/components/loss.py:285`、`torchtitan/components/loss.py:306`、`torchtitan/components/loss.py:315`）。FSDP adapter 随后遍历所有 `FSDPModule`，把 gradient divide factor 设为 1.0（`torchtitan/distributed/fsdp.py:85`、`torchtitan/distributed/fsdp.py:97`、`torchtitan/distributed/fsdp.py:99`）。因此规约仍聚合梯度，但不再附加 rank-count 平均。

EP=1 时 adapter 保留 FSDP 的隐式 prefetch并直接返回（`torchtitan/distributed/fsdp.py:384`、`torchtitan/distributed/fsdp.py:386`）。EP>1 时，它把 embedding → blocks → norm/lm_head 的 forward 邻接关系显式写入 `set_modules_to_forward_prefetch()`，反向则按相反顺序串联（`torchtitan/distributed/fsdp.py:389`、`torchtitan/distributed/fsdp.py:393`、`torchtitan/distributed/fsdp.py:396`、`torchtitan/distributed/fsdp.py:408`、`torchtitan/distributed/fsdp.py:413`、`torchtitan/distributed/fsdp.py:416`）。源码注释直接给出原因：EP 的 D2H sync 可能干扰隐式 FSDP prefetch（`torchtitan/distributed/fsdp.py:384`）。

### ④ 约束/代价/失败边界

- 只有使用 TorchTitan 标准 loss contract、正确传入 global valid tokens 时，关闭 FSDP divide 才成立。自定义 loss 若返回 mean 或自己再归一化，会改变梯度尺度。
- `global_valid_tokens` 在 DP/batch mesh 上规约；PP 的 loss 只在 last stage 产生，TP rank则持有相同逻辑 loss。三者不能都当普通 DP 平均。
- 显式 prefetch 依赖 decoder 层顺序，适用于当前顺序模型；带条件跳层或动态图控制流的模型不能直接复用这条链。
- 专用 EP backend 若完全移除 host sync，显式链仍是合法顺序，但是否优于隐式策略需要性能测量，源码没有自动切换。

---

## 7. Symmetric memory 与当前边界：优化通信实现，不改变状态所有权

### ① 背景/问题

FSDP 的分组、mesh 与归一化决定正确性，但 collective 具体实现仍可优化。把通信优化揉进 sharding 规则会让开关改变模型语义；完全由外部环境隐式启用，又难以复现实验。

### ② 为什么这么设计

**选中的路线**是在 `fully_shard()` 完成后，遍历 FSDP modules 显式启用 symmetric-memory 通信；**替代方案**是为每个 FSDP call 复制特殊 backend 参数，或全局 monkey-patch collective。决定性标准是优化只改变通信实现，不改变 unit、mesh、placement 和梯度语义。提交 `58b034444` 把该开关接入默认 Trainer 路径，并明确不等同于 GraphTrainer 或 Async-TP 的 symmetric-memory kernel。

### ③ 实现思路与细节

`enable_fsdp_symm_mem()` 对每个 `FSDPModule` 强制 sum reduction 并调用 `set_symm_mem_for_comm()`（`torchtitan/distributed/fsdp.py:102`、`torchtitan/distributed/fsdp.py:106`、`torchtitan/distributed/fsdp.py:108`）。公共 decoder adapter 在 root wrapper 之后应用它，再关闭 gradient division（`torchtitan/distributed/fsdp.py:368`、`torchtitan/distributed/fsdp.py:370`、`torchtitan/distributed/fsdp.py:373`）。这一顺序保证新建的所有 child/root FSDP module 都被覆盖。

### ④ 约束/代价/失败边界

- NVIDIA 上配置要求 CUDA 可用且 compute capability 至少 9.0；不满足时在 config 构造阶段失败（`torchtitan/config/configs.py:272`、`torchtitan/config/configs.py:279`）。源码对 ROCm 只绕过 NVIDIA capability 比较，实际支持仍需对应环境测试。
- 开关仅遍历当前 model 中的 `FSDPModule`；之后动态新增 wrapper 不会自动继承。
- 它不改变 reshard 或 per-param mesh，也不能替代 TP/EP 的专用 collective overlap。
- 当前页面只证明接线与 guard；`58b034444` 的 GB200 benchmark 是特定模型/硬件数据，不能外推为所有配置固定加速。

### ⑤ 发展趋势（有源码锚点的推断）

当前 MoE per-param mesh 仍调用 PyTorch 私有 FSDP2 mesh-info API，而 storage axes 已通过公开 `DataParallelMeshDims` 显式化。由这个版本耦合边界可以推断后续会受益于更公开的 per-parameter mesh contract；但仓库没有给出迁移 TODO 或时间表，因此不能写成既定计划。

---

## 8. 版本纠偏与排障顺序

| 旧心智模型 | 当前事实 | 证据/原因 |
|---|---|---|
| DP shard degree=1 时不调用 FSDP | decoder 仍调用，用于 mixed precision；all-gather 可为 no-op | `torchtitan/models/llama3/parallelize.py:57` |
| 一张 `fsdp` mesh 适用于所有后端 | `spmd_types` 用完整 storage mesh + 显式 DP axes；旧后端才取 `fsdp` 子 mesh | `torchtitan/distributed/fsdp.py:28`、`torchtitan/models/llama3/parallelize.py:61` |
| CP 只切激活，不参与参数存储 | CP 开启时加入 dense FSDP shard axes | `torchtitan/distributed/fsdp.py:54` |
| dense 与 MoE 有两套 adapter | 已合并为公共 decoder helper；MoE 是 per-param mesh 的超集 | `torchtitan/distributed/fsdp.py:183` |
| 所有 expert weight 都沿 expert dim shard | rank 区域大于 expert 数时改为 `Shard(1)` | `torchtitan/distributed/fsdp.py:288` |
| FSDP 默认 world-size 平均可直接使用 | 被禁用；loss 按 global valid tokens 归一化 | `torchtitan/distributed/fsdp.py:85`、`torchtitan/components/loss.py:315` |
| PP 每个 forward 后默认释放完整参数 | 默认 PP 不 reshard，避免每 microbatch all-gather | `torchtitan/distributed/fsdp.py:129` |
| EP 仍完全依赖隐式 FSDP prefetch | EP>1 显式连接前后向相邻 units | `torchtitan/distributed/fsdp.py:384` |

排障应按以下顺序，而不是从 NCCL trace 开始：

1. 确认当前后端走的是显式 storage mesh 还是 legacy `fsdp` 子 mesh。
2. 区分 dense param 与 routed expert param，验证其 `dp_mesh`/`edp_mesh` 和 shard dim。
3. 检查 tied parameter 是否落在同一个 FSDP unit，root 是否最后包装。
4. 对 PP 显存/通信问题先核对 reshard policy，而不是只调 microbatch 数。
5. 对梯度尺度先确认 loss 是 sum/global-valid-token contract，再确认 FSDP divide factor 被关闭。
6. EP 时检查显式 prefetch 邻接链和 dispatcher 的 host sync；两者共同决定 all-gather 能否提前。
7. 最后再评估 CPU offload、symmetric memory 等实现优化，它们不应改变上述所有权关系。

> [!important] 证据边界
> “storage adapter / 参数存储平面”是本知识库对完整 storage mesh、`DataParallelMeshDims` 和 per-param mesh 分流的归纳，不是 PyTorch FSDP2 的公开类名。源码直接给出各 mesh、axis 与策略；本页用“平面”强调它们和前后向 `SpmdType` 值布局不是同一所有权。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 当前 Trainer、并行与实验子系统入口。
- [[10_torchtitan_parallel_dims_analysis]] —— dense/sparse storage mesh 与 fwd/bwd value mesh 的 rank 重切基础。
- [[13_torchtitan_cp_analysis]] —— CP 输入 shard、R@CP 逻辑参数与 CP storage shard 的双平面关系。
- [[15_torchtitan_ep_analysis]] —— `efsdp × ep` rank 重切、dispatcher 与 expert 参数所有权。
- [[20_torchtitan_fsdp_prefetch_overlap_memory_analysis]] —— FSDP2 内部 all-gather、reshard、stream 和峰值显存生命周期。
- [[21_torchtitan_hsdp_backward_overlap_analysis]] —— replicate/shard 两级梯度通信和 HSDP backward overlap。
- [[25_torchtitan_simple_fsdp_analysis]] —— Graph/compile 路线把 collective 放入图内的对照设计。
