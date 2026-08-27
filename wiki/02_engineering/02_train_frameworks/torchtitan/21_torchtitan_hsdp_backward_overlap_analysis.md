---
title: "HSDP 反向重叠：TorchTitan 声明通信所有权，FSDP2 执行规约流水"
---

# HSDP 反向重叠：TorchTitan 声明通信所有权，FSDP2 执行规约流水

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：当前 TorchTitan 并没有实现一套自己的 “reduce-scatter / all-reduce 双流调度器”。它负责把 `dp_replicate` 声明为复制轴、把 `dp_shard(+cp)` 声明为分片轴，划定 FSDP unit、选择重分片与显式预取策略，然后把这些契约交给 PyTorch `fully_shard`；真正的 autograd hook、collective stream、buffer 生命周期与收尾同步属于上游 FSDP2。因而分析 HSDP overlap 时，首先要区分“本仓库可保证的接线”与“所安装 PyTorch 版本的执行细节”，不能拿一份旧 PyTorch 源码的行号冒充当前 TorchTitan 事实。
>
> 本页回答 HSDP 如何从 rank 预算进入 FSDP2、TorchTitan 在反向前后究竟控制什么、重叠何时可能成立，以及当前测试能证明到哪里。FSDP2 通用状态机见 [[11_torchtitan_fsdp_analysis]]；storage/type mesh 分层见 [[16_torchtitan_spmd_types_analysis]]；预取与峰值量化见 [[20_torchtitan_fsdp_prefetch_overlap_memory_analysis]]；GraphTrainer 的显式图级 collective 调度见 [[27_torchtitan_graph_trainer_compiler_runtime_analysis]]。

---

## 1. Overview：先确定所有权，再讨论“是否被掩盖”

### ① 背景 / 问题

纯 FSDP 只需回答“参数沿哪个 group 分片”；HSDP 还要回答“相同参数分片在哪些 group 之间复制”。当前 rank 预算要求 `dp_replicate × dp_shard × cp × tp × pp == world_size`，并在 `dp_shard=-1` 时用剩余 rank 反解该维度（`torchtitan/distributed/parallel_dims.py:102`、`torchtitan/distributed/parallel_dims.py:114`、`torchtitan/distributed/parallel_dims.py:118`）。这决定了 HSDP 不是额外再乘一条轴，而是既有 world rank 的一种参数所有权解释。

“反向通信被掩盖”至少包含三个不同问题：梯度在什么 group 上规约、规约何时可从计算关键路径移开、optimizer 何时能安全读取梯度。TorchTitan 当前源码直接回答第一问和训练边界，却把第二问中的 stream/hook 细节交给导入的 PyTorch FSDP2（`torchtitan/distributed/fsdp.py:11`、`torchtitan/distributed/fsdp.py:13`）。

### ② 为什么选择声明式适配，而不是在 Trainer 手写 RS / AR

选中路线是：用 `DeviceMesh + DataParallelMeshDims` 表达 shard/replicate 轴，再调用 `fully_shard`。明显替代方案是在 `loss.backward()` 后由 Trainer 遍历参数，显式发起 reduce-scatter 与 all-reduce。前者的决定性标准是参数生命周期：FSDP unit 的 unshard、reshard、梯度规约都必须与 autograd 中每个 module 的可用时刻一致，而 Trainer 只看见一次 `loss.backward()`（`torchtitan/trainer.py:703`、`torchtitan/trainer.py:710`、`torchtitan/trainer.py:725`）。

提交 `57cfb27458` 的正文给出了具体失败案例：`spmd_types` 已把 dense DP storage 轴暴露成 `dp_shard`，旧调用方却仍请求 legacy `fsdp`，初始化会报 `Invalid mesh dim: 'fsdp'`；修复选择显式告诉 FSDP 哪个轴是 shard，而不是依赖名字猜测。当前共享入口也说明：显式 `dp_mesh_dims` 是为了避免新增 mesh 轴时静默误分类（`torchtitan/distributed/fsdp.py:210`、`torchtitan/distributed/fsdp.py:213`、`torchtitan/distributed/fsdp.py:217`）。

### ③ 当前实现 / 状态 / 调用链

```text
ParallelismConfig
  -> ParallelDims.from_config
  -> build_mesh: dense storage = pp,dp_replicate,dp_shard,cp,tp
  -> model parallelize_fn
  -> resolve_fsdp_mesh
       shard = dp_shard (+ cp)
       replicate = dp_replicate（启用时）
  -> apply_fsdp_to_decoder
       per-block fully_shard + root fully_shard
  -> Trainer: forward -> loss -> backward -> clip -> optimizer.step
```

Trainer 在分布式初始化后由配置构造 `ParallelDims`（`torchtitan/trainer.py:628`、`torchtitan/trainer.py:637`），非 PP 路径调用模型的 `parallelize_fn`（`torchtitan/trainer.py:472`、`torchtitan/trainer.py:476`）。Llama 代表路径在 `spmd_types` 下先 resolve storage mesh，再把 mesh、dtype、reshard policy 和 symmetric-memory 开关传给共享 decoder 入口（`torchtitan/models/llama3/parallelize.py:57`、`torchtitan/models/llama3/parallelize.py:60`、`torchtitan/models/llama3/parallelize.py:68`）。

### ④ 约束 / 代价 / 失败边界

本页基线只冻结 TorchTitan commit，并未冻结一份 PyTorch 源码。因此可以把“HSDP 需要 shard 与 replicate 两类规约”当作 `DataParallelMeshDims` 的接口语义，却不能把“恰好五条 stream”“某个私有 hook 位于某行”“all-reduce 必然原地且零额外峰值”写成当前 TorchTitan 的源码保证。当前 adapter 甚至提供 `enable_fsdp_symm_mem` 去改变 FSDP module 的通信实现（`torchtitan/distributed/fsdp.py:102`、`torchtitan/distributed/fsdp.py:108`），说明 collective 底层路径本来就是可替换的。

### ⑤ 有锚点的发展趋势

**事实**：提交 `601cf4d230` 删除 `full_dtensor`，正文明确将 `spmd_types` 作为默认、`partial_dtensor` 作为 fallback；当前配置也只保留这两个 backend（`torchtitan/config/configs.py:174`、`torchtitan/config/configs.py:178`）。**推断**：HSDP 页的稳定抽象会继续上移到“所有权契约与可观测边界”，而不是绑定 FSDP2 私有类名或 stream 数量。

---

## 2. HSDP 触发：replicate 轴与“有效 shard 区域”共同决定语义

### ① 背景 / 问题

把 `data_parallel_replicate_degree > 1` 直接等同于 HSDP 不够准确：若没有有效 shard 区域，它更接近复制式 DP；反过来，即使 `dp_shard=1`，CP 也会进入 FSDP shard group。当前配置注释把 HSDP 简化为 replicate 与 shard degree 都大于 1（`torchtitan/config/configs.py:124`、`torchtitan/config/configs.py:129`），但实际 resolver 与测试矩阵表达了更宽的组合。

### ② 为什么选择把 CP 合入 shard，而不是再做一套 CP 参数同步

选中路线是把 dense 参数的 shard 轴设为 `dp_shard`，启用 CP 时再追加 `cp`；明显替代方案是 CP 只切 activation、参数继续复制。决定性标准是 CP ranks 是否需要共享一份完整参数：mesh 文档明确说 CP 即使没有常规 DP，也依赖 FSDP 的 weight all-gather 与 gradient reduce-scatter（`torchtitan/distributed/parallel_dims.py:160`、`torchtitan/distributed/parallel_dims.py:164`）。

### ③ 当前实现 / 状态 / 调用链

`spmd_types` 建两张 dense 视图：FSDP storage mesh 是 `dp_replicate,dp_shard,cp,tp`，forward/backward 类型 mesh 则把前两者折叠成逻辑 `dp`（`torchtitan/distributed/parallel_dims.py:229`、`torchtitan/distributed/parallel_dims.py:233`、`torchtitan/distributed/parallel_dims.py:238`）。resolver 从 storage mesh 取 active axes，构造：

- `shard="dp_shard"`，CP 启用时为 `shard=("dp_shard", "cp")`；
- `replicate="dp_replicate"`，仅在 replicate degree 大于 1 时设置；
- size-1 全 mesh 时不传 `DataParallelMeshDims`，避免没有 SPMD 参数注解时错误翻译（`torchtitan/distributed/fsdp.py:47`、`torchtitan/distributed/fsdp.py:52`、`torchtitan/distributed/fsdp.py:54`、`torchtitan/distributed/fsdp.py:62`）。

`partial_dtensor` 仍使用预先 flatten 的 `fsdp = dp_shard × cp` 轴；Llama 按 replicate 是否启用选择 `[dp_replicate, fsdp]` 或 `[fsdp]`，且不传 `dp_mesh_dims`（`torchtitan/models/llama3/parallelize.py:61`、`torchtitan/models/llama3/parallelize.py:66`）。

### ④ 约束 / 代价 / 失败边界

并行度必须精确耗尽 world size；错误组合在建 mesh 前即 assert（`torchtitan/distributed/parallel_dims.py:111`、`torchtitan/distributed/parallel_dims.py:118`）。当前代码固定 axis 的 unflatten 顺序，却不读取 node/locality 信息来自动把 shard 放节点内、replicate 放节点间（`torchtitan/distributed/parallel_dims.py:243`、`torchtitan/distributed/parallel_dims.py:246`）；因此“典型 NVLink RS + 跨节点 AR”是部署建议，不是本基线保证。

日志也不是权威分类器：共享入口只检查 storage mesh 名称中是否含 `dp_replicate`，然后输出 HSDP/FSDP 文本（`torchtitan/distributed/fsdp.py:376`、`torchtitan/distributed/fsdp.py:380`）。真正评估通信量时应看 replicate group 大小和 flatten 后的有效 shard group 大小，而不是只看日志字符串。

### ⑤ 有锚点的发展趋势

当前测试明确把 `dp_replicate=2, dp_shard=1, cp=2` 命名为“HSDP+CP without dp_shard”（`torchtitan_recipes/tests/features.py:300`、`torchtitan_recipes/tests/features.py:305`、`tests/integration_tests/features.py:214`、`tests/integration_tests/features.py:218`）。**推断**：配置 docstring 中“必须 dp_shard>1”的简写会需要与实际的“有效 FSDP shard 区域”统一。

---

## 3. FSDP unit 粒度：重叠窗口来自逐层边界，不是 Trainer 后置规约

### ① 背景 / 问题

即使 shard/replicate group 正确，若整模型只有一个 FSDP unit，梯度规约只能在大块计算结束后开始；若 unit 过碎，又会增加 collective launch 和 metadata 开销。HSDP overlap 的第一项 TorchTitan 自有决策因此不是 stream，而是 module 分组。

### ② 为什么选择 TransformerBlock 级 unit，而不是根模块单 unit

当前 decoder 路线为 embedding、每个 TransformerBlock、末端 norm/head 建 unit，最后再 wrap 根模块。明显替代方案是只 wrap 根模块。决定性标准是为相邻层制造参数释放、反向规约与预取边界，同时避免共享权重重复 all-gather：weight tying 时 embedding/norm/head 被合成一个 unit（`torchtitan/distributed/fsdp.py:238`、`torchtitan/distributed/fsdp.py:246`），非 tying 路径则分别处理首尾模块（`torchtitan/distributed/fsdp.py:252`、`torchtitan/distributed/fsdp.py:260`）。

小型 vision encoder 是反例：代码选择整个 encoder 单 unit，因为相对 decoder 较小，一次 all-gather 比逐层 sharding 更划算（`torchtitan/distributed/fsdp.py:149`、`torchtitan/distributed/fsdp.py:151`、`torchtitan/distributed/fsdp.py:159`）。这说明 unit 粒度是通信启动次数与释放/掩盖窗口的取舍，而非固定教条。

### ③ 当前实现 / 状态 / 调用链

共享 decoder 先建立 mixed-precision policy 和 `fsdp_config`，只有 resolver 返回非空声明时才添加 `dp_mesh_dims`（`torchtitan/distributed/fsdp.py:223`、`torchtitan/distributed/fsdp.py:228`、`torchtitan/distributed/fsdp.py:230`）。随后遍历 `model.layers` 对每个 block 调 `fully_shard`，dense block 走直接分支（`torchtitan/distributed/fsdp.py:267`、`torchtitan/distributed/fsdp.py:361`、`torchtitan/distributed/fsdp.py:365`），最后 wrap 根模块（`torchtitan/distributed/fsdp.py:368`）。

训练执行面没有额外的 HSDP 分支：普通路径完成 forward、sum-loss/global-token normalization 后直接 `loss.backward()`（`torchtitan/trainer.py:710`、`torchtitan/trainer.py:715`、`torchtitan/trainer.py:725`）；反向返回后才进入 grad norm、finite gate 和 optimizer step（`torchtitan/trainer.py:850`、`torchtitan/trainer.py:878`、`torchtitan/trainer.py:887`、`torchtitan/trainer.py:889`）。因此 optimizer 前的完成性由 FSDP2 `fully_shard` 契约承担，不是 Trainer 显式等待某条 HSDP stream。

### ④ 约束 / 代价 / 失败边界

本仓库可推出“逐 block 提供多个规约机会”，不能仅凭 unit 划分推出“通信一定完全隐藏”。是否重叠还取决于上游 FSDP2 版本、collective backend、bucket 大小、层计算时长、拓扑和尾部等待；TorchTitan 当前没有为 eager HSDP 暴露 stream 数、bucket bytes 或 AR/RS 调度顺序配置。

MoE 又破坏“一层一条同构 HSDP 链”的简化时间线：EP>1 时 expert 参数走 sparse `efsdp` mesh，其他参数走 dense DP mesh，单个 block 的 `shard_placement_fn` 按参数返回不同 mesh info（`torchtitan/distributed/fsdp.py:318`、`torchtitan/distributed/fsdp.py:329`、`torchtitan/distributed/fsdp.py:345`、`torchtitan/distributed/fsdp.py:359`）。旧页用一组固定 `S` 描述整层 buffer/通信量，不能覆盖这个现状。

### ⑤ 有锚点的发展趋势

提交 `d92336fee0` 把曾分散在 dense/MoE 模型中的 apply-FSDP 逻辑合并进 `distributed/fsdp.py`；提交正文说明 MoE 路线是 dense 路线的超集。**推断**：后续 eager HSDP 的模型间差异应继续通过共享 unit/placement policy 表达，而不应复制模型专用 backward scheduler。

---

## 4. 反向通信语义：TorchTitan 保证缩放所有权，不保证私有双流细节

### ① 背景 / 问题

HSDP 的正确性不只在于发出两类 collective，还在于“谁负责除以数据并行规模”。若 FSDP 自动平均、loss 又按全局 token 数缩放，遇到不同 rank/微批有效 token 不均衡时会重复或错误加权。

### ② 为什么选择全局 token 归一化 + collective 纯求和

当前路线先统计一个 optimizer step 内的有效 token，再在 DP mesh 上求和；cross entropy 使用 sum reduction，随后除以 global valid tokens（`torchtitan/components/loss.py:57`、`torchtitan/components/loss.py:60`、`torchtitan/components/loss.py:266`、`torchtitan/components/loss.py:281`），FSDP 的自动 gradient division 被关闭。明显替代方案是每 rank/每微批先取 mean，再让 FSDP 按 DP degree 平均。提交 `0cb743558` 的正文明确否定了后者：有效 token 分布不均时，每个 token 的贡献不相等；新路线为此增加一次全局 token-count all-reduce。

### ③ 当前实现 / 状态 / 调用链

Trainer 先收集本 step 的所有 gradient-accumulation / PP microbatch，累加非 `IGNORE_INDEX` token（`torchtitan/trainer.py:785`、`torchtitan/trainer.py:788`、`torchtitan/trainer.py:793`）。DP 启用时，它在 batch mesh 上求 `global_valid_tokens`（`torchtitan/trainer.py:798`、`torchtitan/trainer.py:801`、`torchtitan/trainer.py:804`），并把该值传入 loss（`torchtitan/trainer.py:715`、`torchtitan/trainer.py:718`）。

parallelize 完成后，共享入口遍历所有 `FSDPModule`，调用 `set_gradient_divide_factor(1.0)`（`torchtitan/distributed/fsdp.py:85`、`torchtitan/distributed/fsdp.py:97`、`torchtitan/distributed/fsdp.py:99`），且在根 module wrap 后统一执行（`torchtitan/distributed/fsdp.py:368`、`torchtitan/distributed/fsdp.py:373`）。因此当前 TorchTitan 事实是“把规约缩放所有权交给 loss/global token 逻辑”；至于上游将 factor=1 具体映射成哪种 NCCL op，属于 PyTorch 实现细节。

### ④ 约束 / 代价 / 失败边界

这条路线要求所有 FSDP/replicate module 都成功被遍历并关闭 division；源码特意说明 ReplicateModule 继承 FSDPModule，因此也在范围内（`torchtitan/distributed/fsdp.py:89`、`torchtitan/distributed/fsdp.py:92`）。代价是 step 前多一次 token-count collective；它解决数值权重，不等于优化梯度 collective 的 overlap。

训练的 accumulation 外层循环只是重复调用 forward/backward，然后再做一次 optimizer step（`torchtitan/trainer.py:808`、`torchtitan/trainer.py:812`、`torchtitan/trainer.py:830`、`torchtitan/trainer.py:850`）。当前核心 Trainer 在这段循环里没有 HSDP 专用的“只在最后一次 accumulation 同步”控制，因此不能把 chunked-loss 的局部合并能力外推成全模型 accumulation 规约合并。

### ⑤ 有锚点的发展趋势

当前源码没有 HSDP gradient-accumulation coalescing TODO；这里不预测其路线。可验证的演进只有 `0cb743558` 已把正确性判据从“每 rank mean”改为“每个有效 token 权重相同”。

---

## 5. 预取与 reshard：它们改变窗口和显存，但不是“AR/RS 双流开关”

### ① 背景 / 问题

反向 overlap 既可能被参数 re-all-gather 的等待限制，也可能被过早保留完整参数的显存限制。`reshard_after_forward` 与 backward prefetch 都影响这个窗口，但二者不能与梯度规约本身混为一谈。

### ② 为什么选择 policy + 少量显式例外，而不是全局固定预取链

当前路线让 `always/never/default` 决定 forward 后是否 reshard，并仅在 EP 场景显式串起相邻 FSDP unit 的 prefetch。明显替代方案是所有模型无条件显式预取。决定性标准是干扰源：代码注释只在 EP 分支指出 D2H sync 可能破坏 FSDP implicit prefetch（`torchtitan/distributed/fsdp.py:384`、`torchtitan/distributed/fsdp.py:385`），所以没有把该 workaround 扩散到普通 dense HSDP。

### ③ 当前实现 / 状态 / 调用链

policy resolver 对 `always/never` 原样返回；`default` 在无 PP 时为 true，在 PP 时为 false，以避免每个 pipeline microbatch 的昂贵、难掩盖 all-gather（`torchtitan/distributed/fsdp.py:124`、`torchtitan/distributed/fsdp.py:129`、`torchtitan/distributed/fsdp.py:132`）。该值传给 embedding/block 等 FSDP unit（`torchtitan/distributed/fsdp.py:234`、`torchtitan/distributed/fsdp.py:253`、`torchtitan/distributed/fsdp.py:296`）。末端 norm/head 在 default 下特意不 reshard，因为 FSDP 很快会再次 prefetch（`torchtitan/distributed/fsdp.py:258`、`torchtitan/distributed/fsdp.py:264`）。

若 `ep_degree==1`，共享入口在设置任何显式 prefetch 之前直接返回（`torchtitan/distributed/fsdp.py:384`、`torchtitan/distributed/fsdp.py:387`）。EP 启用时，forward 链按层正序连接（`torchtitan/distributed/fsdp.py:389`、`torchtitan/distributed/fsdp.py:401`），backward 链从 lm_head 到反序 blocks 再到 embedding（`torchtitan/distributed/fsdp.py:408`、`torchtitan/distributed/fsdp.py:414`、`torchtitan/distributed/fsdp.py:421`、`torchtitan/distributed/fsdp.py:424`）。

### ④ 约束 / 代价 / 失败边界

`fsdp_reshard_after_forward` 的配置文档把它定义为参数行为的 memory/communication trade-off（`torchtitan/config/configs.py:147`、`torchtitan/config/configs.py:149`、`torchtitan/config/configs.py:159`）；当前 adapter 没有用它选择 gradient RS/AR stream。因此“never 会关闭 backward gradient communication”是错误推论。

显式 backward prefetch 只在 EP 开启时由 TorchTitan 接线；普通 HSDP 是否以及如何隐式预取取决于上游 FSDP2。同理，旧页精确宣称“最后 embedding 的 RS+AR 是唯一暴露尾部”超出了当前基线：PP schedule、chunked loss、MoE per-param mesh 与上游版本都可能改变最后一个关键路径。

### ⑤ 有锚点的发展趋势

提交 `58b034444` 把 FSDP symmetric-memory 通信接入默认 Trainer，并在正文中明确不覆盖 GraphTrainer 或 Inductor async-TP kernels；当前开关通过 `set_force_sum_reduction_for_comms` 和 `set_symm_mem_for_comm` 应用于全部 FSDP module（`torchtitan/distributed/fsdp.py:102`、`torchtitan/distributed/fsdp.py:109`）。**推断**：通信实现会继续可替换，因此性能页应记录 profiler 观察与具体 PyTorch/runtime 版本，而不是固化一个永恒 stream 图。

---

## 6. Chunked loss 与 MoE：当前真正显式控制梯度同步的两个边界

### ① 背景 / 问题

“一次 module backward 就同步一次梯度”在 chunked lm_head 上会造成每个 chunk 一次 reduce-scatter；“每层只有一张 DP mesh”在 EP MoE 上也不成立。这两处是旧单一 HSDP 时间线最容易失真的地方。

### ② 为什么选择局部关闭同步，而不是让所有 chunk 独立规约

chunked loss 的选中路线是在前 `N-1` 个 lm_head chunk 上关闭 gradient sync，只在最后一个 chunk 恢复。明显替代方案是每 chunk 立即同步。决定性标准是 launch 次数与重复 all-gather：代码同时关闭 lm_head 的 forward/backward reshard，让所有 chunk 复用已 unshard 权重，并把 per-chunk grad sync 合成最后一次 reduce-scatter（`torchtitan/components/loss.py:666`、`torchtitan/components/loss.py:673`）。

### ③ 当前实现 / 状态 / 调用链

进入 chunk loop 前，lm_head 设置 `reshard_after_forward=False`、`reshard_after_backward=False`、`requires_gradient_sync=False`，并在 FSDP idle 状态显式 `unshard()`（`torchtitan/components/loss.py:670`、`torchtitan/components/loss.py:681`）。最后一个 chunk 前恢复 gradient sync（`torchtitan/components/loss.py:683`、`torchtitan/components/loss.py:688`）；全部 chunk 结束后恢复两个 reshard flag、同步 flag 并显式 reshard（`torchtitan/components/loss.py:710`、`torchtitan/components/loss.py:714`）。

MoE 路线则由 dense `dp_mesh` 与 expert `edp_mesh` 并存：EP>1 时 expert 用 `efsdp`，非 expert 用 dense shard placement（`torchtitan/distributed/fsdp.py:282`、`torchtitan/distributed/fsdp.py:332`、`torchtitan/distributed/fsdp.py:346`、`torchtitan/distributed/fsdp.py:352`）。`dp_replicate` 同时出现在 dense 与 sparse storage mesh（`torchtitan/distributed/fsdp.py:28`、`torchtitan/distributed/fsdp.py:29`、`torchtitan/distributed/fsdp.py:81`）。

### ④ 约束 / 代价 / 失败边界

chunked-loss 控制只作用于 `lm_head`，不能当作一般 FSDP gradient accumulation API。它还显式处理 CUDA graph capture：隐式 unshard 会把 eager event 留在共享 state，capture 不能等待该 event，所以必须在 FSDP idle 时 unshard（`torchtitan/components/loss.py:674`、`torchtitan/components/loss.py:680`）。

MoE 的多个 mesh 会增加同层通信相互干扰的可能；当前 TorchTitan 只通过 per-param placement 与 EP 显式 prefetch 给出结构，未提供 eager HSDP+EP 的统一 overlap 证明。提交 `18a46d427` 的正文报告过 per-param mesh 的 AG/RS overlap 与数值一致，但它是引入时的实验记录，不是当前所有硬件、dispatcher 和上游版本的性能保证。

### ⑤ 有锚点的发展趋势

当前没有把 chunked-loss 同步合并泛化到任意 module 的 TODO。可确认的趋势是 per-param FSDP mesh 已成为 MoE 的共享 decoder 路线，而非模型专用实现（`torchtitan/distributed/fsdp.py:183`、`torchtitan/distributed/fsdp.py:185`、`torchtitan/distributed/fsdp.py:218`）。

---

## 7. 组合与验证：测试证明“能跑/数值守卫”，不证明 overlap 百分比

### ① 背景 / 问题

一组 integration recipe 出现在 CI，只能证明当前配置被构造并进入测试；除非测试含 profiler trace、collective 顺序断言或吞吐阈值，不能据此宣称通信已被完全掩盖。

### ② 为什么选择代表性组合矩阵，而不是穷举所有并行轴

当前测试选中基础 HSDP，再分别叠加 TP、CP、compile/float8、EP 与 Flux validation。明显替代方案是穷举 `dpR × dpS × cp × tp × pp × ep`；决定性标准是 GPU 成本与代表性失败面。recipe 用极小 debug model 验证接线，再由少量 real-PG / H100 条目覆盖 backend 特性。

### ③ 当前实现 / 状态 / 调用链

当前可见矩阵包括：

| 组合 | 配置证据 | integration 证据 | 能证明什么 |
|---|---|---|---|
| HSDP 2×2 | `torchtitan_recipes/tests/features.py:271-276` | `tests/integration_tests/features.py:190-195` | 基础 4-GPU 路径 |
| HSDP+TP2 | `torchtitan_recipes/tests/features.py:286-289` | `tests/integration_tests/features.py:202-206` | dense storage 与 TP 组合 |
| HSDP+CP2 | `torchtitan_recipes/tests/features.py:309-312` | `tests/integration_tests/features.py:220-224` | CP flatten 进 shard group |
| replicate2+CP2、dp_shard1 | `torchtitan_recipes/tests/features.py:300-306` | `tests/integration_tests/features.py:214-218` | CP 单独形成有效 shard 区域 |
| HSDP+CP+compile+float8 | `torchtitan_recipes/tests/h100.py:54-60` | `tests/integration_tests/h100.py:43-47` | H100 复合路径 |
| DeepSeek HSDP+EP2 | `torchtitan_recipes/tests/models.py:105-112` | `tests/integration_tests/models.py:67-71` | dense/sparse per-param mesh |
| Flux HSDP+CP+validation | `tests/integration_tests/flux.py:26-35` | 同左 | real-PG 训练/验证/推理流程 |

### ④ 约束 / 代价 / 失败边界

表中没有专门的 eager HSDP+PP 条目，也没有 HSDP+symmetric-memory 条目；现有 symmetric-memory integration 是 2-GPU FSDP 且 CUDA-only（`tests/integration_tests/h100.py:30`、`tests/integration_tests/h100.py:35`）。同样没有测试断言 eager FSDP2 的 stream 数、RS/AR 先后、buffer 峰值或通信隐藏率。要回答这些性能问题，必须冻结实际 PyTorch wheel、NCCL/设备拓扑并采 profiler trace。

GraphTrainer 也不能拿来替 eager 路径背书：其配置转换明确把 backend 设成 `partial_dtensor`（`torchtitan/experiments/graph_trainer/configs.py:242`、`torchtitan/experiments/graph_trainer/configs.py:257`）。提交 `b94c11a63` 说明 GraphTrainer 曾显式补 bucketed replicate all-reduce，并把 wait 推迟到 backward compute 后；提交 `63a758ccc` 又明确其 HSDP 测试只断言 launch-count bucketing，且当时 overlap 调度仍 known-suboptimal。那是图编译路线的问题域，不是 eager FSDP2 当前行为的 locator。

### ⑤ 有锚点的发展趋势

**推断**：若要把本页恢复为精确双流时间线，应新增一份独立、冻结 commit 的 PyTorch FSDP2 基线和一条 profiler/ordering 测试；在此之前，当前最可靠的回归单元是 mesh 契约、unit 划分、缩放所有权和组合矩阵。

---

## 8. 当前决策清单：怎样判断是否值得用 HSDP

### ① 背景 / 问题

HSDP 的目标不是“比 FSDP 多一个 all-reduce”，而是在总 world size 固定时改变参数分片范围与跨副本规约范围。选择错误会同时损失显存和通信关键路径。

### ② 为什么选择按瓶颈判定，而不是默认开启 replicate

明显替代方案是把 `dp_replicate` 设为节点数、`dp_shard` 设为节点内 GPU 数并假设一定更快。当前 TorchTitan 不做拓扑感知映射，因此应以三项可测判据决策：每 rank 是否能容纳仅按有效 shard group 分片的参数/梯度/optimizer state；跨 replicate group 的规约尾部是否可接受；unit 粒度是否提供足够计算窗口。前两项来自 mesh/存储语义，最后一项来自逐 block wrap（`torchtitan/distributed/fsdp.py:267`、`torchtitan/distributed/fsdp.py:368`）。

### ③ 当前实现 / 状态 / 调用链

Quick Start 的最小配置是：

```python
config.parallelism.data_parallel_replicate_degree = 2
config.parallelism.data_parallel_shard_degree = 4
```

随后用当前日志确认 mesh 预算与 HSDP apply：mesh builder 会记录 `pp/dp_replicate/dp_shard/cp/tp/ep`（`torchtitan/distributed/parallel_dims.py:210`、`torchtitan/distributed/parallel_dims.py:214`），共享入口记录 Applied HSDP/FSDP（`torchtitan/distributed/fsdp.py:376`、`torchtitan/distributed/fsdp.py:380`）。但性能验收必须看 trace 中的 exposed collective tail 与峰值，而不是仅看这两条日志。

### ④ 约束 / 代价 / 失败边界

- replicate degree 增大不会扩大单副本的参数 shard group；它增加相同参数分片的副本数。当前 mesh 直接把 batch 设为 `dp_replicate × dp_shard`、FSDP 设为 `dp_shard × cp`（`torchtitan/distributed/parallel_dims.py:216`、`torchtitan/distributed/parallel_dims.py:218`）。
- `mixed_precision_reduce` 当前只允许 float32（`torchtitan/config/configs.py:104`、`torchtitan/config/configs.py:108`），但不能仅据此推导上游临时 buffer 的精确字节峰值。
- `enable_fsdp_symm_mem` 会改变通信路径；其当前 integration 没覆盖 HSDP，启用后必须独立测量（`torchtitan/config/configs.py:162`、`torchtitan/config/configs.py:165`）。
- PP 默认关闭 forward 后 reshard 是为避免微批 all-gather，不是 HSDP overlap 保证（`torchtitan/distributed/fsdp.py:129`、`torchtitan/distributed/fsdp.py:132`）。

### ⑤ 有锚点的发展趋势

当前没有自动拓扑映射或 HSDP autotune TODO，本页不把经验配置包装成路线承诺。可以确认的方向只是 adapter 越来越显式：mesh dims、reshard policy、symmetric-memory 与 per-param mesh 都通过公开配置/共享入口传递，而不是散落在模型反向代码中。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP2 机制]] —— 通用 FSDP state、参数生命周期与上游契约边界
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|SPMD 类型与 storage mesh]] —— `DataParallelMeshDims` 为何必须与逻辑 `dp` 分层
- [[02_engineering/02_train_frameworks/torchtitan/20_torchtitan_fsdp_prefetch_overlap_memory_analysis|FSDP 预取、重叠与显存]] —— 参数预取和峰值的独立专题
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|通信优化与重叠矩阵]] —— 跨 FSDP、TP、CP、EP 的竞争关系
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|GraphTrainer 编译运行时]] —— bucket/reorder 属于显式 FX 图路线，不等于 eager HSDP
- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 系列边界与阅读顺序
