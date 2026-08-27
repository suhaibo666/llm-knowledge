---
title: "Pipeline Parallel：用 stage 适配协议衔接模型切分与上游 schedule"
---

# Pipeline Parallel：用 stage 适配协议衔接模型切分与上游 schedule

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：TorchTitan PP 不自己重写 1F1B/GPipe 的 action 引擎；它把完整 meta model 按 FQN 剪成 rank-local `model_parts`，在每个 part 上继续施加 TP/CP/FSDP/AC/compile，再把预切好的 microbatches 交给上游 `torch.distributed.pipelining` schedule。这是一个 **stage 适配协议**：TorchTitan 拥有模型切分、rank 映射、输入预处理、loss/checkpoint 接口；PyTorch 拥有 stage 间 action/P2P 时序。
>
> 本页回答 stage/`model_parts` 归属、schedule 与 CSV 构建、microbatch/loss/metrics 传播、first/last-stage 减层、checkpoint 以及 FSDP/TP/CP/SPMD 边界。rank/mesh 预算归 [[10_torchtitan_parallel_dims_analysis]]，FSDP/TP/CP/EP 内部机制分别归 [[11_torchtitan_fsdp_analysis]]、[[12_torchtitan_tp_analysis]]、[[13_torchtitan_cp_analysis]]、[[15_torchtitan_ep_analysis]]；上游 schedule 的理论气泡不在本页重复。

---

## 1. Overview

大模型的层可以分到多个 rank，但只有“分层”还不够：首层需要 token embedding，尾层需要 norm/head/loss，looped schedule 还会让一个 rank 持有多个 virtual stages。如果每个模型自己同时实现切分和 action 调度，模型结构、优化器/checkpoint 与 schedule 就会紧耦合。

当前主线是两段式：FQN 列表定义“一个 stage 拥有哪些模块”，`PipelineStage`/schedule 定义“这些 stage 何时 forward/backward/P2P”。提交 `3c84ce095` 明确说明收敛到 FQN chunks 的理由是对所有模型通用，并能被不需要 pipeline execution 的分片用例复用；当前公开入口也只导出 `pipeline_llm` 与 `pipeline_vlm`（`torchtitan/distributed/pipeline_parallel.py:36`、`torchtitan/distributed/pipeline_parallel.py:38`）。

| 概念 | 当前所有者 | 关键语义 |
|---|---|---|
| `module_fqns_per_model_part` | 配置/模型 helper | 每个 virtual stage 保留的模块 FQN |
| `model_parts` | 当前 PP rank | 本 rank 真正物化、优化、checkpoint 的 stage chunks |
| `PipelineStage` | PyTorch pipelining | 用 stage index、PP group 和 local module 封装通信边界 |
| schedule class | PyTorch pipelining | 决定 action/P2P 时序；TorchTitan 只选类、填 stages/microbatches |
| `pp_has_first_stage` | TorchTitan adapter | 该 rank 是否应向 schedule 传 model args |
| `pp_has_last_stage` | TorchTitan adapter | 该 rank 是否应传 labels 并接收 losses |
| `ModelWrapper` | checkpoint | 把本 rank 多个 chunks 的 FQN state 合成一张扁平视图 |

```text
完整 meta model（每个 rank 同构）
        |
        | FQN stage plan + PP-rank mapping
        v
deepcopy + prune -> rank-local model_parts[]
        |
        | 每个 part: sharding -> AC -> compile -> FSDP
        v
PipelineStage(submod=part, stage_idx, get_mesh)
        |
        | get_schedule_class(...) 或 CSV Runtime
        v
PyTorch schedule
        ^
        | Trainer 预切 microbatches
        | first rank: args / all ranks: kwargs / last rank: targets
        |
dataloader -> preprocess_inputs -> schedule.step -> last-stage losses

本 rank 的 model_parts[] 同时交给 optimizer + checkpoint。
```

### Quick Start：最小配置与调用链

```python
config.parallelism.pipeline_parallel_degree = 2
config.parallelism.pipeline_parallel_schedule = "1F1B"
config.parallelism.num_pp_microbatches = 8
config.training.disable_cuda_graphs = True
```

`pipeline_parallel_degree` 是物理 PP rank 数，不是 looped schedule 的总 stage 数（`torchtitan/config/configs.py:182`、`torchtitan/config/configs.py:185`）。模型还必须在 `ModelSpec` 注册 `pipelining_fn`；否则 Trainer 在切分前失败（`torchtitan/trainer.py:428`、`torchtitan/trainer.py:433`）。

一次构建与训练的可追踪链是：

```text
Trainer.__init__
  -> meta device 上 model_config.build + protocol verify
  -> ModelSpec.pipelining_fn
       -> _get_pipeline_metadata
       -> _generate_llm_fqn_per_model_part（或显式 FQN 表）
       -> _pipeline_module_split
       -> 每个 local part 调 parallelize_fn
       -> _build_pipeline_schedule
  -> local model_parts 物化/初始化
  -> optimizer(model_parts) + checkpoint(model_parts)

Trainer.train_step
  -> 预取 grad-accumulation × PP-microbatch 数据
  -> 全局有效 token 计数
  -> 每组 pp_forward_backward_step
       -> 每 microbatch preprocess_inputs
       -> schedule.step(arg_mbs, kwarg_mbs, target_mbs)
       -> last-stage rank 求和 losses
  -> PP-aware grad norm / finite gate -> optimizer step
```

Trainer 先在 meta device 上构造完整模型（`torchtitan/trainer.py:351`、`torchtitan/trainer.py:365`），然后由 `pipelining_fn` 返回 schedule、parts 和首尾 stage flags（`torchtitan/trainer.py:435`、`torchtitan/trainer.py:453`）。完整 `model` 随即删除，后续只物化本 rank parts（`torchtitan/trainer.py:454`、`torchtitan/trainer.py:464`）。

---

## 2. Stage 切分与 `model_parts` 所有权

### ① 背景/问题

不同模型层容器、embedding/head 命名不完全一致，而且首尾 stage 的计算不只是 decoder block。旧 split-point 路线只表达“在哪里断开”，不能直接表达一个 virtual chunk 完整拥有哪些模块，也难以复用给非 pipeline 的模型分片。提交 `3c84ce095` 因此废弃 `pipeline_parallel_split_points`，收敛到每 chunk 的 FQN 列表。

### ② 为什么这么设计

**选中路线**是“完整 meta model + 显式 FQN 所有权 + prune copy”；**最明显的替代方案**是模型专属 split points 或先物化完整模型再原地拆分。删除提交明说的决定性标准是模型通用性与 chunk 复用；meta 构建后只物化 local parts 还避免了在每个 rank 上先实体化全模型，这是由当前调用顺序得出的**知识库推断**（`torchtitan/trainer.py:353`、`torchtitan/trainer.py:458`）。

### ③ 实现思路与细节

stage 计划先决定 virtual stage 数。指定 `pipeline_parallel_layers_per_stage` 时，实现把 decoder layers 加上 input/output weights 后向上取整得到 stage 数，然后要求它能整除 PP rank 数（`torchtitan/distributed/pipeline_parallel.py:219`、`torchtitan/distributed/pipeline_parallel.py:240`）。single-stage schedule 要求每 rank 恰好一个 stage，multi-stage schedule 则至少两个（`torchtitan/distributed/pipeline_parallel.py:242`、`torchtitan/distributed/pipeline_parallel.py:256`）；未指定时默认分别是 1 与 2（`torchtitan/distributed/pipeline_parallel.py:258`、`torchtitan/distributed/pipeline_parallel.py:264`）。

`pipeline_parallel_first_stage_less_layers` 与 `last_stage_less_layers` 在实现中不是切分后再删 N 层，而是 `input_weight/output_weight`：它们先进入 effective-layer 预算，再从首尾 stage 的 decoder-layer 配额中扣除（`torchtitan/distributed/pipeline_parallel.py:213`、`torchtitan/distributed/pipeline_parallel.py:224`；`torchtitan/distributed/pipeline_parallel.py:432`、`torchtitan/distributed/pipeline_parallel.py:456`）。额外 effective layers 优先分给靠前 stages（`torchtitan/distributed/pipeline_parallel.py:397`、`torchtitan/distributed/pipeline_parallel.py:430`）。

自动 FQN plan 把 `tok_embeddings` 放首 stage，`norm/lm_head` 放尾 stage，中间为 `layers.N`（`torchtitan/distributed/pipeline_parallel.py:432`、`torchtitan/distributed/pipeline_parallel.py:465`）。VLM 在未给显式 plan 时再把 `vision_encoder` 插入 stage 0，然后委托 `pipeline_llm`（`torchtitan/distributed/pipeline_parallel.py:169`、`torchtitan/distributed/pipeline_parallel.py:185`）。

每个本地 stage 由完整模型 `deepcopy`，保留目标 `ModuleDict/ModuleList` 项，其他子模块设为 `None`（`torchtitan/distributed/pipeline_parallel.py:488`、`torchtitan/distributed/pipeline_parallel.py:528`）。Decoder forward 因此把缺 embedding/norm/head 当成 passthrough（`torchtitan/models/common/decoder.py:294`、`torchtitan/models/common/decoder.py:308`）。实际不存在一个全局 `ModelPart` 类；运行合同是本 rank 的 `list[nn.Module]`，其类型与返回值由 `PipeliningFunction` 定义（`torchtitan/protocols/model_spec.py:19`、`torchtitan/protocols/model_spec.py:22`）。

### ④ 约束/边界

- 自动 plan 拒绝 stage 数大于 effective layers、空 stage，以及 input/output weight 大于最小 stage 容量（`torchtitan/distributed/pipeline_parallel.py:392`、`torchtitan/distributed/pipeline_parallel.py:418`）。
- VLM 的 vision encoder 成本没有进入自动 weight model；源码要求重型 encoder 用 first-stage-less-layers 手动再平衡（`torchtitan/distributed/pipeline_parallel.py:164`、`torchtitan/distributed/pipeline_parallel.py:167`）。给了显式 FQN plan 时，用户自己负责包含 `vision_encoder`（`torchtitan/distributed/pipeline_parallel.py:169`）。
- `_pipeline_module_split()` 明示要求 forward/init 容忍被删层，且不支持嵌套 `ModuleDict/ModuleList`（`torchtitan/distributed/pipeline_parallel.py:580`、`torchtitan/distributed/pipeline_parallel.py:588`）。
- 只有 `ModelSpec.pipelining_fn` 非空的模型能走 PP；例如当前 Flux 注册为 `None`（`torchtitan/models/flux/__init__.py:565`、`torchtitan/models/flux/__init__.py:570`）。

> [!deprecated] `split_points` 不再是 core 配置字段
> 当前真实入口是 `module_fqns_per_model_part` 或 `pipeline_parallel_layers_per_stage`（`torchtitan/config/configs.py:189`、`torchtitan/config/configs.py:211`）。配置注释在 `torchtitan/config/configs.py:222` 仍提到 split points，但没有对应字段或 core 消费者；这是文档残留，不能当成当前 API。当前 `PipelineStage` 也是以 chunk/index/group/get-mesh 构造，不走旧页的 example-input shape-inference 链（`torchtitan/distributed/pipeline_parallel.py:625`、`torchtitan/distributed/pipeline_parallel.py:632`）。

### ⑤ 发展趋势（有锚点的推断）

提交 `db5da4b6b` 把 metadata、FQN generation 和 split helpers 抽出供 Graph PP 复用，同时明确把它们保留为 private API；当前模块注释也继续如此标注（`torchtitan/distributed/pipeline_parallel.py:36`、`torchtitan/distributed/pipeline_parallel.py:37`）。**推断**：方向是复用 stage plan 而不是恢复 model-specific split points，但 private helper 并不承诺稳定外部 API。

---

## 3. Schedule、rank 映射与 CSV 控制面

### ① 背景/问题

同一组 stages 可以跑 GPipe、1F1B、Interleaved1F1B 或 V-shaped schedules，它们的差异是 action 顺序和 stage-to-rank 形状，不是模型参数所有权。若 TorchTitan 复制一份 schedule 枚举和 action table，上游新 schedule 会迫使两处同步；只接受固定 class 又无法表达实验性自定义时序。

### ② 为什么这么设计

**选中路线**是普通 schedule 名称交给 PyTorch `get_schedule_class()`，自定义 action 交给 CSV runtime；**替代方案**是 TorchTitan 维护自己的 schedule registry/engine。提交 `a88dc4125` 的修改就是改为从 `get_schedule_class()` 取 class；提交 `5525d7723` 再加 CSV loader。决定性标准是让上游拥有 action engine，TorchTitan 只保留 stage 适配与配置入口。

### ③ 实现思路与细节

`_get_pipeline_metadata()` 先用配置名取 schedule class，以 `PipelineScheduleSingle`/multi 决定每 rank stage 约束（`torchtitan/distributed/pipeline_parallel.py:204`、`torchtitan/distributed/pipeline_parallel.py:207`）。切分时又以 class 选 rank mapping：

| mapping | 当前 class | rank `r` 持有的 stage |
|---|---|---|
| loop | 除下述 V 类外的 schedule | `r, r+pp, r+2pp, ...` |
| V | `ZBVZeroBubble`, `DualPipeV` | 前半 `r` + 后半镜像 stage |

style 选择与具体索引在 `torchtitan/distributed/pipeline_parallel.py:550`至 `torchtitan/distributed/pipeline_parallel.py:567`。V 映射要求每 rank 恰好两个 stages，loop 则可以有更多（`torchtitan/distributed/pipeline_parallel.py:554`、`torchtitan/distributed/pipeline_parallel.py:563`）。

`_pipeline_module_split()` 只为当前 PP rank 的 indices 构建 `PipelineStage`，每个 stage 带全局 index、总 stage 数与 PP process group（`torchtitan/distributed/pipeline_parallel.py:614`、`torchtitan/distributed/pipeline_parallel.py:632`）。构建后扫描 `stage.is_first/is_last` 生成 Trainer 的两个所有权 flags（`torchtitan/distributed/pipeline_parallel.py:132`、`torchtitan/distributed/pipeline_parallel.py:141`）。

无 CSV 时，单 stage class 只收 `stages[0]`，multi class 收整个 local stage list（`torchtitan/distributed/pipeline_parallel.py:322`、`torchtitan/distributed/pipeline_parallel.py:336`）。有 CSV 时实际 class 改为 `_PipelineScheduleRuntime`，检查文件后调 `_load_csv()`（`torchtitan/distributed/pipeline_parallel.py:290`、`torchtitan/distributed/pipeline_parallel.py:300`；`torchtitan/distributed/pipeline_parallel.py:342`、`torchtitan/distributed/pipeline_parallel.py:352`）。注意：配置中的 schedule 名仍用于前面的 metadata/rank mapping；当前 CSV recipe 用 `PipelineScheduleMulti` 选 looped 形状，再用 CSV 提供真正 actions（`torchtitan_recipes/tests/features.py:242`、`torchtitan_recipes/tests/features.py:249`）。

### ④ 约束/边界

- 直接选抽象 `PipelineScheduleSingle` 会被拒绝，必须选 GPipe/1F1B 等具体类（`torchtitan/distributed/pipeline_parallel.py:311`、`torchtitan/distributed/pipeline_parallel.py:315`）。
- microbatch 数少于总 stages 只警告可能 bubble，不会拒绝运行（`torchtitan/distributed/pipeline_parallel.py:302`、`torchtitan/distributed/pipeline_parallel.py:309`）。
- CSV 路径不存在时在 schedule 构建期失败（`torchtitan/distributed/pipeline_parallel.py:292`、`torchtitan/distributed/pipeline_parallel.py:297`）。
- zero-bubble 与当前 custom CSV 的 core integration cases 都是 disabled：拆分 backward 路径对转发的 FlexAttention `BlockMask` 读 `requires_grad`，但 mask 不是 tensor（`tests/integration_tests/features.py:138`、`tests/integration_tests/features.py:155`；`tests/integration_tests/features.py:166`、`tests/integration_tests/features.py:176`）。配置能构建不等于当前 CI 默认 attention 已验证。

### ⑤ 发展趋势（有锚点的推断）

测试 TODO 把重启条件明确写成上游 `stage_backward_input` 跳过非 tensor stage inputs（`tests/integration_tests/features.py:140`、`tests/integration_tests/features.py:149`）。**推断**：近期修复边界在 PyTorch stage-backward metadata 处，而不是 TorchTitan 复制一套 zero-bubble engine；仓库没有给出完成时间。

---

## 4. Microbatch、loss 与 metrics 传播

### ① 背景/问题

若把一个大 batch 交给 schedule 内部切分，tensor 可以切，但 varlen attention 的 batch-dependent 非 tensor metadata 无法在那里重建。另一旧问题是每 rank/每 microbatch 各自 mean loss：有效 token 数不同时，每个 token 的权重不再相等。提交 `9228564523` 因第一个问题把 microbatch 所有权移到 dataloader/Trainer；提交 `0cb743558` 因第二个问题改为全局有效 token 归一化。

### ② 为什么这么设计

**选中路线**是“一个 dataloader batch 就是一个 PP microbatch，每个都先走模型 preprocess”；**替代方案**是 schedule 内部 split tensor 并猜测 metadata。提交 `9228564523` 明说决定性标准是让 mask/varlen metadata 与每个 microbatch 同步生成，并与 gradient accumulation 保持独立。

loss 方面，**选中路线**是 sum-reduction 后除全局 valid-token 数，且 schedule `scale_grads=False`；**替代方案**是让 schedule 每次按 microbatch 数缩放梯度。提交 `3e1b843ec` 记录了后者与多次 gradient accumulation 结合时会将累积梯度变成近似指数移动平均；决定标准是语义与非 PP 等价，并避免反复遍历大模型梯度。

### ③ 实现思路与细节

token 预算分成两层：`num_pp_microbatches` 是一次 schedule call 中的 microbatch 数，`gradient_accumulation_steps` 是多少个完整 schedule calls 共享一次 optimizer step。Trainer 用每 microbatch token 数乘 PP microbatch 数得到每 DP rank 的一组 token，再从全局 train-step token 数反推 accumulation steps（`torchtitan/trainer.py:406`、`torchtitan/trainer.py:425`）。配置文档也把这两层分开（`torchtitan/config/configs.py:43`、`torchtitan/config/configs.py:48`；`torchtitan/config/configs.py:234`、`torchtitan/config/configs.py:238`）。

`train_step()` 先预取 `gradient_accumulation_steps × num_pp_microbatches` 个 batch，累加非 `IGNORE_INDEX` labels，并在 DP batch mesh 上求全局 valid-token 数（`torchtitan/trainer.py:785`、`torchtitan/trainer.py:806`）。每个 accumulation group 才单独搬到 device 并发起一次 forward/backward（`torchtitan/trainer.py:808`、`torchtitan/trainer.py:834`）。

PP 路径对每个 microbatch 都调 `model_parts[0].preprocess_inputs()`：只有拥有 first stage 的 rank 把 `inputs` 放入 `arg_mbs`，所有 rank 都传 `extra_kwargs`，只有 last-stage rank 构造 `target_mbs`（`torchtitan/trainer.py:737`、`torchtitan/trainer.py:754`）。这让 positions/masks 在各 stage 本地可见，也让 CP 在 schedule 前完成 mask 构造、输入切分与 SPMD annotation（`torchtitan/models/common/decoder.py:351`、`torchtitan/models/common/decoder.py:386`）。

schedule 的 loss adapter 只从 TorchTitan loss 返回值取标量，并对 single/multi 都设 `scale_grads=False`（`torchtitan/distributed/pipeline_parallel.py:317`、`torchtitan/distributed/pipeline_parallel.py:336`）。CrossEntropy 本身用 sum reduction，`BaseLoss` 再除 `global_valid_tokens`（`torchtitan/components/loss.py:57`、`torchtitan/components/loss.py:62`；`torchtitan/components/loss.py:298`、`torchtitan/components/loss.py:317`）。last-stage rank 求和所有 microbatch losses，其他 rank 返回 device 上的 `-1` sentinel（`torchtitan/trainer.py:756`、`torchtitan/trainer.py:772`）；CPU 单元测试专门固定了 sentinel 语义（`tests/unit_tests/cpu/test_trainer.py:18`、`tests/unit_tests/cpu/test_trainer.py:47`）。

finite gate 先只在拥有 loss 的 last-stage rank 上沿 loss mesh 求 MIN，再沿 PP group 传播给所有 stages（`torchtitan/trainer.py:858`、`torchtitan/trainer.py:876`）。metrics 默认由“最后 PP stage 的第一个 global rank”记录，ZBV 特例是 rank 0（`torchtitan/components/metrics.py:232`、`torchtitan/components/metrics.py:257`）。

### ④ 约束/边界

- `num_pp_microbatches` 必须大于 0（`torchtitan/trainer.py:121`、`torchtitan/trainer.py:125`）；它在 PP 关闭时被强制当成 1（`torchtitan/trainer.py:406`、`torchtitan/trainer.py:408`）。
- 每个 PP microbatch 的 token 数还必须整除启用 SP 时的 TP 因子与 CP 的 `2*cp` 因子（`torchtitan/trainer.py:292`、`torchtitan/trainer.py:305`）。
- PP+FSDP loss 当前可能意外回到 CPU，Trainer 有显式 TODO 并搬回 device（`torchtitan/trainer.py:768`、`torchtitan/trainer.py:771`）；Validator 复用同样 workaround（`torchtitan/components/validate.py:237`、`torchtitan/components/validate.py:244`）。
- V mapping 中 `DualPipeV` 与 ZBV 都把尾 stage 放到 rank 0 一侧（`torchtitan/distributed/pipeline_parallel.py:550`、`torchtitan/distributed/pipeline_parallel.py:567`），但 metrics helper 只对字符串 `ZBVZeroBubble` 特判 rank 0（`torchtitan/components/metrics.py:250`、`torchtitan/components/metrics.py:257`）。**知识库推断**：eager DualPipeV 的 console metrics 可见 rank 仍有接线缺口；当前 core 也没有它的 eager integration case，不能写成已验证。

### ⑤ 发展趋势（有锚点的推断）

PP+FSDP loss 的 device workaround 与 metrics 只特判 ZBV 都是当前源码可见的边界（`torchtitan/trainer.py:768`；`torchtitan/components/metrics.py:250`）。**推断**：继续扩大 V-shaped schedule 覆盖前，loss device/rank 所有权需从 schedule 名称特判收敛到 stage flags；源码没有承诺具体方案。

---

## 5. Stage-local 并行、FSDP 策略与 checkpoint

### ① 背景/问题

PP 只分层不会自动给层内参数分片，一个 rank 还可能拥有多个 virtual chunks。若先对完整模型包 TP/FSDP/compile 再剪枝，wrapper 与 stage 所有权容易错位；若 checkpoint 直接保存每 rank 的 optimizer index，不同 stages 的 `param_group[0]` 会同名碰撞。DCP 文档直接记录了这个 PP optimizer-state 冲突问题（`torchtitan/components/checkpointer/dcp.py:93`、`torchtitan/components/checkpointer/dcp.py:104`）。

### ② 为什么这么设计

**选中路线**是先得到 local chunks，再对每个 chunk 调模型自己的 `parallelize_fn`；**替代方案**是在 global model 上包装后切分。决定性标准是 optimizer/checkpoint/compile 看到的对象必须与 schedule 实际执行的 `stage.submod` 是同一 local part；源码甚至在 compile 可能返回新 module 后显式回写 `stage.submod`（`torchtitan/distributed/pipeline_parallel.py:106`、`torchtitan/distributed/pipeline_parallel.py:123`）。

checkpoint 选择 FQN-keyed flat state，**替代方案**是保留每个本地 optimizer/chunk 的位置索引。DCP 文档给出的判据是跨 PP ranks 和本 rank 多 virtual chunks 都必须消除 index 冲突（`torchtitan/components/checkpointer/dcp.py:95`、`torchtitan/components/checkpointer/dcp.py:114`）。

### ③ 实现思路与细节

以 Llama3 为例，每个 stage chunk 内的顺序是 sharding config/module parallelize，再 AC，再 per-block compile，最后 FSDP（`torchtitan/models/llama3/parallelize.py:40`、`torchtitan/models/llama3/parallelize.py:78`）。这说明 PP 与 TP/CP/FSDP/AC/compile 的组合发生在 local chunk 内，不是 schedule 内部特判。

DTensor 不能连 ProcessGroup 直接跨 PP stage 序列化，所以 TorchTitan 给 `PipelineStage` 注入 `get_mesh` callback：收到 plain tensor 后按 dim names 取当前 rank 的 local mesh，mesh layout 不匹配则返回 `None`（`torchtitan/distributed/pipeline_parallel.py:41`、`torchtitan/distributed/pipeline_parallel.py:60`）。`spmd_types` 路线另由 Trainer context 注册 dense/sparse meshes 并在 schedule 外激活 dense current mesh（`torchtitan/distributed/utils.py:397`、`torchtitan/distributed/utils.py:423`）。两者不应混为同一件事：`get_mesh` 是 PP/DTensor 边界重包装，current mesh 是 local SPMD runtime 语义。

FSDP 在 PP 下的 `default` 策略是 forward 后不 reshard，因为每个 pipeline microbatch 重复 all-gather 昂贵且难以 overlap（`torchtitan/distributed/fsdp.py:124`、`torchtitan/distributed/fsdp.py:132`）。同一 bool 传给 embedding 和 transformer blocks，末端 norm/head 默认也不 reshard，只有 `always` 强制打开（`torchtitan/distributed/fsdp.py:252`、`torchtitan/distributed/fsdp.py:265`；`torchtitan/distributed/fsdp.py:361`、`torchtitan/distributed/fsdp.py:366`）。这用更高参数常驻显存换取少一轮跨 microbatch 通信。

optimizer 在所有并行包装之后以 local `model_parts` 构建，checkpointer 也收同一 list（`torchtitan/trainer.py:533`、`torchtitan/trainer.py:544`；`torchtitan/trainer.py:564`、`torchtitan/trainer.py:577`）。`ModelWrapper` 把本 rank 各 parts 的 state dict 按 FQN 合成扁平 dict，load 时以 `strict=False` 交给每个 part（`torchtitan/components/checkpointer/base.py:115`、`torchtitan/components/checkpointer/base.py:142`）。`OptimizersContainer` 则为每个 local part 构建 optimizer instances，保存时合并 FQN-keyed flat optimizer states（`torchtitan/components/optimizer/optimizer.py:218`、`torchtitan/components/optimizer/optimizer.py:239`；`torchtitan/components/optimizer/optimizer.py:306`、`torchtitan/components/optimizer/optimizer.py:325`）。

### ④ 约束/边界

- `spmd_types` backend 可以与 PP 共存，但 debug SPMD **typechecking** 与 PP 显式互斥（`torchtitan/trainer.py:129`、`torchtitan/trainer.py:139`）。不能把“runtime mesh 已接线”写成“PP 跨 stage 已类型检查”。
- CUDA graphs 当前直接拒绝 PP，普通 `torch.compile` 则有 stage-local 组合测试（`torchtitan/trainer.py:165`、`torchtitan/trainer.py:173`；`tests/integration_tests/features.py:121`、`tests/integration_tests/features.py:127`）。
- Decoder weight tying 与 PP 不支持，在 model config update 阶段就失败（`torchtitan/models/common/decoder.py:163`、`torchtitan/models/common/decoder.py:176`；`tests/unit_tests/cpu/test_weight_tying.py:84`、`tests/unit_tests/cpu/test_weight_tying.py:91`）。
- DCP 假设各 chunk FQN 不冲突，并把这归因于正确 pipeline split（`torchtitan/components/checkpointer/dcp.py:110`、`torchtitan/components/checkpointer/dcp.py:114`）。但当前 `_split_module()` 对 `ModuleList` 重建新 list，会按存活顺序从 0 编号（`torchtitan/distributed/pipeline_parallel.py:507`、`torchtitan/distributed/pipeline_parallel.py:518`）。**知识库推断**：自定义 ModuleList 模型必须额外验证剪枝后 FQN 稳定与唯一，否则 checkpoint 的 flat-key 前提不成立。

### ⑤ 发展趋势（有锚点的推断）

PP+SPMD typechecking 的 guard 旁有显式 TODO：未来启用该组合（`torchtitan/trainer.py:129`、`torchtitan/trainer.py:136`）。**推断**：下一个正确性边界是给 stage 收发与 local current-mesh 之间建立可检查合同，而不是仅移除 guard；当前 TODO 没有给出完成设计。

---

## 6. 当前支持矩阵：guard + 模型协议 + 测试点

### ① 背景/问题

能构造 PP mesh 只证明 rank 数对得上，不证明模型 forward 容忍剪枝、attention metadata 能走拆分 backward、checkpoint FQN 无冲突或 metrics rank 可见。把“PP 可与 X 组合”理解成任意模型与 schedule 的笛卡尔积，会跳过真正的失败边界。

### ② 为什么这么设计

**选中路线**是用配置 guard 提前拒绝明知不可行组合，用模型 `pipelining_fn/parallelize_fn` 承担结构差异，再以具体 topology integration tests 保护点集；**替代方案**是一个声称所有并行维度正交的全局开关。决定性标准是组合必须在真实 model/schedule/metadata 上闭环，而不是只通过 rank 等式。这一设计哲学是对当前 guards 与测试结构的**知识库归纳**，不是上游原文引用。

### ③ 实现思路与细节

| 组合 | 当前证据 | 能证明什么 |
|---|---|---|
| PP-only 1F1B | 2-rank real-PG case（`tests/integration_tests/features.py:89`、`tests/integration_tests/features.py:95`） | 单 stage/rank 的基本 eager adapter |
| FSDP+PP | 1F1B 与 layers-per-stage 配方（`tests/integration_tests/features.py:96`、`tests/integration_tests/features.py:104`） | FSDP smart-default 与自动 stage 数的具体点 |
| TP+PP | 2D GPipe real-PG（`tests/integration_tests/features.py:105`、`tests/integration_tests/features.py:110`） | stage 传输 + 层内 TP |
| FSDP+TP+PP+checkpoint | 8-GPU save/load resume（`tests/integration_tests/features.py:111`、`tests/integration_tests/features.py:120`） | local parts optimizer/checkpoint 的 3D 接线 |
| FSDP+TP+PP+compile | 8-GPU real-PG（`tests/integration_tests/features.py:121`、`tests/integration_tests/features.py:127`） | stage-local compile，不是 CUDA graphs |
| looped PP | 4-rank Interleaved1F1B 与 layers-per-stage（`tests/integration_tests/features.py:128`、`tests/integration_tests/features.py:137`） | 每 rank 多 local chunks 的 loop mapping |
| FSDP+CP+PP+EP | DeepSeek V3 golden numerics（`tests/integration_tests/models.py:57`、`tests/integration_tests/models.py:65`） | CP preprocess + MoE local chunks 的一个 8-GPU topology |
| FSDP+TP+PP+EP | Qwen3.5 real-PG（`tests/integration_tests/models.py:117`、`tests/integration_tests/models.py:124`） | VLM/EP/TP stage adapter 的一个 8-GPU topology |
| Flex CP/Varlen + PP/EP | GPT-OSS 两个 real-PG cases（`tests/integration_tests/models.py:157`、`tests/integration_tests/models.py:170`） | full-backward schedule 可携带这些 metadata，不推广到 split-backward |

PP 训练和验证复用相同的 stage flags、per-microbatch preprocess 与 schedule，验证只把 `step()` 换成 `eval()`（`torchtitan/components/validate.py:203`、`torchtitan/components/validate.py:235`）。这是一个完整 adapter contract，但每一行测试只保护表中的模型、schedule 和 topology。

### ④ 约束/边界

| 失败边界 | 当前行为 | 证据 |
|---|---|---|
| 模型没有 `pipelining_fn` | Trainer 构建期 `RuntimeError` | `torchtitan/trainer.py:428`、`torchtitan/trainer.py:433` |
| single/multi schedule 与 stages/rank 不匹配 | metadata 阶段 `ValueError` | `torchtitan/distributed/pipeline_parallel.py:244`、`torchtitan/distributed/pipeline_parallel.py:256` |
| weight tying + PP | model config 阶段 `NotImplementedError` | `torchtitan/models/common/decoder.py:173`、`torchtitan/models/common/decoder.py:176` |
| CUDA graphs + PP | config 阶段 `ValueError` | `torchtitan/trainer.py:165`、`torchtitan/trainer.py:173` |
| `spmd_types` typechecking + PP | config 阶段 `ValueError` | `torchtitan/trainer.py:129`、`torchtitan/trainer.py:139` |
| zero-bubble/custom CSV + Flex mask | 配方在，core CI disabled | `tests/integration_tests/features.py:138`、`tests/integration_tests/features.py:176` |
| microbatch 少于 stage 数 | warning/bubble，不 fail | `torchtitan/distributed/pipeline_parallel.py:302`、`torchtitan/distributed/pipeline_parallel.py:309` |

> [!important] 当前证据的正确读法
> “FSDP+CP+PP+EP 有 golden test”不等于所有 MoE、attention backend、schedule 都支持这个笛卡尔积。当前对 zero-bubble/CSV、DualPipeV metrics、SPMD typechecking 的证据正好说明组合边界仍是模型和 schedule 专属的。

### ⑤ 发展趋势（有锚点的推断）

当前两个明确 TODO 分别是上游 split-backward 忽略非 tensor inputs（`tests/integration_tests/features.py:138`、`tests/integration_tests/features.py:149`）与 PP 下启用 SPMD typechecking（`torchtitan/trainer.py:134`）。**推断**：PP 支持面会通过补强 stage-boundary contract 扩展，而不是靠添加更宽松的“任意组合”开关；除这两个锚点外，仓库未给出 roadmap。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列的基准、页面责任与阅读顺序。
- [[10_torchtitan_parallel_dims_analysis]] —— PP rank 预算、mesh 轴和 loss mesh 的全局语义。
- [[11_torchtitan_fsdp_analysis]] —— `reshard_after_forward`、FSDP unit 与参数 all-gather 生命周期。
- [[12_torchtitan_tp_analysis]] —— stage-local TP/SP 和 DTensor 跨 stage 重包装边界。
- [[13_torchtitan_cp_analysis]] —— 每 PP microbatch 的 mask 构造、CP 切分与 token 整除条件。
- [[15_torchtitan_ep_analysis]] —— MoE stage chunk 内的 sparse mesh、dispatcher 与 PP/EP 组合。
- [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] —— 复用 stage-plan helpers 但导出显式 forward/backward graph 的 Graph PP 路径。
