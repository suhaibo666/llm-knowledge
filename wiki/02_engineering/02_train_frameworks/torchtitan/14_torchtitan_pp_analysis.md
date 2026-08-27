# Pipeline Parallel：stage 适配层、微批次协议与组合边界

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：TorchTitan 的 PP 核心不是另一套流水调度器，而是一层 **stage 适配协议**：它把完整模型复制并裁成 rank-local chunks，在每个 chunk 上继续施加 FSDP/TP/CP/compile，再把模型预处理后的微批次和 SPMD 上下文交给上游 `torch.distributed.pipelining` schedule。理解当前实现的关键因此不是背 GPipe/1F1B 的气泡公式，而是分清三种所有权：TorchTitan 拥有 stage 物化、rank 映射、输入预处理与组合策略；PyTorch 拥有 action/P2P 调度；模型自己必须容忍被裁剪后的 forward。
>
> 主要源文件：`torchtitan/distributed/pipeline_parallel.py`、`torchtitan/trainer.py`、`torchtitan/config/configs.py`、`torchtitan/distributed/fsdp.py`。

---

## 1. 先纠正旧版知识

旧页基于 TorchTitan `cf3c4312` 与 PyTorch 2.9.1，把篇幅主要放在 PyTorch schedule 的 action 表、P2P batching、`AsyncCollectiveTensor` wait、zero-bubble 拆分 backward 与 DualPipeV 时序上。那些内容属于**独立的上游 PyTorch 基线**，不是当前 TorchTitan 源码的行号空间；本次审计没有用当前 TorchTitan 行号替换旧 PyTorch 行号。

当前配置注释把 schedule 名字空间明确链接到 PyTorch commit `de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0`，而 TorchTitan 运行时只通过 `get_schedule_class()` 解析名字（`torchtitan/config/configs.py:218-224`、`torchtitan/distributed/pipeline_parallel.py:292-300`）。因此本页保留的上游结论只有边界：**schedule 决定 action/P2P 时序；TorchTitan 负责给它 stages 与预切好的 microbatches**。旧页的气泡公式、延迟 wait 和 DualPipeV 局部 GEMM 时序不再当作当前页的 source-faithful 机制断言。

> [!deprecated] 旧 `split_points` / shape-inference 调用链已失效
> 当前 `ParallelismConfig` 的真实切分入口是 `module_fqns_per_model_part` 或 `pipeline_parallel_layers_per_stage`，没有 `split_points` 字段（`torchtitan/config/configs.py:189-215`）。配置注释 `:222-224` 仍出现 `split_points`，但当前核心实现没有消费它；这是文档残留，不应据此配置。当前 `PipelineStage` 构造也不再传旧页所写的示例输入做 shape inference，而是传 `model_chunk/stage_idx/num_stages/device/group/get_mesh`（`torchtitan/distributed/pipeline_parallel.py:619-632`）。

> [!deprecated] microbatch 不再由 schedule 从大 batch 内部切
> Trainer 的 dataloader 每次已经产出一个 PP microbatch；训练步先连续取 `num_pp_microbatches` 个 batch，再组成一个 schedule 调用（`torchtitan/trainer.py:785-820`）。旧页“schedule 用 split spec 切 input/label”的调用链不适用于当前 Trainer。

---

## 2. TorchTitan 与 PyTorch 的责任边界

当前 eager PP 的真实构建链是：

```text
Trainer(meta device 上 build 完整模型)
  -> 检查 ModelSpec.pipelining_fn
  -> model-specific pipelining_fn（通常是 pipeline_llm / pipeline_vlm）
  -> 计算 virtual stage 数与每 stage 的 FQN
  -> deepcopy + prune，物化本 rank 的 model chunks
  -> 每个 chunk 调 model-specific parallelize_fn
  -> 用 chunks 构造 PyTorch PipelineStage 与 schedule
  -> Trainer 为 chunks 建 optimizer/checkpoint，并执行 schedule.step
```

模型先在 meta device 上完整构造并校验 module protocol（`torchtitan/trainer.py:350-365`）。只有 `ModelSpec.pipelining_fn` 非空的模型才能启用 PP；否则 Trainer 立即报错（`torchtitan/trainer.py:426-453`）。PP 函数返回 `schedule/model_parts/has_first_stage/has_last_stage` 后，完整 `model` 被删除，此后初始化、优化器与 checkpoint 都只持有本 rank 的 `model_parts`（`torchtitan/trainer.py:454-464`、`:533-577`）。

这条边界解释了为什么 TorchTitan 可以跟随上游新增 schedule：核心不硬编码一张 schedule 枚举表，而是解析上游 class；但 stage 如何切、chunk 上先施加哪些 SPMD 并行、输入怎样变成 schedule 参数，仍是 TorchTitan 的责任（`torchtitan/distributed/pipeline_parallel.py:65-141`）。

Graph PP 是另一条实验性执行路径；它复用部分 stage/schedule helper，但显式导出 forward/backward graph，不属于本页 eager Trainer 机制。见 [[27_torchtitan_graph_trainer_compiler_runtime_analysis]]。

---

## 3. stage 如何从完整模型物化出来

### 3.1 virtual stage 数先由 schedule 类别约束

`_get_pipeline_metadata()` 先把 schedule 分成 `PipelineScheduleSingle` 与 multi-stage 两类。若显式给 `pipeline_parallel_layers_per_stage`，virtual stage 数按模型层数加首尾权重后向上取整，且必须能被 PP rank 数整除；single-stage schedule 要求每 rank 恰好一个 stage，multi-stage schedule 至少每 rank 两个 stage（`torchtitan/distributed/pipeline_parallel.py:194-257`）。若未显式给定，默认就是 single 每 rank 1 个、multi 每 rank 2 个（`:258-264`）。

`pipeline_parallel_degree` 因而始终表示**物理 rank 数**，不是 stage 数；looped schedule 的 stage 数由每 rank 的 virtual chunks 继续放大（`torchtitan/config/configs.py:182-186`）。

### 3.2 FQN 列表是当前切分契约

用户可用 `module_fqns_per_model_part` 精确列出每个 chunk 保留的模块（`torchtitan/config/configs.py:189-197`）。未指定时，LLM helper 自动在 `tok_embeddings`、`layers.N`、`norm`、`lm_head` 之间分配；首尾权重把 embedding 与 norm/head 当成额外层成本，且会拒绝空 stage 或权重大于最小 stage 容量的配置（`torchtitan/distributed/pipeline_parallel.py:357-419`）。实际分配让首 stage 拿 embedding、末 stage 拿 norm/head，中间只拿 decoder layers（`:421-467`）。

VLM 不是把 vision encoder 单独建成一条流水。`pipeline_vlm()` 把 `vision_encoder` 插入第一 stage 的 FQN 列表，再委托 `pipeline_llm()`；源码同时提醒这一额外负载没有进入自动权重模型，重型 encoder 需要用 first-stage-less-layers 参数重新平衡（`torchtitan/distributed/pipeline_parallel.py:144-185`）。

### 3.3 物化方式是 deepcopy 后裁剪，而非原模型原地分段

每个本地 stage 都从完整模型 `deepcopy` 一份，再删除未选中的 `ModuleDict/ModuleList` 项，或把不需要的普通子模块设为 `None`（`torchtitan/distributed/pipeline_parallel.py:470-529`）。所以模型 forward 必须把缺失模块当恒等传递；公共 Decoder 正是以“无 embedding 就直接接收上一 stage tensor、无 norm/head 就直接返回 hidden state”的方式实现（`torchtitan/models/common/decoder.py:284-308`）。

这也形成明确限制：裁剪 helper 不支持嵌套的 ModuleDict/ModuleList，模型的 forward 与初始化必须容忍被删除的层（`torchtitan/distributed/pipeline_parallel.py:580-589`）。自定义模型仅仅提供 FQN 列表还不够，必须满足这套 stage-safe 协议。

---

## 4. rank 到 stage 的映射与 schedule 构造

### 4.1 两种 rank 映射

TorchTitan 只对 `ZBVZeroBubble` 和 `DualPipeV` 使用 V-shaped 映射；其余 schedule 使用 loop 映射（`torchtitan/distributed/pipeline_parallel.py:532-553`）。

| 映射 | 本 rank 获得的 stage | 当前约束 |
|---|---|---|
| loop | `pp_rank + s * pp_degree` | stage 总数必须整除 PP degree |
| V | rank `r` 拿前半部 `r` 与后半部镜像 stage | 每 rank 必须恰好 2 个 stage |

断言与具体映射在 `torchtitan/distributed/pipeline_parallel.py:554-567`。因此 ZBV/DualPipeV 不只是“换 action 表”：它们还改变末 stage 落在哪个物理 rank。

### 4.2 stage 先切，再在每个 chunk 上施加 SPMD 并行

`_pipeline_module_split()` 先为本 rank 建 `PipelineStage` 和 chunks；随后 `pipeline_llm()` 对每个 chunk 调模型自己的 `parallelize_fn`，因此 FSDP/TP/CP/AC/compile 都只看见本地 stage。若 parallelize 或 compile 返回替换后的 module，还必须回写 `stage.submod`，否则 schedule 会执行旧 chunk（`torchtitan/distributed/pipeline_parallel.py:96-124`）。

PP 传输的是普通 tensor，DTensor 自身不能连同 ProcessGroup 跨 stage 序列化。为让接收端重建本地布局，TorchTitan 给 `PipelineStage` 注入 `get_mesh` 回调；回调按维度名取当前 rank 的 mesh，并校验 mesh layout（`torchtitan/distributed/pipeline_parallel.py:41-62`、`:625-632`）。这是当前 PP 与 TP/CP/SPMD 组合的关键边界。

### 4.3 schedule 控制面

| 配置 | 当前含义 |
|---|---|
| `pipeline_parallel_schedule` | 交给上游 `get_schedule_class()` 解析；TorchTitan 不维护完整名字枚举 |
| `pipeline_parallel_schedule_csv` | 非空时改用 `_PipelineScheduleRuntime`，校验文件存在后加载 CSV |
| `num_pp_microbatches` | 每个数据并行 rank、每次梯度累积迭代交给 schedule 的 microbatch 数 |

CSV/普通 schedule 的选择与加载在 `torchtitan/distributed/pipeline_parallel.py:290-300,342-352`，字段契约在 `torchtitan/config/configs.py:218-238`。若 microbatch 数小于全局 stage 数，当前实现只发 bubble warning，不拒绝运行（`torchtitan/distributed/pipeline_parallel.py:302-309`）。

TorchTitan 给上游 schedule 的 loss wrapper 只返回标量 loss，并设置 `scale_grads=False`；全局有效 token 归一化由自身 loss 路径负责，不让 schedule 再按 microbatch 数缩放梯度（`torchtitan/distributed/pipeline_parallel.py:317-336`）。

---

## 5. Trainer 的当前 microbatch / preprocess / SPMD 协议

### 5.1 一个 dataloader batch 就是一个 PP microbatch

`training.num_tokens_per_microbatch_per_dp_rank` 定义一次模型 forward 的 token slots；一个梯度累积迭代处理它乘 `num_pp_microbatches` 的 token 数（`torchtitan/config/configs.py:36-48`、`torchtitan/trainer.py:406-425`）。Trainer 为每个梯度累积组先取满 `num_pp_microbatches` 个 dataloader batch，统计全局有效 token，再逐组发起一次 forward/backward（`torchtitan/trainer.py:785-834`）。

所以 `num_pp_microbatches` 与 `gradient_accumulation_steps` 是两层不同循环：前者在一次 schedule 内填流水，后者让多个完整 schedule 调用共享一次 optimizer step（`torchtitan/trainer.py:785-850`）。

### 5.2 每个 rank 都 preprocess，每个 microbatch 都 preprocess

PP 分支逐个 microbatch 调本 rank `model_parts[0].preprocess_inputs()`。只有持有 first stage 的 rank把 `inputs` 放入 `arg_mbs`，只有持有 last stage 的 rank收集 `target_mbs`；`extra_kwargs` 则每个 rank都传给 schedule（`torchtitan/trainer.py:730-766`）。这允许 stage-local forward 获得 positions/masks 等非首 stage 元数据，而不是假定所有信息都随 hidden state 自动传播。

公共 Decoder 的预处理顺序是：构建 attention mask、按 CP 规则切输入、在 `spmd_types` 后端标注输入布局，最后拆出 `input/labels/extra_kwargs`（`torchtitan/models/common/decoder.py:351-386`）。Trainer 再在 `train_context()` 内调用 schedule；使用 `spmd_types` 时，该 context 注册 dense/sparse SPMD meshes 并把当前 mesh 设为 dense mesh，若显式打开 typecheck 才额外进入 typechecking context（`torchtitan/trainer.py:579-585`、`torchtitan/distributed/utils.py:397-425`）。

### 5.3 loss 只在拥有末 stage 的 rank有意义

schedule 收集每个 microbatch 的 loss；拥有末 stage 的 rank求和并搬回当前 device，其余 rank返回占位 `-1`（`torchtitan/trainer.py:756-772`）。验证路径复用同一批 stage flags 与 preprocess 协议，只把 `step()` 换为 `eval()`（`torchtitan/components/validate.py:203-244`）。

> [!warning] DualPipeV 的 loss 可见 rank 仍需补验证
> stage 映射把 DualPipeV 与 ZBV 都视为 V shape（`torchtitan/distributed/pipeline_parallel.py:550-567`），但 metrics helper 只对 `ZBVZeroBubble` 特判 rank 0，其他名字仍按最后物理 PP rank记录（`torchtitan/components/metrics.py:232-257`）。核心 integration tests 也没有 eager DualPipeV 用例。当前证据不足以把旧页“DualPipeV 指标自然可见”继续写成已验证结论。

---

## 6. PP + FSDP：默认不 reshard 是跨 microbatch 的通信策略

`fsdp_reshard_after_forward` 支持 `default/always/never`（`torchtitan/config/configs.py:147-159`）。真正的 PP smart default 是：PP 关闭时 `default -> True`，PP 开启时 `default -> False`，因为每个 microbatch forward 后立刻 reshard 会在 backward 前引入昂贵且难重叠的重复 all-gather（`torchtitan/distributed/fsdp.py:112-136`）。

该布尔值实际传给 embedding 与每个 transformer block 的 `fully_shard()`；末尾 norm+lm_head 默认本来也保持不 reshard，只有 `always` 强制开启（`torchtitan/distributed/fsdp.py:238-265`、`:267-368`）。因此：

- `default`：PP stage 的参数跨 forward/backward 窗口保持 materialized，省每 microbatch all-gather，代价是更高峰值显存。
- `always`：更积极释放参数显存，但为每个流水 microbatch 付额外通信。
- `never`：显式选择常驻；语义与 PP 下的 default 相同，但不随 PP 开关改变。

这不是 schedule 自己的优化，而是 model chunk 在进入 schedule 前已经获得的 FSDP policy；`parallelize_fn` 接收完整 `ParallelismConfig` 与 `parallel_dims.pp_enabled`（`torchtitan/distributed/pipeline_parallel.py:106-123`、`torchtitan/distributed/fsdp.py:168-235`）。

---

## 7. 当前组合边界与测试矩阵

### 7.1 代码级门禁

- `num_pp_microbatches` 必须大于 0（`torchtitan/trainer.py:121-125`）。
- 每个 PP microbatch 的 token 数必须能被启用 SP 时的 TP 度数与 CP 的 `2 * cp` 因子整除（`torchtitan/trainer.py:292-305`）。
- 当前 CUDA graphs 明确拒绝 PP（`torchtitan/trainer.py:165-173`）。
- `spmd_types` 后端可与 PP 运行，但 **SPMD typechecking** 暂不支持 PP；配置会在初始化阶段拒绝该组合（`torchtitan/trainer.py:129-139`）。
- 公共 Decoder 的 weight tying 与 PP 组合直接抛 `NotImplementedError`（`torchtitan/models/common/decoder.py:163-176`）。
- zero-bubble / 拆分 backward 类 schedule 当前会把转发的 FlexAttention `BlockMask` 当 tensor 访问 `requires_grad`；因此核心 CI 中 InterleavedZeroBubble、ZBV 与 custom CSV 用例被禁用，1F1B/GPipe/Interleaved1F1B 不受该问题影响（`tests/integration_tests/features.py:138-176`）。
- PP+FSDP loss 暂时会意外回到 CPU，Trainer 与 Validator 都保留了搬回 device 的 workaround/TODO（`torchtitan/trainer.py:768-772`、`torchtitan/components/validate.py:237-244`）。

### 7.2 已提交测试能证明什么

| 维度 | 当前仓库中的证据 | 结论边界 |
|---|---|---|
| eager 基础 schedule | 2-rank 1F1B、FSDP+PP 1F1B、TP+PP GPipe、4-rank Interleaved1F1B 被注册为 real-PG tests（`tests/integration_tests/features.py:89-136`） | 覆盖 full-backward schedule 与 looped stage mapping |
| 3D 工程组合 | FSDP+TP+PP checkpoint save/load 与 compile 均有 8-GPU real-PG test（`tests/integration_tests/features.py:111-127`） | 证明这些具体 debugmodel recipe 接线，不等价于任意模型/形状都支持 |
| MoE 多维组合 | DeepSeek V3 有 FSDP+CP+PP+EP golden numerics；Qwen3.5 有 FSDP+TP+PP+EP real-PG（`tests/integration_tests/models.py:57-66,117-124`） | PP 可与 CP/TP/EP 分别组合，但覆盖的是具体 topology |
| attention / optimizer 变体 | GPT-OSS 有 PP+FSDP+CP+EP+SAC 与 PP+FSDP+EP+Varlen，Kimi K2.7 有 DistMuon+PP+FSDP+EP（`tests/integration_tests/models.py:157-179`） | 说明模型专用 parallelize/pipeline contract 正在被组合测试 |
| 拆分 backward schedules | InterleavedZeroBubble、ZBV、custom CSV 配方存在，但核心 integration cases 当前 `disabled=True`（`tests/integration_tests/features.py:138-176`） | 不能把“配置可构造”升级成“当前默认 attention 下已回归通过” |
| eager DualPipeV | 核心 PP 只含 V-shaped rank 映射分支，没有对应 eager recipe/test（`torchtitan/distributed/pipeline_parallel.py:550-567`） | live 接线存在，验证覆盖缺口仍在 |

测试矩阵的正确读法是“存在一组受保护的组合点”，不是“六维并行任意笛卡尔积均受支持”。尤其 typechecking、CUDA graphs、attention metadata 与模型是否实现 `pipelining_fn` 都会在更早阶段缩小可用空间。

---

## 8. 一次训练步的最短心智模型

```text
构建期：完整 meta model
  -> 按 FQN deepcopy/prune 成本 rank chunks
  -> chunk 内施加 TP/CP/FSDP/AC/compile
  -> 建 PipelineStage(get_mesh=...) 与 upstream schedule

每个 optimizer step：
  -> 取 gradient_accumulation_steps 组数据
  -> 每组含 num_pp_microbatches 个独立 dataloader batch
  -> 每 microbatch 调 model.preprocess_inputs
  -> first stage 提供 arg_mbs，last stage 提供 target_mbs
  -> 在 dense SpmdContext 中 schedule.step
  -> last-stage rank 汇总 loss；所有 stage 的梯度一起进入 optimizer
```

这条链同时解释三个常见误区：PP microbatch 不是 Trainer 外部大 batch 的二次切片；SPMD parallelism 不是在跨 stage 传输后才补做；schedule 名字改变 action 时序，但不会替模型解决 stage-safe forward、FQN 切分或 FSDP reshard 策略。

---

## 9. 小结

- 当前 TorchTitan PP 是 stage/build/data adapter，上游 PyTorch pipelining 才是 action 与 P2P schedule 所有者。
- stage 由 FQN 列表驱动，以 deepcopy+prune 物化；single/multi schedule 决定每 rank 的默认 virtual stage 数，ZBV/DualPipeV 还切换为 V-shaped rank 映射。
- 每个 chunk 在进入 schedule 前已完成模型专用的 TP/CP/FSDP/compile；`get_mesh` 回调负责跨 PP tensor 边界后恢复布局解释。
- Trainer 自己收集独立 microbatches，并逐个执行 `preprocess_inputs`；只有 first/last stage 分别提供 model args/targets，但 stage-local kwargs 与 SPMD context 贯穿所有 rank。
- PP 下 FSDP default 不 reshard，以显存换掉逐 microbatch、难重叠的 all-gather；这是当前最重要的 PP+FSDP 策略差异。
- 1F1B/GPipe/Interleaved1F1B 及若干多维组合有 real-PG/golden 测试；zero-bubble/custom CSV 受 FlexAttention 非 tensor metadata 阻塞，eager DualPipeV 仍缺核心回归。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 系列入口、基线与功能树。
- [[10_torchtitan_parallel_dims_analysis]] —— PP rank 轴如何进入 dense/sparse mesh 与 loss mesh。
- [[11_torchtitan_fsdp_analysis]] —— `reshard_after_forward`、FSDP unit 与参数存储窗口。
- [[12_torchtitan_tp_analysis]] —— stage-local TP/SP 与跨 stage DTensor 布局边界。
- [[13_torchtitan_cp_analysis]] —— `preprocess_inputs` 中的 CP 切分、mask 与 token 整除约束。
- [[15_torchtitan_ep_analysis]] —— 已测试 MoE 多维组合中的 EP/EDP/dispatcher 路径。
- [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] —— Graph PP 的显式 forward/backward graph、额外门禁与 runtime。
