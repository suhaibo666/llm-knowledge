---
title: "TorchTitan Trainer：把完整配方提交为一个可恢复优化步"
---

# TorchTitan Trainer：把完整配方提交为一个可恢复优化步

> **论点式副标题**：`Trainer` 不是“模型外面的一层 for-loop”，而是把 Python full configuration 解析成一张有依赖顺序的运行时状态图，再以“先形成完整优化步、验证数值有限、协调 checkpoint staging，最后才修改参数”的边界提交它。
>
> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **本页回答**：一次标准 Trainer run 如何从配方走到 mesh、模型物化、组件组装、训练步、验证、checkpoint 与清理；哪些顺序是运行时不变量；哪里会失败。
> **Sibling 边界**：配置/模型协议的内部所有权见 [[02_engineering/02_train_frameworks/torchtitan/04_torchtitan_config_model_protocol_analysis|配置与模型协议]]；并行轴、数据恢复、checkpoint 事务、CUDA Graph/compile 的算子机制分别留给对应专项页。本页只解释它们如何被 Trainer 编排。

---

## 1. Overview

### 1.1 背景、问题与 thesis

大模型训练入口同时面对两类变化：上层 recipe 会替换模型、并行布局和组件实现，下层执行却必须维持稳定的生命周期，例如 optimizer 不能绑定并行化前的参数、checkpoint 不能在训练状态建立前构造、一个梯度累积步也不能只消费半组 microbatch。当前 TorchTitan 因而把 launcher 压薄，把稳定的控制顺序集中到 `Trainer`，而不是让每个 recipe 自写训练循环；`train.main()` 只做解析、构造、选择 seed-checkpoint/训练分支与收尾（`torchtitan/train.py:17-68`）。

本页的核心判断是：

1. **配置是构造计划，不是运行时对象**：`ConfigManager` 先得到完整、带类型的 `Trainer.Config`，`Configurable.build()` 才把各子配置绑定到真实组件（`torchtitan/config/manager.py:19-46`、`torchtitan/config/configurable.py:134-160`）。
2. **初始化顺序编码了状态转换**：模型先在 meta 上存在，随后被切 PP/施加 SPMD、AC、compile、FSDP，再物化和初始化；optimizer 看到的只能是最终参数对象（`torchtitan/trainer.py:334-365`、`torchtitan/trainer.py:426-493`、`torchtitan/trainer.py:533-543`）。
3. **`train_step()` 的提交边界是 optimizer update**：Trainer 先收齐本步数据并计算全局有效 token，再前后向、裁剪梯度与执行跨 rank finite gate，最后才等待 staging 并 step optimizer/scheduler（`torchtitan/trainer.py:774-889`）。
4. **外层循环负责恢复与副作用顺序**：checkpoint load 只在循环前发生；每步是 train → save 调用 → 可选 validation → profiler → 首步缩短通信 timeout（`torchtitan/trainer.py:943-997`）。

### 1.2 概念表：谁拥有决策，谁只消费结果

| 层 | 当前所有权 | Trainer 的动作 | 本页不展开 |
|---|---|---|---|
| full configuration | Python recipe + `ConfigManager` | 取得完整 `Trainer.Config` 并调用其 `build()` | 字段遍历、override 语义见 04 页 |
| distributed/mesh | `CommConfig` + `ParallelismConfig` | 初始化 PG，建立 `ParallelDims`，按名字取 `batch/loss/pp/...` mesh | 轴乘积与 mesh 展平见 10 页 |
| model lifecycle | `ModelSpec` + `BaseModel.Config` | meta build → override 后的协议校验 → parallelize/pipeline → materialize/init | sharding、AC、FSDP、compile 算法见专项页 |
| train state | Trainer + optimizer/scheduler/dataloader | 维护 `step`、`ntokens_seen`，形成完整优化步 | 数据 packing 与优化器内部算法 |
| persistence | checkpointer | 注入整张状态图，循环前 load、步后 save、优化前协调 staging | DCP 事务与 state-dict adapter 见 03 页 |
| observability/eval | metrics/profiler/validator + structured logger | 按频率收集规约结果、按 rank 记录 step/span，在独立临时 dataloader 上验证 | logger 后端、trace handler 和模型专用 validator |

### 1.3 关键图：构造状态图，再重复提交优化步

```text
Python recipe
  -> ConfigManager: load function -> typed CLI override -> validate
  -> Trainer.Config.build()
  -> Trainer.__init__
       distributed/ParallelDims
       -> model config update -> override -> meta model
       -> PP split or parallelize(SPMD -> AC -> compile -> FSDP)
       -> to_empty -> init_states
       -> optimizer -> scheduler -> tokenizer/data -> checkpointer
       -> SPMD context / CUDA-Graph fwd-bwd wrapper -> validator
  -> Trainer.train
       load once
       -> [fetch complete step -> valid-token reduce -> fwd/bwd
           -> grad/finite gate -> wait staging -> optimizer/scheduler
           -> save request -> optional validate] * N
  -> close components -> success-only process-group teardown
```

图中的 Llama 代表链不是抽象猜测：其 `parallelize_llama()` 明确按模型并行、AC、per-block compile、FSDP 的次序应用，且 degree=1 时仍调用 FSDP 以安装 mixed-precision policy（`torchtitan/models/llama3/parallelize.py:23-78`）。不同模型可替换 `parallelize_fn`，但 Trainer 的 meta→parallelize→materialize 外层顺序不变。

### 1.4 Quick Start：只选配方，不把配方重新抄成 CLI

```bash
NGPU=4 \
MODULE=torchtitan_recipes.tests.features \
CONFIG=llama3_debugmodel_fsdp2_cp2 \
./run_train.sh
```

仓库把 run 定义成“返回完整 `Trainer.Config` 的 Python 函数”，并要求 launcher 的 `NGPU` 与配置中的并行度乘积相符（`torchtitan/config/README.md:3-19`）。`run_train.sh` 正常路径使用 `torchrun`；`COMM_MODE=fake_backend` 则以单进程 fake process group、强制一步来验证配置/模型装配，而不是做数值训练（`run_train.sh:14-19`、`run_train.sh:34-45`）。

> [!important] 追踪一次 run 的最短方法
> 先读 `torchtitan/train.py:28-68` 找入口分支，再顺着 `torchtitan/trainer.py:274-637` 看构造状态图，最后看 `torchtitan/trainer.py:774-997` 的单步与外层循环。不要从某个旧 `train_configs/*.toml` 或一长串 CLI flag 反推当前控制流。

### 1.5 可追踪调用链 locator

| 要回答的问题 | 当前入口 |
|---|---|
| recipe 如何变成对象？ | `torchtitan/config/manager.py:34-46`、`torchtitan/config/configurable.py:134-160` |
| mesh 在哪里进入 Trainer？ | `torchtitan/trainer.py:283-318`、`torchtitan/trainer.py:628-637` |
| 模型何时并行化/物化？ | `torchtitan/trainer.py:334-365`、`torchtitan/trainer.py:390-493` |
| optimizer/data/checkpoint 何时构造？ | `torchtitan/trainer.py:533-577` |
| 一个优化步在哪里提交？ | `torchtitan/trainer.py:774-940` |
| 恢复、保存、验证在哪里调度？ | `torchtitan/trainer.py:943-997` |
| 正常与异常如何收尾？ | `torchtitan/train.py:41-68`、`torchtitan/trainer.py:1015-1023` |

---

## 2. 控制面交接：full configuration 到 Trainer

### 2.1 背景与问题

如果 launcher 同时负责解析几十个 flag、实例化组件和执行训练，那么每新增一种 optimizer/checkpointer/model 都会修改全局入口。当前设计要解决的是：如何让 recipe 自由组合实现，同时让训练生命周期只有一个权威。

### 2.2 为什么选择 Python full configuration

选中路线是“recipe 函数返回完整 typed config；每个组件 config 构造自己的 owner”。明显替代方案是 TOML/巨型 `JobConfig` 加中央工厂，或由 CLI 字符串注册每种实现。提交 `9810191bbb0385d8ac91cd31abbc7f4156a0d462` 的正文把选择依据写得很清楚：旧设计封装差、单体配置泄漏到各处且难以迭代实验；Python 的语言级组合也避免每个新组件先注册字符串名字。代价是配置成为可执行 Python，序列化与理解成本更高。

### 2.3 当前精确调用链

1. `ConfigManager.parse_args()` 先抽出 `--module/--config`，调用 recipe，再以返回对象的真实类型交给 tyro 施加剩余 override，优先级是 CLI 高于 recipe 默认值（`torchtitan/config/manager.py:19-46`）。
2. 短名会在 models、experiments、RL examples 下搜索；完全限定模块则先尝试追加 `.config_registry`，再直接导入（`torchtitan/config/manager.py:94-140`）。
3. 找不到可调用的 config function 会在 build 之前失败，并列出可用函数（`torchtitan/config/manager.py:143-159`）。
4. `train.main()` 初始化 structured logger 后调用 `config.build()`；`Configurable.Config.build()` 复制 config，并把只有运行时才存在的 kwargs 传给 owner（`torchtitan/train.py:28-45`、`torchtitan/config/configurable.py:134-160`）。
5. `Trainer.Config` 聚合 profiler、metrics、tokenizer、data、optimizer、scheduler、checkpoint、AC、compile、comm、validator 和 loss 等 typed sub-config；`model_spec` 被明确排除出 CLI 并由 recipe 设置（`torchtitan/trainer.py:65-113`）。

更深的 `traverse()`、模型子配置所有权和 override pass 属于 [[02_engineering/02_train_frameworks/torchtitan/04_torchtitan_config_model_protocol_analysis|配置与模型协议]]，本页只保留它们对生命周期的影响。

### 2.4 约束与失败边界

- `--module` 与 `--config` 当前都必填；缺失不是回退到隐式默认，而是直接 `ValueError`（`torchtitan/config/manager.py:48-92`）。
- build 的 runtime kwargs 若与 config 字段重名会失败，避免“配置值”和“调用值”同时成为权威（`torchtitan/config/configurable.py:153-160`）。
- 配置构造期就拒绝 PP microbatch 非正数、SPMD typechecking+PP、SPMD typechecking+FlexAttention SAC，以及未 compile 模型却选择 MemoryBudgetAC 等组合（`torchtitan/trainer.py:115-163`）。
- CUDA Graph 默认开启，但 PP 或带 CPU 同步的 EP dispatcher 会在配置验证期失败，用户必须选择受支持 dispatcher 或显式禁用 graph（`torchtitan/trainer.py:165-196`；对应 CPU tests 在 `tests/unit_tests/cpu/test_config_manager.py:164-241`）。

### 2.5 有锚点的演进

当前配置文档已冻结 `--section.option` 的增长，并把这些 override 定位为兼容层；目标是最终只保留 `--module/--config`，甚至移除 tyro。旧 per-model `config_registry.py` 仍工作，但计划迁到 `torchtitan_recipes`（`torchtitan/config/README.md:54-80`）。

> [!deprecated] 已失效的启动心智模型
> “TOML/模型目录 registry + 大串 CLI 是配置权威”不再成立。当前权威是 recipe 返回的完整 Python `Trainer.Config`；CLI override 只是仍保留的兼容面（`torchtitan/config/README.md:3-26`、`torchtitan/config/README.md:54-68`）。

---

## 3. 初始化：从 mesh 与 meta 模型到可优化状态图

### 3.1 背景与问题

完整模型、分片参数、optimizer state 和 checkpoint state 不是可任意交换顺序的对象。同一个 rank 必须先知道自己属于哪些 mesh，模型必须先获得 sharding/包装，再按最终布局分配存储；否则会先物化单卡完整权重，或让 optimizer 引用随后被 wrapper 替换的参数。

### 3.2 为什么选择“先描述、后包装、最后物化”

当前路线把模型在 meta device 上构造为无真实存储的结构，再由模型专属 `parallelize_fn` 施加并行与执行优化，最后 `to_empty()` 到目标设备并初始化。明显替代方案是先在 GPU/CPU 初始化完整模型再切分。**知识库推断**：源码没有单句宣言比较二者，但 `parallelize_llama()` 明确建议输入位于 meta，否则模型必须先能装入单机内存（`torchtitan/models/llama3/parallelize.py:33-39`），而 Trainer 的真实顺序是 meta build 在前、materialize 在后（`torchtitan/trainer.py:351-365`、`torchtitan/trainer.py:426-493`）；判据就是避免完整未分片存储成为峰值前提，并让初始化遵循最终 DTensor/包装布局。

### 3.3 当前精确调用链

**第一段：设备、distributed 与 mesh。** Trainer 先设置 local device，再调用 `dist_utils.init_distributed()` 和 `ParallelDims.from_config()`（`torchtitan/trainer.py:283-290`、`torchtitan/trainer.py:628-637`）。`ParallelDims` 从 config 取 DP/CP/TP/PP/EP degree，验证 `dp_replicate * dp_shard * cp * tp * pp == world_size`，并额外要求 EP 整除 sparse region（`torchtitan/distributed/parallel_dims.py:69-128`）。mesh 是按需构造/缓存的；Trainer 随后以语义名获取 `batch` mesh 来推导 DP degree/rank（`torchtitan/trainer.py:314-318`）。轴与 mesh 细节见 [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|ParallelDims]]。

**第二段：模型配置、override 与 meta build。** job-level 配置先通过 `model_config.update_from_config()` 写入模型配置；override 必须在它之后、所有组件 build 之前运行，然后再次验证 CUDA Graph 组合（`torchtitan/trainer.py:334-350`）。模型在 meta/default dtype context 中 build，并在并行 wrapper 介入前验证整个子模块树满足 Module protocol（`torchtitan/trainer.py:351-365`；失败行为定义于 `torchtitan/protocols/model.py:81-99`）。

**第三段：两条并行化/物化路径。**

- 无 PP：除 seed-checkpoint 模式外，Trainer 调用 `model_spec.parallelize_fn(model, parallel_dims, training, parallelism, compile, AC, ...)`；返回后 `to_empty(init_device)`、`init_weights()`、`train()`（`torchtitan/trainer.py:472-493`）。Llama 的代表顺序是模型 SPMD/TP → AC → model compile → FSDP（`torchtitan/models/llama3/parallelize.py:40-78`）。
- 有 PP：缺少 `pipelining_fn` 直接失败；该函数获得 `parallelize_fn`，负责切成 model parts 并对 parts 施加 SPMD 优化，随后 Trainer 逐 part 物化/初始化（`torchtitan/trainer.py:426-471`）。完整 PP schedule 见 [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|Pipeline Parallel]]。

`Module.init_states()` 会递归穿过 checkpoint/compile 等普通 wrapper 初始化真正的 Module 后代，再初始化自身参数与 buffer；buffer 重建后恢复原 DTensor placements（`torchtitan/protocols/module.py:74-121`）。这说明“物化”不是简单 `model.cuda()`。

**第四段：按依赖拓扑构造组件。** optimizer 在 parallelism 之后 build，scheduler 再绑定 optimizer；`step/ntokens_seen` 在 checkpointer 之前建立；tokenizer/dataloader 注入 DP rank、context length、token budget；checkpointer 最后接收 data、model parts、optimizer、scheduler、Trainer state 和可选 adapter（`torchtitan/trainer.py:533-577`）。随后才创建 SPMD context、可选 CUDA-Graph fwd/bwd wrapper和 validator（`torchtitan/trainer.py:579-616`）。

### 3.4 约束与失败边界

- SP/CP 开启时，每个 PP microbatch 的 token 数必须被相应 token-sharding divisor 整除，否则构造期失败（`torchtitan/trainer.py:292-305`）。
- `num_tokens_per_train_step` 必须能被“每 DP rank 每次 token × DP degree”整除；否则无法导出整数 gradient accumulation steps（`torchtitan/trainer.py:406-425`）。
- seed-checkpoint 是特例：launcher 要求 world size=1 且 checkpoint enabled；Trainer 在 CPU 初始化并跳过 parallelize，因为它需要的是未分片 step-0 状态，不进入训练（`torchtitan/train.py:46-58`、`torchtitan/trainer.py:390-400`、`torchtitan/trainer.py:472-484`）。
- 如果进程组已由外部初始化，`init_distributed()` 会复用它，并明确警告本次 `comm_config` 不生效；这不是重新配置既有 PG（`torchtitan/distributed/utils.py:446-463`）。
- `config.build()` 若在 `Trainer.__init__` 内失败，赋值尚未完成，launcher 中的 `trainer` 仍是 `None`，不会调用 `Trainer.close()`（`torchtitan/train.py:41-62`）。因此不能把 launcher close 当作构造失败的兜底；各组件自己的析构/close 仍需容忍部分构造对象，checkpointer 的公共 close 就显式使用 `getattr` 处理此情况（`torchtitan/components/checkpointer/base.py:227-234`）。

### 3.5 有锚点的演进

当前源码仍把 `BaseModel.update_from_config()` 标为破坏封装、考虑换成外部 config pass；`init_weights()` 也是为 AutoParallel 保留的 `init_states()` 兼容别名，待外部支持后移除（`torchtitan/protocols/model.py:47-55`、`torchtitan/protocols/model.py:101-117`）。因此“Trainer 永久直接改模型配置”和“`init_weights` 是最终协议名”都不应写成稳定 API。

---

## 4. 一个优化步：token 归一化、前后向与有限性闸门

### 4.1 背景与问题

梯度累积、PP microbatch、padding/屏蔽 token 和多种并行轴共同存在时，“一个 batch”不是可靠的更新单位。Trainer 必须先知道这个 optimizer step 实际包含多少有效 label，且任一 rank 出现非有限 loss/grad 都不能让其余 rank 更新参数。

### 4.2 为什么以完整 token step 为提交单元

选中路线是：先在 CPU 侧收齐 `gradient_accumulation_steps × num_pp_microbatches`，统计有效 token；把全局计数留在 device；每组搬运并执行 fwd/bwd；最后统一 finite gate 和 optimizer update。明显替代方案是一边读取一边更新、每 microbatch 把 loss/finite 拉回 CPU 判断。前者会在数据中途耗尽时留下半个更新，后者引入每 microbatch D2H 同步。当前判据是“更新要么拥有完整输入与跨 rank 一致数值状态，要么不发生”。

### 4.3 当前精确调用链

1. `zero_grad()` 后，Trainer 预取本步所有 microbatch group；CPU 上用 `labels != IGNORE_INDEX` 统计 local valid tokens（`torchtitan/trainer.py:774-796`）。数据 generator 保持 tensor 在 CPU，并把底层 `StopIteration` 改成 `DataloaderExhaustedError`，使整个未完成 step 可在外层取消（`torchtitan/trainer.py:639-666`）。
2. DP 开启时在 `batch` mesh 上求和；`global_valid_tokens` 留在 device，避免 loss normalization 造成 host sync（`torchtitan/trainer.py:798-806`）。
3. 每个 accumulation group 才把 tensor 搬到 device，随后选择 PP schedule 或非 PP 路径，并在使用后释放该组输入（`torchtitan/trainer.py:808-834`）。
4. 非 PP 路径先调用模型的 `preprocess_inputs()`；这个协议负责 mask、CP sharding、SPMD annotation，并返回 inputs/labels/extra kwargs，没有通用默认实现（`torchtitan/protocols/model.py:57-79`）。Trainer 同时按预处理后的 `labels.numel()` 累加 `ntokens_seen`（`torchtitan/trainer.py:688-701`；单元测试固定该行为于 `tests/unit_tests/cpu/test_trainer.py:52-87`）。
5. 真正的 model→loss→backward 位于 `train_context()` 中；loss 接收全局有效 token，backward 则关闭类型检查，因为 forward 已足够验证类型，继续传播会与 AC 等内部实现冲突（`torchtitan/trainer.py:703-728`）。
6. 所有 microbatch 完成后统一 clip grad。loss finite 先在 loss mesh 上合并，再跨 PP 传播；TP replicas 的 loss 相同而 grad norm 已 world-reduced，所以不重复做 TP loss 规约（`torchtitan/trainer.py:850-878`）。
7. `torch._assert_async` 保持检查与 optimizer kernel 的 device 顺序；只有通过后，才 `maybe_wait_for_staging()`、`optimizers.step()`、`lr_schedulers.step()`（`torchtitan/trainer.py:878-889`）。

### 4.4 约束、代价与测试边界

- `_assert_async` 在 CPU 上抛出可捕获 `RuntimeError`，CUDA assertion 失败会使进程失效；它不是自动跳过坏 step 的容错器（`torchtitan/trainer.py:878-885`）。
- 为避免同步，Trainer 只累计“某个 microbatch 非有限”这一 device flag，不能精确指出是哪一个 microbatch。提交 `da3be38c13945610ae1cfedada13a0fb1c111a20` 明确记录了这一诊断精度换同步开销的取舍。
- CUDA Graph 包的是 `_forward_backward_body`，数据搬运、grad clip、optimizer、checkpoint 与 validation 都在图外（`torchtitan/trainer.py:579-588`、`torchtitan/trainer.py:808-889`）。graph 返回的 loss 引用可在下一 replay 被覆盖，所以只有需要日志时才 clone 首个 loss；CPU tests 同时验证 graph-owned output 与跨 replay 累积（`torchtitan/trainer.py:835-848`、`tests/unit_tests/cpu/test_trainer.py:90-204`）。
- DP>1 的 finite dataset 若各 rank 不同时间耗尽会让 collective 挂住；当前 Grain loader 因而拒绝 `repeat=False`，要求 repeat + Trainer step count（`torchtitan/components/data/loader.py:87-97`）。数据恢复和 packing 见 [[02_engineering/02_train_frameworks/torchtitan/02_torchtitan_data_pipeline_grain_analysis|数据管线]]。

### 4.5 有锚点的演进

提交 `73aed7f6c09c04041dae2d2c185bb4c6384ebb3f` 已把语言模型 batch 维折叠为 token budget：配置从 local/global batch size 转向每 microbatch/每 train step token 数。当前 Trainer 的 accumulation 推导和 `ntokens_seen` 都体现这一模型（`torchtitan/trainer.py:406-425`、`torchtitan/trainer.py:688-699`）。因此把 `num_tokens_per_train_step` 解释为“样本 batch size”已经失效。

---

## 5. 外层循环：恢复、save 请求、验证与观测

### 5.1 背景与问题

单步只负责参数更新；恢复、周期性保存、验证、profiling、结构化 trace 和通信 timeout 都会产生跨步副作用。若这些职责塞进 forward/backward，编译/graph 边界会被动态 I/O 污染，也难以清楚定义恢复从哪一步继续。

### 5.2 为什么把副作用放在步边界

当前路线是在进入循环前 load 一次，并在成功完成 `train_step()` 后按固定顺序调度 save、validation 和 profiler。明显替代方案是组件各自监听回调；它更灵活，但难以从单一位置证明“数据耗尽时是否保存”“验证看的是更新前还是更新后权重”。Trainer 选择显式顺序，以可审计性作为判据。

### 5.3 当前精确调用链

```text
checkpointer.load(load_step)
  -> remember loaded_step
  -> profiler context
  -> while step < configured steps:
       step += 1 -> GC -> train_step
       -> checkpointer.save(step, last_step?)
       -> validator.should_validate(step) ? validate(updated model)
       -> profiler.step()
       -> first relative step ? shorten PG timeouts
```

`load()` 会恢复注入 checkpointer 的 Trainer state，因此 `self.step`/`ntokens_seen` 来自 checkpoint；Trainer 自身的 state dict 只包含这两个字段（`torchtitan/trainer.py:943-952`、`torchtitan/trainer.py:1008-1013`）。完整 model/optimizer/data/scheduler 状态归 checkpointer 聚合，见 [[02_engineering/02_train_frameworks/torchtitan/03_torchtitan_checkpoint_state_recovery_analysis|Checkpoint 与恢复]]。

若本步预取时耗尽数据，外层捕获专门异常、记录“last step canceled”并直接 break；save/validation 不会运行（`torchtitan/trainer.py:965-983`）。异常类故意不继承 `StopIteration`，避免 PEP 479 把 generator 内异常包装成无法按预期捕获的 `RuntimeError`（`torchtitan/components/data/loader.py:26-36`）。

验证在 save 调用之后读取**已经 optimizer-updated**的 model parts。通用 Validator 切到 eval + no-grad，建立独立临时 dataloader，以同样的有效 token 语义计算 loss，完成后切回 train（`torchtitan/components/validate.py:140-172`、`torchtitan/components/validate.py:194-282`）。第一步必验证，此后按 `freq`（`torchtitan/components/validate.py:31-54`）。

恢复 run 也会在本进程的第一个相对 step 后缩短 PG timeout；切换前 barrier+device synchronize，避免快 rank 已用短 timeout 发 collective、慢 rank 仍在允许慢初始化的长 timeout 阶段（`torchtitan/trainer.py:988-997`、`torchtitan/distributed/utils.py:539-567`）。

### 5.4 约束与失败边界

- **save 调用在 validation 前，不等于 checkpoint 已持久化。** DCP async 模式会只记录 staging/upload future，后台保存可能仍在进行；下一次 save 才先等待前一个 save，pinned-memory staging 则在下次 optimizer update 前等待（`torchtitan/components/checkpointer/dcp.py:420-488`、`torchtitan/components/checkpointer/dcp.py:613-663`）。旧页把这一顺序写成“验证失败时状态已经提交”过度承诺，现已纠正。
- Validator 临时 loader 由 `finally` 关闭，即使验证 model 抛异常也不泄漏该 loader；CPU test 同时覆盖成功与异常路径（`torchtitan/components/validate.py:285-292`、`tests/unit_tests/cpu/test_validate.py:78-92`）。
- 通用 Validator 只有正常走到末尾才把 model 切回 train；验证异常会冒泡到 launcher，随后进入全局清理，而不是继续下一步（`torchtitan/components/validate.py:268-282`、`torchtitan/train.py:59-62`）。
- metrics 只在 `should_log(step)` 为真时做 loss 分布式规约与 host 标量化；记录 global average/max、grad norm、吞吐、内存与 MFU，随后清空窗口累计（`torchtitan/trainer.py:891-940`、`torchtitan/components/metrics.py:490-541`）。

### 5.5 有锚点的演进

当前 timeout 逻辑的注释已明确使用 `relative_step` 是为了恢复 run 也执行一次 lazy-init/compile 后的 timeout 收紧（`torchtitan/trainer.py:988-997`）。这推翻了“只有全新 run 的 global step 1 才切 timeout”的旧理解；源码没有进一步调度路线图，本页不外推。

### 5.6 Structured logger：跨 rank/actor 的时间证据面

#### ① 背景 / 问题

metrics 回答 loss、吞吐、MFU 等聚合数值，Kineto profiler 回答选定窗口内的算子细节；二者都不天然回答“哪个 rank 在哪一步先进入 checkpoint”“异步 RL 的 actor 是否真正并发”“异常发生前哪个高层阶段拖尾”。这需要一个每步都可开、能跨普通 SPMD 与 asyncio actor 使用的事件时间线。

#### ② 为什么选择标准 logging 事件，而不是把所有观测塞进 profiler

TorchTitan 选择把 span/scalar/instant 写成 Python `logging.Logger` 事件，并允许 handler factory 把同一 schema 落到 JSONL 或外部后端（`torchtitan/observability/structured_logger/structured_logging.py:151-204`）。明显替代是持续开启 Kineto 或让每个组件发自定义 callback；前者采样成本与数据量不适合全程，后者缺少统一 step/rank/source/task 关联。判据是持续可用的高层控制流证据，而不是 kernel 级精确计时。

提交 `b2cd149f6dc7ce16e01b00d119495720c0920924` 的正文把动机明确为定位 straggler、timeout 与 rank/step 异常；提交 `a2a0d99e333b27fe314359ff9c9d918a3c5363f6` 又把同一 API 扩到 RL controller/generator/trainer，说明它的所有权是共享 observability plane，而非 core Trainer 私有指标。

#### ③ 当前状态 / 调用链

入口在配置解析后、Trainer build 前调用 `init_structured_logger(source="training", output_dir=dump_folder)`；默认 `DebugConfig.enable_structured_logging=True`，随后先写一个启动 instant（`torchtitan/train.py:28-44`、`torchtitan/config/configs.py:349-380`）。未指定 `TITAN_STRUCT_LOGGER_HANDLERS` 时使用默认 JSONL handler；指定后只加载环境变量列出的 factory。初始化按进程幂等，rank 默认读 `RANK`，所以不依赖 process group 已初始化（`torchtitan/observability/structured_logger/structured_logging.py:151-212`）。

Trainer 已在 distributed init、模型并行初始化、取 batch、预处理、fwd/bwd、optimizer、metrics、整步与 teardown 外层布置 span（例如 `torchtitan/trainer.py:628-690`、`torchtitan/trainer.py:785-899`、`torchtitan/trainer.py:943-997`）。每个 span 写 `_start/_end`，异常还写 `_error`，结束记录 Python context 的 wall-time；async decorator 会为每次调用新建 span 对象，避免并发调用共享计时状态（`torchtitan/observability/structured_logger/structured_logging.py:284-307`、`torchtitan/observability/structured_logger/structured_logging.py:334-429`）。

step、relative step 与 tags 同时保存在 module global 和 `ContextVar`：SPMD/普通线程走 global，async sibling task 获得隔离视图，防止一个 actor 的 `gc` tag 泄漏到另一个 actor 的 `eval`（`torchtitan/observability/structured_logger/step_state.py:7-37`、`torchtitan/observability/structured_logger/step_state.py:68-126`）。离线 generator 再把各 rank JSONL 的嵌套 span 配对，并把真正并发的 asyncio tasks 分配到不同 Perfetto tracks（`torchtitan/observability/structured_logger/gantt_generator.py:25-83`）。

#### ④ 成本 / 约束 / 失败边界

- trace helper 在 `torch.compiler.is_compiling()` 时直接 no-op；因此它刻画 compiled region 的**外层持续时间**，不能替代图内算子 profiler（`torchtitan/observability/structured_logger/structured_logging.py:215-245`、`torchtitan/observability/structured_logger/structured_logging.py:262-280`）。
- span 的 duration 是 Python context wall-time，源码没有在 span 边界做 device synchronize；不能把它直接解释为单 kernel GPU duration（`torchtitan/observability/structured_logger/structured_logging.py:347-395`）。
- `enable=False` 是进程级全局 no-op；初始化又是幂等的，之后第二次 init 不能改 source/handler（`torchtitan/observability/structured_logger/structured_logging.py:174-212`）。自定义多 actor 进程必须在第一次 init 时决定所有权。
- async decorator 的 `caller` 定位当前有明确 TODO，要求精确 callsite 时应使用 context manager；事件时间和调用者定位是两个不同保证（`torchtitan/observability/structured_logger/structured_logging.py:398-421`）。

#### ⑤ 有锚点的演进

当前可见演进是“同一事件 schema 覆盖同步 Trainer 与异步 actor”，其代码锚点是 global+`ContextVar` 双状态和 Gantt 对并发 task 的 track packing；它已经超过普通训练日志，但仍刻意在 compile 内退让。源码没有承诺把它升级为 GPU profiler 或分布式因果追踪系统，本页不作此推断。

---

## 6. 收尾与异常：释放组件，但避免把清理变成新死锁

### 6.1 背景与问题

分布式异常通常不是所有 rank 同时、同位置失败。若一个 rank 在构造/forward 中出错，另一个 rank 可能仍阻塞于 collective；此时失败 rank 再进入 collective-style teardown，清理动作本身可能成为死锁。

### 6.2 为什么正常与异常采用不对称 teardown

当前路线在两条路径都尽量 `trainer.close()`，但只在无异常的 `else` 分支销毁 process group。明显替代方案是无条件 `finally: destroy_process_group()`；提交 `6b11290c1e45221e097aa4a54d4ab52bf067d0b5` 记录了拒绝它的原因：部分 trainer 已异常而其他 rank 仍在 collective 时，销毁 PG 会导致 cleanup deadlock。这里的判据不是“资源释放最彻底”，而是“不要让失败 rank 发起需要跨 rank 协调的新动作”。

### 6.3 当前精确调用链

- `trainer` 初始为 `None`；构造后训练或 seed-save 任一点异常，launcher 仅在对象存在时调用 `trainer.close()`，然后原样 re-raise（`torchtitan/train.py:41-62`）。
- 成功路径先 `trainer.close()`，再检查 distributed 是否 initialized 并 `destroy_process_group()`（`torchtitan/train.py:63-68`）。
- Trainer close 按 dataloader → CUDA Graph teardown → checkpointer → metrics 的顺序释放，并以 `hasattr` 做防御性检查（`torchtitan/trainer.py:1015-1023`）。
- checkpointer 公共 close 还用 `getattr(enable, False)` 容忍其自身 `__init__` 中途失败；实现只在 enabled 时进入具体 `_close()`（`torchtitan/components/checkpointer/base.py:202-234`）。

### 6.4 约束与失败边界

- `Trainer.close()` 不负责销毁 PG；那是 launcher 成功分支的职责。把二者合并会破坏异常路径的不对称设计。
- close 只显式关闭 dataloader、graph、checkpointer、metrics；源码没有独立 `validator.close()` 或 optimizer close 调用，不应虚构“所有组件统一实现 Closeable”。validator 的临时 dataloader由其 iterator 自己关闭（`torchtitan/components/validate.py:285-292`）。
- CUDA assertion 造成的进程失效、硬 kill 或解释器崩溃不能保证 Python close 执行；这里提供的是正常异常传播路径，不是故障恢复协议。
- 部分构造对象可被安全 close，不代表已经创建的外部资源都能跨 rank 安全协同释放；因此异常仍交由 launcher/作业系统终止其他 ranks。

### 6.5 有锚点的演进与最终纠偏

异常 teardown 的不对称是有历史故障锚点的稳定安全选择，而非漏写 `finally`。结合前述机制，当前应明确废弃以下心智模型：

| 旧断言 | 当前事实 |
|---|---|
| Trainer 只是 `for batch in loader` | 它先构造 mesh/模型/组件状态图，再定义完整优化步与跨步副作用顺序 |
| CLI/TOML 是配置权威 | Python full configuration 是权威；CLI override 是冻结的兼容层 |
| 模型先初始化，再做 sharding | 标准路径是 meta build → parallelize/pipeline → `to_empty` → `init_states` |
| dataloader 输出可直接喂给 model | `BaseModel.preprocess_inputs()` 拥有 mask、CP/SPMD 准备的最后解释权 |
| CUDA Graph 捕获整个 train step | core Trainer 只包装 model/loss/backward body |
| 调用 `save()` 后 checkpoint 一定已落盘 | async 模式可能只启动 staging/upload；持久完成需等待 future |
| 异常时总应 destroy PG | 当前故意只在成功路径 destroy，以免清理死锁 |

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/04_torchtitan_config_model_protocol_analysis|配置、模型与协议控制面]] —— 深入 recipe、`Configurable`、override、Module/ModelSpec 所有权，本页的控制面前置知识。
- [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|ParallelDims 与命名 mesh]] —— 展开 Trainer 初始化得到的 `batch/loss/pp/fsdp/...` mesh 与乘积约束。
- [[02_engineering/02_train_frameworks/torchtitan/02_torchtitan_data_pipeline_grain_analysis|Grain 数据管线]] —— 展开 token packing、DP rank 消费与 dataloader state 恢复。
- [[02_engineering/02_train_frameworks/torchtitan/03_torchtitan_checkpoint_state_recovery_analysis|Checkpoint 状态与恢复]] —— 展开本页只编排的 load/save、异步 staging 和状态图事务。
- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP2 参数生命周期]] —— 解释 meta 模型在物化前如何分片，以及 optimizer 为何必须后建。
- [[02_engineering/02_train_frameworks/torchtitan/23_torchtitan_compute_memory_optimizations_analysis|计算与显存优化]] —— 展开 core CUDA Graph、compile、低精度等被 `parallelize_fn` 接入的机制。
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|GraphTrainer 编译运行时]] —— 与本页标准 Trainer 对照：把更大的训练状态与内存策略纳入编译图，而非仅包装前后向 body。
