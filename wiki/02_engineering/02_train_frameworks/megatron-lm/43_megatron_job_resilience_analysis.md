---
title: "Megatron-LM 作业韧性：进程还在不在、通信域还通不通"
---

# Megatron-LM 作业韧性：进程还在不在、通信域还通不通

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **维度**：功能树模块 N 的**作业侧**。[[28_megatron_training_stability_observability_analysis]] 覆盖的是**数值层面**的稳定性（loss 可不可信、SDC 怎么归因、哪张卡慢）；本页覆盖的是**作业层面**的韧性——进程还在不在、通信域还通不通、硬件还好不好、出事了怎么原地恢复而不是重排队。
> **核心文件**：`megatron/training/{ft_integration,inprocess_restart,dist_signal_handler,determinism,gpu_sniff_test,activation_logging,dgrad_logging}.py`
> **最近更新**：2026-09-02 首建。

---

## 1. 两种"稳定性"要分开谈

万卡训练会以两种完全不同的方式出问题。

一种是**算错了但还在跑**：某张卡的一次矩阵乘因为宇宙射线翻了一位，loss 曲线上看不出来，几千步后模型悄悄坏掉。对付它要的是数值校验与归因——重跑同一批数据比对结果、检查梯度范数、跨 DP 副本比对参数哈希。这是 [[28_megatron_training_stability_observability_analysis]] 的领域。

另一种是**跑不动了**：某个进程段错误退出、某张卡掉了、某条 NCCL 通信超时挂住。这时候没有"结果对不对"的问题，只有"作业还活着吗、能不能不从头开始"的问题。八千卡的作业排队重启一次可能等几小时，而故障率随卡数线性上升——这一类的成本不在算力，在**排队**。

本页讲的是第二种。它的核心矛盾是：**要恢复就得先把坏掉的东西清理干净，但清理动作本身也可能挂住**——比如你想销毁一个已经卡死的 NCCL 通信域。后面几节反复出现的设计张力都源于此。

**本页不覆盖**：RerunStateMachine / SDC 归因 / QK-clip / Timer / MoE 逐层指标 / StragglerDetector 本体 / `megatron/core/fault_injector.py` → 见 [[28_megatron_training_stability_observability_analysis]]；跨框架的快恢与"重新建链"对比 → 见 [[02_engineering/02_train_frameworks/33_fault_recovery_relink_comparison]]；checkpoint 的存取机制本体 → 见 [[19_megatron_dist_checkpointing_analysis]]。

---

## 2. NVRx 容错集成：心跳与自适应超时

`megatron/training/ft_integration.py`（369 行）是 Megatron 接 NVIDIA Resiliency Extension（NVRx）的适配层。由 `--enable-ft-package` 打开。

### 2.1 钩子布点

`setup()`（`:76`）拿到 rank monitor client 后，训练循环在六个位置打点：`on_training_step_start` / `on_training_step_end`（`:121`/`:136`）、`on_eval_step_start` / `on_eval_step_end`（`:146`/`:159`）、`on_checkpointing_start` / `on_checkpointing_end`（`:169`/`:176`），另有 `on_checkpoint_loaded`（`:193`）与 `shutdown`（`:207`）。

这些点位不是随手撒的，它们对应**四段耗时特征完全不同的区间**：训练步（毫秒到秒级、高度规律）、评估步（较长但也规律）、存档（秒到分钟级、且异步存档还有一条尾巴）、其余（初始化、数据加载等）。分开计时才能给每段配不同的超时阈值——这正是下一节的前提。

### 2.2 为什么超时要自己算

`--calc-ft-timeouts` 打开后（`:106` 读入 `args.calc_ft_timeouts`），`_update_timeouts`（`:228`）与 `_maybe_update_timeouts`（`:244`）会**从实际观测到的耗时反推超时阈值**，而不是用一个写死的常数。

**被否掉的替代就是那个写死的常数**：一个固定超时值在跨集群、跨模型规模时必然选错——阈值给大了，真挂了也要等很久才发现；给小了，一次正常的慢存档就被误判成故障、触发不必要的重启。而这两个方向的代价都很高。

自适应也有它的前提：`_MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE = 16`（`:64`），且 `_maybe_update_timeouts` 里判断 `_seen_tr_iters_cnt >= _MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE`（`:261`）才更新。**样本不够就不动阈值**——刚启动的头几步包含各种一次性开销，拿它们算出来的阈值会偏大得离谱。

### 2.3 模拟故障注入

`maybe_setup_simulated_fault()`（`:300`）按 `--simulate-fault` 系列参数（类型、rank、基础延迟）起一个后台线程（`__fault_thread`，`:354`）在指定时刻制造故障。

这是**容错路径自身的测试手段**：容错代码的特点是平时不执行，真出事时才第一次跑——如果那时才发现它有 bug，损失是双份的。注意它与 `megatron/core/fault_injector.py` 分属两层：那个是 core 侧的通用注入器（由 [[28_megatron_training_stability_observability_analysis]] 覆盖），这个是训练侧针对 NVRx 心跳链路的注入。

---

## 3. 进程内重启：不重排队的恢复

`megatron/training/inprocess_restart.py`（163 行）由 `--inprocess-restart` 打开，包装整个 `pretrain`（`maybe_wrap_for_inprocess_restart`，`:130`）。

### 3.1 它和"重排队"的区别

常规的故障恢复是：作业挂掉 → 调度器重新排队 → 重新申请节点 → 重新初始化 → 从 checkpoint 恢复。进程内重启把中间三步全部省掉：**进程不退出**，在同一批已经持有的节点上销毁分布式状态、重建通信域、从内存或本地 checkpoint 继续。

省掉的排队时间就是它的全部价值。代价是**清理必须彻底**——进程还活着，任何没清干净的状态都会污染下一轮。

### 3.2 清理什么

`destroy_state()`（`:25`）做两件事：`training.destroy_global_state()`（`:28`）与 `rerun_state_machine.destroy_rerun_state_machine()`（`:29`）。前者对应 [[41_megatron_config_surface_analysis]] §2 提到的那套全局单例（args、tokenizer、writer、timers），后者是 28 号页覆盖的重跑状态机。

清理被包在 `inprocess.finalize.ThreadedFinalize(timeout=timedelta(seconds=10), fn=destroy_state)`（`:70`）里——**带超时**。这正是 §1 说的那个张力：清理动作自己也可能挂住，所以清理也要有 deadline。

中止链是组合出来的（`:93-95`）：`inprocess.Compose(inprocess.abort.AbortTransformerEngine(), inprocess.abort.AbortTorchDistributed(), ...)`，另有一个自定义的 `AbortCheckpoint`（`:85`）。**顺序有意义**：TE 的中止要在 torch.distributed 之前，因为 TE 持有的通信资源建立在后者之上。

### 3.3 一处非直觉的前置：强制初始化 NCCL

`maybe_force_nccl_backend_init(device_id)`（`:152`）在 `--inprocess-restart` 打开时做一次 `all_reduce` 加 `cuda.synchronize()`。

注释把理由写得很清楚（`:156-159`）："Inprocess uses destroy_process_group to terminate NCCL backend, which does not terminate NCCL kernels if NCCL backend wasn't fully initialized before additional distributed subgroups are created."

翻译过来：**NCCL 后端是惰性初始化的**。如果在它完全初始化之前就创建了额外的子通信域，那么 `destroy_process_group` 无法终止已经在飞的 NCCL kernel——于是"清理"变成了假清理，重启后的通信域和残留 kernel 打架。用一次哪怕毫无用处的 `all_reduce` 强制走完初始化，是拿一次极小的开销换清理路径的确定性。

**这类"为了能清理干净而提前做一件多余的事"是本页的典型模式**，§2.2 的自适应超时、§4 的信号聚合都是同一思路的不同表现。

---

## 4. 信号退出：从"某个 rank 收到"到"全体一致退出"

`megatron/training/dist_signal_handler.py` 由 `--exit-signal-handler` 打开，`--exit-signal` 指定信号（默认 `SIGTERM`，`:51`）。

问题在于**信号是发给单个进程的，退出必须是集体的**。调度器抢占时可能只给 rank 0 发信号；即使广播，各 rank 收到的时刻也不同。如果每个 rank 各自决定何时退出，就会有 rank 已经退出、另一些还阻塞在集合通信上——直接挂死。

`DistributedSignalHandler`（`:50`）的解法是把"我收到信号了"这个布尔量**做一次 all-gather**：`signals_received()`（`:54`）调用 `all_gather_item(self._signal_received, dtype=torch.int32)`（`:55-56`），内部走 `torch.distributed.all_gather`（`:45`）。信号处理器本身只做一件事——把标志位置 True（`:65-66`），不做任何清理。

于是"是否退出"变成一个所有 rank 都能算出相同答案的集体判断。`checkpoint_and_decide_exit` 里用的是 `any(signal_handler.signals_received())`（`megatron/training/training.py:4093`）——**任一 rank 收到即全体退出**。

**被否掉的替代**：让 rank 0 广播决定。那需要一次额外的 broadcast，且 rank 0 自己挂掉时整个机制失效；all-gather 的对称写法没有单点。

---

## 5. 退出策略矩阵

`megatron/training/training.py:4075` 的 `checkpoint_and_decide_exit` 把四条退出路径与存档去重逻辑收在一处。

| 路径 | 触发 | 位置 |
|---|---|---|
| ① 信号 | `--exit-signal-handler` 且任一 rank 收到信号 | `:4092` |
| ② 周期存档（不退出） | `--save-interval` 到点 | `:4110` |
| ③ 时长 | `--exit-duration-in-mins` 超时 | `:4140` |
| ④ 迭代 / 阶段切换 | `--exit-interval` 到点，或 `iteration in args.phase_transition_iterations` | `:4163` |

**去重是这段代码真正的复杂度所在。** 一个 `saved_checkpoint` 标志（`:4091` 初始化为 False）在四条路径间传递，③④ 两路都写成 `if args.save and not saved_checkpoint`（`:4148`、`:4166`）——**没有它，一次同时命中 save-interval 与 exit-interval 的迭代会存两次档**。万卡尺度下一次全量存档是分钟级的集体操作，重复一次是实打实的浪费。

③ 的时长判断值得单看：它把 `train_time > args.exit_duration_in_mins` 的布尔结果放进一个 CUDA 张量再做集体判定（`:4140-4143`）。**各 rank 的墙钟不完全一致**，让每个 rank 各自比较会得到不同答案；走一次集体通信才能保证退出决定是一致的——与 §4 的信号聚合同一道理。

---

## 6. 确定性模式：为什么必须在 argparse 阶段就设环境变量

`megatron/training/determinism.py` 由 `--deterministic-mode` 触发，在 `megatron/training/arguments.py` 的校验阶段被调用。

它管的不是代码，是**环境变量**：`DETERMINISM_ENV_VAR_DEFAULTS` 包含 `NCCL_ALGO: "Ring"`（`:28`）、`CUBLAS_WORKSPACE_CONFIG: ":4096:8"`（`:30`）等。

### 6.1 时机是硬约束

`apply_determinism_env` 的 docstring 把原因写死了（`:82-84`）："These env vars are captured by their respective libraries at first use (NCCL at communicator init, cuBLAS at handle creation, TE at first attention forward), so the call must happen BEFORE any of those events."

**这些库只在第一次使用时读一遍环境变量，之后再改无效。** 而 NCCL 通信域初始化发生在 `initialize_megatron` 里，cuBLAS handle 在第一次 GEMM 时创建——都远早于训练循环。所以这件事必须在参数解析阶段做完，晚一步就静默失效：程序照跑，只是不确定。

**这是"静默失效"而非"报错"，正是它值得单独一节的原因**——用户会以为自己开了确定性模式。

### 6.2 校验：允许收窄，不允许放宽

不是简单地覆盖用户设置。`NCCL_ALGO` 走**子集校验**（`:71-72`）：用户给的逗号分隔列表里每个 token 都必须在 `ACCEPTED_NCCL_ALGO_TOKENS` 里（`:48`，为 `{"Ring", "CollnetDirect", "CollnetChain", "^NVLS"}`），否则断言失败。`CUBLAS_WORKSPACE_CONFIG` 等走**精确匹配**（`:62`，只接受 `:4096:8` 与 `:16:8`）。

校验通过后才 `setdefault`——**调用方已设的值优先**（`:79-80`）。

这套"先校验、后 setdefault"的组合表达的是一条明确的策略：**用户可以在确定性允许的范围内做选择（比如挑一个不同的确定性算法），但不能把范围本身放宽**。`^NVLS` 这个 token 之所以在白名单里，注释解释了它是"排除 NVLS 而非选择某个算法"（`:40`）——语义上是收窄。

---

## 7. GPU sniff test：慢节点检测的另一条路

`megatron/training/gpu_sniff_test.py`（579 行）由 `--gpu-sniff-test-interval` 触发，每 N 迭代跑一轮微基准。

### 7.1 与 StragglerDetector 的分工

[[28_megatron_training_stability_observability_analysis]] 覆盖的 StragglerDetector 是**被动观测**：它测的是真实训练步里各 rank 的耗时，谁慢就报谁。

sniff test 是**主动探测**：它跑一组固定的合成负载——`bench_gemms`（`:130`）、`bench_all_reduce`（`:162`）、`bench_reduce_scatter`（`:199`）、`bench_all_to_all`（`:238`）、`bench_sendrecv`（`:274`）——各 rank 做同样的事，然后比。

**两者给出的证据不同**。StragglerDetector 说"rank 37 这一步慢了"，但慢的原因可能是它分到的数据更长、或者它在 PP 的某个位置上本来就该等。sniff test 说"rank 37 的 AllReduce 带宽比中位数低 40%"——因为所有 rank 跑的是同一个合成负载，**差异只可能来自硬件或链路**。前者发现问题，后者定位到层（是计算还是通信、是哪种通信）。

代价是 sniff test 要占用训练时间，所以是周期性抽查而非常开。

### 7.2 离群判据

`_gather_and_check`（`:76`）把各 rank 的标量指标 all-gather 到 rank 0，然后：先用 `~np.isnan` 滤掉未参与的 rank（`:96`，注释说明未配对的 send/recv 会填 NaN），再算中位数与 **MAD**（median absolute deviation，`:104`），阈值取 `max(mad, median * OUTLIER_MIN_DEVIATION_FRAC)`（`:106`），偏离超过阈值即为离群（`:107`）。

**两个设计点都在避开均值/标准差的坑**：
- 用**中位数与 MAD 而非均值与标准差**——离群点本身会把均值和标准差一起拉走，几张坏卡足以让阈值宽到检测不出它们自己。中位数与 MAD 对离群点不敏感。
- 阈值有**下界** `median * OUTLIER_MIN_DEVIATION_FRAC`——集群完全健康时 MAD 会非常小，纯按 MAD 判会把正常的测量噪声全报成离群。下界保证"偏离必须同时超过绝对比例"才算数。

报告里除了 rank 号还带主机名（`_gather_hostnames`，`:68`），并打印相对中位数的百分比偏差（`:118-120`）——直接指向要下架的那台机器。

`run_sniff_tests`（`:348`）在训练内按 TP / EP / DP 分组跑对应的集合通信；同一文件还带一个独立 CLI（`main`），可脱离训练单独排查。

---

## 8. 三类张量转储

当数值出问题而 §2-§7 的手段都说"作业是健康的"时，剩下的办法是把中间张量落盘逐层比对。三个开关各管一类：

| 开关 | 转储什么 | 实现 |
|---|---|---|
| `--save-activations-interval` | 逐层激活 | `megatron/training/activation_logging.py:311` `enable_activation_logging` / `:319` `save_activations` |
| `--save-tokens-per-expert-interval` | MoE 逐专家 token 数 | 同文件 `:326` `enable_tokens_per_expert_logging` |
| `--save-dgrads-interval` | 逐层数据梯度 | `megatron/training/dgrad_logging.py:119` `enable_dgrad_logging` / `:134` `save_dgrads` |

三者共用同一套 hook 安装机制（`activation_logging.py:102` 的 `_register_hooks`，按模块类型过滤）。一处工程细节：`_discover_te_types()`（`:24`）与 `dgrad_logging.py:17` 的 `_get_linear_types()` **在运行时发现 TE 的类型**而不是硬编码 import——TransformerEngine 是可选依赖且版本间类名会变，硬编码会让转储功能在 TE 缺失或升级时直接崩掉。

`_parse_tpe_module_name`（`activation_logging.py:83`）把模块名解析成 `(名字, 层号, 索引)` 三元组，让落盘文件能按层对齐——**跨两次运行比对时，能对上号才有意义**。

> [!note] 待展开
> 三个 logger 的**落盘格式与文件布局**（张量以什么形式写、怎么分片、跨 rank 如何区分）本页未展开，只覆盖了触发面与 hook 机制。逐格式走查需要单独一轮。

---

## 9. one_logger E2E 指标

`megatron/training/one_logger_utils.py`（462 行）由 `--enable-one-logger` 打开，把作业级事件（训练开始、存档开始/成功/结束、应用标签、配置 flag）上报到 NVIDIA 内部的 one-logger 后端。

它与 §2 的 NVRx 是**互补的两层**：NVRx 关心"这个作业现在还活着吗"，one-logger 关心"这批作业跨天跨次的端到端指标是什么"。前者是实时控制回路，后者是离线分析面。

> [!note] 待展开
> 本页只给了 one_logger 的定位与触发面。它的指标语义（`_produce_e2e_metrics` 具体产出哪些量、如何跨重启累计）未展开。

---

## 10. 约束与边界

| 边界 | 表现 | 证据 |
|---|---|---|
| 确定性模式的时机不可协商 | 晚于 NCCL/cuBLAS/TE 首次使用即**静默失效**，程序照跑但不确定 | `megatron/training/determinism.py:82-84` |
| 进程内重启要求 NCCL 先完全初始化 | 否则 `destroy_process_group` 终止不了在飞的 kernel | `megatron/training/inprocess_restart.py:156-159` |
| 清理动作本身带超时 | `ThreadedFinalize(timeout=10s)`——清理挂住时放弃清理 | `:70` |
| 自适应超时需要 ≥16 次迭代样本 | 样本不足时不更新阈值 | `megatron/training/ft_integration.py:64`、`:261` |
| sniff test 占用训练时间 | 只能周期性抽查，不能常开 | `--gpu-sniff-test-interval` |
| 退出决定必须走集体通信 | 各 rank 墙钟不一致、信号送达不一致 | `megatron/training/training.py:4093`、`:4140-4143` |
| NVRx 与 one-logger 都是可选外部依赖 | 未安装时对应功能整体不可用 | `ft_integration.py:67` 的 client 获取 |

---

---

## 配置契约：`TrainingConfig` 与 `ValidationConfig`

本页正文讲作业**出事之后**怎么办。本节补的是决定作业**怎么跑、跑多久、何时停**的两个 config 类——它们与 §5 的退出策略矩阵直接咬合：`exit_*` 一组是那张表的输入，`save_*_interval` 一组决定退出前要不要落档。两个类都经 [[41_megatron_config_surface_analysis]] §2 的工厂转成 CLI（`megatron/training/arguments.py:3655`、`:4359`）。

**下表直接取自 `megatron/training/config/training_config.py` 的类体**。`ValidationConfig` 单列的理由是评估循环有自己的批大小与频率——本页 §5 说的「四段耗时特征不同的区间」里，评估是独立一段。


### `TrainingConfig`（`megatron/training/config/training_config.py`，12 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `rampup_batch_size` | `list[int] \| None` | `field(default=None, metadata={'argpar…` | Batch size ramp up with the following values: <start batch size>, <batch size increment>, <ramp-up samples> For example: rampup-batch-size = [16, 8, 300000] … | `:20` |
| `step_batch_size_schedule` | `str \| None` | `None` | Step-wise batch size schedule in format "THRESHOLD:BS THRESHOLD:BS ...". Thresholds support suffixes: K (1e3), M (1e6), B (1e9), T (1e12). If sequence length… | `:34` |
| `decrease_batch_size_if_needed` | `bool` | `False` | If set, decrease batch size if microbatch_size * dp_size does not divide batch_size. Old batch_size will be restored if training is re-started with dp_size t… | `:43` |
| `empty_unused_memory_level` | `Literal[0, 1, 2]` | `0` | Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation. 0=off, 1=moderate, 2=aggressive. | `:49` |
| `train_sync_interval` | `int \| None` | `None` | Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU. | `:62` |
| `train_iters` | `int \| None` | `None` | Total number of iterations to train over all training runs. Note that either train_iters or train_samples should be provided. | `:65` |
| `train_samples` | `int \| None` | `None` | Total number of samples to train over all training runs. Note that either train_iters or train_samples should be provided. | `:70` |
| `exit_signal_handler_for_dataloader` | `bool` | `False` | Use signal handler for dataloader workers | `:86` |
| `exit_signal_handler_for_training` | `bool` | `False` | Shutdown the training when SIGINT or SIGTERM received to avoid unclear traceback | `:89` |
| `manual_gc_interval` | `int` | `0` | Training step interval to trigger manual garbage collection. Values > 0 will trigger garbage collections between training steps. | `:98` |
| `manual_gc_eval` | `bool` | `True` | When using manual garbage collection, this controls garbage collection at the start and the end of each evaluation run. | `:103` |
| `iterations_to_skip` | `list[int]` | `field(default_factory=list)` | List of 1-indexed iterations to skip during training, empty by default. | `:108` |

> 该类共 21 个字段，本表收 12 项；其余 9 项已在别处归属：主要归 本页他处 7 项、[[36_megatron_fsdp_analysis]] 1 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 1 项（完整归属见 `docs/coverage/megatron-lm.yaml`）。



### `ValidationConfig`（`megatron/training/config/training_config.py`，10 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `eval_iters` | `int \| None` | `100` | Number of iterations to run for evaluation. Used for both validation and test. If not set, evaluation will not run. | `:116` |
| `eval_interval` | `int \| None` | `None` | Interval between running evaluation on validation set. If not set, evaluation will not run during training. | `:120` |
| `start_eval_at_iter` | `int \| None` | `None` | If set, evaluation will only start after this iteration number. Useful for skipping evaluation during early training iterations when the model is not yet mea… | `:125` |
| `eval_global_batch_size` | `int \| None` | `None` | Global batch size to use during evaluation. If not set, defaults to global_batch_size. Must be divisible by (eval_micro_batch_size * data_parallel_size). | `:131` |
| `eval_micro_batch_size` | `int \| None` | `None` | Micro batch size to use during evaluation. If not set, defaults to micro_batch_size. Changing this affects per-device memory usage during eval and the number… | `:136` |
| `skip_train` | `bool` | `False` | If set, bypass the training loop, perform evaluation for validation/test, and exit. | `:142` |
| `test_mode` | `bool` | `False` | Run all real-time test alongside the experiment. | `:145` |
| `full_validation` | `bool` | `False` | If set, each time validation occurs it uses the full validation dataset(s). This currently only works for GPT datasets! | `:148` |
| `multiple_validation_sets` | `bool` | `False` | If set, multiple datasets listed in the validation split are evaluated independently with a separate loss for each dataset in the list. This argument require… | `:151` |
| `validation_set_names` | `Optional[List[str]]` | `None` | Optional list of names for multiple validation sets. When provided with --multiple-validation-sets, these names are used instead of numeric indices (e.g. 'va… | `:157` |

## Related Pages

- [[28_megatron_training_stability_observability_analysis]] — 数值层面的稳定性（RerunStateMachine、SDC 归因、StragglerDetector）；与本页的作业层面互补，§7.1 专门辨析了两者的分工
- [[40_megatron_feature_tree_analysis]] — 功能树总览；本页覆盖的是它 §4 仪表盘里"作业韧性与张量转储"那一行的七个文件
- [[41_megatron_config_surface_analysis]] — 本页各开关所属的 `RerunStateMachineConfig`/`StragglerDetectionConfig`/`FaultInjectorConfig` 三个参数组由那里的工厂自动生成
- [[19_megatron_dist_checkpointing_analysis]] — §5 退出策略里每一路都要落一次档，存档机制本身在那里
- [[02_engineering/02_train_frameworks/33_fault_recovery_relink_comparison]] — 跨框架的快恢与"重新建链"对比，把本页的 Megatron+NVRx 路线放进横向坐标
