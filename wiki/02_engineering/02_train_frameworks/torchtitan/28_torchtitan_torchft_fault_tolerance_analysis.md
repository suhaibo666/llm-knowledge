---
title: "TorchTitan × TorchFT：动态副本、quorum optimizer 与双通道 checkpoint"
---

# TorchTitan × TorchFT：动态副本、quorum optimizer 与双通道 checkpoint

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **状态**：experimental · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：TorchFT 接入不是给现有 data-parallel ProcessGroup 加一个重试器，而是把普通 HSDP 的静态 replicate 轴拆成多个独立 TorchTitan replica group，再由 TorchFT 管理跨 group 的动态 quorum。为了让“某个 group 消失后剩余 group 继续”成立，TorchTitan 同时改写了四条所有权链：dataloader 的全局 shard 编号、FSDP replicate all-reduce、optimizer step 的提交点，以及 checkpoint 的全局状态/每副本数据游标。semi-sync 模式又关闭逐步 quorum，把跨副本一致性下放到 LocalSGD/DiLoCo 的周期同步。
>
> 这是一条独立实验 Trainer 路径，不等同于核心 Trainer 的自动容错开关。标准状态图与 DCP 语义见 [[03_torchtitan_checkpoint_state_recovery_analysis]]；FSDP/HSDP 的静态存储与规约见 [[11_torchtitan_fsdp_analysis]]、[[21_torchtitan_hsdp_backward_overlap_analysis]]。

---

## 1. Overview：故障域从 rank 变成 replica group

标准 HSDP 在一个作业里建立固定的 replicate × shard mesh；任一 rank 失败通常会使整个进程组失效。TorchFT 路线改成启动多个相互独立的 TorchTitan 实例，每个实例内部做 FSDP shard，实例之间才是可变化的副本集合。项目 README 明确说 TorchFT 针对 replicated weights（DDP/HSDP），并要求多个 replica group 各自维护一份模型权重（`torchtitan/experiments/torchft/README.md:3`、`torchtitan/experiments/torchft/README.md:15`、`torchtitan/experiments/torchft/README.md:17`）。

当前配置函数返回专门的 `FaultTolerantTrainer.Config`，同时选择 TorchFT optimizer、TorchFT checkpoint 与 FaultTolerance config（`torchtitan/experiments/torchft/llama3/config_registry.py:25`、`torchtitan/experiments/torchft/llama3/config_registry.py:40`、`torchtitan/experiments/torchft/llama3/config_registry.py:57`、`torchtitan/experiments/torchft/llama3/config_registry.py:63`）。所以接入边界不是普通 Trainer 上加一个 flag，而是一套成组替换的组件。

```text
replica group 0: [FSDP shard ranks] --+
                                      | TorchFT Manager / quorum
replica group 1: [FSDP shard ranks] --+
                                      |
                     +----------------+----------------+
                     |                |                |
                 DP all-reduce   optimizer commit   state transfer
                     |                |                |
              model gradients     step/zero_grad    model/optim/train

每个 replica 另有自己的 dataloader shard 与游标 checkpoint。
```

### Quick Start：当前入口与最小检查

运行入口使用 `MODULE=torchft.llama3` 和 `CONFIG=llama3_torchft_debugmodel`；每个 replica 实例有独立 `replica_id`，共同连接 lighthouse（`torchtitan/experiments/torchft/README.md:28`、`torchtitan/experiments/torchft/README.md:38`、`torchtitan/experiments/torchft/README.md:43`、`torchtitan/experiments/torchft/README.md:55`）。读源码应从四个对象进入：

| 对象 | 当前职责 | 故障后必须守住的状态 |
|---|---|---|
| `TorchFTManager` | 创建 TorchFT PG/Manager，提供 managed replicate PG | 活跃副本集合与 replica identity |
| `FaultTolerantTrainer` | 改 global ranks、data shard、组件注入 | 每一步使用同一 FT 拓扑 |
| `TorchFTOptimizersContainer` | 把 optimizer step/zero_grad 变成 quorum 边界 | 成功提交的参数/optimizer 状态 |
| `TorchFTCheckpointManager` | 全局 checkpoint + 每 replica dataloader side channel | 权重状态与数据消费位置 |

---

## 2. Manager 与数据分片：为什么 replica 变化必须先改变样本所有权

### ① 背景/问题

只让 surviving replicas 继续 all-reduce 还不够。如果每个独立 TorchTitan 实例仍认为自己的 DP rank 从 0 开始，它们会读取相同数据；故障恢复后新增/存活副本也可能重放已经消费的样本。另一个直觉方案是把所有实例放进一个静态 HSDP world，但该 world 在成员失败后无法保持同一固定 ProcessGroup。

### ② 为什么这么设计

**选中的路线**是实例内部 rank 仍由 TorchTitan 管理，跨实例 replicate membership 交给 TorchFT；dataloader 则用 `(local dp degree × configured group size, replica offset + local rank)` 构造全局 shard 编号。**替代方案**是每个 replica 保持独立 sampler 或用一个静态全局 PG。决定性标准是数据所有权必须与 replica identity 一起变化，而模型内部 FSDP shard topology 仍保持本地可推理。

### ③ 实现思路与细节

`FaultTolerantTrainer.init_distributed()` 先根据 replica id/group size生成 `_ranks` 映射，再初始化本实例的分布式 world，最后才 build TorchFT manager 和本地 `ParallelDims`（`torchtitan/experiments/torchft/trainer.py:351`、`torchtitan/experiments/torchft/trainer.py:354`、`torchtitan/experiments/torchft/trainer.py:365`、`torchtitan/experiments/torchft/trainer.py:372`、`torchtitan/experiments/torchft/trainer.py:375`）。设备必须先被设置，因为 manager 的 NCCL/MCCL 路径依赖当前 device（`torchtitan/experiments/torchft/trainer.py:52`、`torchtitan/experiments/torchft/trainer.py:55`）。

Manager 未启用时退化为空对象；启用却未安装 torchft 时立即失败（`torchtitan/experiments/torchft/manager.py:79`、`torchtitan/experiments/torchft/manager.py:83`、`torchtitan/experiments/torchft/manager.py:87`）。当前实现实际接受 Gloo、NCCL 和 MCCL 三条 PG 路径，并把未知值拒绝（`torchtitan/experiments/torchft/manager.py:90`、`torchtitan/experiments/torchft/manager.py:93`、`torchtitan/experiments/torchft/manager.py:95`、`torchtitan/experiments/torchft/manager.py:107`）。这已经超过 Config docstring 仍写的“only gloo and nccl”（`torchtitan/experiments/torchft/manager.py:48`），是当前文档与代码的一处明确漂移。

Trainer 从本地 `batch` mesh 得到 batch degree/rank 后，调用 `get_dp_info()`扩成跨 replica 的数据 shard 编号，再用它构建 Grain dataloader（`torchtitan/experiments/torchft/trainer.py:64`、`torchtitan/experiments/torchft/trainer.py:70`、`torchtitan/experiments/torchft/trainer.py:97`）。Manager 的换算公式位于 `torchtitan/experiments/torchft/manager.py:137`；未启用时原样返回本地编号。

### ④ 约束/代价/失败边界

- `group_size` 同时参与 data shard 计算与实例 rank 映射；配置错误不会只影响通信，还会造成样本重复或遗漏。
- README 明确把该功能标为 ongoing development（`torchtitan/experiments/torchft/README.md:7`），这里的进程启动与内部 `_ranks` 接口不应当作稳定公共 ABI。
- Gloo timeout 的 Config 注释说只对 Gloo 生效；切到 NCCL/MCCL 不应假设完全相同的 abort 语义（`torchtitan/experiments/torchft/manager.py:53`、`torchtitan/experiments/torchft/manager.py:55`）。
- 数据 shard 编号只解决“谁读哪一份”，不能恢复“已经读到哪里”；后者由每 replica dataloader checkpoint 单独负责。

---

## 3. Async quorum：为什么 FSDP hook 与 optimizer 都要接入 Manager

### ① 背景/问题

如果只把 FSDP replicate all-reduce 换成动态 PG，某一步发生成员变化时，所有 surviving replica 仍需就“这一步是否提交、使用哪份 optimizer state”达成一致。反过来，只包 optimizer 而不改 FSDP all-reduce，梯度通信仍绑定静态副本组。多 optimizer 容器还会放大问题：TorchFT 要求每个 manager 每个 train step 只进入一次 wrapper step，而容器内部可能循环多个 optimizer。

### ② 为什么这么设计

**选中的路线**是 async-quorum 模式下同时安装 managed replicate PG 的 FSDP all-reduce hook，并用 `torchft.Optimizer` 包住整个 TorchTitan optimizer container；**替代方案**是只改 collective 或分别包装每个叶子 optimizer。决定性标准是梯度聚合与 step 提交必须观察同一个动态 quorum，而且一个逻辑 train step只能触发一次 quorum 决策。

### ③ 实现思路与细节

没有 semi-sync method 时 `use_async_quorum=True`；Manager 创建 `ManagedProcessGroup` 并注册为 `dp_replicate`（`torchtitan/experiments/torchft/manager.py:110`、`torchtitan/experiments/torchft/manager.py:113`、`torchtitan/experiments/torchft/manager.py:124`）。Trainer 在 model parallelization/materialization 完成后，对每个 FSDPModule 安装 all-reduce hook（`torchtitan/experiments/torchft/trainer.py:244`、`torchtitan/experiments/torchft/trainer.py:251`）。hook 在 managed PG 上执行 AVG（`torchtitan/experiments/torchft/manager.py:143`、`torchtitan/experiments/torchft/manager.py:146`、`torchtitan/experiments/torchft/manager.py:149`）。

`TorchFTOptimizersContainer` 先构建普通 optimizers并强制初始化其 state，避免 state_dict/load_state_dict 间接触发 step（`torchtitan/experiments/torchft/optimizer.py:31`、`torchtitan/experiments/torchft/optimizer.py:38`、`torchtitan/experiments/torchft/optimizer.py:40`）。最外层 step 临时关闭 `_use_ft_optimizer` 后调用 TorchFT wrapper；wrapper 回调容器时，内部调用才落到普通 `super().step()`，从而避免递归并保证一个 logical step 只进一次 quorum（`torchtitan/experiments/torchft/optimizer.py:64`、`torchtitan/experiments/torchft/optimizer.py:67`、`torchtitan/experiments/torchft/optimizer.py:71`）。`zero_grad()` 使用同一 re-entry guard（`torchtitan/experiments/torchft/optimizer.py:78`、`torchtitan/experiments/torchft/optimizer.py:83`）。

### ④ 约束/代价/失败边界

- 当前 async hook只遍历 `FSDPModule`；未被 FSDP 包装的参数不会自动得到跨 replica gradient sync。
- optimizer container 必须是 TorchFT 版本才能建立 quorum wrapper。Trainer 虽允许普通 optimizer config，但那条分支不会得到 TorchFT step 语义（`torchtitan/experiments/torchft/trainer.py:265`、`torchtitan/experiments/torchft/trainer.py:267`、`torchtitan/experiments/torchft/trainer.py:271`）。
- cached optimizer state 是容错传输的一部分；`load_state_dict()` 必须先清 cache，因为底层 assign 会使旧 cache持有失效 tensor 并泄漏内存（`torchtitan/experiments/torchft/optimizer.py:50`、`torchtitan/experiments/torchft/optimizer.py:56`、`torchtitan/experiments/torchft/optimizer.py:60`）。
- 本页只能验证 TorchTitan 如何调用 TorchFT；quorum 算法本身属于外部依赖，不能从这些 adapter 行推导其全部故障模型。

---

## 4. 双通道 checkpoint：为什么全局状态只写一次，数据游标却每副本每步写

### ① 背景/问题

所有 replicas 写同一份完整模型 checkpoint 会重复 I/O、互相竞争清理策略；只让一个 replica 保存完整状态又会丢掉其他 replica 各自的 dataloader 游标。故障发生后，权重可以从 surviving replica/全局 checkpoint恢复，但数据消费位置若被统一成一份，就会重读或跳过属于其他 replica 的 shard。

### ② 为什么这么设计

**选中的路线**是两条持久化通道：`participating_rank()==0` 的副本写完整 checkpoint，每个 replica 每一步异步写自己的 dataloader-only checkpoint；**明显替代方案**是每个 replica 写完整副本，或只保存一份全局 dataloader state。决定性标准是模型/optimizer 是副本间应收敛的全局状态，而数据游标天然按 replica 分叉。

### ③ 实现思路与细节

TorchFT manager 注册的 in-memory state-transfer 回调只包含 model、optimizer、LR scheduler 和 train state，不包含 dataloader（`torchtitan/experiments/torchft/checkpoint.py:114`、`torchtitan/experiments/torchft/checkpoint.py:117`、`torchtitan/experiments/torchft/checkpoint.py:120`、`torchtitan/experiments/torchft/checkpoint.py:130`）。dataloader 单独放入 `ft_states`，并为异步保存补建 Gloo PG（`torchtitan/experiments/torchft/checkpoint.py:134`、`torchtitan/experiments/torchft/checkpoint.py:137`、`torchtitan/experiments/torchft/checkpoint.py:139`）。

每次 `_save()` 先无条件触发 per-replica `_ft_save()`；只有当前 participating rank 0 才继续调用 DCP 完整保存，其他 replica 返回 false（`torchtitan/experiments/torchft/checkpoint.py:143`、`torchtitan/experiments/torchft/checkpoint.py:145`、`torchtitan/experiments/torchft/checkpoint.py:150`、`torchtitan/experiments/torchft/checkpoint.py:163`）。2026-08-19 的提交 `296624f1c` 专门修正了这里的返回契约：dataloader side channel 不算“写了完整 checkpoint”，bystander 必须报告 false。测试固定了 participating/bystander 两个结果（`torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py:196`、`torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py:205`、`torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py:209`）。

per-replica 路径不受普通 interval 控制，始终用 Async mode写入 `ft-replicat-{id}` 文件夹；发起下一次前等待上一次 future（`torchtitan/experiments/torchft/checkpoint.py:201`、`torchtitan/experiments/torchft/checkpoint.py:204`、`torchtitan/experiments/torchft/checkpoint.py:206`、`torchtitan/experiments/torchft/checkpoint.py:208`）。加载时先从 replica 文件夹恢复 dataloader，再走普通完整 checkpoint；普通 load state 集合会移除 dataloader，避免后一步覆盖刚恢复的 replica 游标（`torchtitan/experiments/torchft/checkpoint.py:168`、`torchtitan/experiments/torchft/checkpoint.py:174`、`torchtitan/experiments/torchft/checkpoint.py:176`）。

### ④ 约束/代价/失败边界

- 关闭 per-replica dataloader checkpoint 被明确警告会重复训练数据并可能过拟合（`torchtitan/experiments/torchft/checkpoint.py:60`、`torchtitan/experiments/torchft/checkpoint.py:62`、`torchtitan/experiments/torchft/checkpoint.py:104`）。
- 每步保存游标减少 replay，却增加 metadata/I/O 与 future 管理开销；当前没有按风险自适应 interval。
- TorchFT manager 当前继承具体 DCP `CheckpointManager`，而不是只实现 `BaseCheckpointManager`；公共 checkpoint bridge 的提交 `6bc2108e9` 也明确记录 TorchFT 仍与 DCP 实现绑定（`torchtitan/experiments/torchft/checkpoint.py:26`、`torchtitan/experiments/torchft/checkpoint.py:44`）。
- `_ft_save()` 即使普通 checkpoint async mode disabled 也会产生 future，所以它必须覆盖 DCP wait 逻辑；不能复用“disabled 时无 future”的假设（`torchtitan/experiments/torchft/checkpoint.py:180`、`torchtitan/experiments/torchft/checkpoint.py:186`）。

---

## 5. Semi-sync：为什么关闭逐步 quorum，再把一致性放到周期边界

### ① 背景/问题

逐步同步让所有活跃 replicas 每个 optimizer step 达成一致，容错语义直接，但跨站点或慢副本会限制吞吐。LocalSGD/DiLoCo 的直觉相反：各 island 连续做多个本地 step，只在周期边界同步。若两者同时启用，既付出逐步 quorum 成本，又失去 semi-sync 的算法含义。

### ② 为什么这么设计

**选中的路线**是 `semi_sync_method is not None` 时关闭 async quorum/FSDP replicate hook/FT optimizer wrapper，由 LocalSGD 或 DiLoCo context 管理同步；**替代方案**是在逐步动态 all-reduce 之上再做周期参数同步。决定性标准是每种训练算法只有一个跨 replica 一致性边界。

### ③ 实现思路与细节

Manager 以 `semi_sync_method is None` 决定 `use_async_quorum`（`torchtitan/experiments/torchft/manager.py:110`）。因此 semi-sync 时不创建 managed replicate PG，loss 也不额外跨 replica sync（`torchtitan/experiments/torchft/manager.py:124`、`torchtitan/experiments/torchft/manager.py:156`、`torchtitan/experiments/torchft/manager.py:160`）。optimizer container 同样把 `_use_ft_optimizer` 初始化为 false（`torchtitan/experiments/torchft/optimizer.py:46`）。

训练循环用 `maybe_semi_sync_training()` 包住整个 step loop（`torchtitan/experiments/torchft/trainer.py:503`、`torchtitan/experiments/torchft/trainer.py:521`、`torchtitan/experiments/torchft/trainer.py:539`）。DiLoCo 可先按 fragment_fn 切模型，为每个 fragment 建外层 SGD，并传 inner optimizer、sync interval、quantization 和 delayed fragment update；LocalSGD 则接整模型与单一 optimizer（`torchtitan/experiments/torchft/manager.py:193`、`torchtitan/experiments/torchft/manager.py:199`、`torchtitan/experiments/torchft/manager.py:208`、`torchtitan/experiments/torchft/manager.py:218`）。未知 method 直接失败（`torchtitan/experiments/torchft/manager.py:225`）。

### ④ 约束/代价/失败边界

- `fragment_sync_delay` 用通信/计算 overlap 换模型质量，Config 注释直接说明这是算法质量权衡，不是透明性能开关（`torchtitan/experiments/torchft/config/job_config.py:38`、`torchtitan/experiments/torchft/config/job_config.py:45`）。
- `fragment_update_alpha` 改变本地/全局参数混合，同样可能改变收敛（`torchtitan/experiments/torchft/config/job_config.py:52`、`torchtitan/experiments/torchft/config/job_config.py:54`）。
- semi-sync 路径不提供逐步跨 replica loss sync；监控系统必须理解每个 island 的 loss 不必逐步相等。
- DiLoCo 的外层 optimizer 超参数当前在代码中固定为 SGD lr=0.7/momentum=0.9/nesterov，而不是 config 字段（`torchtitan/experiments/torchft/manager.py:199`、`torchtitan/experiments/torchft/manager.py:203`）。这是可调性边界。

---

## 6. 当前实验债务：为什么“继承 Trainer”不等于自动获得核心演进

### ① 背景/问题

FaultTolerantTrainer 名义上继承核心 Trainer，但它重写了完整 `__init__`、`train_step` 和 `train`。当核心 Trainer 加入 config-tree override、Module protocol verification、non-finite step gate 或新的组件顺序时，Python 继承不会自动把这些方法内部改动合并进来。

### ② 为什么这么设计

**当前选中的路线**是实验 Trainer 复制 orchestration 并在具体位置插 FT 逻辑；**替代方案**是核心 Trainer 提供 lifecycle hooks/可替换组件，使 FT 只覆写少量边界。现有路线的决定性短期收益是能快速控制 distributed init、dataloader、optimizer 和 checkpoint 顺序；代价是持续追赶核心演进。源码没有给出这是长期架构选择，因此应把它视为实验边界，而不是推荐扩展范式。

### ③ 实现思路与当前差异

FT `__init__` 从设备、distributed、data、model 到 validator 全部自行编排（`torchtitan/experiments/torchft/trainer.py:42`、`torchtitan/experiments/torchft/trainer.py:58`、`torchtitan/experiments/torchft/trainer.py:87`、`torchtitan/experiments/torchft/trainer.py:265`、`torchtitan/experiments/torchft/trainer.py:289`）。这条路径在模型 build 前没有调用核心 Trainer 当前的 config-tree `apply_overrides()` / Module protocol verification；核心路径对应逻辑位于 `torchtitan/trainer.py:342`、`torchtitan/trainer.py:347`、`torchtitan/trainer.py:359`、`torchtitan/trainer.py:365`。

FT `train_step()` 在 backward 后直接 clip grad → wait checkpoint staging → optimizer step（`torchtitan/experiments/torchft/trainer.py:429`、`torchtitan/experiments/torchft/trainer.py:443`、`torchtitan/experiments/torchft/trainer.py:450`）。核心 Trainer 当前还会跨 loss/PP mesh 聚合 finite gate，并把它和 grad norm 共同决定是否执行 step（`torchtitan/trainer.py:810`、`torchtitan/trainer.py:841`、`torchtitan/trainer.py:858`、`torchtitan/trainer.py:878`）。所以“核心 Trainer 已有 non-finite step gate”不能自动推广到 TorchFT baseline。

### ④ 约束/代价/失败边界

- 新增核心 Trainer 功能必须逐项检查 FT fork，而不能只跑核心 tests。
- 使用当前 override 机制接第三方 kernel时，TorchFT Trainer 不应被假设与核心 Trainer 等价；需单独验证 config replacement 是否被应用。
- FT 的 checkpoint tests覆盖 async future 与 participating-rank bool contract，但没有证明完整故障注入下 model/optimizer/data 三者原子一致。
- README 的命令/选项和 Config docstring已有 MCCL 漂移；实验文档应以当前代码与 exact baseline 为准。

### ⑤ 发展趋势（有源码锚点的推断）

8 月连续提交已让 TorchFT 跟随 Grain、flat token batch、公共 checkpoint bridge与 global-valid-token device 化（`1b04fc1c3`、`73aed7f6c`、`6bc2108e9`、`1f3ae096a`），说明它仍被主动同步；与此同时本节列出的 core-only guards 证明 fork drift 尚未消失。由此可推断更细的 Trainer lifecycle seam 会降低维护成本，但源码没有对应 RFC/TODO，不能把重构写成既定路线。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 标准 Trainer 与实验路径的总入口。
- [[01_torchtitan_trainer_quickstart]] —— 核心 Trainer 生命周期，用于逐项对照 TorchFT fork 的能力差异。
- [[02_torchtitan_data_pipeline_grain_analysis]] —— replica data shard 与 iterator state 怎样决定精确数据进度。
- [[03_torchtitan_checkpoint_state_recovery_analysis]] —— 公共 checkpoint manager、DCP 状态图和异步保存基线。
- [[11_torchtitan_fsdp_analysis]] —— 本地 FSDP/HSDP storage mesh、梯度归一化与 all-reduce hook 所在层。
- [[21_torchtitan_hsdp_backward_overlap_analysis]] —— 静态 HSDP replicate/shard 两级规约，与动态 managed PG 对照。
