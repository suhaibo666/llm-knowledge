---
title: "TorchTitan Checkpoint：状态图、异步保存与恢复边界"
---

# TorchTitan Checkpoint：状态图、异步保存与恢复边界

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：TorchTitan checkpoint 的核心不是 `dcp.save()`，而是先定义一个可恢复的训练状态图，再把模型、优化器、scheduler、数据游标和 Trainer step 映射到后端。当前真正闭环的是 DCP manager；新 `torch_checkpointing` manager 只完成了 config、storage、resharder 和 barrier plumbing，所有 load/save 路由仍明确抛 `NotImplementedError`。把“可构建”误写成“可保存”是当前最危险的知识遗漏。
>
> 本页讨论状态所有权、保存/恢复决策与存储后端边界；Grain iterator 内部如何精确续训见 [[02_torchtitan_data_pipeline_grain_analysis]]，FSDP 参数与优化器分片本身见 [[11_torchtitan_fsdp_analysis]] 与 [[26_torchtitan_flex_shard_dist_muon_analysis]]。

---

## 1. Overview：Checkpoint 同时服务“继续训练”和“导出模型”，两者不是同一制品

训练故障恢复要求恢复模型、优化器、LR scheduler、数据位置、step 与 token 计数；发布或下游加载通常只需要模型权重，还可能要求 HF key、safetensors 和指定 dtype。如果让一个“最后保存”开关隐式兼任二者，很容易得到不能续训的 model-only artifact，或把巨大的 optimizer state 当作发布制品。

TorchTitan 把这两个目标放进同一 manager，但通过状态筛选和最后一步策略显式分开。Trainer 在所有 stateful component 构建完成后，把 dataloader、model parts、optimizers、schedulers 和自身 `train_state` 一并注入 manager（`torchtitan/trainer.py:546`、`torchtitan/trainer.py:564`、`torchtitan/trainer.py:570`）。DCP manager 再决定本次是 seed/model-only load、fault-tolerance full resume、周期 full save，还是 final model export。

| 目标 | 状态集合 | 格式/变换 | 是否保证可继续训练 |
|---|---|---|---|
| 周期 checkpoint | model + optimizer + scheduler + dataloader + train_state | native DCP，可同步/异步 | 是，受 RNG/拓扑边界限制 |
| seed checkpoint | step 0 model | native DCP | 否，只作为初始化权重 |
| initial model load | model only | native DCP 或 HF safetensors | 否，其他状态重新初始化 |
| fault-tolerance resume | checkpoint folder 中最新/指定 full state | native DCP | 是 |
| final export | model only，可转 dtype/HF | 同步保存 | 否 |
| final full save | full state，不转 dtype | native DCP | 是 |

```text
Trainer state graph
  model parts ----> ModelWrapper ----+
  optimizers ------------------------+
  lr schedulers ---------------------+--> BaseCheckpointManager
  Grain dataloader ------------------+       |
  train_state step / ntokens --------+       +--> DCP manager  当前闭环
                                              +--> torch_checkpointing  仅 plumbing

load precedence:
  existing valid run checkpoint > explicit initial_load_path > HF assets > fresh start
```

### Quick Start：选择“恢复”还是“导出”

```python
config.checkpoint.enable = True
config.checkpoint.interval = 500
config.checkpoint.async_mode = "async_with_pinned_mem"
config.checkpoint.keep_latest_k = 2
config.checkpoint.last_save_model_only = False  # 最后一步也要能续训
```

默认 concrete config 是 DCP `CheckpointManager.Config`；它只比共享 base config 多一个 `async_mode` 字段（`torchtitan/components/checkpointer/dcp.py:135`）。Trainer 在进入循环前调用 `load(load_step)`，每步按策略 `save()`，关闭前等待/清理资源（`torchtitan/trainer.py:948`、`torchtitan/trainer.py:974`、`torchtitan/trainer.py:1020`）。

---

## 2. Common manager bridge：为什么抽象的是策略与状态，而不是加一个 backend 字符串

### ① 背景/问题

DCP 与新 `torch_checkpointing` 后端的 IO、barrier、resharding 和 storage 类型不同，但 Trainer 关心的语义相同：何时保存、恢复哪些状态、保留几份、禁用时是否 no-op、最后制品是什么。若在 Trainer 中按 backend 写分支，每引入一个 manager 都要复制策略；若只定义松散 Protocol，遗漏 `close` 或 async wait 直到故障时才暴露。

### ② 为什么这么设计

提交 `6bc2108e9` 明确选择 `BaseCheckpointManager(Configurable, ABC)`，而不是 backend selector 或纯 Protocol。**选中路线**是“config 类型即实现选择”：每个 manager 声明自己的 nested Config，`Configurable` 的 owner wiring 负责构造；**替代方案**是在共享 config 中增加 backend 枚举并持续扩展条件分支。决策准则是扩展隔离和构造期失败：漏实现抽象 hook 的 manager 不能实例化，而不是在训练中途才 AttributeError。

### ③ 实现思路与细节

Base 公开 `load/save/maybe_wait_for_staging/close`，统一处理 disabled manager，再派发 `_load/_save/...` hook（`torchtitan/components/checkpointer/base.py:202`、`torchtitan/components/checkpointer/base.py:209`、`torchtitan/components/checkpointer/base.py:215`、`torchtitan/components/checkpointer/base.py:221`）。这条 guard 由提交 `862f966a3` 从各 concrete manager 收回 base，原因是 disabled manager 会在 `__init__` 早退、属性根本不存在；让每个子类记得检查会产生漏网路径。

Storage 也被压缩成四个路径能力：`isdir/isfile/listdir/remove`（`torchtitan/components/checkpointer/base.py:147`、`torchtitan/components/checkpointer/base.py:170`）。提交 `b5bcd76c6` 的理由是 discovery/retention 策略相同，差异只是由谁回答路径操作；自有 `CheckpointStorage` seam 保留 `str`，不会让 `Path` 把 `gs://` 双斜线折叠（`torchtitan/components/checkpointer/base.py:158`）。

### ④ 约束/边界

- concrete manager 必须重新声明 nested `Config`；继承不声明会让 owner 仍指向 abstract base（`torchtitan/components/checkpointer/base.py:187`、`torchtitan/components/checkpointer/base.py:190`）。
- public 方法不应被子类覆盖，否则会绕过 disabled guard；源码把这个不变量直接写在 base 注释中（`torchtitan/components/checkpointer/base.py:202`、`torchtitan/components/checkpointer/base.py:205`）。
- `CheckpointStorage` 的 runtime check 只证明方法名存在，不证明远程 URI 真可达（`torchtitan/components/checkpointer/base.py:165`）。远程能力仍由 concrete backend 决定。
- `keep_latest_k=1` 被拒绝，因为最新 checkpoint 可能仍在保存，至少要保留两份副本；0 表示不清理（`torchtitan/components/checkpointer/base.py:308`、`torchtitan/components/checkpointer/base.py:334`、`torchtitan/components/checkpointer/base.py:336`）。

---

## 3. 状态图与扁平化：为什么 PP 下不能直接拼原生 optimizer state_dict

### ① 背景/问题

Pipeline Parallel 每个 rank 都有本地 optimizer；它们的 param group 常从索引 0 开始，但实际对应不同 layer。直接合并 index-keyed optimizer state 会碰撞，复杂 schedule 在同一 rank 上还有多个 model chunk/optimizer，使冲突更严重。模型权重还可能由 state_dict hook 临时拆分 fused parameter；每次保存新分配 tensor 会破坏 async staging 的 pinned-buffer 复用。

### ② 为什么这么设计

**选中的路线**是先把模型和 optimizer 状态转换成全局 FQN-keyed flat view，再交给 DCP；**替代方案**是保留各 rank 的局部 index key，并让 checkpoint backend 猜它们对应哪个全局参数。决策准则是跨 PP stage/chunk 的唯一身份。DCP manager 的类文档用 PP optimizer collision 说明了这个问题，并明确由 `OptimizersContainer` 扁平化解决（`torchtitan/components/checkpointer/dcp.py:93`、`torchtitan/components/checkpointer/dcp.py:102`、`torchtitan/components/checkpointer/dcp.py:106`）。

### ③ 实现思路与细节

manager 的 `states` 先包含外部 `train_state`，再加入统一 key：`model`、`optimizer`、`dataloader`、`lr_scheduler`（`torchtitan/components/checkpointer/dcp.py:151`、`torchtitan/components/checkpointer/dcp.py:172`）。`ModelWrapper` 把多个 model part 的 state_dict 合成一个 flat FQN view（`torchtitan/components/checkpointer/base.py:86`、`torchtitan/components/checkpointer/base.py:119`）。

它还缓存 tensor storage：每次重新取 state_dict 后，shape/dtype 不变且 storage 不同的 hook 产物被 copy 到旧缓存，而不是替换对象（`torchtitan/components/checkpointer/base.py:123`、`torchtitan/components/checkpointer/base.py:128`、`torchtitan/components/checkpointer/base.py:134`）。选择稳定 storage 的判据是 async DCP staging 可按 source storage 复用 pinned host buffer（`torchtitan/components/checkpointer/base.py:96`）。

保存前 `_flattened_model_states_sd()` 把 model FQN 提到顶层，其他 stateful component 保持顶层对象交给 DCP 递归处理（`torchtitan/components/checkpointer/dcp.py:717`、`torchtitan/components/checkpointer/dcp.py:734`）。加载 native DCP 后，因为 model key 已被展开，当前仍要显式调用 `ModelWrapper.load_state_dict()`；源码保留 TODO 希望消除这一手动步骤（`torchtitan/components/checkpointer/dcp.py:393`、`torchtitan/components/checkpointer/dcp.py:396`）。

### ④ 约束/边界

- model 结构在 wrapper 创建后不能改 key 或替换 tensor reference，否则 stable cache 会陈旧；源码在 `ModelWrapper` notes 中明确禁止（`torchtitan/components/checkpointer/base.py:107`、`torchtitan/components/checkpointer/base.py:110`）。
- `strict=False` 是因为 flat dict 混有非模型 key，不代表模型 key 任意缺失都安全（`torchtitan/components/checkpointer/base.py:138`）。格式兼容仍由 state-dict adapter 和模型测试承担。
- LR scheduler 与 optimizer 状态耦合；排除 optimizer 却加载 scheduler 的配置会提前失败，而不是尝试修补新 optimizer 的 LR（`torchtitan/components/checkpointer/base.py:341`、`torchtitan/components/checkpointer/base.py:343`）。
- `exclude_from_loading` 不允许排除 model；未知 key 在真正构建 load state 时抛错（`torchtitan/components/checkpointer/base.py:341`、`torchtitan/components/checkpointer/dcp.py:760`）。

---

## 4. 三种保存路径：为什么“异步”必须拆成 staging 完成与持久化完成

### ① 背景/问题

Checkpoint 会从 GPU/CPU state 读取大量 tensor 并写入存储。同步保存阻塞训练；直接在线程中异步写可以把 IO 移出主路，但若后台仍读取 live model storage，下一步更新参数会与保存竞争。Pinned-memory staging 先复制出一致快照，再让训练与 upload 重叠，但它引入两种不同的完成时刻和额外 host memory。

### ② 为什么这么设计

DCP manager 提供 `disabled`、`async`、`async_with_pinned_mem` 三种模式（`torchtitan/components/checkpointer/dcp.py:58`、`torchtitan/components/checkpointer/dcp.py:135`）。**选中的高级路线**是把 staging future 与 upload future 分开；**替代方案**是只暴露一个“异步完成”事件。决策准则是训练只需等快照复制完成即可安全改 live state，而开始下一次 checkpoint/退出则必须等 upload 完成。

### ③ 实现思路与细节

`dcp_save()` 根据模式分别调用 `dcp.save()`、普通 `dcp.async_save()`，或带 `DefaultStager`/PROCESS checkpointer 的 `dcp.async_save()`（`torchtitan/components/checkpointer/dcp.py:249`、`torchtitan/components/checkpointer/dcp.py:315`、`torchtitan/components/checkpointer/dcp.py:322`、`torchtitan/components/checkpointer/dcp.py:331`）。

Pinned 路径首次保存才构造 stager，并打开 pinned/shared memory、async staging 与 non-blocking copy；返回值拆成 `staging_completion` 与 `upload_completion`（`torchtitan/components/checkpointer/dcp.py:445`、`torchtitan/components/checkpointer/dcp.py:448`、`torchtitan/components/checkpointer/dcp.py:464`）。训练步末尾调用 `maybe_wait_for_staging()`，确保 live state 不再被读取；下一次实际保存前 `maybe_wait_for_saving()` 则等待前一 upload（`torchtitan/components/checkpointer/dcp.py:613`、`torchtitan/components/checkpointer/dcp.py:639`、`torchtitan/components/checkpointer/dcp.py:642`、`torchtitan/components/checkpointer/dcp.py:662`）。

保存调度先检查 interval/first/final/load-only，启动前等待旧 save，再在成功发起后异步清理旧副本（`torchtitan/components/checkpointer/dcp.py:401`、`torchtitan/components/checkpointer/dcp.py:422`、`torchtitan/components/checkpointer/dcp.py:427`、`torchtitan/components/checkpointer/dcp.py:488`）。

### ④ 约束/边界

- Pinned staging 用 host memory 换训练停顿，不能描述成零成本；stager 还依赖 ModelWrapper 的 stable storage 才能持续复用 buffer。
- 当前同一时间只追踪一个 `save_future`；新 checkpoint 前会等上一个完成，不是无限并发写队列。
- `keep_latest_k` purge 只由 rank 0 发起，后台删除失败会记录 warning 后继续，避免线程一次异常后永久停止（`torchtitan/components/checkpointer/base.py:266`、`torchtitan/components/checkpointer/base.py:53`、`torchtitan/components/checkpointer/base.py:58`）。
- final save 总是同步，即使周期保存配置为 async，目的是进程退出前产生完整制品（`torchtitan/components/checkpointer/dcp.py:811`、`torchtitan/components/checkpointer/dcp.py:814`）。

---

## 5. Load 决策树：为什么已有 run checkpoint 必须压过 initial model 权重

### ① 背景/问题

自动重启作业通常沿用原命令，其中仍带着初始 HF/native 权重路径。若每次启动都优先读 initial path，故障恢复会把已经训练的 run state 覆盖回 step 0；反过来，如果显式要求某个不存在 step 却静默 fresh start，也会丢训练进度。

### ② 为什么这么设计

**选中的优先级**是：输出 checkpoint folder 中有效 run checkpoint > initial path/HF assets > fresh start；显式 `load_step` 不存在则硬失败。**替代方案**是让 initial path 永远优先或缺失时静默回退。决策准则是自动重启的幂等性与防止无声数据损失。源码把 existing run branch 直接标成 fault-tolerance branch，并说明所有 `initial_*` 选项按设计被忽略（`torchtitan/components/checkpointer/dcp.py:576`、`torchtitan/components/checkpointer/dcp.py:579`）。

### ③ 实现思路与细节

`_find_load_step()` 只接受 `step-N` 且目录内存在 DCP `.metadata` 或 HF index 的 candidate，返回最大 step（`torchtitan/components/checkpointer/dcp.py:665`、`torchtitan/components/checkpointer/dcp.py:691`、`torchtitan/components/checkpointer/dcp.py:699`、`torchtitan/components/checkpointer/dcp.py:709`）。显式 step 而 folder/step 不存在分别在入口抛 `FileNotFoundError`（`torchtitan/components/checkpointer/dcp.py:525`、`torchtitan/components/checkpointer/dcp.py:586`）。

没有 run checkpoint 时，manager 才读取 `initial_load_path`；HF load 只能 model-only，并依赖 model-specific state-dict adapter 在 native/HF FQN 之间双向转换（`torchtitan/components/checkpointer/dcp.py:531`、`torchtitan/components/checkpointer/dcp.py:536`、`torchtitan/components/checkpointer/dcp.py:544`、`torchtitan/components/checkpointer/dcp.py:378`）。若既没有 path 也不是 HF load，则返回 fresh start（`torchtitan/components/checkpointer/dcp.py:572`）。

### ④ 约束/边界

- `load_step` 只允许 `-1` 或非负数；这项校验由提交 `d5f414faa` 加入，避免生成 `step--2` 路径（`torchtitan/components/checkpointer/base.py:311`、`torchtitan/components/checkpointer/base.py:332`）。
- latest-step discovery 对 remote storage 不是高效批量查询：每个 step 最多做两次 isfile round trip；源码只认为“加载时执行一次，暂可接受”（`torchtitan/components/checkpointer/dcp.py:680`）。
- native DCP remote URI 可由 fsspec storage adapter 支持；remote HF safetensors load/save 被配置层提前拒绝（`torchtitan/components/checkpointer/base.py:373`、`torchtitan/components/checkpointer/base.py:376`、`torchtitan/components/checkpointer/base.py:381`）。
- 模型 state adapter 是格式映射，不自动证明不同模型 config 兼容；例如 adapter 可对 RoPE 类型做额外验证（`torchtitan/protocols/state_dict_adapter.py:113`）。

---

## 6. 恢复制品与导出制品：为什么 final model-only 不能冒充 full checkpoint

### ① 背景/问题

用户常把“最后一步 checkpoint”理解为最完整的一份，但默认 `last_save_model_only=True` 实际更接近发布 artifact。它会去掉 optimizer、scheduler、data cursor 和 train step，还可能转成另一 dtype/HF key；用它继续训练会丢失状态。

### ② 为什么这么设计

**选中的路线**是由 `last_save_model_only` 明确选择导出或完整恢复制品；**替代方案**是始终保存 full state，再另写转换脚本。当前路线降低发布操作成本，但要求文档和配置明确区分用途。源码在 `_save_last_step()` 中直接声明：full state 不做 dtype conversion 以保证可安全续训，model-only 则假设训练完成后按 export dtype 保存（`torchtitan/components/checkpointer/dcp.py:782`、`torchtitan/components/checkpointer/dcp.py:784`）。

### ③ 实现思路与细节

model-only 分支只取 `ModelWrapper.state_dict()`，对所有 floating tensor 按需 `.to(export_dtype)`，保留 integer buffer 与已是目标 dtype 的 tensor（`torchtitan/components/checkpointer/dcp.py:792`、`torchtitan/components/checkpointer/dcp.py:795`）。提交 `575dd2f03` 修复了 BF16 training 导出 FP32 时旧逻辑不转换的 bug；当前判断比较每个 tensor 实际 dtype，而不是只看目标是不是 FP32。

HF export 先由 state adapter 把 native FQN 转成 HF FQN；有 shard mapping 时先写 `sharded/` 再 consolidate，否则 writer 可直接单文件合并（`torchtitan/components/checkpointer/dcp.py:284`、`torchtitan/components/checkpointer/dcp.py:290`、`torchtitan/components/checkpointer/dcp.py:338`）。

### ④ 约束/边界

- HF export 强制 model-only；没有 state adapter 会在 manager 构造时失败（`torchtitan/components/checkpointer/dcp.py:195`、`torchtitan/components/checkpointer/dcp.py:787`）。
- dtype conversion 只作用于导出字典，不修改 live training model；non-floating state 原样保留。
- 想让最后一步仍可续训必须设置 `last_save_model_only=False`；此时不会做 export dtype conversion，也不能保存 HF 格式。
- seed step 0 与 initial model-only load 同样不含训练状态，不能当作恢复点。

---

## 7. 两个 manager 的真实成熟度：DCP 已闭环，torch_checkpointing 尚未接通

### ① 背景/问题

源码中新出现 `TorchCheckpointingManager`、默认 resharder、pinned stager 和 TCPStore barrier，表面上很容易被解读为“第二个可用 checkpoint backend”。但组件能被 config 构造与真正实现 save/load 是两回事；文档若只列类名会掩盖运行时硬失败。

### ② 为什么这么设计

提交 `7c6f9f734` 明确采用独立 manager，而不是向 DCP manager 塞 backend branches；它也明确本次只做 save config plumbing，路由留给后续 changes。**选中的增量路线**是先稳定共用 contract 与 backend config；**替代方案**是一次性接入所有 load/save 路径。源码没有说明为何分批落地的更高层决策，因此不能推断成性能已经验证。

### ③ 实现思路与细节

新 manager 确实构造了 model/optimizer 的 `DefaultResharder` item spec、pinned staging、TCPStore barrier 和 backend manager（`torchtitan/components/checkpointer/torch_checkpointing.py:81`、`torchtitan/components/checkpointer/torch_checkpointing.py:97`、`torchtitan/components/checkpointer/torch_checkpointing.py:119`、`torchtitan/components/checkpointer/torch_checkpointing.py:185`）。它也复用共享 state keys 和 config policy（`torchtitan/components/checkpointer/torch_checkpointing.py:158`）。

但四个运行 hook `_load/_save/_wait_for_saving/_maybe_wait_for_staging` 当前全部直接抛 `NotImplementedError`（`torchtitan/components/checkpointer/torch_checkpointing.py:193`、`torchtitan/components/checkpointer/torch_checkpointing.py:197`、`torchtitan/components/checkpointer/torch_checkpointing.py:202`、`torchtitan/components/checkpointer/torch_checkpointing.py:207`、`torchtitan/components/checkpointer/torch_checkpointing.py:212`）。所以唯一准确结论是：**配置和后端资源 plumbing 已存在，启用后保存/加载尚不可用**。

### ④ 约束/边界

- checkpoint disabled 时 base guard 会让这些 stub 不被调用，因此 config 可构建/no-op 不代表 enable=True 可用。
- 新 manager 当前拒绝 remote URI，并提示改用 DCP；其 storage adapter 使用 `Path`，所以在构造时提前失败以防 URI 被改写（`torchtitan/components/checkpointer/torch_checkpointing.py:143`、`torchtitan/components/checkpointer/torch_checkpointing.py:151`）。
- DCP 仍是默认实现，也是当前唯一在本源码基线完成 Trainer save/load 闭环的 manager。
- “DefaultResharder 已配置”只表示 backend schema 声明，不等于 TorchTitan 路由已验证跨拓扑恢复。

### ⑤ 发展趋势（有源码锚点的推断）

四个 stub 注释明确说 routing 会在 later changes 落地（`torchtitan/components/checkpointer/torch_checkpointing.py:193`），因此可以把“接通路由”列为源码指向的演进方向；在对应提交出现前，不能把它写成当前能力。

---

## 8. 恢复边界与排障清单

### 当前硬边界

1. **RNG 未保存**：DCP load 路径有 TODO，要保存 rank-local training RNG；同拓扑恢复也不能由 checkpoint 层保证完全相同随机流（`torchtitan/components/checkpointer/dcp.py:594`）。
2. **数据拓扑不可变**：Grain dataloader state 拒绝 effective DP degree 改变，详见 [[02_torchtitan_data_pipeline_grain_analysis]]。
3. **HF 只承载模型**：optimizer/scheduler/data/train state 不在 HF safetensors 语义内。
4. **远程格式有限**：DCP native 支持 fsspec；HF remote 与新 manager remote 都不支持。
5. **retention 不是事务复制**：至少保留两份降低“最新正在写”的风险，但源码没有跨 bucket 原子提交协议。
6. **TorchFT 是独立实验层**：基础 DCP fault-tolerance resume 不等于 elastic membership/故障投票；TorchFT 的 optimizer/checkpoint 协议需要单独分析。

### 排障顺序

1. 先确认使用的是 DCP `CheckpointManager.Config`，不是仍为 stub 的新 manager。
2. 区分输出是 full state 还是 final model-only artifact。
3. 检查 `checkpoint.folder` 是否已有带 `.metadata` 的有效 `step-N`；存在时 initial path 会被忽略。
4. 对显式 `load_step` 让缺失直接报错，不要改成 latest/fresh start 掩盖问题。
5. 核对 `exclude_from_loading`，尤其 optimizer/scheduler 的耦合。
6. 对 async pinned 模式分别观察 staging future 与 upload future，不能只看训练已继续就认为文件已落盘。
7. 续训不一致时分别检查 data cursor、Trainer step/token、optimizer/scheduler 和尚未保存的 RNG，不要只比 model weight。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/01_torchtitan_trainer_quickstart|TorchTitan Trainer Quickstart]] —— manager 在初始化、训练循环和关闭阶段的调用位置。
- [[02_engineering/02_train_frameworks/torchtitan/02_torchtitan_data_pipeline_grain_analysis|TorchTitan Grain 数据管道]] —— dataloader checkpoint 内部的 cursor、packing 与 DP 拓扑约束。
- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|TorchTitan FSDP2]] —— 参数持久分片、all-gather 与 DCP state_dict 的运行时来源。
- [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|TorchTitan PP]] —— 多 stage/chunk 导致 model/optimizer state 必须用全局 FQN 扁平化的原因。
- [[02_engineering/02_train_frameworks/torchtitan/26_torchtitan_flex_shard_dist_muon_analysis|FlexShard 与 DistMuon]] —— optimizer 计算布局与存储布局解耦后，checkpoint 仍以持久状态布局为准。
- [[02_engineering/04_posttrain_frameworks/10_rl_ppo_loss_and_grpo_analysis|TitanRL 异步 RL]] —— RL controller/rollout buffer 并未全部进入核心 Trainer checkpoint 状态图的对照边界。
