---
title: "verl DataProto：当前共享的本地批数据契约"
---

# verl DataProto：当前共享的本地批数据契约

> **代码基准**：verl `main` @ `254a23edc62f25ebfae626e3932ae285d6f86009`（2026-08-28）
> **最后更新**：2026-08-31 · **定位**：DataProto/TensorDict 本地批契约唯一机制 owner
>
> **核心结论**：`DataProto` 仍是 verl 跨算法、worker API 和局部计算共享的批容器，但它不再是 V1
> controller 与所有 worker 之间的唯一传输协议。V1 的控制面主要流转 `KVBatchMeta`，真正执行前才从
> TransferQueue 物化 TensorDict；reward、advantage、算法函数和部分 RPC 边界仍会构造 `DataProto`。

---

## 1. 三种数据必须保持不同的不变量

当前 `DataProto` dataclass 由 `batch`、`non_tensor_batch` 和 `meta_info` 组成
（`verl/protocol.py:310-330`）：

| 容器 | 是什么 / 典型类型 | 怎么保持对齐 | 为什么必须独立 | 典型内容 |
|---|---|---|---|---|
| `batch` | 可批量张量运算的 `TensorDict` | 所有字段共享 batch size | 让 device move、切片和拼接以同一个 batch 维工作 | token、mask、log-prob、reward、advantage |
| `non_tensor_batch` | `dict[str, np.ndarray]` 的逐样本对象 | 第一维与 tensor batch 对齐 | raw Python/对象字段不能安全假装成同设备、同 dtype 的 tensor | raw prompt、uid、tool schema、data source |
| `meta_info` | 一次调用共享的 `dict` | 不按样本索引，切分时作为调用元数据传播 | global config/step 若被当作样本切分，会改变其作用域 | sampling params、timing、global step、validate |

`__post_init__` 不会替调用者修复全部字段；`check_consistency()` 才显式核对 TensorDict batch size、
numpy dtype/长度等约束（`verl/protocol.py:330-376,451-476`）。因此把一个逐样本列表误放进
`meta_info`，或者给 `non_tensor_batch` 写入不同长度数组，可能在后续索引、chunk 或 concat 才暴露。

这三分法不是类型清单，而是三个不同不变量的边界。`【分析推断】` 若全部塞进 TensorDict，工具 schema、raw message 等对象会被迫服从 tensor/device 语义；若全部塞进普通 dict，每次 reorder/chunk/concat 都要靠调用者手工维持行对齐；若把调用元数据混入逐样本数组，同一个 global step 还会产生无意义的复制。DataProto 选择保留三类容器，代价是字段归属必须由调用者明确，框架无法仅凭名字判断语义。

本页拥有容器语义。TransferQueue 的 key/tag/storage 见 [[16_verl_v1_transfer_queue_analysis]]；
AgentLoop 如何生成这些字段见 [[18_verl_agent_loop_reward_runtime_analysis]]。

## 2. 构造与转换不是无损猜测

`from_single_dict`/`from_dict` 按值类型把 torch tensor 与 numpy array 分流；`from_tensordict` 则保留已知
TensorDict 结构（`verl/protocol.py:477-582`）。调用者仍要明确 `meta_info`，框架不会从字段名自动判断
某个对象究竟是逐样本数据还是全局配置。

`【分析推断】` 不按字段名猜测构成一道正确性边界：`uid`、`index`、`temperature` 之类名字本身并不能证明作用域，项目扩展也可以引入同名但不同语义的字段。显式归类牺牲了一点构造便利，换来转换、切片和合并时可检查的作用域；错误归类会 fail late，因此新增字段必须在生产者处记录其容器与对齐规则。

常用转换边界：

- `to(device)` 只迁移 TensorDict；Python/numpy 对象不会自动上设备（`verl/protocol.py:583-596`）；
- `to_tensordict()` 将 tensor 与 non-tensor 字段合成嵌套 TensorDict，要求 batch 对齐
  （`verl/protocol.py:1099-1124`）；
- `select`、`pop`、`rename` 改变字段集合，`union` 合并两个 proto，但冲突字段必须深度相等
  （`verl/protocol.py:597-631,718-796`）。

`union` 的约束很重要：不同 worker 不能用相同 key 返回语义不同的值并期待“后写覆盖”。tensor 和
numpy 字典分别通过 union helper 检查冲突（`verl/protocol.py:109-200`），这是防止静默污染的边界。

## 3. 索引、切分与拼接共同维护样本身份

`__getitem__`、`select_idxs` 和 `slice` 会同时索引 tensor 与 non-tensor batch，meta 保持为调用级信息
（`verl/protocol.py:343-376,632-717`）。所以 batch permutation 不能只重排 `input_ids`；`uid`、
`data_source`、tool fields 与 reward extra info 必须同序移动。

`chunk(chunks)` 按份数拆分，要求 batch size 能被 chunks 整除；`split(split_size)` 按目标大小拆分；
`concat` 重新拼接 TensorDict 和 numpy batch，并校验 meta compatibility
（`verl/protocol.py:861-959`）。`repeat`、`reorder`、`sample_level_repeat` 等操作同样必须覆盖两类逐样本
容器（`verl/protocol.py:960-1098`）。

| 操作 | 主要用途 | 最容易破坏的不变量 |
|---|---|---|
| `select_idxs` / `reorder` | balance、filter、group sampling | uid 与 tensor 行错位 |
| `chunk` / `split` | DP/RPC/micro-batch | 不能整除或 meta 被误当逐样本 |
| `concat` | collect、多批 rollout 合并 | key 集合、dtype、meta 冲突 |
| `repeat` | 每 prompt 多条 rollout | interleave 顺序与 group identity |
| `union` | 添加 worker 计算结果 | 同名字段语义冲突 |

V1 ReplayBuffer 主要选择 `KVBatchMeta` key；真正取回字段后仍需要保持同样的样本对齐。TQ 改变的是
数据所在位置和物化时机，不是取消 batch invariant。

## 4. padding 是显式可逆变换

`pad_dataproto_to_divisor` 复制尾部样本，把 batch size 补到 divisor 的倍数并返回 `pad_size`；
`unpad_dataproto` 依靠该计数移除补样本（`verl/protocol.py:74-108`）。实例级 `padding()` 也会设置
padding 元数据，`is_padding_enabled()` 用来识别此状态（`verl/protocol.py:837-860`）。

padding 的语义不是“新增真实训练样本”。collect 之后必须在统计、reward、验证输出或梯度聚合进入不可逆
步骤前去掉 padding；否则重复尾样本会改变指标和 loss 权重。是否需要 padding 由调用协议和 world size
决定，不是 DataProto 能独立判断的系统策略。

## 5. 序列化与 future 的真实成本

自定义 `__getstate__`/`__setstate__` 将 TensorDict 拆成 tensor metadata 与底层数据，配合 Ray/序列化路径
重建（`verl/protocol.py:377-422`）；`save_to_disk`/`load_from_disk` 是容器级持久化工具
（`verl/protocol.py:423-432`）。这些方法能搬运对象，但不提供训练 checkpoint 的跨状态一致性；恢复协议
由 [[23_verl_training_checkpoint_recovery_analysis]] 负责。

`DataProtoFuture` 保存一组 Ray object refs 与 collect function。`get()` 才 materialize refs 并执行 collect；
future 还可再次 `chunk`，前提是每个 future 的 chunk 数与目标一致
（`verl/protocol.py:1171-1226`）。因此它提供的是 latency overlap，不是事务：一部分 remote method 已修改
状态、另一部分失败时，没有容器级 rollback。

具体说，**它是什么**：跨 WorkerGroup 调用的延迟结果句柄；**怎么做**：先保存各 rank 的 Ray refs 与 collect/可选 dispatch 函数，直到 `get()` 才 `ray.get`、拼接并二次切分；**为什么**：让 driver 不必在生产者 RPC 后立即拉回完整 batch，输出还可以直接作为下一次 group 调用的输入。它没有缓存一个可回滚的业务快照，所以延迟物化减少的是 driver 等待与搬运屏障，不是 partial failure 风险。

大 batch 的隐藏成本包括：

- Python object/numpy `dtype=object` 的序列化；
- controller 侧 concat 与复制；
- full proto 经 Ray object store 的额外驻留；
- `to(device)` 只移动 tensor，导致非 tensor 数据仍留在 host。

V1 用 TQ 引用流减少 controller 搬运，正是为了把这些成本从“每个 role 必经”降为“执行点按需物化”。

## 6. 当前 V1 中的实际角色

V1 reward、advantage 等局部路径从 TQ 取字段后构造 TensorDict/DataProto，执行算法，再把新增字段写回
（`verl/trainer/ppo/v1/trainer_base.py:1436-1472,1550-1707`）。AgentLoop runtime 也使用 DataProto 接收
prompt batch，并把 padded token、mask、score 与 extra fields 组回 DataProto
（`verl/experimental/agent_loop/agent_loop.py:537-629,698-836,1026-1118`）。

controller-worker 的主契约则不同：`KVBatchMeta` 只带 key、field 和 tag 选择；`tqbridge` 装饰的 worker
method 在调用前取 TensorDict，执行后把新字段写回（`verl/utils/transferqueue_utils.py:347-477`）。

因此当前正确表述是：

> DataProto 是共享的本地批语义和兼容容器；TransferQueue/KVBatchMeta 是 V1 的跨边界数据面。

## 7. 失败边界与检查表

1. 新增逐样本字段时，确认它进入 `batch` 还是 `non_tensor_batch`，并与 batch size 对齐；
2. 任何 filter/reorder/repeat 同时检查 uid、data source、tool/reward extra fields；
3. concat 前核对 key 集合、dtype、device 和 meta compatibility；
4. padding 后记录并在正确边界 unpad；
5. 不把 Ray future 完成写成业务状态已原子提交；
6. 不把 `save_to_disk` 外推成 Trainer 可恢复性；
7. V1 问题先判断发生在 TQ 引用、物化 bridge，还是物化后的 DataProto 算法。

## Related Pages

- [[11_verl_single_controller_analysis]] —— DataProto 在 group dispatch/collect 中的调用边界。
- [[15_verl_rl_algorithms_analysis]] —— advantages、returns、uid 与 mask 的算法语义。
- [[16_verl_v1_transfer_queue_analysis]] —— V1 key/tag 与延迟物化数据面。
- [[18_verl_agent_loop_reward_runtime_analysis]] —— prompt 如何形成 token、mask、score 与 extra fields。
- [[20_verl_ray_trainer_analysis]] —— 仍以完整 DataProto 驱动的 V0 legacy lifecycle。
- [[23_verl_training_checkpoint_recovery_analysis]] —— 容器持久化与训练恢复协议的区别。
