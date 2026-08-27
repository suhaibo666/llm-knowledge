---
title: "Megatron-LM 分布式 Checkpoint 深度解析(Distributed Checkpointing)"
---

# Megatron-LM 分布式 Checkpoint 深度解析(Distributed Checkpointing)

> **源码基线**：`NVIDIA/Megatron-LM@ee3f1ffa2acd18131ab67cabab4cec45283512ab`（`dev`，2026-05-19）
> 核心文件:`megatron/core/dist_checkpointing/` 下 `megatron/core/dist_checkpointing/mapping.py`(`ShardedTensor`)、`megatron/core/dist_checkpointing/serialization.py`(`save`/`load`)、`strategies/`、`megatron/core/dist_checkpointing/validation.py`
> 配套阅读:`17_megatron_parallelism_orchestration_analysis.md`、`16_megatron_distributed_optimizer_analysis.md`、`27_megatron_tp_fsdp_resharding_supplements_analysis.md` §3(resharding)
> 定位:模型/优化器状态如何**存盘与读取**。

---

## 0. 总览:它和 resharding 不是一回事

两者都处理"并行布局不同",但:

| | **dist_checkpointing**(本文) | **resharding / refit**(`27_megatron_tp_fsdp_resharding_supplements_analysis.md` §3) |
|--|------------------------------|--------------------------------|
| 干什么 | 模型/优化器状态**存盘 / 从盘读取** | 两个**运行中**的模型之间**实时**搬权重 |
| 介质 | 磁盘 | GPU↔GPU(NCCL/NVSHMEM/Gloo) |
| 场景 | 训练存档、断点续训、换集群续训 | RL:训练模型 → 推理模型 |
| 共同点 | 都要跨**不同并行布局** | |

本文讲 dist_checkpointing。

---

## 1. 动机:并行无关的存档

一个模型被 TP×PP×EP×DP 切散在成百上千张卡上 —— **每张卡只持有每个张量的一小片**。

朴素存档(每张卡存自己那片)的致命问题:**checkpoint 和当时的并行配置死死绑定**。用 `TP8×PP4` 训练存的档,**只能用 `TP8×PP4` 加载**。想换并行度续训、换集群规模、或训练用一套布局而推理用另一套 —— 全做不到。

`dist_checkpointing` 的目标:**存成"并行无关"的格式**。checkpoint 里存的是逻辑上的**全局张量**;加载时每张卡声明"我要全局张量 X 的这一片",框架按需读对应切片。于是 **`TP8×PP4` 存、`TP2×PP1` 载** 也能跑。

---

## 2. 核心抽象:`ShardedTensor`

`megatron/core/dist_checkpointing/mapping.py:52`。整个子系统的核心 —— 描述**本地张量(本 rank 这一片)↔ 全局张量(逻辑全量)**的映射:

```python
@dataclass
class ShardedTensor(ShardedBase):
    key: str                       # 全局张量的唯一标识
    data: torch.Tensor             # 本 rank 持有的局部数据(校验时可为 None)
    dtype, local_shape             # 局部片的形状
    global_shape: Tuple            # 全局张量的形状
    global_offset: Tuple           # 本局部片在全局张量里的偏移(元素数)
    axis_fragmentations: Tuple     # 每个轴被切成几片
    replica_id: ReplicaId = 0      # 本片是第几个副本(DP 副本数据相同)
    prepend_axis_num: int = 0      # 全局张量比局部多出的前置轴数(如专家轴)
    allow_shape_mismatch: bool     # 允许全局形状不严格匹配(padded 张量)
    flattened_range: slice         # 局部片是某个扁平 buffer 的一段切片
```

读法:`(global_shape, global_offset, local_shape)` 三者一起,精确说明"本 rank 的 `data` 是全局张量 `key` 的哪一块"。关键字段:
- **`replica_id`**:DP 副本持有**完全相同**的数据。存档时只让 `replica_id` 为某个值的副本真正写盘,其余跳过 —— **避免 DP 冗余写**。
- **`prepend_axis_num`**:MoE 专家权重等,局部张量是单个专家、全局多了一个"专家"轴 —— 用前置轴表达。
- **`flattened_range`**:DDP 的扁平梯度 buffer、分布式优化器的扁平状态(见 `16_megatron_distributed_optimizer_analysis.md`)—— 局部片是大扁平 buffer 的一段,用这个字段定位。
- `from_rank_offsets`(`:190`):便捷构造器,从"本 rank 在各并行轴的 rank 号"直接算出 offset。
- `narrow`(`:262`):把一个 ShardedTensor 再切窄(load 时按需取子片)。

**正是 `ShardedTensor` 把"并行布局"这个信息编码进了 checkpoint** —— 存的时候各片拼成全局,读的时候按新布局重新切。

---

## 3. 其他 sharded 类型

`megatron/core/dist_checkpointing/mapping.py` 还有三类(都继承 `ShardedBase`):

| 类型 | 用途 |
|------|------|
| **`ShardedObject`**(`:360`) | 非张量对象的分片(如各 rank 各有一份的元数据);`unique_key` 标识 |
| **`ShardedTensorFactory`**(`:438`) | **延迟构造**:存/载时需要变换的张量(如分布式优化器的扁平 buffer 要先重组成逻辑张量)。`build()` 生成 ShardedTensor,`apply()` 做逆变换 |
| **`LocalNonpersistentObject`**(`:342`) | **不该持久化**的对象 —— rank 本地、加载时重算的东西。`save` 时直接丢弃 |

---

## 4. `sharded_state_dict` —— 模型如何声明自己的分片

普通 PyTorch 模型用 `state_dict()` 返回 `{名字: 张量}`。Megatron 的每个模块额外实现 **`sharded_state_dict()`** —— 返回的 dict 里每个张量被包成 `ShardedTensor`,带上正确的"局部↔全局"映射。

前面文档见过的例子:`ColumnParallelLinear.sharded_state_dict`(`12_megatron_tp_analysis.md`,权重按输出维切→沿 axis 0 分片)、`RowParallelLinear.sharded_state_dict`(沿 axis 1)、`DistributedOptimizer.sharded_state_dict`(`16_megatron_distributed_optimizer_analysis.md`,优化器状态按 DP 切→用 `flattened_range`)、`TEGroupedMLP` 的专家用 `prepend_axis_num` 表达专家轴。

所以 `sharded_state_dict` 是**模型对"我的每一片在全局的哪个位置"的自我声明** —— dist_checkpointing 的 `save`/`load` 完全靠它驱动。

---

## 5. `save` / `load` 流程

### 5.1 `save`(`megatron/core/dist_checkpointing/serialization.py:300`)

```
save(sharded_state_dict, checkpoint_dir, ...):
  ① 应用 ShardedTensorFactory(延迟构造展开)
  ② 抽出并丢弃 LocalNonpersistentObject
  ③ 抽出所有 ShardedBase 对象
  ④ rank 0 把"非分片"的普通对象存进 common.pt
  ⑤ (可选)抽出并存 ShardedObject
  ⑥ 存所有 ShardedTensor —— 按 strategy(§6),各片写进各自位置
  ⑦ 写 metadata.json(后端、版本)
```

- 步骤⑥可**异步**(`async_sharded_save=True`)—— 返回一个 `AsyncRequest`,真正写盘在后台跑,训练继续;此时⑦作为完成回调,**整档写完才落 metadata.json**(保证 checkpoint 原子性)。
- `validate_access_integrity` 触发 §7 的校验。

### 5.2 `load`(`megatron/core/dist_checkpointing/serialization.py:54`)

逆过程:每张卡用**当前(可能是新的)并行布局**生成自己的 `sharded_state_dict`(只填 metadata、`data=None`),`load` 据此从 checkpoint 里读出对应的全局张量切片,填回 `data`。**这一步就是"换并行布局加载"发生的地方** —— 新布局的 `global_offset`/`local_shape` 决定读哪一段。

辅助 API:`load_tensors_metadata` / `load_sharded_metadata`(不载数据、只看 checkpoint 里有什么)、`load_plain_tensors`、`load_common_state_dict`、`remove_sharded_tensors`。

---

## 6. 策略(`strategies/`)

`save`/`load` 的"怎么写盘/读盘"由可插拔 strategy 决定:

| strategy | 文件 | 作用 |
|----------|------|------|
| **TorchDist** | `megatron/core/dist_checkpointing/strategies/torch.py` | 基于 PyTorch 分布式 checkpoint(DCP)格式,`--ckpt-format torch_dist`,默认 |
| **fully-parallel** | `megatron/core/dist_checkpointing/strategies/fully_parallel.py` | **全并行存/载**:把写盘工作**摊给所有 rank**(而非只 rank 0),每 rank 写大致等量 → checkpoint 大幅提速;天然跳过 DP 副本冗余 |
| **async** | `megatron/core/dist_checkpointing/strategies/filesystem_async.py`、`megatron/core/dist_checkpointing/strategies/async_utils.py` | **异步存档**:`save` 把数据交给后台线程写盘,主训练流不阻塞 |
| **nvrx** | `megatron/core/dist_checkpointing/strategies/nvrx.py` | NVIDIA Resiliency 扩展集成 —— 本地/内存级 checkpoint,从瞬时故障快速重启 |

`async` 与 `fully-parallel` 是两个关键性能特性:前者把存档延迟**藏进计算**,后者把存档 I/O **摊到所有卡**。大模型 checkpoint 动辄 TB 级,这两者让"每隔 N 步存一次"不至于拖垮吞吐。

> Megatron-FSDP 另用 `fsdp_dtensor` 格式(基于 DTensor,见 `27_megatron_tp_fsdp_resharding_supplements_analysis.md` §1.5)。

---

## 7. 校验(`megatron/core/dist_checkpointing/validation.py`)

`validate_access_integrity` 检查 `sharded_state_dict` 的**完整性与一致性**:把所有 rank 的 `ShardedTensor` 元数据汇总,验证每个全局张量 ——
- **无缺口**:全局张量的每个元素都被某个非副本片覆盖。
- **无重叠**:不同 rank 的非副本片不重叠。
- 形状、`replica_id` 自洽(`allow_shape_mismatch` 的张量放宽)。

这一步挡住"并行配置算错 offset 导致存了个残缺/重叠的 checkpoint"。

---

## 8. 小结

- **要解决的问题**:模型被 TP×PP×EP×DP 切散,朴素存档与并行配置死绑,不能换布局/换集群续训。
- **核心抽象 `ShardedTensor`**:用 `(global_shape, global_offset, local_shape)` 描述"本 rank 这一片在全局张量的哪个位置",把并行布局信息编码进 checkpoint;`replica_id` 去 DP 冗余、`prepend_axis_num` 表达专家轴、`flattened_range` 对接扁平 buffer。
- **`sharded_state_dict()`**:每个 Megatron 模块自我声明分片映射,驱动 save/load。
- **save/load**:存成并行无关的全局张量;加载时按**当前(可不同的)布局**重新切片 —— 这就是"`TP8×PP4` 存、`TP2×PP1` 载"成立的原理。
- **策略**:`torch_dist` 默认;`fully-parallel`(I/O 摊到所有卡)+ `async`(后台写、不阻塞训练)是两个关键提速特性;`nvrx` 做快速故障重启。
- **校验**:`megatron/core/dist_checkpointing/validation.py` 保证全局张量被分片无缺口、无重叠地覆盖。
- 与 resharding 的区别:dist_checkpointing 是**磁盘**存档(续训),resharding 是 GPU↔GPU **实时**搬权重(RL 训推)。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。配套文档:`17_megatron_parallelism_orchestration_analysis.md`、`16_megatron_distributed_optimizer_analysis.md`、`27_megatron_tp_fsdp_resharding_supplements_analysis.md`。*

## Related Pages

- [[17_megatron_parallelism_orchestration_analysis]] · [[16_megatron_distributed_optimizer_analysis]] · [[27_megatron_tp_fsdp_resharding_supplements_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
