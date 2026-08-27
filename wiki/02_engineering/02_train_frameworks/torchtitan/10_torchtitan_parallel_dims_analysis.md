---
title: "并行基座：ParallelDims 与双平面 DeviceMesh"
---

# 并行基座：ParallelDims 与双平面 DeviceMesh

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：`ParallelDims` 不再只是把 6 个并行度展开成三张 mesh。随着 `spmd_types` 成为默认后端，它同时维护两套重叠视图：一套给 FSDP/参数存储决定“状态放在哪”，另一套给前后向类型系统决定“值在每个逻辑轴上是什么布局”。这一区分是理解新版 TP、CP、EP 与 FSDP 组合关系的共同坐标系。
>
> 主要源文件：`torchtitan/distributed/parallel_dims.py`、`torchtitan/distributed/spmd_types.py`、`torchtitan/config/configs.py`。

---

## 1. 先纠正旧版知识

知识库旧基线把 `full_dtensor: bool` 当成新增路径，并把传统的 `[dp_replicate, fsdp, tp]` dense mesh 当成唯一主线。当前代码已经发生两步演进：

1. `spmd_types` 在提交 `5ab3a0fd1` 中成为默认后端；当前配置只保留 `partial_dtensor` 与 `spmd_types` 两个选择，默认是后者（`torchtitan/config/configs.py:168-180`）。
2. 旧 `full_dtensor` 后端在提交 `601cf4d23` 中被删除；当前 `ParallelDims` 字段是 `spmd_backend`，不再有 `full_dtensor`（`torchtitan/distributed/parallel_dims.py:68-82`）。

> [!deprecated] 旧版 `full_dtensor` 章节已失效
> `full_dtensor` 不是当前可选后端。其“让 DP/CP 进入显式布局”的方向由默认的 `spmd_types` 接续，但实现不再是让整网状态都以 DTensor 传播，而是用本地 tensor + `SpmdType` 描述前后向布局，并在边界显式发起 collective（`torchtitan/distributed/spmd_types.py:398-450`）。

`partial_dtensor` 尚未删除，但源码已经把它标成迁移期兼容路径：当前 redistribution 辅助函数的 TODO 明确以“未来移除 partial_dtensor”为收敛方向（`torchtitan/distributed/spmd_types.py:256-269`）。因此，本页以 `spmd_types` 默认路径为主，只把 `partial_dtensor` 作为对照。

---

## 2. `ParallelDims` 的职责：把数字约束变成可复用的命名网格

`ParallelDims` 的字段、默认后端与配置映射集中在 `torchtitan/distributed/parallel_dims.py:68-97`。它承担四件事：

1. 从 `ParallelismConfig` 读取 DP-replicate、DP-shard、CP、TP、PP、EP 六个度数与 SPMD 后端。
2. 校验 world-size 乘积与 EP 的整除关系。
3. 从一维 world mesh 构造数据、loss、dense-storage、sparse-storage 与 SPMD 前后向网格。
4. 通过轴名解析、过滤和缓存子 mesh，让下游不自行计算 rank group。

当前轴名由 `MeshAxisName` 统一定义；`dp` 是面向类型系统的逻辑轴，`dp_replicate`/`dp_shard` 是面向存储与 FSDP 的具体轴（`torchtitan/distributed/parallel_dims.py:29-65`）。这不是同义词替换，而是两种观察同一批 rank 的方式。

| `ParallelDims` 字段 | 配置字段 | 角色 |
|---|---|---|
| `dp_replicate` | `data_parallel_replicate_degree` | DDP/HSDP 的复制度 |
| `dp_shard` | `data_parallel_shard_degree` | FSDP 的参数状态分片度；`-1` 表示吃掉剩余 rank |
| `cp` | `context_parallel_degree` | 序列/上下文轴 |
| `tp` | `tensor_parallel_degree` | 算子内张量并行轴 |
| `pp` | `pipeline_parallel_degree` | 模型深度/流水轴 |
| `ep` | `expert_parallel_degree` | 稀疏专家轴 |
| `spmd_backend` | `spmd_backend` | `spmd_types`（默认）或 `partial_dtensor` |

配置本身还明确了 HSDP/FSDP 的语义以及 `dp_shard=-1` 的行为（`torchtitan/config/configs.py:123-160`）。Trainer 在分布式初始化后设置全局后端，并最终调用 `ParallelDims.from_config`，所以 mesh 不是模型文件各自临时拼出来的（`torchtitan/trainer.py:307-309`、`torchtitan/trainer.py:637`）。

---

## 3. 两条硬约束：world-size 乘积与 EP 重切分

### 3.1 world-size 约束

`_validate()` 先允许 `dp_shard=-1` 自动反推，再要求：

```text
dp_replicate * dp_shard * cp * tp * pp == world_size
```

实现位于 `torchtitan/distributed/parallel_dims.py:102-121`。`dp_shard=-1` 的价值是：用户先指定 PP/TP/CP 与复制度，剩余 rank 自动全部用于参数分片。

### 3.2 EP 不是第六个正交乘数

EP 不进入上式；它重新切分 `dp_shard * cp * tp` 覆盖的 sparse region，因此必须满足：

```text
(dp_shard * cp * tp) % ep == 0
efsdp = dp_shard * cp * tp / ep
```

代码在构造期直接拒绝不能整除的配置（`torchtitan/distributed/parallel_dims.py:123-128`），配置说明也把 dense region 与 sparse region 的等积关系写成公共契约（`torchtitan/config/configs.py:284-292`）。这解释了为什么 EP 开启后，专家权重使用 `efsdp + ep`，而不是简单在 dense mesh 后面再加一根 EP 轴。

### 3.3 序列长度约束

`seq_len_divisor` 仍返回 `tp * (cp * 2)`：TP 的 sequence parallel 要求按 TP 整除；默认 CP 负载均衡把序列组织成 `2 * cp` 个分段（`torchtitan/distributed/parallel_dims.py:601-609`）。CP 的两种负载均衡实现见 [[13_torchtitan_cp_analysis]]。

---

## 4. 核心演进：存储平面与前后向类型平面分离

### 4.1 共同入口

`build_mesh()` 先计算：

```text
batch = dp_replicate * dp_shard
fsdp  = dp_shard * cp
efsdp = fsdp * tp / ep
```

然后创建一维 `world`、数据加载 mesh 与 loss mesh（`torchtitan/distributed/parallel_dims.py:216-228`）。后两者与 SPMD 后端无关：

```text
dataloading = [pp, batch, cp, tp]
loss        = flatten(batch, cp)
```

`loss` 合并 DP 与 CP，是因为这些轴上的 rank 处理不同 token，需要共同规约 loss/有效 token 统计；TP rank 看到的是同一逻辑样本，PP 则由最后 stage 先拥有 loss，再单独跨 PP 传播（`torchtitan/trainer.py:858-875`）。

### 4.2 `spmd_types` 默认路径：同一批设备，两张 dense 视图

默认路径会建立两张 dense mesh（`torchtitan/distributed/parallel_dims.py:229-253`）：

```text
FSDP/存储视图       [pp, dp_replicate, dp_shard, cp, tp]
前后向类型视图      [pp, dp,            cp, tp]
                                  dp = dp_replicate * dp_shard
```

- **存储视图**保留 `dp_replicate` 与 `dp_shard`，传给 `fully_shard()`，让 FSDP 明确哪根轴复制、哪根轴切参数。
- **前后向类型视图**把二者折成逻辑 `dp`，供 `spmd_types` 对激活和值做类型检查。类型系统不需要知道参数状态具体如何在 HSDP/FSDP 内部保存。

这就是“双平面”的关键：**同一个 tensor 的运行时值布局与参数状态存储布局可以相关，但不必由同一张 mesh 表达。** 如果强迫它们共用一张网格，类型规则就会泄漏 FSDP 存储细节；反之只保留逻辑 `dp` 又无法告诉 `fully_shard` 哪部分是 replicate、哪部分是 shard。

### 4.3 `partial_dtensor` 兼容路径

兼容路径仍把 `dp_shard * cp` 折叠成一根 `fsdp`：

```text
[pp, dp_replicate, fsdp, tp]
```

构造分支在 `torchtitan/distributed/parallel_dims.py:254-260`。此后 `resolve_mesh()` 只把 TP/EP 作为带内布局轴；默认 `spmd_types` 则把 DP/CP/TP/EP 都纳入可解析轴（`torchtitan/distributed/parallel_dims.py:461-482`）。

### 4.4 sparse 视图

两种后端都构造：

```text
sparse storage = [pp, dp_replicate, efsdp, ep]
```

代码在 `torchtitan/distributed/parallel_dims.py:262-279`。当默认后端且 `ep > 1` 时，还会注册 `[dp_replicate, efsdp, ep]` 为 sparse 前后向区域；进入 token dispatch / expert 区域时，线程局部的 current mesh 会从 dense 切到 sparse（`torchtitan/distributed/spmd_types.py:108-127`、`torchtitan/distributed/spmd_types.py:159-192`）。

---

## 5. 全部网格关系

```text
                         1D world
                            |
          +-----------------+--------------------+
          |                 |                    |
          v                 v                    v
  dataloading          dense storage        sparse storage
 [pp,batch,cp,tp]   [pp,dpR,dpS,cp,tp]    [pp,dpR,efsdp,ep]
          |                 |
          | flatten         | 另建逻辑视图
          v                 v
 loss=[batch,cp]      dense fwd/bwd
                       [dp,cp,tp]
                  dp = dpR * dpS

 partial_dtensor 对照：dense storage = [pp,dpR,fsdp,tp]
                       fsdp = dpS * cp；没有 dense fwd/bwd SPMD mesh
```

`_global_meshes` 与 `_single_axis_meshes` 的实际注册表位于 `torchtitan/distributed/parallel_dims.py:268-295`；随后 `_validate_meshes()` 逐轴核对真实 size，避免“名字对了但 rank 数错了”的静默错误（`torchtitan/distributed/parallel_dims.py:306-329`）。

---

## 6. 为什么 singleton 轴有时仍需真实进程组

`unflatten_mesh()` 默认把 degree=1 的无效轴换成 fake backend，以免创建无用通信组（`torchtitan/distributed/parallel_dims.py:188-208`）。但有三个例外：

- `partial_dtensor` 的 `fsdp` 始终保留真实后端，使 `fully_shard()` 即使在 size=1 时也能安装混合精度策略。
- `spmd_types` 的 `dp_shard` 同样始终保留，使 TP/DDP/PP-only 配置仍能通过 FSDP 安装混合精度并辨认 DP storage submesh。
- `efsdp` 在 EP 开启时保留，即使其 size 为 1，也需要 FSDP 包装帮助专家层做混合精度。

这些分支集中在 `_mesh_exist()`（`torchtitan/distributed/parallel_dims.py:130-145`）。所以“degree=1 就一定不存在进程组”已经不是可靠推断；应通过 `get_optional_mesh()` 或 `get_all_one_dimensional_meshes()` 查询。

---

## 7. 查询接口与不变量

| 接口 | 语义 | 源码 |
|---|---|---|
| `get_optional_mesh(dims)` | 任一请求轴未启用时返回 `None`；可选择保留 singleton 轴 | `torchtitan/distributed/parallel_dims.py:331-397` |
| `get_mesh(dims)` | 同上，但不可用时抛错 | `torchtitan/distributed/parallel_dims.py:399-423` |
| `spmd_dense_mesh()` / `spmd_sparse_mesh()` | 取前后向类型检查的 dense/sparse 区域 | `torchtitan/distributed/parallel_dims.py:425-435` |
| `get_activated_mesh(axes)` | 过滤未启用轴，再返回剩余轴的子 mesh | `torchtitan/distributed/parallel_dims.py:443-459` |
| `resolve_mesh(axes)` | 按后端过滤带内轴，再解析 dense/sparse mesh | `torchtitan/distributed/parallel_dims.py:461-482` |
| `resolve_shared_mesh(layouts)` | 要求一组布局使用相同轴集合，再找共享 mesh | `torchtitan/distributed/parallel_dims.py:484-512` |

多轴 mesh 会按轴名元组缓存，因为 `DeviceMesh` 的相等性依赖对象身份；反复切同一子网格而不复用对象会破坏下游对 mesh identity 的假设（`torchtitan/distributed/parallel_dims.py:378-397`）。

运行时，`get_spmd_context()` 会确保 mesh 已构建、把 dense/sparse mesh 注册到线程局部状态，再激活当前 dense mesh；这把“静态的 mesh 目录”连接到“每次前后向当前在哪个 SPMD 区域”（`torchtitan/distributed/utils.py:397-417`）。

---

## 8. 从配置到运行时的真实调用链

```text
Trainer.__init__
  -> set_spmd_backend(config.parallelism.spmd_backend)
     torchtitan/trainer.py:307-309
  -> ParallelDims.from_config(config.parallelism, world_size)
     torchtitan/trainer.py:637
  -> ParallelDims.__post_init__ -> _validate
     torchtitan/distributed/parallel_dims.py:84-128
  -> 首次取 mesh / 创建 SpmdContext 时 build_mesh
     torchtitan/distributed/utils.py:397-417
  -> 注册 dense/sparse SPMD mesh，并在 train context 中激活 dense mesh
     torchtitan/distributed/spmd_types.py:108-192
  -> 进入 MoE expert 区域时临时切 sparse mesh，退出后恢复
```

这条链说明 `ParallelDims` 的最终产物不只是 ProcessGroup 集合：它还决定后续类型标注、redistribution 与 sparse-region 切换在哪张 mesh 上解释。

---

## 9. 小结

- world-size 主约束仍是 `dp_replicate * dp_shard * cp * tp * pp`；EP 对 `dp_shard * cp * tp` 做等积重切分。
- 当前默认后端是 `spmd_types`；旧 `full_dtensor` 已删除，`partial_dtensor` 是迁移期兼容路径。
- 新版核心不是“三张 mesh”，而是**数据/loss 视图 + 参数存储视图 + 前后向类型视图**的重叠体系。
- 默认后端把 `dp_replicate`/`dp_shard` 留给 FSDP 存储平面，把二者折成逻辑 `dp` 交给前后向类型平面。
- dense/sparse current mesh 通过线程局部上下文切换，使同一模型可在稠密区与专家区使用不同布局语义。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口与最新基线总览。
- [[11_torchtitan_fsdp_analysis]] —— 存储平面的 `dp_replicate`/`dp_shard` 最终如何进入 FSDP2。
- [[12_torchtitan_tp_analysis]] —— TP 如何消费 dense 前后向 mesh 与布局重分布。
- [[13_torchtitan_cp_analysis]] —— CP 的 `cp` 轴、负载均衡与输入布局接线。
- [[15_torchtitan_ep_analysis]] —— dense/sparse mesh 切换与专家参数的 `efsdp + ep` 布局。
- [[16_torchtitan_spmd_types_analysis]] —— 双平面 mesh 之上的类型契约、边界 collective 与失败边界。
- [[17_megatron_parallelism_orchestration_analysis]] —— Megatron-LM 的 RankGenerator/正交进程组编排对照。
