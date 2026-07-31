# 并行基座:ParallelDims 与 DeviceMesh

> **代码基准**:torchtitan `main` @ `cf3c4312` · PyTorch `2.9.1`(FSDP2/DTensor/pipelining 内核)
> **最后更新**:2026-05-22 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本文是 torchtitan 多维并行分析的**基座篇**。所有并行维度(DP/TP/CP/EP/PP)都建立在 `ParallelDims` 与 `DeviceMesh` 之上;读懂这一层,后面 5 篇才有共同的坐标系。
>
> 主要源文件:`torchtitan/distributed/parallel_dims.py`(527 行)

---

## 1. 功能范围

`ParallelDims`(`parallel_dims.py:26`)是整个并行系统的**单一事实来源(single source of truth)**,职责有四:

1. **读取配置**:从 `ParallelismConfig` 读 6 个并行度。
2. **校验合法性**:6 个度数的乘积必须等于 `world_size`。
3. **构建 DeviceMesh**:把 1D 的"world mesh"重塑成多张带名字轴的逻辑 mesh。
4. **对外提供查询**:其余所有代码通过 `get_mesh("tp")`、`get_optional_mesh(["dp_replicate","fsdp"])` 之类按轴名取 mesh,而不直接碰进程组。

> **为什么需要这一层?** 多维并行最难的不是单个维度,而是**组合时的网格管理**——"哪 8 张卡组成一个 TP 组、哪 4 张组成 PP 组"极易算错。`ParallelDims` 把这件事收敛到一处,代码其余部分只面对"轴名"这个抽象。

---

## 2. 六个并行度与配置入口

```python
# parallel_dims.py:26
@dataclass
class ParallelDims:
    dp_replicate: int   # DDP / HSDP 的"复制"度
    dp_shard: int       # FSDP 的"分片"度(可填 -1 由框架反推)
    cp: int             # 上下文并行
    tp: int             # 张量并行
    pp: int             # 流水线并行
    ep: int             # 专家并行
    world_size: int
    full_dtensor: bool = False
```

`from_config`(`parallel_dims.py:43`)负责把 `ParallelismConfig` 的字段映射进来:

| ParallelDims 字段 | 配置字段 |
|-------------------|---------|
| `dp_replicate` | `data_parallel_replicate_degree` |
| `dp_shard` | `data_parallel_shard_degree` |
| `cp` | `context_parallel_degree` |
| `tp` | `tensor_parallel_degree` |
| `pp` | `pipeline_parallel_degree` |
| `ep` | `expert_parallel_degree` |

---

## 3. 维度约束与校验

`__post_init__` 调用 `_validate()`(`parallel_dims.py:61`):

```python
# parallel_dims.py:70-81
for d in (dp_replicate, cp, tp, pp, ep):
    assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."
if dp_shard < 0:
    self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)

assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, ...
```

两个要点:

### 3.1 `dp_shard = -1`:自动反推

`dp_shard` 可以填 `-1`,框架用 `world_size // (dp_replicate*cp*tp*pp)` 反推。这让用户只需指定"非数据并行"的几个度数,剩下的卡全部用于 FSDP 分片。

### 3.2 `ep` 不在乘积里 —— 关键设计

约束式是 `dp_replicate × dp_shard × cp × tp × pp == world_size`,**`ep` 缺席**。

这意味着 EP 不是"第 6 个独立维度",而是**对 `dp_shard × cp × tp` 这块 GPU 子网格的重新切分**。`ep` 必须整除 `dp_shard × cp × tp`。这一点是后面理解 `sparse_mesh` 的钥匙(见 [[torchtitan_ep_analysis]])。

### 3.3 序列长度约束

```python
# parallel_dims.py:520
@property
def seq_len_divisor(self):
    return self.tp * (self.cp * 2)
```

`seq_len` 必须被 `tp × 2cp` 整除:
- **`× tp`**:Sequence Parallel 要求序列维能被 TP 度整除。
- **`× 2cp`**:CP 的负载均衡(默认开启)把序列切成 `2×cp` 段做"头尾配对"(见 [[torchtitan_cp_analysis]])。

---

## 4. DeviceMesh 基础:unflatten 与 flatten

`DeviceMesh` 是 PyTorch 对"GPU 网格"的抽象。torchtitan 用到两个核心操作:

- **`_unflatten`**:把一根轴拆成多根。`world` 是一根 1D 轴(长度 = world_size),`_unflatten(0, (pp, batch, cp, tp), names)` 把它拆成 4 根带名字的轴。
- **`_flatten`**:把多根轴合并成一根。

> **Rank 排布**:`_unflatten` 给出的轴顺序决定了 rank 的物理映射。靠后的轴"变化最快"(相邻 rank),靠前的轴"变化最慢"。torchtitan 一律把 `tp` 放在最后 → **TP 组永远是相邻的 rank**,天然落在节点内 NVLink。`pp` 放在最前 → PP 组跨度最大,落在最慢的互联。

---

## 5. 三张逻辑 mesh 的构建

`build_mesh()`(`parallel_dims.py:100`)是基座的核心。它先建 1D world mesh,再 unflatten 出三张**相互重叠**的逻辑 mesh。

### 5.1 三组中间度数

```python
# parallel_dims.py:167-169
batch = self.dp_replicate * self.dp_shard       # 数据并行总度数
fsdp  = self.dp_shard * self.cp                 # FSDP 分片轴(CP 折叠进来)
efsdp = fsdp * self.tp // self.ep               # EP 区的 FSDP 轴
```

### 5.2 构建过程

```python
# parallel_dims.py:171-204
self._world_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("world",))

dataloading_mesh = unflatten_mesh(world_mesh,
    ("pp", "batch", "cp", "tp"),  (pp, batch, cp, tp))

loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")

full_dense_mesh  = unflatten_mesh(world_mesh,
    ("pp", "dp_replicate", "fsdp", "tp"),  (pp, dp_replicate, fsdp, tp))

full_sparse_mesh = unflatten_mesh(world_mesh,
    ("pp", "dp_replicate", "efsdp", "ep"), (pp, dp_replicate, efsdp, ep))
```

```
                 init_device_mesh → 1D "world" mesh (size = world_size)
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        │ unflatten               │ unflatten                │ unflatten
        ▼                         ▼                          ▼
  dataloading_mesh           dense_mesh                  sparse_mesh
  [pp, batch, cp, tp]        [pp, dp_replicate,           [pp, dp_replicate,
        │                     fsdp, tp]                    efsdp, ep]
        │ _flatten(batch,cp)
        ▼
   loss_mesh
```

### 5.3 三张 mesh 各管什么

| Mesh | 轴 | 用途 |
|------|-----|------|
| **dataloading** | `pp, batch, cp, tp` | 数据加载:决定每个 rank 读哪部分数据 |
| **loss** | `batch, cp` 展平 | loss 的 all-reduce(DP/CP 都"切了数据",梯度都要规约) |
| **dense** | `pp, dp_replicate, fsdp, tp` | **稠密层**参数:沿 `(dp_replicate, fsdp, tp)` 切 |
| **sparse** | `pp, dp_replicate, efsdp, ep` | **MoE 专家**参数:沿 `(dp_replicate, efsdp, ep)` 切 |

### 5.4 为什么 dense 和 sparse 要分两张?

验证一下两张 mesh 都覆盖全部 GPU:

```
dense:  pp × dp_replicate × fsdp × tp
      = pp × dp_replicate × (dp_shard·cp) × tp = world_size  ✓

sparse: pp × dp_replicate × efsdp × ep
      = pp × dp_replicate × (dp_shard·cp·tp/ep) × ep
      = pp × dp_replicate × dp_shard·cp·tp        = world_size  ✓
```

**同一组 GPU,两种切法**:
- `dense_mesh` 把 `tp` 单独拎成一个轴 → 稠密层的 attention/FFN 权重做 TP 切分。
- `sparse_mesh` 把 `ep` 单独拎成一个轴 → MoE 专家权重做 EP 切分。
- EP 开启时专家**不做 TP**,所以原本属于 TP 的那批 rank 被 sparse 侧的 `efsdp` 吸收。

这就是 torchtitan 处理"稠密层 + MoE 层"混合模型的方式:稠密参数活在 dense mesh 上,专家参数活在 sparse mesh 上,二者是同一批物理 GPU 的不同视图。

### 5.5 为什么 `fsdp = dp_shard × cp`?

CP 被**折叠进 FSDP 轴**。`build_mesh` 的 docstring(`parallel_dims.py:114`)解释:

> 只要用了 CP,就一定套 FSDP——即使全局 batch size 为 1、没有数据并行,也要借 FSDP 的 weight all-gather / gradient reduce-scatter。

所以 CP 的 rank 和 dp_shard 的 rank 一起构成 FSDP 的分片轴。

### 5.6 为什么 `loss_mesh` 是 `batch + cp`?

`dp_replicate`、`dp_shard`、`cp` 三者都在"切数据"——不同 rank 处理不同 token,梯度都需要规约。所以 loss 的规约要在这三者合并的轴上做(`batch` 已含 dp_replicate×dp_shard,再 flatten 上 cp)。

---

## 6. `fake` backend 优化

`_unflatten` 会为**每一根轴**创建一个进程组(NCCL communicator)。但 degree=1 的轴根本不需要真实通信。`unflatten_mesh`(`parallel_dims.py:139`)对这些轴用 `"fake"` 后端:

```python
# parallel_dims.py:149-158
backend_override = {}
for name, degree in zip(dim_names, dim_degrees):
    if not self._mesh_exist(name, degree):
        backend_override[name] = "fake"
return world_mesh._unflatten(0, dim_degrees, dim_names, backend_override=backend_override)
```

`_mesh_exist`(`parallel_dims.py:83`)的规则:

```python
if name == "fsdp":          return True            # 总保留:fully_shard 要挂混合精度策略
if name == "dp_shard" and self.full_dtensor:  return True
if name == "efsdp":         return True if self.ep > 1 else False
return degree > 1           # 其余:只有 degree>1 才建真进程组
```

即:**`fsdp` 轴即使 size=1 也保留真实后端**,因为 `fully_shard()` 需要它来安装 `MixedPrecisionPolicy`;`efsdp` 在 EP>1 时同理。

---

## 7. mesh 查询接口

| 方法 | 行为 |
|------|------|
| `get_mesh(dims)` | 按轴名取 mesh;取不到(degree=1 或未启用)直接 `raise` |
| `get_optional_mesh(dims)` | 同上,但取不到返回 `None` |
| `get_activated_mesh(axes)` | 取 `axes` 中**实际启用**的那些轴组成的子 mesh |
| `resolve_mesh(axes)` | 给一组轴名,解析出对应的 SPMD mesh(dense 或 sparse) |
| `spmd_meshes()` | 返回所有全 SPMD mesh(今天是 dense + sparse 两张) |
| `get_all_one_dimensional_meshes()` | 所有启用的 1D mesh,用于取进程组 |

多轴查询(如 `get_mesh(["dp_replicate","fsdp"])`)会从 `_global_meshes` 里找一张**同时包含这些轴**的全局 mesh 切子 mesh,并缓存——因为 `DeviceMesh` 的相等性是按对象身份(identity)判断的,必须缓存复用(`parallel_dims.py:310` 注释)。

`resolve_mesh`(`parallel_dims.py:380`)在非 `full_dtensor` 路径下只保留 `tp`/`ep` 轴为"带内(in-band)":

```python
# parallel_dims.py:396-398
if not self.full_dtensor:
    in_band = ("tp", "ep")
    axes_list = [axis for axis in axes_list if axis in in_band]
```

含义:**传统路径下,TP/EP 由 DTensor 显式处理(带内),DP/CP 由 FSDP/CP 包装隐式处理(带外)**。这条规则在 [[torchtitan_tp_analysis]] 会再次出现。

---

## 8. `full_dtensor` 模式

`full_dtensor`(`parallel_dims.py:35`)是一条较新的代码路径。差异在于 **`dp_shard` 与 `cp` 是否折叠**:

| | 传统路径 | `full_dtensor` 路径 |
|--|---------|---------------------|
| dense mesh 轴 | `[pp, dp_replicate, fsdp, tp]`(fsdp=dp_shard·cp) | `[pp, dp_replicate, dp_shard, cp, tp]`(不折叠) |
| 原因 | CP 折叠进 FSDP | 全 DTensor 下激活带 `cp` 维,参数也需要独立 `cp` 轴 |
| `fully_shard` | mesh 已是 1D `fsdp` | 在初始化时内部折叠 `dp_shard`+`cp` |

`full_dtensor` 路径下 `_shard_states` 等会把 DP/CP 也作为"带内"轴处理。本系列文档主体分析传统路径(目前 deepseek_v3 等仍 `raise NotImplementedError` 不支持 full_dtensor,见 `deepseek_v3/parallelize.py:43`)。

---

## 9. enabled 属性与便捷查询

`ParallelDims` 暴露一组布尔属性,供 `parallelize_*` 函数判断要不要施加某个并行:

```python
# parallel_dims.py:480-517
dp_enabled            = dp_replicate > 1 or dp_shard > 1
fsdp_enabled          = dp_shard_enabled or cp_enabled    # 注意:CP 也算 FSDP
tp_enabled            = tp > 1
pp_enabled            = pp > 1
ep_enabled            = ep > 1
cp_enabled            = cp > 1
non_data_parallel_size = cp * tp * pp
```

---

## 10. 完整流程图

```
ParallelDims.from_config(parallelism_config, world_size)
        │
        ▼  __post_init__ → _validate()
   校验 dp_replicate·dp_shard·cp·tp·pp == world_size
   (dp_shard=-1 时反推)
        │
        ▼  build_mesh()  [首次 get_mesh / world_mesh 时惰性触发]
   ┌────────────────────────────────────────────────┐
   │ 1. init_device_mesh → 1D "world" mesh           │
   │ 2. unflatten → dataloading_mesh [pp,batch,cp,tp] │
   │ 3. flatten(batch,cp) → loss_mesh                 │
   │ 4. unflatten → dense_mesh  [pp,dpR,fsdp,tp]      │
   │ 5. unflatten → sparse_mesh [pp,dpR,efsdp,ep]     │
   │ 6. 抽取所有 1D 子 mesh 存入 _single_axis_meshes  │
   │ 7. _validate_meshes() 校验每根轴 size 正确       │
   │    degree=1 的轴 → fake backend(省 NCCL 组)     │
   └────────────────────────────────────────────────┘
        │
        ▼  其余代码通过 get_mesh("tp") / get_optional_mesh([...]) 取用
   apply_fsdp / model.parallelize() / apply_moe_ep_tp / pipeline_llm ...
```

---

## 11. 小结

- `ParallelDims` 是并行系统的**单一事实来源**:校验度数、构建 mesh、提供按轴名查询。
- 核心约束 `dp_replicate·dp_shard·cp·tp·pp == world_size`,**`ep` 例外**——EP 是对子网格的重新切分。
- `build_mesh` 从 1D `world` mesh `_unflatten` 出**三张重叠的逻辑 mesh**:dataloading / dense / sparse,外加 flatten 出的 loss mesh。三张都覆盖全部 GPU。
- `dense_mesh` 隔离 `tp` 轴(稠密参数),`sparse_mesh` 隔离 `ep` 轴(专家参数),`loss_mesh` 合并所有"切数据"的轴。
- `fsdp = dp_shard × cp`:CP 折叠进 FSDP;`fake` backend 省掉 degree-1 轴的无用进程组。
- 轴顺序 `[pp, ..., tp]` 决定 rank 物理排布:TP 组相邻(NVLink),PP 组跨度最大(慢互联)。

## Related Pages

- [[torchtitan/index]] —— torchtitan 多维并行知识地图(本系列入口)
- [[torchtitan_fsdp_analysis]] · [[torchtitan_tp_analysis]] · [[torchtitan_cp_analysis]] · [[torchtitan_pp_analysis]] · [[torchtitan_ep_analysis]] —— 五个并行维度的机制级深度分析
- [[17_megatron_parallelism_orchestration_analysis]] —— Megatron-LM 进程组编排 capstone(RankGenerator、正交分组),与 `ParallelDims` 同类
- [[15_megatron_pp_schedulers_analysis]] —— LLM 正反向计算依赖 DAG 与 TP/SP/EP/CP 通信依赖
