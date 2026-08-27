---
title: "并行基座：ParallelDims 的 rank 预算、双平面 Mesh 与运行时区域"
---

# 并行基座：ParallelDims 的 rank 预算、双平面 Mesh 与运行时区域

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：`ParallelDims` 的本质不是“保存六个并行度”，而是把一份 rank 预算投影成几张用途不同、彼此重叠的 DeviceMesh：数据与 loss 按 token 所有权分组，FSDP 按参数存储所有权分组，`spmd_types` 按前后向值的逻辑布局分组，MoE 再把同一 sparse region 重切成 `efsdp × ep`。只有把“存储平面”和“值类型平面”分开，TP、CP、EP、FSDP 的组合才不会互相泄漏实现细节。
>
> 本页回答三件事：一组并行度为什么不是六个正交乘数；为什么当前实现为同一批 rank 建多张 mesh；以及这些静态 mesh 如何成为前后向期间可切换的运行时区域。各并行算法本身由 [[11_torchtitan_fsdp_analysis]]、[[12_torchtitan_tp_analysis]]、[[13_torchtitan_cp_analysis]] 与 [[15_torchtitan_ep_analysis]] 分别负责。

---

## 1. Overview：问题不是建网格，而是同时表达三种所有权

大模型训练同时存在三类并不等价的“谁拥有什么”：某个 rank 读取哪些样本和 token，某个 rank 长期保存哪段参数，某个算子的输入/输出当前沿哪些逻辑轴切分。把三者硬塞进一张 mesh 的直觉实现很快会冲突：FSDP 必须区分 `dp_replicate` 与 `dp_shard`，而前后向类型规则只需要一个逻辑 `dp`；MoE 专家又要求把 dense region 中的 rank 重新分成 `efsdp × ep`，不能再额外乘一个 EP 度数。

TorchTitan 的选择是先创建一维 world mesh，再从同一 rank 集合派生用途明确的重叠视图。这个方向由 2025-12-17 的提交 `183a0d2e7` 明确为设计哲学：以 world mesh 为唯一根，通过 unflatten/flatten 得到一维和多维子 mesh，并用统一查询 API 代替各调用方自行拼 ProcessGroup。当前 `build_mesh()` 的文档列出了 dataloading、dense storage、dense forward/backward 和 sparse storage 四类视图（`torchtitan/distributed/parallel_dims.py:147`、`torchtitan/distributed/parallel_dims.py:170`）。

| 视图 | 核心问题 | 轴形状 | 主要消费者 |
|---|---|---|---|
| rank 预算 | 所有并行度能否恰好覆盖 world | `dpR × dpS × cp × tp × pp` | 配置校验 |
| dataloading | 哪些 rank 属于同一数据副本 | `pp × batch × cp × tp` | sampler / batch 组织 |
| loss | 哪些 rank 拥有不同有效 token | `batch × cp` | loss、finite gate |
| dense storage | 参数在哪些轴复制、在哪些轴分片 | `pp × dpR × dpS × cp × tp` | FSDP / HSDP |
| dense fwd/bwd | 当前值的逻辑 SPMD 布局 | `dp × cp × tp` | `spmd_types` |
| sparse storage | 专家参数怎样在同一 rank 区域重切 | `pp × dpR × efsdp × ep` | MoE FSDP / EP |
| sparse fwd/bwd | dispatch/expert 区域当前值的布局 | `dpR × efsdp × ep` | `spmd_types` / dispatcher |

```text
                         一维 world
                              |
            +-----------------+------------------+
            |                 |                  |
            v                 v                  v
     data / loss 视图     dense storage      sparse storage
   pp,batch,cp,tp       pp,dpR,dpS,cp,tp   pp,dpR,efsdp,ep
       | flatten               |
       v                       | 同一 rank 集合的另一种解释
   loss=batch,cp               v
                        dense fwd/bwd=dp,cp,tp

   dp = dpR × dpS
   efsdp × ep = dpS × cp × tp
```

### Quick Start：从哪里进入源码

最小入口不是手工实例化 mesh，而是把并行配置交给 Trainer：

```python
config.parallelism.data_parallel_shard_degree = -1
config.parallelism.tensor_parallel_degree = 2
config.parallelism.context_parallel_degree = 2
config.parallelism.expert_parallel_degree = 4
config.parallelism.spmd_backend = "spmd_types"
```

阅读调用链应从 `Trainer.init_distributed()` 开始：它取得 world size 后调用 `ParallelDims.from_config()`（`torchtitan/trainer.py:628`、`torchtitan/trainer.py:637`）；构造完成立即执行 `_validate()`（`torchtitan/distributed/parallel_dims.py:99`）；真正首次查询 mesh 时才惰性调用 `build_mesh()`（`torchtitan/distributed/parallel_dims.py:360`）。进入训练步前，`get_spmd_context()` 注册 dense/sparse 两张前后向 mesh，并把 dense mesh 压入线程局部栈（`torchtitan/distributed/utils.py:397`、`torchtitan/distributed/utils.py:414`）。

---

## 2. Rank 预算：为什么 EP 不是第六个正交乘数

### ① 背景/问题

如果把 DP、CP、TP、PP、EP 都当成互相正交的乘数，配置看似简单，却会为专家并行“凭空”要求额外 rank。实际 MoE 需要的是：dense 计算和 sparse expert 计算复用同一批设备，只是在进入专家区域时改变 rank 的分组方式。旧校验只检查 dense mesh 乘积，非法 EP 会把 `efsdp` 截断为零或非整数，直到 DeviceMesh unflatten 才在远离配置入口的位置失败；提交 `253a12a2a` 的 Motivation 明确记录了这个延迟失败问题。

### ② 为什么这么设计

**选中的路线**是先固定 dense world 预算，再要求 EP 整除 sparse region；**被否决的直觉方案**是把 EP 作为 world-size 等式的新乘数。决策准则是“dense 与 sparse 两个区域必须覆盖相同 rank”，配置文档把这个等积关系直接写成公共契约（`torchtitan/config/configs.py:288`）。这样提高 EP 只会改变 expert 与 expert-FSDP 的切分比例，不会改变作业所需 GPU 总数。

`dp_shard=-1` 则解决另一类配置问题：用户通常先决定 PP/TP/CP 拓扑，再希望剩余设备全部用于参数分片；要求用户重复手算容易造成配置与实际 world size 漂移。配置说明明确只有 `dp_shard` 可为负，并把它定义为“吃掉剩余 rank”（`torchtitan/config/configs.py:135`、`torchtitan/config/configs.py:143`）。

### ③ 实现思路与细节

构造后的第一步是把 `-1` 反推为：

```text
dpS = world_size / (dpR × cp × tp × pp)
```

随后要求：

```text
dpR × dpS × cp × tp × pp = world_size
```

对应实现分别位于 `torchtitan/distributed/parallel_dims.py:114` 和 `torchtitan/distributed/parallel_dims.py:118`。EP 不进入该等式，而是作用于：

```text
sparse_region = dpS × cp × tp
efsdp          = sparse_region / ep
```

`_validate()` 先在 `torchtitan/distributed/parallel_dims.py:123` 检查整除；`build_mesh()` 再于 `torchtitan/distributed/parallel_dims.py:216` 计算 `batch`、`fsdp` 与 `efsdp`。单元测试同时固定了三个边界：`dp_shard=-1` 的反推结果（`tests/unit_tests/cpu/test_parallel_dims.py:69`）、EP 不贡献 world size（`tests/unit_tests/cpu/test_parallel_dims.py:181`），以及 `ep=3` 不能切分 size-2 sparse region（`tests/unit_tests/cpu/test_parallel_dims.py:632`）。

### ④ 约束/边界

- 自动反推使用整数除法，但紧接着的 world-size 等式会拒绝有余数的结果，不会静默少用 rank（`torchtitan/distributed/parallel_dims.py:114`、`torchtitan/distributed/parallel_dims.py:118`）。
- `ep` 还必须整除模型的 expert 数；那是模型配置的另一层约束，不由 `ParallelDims` 代管。这里只证明 rank 区域可重切，不证明专家可均分。
- PP 与 `dp_replicate` 是 sparse region 的外层轴，不能借给 EP；配置注释明确只有 `dp_shard × cp × tp` 参与重切（`torchtitan/config/configs.py:288`、`torchtitan/config/configs.py:291`）。
- `seq_len` 还有独立约束：Sequence Parallel 按 TP 切，默认 CP 负载均衡按 `2 × CP` 切（`torchtitan/distributed/parallel_dims.py:601`、`torchtitan/distributed/parallel_dims.py:606`）。Trainer 对实际 PP microbatch token 数重新计算并校验该除数，因此 `seq_len_divisor` 不是唯一闸门（`torchtitan/trainer.py:295`、`torchtitan/trainer.py:299`）。

---

## 3. 多张 mesh：为什么数据、参数存储和值类型不能共用一张视图

### ① 背景/问题

一个 rank 可以同时处于“同一数据副本”“同一参数 shard group”和“同一逻辑 DP 轴”，但三种关系的等价类并不相同。尤其在 `spmd_types` 下，前后向值只需知道逻辑 `dp` 布局，而 FSDP 必须知道这个逻辑轴内部哪部分复制、哪部分切分。如果只保留 `[dp, cp, tp]`，`fully_shard()` 丢失 HSDP 的存储语义；如果把 `[dp_replicate, dp_shard, cp, tp]` 暴露给所有算子，类型规则又被迫理解 FSDP 内部状态布局。

### ② 为什么这么设计

**选中的路线**是让同一 world rank 集合拥有 storage view 与 forward/backward view；**明显替代方案**是一张“全能 mesh”。决策准则是关注点隔离：参数状态的长期所有权和激活值的瞬时布局必须能独立演进。当前源码注释明确说，FSDP mesh 传给 `fully_shard()`，而 SPMD 类型系统看不到 `dp_replicate`/`dp_shard`；类型系统只看折叠后的逻辑 `dp`（`torchtitan/distributed/parallel_dims.py:231`、`torchtitan/distributed/parallel_dims.py:238`）。这是源码明说的设计，不是知识库反推。

data/loss 也没有复用 storage mesh。它们的准则是 token 所有权：DP 与 CP rank 处理不同 token，TP rank 处理同一逻辑样本，PP 则只有最后 stage 先拥有 loss。Trainer 的 finite gate 因而先在 `loss` mesh 规约，再跨 PP 传播，明确排除 TP（`torchtitan/trainer.py:858`、`torchtitan/trainer.py:862`、`torchtitan/trainer.py:870`）。

### ③ 实现思路与细节

`build_mesh()` 从 `(world_size,)` 的唯一 world mesh 开始（`torchtitan/distributed/parallel_dims.py:220`），依次派生：

1. `dataloading = [pp, batch, cp, tp]`，其中 `batch = dpR × dpS`（`torchtitan/distributed/parallel_dims.py:223`）。
2. `loss = flatten(batch, cp)`，把真正处理不同 token 的轴合并（`torchtitan/distributed/parallel_dims.py:228`）。
3. `spmd_types` storage view 为 `[pp, dpR, dpS, cp, tp]`（`torchtitan/distributed/parallel_dims.py:243`）。
4. 它另建 `[pp, dp, cp, tp]`，再去掉 PP 得到 dense fwd/bwd view（`torchtitan/distributed/parallel_dims.py:248`、`torchtitan/distributed/parallel_dims.py:253`）。
5. `partial_dtensor` 不建逻辑 DP view，而把 `dpS × cp` 折成 `fsdp`（`torchtitan/distributed/parallel_dims.py:254`）。
6. sparse storage view 始终是 `[pp, dpR, efsdp, ep]`（`torchtitan/distributed/parallel_dims.py:262`）。

`MeshAxisName.DP` 与具体 DP storage 轴之间不是命名偶合：`unfold_dp_axis()` 显式把逻辑 `dp` 展开为 `dp_replicate + dp_shard`（`torchtitan/distributed/parallel_dims.py:53`）。因此下游可以用逻辑布局描述值，又能在需要操作参数存储时还原具体轴。

### ④ 约束/边界

- 这些 mesh 是重叠视图，不代表创建了额外 rank；但当前 DeviceMesh 会为不同 unflatten 视图重复创建部分 ProcessGroup。源码承认这有冗余，并选择等待上游 DeviceMesh 改进，而不是用大量 Fake 特判把代码复杂化（`torchtitan/distributed/parallel_dims.py:180`）。
- `partial_dtensor` 仍是有效兼容后端，GraphTrainer 与 HF backend 在 `5ab3a0fd1` 把 `spmd_types` 设为默认时仍被明确保留在旧路径；不能把“默认”写成“唯一”。当前配置只允许这两个值（`torchtitan/config/configs.py:174`）。
- `full_dtensor` 已由提交 `601cf4d23` 删除。把它继续描述成第三种当前后端会直接误导配置；该提交给出的理由是 `spmd_types` 已成默认、`partial_dtensor` 已是 fallback，删除专用分支以降低维护负担。

### ⑤ 发展趋势（有源码锚点的推断）

`build_mesh()` 留有 TODO：待 SPMD 不再和 DTensor/default 后端共享路径后清理 mesh 构造（`torchtitan/distributed/parallel_dims.py:241`）。据此可以推断双后端期间仍会保留适配性分支，但源码没有承诺 `partial_dtensor` 的删除日期；只能说方向是收敛，不能写成既定 roadmap。

---

## 4. singleton 与 Fake backend：为什么 degree=1 不等于“不存在”

### ① 背景/问题

常见优化是不给 size-1 轴创建通信组，因为它没有真实 collective；但 TorchTitan 仍需要某些 singleton mesh 承载策略语义。最典型的是只有 TP/DDP/PP 时，FSDP 仍可能被用来安装 mixed-precision policy；EP 开启但 `efsdp=1` 时，专家层也仍依赖 FSDP 包装做混合精度。若简单按 `degree > 1` 删除所有轴，功能会在“没有实际分片”的边界配置上消失。

### ② 为什么这么设计

**选中的路线**是默认用 Fake backend 消除无用组，但为 `fsdp`、`spmd_types.dp_shard` 和 EP 场景下的 `efsdp` 保留真实 backend；**替代方案**是统一保留或统一删除 singleton 轴。前者制造更多 ProcessGroup，后者丢失 mixed-precision/FSDP 语义。源码在三个例外分支中逐一写明判据（`torchtitan/distributed/parallel_dims.py:130`、`torchtitan/distributed/parallel_dims.py:135`、`torchtitan/distributed/parallel_dims.py:141`）。

### ③ 实现思路与细节

`unflatten_mesh()` 为 `_mesh_exist()` 判定为 false 的轴覆盖为 Fake backend（`torchtitan/distributed/parallel_dims.py:188`、`torchtitan/distributed/parallel_dims.py:198`）。查询时又分两种语义：

- `get_optional_mesh()` 默认把未启用/不存在轴返回 `None`，但可通过 `include_singleton_axes=True` 把 size-1 轴交给 `spmd_types` 自行过滤（`torchtitan/distributed/parallel_dims.py:331`、`torchtitan/distributed/parallel_dims.py:343`）。
- `get_all_one_dimensional_meshes()` 只报告既大于 1 又有真实 backend 的轴，防止调用者把 Fake group 当成可通信组（`torchtitan/distributed/parallel_dims.py:514`、`torchtitan/distributed/parallel_dims.py:549`）。

测试专门覆盖了“EP 关闭时 `efsdp` 数值 size 大于 1、实际却是 Fake”的反直觉情形，并要求它不出现在可通信轴列表中（`tests/unit_tests/cpu/test_parallel_dims.py:696`、`tests/unit_tests/cpu/test_parallel_dims.py:713`、`tests/unit_tests/cpu/test_parallel_dims.py:718`）。

### ④ 约束/边界

- “对象存在”不等于“并行已启用”，也不等于“拥有真实 ProcessGroup”。调用方应使用公开查询 API，不能只检查内部字典或 `mesh.size()`。
- 多轴查询只要任一请求轴未启用就返回 `None`；如果需求是“过滤掉未启用轴后返回剩余子 mesh”，必须用 `get_activated_mesh()`，两者语义不同（`torchtitan/distributed/parallel_dims.py:443`、`torchtitan/distributed/parallel_dims.py:446`）。
- `get_mesh()` 是强契约版本：optional 结果为 `None` 时抛错（`torchtitan/distributed/parallel_dims.py:399`、`torchtitan/distributed/parallel_dims.py:414`）。配置必需的通信轴宜用它，真正可选的优化路径才应使用 optional 查询。

---

## 5. Mesh 查询与运行时区域：为什么静态目录还不够

### ① 背景/问题

Mesh 构建只回答“有哪些 rank 分组”，没有回答“当前算子应该按哪张 mesh 解释布局”。Dense attention 与 MoE dispatch/expert 会在同一个 forward 中先后发生；如果布局类型始终绑定到一张全局 mesh，进入 sparse 区域后 `ep`/`efsdp` 的含义无处表达。另一问题是 DeviceMesh 的等价性依赖对象身份：反复切出形状相同的新对象，会破坏依赖 identity 的缓存和类型检查。

### ② 为什么这么设计

**选中的路线**是静态 mesh 目录 + 线程局部 current-mesh 栈：`ParallelDims` 负责稳定地产生、缓存和解析 mesh，`spmd_types` 上下文负责在运行时切 dense/sparse 区域；**替代方案**是让每个模块自行持有 ProcessGroup，或让一个全局 mesh 永不切换。前者重复拓扑逻辑并容易产生不一致对象，后者无法表达 MoE 的区域性布局。提交 `b052f36fe` 的说明明确把 sparse mesh transition、all-to-all 与类型 reinterpretation 作为同一组改动。

### ③ 实现思路与细节

多轴查询按轴名元组缓存，因为源码明确指出 DeviceMesh equality 依赖 identity（`torchtitan/distributed/parallel_dims.py:78`、`torchtitan/distributed/parallel_dims.py:382`）。布局边界则走两级解析：

1. `resolve_mesh()` 按后端决定哪些轴是 in-band：`spmd_types` 保留 `dp/cp/tp/ep`，`partial_dtensor` 只保留 `tp/ep`（`torchtitan/distributed/parallel_dims.py:461`、`torchtitan/distributed/parallel_dims.py:474`）。
2. `resolve_shared_mesh()` 要求同一 boundary 上所有非空布局具有相同轴集合；placement 可以不同，因为 redistribution 正是“同 mesh、不同 placement”（`torchtitan/distributed/parallel_dims.py:484`、`torchtitan/distributed/parallel_dims.py:504`）。

训练上下文通过 `set_spmd_meshes()` 把 dense/sparse mesh 注册到 TLS（`torchtitan/distributed/spmd_types.py:108`），再用栈式 context manager 激活当前 mesh并在退出时校验成对 pop（`torchtitan/distributed/spmd_types.py:159`、`torchtitan/distributed/spmd_types.py:176`）。进入专家区域时 `maybe_set_sparse_mesh()` 临时压入 sparse mesh，离开后自动恢复（`torchtitan/distributed/spmd_types.py:184`、`torchtitan/distributed/spmd_types.py:191`）。

### ④ 约束/边界

- TLS 栈要求进入/退出严格配对；`finally` 中的 identity 断言用于捕获嵌套上下文被错误弹出的情况（`torchtitan/distributed/spmd_types.py:177`、`torchtitan/distributed/spmd_types.py:180`）。
- 当前后端不是 `spmd_types` 时，`current_spmd_mesh()` 返回 `None`，`maybe_set_sparse_mesh()` 退化为 no-op（`torchtitan/distributed/spmd_types.py:138`、`torchtitan/distributed/spmd_types.py:187`）。因此不能把 TLS current mesh 当作所有后端共有机制。
- `resolve_shared_mesh()` 只接受相同轴集合；跨 mesh 的一次性布局转换必须在更高层拆成两个边界，不能伪装成同 mesh redistribution（`torchtitan/distributed/parallel_dims.py:505`、`torchtitan/distributed/parallel_dims.py:507`）。
- autograd 多线程会破坏 TLS DeviceMesh 栈的可见性，因此 TorchTitan 在 `spmd_types` 路线关闭 autograd multithreading；这是运行时约束，不是 mesh API 自身能解决的问题（`torchtitan/distributed/utils.py:460`）。

---

## 6. 版本演进与旧心智模型纠偏

| 旧心智模型 | 当前事实 | 证据与原因 |
|---|---|---|
| 所有并行度相乘等于 world size | EP 重切 `dpS × cp × tp`，不增加 rank | `torchtitan/config/configs.py:288`；提交 `253a12a2a` 把非法整除提前失败 |
| 一张 dense mesh 足够 | storage 与 fwd/bwd 是同 rank 的两张视图 | `torchtitan/distributed/parallel_dims.py:231`、`torchtitan/distributed/parallel_dims.py:238` |
| `full_dtensor` 是当前可选路径 | 已删除；只剩默认 `spmd_types` 与 fallback `partial_dtensor` | `torchtitan/config/configs.py:174`；提交 `601cf4d23` |
| degree=1 的轴都可删除 | 部分 singleton 轴仍承载 mixed precision/FSDP 语义 | `torchtitan/distributed/parallel_dims.py:130` |
| `get_optional_mesh()` 会自动过滤无效轴 | 任一轴无效即返回 `None`；过滤语义属于 `get_activated_mesh()` | `torchtitan/distributed/parallel_dims.py:373`、`torchtitan/distributed/parallel_dims.py:443` |
| mesh 只在初始化阶段有意义 | `spmd_types` 在 forward 内切换 dense/sparse current mesh | `torchtitan/distributed/spmd_types.py:159`、`torchtitan/distributed/spmd_types.py:184` |

> [!important] 证据边界
> “存储平面 / 值类型平面”是本页对源码两张 mesh view 的概念命名；两张 view 的职责由源码注释直接给出，但“平面”一词是知识库为了跨 FSDP、TP、CP、EP 统一推理所作的抽象。它不是上游公开 API 名称。

### 排障顺序

当组合并行配置失败时，按下面顺序定位比直接追 NCCL 错误更有效：

1. 先验算 dense world 等式，确认 `dp_shard=-1` 反推后没有余数。
2. 再验 `ep` 是否整除 `dp_shard × cp × tp`，并单独验 expert 数整除。
3. 检查实际 token 数是否满足 TP 与 CP 的序列除数，而不是只看名义 `seq_len`。
4. 区分所需的是 storage mesh、fwd/bwd mesh，还是 data/loss mesh。
5. 对 size-1 轴同时检查 `_mesh_exist` 语义与 backend，不能只看数值 size。
6. 在 MoE 边界确认 current mesh 已从 dense 切到 sparse，并在退出后恢复。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— Trainer、基础并行与编译器实验的全局入口。
- [[11_torchtitan_fsdp_analysis]] —— dense/sparse storage mesh 如何进入 FSDP/HSDP 参数与梯度生命周期。
- [[12_torchtitan_tp_analysis]] —— `tp` 轴如何驱动参数布局、Sequence/Loss Parallel 与异步 TP。
- [[13_torchtitan_cp_analysis]] —— `cp` 轴怎样切输入、改变序列整除条件并进入 FSDP storage region。
- [[15_torchtitan_ep_analysis]] —— `efsdp × ep` 重切、dense/sparse current-mesh transition 与 dispatcher。
- [[16_torchtitan_spmd_types_analysis]] —— 逻辑轴、边界 redistribution、类型检查与 TLS mesh 栈的完整机制。
- [[17_megatron_parallelism_orchestration_analysis]] —— Megatron-LM 用 RankGenerator 构造正交进程组的对照设计。
