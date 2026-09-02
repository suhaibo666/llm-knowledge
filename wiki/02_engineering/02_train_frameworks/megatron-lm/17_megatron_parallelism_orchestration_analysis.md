---
title: "Megatron-LM 并行编排与进程组构造深度解析(Capstone)"
---

# Megatron-LM 并行编排与进程组构造深度解析(Capstone)

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，未改动本页主线的 `parallel_state.py` / `process_groups_config.py` / `hyper_comm_grid.py`，本页落在改动文件（`training.py`，因 DSA FLOPs PR #6753 前插 184 行）上的引用已在新基线下逐条打开重核，均为纯行号漂移。
> **重定基线**：2026-08-28 由 `ee3f1ffa…`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/parallel_state.py`(2266 行)、`megatron/core/process_groups_config.py`(718 行)、`megatron/core/hyper_comm_grid.py`(443 行)
> 配套阅读:`15_megatron_pp_schedulers_analysis.md`、`14_megatron_ep_analysis.md`、`12_megatron_tp_analysis.md`、`13_megatron_cp_analysis.md`、`16_megatron_distributed_optimizer_analysis.md`
> 定位:**这是收口文档**。前五份各讲一个并行轴,都默认了"每张 GPU 同时属于 TP/PP/CP/EP/DP 的某个组"。本文讲清楚这套**几何**是怎么从 `world_size` 个裸 GPU 构造出来的。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**：2026-08-28。按五拍重排章节顺序；机制正文与既有引用未改。

---

## 1. 背景：一维 `global_rank` 与 5 个并行轴之间,缺一张几何

### 1.1 这份文档解决什么

前五份文档反复出现一个隐含前提:某张卡"在 TP 组里"、"在 PP 组里"、"在 EP 组里"……但**一张物理 GPU 怎么会同时属于 5 个组?这些组的成员是谁?谁来构造?** —— 这就是"并行编排"。

一张 GPU 的全局编号 `global_rank ∈ [0, world_size)`。编排层的工作:把这个一维编号**解释成一个 5 维坐标** `(tp_rank, cp_rank, ep_rank, dp_rank, pp_rank)`,再据此为每个轴建立 `torch.distributed.ProcessGroup`。前五份文档里所有 `tp_group` / `pp_group` / `ep_group` / `dp_cp_group` / `pg_collection` 全部源于此。

### 1.2 三层抽象(历史演进)

Megatron 的进程组管理有三套实现,层层递进:

| # | 抽象 | 文件 | 形态 | 定位 |
|---|------|------|------|------|
| ① | `parallel_state` | `megatron/core/parallel_state.py` | **全局单例** + `get_*_group()` | 经典实现,全局状态 |
| ② | `ProcessGroupCollection` | `megatron/core/process_groups_config.py` | **显式传递**的 dataclass | 把组作为参数注入(`pg_collection=`) |
| ③ | `HyperCommGrid` | `megatron/core/hyper_comm_grid.py` | **N 维网格**对象 | 最新、最干净的统一抽象 |

三者并存,语义等价(`HyperCommGrid` 的 docstring 明确给出与 `initialize_model_parallel` 的等价示例)。

### 1.3 记号约定

| 符号 | 含义 |
|------|------|
| `world_size` | 总 GPU 数 |
| `tp/cp/ep/dp/pp` | 五个并行轴的并行度 |
| `order` | 轴布局字符串,默认 `"tp-cp-ep-dp-pp"` |
| `global_rank` | GPU 的全局一维编号 |
| stride | 某轴在 `global_rank` 分解式里的步长(prefix product) |

### 1.4 一张卡的 5 维坐标

`world_size` 个 GPU 要同时承载 5 个正交的并行轴。把它们看成一个 5 维超长方体(hyperrectangle):

```
world_size = tp · cp · ep · dp · pp
每张 GPU ←→ 5 维坐标 (tp_rank, cp_rank, ep_rank, dp_rank, pp_rank)
```

"正交"的含义:沿任一轴移动,其余 4 维坐标不变。于是:
- **TP 组** = 固定 `(cp,ep,dp,pp)`、遍历 `tp_rank` 的那组卡。
- **DP 组** = 固定 `(tp,cp,ep,pp)`、遍历 `dp_rank` 的那组卡。
- ……每个轴一组,共 5 类组(及各种组合组,如 `dp_cp`、`tp_ep`)。

### 1.5 `order` 字符串:决定物理布局

5 维坐标怎么映射回一维 `global_rank`?由 `order` 字符串决定。默认 `order = "tp-cp-ep-dp-pp"`,含义是:

```
global_rank = tp_rank·s_tp + cp_rank·s_cp + ep_rank·s_ep + dp_rank·s_dp + pp_rank·s_pp

其中步长 s 是 order 各轴大小的"前缀积":
  s_tp = 1                          ← order 第一个轴,步长最小
  s_cp = tp
  s_ep = tp·cp
  s_dp = tp·cp·ep
  s_pp = tp·cp·ep·dp                ← order 最后一个轴,步长最大
```

**这就是一切"TP 留机内、PP 跨机"建议的量化根源**:
- `order` 的**第一个轴(TP)步长 = 1** → 同一 TP 组的卡 `global_rank` **相邻** → 落在**同一节点**(NVLink 域)。
- `order` 的**最后一个轴(PP)步长最大** → 同一 PP 组的卡 `global_rank` 隔得最远 → **跨节点**(IB)。

```
order = "tp-cp-ep-dp-pp"
物理邻近度:  TP(最近,NVLink) → CP → EP → DP → PP(最远,IB)
              └─ 通信最密集的放最近 ──┘   └─ 通信最稀疏/可重叠的放最远 ─┘
```

通信越频繁、越在关键路径上的轴(TP)放 `order` 越靠前;通信越稀疏、越能重叠的轴(PP)放越靠后。这是 `order` 的设计原则。

---

## 2. 为什么这么设计：一条 `order` 字符串 + 一个通用的正交分组函数,替掉"每类组各写一遍"的手写循环

朴素做法是**每种组各写一段循环**:要 DP 组就按 `start_rank / end_rank / step` 手算一遍,要 CP 组再算一遍,要 `tp-dp` 组再算一遍。Megatron 早期正是这么写的,后来整段删掉,换成"一条 `order` 定布局 + 一个通用函数解方程"。源码陈述了其中三条理由;第四条源码沉默,由本页重建并标为推断。

**① 分组被归约成一道可解的方程,而不是每类组一段专门的循环。**
`generate_masked_orthogonal_rank_groups`(`megatron/core/parallel_state.py:250`)的 docstring 把整套编排直接写成数学题:「For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and local_rank satisfy the following equation: `global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size` (1)」,取 `dp_group` 时「The tp_rank and pp_rank will be combined to form the `dp_group_index`」,该段末尾一句是「**This function solve this math problem.**」(`:270-288`)。于是任意轴组合只要给一个 `mask` 就能算出来 —— `get_ranks(token)` 把 `"tp"` / `"dp-cp"` / `"tp-ep"` 这类 token 翻成 mask 再调它(`:503`、`:513-514`),不必为每个 token 各写一段。

**被否掉的替代①:每类组各一段手写循环、把布局焊死在算术里。**
提交 `6513cde7b`(2024-04-12,commit message 即「Support alternative mapping TP->PP->DP」)引入了 `RankGenerator` / `order` / `generate_masked_orthogonal_rank_groups`;在它之前,`initialize_model_parallel` 里 DP 组、`dp_cp` 组、CP 组、TP 组、PP 组、`tp-dp` 组**各有一段独立的嵌套循环**,步长逐段硬编码 —— 例如 DP 组那段是 `start_rank = i * num_pipeline_model_parallel_groups` / `end_rank = (i + 1) * num_pipeline_model_parallel_groups` / `ranks = range(start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size)`(见 `6513cde7b` 删除侧的 `megatron/core/parallel_state.py`,同一 diff 里 CP 组、TP 组、PP 组、`tp-dp` 组各是同构的一段)。这种写法把 `tp-cp-dp-pp` 这一种布局**写死在每一段的算术里**,想换成 `tp-pp-dp` 就得逐段重写 —— commit 标题「Support alternative mapping TP->PP->DP」说的正是这次改写要解锁的能力。
→ 决定取舍的判据是**把"布局"变成一个可替换的参数**:`order` 只在 `RankGenerator.__init__` 里被拆成 `ordered_size`(`megatron/core/parallel_state.py:483-486`),之后所有组共用同一个求解器,新增一种组合只是新增一个 token。

**② `order` 为什么把 TP 放第一位:源码给的理由是"相邻 rank 要在同一台机器上"。**
`initialize_model_parallel` 的 docstring 先摆 16 卡的例子(TP 组 `[g0,g1]`…、PP 组 `[g0,g4,g8,g12]`…,`megatron/core/parallel_state.py:682-692`),紧接着写:「**Note that for efficiency, the caller should make sure adjacent ranks are on the same DGX box.** For example if we are using 2 DGX-1 boxes with a total of 16 GPUs, rank 0 to 7 belong to the first box and ranks 8 to 15 belong to the second box.」(`:693-696`)。而 `order` 形参本身的说明只写取值范围:「The rank initialization order of parallelism. Now we support tp-dp-pp and tp-pp-dp orders.」(`:653-655`)。同向的另一条证据是 `high_priority_stream_groups` 的说明 ——「are scheduled with higher priority, minimizing the exposed communication when it is overlapped with other computation kernels」(`:672-673`),说明编排层确实按"通信是否暴露在关键路径上"给不同的组排优先级。
→ 注意分寸:源码陈述的是**"相邻 rank 应当同机"**这一条;§1.5 那句"步长最小的轴落进 NVLink 域"是把它与 `order` 的前缀积展开合起来得到的,不是源码原话。

**③ 为什么是两个 RankGenerator,而不是一个六维的。**
`RankGenerator.__init__` 的第一行断言就把这条写死:`assert ep == 1 or cp == 1, "Both EP and CP > 1 in not allow in one rank generator. CP is only included in default RankGenerator, and EP only in expert RankGenerator."`(`megatron/core/parallel_state.py:450-453`)。
**被否掉的替代②:一个 generator + 一个 `independent_ep` 布尔开关。** 提交 `7f22e210c`(2024-11-23,commit message 即「MoE parallel folding: separate MoE parallel states from dense」)之前只有**一个** RankGenerator,它同时持有 `order_w_ep` / `order_wo_ep` 两套分解,用 `get_ranks(token, independent_ep=False)` 的开关切换;那个形参的 docstring 直说了当时的语义 ——「This flag controls whether we treat EP and DP independently. **EP shares ranks with DP**, if we want to get ranks related to EP, we should set the flag. For example, get_ranks('dp', True) will get DP modulo EP group」(`7f22e210c^:megatron/core/parallel_state.py:332`,形参说明 `:342-347`),调用侧写成 `generator_wrapper('tp-ep-pp', independent_ep=True)` 这样的形式(`:755`、`:870`、`:877`、`:884`、`:893`)。
→ 判据是**让 MoE 侧的 TP/DP 度能与 dense 侧不同**:EP 只要还是"从 DP 里切出来的一段",专家层就没法独立设 ETP/EDP;拆成两个 generator 之后,两套分解只需在 PP 轴上对齐 —— 这条对齐今天是一句显式断言(`megatron/core/parallel_state.py:808-811`),外加一条"order 不以 pp 结尾时两侧 DP 必须相同"的兜底(`:802-806`)。

**④ 为什么三层抽象并存,而不是一次性迁移。**

> [!note] 推断
> 源码陈述的是**方向**,不是**并存的理由**:`AGENTS.md:34-48` 的「Megatron Core Process Groups」条目要求 `megatron/core` 生产代码「avoid adding new direct reads of global process groups from `parallel_state`」,改为「accepting a `ProcessGroupCollection` or an explicit `torch.distributed.ProcessGroup` from the caller and passing that through」,并**逐条列出豁免点**(`parallel_state.py`、`process_groups_config.py`、初始化引导、测试、带注释的迁移回退);`HyperCommGrid` 的 docstring 也只声明它与 `initialize_model_parallel` 的 `order` 等价(`megatron/core/hyper_comm_grid.py:49-52`)。**"三者并存是因为迁移只能逐点进行、全局单例一时拆不掉"这层解释由本页承担,源码没有这样表态**;那条规范本身写的也是"不要新增",而非清理存量。要引用这条判断,请回到 `AGENTS.md:34-48`、`megatron/core/hyper_comm_grid.py:47-64` 与 §5② 的 `use_mpu_process_groups()` 回退路径,不要引用本段推断。

---

## 3. 核心机制:正交 rank 分组

### 3.1 `RankGenerator` + `order`

`RankGenerator`(`megatron/core/parallel_state.py:444`)是编排的核心类。构造时拿到各轴大小和 `order`:

```python
class RankGenerator:
    def __init__(self, tp, ep, dp, pp, cp, order, rank_offset=0):
        assert ep == 1 or cp == 1, \
            "Both EP and CP > 1 not allowed in one rank generator."   # ← 见 §4
        self.world_size = tp * dp * pp * cp * ep
        self.order = order                                            # 如 "tp-cp-ep-dp-pp"
        self.ordered_size = [name_to_size[t] for t in order.split("-")]  # 各轴大小,按 order 排
```

取某个轴(或轴组合)的 rank 列表用 `get_ranks(token)`:

```python
def get_ranks(self, token):                # token 如 "tp"、"dp"、"tp-dp"、"dp-cp"
    mask = self.get_mask(self.order, token)                          # 哪些轴要"组进同一组"
    ranks = generate_masked_orthogonal_rank_groups(
        self.world_size, self.ordered_size, mask)
    return ranks
```

### 3.2 `generate_masked_orthogonal_rank_groups` 的数学

这个函数(`megatron/core/parallel_state.py:250`)是整套编排的数学内核。给定 n 维布局 + 一个 mask(哪些维"沿之分组"),求出所有正交 rank 组:

```python
masked_shape   = [维大小 for 维 in 各维 if mask[维]]      # 组内成员沿这些维变化
unmasked_shape = [维大小 for 维 in 各维 if not mask[维]]  # 不同组沿这些维变化
group_size  = prod(masked_shape)                          # 每组多少卡
num_of_group = world_size // group_size                   # 多少个组

for group_index in range(num_of_group):
    for rank_in_group in range(group_size):
        rank = inner_product(decompose(rank_in_group, masked_shape),   masked_stride)
             + inner_product(decompose(group_index,   unmasked_shape), unmasked_stride)
```

直觉:`mask` 为 True 的维构成"组内坐标",False 的维构成"组编号";`global_rank = Σ idx[i]·stride[i]` 把两部分拼回一维编号。

### 3.3 具体例子(`world_size=8`,`order="tp-dp-pp"`,`tp=dp=pp=2`)

步长:`s_tp=1, s_dp=2, s_pp=4`,`global_rank = tp + dp·2 + pp·4`。

| global_rank | tp | dp | pp |
|:-:|:-:|:-:|:-:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 |
| 3 | 1 | 1 | 0 |
| 4 | 0 | 0 | 1 |
| 5 | 1 | 0 | 1 |
| 6 | 0 | 1 | 1 |
| 7 | 1 | 1 | 1 |

`get_ranks` 产出的三类组(已用 §3.2 的公式核验):

```
TP 组(mask=tp):   [[0,1], [2,3], [4,5], [6,7]]    ← 相邻 rank,落同一节点 ✅
DP 组(mask=dp):   [[0,2], [1,3], [4,6], [5,7]]    ← 步长 2
PP 组(mask=pp):   [[0,4], [1,5], [2,6], [3,7]]    ← 步长 4,跨整机 ✅
```

一张卡(如 `rank 3`)同时属于:`TP 组 [2,3]`、`DP 组 [1,3]`、`PP 组 [3,7]` —— 这就是"一张 GPU 同时在多个并行组里"的真相。

```
rank 布局(order=tp-dp-pp):
  节点视角:  [0 1 | 2 3]  [4 5 | 6 7]
              └TP┘ └TP┘    └TP┘ └TP┘     ← TP 组永远是相邻对
  PP 组 [0,4] 横跨两半 ─────────────────► ← PP 组跨得最远
```

---

## 4. MoE Parallel Folding 的编排实现:两个 RankGenerator

`RankGenerator.__init__` 第一行断言 `ep == 1 or cp == 1` —— **EP 和 CP 不能在同一个 RankGenerator 里同时 > 1**。原因:attention 层用 CP、MoE 层用 EP,二者是**同一组 GPU 的两套不同分解**(即 `14_megatron_ep_analysis.md` §9 的 MoE Parallel Folding)。

`initialize_model_parallel`(`megatron/core/parallel_state.py:545`)因此构造**两个** RankGenerator:

```python
# ① 给 attention / dense 层用:含 CP,ep=1
decoder_rank_generator = RankGenerator(
    tp=tensor_model_parallel_size, ep=1, dp=data_parallel_size,
    pp=pipeline_model_parallel_size, cp=context_parallel_size, order=order)

# ② 给 MoE 专家层用:含 EP,cp=1,TP/DP 可独立设
expert_decoder_rank_generator = RankGenerator(
    tp=expert_tensor_parallel_size,            # ETP,默认 = TP,推荐设 1
    ep=expert_model_parallel_size,             # EP
    dp=expert_data_parallel_size,              # EDP = world_size / (ETP·EP·PP)
    pp=pipeline_model_parallel_size, cp=1, order=order)
```

两个 generator 在**同一个 `world_size`** 上,但用不同的轴分解:

```
同一组 GPU                ┌─ attention 视角:  TP × CP × DP × PP
                          └─ MoE 视角:       ETP × EP × EDP × PP
                                              └ CP 被"折叠"进 EP/EDP

唯一硬约束:两套分解的 PP 组必须逐一相等
  assert decoder_rank_generator.get_ranks("pp")
       == expert_decoder_rank_generator.get_ranks("pp")
```

PP 是两套分解唯一共享的轴(模型按层切,attention 层和 MoE 层在同一条流水线上)。这就是编排层如何实现"attention 和 MoE 用不同并行度"的 —— `14_megatron_ep_analysis.md` §9 描述的能力,落地点就在这两个 RankGenerator。

---

## 5. 三层抽象逐个看

### ① `parallel_state` —— 全局单例(经典)

`initialize_model_parallel(...)` 一次性把所有组建好,存进模块级全局变量(`_TENSOR_MODEL_PARALLEL_GROUP` 等)。之后任意代码用无参 getter 取:

```python
parallel_state.initialize_model_parallel(tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4, context_parallel_size=2, order="tp-cp-ep-dp-pp")
...
tp_group = parallel_state.get_tensor_model_parallel_group()    # 全局取,无需传参
```

`megatron/core/parallel_state.py` 提供几十个 `get_*` / `is_*` / `get_*_rank` / `get_*_world_size`,以及 PP 专用的 `is_pipeline_first/last_stage`、`get_pipeline_model_parallel_next/prev_rank`(PP 文档 P2P 通信的环形邻居就来自这里)。

- **优点**:简单,任意深度的代码不必层层传参。
- **缺点**:全局可变状态;一个进程只能有一套并行配置;多模型/测试不友好。

### ② `ProcessGroupCollection` —— 显式传递

`megatron/core/process_groups_config.py:27`。一个 dataclass,把所有组作为**显式字段**装在一起:

```python
@dataclass
class ProcessGroupCollection:
    tp / pp / mp / cp / dp / dp_cp / ep / expt_tp / tp_ep / embd / pos_embd / ...
```

它就是前五份文档里到处出现的 `pg_collection=` 参数。代码不再调全局 getter,而是接收一个 `pg_collection` 对象 —— 即依赖注入。`use_mpu_process_groups()`(`megatron/core/process_groups_config.py:169`)能从全局 `parallel_state` 抽出一个 `ProcessGroupCollection`(向后兼容);`setup_process_groups_for_ddp`(`megatron/core/process_groups_config.py:454`)/ `setup_process_groups_for_optimizer`(`megatron/core/process_groups_config.py:263`)按 DDP / 优化器的需要裁出子集。

`MultiModuleProcessGroupCollection`(`megatron/core/process_groups_config.py:585`)是它的多模块版本:多模态模型(视觉编码器 + 语言模型)每个子模块一套 `ProcessGroupCollection`,用 `language_model_module_name` 标出主干 —— 这是 PP 文档提到的"多模块 PP"的进程组基础(详见后续 PP 补遗文档)。

- **优点**:无全局状态;一个进程可并存多套配置;可测试。
- **缺点**:要层层传 `pg_collection`(故大量函数签名里都有这个可选参数)。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `85902ef5`。 — 训练循环向 `pg_collection` 迁移(#5259 / #5250 / #5251 / #5111)
>
> `ee3f1ff` 之后有一批 PR 持续把训练侧对全局 `parallel_state.get_*_group()` 的隐式依赖,改成显式接收 `ProcessGroupCollection` / `group=` —— 即把上文的「依赖注入」从 `megatron/core` 推进到 `megatron/training`:
> - **`train_step` 新增 `pg_collection` 形参**(#5259,新基线 `megatron/training/training.py:3026`、形参在 `:3035`)。step 末尾三处规约不再硬编码全局组:`logical_and_across_model_parallel_group` / `reduce_max_stat_across_model_parallel_group` 改用 `pg_collection.mp`(`megatron/training/training.py:3323`、`:3328`、`:3331-3335`),loss 平均 all-reduce 改用 `pg_collection.dp_cp`(`:3324`、`:3364`),末 stage 判定改用 `is_pp_last_stage(pg_collection.pp)`(`:3325`);解析不到时回退 `ProcessGroupCollection.use_mpu_process_groups()`,并断言含 `mp/pp/dp_cp`(`:3317-3322`)。
> - **`get_model` 的 DDP 桶大小改用注入组**(#5250,新基线 `megatron/training/training.py:2397`,形参 `pg_collection=None` 在 `:2402`、回退在 `:2407-2408`):`bucket_size` 由 `pg_collection.dp_cp` 推出、PP rank 用 `get_pg_rank(pg_collection.pp)`(`megatron/training/training.py:2565`),替换原 `mpu.get_data_parallel_world_size(...)` / `mpu.get_pipeline_model_parallel_rank()`。新基线上桶大小的算式已抽成独立 helper `resolve_ddp_bucket_size(ddp_config, dp_cp_group, overlap_grad_reduce, num_parameters)`(`megatron/training/training.py:2254`),`get_model` 只是把 `pg_collection.dp_cp` 传进去(`:2558-2560`);注入的组仍是 `dp_cp`,机制不变。
> - **`common_utils` 规约 helper 新增可选 `group=`**(#5251,新基线 `megatron/training/utils/common_utils.py:236`/`252`/`277`):`average_losses_across_data_parallel_group` / `reduce_max_stat_across_model_parallel_group` / `logical_and_across_model_parallel_group` 都接受显式 `group`,缺省才回退 mpu 全局组 —— 为上面两个 PR 的注入提供下游支持。
> - **why(规范背书)**:#5111 在 `AGENTS.md` 写入「Megatron Core Process Groups」指引(`AGENTS.md:34`;advisory,非 CI 卡口):`megatron/core` 生产代码**禁止新增**对 `parallel_state.get_*_group()` 的直接读取,应改为接收 `ProcessGroupCollection` 或显式 `ProcessGroup` 并向下透传;仅 `megatron/core/parallel_state.py` / `megatron/core/process_groups_config.py` / 初始化引导 / 测试 / 带注释的迁移回退是豁免点。这条规范正是 §1.2 表中「① 全局单例 → ② 显式注入 → ③ HyperCommGrid」演进的官方背书,也解释了为何越来越多函数签名带 `pg_collection=` / `group=`。

> [!contradiction] `pg_collection` 的**来源**自基线 `71092579` 起已变(新基线 `85902ef5` 仍然如此):上文说的「`pg_collection=None` 时回退 `use_mpu_process_groups()`」不再是从形参取值。新基线的 `train_step` 先从模型上取——`pg_collection = get_attr_wrapped_model(model[0], "pg_collection")`(`megatron/training/training.py:3316`),只有它为 `None` 才回退 `ProcessGroupCollection.use_mpu_process_groups()`(`:3317-3318`);源码注释写明理由是「Reductions source per-rank groups from the model (encoder rank -> encoder groups)」(`:3315`)。同名的 `pg_collection` 形参仍在(`:3035`),但语义已改成「转发给 schedule 的 cross-grid 载体」,docstring 明说 `None` 时保持默认行为(`:3040-3041`),**不再**参与这三处规约的组选择。也就是说依赖注入的方向从「调用方传入」变成了「从模型自带的进程组读取」,注入点下沉一层。

### ③ `HyperCommGrid` —— N 维网格(最新)

`class HyperCommGrid`,`megatron/core/hyper_comm_grid.py:46`。把"5 维超长方体"这个概念**直接对象化**:

> [!note] 位置沿革:`class HyperCommGrid` 在旧基线 `ee3f1ffa…` 为 `megatron/core/hyper_comm_grid.py:33`,自 `dev@232c478d4` 起移至 `:46`,新基线 `71092579` 仍为 `:46`(已 `git show` 确认);文件由 273 行增至 443 行,主因是 #5148 新增 named views(见本节末更新)。

```python
grid = HyperCommGrid([2, 3, 4, 5], ["tp", "cp", "pp", "dp"])   # shape + 维名
dp_group    = grid.create_pg("dp")                              # 按维名建组
dp_cp_group = grid.create_pg(["cp", "dp"], pg_options=..., group_desc="...")
```

`dim_names` 的顺序等价于 `initialize_model_parallel` 的 `order`(docstring 明确写出等价关系)。`create_pg(dims)` 对任意维组合建组,`get_pg(dims)` 取回。它把 `RankGenerator + order 字符串 + generate_masked_orthogonal_rank_groups` 那套机制收敛成一个干净的 N 维 API,且支持 `rank_offset`(网格不占满整个 world 时)。

- **优点**:任意 N 维、任意维组合;无全局状态;API 最清晰。
- **定位**:Megatron 进程组管理的演进方向。

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。 — HyperCommGrid 新增 named views(异构并行)(#5148)
>
> `HyperCommGrid` 现支持在**同一段 rank 跨度**上注册多个并存的「命名因子分解(rank view)」,服务异构并行(不同子模型用不同并行度)。核心新增(`megatron/core/hyper_comm_grid.py`):
> - **`register_view(name, shape, dim_names, shared_dims=None)`**(`megatron/core/hyper_comm_grid.py:143`):为同一组 rank 登记另一套 `shape × dim_names` 分解。约束:`prod(shape)` 必须等于网格 size;`shared_dims` 列出的轴必须在 base view 与新 view 里**枚举出完全相同的 rank 组**(逐组校验,不等则 `raise`)—— 这保证共享轴(典型如 PP)在两套分解下成员一致,与 §4「两个 RankGenerator 的 PP 组必须逐一相等」是同一约束思想的对象化版本。
> - **`create_pg` / `get_pg` / `get_rank_enum` 新增 `view="..."` 关键字参数**(`megatron/core/hyper_comm_grid.py:206`、`:287`、`:313`,默认 `base` view)。base view 的组仍按「短横线拼接的维名」做键;view 私有组用 `(view_name, dims)` 元组做键;若某组只涉及 `shared_dims`,会**复用 base view 的同一进程组**(单键存储,`destroy` 时只销毁一次,`_canonical_pg_key_and_enum_view` `megatron/core/hyper_comm_grid.py:418`)。
> - **底层去掉 einops 依赖**:`_gen_rank_enum` 重构为 `_gen_rank_enum_for(shape, dim_names, dims)`(`megatron/core/hyper_comm_grid.py:356`),用 `numpy.arange + reshape + moveaxis` 直接生成 rank 枚举,不再 `einops.rearrange`。语义与原 MCore 约定(`dim_names` 逆序)一致,行为不变。
>
> 这是上文「演进方向」的落地一步:`HyperCommGrid` 从「单一 N 维网格」升级为「一段 rank 上挂多套命名分解」,正是多模态/异构子模型(见 [[15_megatron_pp_schedulers_analysis]] §8.2 BridgeCommunicator)所需的几何基础。

---

## 6. 约束:组合规则、前提与失效条件

### 6.1 维度整除约束

```
world_size 必须能被 model_size 整除:
  model_size = tp · pp · cp                 (attention 侧)
  data_parallel_size = world_size / model_size

MoE 侧:
  expert_model_size = etp · ep · pp
  expert_data_parallel_size = world_size / expert_model_size
```

各轴还有自身约束(散见前五份文档):`tp | num_heads`、`ep | num_experts`、`pp | num_layers`、`cp | seq_len`、VPP 需 `pp·vp | num_layers`、`m % N` 约束(VPP)等。

### 6.2 `order` 的选择

- 默认 `"tp-cp-ep-dp-pp"`:TP 最内(NVLink)、PP 最外(IB)。绝大多数场景直接用默认。
- 非默认 order 有一条断言(`megatron/core/parallel_state.py:802`):不以 `pp` 结尾时,attention 与 MoE 的 DP 大小必须相同 —— 否则 folding 的 PP 组对不齐。

### 6.3 一张卡的归属总览

```
给定 order="tp-cp-ep-dp-pp",一张 global_rank 的卡同时属于:
  TP 组   ← tensor_parallel.py 的 ColumnParallel/RowParallel all-reduce 域
  CP 组   ← attention 的 KV 搬运域(13_megatron_cp_analysis.md)
  EP 组   ← MoE dispatch/combine 的 A2A 域(14_megatron_ep_analysis.md)—— 走 expert RankGenerator
  DP 组   ← 梯度 all-reduce / ZeRO 分片域(16_megatron_distributed_optimizer_analysis.md)
  PP 组   ← 流水线 P2P 邻居(15_megatron_pp_schedulers_analysis.md)
  + 组合组:dp_cp(梯度规约含 CP)、tp_ep(MoE AllGather dispatcher 域)、
            embd(PP 首尾共享 embedding)、mp(tp+pp,模型并行整体)…
```

### 6.4 前提、代价与故意不做的事

上面两张表给的是"配置怎么填"。这一节补的是**越出前提会发生什么** —— 编排层几乎所有边界都是运行时断言,配置错了要等到 `initialize_model_parallel` 才炸。

| # | 前提 / 不变量 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | 一个 `RankGenerator` 里 EP 与 CP 不能同时 > 1 | `megatron/core/parallel_state.py:450-453` | `AssertionError`。这不是可放宽的性能约束,而是"双 generator"设计本身的定义(§4) |
| 2 | 两套分解的 PP 组必须**逐一相等** | `:808-811` | assert 失败,报文直接打印两侧的 `get_ranks('pp')` |
| 3 | `order` 不以 `pp` 结尾时,attention 与 MoE 的 DP 大小必须相同 | `:802-806` | assert 失败(§6.2 已述);这是 folding 的 PP 组对不齐的前置检查 |
| 4 | 并行度 > 1 的轴必须出现在 `order` 里 | `:473-478`,抛 `RuntimeError` | 缺轴且该轴 size≠1 直接抛错;size==1 的轴则被静默补到 order 末尾(`:479-480`)—— 单卡跑通不代表该轴的排布被覆盖到 |
| 5 | `world_size` 必须同时被 dense 侧与 expert 侧的 model size 整除 | expert 侧 `:786-789`(`RuntimeError`) | 抛错,报文给出 `world_size` 与 `expert_tensor_model_pipeline_parallel_size` |
| 6 | `HyperCommGrid` 上同一维组合只能建一次组 | `megatron/core/hyper_comm_grid.py:54-55`:「For any combination of dimensions, a process group can only be created once. Creating process groups for the same combination with different options is not supported.」 | 想给同一组换一套 `pg_options`,只能另建 grid |
| 7 | `register_view` 的 `shared_dims` 必须在两套分解下枚举出**完全相同**的 rank 组 | `megatron/core/hyper_comm_grid.py:143` | 不等即 `raise`(§5③ 的 2026-06-16 更新已述) |

**代价**

- **正确性靠断言而非类型。** 整除、PP 对齐、order 覆盖这些跨轴关系全部是运行时 assert,且分散在 `initialize_model_parallel` 的不同位置;一次错误的并行度组合在建组阶段才暴露,此时模型和数据集往往已经加载。
- **三层抽象并存的代价是"同一个组有三条取法"。** 读一段用到 `tp_group` 的代码,必须先判断它走的是全局 getter、`pg_collection` 还是 `HyperCommGrid`(§5)—— 这也是 §5② 那条 `[!contradiction]` 记录的注入点下沉之所以容易读错的原因。
- **`get_pg` 不负责建组。** `HyperCommGrid` 明确不在 `get_pg()` 里惰性建组,理由写在 docstring 的 Note 里 ——「there are many options (kwargs) that can be passed when creating a process group, which `get_pg()` should not be exposed to」(`megatron/core/hyper_comm_grid.py:60-64`);代价是调用方必须记得先 `create_pg`,漏了就取不到。

**故意不做的事**

- **不给 expert 侧独立的 `order`。** 两个 RankGenerator 共用同一个 `order` 字符串,`expert_decoder_rank_generator` 的构造点正上方挂着 `# TODO: support expert specific ordering`(`megatron/core/parallel_state.py:791`)。§4 的"两套分解"目前只在**轴大小**上独立,**轴顺序**仍然共享。
- **不把 `expt_dp` 从 `ProcessGroupCollection` 里拿掉。** 该字段旁的注释说明它只是 dist-checkpoint 尚未重构前的 workaround(「we need this workaround until distributed checkpoint is refactored to have sharded_state_dict can take the PG and pass it down」),并挂 `# TODO (Hepteract): remove this once distributed checkpoint is refactored`(`megatron/core/process_groups_config.py:120-125`,填充点 `:240-243`)。
- **不把 `megatron/training` 一起纳入注入规范。** `AGENTS.md:46-48` 明写「This guidance targets Megatron Core library code. Do not apply it to `megatron/training` or other training-loop code unless the PR explicitly opts into that migration.」—— §5② 记录的那批训练侧迁移是逐个 PR "opt in" 的,不是全局要求。

---

## 7. 与前五份文档的衔接

| 文档 | 它假设的"几何",由本文哪个机制提供 |
|------|-----------------------------------|
| `12_megatron_tp_analysis.md` | `tp_group` ← decoder RankGenerator 的 `get_ranks("tp")`,`order` 第一位保证机内 |
| `15_megatron_pp_schedulers_analysis.md` | `pp_group`、`get_pipeline_model_parallel_next/prev_rank` ← `get_ranks("pp")` |
| `13_megatron_cp_analysis.md` | `cp_group` ← decoder RankGenerator(`ep=1`)的 `get_ranks("cp")` |
| `14_megatron_ep_analysis.md` | `ep_group`/`tp_ep` ← **expert** RankGenerator(`cp=1`);MoE Parallel Folding = §4 的双 generator |
| `16_megatron_distributed_optimizer_analysis.md` | `dp`/`dp_cp` 组、HSDP 的内外层 ← `get_ranks("dp")` / `setup_process_groups_for_ddp` |

### 7.1 一句话总结

- **编排层的工作**:把 `world_size` 个一维 `global_rank` 解释成 5 维坐标,据此为每个轴建 `ProcessGroup`。
- **`order` 字符串**决定物理布局:第一个轴步长 1(相邻、机内、NVLink),最后一个轴步长最大(跨机、IB)—— 这是"TP 内 PP 外"所有建议的量化根源。
- **正交分组数学**(`generate_masked_orthogonal_rank_groups`):mask 选中的维做"组内坐标",其余做"组编号",`global_rank = Σ idx·stride` 拼回。
- **两个 RankGenerator**(decoder 含 CP、expert 含 EP)= MoE Parallel Folding 的编排落地;唯一硬约束是两者 PP 组相等。
- **三层抽象**:`parallel_state` 全局单例(经典)→ `ProcessGroupCollection` 显式注入(前五份文档的 `pg_collection`)→ `HyperCommGrid` N 维网格(演进方向)。

---

## 8. 发展趋势

> [!note] 推断:锚点是基线 `71092579` 下的源码事实(TODO、`DeprecationWarning`、新 API),方向判断由本页承担,不是源码的自陈计划。

**一、`order` 会从"两套分解共享"变成"expert 侧可独立指定"。**
§4 的双 RankGenerator 目前只在轴**大小**上独立(ETP/EP/EDP 可以与 TP/CP/DP 不同),轴**顺序**仍然共用同一个 `order`;`expert_decoder_rank_generator` 构造点上方就挂着 `# TODO: support expert specific ordering`(`megatron/core/parallel_state.py:791`)。**由此可推断**:"MoE 侧的物理邻近度应当与 dense 侧不同"(例如 dense 想要 TP 最内、MoE 想要 EP 最内)这件事已经被识别为缺口 —— §1.5 那条"order 第一位落 NVLink 域"的推论,将来对 expert 侧可能要分开来算。

**二、全局单例的退场不止于进程组,VPP rank 也在改成显式传参。**
`set_virtual_pipeline_model_parallel_rank` 现在一进函数就发 `DeprecationWarning`:「set_virtual_pipeline_model_parallel_rank in global scope is deprecated. Pass vp_stage explicitly to is_pipeline_first_stage, is_pipeline_last_stage, etc.」(`megatron/core/parallel_state.py:1730-1736`)。**由此可推断**:§1.2 表里那条"① 全局单例 → ② 显式注入"的演进正在从 *ProcessGroup* 扩展到 *并行位置状态* 本身;`parallel_state` 里剩下的 `_VIRTUAL_PIPELINE_*` 一类模块级变量,应当被视为迁移中而非稳定 API(PP 侧的用法见 [[15_megatron_pp_schedulers_analysis]])。

**三、`ProcessGroupCollection` 的字段表会随 dist-checkpoint 重构缩小。**
`expt_dp` / `expt_dp_ag` 两个字段带着明确的临时性注释和 `# TODO (Hepteract): remove this once distributed checkpoint is refactored`(`megatron/core/process_groups_config.py:120-125`,`use_mpu_process_groups` 的填充点 `:240-243`),理由是「sharded_state_dict」目前无法接收进程组并向下传递。**由此可推断**:§5② 的 dataclass 字段清单不是稳定接口 —— 一旦 checkpoint 侧改成显式接收 PG(检查点侧现状见 [[19_megatron_dist_checkpointing_analysis]]),这两个字段会消失,依赖 `pg_collection.expt_dp` 的代码要跟着改。

**四、`HyperCommGrid` 从"一张网格"走向"一段 rank 上挂多套命名分解"。**
类 docstring 现在直接写着「Methods default to the base factorization. Register additional factorizations of the same rank span with :meth:`register_view` and target them via ``view="..."``」(`megatron/core/hyper_comm_grid.py:57-58`),对应 §5③ 记录的 #5148:`register_view`(`:143`)、`create_pg`/`get_pg`/`get_rank_enum` 的 `view=` 形参(`:206`、`:287`、`:313`)。**由此可推断**:§4 那种"两个 RankGenerator + 一条 PP 对齐断言"的写法,在 `HyperCommGrid` 一侧已经有了对象化的对应物(`shared_dims` 逐组校验);多模态/异构子模型继续铺开时,编排层的重心会从 `parallel_state` 移到 grid + view —— 源码没有给迁移时间表。

**五、注入点正在从"调用方传入"下沉到"从模型读取"。**
§5② 的 `[!contradiction]` 记录了这次变化:`train_step` 里参与规约的 `pg_collection` 已改为 `get_attr_wrapped_model(model[0], "pg_collection")`,同名形参降级为"转发给 schedule 的 cross-grid 载体"。**由此可推断**:异构并行(不同子模块不同并行度)一旦成为常态,"当前进程属于哪些组"就不再是进程级属性,而是**模块级**属性;再写编排相关代码时,应先问"这段代码属于哪个子模块",而不是"这个进程的 TP 组是什么"。这一步与上面第四条是同一方向的两端 —— 但源码只表达了实现,没有表达意图。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准。本文是"第一层补遗"3 份文档之①(并行编排 capstone),后续:② PP 补遗、③ TP·FSDP·resharding 补遗。*

---

## 配置契约：`DistributedInitConfig` 与 `RNGConfig`

本页正文讲进程组**怎么被构造**——RankGenerator 的正交分组数学、order 字符串、三层抽象。本节给这一层的**用户配置面**：`DistributedInitConfig` 决定 `torch.distributed` 怎么起来（后端、超时、初始化方法、NCCL 通信器配置），`RNGConfig` 决定随机性怎么按并行轴分化——后者与本页的分组直接相关：**同一 TP 组内要同种子、不同 DP 组要异种子**，否则要么权重初始化不一致、要么数据并行退化成重复计算。

两个类都在 `megatron/training/config/common_config.py`，经 [[41_megatron_config_surface_analysis]] §2 的工厂自动转 CLI（`megatron/training/arguments.py:4071`、`:3882`）。**下表直接取自类体**，行号为该文件内行号。


### `DistributedInitConfig`（`megatron/training/config/common_config.py`，15 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `distributed_backend` | `Literal['nccl', 'gloo']` | `'nccl'` | Which backend to use for distributed training. | `:77` |
| `distributed_timeout_minutes` | `int` | `10` | Timeout minutes for torch.distributed. | `:80` |
| `lazy_mpu_init` | `bool` | `False` | If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on --use-cpu-initialization flag. This… | `:91` |
| `nccl_communicator_config_path` | `str \| None` | `None` | Path to the yaml file with NCCL communicator configurations. The number of min/max thread groups and thread group cluster size of each communicator can be co… | `:103` |
| `use_tp_pp_dp_mapping` | `bool` | `False` | If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp. | `:108` |
| `use_gloo_process_groups` | `bool` | `field(default=True, metadata={'argpar…` | If enabled, create Gloo process groups for communications. | `:112` |
| `use_sharp` | `bool` | `False` | Set the use of SHARP for the collective communications of data-parallel process groups. When `True`, run barrier within each data-parallel process group, whi… | `:117` |
| `sharp_enabled_group` | `Literal['dp', 'dp_replica'] \| None` | `None` | IB SHARP can be enabled from only one communication group. By default, it is enabled from dp group if not specified and use_sharp=True. Available options: [d… | `:123` |
| `distributed_timeout_seconds_after_init` | `int \| None` | `None` | Timeout in seconds for process groups after initialization. This timeout is applied to all process groups after initialization and the first iteration comple… | `:136` |
| `flight_recorder_dump_path` | `str \| None` | `None` | Path for NCCL flight recorder trace dumps. Sets TORCH_FR_DUMP_TEMP_FILE and TORCH_NCCL_DEBUG_INFO_TEMP_FILE env variables before distributed init. | `:139` |
| `flight_recorder_trace_buffer_size` | `int` | `2000` | Size of the NCCL flight recorder trace buffer (TORCH_NCCL_TRACE_BUFFER_SIZE). | `:142` |
| `flight_recorder_dump_on_timeout` | `bool` | `True` | Dump flight recorder traces on NCCL timeout (TORCH_NCCL_DUMP_ON_TIMEOUT). | `:145` |
| `flight_recorder_include_stack_trace` | `bool` | `False` | Include stack traces in flight recorder dumps (TORCH_INCLUDE_STACK_TRACE). | `:148` |
| `flight_recorder_include_only_active` | `bool` | `True` | Include only active operations in flight recorder dumps (TORCH_INCLUDE_ONLY_ACTIVE). | `:151` |
| `flight_recorder_extra_dump_on_exec` | `bool` | `True` | Enable extra flight recorder dump on execution (TORCH_NCCL_EXTRA_DUMP_ON_EXEC). | `:154` |

> 该类共 21 个字段，本表收 15 项；其余 6 项已在别处归属：`align_grad_reduce` → [[20_megatron_comm_overlap_analysis]]；`local_rank`、`high_priority_stream_groups` → 本页他处；`use_megatron_fsdp` → [[36_megatron_fsdp_analysis]]；`use_torch_fsdp2` → [[16_megatron_distributed_optimizer_analysis]]；`disable_jit_fuser` → [[21_megatron_fusion_operators_analysis]]。



### `RNGConfig`（`megatron/training/config/common_config.py`，3 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `seed` | `int` | `1234` | Random seed used for python, numpy, pytorch, and cuda. | `:11` |
| `inference_rng_tracker` | `bool` | `False` | Use a random number generator configured for inference. | `:18` |
| `data_parallel_random_init` | `bool` | `False` | Enable random initialization of params across data parallel ranks | `:21` |

> 该类共 4 个字段，本表收 3 项；其余 1 项已在别处归属：`te_rng_tracker` → [[23_megatron_precision_cudagraph_fusion_analysis]]。

> **接缝**：`nccl_communicator_config_path` 指向的 YAML 由本页 §正文讲的 `get_nccl_options` 消费，是「按进程组分别配 NCCL」的入口；`distributed_timeout_minutes` 则与 [[43_megatron_job_resilience_analysis]] §2.2 的自适应超时是两套独立的超时机制——前者是 torch.distributed 的建链超时，后者是 NVRx 的心跳超时。

---

## 配置契约：RNG tracker 补充

本页前一节给了 `RNGConfig`。本节补 `TransformerConfig` 里与 RNG tracker 相关、此前零提及的字段——它与 §正文的进程组分化直接相关：tracker 的实现选择会影响各并行轴上随机状态的隔离方式。





### `ModelParallelConfig`（`megatron/core/model_parallel_config.py`，1 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `use_te_rng_tracker` | `bool` | `field(default=False, metadata={'argpa…` | If true, uses RNG state tracker in TransformerEngine if exists. Required for CUDA graphs support. | `:248` |

> 该类共 74 个字段，本表收 1 项；其余 73 项已在别处归属：主要归 [[15_megatron_pp_schedulers_analysis]] 16 项、[[12_megatron_tp_analysis]] 10 项、[[20_megatron_comm_overlap_analysis]] 10 项、[[22_megatron_memory_optimization_analysis]] 6 项，另散见 13 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。



### `TransformerConfig`（`megatron/core/transformer/transformer_config.py`，1 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `use_te_rng_tracker` | `bool` | `False` | Whether to use the TE or MCore version of the RNG tracker. | `:1343` |

> 该类共 266 个字段，本表收 1 项；其余 265 项已在别处归属：主要归 [[10_megatron_model_structure_analysis]] 92 项、[[14_megatron_ep_analysis]] 38 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 38 项、[[21_megatron_fusion_operators_analysis]] 26 项，另散见 20 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。

## Related Pages

- [[15_megatron_pp_schedulers_analysis]] · [[14_megatron_ep_analysis]] · [[12_megatron_tp_analysis]] · [[13_megatron_cp_analysis]] · [[16_megatron_distributed_optimizer_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]


