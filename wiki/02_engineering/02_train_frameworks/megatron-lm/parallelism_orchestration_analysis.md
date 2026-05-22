# Megatron-LM 并行编排与进程组构造深度解析(Capstone)

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/parallel_state.py`(2255 行)、`process_groups_config.py`(718 行)、`hyper_comm_grid.py`(273 行)
> 配套阅读:`pp_schedulers_analysis.md`、`ep_analysis.md`、`tp_analysis.md`、`cp_analysis.md`、`ddp_optimizer_analysis.md`
> 定位:**这是收口文档**。前五份各讲一个并行轴,都默认了"每张 GPU 同时属于 TP/PP/CP/EP/DP 的某个组"。本文讲清楚这套**几何**是怎么从 `world_size` 个裸 GPU 构造出来的。

---

## 0. 总览

### 0.1 这份文档解决什么

前五份文档反复出现一个隐含前提:某张卡"在 TP 组里"、"在 PP 组里"、"在 EP 组里"……但**一张物理 GPU 怎么会同时属于 5 个组?这些组的成员是谁?谁来构造?** —— 这就是"并行编排"。

一张 GPU 的全局编号 `global_rank ∈ [0, world_size)`。编排层的工作:把这个一维编号**解释成一个 5 维坐标** `(tp_rank, cp_rank, ep_rank, dp_rank, pp_rank)`,再据此为每个轴建立 `torch.distributed.ProcessGroup`。前五份文档里所有 `tp_group` / `pp_group` / `ep_group` / `dp_cp_group` / `pg_collection` 全部源于此。

### 0.2 三层抽象(历史演进)

Megatron 的进程组管理有三套实现,层层递进:

| # | 抽象 | 文件 | 形态 | 定位 |
|---|------|------|------|------|
| ① | `parallel_state` | `parallel_state.py` | **全局单例** + `get_*_group()` | 经典实现,全局状态 |
| ② | `ProcessGroupCollection` | `process_groups_config.py` | **显式传递**的 dataclass | 把组作为参数注入(`pg_collection=`) |
| ③ | `HyperCommGrid` | `hyper_comm_grid.py` | **N 维网格**对象 | 最新、最干净的统一抽象 |

三者并存,语义等价(`HyperCommGrid` 的 docstring 明确给出与 `initialize_model_parallel` 的等价示例)。

### 0.3 记号约定

| 符号 | 含义 |
|------|------|
| `world_size` | 总 GPU 数 |
| `tp/cp/ep/dp/pp` | 五个并行轴的并行度 |
| `order` | 轴布局字符串,默认 `"tp-cp-ep-dp-pp"` |
| `global_rank` | GPU 的全局一维编号 |
| stride | 某轴在 `global_rank` 分解式里的步长(prefix product) |

---

## 1. 动机:5 个并行轴怎么落到物理 GPU 上

### 1.1 一张卡的 5 维坐标

`world_size` 个 GPU 要同时承载 5 个正交的并行轴。把它们看成一个 5 维超长方体(hyperrectangle):

```
world_size = tp · cp · ep · dp · pp
每张 GPU ←→ 5 维坐标 (tp_rank, cp_rank, ep_rank, dp_rank, pp_rank)
```

"正交"的含义:沿任一轴移动,其余 4 维坐标不变。于是:
- **TP 组** = 固定 `(cp,ep,dp,pp)`、遍历 `tp_rank` 的那组卡。
- **DP 组** = 固定 `(tp,cp,ep,pp)`、遍历 `dp_rank` 的那组卡。
- ……每个轴一组,共 5 类组(及各种组合组,如 `dp_cp`、`tp_ep`)。

### 1.2 `order` 字符串:决定物理布局

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

## 2. 核心机制:正交 rank 分组

### 2.1 `RankGenerator` + `order`

`RankGenerator`(`parallel_state.py:444`)是编排的核心类。构造时拿到各轴大小和 `order`:

```python
class RankGenerator:
    def __init__(self, tp, ep, dp, pp, cp, order, rank_offset=0):
        assert ep == 1 or cp == 1, \
            "Both EP and CP > 1 not allowed in one rank generator."   # ← 见 §3
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

### 2.2 `generate_masked_orthogonal_rank_groups` 的数学

这个函数(`parallel_state.py:250`)是整套编排的数学内核。给定 n 维布局 + 一个 mask(哪些维"沿之分组"),求出所有正交 rank 组:

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

### 2.3 具体例子(`world_size=8`,`order="tp-dp-pp"`,`tp=dp=pp=2`)

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

`get_ranks` 产出的三类组(已用 §2.2 的公式核验):

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

## 3. MoE Parallel Folding 的编排实现:两个 RankGenerator

`RankGenerator.__init__` 第一行断言 `ep == 1 or cp == 1` —— **EP 和 CP 不能在同一个 RankGenerator 里同时 > 1**。原因:attention 层用 CP、MoE 层用 EP,二者是**同一组 GPU 的两套不同分解**(即 `ep_analysis.md` §6 的 MoE Parallel Folding)。

`initialize_model_parallel`(`parallel_state.py:545`)因此构造**两个** RankGenerator:

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

PP 是两套分解唯一共享的轴(模型按层切,attention 层和 MoE 层在同一条流水线上)。这就是编排层如何实现"attention 和 MoE 用不同并行度"的 —— `ep_analysis.md` §6 描述的能力,落地点就在这两个 RankGenerator。

---

## 4. 三层抽象逐个看

### ① `parallel_state` —— 全局单例(经典)

`initialize_model_parallel(...)` 一次性把所有组建好,存进模块级全局变量(`_TENSOR_MODEL_PARALLEL_GROUP` 等)。之后任意代码用无参 getter 取:

```python
parallel_state.initialize_model_parallel(tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4, context_parallel_size=2, order="tp-cp-ep-dp-pp")
...
tp_group = parallel_state.get_tensor_model_parallel_group()    # 全局取,无需传参
```

`parallel_state.py` 提供几十个 `get_*` / `is_*` / `get_*_rank` / `get_*_world_size`,以及 PP 专用的 `is_pipeline_first/last_stage`、`get_pipeline_model_parallel_next/prev_rank`(PP 文档 P2P 通信的环形邻居就来自这里)。

- **优点**:简单,任意深度的代码不必层层传参。
- **缺点**:全局可变状态;一个进程只能有一套并行配置;多模型/测试不友好。

### ② `ProcessGroupCollection` —— 显式传递

`process_groups_config.py:27`。一个 dataclass,把所有组作为**显式字段**装在一起:

```python
@dataclass
class ProcessGroupCollection:
    tp / pp / mp / cp / dp / dp_cp / ep / expt_tp / tp_ep / embd / pos_embd / ...
```

它就是前五份文档里到处出现的 `pg_collection=` 参数。代码不再调全局 getter,而是接收一个 `pg_collection` 对象 —— 即依赖注入。`use_mpu_process_groups()`(`:169`)能从全局 `parallel_state` 抽出一个 `ProcessGroupCollection`(向后兼容);`setup_process_groups_for_ddp`(`:454`)/ `setup_process_groups_for_optimizer`(`:263`)按 DDP / 优化器的需要裁出子集。

`MultiModuleProcessGroupCollection`(`:585`)是它的多模块版本:多模态模型(视觉编码器 + 语言模型)每个子模块一套 `ProcessGroupCollection`,用 `language_model_module_name` 标出主干 —— 这是 PP 文档提到的"多模块 PP"的进程组基础(详见后续 PP 补遗文档)。

- **优点**:无全局状态;一个进程可并存多套配置;可测试。
- **缺点**:要层层传 `pg_collection`(故大量函数签名里都有这个可选参数)。

### ③ `HyperCommGrid` —— N 维网格(最新)

`hyper_comm_grid.py:33`。把"5 维超长方体"这个概念**直接对象化**:

```python
grid = HyperCommGrid([2, 3, 4, 5], ["tp", "cp", "pp", "dp"])   # shape + 维名
dp_group    = grid.create_pg("dp")                              # 按维名建组
dp_cp_group = grid.create_pg(["cp", "dp"], pg_options=..., group_desc="...")
```

`dim_names` 的顺序等价于 `initialize_model_parallel` 的 `order`(docstring 明确写出等价关系)。`create_pg(dims)` 对任意维组合建组,`get_pg(dims)` 取回。它把 `RankGenerator + order 字符串 + generate_masked_orthogonal_rank_groups` 那套机制收敛成一个干净的 N 维 API,且支持 `rank_offset`(网格不占满整个 world 时)。

- **优点**:任意 N 维、任意维组合;无全局状态;API 最清晰。
- **定位**:Megatron 进程组管理的演进方向。

---

## 5. 组合规则与约束

### 5.1 维度整除约束

```
world_size 必须能被 model_size 整除:
  model_size = tp · pp · cp                 (attention 侧)
  data_parallel_size = world_size / model_size

MoE 侧:
  expert_model_size = etp · ep · pp
  expert_data_parallel_size = world_size / expert_model_size
```

各轴还有自身约束(散见前五份文档):`tp | num_heads`、`ep | num_experts`、`pp | num_layers`、`cp | seq_len`、VPP 需 `pp·vp | num_layers`、`m % N` 约束(VPP)等。

### 5.2 `order` 的选择

- 默认 `"tp-cp-ep-dp-pp"`:TP 最内(NVLink)、PP 最外(IB)。绝大多数场景直接用默认。
- 非默认 order 有一条断言(`parallel_state.py:802`):不以 `pp` 结尾时,attention 与 MoE 的 DP 大小必须相同 —— 否则 folding 的 PP 组对不齐。

### 5.3 一张卡的归属总览

```
给定 order="tp-cp-ep-dp-pp",一张 global_rank 的卡同时属于:
  TP 组   ← tensor_parallel.py 的 ColumnParallel/RowParallel all-reduce 域
  CP 组   ← attention 的 KV 搬运域(cp_analysis.md)
  EP 组   ← MoE dispatch/combine 的 A2A 域(ep_analysis.md)—— 走 expert RankGenerator
  DP 组   ← 梯度 all-reduce / ZeRO 分片域(ddp_optimizer_analysis.md)
  PP 组   ← 流水线 P2P 邻居(pp_schedulers_analysis.md)
  + 组合组:dp_cp(梯度规约含 CP)、tp_ep(MoE AllGather dispatcher 域)、
            embd(PP 首尾共享 embedding)、mp(tp+pp,模型并行整体)…
```

---

## 6. 与前五份文档的衔接

| 文档 | 它假设的"几何",由本文哪个机制提供 |
|------|-----------------------------------|
| `tp_analysis.md` | `tp_group` ← decoder RankGenerator 的 `get_ranks("tp")`,`order` 第一位保证机内 |
| `pp_schedulers_analysis.md` | `pp_group`、`get_pipeline_model_parallel_next/prev_rank` ← `get_ranks("pp")` |
| `cp_analysis.md` | `cp_group` ← decoder RankGenerator(`ep=1`)的 `get_ranks("cp")` |
| `ep_analysis.md` | `ep_group`/`tp_ep` ← **expert** RankGenerator(`cp=1`);MoE Parallel Folding = §3 的双 generator |
| `ddp_optimizer_analysis.md` | `dp`/`dp_cp` 组、HSDP 的内外层 ← `get_ranks("dp")` / `setup_process_groups_for_ddp` |

### 6.1 一句话总结

- **编排层的工作**:把 `world_size` 个一维 `global_rank` 解释成 5 维坐标,据此为每个轴建 `ProcessGroup`。
- **`order` 字符串**决定物理布局:第一个轴步长 1(相邻、机内、NVLink),最后一个轴步长最大(跨机、IB)—— 这是"TP 内 PP 外"所有建议的量化根源。
- **正交分组数学**(`generate_masked_orthogonal_rank_groups`):mask 选中的维做"组内坐标",其余做"组编号",`global_rank = Σ idx·stride` 拼回。
- **两个 RankGenerator**(decoder 含 CP、expert 含 EP)= MoE Parallel Folding 的编排落地;唯一硬约束是两者 PP 组相等。
- **三层抽象**:`parallel_state` 全局单例(经典)→ `ProcessGroupCollection` 显式注入(前五份文档的 `pg_collection`)→ `HyperCommGrid` N 维网格(演进方向)。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。本文是"第一层补遗"3 份文档之①(并行编排 capstone),后续:② PP 补遗、③ TP·FSDP·resharding 补遗。*

## Related Pages

- [[pp_schedulers_analysis]] · [[ep_analysis]] · [[tp_analysis]] · [[cp_analysis]] · [[ddp_optimizer_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
