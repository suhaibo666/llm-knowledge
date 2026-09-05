---
title: "Megatron-LM 并行几何快速入门：从 global rank 到进程组"
---

# Megatron-LM 并行几何快速入门：从 global rank 到进程组

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **学习前置**：先走通 [[02_megatron_training_quickstart]]；只要求理解 `torchrun` 会启动多个 rank。
> **回答的问题**：给定 world size 和 TP/PP/CP/EP 配置，怎样判断配置是否合法，并读懂一个 rank 属于哪些组？
> **不覆盖**：本页不解释分组算法、全局状态或 MoE folding；这些属于 [[17_megatron_parallelism_orchestration_analysis]]。
> **最后复核**：2026-09-03。

---

## 1. 先把“八张卡”改写成坐标

分布式启动只给每个进程一个一维 `global_rank`，并行算法却会说“沿 TP 轴切张量”“在 DP 组归约梯度”“把 PP 相邻 stage 连接起来”。如果直接把每种 group 当成独立名单记忆，组合一变就要重算全部列表。

Megatron 选择的办法是：先用各轴大小和 `order` 把一维 rank 解释成坐标，再用 mask 取出某个轴或轴组合。`RankGenerator` 保存 TP/EP/DP/PP/CP 大小，检查非 1 的轴都出现在 `order` 中，然后把 token mask 交给同一个正交分组函数（`megatron/core/parallel_state.py:444-514`）。被否掉的替代是为 TP、DP、PP 以及每种组合分别维护手写循环；统一坐标模型的判据是新增一个组合只需改变 mask，而不重写布局算术。

## 2. 第一道检查：并行度是否能整除 world size

对 attention/dense decoder 路径，Megatron 先计算：

```text
model_size = tensor_parallel_size × pipeline_parallel_size × context_parallel_size
data_parallel_size = world_size / model_size
```

`world_size` 不能被 `model_size` 整除就直接报错（`megatron/core/parallel_state.py:726-737`）。因此 DP 通常不是独立再乘进去的输入，而是使用 world size 与模型并行度推导出的剩余维度。

最小训练页的配置是 world size=2、TP=2、PP=1、CP=1，所以 DP=1（`examples/run_simple_mcore_train_loop.py:207-221`）。两个 rank 共同切一个模型，但没有第二份 data-parallel replica。脚本仍包装 DDP，是为了给训练主线提供统一的 `finish_grad_sync()` 接口，而不是说明该例发生了跨 rank 的 DP 梯度通信（`:215-221`）。

## 3. 第二道检查：`order` 决定哪一维变化最快

正交分组函数在 docstring 中给出三轴例子。若顺序为 `tp-dp-pp`，则：

```text
global_rank = tp_rank + dp_rank × tp_size + pp_rank × tp_size × dp_size
```

源码随后用 prefix product 构造 stride，并从一维 index 反解坐标（`megatron/core/parallel_state.py:250-330`）。因此 `order` 最前面的轴 stride 最小，同一组 rank 更接近；初始化文档也明确要求调用者尽量让相邻 rank 位于同一 DGX 节点（`:682-696`）。源码直接保证的是 rank 布局，不保证集群调度器一定把这些 rank 放在 NVLink 域；物理映射仍是启动配置的责任。

### 一个可以手算的 8-rank 例子

取 world size=8、TP=2、PP=2、CP=1，则 DP=2。使用 `order="tp-dp-pp"`：

| 组 | rank 列表 | 固定的坐标 |
|---|---|---|
| TP | `[0,1] [2,3] [4,5] [6,7]` | 固定 DP、PP，遍历 TP |
| DP | `[0,2] [1,3] [4,6] [5,7]` | 固定 TP、PP，遍历 DP |
| PP | `[0,4] [1,5] [2,6] [3,7]` | 固定 TP、DP，遍历 PP |

这与源码给出的 16-GPU、TP=2、PP=4 示例采用同一种坐标关系；源码列出了完整的 TP、PP、DP 组作为初始化契约（`megatron/core/parallel_state.py:682-692`）。

## 4. “一张卡属于多个组”不等于复制多份进程

rank 1 在上例中同时属于 TP `[0,1]`、DP `[1,3]` 和 PP `[1,5]`。这是同一个进程持有三个 `ProcessGroup` handle；沿不同轴执行 collective 时选择不同 handle。group 的交集描述的是一张卡在训练状态机中的多种职责，而不是创建三个 worker。

`RankGenerator.get_ranks("tp-dp")` 会把组合 token 转成 mask，返回同时遍历 TP 和 DP、固定其他坐标的组（`megatron/core/parallel_state.py:488-519`）。具体的 process group 创建、缓存和显式注入由 17 号页负责，本页只要求能读懂名单。

## 5. MoE 为什么需要第二套坐标分解

一个 `RankGenerator` 明确禁止 EP>1 与 CP>1 同时出现：CP 只在默认 generator，EP 只在 expert generator（`megatron/core/parallel_state.py:447-453`）。初始化先为 attention/dense 层创建 `ep=1` 的 decoder generator，再用 expert tensor parallel、expert parallel 和 expert data parallel 创建第二个 generator（`:769-800`）。两者必须得到相同 PP groups，否则断言失败（`:802-811`）。

expert 侧必须独立做一次整除检查：未显式设置 ETP 时先令 `ETP=TP`，随后计算 `expert_model_size = ETP × EP × PP` 和 `EDP = world_size / expert_model_size`；若 world size 不能被该乘积整除，初始化直接抛 `RuntimeError`（`megatron/core/parallel_state.py:780-789`）。expert generator 固定 `CP=1`（`:791-800`）。此外，两套分解不仅要各自整除，还必须枚举出完全相同的 PP groups；当 `order` 不以 `pp` 结尾且 PP>1 时，源码还要求 `EDP=DP`（`:802-811`）。因此一个配置通过 dense 侧的 `TP × PP × CP` 检查，并不代表它也通过 expert 侧检查。

所以“每张卡有一个简单的五维坐标”只能作为入门直觉，不能拿来推导 MoE folding 的真实 DP/EP 关系。进入 MoE 时，应分别问：dense 路径的 DP/CP 坐标是什么，expert 路径的 EDP/EP 坐标是什么，以及二者在哪个 PP stage 对齐。完整实现见 [[17_megatron_parallelism_orchestration_analysis|并行编排中的 dense/expert 分组与 PP 对齐]]。

## 6. 配置前的三步心算

1. 先算 `model_size = TP × PP × CP`，确认 world size 可整除，并得到 dense DP。
2. 按 `order` 从左到右写 prefix stride，判断哪些 rank 在物理编号上相邻。
3. 若 EP>1，先取默认或显式 ETP，再检查 `world_size % (ETP × EP × PP) == 0` 并算出 EDP；不要把 EP 和 CP 强塞进同一 generator，最后核对两侧 PP groups 一致，以及非 `pp`-last 布局下的 `EDP=DP` 条件。

完成本页后，读者应能解释：为什么最小训练的两张卡是一个 TP 组、为什么 DP=1、为什么 DDP wrapper 仍然存在，以及 MoE 场景为何不能只用一个五维乘积。

## Related Pages

- [[01_megatron_architecture_analysis]] —— 给并行几何在完整训练状态机中的位置。
- [[02_megatron_training_quickstart]] —— 用 TP=2、PP=1 的真实脚本验证本页心智模型。
- [[12_megatron_tp_analysis]] —— 深入同一 TP 组内张量的切分与 collective。
- [[13_megatron_cp_analysis]] —— 深入 CP 组如何沿序列维度交换上下文。
- [[14_megatron_ep_analysis]] —— 深入 expert generator 消费的 EP/ETP/EDP 语义。
- [[17_megatron_parallelism_orchestration_analysis]] —— `RankGenerator`、group creation 与多套抽象的实现权威页。
