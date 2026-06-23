# MindFormers MoE 专家并行 — 知识地图

> **代码基线**:MindFormers `master` @ `01e71622`(2026-06-18)
> **最后更新**:2026-06-23(新建子目录 + 收纳 PyNative EP / Graph 去冗余 dispatcher 两篇)
> 华为 MindFormers 训练框架中 **MoE 专家并行(EP)** 的源码级分析。覆盖**两条代码路径**——PyNative(动态图)与 Graph(静态图 `parallel_core/training_graph`)——的 token dispatch / combine 机制、去冗余与零冗余优化、以及各方案的通信量代数。

---

## 设计主线:驯服 MoE 的 all-to-all

MoE dispatch 的物理事实只有两条:**token 散在各 rank、专家也散在各 rank**,路由是乱序多对多 —— 天生一次 all-to-all。MindFormers 的全部巧思都在**减少一个 token 在链路上被重复搬运的次数**,并把不规则、需 D2H 的通信关进快的机内 NVLink。一句话谱系:

> **基础 alltoall(逐 token-expert 对,带冗余)→ 零冗余(按目标 rank 去重)→ 去冗余(两级 oep/iep,跨机每 token 恰 1 次)**,三者针对不同瓶颈、互相正交。

## 两条代码路径(必须先分清)

| dispatcher | PyNative(`mindformers/pynative/`) | Graph(`parallel_core/training_graph/`) |
|---|---|---|
| `alltoall`(基础/带冗余) | `ExpertParallel` | `MoEAlltoAllTokenDispatcher` |
| `alltoall_zero_redundancy`(零冗余) | ❌ **不存在** | `MoEAlltoAllZeroRedundancyTokenDispatcher` |
| `alltoall_deredundancy`(去冗余) | `DeredundancyExpertParallel` | `MoEAlltoAllDeredundencyTokenDispatcher` |
| 通信重叠 | `OverlapExpertParallel`(A/B/C/D 钩子 + 异步 a2a) | (图执行器 Depend/overlap) |

> [!important] PyNative 有 deredundancy,**没有** zero_redundancy;零冗余只在 Graph 路径。详见各页。

## 文档系列(2 篇)

| 页面 | 路径 | 核心机制 |
|------|------|---------|
| [[mindformers_pynative_ep_analysis]] | **PyNative** | EP 三方案与通信量对照:`alltoall`(逐 token-expert 对,`_build_resort_index` host 端重建索引省一次 a2a)、`zero_redundancy`(仅 Graph,按 rank 去重)、`alltoall_deredundancy`(两级 oep/iep,跨机量与 k 无关)+ `OverlapExpertParallel` 通算重叠;含三方案通信量总对照表 |
| [[mindformers_moe_token_dispatcher_analysis]] | **Graph** | 去冗余 token dispatcher 逐算子图解(7 图):oep/iep 两级 EP(跨机 AllGather + 机内 AlltoAllV)、mask+NonZero 去冗余、计数转置与 D2H overlap(Depend)、ReduceScatter top-k 求和、梯度反向 adjoint |

## 三方案通信量速览

记号:`T`=每 rank 本地 token、`H`=hidden、`E`=专家数、`k`=top_k、`EP`=ep_degree。

| | 链路上的重复 | 与 k 的关系 | 主集合通信 | 攻的瓶颈 |
|---|---|---|---|---|
| `alltoall` | 每目标**专家** 1 次(≤k) | 线性 ∝ k | all-to-all×2 | 通用 / 实现最简 |
| `zero_redundancy`(仅 Graph) | 每目标 **rank** 1 次 | 次线性 | AllToAll + AllGather + AllToAllV | 本地专家聚集(EP 小 / E 大) |
| `deredundancy` | 跨机每方向**恰 1 次** | 跨机与 k 无关 | 跨机 AllGather/ReduceScatter + 机内 AlltoAllV | 多机带宽不对称(`EP≥iep`) |

## Cross-Domain Links

- [[megatron_ep_analysis]] —— Megatron-LM 的 EP / token dispatcher 源码级分析(CUDA 生态对照)
- [[torchtitan_ep_analysis]] —— torchtitan EP(AllToAll / DeepEP 路径,PyTorch-native 对照)
- [[deepseek_moe_analysis]] —— DeepSeek MoE 路由与共享专家(k、E 的来源)
- [[comm_compute_overlap_analysis]] —— 计算通信掩盖对比(含 DeepEP/HybridEP)

## Related Pages

- [[mindformers_pynative_ep_analysis]] · [[mindformers_moe_token_dispatcher_analysis]]
- [[megatron-lm/index]] · [[torchtitan/index]] —— 姊妹训练框架知识地图
- [[02_engineering/02_train_frameworks/index]] —— 训练框架目录索引
