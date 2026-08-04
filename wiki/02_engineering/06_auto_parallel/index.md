# 自动并行 — 目录索引

> 大模型分布式训练的自动并行策略搜索：搜索空间、代价模型、搜索算法
> 最后更新: 2026-06-22

---

## 罗盘：一句话定位

自动并行 = 给定**模型计算图 + 集群拓扑**，自动求解"算子怎么切 / 切到哪些设备 / 按什么顺序跑"，使吞吐最大且不 OOM。难点是搜索空间组合爆炸（$\sim 10^{125}$），核心套路是 **策略表示 → 代价模型 → 搜索算法 → 运行时执行**。

---

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[auto_parallel_survey_analysis]] | 公开论文/文档综述 | 业界全景：7 大技术谱系、4 个建模维度（搜索空间/代价模型/拓扑/目标）、5 类搜索算法、2024–2026 趋势 |

> 后续按系统拆专页：`alpa_analysis`(ILP+DP)、`nnscaler_analysis`(原语+约束)、`galvatron_analysis`(显存感知 DP)、`gspmd_sharding_propagation_analysis`(传播范式)、`pytorch_dtensor_autoparallel_analysis`(框架原生)。

---

## 关键概念速览

| 维度 | 内容 |
|------|------|
| **两层并行** | intra-operator（张量切分：DP/TP/SP/EP/分片）+ inter-operator（图切分：PP + 放置）|
| **代价三件** | 计算时间 / 通信时间(α-β + 拓扑) / 显存(参数+梯度+优化器+激活) |
| **搜索五法** | 精确(ILP/DP/MILP) · 元启发(MCMC/MCTS) · 贪心传播 · 分解剪枝 · 模拟器纳入搜索闭环 |
| **标杆系统** | Alpa(分层) · nnScaler(原语+约束) · Galvatron(显存感知) · GSPMD(传播) · DTensor(框架原生) |

---

## 关联域

- [[../../01_theory/06_distributed_parallelism/index]] — **原理层**：自动并行搜索的对象——DP/TP/SP/CP/EP/PP/ZeRO 各维的原理、α-β 代价模型与显存账本（搜索空间与代价函数的概念来源）
- [[../02_train_frameworks/index]] — 训练框架（Megatron-LM 手工并行 / torchtitan DTensor，自动并行的对照与执行后端）
- [[../01_ai_frameworks/index]] — AI 框架（PyTorch 编译栈 / MindSpore 自动并行）

## Related Pages

- [[../index]] — 工程实现知识地图
- [[../../changelog]] — 变更日志
