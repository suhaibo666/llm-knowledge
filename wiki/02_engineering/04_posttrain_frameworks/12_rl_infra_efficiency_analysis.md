# Coding RL Infra 效率优化 — 分析

**领域**: 后训练框架 / RL 基础设施
**主题**: Coding RL 训练 GPU 利用率与端到端 throughput 优化
**关键资料**:

- *RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure*（2025-12）
- *ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents*（NVIDIA NeMo Gym, 2026-03）
- *RollPacker: Mitigating Long-Tail Rollouts*（2025-09）
- Sholto Douglas & Trenton Bricken on Claude 4（Dwarkesh Patel 访谈, 2025-05）
- 姚顺宇 张小珺访谈第 140 期（2026-03）
- AReaL `areal/infra/staleness_manager.py`、Kimi K3 Technical Report `0797decb`（§2「优化 6」补充，引自 [[01_posttraining_infra_mechanism_analysis|D05]] 证据基线）

**入库日期**: 2026-05-24（2026-07-31 补「优化 6」，kb-reorg P5 D05 §4 回流）

---

## 1. 核心矛盾

整个 RL coding 训练的效率问题，本质就是一句话：

> **LLM 推理（每张卡几万美元）不能停在那里等 sandbox（每核几美分）的代码跑完。**

但偏偏：

- LLM 推理产生一个 action（一段代码 + 一个 tool call）很快
- Sandbox 执行那个 action（跑测试、装依赖、tool 响应）很慢
- 而且 sandbox 时间方差极大（100ms 到几分钟，参见 [[11_rl_sandbox_design_analysis]] 第 5 节）

这就是为什么 naive 的同步 RL 训练 GPU 利用率经常只有 20-30%——大部分时间都在等 sandbox 回来。

### 1.1 典型 GPU 利用率数据

| 配置 | GPU 利用率 |
|------|-----------|
| Naive 同步 PPO，本地 sandbox | 15-25% |
| Disaggregated 同步 RL | 50-60% |
| Disaggregated 异步 RL + 长尾治理 | 80-90% |
| 加上 hardware-aware scheduling | 90%+ |

从 20% 到 90%，**4 倍的算力效率差**——这就是为什么 RL infra 是真正的护城河。

---

## 2. 核心优化

### 优化 1: 异步训练（Async RL）

**问题**：同步训练里，rollout → reward → update → 权重同步 → 下一轮 rollout 是串行的，training GPU 大量空转。

**做法**：让 training 和 rollout 在不同的 GPU 上**同时**跑。

```
同步训练:
[Rollout]→[Reward]→[Update]→[Sync]→[Rollout]→[Reward]→...
（GPU 利用率低，因为各阶段排队）

异步训练:
Training GPU:   [Update][Update][Update][Update]...
Inference GPU:  [Rollout][Rollout][Rollout][Rollout]...
（两边并行，互不等待）
```

**代价**：rollout 用的 policy 权重落后 training 1~N 步（off-policy staleness）。

**为什么可以接受**：大量实验证明在合理 staleness（一般 1~4 步）下，模型最终质量几乎不掉，但端到端 throughput 翻倍以上。这是个**用一点点 sample efficiency 换 wall-clock 效率**的好买卖。

Claude 这个量级的训练，几乎可以确定走的就是异步。

> **原理**：PPO/GRPO 这类算法本来就是 off-policy 友好的（有 importance sampling 做修正），几步的 staleness 在数学上等价于很轻微的探索噪声。Sholto Douglas 在 Claude 4 访谈中：**RL 的主要作用是激活已有知识并组织成解决方案，不是从零学**。所以策略稍微旧一点，激活效果不会差太多。算法基础见 [[11_ppo_analysis]] / [[20_grpo_analysis]]。

### 优化 2: 长尾治理（Long-tail Mitigation）

**问题**：同步 batch 训练里，90% 的 rollout 几秒钟完成，剩下 10% 跑几分钟。整个 batch 时间 = 最慢那个。

#### A. Redundant Rollouts（超额发射）

发 N+K 个 rollout，前 N 个完成就 cut，丢弃慢的：

```
需要 128 条 trajectory → 发 160 条
等前 128 条回来 → 立刻继续训练
后面 32 条丢弃（cost 浪费 20%，但 wall-clock 时间砍掉 50%+）
```

CPU sandbox 便宜、GPU 训练贵，这个 trade-off 大部分时候是赚的。

#### B. Trajectory-level Scheduling

不要把 batch 当原子单位。每条 trajectory 独立调度，一条完成立刻发下一条进 sandbox。这是 RollArt / RollPacker 等论文的核心思想。

#### C. 早停启发式

如果 trajectory 已经明显跑偏（比如中期 reward 信号判定希望渺茫、token 数超阈值、明显在死循环），直接砍掉，不浪费 sandbox 资源。

#### D. Timeout 严格执行

任何 trajectory 都有硬超时（比如 10 分钟）。超时直接终止 + 给负 reward。

> **数学直觉**：trajectory 时间通常服从重尾分布，假设 p90 = 5s, p99 = 60s。等 100% 完成平均 batch 时间 ≈ 60s；只等 90%，平均 ≈ 5s——**wall-clock 砍掉 12 倍**，浪费的 sandbox 成本忽略不计（CPU 便宜）。redundant rollouts 是**几乎免费的 12x 加速**。

### 优化 3: 硬件感知调度（Hardware-aware Scheduling）

RollArt 论文里有个挺精细的发现：**同一个 rollout 里不同 trajectory 的 inference 特征不同，应该路由到不同的卡**。

| Trajectory 类型 | 特征 | 适合的卡 |
|----------------|------|---------|
| 长 prompt + 短输出（读大代码库判断 bug） | Prefill 重 | H800（compute 强） |
| 短 prompt + 长输出（生成新功能模块） | Decode 重 | H20（带宽够、便宜） |
| 多轮 tool call（agent 来回循环） | 频繁 KV cache 操作 | 大显存卡 |

Reward 模型这种**无状态**组件可以塞到 serverless / function compute 上，按需扩缩，不占常驻 GPU。

> **原理**：不同卡的性价比和性能特征本来就不同——Prefill 阶段算力受限 → 用 H800/H100；decode 阶段带宽受限 → 用 H20。**让每张卡跑它擅长的活**，整体成本/性能可以优化 30%+。

### 优化 4: In-flight Reward Computation（流水线 reward）

**传统做法**：trajectory 跑完 → 喂给 reward model → 算 advantage → 训练。reward 计算是阻塞的。

**优化做法**：

- Trajectory **一边跑一边**回流部分 reward 信号（编译过了、测试 1/5 通过了、lint 干净了）
- Trajectory 一结束**立刻 async 触发 reward 计算**，不阻塞下一轮 rollout
- Reward 计算本身可以拆成多个 stage（语法 → 测试 → judge LLM）并行做

实测能把 reward 计算的时间几乎完全 overlap 掉，端到端时间砍掉 20-30%。

### 优化 5: 环境多样性（容易被忽略）

这不是个纯 infra 优化，但效率上下文里必须提：**环境池太小会让 RL 训练失效**。

姚顺宇访谈里提到的「成功路径」里「反馈信号清晰、数据干净」——后半句的实质是 **environment 池要够大、够多样**。

| 环境池规模 | 后果 |
|-----------|------|
| 100 个 | 模型迅速过拟合，包括过拟合到 hack 路径 |
| 1000 个 | 略好，但 generalization 弱 |
| 1 万 + | 开始有真正的 coding 能力涌现 |
| 10 万 + | 接近 Claude 3.7 这个量级 |

每个 environment 还要有大量随机化（commit SHA、依赖版本、初始状态、目标描述措辞）。Anthropic 在 Claude 3.7 → 4 阶段的一个关键工程突破，据说就是把 environment 池从千级别拉到了**数十万级别**。

构建这种规模的高质量 environment 池，是个**数据工程问题**而不是算法问题。可能比设计算法本身耗费更多人力。

> 环境多样性同时是 [[31_reward_hacking_defense_analysis]] Layer 1 的关键——多样化环境让 hack 路径无法在所有任务上通用，模型必须真的学解题。

### 优化 6: Admission-aware Backpressure（准入与容量管理）

前五个优化解决"怎么让 GPU 别等"，这一个解决"buffer 该在什么时候拒绝新样本"——顺着 [[01_posttraining_infra_mechanism_analysis|D05]] 三平面模型的 data plane 接口定义（一个可恢复 buffer 至少要知道 producer role、sample policy version、group/trajectory completeness、consumer reservation、accepted/rejected state、retry count、checkpoint watermark），准入控制至少要卡四层容量：

1. **并发容量**：同时执行多少 rollout；
2. **staleness 容量**：最多预生成多少未来 batch；
3. **内存容量**：object store、host RAM、NVMe 和 network queue；
4. **状态容量**：GPU KV、CPU external KV、KDA recurrent state 和可暂停 sandbox。

只限制 queue 长度会让短样本挤占版本预算；只限制版本会在慢 verifier 下耗尽内存。AReaL 的 `StalenessManager` 同时取 concurrency 和 staleness capacity 的最小值，是一个清楚的参考实现，见 `areal/infra/staleness_manager.py:80-112`。

K3 给出 cache-pressure-aware admission 的另一种信号组合：active request count、queued request count 和 KV utilization 共同调节送入 inference engine 的请求数。早期 context 短时提高并发，轨迹变长、KV 压力升高时自动收紧，而不是用固定"平均完整轨迹长度"静态限流（Kimi K3 Technical Report §5.3.1，p.21；详见 [[24_kimi_k3_posttraining_case_study_analysis|D12]]）。

---

## 3. Disaggregation 带来的算力账

### 3.1 数学

假设：

- Training 占总时间 30%（用 GPU）
- Inference 占 40%（用 GPU）
- Sandbox 占 30%（用 CPU）

**绑在一起跑**：你要按峰值买 GPU，但 sandbox 阶段 GPU 全空转 → GPU 利用率上限 70%

**Disaggregated**：

- Training cluster 100% 跑 training
- Inference cluster 100% 跑 inference
- Sandbox cluster 100% 跑 CPU 活
- 每一块都接近 100% 利用率，整体 throughput 翻倍以上

这就是为什么大厂都在这么做——节省的 GPU 时间是天文数字。详细架构图见 [[11_rl_sandbox_design_analysis]] 第 4 节。

### 3.2 三阶段独立调度的 throughput 增益

考虑一条 trajectory 的生命周期：

```
   Phase 1     Phase 2         Phase 3
   (init)      (exec)          (eval)
   [---]       [---------]     [-------------]
```

如果按 batch 同步处理，整个 batch 的时间 = max(每条 trajectory 总时间)。一个慢的 trajectory 拖死整个 batch。

三阶段独立调度：

```
Trajectory A:  [Init][======Exec=======][===Eval===]
Trajectory B:       [Init][====Exec====][Eval]
Trajectory C:              [Init][==Exec=][======Eval======]
                                         ↑
                                  A 已经进入下一轮
```

每个 phase 完成就立刻调度下一个任务，整体 throughput 接近所有 phase 并行最大值。

---

## 4. 未来趋势

**1. On-policy 的回归？**

异步训练用 staleness 换效率，但近期一些工作（DeepSeek、月之暗面）显示，更严格的 on-policy（同步或者只允许 1 步 staleness）+ 更激进的算法（GRPO、DAPO）反而能拿到更好的最终质量。未来可能是**两条路线并行**：超大规模训练走异步、追求 SOTA 走严格 on-policy。参见 [[21_dapo_analysis]] / [[20_grpo_analysis]] / [[22_gspo_analysis]]。

**2. Continuous Training（持续训练）**

模型上线后，把真实用户的 trajectory 持续灌回训练池。Claude Code 已经有这个潜力——用户付费用它写代码，每条 session 都是潜在的训练数据（前提是隐私和版权处理好）。这会形成真正的数据飞轮。

**3. Inference-aware Training**

训练时考虑推理成本。比如训练一个模型不只是产出正确答案，还产出**短**、**少 tool call**、**低延迟**的答案。这是个新方向，会让「模型质量」从单维变成多维优化。

**4. Heterogeneous RL（异构 RL）**

不再是「一个 policy 模型 + 一个 reward 模型」，而是几十个不同的判断模块（语法 judge、安全 judge、风格 judge、效率 judge）异步打分。Claude 内部据说已经在做这种 multi-objective RL。

**5. World Model 进入 coding？**

未来可能不直接在真实 sandbox 跑代码，而是用一个学到的「代码世界模型」来 simulate 执行结果——sandbox 慢的部分被 model 取代。这是个高风险高回报的方向，目前还没看到 work 得很好的案例。

---

## 5. 整体行业判断

### 5.1 为什么 coding 是 AI 第一个真正起飞的领域

回到姚顺宇访谈里的洞察：**coding 不是因为最重要才起飞的，是因为它是少数同时满足三个条件的任务**：

1. **Reward 信号天然清晰**（测试过/不过、编译过/不过）——让 reward 设计相对容易，hack 也容易被发现
2. **Environment 可大规模合成**（GitHub 上几亿个 repo + 测试）——让 sandbox 内容不愁
3. **执行环境标准化**（Docker / Python 解释器 / Node 运行时）——让 sandbox 工程化可行

数学、竞赛题也满足前两条，但只是单轮、不需要复杂工具使用。Agentic coding 是**第一个**三个条件都满足的真实生产任务。

而像写作、客服、产品决策这些场景，至少缺一条（reward 信号天然不清晰），所以现在还卡在那里。

### 5.2 三块「脏活」的相互约束

```
        ┌───────────────────┐
        │  Reward Hacking   │
        │     防御          │
        └────────┬──────────┘
                 │ 决定「训出来的是什么」
                 │
        ┌────────▼──────────┐
        │   Sandbox 设计    │
        │                   │←┐
        └────────┬──────────┘ │
                 │            │ 决定「能不能稳定跑」
                 │            │
        ┌────────▼──────────┐ │
        │  RL Infra 效率    ├─┘
        │                   │
        └───────────────────┘
        决定「能跑多大、多久」
```

一句话总结：

> **Sandbox 决定能不能跑得稳；RL Infra 决定能不能跑得多、跑得快；Reward Hacking 防御决定跑出来的东西到底是不是你想要的。**

任何一块短板都会成为整体瓶颈：

- Sandbox 不行 → 训练 epoch 时间过长，迭代太慢，team 失去 momentum（[[11_rl_sandbox_design_analysis]]）
- Infra 不行 → 同样算力别人能跑 10x 数据，你只能跑 1x，能力差距迅速拉开（本页）
- Reward hacking 防御不行 → 表面 benchmark 涨了，实际部署翻车，用户失去信任（[[31_reward_hacking_defense_analysis]]）

### 5.3 对 Coding 大模型方向的判断

1. **模型层短期内不会同质化**：SWE-bench 这种基准上三家头部模型分数接近，但**实际 agentic coding 体验差异巨大**。这种差异主要来自三件事：reward 信号干净度、environment 池规模和质量、长 trajectory 训练经验。这些都不是几个月能追赶的
2. **中国玩家的真实差距在 infra 而不是算法**：国内常说「算法追上来了」——纸面上是。但 RL infra（disaggregated 架构、长尾治理、稳定异步训练）和 reward hacking 体系，国内大厂普遍还在补课阶段。这是真正的差距，而且**没有 shortcut 可以走**
3. **Long horizon + Multi-modal 是下一个战场**：让模型用有限 context 做近似无限长任务、把 coding 能力扩展到 computer use、机器人控制等其他工具使用场景
4. **数据飞轮终于在 coding 形成**：Claude Code、Cursor 这些产品的真实用户使用，正在反哺训练数据。**第一次**在 LLM 历史上出现真正的「用得越多越好用」的飞轮——之前 chatbot 没形成这个，因为 chat 反馈信号太弱
5. **行业进入「工程比拼」阶段**：algorithm 创新的边际收益在递减，工程能力（infra、数据、environment、反馈系统）的边际收益在递增。这正好是姚顺宇说的「AI 这行不需要脑子，需要靠谱」的现实含义

---

## 6. 与其他「脏活」的关系

本页是 coding LLM 训练「三块脏活」分析系列之一：

| 脏活 | 决定的事 | 对应页 |
|------|---------|--------|
| Reward Hacking 防御 | 训出来的是不是你想要的 | [[31_reward_hacking_defense_analysis]] |
| Sandbox 设计 | 训练能不能稳定跑 | [[11_rl_sandbox_design_analysis]] |
| RL Infra 效率 | 训练能跑多大、多快 | **本页** |

---

## Related Pages

- [[01_posttraining_infra_mechanism_analysis]] — control/data/weight 三平面与工业正确性不变量；§4 backpressure 接口定义的落地页即本页「优化 6」
- [[24_kimi_k3_posttraining_case_study_analysis]] — 「优化 6」cache-pressure-aware admission 的完整案例
- [[25_on_policy_off_policy_staleness_analysis]] — async、staleness、off-policy 与 TIM 的严格区分
- [[courses/posttraining_frontier]] — 后训练前沿阅读课程(原 D00–D12 学习域已解散,内容归位至功能树)
- [[31_reward_hacking_defense_analysis]] — 同系列，environment 多样性是 hack 防御 Layer 1
- [[11_rl_sandbox_design_analysis]] — 同系列，disaggregated 架构的物理基础
- [[20_grpo_analysis]] — GRPO 算法基础，off-policy 友好
- [[11_ppo_analysis]] — PPO Loss 与 importance sampling
- [[21_dapo_analysis]] — DAPO 的动态采样与长尾治理思路
- [[22_gspo_analysis]] — 序列级 importance ratio，影响 staleness 容忍度
- [[29_kimi_k1_5_analysis]] — Kimi 长上下文 RL 训练
- [[10_rl_ppo_loss_and_grpo_analysis]] — PPO/GRPO 源码级实现
- [[batch_invariance_guide]] — 后训练框架数值稳定性
- [[02_engineering/04_posttrain_frameworks/index]] — 后训练框架入口
