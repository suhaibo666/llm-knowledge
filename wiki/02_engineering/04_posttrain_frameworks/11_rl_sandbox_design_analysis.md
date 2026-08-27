---
title: "Coding RL Sandbox 设计 — 分析"
---

# Coding RL Sandbox 设计 — 分析

**领域**: 后训练框架 / RL 基础设施
**主题**: 生产级 coding RL 训练的 sandbox 体系设计
**关键资料**:

- *RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure*（2025-12）
- *ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents*（NVIDIA NeMo Gym, 2026-03）
- Anthropic Claude 系列基础设施公开材料
- 姚顺宇 张小珺访谈第 140 期（2026-03）
- Kimi K3 Technical Report `0797decb`（§2.1 补充，引自 [[01_posttraining_infra_mechanism_analysis|D05]] 证据基线）

**入库日期**: 2026-05-24（2026-07-31 补 §2.1 K3 案例，kb-reorg P5 D05 §7 回流）

---

## 1. 量级与挑战

### 1.1 真实并发数字

生产级 coding RL 训练的并发量级：

- **同步 RL**：单次训练 step 通常 128 ~ 几千个并发 environment
- **异步 RL**：几万到十万级
- **大规模训练**：peak 时可能有 **10 万 + 并发 sandbox 同时跑**

每个 sandbox 都要：

- 独立的文件系统
- 独立的进程空间
- 独立的网络命名空间
- 可清理可重置
- 可观测可审计

如果只把这理解为执行一次 `docker run`，系统将无法支撑 RL coding 训练。

### 1.2 Sandbox 没做好的真实失败模式

| 坑 | 后果 |
|----|------|
| 容器启动慢（拉镜像 + cold start 几十秒） | rollout 大部分时间在等启动，GPU 全空转 |
| 没有网络隔离 | agent 直接 curl 答案、或把训练数据外泄 |
| 资源限制不严 | 一个 `while True: fork()` 把整个 worker 拖死 |
| 状态没重置干净 | 上一个 trajectory 的文件污染下一个 |
| 隔离不够（共享 kernel 漏洞） | agent 在沙箱里挖矿、攻击 host |
| 测试套件超时没处理 | 一个 trajectory 卡 1 小时，整个 batch 等它 |
| 镜像层太厚 | 几 GB 镜像，几千 worker 同时拉就把网络打爆 |
| 没有 trajectory 级日志 | 出了问题完全不知道哪一步 hack 了 |

姚顺宇：「很多研究瓶颈本质源于未发现的 bug，修复 bug 比新奇技巧带来的进展更大」——上面每一条都能让训练发散，但都不在 paper 里。

---

## 2. 硬性需求清单

生产级 RL coding sandbox 必须满足：

1. **强隔离**：容器 / microVM 级，agent 之间互不影响、agent 不能影响 host
2. **快速冷启**：目标 < 1 秒，理想 < 100ms
3. **快速重置**：rollout 结束 → 干净状态，不能有遗留
4. **网络管控**：默认 deny-all，按任务白名单放行
5. **资源配额**：CPU、内存、磁盘、进程数、文件描述符全部硬限
6. **可观测**：trajectory 级别记录 syscall / 文件改动 / 网络请求
7. **可弹性扩缩**：从几百到几万实例无感切换
8. **故障容错**：单个 sandbox 挂掉不能影响其他 / 不能丢 trajectory

> 这些需求和 [[31_reward_hacking_defense_analysis]] 中 Layer 1 的「环境加固」直接对接——sandbox 决定了攻击面大小。强隔离 + 测试文件不可写 + 默认 deny-all 网络，是 hack 路径不存在的物理保证。

### 2.1 K3 案例：harness 版本化与故障恢复语义

Kimi K3 的 white-box environment 把 harness 本身也版本化：tools、system prompt、context management、skills、memories 和 subagents 都是可组合模块，训练时会动态构造不同 scaffold；AET 则把 public/hidden verifier、提交预算和最终 environment state 纳入 reward contract（Kimi K3 Technical Report §4.2.1、§4.2.6，pp.14–16）。这是「可观测」需求在 harness 层面的具体落地——不止记录 syscall/文件/网络，连 agent 用的工具集本身也要能溯源版本。

当 reward judge 可能产生副作用时，K3/AgentENV 的 `Fork` 语义比"复制日志后评分"更强：它从相同 microVM state 派生 judge sandbox，同时让原环境继续运行。报告还区分 `Pause/Resume` 与 `Snapshot`，分别处理 inference 等待和故障恢复（Kimi K3 Technical Report §5.3.2，p.22）。三者对应「故障容错」需求里三种不同的失败/暂停场景，不能用单一"支持续跑"布尔值代替。

机制在三平面模型中的位置见 [[01_posttraining_infra_mechanism_analysis|D05]] 第 7 节、第 8.1 节；K3 完整案例见 [[24_kimi_k3_posttraining_case_study_analysis|D12]]。

---

## 3. 关键技术选型对比

| 方案 | 隔离强度 | 冷启时间 | 性能开销 | 适用场景 |
|------|---------|---------|---------|---------|
| 普通 Docker | 中（共享 kernel） | 几秒 | 低 | 内部信任环境 |
| gVisor | 强（用户态 kernel） | 几秒 | 中（syscall 慢 2-5x） | 不信任代码、需强隔离 |
| **Firecracker microVM** | **极强**（独立 kernel） | **100~300ms** | 低 | **AWS Lambda 同款，生产首选** |
| Kata Containers | 极强 | 几秒 | 中 | K8s 友好 |
| Bare 容器 + seccomp | 弱 | 极快 | 极低 | 完全可信代码 |

Anthropic、OpenAI 这类规模的系统，基本都采用 Firecracker 或类似的 microVM 方案。这类方案可以同时满足隔离性、冷启动时延和性能需求。

---

## 4. Disaggregated 架构（2025 RL Infra 共识）

2025 年所有严谨的 RL infra 研究都收敛到同一类架构：**rollout 和 training 物理分离**。

```
┌─────────────────────┐
│   Training Cluster  │  H100/H200 + NVLink
│   (gradient update) │  纯训练
└──────────┬──────────┘
           │ 权重同步（几秒一次）
           ▼
┌─────────────────────┐
│  Inference Cluster  │  H20 / A100，吃 KV cache
│  (policy serving)   │  对外暴露 inference API
└──────────┬──────────┘
           │ action 调用
           ▼
┌─────────────────────┐
│   Sandbox Cluster   │  纯 CPU，K8s + Firecracker
│  (code execution)   │  几万 worker
└─────────────────────┘
```

### 4.1 为什么这么拆：三类负载的 hardware affinity 完全不同

| 组件 | 主要负载 | 硬件偏好 | 成本 |
|------|---------|---------|------|
| Training | 大 batch 的 forward + backward | 高显存 + 高带宽互联 | 极贵 |
| Policy inference | KV cache + decode | 显存带宽 | 中 |
| Sandbox 执行 | CPU 跑 pytest / npm test | CPU + I/O | 便宜 |

绑在一起的话，最慢/最贵的那个就是瓶颈。拆开后每一块可以独立 scale，每一块用最合适的硬件。

> 详细的 throughput 数学和 utilization 分析见 [[12_rl_infra_efficiency_analysis]] 第 3 节。

---

## 5. Rollout 三阶段拆解

ProRL Agent 论文（NVIDIA NeMo Gym）里把每个 rollout 切成三个独立调度的阶段：

```
[Phase 1: Init]      [Phase 2: Execution]       [Phase 3: Evaluation]
container 启动  →    LLM 推理 + tool 调用   →    跑测试 / 算 reward
I/O 密集型           GPU 推理密集型              CPU 密集型
100ms - 几秒         秒 - 分钟                   几 ms - 几分钟
```

三个阶段的**资源诉求和时延特征完全不同**：

- **Phase 1**：I/O 密集，可大批量并行预启
- **Phase 2**：GPU 推理密集，需要调度到 inference cluster
- **Phase 3**：CPU 密集，**方差极大**——trivial 任务 100ms 跑完、跑全量 CI 套件要 5 分钟

现代设计让三个阶段独立排队、独立调度。Phase 3 的高方差是产生长尾的主要原因，直接促成了 [[12_rl_infra_efficiency_analysis]] 中的长尾治理手段（redundant rollouts、trajectory-level scheduling、早停、严格 timeout）。

---

## 6. 未来趋势

**1. Sandbox-as-a-Service 成为标准基建**

Daytona、Northflank、NVIDIA NeMo Gym 这些已经在做。未来大概率会有 1-2 家成为这个层的「AWS」。自己搭 sandbox infra 的小公司会越来越少。

**2. Computer Use / Browser Use 推动新一代 sandbox**

光跑代码不够了。Anthropic 的 computer use、OpenAI 的 Operator 这类 agent 需要的 sandbox 是**完整桌面环境**——X server、浏览器、文件管理器、邮件客户端。这是比 coding sandbox 复杂一个量级的工程。

**3. 跨进程 / 跨服务的真实环境**

Coding 真实场景不只是单进程：数据库、消息队列、微服务、API 依赖。下一代 sandbox 会是 **mini-Kubernetes inside a sandbox**，能拉起整套微服务再让 agent 操作。

**4. 性能进一步压缩**

Firecracker 已经把冷启压到 100ms 级。未来可能用 unikernel、WebAssembly 这类方案压到 10ms 级，让 sandbox 成本进一步下降。

**5. 安全成为不可妥协的红线**

随着 agent 真的开始操作真实代码和系统，「agent 在沙箱里挖矿」、「agent 偷数据」这类事件会出现。Sandbox 安全会从「工程问题」上升到「合规问题」。

---

## 7. 与其他「脏活」的关系

本页是 coding LLM 训练「三块脏活」分析系列之一：

| 脏活 | 决定的事 | 对应页 |
|------|---------|--------|
| Reward Hacking 防御 | 训出来的是不是你想要的 | [[31_reward_hacking_defense_analysis]] |
| Sandbox 设计 | 训练能不能稳定跑 | **本页** |
| RL Infra 效率 | 训练能跑多大、多快 | [[12_rl_infra_efficiency_analysis]] |

Sandbox 既是 reward hacking 防御 Layer 1 的物理基础（强隔离 → hack 路径不存在），也是 RL infra disaggregated 架构的核心组件（独立的 cluster + 三阶段调度）。

---

## Related Pages

- [[24_agentic_rl_algorithm_analysis]] — Agentic trajectory、reward event、failure 与 coding sandbox schema
- [[01_posttraining_infra_mechanism_analysis]] — sandbox 在后训练数据面、故障域和 backpressure 中的位置
- [[24_kimi_k3_posttraining_case_study_analysis]] — §2.1 K3 harness 版本化与 Fork/Pause/Snapshot 语义的完整案例
- [[courses/posttraining_frontier]] — 后训练前沿阅读课程(原 D00–D12 学习域已解散,内容归位至功能树)
- [[31_reward_hacking_defense_analysis]] — 同系列，sandbox 是 Layer 1 的工程基础
- [[12_rl_infra_efficiency_analysis]] — 同系列，sandbox 三阶段拆解直接驱动长尾治理
- [[20_batch_invariance_guide]] — 后训练框架已有页面，训练批次不变性
- [[20_grpo_analysis]] — GRPO 训练的 rollout 上下文
- [[10_rl_ppo_loss_and_grpo_analysis]] — PPO / GRPO 的源码级训练流程
- [[02_engineering/04_posttrain_frameworks/index]] — 后训练框架入口
