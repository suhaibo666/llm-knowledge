# OPD 基础设施机制分析：训练回路、信号带宽账与八项工程工作

> **定位**：本页回答"如果要在自建集群上跑 on-policy distillation（OPD），基础设施要多建什么、多花什么、卡在哪里"。算法层的散度之争与目标函数演化见 [[15_opd_divergence_and_objective_evolution_analysis]]，厂商格局见 [[32_opd_industrial_landscape_analysis]]，框架逐项对照与选型见 [[32_opd_framework_support_comparison]]。
> **最后更新: 2026-08-11**
> **保真度约定**（与知识库其余 OPD 页一致）：正文默认为一手核实内容（技术报告原文指定章节 / 框架官方文档 URL）；**⚠️** = 仅有二手来源或本轮未能独立核实；**【推断】** = 本页作者的分析推断，非来源声称；**【本文推算】** = 本页作者的算术推算，**不是文献给出的数字**。§9 单列本页相对上游综述稿的三条独立核验更正。

---

## 1. 一句话机制：OPD 是"RL 的回路，加一个新角色"

OPD 的系统形态可以用一句话概括：**学生 rollout、损失计算、训练↔推理权重同步这三个环节与 PPO/GRPO 完全同构，唯一新增的角色是教师**——一个只做 prefill 打分、不做 decode 的推理服务。

这个同构性不是巧合，而是算法形态的直接后果：OPD 在形式上是 KL 约束 RL 的稠密奖励特例，把 per-token advantage 设为 $-\mathrm{KL}$ 即可在任意 RL 框架上实现（MiniLLM 把 RKL 写成策略梯度，arXiv:2306.08543 Eq. 2；TML 博客 Pseudocode 节令 per-token advantage $=-\mathrm{RKL}$ 并直接复用 RL 的 importance-sampling 损失，https://thinkingmachines.ai/blog/on-policy-distillation/ ；arXiv:2604.00626 将其形式化为 KL-constrained RL 特例）。**这也是 OPD 能在一年内被全部主流 RL 框架内建的根本原因**——它对框架的最小改动量是"替换 advantage 来源"。

但新增的这一个角色带来了三个方向上的压力集中：**带宽/存储**（教师信号最高可达全词表 logits）、**调度**（万亿参数教师、10+ 个教师、百万 token 轨迹）、**一致性**（教师打分必须与当前学生策略同版本）。本页的其余部分就是把这三条压力拆开。

---

## 2. 与 SFT / RL 的系统需求对照

### 2.1 组件对照表

【推断】下表为本页作者对三种后训练范式系统需求的并排归纳；各组件的存在性均可在 [[32_opd_framework_support_comparison]] 的框架实现中逐项印证。

| 系统组件 | SFT | RL（PPO/GRPO） | OPD |
|---|---|---|---|
| 学生训练引擎（FSDP / Megatron） | 需要 | 需要 | 需要 |
| 学生 rollout 推理引擎（vLLM / SGLang） | — | 需要 | 需要 |
| 训练 ↔ 推理权重同步 | — | 需要 | 需要 |
| 奖励/监督来源 | 静态标签 | RM / verifier（序列级标量） | **教师模型逐 token 分布** |
| Critic / GAE | — | PPO 需要；GRPO 用组内基线 | **不需要**（advantage 直接来自 KL） |
| 新增重资产 | — | RM（通常 $\le$ 策略规模） | **教师：可以比学生大、可以有 10+ 个** |

### 2.2 三个关键差异

**(1) 教师是 prefill-only 负载。** 教师不生成，只对学生已采出的轨迹做一次前向取 logprob/logits——无自回归 decode、可大 batch、可廉价并行。TML 博客"按 GPU 时算成本降 18×"的论证正建立在 "this computation can be cheaply parallelized across GPUs" 之上（TML 博客 Distillation for reasoning 节）。**系统含义**：教师服务的优化目标与在线服务（decode 为主）完全相反——要的是 prefill 吞吐、prefix caching 与批调度，而不是首 token 延迟；容量规划应按 prefill FLOPs 而非并发会话数来做。

**(2) 信号密度是可选项，且每档差一个量级以上。** 这是 OPD infra 独有的设计自由度——SFT 的监督密度由标签固定，RL 的奖励密度由 verifier 固定，只有 OPD 可以在"全词表 / top-k / 隐藏状态 / 单采样 token"四档之间自由选择（§4）。这个自由度同时是负担：**它把一个算法选择变成了 infra 的第一设计决策**。

**(3) Critic 的消失是省出来的预算。** 相对 PPO，OPD 省掉 critic 的显存占用与其自身的训练开销；相对 GRPO，逐 token advantage 不再依赖组内多样本估计——GLM-5 原文："it is no longer necessary to maintain a large group of samples per prompt to estimate advantages"（arXiv:2602.15763, §3.5；实测配置 group size $=1$、batch 1024）。**rollout 采样量直接除以组大小**，这是一笔常被忽略的算力节省，且它正好抵消（部分抵消）教师前向新增的开销——做 OPD 的 TCO 对比时若只加教师、不减 group，会系统性高估 OPD 成本【推断】。

### 2.3 上游综述给出的三条增量需求

Song & Zheng 综述（arXiv:2604.00626v4, §8.3）把 OPD 相对纯 SFT/RL 管线的增量系统需求收敛为三条，可作为 §6 八项清单的顶层框架：

1. **teacher co-hosting**——教师须常驻训练全程，而非一次性产完数据就下线；显存需求按教师规模与教师数量膨胀。
2. **logit-tensor transfer**——传的是词表分布而不是标量奖励，跨节点带宽压力高出数量级（§4 量化）。
3. **staleness tolerance**——教师对学生 rollout 的打分必须与当前学生策略一致；**OPD 能容忍的异步窗口比 reward-model RLHF 更窄**。

第三条最容易被低估，它直接约束 W3 异步流水线能激进到什么程度，§7 单列展开其机制。

---

## 3. 训练回路分解

### 3.1 数据流的四个阶段

与 GRPO 回路的**唯一拓扑差异**是：Reward Model / verifier 被教师打分服务替换——或者叠加（MiMo 的 $\hat{A}_{\mathrm{MOPD}}=\mathrm{sg}[\log(\pi_{domain}/\pi_\theta)]+\alpha\hat{A}_{ORM}$，arXiv:2601.02780 §4.4 Eq. 9；veRL 的 `use_task_rewards` 支持 KL 与任务奖励并存，https://verl.readthedocs.io/en/latest/algo/opd.html ）。

| 阶段 | 引擎 | 负载性质 | 产出 | 下游消费者 |
|---|---|---|---|---|
| ① 学生 rollout | vLLM / SGLang | 自回归 decode，KV cache 增长，**内存受限**；通常主导 wall-clock（生成在 token 维不可并行） | 轨迹 token 序列 + **采样时 logprob** $\mu_\theta$ | ②③ |
| ② 教师打分 | 独立推理服务 / 训练框架内共置前向 | prefill 前向，**算力受限**；教师远大于学生时可媲美甚至超过学生反传的成本 | logits / top-k / 单 token logprob / 隐藏状态 | ③ |
| ③ 学生更新 | FSDP / Megatron | 反传 + 优化器态，三者中通常最便宜但**决定显存下界** | 新权重 | ④ |
| ④ 权重同步 | colocate 或 disaggregate | 带宽突发 | rollout 引擎新权重 | ① |

（三段负载的性质刻画出自 arXiv:2604.00626 §6.3/§8.3。）

**多教师场景下②多出一层路由**：按样本 metadata 选教师——veRL 支持按 `data_source` 等字段路由（https://verl.readthedocs.io/en/latest/algo/opd.html ）；Kimi K3 按（域, effort）二维路由到九个专家教师之一（arXiv:2607.24653, §4.1.3）。

**阶段①产出的"采样时 logprob"必须作为一等公民随轨迹落盘**——它是后续所有重要性采样修正（W3）与训推一致性监控（W5）的唯一输入。框架若不保留它，off-policyness 修正无从谈起。

### 3.2 两条算法路线的系统含义

主稿 §2.5 把工业实现分裂为两条路线；从 infra 视角看，这两条路线的教师服务、传输格式、损失算子**完全不同**。

| | 路线 A：全词表 / top-k logit 反传 | 路线 B：采样 token KL-as-reward |
|---|---|---|
| 教师需返回 | 每位置完整（或 top-k）分布 | 仅采样 token 的 logprob（每位置一个标量） |
| 带宽 / 存储 | 高（§4） | 极低（§4） |
| 损失计算位置 | 训练引擎内做逐位置散度 | 复用 RL 的 advantage 通道 |
| 对 RL 框架的改动 | 需新增蒸馏损失分支 | **近零**（替换 advantage 即可） |
| 梯度性质 | 有偏（对采样分布不作修正）、低方差 | 无偏、高方差（arXiv:2603.25562 给出定量刻画：token 级 OPD 相对序列级 reverse-KL 有偏但方差界更紧） |
| 生产代表 | Qwen3（arXiv:2505.09388 §4.5）、DeepSeek-V4（arXiv:2606.19348 §5.1.2） | GLM-5（arXiv:2602.15763 §3.5）、MiMo-V2-Flash（arXiv:2601.02780 §4.4）、Kimi K3（arXiv:2607.24653 §4.1.3）、Nemotron-Cascade 2（arXiv:2603.19220 §4.4） |

两家头部厂商给出了**互相矛盾的一手经验报告**，这是选型时必须知道的事实：

- DeepSeek-V4 对路线 B 的批评（arXiv:2606.19348, §5.1.2，**逐字**）："it leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD."
- Kimi K3 的反证（arXiv:2607.24653, §4.1.3，**逐字**）："we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting."；其坚持路线 B 的系统理由同样逐字给出："This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks."

**Infra 视角的重述**【推断】：路线之争在系统层是一笔**带宽换方差**的交易。DeepSeek-V4 为了买"低方差"，付出了全词表信号的带宽与一整套教师调度基建（W2）；K3/Nemotron 为了"近零基建改动 + 复用 partial rollout 等既有优化"，改用截断去压方差。因此 **infra 团队的第一个动作应该是问算法团队站哪条路线**——不要期望一套实现通吃（veRL 是目前唯一双路线都实现的开源框架，见 [[32_opd_framework_support_comparison]]）。K3 的那句逐字引文尤其值得注意：它说明**路线 B 的真正卖点不是省带宽，而是"能继承 RL 框架里已有的一切长程优化"**——省带宽只是副产品。

---

## 4. 信号格式的带宽账【本文推算】

> **重要标注**：本节的四档数字是**本页作者按典型配置做的算术推算，不是任何文献报告的实测值**。文献侧只有 §4.3 引用的两个独立算例可用作交叉验证。引用本表时请连同"【本文推算】"一并引用。

### 4.1 四档格式的量级

设词表 $|V|\approx 128\mathrm{K}$、序列长 $L=16\mathrm{K}$ token、教师隐藏维 $d\approx 7\mathrm{K}$、bf16 存储（logprob 按 fp32 计），**单条轨迹、单教师**：

$$S_{\text{full}}=|V|\cdot L\cdot 2\mathrm{B},\quad S_{\text{top-}k}=k\cdot L\cdot 8\mathrm{B},\quad S_{\text{hidden}}=d\cdot L\cdot 2\mathrm{B},\quad S_{\text{token}}=L\cdot 4\mathrm{B}$$

| 信号格式 | 体积 / 轨迹 | 相对量级 | 采用者 |
|---|---|---|---|
| 全词表 logits | $\approx 4.2\ \mathrm{GB}$ | $1\times$ | DeepSeek-V4（配专用教师调度与内核，arXiv:2606.19348 §5.2.2） |
| top-$k$（$k=64$，id + logprob） | $\approx 8.4\ \mathrm{MB}$ | $\sim 1/500$ | veRL `forward_kl_topk`（https://verl.readthedocs.io/en/latest/algo/opd.html ）；Gemini 2.5 的 k-sparse 预计算存储（arXiv:2507.06261 p.3） |
| 教师隐藏状态（学生端重算 logits） | $\approx 0.23\ \mathrm{GB}$ | $\sim 1/18$ | KDFlow（共享内存零拷贝、数学等价，arXiv:2603.01875）；DeepSeek-V4 缓存教师 hidden states ⚠️（Labonne 二手分析，与报告 §5.2.2 相容） |
| 仅采样 token logprob | $\approx 64\ \mathrm{KB}$ | $\sim 1/65000$ | slime `k1`/`k3`、GLM-5、MiMo、K3、Nemotron-Cascade 2 |

**四档之间跨越约 4.8 个数量级**（$4.2\ \mathrm{GB}$ vs $64\ \mathrm{KB}$）。这不是"优化 20%"级别的差异，而是决定了架构形态本身。

### 4.2 放大到生产规模

batch 1024 条轨迹的全词表信号约 $4\ \mathrm{TB}/\text{步}$【本文推算】。**这就是为什么"全词表"必然伴随 DeepSeek-V4 式的专用基建**——没有教师权重 offload 调度、隐藏状态缓冲与即时重建 logits 这一整套设计，全词表路线在工程上不可行。KDFlow 论文的立论方式相同：其配置下全量 logits 达 $\sim160\ \mathrm{GB}$ 级，改传隐藏状态后对 TRL / MS-SWIFT / ROLL 提速 $1.44\text{–}6.36\times$（arXiv:2603.01875）。

### 4.3 独立算例交叉验证与"单机可行 / 跨节点不可行"分界

Song & Zheng（arXiv:2604.00626, §8.3）给出一个独立算例：70B 教师 / 7B 学生、$8\times$H100、$B=16$、$T=4096$、$|V|=128\mathrm{K}$、BF16 $\rightarrow$ 约 $16\ \mathrm{GB}/\text{batch}$；结论是**单节点 NVLink（900 GB/s）内可行，跨节点则必须 top-k 稀疏化或精细张量切分**。

【本文推算】按 §4.1 的公式复算该配置：$16\times 4096\times 131072\times 2\ \mathrm{B}\approx 17.2\ \mathrm{GB}$——与其报告的 $\sim16\ \mathrm{GB}$ 同数量级且量纲自洽，说明两组数字互为旁证。**其真正价值是给出了一条实用分界线**：全词表 OPD 的可行域边界基本落在"教师打分是否与训练同节点"这件事上。跨节点即意味着必须降档（top-k / 隐藏状态 / 采样 token），或者付出 DeepSeek-V4 级别的自研代价。

---

## 5. 成本模型与 $\rho$ 旋钮

### 5.1 公式

$$C_{off}\approx N\,(F_{T}+F_{S}+B_{S}),\qquad C_{on}\approx N\,(G_{S}+\rho F_{T}+F_{S}+B_{S})$$

其中 $F/B$ 为前向/反向 FLOPs（下标 $T$/$S$ 为教师/学生），$G_S$ 为学生自回归生成成本，$\rho\in(0,1]$ 为**教师监督刷新率**（arXiv:2604.00626, §7.4/§8.3）。

### 5.2 两个可读出的结论

**(1) $G_S\gg F_S$ 是 on-policy 溢价的主体。** on-policy 的成本大头不是教师，而是"学生必须自己生成"这件事——教师那一项只是 $\rho F_T$。这解释了为什么 §2.2(3) 里 GLM-5 把 group size 降到 1 是一笔大账：它直接砍 $G_S$ 前面的系数。

**(2) $\rho$ 是 infra 能直接优化的唯一旋钮。** 离线缓存、教师打分复用、前缀截断，本质上都是在压 $\rho$。上游综述转述了两个把这条轴推到极限的方法（**经 arXiv:2604.00626 §6.3 转述，原文均未打开** ⚠️）：

- **FOPD 前缀截断**——蒸馏信号（按逐 token reverse-KL 度量）集中于序列前缀（学生最弱的是前段的高层规划决策，后段被前缀上下文约束、边际信号递减）；只对前缀 $k$ 做 on-policy 蒸馏、其余回退 off-policy KD，**FLOP 降 $2\times$–$47\times$** 而质量持平，配渐进前缀调度（每步 $+256$ token）。
- **Lightning-OPD 离线教师缓存**——在"教师一致性"条件下（SFT 产轨迹与 OPD 参考分布用同一教师）预计算教师 logprob，训练期完全不需要在线教师服务；可证与在线 OPD 同最优点，效率 $4\times$。

两者共同说明一件对 infra 极重要的事：**on-policy 不是二值属性，而是可以按 token 位置、按刷新率 $\rho$ 折价购买的程度量**。这是目前最值得跟进的降本方向，也是 W6 存储层设计的前提。

### 5.3 三段负载配三套并行策略

已是通行做法（arXiv:2604.00626 §8.3 对 veRL/OpenRLHF 模式的概括）：教师推理用 TP（4–8 卡压延迟）、学生生成用 PP（吞吐）、优化器用 ZeRO-3（显存），三者分进程组编排。**这意味着 OPD 集群的资源单元不是"一种 worker"而是三种**，容量规划、故障域、弹性伸缩都要按三类分别做。

---

## 6. Infra 团队的八项工作清单（W1–W8）

每项给出：**问题 → 生产做法与一手出处 → 落地要点**。

### W1 教师打分服务化（第一优先级）

- **问题**：教师前向不能挤占训练引擎的显存与算力；教师可能比学生大、架构异构、甚至不止一个。
- **生产做法**：生态已收敛出三条教师取数路径（主稿 §6.2 归纳）：
  - **(a) 独立推理服务返回 logprob / top-k**——veRL 用独立教师资源池（`distillation.nnodes`）+ ZMQ top-k logprob 服务（`recipe/gkd/teacher/start_server.sh`，https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html ）；slime `--opd-type sglang` 在 rollout 阶段经 API 取 token 级 logprob，官方文档明确该模式适合"教师过大或架构异构"（https://thudm.github.io/slime/zh/advanced/on-policy-distillation.html ）；Tinker 把它抽象成 `compute_logprobs` API（https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation ）。
  - **(b) 共置训练框架内前向**——TRL（https://huggingface.co/docs/trl/gkd_trainer ）、slime `--opd-type megatron`、NeMo-RL（DTensor 路径，https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html ）。
  - **(c) 传隐藏状态**——KDFlow：SGLang 教师 + FSDP2 学生解耦，共享内存零拷贝传 hidden states、学生端重算 logits（arXiv:2603.01875，https://github.com/songmzhang/KDFlow ）。
- **落地要点**：① 教师服务按 **prefill-only** 负载做容量规划（大 batch、prefix caching、无 KV 增量），不要复用在线服务的 SLO 模板；② 打分请求天然可与下一轮 rollout 重叠（W3）；③ **接口层面尽早定死 top-k 传输格式**（id + logprob 的紧凑编码）——这是后续所有带宽优化的边界，改起来最贵。

### W2 多教师权重管理与调度

- **问题**：2026 年的主流用法是多教师（主稿 P4）——DeepSeek-V4 "more than ten teacher models"（arXiv:2606.19348 §5.1.1/§5.2.2）、Kimi K3 九教师（3 域 $\times$ 3 effort，arXiv:2607.24653 §4.1.2）、MiMo/Nemotron 十余专家（arXiv:2601.02780 §4.1；arXiv:2606.15007 §3.3）。**教师总参数量可以远超集群显存**。教师本身如何生产出来（五类来源与成本结构）见主稿 §5.12。
- **生产做法**：DeepSeek-V4 §5.2.2 标题即 "Efficient Teacher Scheduling for Full-Vocabulary OPD"，支持 "effectively unbounded number of teachers, each potentially comprising trillions of parameters" 的教师权重 offload 调度。实现细节由两个独立来源交叉印证（Labonne 二手分析 ⚠️；arXiv:2604.00626 §8.3 的展开转述）：
  - **不物化任何教师的全量 $|V|>100\mathrm{K}$ logit 张量**——中心化缓冲只缓存最后一层 hidden states，训练时经对应 prediction head 即时重建 logits（重算成本可忽略）；
  - **minibatch 内按教师身份排序样本**，使任意时刻至多一个教师的 prediction head 驻留显存；
  - 全部权重加载/卸载异步进行、不阻塞关键路径；
  - FP4 QAT 直接集成在 OPD 阶段内（蒸馏出的模型原生适配部署精度）；
  - ⚠️ 自定义 TileLang 精确 KL 内核仅 Labonne 单源提及，未见报告原文佐证。
  - 路由层面：veRL 支持按样本 metadata（如 `data_source`）路由到不同教师；K3 按（域, effort）二维路由（arXiv:2607.24653 §4.1.3）。
- **落地要点**：多教师调度的核心是**把"哪条样本找哪个教师"变成批间局部性问题**——同教师样本聚簇 $\Rightarrow$ 权重驻留时间最大化。教师数量增长时，offload 层级（HBM $\to$ DRAM $\to$ NVMe）与预取策略决定吞吐上限。另需注意一个算法侧的隐藏成本：多教师能力区不重叠时逐 token 信号会相消而非组合（Counteraction-Aware MOPD，经 arXiv:2604.00626 §7.2 转述 ⚠️）——**路由策略同时是一个正确性问题，不只是调度问题**【推断】。

### W3 异步流水线与 off-policyness 治理

- **问题**：rollout、教师打分、训练三段串行会互相空等；重叠执行则破坏严格 on-policy。
- **生产做法**：
  - veRL 异步 recipe——one/two-step-off 调度重叠三段，文档明言以**牺牲严格 on-policy** 为代价；权重同步优化约 $12\times$ 加速（https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html ）。
  - Kimi K3 **partial rollout**——不等全部轨迹完成，$\lambda$ 比例完成即开始优化，未完轨迹入队下轮续跑，"an individual long-horizon trajectory naturally spans multiple iterations"；陈旧性靠 per-token 正则约束在局部邻域内消化（arXiv:2607.24653 §4.1.2）。K3 §4.1.3 逐字点明这正是选路线 B 的系统理由（引文见 §3.2）。
  - Nemotron 3 Ultra 的异步 behavior / proximal 策略解耦（arXiv:2606.15007 §3.3/§3.3.1）——⚠️ **本轮独立核验未能触达**（arXiv HTML 与 ar5iv 均在 §3.3.1 之后截断），该条目前依据的是上游 research-notes 对官方 PDF §3.3.1（p.21）的读取，标为**待核**，不宜作为设计依据的唯一支撑。

#### W3-a 两类截断的作用对象必须分清（本页对上游稿的更正）

上游 infra 综述稿在 W3 中把 MiMo 的截断、K3 的师生 log-ratio clip、Nemotron-Cascade 2 的截断 IS **并列成一串**，这容易造成误读：**它们不是同一类东西**。截断在 OPD 训练回路里至少有两个截然不同的作用对象：

| 类别 | 作用对象（被截断的比值） | 要治的问题 | 一手实例 |
|---|---|---|---|
| **类型 I：训推不一致（TIM）修正** | $w_t=\dfrac{\pi_\theta(y_t)}{\mu_\theta(y_t)}$ —— **训练策略 vs 采样策略**（同一个模型在训练引擎与推理引擎下的两套 logprob，或跨迭代的策略版本差） | rollout 引擎与训练引擎数值口径不同 + 异步带来的版本陈旧 | MiMo-V2-Flash（arXiv:2601.02780 §4.4 Eq. 7-8：$w_t$ 落在 $[\varepsilon_{low},\varepsilon_{high}]$ 区间外**置零**）；Nemotron-Cascade 2（arXiv:2603.19220 §4.4：截断 IS 权重，$\varepsilon_{low}=0.5$、$\varepsilon_{high}=2.0$，报告明确其用途是修 train–inference 失配） |
| **类型 II：师生 advantage 截断** | $\log\dfrac{\pi_{teacher}(y_t)}{\pi_\theta(y_t)}$ —— **教师 vs 学生**，即 OPD 的 advantage 本身 | 极端 advantage 导致的梯度尖峰与不稳定 | Kimi K3（arXiv:2607.24653 §4.1.3 Eq. 15：$\mathrm{clip}(\cdot,-R_{\max},R_{\max})$） |

**为什么必须分清**：类型 I 与 OPD 无关——任何异步 RL 都要做，它属于 [[26_tim_causal_chain_analysis]] 的问题域；类型 II 才是 OPD 特有的。二者作用在损失的不同因子上，可以（且通常应该）**同时存在**，把其中一个当成另一个的替代品会导致"以为已经治了陈旧性、其实只截了 advantage"这类静默错误【推断】。infra 侧的直接含义是：**两条截断需要两组不同的输入数据**——类型 I 需要 rollout 时的 $\mu_\theta$ 落盘，类型 II 只需要教师 logprob。

#### W3-b 关于 "MiMo 用 IcePop 式截断"的表述更正

上游稿（infra 稿 W3、主稿 §9 第 4 条）写作"MiMo 参照 IcePop 截断"。本轮独立核验的结论是：

1. **MiMo-V2-Flash 正文中并未出现 "IcePop" 一词**——该词只出现在参考文献条目的标题里；正文的原始表述是 "Following Zhao et al. (2025)"。因此"MiMo 用 IcePop"是一个**转写产物**，引用时应还原为报告原话或直接引 Eq. 7-8。
2. 更重要的是：**Eq. 8 的截断作用于训推比 $\pi_\theta/\mu_\theta$，不是师生比值**——它是训推不一致（TIM）修正，与 OPD 的师生 KL 是两件不同的事（见上表类型 I / 类型 II）。把它读成"OPD 的散度截断"会把一条通用 RL 稳定化技巧错记成 OPD 算法组件。

- **落地要点**：**框架应把"rollout 时 logprob"作为一等数据随轨迹落盘**，否则类型 I 修正无从谈起；异步窗口的上限不能照抄 RLHF 的经验值（§7）；`advantage` 是否 detach 这类看似细节的问题在生产里会被真实踩到（slime issue #1449，2026-01-18，closed，讨论中直接引 MiMo 技报——可作为该问题真实存在的旁证）。

### W4 长程 agent 场景支持

- **问题**：多轮 agent 轨迹长（K3 已到百万 token 上下文，arXiv:2607.24653）、含环境 token、跨回合误差累积（TCOD 发现的轨迹级 KL 不稳定：KL 上升伴随成功率下降，学生脱离教师支持集，arXiv:2604.24005）。
- **生产做法**：
  - K3 co-located RL 系统 $=$ partial rollout $+$ **外置 KV-cache 保留** $+$ 自适应限流 $+$ 可续跑 microVM 沙箱（arXiv:2607.24653 引言/贡献列表，§4.1.2）。
  - DeepSeek-V4 §5.2.3：可抢占、容错的 rollout 服务**同时服务 RL 与 OPD 两种工作负载**（arXiv:2606.19348）。
  - Tinker harbor 多轮配方在损失中**掩蔽环境 token**——"only the student's generated tokens contribute to the loss"（tinker-cookbook `harbor_multiturn` 系列）。
- **落地要点**：① 环境 token 掩蔽应做进损失算子的 mask 语义，而不是在数据清洗阶段处理——多轮场景下 mask 与 KV-cache 复用是耦合的；② KAT 的"检测到持续低 KL 即提前终止 rollout"（复用已算好的 reverse KL，滑窗 + 自校准阈值，rollout 长度 $-59.7\%$、avg@k $+2.66\%$、零新增损失，arXiv:2606.09471）是**零成本的吞吐优化**，值得直接做成 rollout 引擎的可插拔终止条件；③ 沙箱与容器层的设计与 RL 共用，见 [[11_rl_sandbox_design_analysis]]。

### W5 数值一致性

- **问题**：两类失配。
  - **(a) 训推失配**：rollout 引擎与训练引擎对同一 token 算出的 logprob 有差（kernel、精度、并行切分不同）。路线 B 对此尤其敏感，因为其 advantage 里显式含 $\pi_\theta$——这正是类型 I 截断存在的另一半理由（W3-a）。
  - **(b) 量化一致性**：为部署做 QAT 时，师生前向的数值口径必须对齐。
- **生产做法**：DeepSeek-V4 §5.2.1 在 FP4（MXFP4）QAT 下保证师生一致性（arXiv:2606.19348）；Kimi K3 从 SFT 阶段起全程 QAT——MoE 专家权重 MXFP4、激活 MXFP8、非专家组件保持高精度（arXiv:2607.24653 §4.1.1/§4.1.4）。**QAT 贯穿 OPD 阶段**意味着教师打分也必须在同一量化语义下进行，否则师生 logprob 不可比。
- **落地要点**：把"同一序列在 rollout 引擎 vs 训练引擎的 logprob 差"做成常态化监控指标（W7）；【推断，基于训推失配的一般机理】异构硬件（非 CUDA 生态）上这条差异通常更大，是移植时的第一验证项。机制层面的因果链见 [[26_tim_causal_chain_analysis]]。

### W6 存储与带宽工程

- **问题**：§4 的带宽账。
- **生产做法谱系**（按 $\rho$ 与信号密度从高成本到低成本排列）：
  1. 在线全词表 $+$ 专用调度——DeepSeek-V4（arXiv:2606.19348 §5.2.2）；
  2. 在线 top-k——veRL `forward_kl_topk`（https://verl.readthedocs.io/en/latest/algo/opd.html ）；
  3. 在线隐藏状态 $+$ 学生端重算——KDFlow（$1.44$–$6.36\times$ 提速，arXiv:2603.01875）；
  4. 在线采样 token——slime / GLM-5 / MiMo / K3 / Nemotron-Cascade 2；
  5. **离线预计算 k-sparse 软标签**——Gemini 2.5：教师 next-token 分布以 k-sparse 近似存储，换取吞吐、代价是存储放大（arXiv:2507.06261 p.3）。⚠️【推断】严格说这是 off-policy 预训练蒸馏的方案，但其存储设计对 OPD 的重放/缓存场景同样适用。
  6. 把 $\rho$ 推到极限的两个方法（FOPD 前缀截断、Lightning-OPD 离线教师缓存）见 §5.2，均**经 arXiv:2604.00626 §6.3 转述** ⚠️。
- **落地要点**：教师打分结果的**可缓存性取决于算法路线**——严格 on-policy 下轨迹一次性使用、缓存无益；但 replay 类变体会让缓存变得有价值（DistiLLM 的自适应 off-policy；TML 博客"单 prompt 多 epoch 复用不产生记忆化"的观察；Lightning-OPD 的教师一致性条件 ⚠️）。**存储层设计前先和算法团队确认复用策略**，否则容易建出一套没人用的缓存。

### W7 评测与可观测性

- **问题**：OPD 的失败模式（见 [[33_opd_effectiveness_and_failure_modes_analysis]]）大多能在训练曲线上提前看到，但默认的 RL 指标盘不够——默认盘只有奖励均值、长度、熵，而 OPD 的病征藏在**分布**里。
- **应监控的信号**（每条对应一个已发表的失败模式）：

| 指标 | 对应失败模式 | 出处 |
|---|---|---|
| 逐 token reverse KL 的**分布**（而非仅均值），特别是持续低 KL 段 | KL 同意陷阱（学生漂入不可恢复前缀后教师局部"同意"，低 KL 但无纠正信号） | arXiv:2606.09471 |
| 响应长度与截断率 | 长度膨胀 $\to$ 截断主导 $\to$ 重复饱和 $\to$ 崩溃链条 | arXiv:2604.08527 |
| 按教师熵分层的 KL | 教师高熵位置信号不稳定 | arXiv:2603.07079 |
| 高损失 token 的驻留集合 | Rock tokens（$\sim18\%$ token 持续高损失、功能贡献可忽略，空耗梯度） | arXiv:2605.09253 |
| pass@k 曲线（不止 pass@1） | 多样性坍缩（RKL mode-seeking），比 pass@1 提前暴露 | arXiv:2603.07079 |
| rollout 引擎 vs 训练引擎的 logprob 差 | 训推失配（W5） | 本页 W5【推断】 |

- **落地要点**：这些都是训练框架 callback 级的工作量，**收益/成本比在八项里最高**，且没有任何框架内建（§8 Gap 4）。

### W8 训练-推理权重同步

- **问题与做法**：与 RL 完全同构（colocate 或 disaggregate 两种部署、增量同步、异步窗口），不再展开，见 [[12_rl_infra_efficiency_analysis]] 与 [[01_posttraining_infra_mechanism_analysis]]。veRL 异步 recipe 报告权重同步优化约 $12\times$ 加速（https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html ）。
- **唯一 OPD 特有的注意点**【推断】：**教师权重不参与同步**（frozen），但多教师 offload（W2）与学生权重同步会争抢同一 PCIe / NVLink 带宽，调度上要错峰——这在单机多教师配置下是真实的带宽冲突点。

---

## 7. 为什么 OPD 的 staleness tolerance 比 RLHF 更窄

这是 §2.3 三条增量需求里最容易被低估的一条，也是 W3 的硬约束来源。上游综述的结论是"教师打分必须与当前学生策略一致，异步窗口比 RM-based RLHF 更紧"（arXiv:2604.00626 §8.3）。其机制可以说得更精确【推断】：

- **RLHF 的奖励是轨迹的函数，不是策略的函数**：$r=\mathrm{RM}(x,y)$。学生权重更新后，同一条旧轨迹的奖励值**不变**；陈旧性只污染"这条轨迹来自旧分布"这一件事，可以用重要性采样修正，且修正对象是一个良态的标量权重。
- **OPD 的信号是策略的函数**：路线 B 的 advantage $\hat{A}_t=\mathrm{sg}\big[\log\frac{\pi_T(y_t)}{\pi_\theta(y_t)}\big]$ 里**显式含有 $\pi_\theta$**。学生权重一变，同一条轨迹上同一个 token 的 advantage 值就变了。也就是说，陈旧性在 OPD 里**同时污染采样分布与信号值本身**——前者可用 IS 修正，后者不能，因为"正确值"已经随权重漂走了。
- 路线 A 也不豁免：全词表 KL $\mathrm{KL}(\pi_\theta\|\pi_T)$ 同样以当前 $\pi_\theta$ 为一个自变量，只是它在训练引擎内用最新权重现算，把陈旧性局限在"轨迹是旧的"这一层。

**工程推论**：
1. one/two-step-off 这类在 RLHF 里被验证过的窗口，**不能直接照搬**到 OPD，需要重新标定；veRL 文档自己也明说异步 recipe 是以牺牲严格 on-policy 为代价的。
2. K3 的 partial rollout 之所以能容忍"单条轨迹跨多个迭代"，是因为它另配了 per-token 正则把陈旧性约束在局部邻域内（arXiv:2607.24653 §4.1.2）——**异步窗口与正则强度是一组必须联调的参数，不能各自独立调**【推断】。
3. 这条约束的极限在哪里，目前没有任何公开刻画（§8 开放问题 2）。

---

## 8. 开放系统问题

1. **全词表信号的经济学**：什么规模/场景下值得为低方差支付 §4 的带宽账？K3/Nemotron 的"top-k 无明显优势"经验与 V4 的坚持互相矛盾（主稿 P5）——【推断】本质可能是各家 infra 成熟度不同导致的成本函数不同：对已有教师调度基建的团队，全词表的边际成本远低于从零自建的团队。更一般地，OPD 缺一条含 rollout 预算 $R$ 的 scaling law：$\text{Quality}\propto N_T^\alpha N_S^\beta D^\gamma R^\delta$（arXiv:2604.00626 §2.4/§9），$R$ 是 OPD 独有的 scaling 轴；没有它，infra 的容量规划只能靠经验法则。
2. **partial rollout 的陈旧性上限**：K3 靠 per-token 正则消化"跨多迭代"的轨迹，该机制在 OPD 下（教师信号也随学生分布漂移，§7）的极限未被刻画。
3. **教师打分与学生 rollout 的联合调度**：prefill 密集与 decode 密集两种负载共享集群的最优编排；DeepSeek-V4 §5.2.3 的可抢占方案是目前唯一公开先例。
4. **缓存与复用的收益模型**：TML"多 epoch 复用不产生记忆化"的观察若成立，教师打分缓存的收益模型值得系统研究——它直接决定 W6 该不该建。
5. **万亿教师 $\times$ 长上下文的乘积**：K3 已到百万 token 轨迹；教师对百万 token 序列做全序列打分的显存/并行方案（context parallel prefill）未见公开讨论。
6. **异步窗口的可标定性**：§7 的机制说明 OPD 的 staleness 上限应当是可以从"教师-学生 KL 的漂移速率"推出来的量，但目前没有任何工作给出可测量的判据【推断】。

---

## 9. 本页相对上游综述稿的独立核验更正

| # | 上游稿表述 | 本轮核验结论 | 本页处理 |
|---|---|---|---|
| 1 | "MiMo 参照 **IcePop** 截断（arXiv:2601.02780 Eq. 7-8）"，并与 K3 的师生 log-ratio clip、Nemotron 的截断 IS 并列 | (a) MiMo **正文未出现 "IcePop"**，该词仅见于参考文献标题，正文原话为 "Following Zhao et al. (2025)"；(b) **Eq. 8 的截断作用于训推比 $\pi_\theta/\mu_\theta$**（训练策略 vs 采样策略），属**训推不一致（TIM）修正**，与 OPD 的师生 KL 是两件不同的事 | 见 W3-a（两类截断的作用对象对照表）与 W3-b（表述还原）；相关机制链接 [[26_tim_causal_chain_analysis]] |
| 2 | Nemotron-Cascade 2 的 MOPD 与 Nemotron 3 Ultra 的 MOPD 被当作同一个东西引用 | Nemotron-Cascade 2 §4.4 节标题为 "**Multi-domain** On-Policy Distillation"，是**多领域**而非多教师；其截断 IS 区间 $\varepsilon_{low}=0.5$、$\varepsilon_{high}=2.0$ **已核实为真**（arXiv:2603.19220 §4.4）。同一缩写 MOPD 在 NVIDIA 两份报告中含义不同（Nemotron 3 Ultra 摘要为 "Multi-teacher On-Policy Distillation"，arXiv:2606.15007） | W3-a 表格中按"多领域"正确标注；详细辨析见 [[32_opd_framework_support_comparison]] §6 |
| 3 | Nemotron 3 Ultra 的"异步 behavior/proximal 策略解耦"作为一手事实引用 | **本轮未能独立核实**——arXiv HTML 与 ar5iv 均在 §3.3.1 之后截断 | W3 中标 ⚠️ **待核**，并注明其现有依据来自上游 research-notes 对官方 PDF §3.3.1（p.21）的读取 |

---

## 参考定位符汇总

**框架文档**：veRL OPD 算法页 https://verl.readthedocs.io/en/latest/algo/opd.html ；veRL 异步 recipe https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html ；slime OPD https://thudm.github.io/slime/zh/advanced/on-policy-distillation.html ；TRL GKDTrainer https://huggingface.co/docs/trl/gkd_trainer ；NeMo-RL OPD https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html ；Tinker cookbook https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation ；KDFlow https://github.com/songmzhang/KDFlow

**技术报告（infra 章节）**：DeepSeek-V4 arXiv:2606.19348v1 §5.1.2 / §5.2.1–5.2.3；Kimi K3 arXiv:2607.24653v2 §4.1.1–4.1.4；Nemotron 3 Ultra arXiv:2606.15007 §3.3；Nemotron-Cascade 2 arXiv:2603.19220 §4.4；GLM-5 arXiv:2602.15763v2 §3.5/§3.6.1；MiMo-V2-Flash arXiv:2601.02780v2 §4.4；Qwen3 arXiv:2505.09388v1 §4.5/§4.7；Gemini 2.5 arXiv:2507.06261 p.3。

**综述与方法**：Song & Zheng, *A Survey of On-Policy Distillation for LLMs*, arXiv:2604.00626v4 §2.4/§6.3/§7.4/§8.3；KDFlow arXiv:2603.01875v3；token 级偏差-方差 arXiv:2603.25562；KAT arXiv:2606.09471；长度膨胀 arXiv:2604.08527；熵感知 arXiv:2603.07079；Rock Tokens arXiv:2605.09253；TCOD arXiv:2604.24005；MiniLLM arXiv:2306.08543 Eq. 2；TML 博客 https://thinkingmachines.ai/blog/on-policy-distillation/ （DOI 10.64434/tml.20251026）。

---

## Related Pages

- [[14_on_policy_distillation_analysis]] —— OPD 的定义、2×2 定位与算法总览（本页的算法前置）
- [[32_opd_framework_support_comparison]] —— 六框架 OPD 支持逐项对照与选型（本页 W1/W3/W6 的框架落地面）
- [[15_opd_divergence_and_objective_evolution_analysis]] —— 散度选择与目标函数演化（决定 §3.2 走哪条路线）
- [[33_opd_effectiveness_and_failure_modes_analysis]] —— 失败模式清单（W7 监控指标的来源）
- [[32_opd_industrial_landscape_analysis]] —— 厂商采用格局与教师来源（W2 多教师需求的产业背景）
- [[26_tim_causal_chain_analysis]] —— 训推不一致的因果链（W3-a 类型 I 截断与 W5 的机制页）
- [[12_rl_infra_efficiency_analysis]] —— RL 训练回路的效率工程（W8 与三段负载编排的同构部分）
- [[11_rl_sandbox_design_analysis]] —— 沙箱与环境设计（W4 长程 agent 的环境侧）
- [[01_posttraining_infra_mechanism_analysis]] —— 后训练基础设施总览（本页的上位页）
- [[verl/index]] —— veRL 框架索引
- [[20_slime_architecture_analysis]] —— slime 架构分析
