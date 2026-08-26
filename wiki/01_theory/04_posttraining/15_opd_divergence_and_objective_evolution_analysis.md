# OPD 的散度与目标函数演进：从 FKL 到奖励视角的六步，每步被什么问题逼出来

> 本页回答一个贯穿性问题：**从 FKL 到 RKL、到 JSD(β)/f-散度、到 skew KL、再到自适应混合与奖励视角，每一步是被什么问题逼出来的？** 这条演进不是散度的随机尝试，而是一条有清晰因果的逻辑链，且它在 2026 年的工业路线之争中原样重演（见 [[14_on_policy_distillation_analysis]] §4.1）。OPD 的定义、与 RL 的统一视角见 [[14_on_policy_distillation_analysis]]。
>
> **来源与保真度**：素材来自 仓库外的 OPD 调研稿目录 `opd-survey/`（2026-08-10 基线，主稿 + 姊妹篇 + 八份底稿共 10 份；**按用户决定未纳入 `raw/`**，引用时请自行取用该目录）（2026-08-10 基线的 OPD 综述调研稿）。本页相对源稿的增量是**把只陈述结论的地方补成推导**——凡标「本页推导」者为标准数学推演（散度的梯度性质、有界性、最优解），不依赖任何论文的内部结论，可自行验算。凡标 ⚠️ 者为二手或未独立核实。
>
> 最后更新: 2026-08-11

---

## 0. 全景：六步演进与它们各自治的病

| 步 | 目标/散度 | 期望在谁的分布下 | 优化方式 | 被什么问题逼出来 | 代表作 |
|---|---|---|---|---|---|
| 0 | FKL（软标签 / 序列级） | 教师 / 固定数据 | 加权交叉熵直接反传 | —（起点） | Hinton KD、SeqKD |
| 1 | RKL | **学生** | **策略梯度**（高方差，需三技巧） | FKL 的 mass-covering + 暴露偏差 | MiniLLM |
| 2 | JSD(β) × $\lambda$ 混合 | $\lambda$ 插值（数据轴独立于散度轴） | **监督式反传**（有偏低方差） | RKL 策略梯度的方差与不稳定 | GKD |
| 3 | f-散度统一（JS/TVD 胜出） | 视变体 | 逐步分解定理保证可计算 | 各散度无法公平比较 | f-DISTILL |
| 4 | skew KL | replay 混合 | 直接反传 + 收敛界 | on-policy 采样成本与梯度爆炸 | DistiLLM |
| 5 | 自适应 FKL+RKL | — | 按头尾（AKL）或按熵加权 | mode-seeking 叙事被证伪；多样性坍缩 | AKL、Entropy-Aware OPD |
| 6 | 奖励视角统一（$\lambda$ 缩放外推） | 学生 | KL 约束 RL | 散度语言表达力不足；超越教师需求 | G-OPD |

**一句话读法**：**第 0 步有两个独立的病，第 1 步与第 2 步各治一个，第 3–5 步在收拾第 1 步的烂摊子，第 6 步把整个问题换了一种语言重述。**

---

## 1. 第 0 步：FKL 的两个病灶

经典 KD 最小化前向 KL：

$$
\mathcal D_{\mathrm{KL}}(p_T\,\|\,q_\theta)=\mathbb E_{y\sim p_T}\Big[\log\frac{p_T(y)}{q_\theta(y)}\Big]
$$

期望在**教师**分布下取，故训练数据可直接来自教师输出或固定语料，优化退化为加权交叉熵——工程上最便宜，这是它统治 2015–2022 的原因。

### 1.1 病灶一：mass-covering（zero-avoiding）

**本页推导。** 观察被求和项 $p_T(y)\log\frac{p_T(y)}{q_\theta(y)}$：当 $q_\theta(y)\to 0$ 而 $p_T(y)>0$ 时，该项 $\to+\infty$。**即 FKL 对"教师有质量、学生没有"施加无穷惩罚**，学生被迫在教师的**全部**支持集上放质量。

后果在容量不足时立刻显现：学生只能把有限的概率质量摊薄到教师的所有模式上，包括教师本身也只给了很低概率的区域。生成时按 $q_\theta$ 采样，就会采出教师根本不会产出的劣质 token——这正是 MiniLLM 的动机论述。

用一维直觉说：教师是双峰分布，学生只有单峰的表达力，FKL 会把学生这个单峰摆在**两峰之间的低谷**上（以覆盖两侧），于是学生的高概率区恰是教师的低概率区，"在模式间搭桥"产生幻觉。

### 1.2 病灶二：off-policy

数据分布固定，学生推理时进入自生成前缀的状态空间，误差按 $O(\epsilon T^2)$ 累积（推导见 [[14_on_policy_distillation_analysis]] §2.1）。

**关键在于这是两个独立的病**：一个关于散度的方向，一个关于数据的来源。后续两条演进路线各治其一——**MiniLLM 治病灶一（顺带治了病灶二），GKD 治病灶二（把散度方向留作可调参数）**。这个"独立性"是理解整条演进链的钥匙。

---

## 2. 第 1 步：MiniLLM 反转散度方向

反向 KL：

$$
\mathcal D_{\mathrm{KL}}(q_\theta\,\|\,p_T)=\mathbb E_{y\sim q_\theta}\Big[\log\frac{q_\theta(y)}{p_T(y)}\Big]
$$

### 2.1 一石二鸟

**本页推导（mode-seeking 的来源）。** 观察 $q_\theta(y)\log\frac{q_\theta(y)}{p_T(y)}$：

- 当 $p_T(y)\to 0$ 而 $q_\theta(y)>0$，该项 $\to+\infty$ ⟹ **学生必须避开教师的零概率区**（zero-forcing）。
- 当 $q_\theta(y)\to 0$ 而 $p_T(y)>0$，该项 $\to 0\cdot\log 0=0$ ⟹ **学生可以安全地放弃教师的某些模式**，不受惩罚。

两条合起来：学生只需拟合教师高概率模式的一个**子集**，不需要覆盖全部——这就是 mode-seeking。第二个好处是天然 on-policy：期望本身就要求从 $q_\theta$ 采样，病灶二同时被治。

### 2.2 代价：蒸馏变成了 RL

**期望的采样分布依赖被优化的 $\theta$ 本身**，不能像 FKL 那样直接反传，必须用策略梯度。完整推导见 [[14_on_policy_distillation_analysis]] §3.1，结论是

$$
\begin{aligned}
\nabla_\theta L
&=-\,\mathbb E_{y\sim q_\theta}\Big[\sum_{t}\big(R_t-1\big)\,\nabla_\theta\log q_\theta(y_t\mid y_{<t},x)\Big],\qquad R_t=\sum_{t'\ge t}\log\frac{p_T(y_{t'}\mid\cdot)}{q_\theta(y_{t'}\mid\cdot)}
\end{aligned}
$$

即"教师似然比充当稠密奖励"。**蒸馏在这一步形式上变成了 RL**，也就继承了 RL 的全部病。MiniLLM 因此需要三个稳定化技巧（均已独立核实）：

| 技巧 | 机制 | 治什么 |
|---|---|---|
| **单步分解** | 把 $R_t$ 拆成当前步与未来步，当前步在**整个词表上直接求期望**而非用采样估计 | 降方差——当前步的期望可解析计算，不必蒙特卡洛 |
| **教师混合采样** | 从 $\tilde p=\alpha\,p_T+(1-\alpha)\,q_\theta$ 采样，$\alpha=0.2$ | 抑制学生早期的退化样本；防 reward hacking |
| **长度归一化** | 按序列长度归一 | 抵消 $R_t$ 随长度累加带来的长度偏置 |

> **三个技巧的共同结构**：它们分别对应策略梯度的三个经典病（方差、行为策略质量、奖励尺度）。**2026 年路线 B 的全部工程手段——clip 截断、截断重要性采样、教师混合——本质是同一批技巧在更大规模上的复现**（见 [[14_on_policy_distillation_analysis]] §4.3）。

---

## 3. 第 2 步：GKD 的关键洞察——病根在数据分布，不在散度方向

GKD（arXiv:2306.13649v3，已独立核实）把两个正交的设计轴显式拆开：

- **数据分布用 $\lambda$ 调**：$\lambda$ 比例的轨迹来自学生自采样——治病灶二。
- **散度方向用 $\beta$ 调**：广义 JSD(β) 插值——把病灶一变成可调参数而非固定选择。

$$
\begin{aligned}
\mathcal D_{\mathrm{JSD}(\beta)}(P\|Q)
&=\beta\,\mathcal D_{\mathrm{KL}}\big(P\,\|\,\beta P+(1-\beta)Q\big) \\
&\quad +(1-\beta)\,\mathcal D_{\mathrm{KL}}\big(Q\,\|\,\beta P+(1-\beta)Q\big)
\end{aligned}
$$

论文给出 $\lim_{\beta\to 0}\mathcal D_{\mathrm{JSD}(\beta)}(P\|Q)/\beta=\mathcal D_{\mathrm{KL}}(P\|Q)$，并称 $\beta$ 接近 0 / 1 时梯度分别类似 FKL / RKL。**注意这是极限/梯度意义上的退化，不是端点处的恒等式**——这是核验时发现源稿未加限定的一处。

### 3.1 最精妙的一点：绕开策略梯度

**本页推导。** GKD 的优化方式常被忽略但极关键：轨迹一旦采出即**视为常量**（不对采样分布求梯度），此时逐 token 散度 $\mathcal D(p_T\|p_S^\theta)$ 在每个前缀位置就是一个普通的可微函数，可以**监督式直接反传**：

$$
\nabla_\theta\, \mathcal D\big(p_T(\cdot\mid y_{<t},x)\,\|\,p_S^\theta(\cdot\mid y_{<t},x)\big)
$$

对照 §2.2 的完整梯度：GKD 丢掉的正是 [[14_on_policy_distillation_analysis]] §3.1 推导里的 **(I) 项**（采样分布移动带来的那一项），只保留 (II) 项。**这是一个有意的偏差换方差**——用"对采样分布的梯度不予修正"这个偏差，换掉 MiniLLM 的 REINFORCE 高方差。

论文的实验结论：**最优 $\beta$ 依赖任务与解码方式，没有普适赢家**；附录另称 mode-seeking 散度在温度采样评测下更好，贪心解码下散度选择影响不大。

### 3.2 GKD 实验的一个常被忽略的读法

在任务几何不强烈偏向某端时，**三种散度的结果相近**——即**采样策略（on/off-policy）比散度选择更重要**（源稿据 2604.00626 §2.3 的解读，⚠️ 转述）。这正是暴露偏差假说的旁证：病灶二比病灶一更致命。

> **核验补注**：GKD 的 XSum 2.1× / WMT 1.7× / GSM8K 1.9× 增益需带两条限定（已核实）——它是"**相对初始学生的提升量之比**"而非绝对分数之比，且是**跨不同规模 T5 学生模型取平均**。§3.2 论文自称是首个同时做蒸馏与 RL 微调（"as far as we know"），属**作者自述**优先权。

---

## 4. 第 3 步：f-DISTILL 的统一与"可计算性"定理

任意凸生成元 $f$（$f(1)=0$）生成 f-散度族：

$$
D_f(P\|Q)=\mathbb E_{y\sim Q}\Big[f\Big(\frac{P(y)}{Q(y)}\Big)\Big]
$$

FKL / RKL / JS / TVD 皆为特例：

| 散度 | 生成元 $f(u)$ | 行为 | 适配的任务几何 |
|---|---|---|---|
| Forward KL | $u\log u$ | mode-covering / zero-avoiding | 多可接受输出（创作、开放 QA） |
| Reverse KL | $-\log u$ | mode-seeking / zero-forcing | 唯一正确答案（数学、代码） |
| JSD | $u\log u-(u{+}1)\log\frac{u+1}{2}$ | 对称有界，平滑插值 | 中间态（如翻译） |
| $\alpha$-散度族 | 参数化 | 在 FKL（$\alpha\to 1$）与 RKL（$\alpha\to 0$）间连续插值 | 细粒度调节 |

### 4.1 一个对 OPD 至关重要的结构性质

**本页推导。** 把 $Q=\pi_\theta$（学生）代入定义，$D_f(\pi_T\|\pi_\theta)=\mathbb E_{y\sim\pi_\theta}[f(\pi_T/\pi_\theta)]$——**整族 f-散度都能写成学生策略下的期望**。因此在 on-policy 设定下，用学生自己的 rollout 估计任何 f-散度都是**无偏**的，**不需要重要性采样修正**。

这是一个容易被略过但很实用的性质：它意味着"换散度"在 on-policy 管线里是纯粹的损失函数替换，不牵动采样与修正逻辑。

### 4.2 Theorem 1：让各散度第一次能公平比赛

f-DISTILL 的核心贡献是**可计算性**（已独立核实原文表述）：

> "(a) The sequence-level KL, RKL, and JS divergences can be decomposed **exactly** into step-wise terms. (b) The sequence-level TVD can be **upper bounded** by step-wise terms."

序列级散度本身不可解（要对指数多的序列求和）；分解定理把它化为可计算的词级损失，各散度这才第一次能在同一框架下比较。

**赛果**（Table 2）：**对称散度（JS/TVD）几乎全面胜过非对称的 KL/RKL**，因为它们在 mode collapse（RKL 端）与 mode averaging（FKL 端）之间取折中——与 GKD 的 JSD(β) 独立收敛到同一结论。

> **核验补注（"几乎"这个限定词必须保留）**：论文自述为对称损失 "consistently better than asymmetric ones across all datasets **except for WMT16 EN-RO, where KL achieves a slightly better TER**"，并归因于机器翻译的单模态性质。⚠️ Table 2 的具体数值本次核验中出现两组互相矛盾的抽取结果，若要引用逐格数字须打开 PDF 原表核对；**排序结论本身由正文文字明确支持，不受影响**。

---

## 5. 第 4 步：DistiLLM 的工程化收口——skew KL

$$
\begin{aligned}
D_{\mathrm{SKL}}^{(\alpha)}(p\,\|\,q_\theta)
&=D_{\mathrm{KL}}\big(p\,\|\,\alpha p+(1-\alpha)q_\theta\big),\qquad \alpha=0.1
\end{aligned}
$$

### 5.1 为什么梯度有界（本页推导）

这是 skew KL 的全部要点，源稿只说"分母被混合分布兜底、不再趋零"，推导如下：

$$
D_{\mathrm{SKL}}^{(\alpha)}=\sum_y p(y)\,\log\frac{p(y)}{\alpha p(y)+(1-\alpha)q_\theta(y)}
$$

对 $q_\theta(y)$ 求偏导：

$$
\begin{aligned}
\frac{\partial D_{\mathrm{SKL}}^{(\alpha)}}{\partial q_\theta(y)}
&=-\,\frac{(1-\alpha)\,p(y)}{\alpha p(y)+(1-\alpha)q_\theta(y)}
\end{aligned}
$$

分母被 $\alpha p(y)$ 从下方兜住，故

$$
\begin{aligned}
\Big\vert\frac{\partial D_{\mathrm{SKL}}^{(\alpha)}}{\partial q_\theta(y)}\Big\vert\;
&\le\;\frac{(1-\alpha)p(y)}{\alpha\,p(y)}\;=\;\frac{1-\alpha}{\alpha}
\end{aligned}
$$

**梯度被一个与 $q_\theta$ 无关的常数界住**（$\alpha=0.1$ 时界为 9）。对照普通 FKL：$\partial D_{\mathrm{KL}}/\partial q_\theta(y)=-p(y)/q_\theta(y)$，当 $q_\theta(y)\to 0$ 时**无界**——这正是 FKL 训练在学生对某 token 概率极低时梯度爆炸的机制。DistiLLM 由此获得收敛速率保证（其 Theorem 1）。

工程侧另一半贡献是**自适应 off-policy replay buffer**（Alg. 1），相对 GKD/MiniLLM 取得 2.5–4.3× 训练加速——**它把"多大比例 on-policy 才够"变成了一个可调的成本旋钮**。

### 5.2 一个结构观察：三者其实同构

JSD(β)、JS、skew KL **都是"对混合分布 $\mathrm{mix}(p,q)$ 求 KL"**：

- skew KL $=\mathrm{KL}(p\,\|\,\alpha p+(1-\alpha)q)$
- JSD(β) $=\beta\,\mathrm{KL}(P\|M_\beta)+(1-\beta)\,\mathrm{KL}(Q\|M_\beta)$，其中 $M_\beta=\beta P+(1-\beta)Q$
- 即 **skew KL 恰是 JSD(β) 的第一项**（取 $\beta=\alpha$，去掉权重与第二项）

源稿把这一点标为【推断】，但它其实是可直接验证的代数事实。**2023–2024 的散度共识就收敛在这个"KL-to-mixture"家族上**，其共同机制正是 §5.1 的分母兜底——对称化与 skew 化是同一个数值稳定技巧的两种包装。

---

## 6. 第 5 步：AKL 的叙事纠偏——问题从"渐近行为"移到"优化动力学"

AKL（arXiv:2404.02657v4, COLING 2025，已独立核实）给出了这条链上最反直觉的一步：

> 摘要原文：**"neither mode-seeking nor mean-seeking properties manifest in KD for LLMs"**

理论 + 实验证明：在 LLM 词表 softmax 的表达力下，**FKL 与 RKL 共享同一优化目标，训练足够久都收敛到同一点**——教科书上的 mode-seeking / mean-seeking 区分（源自**学生表达力受限的连续分布场景**）在 LLM KD 中并不成立。

**为什么？** 因为 §1.1/§2.1 的推导都隐含一个前提：**学生无法完全表达教师**。若学生有足够表达力，$q_\theta=p_T$ 同时是 FKL 与 RKL 的全局最优，两者的最优解重合，"覆盖 vs 精确"的取舍根本不存在。LLM 的词表 softmax 在单个位置上是完全参数化的分类分布——这个前提在 token 级恰好不成立。

真正的差别只在**早期拟合次序**：**FKL 先拟合分布头部、RKL 先拟合尾部**。于是散度选择从"渐近拟合行为"问题变成"优化路径"问题，答案自然是自适应组合：

$$
\begin{aligned}
\mathrm{AKL}(p,q_\theta)
&=\frac{g_{\mathrm{head}}}{g_{\mathrm{head}}+g_{\mathrm{tail}}}\,\mathrm{FKL} \\
&\quad +\frac{g_{\mathrm{tail}}}{g_{\mathrm{head}}+g_{\mathrm{tail}}}\,\mathrm{RKL}
\end{aligned}
$$

其中 $g_{\mathrm{head}}/g_{\mathrm{tail}}$ 为头/尾区域的分布差距，头部由累积概率阈值 $\mu$（默认 0.5）划定。

> [!contradiction] mode-seeking 叙事：教科书说法 vs AKL 的证伪
> §1.1/§2.1 的 mass-covering / mode-seeking 推导在**学生表达力受限**时成立（这也是 MiniLLM 的动机），而 AKL 证明在 LLM 的 token 级 softmax 下该前提不成立、两者渐近等价。**两说并不真正冲突，冲突在于适用尺度**：序列级分布上学生确实表达力受限（指数大的序列空间），token 级条件分布上则不受限。引用 mode-seeking 时应说明是在哪个尺度上讲。
>
> **AKL 的结论必须带一个前提**（核验中确认，源稿未加）：论文同时指出**实践中 LLM 极少训练到收敛**（其图 3 约需 300 epoch），差异才有现实意义。不能把它读成"FKL 与 RKL 在实践中等价"。

2026 年的延续是 **Entropy-Aware OPD**（arXiv:2603.07079v3）：按教师**逐 token 熵**混合——低熵位置用 RKL 保精度、高熵位置混入 FKL 保多样性，直接治 RKL 的多样性坍缩（Pass@8 +1.37~+5.05）。**加权的依据从"头/尾"细化到"教师在该位置有多确定"**，这是同一思路的自然推进。

---

## 7. 第 6 步：奖励视角的收束

**既然理论未定，为什么工业界默认 RKL？** 三个**工程**理由而非理论理由（源稿的归纳，本页认同并各自给出机制）：

| 理由 | 机制 |
|---|---|
| (a) 与 rollout 流程同构 | on-policy 采样下 RKL 的期望天然定义在学生分布上（§4.1），无需重加权 |
| (b) 可直接复用 RL 基建 | log-ratio 可当 advantage 塞进现成框架（[[14_on_policy_distillation_analysis]] §3.2 的"一行改动"） |
| (c) "unhackable" | 低 RKL 必然对应教师视角下的高概率行为，不像 learned reward 可被钻空子 |

> **核验补注（关于 (c) 的论证强度）**：TML 博客原文为 "unlike most reward models in practice, the reverse KL is 'unhackable' **in the sense that** low KL always corresponds to a high probability of desirable behavior **from the teacher model's point of view**"。这是一个**定义性论断**而非实证结论，且相对教师而言——**不等于绝对不可 hack**。[[33_opd_effectiveness_and_failure_modes_analysis]] 系统展示了教师信号并非处处可靠：**"不可 hack" 不等于"处处有益"**。

### 7.1 终点：把"选哪个散度"重述为"设计什么奖励"

G-OPD（arXiv:2602.12125v2）的形式化把整条演进收口：OPD 是**奖励与 KL 恒等权**的 KL 约束 RL。完整推导见 [[14_on_policy_distillation_analysis]] §3.3，两个结论：

- 恒等权时最优解 $\pi^\*=\pi_T$——**散度语言下"拟合教师"这件事，在奖励语言下就是"权重恰好相消"**。
- 引入奖励缩放 $\lambda$ 后，$\log\pi^\*_\lambda=\lambda\log\pi_T+(1-\lambda)\log\pi_{\mathrm{ref}}-\log Z$，$\lambda>1$ 产生外推。

**这一步的意义是表达力**：散度语言只能表达"向教师靠拢"，无法表达"超过教师"；奖励语言可以，因为奖励的**尺度**是一个自由参数，而散度没有这个自由度。**散度演进的尽头，是把"选哪个散度"重述为"设计什么奖励"。**

---

## 8. 本页结论

1. **第 0 步有两个独立的病**（mass-covering、off-policy），这个独立性是整条链的钥匙：MiniLLM 治第一个、GKD 治第二个，两者不是竞争关系而是分工。
2. **反转散度方向的代价是把蒸馏变成 RL**（§2.2），第 3–5 步基本都在收拾这个代价——f-DISTILL 让比较成为可能，skew KL 把梯度界住，AKL 把问题重新定位。
3. **JSD(β) / JS / skew KL 是同一个"KL-to-mixture"家族**（§5.2，可代数验证），共同机制是分母兜底带来的梯度有界性。
4. **mode-seeking 叙事在 token 级不成立**（§6），但在序列级仍成立——引用时须说明尺度；且 AKL 的等价性有"训练到收敛"这个现实中很少满足的前提。
5. **散度选择是未决问题**，工业界的 RKL 默认是工程约束（复用 RL 基建）而非理论最优的胜利；最优选择依赖任务几何、解码方式与训练阶段。
6. **奖励视角是散度语言的严格扩展**（§7.1）：它多出的那个自由度（奖励尺度 $\lambda$）正是"超越教师"的入口。

---

## Related Pages

- [[14_on_policy_distillation_analysis]] — OPD 主线权威页：定义、形式化、与 RL 的统一视角推导、两条实现路线
- [[32_opd_industrial_landscape_analysis]] — 工业实践全景（RKL 成为事实标准的产业侧证据）
- [[33_opd_effectiveness_and_failure_modes_analysis]] — 失败模式全谱（本页第 5 步的多样性坍缩、第 6 步的外推悬崖在此展开）
- [[13_opd_infra_mechanism_analysis]] — 散度选择的系统含义：全词表 / top-k / 采样 token 的带宽账
- [[13_reasoning_rl_algorithm_evolution_analysis]] — Reasoning RL 算法演进（同期的另一条目标函数演进链）
- [[30_preference_optimization_analysis]] — DPO 家族（另一条"把 RL 目标改写为可直接优化的损失"的路线）
- [[11_ppo_analysis]] — KL 约束 RL 的原型
- [[01_theory/04_posttraining/index]] — 后训练算法理论入口
