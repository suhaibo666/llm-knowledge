# Transformer 下 Dense 与 MoE 模型的参数初始化（知乎风格深度科普）

*作者：ChatGPT（GPT-5 Thinking mini） — 面向研究/工程的数学推导 + 实践建议*

---

## 摘要（TL;DR）

* 权重初始化的核心目标是**在前向传播时保持激活（和梯度）方差稳定**，避免随层数爆炸或消失。常见原则来自 **Xavier / He / Kaiming**，在 Transformer 中还要考虑残差累积（residual accumulation），因此通常对残差通路做**尺度修正**（例如 `1/√L`）或采用 pre-LN 结构来稳定训练。([Pinecone][1])
* 对 MoE（Mixture-of-Experts）层，需要同时考虑**专家（experts）内部初始化** 与 **路由器（gate）/top-k 聚合的缩放**。专家内部通常按密集 FFN 的规则初始化，但在合并多专家输出时要对激活做额外缩放（与激活的专家数相关）。MoE 的负载平衡/路由偏置也影响训练早期的稳定性。([arXiv][2])
* 主流大模型（如 Gemini、Qwen、DeepSeek、gpt-oss）的工程实践：预归一化（pre-LN）、残差尺度（或 GPT-2 的 `1/√N` 残差缩放技巧）、以及在超大模型上使用 **meta-device / lazy init** 来节省内存。下面给出完整数学推导、实践代码片段与模型对照建议。([Google AI for Developers][3])

---

## 目录

1. 背景回顾：为什么初始化重要？
2. 线性层与激活方差传播（Xavier / He 回顾）——逐步推导
3. Transformer 特殊项：残差连接、LayerNorm、注意力矩阵的方差分析
4. 深度（L）与宽度（hidden dim）如何影响初始化选择
5. MoE（稀疏专家）层的初始化与路由器注意点
6. 主流模型实践对比（Gemini / Qwen / DeepSeek / gpt-oss）——工程要点
7. 实用 PyTorch 初始化片段（dense / MoE / gating）
8. 实验建议与调试 checklist
9. 总结与参考文献

---

## 1. 背景回顾：为什么初始化重要？

训练深度网络时，如果每层激活的方差不断放大或缩小，两个问题会出现：

* **前向传播**：激活变得非常大（导致数值不稳定）或接近 0（信息丢失）。
* **反向传播**：梯度爆炸/消失，导致训练无法收敛或收敛极慢。

初始化的目标：设计每层权重的分布（均值 μ，方差 σ²），使得在“随机输入/随机权重”的近似下，**每层激活的方差在期望上大致恒定**（或受控）。Xavier（Glorot）和 He（Kaiming）就是为不同激活函数给出的解析解。([Pinecone][1])

---

## 2. 线性层与激活方差传播（逐步推导）

考虑一个简单的线性层（无偏置以便推导）：
$$
y = W x,\quad W_{ij}\sim\mathcal{N}$0,\sigma_w^2$
$$
假设输入 $x$ 的元素独立同分布，均值 0，方差 $\operatorname{Var}(x)=\sigma_x^2$。则输出单元的方差：
$$
\operatorname{Var}$y_i$ = \sum_{j=1}^{n_\text{in}} \operatorname{Var}$W_{ij}x_j$ = n_\text{in},\sigma_w^2,\sigma_x^2
$$
为使 $\operatorname{Var}(y)\approx\operatorname{Var}(x)$，需要
$$
\sigma_w^2 \approx \frac{1}{n_\text{in}}
$$
这就是 **Xavier（均方保留）** 的基本直觉（对线性或对称激活）。如果后续有 ReLU，会丢失一半能量，He 初始化建议
$$
\sigma_w^2 \approx \frac{2}{n_\text{in}}
$$

**注意**：`n_in` 即 fan-in（输入通道数或上一层输出维度），而 `n_out` 是 fan-out。在实现上可用 PyTorch 的 `nn.init.xavier_uniform_` / `kaiming_uniform_`。([Stack Overflow][4])

---

## 3. Transformer 特殊项：残差连接、LayerNorm、注意力矩阵的方差分析

### 3.1 残差连接的方差累积问题

Transformer block（简化）：
$$
\text{out} = x + \mathcal{F}$x$
$$
若 $\operatorname{Var}(x)=\sigma_x^2$ 且 $\operatorname{Var}(\mathcal{F}(x))=\sigma_f^2$，且近似独立（粗略假设），则
$$
\operatorname{Var}$x+\mathcal{F}(x)$ = \sigma_x^2 + \sigma_f^2
$$
在深层网络中，若每层 $\sigma_f^2$ 大致恒定，随着层数 $L$ 方差会**线性增长**，即约为 $\sigma_x^2 + L\cdot \sigma_f^2$——这导致深度越大越不稳定。解决办法（工程上常见的）：

1. **Pre-LayerNorm（pre-LN）结构**：把 LayerNorm 放到子层输入，能把残差项正规化，使每层进入子层的分布更稳定；这是 Transformer 训练中非常普遍的设计，用来缓解残差累积。
2. **残差缩放（residual scaling）**：在初始化时把子层权重按 $1/\sqrt{L}$ 或 $1/\sqrt{N}$ 缩小，使得每层 $\sigma_f^2 \propto 1/L$，从而整体方差不爆炸。GPT 团队在早期工作中就采用过对残差的尺度修正（或变种）。这类技巧在实践中常被用来训练更深或更大的 Transformer。([高洪南][5])

**简单推导（残差缩放）**：若把子层输出乘以 $s$（常取 $s=1/\sqrt{L}$），则总方差约：$\sigma_x^2 + L\cdot s^2 \sigma_{f,0}^2 = \sigma_x^2 + L \cdot \frac{1}{L}\sigma_{f,0}^2 = \sigma_x^2 + \sigma_{f,0}^2$ ——不随层数线性增长。

### 3.2 注意力（QKV）矩阵的方差

注意力机制中，通常把 Q、K、V 的投影矩阵初始化同线性层。关键在于 softmax 前的缩放因子 $1/\sqrt{d_k}$（这是为了数值稳定且控制点积方差）。投影矩阵的方差仍以 fan-in 作为基准；若使用多头注意力，head_dim 较小时要确保初始化方差对每个 head 都合适（即基于 head_dim 的 fan-in）。

总结：Transformer 的初始化应该同时满足线性层的 fan-in 原则，并对**残差累积**做专门处理（pre-LN、残差缩放或初始化时缩小子层权重）。

---

## 4. 深度（L）与宽度（hidden dim）如何影响初始化选择

* **宽度（hidden dim, d）**：影响 fan-in/fan-out。一般线性层（包括 MLP 层、QKV 投影等）按 `fan_in = in_features` 或 `fan_avg = $fan_in+fan_out$/2` 来选 σ²（Xavier/He）。宽度越大，单个权重方差需要越小。
* **深度（L）**：主要通过残差累积影响整体方差（见上）。随着 L 增大，必须采取残差缩放 / pre-LN /特殊初始化（如 GPT-2 的残差缩放技巧）以防方差随深度线性增长。最新研究也表明初始化“是否能让 Transformers fit 任务”高度依赖初始化细节（见近期 arXiv 关于“Initialization is critical...” 的实验证据）。([arXiv][6])

**直观结论**：

* 宽而浅：按常规 Xavier/He 即可（注意 head_dim 的计算）。
* 深而窄：更需注意残差缩放 / pre-LN / 更小的子层初始化方差。

---

## 5. MoE（稀疏专家）层的初始化与路由器注意点

MoE 层通常替换 FFN（feed-forward）为若干专家 $E_1,\dots,E_n$，路由器（gate）决定每个 token 访问 top-k 个专家并合并输出。初始化涉及两方面：

### 5.1 专家内部（expert）初始化

每个专家内部通常是一个 FFN（Linear → nonlinearity → Linear），按密集 FFN 的规律初始化（fan-in/He/Xavier），因为当专家被激活时，其内部计算与普通 FFN 等价。若专家数量很多、每个专家容量不大，仍以专家自己的 fan-in 计算 σ²。([IntuitionLabs][7])

### 5.2 多专家输出合并（缩放问题）

若路由器以权重 $w_i$ 聚合专家输出（例如 soft weighted sum 或 top-k 求和），合并后输出的方差约为专家方差乘以合并系数的平方和。常见情况：

* top-1（只取一个专家）：方差与单个专家一致。
* top-2（取两个专家并平均/加权）：当两个独立专家被平均时，合并方差会被缩小或放大，取决于权重。为保持数值稳定，一般会对合并结果除以 $\sqrt{k}$（或按专家数量做缩放），使方差与一个专家在同一尺度。

**路由器（gate）的初始化**：路由器是个小网络（线性→softmax），其 logits 的偏置与权重决定早期负载平衡。常见技巧：

* gate logits 初始偏置为 0（无偏向）或小噪声（避免在训练早期把大部分 token 推给同一专家）。
* 一些 MoE 设计（DeepSeek-V3、GShard 等）还添加 auxiliary loss/均衡项；DeepSeek-V3 甚至提出 auxiliary-free 的负载平衡策略（工程细节见其技术报告）。([arXiv][8])

---

## 6. 主流模型实践对比（Gemini / Qwen / DeepSeek / gpt-oss）

下面把主流模型的公开工程/文档与初始化相关的实践要点总结成表格（基于可公开文档/技术报告与工程 repo）：

| 模型 / 项目                     |                                                                                                                                                     公开实践（与初始化相关） | 说明 / 推荐                                                         |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------- |
| **DeepSeek-V3**             |                                                              MoE 为核心（671B total，37B active），提出了 load-balance 策略与架构改进。专家内部按标准 FFN 初始化；路由设计对早期稳定性至关重要。([arXiv][8]) | 如果训练类似规模 MoE：专家按 fan-in 初始化；gate 初始偏置为 0；合并 top-k 时考虑合并缩放。      |
| **Qwen（阿里）**                |                                                         商用大模型系列，使用 transformer 的大型训练工程（文档/API 以服务为主）。公开细节有限，但工程上通常采用 pre-LN 与残差尺度技巧以稳定训练。([alibabacloud.com][9]) | 遵循 pre-LN + scaled init，避免 residual 方差爆炸。                       |
| **Gemini（Google/DeepMind）** |                                                       Google 的 Gemini 系列以工程化为主，文档面向 API/部署；内部训练实践未全部开源，但同样使用 pre-LN、分布式初始化/检查点策略。([Google AI for Developers][3]) | 工程上会结合 lazy/meta init 与 sharded checkpoint（见下 gpt-oss）以节省内存。    |
| **gpt-oss（开源实现）**           | 一些开源实现强调 memory-efficient initialization（例如使用 `meta` device / lazy init 在 CPU 上占位再分布），这对超大模型构建很常用。初始化策略一般按 transformer 社区推荐的 scaled init / pre-LN。([GitHub][10]) | 对超大模型：使用 meta device 创建设备无权重占用对象，实际权重在加载 checkpoint 时填充，节省 RAM。 |

> 结论：不同组织会在**同一理论基础（fan-in/残差缩放/pre-LN）**上做工程优化（meta init、sharding、load balancing）。公开论文或 repo（如 DeepSeek-V3 的技术报告、gpt-oss repo）能确认这些工程实践。([arXiv][8])

---

## 7. 实用 PyTorch 初始化片段（推荐实现）

### 7.1 Dense Transformer 层（建议）

```python
import math
import torch
import torch.nn as nn

def init_linear_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)   # 或 nn.init.xavier_normal_
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.apply(init_linear_xavier)
```

### 7.2 残差缩放（初始化时缩放子层权重）

如果你想在初始化阶段就采用残差缩放（例如 GPT-2 的变体）：

```python
def scaled_init_linear(m, scale):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.weight.data.mul_(scale)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# scale = 1/math.sqrt(num_layers)  # 或更细粒度的设定
```

### 7.3 MoE 关键点（专家初始化 + gate）

```python
class SimpleGate(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.w = nn.Linear(d_model, num_experts)
        # gate 初始化：权重小随机；bias=0（避免初期偏置）
        nn.init.normal_(self.w.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.w.bias)

# 专家内部用常规初始化（见上），合并时注意缩放：
# if top_k == 2 and we sum experts e1+e2, 可以 /sqrt(2) 归一化（或其他策略）
```

---

## 8. 实验建议与调试 checklist

1. **先用小模型（缩放版）做快速试验**：检查激活与梯度的方差（逐层打印均值/方差）。
2. **验证残差方差随层数是否增长**：在随机输入下，测量每层输出的方差随层号的曲线。若线性增长，考虑加残差缩放或 pre-LN。
3. **MoE 专家负载**：监控每个专家的 token count，若长尾严重，考虑调整 gate 初始化或使用 load-balance loss。([arXiv][2])
4. **超大模型内存/初始化问题**：使用 `meta` device / lazy init（如许多 gpt-oss 实现）以减小 CPU RAM 占用。([GitHub][10])

---

## 9. 总结（工程化建议）

* **基础法则**：线性层按 fan-in（或 fan_avg）用 Xavier/He 初始化；QKV 基于 head_dim；FFN 基于 fan-in。([Pinecone][1])
* **深度修正**：当模型很深或残差较多时（即 Transformer），要显式考虑残差累积 —— 使用 pre-LN 或在初始化时对残差子层做缩放（例如 $1/\sqrt{L}$ 或更细粒度 scaling），这是 GPT 系列和其它大模型常用的稳定化手段。([高洪南][5])
* **MoE**：专家内部按常规初始化；对路由器的 logits/偏置小心初始化以避免训练早期负载倾斜；合并多个专家输出时做缩放/归一化以保持方差稳定。([arXiv][8])
* **实战**：结合 meta/lazy init、sharded checkpoints 和监控工具（激活/梯度分布、专家负载），逐步放大模型规模。主流工程实践（DeepSeek、Qwen、Gemini 的工程化做法）都遵循这个原则，在细节上有所优化。([arXiv][8])

---

## 参考（选）

1. *Initialization is Critical to Whether Transformers Fit Composite...*（近期 arXiv，讨论初始化与深度稳定性） — 说明初始化对 transformer 能否学习的影响。([arXiv][6])
2. GPT/GPT-2 相关实现与经验（残差尺度/初始化讨论）——社区总结。([高洪南][5])
3. DeepSeek-V3 技术报告 / repo（MoE 设计与工程实践）。([arXiv][8])
4. Mixture-of-Experts in LLMs（综述 / 实践建议）。([arXiv][2])
5. gpt-oss / 开源实现说明（memory-efficient initialization，meta device），工程层面很实用。([GitHub][10])
6. Weight initialization tutorial（Xavier / He 基础），用于理解 fan-in/fan-out。([Pinecone][1])

[1]: https://www.pinecone.io/learn/weight-initialization/?utm_source=chatgpt.com "Weight Initialization Techniques in Neural Networks"
[2]: https://arxiv.org/html/2507.11181v1?utm_source=chatgpt.com "Mixture of Experts in Large Language Models"
[3]: https://ai.google.dev/gemini-api/docs/function-calling?utm_source=chatgpt.com "Function calling with the Gemini API | Google AI for Developers"
[4]: https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks?utm_source=chatgpt.com "How to calculate fan-in and fan-out in Xavier initialization ..."
[5]: https://www.gaohongnan.com/influential/generative_pretrained_transformer/04_implementation.html?utm_source=chatgpt.com "The Implementation of Generative Pre-trained Transformers ..."
[6]: https://arxiv.org/html/2405.05409v3?utm_source=chatgpt.com "Initialization is Critical to Whether Transformers Fit Composite ..."
[7]: https://intuitionlabs.ai/articles/mixture-of-experts-moe-models?utm_source=chatgpt.com "Understanding Mixture of Experts (MoE) Neural Networks"
[8]: https://arxiv.org/html/2412.19437v1?utm_source=chatgpt.com "DeepSeek-V3 Technical Report"
[9]: https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference?utm_source=chatgpt.com "Alibaba Cloud Model Studio:Qwen API reference"
[10]: https://github.com/VizuaraAI/truly-open-gpt-oss?utm_source=chatgpt.com "GPT-OSS: The Missing Open-Source Training Code"


下面我把**数学推导部分扩展为一整页的 LaTeX 严谨推导**（含更精确的联合分布假设与梯度传播分析），并把结果与直观解释、工程化建议整合回你之前的知乎风格博客中。文档以中文为主，LaTeX 公式完整且逐步推导，便于直接粘贴到论文/笔记中或作为博客的数学附录。

> 使用说明：下面先给出完整的 LaTeX 推导（“数学附录”），随后把关键结论和工程建议并入博客正文（以 Markdown / 知乎风格呈现）。如果你要直接把 LaTeX 单独保存为 `.tex`，可把数学附录部分整体复制到 `.tex` 文件里；或者我可以把它转换成单独的 PDF/tex 文件（你需要我生成文件的话告诉我 — 我现在不能异步生成文件）。

---

# 数学附录（LaTeX 严谨推导）：前向 / 反向方差传播与初始化原理

> 下面的推导假设所有随机变量（权重、输入、误差项等）在初期训练阶段可用独立性假设近似处理，并用零均值（或对零均值进行中心化）的近似；对非线性函数（如 ReLU、GELU）用其关于输入分布的二阶统计量近似处理。注：实际训练中这些独立性假设并不完全成立，但在初始化分析与设计中常用且能给出有用的量纲尺度指导。

---

## 设定与符号

* 我们用大写 $X$、$W$、$Y$ 表示随机向量/矩阵；小写 $x_i$ 表示分量。
* 单层线性映射（无偏置）：
  $$
  y = W x,\qquad W\in\mathbb{R}^{n_\text{out}\times n_\text{in}}
  $$
* 假设权重独立同分布（i.i.d.）且均值零：
  $$
  W_{ij}\overset{\text{i.i.d.}}{\sim}\mathcal{D}*W,\quad \mathbb{E}[W*{ij}]=0,\ \operatorname{Var}$W_{ij}$=\sigma_w^2.
  $$
* 输入向量分量 $x_j$ 假设独立同分布，$\mathbb{E}[x_j]=0,\ \operatorname{Var}(x_j)=\sigma_x^2$。
* 梯度符号：若损失为 $ \mathcal{L}$，前向为 $y=W x$，则反向有
  $$
  \delta_y \equiv \frac{\partial\mathcal{L}}{\partial y},\qquad
  \delta_x = W^\top \delta_y,\qquad
  \frac{\partial\mathcal{L}}{\partial W_{ij}} = \delta_{y_i}, x_j.
  $$

下面先做**前向方差传播**的严格推导，再做**反向（梯度）方差传播**推导，最后讨论残差、LayerNorm、注意力缩放、MoE 合并等实际结构的影响。

---

## 1. 前向方差传播（线性层）

对单个输出分量 $y_i=\sum_{j=1}^{n_\text{in}} W_{ij} x_j$，由于 $W_{ij}$ 与 $x_j$ 假设独立且均值为 0，则输出分量的方差为（使用独立性与线性方差和）：

$$
\begin{aligned}
\operatorname{Var}$y_i$
&= \operatorname{Var}!\Big$\sum_{j=1}^{n_\text{in}} W_{ij} x_j\Big$
|= \sum_{j=1}^{n_\text{in}} \operatorname{Var}!\big$W_{ij} x_j\big$ \
&= \sum_{j=1}^{n_\text{in}} \mathbb{E}\big[W_{ij}^2\big], \mathbb{E}\big[x_j^2\big]
;$\text{因独立且均值 0}$\
&= n_\text{in},\sigma_w^2,\sigma_x^2.
\end{aligned}
$$

因此若希望输出方差与输入方差相同（$\operatorname{Var}(y_i)\approx\sigma_x^2$），则需要

$$
\sigma_w^2 \approx \frac{1}{n_\text{in}}.
$$

这就是 Xavier 的核心观点（对线性或对称激活）。若激活后接 ReLU（将负半截置零，近似将方差乘以$\frac12$），则 He 初始化建议

$$
\sigma_w^2 \approx \frac{2}{n_\text{in}}.
$$

---

## 2. 前向—含非线性 $ \phi(\cdot)$ 的层

若有激活 $z=\phi(y)$，则在零均值假设下，近似可用一阶线性化或利用 $\phi$ 关于输入分布的二阶统计量。若 $y$ 近似为均值为 0、方差 $\sigma_y^2$ 的高斯分布，则

$$
\sigma_z^2 = \operatorname{Var}$\phi(y)$ \approx \int \phi$t$^2 p_{y}$t$,dt - \Big$\int \phi(t) p_y(t),dt\Big$^2,
$$

对 ReLU（$\phi(t)=\max(0,t)$）且 $y\sim\mathcal{N}(0,\sigma_y^2)$，有

$$
\mathbb{E}[\phi$y$^2] = \frac{\sigma_y^2}{2},\qquad
\mathbb{E}[\phi$y$] = \sigma_y\frac{1}{\sqrt{2\pi}}.
$$

因此近似有 $\sigma_z^2 \approx \frac{1}{2}\sigma_y^2$（常见近似，忽略均值项的平方）。对 GELU 等更平滑非线性，可用数值或泰勒近似计算其增益因子 $g_\phi$ 使得 $\sigma_z^2 \approx g_\phi \sigma_y^2$。

---

## 3. 反向（梯度）方差传播

关键结论（将逐步推导）：

* 对于线性层 $y = W x$，反向传播有 $\delta_x = W^\top \delta_y$，因此：
  $$
  \operatorname{Var}$\delta_{x_j}$ = n_\text{out},\sigma_w^2,\operatorname{Var}$\delta_{y}$.
  $$
* 对权重梯度 $\nabla_{W_{ij}} = \delta_{y_i}, x_j$，其方差为：
  $$
  \operatorname{Var}$\nabla_{W_{ij}}$ = \operatorname{Var}$\delta_{y_i}$,\operatorname{Var}$x_j$.
  $$
  （在独立性与零均值假设下）

下面是详细推导。

### 3.1 推导：$\operatorname{Var}(\delta_x)$

由 $\delta_x = W^\top \delta_y$，第 $j$ 分量为

$$
\delta_{x_j} = \sum_{i=1}^{n_\text{out}} W_{ij}, \delta_{y_i}.
$$

假设 $W_{ij}$ 与 $\delta_{y_i}$ 近似独立（一种常用近似，训练早期通常近似成立），并且 $\mathbb{E}[\delta_{y_i}]=0$，且 $\operatorname{Var}(\delta_{y_i})=\sigma_{\delta_y}^2$，则

$$
\begin{aligned}
\operatorname{Var}$\delta_{x_j}$
&= \sum_{i=1}^{n_\text{out}} \operatorname{Var}$W_{ij},\delta_{y_i}$ \
&= \sum_{i=1}^{n_\text{out}} \mathbb{E}[W_{ij}^2],\mathbb{E}[\delta_{y_i}^2] \
&= n_\text{out},\sigma_w^2,\sigma_{\delta_y}^2.
\end{aligned}
$$

注意：这与前向传播的公式互为“对偶”（前向用 fan-in，反向用 fan-out）。因此要同时保证正向和反向的方差都不爆炸/消失，常用做法是选择 $\sigma_w^2$ 使两者折衷（Xavier 使用 $\frac{2}{n_\text{in}+n_\text{out}}$），或用分别基于 fan-in/fan-out 的初始化（若关心梯度更重要则采用 He 或 fan-out 针对 ReLU）。

### 3.2 推导：$\operatorname{Var}(\nabla_W)$

单个分量梯度为 $\nabla_{W_{ij}} = \delta_{y_i} x_j$。在独立性假设下：

$$
\operatorname{Var}$\nabla_{W_{ij}}$
= \operatorname{Var}$\delta_{y_i}$,\operatorname{Var}$x_j$
= \sigma_{\delta_y}^2,\sigma_x^2.
$$

因此权重更新的噪声强度与**前向激活方差**和**反向误差信号方差**的乘积成正比。这直接说明了为什么前向的方差尺度会影响权重更新（梯度的尺度）：如果激活非常小（$\sigma_x^2$ 很小），那么即便 $\sigma_{\delta_y}^2$ 正常，梯度也会非常小（更新缓慢，类似“梯度消失”）；反之激活太大会使梯度方差变大，导致步长不稳或梯度爆炸。

---

## 4. 残差连接对方差与梯度的影响

考虑一个残差块（简化）：

$$
x^{$\ell+1$} = x^{$\ell$} + \mathcal{F}\big$x^{(\ell)}\big$,
$$

记 $\operatorname{Var}(x^{(\ell)})=\sigma_\ell^2$，并假设 $\mathcal{F}$ 的输出与 $x^{(\ell)}$ 近似独立且 $\operatorname{Var}(\mathcal{F})=\tau^2$（每层近似恒定）。则

$$
\sigma_{\ell+1}^2 = \sigma_\ell^2 + \tau^2.
$$

递推得到 $L$ 层后，若每层 $\tau^2$ 恒定，则 $\sigma_L^2 \approx \sigma_0^2 + L \tau^2$，随着深度线性增长。于是：

* 前向：激活规模随层数增长 → 激活变大 → 权重梯度方差∝激活方差*误差方差 → 梯度容易爆炸。
* 反向：用链式法则，梯度会通过多条路径累积，若没有正则化/归一化，会导致梯度尺度失控。

解决方法（数学上解释）：

1. **残差缩放**：在每一层把 $\mathcal{F}$ 乘以尺度 $s$。令 $s^2 = 1/L$，则每层对方差贡献为 $s^2 \tau_0^2 = \tau_0^2 / L$，累积后仍是 (O(1))。
2. **Pre-LN（先 LayerNorm）**：LayerNorm 会把进入子层的向量按通道做中心化和标准化，使每层输入的方差被约束（通常为 1），从而避免残差累积导致的方差膨胀。LayerNorm 的存在改变了前向方差的递推关系：若在子层输入处执行 LayerNorm，则每层 $\operatorname{Var}(\mathcal{F})$ 的输入尺度被固定，$\sigma_\ell^2$ 不会线性增长（近似稳定）。这也是为什么 pre-LN 结构在超深 Transformer 中广泛使用的原因。

---

## 5. 注意力机制中的缩放因子 $1/\sqrt{d_k}$ 的推导

在自注意力中，缩放因子 $1/\sqrt{d_k}$ 的目的正是为了控制点积 $q^\top k$ 的方差。若 $q$ 和 $k$ 的每个分量独立且方差为 $\sigma_q^2,\sigma_k^2$，则

$$
q^\top k = \sum_{t=1}^{d_k} q_t k_t,
$$

有

$$
\operatorname{Var}$q^\top k$ = \sum_{t=1}^{d_k} \operatorname{Var}$q_t k_t$
= d_k, \mathbb{E}[q_t^2],\mathbb{E}[k_t^2] = d_k,\sigma_q^2,\sigma_k^2.
$$

因此未经缩放，点积的方差与 $d_k$ 成正比，随着 head_dim 增大 softmax 会变得越来越尖锐或不稳定。乘以 $1/\sqrt{d_k}$ 后，

$$
\operatorname{Var}!\Big$\frac{q^\top k}{\sqrt{d_k}}\Big$= \sigma_q^2\sigma_k^2,
$$

与 $d_k$ 无关，保证了不同 head_dim 下 logits 的尺度可比，从而稳定 softmax 的梯度行为。

---

## 6. MoE（Top-k 专家合并）对方差与梯度的影响

设有 $E$ 个专家，每个专家返回向量 $e_\alpha$（方差 $\sigma_e^2$），若路由器对某 token 选择 top-$k$ 专家并按权重 $w_\alpha$ 合并，输出为

$$
o = \sum_{\alpha\in\mathcal{S}} w_\alpha e_\alpha,\qquad \mathcal{S}\ \text{为被选专家集合},\ |\mathcal{S}|=k.
$$

若 $e_\alpha$ 相互独立且均值 0，则

$$
\operatorname{Var}$o$ = \sum_{\alpha\in\mathcal{S}} w_\alpha^2,\operatorname{Var}$e_\alpha$
= \sigma_e^2 \sum_{\alpha\in\mathcal{S}} w_\alpha^2.
$$

常见情形：

* 如果 $w_\alpha$ 都为 $1$（简单求和），则 $\operatorname{Var}(o) = k,\sigma_e^2$（方差随 $k$ 增大）。
* 若对 $k$ 个专家取算术平均（每个权重为 $1/k$），则 $\operatorname{Var}(o) = \sigma_e^2/k$（方差缩小）。
* 若对权重做归一化（比如 softmax），则 $\sum w_\alpha^2$ 在不同情形下差异较大（当权重集中时，$\sum w_\alpha^2$ 接近 1，当权重均匀时约为 $1/k$）。

**工程结论**：为保持与非 MoE FFN 输出同一尺度，常用策略有：

* 在合并后乘以因子（例如 $\sqrt{k}$ 或 $1/\sqrt{k}$）以调整方差；
* 设计 gate logits / 初始化策略，使得初期 $\sum w_\alpha^2$ 在合理范围（避免一开始就把几乎所有 token 推给单一专家导致不平衡）；
* 在专家层内部使用与 dense FFN 相同的初始化（基于各自 fan-in），这样单专家输出的尺度可与 dense 层对齐。

---

## 7. 总结数学结论（便于直接用于初始化设计）

1. **前向方差受 fan-in 支配**：线性层初始化方差应与 $1/\text{fan_in}$ 同阶，如 Xavier/He。
2. **反向方差受 fan-out 支配**：反向传播中 $\operatorname{Var}(\delta_x)\approx \text{fan_out}\cdot\sigma_w^2\cdot\operatorname{Var}(\delta_y)$。因此初始化也需考虑 fan-out（Xavier 用了 fan_in 和 fan_out 的平均以折中）。
3. **权重梯度的方差依赖于前向激活与反向误差信号的乘积**：$\operatorname{Var}(\nabla_W) \propto \operatorname{Var}(x)\cdot\operatorname{Var}(\delta_y)$。这解释了为何前向尺度直接影响学习速率与稳定性。
4. **残差累积会导致前向方差随层数线性增长**：需要 pre-LN 或残差缩放 $s$（如 $s=1/\sqrt{L}$）以稳定尺度。
5. **注意力的 $1/\sqrt{d_k}$ 缩放**：使 logits 的方差与 $d_k$ 无关，保证 softmax 的稳定性。
6. **MoE 合并需要缩放或合适的 gate 初始化**：避免合并后方差偏移或专家负载不均带来的训练不稳定。

---

# 完整 LaTeX 数学附录（可复制到 `.tex` 的完整段落）

下面给出可以直接放进 LaTeX 文档的数学附录（带编号与注释）：

```tex
\section*{数学附录：前向与反向方差传播的严格推导}

\paragraph{符号与假设} 我们考虑一层线性映射
\[
y = W x,\quad W\in\mathbb{R}^{n_\text{out}\times n_\text{in}},
\]
并作如下常用近似假设：
\begin{itemize}
  \item 权重元素独立同分布： \(W_{ij}\overset{\text{i.i.d.}}{\sim}\mathcal{D}_W\)，\(\mathbb{E}[W_{ij}]=0\)，\(\operatorname{Var}(W_{ij})=\sigma_w^2\)。
  \item 输入分量独立同分布：\(x_j\overset{\text{i.i.d.}}{\sim}\mathcal{D}_x\)，\(\mathbb{E}[x_j]=0\)，\(\operatorname{Var}(x_j)=\sigma_x^2\)。
  \item 在涉及乘积的统计量时，我们近似假设相关项相互独立（例如 \(W_{ij}\) 与 \(x_j\) 独立，\(W_{ij}\) 与 \(\delta_{y_i}\) 近似独立）。
\end{itemize}

\subsection*{前向方差}
对单个输出分量 \(y_i=\sum_{j=1}^{n_\text{in}} W_{ij}x_j\)，由独立性：
\[
\operatorname{Var}(y_i)
= \sum_{j=1}^{n_\text{in}} \operatorname{Var}(W_{ij}x_j)
= \sum_{j=1}^{n_\text{in}} \mathbb{E}[W_{ij}^2]\,\mathbb{E}[x_j^2]
= n_\text{in}\,\sigma_w^2\,\sigma_x^2.
\]
若欲 \(\operatorname{Var}(y_i)\approx \sigma_x^2\)，则必要条件近似为
\[
\sigma_w^2 \approx \frac{1}{n_\text{in}}.
\]

\subsection*{含激活函数的情况}
若在此后接非线性激活 \(z=\phi(y)\)，当 \(y\) 近似为均值 0 的高斯变量，激活后的方差可写为
\[
\sigma_z^2 \approx \mathbb{E}[\phi(y)^2] - (\mathbb{E}[\phi(y)])^2.
\]
例如 ReLU 下，常用近似 \(\sigma_z^2 \approx \tfrac12\sigma_y^2\)。

\subsection*{反向方差}
记损失为 \(\mathcal{L}\)，反向误差信号为 \(\delta_y=\partial\mathcal{L}/\partial y\)，则
\[
\delta_x = W^\top \delta_y,\qquad
\delta_{x_j} = \sum_{i=1}^{n_\text{out}} W_{ij}\,\delta_{y_i}.
\]
在近似独立与零均值假设下：
\[
\operatorname{Var}(\delta_{x_j})
= \sum_{i=1}^{n_\text{out}} \mathbb{E}[W_{ij}^2]\,\mathbb{E}[\delta_{y_i}^2]
= n_\text{out}\,\sigma_w^2\,\sigma_{\delta_y}^2.
\]
因此反向方差由 fan\_out 决定，前向方差由 fan\_in 决定。

\subsection*{权重梯度的方差}
梯度元素 \(\nabla_{W_{ij}}=\delta_{y_i} x_j\)，若 \(\delta_{y_i}\) 与 \(x_j\) 近似独立且均值 0，则
\[
\operatorname{Var}(\nabla_{W_{ij}})
= \operatorname{Var}(\delta_{y_i})\,\operatorname{Var}(x_j)
= \sigma_{\delta_y}^2\,\sigma_x^2.
\]
这表明权重更新的尺度与前向激活与反向誤差信号的方差乘积成正比。

\subsection*{残差与 LayerNorm 的影响}
若每层为残差结构 \(x^{(\ell+1)}=x^{(\ell)}+\mathcal{F}(x^{(\ell)})\)，并假设 \(\operatorname{Var}(\mathcal{F})=\tau^2\) 恒定，则
\[
\operatorname{Var}(x^{(\ell+1)})=\operatorname{Var}(x^{(\ell)})+\tau^2,
\]
递推得到 \(\operatorname{Var}(x^{(L)})\approx \operatorname{Var}(x^{(0)})+L\tau^2\)，会随深度线性增长。LayerNorm 在子层输入处将向量标准化（使局部方差固定），从而抑制这种增长。另一种对策是对 \(\mathcal{F}\) 乘以缩放因子 \(s\)（如 \(s=1/\sqrt{L}\)），使得累积量级受控。

\subsection*{自注意力的缩放因子}
若向量 \(q,k\) 的分量方差分别为 \(\sigma_q^2,\sigma_k^2\)，则点积 \(q^\top k\) 的方差为
\[
\operatorname{Var}(q^\top k) = d_k\,\sigma_q^2\,\sigma_k^2.
\]
除以 \(\sqrt{d_k}\) 后，方差归一化为 \(\sigma_q^2\sigma_k^2\)，从而保证不同 head\_dim 下 logits 的尺度一致。

\subsection*{MoE 合并的方差}
若合并 \(k\) 个独立专家输出 \(e_\alpha\)（每个方差 \(\sigma_e^2\)）并用权重 \(w_\alpha\) 加权求和，则输出方差为
\[
\operatorname{Var}\Big(\sum_{\alpha=1}^k w_\alpha e_\alpha\Big) = \sigma_e^2\sum_{\alpha=1}^k w_\alpha^2.
\]
特殊情形下（均匀平均 \(w_\alpha=\tfrac{1}{k}\)）方差为 \(\tfrac{1}{k}\sigma_e^2\)；若直接求和则为 \(k\sigma_e^2\)。为匹配 dense 层尺度需要在合并后对输出做合适缩放。

\qed
```

## Related Pages

- [[01_theory/index]]
- [[muon_analysis]]
- [[Megatron-LM_MoE_Zero_Redundancy_Analysis]]
- [[mHC]]
