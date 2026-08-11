## 2026-08-11：在线策略蒸馏（OPD）专题摄入 —— 六页新簇 + 独立复核六处更正

**Type**: Source Ingestion + Deep Dive + Correction（源稿为用户自研的 OPD 综述调研稿，已整体落入 `raw/`；本次在其之上做了独立一手复核与数学补全，冲突处按 CLAUDE.md 标注。）

### 一、原始来源落库

- 新增 `raw/01_theory/04_posttraining/opd_survey_2026-08/`（364 KB，10 份）：主稿 `OPD-Survey-2026-08.md`（99 KB，606 行，87 个带版本号的 arXiv 引用、184 处 §级定位）、姊妹篇 `OPD-Infra-Survey-2026-08.md`（30 KB），及 `research-notes/` 下八份底稿（四路调研底稿 + 三份否定性结论复核附录 + Song & Zheng 89 页综述深读摘录）。
- 源稿方法论值得记录：显式区分「一手核实 / ⚠️ 二手 / 【推断】」三级可信度，对每条否定性结论附可复现检索式，并记录了复核中三项否定性结论被推翻的过程（Kimi 从"最大反例"转向 K3 的九专家 MOPD、字节 Seed 的"无公开资料"被修正、MiniLLM 改名版本被精确定位到 v6）。

### 二、新增六页

**算法侧 `01_theory/04_posttraining/`**

- **[[14_on_policy_distillation_analysis]]**（段 1 主线权威页）：定义与 2×2 坐标（判据只有"轨迹是否学生自采"一条）、与相邻概念的边界表、**暴露偏差 $O(\epsilon T^2)$ 的推导**、**OPD ≡ KL 约束 RL 的完整推导**（含反向 KL 必须用策略梯度的证明、MiniLLM 梯度中 $(R_t-1)$ 那个 $-1$ 的来源、KL 约束 RL 最优解 $\pi^*\propto\pi_{\mathrm{ref}}e^{r/\beta}$ 及由此得出的"不动点即教师"与 $\lambda$ 外推公式）、两条实现路线与三种 clip 的区分、独立核验记录。
- **[[15_opd_divergence_and_objective_evolution_analysis]]**（段 1）：散度演进六步的因果链，每步补推导——FKL 的 mass-covering 与 RKL 的 mode-seeking 各自由被求和项的极限行为导出、**skew KL 梯度有界性的完整证明**（界为 $(1-\alpha)/\alpha$，对照 FKL 的无界）、f-散度族"都是学生分布下的期望故 on-policy 无需 IS 修正"、**JSD(β)/JS/skew KL 同属 KL-to-mixture 家族的代数验证**（源稿标为【推断】，实为可验证事实）、AKL 对 mode-seeking 叙事的证伪及其适用尺度辨析。
- **[[32_opd_industrial_landscape_analysis]]**（段 3，644 行）：厂商谱系总表、四类工业用途（压缩/合并/防遗忘/跨模态）、逐厂商深读十节、两条实现路线的一手证据正面对立、教师五类来源与生产经济学、未采用与不披露阵营、11 条待核清单。
- **[[33_opd_effectiveness_and_failure_modes_analysis]]**（段 3，606 行）：三层有效性证据、可利用差距原理与四项训前诊断、**16 条失败模式按五族重组并逐条展开「症状→机理→触发条件→修复→修复代价」**（五族各附一句"被违反的前提假设"作为共同根因）、scaling 与 OPD 独有的 rollout 预算轴、决策框架、subliminal learning 与蒸馏攻击。

**工程侧 `02_engineering/04_posttrain_frameworks/`**

- **[[13_opd_infra_mechanism_analysis]]**（段 1）：OPD = "RL 的回路加一个新角色"、与 SFT/RL 的系统需求对照（critic 消失是省出的预算、GLM-5 把 group size 降到 1 直接除掉 rollout 采样量）、**信号格式四档带宽账**（全词表约 4.2 GB/轨迹 vs 采样 token 约 64 KB，相差约 5 个数量级）、成本模型与教师刷新率 $\rho$ 旋钮、**八项工作清单 W1–W8**、OPD 的 staleness 容忍窗口为何比 RLHF 更窄的机制论证。
- **[[32_opd_framework_support_comparison]]**（段 3）：veRL/slime/TRL/NeMo-RL/Tinker/KDFlow 六框架逐项矩阵与选型、OpenRLHF「可用而非原生支持」辨析、生产系统自研层、六条生态 Gap、预算分配三段模式。

### 三、独立复核：六处事实更正 + 一处降级

复核方式：`raw/` 中已有 PDF 直接逐字核对（DeepSeek-V4、Kimi K3、GLM-5），其余回 arXiv/官方页面核实。

| # | 源稿表述 | 复核结果 |
|---|---|---|
| 1 | TML 博客「以梯度步计 50–100×」 | **口径错位**：原文梯度步为 **7–10×**（<10 步 vs 70 步），**50–100× 是累计计算量**；且出自 Discussion 节的 LoRA rank-128 实验而非主实验 |
| 2 | TML「同设置 RL 达 68%」 | **非博客自跑**：博客是引用 Qwen3 报告的 **67.6%**（17,920 GPU 时，限定语 "a similar SFT initialization"） |
| 3 | MiMo BrowseComp「−6.3 失分域」 | **语义误读**：学生实际 42.5→45.4（**上升 +2.9**），−6.3 是相对 SFT 教师 51.7 的**差距**（Table 7 的 Δ 列为"学生−最强教师"）。真回退是创意写作 90.1→86.2 |
| 4 | Nemotron-Cascade 2 的 MOPD | §4.4 标题为 **Multi-domain** On-Policy Distillation，非 multi-teacher；71.5→85.5 是 ArenaHard V2.0 的 **Hard Prompt 子项**而非总分；Table 3 显示 RLHF 在 100 步已达 81.7 |
| 5 | MiMo「IcePop 式截断」 | 正文未出现 "IcePop"（仅见参考文献标题），原文为 "Following Zhao et al. (2025)"；且截断作用于**训推比 $\pi_\theta/\mu_\theta$**（属 TIM 修正，见 [[26_tim_causal_chain_analysis]]），**不是师生比值** |
| 6 | Nemotron 3 Ultra「两轮师生共进化」 | Figure 10 图注实为：第二轮教师由 **Ultra MOPD1（第一轮的学生）** 初始化并复用第一轮教师；**RLVR Student 是"自教师"**，补专用教师未覆盖的领域 |

**降级一处**：Nemotron 3 Ultra §3.3.5 的"教师天花板"逐字引语（"a limitation of the on-policy distillation setting"）——arXiv HTML 与 ar5iv 两条路径均在 §3.3.1 后截断，本次无一手页面支撑，已在两页降级为 ⚠️ 待核，源稿 P6 的支撑改挂到不依赖此引语的其它证据。

**核实为真、可作一手引用的**：DeepSeek-V4 §5.1.2 全套（"entirely replaced by On-Policy Distillation"、Eq. 29 多教师反向 KL、"more than ten teacher models"、对路线 B 的批评原文）；Kimi K3 §4.1.3（Eq. 15 clip 形式、"no clear advantage" 原句、九专家=三域×三档 effort）；Qwen3 §4.5 逐字与 §4.7 Table 21 十二格数字全中；GKD/MiniLLM/f-DISTILL/AKL 的核心主张；**MiniLLM 改名的版本史**（v5 2025-11-21 旧题、v6 2026-01-31 改为含 "On-Policy Distillation"，ICLR 论文集版保留旧题）。

**补入的限定条件**：Qwen3 的 1,800 vs 17,920 只是两个增量阶段之比（off-policy 起点行 GPU 时为 "–"），且报告未说明是否含教师推理成本；GKD 的 2.1×/1.7×/1.9× 是"相对初始学生的提升量之比"且跨模型规模平均；f-DISTILL 的对称散度优势有 WMT16 EN-RO 的 TER 例外；AKL 的"FKL 与 RKL 共享同一目标"须带"训练到收敛（约 300 epoch）"前提。

### 四、索引与校验

- 更新 `01_theory/04_posttraining/index.md`（段位表补 14/15/32/33、新增「在线策略蒸馏（OPD）」小节）与 `02_engineering/04_posttrain_frameworks/index.md`（补 13/32、新增 OPD 系统侧小节）。
- 校验：两目录 45 个文件的 wikilink **未解析目标 0 种**（修正了两处指向重排前旧页名的链接：`deepseek_v4_analysis`→`13_deepseek_v4_analysis`、`glm_5_analysis`→`01_glm_5_analysis`）；全部 `$$` 配对、代码块闭合；唯一一处 mermaid 经 mermaid-cli 实渲通过。以 LF 行尾写回。

---
