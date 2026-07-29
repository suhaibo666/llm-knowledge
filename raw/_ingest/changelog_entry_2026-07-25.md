## 2026-07-25：训推不一致（TIM）因果链——第 1 块首页与 raw 摄入清单

**Type**: Source Ingestion + Deep Dive（严格路线：所有断言带一手来源与 §/Fig/Table/Eq 级定位符；未核实项显式标注，不做推测补齐。）

- **新增 [[tim_causal_chain_analysis]]**（515 行）：打通本库此前断掉的一环——「kernel 非确定性 → logprob 偏差 → 重要性比方差放大 → 训练崩溃」。上游（浮点非确定性、batch 不变性）已由 [[determinism_and_numerical_reliability_analysis]] 覆盖，下游（loss spike 治理）已由 [[training_dynamics_stability_analysis]] 覆盖，本页补中间两环与算法侧修法全谱。
- **归因框架**：采用 Qwen 2512.01374 §2.4 的二因子分解（训推数值分歧 × 策略陈旧度）作为全页坐标系，据此把 PPO clip、TIS/TRM/ALP/MIPU、batch-invariant kernel、TBIK、FP16、MoE 路由回放归位到各自作用的因子上。
- **六类病因逐条落定位符**：引擎间 kernel 差异、batch 触发不同 tiling、**TP size 改变累加顺序**（本库此前未覆盖）、BF16 尾数不足、MoE 路由分歧、量化 rollout。其中 TBIK Table 1/2 的对照实验证明「只做 batch 不变，对跨 TP 数值发散几乎无效」（Llama-3.1-8B 上 BIO 的 27.54 甚至高于 BF16 的 26.48）。
- **崩溃形态学**：首次在本库区分 recomputation 与 bypass 两条路径的不同崩溃曲线（多阶段 vs 单阶段、是否伴随 loss spike），并记录一条对工程有直接影响的发现——**K1/K3 KL 对 recomputation 型崩溃是盲的**（前 700 步几乎平坦而 reward 已在退化），而 recomputation 正是 verl 等框架的常见默认路径。
- **确定性税汇总**：batch-invariant GEMM 194 vs cuBLAS 527 TFLOPS；单个确定性请求混入 11 请求批次使整批吞吐掉 56%；TBIK 端到端 22%–63%；vLLM+TorchTitan bitwise 一致 RL run 慢 2.4×。并指明三篇系统侧论文**无一测量 RL 闭环端到端代价**，列为头号 open question。
- **三处 `> [!contradiction]` 标注**：① TIM 根因是精度（2510.26788）还是优化（2602.01826）——两篇互相点名，本页据 $C\cdot T^2$ 的乘积结构论证二者数学上不互斥；② MoE 的 Routing Replay 是否必要——GSPO §5.3 主张可取代 vs 同作者组 2512.01374 §4.4 回到「必需」，并如实标注后者**未点名**前者；③ 确定性是否必须付性能代价——实测派 vs 2606.00279，本页指出后者目标是可审计而非不变性，且「无性能代价」这一标题级主张零测量。
- **两处 raw 原文核实**（直接读 `raw/` 中已有 PDF）：DeepSeek-V4 §3.3 的 dual-kernel strategy 抵消的是**放弃 split-KV 后 decoding attention 的 wave-quantization 损失**，不是 matmul 固定 tiling 的损失（matmul 侧是 DeepGEMM 端到端替换 + 放弃 split-k 后另做优化）；GSPO §5.3 的 Routing Replay 表述与 10% 专家漂移数字逐句核对。
- **更新 [[01_theory/04_posttraining/index]]**：新增「训推一致性（TIM）与 RL 稳定性」小节；**首次建立 Knowledge Gaps 节**，记录 9 条确认无一手来源的缺口（崩溃阈值、尾部刻画、逐位置增长曲线、极端 token 频率、重尾→熵坍塌因果、RL 闭环确定性税、VeXact 开销、RL 阶段路由坍塌、vLLM logprobs RFC）；标注三条旧目录结构下的失效链接并给出替代。
- **新增 `raw/_ingest/INGEST_MANIFEST_block1_tim.md` 与 `fetch_block1_tim_sources.ps1`**：29 项已核实来源（26 篇 PDF + 3 份官方文档）的摄入清单与下载脚本，含目标路径、arXiv ID、一句话定位、保真度标注，以及 7 项需浏览器手动存档的博客/issue。**容器侧 arXiv 被代理阻断，PDF 需在本机执行脚本落盘。**
- **Mermaid 三块**：均按 CLAUDE.md 清单逐条自检，并用 mermaid-cli 实渲通过。

---
