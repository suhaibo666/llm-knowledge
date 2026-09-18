# 独立审阅 · F 组：量化与 KV 压缩

> **状态**：进行中（中途保存）。以下为截至目前已核验的内容；未完成的核查标为“待核”。

- **组别**：F — quantization and KV compression
- **页面**：
  - `wiki/01_theory/05_inference/19_inference_quantization_analysis.md`（含 `assets/19_inference_quantization.svg` 与 `.mjs`）
  - `wiki/01_theory/05_inference/20_quantization_methods_analysis.md`
  - `wiki/01_theory/05_inference/21_kv_compression_analysis.md`
- **审阅者**：独立审阅者（未参与写作）；2026-09-18。

## 已打开的来源（版本）

| 来源 | 版本 | 获取方式 | 状态 |
|---|---|---|---|
| Jacob et al., Quantization and Training of NN for Integer-Arithmetic-Only Inference | arXiv:1712.05877v1 | PDF → 分页文本 | 已读 §2.1–2.4、§3 引言、§3.1 |
| Dettmers et al., LLM.int8() | arXiv:2208.07339v2 | PDF → 分页文本 | 已读 §3 引言、§3.1、§3.2、Appendix D |
| TensorRT Quantization Schemes / Accuracy Considerations | 11.3.0 | curl HTML | 已下载，待核 |
| TensorRT Working with Quantized Types | `latest` 动态页 | curl HTML | 返回 HTTP 404，待进一步确认 |
| GPTQ | arXiv:2210.17323v1 | PDF → 分页文本 | 已下载，待核 |
| AWQ | arXiv:2306.00978v1 | PDF → 分页文本 | 已下载，待核 |
| SmoothQuant | ICML 2023 PMLR v202 正式版 PDF | PDF → 分页文本 | 已下载，待核 |
| KIVI | arXiv:2402.02750v1 | PDF → 分页文本 | 已下载，待核 |
| H2O | arXiv:2306.14048v1 | PDF → 分页文本 | 已下载，待核 |
| StreamingLLM | arXiv:2309.17453v2 | PDF → 分页文本 | 已下载，待核 |
| DeepSeek-V2 | arXiv:2405.04434v1 | PDF → 分页文本（共享缓存） | 待核 |

## 已复算内容（`.venv` Python，Decimal / Fraction 精确算术）

- **19 三条路径**：$s=3.8/7=0.542857$ 与 $1.2/7=0.171429$；代码、还原、逐项误差、MAE/最大误差 $0.122857/0.2$、$0.54/2.6$、$0.02/0.057143$ 全部与表一致；$s/2=0.085714$ 一致。
- **19 非对称例**（$[0,15]$、$s=0.4$、$z=3$、最近偶数）：代码 $(0,2,3,5,13)$、还原 $(-1.2,-0.4,0,0.8,4.0)$、误差 $(0,0,0.2,0.1,0.2)$ 在**十进制精确算术**下一致。附带发现：IEEE-754 float64 中 $3.8/0.4=9.499999999999998$，按最近偶数得 9 → 代码 12、还原 3.6（误差仍为 0.2）；即“9.5 的平局”只在精确十进制下成立（见 P2）。
- **19 SVG**：`<text>` 内容与表逐项一致（scale、代码、MAE、max、scale 个数、橙色截断路径）；但向量只保留 3 位小数（见 P2）。
- **20**：$y=(0.98,2.45)$；RTN SSE $6.9629$；$H=2XX^\top=\begin{bmatrix}34&10\\10&4\end{bmatrix}$、$\det=36$、$H^{-1}=\frac1{36}\begin{bmatrix}4&-10\\-10&34\end{bmatrix}$；$\delta w_2=1.225$、$w_2=1.715$（恰为 $w_1=0$ 时的最小二乘最优 $(0.98+2.45)/2$）；GPTQ 输出 $(1,1)$、SSE $2.1029$；AWQ $w'=(0.98,0.49)\mapsto(1,0)$、输出 $(0.5,2)$、SSE $0.4329$；SmoothQuant $\hat w=(1,0.5)$、输出 $(1,2.5)$、SSE $0.0029$；无迁移时第二输出 $1.5$（SSE $0.9029$）。全部一致。附带：SmoothQuant Eq.(4) 取 $\alpha=0.5$ 时本例 $s\propto(2,1)$，教学尺度恰与默认迁移强度成比例。
- **21**：全量权重 $(0.100056,\dots,0.122208)$、$o=2.471958$；量化 K 代码 $(0,1,3,0,1,1)$、$o=2.489583$、差 $+0.017625$；保留 $\{1,3,5,6\}$ 权重 $(0.132636,0.486682,0.218680,0.162002)$、$o=2.658094$、差 $+0.186136$；payload 24/17/16 B。全部一致。

## 已确认的发现（截至目前）

### 19_inference_quantization_analysis

- **P2 · 已确认** · `19_inference_quantization_analysis.md:78`「Jacob 等 §3.1 解释了为什么不同输出通道的动态范围会使单一量化参数损失精度」— 定位错误：该段（“large differences (more than 100×) in ranges of weights for different output channels … outlier weight values”）位于 1712.05877v1 **§3 引言**（p.5，§3.1 之前）；§3.1 是 “Learning quantization ranges”。另外 Jacob 用这一失败模式论证 QAT，其方案仍是每数组一套参数（§2.1 “a single set of quantization parameters for all values within each activations array and within each weights array”）。raw 索引 `raw/01_theory/05_inference/Quantization_and_Training_of_Neural_Networks-1712.05877.md` 同样写成 §3.1。建议改为“§3 引言”，并注明 Jacob 以此动机 QAT、“改变粒度”是本页推论。
- **P2 · 已确认** · `19_inference_quantization_analysis.md:37`「图中的数均保留到六位小数」— SVG 中向量由 `list()` 以 `fixed(value, 3)` 输出，例如 `(-1.086, -0.543, 0, 0.543, 3.8)`；只有 scale、MAE、max 为六位。建议改为“scale 与汇总保留六位，向量保留三位”，或让脚本输出六位。
- **P2 · 已确认** · `19_inference_quantization_analysis.md:62`「$3.8/0.4=9.5$ 舍入为 10」— 数学上成立，但 float64/float32 中商为 $9.4999\ldots$，按任何舍入都得 9，代码 12；读者用 Python/PyTorch 复算会得到不同代码。建议注明“按精确十进制算术”，或换一个在二进制浮点中也恰为平局的输入。

- **P1 · 已确认** · `19_inference_quantization_analysis.md:64`「在其 Q/DQ 约束中要求所支持的 zero-point 为零；该条来自没有可用 11.3.0 固定路径的动态文档快照」及 `:7`、`:25` 的同一链接 — 引用不可核验且出处描述错误：(1) 所链 `https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html` 2026-09-18 返回 “Page Not Found”（文件名拼错，实际为 `work-with-quantized-types.html`）；(2) 版本固定页 `https://docs.nvidia.com/deeplearning/tensorrt/11.3.0/inference-library/work-with-quantized-types.html` 存在（HTTP 200，与 latest 文本相同），故“没有可用 11.3.0 固定路径”不成立；(3) 该页只写 “TensorRT uses a symmetric quantization scheme … centered around zero”，**没有** zero-point 约束；11.3.0 的 Explicit Quantization、Quantization Workflows 页也没有。该约束实际在 **TensorRT Operators Documentation 11.3.0 · Quantize**（`https://docs.nvidia.com/deeplearning/tensorrt/11.3.0/_static/operators/Quantize.html`）：“The zero_point must only contain zero-valued coefficients if set”。结论本身正确，但读者按页面链接无法核验。raw 索引 `raw/01_theory/05_inference/TensorRT_Quantization_Schemes-11.3.0.md` 有同样的坏链与“没有可用的 11.3.0 固定 URL”表述。修法：改链到 11.3.0 固定的 Working with Quantized Types（对称）与 Operators · Quantize（zero_point 必须为 0），删去“动态快照/无固定路径”表述，同步 raw 索引。

（其余发现待核后补充。）

## 锚点抽查（进行中）

| 页面 | 来源与定位 | 结果 |
|---|---|---|
| 19 | Jacob 1712.05877v1 §2.1（p.3，Eq.(1)，zero-point 段） | 相符：$r=S(q-Z)$；$Z$ 是实数 0 对应的量化值，使 0 可精确表示 |
| 19 | Jacob §2.2–2.4（p.3–4，Eqs.(4)–(11)） | 相符：$M=S_1S_2/S_3$ 离线计算；int32 累加 uint8 积；$S_{bias}=S_1S_2$、$Z_{bias}=0$；定点乘+舍入移位后饱和转 uint8；Eq.(7) 的 zero-point 校正项 |
| 19 | Jacob §3.1 | **定位不符**：所述内容在 §3 引言 |
| 19 | LLM.int8() v2 §3.1 Eq.(7) | 相符：X 每行、W 每列一个缩放常数，外积反归一化 |
| 19 | LLM.int8() v2 §3.2 Eq.(8) | 相符：离群 feature 约 0.1%，阈值 $\alpha=6.0$，其余 99.9% 走 8-bit |
| 19 | LLM.int8() v2 Appendix D、Table 5 | 相符：量化/分解开销显著；小模型变慢；加入分解后仅 13B、175B 有加速 |
