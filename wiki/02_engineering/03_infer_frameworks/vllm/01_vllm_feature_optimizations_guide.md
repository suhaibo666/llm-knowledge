---
title: "vLLM 使用与优化指南：用可撤销实验寻找限制资源"
---

# vLLM 使用与优化指南：用可撤销实验寻找限制资源

> **读者问题**：怎样把一个模型可靠地跑起来，建立与真实负载一致的 SLO、吞吐和资源基线，再用一次只改变一个变量的实验找出限制资源，而不是不断追加看似相关的 flags？
> **源码基线**：`vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`（冻结的 detached checkout，`v0.28.1rc0-80-g6b110badbb`，提交时间 2026-08-29T02:40:53Z）
> **中心命题**：vLLM 不存在脱离 workload envelope 的普适最优配置。可复现的调优是一条闭环：**负载假设 → 基线测量 → 限制资源诊断 → 一次只改一个变量 → 正确性与性能双重验证 → 触发条件满足即回滚**。配置只是用来证伪瓶颈假设的实验手段，不是优化成果本身。
> **本文拥有**：离线/在线的最小使用路径、workload envelope、benchmark 选择、warmup/measurement 分离、瓶颈到配置族的决策、one-variable experiment、correctness guard、验收与回滚。
> **明确排除**：请求、Scheduler/KV、Model Runner、采样、编译、并行、跨实例 KV 与可靠性的内部状态机；本页只说明何时去验证这些机制，并链接到各自 owner。
> **最近更新**：2026-08-30。

## 一、背景：为什么“把 flags 堆上去”不能证明优化

同一个参数变化可能同时改变容量、排队、启动成本、尾延迟和输出语义。只比较一次总
tokens/s，会把相关性误写成因果：吞吐变高也可能只是输出变短、请求被拒、prefix cache
复用了上一轮状态，或延迟 SLO 已经失守。

当前源码本身也说明“默认值”不是一张静态处方。顶层默认 optimization level 是 `-O2`、
performance mode 是 `balanced`，但 batch token/sequence budget 会继续按 usage context、
world size 解析，并受模型长度/模态约束；`throughput` mode 也只会放大用户**没有显式
指定**的两个 budget。证据分别见 `vllm/config/vllm.py:415-425`、
`vllm/engine/arg_utils.py:2791-2828` 与 `vllm/engine/arg_utils.py:2830-2874`。因此，
本页不复制一张很快过期的 flag/default 目录；要核对当前安装版本的某个参数，应使用
`vllm serve --help=<参数名>`，CLI 也支持
按配置组或关键字查询，见 `docs/cli/README.md:37-48`。

> [!note] 分析 / 推断
> 调优要采用闭环而不是“推荐值”，是实验因果性的要求，不是源码作者声明的唯一方法。
> 它胜过 flag stacking 的判据是：每轮结果都能归因、能被独立重复、失败后能恢复到已知
> 基线。

## 二、先冻结 workload envelope 与验收门

workload envelope 是这轮实验的合同。没有它，“更快”只代表换了一道题。开始服务前先
把下表填进实验记录；右栏不是普适阈值，而是由业务或评测协议填写的本轮边界。

| 维度 | 必须冻结或记录 | 本轮验收门示例 |
|---|---|---|
| 模型语义 | model/revision、tokenizer、chat template、dtype/quant format、sampling、stop、输出上限 | 固定 prompts 的 token/文本/任务分数不越界 |
| 请求形状 | 输入/输出 token 分布、共享前缀比例、模态与媒体尺寸、LoRA 组合 | 采样后的分布与生产窗口一致 |
| 到达过程 | offline 固定语料，或 online request rate、burstiness、并发上限、持续时间 | 目标到达率和 burst 被实际发出 |
| SLO | 成功率、TTFT、TPOT、ITL、端到端延迟的目标分位数 | 所有硬 SLO 同时满足 |
| 吞吐 | requests/s、input/output/total tokens/s；需要时用 SLO goodput | 只统计成功且满足 SLO 的请求 |
| 资源环境 | vLLM commit/镜像、driver/runtime、GPU 与互连、CPU/NUMA、可用内存、共租户 | 环境差异为零或被单独标注 |
| 运行状态 | cold start、warmup 次数、measurement 窗口、seed、重复轮数、prefix cache 是否保留 | warmup 不进入计时，状态策略一致 |

语义必须显式固定，因为 `LLM.generate()` 在未传 `SamplingParams` 时会读取默认采样参数，
而 `LLM.chat()` 还会先按 chat template 把消息渲染成 prompt；见
`vllm/entrypoints/llm.py:418-481` 与 `vllm/entrypoints/llm.py:612-690`。官方 quickstart
还说明模型仓库的 `generation_config.json` 会影响缺省 sampling；需要 vLLM 自身默认语义
时应显式选择 `generation_config="vllm"`，见
`docs/getting_started/quickstart.md:112-130`。

### 2.1 correctness guard 先于性能目标

先为固定的 canary corpus 保存基线输出与任务级评分，再决定什么差异可接受：

- 只改 batching、graph 或并行布局时，要求请求成功集合、输出 token 与 stop 原因保持一致；
- 改量化或 sampling 时，不把 bitwise equality 当成普适标准，而是预先声明 token/logprob
  容差、任务质量下限或统计检验；
- 每轮都检查空输出、截断、NaN/Inf、错误类型与失败率；失败请求不能从吞吐分母中消失。

这是实验协议，不是 vLLM 对所有模型的正确性承诺。采样与 grammar 的真实约束由
[[02_engineering/03_infer_frameworks/vllm/18_vllm_sampling_structured_output_analysis|采样与结构化输出]]
拥有；量化数值合同由
[[02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis|量化 ABI]] 拥有。

## 三、只用最小路径证明“可以执行”

### 3.1 离线批推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="<model>", generation_config="vllm")
params = SamplingParams(temperature=0, max_tokens=128)
outputs = llm.generate(["<prompt-1>", "<prompt-2>"], params)
```

`LLM.generate()` 接受 prompt sequence，并由引擎在内存约束下自动组成 batch；当前 API
明确建议把可共同执行的 prompts 放进同一个列表，见
`vllm/entrypoints/llm.py:429-459`。chat/instruct 输入不要手工假定模板等价，应使用
`LLM.chat()` 或先冻结 renderer 结果；两条路径的语义边界见
[[02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis|vLLM 请求语义]]。

### 3.2 在线服务

```bash
vllm serve <model> --generation-config vllm
```

`vllm serve <model>` 是当前 OpenAI-compatible 启动入口；默认 endpoint 与
`generation_config.json` 的覆盖行为见 `docs/getting_started/quickstart.md:203-220`。这个
命令只证明服务可接受请求，不证明 production SLO、鉴权边界或容量已经成立。

## 四、基准工具要回答同一个问题

官方 CLI 把 `serve` 定义为在线 serving benchmark，把 `throughput` 定义为离线 inference
throughput benchmark；`latency` 则测单个固定 batch，见 `docs/cli/README.md:137-150`、
`docs/cli/README.md:152-166` 与 `docs/cli/README.md:168-181`。
三者不能互换：离线结果剥离了网络到达与在线排队，在线结果才承载生产 arrival/SLO
假设。

| 问题 | 工具 | 固定输入 | 主要输出 | 不能据此宣称 |
|---|---|---|---|---|
| 同一进程尽快吃完固定语料 | `vllm bench throughput` | dataset、prompt/output length、seed、sampling、warmup | requests/s、input/output/total tokens/s | 在线 TTFT、排队和网关容量 |
| 固定 batch 的执行延迟 | `vllm bench latency` | batch size、shape、sampling、warmup iterations | batch latency | 混合请求下的尾延迟或吞吐 |
| 目标到达过程下是否满足 SLO | `vllm bench serve` | endpoint、dataset、request rate、burstiness、max concurrency、sampling、warmup | success、TTFT/TPOT/ITL/E2E、tokens/s、goodput | GPU kernel 是唯一瓶颈 |

离线与在线 benchmark 尽量使用同一数据采样协议；当前 throughput tests 明确记录它复用
`bench serve` 的 `get_samples` dispatch，以避免复制 dataset 语义，见
`tests/benchmarks/test_throughput_cli.py:3-8`。在线 CLI 的 integration test 也实际用
`input-len`、`output-len` 和 `num-prompts` 启动 benchmark，见
`tests/benchmarks/test_serve_cli.py:103-124`。

### 4.1 离线：warmup 与计时分离

```bash
vllm bench throughput \
  --model <model> \
  --dataset-name random \
  --input-len <input-tokens> \
  --output-len <output-tokens> \
  --seed <fixed-seed> \
  --num-warmups <warmup-prompts> \
  --num-prompts <measurement-prompts>
```

当前实现先生成独立 warmup requests，再执行 measurement requests；warmup 不启 profiler，
也不进入返回的计时结果，见 `vllm/benchmarks/throughput.py:43-79` 与
`vllm/benchmarks/throughput.py:1031-1039`。因此需要同时记录 cold-start 时间和 steady-state
结果，但不能把两者混成一个数字。

### 4.2 在线：把到达过程写进命令

```bash
vllm bench serve \
  --backend vllm \
  --model <model> \
  --dataset-name random \
  --input-len <input-tokens> \
  --output-len <output-tokens> \
  --request-rate <requests-per-second> \
  --burstiness <arrival-shape> \
  --max-concurrency <gateway-limit> \
  --temperature 0 \
  --seed <run-specific-seed> \
  --num-warmups <warmup-requests> \
  --num-prompts <measurement-requests>
```

必须显式写 `request-rate`：当前默认值是 `inf`，会在时间 0 发出全部请求；finite rate
才使用 Poisson 或 Gamma 间隔，而 `max-concurrency` 还可能使实际到达率低于目标值，见
`vllm/benchmarks/serve.py:1602-1613` 与 `vllm/benchmarks/serve.py:1671-1690`。warmup 默认是
0，但指定后实现会先等待整组 warmup 完成，再打印 `Starting main benchmark run`，见
`vllm/benchmarks/serve.py:882-907` 与 `vllm/benchmarks/serve.py:1707-1710`。

采样同样不能依赖隐含默认：当前 `bench serve` 已不再自动发送 `temperature=0`，未指定时
由 server/model API 决定；源码会为此打印 warning，见
`vllm/benchmarks/serve.py:2171-2176`。若你的 production 采样不是 greedy，就把相同的真实
sampling 参数显式写进基线和候选，而不是照抄上例。

另一个状态污染源是 prefix cache。官方 benchmark 文档明确警告：固定 seed 在同一 server
重复跑会复用上一轮 prompts，放大吞吐；若本轮不是刻意测 cache reuse，应更换 seed、重启/
重置 server，或使用会在 runs 间清 cache 的 sweep，见
`docs/benchmarking/cli.md:114-121`。

## 五、建立 baseline：SLO、吞吐与资源必须同屏

对冻结的 workload envelope 先跑两类 baseline：

1. **正确性 baseline**：固定 canary corpus、sampling 与输出上限，保存逐请求状态和质量分数；
2. **性能 baseline**：独立 warmup 后跑足够长的 measurement window，保存原始逐请求结果、
   聚合分位数、资源时间线与完整 resolved config。

`bench serve` 在 client 测 TTFT、ITL 和 TPOT：TTFT 截止到首个 streamed output，ITL 是相邻
streamed outputs 的间隔，TPOT 是每请求排除首 token 后的平均 output-token 时间；投机
解码一次 stream 可能带回多个 token，所以 ITL 与 TPOT 不能互相替代，见
`docs/benchmarking/cli.md:123-153`。实现还把用户为 TTFT、TPOT、E2E 中**已指定**的阈值
全部满足的成功请求计入 goodput，见 `vllm/benchmarks/serve.py:568-650`。

每个 baseline artifact 至少保存：

- 完整启动/benchmark 命令、解析后的配置、commit、镜像、模型 revision 与硬件拓扑；
- dataset manifest、实际 token 长度、seed、sampling、到达率、并发、warmup 与测量时间；
- success/error counts、TTFT/TPOT/ITL/E2E 分位数、requests/s、tokens/s、goodput；
- GPU memory 峰值/余量与利用率、CPU 利用率、network/collective 时间，以及 owner 页定义的
  queue、preemption、prefix hit、fallback、engine health 信号；
- canary outputs、任务质量、日志和 profiler trace 的路径。

## 六、先找限制资源，再选一个配置族

单个症状不是原因。例如 GPU 利用率低既可能是 CPU/render 饥饿，也可能是 batch 不足、
collective 等待或同步边界。先提出**唯一可证伪的假设**，再选择能让该假设变强或变弱的
一个变量。下表中的“反证”失败时，回到诊断，不继续加 flags。

| 观测到的限制信号 | 本轮假设 | 只选一个配置族 / 一次变量 | 必须同时观察的反证 | 机制 owner |
|---|---|---|---|---|
| frontend CPU 饱和，GPU 间歇空闲 | render/tokenize/media 输入供给不足 | API/input-processing capacity | GPU 空闲减少但输出语义、TTFT tail 与 CPU queue 不恶化 | [[02_engineering/03_infer_frameworks/vllm/17_vllm_serving_control_plane_analysis|Serving 控制面]]、[[02_engineering/03_infer_frameworks/vllm/19_vllm_multimodal_execution_analysis|多模态执行]] |
| TTFT 随 load 上升，decode ITL 尚稳定 | admission/prefill 竞争或排队主导 | token/sequence budget 或 arrival/concurrency，一次只改其一 | queue/prefill 指标不变则假设被否证 | [[02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis|Scheduler]] |
| preemption、KV 余量低或 OOM | 权重/KV/graph/临时 buffer 之一耗尽容量 | KV/上下文/并行/量化/graph memory 中先选一个家族 | 对应占用不降，或 latency/quality 代价超过门限 | [[02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis|KV Cache]]、[[02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis|量化 ABI]] |
| 小 batch 下 CPU/launch gap 明显 | host launch 或动态执行开销主导 | optimization/compile/graph mode | GPU timeline gap 不收缩，或 startup/memory 代价超过预算 | [[02_engineering/03_infer_frameworks/vllm/16_vllm_model_runner_v2_analysis|Model Runner V2]]、[[02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]] |
| profiler 指向 attention/GEMM/MoE/格式转换 | 某 backend/kernel 对当前 shape 或 dtype 不合适 | backend/kernel/quant format 中选一个 | 实际路径仍 fallback，或 kernel 变快但 E2E 无改善 | [[02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis|Attention Backend]]、[[02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis|融合 Kernel]] |
| decode 串行时间主导 | draft 成本可能小于节省的 target steps | speculative family 的一个候选 | acceptance、draft+verify 成本与 E2E 不支持 break-even | [[02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis|投机解码]] |
| collective 占比高或单卡放不下 | 当前 rank layout 是限制项 | TP/PP/DP/EP/CP layout 的一次变化 | 每 rank 工作、通信时间或容量没有按假设变化 | [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|分布式推理]] |
| prefill/decode 可独立扩展且传输可覆盖 | 跨实例拆分可能提高有效容量 | disaggregated KV topology | transfer/lease/failure 成本吞掉收益即否证 | [[02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis|分离式 KV Serving]] |

> [!warning] 相关性不是因果
> “开了某项功能且指标上升”不是证据。需要看到该功能确实进入目标路径、预期限制资源
> 随之变化、其他 envelope 不变，并能在回滚后复现反向变化。fallback、stale metrics 与
> 测量噪声由 [[02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis|可观测性与可靠性]]
> 的反馈合同约束。

## 七、one-variable experiment：一次变更必须能撤销

每轮复制下面的 experiment card；如果一个机制必须由多个参数共同启用，把它们定义成
一个不可拆的“候选配置”，但不要同时混入第二个机制族。

| 字段 | 写法 |
|---|---|
| 假设 | “若 X 是限制资源，改变 Y 后，指标 Z 应改善，反证信号 Q 应按预期变化” |
| baseline | 完整 config digest、artifact 路径、measurement window |
| 唯一变化 | 一个参数，或一个明确命名且不可拆的候选配置 |
| 不变量 | workload envelope、硬件、代码、模型语义、warmup、seed 策略 |
| correctness guard | canary、成功集合、质量/数值门限、错误率 |
| performance gate | primary metric + SLO/goodput + 资源 headroom |
| falsifier | 哪个观测会证明瓶颈假设错了 |
| rollback | 原始命令/config、是否需要 restart/clear cache、负责人 |

运行顺序至少要避免固定顺序偏差：先完成独立 warmup，再交错 baseline/candidate 的重复轮次；
保留每轮原始结果，不只保留均值。若波动区间覆盖了收益，结论应是“未证明”，而不是把
最优单轮写进配置。

## 八、双重验证与回滚触发器

候选只有同时通过 correctness 和 performance 两道门才可晋级：

| 门 | 通过条件 | 立即回滚触发器 |
|---|---|---|
| 正确性 | canary/任务质量在预声明边界内；请求集合与 stop/error 行为可解释 | 错 token/结构、质量越界、NaN/Inf、新错误类型或失败率越界 |
| SLO | 目标 load 下 TTFT/TPOT/ITL/E2E 与 goodput 全部过门 | 任一硬分位数或 goodput 失守，即使总 tokens/s 上升 |
| 因果 | 预期限制资源和 primary metric 同向变化，rollback 后反向复现 | 机制未进入目标路径、只见相关性、重复结果不稳定 |
| 容量 | GPU/CPU/host memory、network/collective 与 queue 有预留余量 | OOM、持续 preemption、queue 无界增长、health/engine death |
| 运维 | startup、compile/capture、cache warm 与恢复时间在预算内 | 冷启动或故障恢复超过既定预算 |

回滚不是“再调一个值”，而是恢复完整 baseline artifact，并恢复相同运行状态。若候选改变
了 graph、cache、权重布局或 server 进程内状态，应 restart 后重测；prefix cache 是否保留
则必须与 baseline 使用同一策略。回滚后若指标不能恢复，说明实验受环境漂移或隐藏状态
污染，本轮结果作废。

## 九、完成标准：留下可复查结论，不留下神奇数字

一次调优闭环完成时，读者应能回答：

1. 真实 workload envelope 与硬 SLO 是什么；
2. baseline 的正确性、吞吐、延迟与资源证据在哪里；
3. 哪个观测支持“限制资源是 X”，哪个观测能证伪它；
4. 唯一变化是什么，为什么属于对应配置族；
5. candidate 是否同时通过 correctness、SLO/goodput 和资源门；
6. 哪些代价或失败边界仍存在，回滚是否恢复 baseline。

这些问题闭合，才得到一个**对本 workload 有效**的配置；否则只得到一个待验证的相关性。

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis|vLLM 系统设计原则]] — 用资源承诺模型解释为何同一配置在不同动态负载下会改变瓶颈。
- [[02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis|vLLM 请求语义]] — 冻结 render、sampling、stop 与输出语义，避免把换题误判成加速。
- [[02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis|vLLM Scheduler]] — 当 queue、token budget、prefill 或 preemption 是诊断信号时，查阅唯一机制 owner。
- [[02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis|vLLM KV Cache 管理]] — 将容量、prefix hit 和 eviction 信号映射到单 Engine KV 所有权。
- [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|vLLM 分布式推理]] — 解释并行布局的 rank、collective 与通信代价，避免只比较 GPU 数。
- [[02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis|vLLM 可观测性与可靠性]] — 把 SLO 症状、资源承诺、fallback 与故障域闭合成可证伪反馈。
