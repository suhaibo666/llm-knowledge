---
title: "vLLM 性能调优指南：用测量、单变量实验和回滚验证收益"
---

# vLLM 性能调优指南：用测量、单变量实验和回滚验证收益

> **源码基线**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（`main` 快照，2026-09-07 UTC）
> **主题**：先冻结负载和质量要求，再选择基准工具，理解延迟、吞吐与缓存状态的测量边界。随后从限制资源提出单变量假设，用完整案例说明验收、代价和回滚。
> **适用范围**：以普通文本生成和 Python benchmark 为主；最小使用入口见使用指南，异常定位见排障指南，调度、KV、执行与算子机制由各专题解释。
> **最近更新**：2026-09-08。承接使用页的调优闭环，按新基线核对工具行为并补充完整实验案例。

## 1. 先明确：这次要让谁更快？

假设服务已经能回答问题，但负载升高后，用户看到回答中途停顿。此时提高总 tokens/s 不一定解决问题：输出可能变短，失败请求可能增多，上一轮前缀缓存也可能让本轮少做计算。**调优的成果是对固定负载有效、满足质量和延迟要求的配置，而不是一组更大的吞吐数字。**

本页采用“负载合同 → 基线测量 → 限制资源假设 → 唯一调整 → 双重验证 → 回滚”的实验方法。这是分析者为提高可归因性提出的工程协议，不是源码承诺的普适最优算法。它要求候选配置改善读者真正关心的结果，并留下可以推翻假设的证据。

如果需要进一步理解为何同一配置在不同动态负载下改变瓶颈，可读 [[11_vllm_scheduler_analysis|Scheduler]] 的逐步预算与抢占，以及 [[12_vllm_kv_cache_management_analysis|KV Cache 管理]] 的容量、共享和回收；完成本页实验不以先读机制页为前提。

第一次运行模型应先完成 [[01_vllm_feature_optimizations_guide|vLLM 使用指南]]。遇到启动失败、持续报错或进程退出，先走 [[06_vllm_debugging_troubleshooting_guide|排障指南]]；不稳定的服务不能作为性能基线。

默认值也不能代替实验记录。当前顶层配置为 `optimization_level=O2`、`performance_mode="balanced"`，但 token/sequence budget 还会按使用入口、并行规模和模型约束解析；`throughput` mode 只把未显式指定的这两个 budget 翻倍。保留最终解析配置，查单个选项可使用 `vllm serve --help=max-num-batched-tokens`。相关解析入口见文末源码阅读路线。

## 2. 冻结负载合同，再声明质量与性能门

这里的 workload envelope 指本次结论适用的负载范围。先填写下表，再运行 baseline（基线）与 candidate（候选）；右栏是需要预先选择的条件，不是 vLLM 推荐阈值。

| 维度 | 必须冻结或记录 | 本轮验收要求 |
|---|---|---|
| 模型与语义 | 模型/revision、tokenizer、模板、dtype/量化、sampling、stop、输出上限 | 任务质量、结构和输出长度在预声明边界内 |
| 请求形状 | 实际输入/输出 token 分布、共享前缀比例、模态/媒体尺寸、LoRA 组合 | 代表目标生产窗口，不能只比较平均长度 |
| 到达过程 | 离线语料；或在线请求率、burstiness、并发上限、测量持续时间 | 实际发送时间线与目标相符，客户端没有悄悄限流 |
| 服务目标 | 成功率和 TTFT、TPOT、ITL、E2E 的目标分位数 | 所有硬 SLO 同时满足 |
| 吞吐目标 | 成功 requests/s、input/output/total tokens/s、SLO goodput | 区分成功吞吐与满足 SLO 的有效请求吞吐 |
| 环境 | vLLM commit/镜像、driver/runtime、GPU/互连、CPU/NUMA、主机内存、共租户 | 环境差异为零或作为独立实验因素记录 |
| 运行状态 | 冷启动、warmup、测量窗口、seed、重复轮次、prefix cache 策略 | baseline 与 candidate 使用同一状态恢复协议 |

### 2.1 正确性护栏不能从“改了哪个 flag”推导

先在固定的 canary corpus（小型回归语料）上保存基线输出、停止原因和任务评分，声明允许的差异，再测速度。即使只改 batching、graph 或并行布局，也不能把逐 token 一致写成框架保证；本页没有证明所有模型、后端和布局的数值可复现性。

- 如果业务要求严格复现，把逐 token/停止原因一致设为**本次验收门**，候选不满足就拒绝；这不等于其他配置有通用保证。
- 若业务允许数值或采样变化，预先指定任务分数下限、结构合法性、token/logprob 容差或统计检验，不能看完结果再放宽。
- 每轮检查空输出、意外截断、NaN/Inf、新错误类型、成功请求集合与失败率。随机 token 压测数据只能验证形状和运行行为，不能证明回答质量。
- 模板与缺省采样必须冻结：`generation_config="vllm"` 可避免采用模型仓库的生成配置，但具体请求参数仍应显式记录；聊天与纯文本输入的差别见 [[04_vllm_request_semantics_analysis|请求语义]]。采样和量化的适用边界分别见 [[18_vllm_sampling_structured_output_analysis|采样与结构化输出]]、[[21_vllm_quantization_analysis|量化]]。

## 3. 选择能回答问题的工具

| 问题 | 工具与测量对象 | 要固定什么 | 不能直接推导什么 |
|---|---|---|---|
| 固定语料多久处理完 | `vllm bench throughput`，离线推理吞吐 | 数据采样、输入/输出长度、seed、warmup、执行配置 | 在线排队、网络与网关容量 |
| 同样形状的一批请求多久完成 | `vllm bench latency`，重复运行固定 batch | batch size、shape、warmup iterations、执行配置 | 混合在线请求的尾延迟；引擎仍可能拆成多批执行 |
| 指定到达过程下能否满足 SLO | `vllm bench serve`，客户端发送到流式输出 | endpoint、数据、rate/burstiness/concurrency、sampling、状态策略 | GPU kernel 必然是唯一瓶颈 |

下列命令假定已安装本页基线对应的 vLLM，具备支持该模型的设备与容量，并把 `MODEL` 改为已冻结权重和 tokenizer 的本地目录。它们是完整的调用示例，不是本页实测记录。不同工具分开运行，避免互相争抢设备。命令中的小规模数字用于建立测量路径；正式结论还需真实语料和足够长的窗口。

### 3.1 离线吞吐与固定 batch 延迟

```bash
MODEL=/models/my-frozen-text-model
vllm bench throughput \
  --model "$MODEL" --generation-config vllm \
  --dataset-name random --input-len 512 --output-len 128 \
  --seed 42 --num-warmups 16 --num-prompts 256 \
  --no-enable-prefix-caching --output-json throughput.json
```

默认同步 `vllm` 路径先构造 LLM，再用 `seed+1` 采样 warmup requests，执行后才计量主请求。初始化和 warmup 不在返回的主请求耗时中；主计时包住 `LLM.generate`，也不是纯 kernel 时间。离线与在线工具共享 dataset dispatch，但**不代表采样语义相同**：本路径显式构造 `temperature=1.0`、`top_p=1.0`、`ignore_eos=True` 的 `SamplingParams`，不能把它当作线上 greedy 任务评估。

```bash
vllm bench latency \
  --model "$MODEL" --generation-config vllm \
  --input-len 512 --output-len 128 --batch-size 8 \
  --num-iters-warmup 10 --num-iters 30 \
  --no-enable-prefix-caching --output-json latency.json
```

`latency` 用固定随机 token batch，多轮调用 `LLM.generate` 至返回，报告整批延迟分位数；同样固定 `temperature=1.0`、`ignore_eos=True`，默认关闭 prefix caching。它检查输入加输出长度不超过 `max_model_len`。这两个工具的 warmup 次数单位不同：前者是请求数，后者是整批迭代数。离线 JSON 不等于全量证据：例如 throughput JSON 只有部分聚合字段，还要保留终端的输入/输出 token 统计与运行日志。

### 3.2 在线压测：到达过程和输出语义写进命令

先在一个终端启动服务：

```bash
MODEL=/models/my-frozen-text-model
vllm serve "$MODEL" --generation-config vllm \
  --host 127.0.0.1 --port 8000 --no-enable-prefix-caching
```

另一个终端运行客户端；`VLLM_USE_RUST_BENCH=0` 固定为本页核对的 Python benchmark 路径：

```bash
MODEL=/models/my-frozen-text-model
VLLM_USE_RUST_BENCH=0 vllm bench serve \
  --backend vllm --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model "$MODEL" \
  --dataset-name random --input-len 512 --output-len 128 \
  --request-rate 8 --burstiness 1 --max-concurrency 64 \
  --temperature 0 --ignore-eos --seed 42 \
  --num-warmups 16 --num-prompts 2400 \
  --percentile-metrics ttft,tpot,itl,e2el,client_queue_time,e2el_including_client_queue \
  --metric-percentiles 50,95,99 \
  --goodput ttft:1000 tpot:35 e2el:6000 \
  --save-result --save-detailed --result-dir bench-results \
  --result-filename baseline-42.json
```

这里的 `--ignore-eos` 强制压测输出长度，适合隔离 shape 成本；生产质量回放应恢复实际 stop/EOS 语义，baseline 与 candidate 必须一致。当前 `bench serve` 不再默认发送 `temperature=0`，未指定则由 server/model API 决定。聊天接口要改为匹配的 `--backend openai-chat --endpoint /v1/chat/completions`，并重新核对模板与实际 token 长度，不能只替换 URL。

### 3.3 到达、warmup 与缓存：三个常见污染源

**到达过程。** `request-rate` 默认 `inf`，表示不加到达间隔地发起请求；有限 rate 才构造间隔。`burstiness=1` 使用指数间隔，即文档所称 Poisson 到达；小于 1 更突发，大于 1 更均匀，且源码要求它为正。当前固定 rate、非 trace 路径还会缩放整组间隔，使末次计划发送时间对齐“请求数除以 rate”，所以不能把这一有限样本称为未经修正的生产 Poisson 轨迹。`max-concurrency` 控制客户端同时执行的请求数，排队后实际发送可能落后计划。记录发送时间、客户端 queue 和服务器 queue，不能只抄命令里的目标 rate。

**预热不等于独立数据。** 在线工具先用第一条主请求做 ready check，指定 warmup 后又重复同一个请求，等全部 warmup 完成才开始主计时。它不会覆盖所有请求 shape，也不会清除这些请求留下的缓存；不能把在线 warmup 描述为独立随机语料。需要覆盖的 shape/编译路径应由实验者补充预热，并把冷启动、compile/capture 和恢复耗时单列。

**固定 seed 不等于公平缓存状态。** 同一 server 重跑同一 seed 会重用 prompts；同轮共享前缀也可能命中。若目标不是缓存复用，可像示例一样在两边关闭 prefix cache；也可在相同重启/清缓存协议下成对复用语料。仅更换 seed 不能消除固定 system prompt 或真实语料的共享前缀。若目标就是缓存复用，应保留它并固定共享比例、预热和保留策略。官方文档还说明 `bench sweep serve` 在轮次间重置 server caches；本页未验证 sweep 的完整执行路径。

为兼顾重复性与覆盖面，可按 seed 42、43、44 配对做 A/B 测量，每对采用同一 seed 和状态协议，交错 A/B 次序。数据集 seed 的确定性测试证明的是采样数据相同，不是模型输出逐 token 必然相同。

## 4. 看清指标的分子、分母与计时边界

下面针对示例的 OpenAI completions 流式路径；不同工具或协议比较时，应核对测点，不能只比较同名指标。

| 指标 | 本页基线如何测量 | 必须保留的边界 |
|---|---|---|
| TTFT | 实际发送前开始计时，到首个包含 `choices` 的流式输出 | 可含特殊 token 对应的空文本；不等于用户首次看到非空文字 |
| ITL | 相邻包含 `choices` 的流式输出之间的间隔；成功请求的这些间隔合并统计 | 是流式输出片段间隔，投机解码一次可返回多个 token，不是每个 token 一条样本 |
| TPOT | 每个成功请求扣除 TTFT 后的时间，再除以首 token 以外的输出 token 数；然后按请求聚合 | 输出数不超过 1 时不进入 TPOT 分位数，但 goodput 将其 TPOT 当作 0 |
| E2EL/E2E | 实际发送前到最后一个包含 `choices` 的输出 | 不是接收最后一个 usage 或 `[DONE]` 的时间，也不含客户端 semaphore 排队 |
| 客户端排队 | 请求计划到达后，等待并发 semaphore 至实际发送的延迟 | 单列 `client_queue_time`；用户侧总等待需看 `e2el_including_client_queue` |
| 成功请求吞吐 | 成功数除以主 benchmark duration | duration 覆盖到达计划到全部任务收尾；失败不会进入成功分子，仍要报告全部尝试数和失败率 |
| token 吞吐 | 成功请求的 output tokens 或 input+output tokens，除以同一 duration | 优先用 API usage；缺失时重新 tokenize 文本可能略膨胀，不能当成逐 token 原始记录 |
| Request goodput | 同时满足**已指定** TTFT/TPOT/E2EL 门限的成功请求数，除以同一 duration | 阈值单位 ms；不自动检查 ITL、质量、失败率或客户端排队 |

命令的 `--goodput ttft:1000 tpot:35 e2el:6000` 是逐请求的三个联合门限，不是“P99 已通过”的开关。当前校验只接受 `ttft`、`tpot`、`e2el`，负值会被拒绝。硬 P99、ITL、质量和客户端总等待仍需独立验收。特别是并发上限太小时，服务器看似延迟良好，却可能把等待藏在客户端；goodput 不会自动替你发现这一点。

每组 baseline artifact 至少包含完整启动/压测命令、解析配置、镜像/模型 revision、硬件拓扑，dataset manifest 与实际 token 分布，seed、sampling、warmup、计时窗口，原始逐请求结果、success/error counts 和各分位数。还要保存 GPU memory 峰值与余量、GPU/CPU 利用率、主机内存、network/collective 时间、queue、preemption、prefix hit、fallback 与 engine health 时间线，以及 canary 评分、日志和 profiler trace 的路径。

指标必须来自同一窗口并确认进程/rank 标签，具体指标生命周期和故障信号见 [[27_vllm_observability_reliability_analysis|可观测性与可靠性]]。Profiler 会增加测量成本，应单开诊断轮，最终收益用相同、低干扰采集条件确认。

## 5. 先判断限制资源，再选一个变量

GPU 利用率低可能意味着输入供给不足、batch 太小、collective 等待或同步间隙，单个症状不能决定参数。下表是诊断假设的选择指南；机制是否启用、如何执行以及是否 fallback，必须回到对应专题和实际日志确认。

| 限制信号 | 本轮可检验的假设与变量族 | 什么观测会推翻假设 | 机制入口 |
|---|---|---|---|
| frontend CPU 饱和，GPU 间歇空闲 | 输入供给不足；只改 API/input-processing capacity 的一个因素 | CPU queue 不变或 GPU 空闲未减少，语义/TTFT 反而恶化 | [[17_vllm_serving_control_plane_analysis|Serving 控制面]]、[[19_vllm_multimodal_execution_analysis|多模态]] |
| TTFT 随 load 上升，decode 尚稳定 | 排队或 prefill 竞争；token budget、sequence budget、arrival/concurrency 中只选一个 | queue/prefill 时间不按预期变，或 decode 尾延迟越界 | [[11_vllm_scheduler_analysis|Scheduler]] |
| preemption、KV 余量低、OOM | 权重/KV/graph/临时 buffer 某项占用过高；先选 KV、上下文、量化、并行或 graph memory 一个族 | 对应占用不降，或质量/延迟代价越界 | [[12_vllm_kv_cache_management_analysis|KV Cache]]、[[21_vllm_quantization_analysis|量化]] |
| 小 batch 的 host launch gap 大 | 主机发起计算开销突出；只改一个 compile/graph 候选 | timeline gap 未收缩，或 startup/memory 超预算 | [[16_vllm_model_runner_v2_analysis|Model Runner V2]]、[[23_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]] |
| attention/GEMM/MoE/格式转换占主要计算时间 | shape/dtype 与后端不合适；backend、kernel、量化格式中只选一个 | 实际仍 fallback，或 kernel 加速未传递到 E2E | [[14_vllm_attention_backends_analysis|Attention Backend]]、[[24_vllm_fused_ops_and_kernels_analysis|融合 Kernel]] |
| decode 串行时间突出 | draft 成本可能小于节省的 target 计算；一个 speculative 候选 | acceptance、draft+verify 成本和 E2E 不支持收益 | [[20_vllm_speculative_decoding_analysis|投机解码]] |
| collective 时间高或单卡装不下 | rank layout 限制当前负载；一次 TP/PP/DP/EP/CP 布局变化 | 每 rank 容量/计算或通信没有预期变化 | [[22_vllm_distributed_inference_analysis|分布式推理]] |
| prefill/decode 资源需求可分开扩展 | 拆分实例可能增加有效容量；一个 KV 传输拓扑候选 | transfer、lease 与失败恢复成本吃掉收益 | [[26_vllm_disaggregated_kv_serving_analysis|分离式 KV Serving]] |

这些是待证伪的工程推断，不是看到症状就应启用的功能清单。出现 OOM、持续抢占或 health 失败时，应回到排障流程恢复基线；不把“仍然完成了一部分请求”当成优化成功。

## 6. 一张单变量实验卡

| 字段 | 本轮必须写清楚的内容 |
|---|---|
| 假设 | 若 X 是限制资源，改变 Y 后 Z 应改善，反证信号 Q 应如何变化 |
| baseline | 完整配置摘要、artifact 路径、测量窗口 |
| 唯一变化 | 一个参数；确实必须联合启用时，声明一个不可拆的候选配置，并记录全部变化 |
| 不变量 | workload envelope、硬件、代码、模型语义、warmup 与 seed/cache 策略 |
| 质量门 | canary、成功集合、任务质量/数值门限、stop/error 行为 |
| 性能门 | 主要指标、全部硬 SLO、goodput 和资源余量 |
| 反证 | 什么结果说明瓶颈假设错了，或收益没有得到证明 |
| 回滚 | 原始命令/配置、是否重启或清缓存、恢复状态的方法和负责人 |

独立 warmup 后交错运行 baseline/candidate，保留每轮原始数据和波动范围。如果噪声覆盖了收益，结论是“未证明改善”；不能只挑最优单轮。一个变量可以引起多个结果变化，例如 token budget 同时影响 TTFT、ITL 与显存，所有代价都应记录。

## 7. 完整案例：长输入让回答中途停顿

以下是**教学假设和教学数字，均非本机、模型或源码测试的实测结果**。目的是展示如何完成实验，而不是推荐 `4096` 为通用 token budget。

### 7.1 测量和事先验收

工作负载为单模型、普通文本、无投机解码；合成诊断输入 4096 tokens、输出 128 tokens，8 req/s、burstiness 1、client concurrency 64。生产任务另有冻结 canary corpus，使用生产模板、采样和 stop 规则。

预先写下验收门：成功率 100%，P99 TTFT 不超过 1000 ms，P99 TPOT 不超过 35 ms，P99 ITL 不超过 60 ms，P99 含客户端排队的 E2E 不超过 6000 ms，goodput 至少 7.5 req/s。GPU 余量至少 2 GiB，无持续 preemption，冷启动可接受上限 120 s。100 条带标准答案的 canary 至少答对 90 条，且较基线下降不超过 1 条；不出现新增结构错误、空输出或异常停止。这些门是案例协议，不是框架保证。

baseline 启动命令如下；这里固定 sequence budget，避免第二个 budget 随默认模式改变：

```bash
MODEL=/models/my-frozen-text-model
vllm serve "$MODEL" --generation-config vllm \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 \
  --enable-chunked-prefill --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --no-enable-prefix-caching
```

复制第 3.2 节客户端命令，将 `--input-len` 改为 `4096`，每轮用独立 result filename。A/B 对采用相同 seed，重复三对并交错次序，每次重启服务和执行相同预热。检查启动日志确认普通 chunked prefill 路径、无额外 scheduled-token 覆盖或 speculative 配置，保留解析配置；其他模型/路径不直接套用此例。

### 7.2 假设与唯一调整

假设 baseline 的诊断 trace 显示：长 prefill 所在步骤较长，decode 的片段间隔随这些步骤拉大；GPU 一直有工作，客户端排队可忽略，KV 余量充足且没有 preemption。由此提出可证伪假设：**减小每步 token budget 可减少长 prefill 对 decode 连续输出的干扰，代价可能是 TTFT 增加。** 这是有证据入口的推断，源码只证明 budget 参与调度上限，不保证它必然改善此 workload。

停止 baseline 服务后，以同一命令重启，只把 `--max-num-batched-tokens 8192` 改为 `--max-num-batched-tokens 4096`。不要同时改 `max-num-seqs`、缓存、并行度、sampling 或 arrival。若日志显示参数不被当前模型支持、没有走预期路径，或 trace 中长步骤/ITL 没变化，本轮假设不成立，停止继续堆参数。调度细节查 [[11_vllm_scheduler_analysis|Scheduler]]；本页只拥有选择和验证。

### 7.3 同时验收益、代价与质量

下表仍是教学数字。“范围”表示三轮中各轮结果的最小到最大值，不是置信区间。右列是在恢复原始 8192 配置、重新预热后得到的假设性复测。

| 观测 | A：8192 | B：4096 | 回滚 A |
|---|---|---|---|
| 成功率 | 100% | 100% | 100% |
| P99 TTFT | 780–820 ms | 900–950 ms | 790–830 ms |
| P99 TPOT | 38–40 ms | 30–33 ms | 38–41 ms |
| P99 ITL | 90–100 ms | 51–57 ms | 91–101 ms |
| P99 E2E（含 client queue） | 6100–6400 ms | 5100–5500 ms | 6100–6500 ms |
| goodput | 6.0–6.3 req/s | 7.6–7.8 req/s | 6.0–6.3 req/s |
| output throughput | 1000–1015 tok/s | 1000–1015 tok/s | 1000–1015 tok/s |
| GPU 最小余量 / preemption | 3 GiB / 0 | 3 GiB / 0 | 3 GiB / 0 |
| 冷启动 / canary 正确数 | 100 s / 92 | 102 s / 92 | 100 s / 92 |

这里候选的总输出吞吐几乎没变，却改善了流式停顿和有效请求吞吐；TTFT 增加是必须接受并记录的代价。还需检查原始 stop/error 行为、每条输出长度与 client queue，不能只凭“92 条正确”跳过质量门。诊断轮 trace 若同时显示长 prefill 步骤缩短、decode 间隔收缩，且回滚后恢复原表现，才更支持原瓶颈假设。

在上述教学结果中，B 通过预声明的质量、SLO、容量与运维门，因此可作为**这个负载下**的候选配置。换成短输入、长输出、真实共享前缀或更高到达率，结论仍需重测；三对实验也不能证明所有未来生产窗口都通过。

### 7.4 回滚动作与失败分支

触发器是任一硬门失守：例如 TTFT 升至 1100 ms、goodput 不足 7.5、canary 越界、OOM、持续抢占、queue 持续增长，或启动超过 120 s。回滚不是再试一个中间值，而是停止候选服务，恢复完整 baseline 命令与模型/config artifact，重启、执行相同 warmup，再用配对 seed 重测。

若回滚不能恢复原指标，应怀疑环境漂移、缓存/进程状态或测量条件变化，将本轮结论标为无效并转入诊断。若 B 有收益但业务不接受 TTFT 代价，结论是“候选不通过”；可另开新的实验卡，不能修改本轮验收门让它通过。

## 8. 完成时留下什么

候选晋级需要同时通过正确性与性能两道门；性能还包含因果、容量和运维边界，不能只看一个更快的数字。

| 验收面 | 通过条件 | 拒绝候选或立即回滚的触发器 |
|---|---|---|
| 正确性 | canary/任务质量、请求集合和 stop/error 行为处于预声明边界内 | 质量或结构越界、NaN/Inf、新错误类型、失败率越界 |
| SLO | 目标负载下全部硬延迟门、goodput 同时通过 | 任一硬门失守，即使总 tokens/s 上升 |
| 因果 | 目标路径确实启用，限制资源按预期变化，回滚能复现反向变化 | fallback、只有相关性、重复轮次不足以区分噪声和收益 |
| 容量 | GPU/CPU/主机内存、网络与 queue 有预留余量 | OOM、持续 preemption、queue 持续增长、health/engine 失败 |
| 运维 | 启动、compile/capture、缓存预热和恢复时间在预算内 | 冷启动或故障恢复超过预先约定的预算 |

交付一份能回答以下问题的实验记录：负载与硬 SLO 是什么，baseline 证据在哪里，哪个限制资源假设得到支持或被否证，唯一变化是什么，收益与代价是否都通过预声明门，回滚是否恢复原状态。配置改变 graph、cache、权重布局或进程内状态时，恢复完整 baseline 并按相同重启/缓存策略复测。

本页已静态核对源码与测试入口，未运行 GPU benchmark、下载模型或验证上述教学命令在特定硬件上的容量；依赖库、网络与设备 kernel 内部也未作为已验证实现展开。实际吞吐、质量、冷启动与恢复时间必须由部署环境提供运行证据。下面的测试可用于验证对应 CLI 和数据契约，但测试存在或使用 dummy 权重成功，不等于真实模型的性能/质量认证。

## 9. 紧凑源码阅读路线

以下路径均相对于页头固定的 `vllm-project/vllm`。同一行按“入口 → 决定行为 → 检验”串起一个阅读问题，避免把机制解释复制到本页。

| 阅读问题 | 已核对的源码 / 测试 anchors |
|---|---|
| 哪个 benchmark 实际运行 | `vllm/entrypoints/cli/benchmark/main.py::maybe_exec_rust_bench`、`BenchmarkSubcommand.cmd`：Rust 转交条件与 CLI dispatch；`vllm/benchmarks/serve.py::main_async`：采样参数、默认 warning 和在线入口 |
| 在线计划到达到请求完成 | `vllm/benchmarks/serve.py::get_request`、`benchmark`（含 `limited_request_func`）：间隔缩放、semaphore、第一条请求 warmup、`gather` 完成和主窗口 |
| 流式客户端测点 | `vllm/benchmarks/lib/endpoint_request_func.py::async_request_openai_completions`：发送、首个/末个 choices、ITL、usage 和失败输出 |
| 聚合和 goodput 边界 | `vllm/benchmarks/serve.py::calculate_metrics`、`check_goodput_args`、`add_cli_args`：成功集合、TPOT 单 token 特例、阈值和分位数选项 |
| 离线计时与数据语义 | `vllm/benchmarks/throughput.py::main`、`run_vllm`、`_run_vllm_requests`；`tests/benchmarks/test_throughput_cli.py::test_bench_throughput`：warmup seed、采样、同步返回、token 统计与 CLI 冒烟 |
| 固定 batch 延迟 | `vllm/benchmarks/latency.py::add_cli_args`、`main`；`tests/benchmarks/test_latency_cli.py::test_bench_latency`：缓存缺省、长度检查、warmup 与整批计时 |
| 在线 CLI 契约 | `tests/benchmarks/test_serve_cli.py::test_bench_serve`、`test_bench_serve_chat`：输入输出长度、基础请求与 chat backend/endpoint 配对 |
| seed 的证据范围 | `tests/benchmarks/test_random_dataset.py::test_random_dataset_same_seed`、`test_random_dataset_different_seeds`：数据采样确定性；`docs/benchmarking/cli.md` 的缓存警告和延迟定义提供文档边界 |
| 为什么保留解析配置 | `vllm/config/vllm.py::VllmConfig.optimization_level`、`VllmConfig.performance_mode`；`vllm/engine/arg_utils.py::EngineArgs._set_default_max_num_seqs_and_batched_tokens_args`：显式值、使用上下文和模型约束 |
| 案例变量确实影响什么 | `vllm/v1/core/sched/scheduler.py::Scheduler.__init__`、`Scheduler.schedule`：scheduled/input budget 和 chunked-prefill 分支；算法完整解释由 Scheduler 页拥有 |

## Related Pages

- [[01_vllm_feature_optimizations_guide|vLLM 使用指南]] — 先完成最小运行路径，再进入测量和调优。
- [[06_vllm_debugging_troubleshooting_guide|vLLM 排障指南]] — 服务出错或基线无法恢复时，按症状寻找故障边界。
- [[04_vllm_request_semantics_analysis|vLLM 请求语义]] — 固定模板、采样、停止和输出语义，避免把换题当成加速。
- [[11_vllm_scheduler_analysis|vLLM Scheduler]] — 解释 token/sequence budget、prefill 与 preemption 的机制。
- [[12_vllm_kv_cache_management_analysis|vLLM KV Cache 管理]] — 解释容量、前缀命中与缓存状态对实验的影响。
- [[16_vllm_model_runner_v2_analysis|vLLM Model Runner V2]] — 追踪执行组织与设备路径，验证优化是否真正进入目标实现。
- [[27_vllm_observability_reliability_analysis|vLLM 可观测性与可靠性]] — 为指标、fallback、engine health 与故障恢复提供机制依据。
