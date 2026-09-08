---
title: "vLLM 调试与故障排查：从症状到恢复验证"
---

# vLLM 调试与故障排查：从症状到恢复验证

> **源码基线**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（`main` 快照，2026-09-07 UTC）
> **主题**：用一次启动时 KV 容量不足的案例贯穿症状确认、环境采集、分层定位、处置和恢复验证，再给出日志、健康检查、指标、Profiler 与请求 Trace 的具体操作。最后按安装、模型输入、显存、编译和进程通信划分常见故障。
> **适用范围**：普通文本生成、Python HTTP 服务与 GPU 排障；首次跑通见使用指南，评测调优见性能指南，指标产生与故障传播机制见可观测性专题。
> **最近更新**：2026-09-08。按固定源码快照建立操作指南；案例与命令未在 GPU 服务环境实跑。

## 1. 先找到失败发生在哪一层

“服务没有回答”可能发生在模型尚未加载、请求无法转换成 token、KV 容量不足，或者设备已经执行但结果没有返回。排障的第一步是找出**最后一个已证实完成的阶段，以及紧接着缺失的证据**。例如加载权重结束只证明权重装载路径走完，既不证明 KV 初始化成功，也不证明一次生成能够完成。

本文采用的操作顺序是：保存原始症状和启动条件，找到最早的具体异常，用一个可撤销改动缩小范围，再通过新的请求结果验收。这是诊断方法上的分析建议。它的依据是当前实现有不同的校验与观察边界：容量检查在初始化阶段直接抛错，健康检查读取 Engine 的错误状态，完成请求才产生相应的请求统计。把所有现象归为“GPU 不够”会把这些边界混在一起。

先认识整体模块可读 [[02_vllm_architecture_overview_analysis|架构概览]]。本页只消费这些模块暴露的信号；信号如何产生、跨进程传递，以及异常怎样到达等待者，由 [[23_vllm_observability_reliability_analysis|可观测性与可靠性]] 解释。

## 2. 贯穿案例：权重加载后退出，HTTP 服务没有就绪

### 2.1 症状与复现条件

**以下是教学推演，未实际运行，也不是某型号 GPU 的容量结论。** 假设一个支持 8192 token 上下文的普通文本模型在独立测试实例中启动，权重已经加载，但日志随后报告：要服务一条最大长度请求，所需 KV 大于可用 KV，并给出可容纳长度的估计。客户端连接失败或一直等不到服务就绪。

案例假设模型本身可用、权重位于本地 `/models/text-model`，原始启动参数如下；实际使用时替换为故障实例原来的路径与参数。`diag-model` 是显式设置的服务名称，后续请求用同一个名称。

```bash
mkdir -p diag
vllm collect-env > diag/collect-env.txt 2>&1
python -m pip check > diag/pip-check.txt 2>&1

VLLM_LOGGING_LEVEL=DEBUG vllm serve /models/text-model \
  --served-model-name diag-model --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 \
  > diag/server-before.log 2>&1
```

服务命令占用当前终端；健康检查和请求在另一个终端执行。保留原来的完整启动命令、模型与 tokenizer 的 revision、设备拓扑、失败时间和所有 worker 日志。`collect-env` 会采集 vLLM、PyTorch、Python、驱动、编译/运行时 CUDA 等环境信息，但它不能替代启动参数和请求样本；`pip check` 也只检查安装包的依赖声明，不证明 CUDA 二进制兼容。

### 2.2 采证：找到具体容量错误，不停在最后一条 Engine 退出消息

```bash
rg -n 'Available KV cache memory|Free memory|To serve at least one request|No available memory|Traceback|Error' \
  diag/server-before.log
curl -sS --max-time 5 -i http://127.0.0.1:8000/health
```

先用搜索定位，再读匹配项上下文和所属进程。**案例中有判别力的证据**是 `_check_enough_kv_cache_memory` 报出的 `To serve at least one request with the model's max seq len`，以及所需 KV、可用 KV、最大长度估计；末尾笼统的 Engine 初始化失败只是上游失败的结果。若最早异常发生在下载、权重加载或设备初始化，本案例的处置就不适用。

此时 `/health` 可能连不上，不能预先认定一定返回 503；初始化没有完成时，HTTP 服务本身可能尚未可用。也不应等 `/metrics` 告诉你“启动 OOM”：它是服务上的观测端点，不能保证在启动失败期间可抓取。

### 2.3 定位：权重能装下，不代表最大上下文也装得下

普通 GPU 路径先由 Worker 估算模型执行占用与可留给 KV 的内存，再由 EngineCore 收集各 worker 的 KV 规格和可用内存，生成 KV 配置并检查容量。检查目标是能否容纳**至少一条 `max_model_len` 请求**，不是眼下尚未发送的短测试请求能否运行。因此，把首次请求的 `max_tokens` 改小并不能修复这个启动校验。

当前 `get_kv_cache_configs` 的容量校验还预留了 block pool 的 null block；不能把日志中的总块数全部当作请求可用块。测试用 512 token、每块 16 token 的简化规格证明：总计 32 块仍少一块可用空间，33 块才通过。这里只用这一边界解释错误，不展开 KV 分组和容量算法；其 owner 是 [[08_vllm_kv_cache_management_analysis|KV Cache 管理]]。

两个相似的显存报错需要相反方向的检查：

| 最早的明确错误 | 含义 | 下一步 |
|---|---|---|
| 启动时 `Free memory on device ... is less than desired GPU memory utilization` | 当前空闲显存少于实例申请的预算，尚未走到本案例的最大上下文校验 | 检查同卡进程，降低实例预算或释放经确认属于本任务的占用 |
| 本案例的“最大长度需要的 KV 大于可用 KV” | 已获得的 KV 预算不足以兑现最大长度 | 缩短承诺的上下文，或在实际有余量时增加预算 |

不能把“提高 `gpu_memory_utilization`”写成通用 OOM 修复：它可能越过第一种错误的预算边界。

### 2.4 处置：先只降低最大长度，再解释能力变化

假设本次错误估计可容纳长度约为 4096，且业务允许先用 2048 token 做恢复验证。保留模型、硬件、dtype 和内存预算，单独把 `--max-model-len 8192` 改为 `--max-model-len 2048`，重启测试实例：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve /models/text-model \
  --served-model-name diag-model --host 127.0.0.1 --port 8000 \
  --max-model-len 2048 --gpu-memory-utilization 0.8 \
  > diag/server-after.log 2>&1
```

4096 和 2048 都是教学输入，不能照搬到别的模型；以本次错误估计和新启动结果为准。`--max-model-len -1` 在当前实现另有自动适配路径，但会按各 worker 能容纳的长度调整配置，仍须检查最终上限。自动适配不等于保留原来的服务能力。

如果业务必须保留 8192，降低到 2048 只能证明容量方向，不能算业务故障已解决。后续要增加实际可用资源或改变模型/并行配置，再恢复原来的长度要求。此类方案的收益与成本由 [[04_vllm_performance_tuning_guide|性能评测与调优指南]] 负责。

### 2.5 验证：就绪、生成、统计更新是三个证据

在另一终端记录修改后的启动日志和一次真实请求：

```bash
curl -sS --max-time 5 -i http://127.0.0.1:8000/health
curl -sS --max-time 10 http://127.0.0.1:8000/metrics > diag/metrics-before.txt
curl -sS --max-time 120 -D diag/response.headers \
  http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"diag-model","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
  > diag/response.json
curl -sS --max-time 10 http://127.0.0.1:8000/metrics > diag/metrics-after.txt
rg '^vllm:(request_success_total|generation_tokens_total|e2e_request_latency_seconds_count)' \
  diag/metrics-before.txt diag/metrics-after.txt
```

检查 HTTP 状态、响应中的 `choices`、`usage` 和结束原因，确认请求确实完成，而非仅建立了连接。统计上检查同一 `model_name`/`engine` 的新计数；如果一次抓取尚未看到变化，等新的输出统计可见后再抓取，保留时间戳。不能把旧 gauge 的存在当作恢复证据，也不能把不同进程或不同重启周期的绝对 counter 值直接相减。

短请求完成之后，还需测试业务要求的最长输入加输出，以及实际并发下是否再次出错。缩短上下文的案例验收只证明新范围内恢复；未覆盖原来的 8192 范围，也未给出吞吐、延迟或答案质量保证。关闭 DEBUG 后重测，记录最终保留的参数和撤回的诊断开关。

## 3. 按问题选择诊断工具

### 3.1 日志：从低成本线索逐步升级

默认先保留完整 stdout/stderr，再临时开启 `VLLM_LOGGING_LEVEL=DEBUG`。需要更频繁的文本统计时可在重启命令前加 `VLLM_LOG_STATS_INTERVAL=1`，默认是 10 秒；它改变统计日志间隔，不提高所有指标的真实更新频率。多 API 进程的文本统计覆盖还有边界，详见可观测性专题。

下面每一项都用于单独的短复现，不应一起永久加入启动环境：

| 操作 | 要回答的问题 | 成本与证据边界 |
|---|---|---|
| `CUDA_LAUNCH_BLOCKING=1` | 异步 CUDA 错误更接近哪次调用暴露？ | 官方排障文档提供此隔离方法；同步会改变时序，不能拿结果当常态性能 |
| `NCCL_DEBUG=TRACE` | 初始化或 collective 停在哪个 rank/通信阶段？ | 保存每个 rank 的日志；NCCL 内部行为属于外部依赖，vLLM 日志本身不足以证明驱动或网络根因 |
| `VLLM_TRACE_FUNCTION=1` | 已启用追踪的 Python 线程最后执行到哪里？ | 跟随日志里的 `Trace frame log is saved to ...` 取文件；路径含实例、PID 和线程号，开销极大 |

函数追踪通过 `sys.settrace` 安装到调用它的线程，不是所有线程、C++ 和 GPU kernel 的完整时间线。它可以缩小停滞位置，不能由“最后一行函数名”直接断言那个函数有 bug。CUDA/NCCL 开关在这里依据同基线官方排障文档使用；没有检查对应版本的依赖实现。

### 3.2 Health 与 metrics：先检查可达，再检查是否前进

`GET /health` 的普通 Engine 路径调用 `AsyncLLM.check_health`，检测到 `EngineDeadError` 时返回 503。当前检查依赖已知的 engine-dead/output-handler 状态，**不执行一条测试推理，也不能保证及时发现所有活着但不前进的 hang**。Render-only 服务没有 Engine 时也返回 200，因此先确认探测的是哪一种进程。

`GET /metrics` 用于看同一时间窗里的请求与资源变化。可先抓原始输出，再按症状筛选：

```bash
curl -sS --max-time 10 http://127.0.0.1:8000/metrics > diag/metrics.txt
rg '^vllm:(num_requests_running|num_requests_waiting|num_requests_waiting_by_reason|kv_cache_usage_perc|num_preemptions_total|time_to_first_token_seconds|inter_token_latency_seconds|request_queue_time_seconds|request_prefill_time_seconds|request_decode_time_seconds)' \
  diag/metrics.txt
```

首 token 时间称为 TTFT，相邻输出 token 的间隔称为 ITL。直方图的 `_bucket`、`_sum`、`_count` 是累计观测；一次抓取不能直接给出当前窗口的 p99。若已有 Prometheus，用按 `model_name`、`engine` 分组的窗口增量查看分布；如果没有，先通过前后原始快照确认新请求确实被计入。

| 组合信号 | 排查方向，不是自动根因结论 |
|---|---|
| TTFT 与 queue 时间同时升高，waiting 增长 | 先检查请求压力和排队；`capacity` 指等待调度容量，`deferred` 还可能包含 KV transfer、LoRA 或 blocked 状态 |
| queue 稳定，prefill 或 decode 时间增长 | 转入执行、输入长度或后端变化；保留相同请求用于 Profiler 对比 |
| KV 使用率高且 preemption 增加 | 检查上下文/并发是否把缓存推向压力区；`kv_cache_usage_perc=1` 才表示 100%，不是 1% |
| counters 不再前进，health 仍为 200 | 联合进程日志、真实请求和线程/设备观测判断 hang；不能仅从旧值推出“没有流量” |

没有某条时序不等于值为零：先核对统计是否禁用、模型/engine 标签、端点是否正确、功能开关及版本。指标含义与生成时间归 [[23_vllm_observability_reliability_analysis|可观测性与可靠性]]，如何用它们建立性能对照实验归性能指南。

### 3.3 Profiler：当问题已经收敛到执行阶段

Profiler 用于隔离测试实例上的短窗口。它不是本案例 KV 初始化错误的首选工具，也不能把开启后的耗时拿来做正式 benchmark。当前 CLI 通过 `--profiler-config` 选择后端；只有配置了非空 profiler，服务才挂载 `/start_profile` 和 `/stop_profile`。

沿用案例的模型、服务名称及已验证长度，另开一次诊断启动：

```bash
vllm serve /models/text-model --served-model-name diag-model \
  --host 127.0.0.1 --port 8000 --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm-profile"}'
```

等待服务就绪，先发一条普通预热请求，然后执行：

```bash
curl -sS -i -X POST http://127.0.0.1:8000/start_profile
curl -sS --max-time 120 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"diag-model","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
curl -sS -i -X POST http://127.0.0.1:8000/stop_profile
```

确认 start/stop 返回成功，再到 **worker 所在主机/容器**的 `/tmp/vllm-profile` 收集产物。Torch wrapper 把数据交给 `torch.profiler` 的 trace handler；停止与导出可能耗时，不能把客户端断开当成已完成落盘。vLLM 源码证明了配置、start/stop 和导出回调的接法，不能保证外部 profiler 在任意硬件上成功捕获。

如只需短 worker 窗口，可增加 `delay_iterations`、`max_iterations`，Torch 模式配合 `ignore_frontend=true`，避免前端仍采集整个范围。这里的 iteration 是 Engine 执行步，不是请求数；有 delay 时 start 成功只代表会话已激活，尚未开始记录。测试覆盖了延迟启动、自动停止后再次启动的区别。

当前配置还支持 `cuda` 和 `proton`；后者有 NVIDIA CUDA 平台及禁用 CUDA Graph 的显式校验，不能把它当成所有设备上的同义替换。Nsight/Proton 的详细采集方案从同基线 `docs/contributing/profiling.md` 继续，执行图和编译机制见 [[19_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]]。

### 3.4 请求 Trace：把某次慢请求关联到区间

OTel 请求 Trace 与设备 Profiler 是两条独立路径。前者用于在 collector 中按请求 ID 看 queue、prefill、decode、e2e 等属性；后者用于看 CPU/GPU 执行窗口。已有可接收 OTLP gRPC 的 collector 时，在原启动命令中加入：

```bash
--otlp-traces-endpoint http://127.0.0.1:4317
```

这是追加到 `vllm serve` 的参数，不是独立命令；collector 地址应从服务进程可达，本机地址仅是单机示例。当前默认 exporter 协议是 `grpc`；若使用 `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`，需匹配 collector 的 HTTP 接收地址。OpenTelemetry 依赖未安装时，配置校验会报错。

发送同一条完成请求，在 collector 查找 `llm_request` span，关联 external request ID、token 数及各延迟属性。有上游 trace 时，保留请求的 trace context。只有需要更多模块细节时才加 `--collect-detailed-traces model`、`worker` 或 `all`，这些操作可能有阻塞与性能成本，并且要求先配置 endpoint。

span 经批量 exporter 发送；同基线测试会等待 `llm_request` 出现，再核验请求 ID、token 和延迟属性。刚完成请求却暂时查不到 span，并不能证明请求没有执行。持续挂住、尚未完成的请求也不能只依赖完成时的 `llm_request` span 定位；回到进程日志与短执行采集。

## 4. 常见问题的分层定位表

| 阶段与症状 | 最小隔离动作 | 怎样判读与下一 owner |
|---|---|---|
| 安装/import 报错，设备无法推断 | 保存 `collect-env`、`pip check`、最早 import traceback；以 DEBUG 重试平台检测 | 核对当前解释器、wheel/驱动/硬件组合；平台探测日志会保留插件检测异常。包依赖检查通过不证明二进制能加载 |
| 下载或加载权重长时间无进展 | 保留下载/磁盘/CPU 内存证据；本地已有完整模型时使用本地路径作对照；必要时仅在测试实例加 `--load-format dummy` | Dummy loader 跳过真实权重下载并初始化随机权重，但配置/tokenizer 等仍可能访问外部资源。它只能隔离真实权重路径，不能验证结果质量；模型加载由模型库专题接续 |
| `failed to be inspected` 或 architecture 不支持 | 先读模型 inspection 之前的 import 异常，再核对 checkpoint 的 architecture 与当前注册表 | 当前 registry 区分已登记但 inspection 失败、已移除、迁往外部插件和未知架构；不要把这几种都当成模型根本不受支持 |
| completion 可用但 chat 报模板错，或角色格式不符 | 保存原消息、tokenizer/revision 和实际模板；检查显式模板、processor、tokenizer 与内建 fallback 的选择，必要时通过 `--chat-template /path/to/model-template.jinja` 提供模型对应模板 | 当前 HF renderer 确实有 fallback 选择；不能仅凭 tokenizer 无模板就判定必报错。所有来源都未解析出模板时才抛模板解析错误；不要随意套别的模型模板来换取 HTTP 成功，请求语义见 `03` |
| 加载、profile、KV 校验或运行中 OOM | 先按最早错误划分阶段；KV 校验按本页案例操作；其他阶段减少对应工作量后重测 | 模型权重、执行峰值和 KV 预算不同。`kv_cache_memory_bytes` 显式指定后不再服从利用率预算，必须核对是否启用了这一分支；后续由 `08`、性能指南及模型专题解释 |
| 报错落在编译或 graph replay | 先单独加 `--enforce-eager` 重现；若恢复，再分别用 `--compilation-config '{"cudagraph_mode":"none"}'` 与 `'{"mode":"none","cudagraph_mode":"none"}'` 作对照 | 当前 `--enforce-eager` 同时关闭 torch.compile 与 CUDA Graph，成功只能定位到被关闭路径的组合。改变图模式会改变调度/执行表现，不能据此单独认定编译器 bug；下一 owner 是 `23` |
| traceback 位于 `torch/_inductor`、Triton 或 PTX 工具链 | 按官方 troubleshooting 中的最小 `torch.compile` CUDA 脚本脱离 vLLM 测试，保留同一环境的失败 | 如果最小脚本也失败，先收敛 PyTorch/Triton/工具链环境；外部依赖的具体根因还需对应源码或实验，不从目录名推断 |
| 初始化多进程报 bootstrap/spawn 错误 | 离线 Python 入口使用 `if __name__ == '__main__':` 保护创建引擎的代码，记录实际启动方法 | 当前 `_maybe_force_spawn` 在 CUDA 已初始化等条件下会切到 spawn；不要为了通过而盲目强制 fork |
| 多卡/多机初始化或生成 hang | 给客户端设诊断超时，记录所有 rank 最后进展与首次异常；核对日志中的通信 IP 和网卡，按官方独立通信脚本做对照 | `VLLM_HOST_IP`、`NCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME` 只在地址选择确有问题时试用；独立通信失败将范围收敛到依赖/环境，仍不自动证明硬件损坏；下一 owner 是分布式推理专题 |
| 运行中批量报 EngineDeadError 或 health 503 | 保存最早 worker/EngineCore 异常、受影响请求与重启前日志，再由部署策略恢复实例并重放验收请求 | 不循环重试一个已 DEAD 的 Engine；可恢复 fault-tolerance 状态与不可恢复死亡有不同合同，按 `27` 的状态边界处置 |

这些动作都是隔离变量的方法，不能把临时开关直接升级成生产默认值。发现执行退化但无明确功能故障后，转到性能指南保留原始负载、逐项实验和验收阈值；本页不重复调度、缓存、通信或编译算法。

## 5. 源码阅读路线与验证范围

以下路径均相对上述 vLLM 快照，按“用户入口 → 关键边界 → 验证”组织。相关机制页若尚未迁移，以各自页头基线为准。

| 读者问题 | 已打开的关键源码与测试 |
|---|---|
| 环境与诊断参数怎样生效？ | `vllm/entrypoints/cli/collect_env.py::CollectEnvSubcommand.cmd` → `vllm/collect_env.py::get_env_info`；`vllm/engine/arg_utils.py` 的 `--load-format`、`--profiler-config`、`--otlp-traces-endpoint` 注册；`vllm/envs.py` 的 `VLLM_LOGGING_LEVEL`、`VLLM_LOG_STATS_INTERVAL`、`VLLM_TRACE_FUNCTION` |
| 案例在哪个阶段拒绝启动？ | `vllm/v1/engine/core.py::EngineCore._initialize_kv_caches` → executor 取得各 worker 的内存估计 → `vllm/v1/worker/gpu_worker.py::Worker.determine_available_memory`；`vllm/v1/core/kv_cache_utils.py::get_kv_cache_configs` → `_check_enough_kv_cache_memory`；`vllm/v1/worker/utils.py::request_memory` 是更早的预算检查 |
| 容量边界有什么回归证据？ | `tests/v1/engine/test_init_error_messaging.py::test_kv_cache_oom_no_memory`、`test_kv_cache_oom_insufficient_memory`；`tests/v1/core/test_kv_cache_utils.py::test_check_enough_kv_cache_memory_reserves_null_block`、`test_auto_fit_max_model_len_reserves_null_block` |
| 200 与 metrics 分别证明什么？ | `vllm/entrypoints/serve/instrumentator/health.py::health` → `vllm/v1/engine/async_llm.py::AsyncLLM.check_health`、`errored`；`vllm/entrypoints/serve/instrumentator/metrics.py::attach_router`；`vllm/v1/metrics/loggers.py::PrometheusStatLogger`；`tests/v1/engine/test_async_llm.py::test_check_health`、`tests/entrypoints/serve/instrumentator/test_metrics.py::test_metrics_counts` |
| 工具何时实际开始/结束？ | `vllm/config/profiler.py::ProfilerConfig._validate_profiler_config`；`vllm/entrypoints/serve/profile/api_router.py::attach_router`、`start_profile`、`stop_profile` → `vllm/v1/engine/async_llm.py::AsyncLLM.start_profile`、`stop_profile`；`vllm/profiler/wrapper.py::WorkerProfiler.stop`、`TorchProfilerWrapper`；`tests/v1/worker/test_gpu_profiler.py::test_delayed_start`、`test_restart_after_max_iterations` |
| Trace 为何未立刻出现？ | `vllm/config/observability.py::ObservabilityConfig` → `vllm/tracing/otel.py::init_otel_tracer`、`get_span_exporter`；`vllm/v1/engine/output_processor.py::OutputProcessor.do_tracing`；`tests/v1/tracing/test_tracing.py::test_traces` |
| 模板、import 和 eager 怎样隔离？ | `vllm/renderers/hf.py::resolve_chat_template`、`safe_apply_chat_template`；`tests/renderers/test_hf.py::test_resolve_chat_template`；`vllm/model_executor/models/registry.py::_ModelRegistry._raise_for_unsupported`；`vllm/platforms/__init__.py::resolve_current_platform_cls_qualname`；`vllm/model_executor/model_loader/dummy_loader.py::DummyModelLoader`；`vllm/config/vllm.py::VllmConfig.__post_init__`；`vllm/utils/system_utils.py::_maybe_force_spawn` |
| hang 调试工具的覆盖是什么？ | `vllm/config/vllm.py::VllmConfig.enable_trace_function_call_for_thread` → `vllm/logger.py::enable_trace_function_call`；同基线 `docs/usage/troubleshooting.md` 的日志、通信、multiprocessing、torch.compile 章节，以及 `docs/contributing/profiling.md` 的工具操作说明 |

**验证边界**：本页已核对实际实现、CLI 注册与上述测试源码，未运行 vLLM/GPU 测试、下载模型、启动 collector 或生成 trace。测试中的 mock 容量输入只证明分支合同；它们不验证本页假设模型在特定显卡上的可运行性。生产验收还需要在目标设备、驱动、模型 revision 和并行拓扑上执行上述复现与恢复步骤。

## Related Pages

- [[01_vllm_feature_optimizations_guide|vLLM 使用指南]] — 建立首次安装、离线生成和在线服务的最小可用路径。
- [[04_vllm_performance_tuning_guide|vLLM 性能评测与调优指南]] — 在功能恢复后开展负载对照、性能归因与参数验收。
- [[02_vllm_architecture_overview_analysis|vLLM 架构概览]] — 帮助把日志里的前端、Engine、调度器和 Worker 放回整体请求路径。
- [[03_vllm_request_semantics_analysis|vLLM 请求语义]] — 解释消息、模板、输入校验和输出约定，接续协议与模板类故障。
- [[08_vllm_kv_cache_management_analysis|vLLM KV Cache 管理]] — 解释本页启动容量错误背后的 block 与缓存配置机制。
- [[19_vllm_compilation_cudagraph_analysis|vLLM 编译与 CUDA Graph]] — 接续 eager 对照之后的编译、捕获和重放机制定位。
- [[23_vllm_observability_reliability_analysis|vLLM 可观测性与可靠性]] — 定义指标、事件、Trace 和故障状态的产生、传播及解释边界。
