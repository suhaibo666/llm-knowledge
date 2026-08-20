# vLLM 快速使用与优化指南：跑通、测量，再按瓶颈选机制

> **源码基线**：`vllm-project/vllm@d66300a1baa7779c68c7dfa4e51eee2502b48017`
> **中心命题**：vLLM 没有脱离负载和 SLO 的“最优参数”。可靠路径是先固定模型语义与默认基线，再测 TTFT、TPOT、吞吐、容量和失败率，最后只改变能解释当前瓶颈的一层配置。

本页负责使用、实验设计和调优决策；底层原理分别由后续机制页拥有，避免在 quickstart 中再复制一遍调用链。

## 一、先定义什么叫“跑通”

一次可复现实验至少固定六类变量：

$$
W=(L_{\mathrm{in}},L_{\mathrm{out}},arrival,concurrency,prefix\ reuse,SLO)
$$

- vLLM commit/镜像、PyTorch、driver、设备型号与互连拓扑；
- 模型 revision、tokenizer/chat template、dtype、量化格式；
- 输入/输出 token 长度分布，而不是只写“1000 个请求”；
- arrival pattern、并发或 request rate、warm-up 和随机种子；
- TP/PP/DP/EP、KV 容量、调度 budget 与优化级别；
- 输出正确性、成功率、TTFT/TPOT/ITL 分位数、吞吐和显存峰值。

“进程启动且 curl 返回 200”只是连通性验证。chat template、`generation_config.json` 或量化路径不同，都可能让两个服务产生不同 token；这时速度不能直接比较。

## 二、安装与两条最小路径

当前 quickstart 基础条件是 Linux 与 Python 3.10–3.13；NVIDIA 环境推荐让 `uv` 根据 driver 选择 PyTorch backend，见 `docs/getting_started/quickstart.md:8-30`。

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

ROCm、XPU、TPU、Ascend 和 Apple Silicon 使用不同 wheel、镜像或硬件 plugin，不应把 CUDA 安装命令当成跨平台 ABI；当前入口与限制见 `docs/getting_started/quickstart.md:47-107`。

### 2.1 离线批推理

```python
from vllm import LLM, SamplingParams

conversations = [
    [{"role": "user", "content": "用三句话解释 continuous batching。"}],
    [{"role": "user", "content": "Paged KV Cache 解决什么问题？"}],
]
sampling = SamplingParams(temperature=0.0, max_tokens=128)

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    generation_config="vllm",
)
outputs = llm.chat(conversations, sampling)

for output in outputs:
    print(output.outputs[0].text)
```

`LLM.generate()` 会对一组 prompts 自动 batch；`vllm/entrypoints/llm.py:418-443`。但它不会自动给普通字符串应用 chat template；chat/instruct 模型应使用 `LLM.chat()`，其实现先把 messages 转成 prompt 再调用 generate，见 `vllm/entrypoints/llm.py:612-633`。

模型仓库的 `generation_config.json` 默认可能覆盖采样参数。为了让性能回归的生成语义稳定，上例显式使用 `generation_config="vllm"`；项目 quickstart 对这一行为的说明见 `docs/getting_started/quickstart.md:125-130`。

### 2.2 OpenAI-compatible 服务

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --generation-config vllm
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "解释 vLLM 的调度目标"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

`vllm serve <model>` 是当前 CLI 入口，默认地址为 `localhost:8000`；服务端 chat template、generation config 与 API key 行为见 `docs/getting_started/quickstart.md:203-229`。生产环境还需要独立完成鉴权、TLS、限流和网络边界，OpenAI-compatible 不等于自动具备完整网关安全。

## 三、先认识默认基线，不要重复堆 flag

| 默认项 | 当前基线 | 证据 | 调优含义 |
|---|---|---|---|
| optimization level | `-O2` | `vllm/config/vllm.py:435-439` | 生产基线已经包含 compile/graph/fusion 预设 |
| performance mode | `balanced` | `vllm/config/vllm.py:441-446` | 可对比 `interactivity` 或 `throughput`，但要重测 SLO |
| GPU memory utilization | `0.92` | `vllm/config/cache.py:80-87` | 是单实例预算，不是“实际必须占到 92%” |
| prefix caching | 开启 | `vllm/config/cache.py:107-108` | 开启不代表命中；收益取决于 token 前缀和 eviction |
| chunked prefill | 开启 | `vllm/config/scheduler.py:70-80` | 长 prompt 可被切分，但 token budget 仍决定干扰程度 |

优化级别表达“启动成本换稳态性能”：`-O0` 关闭 compile/CUDA Graph/fusion，`-O1` 使用较快编译与 PIECEWISE graph，默认 `-O2` 加更多编译区间、融合和 FULL_AND_PIECEWISE graph，当前 `-O3` 与 `-O2` 相同；`docs/design/optimization_levels.md:5-13,30-81`。

因此：

- 开发或排查 compile/graph 问题时用 `-O0` 建立 eager 对照，再试 `-O1`；
- 生产性能基线保留 `-O2`；
- 不要仅因为数字更大选择 `-O3`，当前 commit 并无额外预设。

## 四、指标决定优化方向

端到端时间可以先粗分为：

$$
T_{\mathrm{e2e}}=T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{frontend/output}}
$$

对一个输出 $N_{\mathrm{out}}>1$ 的请求：

$$
TPOT=\frac{T_{\mathrm{e2e}}-TTFT}{N_{\mathrm{out}}-1}
$$

`vllm bench serve` 在客户端测量 TTFT、ITL、TPOT，并说明 speculative decoding 下一个 stream output 可含多个 token，所以 ITL 与 TPOT 不一定相等；`docs/benchmarking/cli.md:114-140`。

还应使用 goodput，而不是只看总 tokens/s：

$$
Goodput=\frac{\#\{request\mid success\land TTFT\le S_{\mathrm{ttft}}\land TPOT\le S_{\mathrm{tpot}}\}}{duration}
$$

吞吐提高但 P99 超过 SLO、失败率上升，不能算有效扩容。

## 五、用官方 benchmark 建立 A/B 基线

先启动服务，再从另一个进程运行：

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <sharegpt-json> \
  --num-prompts 1000
```

命令形状和输出字段见 `docs/benchmarking/cli.md:63-111`。不要只跑一次：至少分 warm-up、稳定窗口和重复轮次，记录成功请求数及各分位数。

一个有效 A/B 实验只改变一组因果相关参数。例如比较 `-O0` 与 `-O2` 时，模型 revision、sampling、并发、输入顺序和 KV 容量应保持一致；比较量化时还要增加固定 prompts 的 token/logprob/任务质量检查。

## 六、按症状定位到配置层

### 6.1 TTFT 高：先区分排队还是 prefill

1. 拆出 queue time、tokenization 和 prefill time；
2. 若共享 system prompt 明显，检查真实 prefix-cache query/hit，而不是只看开关；
3. 长 prompt 阻塞 decode 时，对比 `max_num_batched_tokens`、长 prefill cap 与并发；
4. 若 prefill/decode 资源形态长期分离，再评估 P/D disaggregation；
5. 小并发交互负载可对比 `--performance-mode interactivity`，但同时检查吞吐。

设计依据：[[02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis|Scheduler token admission]]、[[02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis|KV/prefix ownership]]、[[02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis|分离式 KV Serving]]。

### 6.2 TPOT/ITL 高：确认 GPU 在等什么

1. 小 batch 下检查 CPU 调度、输入准备和 kernel launch 是否形成气泡；
2. 核对启动日志中的 runner、attention backend、graph mode、quant kernel 与 fallback；
3. 用 profiler 区分 attention、GEMM、MoE、collective 和 sampler；
4. draft 足够便宜且 acceptance 足够高时再试 speculative decoding；
5. TP collective 主导时，比较更少 TP、更多 DP 的布局。

设计依据：[[02_engineering/03_infer_frameworks/vllm/15_vllm_model_runner_v2_analysis|Model Runner V2]]、[[02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis|投机解码]]、[[02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]]。

### 6.3 吞吐低：再扩大 batch 和副本

`throughput` mode 只在用户未显式设置时把默认 `max_num_batched_tokens` 与 `max_num_seqs` 翻倍；`vllm/engine/arg_utils.py:2781-2806`。这提高候选 batch 上限，也可能增加 queue、KV 压力和尾延迟。

- 单副本能装下且请求足够多时，DP 通常是最直接的吞吐扩展；
- 模型单卡装不下时再用 TP/PP 或兼容量化解决容量；
- MoE 还要联合考虑 EP、DP attention、all-to-all backend 和负载均衡；
- 每次增大 token/sequence budget 都要带着 TTFT/TPOT P99 约束重测。

并行布局的 rank 与 collective 不变量见 [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|vLLM 分布式推理]]。

### 6.4 OOM/KV 容量不足：先给显存归因

| 占用来源 | 首选对照 | 主要代价 |
|---|---|---|
| 权重 | 兼容量化、TP/PP、CPU offload | 数值、collective 或 PCIe 带宽 |
| KV | 降上下文/并发、KV quant、offload | 容量、精度或传输延迟 |
| activation/graph | `-O0` 对照、缩小 capture sizes | 失去 graph replay 性能 |
| 通信/临时 buffer | 改并行布局、降低峰值 batch | 吞吐或更多副本成本 |
| 运行时余量 | 降 `--gpu-memory-utilization` | 可分配 KV blocks 减少 |

不要先把 `gpu_memory_utilization` 推到 1。它是 vLLM 实例的预算上限，无法消除其他进程、driver 或阶段峰值。

## 七、特性选择矩阵

| 机制 | 只有出现这个信号才优先评估 | 主要代价 | 原理页 |
|---|---|---|---|
| prefix cache | token 前缀重复且命中率可观测 | hash/refcount/eviction 与容量竞争 | [[02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis|KV Cache 管理]] |
| speculative decoding | decode 串行主导、draft 便宜、acceptance 高 | draft/verify 成本、显存与兼容限制 | [[02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis|投机解码]] |
| weight/KV quantization | 权重容量或访存带宽主导 | 数值误差、格式和 kernel 支持矩阵 | [[02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis|量化派发]] |
| compile/CUDA Graph | CPU/launch/small-op 开销明显 | compile/capture 时间、shape/地址约束 | [[02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]] |
| fused/custom kernel | profiler 指向访存往返或 launch 边界 | backend/shape 约束和验证成本 | [[02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis|融合算子与 Kernel]] |
| TP/PP/DP/EP/CP | 单卡容量、吞吐副本或 MoE/context 规模受限 | collective、负载不均与故障域 | [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|分布式推理]] |
| P/D 与 remote KV | prefill/decode 可独立扩展且传输可覆盖 | 路由、lease、传输和恢复复杂度 | [[02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis|分离式 KV Serving]] |

## 八、三套起步配置

以下是实验起点，不是生产处方。

### 8.1 正确性与故障定位

```bash
vllm serve <model> \
  -O0 \
  --generation-config vllm \
  --gpu-memory-utilization 0.85
```

用于排除模型仓库 sampling config、compile 和 graph；跑通后逐层恢复。

### 8.2 通用生产基线

```bash
vllm serve <model> \
  -O2 \
  --performance-mode balanced \
  --tensor-parallel-size <tp>
```

保留默认 prefix cache 与 chunked prefill，先用目标负载得到基线。

### 8.3 高并发吞吐候选

```bash
vllm serve <model> \
  -O2 \
  --performance-mode throughput \
  --tensor-parallel-size <tp> \
  --data-parallel-size <dp>
```

随后搜索 token/sequence budget、量化和布局，同时给 TTFT/TPOT P99 与失败率设置硬约束。

## 九、快速排障闭环

| 现象 | 第一对照 | 下一步证据 |
|---|---|---|
| 启动慢 | `-O0/-O1/-O2` | 权重加载、compile、autotune、graph capture 分段耗时 |
| GPU 利用率锯齿 | 小/大 batch 与 CPU profile | scheduler、input prepare、IPC、tokenizer |
| prefix cache 无收益 | 固定重复 token 前缀 | query/hit、块边界、eviction |
| speculative 更慢 | feature off/on | draft time、acceptance、target verify、batch size |
| TP 扩展差 | TP 与 DP 布局对照 | collective 占比、拓扑、单 rank GEMM 大小 |
| 偶发 OOM | `-O0` 与更低显存预算 | 权重/KV/graph/通信峰值分类 |
| flag 已开但性能没变 | 启动日志与 metrics | backend/kernel/runner/graph fallback |

生产观测、engine death、NaN 和 health/watchdog 路径见 [[02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis|vLLM 可观测性与可靠性]]。

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] — 全部机制页及三条推荐阅读路径。
- [[02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis|vLLM 系统设计原则与性能模型]] — 把负载、SLO、算力与显存约束放进同一模型。
- [[02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis|vLLM 引擎架构]] — 请求从 frontend 到 EngineCore/worker 的状态边界。
- [[02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis|vLLM Scheduler]] — token budget、continuous batching、chunked prefill 与抢占。
- [[02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis|vLLM KV Cache 管理]] — KV 容量、prefix hit、共享与 eviction。
- [[02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis|vLLM 可观测性与可靠性]] — 用 metrics、trace 与故障信号闭合调优因果链。
