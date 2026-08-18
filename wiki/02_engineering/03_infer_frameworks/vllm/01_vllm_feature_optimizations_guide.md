# vLLM 快速使用与优化指南 —— 从跑通到按瓶颈调优

> **代码基准**：vLLM `main@f4b161d7fca438bfe29509984759be1943a5aa88`（2026-08-18，`v0.27.2rc0-189-gf4b161d7fc`）
> **使用原则**：先用默认 `-O2 + balanced` 建立正确性和性能基线，再一次只改变一组参数。任何“推荐配置”脱离模型、硬件、输入/输出长度、并发与 SLO 都没有普适性。

---

## 一、安装与最小可用路径

当前官方 quickstart 的基础环境是 Linux、Python 3.10–3.13；NVIDIA 推荐用 `uv` 自动选择匹配的 PyTorch backend；`docs/getting_started/quickstart.md:8-30`。

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

其他平台不是简单复用 CUDA wheel：

| 平台 | 官方入口 | 注意点 |
|---|---|---|
| AMD ROCm | vLLM ROCm wheel / Docker | quickstart 当前说明 Python 3.12、ROCm 7.0、`glibc >= 2.35`；`docs/getting_started/quickstart.md:47-66` |
| Intel GPU | XPU backend 与官方镜像 | 版本与设备支持需查 XPU 安装页；`docs/getting_started/quickstart.md:68-75` |
| Google TPU | `vllm-tpu` | 独立插件包；`docs/getting_started/quickstart.md:77-86` |
| Ascend NPU | vLLM Ascend | 社区维护硬件插件，受 CANN/设备版本约束；`docs/getting_started/quickstart.md:88-95` |
| Apple Silicon | vLLM-Metal | 基于 MLX，需 MLX 优化模型；`docs/getting_started/quickstart.md:97-107` |

### 1.1 离线批推理

```python
from vllm import LLM, SamplingParams

prompts = [
    "用三句话解释连续批处理。",
    "Paged KV Cache 解决了什么问题？",
]
sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
outputs = llm.generate(prompts, sampling)

for output in outputs:
    print(output.outputs[0].text)
```

`LLM.generate` 不会自动给普通字符串套 chat template。chat/instruct 模型应使用 `llm.chat(messages, sampling)`，或先显式调用 tokenizer 的 `apply_chat_template`；`docs/getting_started/quickstart.md:112-200`。

另一个经常影响复现的默认值是 `generation_config.json`：模型仓库若提供它，vLLM 会采用模型作者推荐的采样参数。想固定为 vLLM 自身默认值，应构造 `LLM(..., generation_config="vllm")` 或服务端加 `--generation-config vllm`；`docs/getting_started/quickstart.md:125-130,217-220`。

### 1.2 在线 OpenAI-compatible 服务

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "解释 vLLM 的调度器"}],
    "max_tokens": 128
  }'
```

推荐入口是 `vllm serve`；直接运行 `python -m vllm.entrypoints.openai.api_server` 已被官方标记为 deprecated；`docs/design/arch_overview.md:54-79`。

## 二、默认配置已经做了什么

| 默认项 | 当前行为 | 源码证据 | 不应怎样理解 |
|---|---|---|---|
| optimization level | `-O2` | `vllm/config/vllm.py:401-405` | 不代表每个可选投机/量化特性都已开启 |
| performance mode | `balanced` | `vllm/config/vllm.py:407-412` | 不代表适合极低延迟或最大吞吐两个极端 |
| prefix caching | 开启 | `vllm/config/cache.py:107` | 命中率取决于块边界、请求前缀和缓存压力 |
| chunked prefill | 开启 | `vllm/config/scheduler.py:74` | token budget 不合适时仍会伤害目标 SLO |
| GPU memory utilization | 0.92 | `vllm/config/cache.py:80-83` | 不是越接近 1 越好，过高可能 OOM |
| async scheduling | 兼容时自动开启 | `vllm/config/vllm.py:1125-1217` | pooling、部分 speculative、Executor 或 ROCm DBO 组合会关闭/拒绝 |
| Model Runner V2 | 按模型与功能选择 | `vllm/config/vllm.py:614-695` | 不是全模型强制 V2 |

`-O0` 到 `-O3` 是“启动时间换运行性能”的预设：`-O0` 关闭编译和图捕获，`-O1` 使用较快编译与 PIECEWISE graph，默认 `-O2` 增加编译区间、融合和 `FULL_AND_PIECEWISE` graph，当前 `-O3` 与 `-O2` 相同；`docs/design/optimization_levels.md:5-13,30-81`。

实用选择：

- 开发、定位编译错误：先试 `-O0`，再用 `-O1` 检查 compile/graph 兼容性。
- 生产基线：保留默认 `-O2`。
- 不要因为名字叫 aggressive 就默认用 `-O3`；当前没有额外收益保证。

## 三、主要特性及其原理

| 特性 | 解决的瓶颈 | 原理 | 配置入口 | 深挖 |
|---|---|---|---|---|
| Continuous batching | 静态 batch 尾部气泡 | 每个 step 按 token budget 重组运行集合 | 默认核心机制 | [[11_vllm_scheduler_analysis]] |
| Chunked prefill | 长 prompt 独占一步、干扰 decode | 将 prefill 截成若干 token chunk 与 decode 混批 | `--max-num-batched-tokens` 等 | [[11_vllm_scheduler_analysis]] |
| Prefix caching | 重复 system prompt / 历史前缀反复 prefill | block hash 命中后复用 KV，refcount 保护共享块 | 默认开启 | [[12_vllm_kv_cache_management_analysis]] |
| Paged KV | KV 连续预留和碎片 | 逻辑 token 经 block table 映射到固定物理块 | 默认核心机制 | [[12_vllm_kv_cache_management_analysis]] |
| Speculative decoding | decode 串行依赖导致单步延迟高 | draft 一次提议多个 token，target 并行验证并接受前缀 | `--speculative-config` | [[20_vllm_speculative_decoding_analysis]] |
| Quantization | 权重/KV 容量与带宽 | 低比特存储配合专用 GEMM/attention kernel | `--quantization`、`--kv-cache-dtype` | [[21_vllm_quantization_analysis]] |
| Compilation / CUDA Graph | Python、kernel launch、小算子开销 | Dynamo/Inductor 融合，静态区间 capture/replay | `-O*`、`--compilation-config` | [[23_vllm_compilation_cudagraph_analysis]] |
| Custom / fused kernels | 中间张量和内存往返 | RMS/quant/activation/all-reduce/MoE 等垂直或水平融合 | 随平台/量化/优化级别派发 | [[24_vllm_fused_ops_and_kernels_analysis]] |
| TP/PP/DP/EP/CP | 单卡容量、请求吞吐、MoE 专家规模 | 模型、流水、请求、专家和上下文维度切分 | 并行 size 与 EP/CP flags | [[22_vllm_distributed_inference_analysis]] |
| Structured output | JSON/regex/grammar 正确性 | 每步生成合法 token bitmask，在采样前屏蔽非法 logits | 请求级 structured outputs | 本页 §3.1 |
| Multi-LoRA | 多租户适配器服务 | 同一基座按 token/request 映射 LoRA slot，分组低秩 GEMM | `--enable-lora` | 本页 §3.2 |
| KV connector / offload | P/D 分离、远端缓存、GPU KV 容量 | 在调度和 block 生命周期中插入 load/save/延迟释放 | `--kv-transfer-config` | 本页 §3.3 |

### 3.1 结构化输出

结构化输出不是生成结束后再校验 JSON。后端先把 JSON Schema、regex 或 grammar 编译为状态机，每个采样步根据当前状态生成合法 token bitmask，再对 logits 就地屏蔽。投机解码下还需要为 draft token 预留 mask、验证并在拒绝时 rollback。核心实现位于 `vllm/v1/structured_output/`，调度器入口为 `Scheduler.get_grammar_bitmask`，最终在 Model Runner 采样前应用。

这保证格式约束，但不能保证 JSON 中的业务事实正确；schema correctness 与 semantic correctness 是两层问题。

### 3.2 Multi-LoRA

Multi-LoRA 的关键不是“动态加载一个 adapter”，而是一个 batch 内不同请求可以选择不同 adapter。worker 管理 CPU/GPU adapter cache 和 LRU slot，Model Runner 构建 token-to-LoRA mapping，Punica 风格的 grouped shrink/expand GEMM 计算低秩增量。容量规划需同时看 `max_loras`、`max_cpu_loras`、rank、目标模块和 CUDA Graph 捕获策略。

最小服务启动形式：

```bash
vllm serve <base-model> \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16
```

### 3.3 KV connector、P/D 分离与卸载

`--kv-transfer-config` 是连接器抽象入口，可对接 LMCache、NIXL、Mooncake 等实现；当前可选依赖见 `requirements/kv_connectors.txt:1-8`。其本质是在 Scheduler/KV block 生命周期里插入远端 load/save、传输完成判定和延迟 free。

要区分三件事：

- **权重 CPU offload**：`--cpu-offload-gb`，让部分模型权重借助 CPU 内存扩容；`vllm/config/offload.py:16-26`。
- **KV offload**：把 KV block 放到 CPU 或分层存储。
- **P/D disaggregation**：prefill 实例产生 KV，decode 实例消费 KV，还需要路由、服务发现和容错。

它们都涉及“搬数据”，但移动对象、时序和瓶颈完全不同。

## 四、按瓶颈选择旋钮

### 4.1 TTFT 高

按顺序排查：

1. 拆开 queue time、tokenization 和 prefill time，确认是不是引擎外问题。
2. 共享前缀明显时检查 prefix cache hit，而不是只确认 flag 开着。
3. 长 prompt 干扰 decode 时调 `max_num_batched_tokens`、long-prefill 策略和 admission。
4. prefill 与 decode 资源形态差异很大时，再评估 P/D 分离。
5. 交互流量可试 `--performance-mode interactivity`，但必须重测吞吐和 P99。

### 4.2 TPOT 或 ITL 高

1. 确认 batch 是否太小，GPU 是否被 CPU 调度或 launch 饿住。
2. 检查实际优化级别、CUDA Graph mode、Model Runner V1/V2 和 attention backend。
3. 用 profiler 判断是 attention、GEMM、MoE、collective 还是 sampler，而不是盲开量化。
4. draft 足够便宜且 acceptance 足够高时评估 speculative decoding。
5. TP collective 成为主导时，比较更少 TP + 更多 DP 是否更好。

### 4.3 吞吐低

- `--performance-mode throughput` 会在用户未显式设置时放大默认 `max_num_batched_tokens` 与 `max_num_seqs`；`vllm/engine/arg_utils.py:2796-2801`。
- 增大 token/sequence budget 可能提高 GPU 占用，也可能恶化 TTFT、显存压力和尾延迟。
- 模型单副本能装下且请求足够多时，DP 通常是最直接的吞吐扩展；模型装不下才优先考虑 TP/PP/量化。
- MoE 还需联合选择 EP、DP attention、all-to-all backend 和负载均衡。

### 4.4 显存不足或 KV 容量低

依次判断是哪类占用：权重、KV、activation/CUDA Graph、通信 buffer 还是碎片。

| 问题 | 首选手段 | 代价 |
|---|---|---|
| 权重装不下 | 兼容量化、TP/PP、权重 offload | 数值差异、通信或 PCIe 带宽 |
| KV 不够 | 降 `max_model_len`、KV quant、offload、降低并发 | 上下文/并发、精度或传输开销 |
| 图捕获 OOM | 减少 capture size 或降优化级别定位 | 可能损失 decode 性能 |
| 运行时偶发 OOM | 降 `gpu_memory_utilization`，检查非 vLLM 进程和峰值 | KV block 数减少 |

## 五、三套起步配置思路

这些是“对比实验入口”，不是可直接复制的生产答案。

### 5.1 正确性 / 调试

```bash
vllm serve <model> \
  -O0 \
  --generation-config vllm \
  --gpu-memory-utilization 0.85
```

先排除模型仓库 sampling config、编译与 CUDA Graph，再逐项恢复。

### 5.2 通用生产基线

```bash
vllm serve <model> \
  -O2 \
  --performance-mode balanced \
  --tensor-parallel-size <tp>
```

prefix caching、chunked prefill 与兼容场景下的 async scheduling 已有默认行为，不必为“看起来优化很多”而重复堆 flag。

### 5.3 高并发吞吐候选

```bash
vllm serve <model> \
  -O2 \
  --performance-mode throughput \
  --tensor-parallel-size <tp> \
  --data-parallel-size <dp>
```

然后用目标流量逐步搜索 `max_num_batched_tokens`、`max_num_seqs`、量化和并行布局；同时给 TTFT/ITL P99 设置硬约束。

## 六、建立可复现 benchmark

启动服务后，可用官方 `vllm bench serve` 测客户端观察到的 TTFT、TPOT、ITL 与吞吐；`docs/benchmarking/cli.md:70-125`：

```bash
vllm bench serve \
  --backend vllm \
  --model <model> \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <sharegpt-json> \
  --num-prompts 1000
```

至少固定以下实验元数据：

- vLLM commit/version、镜像、PyTorch/CUDA/driver、GPU 型号与拓扑；
- 模型 revision、dtype、quantization、chat template 与 generation config；
- 输入/输出长度分布、并发或 request rate、warm-up、随机种子；
- TP/PP/DP/EP、token budget、sequence budget、KV 容量、optimization/performance mode；
- 成功率、TTFT/TPOT/ITL P50/P95/P99、output tok/s、GPU 利用率和峰值显存。

> [!warning] 先验证输出，再比较速度
> 更换量化、attention backend、speculative method 或 kernel 后，应在固定 prompts 上比较 token、logprob/acceptance、停止条件和结构化输出。一个更快但 chat template、generation config 或数值路径不同的服务，不是有效的 apples-to-apples benchmark。

## 七、快速排障地图

| 现象 | 第一检查点 |
|---|---|
| 启动很慢 | `-O0/-O1` 对比，区分权重加载、compile、autotune 与 CUDA Graph capture |
| GPU 利用率锯齿 | CPU profile、scheduler step、ZMQ/序列化、tokenizer、batch 是否太小 |
| prefix cache 没收益 | 请求 token 前缀是否真正相同、块边界、hash 命中、cache eviction |
| 投机反而更慢 | draft cost、acceptance、batch size、target verify kernel、兼容性 fallback |
| TP 扩展差 | collective 占比、NVLink/PCIe 拓扑、GEMM 变小、是否应改 DP/EP |
| OOM | 权重/KV/graph/通信 buffer 分类，检查 `gpu_memory_utilization` 与 capture sizes |
| 配置已开但行为不同 | 看启动日志确认 runner、attention backend、quant kernel、async scheduling 是否 fallback |

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]]
- [[10_vllm_engine_architecture_analysis|vLLM 引擎架构与请求生命周期]]
- [[11_vllm_scheduler_analysis|vLLM Scheduler 源码分析]]
- [[12_vllm_kv_cache_management_analysis|vLLM KV Cache 管理]]
- [[20_vllm_speculative_decoding_analysis|vLLM 投机解码]]
- [[21_vllm_quantization_analysis|vLLM 量化]]
- [[23_vllm_compilation_cudagraph_analysis|vLLM 编译与 CUDA Graph]]
