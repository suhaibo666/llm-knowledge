---
title: "vLLM 使用指南：从安装到离线推理与流式服务"
---

# vLLM 使用指南：从安装到离线推理与流式服务

> **源码基线**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（`main` 快照，2026-09-07 UTC）
> **主题**：准备运行环境，用一个小型聊天模型完成离线推理，再启动服务、发送客户端请求并消费流式输出。最后解释返回字段、常用配置和默认值的解析边界。
> **适用范围**：以 Linux、单张 NVIDIA GPU、普通文本生成和 Python HTTP 前端为主线；性能实验见调优指南，故障定位见排障指南，内部机制见对应专题。
> **最近更新**：2026-09-08。重构为新读者使用入口，按固定源码与仓库内官方文档核验示例。

## 1. 先选调用方式：一段 Python，还是一个服务？

假设你想让模型回答“用一句话解释什么是 KV Cache”。第一步需要决定的是谁来调用模型，而不是先找一组优化参数。

- **离线推理**：你有一个 Python 程序和一组输入，调用 `LLM.chat()` 或 `LLM.generate()`，等待这批最终结果。模型由该程序加载，不需要 HTTP 服务。
- **在线服务**：你先用 `vllm serve` 启动常驻服务，再由客户端发送请求。模型留在服务端，客户端可以一次读取完整回答，也可以逐步读取流式回答。

两种入口复用推理能力，但输入组织、输出对象和等待方式不同。对聊天模型，消息中的 `role`、`content` 要先通过 **chat template（聊天模板）** 转成模型训练时认识的提示词格式；把同一句话直接交给纯文本续写接口，不能假定得到相同输入。这里先走聊天路径，随后对照纯文本路径。

本文统一使用官方 quickstart 中的 `Qwen/Qwen2.5-1.5B-Instruct`，把上下文长度设为 2048、单次输出上限设为 128 token。**token 是模型处理文本的单位，不等于一个汉字；128 是输出上限，不是承诺输出长度。** 这些值用于短文本入门示例，不是容量或性能推荐值。

> [!note] 验证范围
> 本文命令、Python 调用与返回字段已对照固定源码、仓库内官方文档和测试静态核验；未安装该环境、未下载模型、未执行 GPU 推理或 HTTP 请求，也未核验对应远程 wheel 当前是否可下载。下文描述预期检查点，不提供伪造的实跑输出。模型仓库 revision 和第三方依赖未锁定，因此示例不是完整的可复现实验清单。

## 2. 准备环境并确认安装版本

### 2.1 本文主线需要什么环境？

固定基线的安装文档列出 Linux、Python 3.10–3.13；NVIDIA 路线要求 GPU compute capability 至少 7.5。下面选择 Python 3.12，并假设已经安装 `uv`、可用的 NVIDIA 驱动，且能下载 Python 包及模型文件。

先在目标 GPU 机器运行：

```bash
nvidia-smi
uv venv --python 3.12 --seed
source .venv/bin/activate
```

`nvidia-smi` 应能列出目标 GPU；它展示的驱动 CUDA 能力不等于本地 CUDA Toolkit 的安装版本。预编译包还依赖匹配的 PyTorch/CUDA 二进制组合，因此官方文档建议使用干净环境。不要直接把现有训练环境的 PyTorch 和另一套 vLLM wheel 拼起来。

这不是当前 macOS 工作区的直接运行命令。固定文档另外列出 ROCm、XPU、TPU、Ascend 和 Apple Silicon 路线；后两者分别依赖 vLLM Ascend、vLLM-Metal 等独立插件，不能照搬本节 CUDA 安装命令。本页没有核验这些外部插件的安装或执行合同。

### 2.2 固定到本文源码对应的安装

官方文档支持把完整 main commit 放进 wheel 索引地址；下面按此机制固定到本文基线：

```bash
VLLM_GUIDE_COMMIT=199cb9b964822e59ab9b58d88e7be31eb419a2ae
uv pip install vllm \
  --torch-backend=auto \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_GUIDE_COMMIT}"
uv pip install openai
python -c 'import vllm, torch; print("vLLM:", vllm.__version__); print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
vllm serve --help=max-model-len
```

保留安装输出、包版本和 wheel 来源，并确认 `CUDA available` 为 `True`。固定基线文档的默认 CUDA wheel 变体是 12.9；`--torch-backend=auto` 是交给 `uv` 选择 PyTorch 索引的参数，不能据此保证所有驱动、平台和包版本都兼容。若索引没有目标 wheel，或装出的版本与预期不符，转到 [[05_vllm_debugging_troubleshooting_guide|排障指南]] 检查安装证据，不要悄悄换成 nightly 后仍宣称使用本文基线。

只想试发行版时，官方 quickstart 的简化命令是 `uv pip install vllm --torch-backend=auto`；它**不固定本文 commit**，应以实际安装版的帮助与文档为准。若要修改 C++/CUDA、使用不同二进制组合或既有 PyTorch，仓库文档另有 full build 路线，要求 GCC/G++ 至少 11.3；需要按那条路线配置工具链，本页不把源码构建混进最小使用步骤。

模型默认从 Hugging Face 下载，也可传本地模型目录。首次运行会经历下载、加载及引擎初始化，等待时间与磁盘占用取决于环境。仓库文档还支持在初始化前设置 `VLLM_USE_MODELSCOPE=True` 改用 ModelScope；这会改变下载来源，应记录下来，本文不展开其外部服务行为。

## 3. 跑一次离线推理，并读懂结果

把以下内容保存为 `offline_chat.py`，在刚才的环境运行 `python offline_chat.py`：

```python
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        generation_config="vllm",
        max_model_len=2048,
        dtype="half",
        gpu_memory_utilization=0.8,
    )
    params = SamplingParams(temperature=0, max_tokens=128)
    conversations = [
        [{"role": "user", "content": "用一句话解释什么是 KV Cache。"}],
        [{"role": "user", "content": "用一句话解释什么是批量推理。"}],
    ]
    results = llm.chat(conversations, sampling_params=params)
    for result in results:
        answer = result.outputs[0]
        print("request_id:", result.request_id)
        print("answer:", answer.text)
        print("input tokens:", len(result.prompt_token_ids or []))
        print("output tokens:", len(answer.token_ids))
        print("finished:", result.finished)
        print("finish_reason:", answer.finish_reason)
        print("stop_reason:", answer.stop_reason)


if __name__ == "__main__":
    main()
```

这里有两段**独立对话**，所以传入“消息列表的列表”。一次对话则直接传 `[{'role': 'user', 'content': '...'}]`。多个可共同执行的输入放在同一批，vLLM 会按内存约束自动组批；这不要求它们同时结束。

`LLM.chat()` 负责应用模板，离线执行循环持续推进引擎，收集完成结果，最后按输入顺序返回。函数返回才意味着本次离线调用取得了最终结果，进度条刷新不是流式业务接口。示例的 `temperature=0` 表示 greedy token 选择；`dtype="half"` 明确使用 FP16，`gpu_memory_utilization=0.8` 明确选择当前实例的显存预算比例。它们都不能证明某张卡一定装得下，也不能承诺跨硬件、并行布局或版本逐 token 相同。

### 3.1 `RequestOutput` 与候选回答的层次

| 读取位置 | 含义 | 最容易混淆的边界 |
|---|---|---|
| `results` | 每个输入对应一个 `RequestOutput` | 本例应有两个请求结果，不是一个回答的两块文本 |
| `result.prompt` / `prompt_token_ids` | 实际提示词及其 token ID；字段可为空 | 聊天输入包含模板结构，不能用原始用户句子的字数代替输入 token 数 |
| `result.outputs` | 该请求的候选输出列表；默认 `n=1` | `outputs[0]` 是第一个候选，不是第一个生成 token |
| `answer.text` / `token_ids` | 生成文本及输出 token ID | 回答文本与 token ID 不存在逐字符一一对应关系 |
| `result.finished` | 整个请求是否已完成 | `answer.finish_reason` 描述一个候选为何结束 |
| `answer.finish_reason` / `stop_reason` | 前者给出结束类别；后者可给出命中的停止字符串或 token ID | 普通生成中 `stop` 包括 EOS 或停止条件；`length` 表示长度限制，不代表答案完整；EOS 的 `stop_reason` 可以是 `None` |
| `answer.logprobs` / `cumulative_logprob` | 可选的 token 概率信息及累计值 | 未请求时可为 `None`，不应当作每次必有的评分 |

如果需要纯文本续写，在同一个 `llm` 实例中改为：

```python
results = llm.generate(
    ["The capital of France is", "A cache is useful because"],
    sampling_params=params,
)
```

`generate()` 不自动套聊天模板。这段演示的是接口差异；对当前 Instruct 模型，日常问答继续使用 `chat()`。已有格式化 prompt 的调用者应负责模板一致性，详见 [[03_vllm_request_semantics_analysis|请求语义]]。`SamplingParams` 可以整批共用，也可传与输入数量相同的参数列表；长度不匹配会触发校验错误，仓库测试覆盖了这一边界。

## 4. 启动服务，再发送完整响应请求

先让离线 Python 进程退出，再在同一 GPU 环境启动服务，避免两个示例争用显存。以下服务绑定本机回环地址，并把客户端可见的模型名设为 `qwen-demo`：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --served-model-name qwen-demo \
  --generation-config vllm \
  --max-model-len 2048 \
  --dtype half \
  --gpu-memory-utilization 0.8
```

在第二个终端检查：

```bash
curl -i http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

对本文普通模型服务，`/health` 返回 200 表示这次 Engine 健康检查通过；`/v1/models` 应列出 `qwen-demo`。随后必须真正发一条生成请求，才能验证聊天模板、推理和结果交付这条路径。仅能连上端口或列出模型不能证明生成成功，更不能证明生产容量成立。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-demo",
    "messages": [{"role": "user", "content": "用一句话解释什么是 KV Cache。"}],
    "temperature": 0,
    "max_completion_tokens": 128
  }'
```

请求中的 `model` 应与服务公开名称一致；指定 `--served-model-name` 后，不要仍把下载仓库名当成唯一的 API 名称。若没有设置别名，默认使用启动时的模型名。`base_url` 中 `/v1` 是 API 前缀，不应再在 Python 方法名里重复拼接路径。

同一请求也可以用 Python 客户端发送：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="qwen-demo",
    messages=[{"role": "user", "content": "用一句话解释什么是 KV Cache。"}],
    temperature=0,
    max_completion_tokens=128,
)
choice = response.choices[0]
print(choice.message.content)
print("finish_reason:", choice.finish_reason)
print("usage:", response.usage)
```

这里 `EMPTY` 是未启用服务鉴权时交给 SDK 的占位字符串。本地示例没有配置服务 API key；若启动时设置 `--api-key` 或 `VLLM_API_KEY`，客户端需要传匹配的 key，curl 需要相应的 `Authorization: Bearer ...`。本文服务只监听本机，跨机器访问还需明确监听地址与部署边界。

在线返回对象不是离线 `RequestOutput`。完整聊天响应里，`choices[i].message.content` 是候选回答，`choices[i].finish_reason` 是结束类别；`id` 用于关联本次响应，`model` 是服务公开名称。`usage.prompt_tokens` 统计模板处理后的输入，`completion_tokens` 统计生成 token，`total_tokens` 为二者之和。它们是 token 用量，不是延迟或 GPU 利用率；想保存输入/输出 token ID 或 logprobs，需要使用相应扩展字段，见请求语义专页。

多轮聊天也由客户端提交消息历史：保留上一轮 `assistant` 消息，再追加下一条 `user` 消息后重发 `messages`。不要把上次响应的 `id` 当成服务自动保存会话历史的句柄。

## 5. 逐步消费流式输出

完整响应要等回答结束。若界面希望尽早展示文字，同一个聊天请求添加 `stream=True`；客户端逐块读取 `delta.content`，而不是读取 `message.content`。下面保存为独立的 `stream_chat.py` 后运行：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
stream = client.chat.completions.create(
    model="qwen-demo",
    messages=[{"role": "user", "content": "用两句话解释 KV Cache 的作用。"}],
    temperature=0,
    max_completion_tokens=128,
    stream=True,
    stream_options={"include_usage": True},
)
parts = []
finish_reason = None
usage = None
for chunk in stream:
    if chunk.usage is not None:
        usage = chunk.usage
    for choice in chunk.choices:
        if choice.delta.content:
            parts.append(choice.delta.content)
            print(choice.delta.content, end="", flush=True)
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
print()
print("finish_reason:", finish_reason)
print("usage:", usage)
answer = "".join(parts)
```

本例使用默认 `n=1`；如果请求多个候选，要按 `choice.index` 分别累计。流中开始的块可能只有角色或空内容，结束候选的块带 `finish_reason`；请求 `include_usage=True` 后，通常还会有一个 `choices=[]` 的最终用量块，所以不能每次都无条件访问 `chunk.choices[0]`。SDK 负责消费 HTTP 的 Server-Sent Events（SSE）封装，直接用 curl 观察时可以添加 `-N` 并在 JSON 中设置 `"stream": true`，看到 `data: ...` 及最终 `[DONE]`。

**网络块不是模型 token 的计时单位。** 文本解码与协议封装影响可见粒度；一个块可能有空文本，也可能携带多个 token 的文本。普通成功路径应消费到流结束，并检查候选结束原因；若 SDK 抛错、连接中断或缺少预期完成信息，已经拼出的前半句仍只是部分结果。服务生成器出错时也可能发出错误事件后再发 `[DONE]`，因此单看到 `[DONE]` 不能替代错误检查。

仓库 `test_chat_streaming` 对比了固定测试配置下流式拼接文本与非流式文本，`test_chat_completion_stream_options` 验证了最终用量块的空 `choices`。这是测试覆盖范围，不是所有模型和并行配置逐字一致的保证。流中断后的定位路线见 [[05_vllm_debugging_troubleshooting_guide|排障指南]]；取消、输出收集和故障传播的内部合同由 [[23_vllm_observability_reliability_analysis|可观测性与可靠性]] 解释。

## 6. 常用配置：先分清在哪一层生效

下面只覆盖完成普通文本使用所需的选项，不是全部配置字段目录。Python 引擎参数通常使用下划线，CLI 使用连字符；请求采样参数则放在 `SamplingParams` 或客户端请求体中。进程已经启动后，修改客户端的 `temperature` 不会改变服务的显存预算或并行布局。

### 6.1 模型、模板与生成默认值

`model` 可以是仓库 ID 或本地目录；`revision`、`tokenizer`、`tokenizer_revision` 用于明确权重和 tokenizer 来源。需要复现实验时，连模型 revision、模板和依赖环境一起记录。`dtype` 默认 `auto`，按模型配置解析；`quantization` 还会参考 checkpoint 的量化配置，因此“没有写量化参数”不等于证明加载的是非量化权重。支持矩阵与数值边界见 [[17_vllm_quantization_analysis|量化]]。

聊天路径需要解析出与模型匹配的模板。当前 HF renderer 先检查显式模板，再在适用条件下检查 AutoProcessor、tokenizer，最后尝试内建 fallback；所以 tokenizer 中没有模板，不等于一定报错。所有来源都未解析出模板时，`safe_apply_chat_template` 才抛出 `ChatTemplateResolutionError`，此时应提供与该模型匹配的 `--chat-template` 文件或模板字符串。模板如何处理角色、特殊 token、工具和媒体，由 [[03_vllm_request_semantics_analysis|请求语义]] 负责。

`generation_config` 默认 `auto`，读取模型的 `generation_config.json`；也可指定目录，或用 `vllm` 不加载该文件。`override_generation_config` 再覆盖这一层的配置。常见采样字段需要区分两条路径：

- 离线不传 `sampling_params` 时，`LLM.get_default_sampling_params()` 从模型配置建立默认值。显式构造 `SamplingParams(temperature=0, max_tokens=128)` 时，未填写字段来自 `SamplingParams` 自身默认值，不能当作“自动继承模型推荐值”。例如其 `max_tokens` 自身默认是 16。
- 在线 `ChatCompletionRequest.to_sampling_params()` 对未填写的 `temperature`、`top_p`、`top_k` 等字段，先查服务默认，再使用协议默认；显式请求值通常优先。`stop_token_ids` 还会合并服务默认停止 token，不能用“所有请求字段简单覆盖”概括。

本文显式选择 `generation_config=vllm` 并固定输出上限和 temperature，目的是减少入门样例的隐含差异。准备真实应用时可以保留模型作者的推荐配置，但应知道实际生效值从哪里来。

### 6.2 三种长度不是同一个参数

`max_model_len` 限制**输入加输出**的上下文长度；未指定时由模型配置推导，显式 `-1` 或 CLI `auto` 另有按 GPU 可容纳长度选择的语义。离线 `SamplingParams.max_tokens` 和在线 `max_completion_tokens` 限制**每个候选的输出**。chat 协议仍接受已标为弃用的 `max_tokens`，两者同时存在时优先 `max_completion_tokens`。

在线模板和 tokenization 会检查为输出预留后的输入空间；普通请求超长会触发校验错误，而不是默认替你截断历史。输出预算还要经过 `get_max_tokens`，受剩余上下文、请求或服务默认、显式服务上限及平台限制共同约束。因此遇到 `length`，应先分清是输出预算用完还是上下文触限，再决定缩短历史或改变预算。离线与在线的提前校验路径也不同，不应承诺所有超长输入表现为同一种错误。

> [!contradiction] 模型默认与服务硬上限要分开
> 本基线 `ModelConfig.generation_config` 的字段说明笼统写 `max_new_tokens` 会成为服务全局上限；实际聊天实现及测试更细：`auto` 读取的模型 `max_new_tokens` 是缺省值，显式请求上限可以覆盖它；服务端通过 `override_generation_config.max_new_tokens` 或自定义 generation-config 目录设置的值，才进入 `override_max_tokens` 硬上限。以 `OpenAIServingChat.__init__`、`get_max_tokens` 和 `TestGetMaxTokens` 为当前执行证据。

### 6.3 显存、并行与批处理预算

`gpu_memory_utilization` 是当前实例用于模型执行器的显存比例，固定基线默认 **0.92**，有效范围大于 0 且不超过 1。它不是实时 GPU 计算利用率，也不是给同卡其他进程建立的全局隔离。`kv_cache_memory_bytes` 可直接指定每 GPU 的 KV 字节预算；正值走手动预算路径，不再按显存比例自动估算 KV。不要把这两个参数当作彼此叠加的容量承诺。容量计算与 KV 分配见 [[08_vllm_kv_cache_management_analysis|KV Cache 管理]]。

单卡先保留 `tensor_parallel_size=1`、`pipeline_parallel_size=1`。模型放不下一张卡时，TP、PP 用不同方式把模型执行分布到设备；DP 面向多份请求处理能力，MoE 还有关联的分片行为。增加卡数伴随通信与拓扑约束，不保证成比例加速，选择与有效组合见 [[18_vllm_distributed_inference_analysis|分布式推理]]。

`max_num_batched_tokens` 是调度步的 token 工作预算，`max_num_seqs` 是请求序列预算，都不是客户端输出上限。未显式填写时，`EngineArgs.get_batch_defaults()` 会参考硬件和离线/在线 usage context，CPU 路线还参考 world size；之后继续按模型长度、模态和其他配置调整。不要照抄某台机器启动日志里的数值当成全局默认。`enable_chunked_prefill`、`enable_prefix_caching` 的缺省启用也依赖模型能力，并非所有模型都相同；内部规则见 [[07_vllm_scheduler_analysis|Scheduler]] 与 KV 专页。

### 6.4 优化开关与实际解析值

固定基线的顶层默认是 `optimization_level=O2`（CLI `-O2`）与 `performance_mode=balanced`。优化级别为尚未设置的相关字段提供默认值；`throughput` 模式会放大用户**未显式指定**的 token/sequence budget，再经过其他约束。选 `interactivity` 或 `throughput` 表达的是偏好，不能当作性能验收结果。

`enforce_eager=True` / `--enforce-eager` 会关闭 `torch.compile` 和 CUDA Graph；默认是 `False`。它可用于建立执行对照，但本文最小样例未要求开启。编译与图执行的条件和启动成本见 [[19_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]]。

核对当前安装版时可以直接查询：

```bash
vllm serve --help=ModelConfig
vllm serve --help=max-num-seqs
vllm serve --help=all
```

帮助展示入口选项；运行后的最终配置还受硬件、模型能力与上述解析过程影响。调优时保存完整启动命令、实际版本和最终配置，不只保存一个改动 flag。

## 7. 跑通以后：去哪里解决下一个问题？

当离线结果、非流式回答和流式完成信息都能取得，本页的使用路径就闭合了。下一步按问题进入对应指南：

- **能跑，想证明更快**：[[04_vllm_performance_tuning_guide|性能调优指南]] 拥有 workload envelope、正确性 canary、benchmark 选择、warmup 与 measurement 分离、SLO/goodput、资源证据、单变量实验、验收和回滚。旧版本页的这些完整协议及配置族诊断表统一由该页承接。
- **启动失败、请求出错、流中断或卡住**：[[05_vllm_debugging_troubleshooting_guide|排障指南]] 从症状选择证据和定位步骤；指标、health、取消、fallback 和故障域怎样产生，继续读可观测性与可靠性机制页。
- **想理解服务为什么这样组织**：[[02_vllm_architecture_overview_analysis|架构概览]] 从普通并发请求介绍模块分工，再进入 Scheduler、Runner、采样、分布式等专题。

调优前至少保存模型/版本、模板、采样与 stop 配置、输入输出上限及一次成功响应；“可以执行”是后续实验的起点，不是已经满足业务正确性、延迟或容量要求。

## 8. 紧凑源码阅读路线

以下路径均相对固定基线的 `vllm-project/vllm`。测试名表示本次已打开的契约证据，不表示本次执行过测试。

| 要核对的问题 | 阅读入口与关键边界 |
|---|---|
| 环境、固定 commit wheel 和官方小模型例子 | `docs/getting_started/quickstart.md` 的 Prerequisites、Installation、Offline Batched Inference、Online Serving；`docs/getting_started/installation/gpu.cuda.inc.md` 的 requirements、Install specific revisions、Full build；`docs/cli/README.md` 的 help 查询 |
| 离线输入到最终结果 | `vllm/entrypoints/llm.py::LLM.generate` / `LLM.chat` → `vllm/entrypoints/offline_utils.py::OfflineInferenceMixin._run_completion` / `_run_chat` → `_run_engine`；`vllm/outputs.py::RequestOutput` / `CompletionOutput` |
| 离线普通与负向验证 | `tests/entrypoints/llm/test_chat.py::test_chat` / `test_multi_chat` / `test_llm_chat_tokenization_no_double_bos`；`tests/entrypoints/llm/test_generate.py::test_multiple_sampling_params` / `test_max_model_len` |
| HTTP 聊天与 SSE 交付 | `vllm/entrypoints/openai/chat_completion/api_router.py::create_chat_completion` → `vllm/entrypoints/openai/chat_completion/serving.py::OpenAIServingChat.create_chat_completion` / `chat_completion_stream_generator` / `chat_completion_full_generator`；`vllm/entrypoints/openai/chat_completion/protocol.py::ChatCompletionResponse` / `ChatCompletionStreamResponse` |
| 流式完成、用量与多轮 | `tests/entrypoints/openai/chat_completion/test_chat.py::test_single_chat_session` / `test_chat_streaming` / `test_chat_completion_stream_options`；`vllm/entrypoints/generate/base/protocol.py::StreamOptions`；`vllm/entrypoints/serve/engine/protocol.py::UsageInfo` |
| 采样默认与长度优先级 | `vllm/config/model.py::ModelConfig.get_diff_sampling_param`；`vllm/entrypoints/llm.py::LLM.get_default_sampling_params`；`vllm/entrypoints/openai/chat_completion/protocol.py::ChatCompletionRequest.build_tok_params` / `to_sampling_params`；`vllm/entrypoints/serve/utils/api_utils.py::get_max_tokens`；`vllm/renderers/params.py::TokenizeParams._token_len_check` |
| 默认与硬上限的反例 | `tests/entrypoints/serve/utils/test_api_utils.py::TestGetMaxTokens`；`tests/entrypoints/openai/chat_completion/test_serving_chat.py::test_serving_chat_should_set_correct_max_tokens` |
| 显存与最终配置 | `vllm/config/cache.py::CacheConfig.gpu_memory_utilization` / `kv_cache_memory_bytes`；`vllm/v1/worker/gpu_worker.py::Worker.determine_available_memory`；`vllm/engine/arg_utils.py::EngineArgs.get_batch_defaults` / `_set_default_max_num_seqs_and_batched_tokens_args` / `_set_default_chunked_prefill_and_prefix_caching_args`；`vllm/config/vllm.py::VllmConfig._apply_optimization_level_defaults` |
| 模板与健康边界 | `docs/serving/online_serving/README.md` 的 Chat Template（概括性说明）；`vllm/renderers/hf.py::resolve_chat_template` / `safe_apply_chat_template`（当前选择及失败边界）；`vllm/entrypoints/serve/instrumentator/health.py::health`；`vllm/v1/request.py::RequestStatus.get_finished_reason` |

## Related Pages

- [[02_vllm_architecture_overview_analysis|vLLM 架构概览]] — 从本页已跑通的调用路径转入模块职责与请求协作过程。
- [[03_vllm_request_semantics_analysis|vLLM 请求语义]] — 解释模板、tokenization、生成参数和输出转换的精确合同。
- [[04_vllm_performance_tuning_guide|vLLM 性能调优指南]] — 接管旧版使用与优化页的实验协议、指标、瓶颈选择和回滚方法。
- [[05_vllm_debugging_troubleshooting_guide|vLLM 排障指南]] — 按启动、请求、运行与流式故障选择诊断步骤。
- [[18_vllm_distributed_inference_analysis|vLLM 分布式推理]] — 在单卡入门之后解释 TP、PP、DP 等部署选项的约束与代价。
- [[23_vllm_observability_reliability_analysis|vLLM 可观测性与可靠性]] — 解释用量之外的指标、健康检查、取消和失败传播机制。
