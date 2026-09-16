# Group A 复审（round 2）：vLLM 01 / 03 / 04 / 05 / 24

- 评审者：同一位独立 reviewer（未参与写作）
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（2026-09-07 00:54 UTC）。checkout `/Users/suhaibo/97-llm/vllm` 复审前后 `git rev-parse HEAD` 均为该 commit，未移动。
- 输入：`git diff` 工作区未提交改动（26 页 + `index.md` + `changelog.md`）；前轮报告 `docs/research/2026-09-15-vllm-domain-review/group_A_01_03_04_05_24.md`。
- 本轮实际执行的门禁：`check_links --strict`（0/0/0/0/0，452 页）、`check_math --changed --strict`（0/0）、`check_markdown --changed --strict`（0/0）、`check_assets --changed --strict`（0/0）、`python -m tools.mkdocs_site.cli build --changed`（**FAIL**，见 N1）。
- 未运行 GPU / HTTP / vLLM 测试。所有"已修"结论均回到源码复核，不接受新 prose 的自述。

---

## 0. 结论汇总（verdict rows）

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 01_vllm_feature_optimizations_guide（使用说明） | n/a | n/a | pass | none | n/a | 10/11 | **REJECT** | **使用说明：FAIL（完备性）**。两处事实性 P1（F01-1、F01-2）已按源码修正且证据链完整；但 F01-3 覆盖缺口与 §7 路由一字未动，01 仍只覆盖文本 chat/generate/stream，且不链接 14/15/16/17/24。4 个 P2 全部未修。新增 P2：R580 驱动门槛的适用范围被放大（N5） |
| 03_vllm_request_semantics_analysis | pass | pass | pass | transform（stop 缓冲）、layout（mm 排序） | pass | 14/14 | **PASS（作为分析）/ FAIL（作为使用说明）** | 新增 §2.3（response_format 归一）与 §3.3 token/logprob 返回表**逐格核对全部通过**，是本轮质量最高的改写。F03-2/3/5/6/7 全修。F03-1 未动（index 仍把 03 归"具体机制问题"，未重分类）；F03-4 未修 |
| 04_vllm_performance_tuning_guide | n/a | n/a | pass | none | n/a | 2/2 | **PASS（附 P2）** | 只修 F04-2（且修得准确）。F04-1、F04-3、F04-4 原样保留；仍不链接 24 |
| 05_vllm_debugging_troubleshooting_guide | n/a | n/a | pass | none | n/a | 2/2 | **PASS** | 两处 P1 owner 引用已修为真 wikilink，23 §6.3 锚点实测存在。F05-3/4/5 未修；§4 "安装/import 报错" 行仍不链接 24 |
| 24_vllm_extension_plugin_system_analysis | pass | pass | pass | ordered-rule、timing（Endpoint 两阶段，新增测试/真实分叉） | pass | 12/15 | **REJECT** | **feature: FAIL §3/§3.1/§4/§5.1/§6.1**（`vllm.logits_processors` 的加载亲缘与失败边界只写了 MRV1 runner，漏掉前端 InputProcessor 这一加载点，见 N2）。**并含 P0：§3.1 末尾跨页锚点损坏，mkdocs 构建直接中断（N1）**。F24-2 完全修好；F24-1、F24-3 PARTIAL；F24-4..8 全部未修 |

---

## 1. 前轮 findings 状态表

### 01

| id | 严重度 | 状态 | 证据 |
|---|---|---|---|
| F01-1 `generation_config=vllm` 语义 | P1 | **FIXED** | §6.1 改为"采样缺省不再取自模型文件 / EOS 仍读取"，并加 `[!contradiction]` + 链接 03 §2.2。源码复核：`vllm/config/model.py::ModelConfig.try_get_generation_config`（`if self.generation_config in {"auto","vllm"}` 仍读模型路径）；`ModelConfig.get_diff_sampling_param`（`config = {} if src == "vllm"`，`available_params` 恰为页面列出的 repetition_penalty/temperature/top_k/top_p/min_p/max_new_tokens 六项）；`vllm/sampling_params.py::SamplingParams.update_from_generation_config`（`if eos_ids: ... if not self.ignore_eos: self.stop_token_ids = list(eos_ids)`）——页面的 "未启用 `ignore_eos` 时用于补充停止 token" 逐字成立 |
| F01-2 CUDA 默认变体冲突 | P1 | **FIXED** | 新增 `[!contradiction]` + `/cu129` 备选安装块 + 路线表新行。源码复核：`vllm/envs.py::VLLM_MAIN_CUDA_VERSION` 默认 `"13.0"`（L91 与 L618–620 两处一致）；`.buildkite/release-pipeline.yaml` 主构建组 "Build CUDA 13.0 Python wheels"，12.9 在 `build-additional-wheels`（L60–68）；`requirements/cuda.txt::nvidia-cutlass-dsl[cu13]` / `humming-kernels[cu13]`；`docker/Dockerfile::ARG CUDA_VERSION=13.0.3`；`docs/getting_started/installation/gpu.cuda.inc.md` L40 自陈默认变体"Currently it is CUDA 12.9"。`${COMMIT}/cu129` 形态由同文档 L83 注释 "# add variant subdirectory here if needed" 与 L54 "The default variant (`cu129`) also has a subdirectory" 支持。远程索引未核验这点页面已显式声明 |
| F01-3 使用说明内容不完备 + 无能力路由 | P1 | **NOT_FIXED** | §7 与 Related Pages 逐行未变；无"能力 → 使用入口"表；01 全页 `grep -c 24_vllm_extension` = 0，也不链接 14/15/16/17。`changelog.md` 明确"页面边界与分类变更未擅自落地"，属主动推迟给 planning，非遗漏 |
| F01-4 服务默认 `stop_token_ids` 合并不可达 | P2 | **NOT_FIXED** | §6.1 原句保留。复核仍成立：`rg "default_sampling_params\s*="` 命中的全部赋值都是 `model_config.get_diff_sampling_param()`，该函数只能产出 6 个采样键（`vllm/config/model.py` L1747–1765），基线内无路径写入 `stop_token_ids`；`chat_completion/protocol.py` L698–707 的合并分支因此不可达 |
| F01-5 finish_reason 只写 stop/length | P2 | **NOT_FIXED** | §3.1 表 L130 原文未改 |
| F01-6 `--api-key` 只保护固定前缀 | P2 | **NOT_FIXED** | L198 未补 `GUARDED_PREFIX` 说明 |
| F01-7 指南里写渲染器四级选择链 | P2 | **NOT_FIXED** | §6.1 第二段原样 |

### 03

| id | 严重度 | 状态 | 证据 |
|---|---|---|---|
| F03-1 缺逐 API 可运行示例/默认值/输出/验证 | P1 | **NOT_FIXED** | 全页仍无 curl/SDK/Python 调用；Embedding、Score/Rerank、Transcription、Translation、Realtime、Responses、Anthropic、Tokenize、Render 均无示例。`index.md` 未重分类（03 仍在"请求与资源：已有具体机制问题"组）。同属推迟给 planning |
| F03-2 14 声明归 03 的 `response_format` 映射缺失 | P1 | **FIXED** | 新 §2.3 + 路线表新行。逐格核对：`vllm/entrypoints/generate/base/protocol.py::structured_outputs_from_response_format` —— `None`/`text` 原样返回（L141–142）✅；`json_object` → `{"json_object": True}` ✅；`json_schema` → `{"json": json_schema.json_schema}`（字段以 `alias="schema"` 声明，L97）✅；structural_tag → `json.dumps(model_dump(by_alias=True))` ✅；`replace(structured_outputs, **overrides)` 只覆盖同名字段（L166）✅。缺 `json_schema` 报请求错误由 `ChatCompletionRequest.validate_response_format` L776–787 证实 ✅；内层 `schema=None` → `{"json": None}` → `StructuredOutputsParams.__post_init__` `count < 1` 报错（L123–127）✅；六类互斥 + 至少一类（L106–127）✅；直接字段 json/regex/choice 与 named tool choice 的 pre-validator 由 `check_structured_outputs_count` L867–898 证实 ✅。"regex + json_schema 会形成两个约束并报错"：`replace` 触发 `__post_init__`，`count=2` → raise ✅ |
| F03-3 01 推给 03 的 token id / logprobs 无落点 | P1 | **FIXED** | §3.3 新增"请求字段 → 响应位置 → 约束"表 + 路线表新行。逐格核对：`logprobs=False` / `top_logprobs=0` 默认（protocol L220–221）✅；`top_logprobs>0 或 -1` 必须 `logprobs=true`（L857–860）✅；受 `max_logprobs` 校验（`sampling_params.py` L813–824）✅；"sampled token 可能使返回项比 top-k 多一个"由 `vllm/logprobs.py::append_logprobs_for_next_position` 的 `ranks = chain((rank,), topk_ranks)` + 注释证实 ✅；`prompt_logprobs` 默认 None、`stream=true` 不兼容（L836–839）✅；`echo=true` 未显式给值时沿用 `top_logprobs`（L710–712）✅；Completion prompt embeds 不兼容（`renderers/online_renderer.py` L287–289）✅；`logprob_token_ids` 需 `logprobs=true`、不支持 beam search、优先于自然 top-k（L812–821 + L733–737 `if self.logprobs and not self.logprob_token_ids`）✅；`return_tokens_as_token_ids` 无条件改写为 `token_id:<id>`（`generate/base/serving.py::_get_decoded_token` → `format_token_id_placeholder`，不看可显示性）✅、未显式设置继承 serving 默认（`chat_completion/serving.py` L1253–1256）✅；`return_token_ids` 流式/非流式落点与 protocol L415–420 docstring 逐字一致 ✅；reasoning 抑制由 L688–691（流式 `hide_stream_metadata`）与 L976–982（非流式 `suppress_metadata`）证实 ✅ |
| F03-4 服务默认 `stop_token_ids` 合并 | P2 | **NOT_FIXED** | §2.2 原句保留，同 F01-4 |
| F03-5 `InputProcessor` clone 是浅拷贝 | P2 | **FIXED** | §2.2 新句准确：`chat_completion/protocol.py` L756 `skip_clone=True,  # Created fresh per request`；`sampling_params.py::SamplingParams.clone` L786–791 `if self.skip_clone: return copy.copy(self)`；`input_processor.py` L356 `sampling_params = params.clone()`。页面还正确指出 `structured_outputs` 等嵌套对象仍共享 |
| F03-6 `check_stop_strings` 锚点 | P2 | **FIXED** | 路线表改为 `vllm/v1/engine/detokenizer.py::... / check_stop_strings`，与模块级函数一致 |
| F03-7 翻译入口缺失 | P2 | **FIXED** | §5.4 补 `/v1/audio/translations` 与 `task_type="translate"`，并加路线表锚点。源码：`speech_to_text/transcription/serving.py` L45 `task_type="transcribe"`、`translation/serving.py` L44 `task_type="translate"`；目标语言经 `base/serving.py` L269–271 `model_cls.validate_language(request.to_language)` 在前端校验 ✅ |
| 03 §2.1 Mistral/prompt-embeds 模板阶段 tokenization | suspicion | **仍未核实** | 页面该句未改；本轮也未核对 |

### 04

| id | 严重度 | 状态 | 证据 |
|---|---|---|---|
| F04-1 ready check 写成无条件发生 | P2 | **NOT_FIXED** | §3.3 L114 "在线工具先用第一条主请求做 ready check" 原样；`vllm/benchmarks/serve.py::benchmark` 仍是 `if ready_check_timeout_sec > 0:`，默认 0 即跳过 |
| F04-2 `generation_config="vllm"` 过宽 | P2 | **FIXED** | §2.1 改为"仅避免采用模型仓库的采样缺省 …… EOS 信息仍可能从模型的 `generation_config.json` 读取，并在未启用 `ignore_eos` 时补充停止 token"，与 F01-1 的源码结论一致；路线表加了对应新行 |
| F04-3 E2E 验收门与 `--goodput e2el` 口径不同 | P2 | **NOT_FIXED** | §7.1 L181 仍写"P99 含客户端排队的 E2E 不超过 6000 ms"，命令仍是 `e2el:6000`（不含客户端排队） |
| F04-4 §5 表未告诉读者如何确认 runner 代际 | P2 | **NOT_FIXED** | L150 行仍只指向 12 与 19，没有"先从启动日志确认 runner 版本" |

### 05

| id | 严重度 | 状态 | 证据 |
|---|---|---|---|
| F05-1 编译/graph 行 owner 写成 `23` | P1 | **FIXED** | 改为 `[[19_vllm_compilation_cudagraph_analysis\|编译与 CUDA Graph]]`，与 index、本页 §3.3 一致 |
| F05-2 引用不存在的 `27` | P1 | **FIXED** | 改为 `[[23_vllm_observability_reliability_analysis#6.3 受控恢复：只恢复可恢复的执行环境\|可观测性与可靠性 §6.3]]`；该标题在 23 L381 实测存在，锚点 slug 校验通过。建议的 `/fault_tolerance/status` 采证动作未加（并入 F05-4） |
| F05-3 `request_success_total` 含 abort/error | P2 | **NOT_FIXED** | §2.5 L96 的 `rg` 仍未按 `finished_reason` 过滤 |
| F05-4 补 `/fault_tolerance/status` 入口 | P2 | **NOT_FIXED** | 全页 `grep fault_tolerance` 无命中（除 §4 表那句中文） |
| F05-5 `rg` 依赖未说明 | P2 | **NOT_FIXED** | 无 `grep -nE` 备选 |

### 24

| id | 严重度 | 状态 | 证据 |
|---|---|---|---|
| F24-1 漏掉 `vllm.logits_processors` 组 | P1 | **PARTIAL** | 已补：§3.1 第 6 行、§3 图的独立分支、§4 新调用树、§5.1 新责任行、§6.1 新失败行、§6.2 新配置行、页头适用范围、Related Pages 加 14。已核实为真的部分：`LOGITSPROCS_GROUP = "vllm.logits_processors"`、`_load_logitsprocs_plugins` 直接用 `importlib.metadata.entry_points` 且不读 `VLLM_PLUGINS`、异常包 `RuntimeError` 不隔离、pooling 返回空 `LogitsProcessors()`、speculative 只留 `MinTokensLogitsProcessor`、TPU `_load_custom_logitsprocs` 返回 `[]`、`--logits-processors` flag 存在（`arg_utils.py` L943）、`VllmConfig._get_v2_model_runner_unsupported_features` L2599–2608 以 `entry_points(group=...)` 或 `model_config.logits_processors` 判定 MRV2 blocker、`build_logitsprocs` 唯一调用者是 MRV1 `vllm/v1/worker/gpu_model_runner.py` L771（MRV2 `vllm/v1/worker/gpu/model_runner.py` 全文无 logitsprocs）。**仍缺**：加载亲缘不完整（N2）、失败边界漏 ValueError 分支（N4）、跨页锚点损坏（N1） |
| F24-2 `VLLM_PLUGINS` 过滤范围写窄 | P1 | **FIXED** | §6.2 改为五组 + 新增运维警示段。源码：`load_plugins_by_group` 对任何 group 执行 `if allowed_plugins is None or plugin.name in allowed_plugins`；`rg load_plugins_by_group vllm/` 恰好命中五个调用点（general L87、endpoint L132、platform `platforms/__init__.py` L243、io_processor L59、stat_logger `loggers.py` L77）；"未设置时前四组全加载、Endpoint 整组不加载"由 `load_endpoint_plugins` 的 `if envs.VLLM_PLUGINS is None: ... return []` 证实 |
| F24-3 `collective_rpc("get_scheduler_config")` 无树内 worker 方法 | P1 | **PARTIAL** | §2 已补 `_FakeEngineClient` 标注、图中"测试/真实部署"分叉、以及 endpoint+general 配对合同（与 `docs/design/endpoint_plugins.md` L113–125 "Pairing with `vllm.general_plugins`" 逐条相符）；`worker_extension_cls` 路径经 `vllm/v1/worker/worker_base.py` L284–309 证实。**但新句的异常类型写错**，见 N3 |
| F24-4 路由 shadow 的文档主张未记录 | P2 | **NOT_FIXED** | §6.2 仍只写"属于 FastAPI/Starlette 的外部语义边界"，未列 `EndpointPlugin.attach_router` docstring 的 shadow 主张。补充证据：`docs/design/endpoint_plugins.md` L129 更明确 —— "A plugin's `attach_router` can register a path that collides with a core route and routes attached later win"，与本页"不能从调用顺序推广"的克制口径构成可记录的文档—证据冲突 |
| F24-5 平台探测缺最小例子 | P2 | **NOT_FIXED** | §3.3 仍只有文字规则 |
| F24-6 "覆盖旧工厂"/"实例化为 lora_resolvers" 措辞不准 | P2 | **NOT_FIXED** | §4 首段原文保留。`vllm/lora/resolver.py::_LoRAResolverRegistry.register_resolver(resolver_name, resolver: LoRAResolver)` 存的是**实例**，warning 原文即 "overwritten by the new resolver instance"；`OpenAIServingModels.__init__` 只 `get_resolver` 取引用 |
| F24-7 §1 前后序与 index 不一致 | P2 | **NOT_FIXED** | §1 仍以 23 为前序；`index.md` 的 24 行阅读依赖仍是"请求语义、模型库与Serving → 在线更新、可观测性"（23 为后续）。Related Pages 的 09 标签"模型与权重 ABI" 与 index 的"模型库与权重加载"也仍不一致 |
| F24-8 General 插件用途未点名 | P2 | **NOT_FIXED** | §3.1 General 行与 §3.2 仍只有"注册表或副作用""量化方法或设备候选"，未点名 `ModelRegistry.register_model` 也未链接 09/17/20 |
| 24 路由 shadow 的 Starlette 实际行为 | suspicion | **仍未核实** | 本轮未做外部依赖实验；但见 F24-4 的文档新证据 |

### E2E / 连贯性（前轮 §7 的 6 条）

| 前轮条目 | 状态 | 说明 |
|---|---|---|
| 1. "调用 API / 使用特性"无 usage owner | **NOT_FIXED** | 02 §5 的 9 个命令块仍留在架构页；01 未扩；无新 usage 页。changelog 声明推迟给 planning |
| 2. 入站合同断链（14→03、01→03） | **FIXED（内容）/ P2 残留** | 两处落点都补齐并逐条核实。残留：01 L200 仍是纯文本"见请求语义专页"，没有 wikilink/锚点指向 03 §3.3（N7） |
| 3. 05→19、05→23 §6.3 owner 错误 | **FIXED** | 两处都改成真 wikilink，23 §6.3 锚点实测有效 |
| 4. 24 与 14 的组数量矛盾 / `VLLM_PLUGINS` 误配不进 05 排障 | **PARTIAL** | 24 已枚举第 6 组且与 14 §4.3 结论一致；但 24→14 的链接锚点是坏的（N1），05 §4 "安装/import 报错" 行仍不链接 24，`VLLM_PLUGINS` 跨组静默过滤这一运维风险仍未进入 05 |
| 5. 术语一致性 | **PARTIAL** | 01/03/04 的 `generation_config="vllm"` 口径已统一；24 的前后序仍与 index 冲突（F24-7） |
| 6. 24 页头缺旅程位置 | **NOT_FIXED** | 页头只改了适用范围与更新说明 |

### 计数

| page | FIXED | PARTIAL | NOT_FIXED |
|---|---:|---:|---:|
| 01 | 2 | 0 | 5 |
| 03 | 5 | 0 | 2 |
| 04 | 1 | 0 | 3 |
| 05 | 2 | 0 | 3 |
| 24 | 1 | 2 | 5 |
| **合计** | **11** | **2** | **18** |

（前轮 P1 11 项：FIXED 7、PARTIAL 2、NOT_FIXED 2——两项 NOT_FIXED 都是页面边界/分类类，写作者主动推迟给 planning 批准。前轮 P2 20 项：FIXED 4、NOT_FIXED 16。）

---

## 2. 本轮新发现 / 回归

### N1（**P0**，回归）24 §3.1 末尾 — 跨页锚点损坏，mkdocs 构建直接中断

- 页面（24 L120）："采样处理顺序与扩展接口归 `[[14_vllm_sampling_structured_output_analysis#4.3 扩展接口与不支持项|采样与结构化输出 §4.3]]`"
- 源码/工具证据：`wiki/.../14_vllm_sampling_structured_output_analysis.md` L306 的标题是 `### 4.3 logits processor 的变体集合从哪里枚举出来`；全页无"扩展接口与不支持项"这个标题。`python -m tools.mkdocs_site.cli build --changed` 抛 `tools.mkdocs_site.wikilinks.LinkResolutionError: ...24_...md:120: ... missing target anchor`，构建在第一处错误即终止 —— 即 `.github/workflows/pages.yml` 部署的那套栈现在**构建不出来**。
- 为什么四项 T0 没抓到：`tools/check_links.py` 的 `stale_section` 只检查 wikilink 紧邻的 `§N` 纯文本形态（`SECTION_REF_RE` / `TOP_SECTION_RE`），不解析链接内部的 `#标题` 锚点；`check_markdown`/`check_math`/`check_assets` 也不解析锚点。CLAUDE.md 的 T0-conditional 已经写明"需要构建页证据时"要跑 mkdocs build，本轮改动新增了跨页锚点却没跑。
- 建议修复：`[[14_vllm_sampling_structured_output_analysis#4.3 logits processor 的变体集合从哪里枚举出来|采样与结构化输出 §4.3]]`。
- **同一次扫描发现的组外同类缺陷（构建是 all-or-nothing，必须一起修才能过）**：
  - `25_vllm_weight_transfer_online_update_analysis.md` → `13_vllm_serving_control_plane_analysis#3.1 EngineCoreClient 是一组实现，不是一个进程`；13 L330 的实际标题是 `### 3.1 进程、对象与状态归属`。
  - `25_vllm_weight_transfer_online_update_analysis.md` → `26_vllm_multiproc_executor_rpc_deepdive#6. Future、广播顺序与 FIFO 不变量`；26 L241 的实际标题是 `## 6. Future：响应没有请求 ID 时怎样保持配对`。
  - （扫描脚本按 `markdown.extensions.toc.slugify_unicode` 复现 mkdocs 的 slug 规则，遍历全 `wiki/` 的 `[[页面#锚点]]`，共 3 处坏锚点，全部在本次改动的 vLLM 页内。`changelog.md` 里的 `[[页面#锚点]]` 是示例占位，不计。）

### N2（**P1**，回归）24 §3 / §3.1 / §4 / §5.1 / §6.1 — `vllm.logits_processors` 的加载亲缘与失败边界写成"只在 MRV1 runner"，漏掉前端 InputProcessor 这一加载点

- 页面：§3 "普通 MRV1 生成路径**才**从 `importlib.metadata.entry_points` 加载候选"；§3.1 "普通 MRV1 generation 进入 `_load_logitsprocs_plugins` 后，才直接枚举并逐个 `EntryPoint.load()`"；§4 调用树只有 `GPUModelRunner.__init__ -> build_logitsprocs`；§6.1 失败后果写成"MRV1 runner / 引擎初始化失败"。
- 源码：另有一条与 runner 无关的加载链，落在**前端 process0 的 InputProcessor**：
  `vllm/v1/engine/input_processor.py::InputProcessor._validate_params`（L97 `params.verify(...)`）
  → `vllm/sampling_params.py::SamplingParams.verify`（L804 `self._validate_logits_processors(model_config)`，**无条件调用**，不看 `model_config.logits_processors` 是否为空）
  → `SamplingParams._validate_logits_processors`（L974–979）
  → `vllm/v1/sample/logits_processor/__init__.py::validate_logits_processors_parameters`（L231 `for logits_procs in cached_load_custom_logitsprocs(...)`）
  → `cached_load_custom_logitsprocs = lru_cache(_load_custom_logitsprocs)`（L221）
  → `_load_logitsprocs_plugins()`（枚举 + `EntryPoint.load()`，失败同样 `raise RuntimeError`）。
- 后果（页面因此说错三件事）：(a) 枚举/导入不是 MRV1 runner 独占，前端每条 generation 请求的参数校验都会走一次（lru_cache 后只有首次真正 import）；(b) 该组的导入失败可以表现为**API server 进程内的请求期 RuntimeError**，而不只是"引擎初始化失败"；(c) 该加载点也解释了 `_load_logitsprocs_plugins` 为什么要包 `guard_cuda_initialization()`——正是因为它会在不该初始化 CUDA 的前端进程里执行。注意 pooling 请求走 `PoolingParams.verify`，不经这条链；TPU 仍由 `_load_custom_logitsprocs` 的 `is_tpu()` 早返回拦住。
- 建议修复：§4 调用树加第二棵（`InputProcessor._validate_params -> SamplingParams.verify -> ... -> _load_logitsprocs_plugins`）；§3.1 的"才"改为"MRV1 runner 构造与前端参数校验两处各自加载"；§6.1 后果列加"前端请求校验期 `RuntimeError`"。附带：14 §4.3 说 `build_logitsprocs` 是"唯一的构造点"（构造点确实唯一，但**加载点不唯一**），两页最好互相点明这个区别。

### N3（P2，回归）24 §2 — 缺失 worker 方法的异常类型写错

- 页面："若 General 回调与 `WorkerWrapperBase.init_worker` 的 `worker_extension_cls` 注入都没有提供该方法，就会抛 `AttributeError`。"
- 源码：`vllm/v1/serial_utils.py::run_method` L501–507 —— `try: func = getattr(obj, method) except AttributeError: raise NotImplementedError(f"Method {method!r} is not implemented.") from None`。抛出的是 `NotImplementedError`，`AttributeError` 被吞掉。
- 说明：前轮报告 F24-3 里我自己写的也是 `AttributeError`，写作者照抄了。这是我的错，但页面现在与源码不符。
- 建议修复：改成"`run_method` 捕获 `getattr` 的 `AttributeError` 并抛 `NotImplementedError: Method 'get_scheduler_config' is not implemented.`"。

### N4（P2，回归）24 §3.1 说明段 + §6.1 新失败行 — "早返回"框架漏掉 pooling/speculative 的 ValueError 分支

- 页面：§3.1 "pooling 直接返回空集合，speculative 只保留 `MinTokensLogitsProcessor`"；§6.1 "pooling/speculative/TPU 的早返回不进入此分支"。
- 源码：`build_logitsprocs`（L185–217）在这两个分支上**先判显式项再早返回**：`if is_pooling_model: if custom_logitsprocs: raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)`；`if vllm_config.speculative_config: if custom_logitsprocs: raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)`。`custom_logitsprocs` 来自 `model_config.logits_processors`（`gpu_model_runner.py` L740–742），即 `--logits-processors` 的显式项。
- 后果：pooling 模型或开投机时配了 `--logits-processors`，得到的是**启动硬失败**（"Pooling models do not support custom logits processors." / "Custom logits processors are not supported when speculative decoding is enabled."），不是页面说的"早返回、不进入失败分支"。14 §4.3 已记录这两条，24 的失败边界表反而更松。
- 建议修复：§6.1 加一行（"pooling / speculative 且配置了显式 `--logits-processors`" → `ValueError` → runner 初始化失败），§3.1 说明段把"早返回"限定为"无显式项时"。

### N5（P2，新增）01 §2.1 — R580 驱动门槛的适用范围被放大，且漏掉 forward-compat 通路

- 页面（新增段）："本文普通 wheel 安装路径若选到 CUDA 13，需要 R580 或更新的 NVIDIA 驱动。"
- 源码：`docs/getting_started/installation/gpu.cuda.inc.md` L323–327（标题 "Running on Systems with Older CUDA Drivers"）把这段绑在**CUDA 13 容器镜像**上，原句是 "For CUDA 13 images, the minimum host kernel is Linux 4.15 when running normally because CUDA 13 requires an R580 or newer driver"，紧接着写 "Compatibility mode supports R535 and R570 host drivers"。
- 后果：读者会以为 R580 是硬前置，从而在 R535/R570 机器上直接放弃 CUDA 13 wheel，而文档明确给了 forward-compatibility 通路；反过来，"R580+" 这个数字的证据上下文是 image，页面把它搬到 wheel 路径上属于未标注的外推。
- 建议修复：改为"文档在 CUDA 13 镜像一节记录 CUDA 13 需要 R580+ 驱动，并说明 compatibility mode 支持 R535/R570；wheel 路径本轮未核验等价性"，并保留已有的"以 `torch.version.cuda` 为准"。

### N6（P2，新增）`wiki/changelog.md` 本轮条目 — 门禁陈述与实际不符

- 页面："四项 T0 严格门禁与空白检查通过；本轮改动涉及的 Mermaid 全部解析，新增或改写图完成渲染目检。"
- 事实：四项 T0 确实通过（我复跑：links 0/0/0/0/0、math 0/0、markdown 0/0、assets 0/0）；但本轮新增了三处跨页 `#标题` 锚点，按 CLAUDE.md 的 T0-conditional 需要 mkdocs 构建证据，而该构建现在**失败**（N1）。条目不应读起来像已取得构建页证据。
- 建议修复：改完 N1 的三处锚点后跑 `python -m tools.mkdocs_site.cli build --changed`，再据实改写这句（或明确写"未取得构建页证据"）。

### N7（P2，新增）01 L200 → 03 §3.3 的转交仍是纯文本

- 页面（01 §4）："想保存输入/输出 token ID 或 logprobs，需要使用相应扩展字段，见请求语义专页。"
- 现状：03 §3.3 现在正好有这张表（F03-3 已修），但 01 这句仍没有 wikilink，`grep "\[\[.*#" 01...md` 只命中 §6.1 那一处。
- 建议修复：改为 `[[03_vllm_request_semantics_analysis#3.3 最后一步恢复协议，而非只把 text 填进 JSON|请求语义 §3.3]]`（该标题在 03 L168 存在，slug 校验通过）。

---

## 3. 重跑的评审合同

### 01 / 04 / 05（guide 逻辑 8 要素）

| 要素 | 01 | 04 | 05 |
|---|---|---|---|
| 目标 / 适用范围 | ✅ | ✅ | ✅ |
| 前置条件 | ✅ 本轮加强（驱动/CUDA 变体、cu129 备选），但 R580 外推需收口（N5） | ✅ | ✅ |
| 最小示例 | ✅ 静态可照写；新增 cu129 安装块语法自洽（变量名 `VLLM_GUIDE_COMMIT` 前后一致） | ✅ | ✅ |
| 逐场景说明 | ❌ 仍只有 chat / generate / stream（F01-3） | ✅（评测三工具 + 案例 A/B） | ✅（KV 不足贯穿案例 + 五类故障表） |
| 参数默认值 | ✅ 且 §6.1 本轮改准；F01-4 残留一处不可达分支 | ✅；F04-1/F04-3 两处口径仍不准 | ✅ |
| 输出解读 | ⚠️ finish_reason 仍不全（F01-5） | ✅ | ✅ |
| 错误与限制 | ⚠️ 基本全转给 05；本轮未补 | ✅ §7.4 | ✅ |
| 下一步 | ❌ §7 未动，不链接 14/15/16/17/24（F01-3） | ⚠️ 不链接 24 | ⚠️ 不链接 24 |

**使用说明 verdict — 01：FAIL（完备性）。** 两处事实性 P1 已修，但"能力覆盖 → 使用入口"这个核心缺口一格未动；前轮覆盖表里标 **全域缺使用说明** 的 12 项（beam search、Responses、Anthropic、score/rerank、generative_scoring、translations、tokenize/detokenize、batch、invocations、tool calling、Docker、非 CUDA 平台）本轮全部未获得 owner 或链接。

**使用说明 verdict — 03：FAIL。** 新增的两节都是高质量的**机制/合同**内容（response_format 归一、token/logprob 返回合同），正确落在 03 的分析定位上，但没有让 03 变成使用说明：仍无任何可运行请求、无 per-API 默认值表、无响应样例与成功判据。`index.md` 也维持把 03 归入"具体机制问题"。结论与前轮一致：这是**页面合同/分类问题**，应由 `planning-codebase-analysis` 裁决（选项 A 新立"API 使用手册"页，或选项 B 原地重构 03），reviewer 不代为重划。作为分析页，03 本轮是五页里改得最好的一页。

### 24（base rubric + feature review）

- beat2 **pass**：§1 的被否决方案（"启动时一次性导入所有插件"）与推断标签保留；新增的 logits-processor 独立 loader 理由也规范地标了 **分析推断**。
- hop-walk **pass**：§5.2 路线表本轮全部改成 `path::symbol`，我逐条打开 —— `load_plugins_by_group / load_general_plugins / load_endpoint_plugins`、`AsyncEngineArgs.add_cli_args / EngineArgs.__post_init__`、`EndpointPlugin / attach_endpoint_plugins / init_endpoint_plugins_state`、`build_and_serve` / `build_app` / `build_and_serve_renderer`(L21)、`resolve_current_platform_cls_qualname`(L233) / `__getattr__`(L296)、`PluginWithIOProcessorPlugins`、`BaseServing._check_model` / `_LoRAResolverRegistry`(L44) / `LoRAResolverRegistry`(L88) / `OpenAIServingModels.resolve_lora`、`load_stat_logger_plugin_factories`(L77) / `AsyncLLM.__init__`、`LOGITSPROCS_GROUP / _load_logitsprocs_plugins / _load_custom_logitsprocs / build_logitsprocs` / `GPUModelRunner.__init__` / `_get_v2_model_runner_unsupported_features`、五个 env 变量、以及 8 个测试函数名（`test_plugin_loaded_when_allowlisted_and_task_matches` L90、`test_render_server_attaches_endpoint_plugins_with_no_engine_client` L181、`test_endpoint_plugin_end_to_end` L211、`test_platform_plugins` L10、`test_loading_plugin` L60、`test_stat_logger_plugin_integration_with_engine` L55、`test_filesystem_resolver` L34、`test_hf_resolver_with_multiple_repos` L53）**全部存在**。
- delete-code **pass**。
- figure **pass**：Endpoint 两阶段图新增测试/真实部署分叉（Q→J / Q→W→K），节点与 class 列表一致，`check_markdown --changed` 0/0；新的 discovery 图把独立 loader 画成旁路分支，与源码一致。两图都新加了 Figure spec 注释。
- algorithm-replay **pass**（adapter-x、Endpoint 两例仍可重放）。
- **feature: FAIL §3 / §3.1 / §4 / §5.1 / §6.1** —— 变体枚举依据已经补全（六个 group 都在），但新变体的 **function-point 合同本身不完整**：
  - 加载亲缘（谁调用、在哪个进程）写错为 MRV1 runner 独占（N2）；
  - 失败边界表漏掉 pooling/speculative + 显式项的 `ValueError` 分支（N4）；
  - 与 owner 页 14 的交接链接是坏锚点（N1）。
  相比前轮，(a) 变体枚举依据已修好、(b) `VLLM_PLUGINS` 作用域已修好、(c) 最小示例的集成边界已修好（仅异常类型错，N3）。

---

## 4. 剩余 blockers（按处理顺序）

1. **N1（P0）**：修 24 L120 的锚点，并连同 25 的两处一起修，然后跑 `python -m tools.mkdocs_site.cli build --changed` 取得构建证据。在此之前整个域**不可合并**。
2. **N2（P1）**：24 补前端 InputProcessor 这条加载链与它的失败后果；顺带与 14 §4.3 对齐"构造点唯一 ≠ 加载点唯一"。
3. **N3 / N4 / N5（P2）**：三处新 prose 的事实收口。
4. **N6（P2）**：changelog 的门禁陈述按实际改写。
5. **planning 裁决（P1，非页面缺陷）**：F01-3 / F03-1 / E2E 第 1 条 —— "调用 API / 使用特性"这一步至今无 usage owner。写作者已按 CLAUDE.md 把它上交 planning，这是正确处理，但缺口仍然开着，01/03 的使用说明 verdict 会一直是 FAIL 直到 planning 落地。
6. **未修 P2 共 16 项**（F01-4..7、F03-4、F04-1/3/4、F05-3/4/5、F24-4..8）：其中 **F04-1（ready check 无条件发生）与 F04-3（两个 6000 口径不同）** 会直接误导读者照抄命令，建议优先；**F01-4 / F03-4** 是同一句不可达分支，两页一起改。
