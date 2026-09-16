# Group E 第 3 轮独立验收：11 MRV1 / 12 MRV2 / 19 编译与 CUDA Graph

- 评审人：独立 reviewer（非作者）。**只报告，未修改任何文件。**
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，checkout `/Users/suhaibo/97-llm/vllm`，HEAD 未移动（`git log -1` 复核：`199cb9b9… 2026-09-06 17:54:32 -0700` = 2026-09-07 UTC；工作区仅有未跟踪的 `artifacts/`）。
- 方式：`git diff -- <page>`（三页仍未提交，diff 为累计改动 +209/−94）。上一轮的每条未决项按**源码**重判（不按新措辞判），新增/重写文字的每条断言逐条回源码，新算例自行重放。
- 未运行 vLLM、GPU、分布式或故障注入实验。
- 协调者已确认：scoped mkdocs 构建通过，两条指向 16 的 anchor INFO 是别名锚点造成的误报——本报告不重复该项。

---

## 1. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 11_vllm_model_runner_v1_analysis | pass | pass | pass | timing | pass（两棵新 Core 树与时序图重放一致） | 9/9 | **ACCEPT** | feature: pass §1.4／§1.5；F15 已修（15 §7.7）；仅余 P2 N1（「未自定义 Scheduler」仍挂在 async 解析与 `step()` 条件上） |
| 12_vllm_model_runner_v2_analysis | pass | pass | pass | transform, timing | pass（`max_concurrent_batches` 四行、Mamba partial tail 三条件、sample 树顺序全部重放一致） | 11/11 | **ACCEPT WITH FIXES** | feature: pass §1.6；X-1、X-2、F8、F16 全部 FIXED；**新增 1 条 P1**（§3.2 调用树把 `_set_active_loras` 排在 forward 之后）+1 条 P2 |
| 19_vllm_compilation_cudagraph_analysis | pass | pass | pass | transform, layout, timing | pass（SP padding 3→4、DP min/max 协商、capture 网格与 ⑭ 窄化全部重放一致） | 14/14 | **ACCEPT** | feature: pass §7.3；上一轮三条 P1 阻塞（图 1、图 2、`compile_sizes` 范围）**全部 FIXED**；F6 仍 PARTIAL（缺 LoRA case 两代对照）；另 3 条 P2 |

Feature 复判：

- 11：`feature: pass`。新增的两棵 Core 树（`step_with_batch_queue` 与 `step` 对照）逐行核对 `vllm/v1/engine/core.py::EngineCore.step / step_with_batch_queue` 一致；`with_kv_conn_output_only` 的实际返回（`EMPTY_MODEL_RUNNER_OUTPUT`）与旧 docstring 的冲突写法正确；引用的两个 PP 测试真实存在。
- 12：`feature: pass`。X-1/X-2 两条跨页 P1 已按源码闭合，§3.2 sample 树补齐 `pp_handler.broadcast` 与 `prompt_logprobs_worker`（顺序与源码一致）；唯一缺陷是同一棵树里 LoRA 激活的位置（见 NEW-1）。
- 19：`feature: pass §7.3`。MRV1 派发的四道决定（SP padding / 运行期排除 FULL / `force_eager` / DP 协商后再次派发）逐条与 `gpu_model_runner.py::GPUModelRunner._determine_batch_execution_and_padding` 一致；五条短路与两处断言与 `cudagraph_dispatcher.py::dispatch` 一致。F6 要求的「两代对照」只补了一半。

---

## 2. 本轮受检六项：逐项状态

### 项 1 — 19 §1.2 图 1（上轮 F2，P1）：**FIXED**（残留 1 条 P2）

已按源码核对的部分：

- 新节点 `RST["RequestState<br/>last_sampled_tokens all_token_ids total_len"]`：`vllm/v1/worker/gpu/states.py::RequestState.__init__` 确有 `all_token_ids`、`total_len`、`num_computed_tokens`、`last_sampled_tokens` 四项 ✓。
- 新边 `SMP -->|post_update 写 last_sampled_tokens 与 all_token_ids| RST`：`gpu/model_runner.py::GPUModelRunner.postprocess_sampled`（:1522）调 `input_batch.py::post_update`，实参依次是 `req_states.num_computed_tokens.gpu`、`req_states.last_sampled_tokens`、…、`req_states.all_token_ids.gpu`、`req_states.total_len.gpu` ✓。旧的 `SMP -->|sampled token ids 同址写回| BUF` 已删除 ✓。
- 新边 `RST -->|下一步 combine_sampled_and_draft_tokens 拷进 input_ids| BUF`：`gpu/model_runner.py:1334` 在 `prepare_inputs` 内以 `self.input_buffers.input_ids` 与 `self.req_states.last_sampled_tokens` 调 `combine_sampled_and_draft_tokens`（`input_batch.py:445`）✓。
- 正文列出的 `InputBuffers` 字段与 `input_batch.py::InputBuffers.__init__`（:28-38）逐项一致：`input_ids / positions / is_padding / query_start_loc / seq_lens / dcp_local_seq_lens`，**无 sampled 字段** ✓。
- 与 12 §2.6 的冲突已消除：两页现在都写「`post_update` 落 `RequestState`，下一步 `combine_sampled_and_draft_tokens` 才进 `input_ids`」，且 19 显式把落点合同指给 12 §2.6 ✓。

残留（见 NEW-4，P2）：`BUF -->|下一步的 num_tokens 与 num_reqs| RUN` 这条边未动。

### 项 2 — 19 §6.5 图 2 与启动顺序（上轮 F3，P1）：**FIXED**

- 图中原 B 节点（resolve 在首次编译前）已删除，A 直接连到 C；C 改为 `determine_available_memory 内的 profile_run`，D 标「最小 KV 引发第一次 resolve is_profiling=True，再 capture_model profile_only」，E 标「第二次 resolve 才是生效值」。
- 源码核对：`gpu_worker.py::Worker.determine_available_memory`（:520）先 `self.model_runner.profile_run()`（:564），随后才 `profile_cudagraph_memory()`（:577）；`resolve_cudagraph_mode_and_sizes` 全库只有两个调用点——MRV2 `vllm/v1/worker/gpu/model_runner.py:664`（在 `initialize_kv_cache` 内，直接调用）与 MRV1 `vllm/v1/worker/gpu_model_runner.py:7279`（在 `_check_and_update_cudagraph_mode` 内，链路 `initialize_kv_cache`:7491 → `initialize_attn_backend`:7509 → :7206）。`initialize_attn_backend` 在 MRV1 只有这一个调用点，所以「只在 `initialize_kv_cache` 里调用」作为路径陈述成立 ✓。
- 第一次 resolve：`cudagraph_utils.py::profile_cudagraph_memory`（:757）→ `_init_minimal_kv_cache_for_profiling`（:873）→ `runner.initialize_kv_cache(minimal_config, is_profiling=True)`（:896）→ 内部 resolve；随后 `runner.capture_model(profile_only=True)`（:831）✓。MRV1 有对称实现（`gpu_model_runner.py:6614` 的同名方法、:6637 同样 `is_profiling=True`）✓。
- 跨页一致：12 §2.9 的阶段表（测显存 → 分配 KV → 预热与 capture，graph mode/manager 落在 `initialize_kv_cache`）与 11 §2.9 第 2–4 步（`profile_run` → `profile_cudagraph_memory` → `initialize_kv_cache` → `compile_or_warm_up_model`）都与新图同序，**三页不再矛盾** ✓。
- 顺带的 §1.4 ⑪ 行（上轮 F13）也已修好：`kv_cache_memory_bytes` 显式给定时 `determine_available_memory` 只跑一次 `profile_run` 就 `return`（:534-556）✓；无 KV / `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` 整段跳过与 12 §2.9 一致 ✓。
- 唯一精度瑕疵：MRV1 的直接调用者是 `_check_and_update_cudagraph_mode`，页面只写到 `initialize_kv_cache`（NEW-6，P2/锚点精度）。

### 项 3 — 19 §4.3 / §9.1 `compile_sizes` 范围（上轮 F5，P1）：**FIXED**

- §4.3 改为「它的作用范围只到 `CudagraphDispatcher`」，并补了 `size ≤ max_size` 这一前置条件：源码 `vllm/v1/cudagraph_dispatcher.py::_compute_bs_to_padded_graph_size`（:93-109）确有 `if self.compilation_config.compile_sizes and self.cudagraph_mode != NONE` 与 `if size <= max_size:` 两层门，报错文案逐字一致 ✓。
- 「全库只有三处构造 `CudagraphDispatcher`」：产品代码 `CudagraphDispatcher(` 命中 `vllm/v1/worker/gpu_model_runner.py:919`、`vllm/v1/spec_decode/llm_base_proposer.py:167`、`vllm/v1/spec_decode/extract_hidden_states.py:78`，恰好三处 ✓（`tests/` 另有 7 处构造，措辞用「全库」略欠严谨，见 NEW-7）。两个 drafter 都在 `vllm/v1/spec_decode/`，由 MRV1 的 `gpu_model_runner.py` 导入（:202、:642），因此 §7.3「该校验只在本节这条 MRV1 路径上发生」与 §4.3「MRV1（以及那两个 drafter）」不矛盾 ✓。
- 「默认的 MRV2 路径不经过它，`vllm/v1/worker/gpu/cudagraph_utils.py` 里没有等价校验」：对 `vllm/v1/worker/gpu/` 全域 grep `compile_sizes` → **0 命中**；全库 `compile_sizes` 的 `ValueError` 只在 `cudagraph_dispatcher.py:104`（另有 `piecewise_backend.py:170` 一条不同语义的 `cudagraph_capture_sizes not supported in compile_sizes`，与本条无关）✓。
- §9.1 表行同步改为「走 `CudagraphDispatcher` 的路径（MRV1 与两个 drafter）下……；MRV2 不校验」✓。MRV2 下「错配不报错，代价是该 size 的单点编译白做」是结构性推断，与 `PiecewiseBackend` 的按实际 token 数选 entry 一致，标注得当。

### 项 4 — 19 F6 两代派发对照（P1）：**PARTIAL**（与上轮同，缺的那半未补）

- 已存在且正确的对照：§7.3 的 caller 注入 mode 集合（`valid_modes` / `invalid_modes` → `allowed_modes = valid_modes or valid_runtime_modes(); allowed_modes -= invalid_modes`）、五条短路（`keys_initialized` / `cudagraph_mode==NONE` / `max_size is None` / `num_tokens > max_size` / `allowed_modes <= {NONE}`）、`assert len(allowed_modes) >= 1`、末尾 `assert NONE in allowed_modes`——逐条与 `cudagraph_dispatcher.py::dispatch`（:238-324）一致 ✓；键过滤规则的两代差异有一句明确对照（「MRV1 靠 padding 把真实形状归一到键上，MRV2 靠容量谓词让一张大图服务小 batch」）✓。
- **仍缺**：LoRA capture case 的两代差异。源码事实是 MRV1 `CudagraphDispatcher._get_lora_cases`（:111-130）在 `cudagraph_specialize_lora=False` 时返回 `[lora_config.max_loras + 1]`（**没有 0 case**），MRV2 `vllm/v1/worker/gpu/lora_utils.py::get_lora_capture_cases` 返回 `[0, max_loras + 1]`；页面 §6.2 只写了 MRV2 一侧（写法正确），§7.3 第二个 bullet 只说 `_get_lora_cases()` 由 `cudagraph_specialize_lora` 与 `lora_config.specialize_active_lora` 决定（该两项与 `CudagraphDispatcher.__init__` 的 `self.specialize_lora_count = lora_config.specialize_active_lora` 一致 ✓），**不给集合**。读者无法从本页看出 MRV1 无 LoRA 图这一差异。
- 连带未修：§9.2 ⑰ 的键集合公式仍是旧表述（见 NEW-5）。

### 项 5 — 12 X-1 / X-2（P1）：两条都 **FIXED**

- **X-1（Mamba align 缓存边界）**：新写法把「只缓存落在 `i * block_size` 上的步末状态」明确标注为**字段 docstring 的概括**，再指出实现按 hash 粒度记账。核对：`vllm/config/cache.py` 的 `mamba_cache_mode` docstring 逐字为 “align: only cache the mamba state of the last token of each scheduler step and when the token is at position i * block_size. This is the default when prefix caching is enabled.” ✓；`single_type_kv_cache_manager.py::MambaManager.cache_blocks`（:1917）在 `mamba_cache_mode == "align"` 时调 `_cache_partial_tail_block`（:1928）✓；该 partial-tail 分支（:1984-1992）的前置条件确为 `block_size != hash_block_size`、`num_tokens % self.block_size != 0`、`num_tokens % hash_block_size == 0`，且 `num_tokens == (request.num_prompt_tokens // hash_block_size) * hash_block_size`（「落在 prompt 的最后一个 hash 边界上」）✓。页面「块内的尾状态也会登记」因此成立，与 08 §5.1.3 的 state@112 算例不再冲突 ✓。委托链接指向 08 §5.1.3 的生命周期节 ✓（锚点用的是 §5.1 标题、标签写 §5.1.3，为上一轮已记录的 P2 精度问题，非新增）。
  - 一点未覆盖：checkpoint 分支（`checkpoint_position > 0` 时按 `replace_existing_hashes=True` 重键）没在本页提及，但该分支完整合同已显式交给 08 §5.1.3，符合本库的 owner 约定，不再计为缺陷。
- **X-2（`max_concurrent_batches`）**：§6.1 表行改为「异步 MRV2 为 PP size + 1；异步 MRV1 且 PP≤1 为 2；其余为 PP size（完整四行表归 06 §4.1）」——与 `vllm/config/vllm.py::VllmConfig.max_concurrent_batches`（:567-577）三条分支逐条一致 ✓；§2.3 的快照池深段同步补了「异步 MRV1 另有 PP≤1 取 2 的分支，本页不展开，见 06 §4.1」，并把三个池深算例限定为 MRV2 ✓（`max(2, n)`：n=2→2、n=1→2、n=3→3，重放一致）。与 06 §4.1、11:46 的「容量 2」口径一致 ✓。

### 项 6 — 新增/重写文字的回源核对

逐条核对通过（抽样列出，均与冻结源码一致）：

- **12 §3.2 sample 树的新三行**：`sample_tokens` 空状态 `return None`（`gpu/model_runner.py:1899-1901`）→ 取出并清空 `ExecuteModelState`（:1903-1913）→ 非末 PP 分支（:1915-1933）→ `self.sample`（:1940）→ `pp_handler.broadcast`（:1944-1951，注释明写 handles spec decode multi-token）→ `prompt_logprobs_worker.compute_prompt_logprobs`（:1953-1961）→ `AsyncOutput`（:1974）→ `postprocess_sampled`（:2001）→ `speculator.propose`（:2015）→ `kv_connector.post_forward`（:2053）→ `return async_output`：**页面顺序与源码逐行一致** ✓。
- **12 §3.2 启动树新增两行**：`EngineCore.__init__` 的实际顺序是 `_initialize_kv_caches`(:145) → `StructuredOutputManager`(:146) → `Scheduler = scheduler_config.get_scheduler_cls()`(:149) → `Scheduler(...)`(:162) → `batch_queue`(:210-216) → `step_fn`(:235-236)，与页面「StructuredOutputManager; get_scheduler_cls; Scheduler」→「[max_concurrent_batches > 1] batch_queue；step_fn = step_with_batch_queue」同序 ✓。
- **12 §1.5 新时序图与三段正文**：「不先等 execute future 返回 None 就提交 `sample_tokens(non_block=True)`」与 `EngineCore.step_with_batch_queue`（:689-697）一致 ✓；「队列未满且仍可继续工作时可不等结果返回去排下一批」与 :706-711 一致 ✓；「sampling 结果 None 时再等 exec_future」与 :726-730（随后 `raise RuntimeError("unexpected error")`）一致 ✓；同步对照与 `EngineCore.step`（:619-628）一致 ✓；`AsyncOutput` 无条件构造与 mp/UniProc 两条物化通路与 `WorkerProc.handle_output`（:1001-1009）、`async_output_busy_loop`（:1011）一致 ✓。
- **12 §4 CoW helper（上轮 F8）**：改写后的「`LHBNC` 下一个 block 的字节散落在 `L*H` 个区域，回归测试固定的是这种布局下的复制正确性，helper 自己只按 `data_ptr` 与 storage 字节数两条规则分流」——`tests/v1/worker/test_attn_utils.py:246::test_copy_kv_cache_blocks_separate_head_groups` 存在，注释逐字为 “LHBNC stores each head group separately, so a block's bytes are scattered across L*H regions” ✓。原「源码中找不到」的错误表述已删除 ✓。
- **11 §1.4 / §3.2 新树与新证据段**：`EngineCore.step` 同步分支、队列三条件边界、mp 输出线程与 UniProc 出口、`with_kv_conn_output_only` 返回 `EMPTY_MODEL_RUNNER_OUTPUT`（`vllm/v1/outputs.py:373-383`，docstring 仍写 “Returns None”）、MRV1 `sample_tokens` 三处 return 无一为 `None`、引用的两个测试（`tests/v1/worker/test_gpu_model_runner.py:355`、:386）存在——全部核实 ✓。F15 的段号已改为 15 §7.7（`15_…:471 ### 7.7 EC transfer…`）✓。
- **19 §6.2 容量/身份分组（上轮 F10）**：`gpu/cudagraph_utils.py::_is_compatible`（:98-125）确为 `num_tokens/num_reqs/max_query_len` 三条容量向 + `uniform_token_count/num_active_loras/num_ubatches` 三条身份向，且 `num_reqs`、`max_query_len`、`uniform_token_count` 各带 `desc is None` 逃逸 ✓ — 页面新写法「各三条」正确，旧的「三条 `>=`、两条 `==`」自相矛盾已消除。
- **19 §7.2 四种走法（上轮 F14）**：`vllm/compilation/cuda_graph.py::CUDAGraphWrapper.__call__`（:233-278）的判断顺序确为 ①无 forward context ②mode 不匹配/NONE ③entry hit replay ④miss 时 `validate_cudagraph_capturing_enabled()` 后当场 capture ✓；`set_cudagraph_capturing_enabled(False)` 恰有两处，均在 MRV1（`gpu_model_runner.py:6870` 属 `profile_cudagraph_memory`、:6991 属 `capture_model`）✓。
- **19 §7.3 blocker 清单（上轮 F9）**：`_get_v2_model_runner_unsupported_features`（`config/vllm.py:2547-2617`）逐项一致，含 `ngram/ngram_gpu`、白名单 `eagle/eagle3/mtp/dflash/dspark/extract_hidden_states` 之外的任何 method、`parallel_drafting`（dflash/dspark 除外）；`_get_dbo_unsupported_features`（:2736-2747）在 `VLLM_USE_V2_MODEL_RUNNER is None` 时直接 `return ["dual batch overlap"]`，显式设置后才逐项判定（LoRA/投机/PP>1/context parallelism/多模态与 encoder-decoder/hybrid/`cudagraph_mode != NONE`/encoder-only）✓。
- **19 §9.2 ①（上轮 F12）**：`config/vllm.py::_set_cudagraph_sizes`（:1948-2165）核对——interactivity 只替换头部 `[1,2,4]` 为 `range(1, min(max,32)+1)`，8/16 步长两段照常叠加（:2131-2140）✓；追加落在上限内的 `max_num_batched_tokens`（:2141-2147）与 `uniform_decode_sizes`（:2149-2152）✓；`max_cudagraph_capture_size` 未显式给时 `min(max_num_seqs*decode_query_len*2, 512|1024)`（:2014-2021）后**无条件** `min(max_num_batched_tokens, …)`（:2100-2101）✓；TP>1 且 `enable_sp` 时 `update_sizes_for_sequence_parallelism` 截断（:2155-2163）✓。§9.1 的 `max_cudagraph_capture_size` 行同步改对 ✓。
- **19 §9.2 ⑭（上轮 F11）**：`gpu/spec_decode/autoregressive/speculator.py::init_cudagraph_manager`（:127-147）确实先用目标 mode 建 prefill manager，再窄化 decode manager；`multi_module_mtp/speculator.py:98-111` 与 `dflash/speculator.py:130-141` 各窄化唯一 manager ✓。
- **19 §7.3 新算例（我自行重放）**：`_pad_for_sequence_parallelism`（`gpu_model_runner.py:3579-3585`）在 `enable_sp and tp>1` 时 `round_up(3,2)=4` ✓；`force_eager ⇒ valid_modes={NONE}` 命中 `allowed_modes <= {NONE}` 短路，返回 `BatchDescriptor(4)` ✓；DP 例 `FULL/4` + `PIECEWISE/2` → `_post_process_cudagraph_mode` 取 min = 1（PIECEWISE，`worker/dp_utils.py:97-103`）、`should_dp_pad = synced != 0 or should_ubatch`（:166）为真 → `_post_process_dp_padding` 取 max = 4（:82-94）→ 以 `valid_modes={PIECEWISE}` 重派发得容量 4 的 relaxed 键，并由 runner 断言 token 数相等（:4129-4133）✓。
- **19 §7.5 MRV1 树**：`_determine_batch_execution_and_padding` 的五元组返回、`_is_uniform_decode`（:3975-3993）、`disable_full = use_cascade_attn or has_encoder_output`、`has_encoder_output = is_encoder_decoder and num_encoder_reqs > 0`、`skip_compiled=has_encoder_input` 在 `execute_model` 的 forward context 上另设（:4542）、`_model_forward`（:3942）——全部一致 ✓。

---

## 3. 上一轮未决项状态汇总

| 编号 | 严重度 | 位置 | 本轮状态 | 依据（源码级） |
|---|---|---|---|---|
| F2 | P1 | 19 §1.2 图 1 | **FIXED**（残留 NEW-4） | 见项 1 |
| F3 | P1 | 19 §6.5 图 2 | **FIXED** | 见项 2 |
| F5 | P1 | 19 §4.3、§9.1 | **FIXED** | 见项 3 |
| F6 | P1 | 19 §7.3 枚举依据 | **PARTIAL** | 缺 `_get_lora_cases` 两代集合对照；§9.2 ⑰ 公式未改（NEW-5） |
| X-1 | P1 | 12 §2.5 Mamba align | **FIXED** | 见项 5 |
| X-2 | P1 | 12 §6.1 表行（+§2.3） | **FIXED** | 见项 5 |
| F8 | P2 | 12 §4 CoW helper | **FIXED** | 见项 6 |
| F9 | P2 | 19 §7.3 blocker 清单 | **FIXED** | 见项 6 |
| F10 | P2 | 19 §6.2 子句计数 | **FIXED** | 见项 6 |
| F11 | P2 | 19 §9.2 ⑭ | **FIXED** | 见项 6 |
| F12 | P2 | 19 §9.1、§9.2 ① | **FIXED** | 见项 6 |
| F13 | P2 | 19 §1.4 ⑪ | **FIXED** | 见项 2 |
| F14 | P2 | 19 §7.2 | **FIXED** | 见项 6 |
| F15 | P2 | 11 §3.4 段号 | **FIXED** | 15 §7.7 = EC transfer（15:471），§7.4 = `prompt_embeds`（15:451） |
| F16 | P2 | 12 §2.3、§6.1 | **FIXED** | 见项 5 |
| F17 | P2 | 12 §3.2 调用树 | **PARTIAL** | `pp_handler.broadcast`、`prompt_logprobs_worker` 已补且位置正确；`_set_active_loras` 补了但**排错位置**（NEW-1） |
| N1 | P2 | 11 §1.4（两处） | **NOT_FIXED** | 原文一字未改。`config/vllm.py::__post_init__` 的 async 解析分支（:1270-1319）不读 `scheduler_cls`，也不读 PP；`EngineCore.__init__` 的 `step_fn` 只看 `batch_queue is None`（:235-236），即只看 `max_concurrent_batches`。自定义 Scheduler + async=False + PP=1 仍走 `step()` |
| N2 | P2 | 19 §7.5 Core 树末行 | **NOT_FIXED** | 树末行仍写成无条件末步；`step_with_batch_queue`（`core.py:706-711`）在 `len(batch_queue) < batch_queue_size and (model_executed or scheduler.has_requests())` 时 `return None, model_executed`，本次调用不消费任何结果。11 §1.4 与 12 §1.5 都写了这个条件，19 仍缺 |

统计：**FIXED 13｜PARTIAL 2（F6、F17）｜NOT_FIXED 2（N1、N2）**。按页：11 → FIXED 1、NOT_FIXED 1；12 → FIXED 4、PARTIAL 1；19 → FIXED 8、PARTIAL 1、NOT_FIXED 1。

---

## 4. 本轮新发现

### NEW-1（**P1**｜12 §3.2 worker 调用树）——把 `_set_active_loras` 排在 model forward **之后**

- 页面原文（新增行）：
  ```
      +-- [FULL]      ModelCudaGraphManager.run_fullgraph --> … --> sync_prev_onload; replay
      |   [微批]      UBatchRunner.run
      |   [PIECEWISE] run_pw_graph    [NONE] model(**model_inputs)
      |-- [有 lora_config] lora_state.make_lora_inputs; _set_active_loras   [真实批才做；dummy run 分支不走]
      `-- 保存 ExecuteModelState                          [非末 PP 返回 IntermediateTensors]
  ```
- 该节开头自定的约定是「先后发生的兄弟调用分行列出」，所以这一行断言 LoRA 激活发生在 forward 之后。
- 源码事实：`vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model` 在**非 dummy 分支内**依次是 `prepare_inputs`(:1645) → `prepare_attn`(:1650) → `model_state.preprocess_state`(:1654) → `if self.lora_config: lora_state.make_lora_inputs(...)`(:1661-1667) → `self._set_active_loras(*lora_inputs)`(:1668)；此后（:1698 起）才构建 attn metadata / ubatch state，再进入四条执行分支。`_set_active_loras` 全文件只有这一个调用点（:1668），**必然在 forward 之前**。
- 影响：读者会得出「adapter 在模型执行后才激活」的错误机制；且与 19 §6.2「`num_active_loras` 是 descriptor 身份字段、dispatch 之前就要定」的口径不一致。
- 建议修法：把该行移到 `+-- model_state.preprocess_state` 之后、`+-- [num_ubatches>1] UBatchRunner.prepare` 之前，并把行首标记从 `|--` 改回本树统一的 `+--`。

### NEW-2（P2｜12 §1.5 第一句）——把「未自定义 Scheduler」写成 async 解析的条件

- 页面原文：「普通生成、兼容配置且未自定义 Scheduler 时，`async_scheduling=None` 解析为 True，选择 `AsyncScheduler`。」
- 源码：`config/vllm.py::__post_init__` 的 `elif self.scheduler_config.async_scheduling is None:` 分支（:1270-1319）只看 pooling / speculative method / `disable_padded_drafter_batch` / executor 支持 / ROCm DeepEP HT DBO，**完全不读 `scheduler_cls`**；只有 `SchedulerConfig.get_scheduler_cls()`（`config/scheduler.py:201-222`）才因自定义类而放弃 `AsyncScheduler`（且仅 `warning_once`）。
- 与 N1 同类；19 §7.1 的同一句写法正确（把「未自定义 Scheduler」只挂在「默认解析为 `AsyncScheduler`」上），可直接对齐。
- 建议：改成「普通生成、兼容配置时 `async_scheduling=None` 解析为 True；此时若未自定义 Scheduler 即使用 `AsyncScheduler`」。

### NEW-3（P2｜19 §1.2 正文末句）——`post_update` 的写入清单不完整

- 页面原文：「`post_update` 写的是 `RequestState` 的 `last_sampled_tokens / all_token_ids / total_len / num_computed_tokens`」。
- 源码：`postprocess_sampled`（`gpu/model_runner.py:1522-1547`）传给 `post_update` 的还有 `output_bin_counts = self.sampler.penalties_state.output_bin_counts`（末 PP rank 时非 None），它**不在** `RequestState` 里；`post_update` 的 kernel 同样写它。
- 建议：加半句「另写 sampler 的 `penalties_state.output_bin_counts`（不属 `RequestState`）」，或把句子改成「落进 `RequestState` 的是……」。

### NEW-4（P2｜19 §1.2 图 1）——`BUF -->|下一步的 num_tokens 与 num_reqs| RUN` 仍在

- 这条边是上一轮 F2 引文的第三项，本轮未动。图规格自述「边标注跨越边界的**实际对象名**」，但 `InputBuffers` 只持有容量常量 `max_num_reqs / max_num_tokens`（`input_batch.py:24-25`）与六个 tensor，**没有**本步 `num_tokens / num_reqs`：这两个值由 `SchedulerOutput.num_scheduled_tokens` 经 `prepare_inputs` 算出，而该流向已由图里的 `SCH -->|SchedulerOutput.num_scheduled_tokens| RUN` 表达。
- 建议：把标签改成「下一步 prepare 复用同址 storage 的切片」，或删掉这条边（回边闭环已由 `RST → BUF` + `SCH → RUN` 表达）。

### NEW-5（P2｜19 §9.2 ⑰）——MRV1 键集合公式仍不完整

- 页面原文：「键集合大小 = capture sizes × LoRA cases（FULL 另限于 `<= max_num_seqs * decode_query_len`）」。
- 源码 `cudagraph_dispatcher.py::initialize_cudagraph_keys`（:167-233）建的是**两套键的并集**：mixed-mode 键 = `product(cudagraph_capture_sizes, lora_cases)`（PIECEWISE 时还把 `num_reqs=None, uniform=False` relaxed），另在 `decode_mode()==FULL and separate_routine()` 时追加 decode 键，其筛选是 `x <= uniform_decode_query_len * max_num_seqs` **且** `x >= uniform_decode_query_len`。页面漏了下界与「两套并集」。
- 建议：写成「≤ 2 × capture sizes × LoRA cases（mixed 一套；decode FULL 另一套，size 限于 `[decode_query_len, max_num_seqs*decode_query_len]`）」。

### NEW-6（P2｜19 §6.5 正文）——MRV1 侧 resolve 的符号锚点不是直接调用者

- 页面写 `resolve_cudagraph_mode_and_sizes` 在「MRV1 `gpu_model_runner.py::GPUModelRunner.initialize_kv_cache`」里调用；实际直接调用者是 `GPUModelRunner._check_and_update_cudagraph_mode`（:7252-7288），经 `initialize_attn_backend`(:7206) 由 `initialize_kv_cache`(:7509) 触发。路径陈述成立，但按本库「稳定锚点 = 路径 + 限定符号」的要求应点名真正的符号（§12 的读码路线也未登记它）。

### NEW-7（P3｜19 §4.3）——「全库只有三处构造 `CudagraphDispatcher`」

- 产品代码确为三处，但 `tests/` 另有 7 处（`tests/v1/cudagraph/test_cudagraph_dispatch.py` 4 处、`tests/compile/test_config.py` 4 处、`tests/models/language/generation/test_hybrid.py` 1 处）。措辞宜改「产品代码只有三处」。不影响结论。

---

## 5. 机械门禁（只读复跑）

| 检查 | 结果 |
|---|---|
| `check_links.py --strict` | pages=453，broken/ambiguous/bare_index/stale_section/orphans 全 **0** |
| `check_math.py --changed --strict` | 59 文件，**0 错 0 警** |
| `check_markdown.py --changed --strict` | 59 文件，**0 错 0 警** |
| `check_assets.py --changed --strict` | 59 文件，**0 错 0 警** |

本轮新增/改动的深链锚点逐一手工核对存在且语义正确：08 §5.1（`### 5.1 更细粒度复用…`，§5.1.3 为其子节）、17 §6.2（`### 6.2 reload 与 CUDA Graph：本页负责哪一半`）、06 §4.1（`### 4.1 队列里必须同时保留 future 和原计划`）、12 §1.5、14 §3.8、15 §7.7、26 页面存在。

---

## 6. 结论与剩余阻塞

- **19**：上一轮的三条 P1 阻塞（图 1 sampled token 落点、图 2 启动顺序、`compile_sizes` 范围）**全部按源码修好**，并顺带清掉 F9～F14 六条 P2；与 12 §2.6 / §2.9、11 §2.9 的三处跨页冲突全部消除。**无 P0/P1 残留**，verdict **ACCEPT**；F6 的 LoRA case 两代对照与 N2、NEW-4/5/6 可随后续编辑合并。
- **12**：X-1、X-2 两条跨页 P1 已闭合且可按源码重放，F8/F16 同时修好。**唯一 blocker 级缺陷是 NEW-1**（§3.2 调用树把 `_set_active_loras` 排到 forward 之后），一行位置调整即可；另有 NEW-2 一条 P2。verdict **ACCEPT WITH FIXES**。
- **11**：本轮只剩 F15 的段号修正，已修；N1 仍未动（P2，措辞层面）。verdict **ACCEPT**。
- 建议收口顺序：NEW-1（12 一行）→ N1/NEW-2（11、12 各一句）→ N2 与 NEW-4（19 两处图/树标注）→ F6 的两代 LoRA case 对照与 NEW-5 → NEW-3/6/7。
