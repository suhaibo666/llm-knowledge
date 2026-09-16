# Group F 复核（round 2）：vLLM 14 / 15 / 16

- 复核基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（checkout `/Users/suhaibo/97-llm/vllm`，HEAD 已核对一致，仅有未跟踪 `artifacts/`，未移动 HEAD）
- 复核对象：工作树未提交改动。14 `+92`、15 `+42`、16 `+89`；邻接页 07 `+95`、11 `+64`、12 `+110` 一并对照。
- 方法：先读旧报告，再对**每条新增论断**回到冻结源码逐一验证（不以新散文为依据），然后重放新增数值例、复核跨页归属。评审者未写过这些页面；**未修改任何文件**。
- T0 门（只读运行）：`check_links --strict` → 452 页 `broken=0 / ambiguous=0 / bare_index=0 / stale_section=0 / orphans=0`；`check_math --changed --strict` → 39 文件 `0 error / 0 warning`；`check_markdown --changed --strict` → 39 文件 `0 error / 0 warning`。新增的五个跨页 section 锚点全部存在。

---

## 0. Verdict rows

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 14_vllm_sampling_structured_output_analysis | pass | pass | pass | transform, coupled-planes | pass | 3/3 | ACCEPT（P2 待办） | 旧 P0 已按源码彻底改正（§4.5 + §1.3/§2.1/§3.4/§4.4/§7.1 全部同步）；新增 §3.8 prompt logprobs 单元逐条可核；F14-3/F14-4 两个 P2 未动 |
| 15_vllm_multimodal_execution_analysis | pass | pass | pass | transform, layout, coupled-planes | pass | 3/3 | ACCEPT（P2 待办） | 旧 P1 已闭合（§4.2 新增三段 + §1.3 拆两行 + §1.5 树 + §9.1/§9.3 成本与边界），数值例复算一致；F15-2…F15-6 五个 P2 原文未动 |
| 16_vllm_speculative_decoding_analysis | pass | pass | pass | transform, timing, coupled-planes | pass | 3/3 | ACCEPT（P2 待办） | 三个旧 P1 全部按源码改正（`init_speculator` 顺序、V1 11 分支全表、deferred 发布链）；F16-4/F16-5/F16-6 三个 P2 未动 |

Feature verdicts：

- 14：`feature: pass §4.5`（投机下 target 分布的构造有人负责，V1/MRV2 两条入口分别写清）、`feature: pass §3.8`（新单元的触发/输入/变换/跨 chunk 累积/交付/完成点/成本/阻断齐备且源码可核）
- 15：`feature: pass §4.2`（渐进释放、finish/preempt、逐出、reset 四类边界齐备，enc-dec 分支单列）
- 16：`feature: pass §5.1`（变体集合完整且与枚举依据一致）、`feature: pass §8.3`（三条发布路线的触发条件与两代 runner 差异正确）

Spot-check 记录（本轮轮换的三个新锚点/页）：

- 14：`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py::RejectionSampler._verify` ✔（第一行即 `apply_sampling_params`，默认 `skip_top_k_top_p=False`）；`vllm/v1/sample/rejection_sampler.py::RejectionSampler.apply_logits_processors` + `apply_sampling_constraints` ✔（顺序 penalties→allowed→bad words→MinTokens→thinking budget；constraints 全 greedy 直返）；`vllm/v1/worker/gpu/sample/prompt_logprob.py::PromptLogprobsWorker.compute_prompt_logprobs` / `_prompt_logprobs_token_ids_kernel` / `compute_prompt_logprobs_with_chunking` ✔
- 15：`vllm/v1/core/sched/scheduler.py::Scheduler._free_encoder_inputs` ✔（条件逐符号一致）；`Scheduler.__init__` 的 `num_prefill_lookahead` ✔（默认 0 / eagle 1 / multi-module MTP 取 `num_spec_tokens`）；`EncoderCacheManager.free_encoder_input` / `free` ✔（`num_freeable_slots` 增、`num_free_slots` 不增；enc-dec 版直接加 `num_free_slots`）
- 16：`vllm/v1/worker/gpu/spec_decode/__init__.py::init_speculator` ✔（七分支顺序）；`vllm/v1/worker/gpu_model_runner.py::GPUModelRunner.__init__` ✔（11 个 if/elif + else，PP 末 rank 门）；`vllm/v1/engine/core.py::EngineCore.step_with_batch_queue` + `vllm/v1/worker/gpu/spec_decode/utils.py::DraftTokensHandler` ✔

---

## 1. 上一轮每条 finding 的状态

| ID | 级别 | 状态 | 源码验证 |
|---|---|---|---|
| F14-1 | P0 | **FIXED** | §4.5 重写为“先构造 target 分布，再验证和回采”。MRV2：`RejectionSampler.__call__` → `_verify_in_chunks` → `_verify`，第一行 `self.sampler.apply_sampling_params(...)`，页面所述顺序（bias/min-tokens → penalties → bad words → thinking budget → temperature → min-p → top-k/top-p）与 `sampler.py:209-273` 逐行一致；`_verify` 未传 `skip_top_k_top_p`，默认 `False`，页面“过滤也在验证前完成”正确；`if not np.any(needs_logits_processing[...]) : return logits` 对应“没有请求需要变换时沿用原 logits”。V1：`RejectionSampler.forward` 的 bonus 行 `self.sampler(..., predict_bonus_token=True)`（`Sampler.apply_logits_processors` 的 `predict_bonus_token and any_penalties_or_bad_words` 分支合成假设历史）与 target 行 `RejectionSampler.apply_logits_processors` + `apply_sampling_constraints` 都对；“类 docstring 仍称不支持 top-k/top-p”确有其事（`rejection_sampler.py:53-55`），页面以实现为准并引用 `test_top_k`/`test_top_p` 正确。**因果也改对了**：能力损失来自 `SamplingParams._validate_spec_decode`（`min_p > _SAMPLING_EPS=1e-5` 或 `logit_bias` 非空 → `VLLMValidationError`）与 `build_logitsprocs` 投机分支（只建 `MinTokensLogitsProcessor`、拒 custom），不再归因于“绕过 sampler”。§1.3 行、§2.1 末句、§3.4 变体行、§4.4 行、§7.1 两棵调用树同步更新；全域已无“整体绕过”残留 |
| F14-2 | P2 | **FIXED** | §6.4 改为“两条 draft 发布路线共享 validate，只有 deferred 路线补 `-1`”。同步 `update_draft_token_ids` 只 `validate_tokens` 后写 `request.spec_token_ids`（`scheduler.py:2395-2415`，无 `-1`、无计数）；deferred `update_draft_token_ids_in_output` 截断→validate→补 `-1`→写 `num_invalid_spec_tokens`（`scheduler.py:2379-2414`）。图 2 的边也拆成 `NX`（下一轮）与 deferred 两条，§1.3 行同步 |
| F14-3 | P2 | **NOT_FIXED** | §3.5“其三”与 §5 表 `logprob_token_ids` 行文字未变，仍无投机例外；16 §9.1 同样未动。全 wiki（含 11/12/07）都没有这条边界的 owner |
| F14-4 | P2 | **NOT_FIXED** | §6.4 末段仍只有文字，没有加 `[x1, -1]` 的逐行 mask 表 / rollback 次数的可重放例子 |
| F15-1 | P1 | **FIXED** | §4.2 新增三段：条件写作 `offset + length + num_prefill_lookahead <= num_computed_tokens - num_output_placeholders`，与 `scheduler.py:2347-2355` 逐符号一致（含“用完整 `length` 而非 `get_num_embeds()`”这一细节）；触发点“本步实际执行、有效 spec rejection 回退之后”与 `scheduler.py:1930-1977`（含源码注释 “Free encoder inputs only after the step has actually executed.”）一致；lookahead 取值与 `Scheduler.__init__:272-286` 一致；enc-dec 优先分支 `is_encoder_decoder and num_computed_tokens > 0` 及 `EncoderDecoderCacheManager` 回补 slots 一致；finish/preempt 走 `encoder_cache_manager.free(request)`（`_free_request:2532`、`_preempt_request:1457`）一致，`reset_prefix_cache(reset_running_requests=True)` 确实经 `_preempt_request`。§1.3 拆成“正常渐进释放”“finish/preempt 与逐出清理”两行，§1.5 树、§1.2、§9.1（两行）、§9.3、§7.5 依据与测试清单同步。四个测试锚点 `test_free_encoder_inputs_respects_unconfirmed_placeholders` / `_defers_for_eagle_lookahead` / `_unchanged_without_spec_decode` / `test_encoder_cache_retained_across_preemption_and_resume` 全部存在 |
| F15-2 | P2 | **NOT_FIXED** | §3.1 依据仍写 `receiver_cache_from_config`；基线只有 `engine_receiver_cache_from_config`（309/337/350 行）与 `worker_receiver_cache_from_config` |
| F15-3 | P2 | **NOT_FIXED** | §1 仍写“本页从渲染器交出的 `EngineInput` 开始”，与 §2.1–§2.3 的 `MediaConnector`/`MultiModalDataParser`/`apply` 内容矛盾 |
| F15-4 | P2 | **NOT_FIXED** | §7.8 仍写“这个 runner 的**类**文档”；基线该段是 `mm_encoder_model_runner.py` 的**模块** docstring |
| F15-5 | P2 | **NOT_FIXED** | §4.1 规则 5 仍把 compute 预算与 cache slot 合写成 `allocate()` 一件事 |
| F15-6 | P2 | **NOT_FIXED** | §5.1 仍把 `zip(mm_hashes, encoder_outputs)` 挂在 `EncoderRunner.execute_mm_encoder` 名下 |
| F16-1 | P1 | **FIXED** | §5.1 改为“MRV2 无专用 Step3.5 分支：先 `use_multi_module_mtp()`，`min(num_nextn_predict_layers, k) > 1` 选 `MultiModuleMTPSpeculator`，否则 `MTPSpeculator`”，与 `init_speculator` 顺序 + `use_multi_module_mtp`（`speculative.py:1883-1889`，缺省 1）一致；“两者都没有 V1 的 `set_per_group_attn_metadata` 接口”正确（全仓仅 `v1/spec_decode/step3p5.py:40` 与 V1 runner:2653）；配置改写锚点改为 `SpeculativeConfig.hf_config_override`（实际在 `speculative.py:937-949`）正确，并明确登记“未核验远程 checkpoint 实际层数”这一残余不确定性 |
| F16-2 | P1 | **FIXED** | 新增 11 行“V1 完整选择顺序”表，与 `gpu_model_runner.py:646-705` 的 if/elif 顺序逐行一致（custom_class → ngram → uses_draft_model → use_ngram_gpu → use_gemma4_mtp → use_step3p5_mtp → use_dflash → suffix → use_eagle → medusa → extract_hidden_states → else `ValueError`）；入口门 “有 speculative config 且末 PP rank” 与 `if self.speculative_config and get_pp_group().is_last_rank` 一致；MRV2 顺序句与 `init_speculator` 一致；`use_eagle()` 覆盖集合（eagle/eagle3/mtp/dflash/dspark）正确；“DSpark 被 V1 validator 拒绝”与 `_get_v1_model_runner_unsupported_features:2626-2629` 一致 |
| F16-3 | P1 | **FIXED** | §8.3 路线 2 改为“async/PP batch-queue deferred（有 pending 结构化输出时）”，链路 `take_draft_token_ids()` → `update_draft_token_ids_in_output()` → `get_grammar_bitmask()` → `sample_tokens()` 与 `core.py:743-761` 一致；`pending_structured_output_tokens` 的置位条件与 `async_scheduler.py:30-32` 一致（非 prefill chunk、结构化、**本步增加前**已有 placeholders）；`max_concurrent_batches` 在 async 下 ≥2 与 `vllm.py:567-577` 一致。**并顺带改正了一处旧散文错误**：`DraftTokensHandler.set_draft_tokens` 只看 `has_structured_output_reqs`、**不因 async 跳过** D2H（`utils.py:24-43`），原文“MRV2 在不需要 grammar 校验时根本不把真实候选送回 CPU”的推论已删除。§1.1 六边界、图 1（新增 deferred 边与虚线）、§7 表（async 拆两行）、§10.1（新增 batch-queue 子树）同步；三个测试锚点存在 |
| F16-4 | P2 | **NOT_FIXED** | §1.2 行（第 71 行）仍写“条件：三种来源各有自己的触发”，§3.3（第 188 行）写“四个来源”，§7 表现在已是 **6** 行 |
| F16-5 | P2 | **NOT_FIXED** | §5.2 第 1 步仍写“draft 组的位置12是本轮第一次写（它的 prefill 只覆盖 draft 位置10与11）” |
| F16-6 | P2 | **NOT_FIXED** | §9.1 行文字未变。源码确认 `include_token_ids=` 在全仓仅出现于 `gpu/model_runner.py:1512`，且在 `if shard_metadata is not None:` 的 batch-sharded gather 分支内 |
| E2E-1（14↔16 投机下 p 的构造无人负责） | P0 | **FIXED** | 14 §4.5 明确“target 约束归本页、接受与残差归 16”，§2.1 与 16 §7 互引；16 §7 的“MRV2 将 draft IDs 和 expanded local position 传给普通 sampler 的参数处理”与 14 §4.5 一致，两页不再矛盾 |
| E2E-2（grammar 与草稿重复讲解且精度不一致） | P1 | **FIXED** | 归属已切开：14 §6.4 拥有 validate/截短/补 `-1`/`num_invalid_spec_tokens`/rollback 的语义并把触发条件外链 16 §8.3；16 §7/§8.3 拥有三条发布路线与两代 runner 差异并把 mask 语义外链 14 §6.4。07 §8.1.3 的 `num_invalid_spec_tokens` 行也显式写“grammar 语义归 14，draft D2H/发布归 16” |
| E2E-3（15↔07/16 释放边界缺 owner） | P1 | **FIXED** | 15 §4.2 成为 owner 并互链 16 §8.4；07 第 717 行的同一规则描述与 15 一致，无矛盾 |
| E2E-8（投机下 `logprob_token_ids` 无 owner） | P2 | **NOT_FIXED** | 见 F14-3 / F16-6 |

统计：**FIXED 7（含 1 个 P0、4 个 P1、2 个 P2）/ PARTIAL 0 / NOT_FIXED 10（全为 P2）/ SUPERSEDED 0**。按页：14 = 2 FIXED / 2 NOT_FIXED；15 = 1 FIXED / 5 NOT_FIXED；16 = 3 FIXED / 3 NOT_FIXED；E2E = 3 FIXED / 1 NOT_FIXED。

---

## 2. 新增文本的逐条源码核验（regression hunt）

以下每项都打开了冻结源码，**未发现新的 P0/P1**。

### 2.1 14 §4.5 投机路径（核心修复区）

| 新论断 | 源码 | 结果 |
|---|---|---|
| MRV2 `sample` 三分支（`num_reqs==0` → None；`num_draft_tokens==0 or rejection_sampler is None`；else rejection sampler） | `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.sample`（1479-1496） | ✔ 逐条件一致 |
| `_verify` 第一步 `apply_sampling_params`，之后 `rejection_sample(processed_logits, …)` | `gpu/spec_decode/rejection_sampler.py::RejectionSampler._verify` | ✔ |
| 本次调用保留默认 `skip_top_k_top_p=False` | `gpu/sample/sampler.py:218`（默认值）、`:295`（`Sampler.sample` 反而传 `True`） | ✔ 且对比关系正确 |
| `draft_sampled = input_ids[logits_indices]` | `RejectionSampler.__call__` | ✔ |
| V1 bonus 行 `Sampler(..., predict_bonus_token=True)`，合成“已提交 output + 整段 draft”的假设历史 | `sample/sampler.py::apply_logits_processors:384-390` | ✔ |
| V1 target 行顺序 penalties → allowed → bad words → `MinTokensLogitsProcessor.apply_with_spec_decode` → thinking budget | `sample/rejection_sampler.py::apply_logits_processors:289-346` | ✔ 顺序一致 |
| `apply_sampling_constraints` 先 temperature 再 top-k/top-p，全 greedy 直返 | 同文件 `:510-566` | ✔ |
| `_validate_spec_decode` 用 `min_p > _SAMPLING_EPS`（1e-5）或 `logit_bias` 抛 `VLLMValidationError` | `sampling_params.py:27`、`:1012-1024` | ✔ |
| `build_logitsprocs` 投机分支只建 MinTokens、拒 custom | `v1/sample/logits_processor/__init__.py:201-210` | ✔ |
| `_verify_in_chunks` 的分块目标是限制 `apply_sampling_params` 物化的 FP32 target buffer | 文件头注释 “Cap on the FP32 target-logits buffer materialized by apply_sampling_params.” | ✔ |
| §7.1 两棵调用树（MRV2 `_verify_in_chunks/_verify/_get_logprobs_tensors` 同级关系；V1 `forward` 下四个子步） | 同上两文件 | ✔ 层级与源码调用点一致 |
| 新增依据行全部符号 | `apply_bad_words_with_drafts`（`ops/bad_words.py:39`）、`apply_with_spec_decode`（`builtin.py:290`）、`_combine_outputs_with_spec_tokens`、四个 `test_rejection_sampler.py` 测试、两个 `test_gpu_bad_words.py` 测试 | ✔ 全部存在 |

### 2.2 14 §3.8 prompt logprobs（全新单元）

| 新论断 | 源码 | 结果 |
|---|---|---|
| 目标 token 右移一位：`all_token_ids[num_computed_tokens + 1 + j]` | `_prompt_logprobs_token_ids_kernel`（`target_pos = num_computed_tokens + 1 + block`） | ✔ |
| 每步先筛“仍覆盖 prompt”，抢占恢复越过原 prompt 则跳过 | `includes_prompt = computed_prefill < prompt_lens`；`resumed_after_prompt = prompt_lens < prefill_len_np` | ✔ |
| 按 1024 行切块调 `logits_fn` | `compute_prompt_logprobs_with_chunking`：`CHUNK_SIZE = 1024` | ✔ |
| 任一请求取 `-1` → 本轮宽度为词表，输出 $V+1$ 列；否则公共宽度取 max k，发布前裁 $k+1$ | `max_num_prompt_logprobs = -1 if any(...==-1) else max()`；`requested_num = shape[-1] if -1`；`width = shape[1] if -1 else k+1` | ✔ |
| 第 0 列是目标 token、其余为 top-k、另给全词表 rank | `compute_topk_scores`：`cat((sampled.unsqueeze(-1), topk_indices))` + `_ranks_kernel` | ✔ |
| 最终 chunk 去掉末行（它预测第一个生成 token） | `if not req_is_prompt_chunked: end_idx -= 1` | ✔ |
| `LogprobsTensors.cat` 合并、清空列表、放入 `prompt_logprobs_dict` | 同函数尾部 | ✔ |
| V1 预分配 `(prompt_len-1, k+1)` CPU tensor，按 `num_computed_tokens` 写 slice，prefill 完成才同步 | `gpu_model_runner.py::_get_prompt_logprobs_dict`（`LogprobsTensors.empty_cpu(num_prompt_tokens-1, k+1)`、`chunk_slice = slice(start_idx, …)`、`if prompt_logprobs_dict: self._sync_device()`） | ✔ |
| `raw_*` 与 `processed_*` 在此数值相同 | 源码注释 “prompt tokens skip sampling processors, so processed_* and raw_* yield the same scores here.” | ✔ |
| 默认 `skip_reading_prefix_cache=True` | `sampling_params.py:539-543` | ✔ |
| renderer 拒绝 prompt embeds + prompt logprobs | `renderers/online_renderer.py:287-289` | ✔ |
| Scheduler 仅在 `should_emit_output` 时放入 `new_prompt_logprobs_tensors`，否则断言没有 | `scheduler.py:2122-2151`（`assert not prompt_logprobs_tensors`） | ✔ |
| 前端首项 `None`、DELTA 经 `pop_prompt_logprobs()` 一次交付、重复 token 按 id 合并 | `logprobs.py::create_prompt_logprobs`（`logprobs.append(None)`）、`LogprobsProcessor.pop_prompt_logprobs`、`append_logprobs_for_next_position` 注释 | ✔ |
| `-1` 仍受 `ModelConfig.max_logprobs` 校验 | `sampling_params.py:862-872` | ✔ |
| 依据行测试锚点 | `test_logprobs.py::test_prompt_logprobs_mode:568`、`::test_prompt_logprobs_with_chunking_and_preemption:1220` | ✔ |

### 2.3 15 §4.2 每步释放（新增单元）

数值例复算（**我自己重算**，与页面一致）：图 1 的媒体 `offset=2, length=4`（`1×4×4/2²=4`，展开 `[2,6)`）。
`num_prefill_lookahead=1`、`num_output_placeholders=4` → 条件 `2+4+1 = 7 <= computed − 4`；`computed=10` → 确认进度 6 < 7 保留；`computed=11` → 7 ≥ 7 释放 ✔。无 lookahead、无未确认 placeholder → `computed >= 6` ✔。

其余论断见 §1 表 F15-1 行；另核：`get_cached_input_ids`（`encoder_cache_manager.py:220`）、`free`→`free_encoder_input`（`:262-272`）、`EncoderDecoderCacheManager.free_encoder_input` 直接加 `num_free_slots`（`:403-405`）、§7.9 确实讲“释放延迟一轮”（与 §4.2 的外链一致）。`update_from_output` 的循环确有 KV-load 失败 / 已 finished / drop-stale 三个 `continue`，页面“对仍进入结果处理且有 encoder 输入的请求”这一限定正确。

### 2.4 16 §5.1 / §7 / §8.3 / §10.1

见 §1 表 F16-1/F16-2/F16-3 行。另核：`use_dflash`/`use_ngram_gpu`/`uses_draft_model`/`use_gemma4_mtp`/`use_step3p5_mtp` 五个谓词全部存在且语义与表格一致；`check_for_draft_tokens = use_spec_decode or is_diffusion`（`core.py:172-174`）；`post_step` 门 `check_for_draft_tokens and not async_scheduling and model_executed`（`:644`）；V1 `take_draft_token_ids`/`_copy_draft_token_ids_to_cpu` 的 async 条件（`gpu_model_runner.py:4985-5005`）与页面一致；`update_draft_token_ids_in_output` 的“先按本步 spec 长度截断、再 validate、再补 `-1`”与 `scheduler.py:2398-2414` 一致。

---

## 3. 新发现（regression / new）

**N1 [P2] 16 §8.3 路线 2 标题把 PP 与 async 并列，但 deferred 只可能由 AsyncScheduler 触发**

- 页面原句（§8.3 路线 2 标题）：“**async / PP 的 batch-queue deferred 路线（有 pending 结构化输出时）**”。
- 源码证据：`SchedulerOutput.pending_structured_output_tokens` 默认 `False`，全仓**只有** `vllm/v1/core/sched/async_scheduler.py:31` 会置位（`AsyncScheduler._update_after_schedule`），`core.py:689` 只读它。纯 PP（同步调度）用的是基类 `Scheduler`，永不置位，因此永不进 deferred 分支。
- 说明：同段正文已写“PP 也使用 batch queue，但队列存在本身不是 deferred 条件”，§7 表行也限定为“async 调度，有结构化输出请求且先前输出占位未结清”，所以**只是标题措辞与正文不同调**，不是事实错误；但读者按标题索引会得到“PP 也会 deferred”的印象。
- 建议：标题改为“**async 调度的 batch-queue deferred 路线（有 pending 结构化输出时）**”，把“PP 也用 batch queue 但不构成 deferred 条件”留在正文那一句。§10.1 树的注释 “async 或 PP；这里展开 pending 结构化输出分支”同理可收紧为“async 或 PP 共用此入口；deferred 分支仅 async”。

**N2 [P2] 14 §3.8 “返回字典仍为空”在混合 batch 下不成立**

- 页面原句（§3.8 第三段）：“非最终 chunk 计算出的 `LogprobsTensors` 追加到 `in_progress_prompt_logprobs[req_id]`，**返回字典仍为空**”。
- 源码证据：`PromptLogprobsWorker.compute_prompt_logprobs` 的循环对 `req_is_prompt_chunked` 的请求 `continue`，只是**该请求**不进入 `prompt_logprobs_dict`；同一步里其它请求若正好收尾，仍会 `prompt_logprobs_dict[req_id] = logprobs`。返回值是全 batch 的字典，不是单请求视角。
- 建议：改为“**该请求本步不进入返回字典**”。（V1 侧同构：`prompt_logprobs_dict` 只收 `completed_prefill_reqs`。）

**N3 [P2] 14 新声明的 §3.8 归属没有反映到 11/12 的所有权表**

- 页面原句：14 页头“其实现、跨 prefill chunk 累积和完成边界由本页 §3.8 负责；runner 只提供调用位置与输出搬运”，§4.4 行“数值与跨 chunk 合并见本页 §3.8”。
- 现状：`11_vllm_model_runner_v1_analysis.md:560` 的 `prompt_logprobs_dict` 行归属列仍只写“07 §8.5”；`12_vllm_model_runner_v2_analysis.md:357` 同字段也只写“§8.5”。两页在本轮都被改过（+64 / +110），却没有回指 14 §3.8。
- 影响：不构成矛盾（07 §8.5 第 739 行明确“这里组装已有结果，不执行采样算法或文本解码”，与 14 §3.8 不冲突），但从 11/12 出发的读者找不到新 owner，单向链接会随时间退化成“又一处各写一遍”。
- 建议：在 11/12 的这两行归属列补“数值与跨 chunk 合并归 14 §3.8”。

**N4 [P3] 14 §6.4 的 deferred 描述漏掉 validate 之前的“按本步长度截断”**

- 源码：`update_draft_token_ids_in_output` 先 `del spec_token_ids[orig_num_spec_tokens:]`（注释 “needed for chunked prefill case for example”），**之后**才 `validate_tokens`，最后补 `-1`。14 §6.4 只写“校验真实草稿，把非法尾部用 `-1` 补齐到原长度”。
- 说明：07 §8.1.3 的 `num_invalid_spec_tokens` 行已写全三步，所以域内信息不缺；14 作为 grammar 语义 owner 少一句前置截断。建议在 14 §6.4 补半句，或明写“截断归 07 §8.1.3”。

**N5 [P3] 跨页链接的 label 比 anchor 更精确，落点会偏一级**

- 16 §7 新增 `[[08_vllm_kv_cache_management_analysis#3.3 分配：把候选命中变成受保护的请求映射|KV Cache §3.3.3]]`：anchor 指 `### 3.3`，label 承诺 §3.3.3，而 `#### 3.3.3 投机尾部：有 slot、token 已确定、可登记 hash 是三条不同边界` 确实存在、可直接作 anchor。07 本轮新增的 `|KV Cache §3.3.1]]`、`|KV Cache §5.1.3]]` 同型。
- 说明：`check_links --strict` 通过（anchor 存在），只是跳转落在父节。建议 anchor 与 label 取同一级。

---

## 4. 跨页复核（本轮指定的四组）

1. **14 ↔ 16（rejection sampling 与 grammar-with-drafts）：一致，且归属已切开。** 14 §4.5 owner = target 分数的参数处理与交接边界；16 §2–§4 owner = 接受判定与残差分布；16 §7 的 p_i 段与 14 §4.5 的两代入口描述互为补充、无重复展开。grammar 侧：14 §6.4 owner = validate/截短/补 `-1`/`num_invalid_spec_tokens`/rollback 容量；16 §8.3 owner = 三条发布路线与两代 `take_draft_token_ids` 差异；两处互链且都不再各讲一遍。残余：16 §3.3 的“`-1` 四个来源”属内核侧语义，与 14 §6.4 不冲突。
2. **15 ↔ 07（encoder 预算与每步释放）：一致。** 07 第 717 行“结果确实执行后才 `_free_encoder_inputs()`；确认进度是 computed 减 placeholders，还须越过媒体末端与 drafter lookahead；enc-dec 因 cross-attention KV 已缓存可释放”与 15 §4.2 同构，量纲与条件都对得上。07 未与 15 双向互链（07 只在依据清单列 `_free_encoder_inputs`），但 15 已声明 owner，不构成矛盾。§5.3 的 encoder 预算例子与 15 §4.1 规则 2–3 仍同构（上一轮已核）。
3. **15 ↔ 16（lookahead 与在途结果）：一致。** 15 §4.2 明确区分“KV slot 预留量 `num_lookahead_tokens`”与“`num_prefill_lookahead`”，并外链 16 §8.4；`Scheduler.__init__` 注释也把这两个量分开列，页面没有混用。
4. **14 ↔ 11/12（prompt logprobs 与 `logprob_token_ids`）：无矛盾，但两处链接不完整。** prompt logprobs 见 N3。`logprob_token_ids`：11/12/07 全无提及，因此 F14-3/F16-6 的投机例外在整个域内**仍无 owner**（E2E-8 未闭合）。

---

## 5. 未核实的疑点（unverified suspicion，本轮）

- 16 §5.1 已把“Step3.5 checkpoint 实际 `num_nextn_predict_layers`”登记为未核验 — 这正是上一轮遗留的不确定性，处理方式恰当，但该值本身仍未核。
- 16 §5.2（F16-5）的 padded 第 3 行在 draft prefill 中是否真的写 KV，仍基于“attention 按 slot_mapping 写所有非 −1 行”的通用行为推断，未逐 backend 打开。
- 15 §7.9 “`EncoderDecoderModelState` 对 `ubatch_idx != 0` 断言 DBO 不支持”仍未打开核实。
- 14 §3.8 的成本量级（每 prompt 位置一次 logits projection + top-k/rank）为结构推断，未实测；页面未宣称实测，合规。
- 所有 GPU / 模型 / 第三方依赖测试本轮**未运行**；测试锚点只做静态存在性与命名核对。

## 6. 统计

- 上一轮：P0 = 1、P1 = 4、P2 = 11（另 E2E 3 条纳入上表）。
- 本轮结果：P0 = 0（已闭合）、P1 = 0（4 条全部闭合）、遗留 P2 = 10（F14-3、F14-4、F15-2…F15-6、F16-4、F16-5、F16-6，含 E2E-8 与 F14-3/F16-6 同源）。
- 本轮新增：P2 = 3（N1、N2、N3）、P3 = 2（N4、N5）。**无新 P0/P1，无回归。**
- 剩余阻断项：**无**。三页的 REJECT 理由均已消除；遗留全为可在后续小改中清掉的 P2/P3，其中最值得优先处理的是 **N1（标题误导）与 F16-4（三/四/六 来源口径不一致）**，以及**为投机下 `logprob_token_ids` 指定一个 owner**（F14-3 + F16-6 合并处理）。
