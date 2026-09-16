# Round-3 独立复核（Group H + D 交叉批次）：vLLM 20 / 24 / 17 / 09 / 18

- 复核者：独立 reviewer（未参与写作）。**只读**：未修改任何 wiki / skills / docs / 源码 checkout。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（`/Users/suhaibo/97-llm/vllm`，复核前后 `git rev-parse HEAD` 一致，仅未跟踪 `artifacts/`，HEAD 未移动）。
- 输入：工作区未提交改动（`git diff`），第二轮验收 `docs/research/2026-09-16-vllm-domain-review-recheck.md` 的 B2 表与四份分组明细。
- 方法：B2 五条逐条回源；第二轮列出的本批 P2 按 diff 是否触及判定；**改稿新增/重写文字逐句回源**（新符号、默认值、分支条件、类归属、计数、图标签）。
- 自跑门禁（`.venv` python 3.13，只读）：
  - `check_links.py --strict` → broken=0 ambiguous=0 bare_index=0 stale_section=0 orphans=0
  - `check_math.py --changed --strict` → 59 files，0 error 0 warning
  - `check_markdown.py --changed --strict` → 59 files，0 error 0 warning
  - `check_assets.py --changed --strict` → 59 files，0 error 0 warning
  - `python -m tools.mkdocs_site.cli build --changed` → **exit 0**，broken_links / missing_anchors / missing_assets / missing_legacy_routes 全 0（orphans 按 scoped 规则 skipped）。B1 的三处锚点已不再中断构建。
- 未运行 vLLM / GPU / HTTP / 故障注入。

---

## 1. 判定行

```
20_vllm_fused_ops_and_kernels_analysis : beat2 pass | hop-walk pass | delete-code pass | figure-trigger transform,layout | algorithm-replay pass | spot-check 14/16 | feature: pass §3.5/§5.3/§7.4/§13 | VERDICT PASS
24_vllm_extension_plugin_system_analysis : beat2 pass | hop-walk pass | delete-code pass | figure-trigger ordered-rule,timing | algorithm-replay pass（平台探测两候选走查可复演） | spot-check 13/16 | feature: FAIL §6.1（失败边界表 MRV2 行不可能成立，且缺 pooling/spec 显式项 ValueError 行）；§3/§3.1/§4/§5.1 的加载亲缘已闭合 | VERDICT CONDITIONAL PASS — blocker N-A
17_vllm_quantization_analysis : beat2 pass | hop-walk pass | delete-code pass | figure-trigger transform,layout,timing | algorithm-replay pass | spot-check 12/12 | feature: pass §4.2/§4.3/§5.1.1/§6.2 | VERDICT PASS
09_vllm_model_library_analysis : beat2 pass | hop-walk pass | delete-code pass | figure-trigger transform,layout | algorithm-replay pass | spot-check 6/6 | feature: pass §2.6.1/§2.8 | VERDICT PASS
18_vllm_distributed_inference_analysis : beat2 pass | hop-walk pass | delete-code pass | figure-trigger layout,transform,timing,coupled-planes | algorithm-replay pass | spot-check 11/12 | feature: pass §4.3/§5.1/§5.2 | VERDICT CONDITIONAL PASS — blocker N-B
```

---

## 2. B2 五条逐条状态

### B2-1｜20 §5.3 + §13 `rocm_aiter_ops.do_fused_qk_norm_rope_and_cache` 不存在 → **FIXED**

- 页面现状（§5.3 行 488）：`fused_qk_norm_rope_and_unified_kv_cache_update_impl` → attention backend impl 的 `do_qk_norm_rope_kvcache_update` → `vllm/_aiter_ops.py::rocm_aiter_ops.do_qk_norm_rope_kvcache_update` → `rocm_aiter_ops.fused_qk_norm_rope_and_cache` → 外部 AITER `fused_qk_norm_rope_cache_pts_quant_shuffle`。§13 行 1068 同步为 `rocm_aiter_ops.do_qk_norm_rope_kvcache_update / rocm_aiter_ops.fused_qk_norm_rope_and_cache`，不存在的符号在两处都已消失。
- 源码核验：`vllm/_aiter_ops.py` 行 1585 `class rocm_aiter_ops`，行 2717 `fused_qk_norm_rope_and_cache`（内部 `from aiter.ops.fused_qk_norm_rope_cache_quant import fused_qk_norm_rope_cache_pts_quant_shuffle`）、行 2776 `do_qk_norm_rope_kvcache_update`；后者 docstring 与页面新写的“中间那一层做两件事”逐条一致（`kv_cache_dtype.startswith("fp8")` 时 `view` 成平台 fp8；按 `cos_sin_cache.shape[-1] == head_dim` 决定传 `0` 还是实际 `rotary_dim`，注释举 GLM-4.7 部分旋转）。两个 backend 方法存在：`rocm_aiter_fa.py:1692 AiterFlashAttentionImpl.do_qk_norm_rope_kvcache_update`、`rocm_aiter_unified_attn.py:339 RocmAiterUnifiedAttentionImpl.do_qk_norm_rope_kvcache_update`。§13 阅读路线按字面 grep 全部可落地。
- 附带：第二轮 N4（“后者关闭 shuffle”对比失准）也已改对——页面现写“`use_shuffle_layout` 由调用方给定，unified 读 NHD 必须传 `False`（其 docstring 明说）”，与 `do_qk_norm_rope_kvcache_update` docstring “passes use_shuffle_layout (unified reads NHD and must pass False)”、unified 行 374 硬传 `False`、FA 行 1727 传 `rocm_aiter_ops.is_shuffle_kv_cache_enabled()` 一致。
- 同段另两条新断言也核过：`QkNormRopeKvCacheFusionPass`（`qk_norm_rope_kvcache_fusion.py:415`）限定 `SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS=(64,128,256)`（行 36/447）；`compilation.py:299-304` 在非 ROCm 上把 `fuse_qk_norm_rope_kvcache` 强制 `False`。

### B2-2｜20 §3.5 lowering-pass UUID 范围 → **FIXED**，20 与 21 已一致

- 页面现状（§3.5 第 3 段）：“21 的 lowering pass 走同一条口径——`VllmIRLoweringPass.uuid` 遍历 `IrOp.registry` 的每个 op，把该 op 的 `get_priority()` 列表本身与**列表中每个** provider 的 `uuid()` 都拼进 key，而不是只取某次 lowering 实际选中的那一个（选择由 `supports_args` 在 lowering 期逐节点决定，早于它的 cache key 无法预知）。”
- 源码核验：`vllm/compilation/passes/ir/lowering_pass.py:115-131` —— `priorities = {name: op.get_priority() for name, op in IrOp.registry.items()}`，随后 `priorities_str` 与 `impl_uuids_str`（`IrOp.registry[name].impls[provider].uuid()` for provider in p）拼进返回值；`selected_impls` 只在 `lower_matched_op`（行 55）记录，用于 debug 日志，不进 uuid ✓。对照 `IrOpPriorityConfig.compute_hash`（`vllm/config/kernel.py:37-59`）只遍历 `asdict(self)` 的两个字段 ✓，页面对两者粒度差异的表述成立。
- 跨页：21 行 527 “`VllmIRLoweringPass.uuid` 包含每个 IR op 的 priority 与 priority 中每个 provider 的 implementation source UUID” —— 与 20 新句同向，矛盾消除 ✓。

### B2-3｜24 把 `vllm.logits_processors` 写成 MRV1 runner 独占 → **PARTIAL（结构缺口已闭合，但新引入一条 P1，见 N-A）**

已修且逐条回源通过的部分：

| 页面新断言 | 源码 | 结果 |
|---|---|---|
| 两个独立进入点；前端链 `InputProcessor._validate_params` → `SamplingParams.verify` → `_validate_logits_processors` → `validate_logits_processors_parameters` → `cached_load_custom_logitsprocs` | `input_processor.py:90-102`（SamplingParams 分支无条件 `params.verify(...)`）；`sampling_params.py:793-810` verify 第 5 行无条件调 `_validate_logits_processors`；行 974-979 → `validate_logits_processors_parameters`；`logits_processor/__init__.py:221` `cached_load_custom_logitsprocs = lru_cache(_load_custom_logitsprocs)`、行 231 循环 | ✓ |
| 前端侧“只有 TPU 早返回；与 runner 版本无关，speculative 也走” | TPU 早返回在 `_load_custom_logitsprocs`（行 176-180），pooling/spec 早返回在 `build_logitsprocs`（行 185-210） | ✓ |
| 失败表现：`RuntimeError` 不被捕获，该函数只把 `validate_params` 抛的 `ValueError` 转 `VLLMValidationError` | `validate_logits_processors_parameters` 只 `except ValueError → VLLMValidationError`，`RuntimeError` 透传 | ✓（“该函数”指链尾函数，读法成立） |
| `_load_logitsprocs_plugins` 不读 `VLLM_PLUGINS`、逐个 `EntryPoint.load()`、“记录目标后抛 `RuntimeError`”、不继续其余候选 | 行 57-84：`logger.debug(... target=%s ...)` → `logger.error` → `raise RuntimeError(...) from e`，抛出即中断循环 | ✓ |
| §3.1 表“安装任一该组插件都会成为 MRV2 blocker”（不论是否走到实际 load） | `vllm/config/vllm.py:2598-2607`：`has_logitsproc_plugins = bool(entry_points(group="vllm.logits_processors"))`，与 `model_config.logits_processors` 或关后 append `"custom logits processors"` | ✓ |
| §4 两棵调用树（前端树 + MRV1 runner 树汇到 `lru_cache`） | 与上述链条一致；`GPUModelRunner.__init__ → build_logitsprocs` ✓ | ✓ |
| 交接锚点 `14#4.3 logits processor 的变体集合从哪里枚举出来` | 14 的实际标题逐字一致；scoped mkdocs 构建 missing_anchors=0 | ✓（B1 该处已闭合） |

未修：见 N-C（`run_method` 异常类型）、N-D（pooling/spec 显式项 `ValueError`）。

### B2-4｜17 §6.1 与 09 §2.8 关于加载跟踪豁免的矛盾 → **FIXED，两页同口径且都对**

- 17 §6.1（行 408）现写：“`has_postprocess_quant` 判断取的是 `getattr(quant_method, "process_weights_after_loading", None)`——该 hook 已定义在 `QuantizeMethodBase` 上，`UnquantizedLinearMethod` 还自行覆写了它，所以**凡是挂着 `quant_method` 的模块，其参数都被无条件补进 loaded set**，普通 linear 权重并不是“可能豁免”而是必然豁免（同一事实的 owner 是 09 §2.8）”。
- 09 §2.8（行 262）：“所有挂有 `quant_method` 的模块都会被必然豁免…`UnquantizedLinearMethod` 还自行覆写了该 hook”。
- 源码：`default_loader.py:447-470`（`has_online_quant or has_postprocess_quant` → 把该模块 `named_parameters()` 全部 add 进 loaded_weights）；`base_config.py:77-82` 基类即定义该 hook（故 `getattr` 恒真）；`linear.py:205` `UnquantizedLinearMethod.process_weights_after_loading` 是覆写 ✓。另核 09 行 259 的默认开启条件：`default_loader.py:436-438` `model_config.quantization is None and loaded_weights is not None`，`enable_weights_track` 显式覆盖 ✓。
- 残留（P2，见 N-E）：17 现在把整条链重述了一遍，而 owner 是 09 §2.8，属于域级“重复解释”倾向。

### B2-5｜18 把 `reinitialize_distributed` / `commit_prepared_elastic_ep` 挂到 `EngineCore` → **FIXED（但同句新引入同类错误，见 N-B）**

- 页面现状：行 90（⑬ 流程表）、行 302（§4.3 正文，并显式注明“这两个入口都定义在 `DPEngineCoreProc` 上，不在基类 `EngineCore`——只有 DP engine proc 有 dp_group 可换；同一入口在 23 §6.3 的写法一致”）、行 304、行 586（§13 路线）全部为 `DPEngineCoreProc.*` ✓。
- 跨页：23 行 414 写 `DPEngineCoreProc.reinitialize_distributed` ✓，两页一致。
- 同节其余新断言逐条回源：`worker_type = "removing" if is_shutdown else "existing"` ✓（core.py:2343）；`scale_type = "scale_down" if is_scale_down else "scale_up"` ✓；重复发起抛 `"Elastic EP reconfiguration is already active"` ✓（2336-2337）；未就绪 commit 抛 `"No prepared Elastic EP reconfiguration is ready"` ✓（2357-2359）。

---

## 3. 第二轮 P2 在本批的落点

| 项 | 页 | 状态 | 依据 |
|---|---|---|---|
| P2-3 “两个 frozenset” | 20 §6.2 | **FIXED（尚余一点）** | 行 513 改为“由两个容器给出（两者都是 11 项，但类型不同）…`frozenset` 的 `_APPLY_MOE_ACTIVATIONS`…`dict` 的 `_MASKED_MOE_ACTIVATION_NAMES`”，与 `fused_moe/activation.py:122-151` 逐项一致（11 + 11）；**masked 路径的 `silu_with_clamp` 名称分支（activation.py:261）仍未提** |
| P2-4 `w2_scale` clamp | 20 §8.x | **FIXED** | 行 765 现写两个都 clamp；`quantization/utils/flashinfer_utils.py:532-535` 对 `w13_scale` 与 `w2_scale` 都 `clamp_(min=1e-10)`，仅 `block_quant` 为真，注释即页面引用的 Hopper NaN 理由 |
| P2-5 `vllm/models/` 与 15 个模型文件 | 20 §1.6/§5.2 | **FIXED** | 行 419 点名 `vllm/models/{deepseek_v4,minimax_m3,glm5next,kimi_k3}` 并指出 `minimax_m3/common/vision_tower.py`、`glm5next/nvidia/multimodal.py` 也直接构造；实测 `grep -rl "ApplyRotaryEmb(" vllm/model_executor/models/` **恰 15 个文件**，`vllm/models/` 下**恰 2 个**且与页面点名一致 |
| P2-6 XPU 默认 priority | 20 §3.2 | **FIXED** | 行 271 补 XPU 段；`platforms/xpu.py:481-493` 同形（`inductor && mode!=NONE → ["native"]`，否则 `["vllm_c","native"]`，无 AITER/oink）✓ |
| P2-7 Helion “存在未接线” | 20 §3.2 | **FIXED（措辞略宽，见 N-F）** | 行 271 新增该段，`vllm/kernels/helion/ops/` 确有 `fused_qk_norm_rope`、`silu_mul_fp8`、`rms_norm_dynamic_per_token_quant` 等 8 个文件，且无平台把它们写进默认 priority ✓ |
| P2-8 ⑦ 的 dtype/head_dim 门 | 20 §1.6 | **FIXED** | 行 212 补 `model_config.dtype ∈ {bf16,fp16}` 与“每一层 head_size ∈ (64,128,256)，一层不合整体退回”；`qk_norm_rope_fusion.py:213-238` 一致（页面说“两道门”，源码另有第三条早返回：未发现 attention 层 → 仅告警不启用，极轻） |
| N5 HIP 原地合同 | 20 §5.2 | **NOT_FIXED** | 行 417 仍写“CUDA 的设备路径与**直接复用它的** HIP 分支…原地”；`rotary_embedding/base.py:254-271` 的 `use_aiter` 分支调 `rocm_aiter_triton_rotary_embedding(...)` 后 `return query, key`，同样原地 |
| N6 图边标 `pairOffset = 4 / 2` | 20 §5.3 图 | **NOT_FIXED** | 图内边标未改，仍可读成“4 或 2” |
| N7 两张新图未用 house 配色 | 20 §5.3/§9.1 图 | **NOT_FIXED** | 仍只有 `classDef compute fill:#e8f1fb` 与 `cost fill:#fff1dd`，其余节点无 class |
| N3 24 §2 缺失 worker 方法的异常类型 | 24 §2 | **NOT_FIXED** | 行 67 仍写 `AttributeError`；`vllm/v1/serial_utils.py:501-507` 捕获 `getattr` 的 `AttributeError` 后 `raise NotImplementedError(f"Method {method!r} is not implemented.") from None` |
| N4 pooling/speculative + 显式 `--logits-processors` 的 `ValueError` | 24 §3.1/§6.1 | **NOT_FIXED** | §3.1 门控列仍只写“pooling 返回空集合、speculative 只保留 `MinTokensLogitsProcessor`”，§6.1 仍写“pooling/speculative 的早返回不进入此分支”；源码 `build_logitsprocs`（行 185-210）在这两支上**先判显式项**：`raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)` / `STR_SPEC_DEC_REJECTS_LOGITSPROCS` |
| NEW-P2-a `initialize_online_processing` 父子关系 | 17 §8.3 | **FIXED** | 调用树行 563/566 已拆两条 lane：AutoGPTQ 侧 `register_parameter(qweight/scales/qzeros/g_idx) [不调 initialize_online_processing]`，在线侧 `OnlineLinearBase.create_weights` 下 `register_parameter(weight)` 与 `initialize_online_processing(layer) [同级末尾语句…]` |
| 旧 P2-8 `_already_called_*` flag 位置 | 17 §3.3 | **FIXED** | 行 191 明写“**不在 `_setup_kernel` 里**——由三个子类各自在 `_setup_kernel` 返回之后才设”，并把它与 §6.2 的 `delattr` 串起来 |
| 旧 P2-9 batch-invariant 归属过宽 | 17 §1.1 #10 / §8.3 | **FIXED** | #10 行、§7.2 配置行（行 530）、§8.3 行 608 三处都限定为“在线 FP8 的 `apply`（Cutlass 除外）” |
| NEW-P2-c §4.3 建立段两处不完整 | 17 §4.3 | **FIXED** | 行 247 改为“**6 个** host 镜像”并列出 `_prob_scale_float`（`attention.py:143-148` 恰 6 个）；同段新增唯一硬失败：`kv_cache_dtype == "fp8_e5m2"` 时除非 `CompressedTensorsKVCacheMethod` 且 `kv_cache_scheme is None` 否则 `ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")`（`attention.py:199-215`，含注释理由）✓ |
| NEW-P2-d “实例属性赋值遮蔽类级 loader”用错对象 | 17 §4.3 | **FIXED** | 行 249 改为“括注…说的是 `KVCacheScaleParameter` 自己 docstring 对基类哨兵参数的描述，不要套到 CT 身上…CT 的 `create_weights` 建的是普通 `torch.nn.Parameter`…只是一次普通实例赋值” ✓ |
| NEW-P2-b 锚点与显示标签不一致 | 17 §3.3 | **NOT_FIXED，且扩散到 18** | 17 行 193 仍是 `#2.6 融合前分别分片…|模型库 §2.6.1`；18 行 352（新增文字）用同一个错配锚点。09 的 MoE 小节标题是 `#### 2.6.1 MoE 专家写入：名字中的全局专家先映射成本地槽`（行 217），读者会落到行 182 的 dense QKV 小节。`check_links` 过（锚点可解析） |
| 旧 P2-9 `DPEngineCoreProc.barrier()` test-only | 18 ⑦ | **FIXED** | 行 84/122 两处均写明 docstring 标注 test-only utility；`core.py:2147-2151`（位于 `DPEngineCoreProc`）docstring 逐字为 “Blocking barrier on the DP process group (test-only utility).” ✓ |
| 旧 P2-10 `get_response_mqs` assert 以 output rank 为界 | 18 §1.3③ | **FIXED** | 行 118 改为“以 **`world_size`** 为界（`-1 <= unique_reply_rank < world_size`），不是以 output rank 为界；且 `collective_rpc` 不走 `get_response_mqs`”；`multiproc_executor.py:268-273` 一致，且全库 `get_response_mqs` 除定义处无其他调用点 ✓ |
| P2-lite “普通 MP 路径” | 18 §5.1 | **FIXED** | 现写“普通 MP 路径由 `Worker.execute_model` …；MRV1 的 `external_launcher` 路径也会在 `GPUModelRunner.execute_model` 的 `broadcast_pp_output` 分支构造同名映射”，对照轴已点明 |
| 09 §2.8 dtype/target-device scope、§4.1 `bind_kv_cache` | 09 | **保持 FIXED** | 本轮 09 仅动 contradiction 块措辞与 §13 路线（新增 `QuantizeMethodBase.process_weights_after_loading`），两处都对 |

`ensure_model_parallel_initialized`（18 ②，新增文字）另核：`parallel_state.py:2183-2205` 恰四条 **size** assert（TP/PP/PCP/DCP），不校验成员列表，也不覆盖 `_DP`/`_EP`/`_EPLB` ✓。
`data_parallel_index`（18 §12 配置表，新增文字）：`config/parallel.py:398` `Field(init=False)`、行 939 `__post_init__` 置为 `data_parallel_rank`、`core.py:1337` `run_engine_core` 覆写为 `dp_rank` ✓。
09 §2.6.1 新表的“非本地专家 `weight_loader(..., return_success=True)` 返回 false”：`routed_experts.py:622/645-647` ✓。
24 平台探测两候选走查（新增文字）：`platforms/__init__.py:243-286` —— 逐 factory 调一次；OOT 命中 1 个直接选它（内置同时命中被忽略）；对选中的 factory **再调一次**取 qualname（故必须幂等）；≥2 OOT 或 0 OOT+≥2 内置才抛 `RuntimeError` ✓。
24 `VLLM_PLUGINS` 跨组 allowlist（新增文字）：`plugins/__init__.py:37-74` 的过滤对任何 group 生效，五个 group 全部经 `load_plugins_by_group`（general 87 / platform 243 / IO 59 / stat logger 77 / endpoint 132），`VLLM_PLUGINS=""` 解析为 `[""]` 而非 unset（docstring 行 107-109）✓。
24 LoRA resolver（新增文字）：`lora/resolver.py:44-69` 注册表存**实例**，覆盖告警原文 “overwritten by the new resolver instance”；`openai/models/serving.py:113-117` 构造时 `get_resolver` 取引用快照 ✓。

---

## 4. 新发现（改稿引入 / 仍未闭合）

### N-A（**P1**，新引入）24 §3.1 正文 + §6.1 表 —— “MRV2 部署里也会在第一份采样请求上炸出来”不可能成立，且与同段自述矛盾

- 页面（§3.1 末段）：“**因此“普通 MRV1 generation 才加载”是错的**：一个导入失败的插件，**在 MRV2 部署里同样会在第一份采样请求上炸出来**——只不过它是请求级错误而不是启动失败。这些早返回都不改变配置阶段“已安装插件会阻断 MRV2”的事实。”
  §6.1 表同形：“该请求在参数校验期失败；与 runner 版本无关，**MRV2 部署同样命中**”。
- 源码：`vllm/config/vllm.py:2598-2607` 把“装了任一 `vllm.logits_processors` entry point **或** 设了 `--logits-processors`”记为 `"custom logits processors"` 这项 MRV2 unsupported feature。于是两种情形都不可能出现 MRV2：
  - `VLLM_USE_V2_MODEL_RUNNER` 未设 → `use_v2_model_runner`（行 653-684）看到 unsupported 非空即 `warning_once` 后 `return False`，本部署跑的是 MRV1；
  - 显式 `VLLM_USE_V2_MODEL_RUNNER=1` → `VllmConfig.__post_init__`（行 1661-1662）调 `_validate_v2_model_runner()`（行 2777-2786）直接 `raise ValueError("Model Runner V2 does not yet support: custom logits processors")`，**在任何请求之前**启动即失败。
- 影响：页面用来论证“加载点不唯一”的那个例子指向一个不存在的部署形态，并与紧随其后的“已安装插件会阻断 MRV2”直接冲突；读者会以为 MRV2 + 该组插件是可运行组合。前端加载点本身的结论不受影响（那部分已核对无误）。
- 建议修复：把例子换成**真实成立**的那一种——MRV1 + 投机解码：`build_logitsprocs` 在 spec 分支早返回（不加载），而前端 `SamplingParams.verify` 仍会在第一份采样请求上加载并抛 `RuntimeError`；两个进入点分处 API server 与 worker 两个进程，因此“runner 侧没加载”不等于“不会加载”。§6.1 那一行相应改为“与 runner 侧早返回无关（例如 spec 下 runner 不加载，前端仍加载）”，删掉 MRV2 字样。

### N-B（**P1**，新引入，与 B2-5 同类）18 §4.3 正文 + §13 路线 —— `worker_type="new"` 挂到 `EngineCore._eep_scale_up_before_kv_init`，而基类该方法只 `raise NotImplementedError`

- 页面（行 302）：“`worker_type="new"` 由新加入 engine 自己的 **`EngineCore._eep_scale_up_before_kv_init`** 在 KV 初始化之前产生。” §13 行 586 同样列 `EngineCore._eep_scale_up_before_kv_init`。
- 源码：`vllm/v1/engine/core.py:1023-1024`（位于基类 `EngineCore`，105-1031）函数体是 `raise NotImplementedError`；真正构造 `ElasticEPScalingState(..., worker_type="new", scale_type="scale_up", reconfig_request=None)` 并 `state.run_pre_kv_init_states()` 的是 `DPEngineCoreProc._eep_scale_up_before_kv_init`（行 2397-2414）。调用点在 `EngineCore.__init__` 行 141-142（`if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:`），确实早于行 145 的 `_initialize_kv_caches`，所以“KV 初始化之前”这半句对。
- 影响：与 B2-5 修掉的正是同一类类归属错误——本轮在修 A 的同一句里引入了 B。按 CLAUDE.md 的 provenance 要求（稳定锚点=路径+限定符号），§13 的这一格也指不到实体。
- 建议修复：正文与 §13 都改为 `DPEngineCoreProc._eep_scale_up_before_kv_init`，并可补一句“基类同名方法只 `raise NotImplementedError`，调用点在 `EngineCore.__init__` 的 `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` 分支”。

### N-C（P2，未修）24 §2 —— `AttributeError` 应为 `NotImplementedError`

同第二轮 N3。行 67 原文：“…都没有提供该方法，就会抛 `AttributeError`。” 源码 `vllm/v1/serial_utils.py:501-507` 吞掉 `AttributeError` 并抛 `NotImplementedError(f"Method {method!r} is not implemented.")`。建议按实际异常与消息改写。

### N-D（P2，未修）24 §3.1 门控列 + §6.1 —— 缺 pooling / speculative 配了显式 `--logits-processors` 时的启动硬失败

同第二轮 N4，证据见 §3 表。建议 §6.1 增一行（“pooling / speculative 且配置了显式 `--logits-processors`” → `ValueError` → runner 初始化失败），§3.1 把“早返回”限定为“无显式项时”。

### N-E（P2，新引入）17 §6.1 —— 与 09 §2.8 口径已统一，但改成了第二份完整解释

行 408 现在把“hook 定义在 `QuantizeMethodBase` / `UnquantizedLinearMethod` 覆写 / 无条件补进 loaded set”整条链重述一遍，同时又声明 owner 是 09 §2.8。域级评审把“重复解释”列为唯一变差项，这里是同类风险。建议压成一句结论 + 链接（例如“tracking 对挂有 `quant_method` 的模块无条件豁免，理由与范围见 09 §2.8”），把机制留在 09。

### N-F（P2，新引入，很轻）20 §3.2 Helion 段 —— “全库除 `collect_env.py` 与 `utils/import_utils.py` 外没有任何引用点”把测试目录漏在外

`tests/kernels/helion/` 下至少 7 个文件（`test_helion_available.py`、`test_autotune.py`、`test_fused_qk_norm_rope.py`、`test_per_token_group_fp8_quant.py`、`test_case_key.py`、`test_benchmark_script.py`、`utils.py`）直接引用这些 kernel，包内还有 `register.py`/`config_manager.py` 的自有注册设施。更准确的“未接线”判据是：`vllm/kernels/__init__.py` 只 `from . import aiter_ops, oink_ops, vllm_c`，**不导入 helion**，而 `Platform.import_ir_kernels()` 默认导入的正是 `vllm.kernels`，所以这些实现根本不进 `IrOp` 注册表。建议把结论换成这条判据，并把“全库”限定为“生产代码路径”。

### N-G（P2，未修，第二轮 N9）20 页头 `最近更新：2026-09-15`

本轮改动落在 2026-09-16。若维持域级统一日期口径则可留；否则应同步。

---

## 5. 跨页复核

1. **20 ↔ 21 lowering UUID**：口径已一致（B2-2）。
2. **20 ↔ 17 Marlin workspace / sort indices**：17 §6.2 现在明确“workspace 不走 layer 快照、sort indices 是 Parameter 走 copy-back”，并把 kernel 内用途给 20、通用捕获约束给 19、`WorkspaceManager` 给 19；20 §8.2 侧表述未与之冲突 ✓。
3. **17 ↔ 09 tracking 豁免**：结论一致（B2-4），残留为重复解释（N-E）与锚点标签错配（NEW-P2-b）。
4. **17 / 18 → 09 §2.6.1**：两页都用 `#2.6 …` 锚点配 `§2.6.1` 标签，落点错（NEW-P2-b 扩散）。
5. **18 ↔ 23 Elastic EP 类归属**：`DPEngineCoreProc.reinitialize_distributed` 两页一致 ✓；但 18 新增的 `_eep_scale_up_before_kv_init` 归属错（N-B）。
6. **24 ↔ 14**：24 现在把“加载点不唯一 / 构造点唯一”的区别写在自己这侧并指向 14 §4.3 的真实标题；14 侧是否回指未在本批要求内。

---

## 6. 小结

- B2 五条：**FIXED 4**（20 两条、17/09 一条、18 一条）、**PARTIAL 1**（24，结构缺口闭合但新增一条 P1）。
- 新增 **P1 两条**：N-A（24 的 MRV2 断言不可能成立且自相矛盾）、N-B（18 把 `worker_type="new"` 挂到只抛 `NotImplementedError` 的基类方法）。两条都是单句级改写。
- 新增 P2 三条（N-E、N-F、N-G），未修 P2 六条（20 的 N5/N6/N7，24 的 N3/N4，17 的 NEW-P2-b 且已扩散到 18）。
- 本批无新 P0；T0 四项与 scoped mkdocs 构建全绿。20/17/09 可判 PASS，24/18 待 N-A/N-B 修完即可。
