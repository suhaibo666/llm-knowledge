# Group H 独立审读：vLLM 20（融合算子与 Kernel）与 21（IR 与融合 Pass）

- 审读者：独立 reviewer（未参与两页写作）；仅报告，不改页面。
- 冻结源码：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，本机 `/Users/suhaibo/97-llm/vllm`，`git log` 提交时间 `2026-09-06T17:54:32-07:00`（= 2026-09-07 00:54 UTC）；HEAD 未移动，只读 grep/sed。
- 方法：source-fidelity / codebase / analysis-focus / page-review-rubric / feature-analysis profile + review。
- 页面：
  - `wiki/02_engineering/03_infer_frameworks/vllm/20_vllm_fused_ops_and_kernels_analysis.md`
  - `wiki/02_engineering/03_infer_frameworks/vllm/21_vllm_ir_and_fusion_passes_analysis.md`
  - 邻页核对：`19_vllm_compilation_cudagraph_analysis.md`（§4.2、字段移交段、Related）、`15_vllm_multimodal_execution_analysis.md` §6.2、`17_vllm_quantization_analysis.md` 行 193/437。

---

## 1. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 20_vllm_fused_ops_and_kernels_analysis | pass | FAIL §1.4 图2 / §4.3 / §10 RMSNorm 调用树 | pass | transform, layout | FAIL §9.1（LoRA 稳定排序分组仅表格）；§5.1/§5.3 rope 配对与 warp shuffle 仅表格 | 16/20（4 处与源码不符） | REJECT | feature: FAIL §1.5⑤⑩ 变体枚举依据不完整、§1.6 基础/条件分类与默认编译路径矛盾、§7.4 contradiction 结论不实、§11.2 fallback 分类过度概括 |
| 21_vllm_ir_and_fusion_passes_analysis | pass | pass | pass | transform, layout | pass | 16/16（2 处锚点精度小瑕疵） | PASS（附 P1 跨页交接修正） | feature: pass；委托给 20 的「OOT implementation 注册、容差执行侧」在 20 中不存在（跨页悬空交接，需两页协同修） |

说明：
- 20 的 hop-walk 失败不是调用链本身写错，而是把「torch-wrap 关闭 / 非 codegen 的 eager dispatch」画成了默认 decoder 循环里的真实路径；默认 O2（`VLLM_COMPILE` + `inductor`）下，provider 选择发生在 21 的 `VllmIRLoweringPass.lower_matched_op`（fake args），且 CUDA 默认 priority 为 `["native"]`。
- 20 的 algorithm-replay：图 1（norm/quant）、图 3（MoE align/GEMM）、Marlin SVG 均能复演最小例；但 LoRA 分组（9 token → 三段）是排序/分组算法，仅有表格；rope NEOX/GPT-J 配对与融合 kernel 的 warp shuffle 是布局算法，仅有表格与文字。按 rubric「table alone does not pass」记 FAIL（最小失败单元如上）。

---

## 2. 锚点抽查记录

### 2.1 页面 20（20 个锚点）

| # | 锚点（path::symbol） | 页面主张 | 源码结果 |
|---|---|---|---|
| 1 | `vllm/model_executor/custom_op.py::CustomOp.dispatch_forward` | 6 平台分支 + disabled `maybe_compile(forward_native)` | ✔ ROCm→`forward_hip`、cpu/tpu/xpu/oot/else cuda；disabled 分支同时写 `disabled_custom_ops` |
| 2 | `vllm/ir/op.py::IrOp.dispatch / _filter_priority_impls` | 首个 `supports_args` 命中；末项不覆盖抛 `RuntimeError`；首个 `supports_all_args` 截断，无则追加 native + `warning_once` | ✔（另：priority 中若含 `supported=False` 实现 dispatch 抛 `ValueError`，页未提，无碍） |
| 3 | `vllm/platforms/cuda.py::CudaPlatformBase.get_default_ir_op_priority`；`rocm.py::RocmPlatform.get_default_ir_op_priority` | Inductor 时 `["native"]`，否则 `["vllm_c","native"]`；`VLLM_USE_OINK_OPS` 前插 oink；ROCm AITER 四条件 | ✔（XPU 也有同形默认，页未提，P2） |
| 4 | `vllm/config/kernel.py::KernelConfig / IrOpPriorityConfig / MoEBackend / LinearBackend` | 7 / 2 字段；21 / 21 取值 | ✔ |
| 5 | `csrc/libtorch_stable/layernorm_kernels.cu::fused_add_rms_norm / fused_add_rms_norm_kernel` | `vector_width=8`、对齐 16B、`max_block_size` 1024/256、batch-invariant 关向量分支、两循环间 `BlockReduce`+`__syncthreads()` 并重读 residual | ✔ |
| 6 | `csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu::rms_norm_per_block_quant / _dispatch` | 四条 groupwise guard、`%4` 三检查、512/256 阶梯；per-token 入口 `min(hidden,1024)` | ✔ guard 与 block 数值均对；**但「norm+quant 有两个 `_C` 入口」不成立**，见 F4 |
| 7 | `csrc/libtorch_stable/pos_encoding_kernels.cu::apply_token_rotary_embedding / rotary_embedding` | NEOX `(rot_offset, embed_dim+rot_offset)` 读 `cos[x]`；GPT-J `(2r,2r+1)` 读 `cos[x/2]`；`block(min(num_heads*rot_dim/2,512))` | ✔ 复演 rot_dim=8、rot_offset=1：NEOX (ch1,ch5)，GPT-J (ch2,ch3)，均读 cos[1] |
| 8 | `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu::fused_qk_norm_rope` | SM 9.0 自动阈值 4096/8192、10240/40960；`rotary_bytes%16` 退 1-head；`blockSize=256` | ✔ |
| 9 | `vllm/model_executor/layers/activation.py` | 16 个 `@CustomOp.register`；3 类在 `__init__` 改 `_forward_method` | ✔ |
| 10 | `vllm/model_executor/layers/fused_moe/activation.py::_APPLY_MOE_ACTIVATIONS / _apply_moe_activation_masked` | 11 项；masked 下 SWIGLUOAI 7.0/1.702 | ✔（`_MASKED_MOE_ACTIVATION_NAMES` 是 dict 不是 frozenset，P2） |
| 11 | `vllm/model_executor/layers/fused_moe/modular_kernel.py::FusedMoEExperts.is_supported_config` | 通用 11 项检查，无一读 `swiglu_limit` | ✔ 字面正确；**但 §7.4 contradiction 的结论不实**，见 F3 |
| 12 | `vllm/model_executor/layers/fused_moe/oracle/unquantized.py::_get_priority_backends / select_unquantized_moe_backend` | CUDA 4 候选、Hopper 后移、两条 HACK、显式 `TRITON→BATCHED_TRITON` 无日志改写、LoRA 前置返回、humming→auto | ✔；**§11.2 把「唯一改写」推广到所有显式 family 不成立**，见 F5 |
| 13 | `vllm/model_executor/layers/fused_moe/fused_moe.py::_prepare_expert_assignment` | `expert_map is None and num_tokens*topk*4 <= global_num_experts` 且排除 WNA16 block | ✔；复演 2×2×4=16 ≤ 16 成立 |
| 14 | `csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu::gptq_marlin_repack_kernel` | lane 5：`tc_col=1,tc_row=2,cur_n=1`；8 nibble 顺序；`out_offset+20` | ✔ 逐 nibble 与 `vals[i]/vals[4+i]` + `pack_idx={0,2,4,6,1,3,5,7}` 完全一致；SVG 网格 lane 号与公式一致 |
| 15 | `vllm/model_executor/layers/quantization/utils/marlin_utils.py::get_scale_perms / marlin_permute_scales`；`vllm/model_executor/kernels/linear/mixed_precision/marlin.py::MarlinLinearKernel.can_implement` | 两组置换定义与选择条件；192%128≠0 拒绝 | ✔ |
| 16 | `modular_kernel.py::FusedMoEKernelModularImpl._fused_experts` | `use_output_alias` 四条件、ROCm 另需 `is_fused_moe_enabled()`、~94% copy 注释 | ✔ |
| 17 | `vllm/lora/ops/triton_ops/lora_kernel_metadata.py::LoRAKernelMeta.prepare_tensors`；`punica_gpu.py::add_lora_linear`；`base_linear.py::create_lora_weights` | 稳定排序 `[3,4,0,1,2,5,6,7,8]`、counts 2/3/4；FP32 注释只引 Triton issue；scratch 末维 = `max_lora_rank` | ✔ |
| 18 | `vllm/model_executor/layers/fused_moe/config.py::FusedMoEConfig / FusedMoEQuantConfig` | 31 / 9 字段；13+18 划分 | ✔（AST 计数） |
| 19 | `vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py::fused_qk_norm_rope_and_unified_kv_cache_update_impl / QkNormRopeKvCacheFusionPass` | 目标 op 为「把 KV 写入也并进来的那一族」（紧接 csrc DeepSeek-V4 / MiniMax-M3 两个 kernel） | ✘ 实际经 `attn_layer.impl.do_qk_norm_rope_kvcache_update` → ROCm AITER `fused_qk_norm_rope_cache_pts_quant_shuffle`；csrc 两个 kernel 由 `vllm/models/minimax_m3/...` 模型代码直调。见 F6 |
| 20 | `fused_moe.py::get_default_config / get_moe_configs / invoke_fused_moe_triton_kernel` | 四分支、H200 折叠、batch-invariant 返回 None、最近 M、`SPLIT_K=1`、`BLOCK_SIZE_K` pop | ✔ |

另核对：MoE 手算例（`y_A=1.25c`、`y_B=1.75c`、`E0:(0,3,4,4)/E1:(1,2,4,4)`、`num_tokens_post_padded=8`、`(log3,0)`→0.75/0.25）全部可复演；§1.1 INT8 例 `q=(64,127,64,127)`、`128/127` 正确；图 1「32 B/16 B」算术正确。

### 2.2 页面 21（16 个锚点）

| # | 锚点 | 页面主张 | 源码结果 |
|---|---|---|---|
| 1 | `vllm/compilation/passes/pass_manager.py::PostGradPassManager.configure / __call__ / uuid` | 1 pre-grad + 21 条件类 + 4 固定尾 = 26；router-pad → AR+RMS → reshape → RMS+quant；三合一先于 RoPE+KV；SplitCoalescing 最多 3 次 | ✔ 逐行核对 append 顺序与类数 |
| 2 | `InductorPass.is_applicable_for_range` 及 6 个 override | 仅 SP、AsyncTP、AR（CUDA/ROCm）、RopeKV、QkNormRopeKv | ✔ |
| 3 | `vllm/compilation/passes/fusion/add_rms_fusion.py::AddRMSNormFusionPass / RMSNormReshapeFusionPass` | 2×2 与 2×2 注册 | ✔ |
| 4 | `vllm/compilation/passes/ir/inplace_functionalization.py::VllmIRInplaceFunctionalizationPass.__call__` | later-user `ValueError`、placeholder 进 donation、改 target | ✔ |
| 5 | `vllm/compilation/passes/ir/clone_elimination.py::UnsafeCloneEliminationPass.__call__ / clone_preserves_layout / user_writes_to_node` | 缺 metadata 视为 preserved；donation 只约束被写的 clone；HOP 例外 | ✔（「write 后无 user」检查在 `__call__`，页锚到 `user_writes_to_node`，P2） |
| 6 | `vllm/compilation/passes/ir/lowering_pass.py::VllmIRLoweringPass.lower_matched_op / uuid` | fake args dispatch、`func_impl_fn`、`run_functional_passes=False`、两条 TODO、uuid 含 priority 与 impl uuid | ✔ |
| 7 | `vllm/compilation/backends.py::VllmBackend.configure_post_pass` | pre-grad 独占 assert、`_cache_config_ignore_prefix`、manager 拒收、用户 pass append | ✔ |
| 8 | `vllm/config/compilation.py::PassConfig`（字段与 `__post_init__`） | 16 字段、默认、平台裁决、五条 warn | ✔（AST 计数 16；CompilationConfig 36） |
| 9 | `vllm/config/vllm.py::OPTIMIZATION_LEVEL_00..03 / IS_QUANTIZED / IS_DENSE / enable_*` | 各级默认；两常量 False；MLA 默认函数无平台判据 | ✔ |
| 10 | `vllm/compilation/passes/fusion/sequence_parallelism.py::get_sequence_parallelism_threshold / SP_MIN_*` | SM90/SM100 家族 8192；8/32 MiB；XPU 4096/8；公式 | ✔ |
| 11 | `vllm/compilation/passes/fusion/allreduce_rms_fusion.py::FI_ALLREDUCE_FUSION_MAX_SIZE_MB / _fused_ar_workspace_hidden_dim`；`vllm/config/vllm.py::VllmConfig._set_compile_ranges` | 真值表含 103/107；门用 target/draft 较大 hidden，端点只用 target | ✔（分歧后果为读码推断，页已标注） |
| 12 | `vllm/compilation/passes/fusion/qk_norm_rope_fusion.py::QKNormRoPEFusionPass` | 仅 FP16/BF16；head_dim 白名单；无 range override；`forced_token_heads_per_warp=-1` | ✔ |
| 13 | `CompilationConfig.__post_init__` | `enable_auto_functionalized_v2` 缺键才设 False；`+rotary_embedding` 强制追加与 issue 28042；combo kernels | ✔ |
| 14 | `vllm/envs.py::VLLM_ENABLE_PREGRAD_PASSES`；`compiler_interface.py::InductorStandaloneAdaptor.compile` | torch<2.12 且 0 时 patch `_recursive_pre_grad_passes` | ✔ |
| 15 | `docs/design/fusions.md::Quick Reference` | 各 indicative 百分比 | ✔ 数字一致（但页未列出另两处文档/源码冲突，P2） |
| 16 | `vllm/compilation/passes/fusion/collective_fusion.py::AsyncTPPass / FlashInferAllGatherFP4Pattern` | NVFP4 仅 AG→GEMM，无 GEMM→RS | ✔ |

---

## 3. E2E / 连贯性

目标链：模型 forward 调用 → CustomOp / IR op → 编译期 pass 改写（21）→ provider/kernel 选择（20）→ 设备 kernel。

1. **21 的位置陈述完整**：§1 与 §8.1 明确「Dynamo FX → pre-grad functionalization → AOTAutograd → post-grad 融合 → cleanup → lowering（fake args dispatch）→ clone elimination → FixFunctionalization → 交回 19」，并区分 torch-wrap 关闭时 IR 相关改写失去对象。与 19 §4.2（range 端点产生/消费归 19、阈值语义归 21）及 19 字段移交段一致。
2. **20 缺少自己在链上的位置**（F1，P0/P1）：全文没有 `torch_wrap` / lowering / fake 的任何论述。§4.3 的四阶段与 §10 调用树是 eager（torch-wrap 关闭）路径，图 2 却把 `CustomOp → IrOp → IrOpImpl → torch.ops._C.fused_add_rms_norm` 画成 decoder 层循环的真实执行；默认 O2 下 `custom_ops=["none"]`（`VllmConfig.__post_init__`）、`ir_enable_torch_wrap=True`、CUDA priority `["native"]`，实际是 IR 节点被 21 的 lowering 以 native 展开后交 Inductor codegen，`_C.fused_add_rms_norm`、`_C.rotary_embedding`、`_C.silu_and_mul` 都不在默认路径上。两页对「同一次 add+RMSNorm 在哪里选 provider、是否有保护 clone」给出两套互不引用的叙事：20 §4.3 完成点「没有 copy」，21 §4.3/§7 说 inplace provider 在编译路径先插保护 clone、仅在 donation 等条件满足时回收。
3. **所有权术语冲突**（F2，P1）：20 四处写「融合开关取值归 19」，而 19 与 21 均写 `pass_config` 16/16 字段移交 21。
4. **同一融合两页讲法不一致**（F6，P1）：QK-Norm+RoPE+KV 三合一——20 暗示落到 csrc DeepSeek-V4/MiniMax-M3 KV-insert 族，21 正确写为 AITER 三合一；20 未说明该 pass 仅 ROCm/AITER。
5. **21→20 的悬空交接**（F7，P1）：21 §7/§8.4/§10.2/Related 三次把「OOT implementation 注册」「容差执行侧（`get_tolerance` / `ir_test_utils.assert_close`）」「新增 provider 用 `register_impl`」交给 20，但 20 只讲了 `register_impl` docstring 中 `supported/supports_args` 的分工，没有 OOT 平台 `Platform.import_ir_kernels` 覆写、`docs/design/vllm_ir.md::Out-of-Tree Implementations`、声明容差如何被 provider 对拍消费的任何内容。
6. **norm+quant 变体两页不对齐**（F4，P1）：21 §6.1 正确枚举 static FP8 / dynamic per-token / per-group 三类替换；20 ⑤ 与 §4.3 只列两个 `_C` 入口，漏掉 static FP8 的 `rms_norm_static_fp8_quant` / `fused_add_rms_norm_static_fp8_quant`（`csrc/libtorch_stable/layernorm_quant_kernels.cu`）。
7. 10/17/19/15 的入向交接（rope kernel 归 20、§8.3/§8.4 布局与重排理由、compiled graph 选中的 kernel 归 20）在 20 中均已兑现，交叉 § 号与 17 行 193 一致。

---

## 4. 编号发现

### P0

**F1. [20 §1.6「基础与条件」表 ④⑥⑧ 行；关联 §1.4 图 2、§4.3、§10 调用树] 默认执行路径分类错误**
- 页面主张：「“基础”指未量化、无 LoRA、单卡、默认 moe_backend=auto 的普通文本推理也会经过」；「基础 | ④ | CUDA/ROCm 且 `variance_size is None`、weight dtype 匹配」；「基础 | ⑥ | 除少数无 rope 架构外恒走」。
- 源码证据：
  - `vllm/config/vllm.py::VllmConfig` 默认 `optimization_level=O2`；`__post_init__` 在 mode 为 None 且 level>O0 时设 `CompilationMode.VLLM_COMPILE`；`vllm/config/compilation.py::CompilationConfig.__post_init__` 空 backend 取平台 `simple_compile_backend="inductor"`；随后 `backend=="inductor" and mode!=NONE` 时 `custom_ops.append("none")`，并设 `ir_enable_torch_wrap = mode==VLLM_COMPILE and backend=="inductor"`。
  - `vllm/model_executor/custom_op.py::CustomOp.enabled / default_on`：`"none"` 下未 `+name` 的 CustomOp 关闭 → `dispatch_forward` 走 `maybe_compile(forward_native)`，`RotaryEmbedding.forward_cuda`（`_C.rotary_embedding`）、`SiluAndMul.forward_cuda` 不执行。
  - `vllm/platforms/cuda.py::CudaPlatformBase.get_default_ir_op_priority`：`using_inductor` 时 `["native"]`，`vllm_c` 的 `fused_add_rms_norm` 不在默认 priority；`VllmIRLoweringPass.lower_matched_op` 按此 priority 以 fake args 选 native。
- 影响：读者会以为默认服务在跑 `_C.fused_add_rms_norm` / `_C.rotary_embedding` 设备 kernel；页面自己 §3.2 与 §1.4 段落又说 CUDA Inductor 默认 native，形成内部矛盾。
- 建议：把 ④⑥⑧ 的设备 kernel 行改为「条件：非 codegen（mode NONE / backend eager / torch-wrap 关）或显式 `ir_op_priority` / `+rotary_embedding`（含 21 的 rope 融合开关强制追加）」；基础行只保留 ①（disabled 分支）与 IR op 节点本身；在 §1.4/§4.3 开头加一段「默认 VLLM_COMPILE 下 dispatch 在 21 lowering 发生（fake args、保护 clone、donation 回收），本节 walk 的是 torch-wrap 关闭的 eager 路径」，并在图 2 把 `IRO→IMP→DEV` 边标为「eager / 或 lowering 期选定」两态。

### P1

**F2. [20 §1.6 ⑦ 行、§2 交接表 19 行、§5.3 末段、Related Pages 19 条] PassConfig 开关取值的所有权与 19/21 冲突**
- 页面主张：「开关取值归 19，pass 归 21」；「19 … 以及 `pass_config` 上那几个融合开关的取值」。
- 证据：`19_vllm_compilation_cudagraph_analysis.md` 字段移交段「明确移交的 6 个：… `pass_config` → 21（阈值语义归 21）」；21 §9「本页拥有 `PassConfig` 的 16/16」。20 §12 自身也写「`PassConfig` 的 16 个字段全部归 21」，与前四处自相矛盾。
- 建议：四处统一改为「开关取值与解析归 21 §9.1；range 端点消费归 19 §4.2」。

**F3. [20 §7.4 `[!contradiction] clamp 配置注释与通用能力检查不一致`] 结论不实**
- 页面主张：「但当前通用函数的 11 项检查 … 没有任何一项读 `moe_config.swiglu_limit`」「旧页将注释写成统一运行时保证，现予纠正」。
- 源码：`vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py::TrtLlmFp8Experts.is_supported_config`（覆写后先调通用版）在 `swiglu_limit/alpha/beta` 任一非 None 且 quant key 不在 `{(kMxfp8Static,kMxfp8Dynamic),(kFp8Static128BlockSym,kFp8Dynamic128Sym)}` 或 activation 不在 `{SILU, SWIGLUOAI_UNINTERLEAVE}` 时返回 `False`。另有 10 个 experts 类覆写 `is_supported_config`。注释所说的过滤在「per-kernel 覆写」这一机制下对 TRTLLM FP8 已兑现。
- 建议：改写为「过滤机制是各 experts 类覆写 `is_supported_config`；当前仅 TRTLLM FP8 实现了 swiglu 参数过滤，Triton 在 `activation()` 内实现 clamp，FlashInfer CUTLASS/CuteDSL/AITER 等把 `gemm1_clamp_limit` 透传给外部库（其内部未验证）；其余类既不过滤也未证明实现」；同时在 §2 所有权表把「11 项求交」改为「通用 11 项 + 类覆写」。

**F4. [20 §1.5 枚举依据与 ⑤ 行、§4.3「norm+quant 有两个 `_C` 入口」、§1.1/§1.6 INT8 触发] norm+quant 变体集合不完整且枚举依据选错**
- 页面主张：「④⑤ 来自 `vllm/kernels/vllm_c.py` 注册到这两个 IR op 的 impl 加 `_C` 侧两个 quant 入口」；「norm+quant 有两个 `_C` 入口」；⑤ 触发「下游 GEMM 吃 FP8/INT8」。
- 源码：`vllm/compilation/passes/fusion/rms_quant_fusion.py::FUSED_OPS` 映射三类 op：`_C.rms_norm_static_fp8_quant` / `_C.fused_add_rms_norm_static_fp8_quant`（实现于 `csrc/libtorch_stable/layernorm_quant_kernels.cu`）、`_C.rms_norm_dynamic_per_token_quant`、`_C.rms_norm_per_block_quant`；key 全为 FP8（`kFp8StaticTensorSym/kFp8DynamicTokenSym/kFp8Dynamic64/128Sym`），无 INT8 键——INT8 输出只是 kernel 能力，没有 pass 产出它。
- 建议：以 `FUSED_OPS` 为枚举依据列出 4 个 `_C` 入口（含 static FP8 两个）；⑤ 触发改为「FP8 static/per-token/per-group，经 21 的 `RMSNormQuantFusionPass`」，INT8 标为「kernel 支持、当前无 pass 产出」；ROCm AITER 族注明归 21 §6.1。

**F5. [20 §7.4 显式 backend 段、§11.2 表第 4 行] 「唯一的名称改写是 TRITON→BATCHED_TRITON」过度推广**
- 页面主张：「唯一的名称改写是 batched activation format 下的 `TRITON → BATCHED_TRITON`」；§11.2「唯一的例外是 §7.4 那条 `TRITON → BATCHED_TRITON` 的无日志改写」。
- 源码：`vllm/model_executor/layers/fused_moe/oracle/fp8.py::select_fp8_moe_backend` 显式分支在 batched format 下有三条改写（`DEEPGEMM→BATCHED_DEEPGEMM`、`TRITON→BATCHED_TRITON`、`VLLM_CUTLASS→BATCHED_VLLM_CUTLASS`），另有 `VLLM_USE_DEEP_GEMM`/`VLLM_MOE_USE_DEEP_GEMM` 显式 set 时直选或摘除 DeepGEMM、`allow_vllm_cutlass=False` 时显式 CUTLASS 直接 `ValueError`。
- 建议：§7.4 限定为「未量化 oracle 中唯一…」；§11.2 改为按 oracle 列出改写集合（unquantized 1 条、FP8 3 条 + DeepGEMM/AITER env 前置分支）。

**F6. [20 §5.3「谁决定要不要用融合形态…」段与前一段] QK-Norm+RoPE+KV 融合的落点叙述与 21 矛盾**
- 页面主张：先列 `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` 与 `fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`「把 KV 写入也并进来」，再称 `QkNormRopeKvCacheFusionPass` 目标 op「即把 KV 写入也并进来的那一族」。
- 源码：`qk_norm_rope_kvcache_fusion.py::fused_qk_norm_rope_and_unified_kv_cache_update_impl` 调 `attn_layer.impl.do_qk_norm_rope_kvcache_update`，实现只在 `vllm/v1/attention/backends/rocm_aiter_fa.py` 与 `rocm_aiter_unified_attn.py`，内部调 AITER `fused_qk_norm_rope_cache_pts_quant_shuffle`（pass 注释亦指该函数）；`PassConfig.__post_init__` 非 ROCm 强制关闭。csrc MiniMax-M3 kernel 由 `vllm/models/minimax_m3/{nvidia,amd}/model.py` 直调 `ops.fused_minimax_m3_qknorm_rope_kv_insert`，不经 pass。21 §6.3 写的是 AITER 三合一，正确。
- 建议：拆成两句——「csrc 两个 KV-insert 变体由模型代码直调（点名调用点，不经 pass）」；「`QkNormRopeKvCacheFusionPass` 仅 ROCm/AITER，经 attention backend 进入外部 AITER kernel（依赖边界，归 21 §6.3 / 10）」。

**F7. [21 §7 末段、§8.4 第 2 段、§10.2 末段、Related Pages 20 条；对应 20 §3] 21 交给 20 的内容在 20 中不存在**
- 21 主张：「OOT implementation 注册机制归 20」「provider 对拍与 benchmark 则检验这些声明是否兑现（容差执行侧归 20）」「注册 provider … 前者由 20 号页展开」。
- 20 实况：仅 §3.3 引 `register_impl` docstring 的 `supported/supports_args` 分工，§4.1 泛称「每个 provider 必须与 native 相符」；没有 `vllm/platforms/interface.py::Platform.import_ir_kernels`（OOT 平台覆写入口）、`docs/design/vllm_ir.md::Out-of-Tree Implementations`、`tests/ir/test_op.py::test_uuid_and_oot`、`IrOp.get_tolerance` / `tests/ir/ir_test_utils.py::assert_close` 的任何论述；`grep 容差|tolerance` 在 20 仅两处泛述。
- 建议：二选一——在 20 §3 增一小节「新增/OOT provider 的接入合同」（`register_impl` 参数、schema 一致性检查、`import_ir_kernels`、uuid 入 cache、按声明容差对拍）；或 21 收回这三处委托并自行给出。需在两页之间统一。

**F8. [20 §1.5 ⑩ 行枚举依据「来自 `fused_moe/oracle/` 目录」] 变体集合与枚举依据不符，兄弟选择轴未点名**
- 源码：`vllm/model_executor/layers/fused_moe/oracle/` 含 `unquantized.py fp8.py int8.py int_wna16.py mxfp4.py mxfp8.py nvfp4.py w4a8.py w4a8_int8.py` 9 个 oracle；20 只展开 unquantized 与 fp8，其余 7 个在 20 与 17 中都未出现（`rg MoeBackend` 仅命中 Fp8/Unquantized）。
- 建议：在 ⑩ 与 §7.4 开头列出 9 个 oracle 及其 backend 枚举名，声明本页展开范围（unquantized/fp8）与其余 owner（若归 17 需 17 接收，否则写明「存在、未展开」）。

**F9. [20 §9.1 LoRA 分组；§5.1/§5.3 rope 配对与 warp shuffle] 算法触发存在但无原理图（rubric check 5）**
- 页面有可复演的小例（9 token → `[3,4,0,1,2,5,6,7,8]` / `0→2→5→9`；rot_dim=8 的配对），但只以表格/文字呈现；rubric 与 feature profile 明确「table alone does not pass」。
- 建议：为 LoRA 稳定排序→三段→shrink/expand 回写原 token 位画一张布局图；rope 用同一 rot_dim=8 例画 NEOX（跨半区配对→融合 kernel 需 `__shfl_xor_sync`）与 GPT-J（相邻配对→无通信）两 lane 对照图。

### P2（共 13 条）

1. [20 页头] 基线日期写「2026-09-08」，实际提交 2026-09-06 17:54 -0700 = 2026-09-07 00:54 UTC（index 写 09-07 UTC）；21 写 2026-09-06（本地时区），与 index 口径不一。建议两页统一为 UTC 2026-09-07。
2. [20 §5.2「完成点：四条路径都以 query 被原地旋转…没有任何一条返回新分配的输出」] `RotaryEmbedding.forward_xpu` 在 `key is None` 时走 `forward_native → forward_static`，以 `torch.cat` 返回新张量；与同节上一条 XPU 描述自相矛盾。
3. [20 §6.2] 「由两个 frozenset 给出」——`_MASKED_MOE_ACTIVATION_NAMES` 是 `dict[MoEActivation,str]`；masked 路径另有 `silu_with_clamp` 名称分支未提。
4. [20 §8.3 FlashInfer 第 5 段] 源码对 `w13_scale` 与 `w2_scale` 都 `clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)`（`flashinfer_utils.py::prepare_fp8_moe_layer_for_fi`），页面只写 w13。
5. [20 §5.2 ApplyRotaryEmb] 「15 个模型文件直接构造」只数 `vllm/model_executor/models/`；`vllm/models/glm5next/nvidia/multimodal.py` 与 `vllm/models/minimax_m3/common/vision_tower.py` 也直接构造，兄弟模型目录 `vllm/models/` 全页未点名。
6. [20 §3.2] 平台默认只列 CUDA/ROCm；`vllm/platforms/xpu.py::XPUPlatform.get_default_ir_op_priority` 同形默认、`vllm_c` 的 `supported=GPGPU_DEVICE` 含 XPU（§1.6 ④ 触发只写 CUDA/ROCm）。
7. [20 §1.5 ⑦ / §5.3] `vllm/kernels/helion/` 下存在 Helion 预调参 kernel（`fused_qk_norm_rope`、`rms_norm_per_block_quant`、`silu_and_mul_per_block_quant` 等），当前仅 scripts/tests 使用、未接入生产 dispatch；建议点名「存在、未接线」以免读者误判缺席。
8. [20 §1.6 ⑦ 条件] 漏写 `QKNormRoPEFusionPass` 自身的 FP16/BF16 dtype 门、head_dim 白名单与 `enable_qk_norm_rope_fusion` 平台裁决（CUDA-alike/XPU），以及强制 `+rotary_embedding`。
9. [21 §9.4 contradiction] 文档/源码冲突还有两处未列：`docs/design/fusions.md::Quick Reference` 把 QK Norm + RoPE 的 `num_tokens` 标为 Low，而 `QKNormRoPEFusionPass` 无 range override（21 §11 已写「对所有 range 都应用」）；Quick Reference 称 RMSNorm+Quant 融合「FP8/FP4」，而 `RMSNormQuantFusionPass` 无 NVFP4 replacement（21 §6.1 已写）。
10. [21 §9.1 #1/#2「ROCm 上另加」] 实际条件是 `rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled()`（`PostGradPassManager.configure`），不是平台为 ROCm 即加。
11. [21 §4.3 第 2 条] 「original 在该 write 后不能再有 user」的检查在 `UnsafeCloneEliminationPass.__call__`，锚点写成 `user_writes_to_node`。
12. [21 §8.3 调用树注释「逐 pass 递增，是顺序的可观察证据」] `dump_prefix` 仅在 pass 实际执行后 `+= 1`，被 range 跳过的 pass 不占序号，故 `{i}` 不等于装配位置。
13. [21 §9「其余 30 个…归 19」vs 20 §12 与 19「21 个归 19、9 个内部/计算字段」] 计数口径不一，建议 21 改为「21 个归 19、9 个内部字段」。

### 未验证的怀疑（未计入发现）

- 21 §4.2「donation 集合跨过 AOTAutograd」：pre-grad 记录的是 pre-grad 图中 placeholder 的 `node_to_idx`，post-grad 图 placeholder 次序是否保持一致（参数 lift 等）未核实；若不一致，`UnsafeCloneEliminationPass` 的 `node_to_idx[original_node] not in donated_input_ids` 可能错配。
- 21 §9.3：`VLLM_ENABLE_PREGRAD_PASSES=0`（torch<2.12 standalone）时 `maybe_inplace` 节点是否会以 `auto_functionalized(maybe_inplace)` 形式残留而使 lowering 仅告警「Failed to lower」，未追到 AOTAutograd 内部。
- 21 §9.1 #9「level 默认晚于平台裁决、不重跑 `__post_init__`」：未核 `@config` 的 pydantic `validate_assignment` 是否会触发额外校验；沿用前轮结论。
- 20 §7.1「`moe_sum` 以 FP32 累加」：未打开 csrc `moe_sum` kernel。

---

## 5. 小结

- 21：源码忠实度高，抽查 16 个锚点未发现事实错误；主要问题是向 20 委托了 20 不承载的内容（F7）与少量文档冲突/锚点精度（P2）。
- 20：单点 kernel 算术与布局（norm/rope/qk-rope/Marlin/MoE align/LoRA）复演准确，但在「默认路径是什么」「在编译链上的位置」「变体集合是否穷尽」「fallback/contradiction 结论」上有 1 个 P0 与 7 个 P1，需回到 Phase 2 修订后再审。
