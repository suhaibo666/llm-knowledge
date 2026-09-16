# Group H 复审（第二轮）：vLLM 20（融合算子与 Kernel）与 21（IR 与融合 Pass）

- 审读者：独立 reviewer（未参与写作）；仅报告，不改页面。
- 冻结源码：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（本机 `/Users/suhaibo/97-llm/vllm`，`git rev-parse HEAD` 已复核未移动，提交时间 `2026-09-06 17:54:32 -0700` = 2026-09-07 00:54 UTC）。只读 grep/sed。
- 改动量：`20` +235/−56（`git diff --stat`：235 插入侧含新增两图与 §3.5），`21` 仅页头 3 行（基线日期口径 + 最近更新）。
- 方法：逐条复核上一轮 F1–F9 与 13 条 P2；新增文本按 source-fidelity 逐句回源；两张新图按 algorithm-replay 门自行复演；T0 门自跑。

---

## 1. 判定行

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 20_vllm_fused_ops_and_kernels_analysis | pass | pass（§1.4/§4.3/§10 已分离「默认编译」与「eager walk」两条上下文，图 2 边改为两态） | pass | transform, layout（两张新图均带 spec 注释） | pass（RoPE 配对图、LoRA 分组写回图均可独立复演，算术与源码一致） | 30/32（2 处与源码不符，见 N1/N2） | **PASS（须修 N1、N2 两条 P1；P0 已解除）** | feature: pass §1.5/§1.6/§7.4/§11.2（枚举依据改为可回源集合、基础/条件分类与默认路径自洽、contradiction 改写为三层能力检查、fallback 合同按 oracle 分述） |
| 21_vllm_ir_and_fusion_passes_analysis | pass | pass | pass | transform, layout | pass | 2/2（仅页头改动） | **PASS** | feature: pass；F7 悬空交接已由 20 §3.5 承接落地；但 20 新增一句与 21 §9 冲突（N1），需 20 侧改 |

**上轮 P0 已解除**：默认 O2 下 `custom_ops` 追加 `"none"`、`ir_enable_torch_wrap=True`（均在 `vllm/config/vllm.py::VllmConfig.__post_init__` 第 1445–1460 行）、CUDA `using_inductor = backend=="inductor" and mode != NONE` 时 priority 为 `["native"]`（`vllm/platforms/cuda.py::CudaPlatformBase.get_default_ir_op_priority`）三项事实与页面新文本完全一致；`RMSNorm.forward_cuda` 非 batch-invariant 直接 `return self.forward_native(...)`、`forward_native` 调 `ir.ops.fused_add_rms_norm.maybe_inplace`（`vllm/model_executor/layers/layernorm.py`），也支持 §10 新增的「默认编译树」。

---

## 2. 上一轮findings 状态表

| # | 原级别 | 状态 | 依据（冻结源码 / 页面现状） |
|---|---|---|---|
| F1 默认执行路径分类错误 | P0 | **FIXED** | §1.4 新增前置段（默认 O2/`VLLM_COMPILE`/inductor → `none` + torch-wrap → lowering 以 fake 实参选 native → codegen），并明确「不能画成每层必调 `_C.*`」；§1.6 把 ④⑥⑧ 与 ⑨⑩⑪⑫⑬⑭⑮ 全部改判为「条件」，仅留 ①②③ 与 ⑳ 为基础；图 2 的 `IRO/IMP/DEV/ROPE` 节点与边改为两态；§4.3 与 §10 均先声明「本节/本树是 torch-wrap 关闭的 eager 调用」。④ 行还补了一条精确判据「仅关闭 torch-wrap 不会自动改变 Inductor 的 native priority」——与 `using_inductor` 只看 backend/mode 一致，正确。另核：O2/O3 的 `OPTIMIZATION_LEVEL_02` 把 `enable_qk_norm_rope_fusion` 硬设 `False`，故 ⑥⑦ 判为「条件」不过度保守。 |
| F2 PassConfig 开关取值所有权冲突 | P1 | **FIXED** | 四处（§1.6 ⑦、§2 交接表 21/19 两行、§5.3 末段、Related 19 条）统一为「开关取值与解析归 21 §9.1，range 端点消费归 19 §4.2」；与 19 行 560「6 个明确移交 → 21」及 20 §12「`PassConfig` 16 个全归 21」自洽。 |
| F3 §7.4 clamp contradiction 结论不实 | P1 | **FIXED** | contradiction 块改写为「clamp 能力检查有三层」。逐条回源：`experts/trtllm_fp8_moe.py::TrtLlmFp8ExpertsBase.is_supported_config`（先调通用版，SwiGLU 三参任一非 None 时只允许 `(kMxfp8Static,kMxfp8Dynamic)`/`(kFp8Static128BlockSym,kFp8Dynamic128Sym)` 且 activation ∈ {SILU, SWIGLUOAI_UNINTERLEAVE}）✔；`experts/aiter_mxfp8_moe.py::AiterMxfp8Experts.is_supported_config`（flydsl 可用性 + SWIGLUOAI_UNINTERLEAVE + alpha/beta 与硬编码值 `math.isclose`）✔；`oracle/nvfp4.py::select_nvfp4_moe_backend`（`swiglu_limit is not None` 时按 `NVFP4_BACKENDS_WITH_CLAMP` 过滤 auto 候选；显式分支在名字规范化后对白名单外 family 抛 `ValueError`）✔。类名比上一轮我给的 `TrtLlmFp8Experts` 更精确。 |
| F4 norm+quant 变体集合不完整 / 枚举依据选错 | P1 | **FIXED** | §1.5 枚举依据改为 `rms_quant_fusion.py::FUSED_OPS`；§4.3 新增四入口表。核对 `FUSED_OPS`：`rms_norm_static_fp8_quant`、`fused_add_rms_norm_static_fp8_quant`、`rms_norm_dynamic_per_token_quant`（fused_add True/False 同一 op）、`rms_norm_per_block_quant`（`hasattr` 守卫，`kFp8Dynamic64Sym`/`kFp8Dynamic128Sym` ×2）——共 4 个 `_C` 入口、键全为 FP8 ✔。static 两入口的 guard 与 §页面描述逐条一致（无 residual 仅查 `out.is_contiguous()`；fused-add 另查 residual contiguous、residual/input、weight/input dtype 相同），`VLLM_STABLE_DISPATCH_FP8_TYPES` 限定输出 ✔；`rms_norm_static_fp8_quant_kernel` 先 CUB 归约 RMS 再按传入 `scale_inv` 量化，无动态 absmax ✔。§1.1 与 §1.6 也补上「INT8 是 kernel 能力、当前无 pass 替换键」✔。 |
| F5 「唯一名称改写」过度推广 | P1 | **FIXED** | §7.4 限定为「unquantized oracle 这条显式分支内」，并补 FP8 三条（`DEEPGEMM→BATCHED_DEEPGEMM`、`TRITON→BATCHED_TRITON`、`VLLM_CUTLASS→BATCHED_VLLM_CUTLASS`）与 NVFP4 一条（`FLASHINFER_CUTEDSL→FLASHINFER_CUTEDSL_BATCHED`），均与 `oracle/fp8.py::select_fp8_moe_backend`、`oracle/nvfp4.py` 一致 ✔；`allow_vllm_cutlass` 拒绝、`VLLM_USE_DEEP_GEMM`/`VLLM_MOE_USE_DEEP_GEMM` 与 `VLLM_ROCM_USE_AITER*` 的「摘除或直选」前置分支、非 CUDA/ROCm 返回 `Fp8MoeBackend.NONE, None` 也逐条对上 ✔。§11.2 fallback 行同步改写 ✔。 |
| F6 QK-Norm+RoPE+KV 融合落点 | P1 | **PARTIAL** | 方向已纠正且比我上一轮的描述更完整：csrc 两个 KV-insert 变体确由模型代码直调（`vllm/models/deepseek_v4/attention.py::_fused_qnorm_rope_kv_insert` 按 cache dtype 分派 `_C` op ✔；`vllm/models/minimax_m3/nvidia/model.py::MiniMaxM3Attention.forward` 行 370 调 `ops.fused_minimax_m3_qknorm_rope_kv_insert` ✔），pass 走 attention backend → AITER ✔，并点明「该 pass 不落到 DeepSeek/MiniMax C kernel」✔。但链条里的中间符号写错（N2），且「后者关闭 shuffle」的对比失准（N4）。 |
| F7 21→20 悬空交接 | P1 | **FIXED** | 新增 §3.5「新增与 OOT provider：注册、导入、缓存与容差要一起接上」，覆盖 21 三处委托：`register_impl` 参数与 `infer_schema` schema 一致性、`supports_args` 签名约束、`inplace` 需 `allow_inplace`（`vllm/ir/op.py::IrOpImpl.__init__` 逐条对上 ✔）；`Platform.import_ir_kernels`（默认导入 `vllm.kernels`，OOT 覆写）与导入点 `IrOpPriorityConfig._iter_op_priorities` ✔；`docs/design/vllm_ir.md::Out-of-Tree Implementations`、`tests/ir/test_op.py::test_uuid_and_oot`（改文件→UUID 变、还原→复原）✔；`IrOp.get_tolerance`（override → `DEFAULT_TOLERANCES` → `ValueError`）与 `tests/ir/ir_test_utils.py::assert_close`（按 **actual 输出 dtype** 取容差、递归 tuple/list、失败提示 `override_tolerance`）✔；`tests/kernels/ir/test_layernorm.py::test_impls`（skip 不支持实参 → 与 native 对拍 → 再验 dispatch == 直调）✔。21 行 26/527/619/760 的委托现在都能落到实体内容。**唯一瑕疵见 N1。** |
| F8 oracle 枚举依据与兄弟选择轴 | P1 | **FIXED** | §7.4 新增九类 oracle 表。逐一核对文件、稳定入口与返回枚举：`unquantized/UnquantizedMoeBackend`、`fp8/Fp8MoeBackend`、`int8/Int8MoeBackend`、`int_wna16::select_wna16_moe_backend/WNA16MoEBackend`、`mxfp4::select_mxfp4_moe_backend + select_deepseek_v4_mxfp4_moe_backend/Mxfp4MoeBackend`、`mxfp8::select_mxfp8_moe_backend` 返回 `Fp8MoeBackend`（页面明确「不虚构独立 enum」）、`nvfp4/NvFp4MoeBackend`、`w4a8/W4A8MoeBackend`（`class W4A8MoeBackend(Enum): CUTLASS` 单值 ✔）、`w4a8_int8/W4A8Int8MoeBackend`（单值 `CPU_INT4` ✔）——九行全部正确，含大小写 `WNA16MoEBackend`。覆盖边界声明清楚。 |
| F9 算法触发缺原理图 | P1 | **FIXED** | 新增两张 Mermaid 图并各带 figure spec 注释：§5.3 的 RoPE 配对通信图、§9.1 的 LoRA 分组写回图。二者均独立复演通过（见 §3）。 |
| P2-1 页头基线日期 | P2 | **FIXED** | 20 与 21 均改为「`main` 快照，2026-09-07 UTC」，与 index 口径一致。 |
| P2-2 §5.2 完成点与 XPU 自相矛盾 | P2 | **FIXED** | 改为「CUDA 设备路径与直接复用它的 HIP 分支、CPU 路径原地；XPU 的 `key is None` fallback 与 disabled/native 会 materialize」。（残留小瑕见 N5。） |
| P2-3 §6.2「两个 frozenset」 | P2 | **NOT_FIXED** | 行 513 仍写「由两个 frozenset 给出」，同句后半又称 `_MASKED_MOE_ACTIVATION_NAMES` 是「11 项到字符串名的**映射**」——句内自相矛盾；源码是 `dict[MoEActivation,str]`。masked 路径的 `silu_with_clamp` 名称分支仍未提。 |
| P2-4 `w2_scale` clamp 漏写 | P2 | **NOT_FIXED** | 行 765 仍只写 `w13_scale.clamp_(min=1e-10)`；`flashinfer_utils.py` 行 533–535 对 `w13_scale` 与 `w2_scale` 都 clamp 到 `_FI_CUTLASS_MIN_BLOCK_SCALE`。 |
| P2-5 `vllm/models/` 兄弟目录未点名 / 15 个模型文件 | P2 | **PARTIAL** | §5.3 现已点名 `vllm/models/deepseek_v4/…` 与 `vllm/models/minimax_m3/nvidia/…`（目录不再全页缺席）；但 §1.6 与 §5.2 的 `ApplyRotaryEmb`「15 个模型文件」仍只数 `vllm/model_executor/models/`，`vllm/models/glm5next/nvidia/multimodal.py`、`vllm/models/minimax_m3/common/vision_tower.py` 仍未计入。 |
| P2-6 XPU 平台默认 priority | P2 | **NOT_FIXED** | §3.2 平台默认仍只列 CUDA/ROCm；`vllm/platforms/xpu.py::XPUPlatform.get_default_ir_op_priority` 同形默认未提。（§1.6 ④ 行已不再写「CUDA/ROCm」，该子项自然消解。） |
| P2-7 Helion 预调参 kernel「存在、未接线」 | P2 | **NOT_FIXED** | 全页 `grep -i helion` 无命中。 |
| P2-8 ⑦ 条件漏 dtype/head_dim/平台裁决 | P2 | **NOT_FIXED** | §1.6 ⑦ 仍只写「模型有 q/k norm 且 `enable_qk_norm_rope_fusion` 打开」，未写 `QKNormRoPEFusionPass` 自身的 FP16/BF16 门、head_dim 白名单与 CUDA-alike/XPU 裁决。 |
| P2-9 21 §9.4 另两处文档冲突 | P2 | **NOT_FIXED** | 21 正文未动。 |
| P2-10 21 §9.1 ROCm 条件 | P2 | **NOT_FIXED** | 同上。 |
| P2-11 21 §4.3 锚点精度 | P2 | **NOT_FIXED** | 同上。 |
| P2-12 21 §8.3 `dump_prefix` 注释 | P2 | **NOT_FIXED** | 同上。 |
| P2-13 `CompilationConfig` 字段计数口径 | P2 | **NOT_FIXED** | 21 行 634/678 仍写「其余 30 个归 19」；20 §12 与 19 行 560 的口径是「21 个归 19 + 9 个内部/计算字段 + 6 个移交 21」。两边各自自洽（21+9=30），但对外表述仍不一致。 |

小计：20 号页 —— FIXED 8（F1–F5、F7–F9 中属 20 的部分）、PARTIAL 2（F6、P2-5）、NOT_FIXED 5（P2-3/4/6/7/8）；21 号页 —— FIXED 1（P2-1）、NOT_FIXED 5（P2-9~13），其 F7 侧交接由 20 承接后成立。

---

## 3. 新图复演（algorithm-replay 门）

两张图都先给 spec 注释再给图，问题陈述、输入、中间态、输出与锚点齐备，非「表格代图」。

**§5.3 RoPE 配对通信图**（锚 `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu::fusedQKNormRopeKernel`）。取 `head_dim=64` → `numElemsPerThread = head_dim/32 = 2`，故 lane0 持 ch0/ch1、lane1 持 ch2/ch3、lane2 持 ch4/ch5；`rot_dim=8` → `rotary_lanes = 8/2 = 4`，lane0–3 参与旋转，ch8–ch63 只归一化（图上写「其余 56 通道」✔）。源码行 101 注明 `interleave = !is_neox`，故 `if constexpr (interleave)` 是 GPT-J、else 分支是 NEOX ——与图一致。
- NEOX：`pairOffset = (rotary_dim/2)/numElemsPerThread = 4/2 = 2`；lane0 `__shfl_xor_sync(...,2)` 取 lane2，i=1 → 伙伴为 ch5；`dim_idx = 0*2+1 = 1 → (1*2)%8 = 2 → half_dim = 1` → 读 `cos[1]/sin[1]`；`laneId(0) < pairOffset` 故伙伴取负 → `ch1' = n1·c1 − n5·s1`；lane2 侧不取负 → `ch5' = n5·c1 + n1·s1`。与图上两条公式逐项一致 ✔。两次 `__syncwarp()`（shuffle 前后）与图上「syncwarp → shuffle XOR 2 …算完再 syncwarp」一致 ✔。
- GPT-J：lane1 内 `(idx0,idx1)=(0,1)` → `dim_idx = 1*2+0 = 2`、`half_dim = 1` → 同样 `cos[1]`；`ch2' = n2·c1 − n3·s1`、`ch3' = n3·c1 + n2·s1`，无 shuffle。与图一致 ✔。
- 瑕疵：边标 `pairOffset = 4 / 2` 未给结果，需读者自行读成 `(rot_dim/2)/numElemsPerThread = 2`，与节点里的「shuffle XOR 2」配合才闭合（N6）。

**§9.1 LoRA 分组写回图**（锚 `LoRAKernelMeta.prepare_tensors`、`_lora_shrink_kernel`/`_lora_expand_kernel`、`kernel_utils.py::do_shrink_kernel/do_expand_kernel`、`PunicaWrapperGPU.add_lora_linear`）。原 mapping `[0,0,0,-1,-1,2,2,2,2]` → 稳定排序索引 `[3,4,0,1,2,5,6,7,8]`、slots `[-1,0,2]`、counts `[2,3,4]`、prefix `0→2→5→9`（上一轮已复演 ✔）。新增的三项主张逐条落地：
- 两个 kernel 都在 `lora_id == -1` 处 `return`（`_lora_shrink_kernel` 行 74、`_lora_expand_kernel` 行 67）→ 图上「区间 [0,2)：ram = 3,4；shrink 与 expand 早退」✔，y 的行 3/4 保持 base output ✔。
- `ram = tl.load(cta_lora_seq_indices + offset_m)` 装的是**原行号**；`do_shrink_kernel` 的 `a_ptr` 与 `c_ptr` 都用 `ram[:,None]*stride` 寻址，并 `accumulator *= scaling`；`do_expand_kernel` 的输入（行 184）与输出（行 227）同样用 `ram` → 图上「按 ram gather 原 x / scratch 仍以原 token 行寻址 / 无需逆排序」✔。
- scratch：`add_lora_linear` 里 `buffer = torch.empty((len(output_slices), x.size(0), r), dtype=torch.float32)` → 本例每 slice 为 `9 × rank` 的 FP32 ✔；`add_inputs` 默认 `True`（行 133/326）→「默认 add_inputs 累加回 y 的原行」✔。

---

## 4. 新增 / 回归 findings

### P1

**N1. [20 §3.5 第 3 段] 「lowering pass 纳入实际选中实现的 UUID」与源码及 21 冲突（新引入的事实错误 + 新的跨页矛盾）**
- 页面原文：「`IrOpImpl.uuid` 对实现所在源码文件求 hash；priority 配置把列表中实现的 UUID 纳入身份，**21 的 lowering pass 另纳入实际选中实现的 UUID**。」
- 源码：`vllm/compilation/passes/ir/lowering_pass.py::VllmIRLoweringPass.uuid` 取 `priorities = {name: op.get_priority() for name, op in IrOp.registry.items()}`，再对 **priority 里的每一个 provider** 求 `impls[provider].uuid()` 拼串，与「这次 dispatch 选中了谁」无关；`selected_impls` 只在 `lower_matched_op` 里记录用于日志/统计，不进 uuid。
- 跨页：21 行 527 写的是「包含每个 IR op 的 priority 与 priority 中每个 provider 的 implementation source UUID」——正确；20 的新句与之直接矛盾。
- 影响：读者会以为编译缓存身份随「本次选中的 provider」变化，从而误判换 priority 顺序/换未选中 provider 的源码是否会失效缓存（实际都会）。
- 建议：改为「lowering pass 的 uuid 覆盖 `IrOp.registry` 里**全部** op 的 priority 列表及其中**每个** provider 的源码 UUID，粒度比 `IrOpPriorityConfig.compute_hash`（只遍历配置的 2 个字段）更宽；`selected_impls` 只用于记录，不进身份」。

**N2. [20 §5.3 末段 + §13「KV-cache 融合的真实端点」行] `rocm_aiter_ops.do_fused_qk_norm_rope_and_cache` 这个符号在基线中不存在**
- 页面原文（两处同形）：「… → attention backend 的 `do_qk_norm_rope_kvcache_update` → **`rocm_aiter_ops.do_fused_qk_norm_rope_and_cache`** / `fused_qk_norm_rope_and_cache` → 外部 AITER `fused_qk_norm_rope_cache_pts_quant_shuffle`」。
- 源码：`grep -rn "do_fused_qk_norm_rope_and_cache" vllm/ tests/ csrc/ docs/` 无命中。`vllm/_aiter_ops.py` 上只有两个相关 static：`fused_qk_norm_rope_and_cache`（行 2717，内部 `from aiter.ops.fused_qk_norm_rope_cache_quant import fused_qk_norm_rope_cache_pts_quant_shuffle` 后调用）与 `do_qk_norm_rope_kvcache_update`（行 2776，处理 fp8 view 与 partial-rotary 后调用前者）。两个 attention impl 调的都是 `rocm_aiter_ops.do_qk_norm_rope_kvcache_update`。
- 影响：§13 是本页的源码阅读路线，这一格按字面 grep 落空；且读者会以为 backend 方法与 aiter 层方法是两个不同名字。
- 建议：链条改为「attention impl 的 `do_qk_norm_rope_kvcache_update` → `rocm_aiter_ops.do_qk_norm_rope_kvcache_update`（同名，另一层；负责 fp8 `view` 与 `kernel_rotary_dim` 换算）→ `rocm_aiter_ops.fused_qk_norm_rope_and_cache` → 外部 AITER `fused_qk_norm_rope_cache_pts_quant_shuffle`」，§13 同步。

### P2

**N3. [20 §3.5 第 2 段] 「设置 priority 和计算相应身份之前先让平台实现进入注册表」把 compute_hash 也算进导入点**
- `current_platform.import_ir_kernels()` 只在 `IrOpPriorityConfig._iter_op_priorities` 中调用，而该函数只被 `set_default()` / `set_priority()` 使用；`IrOpPriorityConfig.compute_hash` 走 `asdict(self)` 直接索引 `IrOp.registry[name].impls[provider]`，本身不触发导入（依赖此前已导入）。同段「用户可以覆盖」也偏松：`KernelConfig.set_platform_defaults` 对用户已给的列表是**追加**平台默认（源码注释与 `IrOpPriorityConfig` docstring 都写 appended），不是替换。
- 建议：「导入发生在 `set_default`/`set_priority` 经过的 `_iter_op_priorities`；`compute_hash` 假定注册已完成」，并把「覆盖」改为「用户列表在前、平台默认追加在后」。

**N4. [20 §5.3「已核实 Aiter FlashAttention 与 Aiter Unified 两个 backend；后者关闭 shuffle」] 对比失准**
- `rocm_aiter_unified_attn.py` 传 `use_shuffle_layout=False`（硬编码）✔；但 `rocm_aiter_fa.py` 的 `fused_qk_norm_rope_kvcache_supported()` 返回 `rocm_aiter_ops.is_enabled() and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()`（注释：shuffle 写路径另有专用 cache update），即**融合能生效时 FA 侧 shuffle 也必然是关的**，只是它把该值当参数传下去。
- 建议：「两个 backend 在融合生效时都不用 shuffle 布局：unified 硬传 `False`，FA 则以 `not is_shuffle_kv_cache_enabled()` 作为启用前提」。

**N5. [20 §5.2 完成点] HIP 的 AITER Triton 分支被排除在原地合同之外，但它同样原地**
- `RotaryEmbedding.forward_hip` 在 `self.use_aiter` 为真时调 `self.rocm_aiter_triton_rotary_embedding(...)` 后 `return query, key`（同一对张量），与 `forward_cuda` 一样原地。现表述「CUDA 的设备路径与**直接复用它的** HIP 分支」会让读者以为另一条 HIP 分支不原地。
- 建议：写成「HIP 两条分支（AITER Triton 与回落 `forward_cuda`）均原地」。

**N6. [20 §5.3 RoPE 图] 边标 `pairOffset = 4 / 2` 语义含混**
- 该式意为 `(rot_dim/2)/numElemsPerThread = 4/2 = 2`，但字面也可读成「4 或 2」。建议改为 `pairOffset = (rot_dim/2)/2 = 2`，与节点里的「shuffle XOR 2」一致。

**N7. [20 §5.3、§9.1 两张新图] 未用本域/house 的图配色与 neutral 类**
- 两图只定义 `compute fill:#e8f1fb,stroke:#416b91` 与 `cost fill:#fff1dd,stroke:#ab702c`，其余节点（N/NX/GJ/O、M/S/P/Z/A/B/O）**未指定 class**，会落到 Mermaid 默认淡紫底，与 `drawing-wiki-figures` §3「多数节点用 `.neutral`（白底灰框），每图最多两个强调色」不符；同页图 2 与全域主流是 `neutral #ffffff/#64748b` + `acc1 #dbeafe/#2563eb` + `acc2 #ffedd5/#ea580c`（本域统计：42/39/32 次），`tools/figs/figstyle.css` 的 token 为 `--acc1:#2563EB`、`--acc2:#C3651F`。
- 建议：两图补 `classDef neutral` 并把非强调节点归入；强调色改用与图 2 相同的 `acc1/acc2` 取值。

**N8. [20 §4.3 eager walk 的前提列表] 「已启用 CustomOp」不是该 walk 的必要条件**
- `forward_cuda` 在非 batch-invariant 下直接 `return self.forward_native(...)`，两条都落到同一个 `ir.ops.fused_add_rms_norm.maybe_inplace`；决定「eager 派发还是图节点」的只有 `ir_enable_torch_wrap`。把「已启用 CustomOp」列为前提无害但会让读者误以为 disabled 时不走 IR dispatch。
- 建议：前提收敛为「torch-wrap 关闭（mode NONE 或 backend eager）+ 非 batch-invariant + 带 residual + priority 含 `vllm_c`」。

**N9. [20 页头「最近更新：2026-09-15」]** 本轮改动在 2026-09-15 之后落地（复审日 2026-09-16），若改动实际发生在今日应同步日期；21 同页头写 2026-09-15 且正文确未动，与「统一冻结源码基线日期；正文机制与既有边界保持不变」的自述一致 ✔。

---

## 5. 跨页复核

1. **20 ↔ 21，同一次 add+RMSNorm 的 provider 选择**：现在两页叙事合并——20 §1.4/§4.3/§10 说默认编译下 dispatch 发生在 21 的 lowering（fake 实参、默认 native），21 §1/§8.1 的链路不变。`IrOp.__call__` 按 `_ENABLE_TORCH_WRAP` 二分（`_inner_call` vs `torch_op`）支持这一分界 ✔。
2. **保护 clone / donation**：20 §4.3 新增「选中 inplace provider 时 `IrOpImpl.func_impl_fn` 先 clone activation 输入；只有满足 donation 等条件才可能消除」，与 `IrOpImpl.func_impl_fn`（`for i in self.op.activation_indices: new_args[i] = args[i].clone()`）及 `vllm/kernels/vllm_c.py` 把 `fused_add_rms_norm` 注册为 `inplace=True` 一致，也与 21 §4.3/§7 的 clone-elimination 叙述衔接 ✔。完成点分层（eager 原地 / ROCm native-copy 有 copy / 编译期 functional 包装有 clone）与 `vllm/kernels/vllm_c.py` 三条分支一致 ✔。
3. **20/21 ↔ 19 编译与 range 归属**：19 行 560 的 6 项移交、行 227–229 的端点消费与 20/21 现表述一致 ✔。唯一残留是 21 的「30 个归 19」计数口径（P2-13）。
4. **20 ↔ 17 量化 kernel ABI 与 Marlin workspace**：20 新句把「weight reload 复用 workspace、`g_idx_sort_indices` 地址、device/dtype/numel 不符报错」独占交给 17 §6.2；17 本轮正好把该节扩写为三方划分并回写「20 拥有它在 Kernel 内的用途，19 拥有通用捕获约束」，wikilink 锚 `#6.2 reload 与 CUDA Graph：本页负责哪一半` 与 17 的实际标题逐字相符（`check_links --strict` 0 broken/0 stale_section 复核通过）✔ 双向闭合。
5. **21 → 20 的三处委托现在都落到实体**：容差执行侧（21 行 26、619）→ 20 §3.5 第 4 段；OOT 注册与 uuid（21 行 527）→ 20 §3.5 第 2、3 段；新增 provider 用 `register_impl`（21 行 760）→ 20 §3.5 第 1 段 ✔。F7 关闭。

---

## 6. 自跑质量门（T0，只读）

```
.venv/bin/python tools/check_links.py --strict     → pages=452 broken=0 ambiguous=0 bare_index=0 stale_section=0 orphans=0
.venv/bin/python tools/check_math.py --changed --strict     → 39 files: 0 errors, 0 warnings
.venv/bin/python tools/check_markdown.py --changed --strict → 39 files: 0 errors, 0 warnings
.venv/bin/python tools/check_assets.py --changed --strict   → 39 files: 0 errors, 0 warnings
```

两张新 Mermaid 图未触发 `MD003`（无引号管道标签），也未引入 math/asset 问题。**未跑**「built-page proof」（`mkdocs_site build --changed` + `mathjax-corpus`）——本轮无新公式与新资源，仅两张 Mermaid；若要图渲染证据仍需那一档。

---

## 7. 小结

- **20**：上轮 P0 与 7 条 P1 中，6 条完全关闭（F2/F3/F4/F5/F7/F8）、F1 关闭、F9 关闭、F6 方向纠正但符号写错。新增的 §3.5、§7.4 oracle 表、norm+quant 四入口表、两张原理图全部可回源复演，质量明显高于上一轮。**剩余阻塞只有两条一句话级别的 P1（N1 的 lowering-uuid 表述、N2 的不存在符号）**，改完即可合。另有 5 条上轮 P2 未动（frozenset 措辞、`w2_scale` clamp、XPU 默认 priority、Helion「存在未接线」、⑦ 的 dtype/head_dim 门）与 6 条本轮新 P2。
- **21**：正文未动，声明「机制与边界不变」属实；被 20 承接后交接不再悬空。其 5 条 P2（两处文档冲突、ROCm 条件、`user_writes_to_node` 锚点、`dump_prefix` 注释、字段计数口径）仍待处理，均不阻塞。
