# Group D 复评（round 2）：09 模型库 / 10 Attention Backend / 17 量化

- 评审人：独立 reviewer（非作者）。**只读**：未改动任何 wiki / skills / docs / 源码 checkout。
- 冻结基线：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（checkout HEAD 已核对一致，仅未跟踪 `artifacts/`，未动 HEAD）。
- 方法：先读 `docs/research/2026-09-15-vllm-domain-review/group_D_09_10_17.md` 的上轮结论，再逐条对冻结源码核验**当前散文**（不以新稿自证）；新增文字全部重新走 hop-walk 与数值复算；跨页所有权两侧都打开（09/10/17/19/20/25）。
- 严重度：P0 = 会误导读者的事实错误；P1 = 实质缺口 / 跨页矛盾 / 所有权空洞；P2 = 次要。

## 0. 结论速览

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 09_vllm_model_library_analysis | pass | pass | pass | transform, layout | pass | 12/12 | **PASS** | 上轮 feature FAIL 已闭合：新增 §2.6.1 MoE 专家写入，全链符号与 8 专家/EP=2 复算均对得上 |
| 10_vllm_attention_backends_analysis | pass | pass | pass | transform, layout | pass | 6/6 | **PASS** | KV scale 消费所有权已落到本页，descale 形状与 cascade 分支核验一致 |
| 17_vllm_quantization_analysis | pass | pass（§8.3 仍有 1 条 lane 混流，P2） | pass | transform, layout, timing | pass | 16/16 | **CONDITIONAL PASS** | 两个 P0 与五个 P1 全部闭合；但 §6.1 末段与 09 新修的 contradiction 块**互相矛盾**（NEW-P1），须修后才算通过 |

feature 结论：09 `feature: pass §2.4/§2.6.1`；10 `feature: pass`；17 `feature: pass §4.2–§4.3`。

计数：09 = FIXED 4 / PARTIAL 0 / NOT_FIXED 1；10 = FIXED 2 / PARTIAL 0 / NOT_FIXED 1；17 = FIXED 9 / PARTIAL 3 / NOT_FIXED 1（另 NEW-P1 ×1、NEW-P2 ×3）。

T0 门（只读运行）：`check_links --strict` 452 页 0/0/0/0/0；`check_math --changed --strict`、`check_markdown --changed --strict`、`check_assets --changed --strict` 各 39 文件 **0 error 0 warning**。无回归。

---

## 1. 上轮每条发现的状态

### P0

| # | 上轮发现 | 状态 | 源码核验 |
|---|---|---|---|
| P0-1 | 17 §1.1 #1/#2 与 §8.3 config lane 调用父子关系错误 | **FIXED** | `vllm/config/model.py::ModelConfig.__post_init__`（def @524）在 865 行调 `self._verify_quantization()` ✓；`vllm/engine/arg_utils.py::EngineArgs.__post_init__`（def @777）在 812 行调 `resolve_quantization_config(self.quantization, self.quantization_config)` ✓。§8.3 已拆成 `EngineArgs.__post_init__` / `ModelConfig.__post_init__` / `VllmConfig.__post_init__` **三个独立根**，并在树前注明 args 经 `create_model_config` 传入 ModelConfig（`arg_utils.py:1798 quantization_config=self.quantization_config` ✓）。图 1 的 RC 节点标签、§4.1 首段、§8.1 所有权表同步更正 ✓。`maybe_compose_online_quantization` 挂 overlay 的位置（`weight_utils.py:255`，只在 checkpoint 主配置分支且 `online_args` 非 None）标注准确——纯在线 lane 在 349–401 行直接返回 `OnlineQuantizationConfig` 作主配置，页面的"组合分支"限定没有讲反 ✓ |
| P0-2 | 17 §4.3 缺 q_scale 时最终值写错 | **FIXED** | `kv_cache.py::BaseKVCacheMethod.process_weights_after_loading`：条件块内 146–153 行 warning + `layer._q_scale.copy_(k_scale)` + `_q_scale_float = k_scale`，块不 `return`；169–196 行无条件尾巴读**未被修改的** `layer.q_scale`（仍是 −1.0 哨兵）→ `else: q_scale = 1.0` → 再次 `copy_` 并覆盖 host float。新增 `[!contradiction]` 块写明"先复制 k_scale，再由无条件尾巴覆盖为 1.0"，并给出非 fnuz、k/v=0.3 时 `_q_scale` 轨迹 **1.0 → 0.3 → 1.0**。我按源码重放：一致 ✓ |

### P1

| # | 上轮发现 | 状态 | 源码核验 |
|---|---|---|---|
| P1-1 | 09 MoE 专家权重写入未解释也未点名 owner（feature FAIL） | **FIXED** | 新增 §2.6.1 + §2.4 兄弟轴一行 + §1.6 流程行 + §3.1 所有权行（`RoutedExperts`/`ExpertMapManager`）+ §3.2 调用树三行 + 阅读路线 #6。逐符号核验：`routed_experts.py::RoutedExperts.load_weights`（896 行 `get_expert_mapping(include_fused=True)`）、`get_expert_mapping`（980）、`make_expert_params_mapping`（1002，docstring 明写 "Legacy entry point for models that still hand-roll load_weights"）、`build_expert_params_mapping`（1034，返回 `(param_name, weight_name, physical expert_id, shard_id)`，用 `EplbState.build_initial_global_physical_to_logical_map` 决定 checkpoint 名）、`weight_loader`（615）、`_map_global_expert_id_to_local_expert_id`（296）→ `expert_map_manager.py::ExpertMapManager.map_global_to_local`（336）。`w1=gate_proj / w2=down_proj / w3=up_proj`、`w13_*` vs `w2_*` 归属见 1130–1145 ✓。非本地 `expert_id == -1 and not use_global_sf → return False`（645–647）✓；`use_global_sf` 只在 `"input_scale" in weight_name` 时成立（640–643）✓ |
| P1-2 | packed 参数写入侧切片与 v1/v2 loader 选择两页互推 | **FIXED** | 17 新增 §5.1.1 独占该规则，09 §2.4 改为"packed/scale 坐标规则归量化页 §5.1.1"，17 §1/§8.1 两处措辞一致，**无双重声明**。逐项核验见 §2 表 |
| P1-3 | KV scale 在 attention kernel 中的消费无人负责 | **FIXED** | 10 §2.10 阶段表写入/读取两行补入 scale，并新增"KV scale 的实际消费边界"整段 + §4 改写 + 调用树两行；17 页头/§4.3/§8.1 反向指向 10。源码：`flash_attn.py::FlashAttentionImpl.do_kv_cache_update`（1244–1253）把 `self.kv_cache_dtype, layer._k_scale, layer._v_scale` 传给 `reshape_and_cache_flash` ✓；非 cascade 路径 1046–1054 `descale_shape = (cu_seqlens_q.shape[0]-1, self.num_kv_heads)`，`q_descale` 仅在 `self.supports_quant_query_input` 为真时给出 ✓；cascade 路径 1212–1214 直传 `layer._q/_k/_v_scale` ✓ |
| P1-4 | Marlin `workspace`/`g_idx_sort_indices` reload 地址合同三页不一致 | **FIXED** | 四页现在一致：17 §6.2 独占（页头也写入适用范围）；25 L226 "…归量化执行 §6.2…本页只拥有 pause window 与版本可见性"；19 L268 明确区分 `WorkspaceManager` 与 Marlin `self.workspace`/`g_idx_sort_indices` 并指向 17 §6.2；20 §8.2 "本页只拥有这块缓冲在 Kernel 内的用途…由量化执行 §6.2 独占说明"。源码：`marlin_utils.py::marlin_make_workspace_new`（408–433，`sms * max_blocks_per_sm` 个 `torch.int`，`existing` 不相容抛 `ValueError` 并含 "Reload must reuse the workspace storage captured by CUDA graphs"）✓；`utils.py::replace_parameter` docstring 明写 `prefer_copy` 仅"相容时"就地 copy、保留旧属性、`weight_loader` 总取旧参数 ✓。四个测试名均存在（`test_reload.py:347/459/479/718`）✓ |
| P1-5 | KV-cache method 变体集合无枚举依据、CT 子类语义不同（feature FAIL） | **FIXED** | §4.2 新表以"各 config 的 `get_quant_method` 对 attention 层的选择"为枚举依据，列出基线里 `BaseKVCacheMethod` 的**四个**直接子类；`grep 'BaseKVCacheMethod)' vllm/` 恰好四条（`fp8.py:893`、`modelopt.py:144`、`quark/quark.py:1022`、`compressed_tensors.py:1031`）✓。选择点逐条核对：`Fp8Config` @fp8.py:224 ✓；`ModelOptQuantConfigBase.get_quant_method` @211–212 经 `KVCacheMethodCls`（默认槽位 `BaseKVCacheMethod` @160，三个具体 config 在 702/1093/1463 覆写）✓；`ModelOptMixedPrecisionConfig` @1707–1710 另要求 `kv_cache_quant_method` 非空 ✓；`QuarkConfig.get_quant_method_target` @278–279 返回 `QuarkKVCacheMethod`，exclude 命中且非 LinearBase 时返回 `None, None, None` ✓。§4.3 的 CT 覆写描述逐句对得上 1083–1221：create 建全 1 的 q/k/v scale 与全 0 的三个 zero-point、不建 `prob_scale`、`n_scales = num_kv_heads if attn_head else 1`、`validate_kv_cache_scheme` 只接受 8-bit float + TENSOR/ATTN_HEAD + symmetric；`_tp_aware_loader` 对 q 做 `torch.amax(view(-1, heads//kv_heads), dim=1)` 后按 TP 切分或复制；post-load 直接 `layer._k_scale = layer.k_scale`、`_*_float` 多元素取 `max().item()`、`_k/_v_scale_cpu.fill_(标量)`、最后删三 scale + 三 zero-point；无 pre-processed / 已消费早返回 ✓ |

### P2

| # | 上轮 P2 | 状态 | 说明 |
|---|---|---|---|
| 1 | 09 §4.1 `customize_spec` 只说 MRV2 | **FIXED** | 改为"MRV1 与 MRV2 都"；源码两处：`gpu_model_runner.py:7662`（MRV1）与 `gpu/attn_utils.py:64`（MRV2）✓ |
| 2 | 09/17 tracking 豁免"可能" vs "必然"、`UnquantizedLinearMethod` 是否继承 no-op | **09 FIXED / 17 NOT_FIXED** | 09 §2.8 contradiction 块已改成"所有挂有 `quant_method` 的模块都会被**必然豁免**…`UnquantizedLinearMethod` 还自行覆写了该 hook"，与源码一致；**17 §6.1 末段未同步**，见 NEW-P1 |
| 3 | 09 §2.8 dtype / device scope | **FIXED** | 现写"dtype scope 覆盖构造→post-load，target-device context 只包 `initialize_model`，离开 dtype scope 后才 `model.eval()`"；`base_loader.py:53–83` 完全一致 ✓；§3.2 调用树两处标注同步 ✓ |
| 4 | 09↔10 `bind_kv_cache` 双重声明 | **FIXED** | 09 §4.1 身份链止于 `get_layers_from_vllm_config`，`bind_kv_cache` 明确归 10 ✓ |
| 5 | 10 §2.2 FP32→Flex 与候选顺序漏 FLASHINFER | **FIXED（两条）** | 现写 SM10+causal 依次 FlashInfer/FlashAttention/Triton/Flex/TurboQuant，其余依次 FlashAttention/FlashInfer/Triton/Flex/TurboQuant——与 `platforms/cuda.py:157–172` 逐项一致 ✓；FP32 段改为"FA 与 FlashInfer 拒绝后 head size ≥32 时 Triton 仍支持 FP32；测试用 head size 16 才落 Flex；本文 head size 64 只改 FP32 会选 Triton"——`triton_attn.py:289–293 supported_dtypes` 含 `float32`、`347–348 supports_head_size: head_size >= 32`、`tests/kernels/attention/test_attention_selector.py:249 get_attn_backend(16, torch.float32, None)` ✓ |
| 6 | 17 §4.3 fnuz ×2 自相矛盾 / "恰好一个 > 0" | **FIXED** | 现写"只有从 checkpoint 取值的两条分支在 fnuz 平台再 ×2；缺失两者时的常量 1.0 不加倍"（`kv_cache.py:121–122 / 137–139`，第 2 态不乘 ✓），并写明第三态"要求 `k_scale > 0`（有 assert），取 k/v 的最大值"（133–136 ✓） |
| 7 | 17 §8.3 `create_weights` 父节点 / `initialize_online_processing` lane | **PARTIAL** | `quant_method.create_weights` 已改挂 `ColumnParallelLinear / RowParallelLinear 构造`（源码 `linear.py:512`/`1691` ✓）；但 `initialize_online_processing(layer)` **仍画作 AutoGPTQ `register_parameter(qweight/scales/qzeros/g_idx)` 的子节点**，见 NEW-P2-a |
| 8 | 17 §3.3 把 already-called flag 写进 `_setup_kernel` | **NOT_FIXED** | §3.3 完成点仍把 `layer._already_called_process_weights_after_loading = True` 排在 `_setup_kernel` 的步骤序列里；源码 `online/fp8.py::_Fp8OnlineMoEBase._setup_kernel`（477–525）不设该 flag，由三个子类在 `_setup_kernel` 返回**之后**设（606 / 704 / 788） |
| 9 | 17 §1.1 #10 与 §8.3 末行 batch-invariant 归属过宽 | **PARTIAL** | §8.2 契约表已限定为"在线 FP8 的 `apply`（Cutlass 除外）"；但 §1.1 #10 "`VLLM_BATCH_INVARIANT` 下走 BF16 dequant + `F.linear`" 与 §8.3 末行 `kernel.apply_weights [VLLM_BATCH_INVARIANT 下改走 BF16 dequant + F.linear]` 仍挂在通用节点上。源码该退路只在 `online/fp8.py:235 / 415`；`UnquantizedLinearMethod.apply` 走 `linear_batch_invariant`（`linear.py:234–237`），`auto_gptq.py` 无此分支 |
| 10 | 术语与交叉引用 | **PARTIAL** | 17 §8.1 里"09 的所有权表把量化参数布局指回本页"这句已删除 ✓；但命名仍四套：17 标题"量化执行"、09/10 链接文字"量化派发"（4 处）、`index.md:48` 作"17 量化"、17 新增链接作"量化页" |
| 11 | 09 EP 过滤 / row-parallel 复算未入图 | **NOT_FIXED** | 仍只有文字复算（上轮即标为影响不大） |
| 12 | 10 §2.8 B12X 两平面未用 A19 复算 | **NOT_FIXED** | §2.8 本轮未改（10 的 diff 仅 19 行，集中在 §2.2/§2.10/§4/调用树/阅读路线） |
| 13 | 17 §4.1 把"挂 overlay"写成 config 解析产出 | **FIXED** | §4.1 首段与图 1 RC 节点都写明 overlay 由 `get_quant_config` 的 `maybe_compose_online_quantization` 挂上 ✓ |
| 14 | 17 §6 ↔ 25 §3.4 layerwise reload 重复 | **FIXED** | 25 §3.4 只留一句风险取向的概述 + 指向 17 §6.2 的链接，机制细节不再重复；17 §6.2 唯一展开 ✓。**未出现新的双讲** |

### 未核验怀疑（沿用上轮，本轮仍未核验）

- 09 §2.10 PP>1 时 sharded state 文件名只含 TP rank 的冲突风险（页面自标"未验证风险"）。
- 10 §2.9 MRV1 `_check_and_update_cudagraph_mode` 取每组最弱能力。

---

## 2. 新增文字的重新核验（hop-walk + 复算）

### 2.1 17 §5.1.1 packed 写入（新增，逐条核对）

| 页面主张 | 源码 | 结果 |
|---|---|---|
| Column/Row 构造按 `quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED` 选 `weight_loader_v2` 或 `weight_loader` | `linear.py:519–522`（`ColumnParallelLinear`，class @427）、`1698–1701`（`RowParallelLinear`，class @1619）；全文仅这两处，`ReplicatedLinear` 不参与 | ✓（页面正确限定为 Column/Row） |
| 表内 11 个类名 + `ModelOptLinearMethod` 经装饰器追加 | `linear.py:49–63` 逐名逐序一致；`register_weight_loader_v2_supported_method` @66–69；`modelopt.py:2352` 装饰 `class ModelOptLinearMethod` | ✓ |
| `adjust_block_scale_shard` 用 `weight_block_size[0]` 分别向上整除 offset 与 size，`weight_block_size` 必须非 None | `linear.py:84–93`：`assert weight_block_size is not None`；`block_n = weight_block_size[0]`；两个 `(x + block_n - 1) // block_n` | ✓ |
| 仅 `packed_dim == output_dim` 才 `round(value // packed_factor)` | `parameter.py:160–168`（`load_merged_column_weight`）、`225–231`（`load_qkv_weight`）；`_adjust_shard_indexes_for_packing` @606–618 | ✓ |
| `adjust_marlin_shard` 乘 tile；v2 侧 `_adjust_shard_indexes_for_marlin` 同乘；无 tile 属性则不变 | `linear.py:72–81`；`parameter.py:602–603`；`getattr(param, "marlin_tile_size", None)` | ✓ |
| `RowvLLMParameter` 直接取本地 `data.shape[input_dim]`，源从 `tp_rank * size` 起 | `parameter.py:219–229` | ✓ |
| `marlin_repeat_scales_on_all_ranks` 为真时 AutoGPTQ 建**无 `input_dim`** 的 `ChannelQuantScaleParameter` / `PackedColumnParameter`，其 row loader 继承 Base 的整张 shape 校验 + copy | `auto_gptq.py:421–428`；`parameter.py:93–109 _assert_and_load` | ✓ |
| QKV：Q 源 `tp_rank * shard_size`，K/V 源 `(tp_rank // num_kv_head_replicas) * shard_size` | `parameter.py:236`（`shard_id_int = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads`）+ `linear.py:1184`（`num_heads=self.num_kv_head_replicas`） | ✓ |
| 融合 checkpoint `loaded_shard_id=None` 时先按全局 Q/K/V 宽度切三段（同样先换 block/packed 坐标），再回 named-shard；per-tensor scale 特例填满三槽 | `linear.py:1098–1145`（QKV 版）、`860–898`（Merged 版）、`1154–1162`（`PerTensorScaleParameter` 循环） | ✓ |

**`[64,512]` QKV 复算（我自己重算，与页面逐格一致）**：hidden 512、Q head 8、KV head 4、head size 64、TP=2、rank 1、4 bit、group 128、无 act-order。
- 全局输出宽度 Q/K/V = 512/256/256；本地 256/128/128，合计 512 ✓。K 不切（column parallel），`input_size_per_partition = 512`。
- `pack_factor = 8` → 本地 `qweight = [512/8, 512] = [64,512]` ✓。`is_row_parallel=False`、`marlin_repeat_scales_on_all_ranks(False,128,False)=False` → `scales_and_zp_input_dim=0`、`scales_and_zp_size = 512//128 = 4` → `scales=[4,512]`、`qzeros=[4, 512/8]=[4,64]` ✓。
- qweight 本地段 0/256/384，`packed_dim=0 ≠ output_dim=1` 故**不除 pack**；Q 源 `[64,512]` 取 `[:,256:512]`（`tp_rank*256`）、K/V 源 `[64,256]` 取 `[:,128:256]`（`num_kv_head_replicas = 1`，因 `tp_size 2 < total_num_kv_heads 4`，`shard_id_int = 1//1 = 1`）✓。
- qzeros `packed_dim = output_dim = 1`：K 逻辑 offset 256/size 128 → `/8` → 32/16，本地段 0:32 / 32:48 / 48:64；Q 源 `[4,64]` 取 `[:,32:64]`、K/V 源 `[4,32]` 取 `[:,16:32]` ✓。
- scales 无 packed 轴，段与 qweight 同 ✓。
- 磁盘已融合：qweight 全局 `[64,1024]` 切 0:512 / 512:768 / 768:1024；qzeros 全局 `[4,128]` 切 0:64 / 64:96 / 96:128（`/8` 后）✓。页面强调"不是对同一 offset 连除两次"，与 `_load_fused_module_from_checkpoint` 先换全局坐标、`load_qkv_weight` 再换本地坐标的两段语义一致 ✓。

### 2.2 09 §2.6.1 MoE 复算（我自己重算）

- 8 专家、EP=2、`linear` placement：`determine_expert_map`（`expert_map_manager.py:67–79`）`base_experts=4, remainder=0`，rank 1 `start_idx = 4` → 全局 {4..7} → 本地 {0..3}，全局 5 → **本地 1** ✓（页面 `w13_weight[1, …]` 正确）。
- `w13` 每专家本地形状 `[2·I_r, H]`；`_load_w13`（`routed_experts.py:478–527`）`shard_size = expert_data.shape[0]//2 = I_r`，w1 → `narrow(0, 0, I_r)`、w3 → `narrow(0, I_r, I_r)` ✓；loaded 侧按 `loaded_per_rank * tp_rank` 取 TP 段 ✓（页面"若 TP 还切 intermediate，则先取当前 TP 的 I_r 行"正确）。
- `w2` `shard_dim = 1`（`SHARD_ID_TO_SHARDED_DIM` @674），`_load_w2`（529–562）不再对 `expert_data` 分段，只对 loaded 侧按 `I_r` 取 TP 段 → 目标就是整块本地槽 `w2_weight[1, :, 0:I_r]` ✓。
- 全局专家 2 在 rank 1 上 `expert_map[2] = -1` → `return False`（645–647）✓。

### 2.3 10 §2.10 KV scale 消费（新增段）

逐句核验见上表 P1-3。另核：页面"基类的非量化-cache 路径会把 K/V scale 收敛到 1.0"——`set_default_quant_scales(register_buffer=True)` 初值即 1.0（`attention.py:130–133`），`is_quantized_kv_cache` 为假时条件块整段跳过，尾巴不动 k/v ✓；"per-token-head 模式由对应 kernel 动态生成 scale" 与 `kv_cache.py:99–108` 强制 1.0 + `triton_attn.py:276–281 customize_spec`（每 head 内联 fp32 scale）一致 ✓；"CT 可把 per-head Parameter 直接接到 `_k/_v/_q_scale`" 与 `_k_scale.expand((num_seqs, num_kv_heads))` 对长度 `num_kv_heads` 的向量成立 ✓。

---

## 3. 新发现（回归 / 新错）

### NEW-P1｜17 §6.1 末段 ↔ 09 §2.8 contradiction 块｜同一事实两页写反（跨页矛盾）

- 17 §6.1 末段（未改动）：「`DefaultModelLoader.track_weights_loading` 默认只对具备 loaded-name tracking 的非量化模型开启，且其 `has_postprocess_quant` 判断**连继承 no-op hook 的普通 linear 参数都可能豁免**。」
- 09 §2.8（本轮新改）：「后一个方法已定义在 `QuantizeMethodBase`，所以**所有挂有 `quant_method` 的模块都会被必然豁免**…；`UnquantizedLinearMethod` 还**自行覆写了该 hook，而不是继承一个无行为特例**。」
- 源码：`base_config.py:77–82` `QuantizeMethodBase.process_weights_after_loading` 在基类即有定义 → `default_loader.py:456–458` 的 `getattr(quant_method, "process_weights_after_loading", None)` 对任何 quant method 恒为真 → 豁免是**无条件**的（`default_loader.py:461–464`，且 `module.named_parameters()` 递归，连子模块一起豁免）；`linear.py:205` `UnquantizedLinearMethod.process_weights_after_loading` **是覆写**。
- 影响：09 与 17 对同一个判定给出相反强度与相反继承关系，读者无法判断 tracking 的实际保证；17 的说法还低估了豁免范围（"可能"读成偶发）。
- 修复：把 17 §6.1 末段改成与 09 同口径（"必然豁免；`UnquantizedLinearMethod` 自行覆写该 hook"），或整句压成一行并链接 09 §2.8，由 09 单点拥有该结论。

### NEW-P2-a｜17 §8.3 调用树｜`initialize_online_processing` 仍与 AutoGPTQ lane 混流（hop-walk 断一跳）

- 页面：`register_parameter(qweight/scales/qzeros/g_idx)` 之下挂 `` `-- initialize_online_processing(layer) [条件：uses_meta_device] ``。
- 源码：`online/fp8.py::OnlineLinearBase.create_weights`（125–156）注册的是 **`weight`**，`initialize_online_processing(layer)` 是它的**同级末尾语句**（156 行），不是 `register_parameter` 的子调用；`auto_gptq.py::AutoGPTQLinearMethod.create_weights`（326–453）**从不**调用它。MoE 侧同理在 `online/moe_base.py:97`。
- 影响：按树走进 AutoGPTQ 的 `register_parameter` 找不到该节点，两条 lane（预量化 / 在线 FP8）被并成一条——这正是上轮 P2-7 的后半条。
- 修复：在 `quant_method.create_weights` 下另开一条 `[在线 lane] OnlineLinearBase.create_weights` 子树，`register_parameter(weight)` 与 `initialize_online_processing(layer)` 作其同级子节点。

### NEW-P2-b｜17 §3.3 末段｜链接锚点与显示标签不一致，落点是错的小节

- 页面：`[[…/09_vllm_model_library_analysis#2.6 融合前分别分片，融合后仍能拆回各投影|模型库 §2.6.1]]`。
- 事实：09 的 MoE 内容在 `#### 2.6.1 MoE 专家写入：名字中的全局专家先映射成本地槽`（L217），而锚点指向 `### 2.6 融合前分别分片，融合后仍能拆回各投影`（L182，dense QKV）。锚点能解析，所以 `check_links --strict` 过（0 stale_section），但读者点进去落在 dense 小节。
- 修复：锚点改成 `#2.6.1 MoE 专家写入：名字中的全局专家先映射成本地槽`。

### NEW-P2-c｜17 §4.3「建立」段｜两处描述不完整

1. 列出 `set_default_quant_scales` 的 host 镜像时漏了 `_prob_scale_float`（`attention.py:148`）；页面枚举了 5 个属性而源码是 6 个，且措辞是封闭枚举。
2. 该段未提 `_init_kv_cache_quant` 在 `should_load_quant_weights` 为真后还有一道 `fp8_e5m2` 门（`attention.py:198–214`：非 CT、或 CT 但 `kv_cache_scheme` 非空时抛 `ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")`）。这是"建立"阶段唯一的硬失败，属本节的失败边界。

### NEW-P2-d｜17 §4.3｜"实例属性赋值遮蔽类级 loader"用错了对象

- 页面：per-head scale「走 compressed-tensors 的 `_tp_aware_loader`（实例属性赋值遮蔽类级 loader）」。
- 事实：该括注是 `KVCacheScaleParameter` 自己的 docstring（`kv_cache.py:19–28`）在描述**基类哨兵参数**；CT 的 `create_weights` 建的是普通 `torch.nn.Parameter`（`compressed_tensors.py:1099–1121`），其上本无类级 `weight_loader` 可遮蔽，`layer.q_scale.weight_loader = partial(...)`（1171–1189）只是普通实例赋值。读者可能误以为 CT 也用 `KVCacheScaleParameter`。
- 修复：把括注移回描述基类哨兵的那句，或改成"CT 用普通 Parameter 并直接挂实例 loader"。

### NEW-P2-e（很轻）｜17 §1.1 #2｜纯在线 lane 未出现在"产出"列

`get_quant_config` 还有三条**不经 overlay** 的返回：`weight_utils.py:354–356`、`386–389`、`398–401` 直接把 `OnlineQuantizationConfig` 当**主配置**返回。#2 行只写了"包成 `OnlineQuantizationConfig`，挂到主配置"（overlay 语义）。§4.1 的逐层表间接覆盖了行为，但"无 checkpoint 量化时在线 config 自己就是主配置"这一形态在清单里看不出来。建议 #2 的产出列补半句。

---

## 4. 跨页所有权复核（两侧都打开）

| 关切 | 上轮问题 | 现状 | 判定 |
|---|---|---|---|
| MoE checkpoint → 本地专家槽 | 空洞 | 09 §2.6.1 独占；17 §3.3 末段与 §1.1 #11 只保留"量化 ABI 创建与 post-load"并链回 09（锚点见 NEW-P2-b）；18 未插手 | 一致，无重复 |
| 量化 packed 参数写入坐标 + v1/v2 loader | 互推 | 17 §5.1.1 独占；09 §2.4 一句话链接；17 §1/§8.1 三处措辞统一为"通用写入入口归 09、量化坐标规则归本页" | 一致，无重复 |
| KV scale 在 kernel 中的消费 | 空洞 | 10 §2.10 + §4 独占消费侧；17 页头/§4.2/§4.3/§8.1 四处都指向 10 的具体小节；10 反向指向 17 §4.3 的具体锚点 | 一致，无重复 |
| Marlin workspace / sort-index reload 地址合同 | 三页不一致 | 17 §6.2 独占；19 L268、20 §8.2、25 §3.4 各一句链接并明确各自只拥有什么（通用 capture 前提 / kernel 内用途 / pause window） | 一致，无重复 |
| `bind_kv_cache` | 双重声明 | 归 10；09 §4.1 身份链止于 `get_layers_from_vllm_config` | 一致 |
| layerwise reload 机制 | 两页各讲一遍 | 17 §6.1/§6.2 独占；25 §3.4 一句概述 + 链接 | 一致 |
| tracking 豁免强度 | — | **09 与 17 相反** | **NEW-P1** |
| 页面命名 | 不统一 | 仍四套（量化执行 / 量化派发 / 17 量化 / 量化页） | 上轮 P2-10，PARTIAL |

---

## 5. 判定行

```
09_vllm_model_library_analysis : beat2 pass | hop-walk pass | delete-code pass | figure-trigger transform,layout | algorithm-replay pass | spot-check 12/12 | feature: pass §2.4/§2.6.1 | VERDICT PASS
10_vllm_attention_backends_analysis : beat2 pass | hop-walk pass | delete-code pass | figure-trigger transform,layout | algorithm-replay pass | spot-check 6/6 | feature: pass | VERDICT PASS
17_vllm_quantization_analysis : beat2 pass | hop-walk pass(P2 §8.3 lane) | delete-code pass | figure-trigger transform,layout,timing | algorithm-replay pass | spot-check 16/16 | feature: pass §4.2–§4.3 | VERDICT CONDITIONAL PASS — blocker NEW-P1 (§6.1 与 09 §2.8 矛盾)
```

剩余阻塞项：**仅 NEW-P1 一条**（17 §6.1 末段）。其余为 P2：NEW-P2-a…e、上轮 P2-7/8/9/10 的未闭合部分、P2-11/12（图示补强，可选）。
