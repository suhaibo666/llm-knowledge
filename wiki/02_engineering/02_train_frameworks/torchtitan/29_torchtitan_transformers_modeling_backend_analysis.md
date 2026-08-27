---
title: "Transformers modeling backend：把 HF 模型接到 TorchTitan 协议边界"
---

# Transformers modeling backend：把 HF 模型接到 TorchTitan 协议边界

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **状态**：experimental · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：当前 Transformers modeling backend 不是在 TorchTitan 里维护另一套 HF fork，也不是把旧的 DTensor hook 方案继续包一层。它保留 HF 的模型类与 forward，把变化集中到四个协议边界：`PretrainedConfig ↔ BaseModel.Config`、普通 `nn.Module ↔ Module`、HF 模块角色 ↔ `ShardingConfig`、HF state-dict ↔ TorchTitan checkpoint。这样 dense HF 模型能复用 core Trainer、SPMD types、PP/AC/compile/FSDP；MoE 则在并行化前把异构 HF MoE 子树替换成 TorchTitan 的统一 MoE。代价是“通用”仍依赖可识别的模块结构、动态类替换和显式兼容矩阵，未知结构必须早失败。
>
> **证据冲突提示**：目录 README 的“Further work”仍写着 “Load HF weights”，但当前 SFT full config 已设置 `initial_load_in_hf=True`，`ModelSpec` 也已接入 state-dict adapter。本页以可执行路径为准，并把 README 该条标为过时，而不是把文档愿望当成 HEAD 事实（`torchtitan/experiments/transformers_modeling_backend/README.md:78-82`、`torchtitan/experiments/transformers_modeling_backend/config_registry.py:168-220`、`torchtitan/experiments/transformers_modeling_backend/__init__.py:163-172`）。

---

## 1. Overview：保留 HF 语义，替换并行协议的接缝

### ① 背景/问题

HF Transformers 提供了大量持续演进的模型实现，TorchTitan 提供的是训练控制面、并行 mesh、Module/SPMD 协议、checkpoint 与性能组件。复制模型代码会形成长期 fork；只把 HF model 塞进 Trainer 又不够，因为普通 HF 子模块没有 TorchTitan 的状态布局声明，HF 配置与 checkpoint FQN 也不符合 TorchTitan 协议。

当前目录把目标限定为 HF causal-LM backend，并在 README 声明 dense 路径支持 FSDP/CP/TP/PP/compile、要求 Transformers 5.9.0 与 `spmd_types`（`torchtitan/experiments/transformers_modeling_backend/README.md:1-9`、`torchtitan/experiments/transformers_modeling_backend/README.md:23-39`）。这不是所有 HF task/model 的普遍承诺。

### ② 为什么选择协议适配，而不是 fork 或 hook 堆叠

选中路线是：**HF forward 与对象树尽量保留，TorchTitan 只接管配置、状态布局、输入预处理、MoE 统一实现和训练生命周期**。明显替代方案有两种：

1. 把每个 HF 模型移植成原生 TorchTitan `Module`；兼容最明确，但每次上游模型变化都要维护副本。
2. 保留普通 HF module，只在 forward 前后堆 DTensor/EP hook；接入快，却把参数布局、collective 顺序和 FSDP hook ordering 分散在运行时，难以做完整性检查。

当前实现选择统一协议的判据是“布局必须在构造/parallelize 阶段可审计”。`ModelSpec` 只注册一个 backend model、parallelize、pipeline、optimizer hook 和 state adapter；随后整个 HF 子树会被赋予 `Module` 能力与声明式 `ShardingConfig`（`torchtitan/experiments/transformers_modeling_backend/__init__.py:163-172`、`torchtitan/experiments/transformers_modeling_backend/module_conversion.py:7-15`）。提交 `4022b15e9afd14df54ccd7b9db8a30721b9f421f` 把 backend 迁入 `spmd_types`；提交 `601cf4d2304e4cfbb05e7b2865bb116a61d81a94` 随后删除 `full_dtensor` backend，说明当前设计权威已从“多套 runtime hook”收敛到 Module/SPMD types 协议。

### ③ 当前状态与调用链

```text
Python recipe / hf_model id
  -> HFTransformerModel.Config.update_from_config()
  -> AutoConfig + 显式 flavor override
  -> HFTransformerModel(meta) 保留 HF forward
  -> 可选 HF MoE -> Titan MoE
  -> 普通 HF nn.Module 动态转成 Module
  -> 为每个可识别角色安装 ShardingConfig
  -> model.parallelize(parallel_dims)
  -> AC -> compile -> FSDP
  -> core Trainer / PP schedule / checkpoint
```

入口的 `TransformersBackendConfig` 只在 core Trainer config 上增加 `hf_model`；专用 Trainer 也只增加 CP load-balancer guard，其余生命周期继承 core `Trainer`（`torchtitan/experiments/transformers_modeling_backend/configs.py:14-17`、`torchtitan/experiments/transformers_modeling_backend/trainer.py:12-30`）。因此它是模型 backend，不是第二套训练循环。

### ④ 约束/失败边界

- backend 在并行化入口硬性要求 `spmd_backend == "spmd_types"`，不提供静默 fallback（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:87-111`）。
- 模型类必须能从 `config.architectures[0]` 或 `model_type → Auto mapping` 找到，否则 build 直接 `ImportError`（`torchtitan/experiments/transformers_modeling_backend/model.py:654-687`）。
- `HFTransformerModel.verify_module_protocol()` 有意跳过 build 后的递归检查，因为此时内部仍源于 HF；真正的布局完整性检查发生在后续 sharding pass（`torchtitan/experiments/transformers_modeling_backend/model.py:1331-1338`、`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:491-524`）。不能把前者写成“所有 HF 模块已天然满足协议”。

### ⑤ 有锚点的趋势

提交 `e8e39abc6899d8557816aedd80d86aaf9a4496bc` 把原 Trainer 的 post-dataloader 特例下沉为 model-owned `preprocess_inputs()`；HEAD 的专用 Trainer 已只剩一个 guard（`torchtitan/experiments/transformers_modeling_backend/trainer.py:12-21`）。**推断**：backend 的演进方向是减少 Trainer fork、让模型/Module 协议拥有扩展点，而不是继续在训练循环中识别 HF 特例。

---

## 2. 配置桥：HF 架构事实优先，TorchTitan flavor 只覆盖显式字段

### ① 背景/问题

TorchTitan recipe 要用统一字段访问 `dim/n_layers/max_seq_len`，HF 模型又可能有 `hidden_size/num_hidden_layers/head_dim/qk_rope_head_dim` 等架构专属字段。若 TorchTitan debug flavor 的默认值无条件覆盖 `AutoConfig`，会在“权重成功读取”的表象下构造错误架构。

### ② 为什么采用“双源合并 + 显式覆盖”

选中路线是让 HF `AutoConfig` 成为 full model 的架构权威，只重放 flavor 中**非默认、明确设置**的 TorchTitan 字段。替代方案是始终从 TorchTitan 公共字段重新推导 `head_dim/intermediate_size`。提交 `ac240f926c8c70d188ef0e2f353991549d77187e` 记录了后者的实际失败：Qwen3 的 RoPE theta、head_dim 和 intermediate size 被错误默认值覆盖，加载正确权重后 loss 仍接近随机。决定标准不是字段名统一，而是不能改变 HF checkpoint 所对应的真实架构。

### ③ 实现思路与状态转换

1. `TitanModelConfig` 中会映射到 HF 的字段默认 `None`，TorchTitan-only 控制才保留实值；MoE config 在其上增加专家/dispatcher 选择（`torchtitan/experiments/transformers_modeling_backend/__init__.py:21-53`、`torchtitan/experiments/transformers_modeling_backend/__init__.py:56-105`）。
2. `HFTransformerModel.Config` 同时继承 `BaseModel.Config` 与 `PretrainedConfig`。由于自定义构造器与动态 HF 属性不适合 `dataclasses.replace()`，其 `build()`/`_replace()` 用 shallow copy 保住动态属性（`torchtitan/experiments/transformers_modeling_backend/model.py:192-260`、`torchtitan/experiments/transformers_modeling_backend/model.py:265-286`）。
3. 初始 flavor 只记录非默认字段到 `_titan_injected_model_args`；`update_from_config()` 加载 HF config、复制包含计算属性的 `vars()`，最后重放这些显式 override（`torchtitan/experiments/transformers_modeling_backend/model.py:288-330`、`torchtitan/experiments/transformers_modeling_backend/model.py:399-456`）。
4. 只有 `dim` 被显式覆盖时才重新推导 dense FFN/head_dim；MLA 的 `qk_rope_head_dim` 和 HF 已提供的 head_dim 优先（`torchtitan/experiments/transformers_modeling_backend/model.py:476-511`）。
5. DeepSeek V2/V3 被列入 remote-config denylist，避免旧 remote config 与本地 Transformers model code 混配（`torchtitan/experiments/transformers_modeling_backend/model.py:149-154`、`torchtitan/experiments/transformers_modeling_backend/model.py:408-419`）。

### ④ 成本/失败边界

- 这是 mutable config 合并，不是可逆 schema 映射；`vars(hf_model_config)` 中下划线字段不会复制，attention implementation 由 backend 另行恢复（`torchtitan/experiments/transformers_modeling_backend/model.py:332-356`、`torchtitan/experiments/transformers_modeling_backend/model.py:442-456`）。
- “非默认即显式”的判断依赖 dataclass default；若新增字段把有意义的 HF 值误设为具体 TorchTitan default，就会再次发生 silent override（`torchtitan/experiments/transformers_modeling_backend/model.py:297-330`）。
- 对 debug flavor 改 `dim` 会有意触发派生尺寸更新；full flavor 则保留模型专属关系。两者不能用同一套“统一推导”心智模型。

### ⑤ 有锚点的趋势

提交 `ac240f926c8c70d188ef0e2f353991549d77187e` 修复的五个根因都围绕“配置/初始化/checkpoint 适配必须保持模型事实”；HEAD 又对 composite model 的 `text_config`、MLA head dim 和 remote config 加了专门处理（`torchtitan/experiments/transformers_modeling_backend/model.py:421-435`、`torchtitan/experiments/transformers_modeling_backend/model.py:495-511`）。**推断**：新增模型首先会扩展显式兼容边界，而不会把全部 HF 差异压回一个扁平公共 config。

---

## 3. Module 化与 sharding：动态类替换保留 forward，声明覆盖负责 fail-fast

### ① 背景/问题

`spmd_types` 的 `Module.parallelize()` 需要每个有状态模块声明 parameter/buffer 与输入输出布局。HF 子树只有普通 `nn.Module`；若只给顶层套 wrapper，内部 projection/norm/RoPE buffer 仍会在 TP 下混用 plain Tensor 与 DTensor。

### ② 为什么动态换类，而不是重建对象树

选中路线对已存在实例执行 `__class__` swap：新类同时继承 `Module` 与原 HF class，因此对象状态、forward 和 state-dict key 不变。替代方案是逐层构造 wrapper/复制 parameter；那会改变 FQN/权重共享关系，并增加 checkpoint 适配面。决定标准是“获得协议能力，同时不改变 HF 对象身份和 key 空间”（`torchtitan/experiments/transformers_modeling_backend/module_conversion.py:7-15`、`torchtitan/experiments/transformers_modeling_backend/module_conversion.py:29-50`）。

### ③ 当前精确机制

- conversion 递归匹配 `Module.parallelize()` 的 `children()` traversal；普通 embedding 转成 TorchTitan `Embedding`，已是 `Module` 的 Titan MoE/HF root 跳过（`torchtitan/experiments/transformers_modeling_backend/module_conversion.py:42-65`）。
- 然后 `set_hf_sharding_configs()` 为 embedding、norm、lm_head、RoPE 和每层安装配置，并在末尾扫描所有拥有直接 parameter/buffer 的模块（`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:104-192`）。
- attention 边界先把 SP hidden state gather 为 TP replicate；Q/K/V projection col-shard，O projection row-shard。MLA 的低秩 Q/KV 下投影 replicate、上投影 col-shard；无法识别投影命名时直接报错并要求增加 case（`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:266-327`）。
- dense MLP 的 gate/up 是 colwise、down 是 rowwise；直接挂在 decoder layer 的 parameter/buffer 也显式 replicate（`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:362-394`）。
- 最终 completeness backstop 把“深处 mixed Tensor/DTensor”提前为 setup-time 缺配置错误（`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:491-524`）。

### ④ 约束/失败边界

- 含不兼容 `__slots__` 的 HF class 会跳过动态换类并记录 warning；若它拥有状态，后续完整性扫描才会失败（`torchtitan/experiments/transformers_modeling_backend/module_conversion.py:56-65`）。这不是无条件支持任意 Python module class。
- 支持依赖结构语义和属性名：attention 必须能识别 q 或 q_a/q_b、k/v 或 kv_a/kv_b，MLP 必须匹配 gate/fc1、down/fc2。未知架构的第一步是补 mapping 与测试，不是期待自动推断（`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:283-321`、`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:362-381`）。
- GLM-5 DSA indexer 在 TP 下因 scatter/index 的 Tensor/DTensor dispatch 缺口显式 `NotImplementedError`；FSDP/EP 无 TP 时仅 replicate state（`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:335-356`）。其中注释仍说“move ... onto spmd_types”，与整个文件已经使用 spmd_types 的 HEAD 状态矛盾，是局部过时注释；真实未解决点是 indexer op 的 local boundary，而不是 backend 选择。

### ⑤ 有锚点的趋势

`hf_sharding.py` 已从“为已知 happy path 打 hook”演进成角色映射 + completeness backstop。**推断**：新架构支持的质量门会继续是“所有 state 有声明、特殊 op 有 local boundary、未知结构早失败”，而不是把 completeness scan 放宽成 warning。

---

## 4. Attention 与输入所有权：先构造 mask/CP shard，再做 SPMD annotation

### ① 背景/问题

packed SFT 需要“因果且同文档”的 mask；CP 又会重排/切分 token 与 mask。若 HF 内部按 local sequence 临时构造 mask，RoPE position、BlockMask query shard 与 K/V 全局视图可能不一致。

### ② 为什么让 model 拥有 preprocess_inputs

当前选择由 `HFTransformerModel.preprocess_inputs()` 在模型语义边界构造 attention 输入，再依次 CP shard 和 SPMD annotate。替代方案是 backend Trainer 重写 post-dataload hook；提交 `e8e39abc...` 的动机是让不同模型声明自己的附加输入和 sharding，避免 decoder 特例泄漏到 Trainer。决定标准是 mask/position 属于模型 forward 协议，而不是 dataloader 或训练循环的通用状态。

### ③ 当前调用链与布局

1. 若 batch 未提供 mask，model 用 `positions` 构造 causal 或 causal+same-document mask；然后 `prepare_context_parallel_input()`，最后只对 decoder input 声明中的字段做 SPMD annotation（`torchtitan/experiments/transformers_modeling_backend/model.py:1169-1213`）。
2. 普通模型构造 Flex `BlockMask`；DSA 因 indexer 要对 mask 做 `.dim()`、加法与 top-k，改用 dense `[1,1,T,T]` additive mask（`torchtitan/experiments/transformers_modeling_backend/model.py:1215-1275`）。
3. forward 使用传入的 per-document positions 驱动 RoPE，并把预构造 mask 放进 HF `attention_mask`；没有 positions 才退回 local arange（`torchtitan/experiments/transformers_modeling_backend/model.py:1277-1309`）。
4. TP 下 `HFFlexKernel` 通过 `LocalMapConfig` 把 head-sharded DTensor 暂时降成本地 tensor 执行 Flex HOP；布局从 `(b,h,s,d)` 的 TP `S(1)`/CP `S(2)` 转为 `(b,s,h,d)` 的 TP `S(2)`/CP `S(1)`（`torchtitan/experiments/transformers_modeling_backend/model.py:39-88`、`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:195-239`）。
5. CP 下 Q 保持 sequence shard，K/V 在 kernel local region 内显式 `flex_cp_allgather()`，backward 由 autograd-aware collective reduce-scatter 回本地 shard（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:39-79`）。

### ④ 成本/失败边界

- CP + Flex 的 K/V all-gather 复制全序列 K/V；这是 query 分片换取 attention 正确性的通信/显存成本，不是 ring attention（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:39-53`）。
- 专用 Trainer 拒绝 CP `headtail`，只允许 PTRR 或无 balancer，因为 headtail 不能 shard Flex BlockMask（`torchtitan/experiments/transformers_modeling_backend/trainer.py:32-48`）。
- PTRR 还要求 query block 数能被 CP degree 整除；入口注释明确短序列会由 balancer 抛错（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:113-121`）。
- DSA 使用 dense quadratic mask；它解决 BlockMask API 不兼容，但增加 `T²` mask 内存，并不解除 TP indexer 的 op 限制（`torchtitan/experiments/transformers_modeling_backend/model.py:1252-1275`、`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:335-354`）。

### ⑤ 有锚点的趋势

提交 `31948fb2789c4aed6445698fb687dadbaf04593f` 最初把 SFT BlockMask 放在专用 Trainer hook；`e8e39abc...` 后 HEAD 已把它移到 model preprocess。**推断**：下一步的多模态/附加输入支持也更可能扩展 model-owned preprocessing/input sharding，而不是再增加 HFTrainer 分支；该可能性在 `e8e39abc...` 提交正文中也以多模态 CP 为展望，但尚无完成承诺。

---

## 5. MoE：历史 hook 方案已被统一 Titan MoE 替换

### ① 背景/问题

HF MoE 模型在 router、专家权重布局、共享专家、MLA/DSA 和 forward 返回结构上差异很大。EP 还要求统一 dispatcher/GroupedExperts 与 dense/sparse mesh 语义。直接在每个 HF forward 上叠 pre/post hook 会把通信协议绑定到模型内部细节。

### ② 为什么当前选择“探测配置，替换实现”

提交 `af35cfb1567f5c6573eb75d6ad95257ef4f8a40f` 的历史设计曾明确保留原 HF MoE forward，以 hooks 做 EP/TP/FSDP ordering；但 HEAD 已选择不同路线：先探测 HF MoE 结构并构造 `MoE.Config`，在 parallelize 前把 block 换成 TorchTitan native MoE。明显替代方案正是旧 hook 路线。当前判据是 dispatcher、GroupedExperts、load balancing 与 sharding 必须复用同一套 core 契约，不能让每个 HF for-loop 决定 collective 生命周期（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:97-105`、`torchtitan/experiments/transformers_modeling_backend/parallelize.py:145-179`）。

### ③ 当前实现思路

- model build 先从 layer 内或 layer-level router/experts 识别 MoE，并预计算 native MoE config（`torchtitan/experiments/transformers_modeling_backend/model.py:775-810`）。
- parallelize 阶段 `build_and_swap_native_moe()` 先于普通 HF module conversion，Titan MoE 子树因已满足 `Module` 而不会被二次换类（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:145-163`、`torchtitan/experiments/transformers_modeling_backend/module_conversion.py:49-65`）。
- `ModelSpec.post_optimizer_build_fn` 统一注册 MoE load-balancing hook，而不是让 backend Trainer 拥有更新逻辑（`torchtitan/experiments/transformers_modeling_backend/__init__.py:163-172`）。
- checkpoint 辅助转换处理 HF fused `gate_up_proj` 与 Titan gate/up 分离、router/shared-expert 重命名及当前 `routed_experts.inner_experts` FQN（`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:7-22`、`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:171-224`）。

### ④ 约束/失败边界

- README 的矩阵是 debug-scale、两步 loss 下降验证，不等于所有模型/规模的收敛证明（`torchtitan/experiments/transformers_modeling_backend/README.md:55-71`）。
- PP 尚未接入 MoE path；README 明确列为缺口（`torchtitan/experiments/transformers_modeling_backend/README.md:71-80`）。因此不能沿用 `af35cfb...` 提交正文中旧实现的 “PP+EP 已支持” 作为 HEAD 结论。
- GLM-5 DSA 与 Gemma4 分别限制 TP/CP 或 attention TP；准确支持面以当前 README 表为准（`torchtitan/experiments/transformers_modeling_backend/README.md:63-70`）。
- `MODEL_COMPATIBILITY.md` 的 PASS/WARN/FAIL 是 native Titan MoE 与未修改 HF forward 的数值对比，并记录 router/activation 不等价来源；“可训练”不等于“数值等价”（`torchtitan/experiments/transformers_modeling_backend/MODEL_COMPATIBILITY.md:11-26`、`torchtitan/experiments/transformers_modeling_backend/MODEL_COMPATIBILITY.md:126-138`）。

### ⑤ 有锚点的趋势

HEAD 注释已明确旧的 “MoE-only compile workaround” 因整 block MoE compile 可用而废弃（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:188-197`）。**推断**：MoE backend 正从 model-specific hook/compile 例外收敛到 core MoE + common compile；PP 与数值兼容仍是显式未闭环项。

---

## 6. 并行化顺序：协议布局在前，AC/compile/FSDP 在后

### ① 背景/问题

class swap、MoE replacement、local_map、AC、compile 和 FSDP 都会修改 module/forward/parameter 生命周期；顺序错了会让 compile 捕获不到 CP wrapper，或让 FSDP 管理同一个 tied parameter 两次。

### ② 为什么固定单向 phase order

当前顺序是 `untie → MoE swap → Module conversion → ShardingConfig → CP kernel wrap → model.parallelize → AC → compile → FSDP`（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:123-215`）。明显替代方案是在 FSDP/compile 后追加各类 hook。选中路线的决定标准是每一 phase 都消费前一 phase 已稳定的对象/布局：CP wrapper 必须被 local_map capture，compile 必须看见 AC wrapper，FSDP 必须最后按最终 parameter ownership 分组。

### ③ PP 与 FSDP 的具体边界

- PP 先按显式 FQN 或层数/首尾权重生成 virtual stages；不保留的模块换成 `Identity`，每个 stage 都保留 RoPE（`torchtitan/experiments/transformers_modeling_backend/pipeline.py:39-145`、`torchtitan/experiments/transformers_modeling_backend/pipeline.py:191-242`）。
- single-stage schedule 要求每 rank 一 stage，多-stage 至少每 rank 两 stage；virtual stage 数必须被 PP degree 整除（`torchtitan/experiments/transformers_modeling_backend/pipeline.py:303-367`）。
- 每个 model part 独立走同一个 parallelize_fn，返回的 compiled/modified model 必须回写 `stage.submod`（`torchtitan/experiments/transformers_modeling_backend/pipeline.py:375-408`）。
- FSDP 对 dense 与 sparse 参数选择不同 mesh；MoE+EP 用单个 `fully_shard` 的 placement callback 路由 expert params，避免 nested FSDP hooks 干扰 SAC op count（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:232-254`、`torchtitan/experiments/transformers_modeling_backend/parallelize.py:300-318`）。

### ④ 成本/失败边界

- 入口在 parallelize 前检测并复制 tied lm_head weight，以满足 FSDP2 group ownership；这会让从 scratch 训练的 embedding/head 此后独立更新（`torchtitan/experiments/transformers_modeling_backend/parallelize.py:123-143`）。而 `init_states()` 在同一 stage 同时有首尾模块时仍会调用 HF `tie_weights()`（`torchtitan/experiments/transformers_modeling_backend/model.py:1378-1383`）。权重共享语义因此依赖 PP placement/初始化/并行化时序，不能概括成“始终 tied”或“始终 untied”。
- meta init 会抹掉 HF 在 `__init__` 计算的 RoPE `inv_freq`；materialize 后必须用 `rope_init_fn` 重算，否则不会 crash 却会静默失去位置信息（`torchtitan/experiments/transformers_modeling_backend/model.py:1340-1376`）。
- PP stage 是 `deepcopy` 后删模块；不适合靠跨 stage Python object identity 表达共享状态（`torchtitan/experiments/transformers_modeling_backend/pipeline.py:191-242`）。

### ⑤ 有锚点的趋势

提交 `ac240f926c8c70d188ef0e2f353991549d77187e` 把 tied weights、RoPE buffer 与 state adapter 的 bug 归因到“加载之后的构造/初始化”，而非 safetensors 本身。**推断**：backend 正确性测试必须继续覆盖 meta→materialize→parallelize 的完整状态转换；仅比较 checkpoint bytes 不足以验证模型语义。

---

## 7. Checkpoint 与 SFT：可执行加载路径已经闭环，但 MoE 转换仍分层

### ① 背景/问题

HF checkpoint key 针对内部 causal-LM；TorchTitan 顶层又包了一层 `HFTransformerModel.model`，且 tied embeddings 可能省略 lm_head。MoE replacement 还会改变 expert tensor 的融合形式与 FQN。

### ② 为什么分成“通用 adapter + MoE layout converter”

当前选中路线把核心 checkpoint bridge 保持最小：strip/add `model.` 和补 tied lm_head；MoE 的 fused/split/rename 放在独立转换函数。替代方案是在一个 state adapter 中为所有模型硬编码 reshape。决定标准是 dense wrapper 的 key 差异稳定，而 MoE layout 与 `GroupedExperts` 命名会随实现变化，必须动态发现（`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:7-22`、`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:93-102`）。

### ③ 当前闭环

- `HFTransformerStateDictAdapter.to_hf()` 移除顶层 `model.`，tied 情况删除可缺省 lm_head；`from_hf()` 反向补回并恢复 prefix（`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:52-90`）。
- `ModelSpec` 注册该 adapter；SFT full config 提供本地 HF assets path，并要求 `initial_load_in_hf + initial_load_model_only`（`torchtitan/experiments/transformers_modeling_backend/__init__.py:163-172`、`torchtitan/experiments/transformers_modeling_backend/config_registry.py:168-220`）。
- MoE converter 按当前 expert param placement 发现 gate/down/up 名字，再完成 fused `gate_up_proj` 拆分与反向 concat（`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:93-102`、`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:171-224`、`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:227-289`）。

### ④ 约束/失败边界

- README “Load HF weights” 已被代码事实超越；真正现状是 dense/SFT loading 已有具体 recipe，不能据此推导所有 MoE pretrained loading 组合都已接入 DCP 主路径。
- state adapter 的 docstring 说 wrapper prefix 是“only difference”，这对 dense 主路径成立；MoE replacement 明显还需要 layout converter，源码自己也将两者分成 complementary pieces（`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:7-22`、`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:52-61`）。
- 本地 HF loading 还受 core checkpoint 的 assets path、model-only 与 remote storage guards 约束；详见 [[03_torchtitan_checkpoint_state_recovery_analysis]]，本页不重复扩写 DCP 事务。

### ⑤ 有锚点的趋势

提交 `31948fb2789c4aed6445698fb687dadbaf04593f` 和 `ac240f926c8c70d188ef0e2f353991549d77187e` 已把 SFT/pretrained dense 从实验愿望推进成 executable config。**推断**：README 的 “Load HF weights” 应收窄为 MoE/更多格式与组合验证，而不是继续描述为全局缺失；源码尚未提供统一完成矩阵。

---

## 8. 当前支持面、旧知识纠偏与仍遗漏的点

### ① 背景/问题

这个 backend 的历史变化快：旧提交正文、README、局部 TODO 和 HEAD 可执行路径会互相冲突。知识库若只罗列“4D + compile”会漏掉决定能否正确运行的状态边界。

### ② 判定规则

本页按证据优先级判断：**HEAD executable path/test > HEAD README compatibility table > 历史提交设计**。替代方案是把所有来源并列合并；那会同时得出“MoE 保留 HF forward”与“MoE 被替换”、“HF weights 未支持”与“已有 SFT load recipe”这类自相矛盾结论。

### ③ 已确认的纠偏

| 旧/表面断言 | HEAD 事实 | 证据 |
|---|---|---|
| backend 只是 HF model + Trainer | 中间有 config bridge、Module class swap、完整 ShardingConfig 覆盖、state adapter | `torchtitan/experiments/transformers_modeling_backend/model.py:399-522`、`torchtitan/experiments/transformers_modeling_backend/module_conversion.py:42-65`、`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:104-192`、`torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py:52-90` |
| MoE 保留原 HF forward、靠 hooks 并行 | HEAD 在 parallelize 前 build/swap native Titan MoE；旧说法属于提交 `af35cfb1567f5c6573eb75d6ad95257ef4f8a40f` 的历史实现 | `torchtitan/experiments/transformers_modeling_backend/parallelize.py:97-105`、`torchtitan/experiments/transformers_modeling_backend/parallelize.py:145-179` |
| HF weights 尚不能加载 | SFT full recipe 已启用 HF initial model-only load；README backlog 过时 | `torchtitan/experiments/transformers_modeling_backend/README.md:78-82`、`torchtitan/experiments/transformers_modeling_backend/config_registry.py:168-220` |
| 任意 HF 架构都能自动 TP | 只支持识别的 projection/MLP/RoPE 结构，缺声明 fail-fast；DSA TP 显式禁用 | `torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:283-321`、`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:335-356`、`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:491-524` |
| Flex CP 等于 ring attention | 当前 Q shard、K/V 显式 all-gather；PTRR/BlockMask 有额外整除与 balancer 限制 | `torchtitan/experiments/transformers_modeling_backend/parallelize.py:39-79`、`torchtitan/experiments/transformers_modeling_backend/parallelize.py:113-121`、`torchtitan/experiments/transformers_modeling_backend/trainer.py:32-48` |
| “可运行”就表示 HF 数值完全等价 | compatibility 把多个模型标成 WARN/FAIL，并列出 router/activation 差异 | `torchtitan/experiments/transformers_modeling_backend/MODEL_COMPATIBILITY.md:11-26`、`torchtitan/experiments/transformers_modeling_backend/MODEL_COMPATIBILITY.md:126-138` |

### ④ 仍遗漏/未闭环的边界

1. **MoE + PP**：README 明确未接入（`torchtitan/experiments/transformers_modeling_backend/README.md:71-80`）。
2. **LoRA 与 Titan RL**：仍在 Further work（`torchtitan/experiments/transformers_modeling_backend/README.md:78-83`）。
3. **HF vs native 性能**：README 只说 MFU 较低，没有机制归因或 benchmark methodology（`torchtitan/experiments/transformers_modeling_backend/README.md:73-76`）。这是性能证据缺口，不应猜测成某个 kernel 的责任。
4. **FSDP-only vs FSDP+PP 数值差异**：当前仅记录不 bitwise match、怀疑 seed checkpoint buffer；没有 closed root cause（`torchtitan/experiments/transformers_modeling_backend/README.md:73-76`）。
5. **未知架构扩展**：动态换类与 attribute mapping 没有通用 schema；每个新模型仍需覆盖结构、mask、初始化、state dict 与数值测试。
6. **局部注释债务**：DSA TODO 仍把解决方案写成“迁到 spmd_types”，而 HEAD 已在 spmd_types；真正缺口需重新定义为 local-map/op 支持（`torchtitan/experiments/transformers_modeling_backend/hf_sharding.py:335-354`）。

### ⑤ 有锚点的趋势

当前演进锚点是 Trainer 特例下沉、full_dtensor 删除、MoE hook 路线收敛到 native Titan MoE，以及 executable SFT/HF load。**推断**：backend 的核心工作正在从“能接进训练”转为“明确每个架构的协议兼容和数值/性能证据”；在上述缺口闭环前，不能把支持表外推成“任意 Transformers 模型的 4D 并行”。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 系列基线、概念所有权与剩余覆盖缺口。
- [[01_torchtitan_trainer_quickstart]] —— backend 复用的 Trainer 构造、单步提交与恢复生命周期。
- [[03_torchtitan_checkpoint_state_recovery_analysis]] —— HF initial load、state adapter 与 DCP checkpoint 的通用事务边界。
- [[04_torchtitan_config_model_protocol_analysis]] —— `Configurable`、`ModelSpec`、`Module` 与 model-owned preprocessing 协议。
- [[13_torchtitan_cp_analysis]] —— 输入 sequence ownership、BlockMask/PTRR 与 CP attention 边界。
- [[15_torchtitan_ep_analysis]] —— native Titan MoE replacement 最终复用的 dispatcher 与 sparse rank plane。
- [[16_torchtitan_spmd_types_analysis]] —— `ShardingConfig`、local map、current mesh 与 fail-fast layout contract。
