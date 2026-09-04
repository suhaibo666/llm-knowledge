---
title: "Megatron-LM TensorRT-LLM 导出：从训练权重到可加载引擎"
---

# Megatron-LM TensorRT-LLM 导出：从训练权重到可加载引擎

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **功能树归属**：模块 M 的离线 TensorRT-LLM 权重导出；`megatron/core/export/**` 共 17 个 `.py`。
> **核心入口**：`TRTLLMHelper.get_trtllm_pretrained_config_and_model_weights` 与 `TRTLLMHelper.build_and_save_engine`。
> **最近更新**：2026-09-03。从原 44 合并页拆出，并补齐 conversion → engine build 的调用、配对与外部依赖边界。

---

## 1. 背景：训练态 state dict 不是推理引擎的装载格式

Megatron 的 state dict 按训练模块命名，并可能沿训练时的并行拓扑分片；TensorRT-LLM 需要自己的层名、rank mapping、模型配置和 engine build profile。`megatron/core/export/` 就是两者之间的离线适配层：

1. 把 Megatron 层名与张量布局转换成 TensorRT-LLM 权重字典；
2. 为目标推理 TP/PP rank 生成匹配的 TensorRT-LLM config；
3. 把一对权重/config 交给外部 `tensorrt_llm` 构建并保存 engine。

这不是 [[31_megatron_inference_engine_analysis]] 里的 `trt_llm_engine_wrapper.py`。后者是 Megatron 推理侧的后端接口占位；本页分析的是 `megatron/core/export/trtllm/` 的**离线格式转换与建引擎**。

> [!note] 证据边界
> 下文标作“源码行为”的内容都锚定冻结 commit 的函数体；“因此”“适合”等选路解释是从前置条件重建的工程判断，不代表仓库维护者发布的路线声明。

---

## 2. 对外契约：转换和建引擎是两个独立 API

`TRTLLMHelper` 的构造器 docstring 明列两个 public API（`megatron/core/export/trtllm/trtllm_helper.py:62-64`）：

| API | 输入 | 输出 / 副作用 |
|---|---|---|
| `get_trtllm_pretrained_config_and_model_weights` | Megatron state dict、数据类型、选路参数 | 两个**位置对齐的 list**：每个目标 rank 的权重字典与 config（`:264-350`） |
| `build_and_save_engine` | 上一步的一对 `weights` / `config` 与 build limits | 委托 builder 构建，保存到 `engine_dir`，返回 engine（`:532-614`） |

两者之间**没有自动调用**。调用者要接住两个 list，按相同下标配对，再为目标 rank 调第二个 API。分布式转换也统一返回 list，但只包一份当前 device 的权重/config（`:320-331`）；单设备转换则为 `world_size = inference_tp_size × inference_pp_size` 生成等长的多 rank lists（`:499-530`）。

构造 `TRTLLMHelper` 时就会检查外部 `tensorrt_llm` 是否可导入，缺失即抛 `ImportError`（`:82-85`）。因此即使只想转换、不立刻 build，这个 helper 也不是无 TensorRT-LLM 依赖的纯格式工具。

---

## 3. `ExportConfig`：描述目标推理拓扑，不描述训练拓扑

`megatron/core/export/export_config.py:9-30` 只有四个字段：

| 字段 | 默认 | 作用 |
|---|---:|---|
| `inference_tp_size` | `1` | 目标推理 tensor parallel 度。 |
| `inference_pp_size` | `1` | 目标推理 pipeline parallel 度。 |
| `use_parallel_embedding` | `False` | 目标 config 是否使用并行 embedding。 |
| `use_embedding_sharing` | `None` | 已弃用；设置非 `None` 会发 `DeprecationWarning`，替代项是 helper 的 `share_embeddings_and_output_weights`。 |

它表达的是**转换结果要被怎样切给推理 ranks**。但只有单设备路径消费调用者提供的 `ExportConfig`；分布式路径会根据已经建立的并行状态生成它，并禁止调用者再传一个可能冲突的目标拓扑（见 §4）。

`ExportConfig` 不属于 `megatron/training/config/training_config.py` 中由 coverage 枚举追踪的训练 config dataclass，所以本页没有配置字段 owner；它仍是明确的导出 API 契约，不能把“zero owner”误读成“没有配置”。

---

## 4. 转换入口：先提取量化 scale，再按 state dict 形态选路

入口先断言 state dict 非空。若 `fp8_quantized=True`，`_load_scaling_factors` 从 `._extra_state` 载入 forward scale / inverse scale（`megatron/core/export/trtllm/trtllm_helper.py:210-261`）；随后入口删除所有 key 中含 `extra_state` 的项（`:296-299`）。转换器完成普通权重改名/重排后，`_add_scales_to_converter` 再把 TensorRT-LLM scale 与可选 KV-cache scale 合并回输出权重字典（`:352-375`）。

**源码事实**是“先从 extra state 取 scale、再过滤 extra state、最后注入转换结果”。把被过滤部分称为“仅训练期状态、推理一概不需要”是对当前导出实现的解释，不是这段代码对所有 `extra_state` 的通用保证。

### 4.1 分布式 on-device 路径：输入已经按目标拓扑分片

开启 `on_device_distributed_conversion=True` 时，入口要求：

- `vocab_size` 与 `gpus_per_node` 都必须提供（`:301-319`）；
- model type 只能是 `gpt`、`gptnext`、`llama`、`nemotron_nas`（`:303-311`）；
- `export_config` 必须为 `None`，因为 inference TP/PP 从当前并行状态推断（`:312-315`）；
- 传入 state dict 已按所需推理模型并行方式分片，每个 device 只持有自己的部分（`:284-290`、`:387-390`）。

内部构造 `DistributedTRTLLMModelWeightsConverter` 并 `convert`（`:405-416`），从 converter 的并行状态生成 `ExportConfig`（`:419-423`），计算当前 model-parallel rank，设置 `tensorrt_llm.Mapping`（`:425-447`），返回当前 device 的一对结果（`:449`）。

**选路解释**：这条路径避免把完整模型聚到一台 CPU/GPU，却不负责把一个任意训练分片重新规划成另一个目标拓扑。若想指定新的 inference TP/PP，必须先以那个拓扑加载 state dict；入口的断言错误信息也直接给出这个操作要求（`:312-315`）。

### 4.2 单设备路径：完整 state dict 重切成目标 TP×PP

默认路径假设完整 state dict 位于 CPU 或一张 GPU（`:451-477`）。这里 `vocab_size` 必须留为 `None`，由输入层推断（`:333-336`）；调用者通过 `ExportConfig` 给目标 inference TP/PP。

`SingleDeviceTRTLLMModelWeightsConverter.convert` 先把完整权重转成统一 TRT-LLM 表示（`:482-497`）。随后代码遍历每个 `gpu_rank`，为该 rank 新建 `Mapping` 和 config，并调用 `get_local_model_weights_per_gpu` 取本地分片（`:499-528`）。输出因此是每个目标 rank 一份权重/config。

**选路解释**：只有这条路径在本函数内完成“完整权重 → 新目标拓扑”的重切分；代价是调用设备必须容纳完整 state dict。源码只保证输入可在 CPU 或单 GPU，不保证任何模型规模都能放得下。

### 4.3 层名映射是可扩展词典，不是运行时语义校验

helper 先复制 `DEFAULT_CONVERSION_DICT`；Nemotron-NAS 再叠加专属映射，最后叠加调用者的 `trtllm_conversion_dict`（`:87-92`）。映射实现位于 `model_to_trllm_mapping/default_conversion_dict.py`，权重改名/重排位于两个 converter 与 `trtllm_layers.py`。

这允许自定义映射覆盖默认项。相应风险是：层名配错可能把张量交给错误目标参数；本入口没有端到端数值等价验证。后半句是从函数职责边界得出的**分析结论**，不是说 TensorRT-LLM build 完全不做 shape/类型检查。

---

## 5. 完整执行链：conversion 的 list 终点如何接到 engine build

仓库示例给出了缺失的 caller hop，而不是让读者自行猜接法：单设备示例接住 `weight_list/config_list`，用 `zip(...)` 成对遍历后调用 build（`examples/export/trtllm_export/single_device_export/gpt_single_device_cpu_export.py:85-100`）；分布式示例接住两个单元素 list，再以 `[0]` 取当前 device 的配对结果（`examples/export/trtllm_export/distributed_export/gpt_distributed_gpu_export.py:85-99`）。

```text
Megatron state dict
  │
  ├─ TRTLLMHelper(...)
  │    └─ 检查 tensorrt_llm；合并默认/模型/用户层名映射
  │
  ├─ get_trtllm_pretrained_config_and_model_weights(...)
  │    ├─ FP8 时提取 scale；过滤 extra_state
  │    ├─ distributed：当前 device → [weights], [config]
  │    └─ single-device：完整权重 → [weights_rank0..N], [config_rank0..N]
  │
  ├─ 调用者按同一 rank/index 选择 weights_i + config_i
  │
  └─ helper.build_and_save_engine(engine_dir, weights_i, config_i, ...)
       └─ TRTLLMEngineBuilder.build_and_save_engine(...)
            ├─ 解析 architecture，构造 PluginConfig / BuildConfig
            ├─ model_cls.from_config(config_i)
            ├─ optimize_model(...)
            ├─ preprocess_weights(weights_i, config_i)
            ├─ model.load(weights_i)
            ├─ tensorrt_llm.commands.build.build(model, build_config)
            └─ engine.save(engine_dir) → return engine
```

逐跳证据如下：

1. helper 的转换 API 在入口清洗后调用分布式或单设备 helper，并分别返回两个 list（`megatron/core/export/trtllm/trtllm_helper.py:296-350`）。
2. 调用者负责 list 配对；两个官方 example 分别以 `zip` 和 `[0]/[0]` 闭合此跳。helper 内没有在两个 public API 之间保存“下一个待 build rank”，其接口只接受显式 `weights/config`（`megatron/core/export/trtllm/trtllm_helper.py:62-64`、`:532-536`）。
3. helper 的 build API 不再转换权重，只原样把 build 参数委托给 `TRTLLMEngineBuilder.build_and_save_engine` 并返回 engine（`:588-614`）。
4. builder 再次检查 TensorRT-LLM 依赖，解析 config 的 architecture，配置 attention/GEMM/paged-KV 等插件，并通过 `check_max_num_tokens` 归一 token 上限（`megatron/core/export/trtllm/engine_builder/trtllm_engine_builder.py:80-146`）。
5. 最后，builder 从 config 创建模型，优化模型，预处理并加载**与该 config 同 rank 的权重**，调用外部 build，保存 engine（`:159-172`）。

“同 rank 配对”不是文档风格建议，而是数据契约：单设备路径的两个 list 在同一个 `for gpu_rank` 循环里同步 append（`megatron/core/export/trtllm/trtllm_helper.py:503-528`）。交叉配对会让 mapping 与 local weights 不一致。

---

## 6. Engine build 的容量参数与外部边界

builder 的默认容量包括 `max_input_len=1024`、`max_output_len=1024`、`max_batch_size=4`；若 `max_seq_len` 未给，则设为前两者之和（`megatron/core/export/trtllm/engine_builder/trtllm_engine_builder.py:23-48`、`:108-122`）。`PluginConfig` 控制 paged KV cache、remove input padding、paged context FMHA、multiple profiles 与 reduce fusion（`:95-106`）；可选 LoRA 配置在 build config 上附加 `LoraConfig`（`:148-157`）。

这里已经跨出 Megatron 的执行边界：模型类、优化、权重预处理、真正 build 与 engine 序列化都调用外部 `tensorrt_llm` API（`:4-12`、`:159-172`）。因此：

- Megatron 本页能解释输入是怎样转换并传出，却不能仅凭这 17 个文件保证某个 TensorRT-LLM 版本、GPU 架构或 plugin 组合一定可 build；
- engine profile 的容量是 build-time 上限，不是从训练 sequence length 自动继承；调用者必须显式选择适合部署流量的限制；
- 冻结基线里 builder 自身有 `reduce_fusion` 参数（`:47`），但 helper 的 public wrapper 没有对应参数（`megatron/core/export/trtllm/trtllm_helper.py:532-612`）；要通过该 wrapper 使用时，该选项保持 builder 默认值 `False`。

---

## 7. 约束与排障判据

| 现象 | 先检查 | 源码边界 |
|---|---|---|
| helper 初始化即失败 | 是否安装可导入的 `tensorrt_llm` | `megatron/core/export/trtllm/trtllm_helper.py:82-85` |
| 分布式转换拒绝 model type | 是否属于四个显式支持项 | `megatron/core/export/trtllm/trtllm_helper.py:303-311` |
| 分布式转换拒绝 `export_config` | state dict 是否已按目标 TP/PP 加载；此路径从并行状态推断 config | `megatron/core/export/trtllm/trtllm_helper.py:312-315` |
| 单设备转换拒绝 `vocab_size` | 该路径要求从输入层推断，传参应为 `None` | `megatron/core/export/trtllm/trtllm_helper.py:333-336` |
| architecture 找不到 | config 的 architecture 是否能映射到 `tensorrt_llm.models`；Llama 名有一次兼容改写 | `megatron/core/export/trtllm/engine_builder/trtllm_engine_builder.py:85-93` |
| build 前 token 上限报错 | `max_seq_len`、batch、beam、remove-padding 与 profile 是否一致 | `megatron/core/export/trtllm/engine_builder/trtllm_engine_builder.py:108-146` |
| 某 rank 装载失败 | 是否把相同 index/rank 的 weights 与 config 配成一对 | `megatron/core/export/trtllm/trtllm_helper.py:503-528` |

另有一处维护风险：支持列表实际包含四种模型，但紧随其后的 assert message 只写 “gptnext and llama”（`megatron/core/export/trtllm/trtllm_helper.py:303-311`）。排障时应以布尔列表而不是过时的错误文本为准。

---

## 8. 小结

- `ExportConfig` 只描述目标推理 TP/PP 与 embedding 布局；单设备路径消费它，分布式路径从既有并行状态重建它。
- FP8 scale 在过滤 `extra_state` 前提取，并在普通权重转换后重新注入输出。
- 分布式路径省去全模型聚合但不在函数内重切拓扑；单设备路径能为目标 TP×PP 逐 rank 重切，但需要完整 state dict。
- conversion 与 engine build 是两个 public API。调用者必须把同 rank 的 `weights_i/config_i` 配对，再交给外部 TensorRT-LLM build 链。
- 本页没有训练 config owner，不等于没有配置：`ExportConfig` 与 engine build 参数都在训练配置枚举面之外。

## Related Pages

- [[11_megatron_dataset_analysis]] — 训练管线输入端；本页是训练产物离线交给部署引擎的输出端。
- [[31_megatron_inference_engine_analysis]] — 区分 Megatron 在线推理接口与本页的离线 TensorRT-LLM export。
- [[30_megatron_rl_posttraining_consistency_analysis]] — RL rollout 的在线权重交付/refit 是另一条模块 M 路径。
- [[19_megatron_dist_checkpointing_analysis]] — 导出前的训练权重怎样保存、加载与重组。
- [[40_megatron_feature_tree_analysis]] — 查看模块 M 在 A–Q 功能树中的完整边界。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] — 返回本域导航。
