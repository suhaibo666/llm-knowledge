---
title: "多模态数据—模型契约：placeholder、packed patches 与 vision scatter 的一致性"
---

# 多模态数据—模型契约：placeholder、packed patches 与 vision scatter 的一致性

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：TorchTitan 多模态训练的核心不是某个视觉编码器，而是一条跨越 CPU 数据处理、packing、collation、SPMD 边界与模型前向的强一致契约：数据侧必须按最终视觉网格插入恰好相同数量的 placeholder，并保持 document/media 顺序；模型侧必须用视觉编码器的真实输出长度再次验算，只有整条视觉流与 placeholder runs 一一对应后才允许覆盖文本 embedding。当前实现宁愿丢弃坏样本或显式报错，也不以截断、广播或静默错位换取“继续运行”。
>
> 本页只回答共享的“数据怎样与模型对齐”问题，不介绍 Qwen3.5、Kimi 或 Muse Glimmer 的模型结构。Grain 图、并行 mesh、CP 算法、SPMD 类型系统与模型协议分别见 [[02_torchtitan_data_pipeline_grain_analysis]]、[[10_torchtitan_parallel_dims_analysis]]、[[13_torchtitan_cp_analysis]]、[[16_torchtitan_spmd_types_analysis]] 与 [[04_torchtitan_config_model_protocol_analysis]]。

---

## 1. Overview：多模态训练首先是两条有序流的 join

### ① 背景/问题

文本流只需要 token ID；视觉流却先经历 resize、时空 patch 化、空间 merge，再变成不等长的 embedding。模型不能凭一张图片的原始高宽猜出它最终占据多少文本位置，也不能只知道“本 batch 有三张图”就确定三段 embedding 应覆盖哪里。真正需要保持的是两条有序流之间的 join key：第几个 media item、它对应哪一段连续 placeholder、该段长度是否等于编码器输出长度。

TorchTitan 因而让 `MultiModalTokenizer` 成为五个特殊 token 字符串与 ID 的单一来源；初始化时每个 token 都必须已存在于 tokenizer 的 added-token 表中，collator 再把这些 ID 作为普通字典随 batch 送入模型（`torchtitan/components/tokenizer.py:499`、`torchtitan/components/tokenizer.py:532`、`torchtitan/components/tokenizer.py:538`）。

### ② 为什么这么设计

**选中的路线**是“数据侧先算占位长度，模型侧按真实输出复核，再原位 scatter”；**最明显的替代方案**是把图像 embedding 作为独立前缀拼在文本前面，或只凭预估长度直接覆盖。前者会改变原始 interleaved 文档语义，后者会把任何 resize、patch order 或 merge 配置漂移变成静默错位。当前共享 helper 的判据很明确：media 数、placeholder run 数、逐项长度和视觉 embedding 总数必须全部相等，否则抛错（`torchtitan/models/common/multimodal.py:54`、`torchtitan/models/common/multimodal.py:69`、`torchtitan/models/common/multimodal.py:88`、`torchtitan/models/common/multimodal.py:119`）。

### ③ 实现思路与真实调用链

```text
recipe: tokenizer + dataset processor + collator geometry
  -> Grain source / per-sample media decode & resize
  -> 根据处理后网格展开 placeholder；生成 shifted labels 并 mask 特殊 token
  -> 可选 FirstFit whole-document packing，media list 保序
  -> collator: 文本压成固定 token budget；视觉 patch 无 padding 拼接；可选 MRoPE
  -> Trainer.preprocess_inputs(): mask / CP / SPMD 输入边界
  -> model.forward() 的 DP-local multimodal region
  -> vision encoder 给出 packed embeddings + 每项真实长度
  -> get_vision_positions() 强校验
  -> scatter_vision_embeds() 覆盖文本流
  -> 恢复 token-aligned decoder 布局
```

Qwen3.5 recipe 从同一 processor 配置复制 patch、temporal 与 merge 几何到 collator，并启用 MRoPE（`torchtitan/models/qwen3_5/config_registry.py:29`、`torchtitan/models/qwen3_5/config_registry.py:34`）；Kimi K2.5 则显式改为 `14 × 14`、temporal 1、spatial merge 2、raster patch order 且不生成 MRoPE（`torchtitan/models/kimi_k2_7/config_registry.py:50`、`torchtitan/models/kimi_k2_7/config_registry.py:57`、`torchtitan/models/kimi_k2_7/config_registry.py:75`）。这不是外观配置：它们共同决定 placeholder 数、patch 顺序和模型输出数。

Grain loader 把 `num_tokens_per_batch`、`max_context_length` 与 tokenizer 注入 dataset/collator，先 build 数据图再以 collator 做 batch，并在线程中预取完成 batch（`torchtitan/components/data/loader.py:101`、`torchtitan/components/data/loader.py:118`、`torchtitan/components/data/loader.py:133`、`torchtitan/components/data/loader.py:140`）。Trainer 保持 batch 在 CPU，直到每个 forward/microbatch 才调用模型的 `preprocess_inputs()`（`torchtitan/trainer.py:639`、`torchtitan/trainer.py:665`、`torchtitan/trainer.py:690`；PP 路径对应 `torchtitan/trainer.py:737`、`torchtitan/trainer.py:741`）。

### ④ 约束、代价与失败边界

这条契约故意把三个预算分开：文本 token budget、视觉 entry 上限、每项 patch budget。文本超出本地 token budget 会报错；视觉 entry 超上限也会报错；resize 则先约束单项像素/patch 数（`torchtitan/hf_datasets/multimodal/mm_collator.py:89`、`torchtitan/hf_datasets/multimodal/mm_collator.py:290`、`torchtitan/hf_datasets/multimodal/utils/image.py:288`）。因此“文本装得下”不代表视觉工作量平衡，反之亦然。

当前模块顶部旧流程图仍声称视觉输入 pad 成三维 `N × max_patches × patch_dim`；实际代码已经 `torch.cat` 为二维 packed patches（`torchtitan/hf_datasets/multimodal/mm_datasets.py:43` 对比 `torchtitan/hf_datasets/multimodal/mm_collator.py:73`、`torchtitan/hf_datasets/multimodal/mm_collator.py:76`）。本页以可执行路径为准，不沿用该过时注释。

### ⑤ 演进锚点

提交 `1b04fc1c3576a86fe8c07f886a97709164eb55c6` 把模型专用 loader 收敛为同一 Grain composition/checkpoint contract；提交 `73aed7f6c09c04041dae2d2c185bb4c6384ebb3f` 又把语言 batch 折成 `[T]`、视觉 patch 折成 `[total_patches, patch_dim]`，明确删除视觉 padding。**推断**：后续扩展新 VLM 的主要工作应继续是声明几何与模型侧真实 token-count 函数，而不是复制一套 dataloader。

---

## 2. Processor：placeholder 数必须来自“处理后”图像

### ① 背景/问题

原始图片尺寸不能直接决定视觉 token 数：resize 会把高宽对齐到 `patch_size × spatial_merge_size` 的倍数，NaViT 路线还会在长宽比约束后补右/下 padding。若在 decode/resize 之前插入 placeholder，同一图片可能在文本里预留 N 个位置，却在视觉塔中产出 M 个 token。

### ② 为什么这么设计

**选中的路线**是先完成 decode、resize、normalize，再从实际 tensor shape 计算 placeholder 长度；**替代方案**是用原图元数据或固定每图 token 数。决定性标准是 placeholder 必须描述视觉塔真正看到的网格，而不是输入文件的名义尺寸。`process_image()` 在内部捕获处理异常并返回 `None`；上层只要发现任一非空图片未成功处理，就丢弃整个 sample，而不是保留部分图片破坏 interleaved 对齐（`torchtitan/hf_datasets/multimodal/utils/image.py:200`、`torchtitan/hf_datasets/multimodal/utils/image.py:269`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:167`）。

### ③ 实现、状态与不变量

processor 先要求 `texts` 非空且与 `images` 等长，以相同索引表达 text/media 交错；每个成功图像被写回 `texts[idx] = None`，并把处理后 shape 交给 `calculate_vision_tokens()`（`torchtitan/hf_datasets/multimodal/mm_datasets.py:88`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:130`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:136`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:151`）。该函数用 temporal patch 的向上取整以及空间 patch/merge 后的行列数求总 token 数（`torchtitan/hf_datasets/multimodal/utils/image.py:280`、`torchtitan/hf_datasets/multimodal/utils/image.py:301`）。

随后每个 `None` 被展开成 `vision_start + N × vision_token + vision_end`，文末补 EOS（`torchtitan/hf_datasets/multimodal/utils/text.py:70`、`torchtitan/hf_datasets/multimodal/utils/text.py:98`、`torchtitan/hf_datasets/multimodal/utils/text.py:111`）。tokenize 后做一位 causal shift；label 中的 vision start/end、image、video 四类 ID 全部替换为 `IGNORE_INDEX`，所以 placeholder 提供上下文位置但不贡献语言建模 loss（`torchtitan/hf_datasets/multimodal/mm_datasets.py:185`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:189`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:192`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:204`）。

### ④ 约束、代价与失败边界

当前 image 路径把 `temporal_patch_size` 固定为 1；源码 TODO 明确说配置的 temporal patch size 在这里尚未使用（`torchtitan/hf_datasets/multimodal/mm_datasets.py:152`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:158`）。这与 collator 对单帧图像做 temporal repeat padding 并不自动等价：recipe 必须让 processor 的 placeholder 计数与目标 vision encoder 的合并语义一致。

processor 对 tokenize 后少于 2 个 token 的 sample 返回 `None`，对超过 `max_context_length` 的 sample 也返回 `None`，不做截断；后者避免把 placeholder run 或视觉边界从中间切断（`torchtitan/hf_datasets/multimodal/mm_datasets.py:185`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:328`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:334`）。代价是坏图和超长样本会直接损失数据。

### ⑤ 演进锚点

`torchtitan/hf_datasets/multimodal/mm_datasets.py:158` 的 `data-mm-temporal-patches` TODO 是明确未闭合项。**推断**：在统一 image/video token counting 前，不能把 processor 的默认 temporal 参数理解为所有模型都已端到端生效。

---

## 3. Document packing：保序比“装满”更优先，但尚未全局均衡

### ① 背景/问题

多模态 packing 不只是把 token 填满固定长度。若把两个文档的 token 合并，却按另一顺序拼 media，模型仍会看到合法 shape，只有语义被悄悄交换；若跨 DP rank 各自 first-fit，同样文本 token 数也可能对应完全不同的视觉 patch 工作量。

### ② 为什么这么设计

**选中的路线**是 whole-document first-fit：以完整样本为最小装箱单元，token 字段参与长度/补齐，media 字段作为 meta features 随文档移动并在输出端按文档顺序 flatten；**替代方案**是截断文档、先把 media 全局拼接，或仅按 token 数切固定窗口。决定性标准是任何 packing 操作都不能改变“文本中第 i 个 placeholder run 对应 media stream 第 i 项”的顺序。

### ③ 实现、状态与调用链

`MMSamplePackingConfig` 先拒绝非正 packing bins，再过滤超过 `max_context_length` 的完整样本；`FirstFitPackIterDataset` 将 `input_ids/labels/positions` 都装到本 DP rank 的 `num_tokens_per_batch`，尾部依次使用 pad ID、`IGNORE_INDEX`、位置 0（`torchtitan/hf_datasets/multimodal/mm_datasets.py:378`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:385`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:399`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:407`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:409`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:414`）。

`pixel_values` 与 `pixel_values_videos` 被列为 meta features；packing 输出恢复为 tensor 时，双层 list 按 document 顺序逐项 flatten（`torchtitan/hf_datasets/multimodal/mm_datasets.py:421`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:444`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:452`）。CPU 测试锁定了两个文档合并后 image/video 的原顺序，也锁定了 iterator checkpoint 恢复后的 media 内容（`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:204`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:220`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:234`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:249`）。

### ④ 约束、代价与失败边界

packing 是可选 wrapper；当前 `MM_DATASETS` registry 返回普通 `SingleDatasetConfig`，测试也明确断言 registry 不默认启用 packing（`torchtitan/hf_datasets/multimodal/mm_datasets.py:340`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:98`）。更大的 `num_packing_bins` 可减少文本 padding，却会让更多 media 暂存在开放 bins 中，源码字段说明直接记录了这一内存代价（`torchtitan/hf_datasets/multimodal/mm_datasets.py:381`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:383`）。

当前 packing 发生在 dataset 已按 DP policy build 之后；源码 TODO 明确指出还没有在 DP sharding 前建立 global pack plan，因此 rank 间文本与 media 负载可能不均（`torchtitan/hf_datasets/multimodal/mm_datasets.py:395`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:405`）。这不是 correctness 错误，但会形成 straggler，尤其当相同 token budget 对应的图像分辨率差异很大时。

Kimi K3 是显式例外：其 recurrent/causal-convolution 状态还不能在 document boundary reset，因此配置看到 `MMSamplePackingConfig` 就报错（`torchtitan/models/kimi_k3/model.py:274`、`torchtitan/models/kimi_k3/model.py:276`、`torchtitan/models/kimi_k3/model.py:278`）。

### ⑤ 演进锚点

`data-global-pack-plan` TODO 已把目标限定为“packing before DP sharding，使各 rank 获得相近 text/media work”（`torchtitan/hf_datasets/multimodal/mm_datasets.py:405`）。**推断**：真正的全局计划不能只平衡 token 长度，还要把视觉 patch 数作为负载维度，否则仍可能 token 平衡而 vision 不平衡。

---

## 4. Collator：两个 budget、两种 patch order 与可选 MRoPE

### ① 背景/问题

processor 保留每项媒体 tensor，模型却希望一次吞下连续 patch 序列；同时语言模型需要固定本地 token budget。若对每张图 pad 到 batch 最大 patch 数，会浪费显存并让“真实输出长度”难以追踪；若盲目拼 patch，又会丢失 item boundary 和空间坐标。

### ② 为什么这么设计

**选中的路线**是分别打包文本与视觉：文本拼到固定 `[T]` 并只 pad 尾部，视觉 patch 完全不 pad、拼成 `[P, patch_dim]`，另用每项 `[t,h,w]` 的 `grid_thw` 保存分段；**替代方案**是保留 `[N,max_patches,*]`。决策标准是将 padding 成本限制在文本 token budget，同时让视觉塔仍能按 grid 重建每项边界（`torchtitan/hf_datasets/multimodal/mm_collator.py:49`、`torchtitan/hf_datasets/multimodal/mm_collator.py:57`、`torchtitan/hf_datasets/multimodal/mm_collator.py:61`）。

### ③ 实现、状态与布局

`collate_text()` 依次拼接完整样本的 input、label、position；超过 budget 报错，不足则只补 batch 尾部，labels 补 `IGNORE_INDEX`，padding positions 按 `max_context_length` 循环（`torchtitan/hf_datasets/multimodal/mm_collator.py:81`、`torchtitan/hf_datasets/multimodal/mm_collator.py:86`、`torchtitan/hf_datasets/multimodal/mm_collator.py:89`、`torchtitan/hf_datasets/multimodal/mm_collator.py:92`）。对应测试锁定了 flatten 顺序和 padding positions（`tests/unit_tests/cpu/test_packed_vision.py:34`、`tests/unit_tests/cpu/test_packed_vision.py:59`、`tests/unit_tests/cpu/test_packed_vision.py:65`、`tests/unit_tests/cpu/test_packed_vision.py:85`）。

`collate_images()` 对每项调用 `vision_to_patches()`，然后沿 patch 轴 concat、对 grids stack；测试明确验证两个不等长 patch 序列没有视觉 padding（`torchtitan/hf_datasets/multimodal/mm_collator.py:63`、`torchtitan/hf_datasets/multimodal/mm_collator.py:73`、`tests/unit_tests/cpu/test_packed_vision.py:92`、`tests/unit_tests/cpu/test_packed_vision.py:113`）。block order 让同一 `merge_size × merge_size` 组连续，适合 Qwen 风格 merger；raster order 保持行优先，适合 MoonViT 的位置索引（`torchtitan/hf_datasets/multimodal/utils/image.py:317`、`torchtitan/hf_datasets/multimodal/utils/image.py:319`、`torchtitan/hf_datasets/multimodal/utils/image.py:322`、`torchtitan/hf_datasets/multimodal/utils/image.py:369`）。

collator 独立统计 image entries 和 video temporal entries；video 用向上取整，与 patcher 重复最后一帧补齐 temporal group 的行为一致（`torchtitan/hf_datasets/multimodal/mm_collator.py:281`、`torchtitan/hf_datasets/multimodal/mm_collator.py:284`、`torchtitan/hf_datasets/multimodal/utils/image.py:356`）。最终 batch 同时携带 packed pixels、grids、1D positions 与 special token IDs（`torchtitan/hf_datasets/multimodal/mm_collator.py:297`、`torchtitan/hf_datasets/multimodal/mm_collator.py:316`、`torchtitan/hf_datasets/multimodal/mm_collator.py:318`）。

Qwen 路线还在 CPU collator 构建 `[T,3]` MRoPE；它用 1D positions 等于 0 的位置识别 packed-document 边界，在每个文档内给文本重复 T/H/W 三轴位置、给视觉 span 分配时空网格（`torchtitan/hf_datasets/multimodal/mm_collator.py:117`、`torchtitan/hf_datasets/multimodal/mm_collator.py:155`、`torchtitan/hf_datasets/multimodal/mm_collator.py:168`、`torchtitan/hf_datasets/multimodal/mm_collator.py:226`、`torchtitan/hf_datasets/multimodal/mm_collator.py:230`）。模型 preprocess 仍用 1D positions 建 attention mask，却把 MRoPE 替换为 decoder 的 RoPE positions（`torchtitan/models/qwen3_5/model.py:379`、`torchtitan/models/qwen3_5/model.py:388`、`tests/unit_tests/test_qwen3_5_mrope_positions.py:104`、`tests/unit_tests/test_qwen3_5_mrope_positions.py:122`）。

### ④ 约束、代价与失败边界

MRoPE 只接受 block patch order；raster 会立即报错，因为坐标序列会与 patch 序列脱节（`torchtitan/hf_datasets/multimodal/mm_collator.py:134`、`torchtitan/hf_datasets/multimodal/mm_collator.py:136`）。TorchTitan 当前让一个 video 对应一个连续 placeholder run，因此一个原始 `[T,H,W]` grid 被当作完整 3D region 消费，而不是模仿 Transformers 的逐帧 span（`torchtitan/hf_datasets/multimodal/mm_collator.py:141`）。

视觉 entry 上限不是视觉 token/patch 总上限：一张高分辨率图只计一个 entry，却可能贡献很多 patch；单项 patch 上限由 resize 配置承担。**推断**：当前 `max_images_per_batch` 是防止媒体项/temporal-group 数失控的 guard，不是严格 GPU 工作量预算。

### ⑤ 演进锚点

提交 `539aa6cdf6110cc9555f694903fd7713f88cbc0d` 修复了 Qwen 视频 MRoPE 错把一个连续 video run 拆成逐帧 grids 的错位；提交 `304fd882e4876e134f63c0a7bbb1736b10fb60a8` 又把 video entry 计数由 floor 修为 ceil。两次修复共同说明 shape 可运行并不等于 ordering 正确，temporal padding、placeholder runs 与 grid 消费必须同步演进。

---

## 5. 模型边界：先在 DP-local region 完成视觉 join，再回到 token 布局

### ① 背景/问题

每个 DP rank 读取自己的图片，这些视觉 tensor 是“本 rank 的可变长对象集合”，而 decoder hidden state 是沿 token 轴表达的全局 SPMD 值。若直接把 pixel tensor 当成普通 token tensor沿 DP/CP 切分，某个视觉项可能被拆到多 rank，而其 placeholder run、grid metadata 与 encoder 输出却不再共同拥有。

### ② 为什么这么设计

**选中的路线**是把 `pixel_values/grid_thw` 声明成 DP-local value、TP-invariant，在一个局部 mesh region 内完成 text embedding、vision encoder 和 scatter；scatter 后结果重新变成 token-aligned tensor，再恢复 decoder 的全局 DP 布局（`torchtitan/models/common/vision_encoder_sharding.py:26`、`torchtitan/models/common/vision_encoder_sharding.py:29`、`torchtitan/models/common/multimodal.py:24`、`torchtitan/models/common/multimodal.py:30`）。

**最明显的替代方案**是先按 CP 切文本/视觉输入再 scatter。当前源码直接否定了它：Kimi K2.5 说明 vision scatter 需要 CP 切分前的完整序列；Qwen3.5 还同时受 GatedDeltaNet full-sequence all-gather 限制（`torchtitan/models/kimi_k2_7/parallelize.py:55`、`torchtitan/models/qwen3_5/parallelize.py:59`）。决定性标准不是“视觉 encoder 能否并行”，而是每个完整 media item、其完整 placeholder run 与 token-count metadata 是否在 scatter 点共址。

### ③ 实现、状态与模型差异

`multimodal_context()` 仅在 `spmd_types` 且 DP 大于 1 时把 DP 设为 local axis；其他后端或 DP=1 是 no-op（`torchtitan/models/common/multimodal.py:33`、`torchtitan/models/common/multimodal.py:35`）。Qwen3.5 在该 region 内先做文本 embedding，再分别编码 image/video；视觉塔返回 packed embeddings，模型由 `grid_thw.prod(-1) // spatial_merge_unit` 求每项真实长度，再定位对应 placeholder runs（`torchtitan/models/qwen3_5/model.py:479`、`torchtitan/models/qwen3_5/model.py:495`、`torchtitan/models/qwen3_5/model.py:498`、`torchtitan/models/qwen3_5/model.py:530`、`torchtitan/models/qwen3_5/model.py:546`、`torchtitan/models/qwen3_5/model.py:576`）。

共享定位器先找连续 placeholder runs，要求 run 数等于 media item 数，再逐项比较 run length 与模型计算出的真实 token 数（`torchtitan/models/common/multimodal.py:61`、`torchtitan/models/common/multimodal.py:66`、`torchtitan/models/common/multimodal.py:71`、`torchtitan/models/common/multimodal.py:85`）。scatter 按 media 顺序依次覆盖文本 embedding，最后再验证消费总数等于 packed vision output 总数（`torchtitan/models/common/multimodal.py:112`、`torchtitan/models/common/multimodal.py:114`、`torchtitan/models/common/multimodal.py:119`）。

模型侧 token-count 规则并不统一：

| 模型路径 | placeholder/patch 契约 | 当前边界 |
|---|---|---|
| Qwen3.5 | image/video 分别定位；真实长度含 temporal grid 并除 spatial merge unit（`torchtitan/models/qwen3_5/model.py:498`、`torchtitan/models/qwen3_5/model.py:536`、`torchtitan/models/qwen3_5/model.py:552`） | CP 显式拒绝（`torchtitan/models/qwen3_5/parallelize.py:59`） |
| Kimi K2.5 | image/video 共用同一 placeholder ID；MoonViT 长度只取 merge 后 H/W，与 T 无关（`torchtitan/models/kimi_k2_7/model.py:170`、`torchtitan/models/kimi_k2_7/model.py:179`、`torchtitan/models/kimi_k2_7/model.py:185`） | 同 batch 混合 image+video 仍断言失败；CP 拒绝（`torchtitan/models/kimi_k2_7/model.py:173`、`torchtitan/models/kimi_k2_7/parallelize.py:55`） |
| Muse Glimmer multimodal | recipe 从 encoder 派生 patch/temporal/downsample 几何，模型按 downsample 后 H/W 定长（`torchtitan/models/muse_glimmer/config_registry.py:82`、`torchtitan/models/muse_glimmer/model.py:490`） | video 与 vision CP 未实现（`torchtitan/models/muse_glimmer/model.py:447`、`torchtitan/models/muse_glimmer/parallelize.py:53`） |
| Kimi K3 v1 | model 自己做 image count/scatter，但当前只支持 image（`torchtitan/models/kimi_k3/model.py:319`、`torchtitan/models/kimi_k3/model.py:343`、`torchtitan/models/kimi_k3/model.py:372`） | 只支持 FSDP2 + `partial_dtensor`，拒绝 TP/PP/CP/EP（`torchtitan/models/kimi_k3/parallelize.py:36`、`torchtitan/models/kimi_k3/parallelize.py:51`） |

### ④ 约束、代价与失败边界

当前“DP-local vision → token-aligned scatter”不能解释为 vision input 已支持 CP：shared input sharding 对视觉字段只有 DP 与 TP 规则，没有 CP 轴（`torchtitan/models/common/vision_encoder_sharding.py:34`、`torchtitan/models/common/vision_encoder_sharding.py:35`）。Qwen/Kimi preprocess 虽包含通用 `prepare_context_parallel_input()` 接线，但各自 parallelize 入口更早拒绝 CP；不能把存在死后备路径误写成受支持组合（`torchtitan/models/qwen3_5/model.py:412`、`torchtitan/models/qwen3_5/parallelize.py:59`、`torchtitan/models/kimi_k2_7/model.py:127`、`torchtitan/models/kimi_k2_7/parallelize.py:55`）。

模型侧验证发生在 forward，意味着错配 batch 已经完成 CPU 处理和设备传输；但它阻止了更危险的静默 embedding 交换。定位器仍会把三份 metadata tensor 各 `.tolist()` 一次；提交 `4d38b17adf34f82ea681140bbc1e4f0edb2e4538` 将旧的逐 item `.item()` 同步降为常数次 host transfer，源码对应实现位于 `torchtitan/models/common/multimodal.py:79`—`83`。代价没有消失，只是不再随 media item 数线性增加同步次数。

### ⑤ 演进锚点

提交 `85c549b9332067f7615241e73e0c4a28d44b065f` 把 vision-position/scatter 抽成 Qwen 与 Kimi 共用 primitive，并同时引入 block/raster patch-order 区分。当前 Kimi mixed-media TODO 要求在 unified placeholder 下构造 document-ordered 单一 vision stream（`torchtitan/models/kimi_k2_7/model.py:170`）。**推断**：未来若要支持 CP，合理方向也是先把“完整视觉 join”显式放到 CP sharding 之前，或设计能同时切 placeholder、vision embeddings 与 per-item metadata 的新协议，而不是只给 pixel tensor 加一个 CP placement。

---

## 6. 正确性检查表：哪些错误被哪里拦住

### ① 背景/问题

多模态错位常常不会表现为 shape error：两张图恰好产生相同 token 数时，交换顺序仍能正常训练。验证因此必须覆盖 ordering、count、geometry、document boundary 和恢复状态，而不能只检查 forward 能运行。

### ② 为什么这么设计

**选中的路线**是分层 fail-fast：processor 处理媒体完整性，packing 保序，collator 管 budget/patch order，模型复核最终 count；**替代方案**是只在模型末端做一个总 shape assert。分层验证能把错误定位到最早知道语义的边界，总数验证则作为最后防线。

### ③ 当前 guards 与 tests

| 风险 | 当前 guard / test |
|---|---|
| 文本/图片交错结构非法或坏图 | processor 返回 `None`（`torchtitan/hf_datasets/multimodal/mm_datasets.py:130`、`torchtitan/hf_datasets/multimodal/mm_datasets.py:167`） |
| processor/collator 几何漂移 | Qwen recipes 从 processor 复制三项几何；参数化测试遍历全部 Qwen3.5 recipe（`torchtitan/models/qwen3_5/config_registry.py:34`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:153`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:181`） |
| packing 交换 media 顺序 | image/video 顺序测试（`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:204`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:226`） |
| packing iterator 恢复改变下一 batch | buffered state restore 比较 token 与 media（`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:234`、`tests/unit_tests/cpu/components/data/test_qwen_multimodal_data.py:249`） |
| 视觉 padding 或 patch 顺序破坏 packed layout | concat patches 与 Qwen/Kimi merger 测试（`tests/unit_tests/cpu/test_packed_vision.py:92`、`tests/unit_tests/cpu/test_packed_vision.py:177`、`tests/unit_tests/cpu/test_packed_vision.py:207`） |
| partial temporal group 被容量计数漏掉 | 3 帧、temporal size 2 的错误测试（`tests/unit_tests/cpu/test_packed_vision.py:118`、`tests/unit_tests/cpu/test_packed_vision.py:129`） |
| placeholder run 与 encoder 输出不一致 | 共享 helper 对 run 数、逐项长度和总消费量抛 `ValueError`（`torchtitan/models/common/multimodal.py:71`、`torchtitan/models/common/multimodal.py:88`、`torchtitan/models/common/multimodal.py:119`） |
| MRoPE 与 attention boundary 混淆 | characterization test 验证 layer 收到 3D MRoPE、mask 仍按 1D positions（`tests/unit_tests/test_qwen3_5_mrope_positions.py:104`、`tests/unit_tests/test_qwen3_5_mrope_positions.py:120`、`tests/unit_tests/test_qwen3_5_mrope_positions.py:127`） |

### ④ 未覆盖边界与成本

当前 CPU tests 很好地覆盖了 packing 顺序、恢复、patch concat 和位置路由，但没有看到针对 `_process_mm_sample()` 的端到端单测来同时锁定“实际 resize 后 placeholder 数 + 四类 label mask + 坏图整样本丢弃”；这些性质目前主要由实现本身与下游模型 guard 共同保证。也没有全局 DP pack-plan 测试，因为该能力仍是 TODO。

视觉工作量的主要成本来自解码/resize、总 patch 数和 vision encoder，而不是文本 token 数本身；线程预取能隐藏部分 CPU 时间，但 loader 源码仍记录 CPU-heavy processing 未来应迁到多进程的 TODO（`torchtitan/components/data/loader.py:124`、`torchtitan/components/data/loader.py:128`）。

### ⑤ 趋势（基于源码 TODO 的推断）

明确可见的后续方向有三项：统一 image/video temporal token counting（`torchtitan/hf_datasets/multimodal/mm_datasets.py:158`）、在 DP sharding 前做全局 packing（`torchtitan/hf_datasets/multimodal/mm_datasets.py:405`）、让 Kimi unified placeholder 支持 mixed image/video（`torchtitan/models/kimi_k2_7/model.py:170`）。**推断**：这三项其实都在收紧同一个契约——让 media item 的计数、顺序和所有权在更早的 CPU 边界就确定，减少 forward 才发现错位的机会。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/02_torchtitan_data_pipeline_grain_analysis|Grain 数据管线]]：processor、packing、collator、checkpointable iterator 的通用组合边界。
- [[02_engineering/02_train_frameworks/torchtitan/04_torchtitan_config_model_protocol_analysis|配置—模型协议]]：recipe 如何把 tokenizer、dataloader 与模型 `preprocess_inputs()` 绑定成运行时契约。
- [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|ParallelDims 并行基座]]：DP-local 与 dense forward mesh 的 rank/所有权语义。
- [[02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis|Context Parallel]]：文本 token 在 attention 边界的 CP 切分；本页说明视觉 join 为什么必须先完成。
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|SPMD types]]：`V@DP`、local axes 与 scatter 后 token-aligned 类型的含义。
- [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|Pipeline Parallel]]：只有 embedding stage 执行多模态注入时，batch metadata 如何随 microbatch/stage 传播。
