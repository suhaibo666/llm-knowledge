---
title: "TorchTitan 数据管道：Grain 组合图、token budget 与精确续训"
---

# TorchTitan 数据管道：Grain 组合图、token budget 与精确续训

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：新版 TorchTitan 数据层不是“换成 Grain 的 DataLoader”，而是把 source、row processor、mix、packing、collator、DP 所有权和 checkpoint cursor 变成一张可组合、可递归保存状态的 dataset graph。它击败的是按模型维护四套 loader 的旧路线；代价是恢复拓扑、有限数据集、多进程处理和 packing 策略都成为显式契约。
>
> 本页负责 CPU 数据图直到 Trainer 收到扁平 token batch 的边界；模型侧 CP/SPMD 输入处理见 [[13_torchtitan_cp_analysis]] 与 [[16_torchtitan_spmd_types_analysis]]，训练状态整体保存与存储后端另见本系列 checkpoint 页面。

---

## 1. Overview：统一的不是数据格式，而是组合与恢复契约

TorchTitan 同时训练纯文本预训练、SFT、多模态模型和 Flux。旧实现让这些 workload 各自拥有 loader，导致 sharding、mixing、packing、checkpoint resume 的语义在多套代码里漂移。提交 `1b04fc1c3` 的 Previous behavior 明确记录了这种分裂；它选择用一个 Grain-backed pipeline 统一四类 workload，并把 source、processor、mix、packing、collator 定义成独立扩展点。

核心设计不是强迫所有样本长得一样，而是让各 workload 最终实现同一 `Stateful` loader 契约。Trainer 只消费 `(input_dict, labels)`，保存时又把同一个 loader 对象交给 checkpointer（`torchtitan/trainer.py:554`、`torchtitan/trainer.py:564`）。因此数据路径与 checkpoint 路径不是两个外挂系统，而是同一状态图的生产与恢复两面。

| 层 | 回答的问题 | 当前抽象 | 关键状态/边界 |
|---|---|---|---|
| source | 数据从哪里来、能否按索引/游标恢复 | `SourceConfig` | 文件 offset、HF stream cursor |
| processor | 一行如何变成训练样本 | `SampleProcessor` | token shift、SFT mask、随机种子 |
| composition | 多源怎样 concat/mix/shard/repeat | `DatasetConfig` | 权重语义、DP rank、epoch |
| packing | 样本怎样填入固定 token budget | concat-then-split / first-fit | residual、open bins、位置重置 |
| collator | dataset row 怎样成为 Trainer batch | `Collator` | padding、`IGNORE_INDEX`、dtype |
| loader | 怎样 batch/prefetch/checkpoint 整张图 | `GrainDataLoader` | iterator state、DP topology |
| Trainer | 怎样跨 PP/梯度累积统计有效 token | `batch_generator` / `train_step` | CPU batch、global valid tokens |

```text
source
  -> pre-filter -> processor -> post-filter
  -> global shuffle / DP shard / repeat
  -> concat or weighted mix
  -> packing
  -> batch + collator
  -> thread prefetch
  -> Trainer CPU batch
  -> device copy -> model.preprocess_inputs -> CP/SPMD boundary

checkpoint: loader iterator.get_state()  --------------------+
resume:     loader iterator.set_state() <--------------------+
```

### Quick Start：从 full configuration 组一条文本流

```python
config.dataloader = GrainDataLoader.Config(
    dataset=ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4"],
    ),
    collator=TextCollator.Config(),
    shuffle=True,
    repeat=True,
)
```

源码入口从 `GrainDataLoader.__init__()` 读起：它接收 Trainer 注入的 effective DP rank、tokenizer、最大上下文与每 microbatch token budget（`torchtitan/components/data/loader.py:75`），构造共享 context/policy（`torchtitan/components/data/loader.py:101`），再递归调用 dataset 和 collator 的 `build()`（`torchtitan/components/data/loader.py:118`）。标准文本 recipe 在 `torchtitan/hf_datasets/text_datasets.py:159`。

---

## 2. Source 与 Dataset 图：为什么用小协议组合，而不是继续扩展 model-specific loader

### ① 背景/问题

不同 workload 的差异主要落在“源如何读取”“单行如何处理”“样本在何时混合和打包”，而 DP sharding、shuffle、repeat、prefetch 与 checkpoint cursor 是横切需求。把这些差异都塞进一个大 DataLoader 会形成条件分支矩阵；继续保留四套 loader 又会重复恢复逻辑。提交 `1b04fc1c3` 的新设计要求所有 workload 共享 composition 和 checkpoint contract，同时允许独立替换各节点。

### ② 为什么这么设计

**选中的路线**是两个窄协议：`SourceConfig.build()` 只产出 random-access source 或 Grain stream，`DatasetConfig.build()` 只产出一个 dataset graph 节点（`torchtitan/components/data/sources.py:35`、`torchtitan/components/data/dataset.py:26`）；**明显替代方案**是按模型继承一个全功能 loader。决策准则是组合性：source、row transform、mix 和 packing 的笛卡尔积不应变成子类数量。

源码没有声明“Protocol 一定比继承快”；这里的收益是结构与状态归属，而不是运行时免费加速。实际性能还受 packing、prefetch 和 Python 处理开销影响，提交中的本地 debug benchmark 也明确不是 production-scale throughput 结论。

### ③ 实现思路与细节

`DatasetBuildContext` 只携带构图时所有节点共享的 tokenizer、最大上下文、token budget 和 Grain read options；`DatasetIterationPolicy` 单独携带 seed、shuffle、repeat、DP rank/world size 与 streaming shuffle buffer（`torchtitan/components/data/types.py:16`、`torchtitan/components/data/types.py:26`）。把“样本语义”和“本次运行策略”分开，使同一 recipe 可在训练/验证与不同 DP 拓扑下重建。

`SingleDatasetConfig` 的固定次序是：

1. 构建 source，并区分 map/iter 两类（`torchtitan/components/data/dataset.py:92`、`torchtitan/components/data/dataset.py:101`）。
2. 对 map source 做 pre-filter、带确定随机源的 processor、post-filter（`torchtitan/components/data/dataset.py:129`、`torchtitan/components/data/dataset.py:135`、`torchtitan/components/data/dataset.py:141`）。
3. 全局 shuffle 后再按 DP rank 切不重叠连续 slice，最后 repeat（`torchtitan/components/data/dataset.py:145`、`torchtitan/components/data/dataset.py:153`、`torchtitan/components/data/dataset.py:163`）。
4. streaming source 则先做窗口 shuffle，再按流顺序 filter/process；processor seed 额外加入 DP rank（`torchtitan/components/data/dataset.py:169`、`torchtitan/components/data/dataset.py:178`、`torchtitan/components/data/dataset.py:188`）。

随机访问 JSONL 不把所有行读入内存，而是保存 path id 和 byte offset，按需 seek/read（`torchtitan/components/data/sources.py:46`、`torchtitan/components/data/sources.py:61`、`torchtitan/components/data/sources.py:79`）。HF streaming source 则要求底层 dataset 本身支持 `state_dict/load_state_dict`，否则直接拒绝“看似能流式读取、实际不能精确续训”的源（`torchtitan/components/data/sources.py:181`）。

### ④ 约束/边界

- Indexed JSONL 当前每个 rank/worker 启动时重扫文件生成 offset；源码 TODO 建议未来用可 mmap 的 sidecar index（`torchtitan/components/data/sources.py:63`）。这是启动成本，不应误写成已经共享索引。
- map source 行数若小于 DP world size 会早失败，避免某些 rank 为空（`torchtitan/components/data/dataset.py:148`）。
- streaming shuffle 是每 rank 的有限窗口近似，不是全数据集均匀 permutation；buffer size 在 policy 中显式保存（`torchtitan/components/data/types.py:35`）。
- processor 的确定性只覆盖 Grain 注入的 NumPy RNG；模型、accelerator kernel 和训练 RNG 不由数据图自动保存。

---

## 3. Mix 与 DP 所有权：为什么“权重”必须说明是在样本、packed row 还是 token 上

### ① 背景/问题

多源训练常说“数据集 A 占 70%”，但 70% 可以指被尝试的索引、成功过滤后的样本、packed row 或物理 token。不同样本长度下，这些口径会产生不同训练分布。若先各自 packing 再 mix，长短样本的 token 权重又与先 mix 样本再 packing 不同。

### ② 为什么这么设计

当前设计不假装存在一个万能权重口径，而是让 dataset graph 的节点位置决定语义：mix 的 child 输出什么，权重就选择什么。**选中的路线**是结构显式、语义可追踪；**替代方案**是把所有 mix 自动换算成 token 权重。后者需要在线统计长度并把运行估计纳入 checkpoint，当前源码尚未实现，TODO 明确将 token-weighted mix 留作后续工作（`torchtitan/components/data/dataset.py:246`）。

### ③ 实现思路与细节

`DatasetMixConfig` 的 docstring 直接给出三种语义：全 map path 在 filter 存在时权重 attempted index；iterable path 权重实际 emitted element；若 child 是 packed fixed-length dataset，则权重 physical token（`torchtitan/components/data/dataset.py:207`、`torchtitan/components/data/dataset.py:210`）。每个 child 用 `seed + index`，所以增删或重排 child 会改变后续 child 随机序列（`torchtitan/components/data/dataset.py:234`、`torchtitan/components/data/dataset.py:237`）。

Concat 的语义不同：先临时关闭 child 的 shuffle/repeat/DP sharding，在单一全局 map dataset 上 concatenate，然后统一 shuffle 与 DP slice（`torchtitan/components/data/dataset.py:262`、`torchtitan/components/data/dataset.py:274`、`torchtitan/components/data/dataset.py:294`）。这保证“拼接后全局分片”，而不是把每个小 child 分别切到所有 rank。

### ④ 约束/边界

- 所有 mix 权重必须有限且大于零（`torchtitan/components/data/dataset.py:228`）。
- `repeat=False` 的 mix 在第一个 child 耗尽时停止，较大 child 不保证覆盖完（`torchtitan/components/data/dataset.py:216`）。
- all-map filter 会留下 rejected index 占位，因此权重 attempted index 而非 accepted sample；这是当前文档化语义，不是 bug 已修复。
- 恢复 checkpoint 后要保持 recipe 的 child 顺序与结构。源码没有为“重排数据集但沿用旧 cursor state”提供迁移协议。

---

## 4. 扁平 token budget 与两种 packing：为什么 batch 不再等于样本数

### ① 背景/问题

传统 `[B, L]` batch 把吞吐预算绑定到“样本数 × 固定长度”，但预训练文档、SFT 对话和多模态样本长度差异很大；每行 padding 会浪费 token slot，PP/梯度累积又真正关心每次 forward 处理多少 token。提交 `73aed7f6c` 因而把语言模型栈从 `[B,L]` 折成 `[T]`，配置也从 local/global batch size 改成每 microbatch/每 train step 的 token 数。

### ② 为什么这么设计

**选中的路线**是用固定 `num_tokens_per_batch` 作为数据—Trainer—模型的预算单位；**替代方案**是保留 synthetic batch 维并在每个模型入口 flatten。决策准则是让 PP microbatch、DP rank 和有效 token normalization 使用同一物理量，同时把 singleton batch 兼容形状从模型内部移除。代价是“一个 batch 包含多少样本”不再固定，位置边界也不再由旧物理行天然给出。

### ③ 实现思路与细节

processor 先负责 next-token 对齐；Trainer 不再做 token shift。`TextSequence` 的契约明确写出这一责任，并在构造时要求 input/label/position 等长，防止 supervision 在 packing 后静默错位（`torchtitan/components/data/dataset.py:38`、`torchtitan/components/data/dataset.py:42`、`torchtitan/components/data/dataset.py:53`）。普通文本 processor 生成 `tokens[:-1]` 与 `tokens[1:]`（`torchtitan/hf_datasets/text_datasets.py:45`、`torchtitan/hf_datasets/text_datasets.py:55`）。

两种 packing 服务不同目标：

| 策略 | 选中的语义 | 明显代价 |
|---|---|---|
| concat-then-split | 连续拼 token，再切满额 `[T]` row；适合预训练高填充率 | 长文会跨 batch 被切；尾部不足的 row 被过滤 |
| first-fit | 完整样本放入若干 open bins；适合 SFT 不切 response | 超过上下文样本丢弃；更多 bin 减 padding但占更多缓冲 |

Concat-then-split 把三字段长度固定成 token budget，并只保留完全填满的输出（`torchtitan/components/data/packing.py:46`、`torchtitan/components/data/packing.py:48`、`torchtitan/components/data/packing.py:54`）。First-fit 先过滤超长样本，用 `num_packing_bins` 在填充率与缓冲状态之间取舍（`torchtitan/components/data/packing.py:59`、`torchtitan/components/data/packing.py:63`、`torchtitan/components/data/packing.py:80`）。

packing 后依据 position==0 还原每个文档段内从零开始的位置，并把 padding segment 的 label 强制设为 `IGNORE_INDEX`（`torchtitan/components/data/packing.py:139`、`torchtitan/components/data/packing.py:143`、`torchtitan/components/data/packing.py:148`）。TextCollator 只填最后 token-batch 尾部；padding position 使用 0，避免很短序列产生超过模型上下文上限的位置（`torchtitan/components/data/collators.py:41`、`torchtitan/components/data/collators.py:68`、`torchtitan/components/data/collators.py:72`）。

### ④ 约束/边界

- concat-then-split 会切长文，first-fit 会丢长文；源码 TODO 承认尚无共享的 split/truncate/drop policy（`torchtitan/components/data/packing.py:44`）。
- SFT 当前只接受单轮 user/assistant；多轮语义与 loss mask 仍是 TODO（`torchtitan/hf_datasets/text_datasets.py:82`、`torchtitan/hf_datasets/text_datasets.py:85`）。超长 SFT 被丢弃而非截断（`torchtitan/hf_datasets/text_datasets.py:123`、`torchtitan/hf_datasets/text_datasets.py:126`）。
- prompt boundary 通过单独重 tokenize user prefix 得出，但当前只留 TODO 检查它是否真是 full conversation token 的精确前缀（`torchtitan/hf_datasets/text_datasets.py:133`、`torchtitan/hf_datasets/text_datasets.py:139`）。
- `TextCollator` 的默认 token id padding 是 0；loss 由 `IGNORE_INDEX` 屏蔽，但具体模型是否把 input 0 当作安全 padding 仍由 recipe/tokenizer 契约保证。

测试不只断言 shape：它检查 position 在文档内连续或归零，且文档末 token 绝不预测下一个文档的 BOS（`tests/unit_tests/cpu/test_text_dataset_packing.py:44`、`tests/unit_tests/cpu/test_text_dataset_packing.py:58`、`tests/unit_tests/cpu/test_text_dataset_packing.py:79`）。

---

## 5. 精确续训：为什么 checkpoint 必须保存整张 iterator graph，而不只是 source offset

### ① 背景/问题

数据消费状态不只是一条文件游标。shuffle window、repeat epoch、mix selector、packing residual/open bins 和 thread prefetch 都可能已经读入但尚未交给 Trainer。只保存“读到第 N 行”会在恢复后重复或跳过 token，即使模型和优化器完全恢复也改变训练轨迹。

### ② 为什么这么设计

**选中的路线**是让 `BaseDataLoader` 实现 PyTorch `Stateful`，并直接保存 Grain iterator 的递归状态；**替代方案**是 Trainer 维护 global sample index 并重放数据。重放在 streaming、filter、weighted mix 和 stateful packing 下既昂贵又不一定可重现。提交 `1b04fc1c3` 的验证专门比较连续 10 step 与 5 step checkpoint 后续训，证明“数据图状态”是本次迁移的主目标，而非附带接口。

### ③ 实现思路与细节

`GrainDataLoader.state_dict()` 保存版本、effective DP world size，以及以 `dp_rank_N` 命名的 iterator state（`torchtitan/components/data/loader.py:149`）。恢复时依次验证版本、DP degree 与 rank state 是否存在，再调用 `_iterator.set_state()`；失败会先关闭 iterator，避免残留 prefetch worker（`torchtitan/components/data/loader.py:156`、`torchtitan/components/data/loader.py:163`、`torchtitan/components/data/loader.py:171`）。

HF streaming source 自己再把 epoch 与底层 HF cursor 暴露给 Grain；repeat 到新 epoch 时重置初始 cursor，shuffle 时同步 `set_epoch()`（`torchtitan/components/data/sources.py:220`、`torchtitan/components/data/sources.py:238`、`torchtitan/components/data/sources.py:251`）。所以状态递归覆盖 source 与上层 transform，而不是只有 loader 外壳。

Trainer 构建 checkpointer 时把真实 dataloader 对象作为一个 stateful component 注入（`torchtitan/trainer.py:564`、`torchtitan/trainer.py:566`）。恢复测试跨过至少一次 repeat boundary 后保存，再比较后续 input、position、label 完全相等（`tests/unit_tests/cpu/test_dataset_checkpointing.py:31`、`tests/unit_tests/cpu/test_dataset_checkpointing.py:42`、`tests/unit_tests/cpu/test_dataset_checkpointing.py:52`）；另一个测试直接覆盖 packing buffer round-trip（`tests/unit_tests/cpu/test_text_dataset_packing.py:89`、`tests/unit_tests/cpu/test_text_dataset_packing.py:96`、`tests/unit_tests/cpu/test_text_dataset_packing.py:101`）。

### ④ 约束/边界

- 当前 checkpoint 不允许改变 effective DP degree 后继续同一数据游标；loader 在恢复入口硬失败（`torchtitan/components/data/loader.py:163`）。弹性扩缩容若要保持样本不重不漏，需要新的全局 ownership 迁移协议。
- 数据图可精确恢复不等于整个训练 bitwise exact。提交 `1b04fc1c3` 的结果显示 text/SFT 精确，MM/Flux 的剩余差异来自数据路径外或 accelerator RNG/非确定性 kernel。
- iterator state 与 recipe 结构耦合；源码仅有 schema version=1，没有跨 recipe 变更迁移。
- prefetch 状态由 Grain iterator graph 管理；绕过 `GrainDataLoader.state_dict()` 自行保存 source offset 会漏掉已预取 batch。

### ⑤ 发展趋势（有源码锚点的推断）

当前 TODO 明确指向两条压力：finite DP dataset 需要全局 remainder/exhaustion policy，CPU-heavy processing 需要把多进程 worker boundary 前移（`torchtitan/components/data/loader.py:88`、`torchtitan/components/data/loader.py:124`）。可以推断数据层会继续围绕“全局协调而非单 rank 局部便利”演进，但源码没有给出实现或时间表。

---

## 6. Trainer 边界：为什么统计的是 token slot 与 global valid token，而不是 batch 数

### ① 背景/问题

固定 token budget 只保证每次 forward 的槽位数相同，不保证有效监督 token 数相同；SFT prompt 和 padding 都用 `IGNORE_INDEX`，不同 DP rank、PP microbatch、gradient-accumulation group 的有效 token 会失衡。若仍按 local batch 或 local mean 归一化，某些 token 会被赋予更高梯度权重。

### ② 为什么这么设计

**选中的路线**是数据层产出固定 `[T]` slot，Trainer 在开始任何 forward/backward 前收齐本 optimizer step 的全部 microbatch，统计有效 label 后跨 batch/DP mesh 求和；**替代方案**是每 microbatch/每 rank 独立 mean。提交 `0cb743558` 的 Motivation 已证明后者会在有效 token 不均衡时给出错误权重，选择全局 token normalization 的准则是“每个有效 token 等权”。

### ③ 实现思路与细节

Trainer 从配置推导每 DP rank 一次 iteration 的 token 数，再用全局 train-step token budget 推导 gradient accumulation steps（`torchtitan/trainer.py:409`、`torchtitan/trainer.py:413`、`torchtitan/trainer.py:423`）。`batch_generator()` 保持 tensor 在 CPU，记录的是 labels.numel() token slots 而非有效 token（`torchtitan/trainer.py:639`、`torchtitan/trainer.py:658`、`torchtitan/trainer.py:665`）。

进入 train step 后，Trainer 先为每个 accumulation group 收齐所有 PP microbatch，并用 `labels != IGNORE_INDEX` 统计 local valid tokens（`torchtitan/trainer.py:785`、`torchtitan/trainer.py:788`、`torchtitan/trainer.py:793`）。DP 开启时在 `batch` mesh 上求和，且把计数留在 device，避免在训练主路引入 CPU sync（`torchtitan/trainer.py:798`、`torchtitan/trainer.py:800`、`torchtitan/trainer.py:802`）。随后才逐 group 搬运到 device 并进入模型预处理（`torchtitan/trainer.py:808`、`torchtitan/trainer.py:815`）。

### ④ 约束/边界

- `metrics_processor.ntokens_since_last_log` 统计 slot，loss normalization 统计 valid token；两个计数回答不同问题，不能混用。
- 如果 finite loader 在 gradient accumulation 中途耗尽，整个 optimizer step 不执行；`DataloaderExhaustedError` 特意不继承 `StopIteration`，避免 PEP 479 把 generator 内异常包装成不可按预期捕获的 `RuntimeError`（`torchtitan/components/data/loader.py:26`、`torchtitan/trainer.py:653`）。
- `repeat=False + DP>1` 当前被拒绝，因为各 rank 不同步耗尽会让 collectives hang（`torchtitan/components/data/loader.py:88`、`torchtitan/components/data/loader.py:92`）。这也是“有限数据集支持”仍不完整的直接失败边界。
- 模型额外输入、CP sharding 与 SPMD annotation 属于 `BaseModel.preprocess_inputs()`，不是 collator 的职责；协议要求每个标准 Trainer 模型自行实现，默认直接 `NotImplementedError`（`torchtitan/protocols/model.py:57`、`torchtitan/protocols/model.py:72`、`torchtitan/protocols/model.py:77`）。

---

## 7. 审计结论：已覆盖能力与仍开放缺口

| 结论 | 当前证据 | 不能外推成什么 |
|---|---|---|
| text/SFT/MM/Flux 共用 Grain composition/checkpoint contract | 提交 `1b04fc1c3`；`GrainDataLoader` | 所有 workload 使用相同 sample shape |
| map 与 stream 都可续训 | `tests/unit_tests/cpu/test_dataset_checkpointing.py:33` | 整个训练一定 bitwise exact |
| `[T]` 是语言模型主批形态 | 提交 `73aed7f6c`；`TextCollator` | Flux 也取消独立 batch 维 |
| mix 权重由 child 输出层次决定 | `torchtitan/components/data/dataset.py:207` | 当前已实现 token-weighted mix |
| packing buffer 在 checkpoint 中恢复 | `tests/unit_tests/cpu/test_text_dataset_packing.py:89` | 可改变 DP degree/recipe 后继续旧 cursor |
| global valid-token normalization 在 Trainer 完成 | `torchtitan/trainer.py:785` | collator 应输出全局归一化 loss |

尚未闭合的源码锚点包括：JSONL sidecar index、多进程 CPU transform、有限 DP exhaustion、统一 overflow policy、token-weighted mix、多轮 SFT、prompt prefix validation，以及改变 DP 拓扑后的数据 ownership 迁移。这些不是“文档未来建议”，而是当前 TODO/guard 暴露的真实边界。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/01_torchtitan_trainer_quickstart|TorchTitan Trainer Quickstart]] —— 数据图怎样进入 Trainer 初始化、训练步与 checkpoint 生命周期。
- [[02_engineering/02_train_frameworks/torchtitan/10_torchtitan_parallel_dims_analysis|ParallelDims 与双平面 Mesh]] —— effective DP rank、`batch` mesh 与 loss mesh 的所有权来源。
- [[02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis|TorchTitan CP]] —— 扁平 token 进入 device 后怎样按 CP 切分并构建 attention 输入。
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|TorchTitan SPMD Types]] —— 模型预处理怎样给输入添加 DP/CP/TP 逻辑布局。
- [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|TorchTitan PP]] —— token batch 怎样预切成 PP microbatch，并参与梯度累积。
- [[02_engineering/04_posttrain_frameworks/10_rl_ppo_loss_and_grpo_analysis|TitanRL 异步 RL]] —— 后训练场景的 rollout buffer/batcher 与预训练 Grain 数据图的不同状态边界。
