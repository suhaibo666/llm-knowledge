# Knowledge Base Changelog

All source ingestions and significant wiki updates are logged here.

---

## 2026-05-19: 分片 Muon 与双网格 HSDP 技术报告入库

**Type**: User Contribution（手动入库，分布式优化器技术分析报告）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/muon_sharded_hsdp_report.html`
  - **§1**: Muon 算法核心原理（Nesterov 动量 + Newton-Schulz 正交化、按 head/expert 粒度）
  - **§2**: 分片 Muon 的挑战与 all-to-all 解法（gather → N-S → scatter 流程、批量化同形状张量、通信异步化）
  - **§3**: 双网格 HSDP 设计（非专家窄网格 FSDP + 专家宽网格 EP、CP/EP 维度解耦）
  - **§4**: TP 场景覆盖情况分析（Column/Row-parallel 下 N-S 的正确性条件）
  - **§5**: 异步流水线 Gantt 图（顺序执行 vs 通信计算重叠，约 33% 耗时节省）
  - **§6**: 非专家权重分工 N-S 优化方案（消除 k 倍计算冗余，通信量约减半）
  - **§7**: 方案对比总结表（Cursor vs nanoGPT-speedrun vs 分工优化提案）
  - 含 4 幅 SVG 图表（all-to-all N-S 流程、双网格拓扑、异步流水线 Gantt、分工优化对比）

**交叉引用更新**:

- `02_train_frameworks/index.md` — 页面列表新增条目
- `01_theory/02_pretraining/muon_analysis.md` — Related Pages 新增回链

---

## 2026-05-19: 算子调优体系指南入库

**Type**: User Contribution（手动入库，算子开发与性能优化指南）

**入库文件**:

- `wiki/02_engineering/01_ai_frameworks/operator_optimization_guide.md`
  - **§1**: 算子编程体系概览（GPU: CUDA/CUTLASS/Triton/TileLang/TVM + NPU: AscendC/TBE/CANN）
  - **§2**: Roofline 性能分析模型（A100/H100/910B Ridge Point、Nsight/msprof Profiling 指标）
  - **§3**: GPU Memory Bound 优化（融合、算法变形、向量化访存）与 Compute Bound 优化（Tiling、软件流水线、Tensor Core）
  - **§4**: 融合算子识别与设计（决策矩阵、常见 Pattern、FX Graph 替换）
  - **§5**: 等价替换寻找方法（数学变形、算法层替换、AutoTuning）
  - **§6**: 昇腾 NPU 优化路径（Da Vinci 架构、AscendC 三段流水、GPU 经验适配表）
  - **§7**: 与 torch.compile 的关系（各框架接入方式、Custom Op 注册）
  - **§8**: GPU/NPU 完整优化工作流（Profile → Roofline → 优化 → 注册 → 验证）

**交叉引用更新**:

- `01_ai_frameworks/index.md` — 编译优化表格新增条目

---

## 2026-05-19: Pin Memory 与内存语义通信分析入库

**Type**: User Contribution（手动入库，综合深度分析）

**入库文件**:

- `wiki/02_engineering/pin_memory_and_memory_semantics_analysis.md`
  - **§1**: Pin Memory 与内存语义通信基础概念（DMA、RDMA Write/Read/Atomic、ibv_reg_mr）
  - **§2**: 传统消息语义 vs 内存语义的局限性分析（NCCL 固定成员、P2P 内存拷贝开销、NIC 厂商碎片化）
  - **§3**: Pin Memory 在 PyTorch DataLoader / DeepSpeed ZeRO-Offload / vLLM KV Offload 中的应用
  - **§4**: 内存语义通信在 vLLM P/D 分离 / Mooncake TransferEngine / DeepSeek DeepEP / 3FS / RLHF 权重同步中的应用
  - **§5**: 两种 Pin Memory 层次区分（CPU DRAM vs GPU HBM Registration）
  - **§6**: 社区应用全景总结与核心趋势

**交叉引用更新**:

- `02_engineering/index.md` — 页面列表新增条目

---

## 2026-05-17: Megatron-LM Pipeline Parallelism 分析报告

**Type**: New Page

**新增文件**:

- `wiki/02_engineering/02_train_frameworks/megatron_pp_parallelism_analysis.html`
  - **§1**: PP 进程组与拓扑 (`parallel_state.py`) — 4 类进程组、UCC/NCCL 双后端、辅助通信组
  - **§2**: 4 种调度策略总览 — Non-Interleaved 1F1B / Interleaved 1F1B (VPP) / Combined 1F1B / No-Pipelining
  - **§3**: Non-Interleaved 1F1B 详细执行流 — Warmup/Steady/Cooldown 三阶段公式、SVG 流水线时序图
  - **§4**: Interleaved 1F1B (VPP) — 微批量到 Chunk 的映射表、Bubble 验证与计算并行度提升
  - **§5**: P2P 通信原语 — `isend/irecv` vs `batch_isend_irecv`、交替组策略 (PP=2 优化)、`P2PCommunicator` 封装
  - **§6**: Combined 1F1B (EP 通算重叠) — `AbstractSchedulePlan` 层级别交错、AllGather/ReduceScatter 重叠
  - **§7**: Bubble 分析与通信量公式 — 气泡比推导、非交错 vs 交错对比、P2P 通信量公式
  - **§8**: 激活检查点与内存优化 — Partial AC、Deallocate Pipeline Outputs、Defer Embedding Wgrad
  - **§9**: 细粒度激活卸载 — `PipelineOffloadManager`、D2H/H2D 双流、`post_warmup_callback` 自适应调参
  - **§10**: 配置推荐与决策树 — 4 种典型场景配置速查表

---

## 2026-05-16: DeepSeek-V4 CP 分析报告准确性修正

**Type**: Correction（对已有文档进行准确性审核并修正错误）

**修正文件**:

- `wiki/01_theory/01_models/deepseek/deepseek_v4_cp_analysis.md`
  - **修正 1**：源文件路径错误 — `raw/05_model_families/deepseek/DeepSeek_V4.pdf` → `raw/01_theory/01_models/deepseek/DeepSeek_V4.pdf`
  - **修正 2**：Stage 1 Step 3 压缩输出数量错误 — `(c+1) 个 compressed entries` → `1（CSA 重叠窗口）或 2（HCA 无重叠）个 boundary compressed entries`（原公式与 2c tokens / ratio c 的数学不一致）
  - **修正 3**：Stage 2 All-Gather 输出长度公式错误 — `总长度 = P × c`（与 S 无关的常数，量级完全错误）→ `总长度 ≈ S/c，即 P × S/(P·c)`

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_context_parallel_analysis.html`
  - **修正 1（§5.2 callout）**：删除误导性"CSA 与 CP 的序列分片策略兼容"表述，改为明确说明当前代码是功能降级版（AllGather 缺失、fill_value 边界填充）
  - **修正 2（§6.3）**：`h_k = 1（MQA）` → `MLA 低秩潜变量压缩效果，通信量等效于 MQA`，避免将 MLA 误称为 MQA
  - **修正 3（§6.4）**：CSA CP 通信量公式标注为"论文设计目标，当前代码未完全实现"，补充实际代码行为（CSA 层跨 rank 压缩 KV 通信量为 0）
  - **修正 4（§5.5 Gap 4）**：P2P 数据量 `ratio × hidden_size` 改为按 Compressor 输入维度分情况讨论，加 ⚠️ 提示需确认实际维度
  - **修正 5（§9.1 特征 5）**："CSA 压缩与 CP 天然兼容" → "CSA 压缩与 CP 的兼容尚未完整实现"，如实反映当前代码状态

---

## 2026-05-15: 知识库目录结构重整

**Type**: Reorganization

**变更内容**:

- **新建 `mlir/` 子目录** (`wiki/02_engineering/01_ai_frameworks/mlir/`)
  - 从 `01_ai_frameworks/` 移入: `mlir_core_concepts.md`、`torch_mlir_pass_pipeline_analysis.md`、`triton_vs_mlir_backend_analysis.md`
  - 从 `inductor/` 移入: `npu_mlir_backend_deep_analysis.md`、`npu_mlir_pipeline_analysis.md`、`NPU_MLIR_Backend_Technical_Analysis.md`
- **模型论文归位**:
  - `Engram_Analysis.md` → `deepseek/`
  - `moba_analysis.md`、`kimi_linear_analysis.md` → `moonshot_kimi/`
- **跨域移动**:
  - `comm_compute_fusion_guide.md` → `02_train_frameworks/`
  - `mooncake_analysis.md` → `03_infer_frameworks/`
  - `batch_invariance_guide.md` → `04_posttrain_frameworks/`
- **索引更新**: 所有受影响的 `index.md` 已同步更新

## 2026-05-15: DeepSeek-V4 Tensor Parallel 分析重大修正（基于 Megatron-LM dev 源码）

**Type**: Correction / Rewrite（基于实际源码的全面重写，纠正此前推断性分析中的重大错误）

**修正文件**:

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_tensor_parallel_analysis.html`
  - **纠正 1**：DSv4 Hybrid Attention 实际强制 `TP size = 1`（`assert get_pg_size(self.pg_collection.tp) == 1`），此前错误推断为 Column+Row Parallel 切分
  - **纠正 2**：`q_down_proj` 为 `tp_group=None` + `parallel_mode="duplicated"`，不是 ColumnParallel
  - **纠正 3**：Compressor (`linear_wkv`, `linear_wgate`) 和 CSAIndexer (`linear_wq_b`, `linear_weights_proj`) 均为 `parallel_mode="duplicated"`，不产生 TP 通信
  - **纠正 4**：mHC 使用原生 `nn.Linear`（非 TP-sharded），依赖 `sequence_parallel` 属性进行梯度同步，不是 Column+Row Parallel
  - **纠正 5**：Routed Expert 的 fused `TEGroupedMLP` 不支持 `TP > 1`（`experts.py:328-329`），此前错误推断为 ETP 切分
  - **修正 6**：通信量分析全面重写——当前实现下 Attention、mHC、Routed Expert 的 TP 通信均为 0，主要跨 rank 通信仅剩 EP All-to-All 和 CP Ring-AG
  - 新增明确的源码引用（文件路径 + 行号）
  - 新增 "关键发现对比表" 和 "Future Work" 章节

**源码依据**:

- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:87-88, 172-195, 421-454`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py:288-309, 451-474`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/hyper_connection.py:150-151, 187-200`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/moe/experts.py:328-329`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/moe/shared_experts.py:112-159`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py:183-196`

---

## 2026-05-15: 新增"为什么 V4 选择 TP=1"架构分析章节

**Type**: Enhancement（在修正后的文档中新增深度分析章节）

**更新文件**:

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_tensor_parallel_analysis.html`
  - 新增章节 **"为什么 V4 选择 TP=1：架构与工程考量"**（位于"关键发现"与"Attention 层"之间）
  - 从 5 个维度系统分析 TP=1 的深层原因：
    1. 压缩操作的全局性（Compressor softmax 归约、Indexer Top-K 无法在 TP 边界分片）
    2. q_down_proj 的 duplicated 设计（需要完整 hidden_states 同时供给 q 和 kv 压缩路径）
    3. o_group_proj 的不可分性（Grouped LoRA einsum 需要完整 attention 输出）
    4. 延续 V3 的"弱化 TP，强化 EP+DP"设计哲学
    5. 通信开销与计算密度的权衡（长序列下 AG/RS 数据量远大于收益）
  - 新增 V3 vs V4 并行策略对比表
  - 阐明"接口预留、实现待定"的工程策略

---

## 2026-05-15: DeepSeek-V4 Context Parallelism 实现深度分析（基于 Megatron-LM dev 源码）

**Type**: Knowledge Synthesis（基于 Megatron-LM dev 分支 CP 实现源码，新建 HTML 深度分析）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_context_parallel_analysis.html`
  - 9 节深度分析：CP 进程组拓扑（含 Hierarchical CP）、4 种 CP 通信类型（p2p/all_gather/a2a/a2a+p2p）、Native CP 实现（AttentionFuncionWithContextParallel autograd.Function）、TransformerEngine CP 支持（cp_stream Ring Attention）、DSv4 CP 适配（RoPE cp_group、CSA p2p、Dynamic CP 限制）、通信量分析（MLA 使 CP 通信降低 ~128x）、Overlap 机制（TE P2P vs Native AllGather）、Dynamic CP 运行时机制、配置推荐
  - **新增 5.5 节**：CSA/HCA CP 论文设计与代码实现的 Gap 分析——基于 csa.py 源码审计，指出 `cp_comm_type` 参数未实际使用、`_overlap_transform` 跨 rank 依赖未解决（fill_value 填充边界）、压缩 KV 的 AllGather 缺失、P2P 与计算掩盖可行性分析
  - **新增 2.4 节**：四种 CP 方法的 QKV 交互图示——4 幅 SVG 详细展示 p2p（Q 固定，K/V 轮转）、all_gather（聚合完整 K/V）、a2a（All-to-All 交换序列/Head 维度）、a2a+p2p（分层 NVLink A2A + IB P2P）的 token 收发细节与计算数据布局
  - 含 9 幅 SVG 图表（新增 4 幅交互图示）

**源码依据**:

- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/dot_product_attention_context_parallel.py` — Native CP autograd.Function，AllGather/ReduceScatter 实现
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/extensions/transformer_engine.py` — TE CP 初始化，cp_stream，cp_comm_type 配置
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/parallel_state.py` — CP group 创建，Hierarchical CP `create_hierarchical_groups`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py` — DSv4 CP 集成，RoPE cp_group，Dynamic CP 不支持
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py` — CSA 默认 `cp_comm_type="p2p"`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/transformer_config.py` — CP 通信类型定义与校验
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/model_parallel_config.py` — Dynamic CP 配置

**交叉引用更新**:

- `02_train_frameworks/index.md` — 页面列表新增 deepseek_v4_context_parallel_analysis.html 条目

---

## 2026-05-14: DeepSeek-V4 Tensor Parallel 切分方案 HTML 深度分析

**Type**: Knowledge Synthesis（基于 Megatron-LM dev 分支实现 + V4 架构特性，新建 HTML 深度分析）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_tensor_parallel_analysis.html`
  - 8 节深度分析：V4 架构概览与 TP 必要性、CSA/HCA Attention 层 TP 列行并行策略、MoE ETP 切分（共享专家+路由专家）、mHC 流形约束超连接的切分特殊性、逐层通信量统一公式推导、TP Bulk vs Pipelined Overlap 掩盖方案、TP×EP×PP×CP 四维协同调度、配置决策树
  - 含 5 幅 SVG 图表：CSA TP 数据流、MoE Expert ETP 切分、Bulk Overlap 原理、四维并行通信组拓扑、TP 配置决策树

**交叉引用更新**:

- `02_train_frameworks/index.md` — 页面列表新增 deepseek_v4_tensor_parallel_analysis.html 条目

---

## 2026-05-14: 分布式优化器深度分析 HTML 入库

**Type**: Ingestion（HTML 深度分析文档入库，无需新建 .md）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/distributed_optimizer_deep_dive.html`
  - 7 节深度分析：ZeRO 分片体系通信量等价性、梯度累积对 ZeRO-1/2 的差异化影响 (K×P)、FSDP2/Megatron/MindSpeed 三方对比、MindSpeed param 临时化与 zero-copy、Adam vs Muon 优化器内存估算 (18→14 bytes/param)、Muon Newton-Schulz 对 ZeRO 切分的根本性挑战、选型决策树
  - 含 6 幅 SVG 图表：DDP vs ZeRO 通信量、梯度累积通信差异、Overlap 机制对比、MindSpeed 内存布局、Element-wise vs 矩阵运算、选型决策树

**交叉引用更新**:

- `megatron-lm/index.md` — Memory & Compute Optimization 节新增 HTML 文件条目
- `megatron-lm/megatron_distributed_optimizer_analysis.md` — Related Pages 新增链接
- `02_train_frameworks/index.md` — 页面列表新增条目

---

## 2026-05-13: torch_npu torch.compile 三条路径深度分析

**Type**: Knowledge Synthesis（基于 torch_npu 源码级分析，新建 4 页 + 更新 3 页）

**新建页面**:

- `wiki/02_engineering/01_ai_frameworks/inductor/npu_compile_paths_overview.md`
  - NPU torch.compile 路径总览：三条路径 (Triton/default、ACLGraph、MLIR) 全景对比
  - 与社区 (CUDA/XPU) 的核心差异：monkey-patching 策略、fallback 机制、调度器继承
  - 当前适配的收益：快速迭代、硬件特性直达、多路径冗余保障
  - 演进路线：v2.7.1 (35+ patches) → v2.9.0 (~10) → master (~8)，`_compat.inductor` 兼容层、条件化 patch 管理

- `wiki/02_engineering/01_ai_frameworks/inductor/npu_triton_backend_deep_analysis.md`
  - Triton/Inductor default 路径深度分析（Path 1）
  - Monkey Patch 五类分类：调度器重写、代码生成、wrapper 层、 lowering 规则、Triton 集成
  - `NPUCombinedScheduling` 继承 `CUDACombinedScheduling`，组合 CATLASS + Triton + NoLinearTriton 三种调度器
  - `golden_var_list` / `unified-axis` 逻辑：SIMD/SIMT 混合执行的统一轴选择机制
  - `NPUIndexTritonKernel` 特殊索引 kernel、35+ monkey patches 逐版本演进
  - `lowering_fallback_list.py`：859 aten ops + 135 prims ops 强制 fallback 到 ACLNN
  - 与社区逻辑对比：继承为主、局部重写，演进方向是减少侵入式 patch

- `wiki/02_engineering/01_ai_frameworks/cudagraphs/npugraphs/aclgraph_deep_analysis.md`
  - ACLGraph 深度分析（Path 2）
  - CANN `aclmdlRI*` API 图捕获/重放机制（`AclmdlRICaptureBegin`、`AclmdlRIExecuteAsync`）
  - `NpuGraphOpHandler` 插件框架：FA3 等特殊融合算子在图捕获期的参数预分配
  - Super Kernel (`AclskOptimize`)：CANN 特有的多 kernel 合并优化
  - `StaticKernelCompiler`：预热期预编译 ACLNN kernel，确保捕获确定性
  - 与社区 CUDAGraph 差异：ACLGraph 是 CANN 运行时原语，NPU Graphs 是 PyTorch 层封装
  - 演进方向：统一 NPU Graphs/ACLGraph 接口，向社区 `torch.cuda.CUDAGraph` API 对齐

- `wiki/02_engineering/01_ai_frameworks/inductor/npu_mlir_backend_deep_analysis.md`
  - MLIR 路径深度分析（Path 3）
  - `has_triton = False` 禁用 Triton，启用 MLIR codegen 路径
  - IR 回溯机制：patch `ir.Loops.create` 附加 `traced_graph` 元数据，实现 FX Graph 重建
  - Bisheng 编译器 (`bishengir-opt` + `bishengir-compile`)：华为私有编译器管线
  - Scheduler patch：修改融合规则适应 MLIR 路径需求
  - `auto_fallback` 模式：编译失败自动回退到 FX Graph，双通道容错
  - 与社区逻辑差异：MLIR 路径在社区不存在，是 NPU 特有方案
  - 演进方向：`torch_npu._compat.inductor` 兼容层、条件化 patch、社区 MLIR 接口标准化

**更新页面**:

- `wiki/02_engineering/01_ai_frameworks/inductor/index.md`
  - 新增 2 个深度分析页面条目（npu_triton_backend_deep_analysis、npu_mlir_backend_deep_analysis）

- `wiki/02_engineering/01_ai_frameworks/cudagraphs/npugraphs/index.md`
  - 新增 1 个深度分析页面条目（aclgraph_deep_analysis）

- `wiki/02_engineering/01_ai_frameworks/index.md`
  - 新增 4 个页面条目（npu_compile_paths_overview、npu_triton_backend_deep_analysis、npu_mlir_backend_deep_analysis、aclgraph_deep_analysis）
  - 更新知识空白（新增 3 项：monkey patch 演进追踪、CATLASS/CK 生态、IR 回溯通用性）

- `wiki/changelog.md`（本条目）

**知识来源**:
  - `torch_npu` 源码：`torch_npu/_inductor/`（monkey-patch、lowering、codegen/triton.py、codegen/wrapper.py）
  - `torch_npu` 源码：`torch_npu/utils/_graph_tree.py`（ACLGraph 捕获/重放）
  - `torch_npu` 源码：`torch_npu/dynamo/__init__.py`（backend 注册）
  - `torch_npu` 文档：`docs/torch_npu_compile_path_*.md`（三条路径原始分析报告）
  - PyTorch 社区源码：`torch/_inductor/`（CUDA 参考实现）

---

## 2026-05-12: DL 编译优化趋势与通算融合知识补充

**Type**: Knowledge Synthesis（基于社区分析 + DeepSeek V4 wiki 内容，新建 4 页 + 更新 3 页）

**新建页面**:

- `wiki/02_engineering/01_ai_frameworks/flex_attention_analysis.md`
  - FlexAttention（PyTorch 2.4+）范式：从 Pattern Matching（_sfdp_init）到语义驱动代码生成
  - BlockMask 机制：block 粒度稀疏结构编译时分析，FULL/PARTIAL/EMPTY 分类
  - score_mod：内联注意力权重修改（ALiBi、Soft-cap、Temperature）
  - 与 _sfdp_init 的详细对比，典型 LLM 模型映射表
  - 局限与未来方向（torch.export AOT、NPU 支持）

- `wiki/02_engineering/01_ai_frameworks/tilelang_analysis.md`
  - TileLang 的定位：填补图 Pass（太高层）和 Kernel（太低层）之间的 Gap
  - DeepSeek V4 mHC 融合 kernel：RMSNorm+Linear+Sinkhorn-Knopp 片上融合，读写量降 3×
  - Host Codegen：<1μs kernel launch overhead（vs Python wrapper 的 5-20μs）
  - Z3 SMT 求解器：整数约束的编译时自动验证
  - TileLang vs Triton DSL 对比，tile-level IR 生态（FlexAttention/Linalg Tiling/CuTe）
  - 对图 Pass 体系的影响：tile-level IR 作为图 Pass 与 Kernel 的解耦层

- `wiki/02_engineering/01_ai_frameworks/comm_compute_fusion_guide.md`
  - 通算融合四层次模型（手动→半自动→框架感知→全自动）
  - **WaveEP（DeepSeek V4）**：wave-based 细粒度 EP 调度原理、CUDA Stream 架构、wave 粒度权衡
  - 实测性能：一般推理 1.50-1.73×，RL rollout 高达 1.96×
  - **DeepEP**：fine-grained SM control，FusedDispatch（Permute+A2A+Unpermute），HybridEP（NVLink/IB 异构）
  - TP/PP/CP/DP 各维度通算重叠机制（Pipelined AG、DualPipe、Ring Attention 双缓冲）
  - MLIR Mesh Dialect 的通算 IR 作用：async token + chunk-level 依赖
  - WaveEP 编译化路径：wave IR 表示 → Cost Model → TileLang 绑定 → DTensor 集成

- `wiki/02_engineering/01_ai_frameworks/mindspore_compiler_analysis.md`
  - MindSpore 编译流水线：ANF 图 → MindCompiler → AKG → CANN 后端
  - ANF（Administrative Normal Form）IR：函数式表示，高阶函数支持，vs FX Graph 对比
  - MindCompiler Pass：代数化简、常量折叠、算子融合白名单匹配
  - AKG Polyhedral 自动 Kernel 生成：loop tiling/vectorization/fusion，昇腾 NPU 特化
  - **ParallelAuto**：DP 递归规划自动并行策略搜索，vs Alpa 对比
  - MindSpore 2.x 动静统一（@jit 装饰器）
  - 与 PyTorch Inductor 的 Pass 体系逐类对比，优劣评价

**更新页面**:

- `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md`（新增"补充"章节）
  - **MLIR Mesh Dialect**：通信作为 IR 一等公民，async token + chunk 依赖分析，对 WaveEP 编译化的意义
  - **IREE**：Flow/Stream/HAL Dialect 三层架构，与 torch-mlir 的组合使用，vs torch.compile 对比
  - **StableHLO**：跨框架稳定 IR 锚点，通信算子标准化，GSPMD 自动并行的 IR 基础
  - **Triton 3.x MLIR 迁移**：Triton Dialect + TritonGPU Dialect，H100 TMA 异步 copy，与 Linalg Pass 的潜在集成

- `wiki/02_engineering/01_ai_frameworks/index.md`
  - 新增 3 个优化页面条目（flex_attention、tilelang、comm_compute_fusion）
  - 新增 1 个架构页面条目（mindspore_compiler）
  - 更新 mlir_core_concepts 描述（含新增内容）
  - 更新知识空白（新增 3 项）

- `wiki/changelog.md`（本条目）

**知识来源**:
  - PyTorch 官方文档（FlexAttention, DTensor, torch.export）
  - `wiki/01_theory/01_models/deepseek/deepseek_v4_analysis.md`（WaveEP、TileLang、DeepEP）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/moe_training_optimization_report.md`（DeepEP/HybridEP）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_comm_overlap_analysis.md`（多维通算重叠）
  - MLIR 官方文档（Mesh Dialect RFC，IREE，StableHLO）
  - Triton GitHub（triton-lang/triton MLIR 迁移 PR）
  - MindSpore 官方文档（ParallelAuto，AKG，ANF IR）

---

## 2026-05-12: Megatron-LM MoE 训练优化技术全景分析

**Type**: Knowledge Synthesis + Research（源码级分析, 新建 3 个 Wiki 页面）

- **Created**:
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_distributed_optimizer_analysis.md` — 分布式优化器深度分析（ZeRO-1/2 分片机制, Reduce-Scatter/All-Gather 通信, FP8/FP4 量化参数, CPU Offloading 双模式）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_memory_optimization_analysis.md` — 显存优化全景分析（NCCL Pool, MoE Paged Stash 三级溢出, Fine-Grained Activation Offloading, Buffer 复用, FP8/FP4 精度, Resharding）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_fusion_operators_analysis.md` — 融合算子优化分析（Bias+Activation 融合 6 种, Fused LayerNorm/Softmax, MoE 专用融合 4 种, Communication Fusion, FP8 Input Store, Triton/CUTLASS/cuTile kernel 层次）
- **Updated**: `megatron-lm/index.md` — 新增 "Memory & Compute Optimization" 章节, 更新 Knowledge Gaps 和 Cross-Domain Links
- **Supplemented**: FSDP2 适配分析 — Megatron 三种梯度/参数分片方案对比（`DistributedOptimizer` vs `TorchFullyShardedDataParallel` vs `MegatronFSDP`），FSDP Unit 机制, ZeRO 分片谱系, NCCL UserBuffer 优化, Delayed Wgrad Overlap, 与 EP/Activation Checkpointing/CUDA Graph 的协同
- **Supplemented**: CP 源码分析 — Ring Attention with AllGather pipeline, zigzag mask conversion, KV buffer double buffering, CP 正反向通信量公式
- **Supplemented**: 通信量/通信组全面分析（正反向） — 6 个并行维度（TP/PP/EP/CP/DP/DistOpt）的通信组层级关系图、通信原语映射、正反向通信量公式推导、通信时序图、Bucket 粒度与 NCCL 带宽、统一通信量总览表、671B MoE 典型通信量排序
- **Removed**: `Megatron-LM_Distributed_Parallel_Exam.md` — 内容分发至各分析页面（SP/TP 边界、TP autograd Function、Dynamic CP、MoE Router/Folding、FSDP+TP/EP 拓扑、Layer-Wise Optimizer、Grouped GEMM、通信组具体示例）
- **Updated**: `megatron-lm/index.md` — 移除已删除 exam 页面引用
- **Design doc**: `docs/superpowers/specs/2026-05-12-moe-training-optimization-report-design.md`

**Key sources analyzed**:
  - `megatron/core/optimizer/distrib_optimizer.py` — 分布式优化器主类 (~2800 lines)
  - `megatron/core/fusions/` — 13 个融合算子文件（@jit_fuser, Triton, CUTLASS/cuTile）
  - `megatron/core/nccl_allocator.py`, `moe/paged_stash.py`, `fine_grained_activation_offload.py` — 显存优化
  - `megatron/core/transformer/dot_product_attention_context_parallel.py` — CP Ring Attention
  - `megatron/core/transformer/moe/fused_a2a.py` — DeepEP/HybridEP 通信融合
  - `megatron/core/distributed/param_and_grad_buffer.py` — 参数/梯度 Buffer 管理

---

## 2026-05-11: torch.compile Dynamic Shape 全链路技术分析

**Type**: Knowledge Synthesis（PyTorch 主分支源码级调研）

- **Created**: `wiki/02_engineering/01_ai_frameworks/inductor/dynamic_shapes_full_analysis.md` — Dynamic Shape 全链路分析（中文）
- **Updated**: `wiki/02_engineering/01_ai_frameworks/inductor/index.md` — 编译阶段表格新增条目

**Key topics**:
  - **Why static-only**: Guard system bakes concrete integers → every shape change triggers recompilation
  - **ShapeEnv architecture**: `_init()` core data structures (`var_to_range`, `replacements`, `divisible`, `deferred_runtime_asserts`), backpropagation of constraints
  - **DimDynamic**: DYNAMIC/DUCK/STATIC/UNBACKED/INFER_STRIDE policies, how `mark_dynamic()` and `assume_static_by_default` control symbol allocation
  - **Guard system**: `_maybe_guard_rel()` → equality replacement + range refinement, three-layer guard architecture (ShapeEnv → GuardBuilder.SHAPE_ENV → runtime asserts)
  - **Correctness guarantees**: `assert_size_stride()` runtime validation, `exclusion_constraints` for automatic_dynamic recompilation
  - **SymInt/SymNode**: Python-level symbolic integer wrapping sympy.Expr, transparent tracking of all shape arithmetic
  - **automatic_dynamic_shapes**: Progressive dynamism — static first, recompile with dynamic on wobble, exclusion guards preserve static cache

- Cross-referenced with `[[inductor_codegen_dynamic_shape_analysis]]`, `[[torch_compile_architecture]]`, `[[PyTorch_Dynamo_Technical_Analysis]]`

---

## 2026-05-11: PyTorch Inductor 端到端编译管线源码分析

**Type**: Knowledge Synthesis（PyTorch 主分支源码级调研）

- **Created**: `wiki/02_engineering/01_ai_frameworks/inductor/inductor_compiler_pipeline_analysis.md` — PyTorch Inductor 后端编译流程深度分析（中文）
- **Updated**: `wiki/02_engineering/01_ai_frameworks/inductor/index.md` — 架构与流程表格新增条目
- **Cross-referenced**: 新页面与现有 10 个分阶段分析页面建立双向链接（`[[aotautograd_analysis]]`, `[[pre_grad_passes_guide]]`, `[[joint_graph_passes_guide]]`, `[[post_grad_passes_guide]]`, `[[lowering_analysis]]`, `[[scheduler_analysis]]`, `[[inductor_codegen_analysis]]`, `[[PyTorch_Dynamo_Technical_Analysis]]`, `[[PyTorch_Inductor_Technical_Analysis]]`, `[[torch_compile_architecture]]`）

**Key topics**:
  - **§1 Dynamo**: PEP 523 帧拦截、符号化执行字节码（`InstructionTranslator`）、VariableTracker 体系、Guards 机制（C++ `RootGuardManager`）、Graph Break 处理
  - **§2 AOT Autograd**: 前向/反向追踪、Joint Graph、Functionalization、Min-cut 分区算法、激活值保存 vs 重新计算权衡
  - **§3 Decomposition**: Core ATen + Inductor 分解表、条件化分解（形状/设备/类型）
  - **§4 FX Passes**: Pre-grad（normalization、group_batch_fusion、fuse_fx、efficient_conv_bn_eval）/ Joint-graph（constant_fold_uniform_value、remove_no_ops、pattern matching、replace_random）/ Post-grad（reorder_for_locality、mkldnn_fusion、b2b_gemm、micro_pipeline_tp、collectives bucketing、reinplace）— 逐 pass 源码级分析
  - **§5 Lowering**: `lowerings` 字典、`TensorBox/StorageBox`、IR 原语（Pointwise/Reduction/Scan/TemplateBuffer）、`register_lowering` 装饰器
  - **§6 Scheduler**: 依赖分析（`compute_dependencies`）、融合算法（`fuse_nodes`/`can_fuse`/`can_fuse_vertical`）、Combo Kernel、图分区（CUDAGraph）
  - **§7 CodeGen**: Triton/C++ Kernel 生成、Tiling 策略、Autotuning 子进程（`TuningProcessPool`）、AOTI C++ Wrapper、两层架构（Kernel + Wrapper）
  - **§8 设计哲学**: 分层解耦、函数化→优化→inplace、融合优先、延迟决策

- 重写 `wiki/02_engineering/01_ai_frameworks/torch_mlir_pass_pipeline_analysis.md`
  — **核心修正**: torch-mlir 可以通过 `torch.compile(model, backend=custom_mlir_backend)` 的自定义 backend 方式使用，入口是 `stateless_fx_import(gm)`——它直接接收 Dynamo 捕获的 `torch.fx.GraphModule`，不需要 `torch.export`。
  — 三条路径: A(`torch.compile`→Inductor→Triton，不走MLIR) / B(`torch.compile`+torch-mlir backend，走MLIR，本文) / C(`torch.compile`→NPU MLIR，monkey-patch)
  — 文档主体: Layer 0 `stateless_fx_import(gm)` → Layer 1 `torchdynamo-export-to-torch-backend-pipeline` (4-7 Pass) → Layer 2 `torch-backend-to-linalg-on-tensors-backend-pipeline` (18 Pass) → Layer 3 Linalg→GPU (上游 MLIR 概述)
- 更新 `wiki/02_engineering/01_ai_frameworks/inductor/npu_mlir_pipeline_analysis.md`
  — 新增 "NPU Codegen 内部的 MLIR Pass 分解" 小节，详细列出 Stage 6a→6e 的五个子阶段及每个子阶段内部的 Pass 序列
  — 补充 `torch-lower-to-backend-contract` 在 NPU 场景中的具体 Pass 序列及每个 Pass 的作用

- 重写 `wiki/02_engineering/01_ai_frameworks/torch_mlir_pass_pipeline_analysis.md`
  — **根本性重写**: 不再分析 torch-mlir 独立路径 (fx.py/export_and_import)，而是追踪 `torch.compile` → Inductor → NPU MLIR 的实际代码路径。
  — 六阶段流水线: Stage 0 Dynamo 图捕获 → Stage 1 FX Graph 预处理 (npu_optimize_fx_graph, parallel_scheduler_pass) → Stage 2 AOT Autograd (wrap_compiler 注入) → Stage 3 Decomposition (NPU 选择性禁用) → Stage 4 Inductor Lowering (TracedGraph 三层耦合) → Stage 5 Scheduler 融合 (NPU 修改规则) → Stage 6 NPU MLIR Codegen (5 子阶段: FX 重建→FxImporter→LowerToBackendContract→Bisheng 降级→毕昇编译)
  — 基于 `torch_npu` 源码: `npu_inductor_plugin.py`、`codegen/mlir.py`、`inductor_patch/lowering.py`、`inductor_patch/ir.py`、`utils.py`、`torch_mlir_patch.py`、`mlir_compiler.py`
  — 35+ Pass 总结表，标注每个 Pass 的 IR 层级、实现语言、核心作用、是否为 NPU 特有
  — 核心设计权衡: Python 前端承担编译责任、TracedGraph "夹带私货"代价、双编译器分工、Fallback 双通道

## 2026-05-11: torch.compile → MLIR 完整 Pass 管线分析 (基于源码追踪重写) [已被上述版本取代]

- 重写 `wiki/02_engineering/01_ai_frameworks/torch_mlir_pass_pipeline_analysis.md`
  — **核心修正**: 基于 `torch.compile` → MLIR 的实际 Python → C++ 调用链完整追踪。从 `fx.py:export_and_import()` → `_module_lowering()` → `lower_mlir_module()` 逐函数追踪，确定实际执行的两级 MLIR pipeline:
  — **Stage 1** (`torchdynamo-export-to-torch-backend-pipeline`): `torch-match-quantized-custom-ops` → `Inliner` → `ReduceOpVariants` → `Canonicalizer` → [可选 Decompose→Recompose→Canonicalizer]，共 4-7 Pass
  — **Stage 2** (`torch-backend-to-linalg-on-tensors-backend-pipeline`): `RestructureNonConstantAxes` → `FuseQuantizedOps` → `ConvertTorchToTMTensor` → `Canonicalizer` → `ConvertTorchToLinalg`(9 组 pattern) → `Canonicalizer` → `ConvertTorchToSCF` → `ConvertTorchToArith` → `ConvertTorchToTensor` → `ConvertTorchConversionToMLProgram` → `memref::ExpandOps` → `Canonicalizer` → `memref::ResolveShapedTypeResultDims` → `CSE` → `FuncBackendTypeConversion` → `Canonicalizer` → `FinalizingBackendTypeConversion` → `VerifyLinalgOnTensorsBackendContract`，共 18 Pass
  — 基于源码: `python/torch_mlir/fx.py`、`python/torch_mlir/compiler_utils.py`、`lib/Dialect/Torch/Transforms/Passes.cpp`、`lib/Dialect/TorchConversion/Transforms/Passes.cpp`、`lib/Conversion/TorchToLinalg/TorchToLinalg.cpp`
  — 每次 Canonicalizer (共 5 次) 标注了其消除的特定碎屑类型
  — 文档结构: §1 Dynamo Export 管线 6 个 Pass 三维分析 (Inliner→ReduceOpVariants→Canonicalizer→[Decompose→Recompose→Canonicalizer]) + ConvertTorchToLinalg 概述；§2 TorchScript 管线完整执行顺序；§3 架构转变分析 "前端承担编译责任" (TorchScript 2019 vs Dynamo Export 2023 哲学对比表)；§4 两条管线的共享组件 (ReduceOpVariants / DecomposeComplexOps / Canonicalizer / satisfiesBackendContract)；§5 LowerToBackendContract 迭代引擎深度分析；§6 设计方案总结对比表；§7 与 Triton 对比。
  — 基于 `Passes.cpp` 中 `createTorchDynamoExportToTorchBackendPipeline` 和 `createTorchScriptModuleToTorchBackendPipeline` 的精确源码，阐明两条管线的 18 个 Pass 差异及其根本原因。
- 更新 `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md` — Related Pages 新增交叉引用
- 更新 `wiki/02_engineering/01_ai_frameworks/index.md` — 编译架构页面列表新增条目

## 2026-05-11: MLIR Pass 设计哲学补充 + torch-mlir Pass 源码实例

- 更新 `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md`
  — 新增 §4.1 四种 Pass 作用域的设计哲学（安全性/可组合性/并行调度/测试调试）、"为什么不像 Triton 做全局优化"分析、与 Eager Mode 概念对应表；新增 §4.2 上游 MLIR ElementwiseOpFusion 源码解析（`areElementwiseOpsFusable`、`fuseElementwiseOps`、融合前后 IR 对比、与 Triton 融合检查项一一对应）；新增 torch-mlir FuseQuantizedOps 实例（Dialect 级 Pass，量化链融合）
- 更新 `wiki/02_engineering/01_ai_frameworks/triton_vs_mlir_backend_analysis.md`
  — 新增社区活跃度章节（`llvm/torch-mlir` main 分支每日活跃，SHARK-Turbine 已迁移）


- 新建 `wiki/02_engineering/01_ai_frameworks/inductor/npu_mlir_pipeline_analysis.md`
  — NPU MLIR 六阶段适配全景 (Dynamo→AOT→Decomp→Lowering→Scheduler→Codegen)，GPU Triton vs NPU MLIR 逐阶段对比。"改了什么、为什么在这一层、怎么改的"。
  核心内容: 三层 Pass 架构 (FX/Inductor/毕昇)、15 个 Monkey Patch 五组分类、编译模式状态机、Fallback 双通道、Autotune 60 配置
- 重写 `wiki/02_engineering/01_ai_frameworks/inductor/npu_compile.md`（原为 10 行存根）
  — 完整 NPU 编译工作流: 三种编译模式 (auto_fallback/default/complete_fallback)、毕昇编译器接口 (-enable-hfusion-compile 等)、60 维 Autotune、在线精度对比 (ANIR_ONLINE_ACC_COMP)、芯片感知 (910B1/310B1/910_9391)
- 更新 `inductor/index.md`、`01_ai_frameworks/index.md`、`NPU_MLIR_Backend_Technical_Analysis.md`、`npu_lowering_guide.md` 交叉引用

## 2026-05-08: 知识库目录结构重构

**Type**: Infrastructure — 从旧编号体系迁移至 Theory/Engineering 双层结构

### 新结构

```
raw/ & wiki/ 镜像
├── 01_theory/           # 理论研究 (原 llm/ 域 + 模型家族)
│   ├── 01_models/       # 模型架构 + 模型家族 (原 01_architecture + 05_model_families + 07_multimodal)
│   ├── 02_pretraining/  # 预训练技术 (原 02_training)
│   ├── 03_sft/          # SFT + 低参微调 (新建，预留)
│   ├── 04_posttraining/ # 后训练对齐 (原 03_alignment)
│   └── 05_inference/    # 推理技术 (原 04_reasoning + 08_agents)
└── 02_engineering/      # 工程实现 (原 torch_compile/ + 06_infra + 10/11)
    ├── 01_ai_frameworks/    # AI框架 (原 torch_compile/)
    ├── 02_train_frameworks/ # 训练框架 (原 06_infra + 10_train_framework)
    ├── 03_infer_frameworks/ # 推理框架 (原 11_infer_framework)
    └── 04_posttrain_frameworks/ # 后训练框架 (新建，预留)
```

### 变更内容

- 迁移 ~99 raw PDFs + ~102 wiki 页面至新结构
- 80 个文件中的 `[[wiki links]]` 路径批量更新（Python 脚本）
- 新建 5 个 index.md；重写 wiki/index.md 和 7 个领域 index
- 更新 CLAUDE.md、README.md
- 旧编号体系 (01-11) 完全废弃

## 2026-05-09: Triton vs Torch-MLIR 编译后端对比 + MLIR 基础概念

- 新建 `wiki/02_engineering/01_ai_frameworks/triton_vs_mlir_backend_analysis.md`
  — Triton 与 Torch-MLIR 在 Dynamo→AOT Eager→Decomposition→Lowering→Scheduler→Codegen 六个阶段的概念级对等映射表和优劣势分析
- 新建 `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md`
  — MLIR 三核心机制: Dialect 词汇表、Pass 变换引擎、IR 注册链路 (TableGen→C++→MLIRContext)，含递降完整示例
- 更新 `wiki/02_engineering/01_ai_frameworks/index.md`、`inductor/index.md` 和 `NPU_MLIR_Backend_Technical_Analysis.md` 的交叉引用

## 2026-05-08: 训练/推理框架目录页创建

- 新建 `wiki/llm/10_train_framework/index.md`（对应 `raw/10_train_framework/`：megatron.eddx, mindformers.eddx）
- 新建 `wiki/llm/11_infer_framework/index.md`（对应 `raw/11_infer_framework/`，当前为空）
- 更新 `wiki/llm/index.md`、`wiki/index.md`、`CLAUDE.md` 目录结构

## 2026-05-06: GLM/GLM-5 技术路线摄入

**Type**: Source Ingestion (GLM Series)

### 下载的新 Raw 文件

- `raw/05_model_families/zhipu_glm/GLM-5_Vibe_Coding_to_Agentic_Engineering-2602.15763.pdf`

### 创建的 Wiki 页面

- **Created**: `wiki/llm/05_model_families/zhipu_glm/glm_5_analysis.md` — GLM-5 Vibe Coding 到 Agentic Engineering（中文）
- **Created**: `wiki/llm/05_model_families/zhipu_glm/glm_5v_turbo_analysis.md` — GLM-5V-Turbo 原生多模态 Agent（中文）
- **Created**: `wiki/llm/05_model_families/zhipu_glm/index.md` — GLM 技术路线总览

**Key topics (glm_5_analysis)**:
  - 744B/40B MoE (256 专家，8 激活)，80 层
  - Muon Split: per-head 独立正交化，MLA 匹配 GQA-8 性能
  - MLA-256: head dim 192→256，头数减少 1/3，解码计算降低
  - MTP 参数共享 (3 层)，Accept Length 2.76
  - DSA 稀疏注意力：20B tokens 适配，计算减少 1.5-2×，无损
  - 28.5T tokens 预训练，200K 上下文 mid-training
  - 异步 RL 基础设施：TITO gateway + Direct Double-sided Importance Sampling
  - Reasoning RL: GRPO + IcePop，训练-推理不匹配缓解
  - Agentic RL: 10K+ SWE + Terminal + Search 环境
  - 国产 GPU 全栈适配 (7 大平台)
  - SWE-bench ~65, τ²-Bench ~60, HLE ~30

**Key topics (glm_5v_turbo_analysis)**:
  - CogViT 视觉编码器：两阶段预训练 (蒸馏 MIM + 对比图文)
  - NaFlex 可变分辨率，64K batch, 80 亿中英图文对
  - MMTP 多模态 MTP：`<|image|>` 共享 token 方案
  - 30+ 任务联合 RL：感知/推理/Agent 全面提升
  - 大规模多模态 RL 基础设施：四维重新设计
  - ImageMining 基准：30.7 分
  - Design2Code 94.8, BrowseComp-VL 51.9, OSWorld 62.3
  - 纯文本编码能力保持 (CC-Backend 22.8, CC-Frontend 68.4)

---

## 2026-05-06: Kimi K2 & K2.5 技术路线摄入

**Type**: Source Ingestion (Kimi K2/K2.5)

### 创建的 Wiki 页面

- **Created**: `wiki/llm/05_model_families/moonshot_kimi/kimi_k2_analysis.md` — Kimi K2 开放 Agent 智能（中文）
- **Created**: `wiki/llm/05_model_families/moonshot_kimi/kimi_k2.5_analysis.md` — Kimi K2.5 视觉 Agent 智能（中文）
- **Updated**: `wiki/llm/05_model_families/moonshot_kimi/index.md` — 论文索引更新，K2/K2.5 标记为已摄入

**Key topics (kimi_k2_analysis)**:
  - 1.04T/32.6B MoE，384 专家 (sparsity=48)，64 注意力头
  - MuonClip 优化器：QK-Clip 解决 logits 爆炸，15.5T token 零 loss spike
  - 稀疏度扩展定律：sparsity 48 vs 8 节省 1.69× FLOPs
  - 大规模 Agentic 数据合成：23,000+ 工具，模拟+真实沙盒
  - RL 框架：RLVR + 自批评 Rubric 奖励，覆盖可验证和主观任务
  - SWE-bench 65.8、τ²-Bench 66.1、AIME 2024 69.6
  - Agent 能力超越 Claude Opus 4 和 GPT-4.1

**Key topics (kimi_k2.5_analysis)**:
  - MoonViT-3D 视觉编码器：原生分辨率，3D 时空编码，4× 时间压缩
  - 早期融合 + 低视觉比例 (10%:90%) 优于晚期融合
  - Zero-Vision SFT：仅用文本 SFT 激活视觉能力
  - 联合多模态 RL：视觉 RL 提升文本性能 (MMLU-Pro +1.7%)
  - Agent Swarm：可训练编排器 + 冻结子智能体，BrowseComp 60.6%→78.4%
  - Toggle 算法：token 减少 25-30%，性能影响可忽略
  - DEP 训练基础设施：多模态训练效率达纯文本 90%
  - LVBench 75.9%、OCRBench 92.3%、BrowseComp 78.4%

---

## 2026-05-06: Kimi/Moonshot AI 技术路线批量摄入 (4 篇核心论文)

**Type**: Source Ingestion (Kimi 技术路线)

### 下载的新 Raw 文件

- `raw/05_model_families/moonshot_kimi/Kimi_k1.5_Scaling_RL-2501.12599.pdf`
- `raw/05_model_families/moonshot_kimi/Mooncake_KVCache_Disaggregated-2407.00079.pdf`
- `raw/05_model_families/moonshot_kimi/MoBA_Mixture_of_Block_Attention-2502.13189.pdf`
- `raw/05_model_families/moonshot_kimi/Kimi_Linear_Attention-2510.26692.pdf`

### 创建的 Wiki 页面

- **Created**: `wiki/llm/06_infra/mooncake_analysis.md` — Mooncake KVCache 中心化分离式服务架构（中文）
- **Created**: `wiki/llm/01_architecture/moba_analysis.md` — MoBA 混合块注意力机制（中文）
- **Created**: `wiki/llm/01_architecture/kimi_linear_analysis.md` — Kimi Linear/KDA 线性注意力架构（中文）
- **Created**: `wiki/llm/03_alignment/kimi_k1.5_analysis.md` — Kimi k1.5 RL 缩放定律（中文）
- **Created**: `wiki/llm/05_model_families/moonshot_kimi/index.md` — Kimi 技术路线总览

**Key topics (mooncake_analysis)**:
  - Prefill/Decode/KVCache 三池分离架构
  - Chunked Pipeline Parallelism (CPP) 替代跨节点 SP
  - Layer-wise Prefill：KVCache 传输与计算重叠
  - 缓存感知全局调度 + 热点块迁移
  - 预测性早期拒绝解决负载波动
  - 真实负载吞吐量提升 75%，模拟场景 525%

**Key topics (moba_analysis)**:
  - 将 MoE 原理应用于注意力机制
  - Query 动态路由到 KV Block (top-k 选择)
  - 块路由：mean_pool(K) 亲和度 + 因果掩码
  - MoBA/Full 混合预训练 (90%/10%)
  - 1M 序列 6.5x 加速，10M 序列 16x 加速
  - 已部署支持 Kimi 长上下文请求

**Key topics (kimi_linear_analysis)**:
  - KDA: Kimi Delta Attention (通道级细粒度遗忘门)
  - 约束 DPLR 结构，消除数值不稳定，Kernel 速度 ~2x
  - 3:1 KDA-MLA 混合架构，MLA 层使用 NoPE
  - KV Cache 减少 75%，1M 解码 6x 加速
  - 在预训练/SFT/长上下文/RL 场景下均超越全注意力
  - 开源 KDA Kernel + vLLM 集成 + Checkpoints

**Key topics (kimi_k1.5_analysis)**:
  - 在线镜像下降变体 (类似 GRPO，理论来源不同)
  - 128K 上下文 RL 训练，上下文长度是关键扩展维度
  - Partial Rollout + 混合部署 (Megatron ↔ vLLM via Mooncake)
  - Long2Short 蒸馏 (模型合并/拒绝采样/DPO/RL)
  - 长度惩罚渐进式引入，防止过度思考
  - AIME 77.5、MATH-500 96.2、Codeforces 94th percentile

---

## 2026-05-06: 低精度训练与 Transformer Engine 知识整合

**Type**: Knowledge Synthesis（Megatron-LM 源码 + TE GitHub 仓库 + DeepSeek-V4 FP4 QAT）

- **Created**: `wiki/llm/02_training/low_precision_training_analysis.md` — Megatron 低精度训练全栈分析（中文）
- **Created**: `wiki/llm/02_training/transformer_engine_analysis.md` — NVIDIA Transformer Engine 技术分析（中文）
- **Updated**: `wiki/llm/index.md` — Optimizers & Training Algorithms 表格新增 3 条目
- **Updated**: `wiki/llm/06_infra/megatron-lm/index.md` — Knowledge Gaps 更新（TE 集成、低精度训练标记为已解决），Cross-Domain Links 扩展

**Key topics (low_precision_training_analysis)**:
  - 精度格式全览（FP32 → BF16 → FP16 → FP8 → MXFP8 → FP4）
  - 五种 FP8 Recipe（tensorwise/delayed/blockwise/mxfp8/custom）及对比
  - FP8 Primary Weights（fp8_param_gather）显存节省分析（6N → 5N bytes）
  - first_last_layers_bf16 首末层 BF16 保护机制
  - TP 通信与 FP8 协同（User Buffer, Pipelined/Bulk Overlap）
  - FP4 QAT（DeepSeek-V4 方案）：无损反量化原理、STE 训练、推理部署
  - MoE + 低精度（Grouped GEMM FP8, Router Fusion, DeepEP A2A）
  - Scaling MoE 论文精度实践总结
  - 配置速查表

**Key topics (transformer_engine_analysis)**:
  - TE 两层架构（Python API + C++/CUDA Kernel）
  - 精度格式矩阵：FP8(E4M3/E5M2/HYBRID) / MXFP8 / NVFP4 / BF16/FP16
  - Recipe 系统（DelayedScaling → Float8CurrentScaling → MXFP8BlockScaling → NVFP4BlockScaling2D）
  - Quantizer 体系（Float8CurrentScalingQuantizer / Float8Quantizer / MXFP8Quantizer）
  - Scale 计算核心公式 + 边界情况处理
  - FP8GlobalStateManager：全局 buffer 批量 amax reduce + 激活重计算支持
  - C++ Kernel 层（quantize/dequantize/gemm/grouped_gemm/融合算子）
  - CommOverlap 体系（CommOverlapHelper/CommOverlap/CommOverlapP2P + NVSHMEM）
  - Megatron 集成桥接（TELinear/TELayerNormColumnParallelLinear/TENorm + FP8 recipe 映射）
  - CUDA Graphs + FP8 协同
  - 环境变量与调试指南

---

## 2026-05-07: 知识库索引体系重构 — overview.md → index.md

**Type**: Infrastructure

- **Renamed** all `overview.md` → `index.md`: `llm/`, `llm/06_infra/megatron-lm/`, `torch_compile/`
- **Renamed** `*_overview.md` → `index.md`: `moonshot_kimi/kimi_overview.md`, `zhipu_glm/glm_overview.md`
- **Created** 13 new `index.md` files for directories lacking one:
  - `wiki/index.md` — 知识库总索引（全新）
  - `llm/01_architecture/index.md`, `llm/02_training/index.md`, `llm/03_alignment/index.md`
  - `llm/04_reasoning_and_retrieval/index.md` (stub), `llm/05_model_families/index.md`, `llm/05_model_families/deepseek/index.md`
  - `llm/06_infra/index.md`, `llm/07_multimodal/index.md` (stub), `llm/08_agents/index.md` (stub)
  - `torch_compile/cudagraphs/index.md`, `torch_compile/cudagraphs/npugraphs/index.md`, `torch_compile/inductor/index.md`
- **Updated** all cross-references (~50 files): `overview` → `index`, `kimi_overview` → `index`, `glm_overview` → `index`
- **Updated** `CLAUDE.md` — Page Types, Naming Conventions, Directory Layout, all workflows

---

## 2026-05-06: Wiki 目录重组 — torch_compile 独立为顶级域

**Type**: Infrastructure

- **Moved** `wiki/llm/02_training/torch_compile/` → `wiki/torch_compile/`
- **Rationale**: 与 `raw/09_pytorch/00_compile/` 对齐，torch_compile 作为独立领域不再嵌套在 LLM training 下
- **Updated** all cross-references (~35 files): `llm/02_training/torch_compile/` → `torch_compile/`
- **Updated** `CLAUDE.md` — Directory Layout 反映新结构

---

## 2026-05-06: Raw 目录结构更新 — 新增 09_pytorch

**Type**: Infrastructure

- **Added** `raw/09_pytorch/00_compile/` — 5 PyTorch compile 内部源码分析图（.eddx 格式）：
  - `torch.compile.eddx` — torch.compile 整体架构
  - `dynamo.eddx` — Dynamo 图捕获
  - `AOTautograd.eddx` — AOT Autograd 前向/反向分离
  - `inductor-lowering.eddx` — Inductor IR Lowering 流程
  - `aoteager精度比对.eddx` — AOT Eager 精度对比
- **Updated** `CLAUDE.md` — Directory Layout 同步更新（raw/ 新增 09_pytorch, wiki/ 反映实际重组后的结构）

---

## 2026-04-29: LLM 并行计算依赖分析（HTML）

**Type**: Knowledge Synthesis（Megatron-LM 源码验证）

- **Created**: `wiki/llm/06_infra/llm_parallelism_analysis.html` — LLM 正反向计算依赖 + 并行策略通信分析（中文）
- **Updated**: `wiki/llm/06_infra/megatron-lm/index.md` — Distributed Parallelism 表格新增条目 + Knowledge Gaps 更新
- **Key topics**:
  - 单层 Transformer Decoder 前向/反向算子 DAG（SVG 依赖图 + 关系表）
  - Megatron-LM 源码级验证: `ColumnParallelLinear` / `RowParallelLinear` / `LinearWithGradAccumulationAndAsyncCommunication`
  - TP (Tensor Parallelism) f/g 算子通信模式、SP (Sequence Parallelism) AG+RS 数据流
  - EP (Expert Parallelism) AllToAll dispatch/combine + 内部 TP 通信
  - CP (Context Parallelism) Ring Attention vs Ulysses 对比
  - 组合并行 (TP+SP+CP+EP+PP) 完整前向执行顺序表
  - 计算通信重叠: async grad AllReduce, Ring Attention P2P overlap, DDP bucket overlap
  - CSS `white-space: pre` 修复, 12 代码块 Python 格式化 + 语法高亮

---

## 2026-04-29: DeepSeek-V4 Raw → Wiki 知识整合

**Type**: Knowledge Integration（Raw MD 文件与 Wiki 合并/去重）

将 `raw/05_model_families/deepseek/` 下 9 个 V4 相关 MD 文件与 Wiki 现有内容整合：

- **Created**: `wiki/llm/05_model_families/deepseek/deepseek_v4_fp4_qat_analysis.md` — FP4 QAT 完整分析（全新主题）
- **Moved (3 files)**:
  - `deepseek_v4_architecture_diagrams.md` — V4 架构 ASCII 结构图（50KB 补充参考）
  - `deepseek_v4_implementation_details.md` — V4 核心组件伪代码实现（34KB 补充参考）
  - `deepseek_v4_technical_deep_dive.md` — CSA/HCA/DSA/MLA 对比深度解析（42KB 补充参考）
- **Updated (merged unique content)**:
  - `deepseek_v4_analysis.md` — 新增 §Compressed KV 数值示例、DualPath 推理框架、Think Modes、Pro-Max 评测
  - `mHC.md` — 扩展 §动态与静态系数（完整公式 3-8、对比表、训练细节）
  - `deepseek_v4_cp_analysis.md` — 新增 §9 实现细节（Fused Select-and-Pad、Top-K Selector、传统 CP 对比表）
- **Cross-references**: 所有新/更新页面双向链接已更新

---

## 2026-04-29: Activation Checkpointing（重计算）完整分析

**Type**: Knowledge Synthesis（PyTorch autograd 机制 + Megatron-LM 源码分析）

- **Created**: `wiki/llm/02_training/activation_checkpointing_analysis.md` — 激活重计算完整分析（中文）
- **Updated**: `wiki/llm/index.md` — Optimizers & Training Algorithms 表格新增条目
- **Updated**: `wiki/llm/06_infra/megatron-lm/Megatron-LM_Distributed_Parallel_Exam.md` — Q12 考点添加交叉引用
- **Key topics**:
  - autograd `ctx.save_for_backward` 机制与 `torch.no_grad` 干预原理
  - ctx 中 tensor 激活值 vs 元信息的二分法（重计算只消除前者）
  - View/Cast/Slice 算子的反向机制：仅依赖元信息，ctx 不存储 tensor
  - View chain 问题与 Megatron `make_viewless_tensor` 的切断方案
  - Megatron 三层 checkpoint 架构：CheckpointFunction → CheckpointWithoutOutput/te_checkpoint → TransformerBlock 调度
  - `distribute_saved_activations` 的 TP 切分/聚合机制
  - `CheckpointWithoutOutput` 的 zero-copy storage sharing 和 `CheckpointManager`
  - Uniform vs Block 调度策略、逐层 checkpoint 的必要性（vs 整 model 一层）
  - Selective recomputation 的子模块级选择依据与 Decoder 层激活值依赖全景
  - 理论激活值开销公式与估算范例

---

## 2026-04-28: DeepSeek-V4 CP 深度分析

---

## 2026-04-28: DeepSeek-V4 CP 深度分析

**Type**: Source Ingestion (扩展已有 V4 分析)

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` §3.5.3, §3.6, §4.1
- **Created**: `wiki/llm/05_model_families/deepseek/deepseek_v4_cp_analysis.md` — DeepSeek-V4 Context Parallelism 深度分析（中文）
- **Updated**: `wiki/llm/05_model_families/deepseek/deepseek_v4_analysis.md` — CP 节扩展并添加指向新页面链接
- **Key topics**:
  - Packed sequences 数据格式与 CP 的三个矛盾（跨 rank 文档切断、压缩窗口跨边界、压缩输出长度不可预测）
  - 两阶段通信协议形式化描述（Stage 1 P2P O(c) 常数通信 + Stage 2 All-Gather 压缩 KV）
  - 通信量开销公式推导与数值估算（CSA ~51× 减少, HCA ~2048× 减少 vs 标准 CP）
  - 三层 sample 可见性控制（sample-level attention mask → block-level causal → precomputed rules / Top-K selector）
  - 训练 vs 推理尾部 token 处理策略对比（丢弃 vs State Cache vs 重计算）
  - CSA 重叠窗口对 CP 边界的额外影响
  - 完整 packed sequences × CP × 压缩的数值示例

---

## 2026-04-24: Wiki Directory Restructure

**Type**: Infrastructure

Restructured `wiki/llm/` to mirror `raw/` classification (01-08), consolidating related content:

- **Created** subdirectories under `wiki/llm/`:
  - `01_architecture/` — Transformer, scaling laws, memory architectures
  - `02_training/` — Optimizers, initialization, training precision
  - `03_alignment/` — RLHF, DPO, GRPO, PPO, and related methods
  - `04_reasoning_and_retrieval/` — Reserved for CoT, verification, RAG
  - `05_model_families/deepseek/` — All DeepSeek model analyses
  - `06_infra/megatron-lm/` — Distributed training, MoE infrastructure
  - `07_multimodal/` — Reserved for vision-language, audio-language
  - `08_agents/` — Reserved for agentic AI, tool use
- **Moved** `wiki/torch_compile/` → `wiki/torch_compile/`
- **Moved** `wiki/megatron-lm/` → `wiki/llm/06_infra/megatron-lm/`
- **Moved** `mHC.md` → `wiki/llm/05_model_families/deepseek/mHC.md`
- **Updated** all path-based wiki links across the entire wiki

---

## 2026-04-16: Wiki Schema & Structure Initialization

**Type**: Infrastructure

Created the wiki schema and structural pages:

- Created `CLAUDE.md` — wiki maintenance schema and rules
- Created `wiki/llm/index.md` — LLM domain knowledge map
- Created `wiki/megatron-lm/overview.md` — Megatron-LM domain knowledge map
- Created `wiki/torch_compile/index.md` — torch compile domain knowledge map
- Created `wiki/changelog.md` — this file

---

## Pre-Changelog Entries (Historical Reconstruction)

The following pages were created before the changelog was established. Dates are approximate.

### ~2026-03: MoE & Distributed Training

- Created `wiki/megatron-lm/Megatron-LM_MoE_Zero_Redundancy_Analysis.md` — Source: `raw/Scalable Training of Moe Models with Megatron core-2603.07685v2.pdf`
- Created `wiki/megatron-lm/Megatron-LM_Distributed_Parallel_Exam.md` — Comprehensive exam covering 5D parallelism

### ~2026-02: Muon Optimizer

- Created `wiki/llm/muon_analysis.md` — Source: `raw/MUON IS SCALABLE FOR LLM TRAINING-2502.16982v1.pdf`
- Created `wiki/megatron-lm/Megatron_LM_TFLOPS_Analysis.md` — TFLOPS estimation methodology

### ~2026-01: DeepSeek & Memory Architectures

- Created `wiki/llm/Engram_Analysis.md` — Source: `raw/Engram_paper.pdf`
- Created `wiki/llm/deepseek_math_v2.md` — Self-verifiable math reasoning

### ~2025-12: Weight Initialization & KIMI

- Created `wiki/llm/llm_initiliaze_analysis.md` — Dense & MoE initialization

---

## 2026-04-17: mHC Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/mHC-2512.24880v2.pdf` (DeepSeek-AI, arXiv:2512.24880v2)
- **Created**: `wiki/llm/mHC.md` — Manifold-Constrained Hyper-Connections analysis (in Chinese)
- **Updated**: `wiki/llm/index.md` — Added mHC entry and cross-domain links
- **Cross-referenced**: Added backlinks to `muon_analysis.md`, `llm_initiliaze_analysis.md`, `Megatron-LM_MoE_Zero_Redundancy_Analysis.md`
- **Key topics**: doubly stochastic matrix, Sinkhorn-Knopp projection, residual stream expansion, DeepSeek-V3 MoE, kernel fusion, selective recomputing

### ~2025-11: Training-Inference Integration

- Created `wiki/megatron-lm/Megatron_vLLM_Weight_Sync_Analysis.md` — verl Megatron + vLLM weight sync

### ~2025-10: Torch Compile & NPU

- Created `wiki/torch_compile/inductor/` — 17 pages covering Dynamo, AOT Autograd, Inductor, NPU backends
- Created `wiki/torch_compile/cudagraphs/` — CUDA Graphs guides and NPU Graphs deep dives

## 2026-04-20: DeepSeek Model Family Batch Ingestion (Part 1/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_LLM-2401.02954.pdf` (DeepSeek-AI, arXiv:2401.02954)
- **Created**: `wiki/llm/deepseek_llm_analysis.md` — DeepSeek LLM analysis
- **Updated**: `wiki/llm/index.md` — Added DeepSeek model family section
- **Key topics**: scaling laws with non-embedding FLOPs/token representation, data quality impact on model/data allocation, multi-step LR scheduler, GQA, bilingual pre-training, SFT+DPO alignment

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V2-2405.04434.pdf` (DeepSeek-AI, arXiv:2405.04434)
- **Created**: `wiki/llm/deepseek_v2_analysis.md` — DeepSeek-V2 analysis
- **Key topics**: MLA (Multi-head Latent Attention), low-rank KV joint compression, decoupled RoPE, DeepSeekMoE, device-limited routing, three-level auxiliary losses, token dropping, GRPO, two-stage RL

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V3-2412.19437.pdf` (DeepSeek-AI, arXiv:2412.19437)
- **Created**: `wiki/llm/deepseek_v3_analysis.md` — DeepSeek-V3 analysis
- **Key topics**: FP8 mixed precision training, fine-grained quantization (tile/block-wise), DualPipe pipeline parallelism, auxiliary-loss-free load balancing, Multi-Token Prediction (MTP), cross-node all-to-all communication kernels, inference deployment with redundant experts, R1 distillation

- **Source**: `raw/05_model_families/deepseek/DeepSeek_R1-2501.12948.pdf` (DeepSeek-AI, arXiv:2501.12948)
- **Created**: `wiki/llm/deepseek_r1_analysis.md` — DeepSeek-R1 analysis
- **Key topics**: pure RL reasoning without SFT, GRPO, emergent self-verification/reflection, "aha moment", multi-stage pipeline (cold start → RL → SFT → RL), distillation to Qwen/Llama, rule-based rewards, language consistency reward

**Remaining**: Coder, Coder-V2, Math, MoE, Prover, VL

---

## 2026-04-24: DeepSeek-V4 Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` (DeepSeek-AI, 2025)
- **Created**: `wiki/llm/deepseek_v4_analysis.md` — DeepSeek-V4 analysis (in Chinese)
- **Updated**: `wiki/llm/index.md` — Added V4 to DeepSeek model family section
- **Updated**: `wiki/llm/deepseek_v3_analysis.md` — Added backlink to V4
- **Updated**: `wiki/llm/deepseek_v2_analysis.md` — Added backlink to V4
- **Cross-referenced**: `mHC.md`, `muon_analysis.md`, `deepseek_v3_analysis.md`, `deepseek_v2_analysis.md`
- **Key topics**: CSA (Compressed Sparse Attention), HCA (Heavily Compressed Attention), hybrid attention architecture, DSA (DeepSeek Sparse Attention), Lightning Indexer, million-token context, mHC integration, Muon optimizer, Anticipatory Routing, SwiGLU clamping, wave-based EP overlap, TileLang kernels, FP4 QAT, heterogeneous KV cache management, on-disk KV cache storage

---

## 2026-04-21: DeepSeek Model Family Batch Ingestion (Part 2/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder-2401.14196.pdf` (DeepSeek-AI, arXiv:2401.14196)
- **Created**: `wiki/llm/deepseek_coder_analysis.md` — DeepSeek-Coder analysis
- **Key topics**: repository-level code corpus, dependency parsing, topological sort, Fill-in-the-Middle (FIM), 87 programming languages, 16K context, GQA

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder_V2-2406.11931.pdf` (DeepSeek-AI, arXiv:2406.11931)
- **Created**: `wiki/llm/deepseek_coder_v2_analysis.md` — DeepSeek-Coder-V2 analysis
- **Key topics**: MoE code model, 338 languages, 128K context, 6T additional tokens, YaRN extension, GRPO with reward model, SWE-bench >10%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Math-2402.03300.pdf` (DeepSeek-AI, arXiv:2402.03300)
- **Created**: `wiki/llm/deepseek_math_analysis.md` — DeepSeekMath analysis
- **Key topics**: 120B math tokens from Common Crawl, iterative fastText pipeline, GRPO origin, unified RL paradigm, MATH 51.7%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_MoE-2401.06066.pdf` (DeepSeek-AI, arXiv:2401.06066)
- **Created**: `wiki/llm/deepseek_moe_analysis.md` — DeepSeekMoE architecture analysis
- **Key topics**: fine-grained expert segmentation, shared expert isolation, expert-level/device-level balance loss, 2B/16B/145B scales

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Prover-2408.08152.pdf` (DeepSeek-AI, arXiv:2408.08152)
- **Created**: `wiki/llm/deepseek_prover_analysis.md` — DeepSeek-Prover-V1.5 analysis
- **Key topics**: Lean 4 theorem proving, truncate-and-resume mechanism, RMaxTS Monte-Carlo tree search, thought-augmented proofs, RLPAF

- **Source**: `raw/05_model_families/deepseek/DeepSeek_VL-2403.05525.pdf` (DeepSeek-AI, arXiv:2403.05525)
- **Created**: `wiki/llm/deepseek_vl_analysis.md` — DeepSeek-VL analysis
- **Key topics**: hybrid vision encoder (SigLIP + SAM), 576 visual tokens, modality warm-up, 70% text preservation, real-world VL taxonomy

- **Note**: `raw/05_model_families/deepseek/DeepSeek_VL2-2412.10322.pdf` was identified as an unrelated physics paper (arXiv:2412.10322v1, hep-lat). No genuine DeepSeek-VL2 source was found.

**Remaining**: None (DeepSeek model family complete)

---

## 2026-04-21: Architecture Foundations & Alignment Methods Batch Ingestion

**Type**: Source Ingestion

### Architecture Foundations (01_architecture/)

- **Source**: `raw/01_architecture/Attention_Is_All_You_Need-1706.03762.pdf` (Vaswani et al., Google, NIPS 2017)
- **Created**: `wiki/llm/attention_is_all_you_need_analysis.md` — Transformer architecture analysis
- **Key topics**: scaled dot-product attention, multi-head attention, positional encoding, encoder-decoder structure, self-attention vs RNN/CNN complexity, O(1) path length

- **Source**: `raw/01_architecture/Scaling_Laws_for_Neural_Language_Models-2001.08361.pdf` (Kaplan et al., OpenAI, 2020)
- **Created**: `wiki/llm/scaling_laws_analysis.md` — Neural scaling laws analysis
- **Key topics**: power-law scaling (L ~ N^-0.076, D^-0.095, C^-0.050), compute-optimal training (N~C^0.73), sub-linear data scaling (D~N^0.74), early stopping, critical batch size, architecture independence

- **Source**: `raw/01_architecture/Long_Context_Scaling_Law-2503.04725.pdf` (Chen et al., MIT, NeurIPS 2025)
- **Created**: `wiki/llm/long_context_scaling_law_analysis.md` — Long-context mutual information scaling
- **Key topics**: bipartite mutual information (I_BP ~ L^beta), L2M condition, history state requirements, Transformer vs SSM long-context capability

- **Skipped**: `raw/01_architecture/Scaling_Laws_for_Transfer-2002.05102.pdf` — PDF contains unrelated mathematics paper (Hurwitz actions on reflection groups)

### Alignment & Preference Optimization (03_alignment/)

- **Source**: `raw/03_alignment/PPO_Proximal_Policy_Optimization-1707.06347.pdf` (Schulman et al., OpenAI, 2017)
- **Created**: `wiki/llm/ppo_analysis.md` — PPO algorithm analysis
- **Key topics**: PPO-Clip objective, surrogate loss, multiple epochs on same data, GAE advantage estimation, KL constraint

- **Source**: `raw/03_alignment/InstructGPT_RLHF-2203.02155.pdf` (Ouyang et al., OpenAI, 2022)
- **Created**: `wiki/llm/instructgpt_rlhf_analysis.md` — RLHF pipeline analysis
- **Key topics**: three-step RLHF (SFT→RM→PPO), KL penalty against SFT, 1.3B > 175B GPT-3, helpful/honest/harmless criteria

- **Source**: `raw/03_alignment/DPO_Direct_Preference_Optimization-2305.18290.pdf` (Rafailov et al., Stanford, 2023)
- **Created**: `wiki/llm/dpo_analysis.md` — DPO algorithm analysis
- **Key topics**: closed-form policy-reward relationship, binary cross-entropy replaces RLHF, no sampling during training

- **Created**: `wiki/llm/preference_optimization_analysis.md` — DPO family comparison
- **Covers**: IPO (squared loss), SimPO (no ref model, length-normalized), ORPO (monolithic), KTO (binary labels, prospect theory), MODPO (multi-objective)

- **Source**: `raw/03_alignment/DeepSeek_R1_Reasoning_via_RL-2501.12948.pdf` (DeepSeek-AI, 2025)
- **Created**: `wiki/llm/grpo_analysis.md` — GRPO algorithm analysis
- **Key topics**: group-relative advantages, no value function, pure RL for reasoning, DeepSeek-R1-Zero emergent behaviors

**Updated**: `wiki/llm/index.md` — Added Architecture Foundations, Scaling Laws, and Alignment sections

---

## 2026-04-21: Alignment Methods Batch Ingestion (Part 2)

**Type**: Source Ingestion

### Advanced RL Algorithms

- **Source**: `raw/03_alignment/DAPO_Decoupled_Clip_Dynamic_Sampling-2503.14476.pdf` (ByteDance Seed, Tsinghua AIR, 2025)
- **Created**: `wiki/llm/dapo_analysis.md` — DAPO algorithm analysis
- **Key topics**: decoupled clipping (eps_low=0.2, eps_high=0.28), dynamic sampling (filter accuracy 0/1), token-level policy gradient loss, soft overlong punishment, AIME 50 with Qwen2.5-32B, open-source RL system

- **Source**: `raw/03_alignment/GSPO_Group_Sequence_Policy_Optimization-2507.18071.pdf` (Qwen Team, Alibaba, 2025)
- **Created**: `wiki/llm/gspo_analysis.md` — GSPO algorithm analysis
- **Key topics**: sequence-level importance ratio, fixes GRPO's token-level instability, length-normalized sequence likelihood, stabilizes MoE RL training, Qwen3 improvements

- **Source**: `raw/03_alignment/RLOO_REINFORCE_Leave_One_Out-2402.14740.pdf` (Cohere For AI, 2024)
- **Created**: `wiki/llm/rloo_analysis.md` — RLOO algorithm analysis
- **Key topics**: REINFORCE with leave-one-out baseline, no value function needed, theoretical foundation for GRPO, 2.5x faster than PPO

- **Source**: `raw/03_alignment/VAPO_Value_Augmented_Proximal_Policy_Optimization-2504.05118.pdf` (ByteDance Seed, 2025)
- **Created**: `wiki/llm/vapo_analysis.md` — VAPO framework analysis
- **Key topics**: value-model-based RL, AIME 60.4 (SOTA), addresses value bias/length heterogeneity/reward sparsity, 5000 steps to SOTA, zero crashes

### RLHF Foundations & Advanced Methods

- **Created**: `wiki/llm/rlhf_foundations_analysis.md` — Comprehensive coverage of:
  - **ReMax** (arXiv:2310.10505): Simplified RLHF using REINFORCE, exploits fast simulation/deterministic transitions/trajectory rewards
  - **Weak-to-Strong Generalization** (OpenAI, arXiv:2312.09390): Can weak model supervision elicit strong model capabilities? Analogy to superhuman alignment
  - **Scaling Laws for RM Overoptimization** (OpenAI, arXiv:2210.10760): Goodhart's Law in RLHF, predictable scaling of overoptimization, best-of-n vs RL
  - **Learning to Summarize** (OpenAI, arXiv:2009.01325): First RLHF for summarization, precursor to InstructGPT
  - **Fine-Tuning from Human Preferences** (OpenAI, arXiv:1909.08593): Earliest RLHF work, stylistic control and summarization
  - **RigorLLM** (arXiv:2403.13031): Resilient guardrails against adversarial attacks, energy-based data generation, minimax optimization

**Updated**: `wiki/llm/index.md` — Added DAPO, GSPO, RLOO, VAPO, and RLHF Foundations entries

**Digestion progress**: 3/4 architecture papers, **20/20 alignment papers digested** (complete)

## Related Pages

- [[01_theory/index]]
- [[02_engineering/02_train_frameworks/megatron-lm/index]]
- [[02_engineering/01_ai_frameworks/index]]
