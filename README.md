# LLM Knowledge Base

LLM 训练与推理技术知识库，由 Claude Code Agent 维护。全库共 **409 篇** Markdown 分析页，覆盖从模型结构、并行理论到框架源码级机制的完整链路。

三种浏览方式：

- **直接在 GitHub 上读** —— 下方[二级目录概览](#wiki-二级目录概览)与[核心文章索引](#核心文章索引)可以点进任意原始文档。
- **本地 HTML 站点** —— `npm run docs` 起一个离线 Quartz 站点，支持 wikilink、Mermaid、公式（见[本地文档站点](#本地文档站点)）。
- **Obsidian** —— 直接把 `wiki/` 作为 vault 打开。

## 目录结构

```
raw/            # 源材料索引（论文的 arXiv/官方链接说明页），与 wiki/ 不要求镜像对齐
wiki/           # 分析页（Obsidian vault），功能分类树是唯一内容权威
├── 01_theory/        # 理论：模型家族 / 预训练 / SFT / 后训练对齐 / 推理技术 / 分布式并行
├── 02_engineering/   # 工程：AI 框架 / 训练框架 / 推理框架 / 后训练框架 / GPU Kernel / 自动并行 / 训练可靠性
├── courses/          # 纯导读层：两条跨域学习路线，只含阅读顺序 + 链接，不承载正文
├── index.md          # 总索引（域级表格）
└── changelog.md      # 当季变更日志；历史条目按季度归档于 wiki/changelog/
docs/           # 流程文档（specs / plans / research）
tools/          # 维护工具：check_links.py（链接健康）、check_math.py（公式规范）、docs-site/（本地站点）
```

## wiki 二级目录概览

### 01 理论

| 二级目录 | 篇数 | 讲什么 |
|---|---:|---|
| [01_models](wiki/01_theory/01_models/index.md) | 59 | 各家前沿模型的结构与技术报告解读：[DeepSeek](wiki/01_theory/01_models/deepseek/index.md)、[Kimi](wiki/01_theory/01_models/moonshot_kimi/index.md)、[GLM](wiki/01_theory/01_models/zhipu_glm/index.md)、[Qwen](wiki/01_theory/01_models/alibaba_qwen/index.md)、[LongCat](wiki/01_theory/01_models/meituan_longcat/index.md) 等 |
| [02_pretraining](wiki/01_theory/02_pretraining/index.md) | 7 | 预训练侧的优化器、低精度、重计算、参数初始化与训推精度一致性 |
| [03_sft](wiki/01_theory/03_sft/index.md) | 1 | SFT 与低参微调（待建设） |
| [04_posttraining](wiki/01_theory/04_posttraining/index.md) | 22 | 后训练对齐：PPO/GRPO/GSPO 等算法演进、on-policy 蒸馏、staleness 与训推不一致 |
| [05_inference](wiki/01_theory/05_inference/index.md) | 1 | 推理技术（待建设） |
| [06_distributed_parallelism](wiki/01_theory/06_distributed_parallelism/index.md) | 9 | 并行理论本身：集合通信代价模型、DP/TP/SP/CP/PP/EP、ZeRO、Ring Attention |

### 02 工程

| 二级目录 | 篇数 | 讲什么 |
|---|---:|---|
| [01_ai_frameworks](wiki/02_engineering/01_ai_frameworks/index.md) | 150 | PyTorch 本体：[eager 运行时](wiki/02_engineering/01_ai_frameworks/01_eager_runtime/index.md)、[编译栈](wiki/02_engineering/01_ai_frameworks/02_compile_stack/index.md)（Dynamo/AOTAutograd/Inductor/MLIR）、[运行时图](wiki/02_engineering/01_ai_frameworks/03_runtime_graphs/index.md)、export 与分布式原语 |
| [02_train_frameworks](wiki/02_engineering/02_train_frameworks/index.md) | 56 | 训练框架源码级机制：[Megatron-LM](wiki/02_engineering/02_train_frameworks/megatron-lm/index.md)、[torchtitan](wiki/02_engineering/02_train_frameworks/torchtitan/index.md)、[MindSpeed](wiki/02_engineering/02_train_frameworks/mindspeed/index.md)、[MindFormers](wiki/02_engineering/02_train_frameworks/mindformers/index.md) |
| [03_infer_frameworks](wiki/02_engineering/03_infer_frameworks/index.md) | 28 | 推理引擎：[vLLM 知识域](wiki/02_engineering/03_infer_frameworks/vllm/index.md)（调度、分页 KV、投机解码、CUDA Graph）与 Mooncake 分离式架构 |
| [04_posttrain_frameworks](wiki/02_engineering/04_posttrain_frameworks/index.md) | 44 | RL 后训练框架：[verl](wiki/02_engineering/04_posttrain_frameworks/verl/index.md)、[slime](wiki/02_engineering/04_posttrain_frameworks/slime/index.md)，含权重同步、rollout 重分片、控制面 |
| [05_gpu_kernel](wiki/02_engineering/05_gpu_kernel/index.md) | 17 | Kernel 开发：CUDA GEMM/FlashAttention、[Triton 学习路线](wiki/02_engineering/05_gpu_kernel/triton/index.md)、Ascend 达芬奇执行模型 |
| [06_auto_parallel](wiki/02_engineering/06_auto_parallel/index.md) | 2 | 自动并行业界综述 |
| [07_training_reliability](wiki/02_engineering/07_training_reliability/index.md) | 5 | 确定性与数值可靠性、批次不变性、故障容错、训练动力学稳定性 |

### 跨域导读

| 课程 | 讲什么 |
|---|---|
| [`torch.compile` 端到端阅读课程](wiki/courses/torch_compile_end_to_end.md) | 从 `torch.compile` 入口一路读到 Inductor codegen 的阅读顺序 |
| [LLM 后训练前沿阅读课程](wiki/courses/posttraining_frontier.md) | 后训练算法与 Infra 的交叉阅读路线 |

## 核心文章索引

按**被库内其它页面引用的次数**排序挑选，可直接点进原始 Markdown。

### 理论

| 文章 | 一句话 |
|---|---|
| [DeepSeek-V3 Analysis](wiki/01_theory/01_models/deepseek/12_deepseek_v3_analysis.md) | 全库被引最多的模型页：MoE 无辅助损失负载均衡、MTP、FP8 训练 |
| [DeepSeek-V4 深度解析](wiki/01_theory/01_models/deepseek/13_deepseek_v4_analysis.md) | 百万上下文：CSA/HCA 混合注意力、mHC、Muon |
| [Kimi K3 结构变化深析](wiki/01_theory/01_models/moonshot_kimi/22_kimi_k3_architecture_deepdive.md) | 同时优化序列轴与深度轴的信息流 |
| [GLM-5 架构深挖](wiki/01_theory/01_models/zhipu_glm/20_glm5_architecture_deepdive.md) | MLA·Muon Split·MTP·DSA 的规模 × 长上下文成本权衡 |
| [Muon 优化器](wiki/01_theory/02_pretraining/11_muon_analysis.md) | 原理解读 + Megatron-LM 实现 |
| [Reasoning RL 算法演进](wiki/01_theory/04_posttraining/13_reasoning_rl_algorithm_evolution_analysis.md) | 后训练域的主干页，串起各 RL 算法的因果链 |
| [GRPO 分析](wiki/01_theory/04_posttraining/20_grpo_analysis.md) | Group Relative Policy Optimization 机制拆解 |
| [训推不一致（TIM）因果链](wiki/01_theory/04_posttraining/26_tim_causal_chain_analysis.md) | 从 kernel 非确定性一路推到训练崩溃 |
| [分布式原语与通信代价模型](wiki/01_theory/06_distributed_parallelism/10_collectives_analysis.md) | 并行理论的地基：集合通信代价怎么算 |
| [专家并行 EP（MoE）](wiki/01_theory/06_distributed_parallelism/14_expert_parallel_analysis.md) | EP 的切分、dispatch/combine 与通信量 |

### 工程

| 文章 | 一句话 |
|---|---|
| [Megatron-LM 专家并行深度解析](wiki/02_engineering/02_train_frameworks/megatron-lm/14_megatron_ep_analysis.md) | 全库被引最多的工程页（65 次入链） |
| [Megatron-Core 通信掩盖](wiki/02_engineering/02_train_frameworks/megatron-lm/20_megatron_comm_overlap_analysis.md) | 通算重叠的实现与生效条件 |
| [FSDP2 机制级分析](wiki/02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis.md) | torchtitan 视角的数据并行 |
| [AOTAutograd 的 Joint/Forward/Backward Graph](wiki/02_engineering/01_ai_frameworks/02_compile_stack/02_aot_autograd/11_aotautograd_joint_forward_backward_graphs_analysis.md) | 编译栈里正反向图怎么被切出来 |
| [Inductor Scheduler 依赖图与 Fusion](wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/13_scheduler_dependency_graph_fusion_and_ordering_analysis.md) | 融合决策与执行顺序 |
| [符号形状、Guards 与图复用](wiki/02_engineering/01_ai_frameworks/02_compile_stack/01_dynamo/20_symbolic_shapes_guards_and_graph_reuse_analysis.md) | 动态 shape 为什么能复用图 |
| [vLLM KV Cache 管理](wiki/02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis.md) | 分页不是索引技巧，而是物理块所有权协议 |
| [vLLM 请求全链路导览](wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_request_flow_walkthrough_analysis.md) | 一条请求怎样穿过进程、队列与 GPU（含离线交互图） |
| [verl 端到端训练迭代](wiki/02_engineering/04_posttrain_frameworks/verl/10_verl_end_to_end_iteration_analysis.md) | RL 后训练框架的主循环 |
| [slime 权重同步](wiki/02_engineering/04_posttrain_frameworks/slime/16_slime_weight_sync_analysis.md) | Megatron→SGLang 跨异构分片的提交协议 |
| [GPU Kernel 开发](wiki/02_engineering/05_gpu_kernel/01_gpu_kernel_guide.md) | Kernel 域入口：访存、Tiling、FlashAttention |
| [确定性与数值可靠性](wiki/02_engineering/07_training_reliability/10_determinism_and_numerical_reliability_analysis.md) | 浮点非确定性 · batch 不变性 · 低精度累加 · SDC |

完整页面清单见 [wiki/index.md](wiki/index.md)；各域内部导航见对应 `index.md`。

## 本地文档站点

需要 Node.js ≥ 22、npm ≥ 10.9.2 和 Git ≥ 2。在仓库根目录运行：

```bash
npm run docs
```

命令会在浏览器打开 `http://127.0.0.1:8080`，并监听 Markdown 变化。首次运行需要联网下载已锁定版本的 Quartz、社区插件和 Mermaid；成功安装后，日常启动与构建均复用 `.cache/llm-knowledge-docs/` 下的仓库私有运行时，不再访问包仓库或 CDN。

常用命令：

```bash
npm run docs -- --port 8088  # HTTP 使用 8088，热更新 WebSocket 使用 8089
npm run docs -- --no-open    # 启动但不自动打开浏览器
npm run docs:build           # 仅生成静态站点
npm run docs:test            # 单元测试 + 本地浏览器端到端验收
npm run docs:repair          # 显式重建损坏或版本漂移的私有运行时
```

HTTP 和热更新端口都只绑定 `127.0.0.1`，不向局域网暴露。站点直接读取且只展示 `wiki/`；它不会复制、格式化或改写任何 Markdown，Obsidian wikilink、callout、Mermaid 与公式兼容均由站点层处理。

也可以继续用 Obsidian 打开 `wiki/` 浏览，或 `cd llm-knowledge && claude` 直接提问。

## 质量门禁

```bash
python tools/check_links.py     # wikilink 健康：broken / ambiguous / bare_index / orphans 必须为 0
python tools/check_math.py wiki # Obsidian 公式规范（--changed 只查改动文件，--strict 把 warning 也算失败）
python -m pytest tools/         # 维护工具自身的单元测试
npm run docs:test               # 本地站点单元测试 + 端到端验收
```

## 维护

按 [CLAUDE.md](CLAUDE.md) 定义的 Workflow 由 Agent 维护。当前结构整改：
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`。
