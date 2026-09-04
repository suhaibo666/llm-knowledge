# LLM Knowledge Base

LLM 训练与推理技术知识库，由 Claude Code Agent 维护。全库共 **446 篇** Markdown 分析页，覆盖从模型结构、并行理论到框架源码级机制的完整链路。

三种浏览方式：

- **直接在 GitHub 上读** —— 下方[二级目录概览](#wiki-二级目录概览)与[核心文章索引](#核心文章索引)可以点进任意原始文档。
- **本地 HTML 站点** —— `npm run docs` 起一个离线 Quartz 站点，支持 wikilink、Mermaid、公式（见[本地文档站点](#本地文档站点)）。
- **Obsidian** —— 直接把 `wiki/` 作为 vault 打开。

## 目录结构

```
raw/            # 源材料索引（论文的 arXiv/官方链接说明页、文章与图表源），与 wiki/ 不要求镜像对齐
wiki/           # 分析页（Obsidian vault），功能分类树是唯一内容权威
├── 01_theory/        # 理论：模型家族 / 预训练 / SFT / 后训练对齐 / 推理技术 / 分布式并行
├── 02_engineering/   # 工程：PyTorch / 训练框架 / 推理框架 / 后训练框架 / GPU Kernel / 自动并行 / 训练可靠性
├── courses/          # 纯导读层：三条跨域学习路线，只含阅读顺序 + 链接，不承载正文
├── index.md          # 总索引（域级表格）
└── changelog.md      # 当季变更日志；历史条目按季度归档于 wiki/changelog/
docs/           # 流程与运营文档：coverage（覆盖台账）/ radar（上游雷达报告）/ research / superpowers（specs、plans）
tools/          # 维护工具：check_links/check_math/check_markdown/check_assets（四条日常门禁）、
                # check_locators（代码引用）、mkdocs_site/（出版站点）、docs-site/（旧本地站点）
```

## wiki 二级目录概览

### 01 理论

| 二级目录 | 讲什么 |
|---|---|
| [01_models](wiki/01_theory/01_models/index.md) | 各家前沿模型的结构与技术报告解读：[DeepSeek](wiki/01_theory/01_models/deepseek/index.md)、[Kimi](wiki/01_theory/01_models/moonshot_kimi/index.md)、[GLM](wiki/01_theory/01_models/zhipu_glm/index.md)、[Qwen](wiki/01_theory/01_models/alibaba_qwen/index.md)、[LongCat](wiki/01_theory/01_models/meituan_longcat/index.md) 等 |
| [02_pretraining](wiki/01_theory/02_pretraining/index.md) | 预训练侧的优化器、低精度、重计算、参数初始化与训推精度一致性 |
| [03_sft](wiki/01_theory/03_sft/index.md) | SFT 与低参微调（待建设） |
| [04_posttraining](wiki/01_theory/04_posttraining/index.md) | 后训练对齐：PPO/GRPO/GSPO 等算法演进、on-policy 蒸馏、staleness 与训推不一致 |
| [05_inference](wiki/01_theory/05_inference/index.md) | 推理技术（待建设） |
| [06_distributed_parallelism](wiki/01_theory/06_distributed_parallelism/index.md) | 并行理论本身：集合通信代价模型、DP/TP/SP/CP/PP/EP、ZeRO、Ring Attention |

### 02 工程

| 二级目录 | 讲什么 |
|---|---|
| [01_pytorch](wiki/02_engineering/01_pytorch/index.md) | PyTorch 本体：[eager 运行时](wiki/02_engineering/01_pytorch/01_eager_runtime/index.md)、[编译栈](wiki/02_engineering/01_pytorch/02_compile_stack/index.md)（Dynamo/AOTAutograd/Inductor/MLIR）、[运行时图](wiki/02_engineering/01_pytorch/03_runtime_graphs/index.md)、export 与分布式原语 |
| [02_train_frameworks](wiki/02_engineering/02_train_frameworks/index.md) | 训练框架源码级机制：[Megatron-LM](wiki/02_engineering/02_train_frameworks/megatron-lm/index.md)、[torchtitan](wiki/02_engineering/02_train_frameworks/torchtitan/index.md)、[MindSpeed](wiki/02_engineering/02_train_frameworks/mindspeed/index.md)、[MindFormers](wiki/02_engineering/02_train_frameworks/mindformers/index.md) |
| [03_infer_frameworks](wiki/02_engineering/03_infer_frameworks/index.md) | 推理引擎：[vLLM 知识域](wiki/02_engineering/03_infer_frameworks/vllm/index.md)（调度、分页 KV、投机解码、CUDA Graph）与 Mooncake 分离式架构 |
| [04_posttrain_frameworks](wiki/02_engineering/04_posttrain_frameworks/index.md) | RL 后训练框架：[verl](wiki/02_engineering/04_posttrain_frameworks/verl/index.md)、[slime](wiki/02_engineering/04_posttrain_frameworks/slime/index.md)，含权重同步、rollout 重分片、控制面 |
| [05_gpu_kernel](wiki/02_engineering/05_gpu_kernel/index.md) | Kernel 开发：CUDA GEMM/FlashAttention、[Triton 学习路线](wiki/02_engineering/05_gpu_kernel/triton/index.md)、Ascend 达芬奇执行模型 |
| [06_auto_parallel](wiki/02_engineering/06_auto_parallel/index.md) | 自动并行业界综述 |
| [07_training_reliability](wiki/02_engineering/07_training_reliability/index.md) | 确定性与数值可靠性、批次不变性、故障容错、训练动力学稳定性 |

### 跨域导读

| 课程 | 讲什么 |
|---|---|
| [`torch.compile` 端到端阅读课程](wiki/courses/torch_compile_end_to_end.md) | 从 `torch.compile` 入口一路读到 Inductor codegen 的阅读顺序 |
| [LLM 后训练前沿阅读课程](wiki/courses/posttraining_frontier.md) | 后训练算法与 Infra 的交叉阅读路线 |
| [Megatron-LM 阅读路径](wiki/courses/megatron_lm.md) | 入门三页起步，先走 dense 主干，再按长上下文 / MoE / 性能 / 可靠性等分支分流 |

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
| [Megatron-LM 专家并行深度解析](wiki/02_engineering/02_train_frameworks/megatron-lm/14_megatron_ep_analysis.md) | 全库被引最多的工程页（89 次入链） |
| [Megatron-Core 通信掩盖](wiki/02_engineering/02_train_frameworks/megatron-lm/20_megatron_comm_overlap_analysis.md) | 通算重叠的实现与生效条件 |
| [FSDP2 机制级分析](wiki/02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis.md) | torchtitan 视角的数据并行 |
| [AOTAutograd 的 Joint/Forward/Backward Graph](wiki/02_engineering/01_pytorch/02_compile_stack/02_aot_autograd/11_aotautograd_joint_forward_backward_graphs_analysis.md) | 编译栈里正反向图怎么被切出来 |
| [Inductor Scheduler 依赖图与 Fusion](wiki/02_engineering/01_pytorch/02_compile_stack/04_inductor/13_scheduler_dependency_graph_fusion_and_ordering_analysis.md) | 融合决策与执行顺序 |
| [符号形状、Guards 与图复用](wiki/02_engineering/01_pytorch/02_compile_stack/01_dynamo/20_symbolic_shapes_guards_and_graph_reuse_analysis.md) | 动态 shape 为什么能复用图 |
| [vLLM KV Cache 管理](wiki/02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis.md) | 分页不是索引技巧，而是物理块所有权协议 |
| [vLLM 架构概览](wiki/02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis.md) | 六层责任与状态所有权，一条请求在其上的状态移交主线 |
| [verl 端到端训练迭代](wiki/02_engineering/04_posttrain_frameworks/verl/10_verl_end_to_end_iteration_analysis.md) | RL 后训练框架的主循环 |
| [slime 权重同步](wiki/02_engineering/04_posttrain_frameworks/slime/16_slime_weight_sync_analysis.md) | Megatron→SGLang 跨异构分片的提交协议 |
| [GPU Kernel 开发](wiki/02_engineering/05_gpu_kernel/01_gpu_kernel_guide.md) | Kernel 域入口：访存、Tiling、FlashAttention |
| [确定性与数值可靠性](wiki/02_engineering/07_training_reliability/10_determinism_and_numerical_reliability_analysis.md) | 浮点非确定性 · batch 不变性 · 低精度累加 · SDC |

完整页面清单与各域页面数见 [wiki/index.md](wiki/index.md)；各域内部导航见对应 `index.md`。

## 本地文档站点

需要 Node.js ≥ 22、npm ≥ 10.9.2 和 Git ≥ 2。在仓库根目录运行：

```bash
npm run docs
```

命令会在浏览器打开 `http://127.0.0.1:8080`，并监听 Markdown 变化（改完存盘，页面自动刷新）。首次运行需要联网下载已锁定版本的 Quartz、社区插件和 Mermaid，这一步耗时较久（几分钟级），**属于正常现象**；成功安装后，日常启动与构建均复用 `.cache/llm-knowledge-docs/` 下的仓库私有运行时，不再访问包仓库或 CDN。运行时损坏或版本漂移时用 `npm run docs:repair` 原子重建（先装到 staging，校验通过才替换，失败则保留旧运行时）。全量 446 页首次构建约 30 秒。

常用命令：

```bash
npm run docs -- --port 8088       # HTTP 使用 8088，热更新 WebSocket 使用 8089
npm run docs -- --host 127.0.0.1  # 只绑回环，不对外暴露
npm run docs -- --no-open         # 启动但不自动打开浏览器
npm run docs:build                # 仅生成静态站点
npm run docs:test                 # 旧站点的单元测试 + 端到端验收（CI 部署的是 mkdocs 那套）
npm run docs:repair               # 显式重建损坏或版本漂移的私有运行时
```

**监听地址**：默认绑定 `0.0.0.0`，即本机所有网卡——同一局域网内的其他机器可以直接用 `http://<本机 IP>:8080/` 访问，启动日志会把可用地址列出来。热更新的 WebSocket 地址由浏览器按当前页面的 hostname 推导，因此远程访问时热更新同样有效。

> 这意味着**站点对局域网可见**。在不可信网络里请用 `npm run docs -- --host 127.0.0.1` 退回只绑回环。

站点直接读取且只展示 `wiki/`；它不会复制、格式化或改写任何 Markdown，Obsidian wikilink、callout、Mermaid 与公式兼容均由站点层处理。

也可以继续用 Obsidian 打开 `wiki/` 浏览，或 `cd llm-knowledge && claude` 直接提问。

## 质量门禁

门禁按**改动碰了什么**分层，不是每次都跑全量。日常这四条约 6 秒：

```bash
python tools/check_links.py --strict              # wikilink：broken / ambiguous / bare_index / orphans 必须为 0
python tools/check_math.py --changed --strict     # Obsidian 公式规范
python tools/check_markdown.py --changed --strict # 列表标记与 mermaid 标签的渲染陷阱
python tools/check_assets.py --changed --strict   # 图片与本地资源是否存在
```

改了工具、渲染栈或要 push 时才跑更重的那几层（`pytest tools/`、`npm run docs:mkdocs:test`），
构建与浏览器相关的校验支持按改动收窄（`cli build --changed`、`mathjax-corpus --pages`）。
完整的分层表、各层成本与各检查器基线见 [CLAUDE.md](CLAUDE.md) 的「Quality gates」一节。

## 上游雷达

`tools/radar.py` 每周扫一遍上游，产出 `docs/radar/<日期>.md`：

```bash
python tools/radar.py              # 最近 7 天，写报告并更新 state
python tools/radar.py --since 14   # 改时间窗
python tools/radar.py --dry-run    # 只打印，不落盘
```

报告分五节，**第一节最重要**：哪些仓库的 KB 基线已经落后上游多少个提交（首次运行实测：torchtitan 落后 308、Megatron-LM 276、verl 266、vLLM 236、slime 23）。其余四节是仓库活动、模型厂商新发布（HuggingFace）、前沿论文（arXiv 五个主题）、以及**本次采集失败项**——失败会如实列出，不会被伪装成「本期无变化」。

追踪清单在 [`docs/radar/watchlist.yaml`](docs/radar/watchlist.yaml)，显式维护（13 仓 + 7 家厂商 + 5 个论文主题）。为什么不从页头基线自动推导、arXiv 查询为什么必须带 LLM 约束，文件头的注释里写了原因。

> **边界**：雷达**只报告事实，不写 `wiki/` 分析页**。本库的价值在于每条断言都有可核验定位符；无人值守产出的机制级结论没人复核，会污染这个前提。要把某个变化落成分析页，走 [`source-faithful-analysis`](skills/source-faithful-analysis/SKILL.md)，**并在合并后同步更新 watchlist 里的 `kb_baseline`**，否则雷达会一直报同一批陈旧漂移。

已注册为每周一的本机定时任务（`llm-knowledge-upstream-radar`），只在 Claude 应用开着时触发，关着则下次启动补跑。

## 维护

按 [CLAUDE.md](CLAUDE.md) 定义的分层结构、来源政策与质量门禁由 Agent 维护；具体写法收在 [`skills/`](skills/README.md)，按需加载。
