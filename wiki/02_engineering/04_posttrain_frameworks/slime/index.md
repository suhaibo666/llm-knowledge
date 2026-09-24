---
title: "slime RL 后训练框架 — 知识地图"
---

# slime RL 后训练框架 — 知识地图

> **源码基线**：`THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`（`main`，2026-09-03）
> **最近更新**：2026-09-17。
> **系列规模**：23 篇内容页（其中 1 篇审计 vime 衍生实现，vime 基线见该页页头），加本索引共 24 个 Markdown 文件。

本表是本目录唯一页面入口清单。先读[[02_engineering/04_posttrain_frameworks/slime/01_slime_architecture_overview_analysis|架构总览]]和[[02_engineering/04_posttrain_frameworks/slime/02_slime_quickstart_and_configuration_guide|快速上手与配置]]，再从[[02_engineering/04_posttrain_frameworks/slime/10_slime_end_to_end_iteration_analysis|端到端迭代时序]]进入段 1 的数据、训练、权重与服务机制；段 2 的能力专题按需选读；排查吞吐或训练发散时读[[02_engineering/04_posttrain_frameworks/slime/30_slime_rollout_optimization_analysis|rollout 优化]]与[[02_engineering/04_posttrain_frameworks/slime/31_slime_posttraining_stability_analysis|后训练稳定性]]。

## 页面入口

| 段 | 页面 | 解决的问题 |
|---|---|---|
| 0 | [[02_engineering/04_posttrain_frameworks/slime/01_slime_architecture_overview_analysis|架构总览]] | 设计背景、四层八模块、运行协作、模块合同与使用场景 |
| 0 | [[02_engineering/04_posttrain_frameworks/slime/02_slime_quickstart_and_configuration_guide|快速上手与配置]] | 从脚本、CLI、Megatron/SGLang YAML 到源码入口怎样对应；异步 / SFT / eval-only 命令、多节点、指标与 checkpoint 落点、分阶段失败定位 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/10_slime_end_to_end_iteration_analysis|端到端迭代时序]] | 同步、one-stage async（发布间隔 1、2）与 fully-async 叠加下的版本年龄、发布栅栏与权重 commit；各入口的 checkpoint 切口 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis|Ray 控制面]] | GPU 怎么放、actor 怎么起、谁拥有服务与生命周期 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/12_slime_sample_datasource_analysis|Sample 与 DataSource]] | prompt/group/sample/rollout/train batch 如何转换且不丢语义 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/13_slime_sglang_rollout_engine_analysis|SGLang rollout 引擎]] | 请求并发、动态过滤、abort 与 partial 回收、streaming、fully-async 与 ABORTED 回队条件、规则奖励打分器 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/14_slime_megatron_training_analysis|Megatron 训练后端]] | DP 调度与 THD/CP 打包、actor/ref/teacher/critic 的 tag 切换、logprob、advantage、optimizer step 的内部路径；优化器状态重置与参数冻结 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/15_slime_loss_parallelism_analysis|Loss 与并行归一化]] | GRPO/PPO/GSPO/CISPO 估计器、rollout 均值 reducer 与缩放链、DP/CP/micro-batch 不变量 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/16_slime_weight_sync_analysis|权重同步]] | 四种 updater 与数据面、HF 逻辑重组与分桶、暂停窗口里的提交、共卡 IPC 与 MoE 定向路由、增量版本链 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/17_slime_train_inference_consistency_analysis|训推一致性]] | 六层训推一致性证据链、top-p 与路由重放、对齐替换层、TIS/ICEPOP/MIS 校正 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/18_slime_fault_tolerance_observability_analysis|容错与可观测性]] | 故障域局部恢复、更新边界上的 engine 重建、续训共同切点、取证与 CI 分层；指标落点与 x 轴 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/19_slime_rollout_backend_extension_analysis|rollout 后端扩展]] | rollout 数据面扩展、rollout buffer 插件及其配置、external SGLang、完整 backend 替换边界 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/20_slime_on_policy_distillation_analysis|On-Policy 蒸馏]] | 两种核心 OPD 模式、独立 teacher server 协议、reverse-KL advantage |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/21_slime_speculative_decoding_mtp_analysis|投机解码与 MTP]] | EAGLE/draft、在线 MTP 训练、Megatron 补丁中的 MTP 目标与梯度隔离、draft 发布与接受率闭环 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/22_slime_low_precision_training_rollout_analysis|低精度训练与 rollout]] | 七个精度环节；FP8 分块 / per-tensor / UE8M0 与 INT4 分组、fake-QAT 语义、KV cache、离线转换工具的产出与消费 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/23_slime_model_architecture_extension_analysis|模型架构扩展]] | custom provider、ModuleSpec/HF wrapper、双向权重映射；`--qwen-gdn-backend`、导入/导出与模型支持矩阵 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/24_slime_agent_workflow_examples_analysis|Agent workflow 示例]] | adapter、trajectory、tool/sandbox、fan-out 与 coding agent |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/25_vime_vllm_backend_support_analysis|vime vLLM 后端审计]] | vime 如何保留 slime 上层、替换为 vLLM/vllm-router，以及逐能力支持度与缺口 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/26_slime_multimodal_vlm_path_analysis|多模态 VLM 路径]] | 图像占位、processor、visual token 与训练特征；mRoPE 位置推进规则、CP 下视觉特征注入、多模态成本账 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/27_slime_evaluation_path_analysis|评估路径]] | 多数据集配置、独立采样口径与 9 个 `--eval-*` 覆盖参数、eval 指标与共享服务时序 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/28_slime_sft_path_and_loss_mask_analysis|SFT 与 loss mask]] | 离线对话变成 Sample、assistant mask、截断与 SFT loss |
| 3 | [[02_engineering/04_posttrain_frameworks/slime/30_slime_rollout_optimization_analysis|rollout 优化]] | 有效训练吞吐、容量/关键路径账本与负收益反例 |
| 3 | [[02_engineering/04_posttrain_frameworks/slime/31_slime_posttraining_stability_analysis|后训练稳定性]] | 数据、策略版本、估计量/数值、基础设施四控制环与判别实验；指标出现条件与结构性零值、NaN 检查与跳步、IMA 与 Ray 调试入口 |

## 阅读术语

术语只在 owner 页定义，本节只指路：

- driver、`RolloutManager`、full+disk 更新由训练组持有的版本 → [[02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis|Ray 控制面]]；weight updater 与权重 commit 的完成边界 → [[02_engineering/04_posttrain_frameworks/slime/16_slime_weight_sync_analysis|权重同步]]。
- rollout round 与 `Sample.rollout_id` 的粒度区别 → [[02_engineering/04_posttrain_frameworks/slime/12_slime_sample_datasource_analysis|Sample 与 DataSource]]；对应的归约口径 → [[02_engineering/04_posttrain_frameworks/slime/15_slime_loss_parallelism_analysis|Loss 与并行归一化]]。
- one-stage async 与 fully-async → [[02_engineering/04_posttrain_frameworks/slime/10_slime_end_to_end_iteration_analysis|端到端迭代时序]]。
- 并发信号量与动态过滤、partial 回收点 → [[02_engineering/04_posttrain_frameworks/slime/13_slime_sglang_rollout_engine_analysis|SGLang rollout 引擎]]；`--use-routing-replay` 与 `--use-rollout-routing-replay` → [[02_engineering/04_posttrain_frameworks/slime/17_slime_train_inference_consistency_analysis|训推一致性]]。

## Knowledge Gaps

### 待 planning 定归属的能力

以下能力在基线源码中存在，但本目录还没有讲清机制的 owner 页；表中只给源码入口和最接近的现有页，归属与是否新建页面待 `planning-codebase-analysis` 决定。

| 能力 | 源码入口 | 最接近的现有页 |
|---|---|---|
| PD 分离：Mooncake 传输、router PD 模式、bootstrap、外部 PD | `docs/en/advanced/pd-disaggregation.md`；`slime/backends/sglang_utils/disaggregation.py`；`tests/test_glm4.7_30B_A3B_pd_mooncake.py`、`tests/test_qwen3.6_35B_A3B_pd_mooncake.py`、`tests/test_qwen3_4B_external_pd.py` | [[02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis|Ray 控制面]]、[[02_engineering/04_posttrain_frameworks/slime/19_slime_rollout_backend_extension_analysis|rollout 后端扩展]]、[[01_mooncake_architecture_overview_analysis|Mooncake]] |
| 依赖补丁合同与镜像闸门 | `docker/Dockerfile`（`ARG PATCH_VERSION=latest`；`ARG ENABLE_SGLANG_PATCH=1` 时依次应用 `sglang.patch`、`sglang-top_p.patch`、`sglang-release_hicache.patch`、`sglang-pull_weights.patch`、`sglang-deterministic.patch`；Megatron 侧 `megatron.patch` 与可选的 `megatron-sglang-aligned.patch`）；补丁目录 `docker/patch/latest/` | [[02_engineering/04_posttrain_frameworks/slime/16_slime_weight_sync_analysis|权重同步]]、[[02_engineering/04_posttrain_frameworks/slime/17_slime_train_inference_consistency_analysis|训推一致性]]、[[02_engineering/04_posttrain_frameworks/slime/21_slime_speculative_decoding_mtp_analysis|投机解码与 MTP]]、[[02_engineering/04_posttrain_frameworks/slime/22_slime_low_precision_training_rollout_analysis|低精度训练与 rollout]] |
| 平台：ROCm/AMD | `docker/Dockerfile.rocm`、`docker/amd_patch/`、`docs/en/platform_support/amd_tutorial.md`；ROCm 走 `slime/utils/accelerator/cuda.py` 中按 `torch.version.hip` 区分的分支 | [[02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis|Ray 控制面]]、[[02_engineering/04_posttrain_frameworks/slime/22_slime_low_precision_training_rollout_analysis|低精度训练与 rollout]] |
| 平台：Ascend NPU（旧版、非主线） | `docker/npu_patch/` | [[31_cuda_ascend_posttraining_stack_comparison|CUDA–Ascend 后训练栈对照]]（钉定版本矩阵与主线失配） |
| 平台：GB10 | `docker/Dockerfile.gb10`、`docker/NOTES_GB10.md` | [[02_engineering/04_posttrain_frameworks/slime/02_slime_quickstart_and_configuration_guide|快速上手与配置]] |
| 平台：MUSA | `slime/utils/accelerator/musa.py`；`slime/utils/accelerator/__init__.py::_register_builtin_backends` 注册 `cuda`（`nccl`）与 `musa`（`mccl`） | [[02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis|Ray 控制面]] |
| GLM-5 DSA 训练：sparse MLA、indexer、tilelang 反向、PP 切分约束 | `slime_plugins/models/glm5/glm5.py`；`slime_plugins/models/glm5/ops/indexer.py`、`slime_plugins/models/glm5/ops/sparse_mla.py`、`slime_plugins/models/glm5/ops/tilelang_indexer_bwd.py`；`tests/test_glm5_indexer_q_norm.py`、`tests/test_glm5_indexer_short_context.py`；`--freeze-indexer` | [[02_engineering/04_posttrain_frameworks/slime/14_slime_megatron_training_analysis|Megatron 训练后端]]、[[02_engineering/04_posttrain_frameworks/slime/17_slime_train_inference_consistency_analysis|训推一致性]]、[[20_glm5_architecture_deepdive|GLM-5 架构]] |
| 多节点大 MoE 配方 | `scripts/run-glm5.2-744B-A40B.sh` | [[02_engineering/04_posttrain_frameworks/slime/02_slime_quickstart_and_configuration_guide|快速上手与配置]]（只覆盖多节点启动变量） |

### 明确排除，不设 owner

| 项 | 基线证据 | 排除理由 |
|---|---|---|
| `--http-proxy` | `slime/utils/arguments.py` 的 `add_network_arguments` 注册；`slime/`、`slime_plugins/`、`train.py`、`train_async.py`、`examples/`、`tools/` 中无 `http_proxy` 读取方 | 死参数 |
| `--min-batch-collection-ratio` | `slime/utils/arguments.py` 的 `add_rollout_buffer_arguments` 注册；全仓无 `min_batch_collection_ratio` 读取方 | 死参数 |
| `--foo-bar`、`--sglang-foo-bar` | `slime/backends/sglang_utils/arguments.py` 中解释 dest 推导的注释 | 注释示例，不是 flag |
| `learnable_softmax_attention.py`、`flash_dot_product_attention.py` | `slime_plugins/models/flash_dot_product_attention.py::FlashDotProductAttention` 无任何引用；`slime_plugins/models/learnable_softmax_attention.py` 只被前者导入 | 未引用的遗留模块 |
| `tools/convert_to_hf.py` | 导入 `slime.backends.megatron_utils.update_weight_utils`，该模块不存在（权重更新代码在 `slime/backends/megatron_utils/update_weight/`） | 已损坏的工具 |
| `slime/utils/external_utils/*` | `slime/utils/external_utils/command_utils.py` 模块 docstring 自述"不属于 slime 框架本身"，供启动作业与测试；消费方为 `tests/test_*`、`examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py` 与 `docs/en/developer_guide/ci.md` | 测试与启动 harness |

## Related Pages

- [[02_engineering/04_posttrain_frameworks/index|后训练框架目录]] — 本域上级入口。
- [[02_engineering/04_posttrain_frameworks/30_rl_framework_comparison|RL 框架对照]] — 跨框架差异与证据边界。
- [[02_engineering/04_posttrain_frameworks/verl/index|verl 知识域]] — 相邻框架的独立入口。
