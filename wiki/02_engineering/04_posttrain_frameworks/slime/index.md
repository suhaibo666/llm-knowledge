---
title: "slime RL 后训练框架 — 知识地图"
---

# slime RL 后训练框架 — 知识地图

> **源码基线**：`THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`（2026-08-12）；vime 衍生实现另见该页基线。
> **最近更新**：2026-09-10。
> **系列规模**：23 篇内容页，加本索引共 24 个 Markdown 文件。新增多模态、评估、SFT 三个独立专题；源码核验范围和运行验证边界见各页。

本表是本目录唯一页面入口清单。先读 01 架构和 02 配置，再沿 10–19 进入数据、训练与服务机制；20–28 按能力选读，30–31 用于诊断。

## 页面入口

| 段 | 页面 | 解决的问题 |
|---|---|---|
| 0 | [[02_engineering/04_posttrain_frameworks/slime/01_slime_architecture_overview_analysis]] | 整体软件架构、设计取舍、职责与关键接口 |
| 0 | [[02_engineering/04_posttrain_frameworks/slime/02_slime_quickstart_and_configuration_guide]] | 从脚本、CLI、Megatron/SGLang YAML 到源码入口怎样对应 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/10_slime_end_to_end_iteration_analysis]] | 同步、one-stage async 与 fully-async 叠加的调度与权重 commit |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis]] | GPU 怎么放、actor 怎么起、谁拥有服务与生命周期 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/12_slime_sample_datasource_analysis]] | prompt/group/sample/rollout/train batch 如何转换且不丢语义 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/13_slime_sglang_rollout_engine_analysis]] | 请求并发、RM、动态采样、partial、streaming、fully async 怎样实现 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/14_slime_megatron_training_analysis]] | actor/ref/teacher/critic、logprob、advantage、optimizer step 的内部路径 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/15_slime_loss_parallelism_analysis]] | GRPO/PPO/GSPO/CISPO、reducer、DP/CP/PP/VPP 不变量 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/16_slime_weight_sync_analysis]] | 四条权重路径、HF 逻辑重组、训推 topology 转换、共卡 CUDA IPC 与 MoE 定向路由 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/17_slime_train_inference_consistency_analysis]] | 四层训推一致性、TIS/MIS 与 routing replay |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/18_slime_fault_tolerance_observability_analysis]] | engine recovery、debug replay、trace/profiling、CI 分层 |
| 1 | [[02_engineering/04_posttrain_frameworks/slime/19_slime_rollout_backend_extension_analysis]] | rollout 数据面扩展、external SGLang、完整 backend 替换边界 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/20_slime_on_policy_distillation_analysis]] | 两种核心 OPD 模式、独立 teacher server 协议、reverse-KL advantage |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/21_slime_speculative_decoding_mtp_analysis]] | EAGLE/draft、在线 MTP 训练、同步与接受率闭环 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/22_slime_low_precision_training_rollout_analysis]] | BF16/FP8/INT4、KV cache 与量化权重提交 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/23_slime_model_architecture_extension_analysis]] | custom provider、ModuleSpec/HF wrapper、双向权重映射 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/24_slime_agent_workflow_examples_analysis]] | adapter、trajectory、tool/sandbox、fan-out 与 coding agent |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/25_vime_vllm_backend_support_analysis]] | vime 如何保留 slime 上层、替换为 vLLM/vllm-router，以及逐能力支持度与缺口 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/26_slime_multimodal_vlm_path_analysis|多模态 VLM 路径]] | 图像占位、processor、visual token、训练特征与 mRoPE 对齐 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/27_slime_evaluation_path_analysis|评估路径]] | 多数据集配置、独立采样口径、eval 指标与共享服务时序 |
| 2 | [[02_engineering/04_posttrain_frameworks/slime/28_slime_sft_path_and_loss_mask_analysis|SFT 与 loss mask]] | 离线对话变成 Sample、assistant mask、截断与 SFT loss |
| 3 | [[02_engineering/04_posttrain_frameworks/slime/30_slime_rollout_optimization_analysis]] | 有效训练吞吐、容量/关键路径账本与负收益反例 |
| 3 | [[02_engineering/04_posttrain_frameworks/slime/31_slime_posttraining_stability_analysis]] | 数据、策略版本、估计量/数值、基础设施四控制环与判别实验 |

## 阅读术语

- driver 指 `train.py` / `train_async.py` 进程；RolloutManager 保留类名，weight updater 指各权重更新实现。完整磁盘更新的版本由训练组持有，见 11；权重 commit 的完成边界见 16。
- rollout round 指一次 `generate` 产出的一批；`Sample.rollout_id` 标记的逻辑执行可包含多个训练片段，不能与一轮批次混同。对应数据与归约口径见 12、15。
- one-stage async 指 `train_async.py` 的阶段重叠；fully-async 是其上的 rollout 函数替换，详细时序统一见 10。
- 并发信号量与动态过滤分开命名；partial 回收点与权重 commit 分开命名。两个 routing replay 开关始终写全 `--use-routing-replay` / `--use-rollout-routing-replay`，见 13、17。

## Related Pages

- [[02_engineering/04_posttrain_frameworks/index|后训练框架目录]] — 本域上级入口。
- [[02_engineering/04_posttrain_frameworks/30_rl_framework_comparison|RL 框架对照]] — 跨框架差异与证据边界。
- [[02_engineering/04_posttrain_frameworks/verl/index|verl 知识域]] — 相邻框架的独立入口。
