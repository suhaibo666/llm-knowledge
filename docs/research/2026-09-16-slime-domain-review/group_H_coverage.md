# H 组：slime 域覆盖度、跨域一致性与基线漂移

评审者：独立覆盖度审计（未参与写作），只读。基线 `slime@681b3adc`；漂移对照本机 `/Users/suhaibo/97-llm/slime@4c193f1f`（后 26 个提交，含 v0.3.2）。

"explained"表示该页讲清了机制，"mentioned"表示只提到，"absent"表示没有覆盖。

## 1. 覆盖矩阵摘要

| 能力/场景 | 源码证据 | 当前状态 | 建议 | 重要度 |
|---|---|---|---|---|
| **部署与平台** | | | | |
| PD 分离（Mooncake、router PD 模式、bootstrap、外部 PD） | `pd-disaggregation.md`；`rollout.py::_start_router(has_pd_disaggregation)`；`test_{glm4.7_30B_A3B,qwen3.6_35B_A3B}_pd_mooncake.py`、`test_qwen3_4B_external_pd.py` | mentioned：11 讲 bootstrap 端口，13 讲旧参数的卡数算术，19 讲外部 PD 发现，30 讲调参；Mooncake 传输后端、PD 模式、3 个测试 absent | 把 13 §4.2 扩成 PD 的 owner：何时该用；`disaggregation_transfer_backend`；同一模型不能混 regular；prefill 和 decode 都接收权重；`pd_*` 计时依赖补丁；3 个门禁测试。11/19/30 只留链接 | 高 |
| 依赖补丁合同 | Dockerfile 依次应用 sglang、top_p、release_hicache、pull_weights、deterministic 五个补丁，再加 megatron 与 aligned 补丁 | 零散：16 有 pull_weights，17 有 top_p 和 compute_topk，22 有 QAT；hicache、deterministic、`/post_process_weights`、PD 计时 absent | 新页（建议 `32_slime_deployment_stack_platforms_and_patches_analysis.md`）：镜像补丁本身就是功能合同，附"补丁→依赖它的特性"矩阵 | 高 |
| ROCm、GB10、Ascend NPU | `Dockerfile.rocm*`、amd_patch、`amd_tutorial.md`、`torch.version.hip` 分支；`NOTES_GB10` 记录 15 个阻塞项；npu_patch 钉在 slime v0.2.2，改动的 `fsdp_utils` 在基线已不存在 | absent（只有 11 提到 ROCm 跳过 NUMA 绑定、22 提到 NPU patch） | 放进上面的新页；NPU 标为旧版、非主线 | 中高 |
| 多节点大 MoE 配方 | `run-glm5.2-744B-A40B.sh` 的 HOSTFILE 与 IB/NVSHMEM 环境变量；DeepEP 分发 buffer 须 ≥ cuda graph bs × draft token 数 | absent（02 只讲单机 glm4-9B） | 新页加一节"多节点启动与网络" | 中 |
| `build_conda.sh` 与 conda-ci | 按与 Dockerfile 相同顺序打补丁 | absent | 新页写一句 | 低 |
| **模型与算子** | | | | |
| GLM-5 DSA 训练（sparse MLA、indexer、跨层 index sharing、tilelang fwd/bwd） | `glm5/glm5.py`（1073 行）与 `ops/*`；`test_glm5_indexer_*`；示例文档要求每个 PP stage 从 computing layer 开始 | mentioned：14 讲 allgather CP，17 讲对齐门禁，22 讲量化名 | 新页（建议 `29_slime_glm5_dsa_sparse_attention_training_analysis.md`）：PP 切分约束、`--allgather-cp`、`--freeze-indexer`、KV FP8 QAT | 高 |
| Qwen GDN 后端 | `--qwen-gdn-backend fla/flashqla`；FlashQLA 要求 torch≥2.8、SM90+、CUDA≥12.8 | 23 §5.3 只讲 cu_seqlens | 补入 23 §5.3 | 中 |
| 模型支持矩阵 | 39 个 `scripts/models` preset、23 个 `run-*.sh`；Moonlight 与 Qwen3-30B 的 r3 门禁 | 23 §4.4 只有转换函数注册表 | 23 §4.4 加三列：preset 脚本、专用 plugin、CI 门禁 | 中 |
| `learnable_softmax_attention.py`、`flash_dot_product_attention.py` | 仓内无引用，来自 gpt-oss 提交 e0af6dc3 | absent | 排除，ledger 标"未引用遗留" | 低 |
| **训练与算法** | | | | |
| 奖励打分器语义 | `rm_hub/*`；`test_rm_*` | 13 §2.2.2 讲分派链；各打分器语义 absent | 13 加打分器表：deepscaler 无 `</think>`/`###Response` 时恒返回 0；dapo 返回 dict（±1），须配 `--reward-key score`；ifbench 在 import 时自动 git clone 并 pip install | 中 |
| 参数冻结 | `freeze_model_params`；only-train 与 freeze 名单互斥；`--freeze-indexer` 按 PP stage 逐段校验；`test_model_provider_freeze` | mentioned：17、20 | 14 §4 新增小节 | 中 |
| PPO/critic | 14、15、10 | explained；缺 critic 输出层 reinit、value 不乘温度（`test_value_temperature`）、3 个 PPO E2E | 补入 14 §2.2.1 与门禁表 | 低中 |
| `--use-rollout-entropy`、`--opd-teacher-ckpt-step` | `model.py` 的 `with_entropy`；`actor.py` 的 teacher `ckpt_step` | absent | 分别加到 14 §6、20 §8 | 低 |
| **观测与调试** | | | | |
| W&B/TensorBoard 配置 | 15 个 flag；`compute_rollout_step`（开 `--wandb-always-use-train-step` 后 x 轴换成训练步）；run 名随机后缀默认开 | absent | 18 §2.2.6 加"指标落点与 x 轴"小节 | 中 |
| 指标目录 | rollout/、train/、perf/、eval/ 四类前缀 | 分散：31 §4.1、30 §7.2、16、21、27 | 在 18 按前缀建索引，只链各 owner，不复制内容 | 中 |
| 自定义日志钩子、`--log-multi-turn` | customization.md §14；`log_multi_turn_data` | absent | 加到 18 §2.2.6 | 低中 |
| FAQ 失败模式 | `qa.md` 共 13 条 | 部分在 02 §7；`--model-name` 多进程读文件竞争、get_model_info 端口冲突、stop token 缺失、IMA、Inductor 缓存报错、NaN 跳步 absent | 补成 02 §7 表格行；IMA 与 Ray distributed debugger 加到 31 §8 | 中 |
| CI、profiling、trace | ci/profiling/trace 三份文档 | explained：18 §2.2.7、§4.2 | 维持 | — |
| **数据、扩展与 checkpoint** | | | | |
| rollout buffer 插件 | `--rollout-buffer-url/--rollout-task-type/--fetch-trajectory-retry-times`、`base_generator.py` | 19 §3.4 讲机制，flag 与 generator 契约 absent | 19 §3.4 加配置行 | 低 |
| eval flag | `--eval-*-key` 与各采样参数 | 27 §5 讲语义但未点名 flag；`--eval-max-context-len` 只有 multi_agent 示例读取 | 27 §5 点名 | 低 |
| checkpoint 导入/导出 | `fp8_cast_bf16.py`、K2 INT4 转 BF16；`--save-hf` 与 `convert_torch_dist_to_hf(_parallel)` 区别；词表 padding | mentioned：02 只讲导入，14/23 只讲 saver 内部；导出流程 absent | 02 §4.2 加导入/导出路线表 | 中 |
| `tools/convert_to_hf.py` | 导入不存在的 `megatron_utils.update_weight_utils` | absent | 排除，ledger 标 broken | 低 |
| 应排除的 flag 与内部件 | `--http-proxy`、`--min-batch-collection-ratio` 无读取方；`--foo-bar` 类是注释伪影；grad-coalesce 猴补已由基线后提交 c403335d 删除；`external_utils` 是测试 harness；仓内 `.claude/skills` | absent | 排除，并在 ledger 分类记录 | 低 |

**对协调者机械结果的修正：** `strands_sglang` 在 24 §7 有讲解（写作"Strands 示例"），关键词扫描漏检；三个参数文件按 AST 去重后是 218 个 flag，不是 241；`debug_rollout_only`、`log_correct_samples`、`log_passrate`、`n_samples_per_eval_prompt`、`only_train_params_name_list` 以 snake_case 被提到，不算缺失。

## 2. 跨域重复与矛盾

1. **30_rl_framework_comparison §7 两处说过头。** "checkpoint 队列状态由 DataSource save/load 保存"：`RolloutDataSource.save` 只写 `sample_offset`、`epoch_id`、`sample_group_index`、`sample_index`、`metadata`，partial 和 fully-async 队列都不进 checkpoint（与 18 一致）。"engine 端比对版本号"：只在 full+disk 且开 `--ci-test` 时成立（见 16）。
2. **13_opd_infra 把 slime 标成 `k1`/`k3`，不对。** `apply_opd_kl_to_advantages` 只计算 `student_logp − teacher_logp`，OPD 无估计量选项；k1–k3 是 `--kl-loss-type`，属于对 ref 的 KL。
3. **32_opd_framework 的 slime 条目三处问题。** 写"两种 teacher 模式二选一"，漏了 20 §5.2 的独立 Megatron teacher server；写"支持异步"，但无 OPD 与 `train_async` 组合的证据；§3.2 与 20 §6 重复，应缩成一句并链到 20，同时补"SGLang 模式必须同时配 custom-rm、postprocess、rm_url"陷阱。
4. **课程页与 25_vime 的描述过时。** 课程 L116 和 25_vime 的 Related Pages 写"四层一致性"，17 实际是 L0–L5 六层；课程 D08c 把"rollout 分母、DP×CP whitening"归给 31，但 31 页头声明归约属于 15。
5. **基线分裂。** 课程页头称 D01 固定在 `681b3adc`，但 `01_posttraining_frontier_map` §3 钉的是 `aaf5c20`（早 31 个提交）；D05 `01_posttraining_infra_mechanism` 沿用同一旧基线；`24_agentic_rl_algorithm` 引用的 `fully_async_rollout.py:178-189` 行号对应 aaf5c20，在 681b3adc 上是 199-204（结论仍成立）。
6. **GLM-5 两个理论深挖页只转述论文。** 25_glm5_training_stability §6 写"服务器周期发心跳、重试自动重路由"；24_glm5_agentic_rl 讲 PD/FP8/MTP；两页对 13/18/21/22 零链接。开源基线实际是 driver 侧 `RolloutHealthMonitor` 轮询，失败即整组标死，到下次 `update_weights` 才重建。建议加"论文 vs 开源基线"注记并链到 18。
7. **缺反链。** 07_training_reliability/12 讲"每次推送后重置优化器"，没链 14 的 `--reset-optimizer-states`/`--use-stateless-adam`；01_posttraining_infra §6 列了 slime 四种权重传输方式，没链 16；31_cuda_ascend 完全没提 slime，而 `docker/npu_patch` 正是其"整套版本矩阵"论点的实例；megatron-lm/30 与 slime 16/17 是不同系统实现，不算重复。

## 3. 计数/索引一致性

- 一致：slime 目录 find 为 24（23 内容页 + index），与 slime/index、后训练框架 index、wiki/index 三处一致；verl 17，一致。
- 后训练框架目录计数不对：find 为 53，wiki/index.md 写 52（DORA 页 2f6a04d 未回写主索引）；同表下方注释"后训练框架 47→49"也与表里 52 自相矛盾。
- vime 重复计数：后训练框架 index 系统表里 slime 的"23 篇"已含 25_vime，又单列 vime"1 篇"。

## 4. 基线漂移影响清单（681b3adc→4c193f1f，26 个提交）

| 提交 | 受影响的页与结论 |
|---|---|
| d8ff51c4（拆分 rollout.py）、daebd20b（删除 rollout_validation.py） | 11 §2.2、§3.3、§5.1 的 `rollout.py::ServerGroup/RolloutServer/_start_router/_allocate_…/_resolve_sglang_config` 锚点已移到 `sglang_utils/{engine_group,deployment,sglang_config}.py`；13 §4.2；18 恢复路线；02 §7 引用的 `rollout.py:1274-1282` 与 `rollout_validation` |
| 624b824a（新建 `slime/observability/`） | 15 的 `cp_utils::reduce_train_step_metrics`；12/14/15/17/30 的 `data.py::log_rollout_data`；18 的 `utils/trace_utils`；30/31 的 `rollout.py::compute_*metrics*` |
| 7fc5715c（删死代码） | 18 §6 的 `--profile-target` 已删；16 §5.1/§5.4 与 21 §6.1/§9 引用的 `_named_params_and_buffers_global/_vanilla`、`translate_gpu_to_cpu` |
| 7e4ac3be（抽出 `create_weight_updater`） | 16 已加注；11 §2.2"由 `MegatronTrainRayActor.init` 选 updater"未加注 |
| a37dd90b、8f20503f、d8ad1b57（测试删除与移动） | 15 引用的 `test_chunked_gae`、17 引用的 `test_deepgemm_moe_forward`、28 引用的 `test_mask_utils` 已删；17 的 `compare_glm52_layerwise` 与 12 的 `_fanout_test_helpers` 已移到 tests/ |
| 4c1ab402（外部流式 rollout） | 13 §5.2/§5.4"只支持累计 SSE"失效；流式按单请求取消，终止 chunk 上的 top-p 与路由专家元数据随取消丢失，影响 17 重放边界和 19 external 路径 |
| 4c193f1f（abort 经 router 扇出到各 worker） | 24 §6.1"best-effort abort"：基线直接向 router 发 abort，可能不释放 KV |
| a0d6d26a（eval-only 跳过 optimizer） | 01 §5.4"eval-only 成本不低"需收窄 |
| 16c15fc2（温度须 >0）、41014d1f（两类 dump 路径不得相同） | 13/14/17 配置行；17、18 约束表 |
| 2fa9a442、876cd89b、e593fa0a（UE8M0 修复、ROCm INT4、MUSA） | 22 §5.1/§5.2"INT4 kernel 仅 CUDA"；11 的设备绑定与环境变量注入现改走 accelerator 抽象 |
| 1a3fb0a6、3778dbf6（文档修正、v0.3.2） | 16 §5.2 引的"delta+nccl 文档"已修正；17 引用的 `patch/latest/*` 已进入 v0.5.15.post1 稳定目录 |
| 045310b2、1da1bb19、08160d3f | 15/20/22 已加注，无需处理 |

## 5. 工具/配置缺口

- watchlist 的 slime 路径错：`checkout: ../slime` 解析为 `/Users/suhaibo/96-knowledge/slime`，不存在；应按 vllm 条目写法改为 `../../97-llm/slime`（该 checkout 含 681b3adc）。否则 slime 行号引用只能报 unresolved_repo，radar 也算不出漂移。
- 缺 vime 与 sglang 的 checkout：vime 无 watchlist 条目、本机无 checkout；`../sglang` 同样不存在，16/21/23 跨仓引用无法复核。
- 缺 `docs/coverage/slime.yaml`：flag 独立枚举没有持久化。建议建 ledger，把 218 个 flag 逐个映射到 owner 页或排除项，并标注 dead（`--http-proxy`、`--min-batch-collection-ratio`）、artifact（`--foo-bar`、`--sglang-foo-bar`）、pass-through（`--sglang-pipeline-parallel-size`、`--padded-vocab-size`），接入 `check_coverage.py`。
