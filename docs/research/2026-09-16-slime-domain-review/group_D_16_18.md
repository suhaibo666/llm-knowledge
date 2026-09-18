# D 组：16 权重同步、17 训推一致性、18 容错与可观测性

评审者：独立评审（未参与写作），只读。基线 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`；Megatron 用 `git show 1dcf0daf` 核对；slime 仓内 `docker/patch/latest/*.patch` 在基线可核。

## A. 判定表

| page | profile(fit?) | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | feature-check | host | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | feature ✓ | pass | pass（四条数据面都走到 continue/reload） | pass | transform, layout, timing, coupled-planes | pass（fc1 重排、[300][260+100]、world 7 / offset 1·3、越界组 world 3、MoE 12 对 24 份及 batch0/1、增量 2/8、8/14 字节、pull 四种情形；全部独立复算，与生成器一致） | 3/3：`expert_routing.py::_pack_expert_transfer_batches`、`sglang-pull_weights.patch::local_checkpoint.pull`、`actor_group.py::RayTrainGroup._reload_rollout_weights_from_disk`（另抽约 20 项，1 处不精确） | FAIL §2.2.8 依赖边界 | minor | **REJECT** |
| 17 | feature ✓ | pass | pass | pass | transform, layout, timing | 数值 pass（−1.440/−1.313/0.127、工具 token 得 0、BF16 1 对 1.0078125、[0.4,1.6] 加 mask [1,0]、seq 2、geo 1.414）；图①的 top-k 图注错 | 3/3：`ppo_utils.py::_VocabParallelLogProbEntropy.forward`、`mis.py::compute_mis_weights`、`deterministic_route_kernels.py::_scatter_routes_backward_kernel`（另抽约 17 项，1 处错误） | FAIL §2.2.3 top-k 边界；§2.2.5/§2.2.8 依赖边界 | nit | **REJECT** |
| 18 | feature ✓ | pass | pass | pass | timing | pass（上界 15/45、CI 等 20；默认 60/65；保存点 1·3·5；撕裂 start 4、load(3) 归零；退回与异步都是 2、load(1)；生成器与源码一致） | 3/3：`health_monitor.py::_kill_engine`、`rollout.py::_try_ci_fault_injection`、`data_source.py::RolloutDataSource.load`（另抽约 15 项，0 错） | pass（观测 sink 只提及，minor） | minor | **PASS**（带 minor） |

## B. 问题（按严重度）

- **[major] 16 §2.2.8/§2.2.4/§4.2/§2.2.6 与图③：只把 `/pull_weights` 归给补丁，没有写版本闸门。** 页面说"slime 无法证明 SGLang 何时完成 GPU 拷贝"，图③说"返回时拷贝可能仍在排队"，两处都与基线补丁不符。
  - `docker/patch/latest/sglang-deterministic.patch` 在 `model_runner.py` flattened-bucket 的 `load_weights` 之后加了 `torch.cuda.synchronize()`，注释说的正是生产者缓冲被复用的竞争。
  - `sglang.patch` 新增 `/post_process_weights`；上游 `/weight_version` 直接返回 404，补丁改写为返回版本号（CI 读回依赖它）。
  - `docker/Dockerfile` 底座 `v0.5.15.post1-cu129`，`PATCH_VERSION=latest`，逐个 `git apply --check`，并有 `ENABLE_SGLANG_PATCH` 开关（GB200/GB300 可跳过）；`docker/patch/v0.5.15.post1/` 里没有 deterministic 补丁。
  - 修：加一张"端点/语义 → 上游还是哪个补丁 → 镜像闸门"表，图③④标出补丁侧。
- **[major] 17 §2.2.3/§1.3/§2.3/§5.1/§6 与图①图注："top-k 不重放、差异来自归一化域"与补丁矛盾。** `sglang-top_p.patch` 的 `logprob.py::_top_p_keep_mask_sorted` 保留 rank<top_k ∧ cumsum≤top_p ∧ min_p；`get_top_p_token_ids_from_probs` 和 `renorm_logprob_over_top_p` 都用这个集合。`GenerateState` 只在 top_p≠1 时请求 ids。结论：top_p≠1 时，返回的 ids 已包含 top-k 截断，重放覆盖了 top-k；top_p=1 且 top_k≠−1 时不请求 ids，采样器返回未截断的 `log(probs)`，两侧都在全词表域，没有假 δ，但用作 old policy/TIS 的并不是行为概率。修：按这两种情形重写。
- **[major] 17 §2.2.5/§2.2.8 门禁表：GLM-5 对齐栈没有写补丁交接、版本闸门和 skip 语义。**
  - `alignment/env.py` 里 `SGLANG_DEEPGEMM_BATCH_INVARIANT`、`SGLANG_DEEPEP_LL_PREFILL_STAGING` 等变量的消费方在 `sglang-deterministic.patch`；该补丁还加了 `enable_fp32_moe_router` 和 DeepEP `align_fp8_quantization` 透传。
  - `MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS` 由 `megatron-sglang-aligned.patch` 消费。
  - `_configure_batch_invariant` 注释写明 B300/cu130 镜像缺 deterministic 补丁。
  - `test_glm52_6layer_deterministic_e2e.py::_skip_reason` 发现栈或 aligned 补丁缺失时 `pytest.skip`，注释说缺补丁会发散到 ~1e-2。
  - `reproducibility.md` 明示需要 aligned 补丁。
  - 修：补交接表；门禁表注明"栈缺失时 skip 而非 fail"。
- **[minor] 17 图①与 §2.1 表 L2 行**：rollout 侧 −1.313 来自补丁的重归一，图上未标依赖侧（正文有）。
- **[minor] 17 §2.1 表 L3、§2.1.2、图②**："old 前向取 2 次 → clear_all_forward"只在 `can_reuse_log_probs_in_loss` 为假时成立；单步、`kl_coef` 默认 0、非 gspo 时跳过；维护的 R3 用例用 gspo。终态 cursor 不变，但应写条件。
- **[minor] 16 §2.2.2**："每个 rank 都各自 convert_to_hf"对整份磁盘不成立：`save_hf_model_direct_to_path` 传入 `should_convert_chunk=lambda _: is_writer_rank`。
- **[minor] 16、18 页头「主题」含论断**：16"每条都在 pause/flush 与 continue 之间提交"；18"不做全局事务……HTTP 重试只保证瞬态可用"。
- **[minor] 18 §2.2.6/§2.3「取证方式」轴**：W&B/TensorBoard 只作为载体提及，无开关（`--use-wandb`、`--wandb-*`、`--use-tensorboard`、`--tb-*`）和归属；`--custom-rollout-log-function-path` 能跳过默认日志，直接影响"以实际键集合为准"。
- **[nit] 16 §2.1.5 与图⑤**："v0 = hf_checkpoint"：主机 base 实际是补丁里的 `server_args.model_path`，默认等于 hf_checkpoint，但 `--sglang-config` 的 `model_path` 可改；应写相等前提，否则 checksum 失败。
- **[nit] 17 §2.1.2**："SGLang 确定性 top-k 用 sorted=False"出自 slime 注释（`routing_replay.py`），应标来源；Megatron 侧默认 sorted 已在 1dcf0daf 核实。
- **[nit] 18 §2.1.3 与图②**："tracker 仍指向 1（依赖侧）"可核实：Megatron `checkpointing.py::save_checkpoint` 在 `iter_finalize_fn` 里才写 tracker；可升为已核实，并加 Megatron 基线行。
- **重复与矛盾**：30 重复了健康检查默认值和 `perf/request/count`，11 重复了 updater 选择，但与本组一致，未发现矛盾。

## C. 缺失内容

| 内容 | 源码位置 | 建议归属 | 重要度 | 现状 |
|---|---|---|---|---|
| 补丁应用闸门与两套补丁目录 | `docker/Dockerfile`、`docker/patch/{latest,v0.5.15.post1}`、`docker/README.md` | 16 §2.2.8（镜像策略可链到 01/02） | 高 | 缺失 |
| `sglang.patch` 的 `/post_process_weights`、`/weight_version` | `docker/patch/latest/sglang.patch` | 16 | 高 | 缺失 |
| `sglang-deterministic.patch`：IPC 同步、FP32 router、DeepEP align、批不变算子 | `docker/patch/latest/sglang-deterministic.patch` | 16 §2.2.4 与 17 §2.2.5 | 高 | 全域缺失 |
| `megatron-sglang-aligned.patch` 与 GLM-5 门禁 skip 探针 | 同名补丁；`tests/test_glm52_6layer_deterministic_e2e.py::_skip_reason` | 17 | 高 | 全域缺失 |
| `sglang-release_hicache.patch`：按 tag 幂等的 release/resume、`release_hicache` | `docker/patch/latest/sglang-release_hicache.patch` | 13，16 与 18 加链接 | 中 | 全域缺失 |
| 指标 sink 与自定义日志函数 | `wandb_utils.py`、`tensorboard_utils.py`、`rollout.py::_log_rollout_data` | 18 | 中 | 只提及 |
| SGLang decode trace 分析器 | `tools/analyze_profile.py` | 18 §4.2 | 低到中 | 18 缺失（仅 01 提及） |
| 单侧逐位复现配方（无 CI 门禁） | `scripts/run-qwen2.5-0.5B-reproducibility.sh` | 17 §2.2.8 | 低到中 | 缺失 |
| delta 示例 | `examples/delta_weight_sync/` | 16 §2.2.7 | 低 | 缺失 |
| MIS 运行脚本 | `examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh` | 17 | 低 | 缺失 |
| 内存快照目录与打印工具 | `--memory-snapshot-dir`、`memory_utils.py::print_memory` | 18 §6 | 低 | 缺失 |
| trace 单测（未进 cpu-unittest）与 CI 运维 | `tests/utils/test_trace_utils.py`、`tests/ci/README.md`、`gpu_lock_exec.py` | 18 §2.2.7 | 低 | 缺失 |

## D. 环境缺口 / 无法核实

- 无 SGLang checkout：16 页 8 处 0b3bb0cb 行号链接未复核；`/get_weight_version` 是否与被改写的 `/weight_version` 共用处理函数无法确认；各补丁能否干净应用只在镜像构建时检查；CI 镜像 `slimerl/slime-test:latest` 实际用哪套补丁无法确认。
- DeepGEMM/DeepEP 分叉未检出：ep_gather 的 FP32 语义、`align_fp8_quantization` 未核。
- 运行时行为无法确认：`torch_memory_saver` 的 pause/disable 语义、PyTorch CUDA IPC 引用计数、SGLang `kill_process_tree` 对已退出 pid 的处理。
- 未运行 GPU/Ray/CI：注入用例的 15 秒上界、GLM-5 门禁在 CI 中是通过还是被 skip，无法观察。
- Megatron 只核了 tracker 写入与 `_compute_topk`；18 页 critic 续训推断涉及的 actor/critic 共用 `--save` 路径未检查。

## 协调者复核

- 17 top-k：属实。`docker/patch/latest/sglang-top_p.patch::_top_p_keep_mask_sorted` docstring 为 "Reproduces SGLang's sampler truncation (rank < top_k, cumulative prob within top_p, prob >= top1 * min_p) so replay sees the exact set the sampler keeps"，实现首行 `keep = ranks < top_ks.view(-1, 1)`；17 页第 256 行写"top-k 不在范围内"。
- 16 IPC 同步：属实。`docker/patch/latest/sglang-deterministic.patch` 含 `torch.cuda.synchronize()`，注释说明 RPC 响应释放生产者侧 IPC buffer 后存储被复用会破坏刚加载的权重；`assets/slime_weight_sync_planes.svg` 有"拷贝可能仍…"的表述，未标补丁闸门。
- 17 GLM-5 门禁：属实。`tests/test_glm52_6layer_deterministic_e2e.py::_skip_reason` 注释写明缺 `megatron-sglang-aligned.patch` 时门禁发散约 1e-2，因此 skip。
