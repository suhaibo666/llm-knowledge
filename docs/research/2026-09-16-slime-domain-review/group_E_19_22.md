# E 组：19 rollout backend 扩展、20 OPD、21 投机解码与 MTP、22 低精度

评审者：独立评审（未参与写作），只读。基线 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`；Megatron 用 `git show 1dcf0daf` 核对。

## A. 判定表

| page | profile（是否匹配） | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check（n/3 + 锚点） | profile-check | host | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 19 | mechanism（匹配） | pass（§3.3 hooks 的 why 较薄） | pass（§5：`RolloutManager.__init__`→`generate_and_rm`→`apply_rollout_sample_hooks`→`async_rm`→`_get_rollout_data`→按 DP 切分） | pass | none | n/a | 3/3：`slime/ray/rollout.py::RolloutManager.__init__`、`sglang_utils/external.py::ExternalRolloutServer.recover/offload/onload*`、`rollout/forge_load.py::_resolve_path` | pass（external、buffer 的失败边界有缺口） | minor | PASS（补 B9–B11） |
| 20 | feature（匹配） | FAIL §6 | pass（§6 有一条条件边写错） | pass | transform | FAIL §4/§6（§5.2 TP 采样图合格） | 3/3：`loss.py::apply_opd_kl_to_advantages` 及调用顺序、`on_policy_distillation.py::post_process_rewards`、`server/megatron_server.py::_build_http_app.update_from_disk`（加抽 `actor.py::train_actor` 学生项来源，部分不符） | FAIL | minor | REJECT |
| 21 | feature（匹配） | pass | FAIL §5 | pass | transform、reduction、训练梯度 | FAIL | 2/3：`update_weight/common.py::_named_params_and_buffers_global` ✓、`types.py::Sample.SpecInfo` + `rollout.py::_compute_spec_metrics` ✓、`ci_utils.py::check_mtp_only_grad` + MTP 测试 ✗ | FAIL | minor | REJECT |
| 22 | feature（匹配） | pass | pass | pass | transform、layout | FAIL §5.1/§5.2 | 3/3：`processors/quantizer_fp8.py::_quantize_param`、`kernels/fp8_kernel.py::_blockwise_cast_to_fp8_triton`、`processors/__init__.py::quantize_params` | FAIL | minor | REJECT |

## B. 问题清单（按严重度）

1. **[blocker] 21 §5.2/§5.3：误读 CI 测试结论。** 页面把测试读成"主 policy loss 被 mask 后不污染主干"。测试实际证明：MTP loss 的梯度不会流到非 MTP 参数。这种隔离以及 `mtp_kwargs` 接口只存在于 slime 自带的 Megatron 补丁里。证据：`tests/test_mimo_7B_mtp_only_grad.py` docstring "isolates gradient flow to only the MTP layers"；Dockerfile 用 `PATCH_VERSION=latest` 应用 `docker/patch/latest/megatron.patch`，该补丁在 MTP 层加 `decoder_input.detach()` 和 `keep_graph=False`，在 `_postprocess` 加 `mtp_output_weight.detach()`，并新增 `mtp_kwargs`；上游 `Megatron-LM@1dcf0daf` 为 `keep_graph=True`，无隔离。修正：改写 §5.3 结论；补 forward → MTP loss（loss_mask 随 roll 移位，只算 response 位置）→ 隔离的 backward → optimizer 链；写明对 Megatron 补丁的依赖边界。
2. **[major] 20 §6：学生项来源写成无条件"当前学生重算"。** 只在默认配置成立。证据：`loss.py::compute_advantages_and_returns` 中 `rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")` 直接交给 OPD；`actor.py::train_actor` 在 `--keep-old-actor` 时切到 `old_actor` 前向；参数校验不禁止这两个组合。修正：列为兄弟选择轴（重算 / old_actor / rollout 引擎 logprob），分别说明版本和温度含义。
3. **[major] 20 §4/§6：核心数值路径无最小实例、无原理图。** "丢首项 + 尾部裁剪"对齐和逐 token advantage 注入都没复现；§1 的 Mermaid 只是架构流程图。修正：用 2 个 prompt token + 3 个 response token 复现 logprob 对齐 → d̂_t → Â_t，含 GRPO 纯蒸馏 A_t=0 情形；token 序列按规范用 SVG。
4. **[major] 20 §6 beat-2：缺有效性论证与 λ 说明。** 缺推导：advantage 是 no_grad 常量，`E[d̂_t·∇logπ]` 即反向 KL 的 score-function 梯度；缺"独立 KL loss"备选的取舍判据；未说明经 clip 和 `--normalize-advantages`（REINFORCE++ 时被断言强制开启）后 λ 不再是绝对系数。
5. **[major] 21 全页：触发的变换没复现，缺成本账本和阅读路线。** MTP 层号保持从 0 开始、只有 expert 下标加 EP offset 的改名规则；`spec_accept_rate` 按样本等权平均；全页只有一张闭环流程图，无成本账本，无紧凑阅读路线或 caller tree。修正：用 PP=2、EP=2、8 个 expert 复现改名；两样本数值对比等权平均与按 token 加权。
6. **[major] 22 §5.1：原理图示例退化。** 单个恒值 128×128 block 无法区分 blockwise 与 per-tensor，也看不到分块网格、边缘块、舍入、零块；二维块网格不适合 Mermaid。scale 张量形状 `ceil(M/B)×ceil(N/B)`，eps 1e-10。修正：用 `tests/test_block_fp8_zero_block.py` 的 256×256（1 非零块、3 零块）同时复现 blockwise、per-tensor（`weight_scale`）和 INT4 三条数据面，生成外部 SVG。
7. **[major] 22 §5.2/§6：INT4 数据面无数值语义。** 训练侧 fake-QAT 在 `megatron.patch` 的 `_FakeInt4QuantizationSTE`：只作用于 `TEGroupedLinear`（MoE experts），`q_max=7`，scale 下限 1e-5，STE 直通；在线打包 `fake_int4_quant_cuda` 后接 `pack_to_int32(..., 4)`。修正：补同一输入复现，并写明"QAT 只覆盖 MoE experts"。
8. **[major] 22 §3.3/§7：compressed-tensors 内部兄弟轴未枚举。** `quantize_params_compressed_tensors` 不读 `num_bits/type/strategy`，注释 "only int4 at the moment"；而 `convert_hf_to_fp8.py --strategy channel` 会输出 compressed-tensors 格式 FP8，`convert_hf_to_int4.py` 提供 W8A16。修正：注明这两种 schema 不支持在线更新（后果为推断，未运行），补进 §7 失败表。
9. **[minor] 19 §4：外部部署协议边界漏补丁依赖。** 写"协议边界没有移动"，但 delta 与 local-checkpoint 路径依赖 slime 补丁提供的 `/pull_weights`（`sglang-pull_weights.patch`，external 文档第 70/84 行）；外部 PD 测试 docstring 写 delta+disk 是 "only sync path that actually works"。
10. **[minor] 19 §3.4：buffer 插件缺失败边界。** `select_rollout_data` 只保留最新的组，超额组被丢弃；generator 走 OpenAI `/v1` 文本接口，Sample 无 rollout logprob 和 weight_versions。
11. **[minor] 19 §3.2/§7-3：建议"跑契约测试"，但仓内替换函数自己过不了。** 契约要求形参名一致、eval 输出非空；`generate_rollout_fully_async` 与 buffer 示例形参是 `data_buffer` 且拒绝 eval；forge 在字面路径模式下 eval 返回空。需注明契约适用范围。
12. **[minor] 21 §6.1：PP 放置写成"不能断言"，其实可核验。** 固定版本 Megatron 的 `get_mtp_num_layers_to_build` 只在最后一个 PP stage 构建 MTP；自定义 layout 时断言"全有或全无"。
13. **[minor] 21 §6.3/§7.3：未用本地可核证据。** 页面写"未额外打补丁"，但官方镜像应用 `docker/patch/latest/sglang*.patch`；可查到这些补丁未改 `update_weights_from_distributed`、未改 spec 元数据键，但新增 `post_process_weights` 转发给 draft_worker。
14. **[minor] 22 §10：基线后变更漏记两项。** `2fa9a442`：`--save-hf` 关闭 `transform_scale_ue8m0`，直接影响 §5.1"在线同步与落盘审同一 schema"；`876cd89b`：ROCm INT4 QAT kernel。
15. **[minor] 22 §4.3：`NVTE_FP8_BLOCK_SCALING_FP32_SCALES` 已有默认值。** `slime/ray/actor_group.py` 默认注入 "1"，页面写成需手动配置。
16. **[minor] 22 §2.2：引用 FP8 reader 但未写约束。** `SafetensorReader.get_tensor` 写死 128 块，只识别 `*_scale_inv`。
17. **[nit] 头部与呈现。** 19、20 的"最近更新"写核实过程；21 的"适用范围"含"本轮核验"，"依赖基线"块含"当前机器"等环境史，应移 changelog；19 链接别名写 §2.2.2，实际锚点 §2.2；20 用"第 12/15 页"纯数字引用；20、22 基本逐句 permalink，只有局部阅读路线；21 无紧凑阅读路线；20：server update 超时可被请求体 `timeout_s` 覆盖，未说明；21 §6.4 未引本地注释 "so that we run training without mtp weights"。

## C. 缺失内容

| 缺失项 | 源码位置 | 建议归属 | 重要性 |
|---|---|---|---|
| buffer 相关 CLI（全 wiki 未出现）：`--rollout-buffer-url`、`--fetch-trajectory-retry-times`（默认 -1，无限重试）、`--min-batch-collection-ratio`（已定义但基线无消费方）、`--rollout-task-type`（默认 math）、`--loss-mask-type` | `arguments.py::add_rollout_buffer_arguments` | 19 | 中 |
| external PD 及对 `/pull_weights` 补丁的依赖 | `tests/test_qwen3_4B_external_pd.py`、`docker/patch/latest/sglang-pull_weights.patch` | 19（链 16） | 中高 |
| 其余 rollout 侧 `*-path` 钩子：dynamic-sampling-filter、buffer-filter、rollout-sample-filter、rollout-all-samples-process、rollout-data-postprocess | `tests/plugin_contracts/*` | 12/13，19 表内路由 | 低 |
| `--opd-teacher-ckpt-step` | `actor.py::load_other_checkpoint` | 20 | 中 |
| 学生项来源轴：`--use-rollout-logprobs` / `--keep-old-actor` | `loss.py`、`actor.py::train_actor` | 20 | 高 |
| normalize 对 OPD 项的缩放（只提调用顺序） | `loss.py::compute_advantages_and_returns` | 20/15 | 中 |
| MiMo EAGLE 完整配方 | `scripts/run-mimo-7B-rl-eagle.sh` | 21 | 低中 |
| Megatron 补丁里的 MTP 隔离、`mtp_kwargs`、loss_mask 的 roll 移位（全 wiki 未出现） | `docker/patch/latest/megatron.patch` | 21（链 14） | 高 |
| GPTQ 转换工具（W4A16/W8A16，默认 group 32） | `tools/convert_hf_to_int4.py` | 22 | 中 |
| FP8 转换工具 `--strategy channel` 与 `--scale-fmt ue8m0` | `tools/convert_hf_to_fp8.py` | 22 | 中 |
| 反量化工具 | `tools/fp8_cast_bf16.py`、`tools/convert_k2_thinking_int4_to_bf16.py` | 22 | 低中 |
| 其他配方：qwen3-4b-fp8、moonlight/235B/kimi INT4（kimi 用 group 32） | `scripts/low_precision/*` | 22 | 低 |
| ROCm/NPU：基线 CUDA 镜像才编 int4_qat（Dockerfile:161），Dockerfile.rocm 不编 | `docker/Dockerfile.rocm`、`docker/amd_patch`、`docker/npu_patch` | 22 | 低中 |

## D. 环境缺口与无法核实项

- 21 的 SGLang 引用（12 条 permalink）：EAGLE verify/logprob、weight_updater 转发 draft、tokenizer_manager 元数据键与别名、draft CPU 备份语义，本地均无法核实；本地只能证明 slime 补丁未改相关路径，且补丁增加了 draft 的 post_process 转发。
- 20：SGLang 在 `temperature=0`、`logprob_start_len=0` 下 `input_token_logprobs` 首项语义，属依赖契约，未核。
- 22：SGLang 的 `should_deepgemm_weight_requant_ue8m0`、`quant_weight_ue8m0`、`transform_scale_ue8m0`，compressed-tensors loader，TE FP8 recipe，`fake_int4_quant_cuda` 实际执行，均未核。
- 19：router 的 `/workers` 和 `/server_info` 字段契约未核。
- 未运行 GPU/E2E；B8"在线不支持"、B1 训练后果为源码推断。

## 协调者复核

- B1 属实：`tests/test_mimo_7B_mtp_only_grad.py` docstring 为 "validates that the MTP loss computation correctly isolates gradient flow to only the MTP layers when the main model loss is zero (due to truncation)"；`docker/patch/latest/megatron.patch` 含 `decoder_input = decoder_input.detach()` 与 `keep_graph=True`→`keep_graph=False` 改动、`mtp_output_weight.detach()`、`mtp_kwargs`；上游 `megatron/core/transformer/multi_token_prediction.py`@1dcf0daf 为 `keep_graph=True`。21 页第 129 行写"这证明 MTP loss 已接上，且主 policy loss 被 mask 后不会污染主干梯度"。
- B2 属实：`slime/backends/megatron_utils/loss.py` 第 731 行 `rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")`。
