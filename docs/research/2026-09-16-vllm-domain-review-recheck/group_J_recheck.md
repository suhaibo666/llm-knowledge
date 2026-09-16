# Group J 复审：vLLM 域级 E2E 修复复核（2026-09-16）

- 对象：`wiki/02_engineering/03_infer_frameworks/vllm/` 01–26 + index + `wiki/changelog.md` 的**未提交工作区改动**（`git diff --stat`：28 文件，+1725/−471）。
- 源码：`/Users/suhaibo/97-llm/vllm`，复核前后 `git rev-parse HEAD` 均为 `199cb9b964822e59ab9b58d88e7be31eb419a2ae`（2026-09-07 00:54 UTC），只读。
- 方法：重建三张 e2e 覆盖表；逐条核对 F1–F25 与 consolidated §4.2 的 21 个互推项（两侧都查）；机械抽取并校验全部跨页 `NN §x.y` 引用（91 处）与全部 wikilink `#标题` 锚点（新增脚本，因为 `check_links` 不校验链接内锚点）；抽查 `SchedulerOutput` 22 字段、`reinitialize_distributed` 归属类等域级合同；未逐页复核事实（由其他组负责）。
- 机械门禁（`.venv` 3.13）：`check_links --strict` 452 页 broken/ambiguous/bare_index/stale_section/orphans **全 0**；`check_math/check_markdown/check_assets --changed --strict` 39 文件 **0 错 0 警**。**但门禁看不到本轮新引入的 3 处链接内锚点失效，见 N1。**

**总体结论**：修复质量高。consolidated §4.2 的 21 个互推项中 **17 项两侧都已闭合**（owner 页给出目的/IO/处理逻辑/边界/范围，推出方只留链接），没有发现"只把箭头换了个方向"的伪修复。启动链的两大缺口（显存 profiling → KV 预算、`SchedulerOutput` 生产端合同）与 KV events、connector 遥测、Marlin 地址合同、MRV1 graph 派发、grammar↔draft 分工全部真正落地。剩余问题集中在**被显式推给 planning 的 8 项无 owner 能力**（索引仍未登记）、**6 项本可就地修但未修的旧 P2**，以及**本轮新引入的 2 条 P1**（3 处锚点失效、18↔23 的 `reinitialize_distributed` 归属类矛盾）。

---

## 1. 覆盖表：状态变化或仍非"完整"的行

### (a) 启动链

| # | 阶段 | 上轮 | 本轮 | 证据 |
|---|---|---|---|---|
| a2 | 配置解析 `EngineArgs → VllmConfig.__post_init__ → check_and_update_config` | 仅片段 | **仍仅片段** | 全域 `create_engine_config` 仍只在 18 §9 出现 1 次，`check_and_update_config` 只在 21 §... 出现 1 次；02 未新增配置解析树。见 F1 |
| a3 | 平台探测 / Platform 对象能力 | 完整（仅探测） | **仍仅探测** | 02 §3.8 未新增 Platform 选择点清单；`current_platform` 的 worker_cls/attention/check_and_update_config 职责仍无 owner。见 F13 |
| a8 | 显存 profiling → 可用 KV 字节 → `num_gpu_blocks` | 仅片段 | **完整** ✅ | 新增 08 §3.1.3：requested/consumed/transient/graph/MM 五项公式、80 GiB 手算例（38 GiB → 62,259 pool id → 跨 worker 取最小 → 非 null 59,999）、`kv_cache_memory_bytes` 与 `VLLM_ENABLE_STARTUP_PLAN` 两条旁路、fingerprint 门、支持范围与源码路线 |
| — | 启动顺序本身 | 重叠一致（6 处） | **重叠一致（7 处）** | 08 §3.1.3 新增一张预算转换图，成为第 7 处启动期图示。见 N7 |

其余 a1、a4–a7、a9–a11 维持"完整"，未发现回退。

### (b) 一次请求

| # | 阶段 | 上轮 | 本轮 | 证据 |
|---|---|---|---|---|
| b8 | `SchedulerOutput` 生产端字段合同 | 仅链接（断链） | **完整** ✅ | 07 新增 §8.1.1（9 个执行集合/输入差量字段 + 2 个子结构表）、§8.1.2（10 个生命周期/资源字段）、§8.1.3（3 个异步补写字段），自述"9+10+3=22"。对源码 `vllm/v1/core/sched/output.py::SchedulerOutput` 逐字段清点确认**恰为 22 个顶层字段**，与 11 §3.4、12 §2.10 的"22 个字段"一致；`new_token_ids`（仅 PP 且关闭 async）、`all_token_ids`（仅 MRV1 且上一步未获调度）等发送条件两侧一致 |
| b16 | `ModelRunnerOutput` 字段去向 | 完整（1 处错指） | **仍有 1 处错指** | 11 §3.4 `ec_connector_output` 行仍写"15 §7.4：`update_from_output()` 交 `ECConnectorBase.update_connector_output()`"。15 §7.4 是 `prompt_embeds`，EC transfer 在 15 §7.7（该节确有"回传 `ECConnectorOutput` 由 Scheduler 的 `update_from_output()` 交 connector 更新"）。F7 未修 |
| b20 | 流式协议恢复（tool/reasoning 解析） | 缺失 | **仍缺失** | `tool_parser`/`ToolParser`/`reasoning_parser`/`ReasoningParser`/`Harmony` 全域仍 0 次 |
| b+ | prompt logprobs | 仅链接 | **完整（owner 侧）/ PARTIAL（推出侧）** | 14 新增 §3.8：右移一位的位置变换、1024 行分块、`-1` 时 V+1 列、跨 chunk 累积与末行剔除、两代差异、交付完成点。但 11 §3.4 / 12 §2.10 的 `prompt_logprobs_dict` 行仍只指 07 §8.5，未指向 14 §3.8 |
| b+ | `n>1` 父子请求 / beam search | 缺失 | **仍缺失** | `ParentRequest` 0 次；`beam search` 仅 03 的"被拒绝组合" |

b1–b7、b9–b15、b17–b19、b21、b22 维持原状态（b21 延迟释放仍为"重叠一致"）。

### (c) 横切能力

| 能力 | 上轮 | 本轮 | 证据 |
|---|---|---|---|
| KV events 发布 | 仅链接（断链） | **完整** ✅ | 23 新增 §5.3「KV cache 事件：从 block 生命周期发布到外部前缀路由」（`BlockStored`/`BlockRemoved`/`AllBlocksCleared`、`KVEventBatch`、publisher、外部 prefix-aware router 的消费语义）；22 §14.4 与 Related Pages 均改为带锚点的 §5.3 |
| 传输观测的 connector 枚举 | （§4.2 项） | **完整** ✅ | 23 §5.2「KV 传输：通用搬运 stats，connector 决定实际指标」；22 两处链到 §5.2 |
| Elastic EP 编排 | 无 owner（18→13→18） | **完整** ✅ | 18 拥有 `ElasticEPScalingState` 四张状态机 + `eep_ready` + commit 边界；13 §4.6 拥有控制面成员表/READY 接缝并说明"CLI 把 elastic EP 的 API 数限制到至多一"。环闭合 |
| Ray / external_launcher 执行与监督 | 无 owner | **仅片段** | 26 §1.1 新增 `Executor.get_class` 六行变体表（mp / ray V1 / ray V2 / uni / external_launcher / 自定义）并闭合 `RayExecutorV2` 复用 MQ；但 **Ray actor 的监督与故障闭环仍无 owner**，且 23 已不再把它指给 18/13，缺口变成未声明 |
| pooling / embedding 执行 | 仅片段 | **仍仅片段** | 12 §2.6 仍写"pooler 算法与 `PoolingRunner` 的内部状态本域暂无专页" |
| tool / reasoning parser、Responses/Anthropic/harmony | 缺失 | **仍缺失** | 同 b20 |
| 非 CUDA 平台 worker/runner | 缺失 | **仍缺失** | 11 §2.10 仍写"本域暂无页面专门展开平台 runner"；index 未声明平台覆盖边界 |
| sleep / wake | 仅片段 | **仍仅片段** | `CuMemAllocator`、`sleep_mode_backend` 全域仍 0 次 |
| EPD 部署拓扑 / P/D proxy | 缺失（两页互指） | **仍缺失（仍互指）** | 22 §14.4「**本域仍无 owner**，与 `15:§9.4` 的空白登记互指」；proxy/router 行仍写"建议归 **13**；基线下全库 wiki 无人展开"，13 未接 |
| LoRA 使用入口 | 机制完整、入口缺失 | **仍缺失** | `--enable-lora`、`LoRARequest`、`/v1/load_lora_adapter` 全域 0 次 |
| 多模态 encoder 准入/释放 | 重叠一致 | **释放已收口、准入仍重叠** | 15 §4.2 拥有释放规则（`offset + length + num_prefill_lookahead <= num_computed_tokens − num_output_placeholders`、`[2,6)` 算例）；07 §8.4 只留一句摘要（**但未链接 15 §4.2**）。准入仍 07 §5.3 与 15 §4.1 各写一遍，各自视角不同、可接受 |

---

## 2. F1–F25 状态

计数：**FIXED 8 / PARTIAL 3 / NOT_FIXED 14**（其中 8 项是 consolidated §6.4 明确路由给 planning 的页面合同变更，6 项是本可就地修而未修）。

| # | 上轮问题 | 状态 | 证据 |
|---|---|---|---|
| F1 | 配置解析阶段无 owner | **NOT_FIXED**（planning 待批） | 02 无新增配置解析节；`create_engine_config` 仍只 18 §9 一处。changelog 声明"配置总链…留待规划批准"，但 index 未向读者登记 |
| F2 | 显存 profiling / KV 预算无 owner 级解释 | **FIXED** | 08 §3.1.3，见上表 a8 |
| F3 | `SchedulerOutput` 生产端合同落空 | **FIXED** | 07 §8.1.1–8.1.3；22 字段经源码清点一致；07 页头适用范围已加入"`SchedulerOutput` 生产端字段合同" |
| F4 | tool-call / reasoning parser 与 Responses/Anthropic 无 owner | **NOT_FIXED**（planning 待批） | 全域 0 次 |
| F5 | 05 §4 引用不存在的"27" | **FIXED** | 改为 `[[23…#6.3 受控恢复：只恢复可恢复的执行环境\|可观测性与可靠性 §6.3]]`；同表编译/graph 行的 owner 也从 23 改为 19 |
| F6 | index 阅读依赖箭头 | **FIXED** | 23 行改为"Scheduler、KV、Serving、MultiprocExecutor与分布式 → 调试与排障"；26 提到"系统专题"首行、18 行依赖补 MultiprocExecutor；21 行移到 20 之前；"模型与设备执行"表上方新增 MRV2 默认提示，11 行"对照/回落路径"、12 行"默认入口" |
| F7 | 11 §3.4 → 15 §7.4 错位 | **NOT_FIXED** | 仍为 15 §7.4；正确目标是 15 §7.7 |
| F8 | OutputProcessor 归属标注不一致 | **FIXED** | 12 已不再出现 OutputProcessor（"归 06"标注移除），只余 14 §1.2 的"归 03 与 06"，冲突消失 |
| F9 | KV events：22 指 23，23 无内容 | **FIXED** | 23 §5.3 |
| F10 | prompt logprobs 只有链接 | **PARTIAL** | 14 §3.8 已实质拥有；11/12 的字段表未回指 14 §3.8 |
| F11 | `n>1` / beam search 无 owner | **NOT_FIXED**（planning 待批） | `ParentRequest` 0 次 |
| F12 | pooling/embedding 执行侧无专页 | **NOT_FIXED**（planning 待批） | 12 自述未变 |
| F13 | 非 CUDA 平台与 Platform 接口无 owner、index 未声明范围 | **NOT_FIXED** | 11 §2.10 自述未变；index"目录范围"未加平台边界句；02 §3.8 未加 Platform 选择点清单 |
| F14 | sleep 只有语义没有机制 | **NOT_FIXED**（planning 待批） | `CuMemAllocator` 0 次 |
| F15 | EPD/P-D proxy 自述无 owner、两页互指成环 | **NOT_FIXED**（planning 待批） | 22 §14.4 与 15 §9.4 仍互指 |
| F16 | LoRA 缺使用入口与串联 | **NOT_FIXED** | 01/02 仍无 `--enable-lora`、`/v1/load_lora_adapter`、`LoRARequest` |
| F17 | 页头基线行日期/格式漂移 | **FIXED** | 26 页页头全部为 `` `vllm-project/vllm@199cb9b…`（`main` 快照，2026-09-07 UTC） ``，无 09-06/09-08、无缺 UTC、无句末多余句号 |
| F18 | 24–26 页头字段是占位文本 | **PARTIAL** | "最近更新"已改为实质变更说明；但**主题仍是占位**（24"vLLM 扩展插件系统（机制分析）"、25"（功能分析）"、26"（并发与分布式机制分析）"），适用范围仍只是对象列表、无"不拥有/归 NN"陈述；页头后孤立 `---` 仍在 |
| F19 | "最近更新"与索引/状态不一致 | **FIXED** | 26 页全部 2026-09-15；index"最后更新：2026-09-15"；changelog 新增本轮条目 |
| F20 | 重复解释（漂移风险） | **PARTIAL** | 改善：encoder 释放收口到 15 §4.2；DP pause 的 25 §5.3 压成 3 句并链到 13；Marlin/KV scale/HMA 均改为单 owner + 链接。**恶化**：启动顺序 6→7 处（N7），默认异步提交序在 7 页各叙一遍（N4），deferred `-1` 在 3 页各叙一遍（N5）；stale output 仍 06 §5.1 与 07 §8.3 各完整一遍 |
| F21 | 两代页面模板并存 | **NOT_FIXED** | 07、08、13、21–26 仍无"核心流程清单"分节（09–12、14–20、17 §1.1、19 §1.3 有） |
| F22 | 裸 "V1/V2" 指代 runner 代际 | **NOT_FIXED** | 14 仍有 23 处裸 `V1`、4 处裸 `V2`（如 §3.2 标题"V1 native…"）；16 裸 V1/V2 共 55 处 vs MRV1/MRV2 37 处，且 §2 内相邻两段混用（line 121 用 MRV2、line 123 用"V1 内核"） |
| F23 | 03 定位与"API 调用参考"缺失 | **NOT_FIXED**（planning 待批） | 01 TOC 无 API 速查节；03 仍为分析页；index 未标注 03 类型 |
| F24 | 02 §6 出口不全、技术栈学习路径、courses | **NOT_FIXED** | 02 §6 仍漏 13、14、15、18、22–26；`01_llm_inference_technology_stack_analysis.md` 学习路径第 2 步仍是 06；`wiki/courses/**` 仍无 vLLM 链接（只有 slime 的 vime 页提到 vLLM backend） |
| F25 | 01 文件名与 H1 不符 | **NOT_FIXED**（本就建议下次重命名批处理） | 文件名未变 |

---

## 3. consolidated §4.2 互推项：两侧验证

判定标准：owner 侧必须给出目的/IO/处理逻辑/边界约束/支持范围（而不只是提一句），推出侧必须指向该 owner 的具体节。只换箭头方向 = NOT_FIXED。

| # | 功能点 | owner 侧（是否真的写了） | 推出侧（是否指向该处） | 状态 |
|---|---|---|---|---|
| 1 | 滑窗/chunked-local admission cap | **08 §3.3.1**：$N_{\mathrm{SWA}}$/$N_{\mathrm{chunk}}$ 两条公式、B=16/W=64/I=256 算例（21 块 / 20 块）、`apply_admission_cap=True` 只作用于前置整段准入门、R-SWA 与 HMA 关闭后的例外 | 07 §7（line 494）"由 `[[08…\|KV Cache §3.3.1]]` 负责；本页只保留传参条件" | **FIXED** |
| 2 | hybrid KV 开关 / block hash 生成 / Mamba align 块生命周期 | **08 §4.5.1**（三态解析表、双重否定、connector gate 只在 None 分支、`_promote_local_kv_cache_specs` 的实际变换）、**08 §2.3**（生成时机、`hash_block_tokens` 链、四种算法表、`NONE_HASH` 种子）、**08 §5.1.3** | 07 §5.4 → 08 §5.1.3；12 §... `MambaHybridModelState` 行 → 08 §5.1.3；15 §1、§5.2、Related → 08 hash extra key | **FIXED** |
| 3 | spec 下可缓存的尾 token | **08 §3.3.3**：三条不同边界、`C=min(computed+new, num_tokens)`、基类/hybrid 普通组/EAGLE 组三分支上界表、C=70/R=2 算例、`extra_retained_tokens` 的释放影响 | 16 §8.3 尾段 → `[[08…\|KV Cache §3.3.3]]` | **FIXED** |
| 4 | `SchedulerOutput` 生产端字段合同 | **07 §8.1.1–8.1.3**（22 字段 × 生产逻辑 × 消费者边界 + 2 张子结构表 + 发布完成点 + 1 个 contradiction 框） | 11 §3.4 与 12 §2.10 的 22 字段消费表；12 §2.10 明写"字段怎样由 `schedule()` 生成归 07 §4.1、§8.1" | **FIXED** |
| 5 | MoE 专家权重写入 | **09 §2.6.1**：全局 expert id + `w1/w2/w3` 两坐标、`get_expert_mapping` 与 `make_expert_params_mapping` 两条路径、调用树到 `_map_global_expert_id_to_local_expert_id`、local id=-1 跳过 | 17 §1（"不是…通用 TP 写入入口，归 09"）；18 把权重搬迁交 EPLB | **FIXED** |
| 6 | packed 量化参数切片 / v1-v2 loader 选择 | **17 §5.1.1**「写入时换算的是存储坐标，不是把 INT4 解开」：`adjust_shard_indexes_for_packing` 仅 `packed_dim==output_dim`、输入轴打包不得再除、QKV 偏移实算 | 09 §... line 162"packed/scale 坐标规则归 `[[17…\|量化页 §5.1.1]]`" | **FIXED** |
| 7 | attention kernel 消费 KV scale | **10**：§2.11 写入/读取两行明确 `_k_scale`/`_v_scale`/`_q_scale` 的传递，line 340「KV scale 的实际消费边界」段给出 cascade 与非 cascade 两路 descale 扩展、闭合条件 | 17 页头 + §4.3 结尾 → 10；10 反向指 17 §4.3 拥有参数生命周期 | **FIXED** |
| 8 | Marlin workspace 与 `g_idx_sort_indices` 地址稳定 | **17 §6.2**：三页分工声明、三个跨边界对象表、`prefer_copy=True` 的相容条件与"不是无条件保址"边界、三条回归测试 | 19 §6.1（line 268）、20 §...（line 735）、25 §...（line 226）三处全部链到 `17…§6.2`，且各自声明只保留通用前提 | **FIXED** |
| 9 | MRV1 cudagraph 派发 | **19 §7.3**：真实入口 `GPUModelRunner._determine_batch_execution_and_padding` 内的 `dispatch_cudagraph` 闭包，4 道决定表（SP padding / 本步排除 FULL / `force_eager` / DP 协商后再派发）、三 token 算例 | 11 调用树 line 449"`_determine_batch_execution_and_padding` [graph 模式与 padding，见 19]" | **FIXED**（P0 #6 同时修正） |
| 10 | 投机下的 p 约束 | **14 §4.5**：MRV2 `RejectionSampler._verify` 先调 `sampler.apply_sampling_params`、V1 的 `RejectionSampler.apply_logits_processors` + `apply_sampling_constraints`、`_validate_spec_decode` 的 min_p/logit_bias 拒绝入口 | 16 §7 / §9.2 → 14 | **FIXED**（P0 #2 同时修正） |
| 11 | grammar + draft | 语义 **14 §6.4**（同步截短 vs deferred 补 `-1`、`apply_bitmask=False` 的后果、`max_rollback_tokens` 深度）；发布路径 **16 §8.3**（三条路线、`post_step` 门、deferred 顺序） | 双向明确互链，各自写"本页只拥有…" | **FIXED** |
| 12 | IR provider 注册、tolerance | **20 §3.5**「新增与 OOT provider：注册、导入、缓存与容差要一起接上」+ line 299 的 `get_tolerance`/`assert_close` 执行侧 | 21 §1（"容差的声明属本页，容差的执行…归 20"）与 §12（line 619）具体点名 `tests/kernels/ir/test_layernorm.py` | **FIXED** |
| 13 | prompt logprobs 实现 | **14 §3.8**（实质完整） | 11 §3.4 / 12 §2.10 的 `prompt_logprobs_dict` 行仍只写 07 §8.5，未回指 14 §3.8 | **PARTIAL** |
| 14 | `response_format` 映射、`return_token_ids` | **03 §2.3**「`response_format` 怎样归一成结构化约束」+ §3.3 的 `return_token_ids` 行与三意图对照段 | 14 §1（"…归请求语义页"）、01 → 03 | **FIXED** |
| 15 | P/D proxy 路由段 | 无（22 仍写"建议归 13；基线下全库 wiki 无人展开"） | 22 → 13（13 未接） | **NOT_FIXED**（planning 待批） |
| 16 | KV events 发布 | **23 §5.3** | 22 §14.4 + Related 带锚点；**但 22 的环境变量表行仍只写"→ **23**"** | **FIXED**（残留 N10） |
| 17 | 传输观测的 connector 枚举 | **23 §5.2** | 22 两处带锚点 | **FIXED** |
| 18 | Elastic EP 编排 | **18**（状态机、`eep_ready`、commit 前后置、重复发起的异常文案） | 13 §4.6 拥有控制面接缝并回指 18 | **FIXED**（残留 N2 的符号归属矛盾） |
| 19 | Ray、external_launcher 执行与监督 | **26 §1.1** 拥有执行器变体枚举（含 Ray V1/V2 与 external_launcher 的适用边界） | 18 §12 调用树列 `Executor.get_class` 分支；**但 Ray actor 监督/健康无人拥有，23 也不再指向任何页** | **PARTIAL** |
| 20 | 配置解析总链 | 无 | 片段仍散在 01/07/12/18/19/21 | **NOT_FIXED**（planning 待批） |
| 21 | 显存 profiling → KV 预算 → `num_gpu_blocks` | **08 §3.1.3** | 09 图 1 标注"：08"；01 §6.3、04、05 §4 指向 08 | **FIXED** |

小计：**FIXED 17 / PARTIAL 2（#13、#19）/ NOT_FIXED 2（#15、#20，均已路由 planning）**。未发现任何"只换方向"的伪修复。

---

## 4. 本轮新引入的域级问题

### P1

**N1 [P1] 3 处 wikilink 内锚点失效，机械门禁看不到**

| 位置 | 链接 | 目标页实际标题 |
|---|---|---|
| 24 L120 | `[[14_vllm_sampling_structured_output_analysis#4.3 扩展接口与不支持项\|采样与结构化输出 §4.3]]` | 14 §4.3 是「logits processor 的变体集合从哪里枚举出来」 |
| 25 L315 | `[[13_vllm_serving_control_plane_analysis#3.1 EngineCoreClient 是一组实现，不是一个进程\|…]]` | 13 §3.1 已改名为「进程、对象与状态归属」（本轮 13 重写所致） |
| 25 L403 | `[[26_vllm_multiproc_executor_rpc_deepdive#6. Future、广播顺序与 FIFO 不变量\|…]]` | 26 §6 实际是「Future：响应没有请求 ID 时怎样保持配对」 |

原因：`tools/check_links.py` 的 `_check_sections()` 只校验**链接后紧跟的裸 `§N` 文本**且只比顶层节号（`target_of()` 直接 `split("#")` 丢掉锚点），因此链接内 `#标题` 从不被检查；能发现它的是 mkdocs 构建的 `missing_anchors`，而本轮只跑了 T0 四项。后两条正是 CLAUDE.md 所说"此处改名破坏别处入链"的情形，只是方向反了——13/26 改了标题，25 的入链留成了旧标题。建议：修这三处，并在本轮提交前跑一次 `python -m tools.mkdocs_site.cli build --changed`。

**N2 [P1] `reinitialize_distributed` 的归属类：18 与 23 互相矛盾，且 18 写错**

- 18 三处写 `EngineCore.reinitialize_distributed`（§1.3 流程表 ⑬、§5.5 正文 line 302、§12 源码路线 line 586 的 `vllm/v1/engine/core.py::EngineCore.reinitialize_distributed / commit_prepared_elastic_ep`）。
- 23 §6.3（line 414）写 `DPEngineCoreProc.reinitialize_distributed`。
- 源码判定：`vllm/v1/engine/core.py` 中 `class DPEngineCoreProc(EngineCoreProc)` 始于 2015 行，`reinitialize_distributed` 在 2302 行、`commit_prepared_elastic_ep` 在 2356 行，**两者都属 `DPEngineCoreProc`**；2015 行之后到 2400 行之间没有其他 class 定义。
- 影响：18 是该功能点的 owner，锚点却指不到真实符号；23 修对了，于是同一符号在域内出现两种限定名。consolidated §3 中"18：`reinitialize_distributed` 实际在 `DPEngineCoreProc`"这条 P1 未被执行。

### P2

**N3 [P2] 07 §2.1 重讲 06 §4.1 已独占的 async 解析轴，且不链接过去**
06 §4.1 明确写"这条选择轴的 owner 是本页"，并给出输入条件表 + 容量/`step_fn` 表。07 §2.1（line 113）又复述了一遍：pooling 默认关闭、不兼容 spec 方法/`disable_padded_drafter_batch`/ROCm DeepEP DBO 会关闭、显式强开报错、允许集合由 `EagleModelTypes`/`NgramGPUTypes`/`draft_model`/`dspark` 枚举。内容与 06 一致，但 07 全页对 06 只有 Related Pages 一条无节号链接（`grep "06_vllm_engine_architecture_analysis#4.1"` 命中 11、12、19，不含 07）。07 是唯一"复述且无指针"的页。

**N4 [P2] 默认异步提交序在 7 页各叙一遍**
`non_block` 提交顺序（execute non_block → 取 grammar → sample non_block → 三元组入队，不等 execute future）现出现在 02 §2.3.3、06 §4.1/§4.2、07 §8.2、11 §1.4、12 §1.5、19 §7、23 各处（`grep -c non_block`：02=6、11=6、12=6、07=3、06=2、19=2、23=1）。逐条比对文字后**当前完全一致**，11/12/19 也都链到 06 §4.1；但这是本轮"统一默认执行轴"的副作用：同一断言的表述点从 4 处增加到 7 处，后续任何基线变化需同步 7 页。

**N5 [P2] deferred `-1` 补齐与 `num_invalid_spec_tokens` 在 3 页各叙一遍**
07 §8.1.3（`num_invalid_spec_tokens` 行 + contradiction 框）、14 §6.4（"只有 deferred 路线补 `-1`"段）、16 §8.3（路线 2）。三处都声明了自己的边界并互链，语义一致，但"截断 → 补 `-1` → 记录补位数 → 统计扣除"这条链被完整叙述了三遍。

**N6 [P2] 锚点与标签的节级不匹配（6 处）**
形如 `[[08…#3.3 分配：把候选命中变成受保护的请求映射|KV Cache §3.3.1]]`：锚点落在父节，标签宣称子节。机械统计：`2.6→2.6.1` ×2、`3.3→3.3.1`、`3.3→3.3.3`、`5.1→5.1.3` ×2。链接可解析（父节存在），但读者点进去落在上一层，需要再找子节。

**N7 [P2] 启动顺序图示增至 7 处**
02 §2.2、09 §1.5 图 1、10 §1.5/§2.5、11 §2.9、12 §2.9、19 §3.4，本轮新增 08 §3.1.3 的预算转换图。内容一致（08 那张限定在预算维度，并声明"编译、graph capture 生命周期继续由 19 展开"），仅作漂移风险登记。

**N8 [P2] 08 §3.1.2 保留了被 §3.1.3 取代的旧摘要，且无前向指针**
§3.1.2 末段仍写"`Worker.determine_available_memory()` 的 profile 路径从执行器预算扣除非 KV 开销，CUDA Graph 估算还受相应开关控制…"，紧接其后的 §3.1.3 才是完整推导。页内冗余，建议 §3.1.2 改为一句"预算推导见 §3.1.3"。

**N9 [P2] index 仍未登记域内已知缺口**
09（`runai_streamer`/`modelexpress` 无页）、10（ViT backend、插件 backend 无 owner）、12（pooler、diffusion、KV sharing fast prefill 无专页）、15 §9.4、22 §14.4 都在页内诚实登记了空白，但 index 的"目录范围"只写"26篇内容页 + 本索引"。读者从 index 无法看到这个域覆盖到哪里、哪些能力（tool/reasoning parser、pooling 执行、非 CUDA 平台、配置总链、P/D proxy、gRPC/Rust frontend）是明知未覆盖的。changelog 已写"留待规划批准"，但 changelog 不是读者入口。

**N10 [P2] 22 的环境变量表仍用页级指针**
22 line 854 `| VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES | True | KV event 发布 | → **23** |`，而同页 §14.4 与 Related Pages 已升级为 `23 §5.3`。同页两种精度。

### 已核验、未发现问题的域级项

- **交接对象名**：`EngineCoreRequest`(24)、`SchedulerOutput`(109)、`ModelRunnerOutput`(83)、`EngineCoreOutputs`(35)、`GrammarOutput`(22)、`AsyncOutput`(64) 全域拼写一致，无 `SchedulerOutputs`/`ModelRunnerOutputs` 等变体；`EngineCoreOutput`(20) 是源码中真实存在的单请求类（`vllm/v1/engine/__init__.py:196`），不是笔误。
- **跨页 `NN §x.y` 引用**：机械抽取 91 处，全部命中目标页真实节号（唯一语义错指是 F7 的 11→15 §7.4，节号存在但内容不对）。
- **锚点风格**：24–26 的稳定源码路线已全部收敛为 `path::symbol`（24 表内 16 条、25 表内 11 条、26 表内 5 条）；24 中残留的 `vllm.xxx` 点分名都是 entry-point group 名，不是 qualname。26 的测试行仍有 4 处只到文件名、未到 `::test_name`（轻微）。
- **索引一致性**：`ls *.md | wc -l` = 27，与 index"26篇正文 + 本索引"、上级 index"26 篇正文 + index"、`wiki/index.md` 的"27"一致；index 五张表的标题/问题句与各页 H1 及新页头适用范围一致（12 行"默认入口"与 12 §2.1"默认 MRV2"一致，11 行"回落路径"与 19 §7.3"MRV1 是现役分支"不冲突）。
- **07 §8.1.x 的字段合同对源码**：22 个顶层字段逐一比对，数量与分组（9+10+3）正确；`has_structured_output_requests` 的 contradiction 框（注释写 "Set only in async scheduling case"，实现在基类 `_update_after_schedule`）与源码一致。
- **HEAD 未移动**：复核前后均为 `199cb9b9…`。

---

## 5. 域级判定

**(a) 普通在线文本请求**：可以端到端读通，且比上轮更严密。Scheduler→Runner 这条最核心的合同现在有生产端（07 §8.1.x）与两个消费端（11 §3.4 / 12 §2.10）的三方一致描述；默认异步路径在 02/06/07/11/12/19 口径统一。剩余断点只有 F7 一处错指（读者从 11 找 EC 输出会落到 15 的 `prompt_embeds` 节）。**结论：通过**。

**(b) 启动**：从"仅片段"升级为几乎完整。显存 profiling→KV 预算这条此前最大的断链已由 08 §3.1.3 补全并给出算例。**仍缺前端两段**：配置解析总链（a2）与 Platform 对象提供哪些能力（a3）。读者仍无法回答"我给的 flag 在哪一步、按什么顺序被改写成最终配置"。**结论：基本通过，头部缺一环**。

**(c) 用户旅程 install → run → call API → read output → tune → debug → extend**：
- install → run → read output → tune → debug：连贯（01 → 04 → 05），05 的两处 owner 错指已修，05/01/04 指向 08 理解 KV 预算现在有内容可读。
- **call API 仍是断点**：tool calling、`/v1/responses`、Anthropic `/v1/messages`、`/tokenize`、beam search、`n>1`、LoRA 服务入口、embedding/rerank 的可操作说明全域仍无 owner；01 无 API 速查节，03 仍是机制页。
- **extend 半通**：24 拥有插件 ABI，但插件打包 how-to、非 CUDA 平台接入仍无处可去。

**总判定**：**域级 CONDITIONAL PASS**。三波修复按 consolidated §6 的路由执行到位，17/21 个互推项两侧闭合，没有伪修复；提交前应先处理两条 P1（N1 的 3 处锚点、N2 的 `reinitialize_distributed` 归属）并跑一次 `mkdocs_site.cli build --changed`，其余 6 项本地 P2（F7、F16、F21、F22、F24、F25）可排入下一批；8 项 planning 待批能力应至少在 index"目录范围"登记一句覆盖边界（N9），否则读者看不到这个域的已知边界。
