---
title: "Knowledge Base Changelog"
---

# Knowledge Base Changelog

All source ingestions and significant wiki updates are logged here.

> 本文件为只追加的历史日志：各条目按**写入当时**的状态记载，其中的文件路径、行数等均以当时为准，**不随后续目录迁移回写**。查当前路径请以各域 index 为准。

> 本文件只保留 **2026-07 起**（知识库结构整改期）的条目；2026-Q2（2026-04~06）及更早的历史条目已归档至 [[changelog/2026_q2_and_earlier|2026-Q2 及更早变更日志归档]]。

---

## 2026-08-31：按共享能力与动态生命周期重构 Verl 分析域

**Type**：新增 2 页 + 重写/收敛 14 页 + 重命名 2 页 + 索引与入链修复

**为什么重构**：旧结构混用了历史 V0 基线与当前 V1 主线，并把 Agent/Reward、在线权重发布、训练 checkpoint/recovery 分散写进多个生命周期页。页面各自内容大多有价值，但概念所有权不清，导致同一机制在 sync、async、V0 和优化指南中重复解释，也容易把 CheckpointEngine 的进程内发布误当成跨重启持久化。

**新结构**：[[01_verl_architecture_overview_analysis]] 改为“共享能力层 + 动态生命周期层”的系统地图；[[10_verl_end_to_end_iteration_analysis]]、[[17_verl_v1_async_trainer_analysis]]、[[22_verl_fully_async_dynamic_schedule_deepdive]] 与 [[20_verl_ray_trainer_analysis]] 分别只拥有 V1 sync、stable V1 async、experimental fully async 和当前 V0 legacy 生命周期。11/12/20 从历史冻结说明更新到统一提交 `254a23ed...`，不再把仍被 V1/V0 共享的控制与数据机制称为纯历史档案。

**补齐缺口**：新增 [[18_verl_agent_loop_reward_runtime_analysis]]，集中解释 AgentLoop、tool loop、trajectory 与 RewardLoop；新增 [[23_verl_training_checkpoint_recovery_analysis]]，集中比较 V0、V1 与 fully async 的模型/优化器/trainer/dataloader/TQ/MQ 恢复边界，并与在线权重发布明确分层。

**重命名与去重**：`14_verl_rollout_resharding_analysis` 改为 [[14_verl_rollout_runtime_analysis]]，只拥有 request、KV、sleep、abort、PD 与 partial request；`21_verl_delta_weight_sync_deepdive` 改为 [[21_verl_weight_publication_analysis]]，成为 full/`delta_sharded` 在线发布的唯一 owner。[[30_verl_optimization_analysis]] 收敛为证据驱动的调优决策指南，[[02_verl_quickstart_guide]] 只保留首跑路径和诊断入口。

**导航**：[[02_engineering/04_posttrain_frameworks/verl/index|verl 分析域]] 现含 16 篇内容页，并同步父索引、总索引和全库旧文件名入链。上游代码基线未变化，因此本轮不推进 radar baseline。

---

## 2026-08-30：补齐 Model Runner V1，并将执行主线重排为 15 V1 → 16 V2

**Type**：新增机制 owner 页 + 4 页连续编号调整 + 导航与交叉链接集成

新增 [[02_engineering/03_infer_frameworks/vllm/15_vllm_model_runner_v1_analysis|15 Model Runner V1]]，独立解释 MRV1 如何用 `CachedRequestState` 与紧凑 persistent `InputBatch` 承接动态调度：请求状态与 batch row 的双层所有权、finished/preempt/resume 的更新事务、condense/reorder 的全状态迁移、request-major 到 token-major 的输入物化、异步 token 回填与 host-buffer barrier，以及 dummy/profile/CUDA Graph 共用生命周期。页面以设计取舍、承重不变量、成本和失败边界组织，源码 locator 只承担证据角色。

为让演进关系先 V1 后 V2，原 `15 Model Runner V2`、`16 Serving 控制面`、`17 采样与结构化输出`、`18 多模态执行` 依次调整为 `16`、`17`、`18`、`19`；vLLM 域内引用、父索引和总索引同步更新。该域现有 **24 篇内容页 + index**，仍统一固定到 `vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`。

本轮验证限定在新增/重排的 vLLM 页面、该域链接闭合性以及直接受影响的两级索引和 changelog；未运行全库门禁，也未改写历史日志中的当时编号叙述。

---

## 2026-08-30：vLLM Waves 2–6 完成——23 篇统一基线与最终知识地图

**Type**：全域机制优先重写 + 4 个新 owner 页 + 导航/迁移元数据集成

Waves 2–6 已完成入口与控制边界、资源与设备热路径、模型与专用化、规模化与生产闭环、使用与最终导航的复审。全域不再按源码目录或函数顺序搬运，而以读者问题、设计取舍、状态所有权、提交不变量、成本和失败边界组织；跨页只保留相邻合同并链接唯一 owner。

新增四个权威 owner：[[02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis|04 请求语义]]、[[02_engineering/03_infer_frameworks/vllm/18_vllm_sampling_structured_output_analysis|18 采样与结构化输出]]、[[02_engineering/03_infer_frameworks/vllm/19_vllm_multimodal_execution_analysis|19 多模态执行]]、[[02_engineering/03_infer_frameworks/vllm/29_vllm_weight_transfer_online_update_analysis|29 在线权重更新]]。相关回链已从退休 ownership 修正到这些页面，协议/任务、token selection、媒体 encoder state 与在线版本可见性各有且仅有一个正文 owner。

最终 23 篇内容页与 [[02_engineering/03_infer_frameworks/vllm/index|vLLM 知识地图]] 全部固定到冻结源码 `vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`（2026-08-29），父索引、总索引计数与 repository radar 同步为 **23 篇 + index**。本轮只对直接改动的 index/backlink/radar/changelog 做 scoped `rg`、wikilink target existence、显式路径 `git diff --check` 与人工 owner/路径/计数复核；未运行全库验证，也未回写历史 changelog 条目。

---

## 2026-08-29：vLLM 架构概览 Wave 1——直达导航、回链与混合基线迁移

**Type**：重建页迁移集成（1 页更名/重建 + 2 个回链 + 2 个索引）

原 `03` 已更名并重建为 [[02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis|vLLM 架构概览]]。新页先解释静态责任层与状态边界，再用一条代表性在线请求说明生命周期；代码 locator 作为机制判断的证据，而非搬运源码散文。

DeepSeek 专属 MLA/MoE 叙事、语法表、机械函数索引与超大交互资源均已退役或归还其对应 owner 页。vLLM 索引和两个直接回链现指向架构概览；启动、进程与 serving 控制面的细节仍分别由 `10` 与 `16` 号页拥有。

本次验证有意限定在 vLLM 目录、两个直接回链和受影响索引的闭合性。它只是 Wave 1：架构概览已核验至 `6b110bad`，其余页面仍保留既有基线；后续 wave 须等待用户接受这一 exemplar 后推进。

---

## 2026-08-28（四）：给 check_links 加 stale_section 规则；重做 ACT 深潜页——立论被上游改写

**Type**: 门禁新规则（1 检查项 + 7 单测 + 3 处存量）+ 1 页重写

### 一、`stale_section`：把长期盲区纳入门禁

`§N` 是纯文本、不是 wikilink，`check_links` 此前完全不检查。本轮 megatron 26 页重排暴露出 **102 处**失效的章节交叉引用，全程门禁绿灯——这是个长期盲区。

规则：`[[页面]]` 后**紧跟** `§N` 时，去目标页解析顶层节号（`## N.` 与 `## 一、` 两种风格都认，中文序号转整数），不存在即报 `stale_section`，计入 `--strict`。三条取舍：只认紧邻形态（隔着一句话的 `§N` 往往指引用页自己）；窗口止于行尾（`visible_text` 把全篇拼成一个串，不截断会跨行误判）；目标页无编号小节时不评判。

上线即抓到 **3 处谁都不知道的存量**，都在与本轮无关的 inductor/npu 域，其中一处我先前手工 grep 漏了——因为文件里写的是中文数字 `§七`。补 7 个单测；`tools/` 全套 121 passed。

### 二、[[21_async_collective_tensor_deepdive]]：立论被改写，不只是行号过期

原页用 torchtitan `AllToAllTokenDispatcher.combine()` 论证「ACT 能在同一 forward 内掩盖通信」，且**全页没有任何基线声明**。重做后：

**① ACT 的出身改写了整页立论。** `e22d791287`（2023-02-16，#93990）标题即 *"[PTD] Introduce tracing friendly collectives"* —— **ACT 与 functional collectives 同一提交引入，是「让集合通信可被 dynamo/FX 追踪」的副产品，不是为掩盖设计的**。模块 docstring 亦自陈 eager 走 subclass、编译交给编译器、"In the future, these paths may be unified"。

**② 机制本身成立，例子不成立。** 三层机制（wrapper subclass → `__torch_dispatch__` view/非-view 分流 → `wait_tensor` 经 `WorkRegistry` 反查 Work 并 `ncclEndEvent_->block(currentStream)`）逐行重核通过；wait 是 stream 级 block 而非 CPU block。被推翻的是 **shared experts 那个例子**：`963c20cba`(#3386) 把 shared experts 从 `combine()` 移到 `MoE.forward()`，当前 `moe.py:440`(routed) 与 `:447`(shared) 的顺序使窗口不存在。

**③ 更根本的一条：默认配置下 torchtitan 的 MoE 可能根本不产生 ACT。** dispatcher 三处收发一律 `if (is_compiling or non_strict_tracing) or get_spmd_backend() != "spmd_types": funcol… else spmd.all_to_all`；而默认 `_spmd_backend = "spmd_types"`（`distributed/utils.py:36`）。该后端是 out-of-tree pip 包（`spmd_types==0.2.5`），不在检出内，**本页如实写「无法核实」而非猜测**。唯一被主动选用 ACT 的是 TP redistribute 的四处显式 `async_op=True`。

**④ 源码内部不一致**：`token_dispatcher.py:589-590` 那条 ACT 注释由 `09ea7d8e73`(#2842) 留下，写于 shared experts 尚在 combine 内、`spmd_types` 分支尚未加入之前，今天既不描述默认分支也不对应任何窗口——已标 contradiction（值得给上游提 issue，本轮未做）。

**基线**：本页跨三个仓库，各自钉死——torchtitan `a3168782c9`、PyTorch `ea5655fceb`、Megatron-LM `71092579…`（§4.6 对照现已带真实 locator，原本零 locator）。旧页 7 条数字 locator 全部作废，但查明它们在 `963c20cba^`（`83e490429cc5`）下逐条属实，这一事实写进附录。新页 151 处引用逐条 `git show` 打开确认。

**旧论证保留**：`## 附录 A` 完整保留旧形态代码与两张旧图，代码块行号一律带 `@ 83e490429cc5` 后缀标明所属基线，开头 `[!contradiction]` 说明推翻依据。四张图一张未删，失效处只在图注下加更正行。

**已知未尽**：fig1/fig2/fig3 各有一格描述已推翻的形态，只加注未重绘（需重跑绘图工具链）；`spmd_types` 包若装上，§4.3 表格首行可以定论。

**校验**：`check_links --strict` 430 页六项全 0；`check_math --changed --strict` 0 错 0 警；`pytest tools/` 121 passed。

---

## 2026-08-28（三）：修 torchtitan 章节引用，并翻掉跨框架对比表里一条已被上游推翻的结论

**Type**: 交叉引用修复（1 页 7 处 §）+ 一条主结论更正

**起因**：torchtitan 域被另一会话重构（`4b789b5`）后章节结构大改，别处的 `§N` 引用失效。实测范围比先前粗估的「约 23 处」小得多——**全库目录外指向 torchtitan 页的带 `§` 引用只有 8 条**，7 条集中在 [[30_comm_compute_overlap_analysis]] 的跨框架对比表，另 1 条是误报（dispatcher 页那个 `§11` 指的是它自己的 §11）。torchtitan 目录内部页面之间的 `]] §` 引用为 **0**，重构时已一并清掉。

**改号之外，查出 5 处是「归属错」而不只是编号错**：

- **PP 的 action runtime 不是 torchtitan 实现的**。原表把「`RECV` 早发起、用前才 wait」记在 torchtitan 名下并指向越界的 §8。该页 §1 表格逐字写着「schedule class | PyTorch pipelining | 决定 action/P2P 时序；TorchTitan 只选类、填 stages/microbatches」，§3 又说「让上游拥有 action engine」。已改指 §1/§3 并注明归属。
- **ZBV/DualPipeV 同理**：整页 grep 不到 `OVERLAP_F_B` 或 `stage_backward_input/weight` 的机制描述；该页只承载 V 型 rank 映射表，且自陈「zero-bubble 与 custom CSV 的 core integration case 当前 disabled」。
- **HSDP「反向另开 all-reduce stream」说反了**。21 页论点原话：「TorchTitan 并没有实现一套自己的 reduce-scatter / all-reduce 双流调度器…属上游 FSDP2」，而原指的 §5 标题本身就是「…**不是** AR/RS 双流开关」。已改指 §4 并把说法改成「属上游 PyTorch FSDP2；TorchTitan 只声明轴与缩放所有权」。
- Async-TP 那格的两处细节：`symm_mem.fused_*` 实属 dist-GEMM 而非 Async-TP；「Hopper 对称内存」门槛挂在 `enable_fsdp_symm_mem` 上，`_maybe_enable_async_tp` 里**没有任何 capability 检查**。

**一条主结论被翻面**：对比表原记「torchtitan 用 `AsyncCollectiveTensor` 实现同一 microbatch 内的 EP 掩盖」。回 `torchtitan@a3168782c9` 核对 `MoE.forward`：`out_TD = self.routed_experts(...)` 在 `torchtitan/models/common/moe.py:440`、`shared_out_TD` 在 `:447` —— **shared experts 严格排在 routed path 完成之后，没有可供掩盖的窗口**；#3386 `963c20cba`（2026-05-20）正是把 shared experts 移出 dispatcher 的那次重构。权威页 [[15_torchtitan_ep_analysis]] §5 已标明旧述不符合 HEAD。

代理只改了它被授权的那一格，同页另有四处仍在主张相反的事实（三分类举例、可达性矩阵、§4.3 结论、Related Pages 描述）——**只改一处会让同一页自相矛盾**，故由协调者一并改掉：矩阵该格 ✓→✗ 并补一条带 locator 的修正注；结论从「两个独立层次」改为「只剩 stage 级跨 mb 一个层次，且由上游 pipelining 提供」。`Stream 管理` 一行保留——那是泛指异步集合通信的载体，不是被推翻的那条断言。

**方法论**：`§N` 是纯文本、不是 wikilink，`check_links --strict` 检查不到，这类失效长期是盲区。建议给 `tools/check_links.py` 加一条低成本规则——wikilink 后紧跟 `§N` 时去目标页 `grep '^## '` 校验该顶层节存在；本轮 8 条里有 3 条纯靠「越界」就能自动抓出来。

**校验**：`check_links --strict` 430 页 broken/ambiguous/bare_index/orphans 全 0；`check_math --changed --strict` 0 错 0 警。

---

## 2026-08-28（二）：Megatron-FSDP 提为独立页（36 号），并按 Merge over coexist 去重

**Type**: 新增 1 页 + 合并去重 2 页 + 域索引

**为什么要提**：`megatron/core/distributed/fsdp/src/megatron_fsdp/` 在源码里是个 **11321 行、16 个文件的独立子系统**，wiki 里却只是「[[16_megatron_distributed_optimizer_analysis]] 的一节」加「[[27_megatron_tp_fsdp_resharding_supplements_analysis]] 的一节」。按「一个概念一页、宁拆勿合」提为独立权威页 [[36_megatron_fsdp_analysis]]，基线 `NVIDIA/Megatron-LM@71092579…`（`dev`，2026-08-27）。

**页号 36 的由来**：段 1（10–19）与段 2（20–29）已排满——26 号是 2026-08-01 PP 三页合并空出的号、域索引明写「不重新分配」，复用会让旧引用产生歧义。按 `CLAUDE.md`「某段超出容量时占用相邻空段，并在该目录 index.md 的段位表里注明」取段 3 首个空号，理由已写进索引段位说明。

**第 2 拍挖到四条被否掉的替代**，前三条有源码/文档原话：

1. **逐参数分片（FSDP2 的做法）** —— 判据是「进出通信缓冲区的那次 `COPY`」。文档把两者代价并排写明：FSDP2 需要 COPY 才能减少 NCCL 调用次数，Megatron-FSDP 则把连续缓冲区的切片视图直接赋给参数。代价是同一 `DTensor` 参数**在不同 rank 上形状可以完全不同**，因此需要一整个 `uneven_dtensor` 库。
2. **按 DP 度直接均分字节** —— 判据是 kernel 的 **locality**：块量化的 scaling factor 计算会被 FSDP 从中间劈开。替代方案是算 FSDP unit 内所有参数 `p.shape[1:]` 的最小公倍数、把缓冲区 pad 到 `DP × LCM`，保证 `dim=0` 任何一行都不被劈开。
3. **直接写进 `megatron/core/distributed/`** —— 判据是「能不能被别的框架装走」。提交 `af28b5a55`（2025-08-21）标题即 *Decouple Custom FSDP to make it independently installable*，四份独立证据坐实：`src/` 自带 `pyproject.toml`（`name = "megatron-fsdp"`）、两个入口的双源导入注释「Megatron-LM is not installed, use Megatron-FSDP as a standalone module」、`TODO(@cspades): Copied from megatron.core.utils to avoid depending on MCore`、以及文档的 "Bring Your Own Parallelism" 定位。**代价是 MCore 专属知识（哪个参数是列并行/行并行）只能由接入层按模块类名重新推断一遍。**
4. **每次 unshard 现分配临时缓冲** —— 四档分配器的 docstring 各自写死理由（碎片、分配开销、跨 unit 复用）。但「四档构成一条由松到紧的取舍阶梯、默认档选 `_resize_` 是因为多数模型不开 `nccl_ub`」这层判断**源码没有表态**，整段挂 `> [!note] 推断` 并写明该引哪几个 locator。

**Merge over coexist 的执行**（严格按规程顺序，先吸收、再改入链、最后才替换）：

- **吸收的独有增量**：`no_shard`（ZeRO-0）的收敛性陷阱（梯度统计只能在 `model_parallel_group` 上规约，否则 grad norm 虚高）、HSDP 缺省组合（不传 `ddp_config` 即 `optim_grads_params` 内层 + `no_shard` 外层）、grouped-expert 分桶的 #5013 归属、以及与激活重计算的协同（整层重算时参数只 all-gather 一次、重算与反向共用，`megatron_fsdp.py:116-119`）。吸收时发现新页的 locator **比原页更准**（用 `:250-264`／`:1068-1075` 而非原来的 `:250-255`／`:1071`），按新页为准。
- **入链**：全库扫描确认除新页页头外**没有任何外部引用**指向被并的三处小节，无需改指。
- **替换**：[[16_megatron_distributed_optimizer_analysis]] §18.2、§18.6 与 [[27_megatron_tp_fsdp_resharding_supplements_analysis]] §3 的正文换成指向新页的一行指引（标题保留作占位，避免同页后续编号连锁变动）。16 号页 −68 行、27 号页 −49 行。
- **对比分析按约定全部保留**：§18 三方对比框架、§18.1 概览表、§18.3 TorchFSDP2 详析、§18.4 选型矩阵、§18.5「为什么 FSDP2 在 MoE 训练中重要」一律留在 16 号页原地。

**顺带确认的一条事实**：`TorchFullyShardedDataParallel`（FSDP2 路径）在新基线下 `expert|Expert` **整体零命中**，没有任何 EP 专门处理；MegatronFSDP 侧的对应实现在 `megatron_fsdp.py:296`/`:351`。

**未覆盖**：`experimental/` 子包（1487 行，六个文件，三处 docstring 自称 "Experimental / Minimal"）只在第 5 拍作为在途方向提及，未展开成小节。

**校验**：`check_links --strict` 430 页 broken/ambiguous/bare_index/orphans 全 0；`check_math --changed --strict` 0 错 0 警。

---

## 2026-08-28：verl 推进 273 个提交并重建默认 V1 知识域——TransferQueue、两套 async 与 delta 权重发布补齐

**Type**: 源码基线推进 + 全域概念重构（14 篇内容页 + 域/父/全局索引 + radar）

**基线与工作区**：先把旁置源码工作区 `E:/97-codes/torch_parallel/verl` 从 `8a694930275061f52ebd538c906ef8819af56dbd` fast-forward 到恢复网络后的最终 `origin/main` `254a23edc62f25ebfae626e3932ae285d6f86009`（2026-08-28 10:08 +08），跨度 273 commits、623 files、+62,155/-12,195。源码工作区原有未跟踪 `GRPO_Analysis.md` 在两次 fast-forward 前后 SHA-256 均为 `ABD72593BCE2228C034DAE5433B446DB76A863CEF97FF8F73E0E89FB8F5E4529`，未被修改。最终提交新增 vLLM prefix-cache hit 到 TokenOutput，并在 partial-resume 中保留首次 prefill 的命中数。

**默认主线纠正**：[[01_verl_architecture_overview_analysis]]、[[02_verl_quickstart_guide]]、[[10_verl_end_to_end_iteration_analysis]] 全部从旧 `RayPPOTrainer/DataProto/full CUDA IPC` 叙事改成当前 `TaskRunnerV1 → PPOTrainerSync → TransferQueue → KVBatchMeta/tqbridge → Worker/Engine → CheckpointEngine`。`trainer.use_v1=true` 与 `trainer_mode=sync` 是当前默认；[[20_verl_ray_trainer_analysis]]、[[11_verl_single_controller_analysis]]、[[12_verl_dataproto_analysis]] 保留为完整 commit `8a694930...` 的冻结 V0 机制档案，不再机械 repin 或声称代表默认路径。

**新增三个机制 owner**：

- [[17_verl_v1_async_trainer_analysis]]：稳定 V1 的 sync/colocate async/separate async、ReplayBuffer `drop/wait`、DAPO/failure refill、streaming fetch、TQ checkpoint recovery、同一 PPO cycle 的稳定旧策略与 GPU lending；
- [[21_verl_weight_publication_analysis]]：Engine/CheckpointEngine/rollout loader 三段 ownership，dense seed → host snapshot prime → sparse steady state，ShardSpec、真实训练/rollout支持矩阵、checksum/periodic verify 与非事务失败窗口；
- [[22_verl_fully_async_dynamic_schedule_deepdive]]：独立 experimental TaskRunner 的 Rollouter/Trainer/MessageQueue completion-order、staleness admission、partial rollout、动态 Hybrid GPU、rebalance、恢复/确定性/测试缺口；明确它不是稳定 V1 第四种 mode。

**重构与最新遗漏**：[[16_verl_v1_transfer_queue_analysis]] 从“官方文档级、源码待核实”升级为固定提交源码页，闭合 `use_v1`/TQ 真实开关时序、prompt/trajectory 双层 key、延迟物化与当前仓内 SimpleStorage/MooncakeStore 配置范围；[[13_verl_workers_engine_analysis]] 补齐 FSDP Turbo、TorchTitan、Megatron/VeOmni、MindSpeed 精确删除范围和 `grad_offload` 配置变化；[[14_verl_rollout_runtime_analysis]] 拆开 colocated naive、disaggregated full、`delta_sharded` 与 PD/KV；[[15_verl_rl_algorithms_analysis]] 从 14 estimator × 11 loss 更新为 14 × 12，新增 DRO、`token-sum`、多轮 REINFORCE++ observation-span 修复与 critic global-batch 归一化；[[30_verl_optimization_analysis]] 改为吞吐、显存、新鲜度、权重发布与恢复性的联合预算，避免复制各深潜页。

**导航与追踪**：[[02_engineering/04_posttrain_frameworks/verl/index|verl 分析域]] 重建为 14 篇内容页的概念 ownership/三条阅读路线；后训练框架域从 44→47 页，verl 子域从 12→15 页（均含 index）；README、父索引、全局索引同步。`docs/radar/watchlist.yaml` 的 verl `kb_baseline` 更新为完整 `254a23ed...`。

**证据与门禁**：当前基线 11 篇实现页共抽取 460 个仓库相对 `file:line`/range locator，路径缺失与行号越界均为 0；三个 V0 档案继续以各自冻结提交解释原行号。`python -m pytest tools/` 为 114 passed；`npm run docs:test` 的 68 项运行时单测、429 页 Quartz 构建与浏览器 smoke 全部通过（139 个请求，覆盖 Mermaid、链接、静态资源与 loopback-only 网络）。最终链接、公式与格式门禁在写入本条后再次执行。

---

## 2026-08-28：megatron-lm 基线推进 578 个提交，456 条引用逐条重核——推翻 6 条结论、揪出 4 处原始撰写错误

**Type**: 基线重定 + 全域引用重核（26 页 + 域索引 + radar；六个并行 agent）

**基线**：`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27）。由 `ee3f1ffa…`（2026-05-19）推进 **578 个提交**，部分页由 `232c478d4…`（2026-06-16）推进 280 个。每页页头加一行 `> **重定基线**`；原先"正文按 ee3f1ff、`[!update]` 块按 232c478d4"的双行号口径就此作废，全部统一。

**方法**：不是把行号按偏移量平移，而是逐条读"这处引用声称的是什么"（哪个类/函数/断言/注释），在新基线下 `git grep` 找到它的真实位置，改完再 `git show` 打开确认内容对得上。找不到的一律判定性质，**不许硬凑**：移动改名 → 更新并注明 PR；机制被删 → 保留原文加 `> [!deprecated]`；行为变了 → `> [!contradiction]`。本轮共新增 25 条 `[!contradiction]`、15 条 `[!deprecated]`。

**被推翻的结论（本轮最有价值的产出）**

1. **[[35_deepseek_v4_context_parallel_analysis]] 自称的"核心贡献"整体反转。** 该页的卖点是「CSA/HCA 两阶段 CP 在代码里尚未实现，审计未发现 isend/irecv/all_gather」。新基线下 `experimental_attention_variant/` 从 10 个文件涨到 **45 个**，多出 `csa_utils/` 子包；两阶段 CP 由 **#5087** 实现：`_LeftBoundaryExchange`（`csa_utils/cp_utils.py:124`）用 `dist.batch_isend_irecv` 做前反向边界交换（`:156`/`:184`），入口 `exchange_cp_boundary_hidden`（`:201`）。§5.5 四条 gap、§8.3、§九 特征 4 随之作废。
2. **[[18_megatron_recompute_analysis]] 的「历史更正」自己过期了，且会误导配置。** 该页曾正确指出 `gdn_norm_out` 全仓已不存在（`ee3f1ff` 下确为 0 命中）；**#6088 把它加了回来**（新基线 8 处命中），现在 `gdn` 与 `gdn_norm_out` 并存且互斥，照旧说法配置会失败。
3. **MoE aux/z-loss 的 `× tp_cp_group.size()` 预乘被上游判定为不正确并改掉**（#5542/#4359）。源码注释原话：THD padding 或动态 CP 下各 rank 有效 token 数不同，`local_num_tokens * group_size is not generally correct`。现改为沿 `aux_loss_scale_reduce_groups` 逐组 all_reduce 后再乘（`router.py:598-624`）。影响 [[01_megatron_moe_training_optimization_analysis]] 与 [[28_megatron_training_stability_observability_analysis]]。
4. **flex dispatcher 多了第四个后端 `ncclep`**（`transformer_config.py:972`、`_NCCLEPManager` @ `token_dispatcher.py:1637`），带 `moe_ncclep_static_shape`（固定接收缓冲、无 D2H 同步）使 MoE 的 all-to-all **可被 CUDA Graph 捕获**。[[14_megatron_ep_analysis]] 的「三种 dispatcher」对比表与选型树枚举过时。
5. **两处上游自己回退/复活**：#5170 移除的 checkpoint 期显存回收 workaround 被 **#5366 整体 revert**，代码原样活着（影响 [[22_megatron_memory_optimization_analysis]]）；GDN 的「单次统一 A2A」被回退的结论也反转了，它现在是默认路径（影响 [[21_megatron_fusion_operators_analysis]]）。
6. **HybridEP 的 THD 自动补齐被 #5668 取消**，现在只认显式开关 `moe_hybridep_pad_variable_tokens`。

**机制被删除**：`broadcast_to_pp_group`（#4226，`ee3f1ff` 下 3 处命中 → 新基线 0；[[29_megatron_packed_dataset_dynamic_cp_analysis]] 的九步流水线第⑦步与 [[11_megatron_dataset_analysis]] §3.3 受影响，**且这次删除发生在 `232c478d4` 之前，上一轮重定基线时就该抓到、漏了**）、`TensorReusePool`（#5451，改引用计数）、`"dynamic_context_parallel is not supported with MLA yet"` 断言（#4226）、`optimizer_state_offloader.py`（#6244，改名 `chunked_optimizer_state_offload.py`）、`ssm/gated_delta_net.py`（#6088，拆成包）。

**揪出的原始撰写错误（不是漂移，是从来就错）**

- **[[20_megatron_comm_overlap_analysis]] §5.4.1 的代码片段是编造的**：`self.output_grads`、`delay_grads_release`、`manual_release_grads`、`untyped_storage().resize_(0)` 四处在 `71092579`／`232c478d4`／`ee3f1ff` **三个基线下全域零命中**。真实的 `backward_impl` 只做 `default_backward_func(...)` 后 `return grads`，`backward_dw` 只切流加 nvtx。"dX/dW 分离、dW 延后"这半句成立，"手工保存梯度再手工释放显存"那半句不成立。
- **[[16_megatron_distributed_optimizer_analysis]] §13.1** 把 `_check_module_parameter_types` 算作 `TorchFullyShardedDataParallel` 的 EP 处理手段——该符号在那个文件里**新旧基线都不存在**，它是 `MegatronFSDP` 的方法；新基线下该文件 `expert|Expert` 整体零命中，FSDP2 路径根本没有 EP 专门处理。
- **`paged_stash.py:1247-1267` 从来越界**：该文件在 `232c478d4` 与新基线下都只有 1240 行。
- **[[13_megatron_cp_analysis]] 的「`full_iteration` 与 `cu_seqlens` 互斥」两层都不成立**：被引的两处守卫分别是关于 `--no-check-for-nan-in-loss-and-grad` 与 `--cuda-graph-modules`，整段里 `cu_seqlens` 出现 0 次——这是 2026-08-27（十三）那一轮补 locator 时引错的，当时只核了"行存在"没核"它证明什么"；而且 #4359 已给 THD 变长训练加上 CUDA Graph 支持，该论断本身也过期了。
- 另有若干旧基线下就指错文件的（`batch_p2p_comm` 一直写成 `transformer_config.py`、实为 `model_parallel_config.py`；`drain_embedding_wgrad_compute` 指的是调用点非定义处；`validate_access_integrity` 是开关参数不是函数）。

**[[33_megatron_vllm_weight_sync_analysis]] 定出了基线**：它分析的是 `volcengine/verl`，不适用 Megatron 基线。沿本机 verl 检出历史回溯，钉为 `volcengine/verl@ab0705220a95952219111409d8f971872002c193`（`main`，2025-12-04）——这是**本页每一处引用都仍能解析的最新 commit**，紧随其后的 `fd893c78` 就删掉了 `vllm_rollout_spmd.py`。全页约 25 条引用据此补齐行号，并标注该调用链在 verl 当前 HEAD 已不存在。

**索引与 radar**：域索引改写为"全域统一基线 `71092579`"，并说明被推翻的结论是就地标注而非删除。`docs/radar/watchlist.yaml` 的 `kb_baseline` 同步推进。

**校验**：`check_links --strict` 426 页 0/0/0/0；`check_math --changed --strict` 0 错 0 警；`pytest tools/` 107 passed。26 页共 456 条带行号引用全部重核，每个新行号改完重新 `git show` 确认。

**未做**：五拍重排（第二波）。本域 26 篇里只有 4 篇有第 2 拍，是三个域里最缺的。

---

## 2026-08-27（十三）：把（十）（十一）报告出来但没修的缺陷全部修掉

**Type**: 缺陷修复（megatron 18 页 + slime 19 页；三个并行 agent + 协调者收尾）

上两轮各自留下了一批"已定位、已报告、但按当轮授权范围没动"的缺陷。本轮全部清掉。

**① 剩余 117 处非仓库相对 locator**（P1 42 + P2 31 + P3 26 + 协调者 13）。含两类：半路径（`transformer/moe/token_dispatcher.py`、`optimizer/__init__.py:761`、`fsdp/src/megatron_fsdp/megatron_fsdp.py:105`）与裸文件名（`csa.py:297`、`nccl_allocator.py`、`emerging_optimizers.py`）。复扫后剩 15 处全是正则误报（`torch.cuda.empty_cache()`、`te.pytorch.ops.Sequential` 这种符号名里含 `.cu`/`.h`），非缺陷。跨仓库的两处按各自仓库补全：DeepEP 的 `csrc/kernels/legacy/internode.cu`（`af9a040`）在 [[14_megatron_ep_analysis]] 与本域 index 均已补全。

**② 12 处写在 `###` 标题里的文件名**（上一轮明确禁止改标题，本轮授权）。三个 agent 改前都做了全库锚点检查——grep `[[页面#锚点]]`、grep 页内 `](#…)`、grep 标题文本引用，**三种口径均零命中**，改后 `check_links --strict` broken=0。

**③ 行号错误——其中一条要更正上一轮的结论**

上一轮怀疑 `param_and_grad_buffer.py` 的 **NVFP4 / MXFP8 两个标签被互换**。两个 agent 各自独立核实后，**该假说不成立**：标签语义一直是对的，真因是**跨基线行号串页**——B 基线的行号被贴进了 A 基线的页面。

| 页 | 页头基线 | 原行号 | 实际 | 处理 |
|---|---|---|---|---|
| [[01_megatron_moe_training_optimization_analysis]] | B | NVFP4 `:946` | B 下是 class docstring | 改 `:1020-1045`（`# NVFP4 uses a dual-buffer layout…` → `_compute_nvfp4_packed_layout`） |
| 同上 | B | MXFP8 `:1097` | **B 下正确** | 未改 |
| [[16_megatron_distributed_optimizer_analysis]] | A | NVFP4 `:946-963` / MXFP8 `:1097-1113` | 后者正是 **B** 的 MXFP8 区间 | 改 A 的真实区间 `:964-989` / `:1036-1055` |

其余已修：[[22_megatron_memory_optimization_analysis]] 三处（`paged_stash.py:587`→`:632`+`:1125-1128`、`param_and_grad_buffer.py:1097-1113`→`:1036-1055`、`fp8_utils.py:594`→`:513-529`，后者 `:594` 实为 `return fp8_recipe`），并顺带修了同节另外三处（`:946-963`→`:964-989`、`:357`→`:393-401`、`training.py:~2855`→`:2775-2781`）；[[16_megatron_distributed_optimizer_analysis]] 的 fsdp `:1404`→`:1407`、`optimizer/__init__.py:776`→`:777-780`；[[20_megatron_comm_overlap_analysis]] 的 `fused_a2a.py:135`→`:92-97`（判定该段描述 A 的形态，依据是紧邻代码块自标 `:69-138` 与 A 吻合）；[[01_megatron_moe_training_optimization_analysis]] 的 `:418`→`:352`（`start_param_sync` 才是入口）、`:357`→`:413-420`；[[10_megatron_model_structure_analysis]] 的 `training.py:576`→`:516-529`；两页的 `megatron_fsdp.py:105`→`:106`（105 是空行）。

**④ 一处消歧成功**：[[23_megatron_precision_cudagraph_fusion_analysis]] 的 `quantization/utils.py` 上一轮因"两个候选都不含 fp4"而无法判定。本轮改搜页面真正的承重符号 `get_quant_config_or_none`，在两个基线下都唯一解析到 `megatron/core/quantization/utils.py:9`，另一候选是 MXFP8 GEMM 派发、与本页无关。

**⑤ 一处描述与源码不符**：[[13_megatron_cp_analysis]] 原写 `_broadcast_cu_seqlens` "直接短路返回 `None`"。实读 `megatron/core/utils.py:2061-2067`：它是在 `cu_seqlens` 为 `None` 时广播计数 `n=0`、不再广播张量本体。同句里 `full_iteration` 互斥那半句原本**没有 locator**，已补 `megatron/training/arguments.py:1329-1332`、`:2050`。另 [[20_megatron_comm_overlap_analysis]] 的「（line 1113）／（line 964）」这种非 locator 写法改成真 locator，且两个数字在本页基线下都是错的（实为 `token_dispatcher.py:1161` / `:990`）。

**⑥ slime 三处尾巴**：[[15_slime_loss_parallelism_analysis]] 的操作清单原本夹在第 4 拍与第 5 拍之间、打断五拍，移到趋势之后；[[12_slime_sample_datasource_analysis]] 误读表里与第 2 拍重复的一行改为回指；19 页统一写清「核验日期」与「最近更新」的分工——本次未重新核验既有引用，故核验日期不动。

**⑦ 索引更新**：本域 index 里的「已知偏差」条目描述的状态已不存在（34/35 页头已改钉 A、行号已复核），改写为「已修正的偏差」。

**校验**：`check_links --strict` 426 页 broken/ambiguous/bare_index=0；`check_math --changed --strict` 0 错 0 警。所有新行号在改完后重新 `git show` 打开确认内容确实是页面所述之物。

**仍未做（不是缺陷，是另一个项目）**：推进基线。megatron 多数页停在 `ee3f1ff`（落后检出 HEAD 298 个提交）、vLLM 停在 `d66300a1`（落后 142 个）。宪法要求的是钉死某个 commit、不是钉最新；推进意味着把两域约 1000 条引用在数百个提交之后重新逐条核验。

---

## 2026-08-27（十二）：TorchTitan 按新版源码分析机制重做——23 页从“功能罗列”刷新为端到端状态与协议审计

**Type**: 机制级重审（16 篇既有 TorchTitan 页重写/扩写 + 7 篇新增页 + 域索引；另同步刷新 1 篇 TitanRL 训练循环页）

**冻结基线**：本机 `pytorch/torchtitan main@a3168782c9a3a2e40afbd0de114818b96e2bda6e`（commit date 2026-08-26，工作树 clean）。这次没有用“最新代码”代替基线，也没有把历史实现与当前实现混写；每个非平凡判断都回到这一提交的仓库相对 `file:line`。

**为什么重做**：旧页覆盖了功能名，却经常停在“有哪些开关/collective”，没有闭合新版 `source-faithful-analysis` 要求的五拍：背景约束 → 为什么选它而不是直观替代 → 状态/调用链与不变量 → 成本和失败边界 → 仅由源码 TODO/弃用声明支撑的趋势。本轮以“状态由谁拥有、何时转移、组合在哪一层被拒绝”为主线，重写 Trainer、ParallelDims、FSDP、TP、CP、PP、EP、SPMD Types、AC、通信/内存优化、SimpleFSDP、FlexShard 与 GraphTrainer 共 16 篇既有页。

**新增的 7 个概念所有者**：

- [[02_torchtitan_data_pipeline_grain_analysis]]：Grain source/mix/packing/iterator graph 与 exact resume，不再把 dataloader 简化成 batch index。
- [[03_torchtitan_checkpoint_state_recovery_analysis]]：manager/协议、storage、PP FQN、async staging、load precedence 与 HF/native/final export 的边界。
- [[04_torchtitan_config_model_protocol_analysis]]：full Python configuration、`Configurable` owner/build、override traversal 与 `ModelSpec`/`Module` 协议。
- [[05_torchtitan_multimodal_data_model_contract_analysis]]：resize 后 placeholder runs、document/media 保序的 packed patches、DP-local vision join 与 scatter 的端到端一致性。
- [[28_torchtitan_torchft_fault_tolerance_analysis]]：replica-group 故障域、quorum optimizer、FSDP hook 与全局/每副本 checkpoint 双通道。
- [[29_torchtitan_transformers_modeling_backend_analysis]]：HF config/module/sharding/state-dict 适配、native Titan MoE replacement 与兼容矩阵。
- [[30_torchtitan_forge_engine_analysis]]：可嵌入构造 seam、下游 loop 所有权，以及 HEAD 中 `ModelSpec.loss` 消费点已经协议漂移的事实。

**从遗漏审计回填进既有页的机制**：[[01_torchtitan_trainer_quickstart]] 补 structured span/scalar/instant 的双层状态、compile no-op 与 Perfetto 转换边界；[[15_torchtitan_ep_analysis]] 补 aux-loss-free expert bias/token counter 从 forward 累积到 optimizer pre-hook 规约、更新、清零的完整状态机，以及自定义模型必须接 hook 的失败边界；[[23_torchtitan_compute_memory_optimizations_analysis]] 补 LoRA 在 Config 树中继承量化 Linear owner、全树冻结、TP A/B placement 与普通 DCP 仍保存完整模型状态；[[27_torchtitan_graph_trainer_compiler_runtime_analysis]] 补 AutoParallel solver、local-tensor AOT 边界、DTensor rewrap、dense/sparse mesh 分工与两个集成定义仍禁用的成熟度证据。EP 页还补了 `torchtitan::deterministic_scatter_add` 如何暂时切换并恢复 deterministic setting，而非把局部算子确定性误写成全训练确定性。

**纠正的旧心智模型**：当前默认不是 `full_dtensor`；PP 是 stage adapter 而非自有 scheduler；CP 不再走旧 forward wrapper；EP 的 shared expert 不与 routed combine 重叠；FSDP 不存在固定“5 stream”模型；HF MoE 已先换成 native Titan MoE；TorchCheckpointing hooks 不是完整恢复闭环；SimpleFSDP 的 optimizer 仍在 joint graph 外；FlexShard 只临时改变 optimizer compute ownership；LoRA 没有 adapter-only/PEFT export；structured trace 不是 GPU profiler。TitanRL 页同步纠正为 global response-token denominator，并明确 zero-valid-token 的防线主要在标准 builder/batcher，Trainer 没有第二道 guard；TorchStore 只交接运行时状态，resume 也不提供 rollout exactly-once。

**仍明确保留的缺口**：P1 是 DeepSeek V3 MTP、metrics/Kineto/memory profiler、fused MLA/Helion RoPE、Kimi K2 QK clipping 与 Flux 训练纵切；P2 是 LR scheduler state contract、TitanRL recorder/metrics plumbing，以及 Qwen3.5 GDN/Kimi K3 KDA 内部。多模态页还记录了 global DP pack plan、mixed-media temporal counting 与端到端坏图/label-mask 测试缺口。这些没有用推断填成“已覆盖”。

**可核验性**：23 篇 TorchTitan 内容页共扫描到 2,752 处 locator（2,144 个唯一 `file:line`/range），相对冻结源码 missing/out-of-range 均为 0；页数按磁盘重算为 TorchTitan 23 篇内容页 + index、训练框架域 67 页。最终门禁：`check_links --strict` 扫 426 页，broken/ambiguous/bare-index/orphan 均为 0；`check_math --changed --strict` 扫 29 个 Markdown，0 error/0 warning；`python -m pytest tools/` 为 107 passed；`npm run docs:test` 为 67 个单测通过，并完成 426 页 Quartz 构建与浏览器 smoke 的 loopback-only 网络审计；两张变更 Mermaid 已人工检查且在站点 smoke 中成功解析；`git diff --check` 通过。

---

## 2026-08-27（十一）：megatron-lm 全域补溯源——26 页钉死 commit，774 处 locator 补成仓库相对路径

**Type**: 溯源修复（26 页 + 域索引 + radar 基线；三个并行 agent 各自独占一组文件）。**本轮不动章节结构、不动论述措辞、不动行号。**

**为什么先修这个而不是先排五拍**：审计发现本域的问题在比拍序更靠前的一层——7 篇没有 commit 级基线（[[20_megatron_comm_overlap_analysis]] 只写「基于 Megatron-LM `dev` 分支代码分析」，正是 `CLAUDE.md` 溯源政策点名禁止的 vague latest code），15 篇用 7 位短码，locator 普遍是裸文件名（`moe_layer.py:660`）而非仓库相对路径。在基线没钉住的前提下补写「为什么这么设计」，下一次刷新基线时无法验证。

**基线：一律靠差分证据判定，没有一页是猜的**

- 原写 `ee3f1ff` / `232c478d4` 的：机械扩成 40 位。
- **无基线的 7 篇**：取 3–11 条带行号的引用，分别在 `ee3f1ffa2acd18131ab67cabab4cec45283512ab`（A，2026-05-19）与 `232c478d43ce2f8b4c8db3507d3623fa82f55823`（B，2026-06-16）下 `git show` 逐条核对——
  - [[01_megatron_moe_training_optimization_analysis]] → **B**（5/5 只在 B 命中：`transformer_config.py:881` 的 `moe_flex_dispatcher_backend`、`token_dispatcher.py:1470` 的 `_DeepepV2Manager`、`fused_a2a.py:90` 的 `get_elastic_buffer` 等，在 A 处均为空行或无关代码）
  - [[20_megatron_comm_overlap_analysis]] → **A**（11 处里 8 处只在 A 命中，如 `model_parallel_config.py:196` 在 A 是 `tp_comm_overlap`、在 B 是 `--te-rng-tracker`）
  - [[21_megatron_fusion_operators_analysis]] → **A**（13 处引用在 A/B 逐字节相同、不具区分度，改用 `git diff A..B -- core/fusions/`：只有 `fused_mhc_kernels.py` 变化 964→3397 行，而本页描述的是 A 的形态）
  - [[22_megatron_memory_optimization_analysis]] → **A**（`paged_stash.py:129` 在 A 是 Triton kernel，B 处该文件已被 #5003 迁走）
  - [[25_megatron_nonuniform_tp_analysis]] → **A，但证据不具区分度**：9 处引用在 A 与 B 完全一致，取 A 的理由（同批一致 + frontmatter 日期）**已如实写进页头**
  - [[32_megatron_tflops_analysis]] → **A**（`num_floating_point_operations` 在 A 是双参签名、B 已改四参；页内伪代码逐 token 形式只在 A 成立）
  - [[33_megatron_vllm_weight_sync_analysis]] → **未确定**：它分析的是 `volcengine/verl`，locator 全是 `verl/...` 且无行号；本机 verl 检出 `8a694930` 下其中两个文件已不存在，无法反推。页头如实写「仓库=volcengine/verl、基线未确定」，**未硬钉 Megatron 基线**。

**两处矛盾，由协调者复核后处理**

1. [[34_deepseek_v4_tensor_parallel_analysis]] / [[35_deepseek_v4_context_parallel_analysis]] 页头声明 B，**正文行号却系统性命中 A**：`deepseek_v4_hybrid_attention.py:87-88` 在 A 正是 `get_pg_size(self.pg_collection.tp) == 1` 的 TP=1 断言，在 B 漂到 `:92-93`；`experts.py:328` 在 A 是 `expert TP > 1` 的 unsupported 分支、在 B 无关；`csa.py:297/309/460/473` → B 的 `:313/325/476/489`。改钉 A，并把页头原有那条 2026-06-25 的审计注标明**是在 B 上做的**——这也解释了它当时为何写「行号较旧稿有数行漂移」。
2. [[20_megatron_comm_overlap_analysis]] 文末第 797 行还留着「代码片段均来自 commit `3beeaa65b` **附近**」，与新页头冲突，且「附近」本身不是可核验基线。实测：`param_and_grad_buffer.py:709`、`layers.py:520`、`fused_a2a.py:139` 在 A 命中、在 `3beeaa65b` 不命中；**但** `combined_1f1b.py` 的函数定义行在 `3beeaa65b` 下更准。故以 A 为准，并注明本页历史上经历过局部刷新、少数行号可能仍停留在更早形态——没有硬凑成单一基线。

**locator**：**774 处**裸文件名补成仓库相对路径（三组分别 367 / 234 / 173）。多路径的一律先抽查该行内容再定，例如 `param_and_grad_buffer.py` 按符号计数区分 DDP 版与 FSDP 版、23 号页的 `enums.py` **按行拆分**到 `core/enums.py` 与 `core/transformer/enums.py`。判断不了的保持原样（`23:190` 的 `quantization/utils.py`，两个候选在 A/B 都不含 "fp4"）。属于 DeepEP 等外部仓库的（`internode.cu`）刻意不改。

**索引与 radar**：`megatron-lm/index.md` 新增基线声明，**如实记录本域基线未统一**（多数 A、少数 B、若干未确定），不写成「全域统一基线」。`docs/radar/watchlist.yaml` 的 `kb_baseline` 从短码 `232c478d4` 改为完整的 A —— 原值让 radar 以为 KB 已在 B，会漏报近 300 个提交的漂移。

**本轮明确未做（留给下一轮）**：① 行号刷新——已报告 8 处内容与行号不符的引用（如 `01`/`16` 的 `param_and_grad_buffer.py:946`/`:1097` 两个标签疑似互换、`22` 的 `fp8_utils.py:594`），一律**未擅自改**；② 55 处半路径（`moe/router.py` 这类）与 16 处写在 `###` 标题里的裸文件名——后者改动会同时触碰标题文本与潜在锚点，需单独授权；③ 五拍重排——本域 26 篇里只有 4 篇有第 2 拍内容，是三个域里最缺的，但那是重写而非搬运，必须开着源码做。

---

## 2026-08-27（十）：slime 全域 19 篇按五拍重排，第 5 拍全部锚定源码自陈的在途改动

**Type**: 批量重构（19 页正文；四个并行 writer agent，各自独占一组文件）

**范围**：`02_engineering/04_posttrain_frameworks/slime/` 下 20 篇内容页中的 19 篇。[[02_slime_quickstart_and_configuration_guide]] 是 guide 体，按 vLLM 先例豁免。基线 `THUDM/slime main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`（2026-08-12，与本机检出 HEAD 一致）；[[25_vime_vllm_backend_support_analysis]] 的源仓库 `vllm-project/vime@8144096e` 不在本机，该页只做不需要开源码验证的部分（标题、页头、已有小节前移），未新增任何引用。

**第 2 拍（为什么这么设计）**：19 篇里 **12 篇整节前移**到 §2 并改题「为什么这么设计：…」，**4 篇本来就达标**（[[01_slime_architecture_overview_analysis]]、[[13_slime_sglang_rollout_engine_analysis]]、[[18_slime_fault_tolerance_observability_analysis]]、[[20_slime_on_policy_distillation_analysis]]）只补标题，**3 篇是真缺、需要回源码补写**：

- [[17_slime_train_inference_consistency_analysis]] 新写一节 5 个被否掉的方案，每行挂源码/文档原话：抬到 FP32 → `docs/en/advanced/reproducibility.md:61` 的 *matching precision, not fp32, is what aligns*；全栈 bitwise → `alignment/deepgemm_forward.py:1-8` 自称 *an opt-in numerical-alignment hook* 且只对齐 forward；目标拓扑全开 → 同文件 `:10-12` 第一版主动要求 TP=1 以排除 row-parallel 舍入这个 confounder；校正抹平 → `slime/utils/arguments.py:1849` 禁止 `use_rollout_logprobs` 与 `use_tis` 同开。L0–L5 分层是据实现形态重建，整段挂 `> [!note] 推断` 并写明「源码没有在任何一处写下这句总结」。
- [[30_slime_rollout_optimization_analysis]] 原本第 2、4 拍都没有，补了五行「更简单的做法为何被排除」对照表 + 一整节约束（覆盖前提/代价/故意不做的事/失效条件），四个子节逐条带 locator。
- [[12_slime_sample_datasource_analysis]] 在既有取舍段之上补写被否方案表，整表挂推断。

**第 5 拍（发展趋势）**：**16 篇写了、3 篇明确不写**。写的每一条都锚在实读过的源码注释上，例如 `sglang_utils/arguments.py:8` 的 router 参数前缀 TODO、`examples/fully_async/README.md:82-83` 的 ABORTED 轨迹尚未接 partial-rollout resume、`update_weight/common.py:236` 的 `# TODO shall we handle (almost) all buffers`、`megatron_utils/arguments.py:150` 的 `# TODO: maybe change this after megatron has good fp8 support`、`loss.py:818` 与 `model.py:903` 的两条数值行为待查 TODO。不写的三篇（[[20_slime_on_policy_distillation_analysis]]、[[24_slime_agent_workflow_examples_analysis]]、[[25_vime_vllm_backend_support_analysis]]）是**扫过相关路径零命中**后按规则整拍略去，页头注明并列出扫过哪些路径。另有两处主动弃用的锚点：`slime/utils/arguments.py:607` 的 TODO 因紧邻的 `--num-epoch` 已存在而判定过时；`profile_utils.py:63` 因会把该节稀释成 TODO 清单而未采用。

**修正一处我方预扫描的错判**：协调者用正则扫描时判定 [[16_slime_weight_sync_analysis]] 没有第 2 拍内容，实际它埋在原 §3.1「四个直觉方案为什么不够」，只是位置靠后；真正从零补写的只有 17 与 30。

**校验**：`check_links --strict` 425 页 broken/ambiguous/bare_index/orphans 全 0；`check_math --changed --strict` 0 错 0 警。协调者对四组各抽查 4 条引用逐条 `git show` 复核，**16/16 逐字命中**。既有机制正文与既有 `file:line` 未改，[[20_slime_on_policy_distillation_analysis]] 全页仅 `2+ 0-`。

**未做**：[[15_slime_loss_parallelism_analysis]] 的 §10 操作清单夹在约束与趋势之间（属内容重组，未动）；`update_weight/common.py:47-48` 与 `:114-115` 是同一对 TODO 的重复实现，只在趋势里指出，未据此改写机制正文。

---

## 2026-08-27（九）：vLLM 全域 16 篇按五拍重排，并按源码自陈的在途改动补出第 5 拍

**Type**: 批量重构（15 页正文 + 域索引；接续同日（八）的样板页）

**范围**：`02_engineering/03_infer_frameworks/vllm/` 下 18 篇分析页中的 16 篇（含（八）已改的 [[11_vllm_scheduler_analysis]]）。两篇按文体豁免并在索引里写明理由：[[02_vllm_system_design_principles_analysis]] 是「原始问题 → 四类资源约束 → 五个系统支点」的推导体，第 2 拍本来就是它的第二、三节，前移反而打断推导；`03_vllm_request_flow_walkthrough_analysis` 是端到端走查体，按时序组织。

**改了什么**（机制正文与既有 `file:line` 一律未动，只动章节顺序、标题与新增段落）：

- **第 2 拍前移（14 篇）**：把「替代方案 / 直观方案为何不够」整节搬到 §二，标题统一为「为什么这么设计：…」。改前 16 篇里有 15 篇把它放在 §八～§十，即讲完全部机制之后——旧四拍的形状。[[22_vllm_distributed_inference_analysis]] 本就没有独立的替代方案节，只做其余各拍。
- **第 1 拍显式化（15 篇）**：§一 标题统一加「背景：」前缀。
- **第 4 拍显式化（12 篇）**：把「失败边界与观测 / 验证与排查 / 实现和部署检查清单」等标题改成含「约束」的名字；[[23_vllm_compilation_cudagraph_analysis]]、[[25_vllm_ir_and_fusion_passes_analysis]]、[[26_vllm_disaggregated_kv_serving_analysis]] 三篇原本把「替代方案」和「验证/失败边界」混在一节，按拍拆开：表走第 2 拍，验证与失败边界留在第 4 拍。
- **第 5 拍（10 篇有、5 篇无）**：新增「发展趋势」节，每条都锚在**已逐条打开读过**的源码注释上——`v1/attention/backend.py:534` 的 `Deprecated fields ... (v0.15.0)`、`model_executor/custom_op.py:183` 的 `enforce_enable ... will be removed`、`compressed_tensors.py:290` 的稀疏支持已删除并抛 `DeprecationWarning`、`parallel_state.py:370` 的等待 pytorch#165086、`compiler_interface.py:350` 的等待 pytorch#176562、`rejection_sampler.py:27` 的 chunking 是 workaround、`metrics/loggers.py:461` 的 `show_hidden_metrics`、`kv_connector/v1/base.py:606` 与 `sched/scheduler.py:2503` 的同向 TODO、`registry.py:786` 的 V0 弃用清单、`docs/design/model_runner_v2.md:3-9` 的自陈未 feature-complete。**找不到锚点的 5 篇（10 / 12 / 16 / 25 / 28）直接不写这一拍**，并在页头注明「本页无可锚定的在途改动，第 5 拍略」——技能里第 5 拍是可选且必须锚定，宁缺勿编。
- **推断边界**：每张被前移的替代方案表下加 `> [!note] 推断`，写明「源码通常只陈述最终形态，不陈述被否掉的选项」，要引用请回到对应小节的 locator 而不是引用该表。

**未做**：19 篇仍钉在 `d66300a1`（落后检出 HEAD `26858770` 共 142 个提交、4 天）的基线刷新；[[01_vllm_feature_optimizations_guide]] 是 guide 体，不适用本拍序。

---

## 2026-08-27（八）：把 vLLM Scheduler 页改成五拍叙事，并标出两处被当成源码意图的推断

**Type**: Skill 规则落地 + 单页重构（1 页正文；`source-faithful-analysis` 本轮新增的五拍顺序的首个样板）

**背景**：`source-faithful-analysis` 本轮把分析页的叙事顺序定成强制五拍——**背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势（可选）**。拿 vLLM 这组 20 页对着 `../vllm` 检出逐条核过：溯源这层是全库范本（19 页钉 `d66300a1`、抽查 7 条 `file:line` 全部精确命中），但拍序有系统性偏差——16 篇里有 15 篇把「替代方案」表放在 §八～§十，即讲完全部机制之后。本页作为回填样板先改。

**改了什么**（[[11_vllm_scheduler_analysis]]，机制正文与既有引用一行未动）：

- **前移第 2 拍**：原 §八「为什么不采用几个直观方案」整表搬到 §二，并补一句判据——*任何让「已承诺的资源」与「本轮真实可执行性」脱钩的方案，都会把一次局部容量不足放大成全局抖动*。原 §二～§七顺次后移为 §三～§八。
- **标出两处推断**（用代码验的，不是措辞问题）：① 抢占后不纳新的 `if not preempted_reqs` 守卫**没有任何解释性注释**（`scheduler.py:748` 上方只有 `# Next, schedule the WAITING requests.`），`git log -S` 追到 PR #15250 *[V1] Scheduler Refactoring [1/N]*；② 与 CPU swap 的对比在 V1 里**无源可依**——`git grep -i swap d66300a1 -- vllm/v1/core/sched/ vllm/v1/core/kv_cache_manager.py` 零命中，V1 压根没有 swap 路径，该对比是对 V0 历史方案的重建。两处均加 `> [!note] 推断`，原文照留。
- **第 4 拍显式化**：§九 改名「约束、取舍与观测」，前置一段讲这套设计**不保证**什么（不保证公平、不保证 running 必然跑完、不负责降级）与三个成立前提。
- **新增第 5 拍 §十 发展趋势**，三条全部锚在源码自陈的在途改动上：HMA 收敛（`scheduler.py:2696-2702` 的 `NOTE(Kuntai): We should deprecate this code path...` + `:2877` 的 `TODO (davidb)`）、connector 保活待一般化（`:2500-2508` 的 `TODO`）、`scheduler_cls` 可替换（`vllm/config/scheduler.py:117,170-190`；顺带更正一处易错点：`AsyncScheduler` 是 `Scheduler` 的子类，不是 `SchedulerInterface` 的第二个独立实现）。外推的那半句单独标了推断。

**未做**：其余 18 篇 vLLM 页的同类回填；19 页停在 `d66300a1`（落后 HEAD `26858770` 共 142 个提交、4 天）的基线刷新。

---

## 2026-08-27（七）：修好「子目录 index.md 在网页上是空白页」，并给全库 419 页补 frontmatter 标题

**Type**: Docs-site Fix（1 个插件 + 419 页 frontmatter + 3 处测试补强）

**根因**（逐行读 Quartz v5 源码定位，不是猜的）：

1. `folder-page` 的 `match` 是 `slug.endsWith("/index")`、priority 10；`content-page` 的 `match` 对 `*/index` **显式返回 false**。所以每个子目录 `index.md` 的正文渲染权**完全交给** `FolderContent`。根 `wiki/index.md` 的 slug 是 `index`（不以 `/index` 结尾），走 `content-page`——这解释了"只有子目录 index 打不开"。
2. `FolderContent` 里 `const trie = ctx?.trie; if (trie) { if (!trie.findNode(...)) return null }`——`pagesFromAllFiles` 那条好用的 fallback **只在 trie 不存在时**才走，而 `dispatcher.ts` 用 `ctx.trie ??= trieFromAllFiles(allFiles)` 保证了 trie 一定存在。于是命中"trie 存在但查不到"的死角。
3. **真正的根因**：`quartz/util/ctx.ts` 的 `trieFromAllFiles` 只收 `file.frontmatter` 为真的条目，而 `fileData.frontmatter` 由 `github:quartz-community/note-properties` 填充——**我们的 `quartz.config.yaml` 从来没启用过它**（只启用 19 个插件，Quartz 官方默认配置是 42 个，`note-properties` 在其第 212 行）。全库无 frontmatter ⇒ trie 零节点 ⇒ `findNode` 返回 undefined ⇒ 空白页。

同一根因还解释了另外两个一直存在的现象：所有页面的 `<title>` 都是站点名（`Head.tsx` 用 `fileData.frontmatter?.title ?? cfg.pageTitle`）；`article-title` 全站不输出 `h1`。

**修复**：

- 启用 `note-properties`（pin `e68145b9`，`order: 5` 最先跑，`hidePropertiesView: true`），同步写入 `quartz.lock.json` 与 `runtime-manifest.json`，`npm run docs:repair` 重建运行时。
- 只启用插件会让标题回退到**文件名 stem**（tab 显示 `index`、每页多一个 `<h1>index</h1>`、自动目录清单列出 `01_llm_inference_technology_stack_analysis` 这种原始文件名），所以配套做两件事：
  - **给全库 419 页补 frontmatter `title:`**，取值为各页正文首行 `# 标题`（去掉 `**`/反引号），YAML 双引号转义，**逐文件保持原有 CRLF/LF 行尾**（273 CRLF + 146 LF）不动，避免整库换行 diff；页面正文一行未改。
  - **关闭 `article-title`**——本库每页正文首行就是 `# 标题`，再渲染一个 h1 会重复。标题改由 frontmatter 供给 `<title>` / explorer / search / folder-listing。
- 效果（构建产物实测）：`02_engineering/index.html` 21,982 → 30,532 字节；`03_infer_frameworks/index.html` 41,128 字节；三处抽查均为 `<title>` = 真实标题、正文 `<h1>` 恰好 1 个、自动目录清单显示真实标题而非文件名。

**测试补强**（这个 bug 之前**没有任何测试覆盖**，所以先补测试）：

- `config.test.mjs` 新增两条用例：① `note-properties` 必须存在且 `order` 小于所有其他 transformer——否则整站子目录 index 变空白；② `article-title` 必须保持 `enabled: false`。
- `smoke.mjs` 新增子目录 index 的端到端回归断言：`<title>` 来自 frontmatter、`article` 正文长度 > 500 字符、存在 `.page-listing`、`article h1` 恰好 1 个。首页 `<title>` 断言同步改为真实标题。

**顺带修的两处测试**：

- `tools/labs_torch_compile/test_volume_demo_contract.py` 有 4 处硬编码 `"01_ai_frameworks"` 作为**真实 wiki 页面路径根**（不是 artifacts 里的历史字符串），被本日（四）的改名打破，同步改为 `"01_pytorch"`，变量 `ai_frameworks_root` → `pytorch_domain_root`。
- `tools/test_math_skill.py::test_skills_live_in_exactly_one_shared_place` 的 `rglob` 排除表漏了 `.worktrees/`——一个 git worktree 是同仓库的另一份合法检出，不是"技能被复制"。补上排除项，否则任何人开 worktree 都会让本用例变红。

- 验证（四道门禁全绿）：`check_links --strict` 419 页 **0/0/0/0**；`check_math --changed --strict` 420 文件 **0 错 0 警**；`pytest tools/` **107 passed**；`npm run docs:test` **PASS**（含新增的空白页回归断言与 141 次全 loopback 请求检查）。

---

## 2026-08-27（六）：给 `02_engineering` 七个一级域补模块总览，并按产品分级

**Type**: Restructure（7 个域 `index.md` 各加一节「模块定位」，不新增内容页）

**动因**：七个域的 `index.md` 此前都是**链接表**——标题写着「目录索引」，内容是"有哪些页面"，但不回答"这个模块在整个栈里解决什么问题、提供什么能力、边界在哪"。从宏观进入知识库时没有入口。

**统一结构**（每个域一节，插在链接表之前）：一句话定位 → 为什么必须独立成一层 → **本域覆盖的系统/技术栈与各自定位** → 本域提供的能力（能力 / 具体提供什么 / 样本与源码锚点 / 详见）→ 不属于本模块的 → 与兄弟域的关系。

**分级是这次的重点**：初稿把 `03_infer_frameworks` 整节写成了 vLLM 的介绍，SGLang / Mooncake / 投机推理都没有位置——这是把"覆盖最多的产品"当成了"整个域"。返工后每个多产品域都先给一张**系统定位表**，并如实标注覆盖不均衡：

- `03_infer_frameworks`：vLLM 19 篇（系统性）· SGLang 1 篇（单点切入）· Mooncake 1 篇（论文）· 投机推理 2 篇（技术专题，非产品）· TensorRT-LLM/llama.cpp/TGI **0 篇，仅在对照表出现**。
- `02_train_frameworks`：Megatron-LM 26 篇 + TorchTitan 15 篇（两条系统性主线）· MindSpeed 5 篇（**机制级深挖，非特性全量走查**）· MindFormers 2 篇（**仅 MoE EP 一个切面**）· 跨框架专题 6 篇。
- `04_posttrain_frameworks`：slime 20 篇 + verl 11 篇（系统性）· AReaL / ROLL 各 1 篇（架构专题）· TRL/NeMo-RL/Tinker/KDFlow **0 篇，仅在 OPD 对照表出现**。
- `05_gpu_kernel`：按**来源性质**分级——只有 Triton 8 篇是源码级可核验（`@70e0929`）；CUDA GEMM/非 GEMM 与 Ascend 三篇来自本地 HTML 快照，TileLang 一篇是概念分析（**本地无实现源码**）。
- `06_auto_parallel`：Alpa/nnScaler/Galvatron/GSPMD/DTensor/MindSpore 六个系统**全部无专页**，只在综述页内各占一节——本域目前提供判断力而非细节。
- `07_training_reliability`：按**证据强度**分五档（本地可核验源码 / 厂商报告 / 工程博客 / **二手综述稿** / 本库一手页交叉），并写明本簇三篇主内容页是对一份二手综述的结构化摄入，以及"公开得多 ≠ 做得好"的样本偏差。
- `01_pytorch`：单框架，但按**代码库**分级——上游 `pytorch/pytorch` ~116 篇 vs `torch_npu` 28 篇（`b3c8a815b`）vs CUDA 特定 4 篇，并重申"硬件无关放本层、硬件特定下沉 `npu/`/`cuda/`"的读法。

**源码锚点的核对方式**：能力表里每条锚点都在侧车 checkout 里核对过**路径存在**（`pytorch@ea5655fc`、`vllm@26858770`、`Megatron-LM@232c478d4`、`torchtitan@a3168782c`、`verl@8a694930`、`slime@681b3adc`、`triton@70e0929`、`torch_npu@b3c8a815b`）。锚点粒度是**模块路径级**，不是 `file:line`——页面正文里的行号级定位仍以各页自己的基线为准，两者在节首都做了区分标注。

**顺带修**：`06_auto_parallel/index.md` 的 5 处 `../` 相对上跳链接改为 wiki 根路径（违反跨引用规则），并修掉因 MindSpore 页删除而失效的一行描述。

- 验证：`check_links --strict` 419 页 **0 破损 / 0 歧义 / 0 裸 index / 0 孤儿**；`check_math --changed --strict` 193 文件 **0 错 0 警**。

---

## 2026-08-27（五）：跟进 TorchTitan 320 个提交，重做并行主线并补齐编译运行时与 TitanRL

**Type**: Source Re-audit（TorchTitan `main@a3168782c`，相对旧基线 `61c010fcb` 前进 320 commits；16 篇训练框架专题 + 1 篇后训练专题）

- **基线冻结**：以 sibling checkout `pytorch/torchtitan` 的干净 `main@a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）为唯一当前代码基线；每个非平凡机制结论重新落到该提交的 `file:line`。旧页中继续需要的 PyTorch 2.9.1 内部机制则保留为清楚标注的独立固定基线，没有用 TorchTitan 行号冒充上游实现。
- **新增 4 篇主线页**：[[02_engineering/02_train_frameworks/torchtitan/01_torchtitan_trainer_quickstart|Trainer 入口与生命周期]]、[[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|SPMD Types]]、[[02_engineering/02_train_frameworks/torchtitan/26_torchtitan_flex_shard_dist_muon_analysis|FlexShard / DistMuon]]、[[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|GraphTrainer 编译运行时]]。这四篇补上旧知识域完全没有覆盖的 Python recipe、声明式布局、storage/compute layout 分离、显式 all-to-all 双槽流水、joint FX graph、Graph PP 与 EP overlap。
- **重做并行主线**：ParallelDims、FSDP、TP、CP、PP、EP 六页全部切到当前接口。最关键的更正是：`spmd_types` 已成为默认后端；全局 `full_dtensor` 后端与旧 CP `apply_cp_to_forward` 已删除；CP 当前通过 input sharder 与 K/V shard→replicate redistribution 表达；Async TP 已移入 compile；PP 是 stage/微批次适配层而非 action/P2P 调度器；EP 已统一为 `token_dispatcher + inner_experts`，DeepEP v2 与 MinimalAsyncEP 的组合边界也按当前测试重写。
- **更新资源优化专题**：FSDP prefetch、HSDP overlap、activation checkpoint、计算/显存优化、通信 overlap、SimpleFSDP 六页重审当前接线。补入 NVFP4、显式 dist-GEMM、core Trainer CUDA Graph、chunked loss、对称内存、GraphTrainer memory policy，并把 activation checkpoint 与权重 checkpoint 分开；SimpleFSDP 页明确区分实验内部的局部布局选项与已经删除的全局 `parallelism.spmd_backend="full_dtensor"`。
- **补当前 TitanRL**：[[02_engineering/04_posttrain_frameworks/10_rl_ppo_loss_and_grpo_analysis|TitanRL 异步控制器与 GRPO/DAPO]] 从旧同步 PPO/vLLM 叙述改为当前 `experiments/rl` 的 controller/policy version、windowed FIFO、rollout worker、批处理、版本化权重同步与 loss/advantage 数据流。
- **主动记录证据冲突**：GraphTrainer README 把 CP 标为可用，但当前构造路径强制 `partial_dtensor`，相应 tests 也禁用 CP，因此页面按源码与测试把它记为“当前未启用”；GraphTrainer tracer 虽能捕获 optimizer，生产 `GraphTrainer` 的 step graph 当前仍不包含 optimizer。PP 的 zero-bubble/custom CSV 配方存在，但相关核心测试因 FlexAttention metadata 问题禁用，也没有升级成“已验证可用”。
- **导航与监控**：重建 [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]，同步训练框架索引、工程索引和全库首页；`docs/radar/watchlist.yaml` 的 TorchTitan 知识基线更新到 `a3168782c`。
- **验证**：`check_links --strict` 覆盖 419 页，0 破损/0 歧义/0 裸 index/0 孤儿；`check_math --changed --strict` 覆盖 193 个变更页，0 错 0 警；`npm run docs:test` 为 63/63 单测与 133 请求浏览器 smoke 全通过；`pytest tools/` 为 **103 passed / 4 failed**，4 个失败均来自同期 `01_ai_frameworks` → `01_pytorch` 目录改名后 `tools/labs_torch_compile/test_volume_demo_contract.py` 仍引用旧路径，与本次 TorchTitan 页面无关。

---

## 2026-08-27（四）：`01_ai_frameworks` → `01_pytorch`，并删除其它框架对照子域

**Type**: Restructure（域目录改名 148 页 + 删除 2 页 + 全库链接回写）

**改名理由**：该域 150 篇里有 148 篇是 PyTorch，页面自身的标题早已是「PyTorch 编译与运行时架构」，`ai_frameworks` 这个名字既没说清是 PyTorch，也没说清它是 `02_train_frameworks`/`03_infer_frameworks`/`04_posttrain_frameworks` 共同的底座——命名轴和内容对不上。

- **目录**：`wiki/02_engineering/01_ai_frameworks/` → `wiki/02_engineering/01_pytorch/`（148 个 `.md` 经 `git mv` 逐文件迁移；顶层目录整体 `rename` 在 Windows 上被并发进程占用而失败，改为逐文件迁移，git 全部识别为 rename）。
- **删除**：原 ⑤ 层 `05_other_frameworks/`（`index.md` + `10_mindspore_compiler_analysis.md`，347 行 MindSpore 编译器概念分析）整体删除，本域收敛为纯 PyTorch 四层。
- **链接回写**：80 个页面内的 `01_ai_frameworks` 全量替换；`wiki/changelog.md` 与 `changelog/2026_q2_and_earlier.md` **只改活链接、保留反引号里的历史路径**（本文件只追加、不回写历史的约定）。
- **指向已删页的入链**：`auto_parallel_survey_analysis` 3 处、`mlir/10_mlir_core_concepts` 1 处直接删除条目；`changelog/2026_q2_and_earlier` 1 处按约定转为反引号示例并注明删除日期。
- **域外路径**：`README.md`（域表 + 3 条推荐入口，页数 150→148）、`docs/radar/watchlist.yaml` 两处 `kb_entry` 同步改名。`raw/02_engineering/01_ai_frameworks/` **不改**——`raw/` 与 `wiki/` 是独立树、且 `raw/` 只读，`wiki/index.md` 里指向它的那一行保持原路径。
- **层数口径**：`01_pytorch/index.md`「五层架构导航」→「四层」，`courses/torch_compile_end_to_end` 同步。
- 验证：`check_links --strict` 419 页 **0 破损 / 0 歧义 / 0 裸 index / 0 孤儿**；`check_math --changed --strict` 182 文件 **0 错 0 警**。

> 遗留（本次未处理，属既有债务）：`tools/labs_torch_compile/README.md` 等处仍写着 `wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/...`——该路径在 kb-reorg P4 解散 `19_` 课程目录时就已失效，与本次改名无关。

---

## 2026-08-27（三）：为 GLM-5.3-Flash 与 Qwen3.8-Flash-Next 补结构图，并证伪一处流传的参数口径

**Type**: Figures + Source Verification（4 张 HTML→PNG 结构图 + 2 处配置级核实）

- 按库内既有的 `tools/figs/*.html` → `render_figs.mjs` → `assets/*.png` 流水线新增 4 张图：`glm53_flash_architecture_fig1/fig2`（整机结构 / DSA 索引器数据通路）与 `qwen38_flash_next_architecture_fig1/fig2`（整机结构 / QSA 数据通路 + 两阶段 CPT）。Qwen 两张复刻自技术报告 Figure 1 与 Figure 3；GLM 两张依据 `config.json` 与 FP8 豁免清单重画。

**两处配置级核实**：

- **mHC 的逐子层放置得到确证。** 此前只知道 `mhc: true`；这次从 `quantization_config.modules_to_not_convert` 逐层读出 `hc_attn_base/fn/scale` 与 `hc_ffn_base/fn/scale` 六项，**45 层主干层层齐全、第 45 号（MTP）层一项都没有**——即注意力子层与 MoE 子层各有一处 mHC，且 MTP 层不参与。`base / fn / scale` 三件套恰好对应超连接公式的「静态项 + 数据相关项 + 缩放」。
- **`index_kpool` 的实现形态被暴露。** DSA 层独有模块含 `indexer.index_kpool_compress_ape` 与 `indexer.index_kpool_compress_gate`，说明 GLM 的 4× 键池化是**带位置嵌入的可学门控压缩**，而非 [[20_qwen3_8_flash_next_architecture_deepdive|Qwen QSA]] 那样的非重叠块 AvgPool。**两家压缩比同为 4，实现路线不同。** 另新增 §2.4 指出 `index_kpool` 与 [[11_glm_5_1_5_2_analysis|5.2 的 IndexShare]] 是两条正交降本路径，且 5.3-Flash 的 `indexer_types` 全为 `full`——**没有沿用 IndexShare**。

**证伪一处流传口径**：

- 流传的第三方架构图把 GLM-5.3-Flash 标注为「320B total **13B** active」，与模型卡的 **18B** 冲突。按 `config.json` 逐模块估算得 **17.30B**（MoE 9.97 + KDA 4.58 + DSA 1.48 + 头 0.63 + 嵌入 0.63），**距 18B 仅 0.70B**（未计入的 mHC 开销正落在这个量级），**距 13B 差 4.30B**——即便剔除嵌入与输出头也仍有 16.03B，没有任何合理口径能填平。**模型卡的 18B 站得住，13B 站不住。**
- 同一张图的两条效率曲线（1M 处 KV-cache 4.44×、attention compute 3.01×）**未采纳**：计算方法未公开；且其中作为对照的 "GLM-5.3"（区别于 GLM-5.3-Flash）**在公开渠道不存在**（HF 上 `zai-org/GLM-5.3` 返回 401，仅 Flash 版公开）。页面用一张三态表把「已核实 / 已证伪 / 无法复核」逐项标了出来。

- 方法上值得记一笔：**第三方图不是拿来照抄的，是拿来逐条对配置验的**——结构部分全对上了（因此采纳并重画），参数口径当场证伪，效率曲线无从复核。三种处置方式写进同一页，读者才知道哪部分能引用。
- 验证：`check_links --strict` 417 页 0 破损/0 歧义/0 孤儿；`check_math --changed --strict` 0 错 0 警；4 张图的引用路径逐一存在性检查通过。

---

## 2026-08-27（二）：摄入 Qwen3.8-Flash-Next 技术报告，回写三处已被推翻/确认的推断

**Type**: Source Ingestion（28 页 PDF 技术报告 → 2 篇深挖页 + 3 处既有页回写）

- 摄入《On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability》（Qwen Team，2026-08-26，**无 arXiv 编号**，随 GitHub 仓库发布）。按库约定**不入库 PDF 原文**，只建来源索引页 `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Flash_Next_tech_report.md`（链接 + 元数据 + 章节定位表 + 关键外部引用核实）。

**新增 2 篇深挖页**：

- [[20_qwen3_8_flash_next_architecture_deepdive]]（§2 架构）—— GDN 门控 delta 规则的两个门分工；**QSA 是在 CPT 阶段才替换掉全注意力层的**（Stage 1 只训 indexer 1000 步 / 2B token，Stage 2 联合训练 8000 步 / 200B token），教师分布用 **MaxPool** 而 key 压缩用 **AvgPool** 的刻意区分；**key 压缩发生在位置编码之前**以避免平均不同旋转相位；GR 的五条消融与瓶颈秩 $r=d/8=320$ 同 config 精确对上；Fig. 7 的跨层路径分析（**一条分支承载长程、另三条局部；中程路径合计 −3.21**）。
- [[21_qwen3_8_flash_next_optimization_deepdive]]（§3–§4 优化）—— Muon 的权重归属清单（**路由器与 GR 低秩投影走 AdamW**，并给了"路由器各输出维独立、无共享线性结构可供正交化"的解释）；**Megatron 的融合 qkv/fc1 直接正交化是错的**（混合不相关子块 + 缩放因子用错形状），须按 per-head 拆分；**取消 batch-size warmup** 的完整论证（两个变体都更差，且多花 18.8% 优化器步数）；压力测试在 4× 最优 LR 下 **AdamW 尖峰 183 次/万步 vs Muon+GR 零尖峰**，并在 8× 模型规模、生产 LR 下复现。

**回写既有页面（源头已给出结论，不留两套说法）**：

- [[12_qwen3_8_flash_next_analysis]] —— 页首 warning 改为 note；四处"边界"按报告实际内容拆成**已补齐**与**仍未解决**。其中两条此前的判断被更正：① GR 与 HC/mHC 的谱系归属由【推断】**升级为事实**（报告原文 "GR belongs to the same family as HC, mHC and VWN"），但"Qwen 采用了 mHC"仍是错的——GR 把表达力押在读上并**丢弃 $H_{\mathrm{res}}$**；② 此前写"卡片没有引用 Engram、承袭关系无法判定"，而报告 §2.3 **明确引用了** DeepSeek 的 Cheng et al., 2026，应予更正。
- [[11_glm_5_1_5_2_analysis]] —— 新增 §3.5：**唯一一份对 IndexShare 的公开第三方评测**（Qwen 实测 QSA 在相对 indexer 延迟 0.25 处持平全注意力，IndexShare 在 0.5 处仍低于基线），并列出**三重必需的限定**（对比在混合架构下先天不利于 IndexShare、两者度量的量不同、智谱侧无可比数据）。另做**命名订正**：arXiv:2603.12201 的实际标题是 "IndexCache" 而非 IndexShare，检索时两个名字都要试。
- [[12_glm_5_3_flash_analysis]] —— 就其 `qk_rope_head_dim: 0` / NoPE 设计补反驳栏：**Qwen 在同类混合架构上实测 NoPE 后拒绝了它**（预训练几乎无差别，但后训练后"不停止生成"比例明显更高）。本库此前"位置信息由线性层承担故可去 RoPE"的推断，据此从"看似合理"降级为**已被至少一家厂商实测推翻的假设**。

- 这三处回写都属于同一类价值：**跨页面的交叉核对把单页里看不出的问题暴露出来**。三条都不是本库写错了，而是当时的公开材料不足以判断；报告出来后必须回写，否则库里会同时存在两套说法。
- 验证：`check_links --strict` 417 页 0 破损/0 歧义/0 孤儿；`check_math --changed --strict` 9 文件 0 错 0 警（新增内容曾引入 22 个警告：`H_{res}` 一类语义下标须 `\mathrm` 包裹，两处 display 公式过长须改 `aligned`，均已修）；`pytest tools/` 107 passed；`npm run docs:test` PASS。

---

## 2026-08-27：摄入 9 个新发布模型，补齐 GLM / Qwen / Kimi / DeepSeek 的覆盖缺口

**Type**: Source Ingestion（模型卡 + 权重配置，9 个 checkpoint → 6 篇分析页）

- **缺口**：库内 GLM 停在 GLM-5（2026-02）、Qwen 只有 2.4T 旗舰档、Kimi 的 K2.6 在索引里还标着“待发布”、DeepSeek-V4 的两个**正式版 checkpoint** 零覆盖。本次一次补齐：GLM-5.1/5.2/5.3-Flash、Qwen3.8-27B/Flash-Next、Kimi-K2.6/K2.7-Code、DeepSeek-V4-Pro-0813/Flash-0731。
- **方法**：走 `source-faithful-analysis`。每个模型同时快照**模型卡**（厂商自述）与 **`config.json`**（架构硬证据）到 `raw/01_theory/01_models/<厂商>/`，并钉住 HF revision sha；参数量取自 safetensors 索引。**卡片声称与配置可核验严格分开**，凡推断均标注 `【推断】` 与边界。

**新增页面（6 篇）**：

- [[11_glm_5_1_5_2_analysis]] — 全键比对证明 **5.1→5.2 骨架一个字段都没动**；变化只有上下文 202752→1048576、`rope_theta` 1e6→8e6，以及 **IndexShare**（78 层中仅 21 层建全索引 = 26.9%，`index_topk_freq=4`）。并纠正一个陷阱：`head_dim` 64→192 **不是**结构变化，两版 `qk_head_dim` 同为 256，是 transformers 5.4/5.12 填这个冗余字段的口径不同。
- [[12_glm_5_3_flash_analysis]] — GLM 首次**换底座**：45 层 = 34 层 KDA + 11 层 DSA。最值得记的一条是**跨厂商趋同**——`mhc/hc_mult=4/hc_sinkhorn_iters=20` 与 DeepSeek-V4 **逐个相同**，层类型字面写作 `deepseek_sparse_attention`，线性层配置字段是 `kda_layers`。另标出该卡片是本批 9 份里**唯一不含任何文本基准分数**的。
- [[11_qwen3_8_27b_analysis]] — 27.8B **稠密** VL；舍弃了 MoE 却保留 3:1 混合注意力。点破 1M 上下文是 262K 的 4× YaRN 外推（需使用方改配置），且 SWE-bench Pro 一行是阿里“修正题目后重测所有基线”的口径，与公开榜单不可直接比。
- [[12_qwen3_8_flash_next_analysis]] — **Qwen4 架构预览**（`model_type: qwen4_exp`）。四项创新逐条对上配置：QSA 的 `indexer_budget 2048 ÷ compress_ratio 4 = 512` 微块、Gated Residual 的 `hc_count=4/hc_lowrank=320`、2000 万条 n-gram 嵌入（`ple_layer_ids=[2]`、`heads_per_ngram=8`）。参数账 **125B + 51B + 4B = 180B** 与实测精确吻合——只读“125B”会低估 44%。指出其 n-gram 嵌入与 [[29_engram_analysis]] 高度同构。
- [[15_kimi_k2_6_k2_7_analysis]] — **K2.6 与 K2.7-Code 的 `config.json` 逐字段相同、参数量同为 1026.9B**，是同一副骨架上的两次纯后训练。K2.6 的能力提升呈清晰梯度（工具类 +16~26，知识推理近噪声）；K2.7-Code 主打思考 token −30%，但**只给编码类基准、通用能力是否退化无数据**。
- [[31_deepseek_v4_released_checkpoints_analysis]] — 拿**发布权重**核对论文（[[30_deepseek_v4_audit_analysis]] 的续篇）：§4.2.1 超参**十四项全中**，连“Pro 前 2 层 HCA、Flash 前 2 层纯 SWA”这种不对称细节都被 `compress_ratios` 逐层坐实。三项论文之外的新事实：**DSpark 已内嵌进权重**（`dspark_markov_rank` Flash=256 正是论文默认 r）、**专家以 FP4 发布**、**Flash 实测 304.2B 比论文的 284B 多约 20B**。并指出正式版相对 Preview 的跃迁（DeepSWE 12.8→62.7）**不可能由 DSpark 解释**——投机解码不改输出分布，成因完全未披露。

- **索引订正**：GLM 家族表里 `GLM-5.1 | 754B | 最新迭代 | -` 的占位符、Kimi 家族表里 `K2.6 | 1.1T | 待发布`（实为 2026-04-14 已发布、1026.9B）与 `K2.5 | 1.1T`（页内实为 1.04T）均已订正；四个厂商 index、演进时间线与架构树同步更新；`wiki/index.md` 模型域各行**重新统计**而非递增。
- **已登记的最大缺口**：Qwen3.8-Flash-Next **有技术报告 PDF 但本次未摄入**，QSA/Gated Residual/N-gram Embedding 的动机与消融都在其中，已在页首 `> [!warning]` 与 Qwen index 标为最高优先级。
- 验证：`check_links --strict` 415 页 0 破损/0 歧义/0 孤儿；`check_math --changed --strict` 19 文件 0 错 0 警；`pytest tools/` 107 passed。

---

## 2026-08-26：新增上游雷达，每周追踪仓库演进 / 模型发布 / 前沿论文

**Type**: Tooling + Scheduled Maintenance

- 新增 `tools/radar.py` 与 [`docs/radar/watchlist.yaml`](https://github.com/suhaibo666/llm-knowledge/blob/main/docs/radar/watchlist.yaml)：每周扫 12 个仓库、7 家模型厂商（HuggingFace）、5 个 arXiv 论文主题，产出 `docs/radar/<日期>.md`。已注册为每周一的本机定时任务 `llm-knowledge-upstream-radar`。
- **报告第一节就是“哪些 KB 基线已过期”**——首次运行实测：torchtitan 落后 308 个提交、Megatron-LM 276、verl 266、vLLM 229、slime 23，并附上游最近提交标题，直接指向对应域的 `index.md`。
- **刻意的边界：只报告事实，不写 `wiki/` 分析页。** 本库价值在于每条断言都有可核验定位符，无人值守产出的机制级结论没人复核会污染这个前提。要落成分析页仍走 `source-faithful-analysis`；相应地，Ingest Workflow 增加第 7 步：合并新分析后必须同步更新 watchlist 里的 `kb_baseline`，否则雷达会每周重复报同一批陈旧漂移。
- 清单显式维护而非从页头自动推导，原因写在文件头：页头基线格式不统一（代码基准 / 代码基线 / 源码基线，`verl main @ 8a694930` 与 `vllm-project/vllm@d66300a1…` 并存），且 Megatron-LM / torchtitan / MindSpeed 在库里是旁置 checkout 引用，页面里根本不出现 github URL，靠 URL 反推必然漏。
- 三处经实测调整：① arXiv 查询改用 `ti:` 标题匹配 + LLM 约束 + 类别约束，此前 `abs:` 把外科识别、无线边缘、加密货币预测都捞了进来；② 每主题上限 8 条并如实标注截断多少，避免静默截断被读成“本期就这些”；③ tag 只跟踪发布形态并限量 80——pytorch 有 6617 个 tag，其中 2751 个 `viable/strict`、1000+ 个 `ciflow`，全存会让 `state.json` 到 286KB 且每周提交，过滤后 20KB。
- 采集失败如实列出（不会被伪装成“本期无变化”），单源失败不影响整轮。首次运行 12 仓 0 失败。
- 验证：`pytest tools/test_radar.py` 13 passed（全离线，不打网络）。

---

## 2026-08-26：修好本地文档站的 provisioning 阻塞，并默认对局域网提供服务

**Type**: Docs-site Tooling Fix + Network Default Change

- **根因：全新环境装不起来。** `docs:repair` 稳定复现 `Plugin commit drift for explorer: expected def459f…, detected 06ea3d8…`。链条是：`quartz.lock.json` 确实钉死了各插件 commit，但 provisioning 调的是 `quartz plugin install`——**它解析到各插件分支的最新提交**，不是锁文件里的那个；装完与 manifest 校验不符，整个 provisioning 失败。也就是说**上游任何一个社区插件推一次新提交，全新环境就再也装不起来**；已有 runtime 因为校验通过被直接复用，所以老环境看不出问题。
- **修法**：改用 Quartz 自带的 `quartz plugin restore`（其帮助明确写 “Restore plugins from lockfile (exact versions)”）。修完 `docs:repair` 退出码 0，explorer 回到钉死的 `def459f…`。补了回归单测断言用的是 `restore` 而非 `install`。
- **监听地址默认改为 `0.0.0.0`**（原先补丁把 Quartz 从“全网卡”改成只绑 `127.0.0.1`）。新增 `--host` 参数可退回回环；端口预检按所选 host；就绪探测仍走 `127.0.0.1`（`0.0.0.0` 只表示“所有接口”，不是可连接目标）；启动日志列出本机各网卡地址。
- **同时修掉一个只改绑定会踩的坑**：注入页面的热更新脚本原先把地址写死成 `ws://127.0.0.1:8081`，远程浏览器会去连“它自己”的 8081，热更新必然失效。改为由浏览器按当前页面 hostname 推导（`location.hostname`），远程访问热更新同样有效。
- listener 审计从硬编码 `127.0.0.1` 改为按配置 host 校验（通配绑定接受 `0.0.0.0` 或具体网卡地址）；smoke 显式钉在 `--host 127.0.0.1`，端到端测试不在局域网上开端口。
- **安全影响**：站点默认对同网段可见。不可信网络里用 `npm run docs -- --host 127.0.0.1`，README 已写明。
- 实测：两个端口均 `0.0.0.0` LISTENING；从 `192.168.205.175:8080` 取页返回 HTTP 200；页面内 WS 片段为 `new WebSocket('ws://__DOCS_WS_HOST__:8081'.replace(…, location.hostname))`，已无硬编码回环。
- **未覆盖项（本机环境限制）**：`npm run docs:test` 里的浏览器端到端一段跑不起来——`puppeteer-core` 无法拉起 Edge（本机没装 Chrome），最小复现里不涉及任何 docs-site 代码也同样失败，属于环境能力问题而非本次改动引入。单测（63）与上面的手工端到端验证均通过。

---

## 2026-08-26：CLAUDE.md 收回基本法，文档操作规则下沉到公共 skills

**Type**: Repository Governance Restructure

- **技能去重**：此前 `.claude/skills/` 与 `.agents/skills/` 各存一份副本，且已经漂移（`source-faithful-analysis` 里宿主说明一处写 `CLAUDE.md`、一处写 `AGENTS.md`），还要靠一条单测强行比对来维持同步。现在合并为唯一的公共目录 [`skills/`](https://github.com/suhaibo666/llm-knowledge/blob/main/skills/README.md)，那条同步单测随之删除。
- **CLAUDE.md 瘦身 187 → 约 60 行**：只保留基本法——三层结构、功能树的唯一权威地位、Courses 导读层约束、溯源政策、质量门禁，外加技能索引。**具体怎么写文档不再默认加载**。
- **新增两个技能**（内容自 CLAUDE.md 原样搬迁，未改写约定）：`maintaining-llm-knowledge`（页面类型与 `NN_` 段位编号、命名、Ingest/Query/Maintenance 流程、交叉引用规则、合并优于并存）与 `writing-mermaid-diagrams`（Mermaid 定界符两档严重度与生成后校验清单）。
- **`writing-obsidian-math` 刷新**：补上 `MATH001–005/101–105` 全部规则对照表、竖线语义选择表（`\mid` / `\lvert...\rvert` / `\lVert...\rVert` / `\,\|\,` / `\big\vert`），以及本轮清零踩到的四个**伪告警陷阱**（转义方括号不是公式、`\\[2pt]` 是行距、以数字开头的行内公式不是货币、`h_{t-1}` 是索引不是标签）与长公式改写的三条硬约束（每行补 `\\`、`\left/\right` 不能跨行、源码换行不等于换行）。
- **只保留 Claude 与 Codex**：删除重复的 `.agents/` 与 `opencode.json`；`.claude/` 只剩本地 settings，`.codex/config.toml` 保留；`AGENTS.md` 改为工具中立的入口，指向基本法与公共技能。
- **补充（同日）**：`.claude/skills` 建为指向 `skills/` 的软链接（git 存为 symlink 对象 mode `120000`），Claude Code 因此恢复原生技能发现，物理副本仍只有一份。**Codex 侧不建对称软链接**：查证 `codex` 无 `skills` 子命令、技能来自全局 `~/.codex/skills/` 与 plugin marketplace、无项目级技能发现，建了也不起作用；其加载路径就是读 `AGENTS.md` 里的技能表。单测守两条不变量：agent 侧 skills 路径必须 resolve 到 `skills/`，且每个技能只有一份物理副本。`CLAUDE.md` 改为全英文。
- **仓库 agent 工程清理（同日）**：删除 `.codex/config.toml`——`codex doctor` 实测只加载全局 `~/.codex/config.toml`（其中已自带 `[mcp_servers.*]`），仓库这份从不被读取，且写死了本机绝对路径；`.codex/` 随之清空删除。`.mcp.json` 改用相对路径（`./wiki`、`./raw`，已实测服务器能在仓库根正常启动），qmd 由本机绝对 node 路径改为 PATH 上的 `qmd mcp`（其 `--help` 给出的官方调用形式），并去掉绝对 `cwd`；至此全部被跟踪的配置文件里不再有任何本机绝对路径。`.obsidian/workspace.json`（“哪些面板/文件被打开”这类个人 UI 状态，不是 vault 配置）取消跟踪并加入 `.gitignore`。
- 验证：`pytest tools/` 94 passed（技能单测由「两份副本必须一致」改为「技能必须覆盖检查器全部诊断码 + 记录已知伪告警陷阱」）；`check_links --strict` pages=409 全 0；`check_math --strict` 全库通过；skills/ 与两份入口文档自身公式检查 0 error / 0 warning。

---

## 2026-08-26：全库公式 warning 清零（464 → 0）+ 修正 MATH103 规则

**Type**: Corpus-wide Math Normalization (warnings) + Checker Rule Fix

- 承接同日 error 清零，本轮把 464 条 warning 也清到 **0**；`check_math --strict`（warning 也算失败）现已通过。
- **MATH101（181）**：168 处单行 `$$...$$` 与 13 处 callout 内的 `> $$...$$` 拆成「定界符各占一行」，callout 保持整块 `>` 引用。
- **MATH102（10）**：转义下划线移入文本类命令——程序标识符用 `\texttt{...}`（`q\_idx`、`local\_cp\_size`、`dt\_bias` 等），角色标签用 `\mathrm{...}`。
- **MATH103（148 → 0）**：先修规则再改内容。原规则把任何含 `-`/`/` 的下标都判为语义下标，导致 `h_{t-1}`、`m_{m'i:m'(i+1)-1}`、`x_{s+r-1}` 等**索引表达式**被误报（约 34 处）；按 skill 的本意改为只匹配「3 个以上连续字母」的词语标签，并补了对应单测。随后把 183 处真正的词语下标（`old`/`new`/`low`/`high`/`teacher`/`GRPO` 等 56 个）包进 `\mathrm{...}`。
- **MATH104（37）**：按语义区分竖线——条件概率用 `\mid`，绝对值/基数用 `\lvert...\rvert`，KL 用 `\,\|\,`，集合构造与大号定界符用 `\big\vert`。改写只在数学区域内进行（复用 checker 自己的 fence / 行内代码 / 货币判定），因此 Markdown 表格里的 `|` 与成本表 `| **$4,432** | - | ~$5,000 |` 均未被触碰。
- **MATH105（88 → 0）**：长公式改为显式 `\begin{aligned}` 结构，在**顶层**关系符（`=`/`\approx`/`\longrightarrow` 等，按花括号、`\left`/`\right`、`\begin`/`\end` 深度判定）处断行；多行时补 `\\` 行分隔符。三处需人工判断的单独处理：无关系符的目标函数按顶层空格软换行；跨行的 `\left(...\right)` 改成定尺寸 `\Big(...\Big)`（`\left`/`\right` 不能跨 aligned 行）；整体位于 `\boxed{}` 内的公式把 aligned 移到框内。
- 验证：`check_math --strict` 通过；`check_links` pages=409，broken/ambiguous/bare_index/orphans 均为 0；`pytest tools/` 92 passed；`npm docs:test` 61 passed。

---

## 2026-08-26：全库公式规范清零 + 修复 check_math 两处误报 + README 索引重写

**Type**: Corpus-wide Math Normalization + Checker Bug Fix + Entry-point Rewrite

- 全库 409 页 `check_math` 错误从 **910 → 0**（warning 仍有 464 条，本轮不在范围内）。
- **744 处真实违规**按 `.claude/skills/writing-obsidian-math/SKILL.md` 的唯一约定改写：646 处行内 `\(...\)` → `$...$`，98 处独占整行的 `\[` / `\]` → `$$`。转换只作用于 checker 实际检查的区域（跳过围栏代码块与行内代码），`ig(`、`\left(` 等命令不受影响。
- **17 处并非数学**：`List\[str\]`、`A\[t\]`、`\[B,H,N,N\]`、`a\[4\]\[4\]`、`W\[:, i·h\]` 以及 mindformers 的步骤标号 `\[B\]/\[C\]/\[D\]/\[E\]` 都是 Markdown 转义的**字面方括号**，改为行内代码而非公式（直接去转义会让 `[C][D]` 变成 Markdown 引用式链接）。
- **修复 `tools/check_math.py` 两处误报**（内容本身正确，是检查器判错）：① `LEGACY_DELIMITER_RE` 把 `aligned`/`cases` 里合法的行距 `\[2pt]` 当成 `\[` 定界符——改为跳过成对转义反斜杠的扫描器；② 货币启发式把 `$4.2 	imes 10^{-4}$` 当作金额，导致所有"以数字开头的行内公式"被报为 `$` 未配对——改为只在其后没有构成公式的闭合 `$`（或下一个 `$` 同样是金额）时才判为货币。两处均保持既有 11 条单测通过。
- **修复 `10_llm_initiliaze_analysis.md` 的历史性破坏**：该页 23 行显示公式此前被某次 Markdown 化处理吃掉了控制字符（`\(`→`$`、`\!`→`!`、`\,`→`,`、`\;`→`;`、`&`→`|`、`\\`→`\`、`_`→`*`），大部分 checker 看不出来但渲染是错的；已逐块还原，`\mathcal{N}(0,\sigma^2)` 这类真列表逗号保持不变。
- README 重写为可用入口：新增 **wiki 二级目录概览**（13 个二级目录 + 篇数 + 一句话职责 + 各域 index 链接）与 **核心文章索引**（按库内入链次数客观挑选 22 篇，GitHub 上可直接点进原始文档），并补上质量门禁命令清单。55 条链接全部校验可达。

---

## 2026-08-26：新增 vLLM 请求全链路导览页并归档其离线交互图

**Type**: Source Ingestion + Cross-domain Cross-reference

- 将 `vllm/deepseek_v3_inference_flow.md`（旁置 vLLM checkout 根目录的分析稿）纳入 [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]]，落为 `03_vllm_request_flow_walkthrough_analysis`（原 vLLM 请求全链路导览），占 2.1「入口与统一心智模型」段位。该页定位为**导览页**（"一条请求怎样穿过进程、队列与 GPU"），与本域其余 owner 页的「约束 → 状态所有权 → 设计选择」叙事互补。
- 按「合并优于并存」裁掉与既有 owner 页重叠的部分：原稿第 3.2–3.6 节、第 4 节（调度/执行/Executor 论证）压缩为一节交界事实并指向 [[02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis|Scheduler]]、[[02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis|KV Cache 管理]]、[[02_engineering/03_infer_frameworks/vllm/16_vllm_model_runner_v2_analysis|Model Runner V2]]；第 8.1/8.2/8.4 节压缩为条件路径摘要；第 7 节并行维度表保留但归口 [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|分布式推理]]。
- **保留的独有增量**（本域此前未覆盖）：服务启动进程树与三级就绪屏障（worker `Pipe` READY → EngineCore `HELLO/READY` → 数据面 ready）、空闲后端的逐层唤醒路径（ZMQ poll → `queue.Queue` → SHM ring + `SpinCondition`）、P0–P18 跨进程管道拓扑表、DeepSeek-V3 的 MLA/MoE 在通用调用链中的落点、按状态边界定位的排查表、源码阅读顺序与启动/请求主线函数索引。
- **基线例外**：该页显式声明源码基线 `vllm-project/vllm@26858770`（2026-08-24），高于本域统一基线 `d66300a1`（2026-08-20）；两提交之间该页引用的架构、引擎、调度、worker 与 DeepSeek 模型文件无源码差异，已在页头与域索引同时注明。
- 离线交互图 `deepseek_v3_inference_flow_interactive.html` 及其依赖 `.js` 归档至 `wiki/02_engineering/03_infer_frameworks/vllm/assets/`（HTML 通过 `./` 相对路径加载同目录 JS，两者需一起保留；未收原仓库的 `.test.js`）。
- 交叉链接：新页 Related Pages 7 条；[[02_engineering/03_infer_frameworks/vllm/17_vllm_serving_control_plane_analysis|Serving 控制面]] 补 Related Pages 回链（6→7）；[[02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis|引擎架构]] 因 Related Pages 已达 7 条上限，改在正文「进程生命周期与故障传播」一节补内联回链。
- 全部 8 个 mermaid 块按本库规范重写标签（管道标签去引号、`-. 文字 .->` 改 `-.->|文字|`、去 HTML 实体），链接检查 pages=409，broken/ambiguous/bare_index/orphans 均为 0。

---

## 2026-08-26：`raw/` 论文 PDF 全部替换为来源链接说明页

**Type**: Source-material Policy Change + Citation Correction

- 本库迁移至 GitHub 公开仓库 [`suhaibo666/llm-knowledge`](https://github.com/suhaibo666/llm-knowledge) 后，为避免公开转载第三方论文原文，删除 `raw/` 下全部 **102 份 PDF**（约 448 MB），每份替换为同名 `.md` 说明页，内容为 arXiv/官方链接、规范标题、提交与更新日期、主分类、作者与摘要；元数据统一取自 arXiv API 而非文件名推断。`.eddx`/`.html`/`.txt` 等自制或非论文源材料保持不变。
- 新增 [`raw/README.md`](https://github.com/suhaibo666/llm-knowledge/blob/main/raw/README.md) 作为策略说明与 102 条索引表。
- **更正 5 处错误引用**：逐条用 arXiv API 回查文件名内嵌 ID 的真实标题，发现 5 个 ID 指向完全无关的论文——`Scaling_Laws_for_Transfer` 的 `2002.05102`（实为复数反射群数学论文）应为 `2102.01293`；`DeepSeek_VL2` 的 `2412.10322`（实为格点 QCD）应为 `2412.10302`；`CodeGeeX` 的 `2306.03078`（实为 SpQR）应为 `2303.17568`；`CogVideo` 的 `2204.14230`（实为平坦丛上同调）应为 `2205.15868`；`GPT4_Vision_System_Card` 的 `2304.10592` 实为 MiniGPT-4，该系统卡并未在 arXiv 发布。
- 另有 4 份文件名不含 ID 者按 PDF 正文标题核定来源：`Engram_paper` = [arXiv:2601.07372](https://arxiv.org/abs/2601.07372)《Conditional Memory via Scalable Lookup》（DeepSeek-AI × 北大）、`DeepSeek_V4` = [arXiv:2606.19348](https://arxiv.org/abs/2606.19348)、`Claude_3_Model_Card`（Anthropic 官方）、`Kimi_K3_Technical_Report`（MoonshotAI/Kimi-K3 `0797decb`）、`GPT2_...`（OpenAI 官方 PDF）。
- 同步重写 39 个 wiki/docs 页中的 77 处 `raw/*.pdf` 引用，指向新的 `.md` 说明页并顺带修正其中的历史陈旧路径（`raw/01_architecture/`、`raw/05_model_families/` 等）。链接检查 pages=408，broken/ambiguous/bare_index/orphans 均为 0。

---

## 2026-08-20：按设计约束重构 vLLM 知识域并统一最新源码基线

**Type**: Knowledge-domain Architecture Redesign + Source-level Mechanism Audit

- 将 [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] 重构为 18 篇内容页加总索引的完整知识域，统一固定到 `vllm-project/vllm@d66300a1baa7779c68c7dfa4e51eee2502b48017`（`v0.27.2rc0-304-gd66300a1ba`）；新增系统设计原则、Model Runner V2、Serving 控制面、分离式 KV Serving、可观测性/可靠性和插件扩展体系，消除旧页面混合 commit 后无法拼成同一系统的问题。
- 系列叙事从“入口 → 调用 → 返回”的源码顺序改为“瓶颈/约束 → 状态所有权与不变量 → 设计选择 → 实现证据 → 替代方案 → 代价/失败边界”。总索引为每个设计问题指定唯一 owner 页；overview 和快速使用页只负责全局模型、跑通/测量/按症状调优，并链接到机制 owner，减少 EngineCore、continuous batching、Paged KV、compile/graph 等概念的跨页重复。
- 深化核心实现：Scheduler 作为 token/KV admission；KV block 的 ownership/refcount/free queue/hash/eviction；模型/权重/attention backend ABI；MRV2 stable row 与 async buffer；serving 进程拓扑、abort 和 bounded shutdown；投机、量化、分布式、compile/CUDA Graph、kernel 与 IR pass 的收益条件和 fallback；remote KV 的 producer/consumer/lease；metrics/trace/engine-death；plugin group 的进程覆盖与 endpoint 安全生命周期。
- 质量门禁：18/18 内容页包含统一 baseline、中心命题和 Related Pages；机械核验 361 个源码 `path:line` 定位符，缺失、歧义和越界均为 0；全库链接检查 pages=408 且 broken/ambiguous/bare_index/orphans 均为 0；21 个 changed Markdown 严格公式检查 0 error / 0 warning；完整测试 91 passed；`git diff --check` 无空白错误。

---

## 2026-08-20：补齐 slime 系列关键流程的可视化表达

**Type**: Series-wide Flow Visualization Audit

- 全局审视 [[02_engineering/04_posttrain_frameworks/slime/index|slime 源码分析系列]] 的 20 个内容页，逐节检查真实调用链、请求时序、数据切分、扩展接入和故障排查是否只有文字描述；保留已有且足以解释相邻章节的架构图、状态机和流程图，避免机械重复。
- 为 [[02_engineering/04_posttrain_frameworks/slime/10_slime_end_to_end_iteration_analysis|端到端迭代]]、[[02_engineering/04_posttrain_frameworks/slime/12_slime_sample_datasource_analysis|Sample 与 DataSource]]、[[02_engineering/04_posttrain_frameworks/slime/13_slime_sglang_rollout_engine_analysis|SGLang rollout]]、[[02_engineering/04_posttrain_frameworks/slime/19_slime_rollout_backend_extension_analysis|rollout 后端扩展]] 和 [[02_engineering/04_posttrain_frameworks/slime/25_vime_vllm_backend_support_analysis|vime/vLLM 支持度]] 补充源码对应的时序或流程图，显式展示并发层级、等待边界、DP 切分顺序和仍被复用的默认控制链。
- 为 [[02_engineering/04_posttrain_frameworks/slime/17_slime_train_inference_consistency_analysis|训推一致性]] 与 [[02_engineering/04_posttrain_frameworks/slime/31_slime_posttraining_stability_analysis|训练稳定性]] 增加分层决策图，把“先验证身份与版本，再下钻数值/kernel，最后调整执行器”的排查顺序从文字清单提升为可执行路径。

---

## 2026-08-19：补充 slime 在线 rollout 数据到 Megatron rank 的切分链路

**Type**: Source-faithful Data Scheduling Clarification

- 扩充 [[02_engineering/04_posttrain_frameworks/slime/12_slime_sample_datasource_analysis|Sample 与 DataSource 分析]]，明确 DataSource 只生产当前 rollout 的 prompt groups，RolloutManager 才按逻辑 `rollout_id` 切 optimizer steps、打包 micro-batches，并构造逐 DP-rank partitions；补充 `rollout_batch_size × n_samples_per_prompt`、`global_batch_size` 与每轮训练步数的关系及整除边界。
- 解释全局训练 rank 与数据 partition 并非一一对应：只有 DP ranks 获得不同 Sample 集合，同一 DP 副本内的 TP/PP/CP/EP ranks 共享样本身份，CP 再在训练侧按上下文维度切 token；增加 32 prompts × 8 responses、4 steps、DP=4、micro-batch=2 的完整算例。
- 同步修订 [[02_engineering/04_posttrain_frameworks/slime/14_slime_megatron_training_analysis|Megatron 训练后端分析]]，明确在线 rollout 不经过 Megatron 常规 Dataset/DataLoader，而由 slime `DataIterator` 消费预计算 schedule；Megatron 接管的是 pipeline forward/backward、模型并行、梯度同步和 optimizer 边界。

---

## 2026-08-19：补充 slime 中 Ray actor 与 SPMD rank 的并发关系

**Type**: Source-faithful Concurrency Clarification

- 扩充 [[02_engineering/04_posttrain_frameworks/slime/11_slime_ray_control_plane_analysis|slime Ray 控制面分析]]，明确 SPMD 的 single 指同一程序而非单进程，并解释一个逻辑训练角色如何由 `RayTrainGroup` 扇出为多个 trainer actors，再由各 actor 以独立 rank 加入同一个 `torch.distributed` world。
- 增加 Ray 并发管理分层：placement group 负责成组资源预留与放置，不同 actors 之间通过异步 RPC 并行，同一同步 actor 的方法默认串行，`ray.get` 形成控制面屏障，而 Megatron/PyTorch distributed 负责 TP/PP/DP/CP/EP 的集合通信与训练同步。
- 将含混的“每个子系统只做成一个 Ray actor”改写为准确反事实“把整个训练角色压进一个 Ray actor”，说明 slime 选择逐 rank actor 的资源、生命周期和故障隔离理由，同时解释 `RolloutServer`/`ServerGroup` 不需要成为 actor 的条件。

---

## 2026-08-19：统一 slime 系列中文技术用语与表格表达

**Type**: Series-wide Terminology and Readability Revision

- 审校 [[02_engineering/04_posttrain_frameworks/slime/index|slime 源码分析系列]] 的 20 篇内容页与总索引，在不改变源码事实、固定提交引用和公式的前提下，统一表头、章节标题和机制说明中的中文技术用语。
- 将会妨碍理解的直译改成常用工程表达：例如“制品路径”改为“模型与检查点目录”，“owner/ownership”按语境改为“责任主体/状态归属”，“admission”改为“准入控制”，“measure”改为“统计口径”，“productive throughput”改为“有效训练吞吐”，“fanout/flatten/sibling”分别写作“扇出/展平/同组片段”。
- 在索引中补充术语约定：类名、参数名、协议字段和行业通用缩写保留源码写法；普通说明文字优先使用能直接对应工程动作的中文，避免把 artifact、contract、identity、runtime、adapter 等英文概念逐字翻译成生硬表述。

---

## 2026-08-18：重构 slime 全目录源码分析——从功能罗列到设计因果与诊断

**Type**: Series-wide Source-faithful Rewrite + Design Rationale Audit

- 重构 [[02_engineering/04_posttrain_frameworks/slime/index|slime 源码分析系列]] 的 20 篇内容页与总索引：统一采用“问题背景 → 约束/不变量 → 设计选择 → 调用链 → 替代方案与代价 → 失败边界 → 证据”的因果主线；总索引新增“系统问题 → owner 页面 → 核心设计 → 主要边界”地图和按架构、数据/梯度、服务/提交、扩展、性能/故障五条阅读路线。
- 深化数据与控制面：解释 `Sample` 为什么是跨 DataSource、rollout、RM、trainer 的语义载体，明确 partial 默认会训练新旧 model token、mask 开关才只训练新 span，tool/environment token 的 logprob `0` 是不可训练占位；说明 nested compact fanout 如何用共享 `rollout_id` 在 flatten 后保留一次逻辑 execution 的统计身份，并厘清 placement group、actor group、RolloutManager、server、server group、engine 与 updater 的管理对象和参与时机。
- 深化训练正确性与状态边界：把 rollout-aware reducer、DP×CP whitening、训推 topology 转换、四种权重 transport 统一到 estimator measure 与版本化提交模型；严格区分 version、consistency、commit、recovery 和 replay，明确 engine 局部恢复不等于 DataSource/trainer/外部副作用的全局事务恢复。
- 深化能力扩展：将 external SGLang、custom generate、整轮 rollout、新 backend 分成不同扩展层；把 OPD、在线 MTP、低精度、新模型和 agent workflow 分别还原为 role/version、precision axes、双向语义映射与 execution→training projection 问题。vime 页使用独立 `vllm-project/vime@8144096e` 基线，明确它是 fork 级 vLLM 替换而非 upstream slime 内置 backend，并记录默认 top-p replay adapter 未闭合、external-engine 文档 `delta + NCCL` 与 runtime `delta + disk` 限制冲突等源码事实。
- 将 [[02_engineering/04_posttrain_frameworks/slime/30_slime_rollout_optimization_analysis|Rollout 优化]] 改写为 productive throughput 与闭环关键路径账本，覆盖 service/admission、filter/oversampling、tail/drain、trainer wait、data conversion、offload/publish 和 overlap 的适用条件及负收益反例；将 [[02_engineering/04_posttrain_frameworks/slime/31_slime_posttraining_stability_analysis|稳定性]] 改写为数据/奖励、策略版本、估计量/数值、基础设施四个控制环的判别式诊断，说明 clip、filter、restart 何时只是在压低症状。
- 证据与质量门禁：21 个 Markdown 页共核验 1,251 个 fixed-commit 定位符、195 个唯一源码文件，覆盖 `THUDM/slime@681b3adc`、SGLang `0b3bb0cb` / `d6ef6888` 与 `vllm-project/vime@8144096e`，路径和行号越界均为 0；全目录严格公式检查 0 error / 0 warning；全库链接检查 pages=402 且 broken/ambiguous/bare_index/orphans 均为 0；checker 测试 22 passed；所有改动通过 `git diff --check`，新增/改写 Mermaid 均按仓库规范人工复核。

---

## 2026-08-18：深化 slime 权重同步——共卡 CUDA IPC、拓扑转换与 MoE 定向路由

**Type**: Source-level Mechanism Clarification + Topology Constraint Audit

- 深化 [[16_slime_weight_sync_analysis]]：纠正“每个参数只在一个 train rank 聚合”和“每个 infer rank 最终常驻完整 HF 参数”两种过度简化，区分 Megatron collective 重组、NCCL transport source 与 SGLang TP-aware local shard loader，并列出 train/infer TP、PP、EP topology 转换的能力边界。
- 拆解 colocate 完整生命周期：说明 actor/rollout 是同一物理 GPU 上的不同进程与分时驻留，训练新值先落 pinned CPU snapshot、更新时按桶回到 train GPU；CUDA IPC 只传 GPU allocation handle 与 tensor metadata，SGLang 映射 source storage 后再 GPU→GPU copy 进自己的 parameter shard，Gloo/Ray 不搬运权重 payload。
- 补充 MoE 例外路径：在两侧 expert TP 为 1、infer PP=1、EP 静态且所有 engines 共卡等条件下，从 expert owner 通过 NCCL P2P 定向发送到目标 train rank，再经同卡 IPC 装入 SGLang EP worker；不满足条件自动回退通用完整 HF bucket。
- 更新 [[slime/index]] 的核验日期和权重同步入口摘要；源码证据固定到 slime `681b3adc` 与其 stable SGLang `0b3bb0cb` 基线。校验：目标页 53 个 fixed-commit 源码链接、23 个唯一文件均存在且行号无越界；全库链接检查 pages=402 且 broken/ambiguous/bare_index/orphans 均为 0；3 个 changed Markdown 严格公式检查为 0 error / 0 warning；公式与链接测试 19 passed；`git diff --check` 无空白错误。

---

## 2026-08-18：slime rollout 参数、GPU 复用与 PPO Actor/Critic 机制深化

**Type**: Source-level Mechanism Clarification + Beginner-oriented PPO Walkthrough

- 扩写 [[10_slime_end_to_end_iteration_analysis]]：把 post-training 外层事务边界与 optimizer step、dataset epoch、checkpoint cadence 分开，明确 `--start-rollout-id` 是 rollout/train/weight-commit cycle 的编号，自动恢复取已保存 id 的下一轮，`num_rollout` 是排他上界；checkpoint 并非每轮必存，但新权重每轮训练后都会提交给 rollout engine。
- 扩写 [[13_slime_sglang_rollout_engine_analysis]]：逐项解释 `--rollout-batch-size`、`--n-samples-per-prompt`、`--over-sampling-batch-size` 的 group/response/补采波次单位，以 32×8、oversample 64 的例子说明 attempted→accepted；明确 dynamic filter 在整组生成和 reward 之后运行，记录 zero-std filter 与 `keep_when_insufficient` fallback 的数据质量—延迟权衡。
- 扩写 [[11_slime_ray_control_plane_analysis]] 与 [[14_slime_megatron_training_analysis]]：说明 `_get_placement_group_layout` 同时定义 placement-group bundle 总量和 rollout slice offset、真正 rank/engine 绑定由下游完成；区分 Actor/Critic 共享训练卡与 train/rollout colocate 两条复用轴，并解释 PPO 强制 `offload_train` 是同卡显存驻留交接，不是角色控制流开关。
- 深化 [[15_slime_loss_parallelism_analysis]]：从 prefix state/response token 解释 GAE 中的 t 与 t+1，区分单步 TD residual 与完整 GAE，推导 return target 为什么是 advantage 加 frozen old value；补 Actor/Critic clipped loss 的业务依据、一条三 token 轨迹的逐项数值计算，以及 sample reward→token GAE/loss→logical-rollout reducer→Megatron scalar 的粒度链。
- 校验：5 篇目标 Markdown 严格公式检查为 0 error / 0 warning；5 页共 205 个固定 commit `path:line` 引用，文件与行号范围 issues=0；全库链接检查 pages=402，broken/ambiguous/bare_index/orphans 均为 0；公式/链接相关测试 22 passed；目标文件 `git diff --check` 无空白错误。

---

## 2026-08-18：vLLM 主分支架构重验与当前推理技术栈总览

**Type**: Source Revalidation + Architecture Deep Dive + Knowledge Map Refresh

- 固定并核验 `vllm-project/vllm@f4b161d7fca438bfe29509984759be1943a5aa88`（`v0.27.2rc0-189-gf4b161d7fc`），新增 [[02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis|大模型推理技术栈全景]]：按模型制品、协议输入、连续调度、分页 KV、执行/并行、图编译、kernel/通信、KV 数据平面和运维分层，并用官方资料定位 Transformers、vLLM、SGLang、TensorRT-LLM、llama.cpp 与已进入维护模式的 TGI。
- 重写 [[02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis|vLLM 引擎架构与请求生命周期]]：区分离线 `SyncMPClient` / 可选 `InprocClient` 与在线 `AsyncMPClient`，追踪 EngineCore 普通 step、batch queue、统一 token scheduler、KV block 所有权、Executor/Worker 与 Model Runner V1/V2 选择逻辑，修正“永远双进程”“离线必定同进程”和“V2 全量替换”等过度简化。
- 重写 [[02_engineering/03_infer_frameworks/vllm/01_vllm_feature_optimizations_guide|vLLM 快速使用与优化指南]]：补 Linux 与多硬件插件安装路径、offline/online 最小示例、chat template 与 generation config 陷阱、`-O0` 到 `-O3`、`balanced/interactivity/throughput`、prefix/chunked/async 默认行为，以及按 TTFT、TPOT、吞吐和显存分类的调优与 benchmark 流程。
- 更新 [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] 与 [[02_engineering/03_infer_frameworks/index|推理框架目录索引]]，显式标注当前入口页和 `485bbe1c6` 专题页的混合源码基线，避免把旧行号和默认值冒充当前主分支事实。
- 校验：4 篇入口/架构文档中的 77 个固定 `path:line` 定位均存在且未越界；全库链接检查 pages=402，broken/ambiguous/bare_index/orphans 均为 0；本次 6 个 Markdown 严格公式检查为 0 error / 0 warning；知识库测试 91 passed；本次文件范围 `git diff --check` 无空白错误。

---

## 2026-08-14：补齐 slime rollout batch、数据通道、logprob 与 delta 同步问答

**Type**: Source-level FAQ Enrichment + Mechanism Clarification

- 扩写 [[10_slime_end_to_end_iteration_analysis]]，统一解释 `rollout_batch_size`、训练 global batch、micro-batch/梯度累计、optimizer step、外层 rollout 与 epoch 的边界；明确 actor 每个 train step 更新，而 SGLang 权重在整轮 train 后一次提交。
- 补齐 GRPO 的 $R\times G$ 采样、group reward normalization 与 train-step split 关系，并解释变长轨迹如何经过 token-budget first-fit、DP/VPP 对齐和 THD packed sequence 进入 Megatron。
- 扩写 [[12_slime_sample_datasource_analysis]] 与 [[13_slime_sglang_rollout_engine_analysis]]：区分 SGLang `/generate` 返回与 Slime rollout 子系统最终 Sample，说明 reward 可来自 remote/custom/group/built-in verifier，复合格式/正确性/工具奖励需要显式组合；固定基线的数据路径是 CPU tensorize → Ray object store/NIXL → trainer CPU → CUDA，而非 NCCL HBM 直传。
- 纠正 logprob 量级误解：持久化 rollout logprob 是每个已选 response token 一个 float32，单条 shape 为 `[R_i]`，不是 `[B,S,V]`；[[14_slime_megatron_training_analysis]] 进一步说明训练侧 full-vocab logits 只在当前 micro-batch 内以 TP shard 短暂存在，并由 in-place vocab-parallel softmax、token chunk 与 dynamic token budget 控制峰值。
- 深化 [[16_slime_weight_sync_analysis]]：逐字节说明 delta baseline、XOR/overwrite、zstd、checksum、host-local mmap apply，区分 wire/storage bytes 的节省与全量 gather/CPU snapshot/最终 HBM reload 等未节省成本；补四条权重路径的关键路径、可重叠阶段、timer 口径和同步占比下界估算。
- 因 changelog 本次进入 changed-file 公式门禁，顺手机械修复该历史文件中既有的 4 个旧定界符错误和 5 个语义下标警告；只调整 MathJax 写法，不改变历史结论。
- 校验：6 个 changed Markdown 严格公式检查为 0 error / 0 warning；5 篇 Slime 页面共 176 个固定 commit `path:line` 引用，文件与行号范围 issues=0；全库链接检查 pages=401，broken/ambiguous/bare_index/orphans 均为 0；公式/链接相关测试 22 passed；`git diff --check` 无空白错误。

---

## 2026-08-14：slime/vime 公式整改与 Obsidian 数学质量门禁

**Type**: Documentation Remediation + Repository Skill + Automated Quality Gate

- 审计 [[slime/index]] 下 21 篇 slime/vime 页面，定位四类主要问题：混用 `\(...\)` / `\[...\]` 与 dollar 定界符、多字母语义下标被当作斜体变量或减法、API 标识符用裸 `\_` 塞入公式，以及条件竖线/多行推导未使用语义化 MathJax 写法。
- 整改实际命中问题的 10 篇 slime 页面：统一为 `$...$` 和起止独占行的 `$$`，用 `\mathrm` / `\text` / `\operatorname` 表达语义标签，以 `\mid`、`\lvert...\rvert` 和 `aligned` 修复条件概率、绝对值及多行公式；VIME 支持分析页经检查无需改写。
- 新增 `tools/check_math.py`：忽略代码围栏与行内代码，区分高置信度结构错误和启发式排版警告，支持显式文件/目录及 Git 变更文件检查；`--strict` 将警告一并纳入门禁，并对货币金额等常见非公式 `$` 写法做豁免。
- 新增 `.agents/skills/writing-obsidian-math/` 并同步 `.claude/skills/` 镜像，覆盖定界符、语义下标、API 名、表格、callout、多行对齐和强制自检流程；`CLAUDE.md` 与 `tools/README.md` 接入相同要求。
- 校验：公式检查器与 skill 集成测试共 12 passed；两份 skill 通过 `quick_validate.py` 且内容逐字节一致；21 篇 slime/vime 页面严格公式检查为 0 error / 0 warning。

---

## 2026-08-14：vime vLLM 衍生架构与支持度源码审计

**Type**: Derivative Architecture Analysis + Support Maturity Audit

- 固定 `vllm-project/vime@8144096e3f4fb0fb670c37b8f2d84015f7e92320` 与 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9` 两条基线，新增 [[25_vime_vllm_backend_support_analysis]]：将 vime 定位为保留 slime 训练、数据、算法与 Agent 上层、但深度改写 rollout 控制、vLLM server/router 请求路径和权重同步的衍生框架，而不是 slime 内可热切换的轻量 rollout 插件。
- 按 P1 接口、P2 功能、P3 正确性、P4 生产与性能四级口径审计 vLLM 参数面、内置/外部 engine、TP/PP/DP、PD/EPD、多模型、token/logprob、多模态、session affinity、top-p/MoE replay、custom generate/rollout、同步/一拍异步/fully async 与故障恢复。
- 追踪四条权重路径（NCCL、colocate IPC、全量磁盘、delta 磁盘）及 pause/flush/version/resume 事务；记录多模型只选择首个可更新 server、`check_weights()` 未实现和磁盘 CI 未做逐 tensor 等价验证等 P3/P4 缺口。
- 对照官方文档与源码标出三处边界：worker 类型文档滞后于 encoder-prefill 实现；`vllm-config` 示例混入 SGLang 风格键；“多模型可更新”不能据文档推导为所有 `update_weights: true` 模型都会同步。平台成熟度按 H100/H200、B 系列、A100/A800、AMD 与 Ascend 的不同证据强度分别表述。
- 把 vime 页面接入 [[19_slime_rollout_backend_extension_analysis]]、[[slime/index]]、[[02_engineering/04_posttrain_frameworks/index]]、[[courses/posttraining_frontier]] 与总索引；slime 子域现为 21 页，后训练框架域 44 页，全 wiki 399 页。
- 校验：21 篇 slime/vime 页面共 585 个固定 commit `path:line` 引用（slime 491、vime 94），仓库、commit、文件与行号范围 issues=0；严格链接检查 pages=399，broken/ambiguous/bare_index/orphans 均为 0；知识库测试 77 passed；`git diff --check` 无空白错误。vime 审计副本 remote 为官方仓库且工作树 clean，`HEAD=8144096e3f4fb0fb670c37b8f2d84015f7e92320`；GPU/vLLM/Megatron 能力以固定源码、文档和 CI 定义为证据，不冒充当前 Windows 主机实跑。

---

## 2026-08-14：slime 官方支持特性源码解读补齐

**Type**: Official Feature Coverage Audit + Source-level Deep Dive

- 以固定基线 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9` 对照官方 `docs/zh` 特性导航，补齐 6 篇实现分析：[[19_slime_rollout_backend_extension_analysis]]、[[20_slime_on_policy_distillation_analysis]]、[[21_slime_speculative_decoding_mtp_analysis]]、[[22_slime_low_precision_training_rollout_analysis]]、[[23_slime_model_architecture_extension_analysis]]、[[24_slime_agent_workflow_examples_analysis]]。
- 明确 rollout 扩展分为 custom generate、整轮 rollout function、external SGLang 与完整 backend replacement 四层；主仓库是 SGLang single-backend，external engine 仍是外部管理的 SGLang，vime 证明可派生替换而非运行时插件切换。
- 把 OPD 两类 teacher、在线 MTP 的训练—转换—同步—acceptance 闭环、BF16/FP8/INT4/KV 三条精度轴、ModuleSpec + HF wrapper 的无 module-TP 限制，以及 agent adapter/trajectory/harness/sandbox/fan-out 逐项追到源码。
- 增强 [[18_slime_fault_tolerance_observability_analysis]]：区分 W&B/TensorBoard step aggregate、sample trace 与 Prometheus time series，并记录固定提交下官方页面列出 `request/count`、源码默认路径却未实际发出的差异。
- [[slime/index]] 新增官方特性覆盖矩阵；slime 子域由 14 页增至 20 页，后训练框架域由 37 页增至 43 页。
- 校验：20 篇 slime 页面共 491 个固定 commit `path:line` 引用，文件与行号范围 issues=0；严格链接检查 pages=398，broken/ambiguous/bare_index/orphans 均为 0；知识库测试 77 passed；`git diff --check` 无空白错误。slime 源码树保持 clean，`HEAD` 与 `origin/HEAD` 均为 `681b3adc`；GPU/Megatron/SGLang 结论仍以源码与 CI 审阅为证据，不冒充当前 Windows 主机实跑。

---

## 2026-08-14：slime 建立独立软件架构与实现知识域

**Type**: Knowledge Domain Refactor + Source-level Implementation Analysis

- 将原有四篇 slime 专题迁入 [[slime/index]]，形成与 `verl/` 对等的独立子目录；保留固定源码基线 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`。
- 新增配置与 Quickstart、端到端迭代、Ray 控制面、`Sample`/`DataSource`、SGLang rollout engine、Megatron 训练、loss/并行、权重同步、容错/可观测性九篇实现分析；连同总览和三个横切专题，子域共 14 页。
- 软件架构分析明确拆出控制、数据、rollout、训练、权重、正确性与可观测性平面，并逐段追踪同步/一拍异步执行、逻辑 rollout reducer、四类权重 transport、engine recovery 和扩展 ABI。
- 更新后训练框架索引、D08 课程路线、全库快速导航、框架对照页以及 verl/AReaL 相邻导航；迁移后的旧链接全部改指新位置。
- 后训练框架域按当前文件树递归计为 37 页，其中 `verl/` 12 页、`slime/` 14 页。
- 校验：14 页共 381 个固定 commit `path:line` 引用，文件存在与行号范围 issues=0；严格链接检查 pages=392、broken/ambiguous/bare_index/orphans 均为 0；知识库测试 77 passed；`git diff --check` 无空白错误。slime 源码树保持 clean，仍固定在 `main@681b3adc`；GPU/Ray 集成边界沿用下条重验记录，不冒充本机实跑。

---

## 2026-08-14：slime 主分支重验 —— Rollout 优化、四层训推一致性与后训练稳定性

**Type**: Source Revalidation + Architecture Deep Dive + Correction

### 一、来源基线与旧结论纠偏

- 将 slime 源码基线从 `aaf5c209` 更新到本地干净主分支 `main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`（提交时间 2026-08-12T16:50:12+07:00，核验 2026-08-14）。
- 重写 [[01_slime_architecture_overview_analysis]]：以 control/data/weight 三平面贯通 `train.py`、`train_async.py`、RolloutManager、DataSource/Sample、Megatron actor 与四类 weight transport。
- 更正 fully-async 的描述：当前 physical output queue 为故意无界，backpressure 在 producer loop；每次只 drain 目标 group 并把 surplus 留给下一轮。外层 `train_async.py` 只是 generate(N+1)/train(N) 一拍 overlap，换权重前仍等待 generation，不能写成无边界 policy-lag trainer。

### 二、新增三篇专题

- **[[30_slime_rollout_optimization_analysis]]**：把 rollout 吞吐拆成 SGLang 请求并发、first-completed/oversampling、dynamic filter、partial+SSE streaming、fully-async warm queue、phase overlap、PD/speculative/FP8、DP packing/NIXL 与 weight-sync 九组机制，并给出 attempted→accepted 有效样本成本口径。
- **[[17_slime_train_inference_consistency_analysis]]**：建立 weight snapshot、token/mask contract、sampling distribution、kernel/MoE numerical path 四层一致性模型；覆盖 temperature/top-p exact replay、MoE top-k order/R3、GLM-5 layerwise exact-zero 与 e2e `<1e-6` gate，以及 TIS/ICEPOP/OPSM 的补偿边界。
- **[[31_slime_posttraining_stability_analysis]]**：以逻辑 rollout 而非物理 sample/mbs 为统计单位，解释 `rollout_mask_sums`、DP×CP advantage whitening、step-global reducer、PPO/GSPO/CISPO/KL/TIS/OPSM、空 token autograd liveness、FP8 zero-block guard、engine recovery 与 debug replay。

### 三、索引与对照页

- 更新后训练框架域 index、D08 课程路线与 wiki 快速导航；该域递归页面数按当前文件树更新为 27。
- 更新 [[30_rl_framework_comparison]] 的 slime commit、TIM 与 async 语义；AReaL/ROLL 仍保留 2026-07-27 固定快照，未借 slime 重验外推其他框架。

### 四、校验

- 四篇 slime 页面共 175 个固定 commit `path:line` 引用，逐一校验文件存在且行号未越界：issues=0。
- `python tools/check_links.py --strict`：pages=382，broken=0，ambiguous=0，bare_index=0，orphans=0。
- 知识库 `pytest -q`：77 passed；`git diff --check`：0 errors。
- slime 当前 Windows 环境可运行的契约测试：29 passed（DP schedule 9、top-p/CP response span 5、Sample contract 12、FP8 zero-block 3）。fully-async/process-rollout/routing-replay 三个 Ray 相关测试文件因本机未安装 `ray` 在 collection 阶段阻塞，未把它们记录为通过；GPU/GLM-5 门禁只核验固定源码与测试定义，未在本机执行。

---

## 2026-08-13：Qwen3.8-Max 发布材料摄入 —— 首个开放 Max 级权重的结构、真实工作 RL 与边界审计

**Type**: Source Ingestion + Model Analysis + Index Integration

### 一、来源基线

- 新增官方博客正文快照 `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Max_blog_2026-08-03.txt`（发布 2026-08-03，抓取 2026-08-13，584 行，含 4 行来源元数据）。
- 新增固定 Hugging Face revision `Qwen/Qwen3.8-2.4T-A95B@207bd685` 的模型卡与许可证快照。结构硬数值同时逐字段核对同 revision `config.json`。
- 官方模型卡没有链接独立 arXiv/PDF 技术报告，而是把发布博客作为 citation；本次因此明确按“产品/系统发布报告”摄入，不补造论文级训练细节。

### 二、新增权威页与家族入口

- **新增 [[10_qwen3_8_analysis]]**：以“Qwen3.5 架构放大 + 面向长程交付的真实工作 RL”为主线，覆盖 2.4T/95B、92 层 `23 × 3:1` Gated DeltaNet/Attention、512 选 10 + 1 shared MoE、原生 262K/可扩 1.01M、MTP，以及 Task / Workspace / Harness 组合环境、统一 reward、在线 batch 均衡三件套。
- **新增 [[01_theory/01_models/alibaba_qwen/index|Qwen 技术路线总览]]**，并接入模型总索引、wiki 总索引与快速导航；模型域递归页数 57→59，Qwen 子域 2 页。
- 在 [[24_agentic_rl_algorithm_analysis]] 增加 Qwen3.8 工业配方反链，使报告没有展开的 trajectory / credit / harness contract 能回到算法域阅读。

### 三、三条必须保留的审计边界

1. **Max endpoint ≠ 开放 checkpoint**：开放的 `Qwen3.8-2.4T-A95B` 是 text-only、强制 thinking、原生 262K；视觉输入、non-thinking、默认 1M 与官方内置工具属于托管 `Qwen3.8-Max`。官方 benchmark 表也报告 Max endpoint，没有另列开放 checkpoint。
2. **厂商长程案例 ≠ 基础模型隔离消融**：16 天自主开发、125 小时论文复现/改进、天池 45 次提交与 500 轮 RTL 优化都作为“模型 + harness + tools + verifier + budget”的系统证据记录，未写成单一权重的因果结论。
3. **开放权重 ≠ Apache 2.0**：模型卡元数据为 `license: other`；自定义许可证对超大商业产品的模型名展示，以及高收入 MaaS / AI Work Assistant 业务的另行许可设有条件。

### 四、仍未披露

预训练 token/数据、优化器、训练硬件与精度、3:1 注意力和 512/10 MoE 消融、RL 算法/超参、reward 权重、online balancer 实现与方差数字均未公开；正文以知识缺口表保留，不用推测填补。

### 五、校验

- `python tools/check_links.py --strict`：pages=381，broken=0，ambiguous=0，bare_index=0，orphans=0。
- `pytest -q`：77 passed。
- 新增 2 个 Mermaid 块按仓库定界符清单逐块静态复核，issues=0；`git diff --check`=0。

---

## 2026-08-11（二）：修复三条「会误导读者」的结构性问题，并更正上一轮审计自身的过度结论

**Type**: Correction（三条均出自 2026-08-10 的两域审计第一层。本次核对 verl 官方文档后，发现**审计自身的一条结论说过头了**，一并更正。）

### 一、verl v1/TransferQueue：补文档级覆盖，并收回"整簇失效"的过度判断

- **新增 [[16_verl_v1_transfer_queue_analysis]]**（文档级，**非源码级**，页头已显著标注）：厘清被混为一谈的三个概念、TransferQueue 的三层架构与四种存储后端（含昇腾原生的 Yuanrong 分层后端）、版本状态、以及**对本簇 9 篇影响范围的逐条界定**。
- **核对官方文档后的两处关键发现**：
  1. **`use_v1`、`TaskRunnerV1`、`trainer/ppo/v1/*` 在 verl 官方文档与 v0.7/v0.8 release notes 中均查无此名**——它们是源码树内部命名。本簇对它们的记述唯一依据是自身的源码观察（`ppo_trainer.yaml` 行号双向记录），不能引官方文档背书。官方侧可查证的对应事实是入口更名："`main_ppo.py` is deprecated with a warning in favor of `main_ppo_sync.py`"（v0.8.0 release notes）。
  2. **TransferQueue 至今仍不是默认传输方式**。v0.7 blog 称"计划在 v0.8 成为默认"，但 v0.8.0 实际只交付了 "New sync trainer with TransferQueue"，并注明 "TBD: Fully async trainer with TransferQueue will be in next release"。凡"verl 已默认走 TransferQueue"的说法无官方依据。
- **更正 2026-08-10 审计自身的结论**：该审计称「verl 簇 7 篇深潜写的是 legacy 路径，整簇有效性存疑」——**这个推论过度**。受影响的只是**编排层与数据搬运层**（`RayPPOTrainer.fit` 主链、以及"数据流经 driver"这一前提）；**计算面**（[[13_verl_workers_engine_analysis]]）、**权重面**（[[14_verl_rollout_runtime_analysis]]）、**算法面**（[[15_verl_rl_algorithms_analysis]]）、**数据契约**（[[12_verl_dataproto_analysis]]，官方称经 `RemoteBatch` 与 TransferQueue 兼容共存）均不受影响。逐条对照表见新页 §5。
- **更新 [[02_engineering/04_posttrain_frameworks/verl/index|verl 分析域]]**：重写架构演进提示，加 `[!warning]` 记录上述两处更正；把新页加入页面列表。**更新父索引** `04_posttrain_frameworks/index.md` 的 verl 篇数说明。

### 二、[[30_rl_framework_comparison]]：把"有效期临近"落到具体过期项

上一轮只加了到期提醒。本次从官方文档侧做了部分重验，**确证 verl 列至少两项已过期**（入口脚本更名、数据面新增 TransferQueue 通路），同时**确证一项"未过期"以免反向误判**（TransferQueue 未成为默认，故把 verl 数据面记为 `DataProto` 在默认路径上仍正确、只是不再完整）。并如实记录重验障碍：四框架 commit 比对需访问 GitHub，本次环境无法访问，**slime / AReaL / ROLL 三列本次未做任何重验**。

### 三、[[12_rl_infra_efficiency_analysis]]：重写长尾数学，并给全页数字定级

- **长尾治理的"12×"推导已重写**。原文用单条轨迹的 p99 代表"等 batch 全部完成"，但等全部完成是 $X_{(N)}=\max$，其典型分位为 $F^{-1}(1-\frac{1}{N+1})$——$N=128$ 时约 **p99.2 而非 p99**，重尾下差距更大，**误差方向是低估收益**。结论量级（≥12×）站得住，推导过程不成立，现按次序统计量重写。
- **区分了两种被混用的截尾口径**：「等前 90%」是 $X_{(116)}\approx$ p90；「发 $N{+}K$ 取前 $N$」（发 160 取 128）等的是 $F^{-1}(128/161)\approx$ **p80**，更快但多付 25% sandbox。原文 A、B 两小节混用了这两种。
- **撤回"几乎免费"的说法**。真正的代价不是浪费的 CPU，而是**选择偏差**——被丢弃的永远是最慢的那批，而 coding RL 里最慢的轨迹系统性地就是最难的任务；持续按完成时间截尾等于在训练分布里删除难题。[[01_posttraining_infra_mechanism_analysis]] §10 已就此警告过，本页此前未回应，现补记并给出监控建议。
- **新增全页数据口径说明**：把正文数字分三级——⚠️ 无出处经验区间（GPU 利用率四档、"30%+"、"20-30%"、环境池四档、三段占比假设）、⚠️ 舆情/转述级（带"据说"的表述）、✅ 有一手出处（AReaL `staleness_manager.py:80-112`、K3 §5.3.1 p.21）。并点明**本页是本目录被引最多、也最不可核验的一页**，这个组合本身就是风险。

### 四、校验

两目录 46 个文件 wikilink **未解析目标 0 种**；代码块与 LaTeX 配对正常；以 LF 行尾写回。

---

## 2026-08-11：在线策略蒸馏（OPD）专题摄入 —— 六页新簇 + 独立复核六处更正

**Type**: Source Ingestion + Deep Dive + Correction（源稿为用户自研的 OPD 综述调研稿，已整体落入 `raw/`；本次在其之上做了独立一手复核与数学补全，冲突处按 CLAUDE.md 标注。）

### 一、原始来源落库

- 新增 仓库外的 OPD 调研稿目录 `opd-survey/`（2026-08-10 基线，主稿 + 姊妹篇 + 八份底稿共 10 份；**按用户决定未纳入 `raw/`**，引用时请自行取用该目录）（364 KB，10 份）：主稿 `OPD-Survey-2026-08.md`（99 KB，606 行，87 个带版本号的 arXiv 引用、184 处 §级定位）、姊妹篇 `OPD-Infra-Survey-2026-08.md`（30 KB），及 `research-notes/` 下八份底稿（四路调研底稿 + 三份否定性结论复核附录 + Song & Zheng 89 页综述深读摘录）。
- 源稿方法论值得记录：显式区分「一手核实 / ⚠️ 二手 / 【推断】」三级可信度，对每条否定性结论附可复现检索式，并记录了复核中三项否定性结论被推翻的过程（Kimi 从"最大反例"转向 K3 的九专家 MOPD、字节 Seed 的"无公开资料"被修正、MiniLLM 改名版本被精确定位到 v6）。

### 二、新增六页

**算法侧 `01_theory/04_posttraining/`**

- **[[14_on_policy_distillation_analysis]]**（段 1 主线权威页）：定义与 2×2 坐标（判据只有"轨迹是否学生自采"一条）、与相邻概念的边界表、**暴露偏差 $O(\epsilon T^2)$ 的推导**、**OPD ≡ KL 约束 RL 的完整推导**（含反向 KL 必须用策略梯度的证明、MiniLLM 梯度中 $(R_t-1)$ 那个 $-1$ 的来源、KL 约束 RL 最优解 $\pi^*\propto\pi_{\mathrm{ref}}e^{r/\beta}$ 及由此得出的"不动点即教师"与 $\lambda$ 外推公式）、两条实现路线与三种 clip 的区分、独立核验记录。
- **[[15_opd_divergence_and_objective_evolution_analysis]]**（段 1）：散度演进六步的因果链，每步补推导——FKL 的 mass-covering 与 RKL 的 mode-seeking 各自由被求和项的极限行为导出、**skew KL 梯度有界性的完整证明**（界为 $(1-\alpha)/\alpha$，对照 FKL 的无界）、f-散度族"都是学生分布下的期望故 on-policy 无需 IS 修正"、**JSD(β)/JS/skew KL 同属 KL-to-mixture 家族的代数验证**（源稿标为【推断】，实为可验证事实）、AKL 对 mode-seeking 叙事的证伪及其适用尺度辨析。
- **[[32_opd_industrial_landscape_analysis]]**（段 3，644 行）：厂商谱系总表、四类工业用途（压缩/合并/防遗忘/跨模态）、逐厂商深读十节、两条实现路线的一手证据正面对立、教师五类来源与生产经济学、未采用与不披露阵营、11 条待核清单。
- **[[33_opd_effectiveness_and_failure_modes_analysis]]**（段 3，606 行）：三层有效性证据、可利用差距原理与四项训前诊断、**16 条失败模式按五族重组并逐条展开「症状→机理→触发条件→修复→修复代价」**（五族各附一句"被违反的前提假设"作为共同根因）、scaling 与 OPD 独有的 rollout 预算轴、决策框架、subliminal learning 与蒸馏攻击。

**工程侧 `02_engineering/04_posttrain_frameworks/`**

- **[[13_opd_infra_mechanism_analysis]]**（段 1）：OPD = "RL 的回路加一个新角色"、与 SFT/RL 的系统需求对照（critic 消失是省出的预算、GLM-5 把 group size 降到 1 直接除掉 rollout 采样量）、**信号格式四档带宽账**（全词表约 4.2 GB/轨迹 vs 采样 token 约 64 KB，相差约 5 个数量级）、成本模型与教师刷新率 $\rho$ 旋钮、**八项工作清单 W1–W8**、OPD 的 staleness 容忍窗口为何比 RLHF 更窄的机制论证。
- **[[32_opd_framework_support_comparison]]**（段 3）：veRL/slime/TRL/NeMo-RL/Tinker/KDFlow 六框架逐项矩阵与选型、OpenRLHF「可用而非原生支持」辨析、生产系统自研层、六条生态 Gap、预算分配三段模式。

### 三、独立复核：六处事实更正 + 一处降级

复核方式：`raw/` 中已有 PDF 直接逐字核对（DeepSeek-V4、Kimi K3、GLM-5），其余回 arXiv/官方页面核实。

| # | 源稿表述 | 复核结果 |
|---|---|---|
| 1 | TML 博客「以梯度步计 50–100×」 | **口径错位**：原文梯度步为 **7–10×**（<10 步 vs 70 步），**50–100× 是累计计算量**；且出自 Discussion 节的 LoRA rank-128 实验而非主实验 |
| 2 | TML「同设置 RL 达 68%」 | **非博客自跑**：博客是引用 Qwen3 报告的 **67.6%**（17,920 GPU 时，限定语 "a similar SFT initialization"） |
| 3 | MiMo BrowseComp「−6.3 失分域」 | **语义误读**：学生实际 42.5→45.4（**上升 +2.9**），−6.3 是相对 SFT 教师 51.7 的**差距**（Table 7 的 Δ 列为"学生−最强教师"）。真回退是创意写作 90.1→86.2 |
| 4 | Nemotron-Cascade 2 的 MOPD | §4.4 标题为 **Multi-domain** On-Policy Distillation，非 multi-teacher；71.5→85.5 是 ArenaHard V2.0 的 **Hard Prompt 子项**而非总分；Table 3 显示 RLHF 在 100 步已达 81.7 |
| 5 | MiMo「IcePop 式截断」 | 正文未出现 "IcePop"（仅见参考文献标题），原文为 "Following Zhao et al. (2025)"；且截断作用于**训推比 $\pi_\theta/\mu_\theta$**（属 TIM 修正，见 [[26_tim_causal_chain_analysis]]），**不是师生比值** |
| 6 | Nemotron 3 Ultra「两轮师生共进化」 | Figure 10 图注实为：第二轮教师由 **Ultra MOPD1（第一轮的学生）** 初始化并复用第一轮教师；**RLVR Student 是"自教师"**，补专用教师未覆盖的领域 |

**降级一处**：Nemotron 3 Ultra §3.3.5 的"教师天花板"逐字引语（"a limitation of the on-policy distillation setting"）——arXiv HTML 与 ar5iv 两条路径均在 §3.3.1 后截断，本次无一手页面支撑，已在两页降级为 ⚠️ 待核，源稿 P6 的支撑改挂到不依赖此引语的其它证据。

**核实为真、可作一手引用的**：DeepSeek-V4 §5.1.2 全套（"entirely replaced by On-Policy Distillation"、Eq. 29 多教师反向 KL、"more than ten teacher models"、对路线 B 的批评原文）；Kimi K3 §4.1.3（Eq. 15 clip 形式、"no clear advantage" 原句、九专家=三域×三档 effort）；Qwen3 §4.5 逐字与 §4.7 Table 21 十二格数字全中；GKD/MiniLLM/f-DISTILL/AKL 的核心主张；**MiniLLM 改名的版本史**（v5 2025-11-21 旧题、v6 2026-01-31 改为含 "On-Policy Distillation"，ICLR 论文集版保留旧题）。

**补入的限定条件**：Qwen3 的 1,800 vs 17,920 只是两个增量阶段之比（off-policy 起点行 GPU 时为 "–"），且报告未说明是否含教师推理成本；GKD 的 2.1×/1.7×/1.9× 是"相对初始学生的提升量之比"且跨模型规模平均；f-DISTILL 的对称散度优势有 WMT16 EN-RO 的 TER 例外；AKL 的"FKL 与 RKL 共享同一目标"须带"训练到收敛（约 300 epoch）"前提。

### 四、索引与校验

- 更新 `01_theory/04_posttraining/index.md`（段位表补 14/15/32/33、新增「在线策略蒸馏（OPD）」小节）与 `02_engineering/04_posttrain_frameworks/index.md`（补 13/32、新增 OPD 系统侧小节）。
- 校验：两目录 45 个文件的 wikilink **未解析目标 0 种**（修正了两处指向重排前旧页名的链接：`deepseek_v4_analysis`→`13_deepseek_v4_analysis`、`glm_5_analysis`→`01_glm_5_analysis`）；全部 `$$` 配对、代码块闭合；唯一一处 mermaid 经 mermaid-cli 实渲通过。以 LF 行尾写回。

---

## 2026-08-10：后训练两域审计与事实性修复（回一手 PDF 裁决）

**Type**: Audit + Correction（对 `01_theory/04_posttraining` 与 `02_engineering/04_posttrain_frameworks` 共 39 页做全量精读审计；所有更正均回 `raw/` 中的原始 PDF 逐条裁决，冲突处按 CLAUDE.md 加 `[!deprecated]` / `[!warning]` 标注并保留原说法，不静默改写。）

### 一、六处事实性矛盾的裁决结果

- **[[20_grpo_analysis]] —— GRPO 比值粒度（原表述错误，已更正）**：此前称比值是「whole-output ratio with no token subscript」。据 DeepSeekMath **arXiv:2402.03300v3 §4.1.1 式 (3)**，分子分母均为 `pi(o_{i,t} | q, o_{i,<t})`，外层为 `1/G Σ_i 1/|o_i| Σ_t`，clip 为**对称** `[1-eps, 1+eps]`；Algorithm 1 第 9 行亦写明 "Compute Â_{i,t} for the **t-th token** of o_i"。该错误此前与 [[22_gspo_analysis]] 的全部论证前提及 [[13_reasoning_rl_algorithm_evolution_analysis]] 的记号直接冲突，现统一为 token 级。
- **[[22_gspo_analysis]] —— clip 值 0.2/0.27（原数字正确，标签误导，已加注）**：核对 GSPO 论文 §5.1，0.2/0.27 是 GSPO 作者为公平对比而「carefully tuned」的 **GRPO 实验基线设定**，非 GRPO 固有配置。新增注记区分三者：GRPO 本身对称 `eps`；本表 0.2/0.27 属 GSPO 实验设定；DAPO 的 Clip-Higher 是 0.2/0.28（另一篇论文的另一个数字，非笔误）。
- **[[23_rloo_analysis]] —— RLOO 与 GRPO 的时序（原表述错误，已更正）**：此前称「RLOO is the theoretical foundation for GRPO」。arXiv 编号按投稿顺序递增，GRPO 出自 **2402.03300**、RLOO 为 **2402.14740**，前者早于后者公开，不构成继承关系。改为「同期独立工作，共享 leave-one-out baseline 思路」，Impact 节的「paving the way for GRPO」同步更正为并列表述。
- **[[21_dapo_analysis]] —— batch 配置（审计误判，原页无误，已加口径说明）**：`Prompt batch size 512` 与 `Mini-batch size 512 (16 updates)` 并不矛盾——前者数 prompt（512 × 每 prompt 16 条 = 8192 条序列），后者数序列，8192/512 = 16 次更新。按 DAPO §5 原文补口径说明。
- **[[27_vapo_analysis]] —— 步数与崩溃断言（三处更正 + 空壳补齐）**：① 「50% fewer training steps than DAPO」错误，论文 §5 实为「using only **60% of DAPO's steps**」且指的是**追平 DAPO 的 50 分**、非达到 60.4；② 表中 DAPO `~8,000` / R1-Zero `~10,400` 是页面自行换算，论文只给相对比例，且 `~10,400` 属 **DeepSeek-R1-Zero（V3-base）** 与此处的 **R1-Zero-Qwen-32B** 不是同一次运行；③ `Crashes: Yes` 是对他方系统的无源断言，论文只声明 VAPO 自身零崩溃。同时把三个 Challenge 的空句 "Solution" 与四条名词式 "VAPO Framework" 替换为 §4.1–4.3 的具体机制与 **§5.1 的七项修改清单**（含 Value-Pretraining 50 步预热、Decoupled-GAE、Length-Adaptive GAE `λ = 1 − 1/(αl)` 与 `α = 0.05`、Clip-Higher 0.2/0.28、token-level loss、Positive-example LM loss 权重 0.1、Group-Sampling 512×16）。
- **[[30_preference_optimization_analysis]] —— KTO loss 形式（原式错误，已更正）**：此前写作 `L_KTO = -E[z(x,y) * lambda(x) * sigma(beta*(r - r_ref))]`，用 `z = ±1` 在 σ **之外**翻转符号。据 KTO **arXiv:2402.01306 §4 Eq. 8**，符号翻转发生在 σ **参数内部**（`r_θ − z_0` 与 `z_0 − r_θ` 互换），损失形如 `E[λ_y − v(x,y)]`。σ 单调递增且值域 (0,1)，整体乘 −1 与参数取反**不等价**——按旧式实现，undesirable 样本会得到方向相反的优化信号。已改为原文分段形式并补 LaTeX。

### 二、内容缺口的就地补齐

- **[[22_gspo_analysis]]**：把「长度归一化为什么关键」的论证从 D02 **收回本地**——核对后 D02 §4 的跨算法表只有 6 列，**没有 gradient weighting 与 stability 列**，D02 §3.4 也不含该推导，原转指指向不存在的内容。本地补入 GSPO §4.1 原文引证。Empirical Results 节此前零数字，现补 §5.1 实验设定与 §5.2 唯一的硬量化结论（GSPO 与 GRPO 的 clipping fraction 相差**两个数量级**，而 GSPO 训练效率反而更高），并说明该论文本身不提供基准分数表、结果仅以曲线呈现。

### 三、卫生与失效指针

- **[[15_verl_rl_algorithms_analysis]]**：「详细 IS/RS 预设见 14」为悬空指针——14 号页对 `importance`/`rollout_is`/`rollout_rs`/`rejection`/`correction` 的命中数**均为 0**。已改为显式标注 **rollout correction 的实现走查在 verl 簇内无人承担**，并把算法侧谱系指向 [[26_tim_causal_chain_analysis]] §6.3。
- **[[01_verl_architecture_overview_analysis]]**：延伸阅读表把 profiler 指向 30 号页，但该页 `profil*` 词频为 0。已拆行并标注为未覆盖。
- **[[13_verl_workers_engine_analysis]]**：本页 `engine_workers.py` 行号与簇内其余四页存在系统性 **+3**（`update_weights` 为 +1）偏移，此前无任何约定说明。已加警示并给出推测（本页记 `def` 行、其余记 `@register` 装饰器行，待在固定 checkout 上核实），要求跨页跳转按 ±3 行窗口查找。
- **[[30_verl_optimization_analysis]]**：删除页头泄漏的作者本机路径（`E:\...\verl\verl\` → `repo 根下的 verl/verl/`）。
- **[[30_rl_framework_comparison]]**：自设复核期 2026-08-26 仅剩约两周，已加有效期警示；同时标注两项已知边界——本页 verl 列锁 `983cb0f` 而 verl 簇 10 篇中 7 篇深潜页基线为 `8a694930`；对比集为四框架封闭集，Miles/SkyRL/NeMo-RL 全库零命中、PRIME-RL 未进入本页。

### 四、审计中被证伪的「问题」（记录以免重复排查）

- **wikilink 零断裂**：全库复扫两目录 39 个文件，未解析目标 **0 种**。此前疑似断链（K3 案例页、courses、batch_invariance、megatron 两页、source-ledger、verl dataproto/quickstart）经核实**全部存在**，只是改用了带编号的新文件名，或是扫描脚本未处理表格中转义的 `\|`。
- **`areal-project/AReaL` 是正确的组织名**，非笔误。

### 五、附带效果

本次以 **LF** 行尾写回，使 `verl/01`、`verl/13`、`verl/15`、`verl/30` 四页脱离仓库既有的 CRLF 幻影 diff（各自从约 500 行整文件改动降为 2–6 行真实改动）。仓库仍有约 120 个文件处于 worktree CRLF 与 git HEAD LF 不一致的状态，建议择期 `git add --renormalize .` 或补 `.gitattributes` 统一处理。

---

## 2026-08-04：知识库结构整改 P7 完成——整改全程收官（P0-P7）

**Type**: Structure Reorg 终章（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`）

- **全库分段编号完成**：P7 推广 13 目录 ~130 页（加上 P4/P5 共约 28 个目录、280+ 页），全库内容页按「0 导览/1 主线/2 深潜/3 方法论」编号，文件名即学习路径；snake_case/后缀/点号残余全部清零。
- **spec §3.4 补执行**：Megatron PP 三页（1740 行）合并为单权威页（审查回补五处后完整）。
- **链接治理终态**：裸 `[[index]]` 70 处全部路径限定+语义显示名；58 处句主位裸编号链接补显示名；检查器四项全零（broken/ambiguous/bare_index/orphans = 0）。
- **治理文档定稿**：CLAUDE.md 全面修订（Courses 规则/页面类型与编号约定/链接新规/溯源政策/索引维护）；changelog 按季度归档（Q2 及更早 2112 行入 `changelog/`）；README 与总索引重算定稿。
- **整改总账（2026-07-29 至 2026-08-04）**：wiki 402→372 页；13 组高重叠全部清零；坏链 160→0 并保持；仓库瘦身 ~55MB;图表可再生；`01_ai_frameworks` 18 平铺目录→5 架构层;19 号课程目录与 03_posttraining 纵向域解散为 2 个 courses 导读页;全程每次合并经对抗审查，累计回补 40+ 组险丢失的独有事实。

## 2026-08-04：spec §3.4 PP 三页合并 —— 逐行审查修正（吸收不完整的五处回补）

**Type**: Correction（对 2026-08-01 条目的订正；协调者逐行核对 `git show 68415c2:` 两源页原文后判定该次合并 ❌ 五处吸收不完整）

**披露 —— 上条目的一处失实断言**：2026-08-01 条目判定"细粒度激活换出" `20_` §9/`26_` §2 已被 `22_megatron_memory_optimization_analysis.md` §2.3 完全覆盖、故不重复搬入 `15_`。经协调者反向 grep 核实：该判断**只在参数/配置面成立**（`min_offloaded_tensor_size`/`activation_offload_fraction`/`offload_margin`/`max_inflight_offloads` 22 页确实都有），但**机制/代码面 22 页当时完全没有** —— `saved_tensors_hooks` autograd 挂钩、`on_save_for_backward`/`tensor_push`/`tensor_pop`/`bulk_offload_group`/`bulk_reload_group` 的具体实现、`OffloadTensorGroup` 双 CUDA event 同步设计、`post_warmup_callback` 自适应调优逻辑，这些在 `20_`/`26_` 原文中存在但 22 页没有。"22 已覆盖"的判断因此是**不完整的**，构成一次遗漏。本条目订正。

**五处回补**（原文取自 `git show 68415c2:` 两源页,逐字/保留行号核对无误后落地）：

1. **VPP 层分布示例表**（原 `20_` §4.1）：`pp=4, vp=2, 16` 层时 GPU0-3 各持哪两个不连续层段（GPU0: layers1-2/9-10 …）→ 补回 `15_megatron_pp_schedulers_analysis.md` §③.2，表后加一句一般化规则（起始层 `= c·pp·(L/pp/vp) + r·(L/pp/vp)`，代入验证与原表吻合）。
2. **§8.4 配置速查表**补 `pipeline_dtype` 行（"与模型一致 | `pp>1` 时必填"）。
3. **offload 机制小节回补到 `22_megatron_memory_optimization_analysis.md` §2.3**（而非 15 页 —— 22 页是 offload 权威页，机制该在它那）：新增"Autograd 挂钩与执行流程"（`PipelineOffloadManager` 类初始化、`saved_tensors_hooks` 拦截、`tensor_push`/`tensor_pop`/`bulk_offload_group`/`bulk_reload_group` 逐字代码）、`OffloadTensorGroup` 双 event 同步说明、`post_warmup_callback` 自适应调优代码，并挂钩既有"关键设计"第 2 条 `offload_margin` 参数的来源。`15_` §0.3 的指针措辞同步改为"参数与机制见 `22_` §2.3"。
4. **CP 负载不均衡四轴对比表**（原 `26_` §3.3：PP/CP/CP变长/EP 的不均衡来源→均衡手段）→ 补回 `15_` §6.1，自引用文案按落点调整（"本文调度器③"/"本节"）。
5. **气泡率数值例**（`pp=4,m=8→37.5%`、`pp=4,m=32→9.4%`、`pp=4,vp=2,m=8→18.75%`）补进 `15_` §②.4/§③.4 公式旁；**UCC 环境变量全称** `UCC_CL_BASIC_TLS=^sharp,nccl` 补回 `15_` §0.4 的 UCC 后端行（该事实的落点是 §0.4 的进程组/后端表，非 §0.3；`0.4` 沿用了原 `20_` §1.1 段落的内容边界）。

**校验**：`python tools/check_links.py`：pages=371,broken=0,orphans=0（ambiguous=70/bare_index=70 仍为既有基线,未变化）；`pytest -q`：77 passed。

---

## 2026-08-01：知识库结构整改 —— spec §3.4 Megatron PP 三页合并（补执行，覆盖遗漏修复）

**Type**: Structure Reorg / Dedup（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.4；分支 `reorg/p7`）

**背景 —— 覆盖遗漏的发现与归因**：spec §3.4 明文裁定"以 `megatron-lm/15_megatron_pp_schedulers_analysis.md`（771 行，基线 `ee3f1ff`）为权威，吸收 `02_train_frameworks/20_megatron_pp_parallelism_analysis.md`（740）与 `megatron-lm/26_megatron_pp_supplements_analysis.md`（229）的增量后删除两者"，划归 P6 Task 3（§3.4+§3.5+§3.6 横向页收缩批）。但 `docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 3 的实际执行范围只覆盖了 §3.5 的两项（`comm_compute_overlap_analysis` 矩阵化 + megatron 优化器三页合并，见 commit `2dd9772`/`50afc2b`），§3.4 的 PP 三页合并**从未被排进任何 P6/P7 任务清单、也从未执行**。P7 Task 7（全库编号推广,commit `68415c2` 前置链）在给 `20_megatron_pp_parallelism_analysis.md` 判段编号时命中此空档，在 changelog 该条目"判段说明"里显式记录"该合并未落地……不属本任务范围"（见上一条目 2026-08-01 P7 Task 7 记录），并原地保留 `20_` 编号（未移入 `megatron-lm/` 子目录、未删除）—— 即本次任务标题所称的"发现者"。本条目是对这一遗漏的**补执行**。

### 三页逐节台账

以 `15_` 为骨架逐节核对 `20_`（740 行）与 `26_`（229 行）：

- **1F1B / Interleaved 1F1B / Combined 1F1B / VPP 层分布 / bubble 公式 / P2P 通信 / activation offload** 七大主题在三页间**逐项对应**（盘点当时判定属实）：`20_` §二/三/四/六/七/八 与 `15_` 调度器①-⑤ + §0.3 同一套源码（`schedules.py`/`combined_1f1b.py`/`p2p_communication.py`）的复述,判重删除；`26_` §1 P2P 内部与 `20_` §五同题,二者合并去重。
- **`20_` 独有增量**（判定后原样/改写吸收）：§一 PP 进程组与拓扑（`parallel_state.py` 建组、Embedding/Position Embedding 组、UCC 后端）→ 新增 `15_` §0.4；§3.3 Grad Sync 与 Bubble 重叠（非首 stage 梯度 AllReduce 借 cooldown bubble 异步执行）→ 并入 `15_` §②.2；§6.4 Combined-1F1B 的 FP8 支持（`use_outer_fp8_context`）→ 并入 `15_` §⑤.2.1 不变量callout；§7.3 PP P2P 与 EP A2A 的带宽竞争 → 新增 `15_` §⑤.6；§8.1 Partial Activation Checkpointing（`max_outstanding_backprops` 自适应窗口）→ 并入 `15_` §②.2（`18_megatron_recompute_analysis.md` 此前已预置指向此处的引用,验证判位准确）；§8.3 Defer Embedding WGrad → 并入 `15_` §0.4；§10.3 配置速查表 → 改写为 `15_` §8.4。
- **`26_` 独有增量**：§1 P2P 内部（`_p2p_ops`/`_batched_p2p_ops`、even/odd 死锁规避、WORLD group 技巧、变长序列形状交换）与 `20_` §五合并后扩写进 `15_` §0.3；§3 混合 CP 动态调度（`hybrid_cp_schedule.py` `BalancedCPScheduler`）与 §4 多模块/多模态流水线（`BridgeCommunicator`/`MultiModulePipelineCommunicator`,含 2026-06-16 `bridge_pg` 更新）→ 新增 `15_` §6（目录周边设施），与 `29_megatron_packed_dataset_dynamic_cp_analysis.md` 互链澄清"类形态兄弟 vs 真正集成入口"关系（不推翻 29 号页已有勘误）。
- **判定跳过、不吸收**：`20_` §9 细粒度激活换出（`PipelineOffloadManager`）与 `26_` §2 同题——核实 `22_megatron_memory_optimization_analysis.md` §2.3 已是**更新、更全**的现行权威页（含 `min_offloaded_tensor_size`/`offload_margin`/`ChunkOffloadHandler`/2026-06-16 `max_inflight_offloads` 节流,`20_`/`26_` 均无这些细节），故不重复搬入 `15_`，改为一句话+链接指向 `22_` §2.3（`15_` §0.3 尾段），避免制造新重复。

### 跨框架段处置

按任务书要求排查 `20_`（740 行）是否含跨框架视角段（如 torchtitan 对照）：全文 grep `torchtitan|TorchTitan|跨框架|vLLM|DeepSpeed` **零命中**——`20_` 通篇是 Megatron-LM 单框架源码分析,无需迁出至横向对比页,不适用"迁入 `30_comm_compute_overlap_analysis` 同款对比页"或"根目录留瘦对比页"的处置分支。

### 图资产处置

`20_` 引用的 4 张 PNG（`assets/megatron_pp_parallelism_analysis_fig1-4.png`）经逐张查看：内容均为 `15_` 已有 ASCII 时序图/决策树的低精度复述（`15_` 的空间-时间图精确到逐 op 槽位、决策树含 MoE/EP 分支,严格更完整），且仅被 `20_` 自身引用，判定为无独有信息、随页删除。

### 合并结果

- `15_megatron_pp_schedulers_analysis.md`：771 → 896 行，新增 §0.4（PP 进程组与拓扑）、§6（目录周边设施：混合 CP 动态调度 + 多模块流水线）、§⑤.6（PP P2P/EP 资源竞争）、§8.4（配置速查表）,§0.3/§②.2/§⑤.2.1 就地扩写；旧 §6/§7 顺延为 §7/§8（`见第 6 节` 内部引用同步改 `见第 7 节`）。
- 删除 `02_train_frameworks/20_megatron_pp_parallelism_analysis.md`(740)、`megatron-lm/26_megatron_pp_supplements_analysis.md`(229) 及其 4 张图资产；`02_train_frameworks/index.md`、`megatron-lm/index.md` 段位表同步（`20`/`26` 两个编号空出、不重新分配，表内注明）。
- **入链改写 9 处文件**（曾用名 `20_megatron_pp_parallelism_analysis`/`26_megatron_pp_supplements_analysis` 及其 §编号指针全部改写为 `15_megatron_pp_schedulers_analysis` 对应新 §）：`megatron-lm/11_megatron_dataset_analysis.md`、`megatron-lm/27_megatron_tp_fsdp_resharding_supplements_analysis.md`、`megatron-lm/18_megatron_recompute_analysis.md`（激活换出指针改指 `22_` §2.3,非 `15_`）、`megatron-lm/29_megatron_packed_dataset_dynamic_cp_analysis.md`、`megatron-lm/17_megatron_parallelism_orchestration_analysis.md`、`02_train_frameworks/index.md`、`megatron-lm/index.md`、`torchtitan/14_torchtitan_pp_analysis.md`（原双链 15_/20_ 去重为单链）、`15_megatron_pp_schedulers_analysis.md` 自身 Related Pages（移除自引用，新增 `29_`/`22_` 互链）。

**校验**：`python tools/check_links.py`：pages=371（373−2）,broken=0,orphans=0（ambiguous=70/bare_index=70 为既有基线,未变化,P7 Task 8 处理范围）；`pytest -q`：77 passed。

---

## 2026-08-01：知识库结构整改 P7 Task 7 —— 全库编号/命名推广

**Type**: Naming Convention（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §5；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 7；P4 Task 9.5 / P5 Task 8 同款规程）

对 P4/P5 尚未覆盖的 13 个目录逐目录施行两位数字段位前缀命名：段 0（01-09）导览/概览、段 1（10-19）
核心机制主线、段 2（20-29）深潜/专题、段 3（30-39）方法论/对照/工程实践；`index.md` 不编号，仅在
各目录 index 新增"段位与阅读顺序"小节或"段位速查"表。纯改名，不改内容；跨 9 个 commit 分批完成
（含 1 个 batch0 隔离修复 + 7 个目录批 + 1 个后清收批），每批 checker broken=0 + pytest 77 后提交：

| 目录 | 篇数 | 段位分布 | 编号区间 |
|---|---|---|---|
| `01_theory/01_models/deepseek` | 20 | 1段9/2段10/3段1 | 10-18,20-29,30 |
| `01_theory/01_models/moonshot_kimi` | 13 | 1段5/2段8 | 10-14,20-27 |
| `01_theory/01_models/zhipu_glm` | 9 | 0段1/1段1/2段7 | 01,10,20-26 |
| `01_theory/02_pretraining` | 6 | 1段5/2段1 | 10-14,20 |
| `01_theory/06_distributed_parallelism` | 8 | 1段6/2段2 | 10-15,20-21 |
| `02_engineering/02_train_frameworks`（根） | 7 | 2段3/3段4 | 20-22,30-33 |
| `02_engineering/02_train_frameworks/megatron-lm` | 27 | 0段1/1段10/2段10/3段6 | 01,10-19,20-29,30-35 |
| `02_engineering/02_train_frameworks/torchtitan` | 12 | 1段6/2段6 | 10-15,20-25 |
| `02_engineering/02_train_frameworks/mindspeed` | 5 | 1段4/2段1 | 10-13,20 |
| `02_engineering/03_infer_frameworks/vllm` | 12 | 0段1/1段5/2段6 | 01,10-14,20-25 |
| `02_engineering/05_gpu_kernel`（根） | 7 | 0段1/1段2/2段4 | 01,10-11,20-23 |
| `02_engineering/05_gpu_kernel/triton` | 8 | 0段1/1段5/3段2 | 01,10-14,30-31 |
| `02_engineering/07_training_reliability` | 4 | 1段3/2段1 | 10-12,20 |

共 138 篇内容页施行/调整编号，全库改写裸基名 `[[wikilink]]` 链接与正文字面路径约 3025 处（含
`wiki/courses/` 两页、`wiki/index.md` 快速导航、`tools/labs_torch_compile/demo_manifest.json` 的
`page` 字段与 `test_volume_demo_contract.py` 硬编码文件名字面量同步）。

**豁免清单**（<4 内容页目录，按 spec §5 门槛不编号，仅数页确认）：`01_theory/01_models` 根（3）、
`tencent_hunyuan`（2）、`meituan_longcat`（2）、`thinking_machines`（1）、`02_train_frameworks/mindformers`
（2）、`03_infer_frameworks` 根（1）、`speculative_decoding`（2）、`sglang`（1）、`06_auto_parallel`（1）、
`02_compile_stack/05_codegen_backends/mlir/npu`（2，P4 既定豁免，本次仅修正大小写见下）。

**Batch0：孤立大小写/点号残留修复**（在既定编号目录内，Task 7 清单明文列出，独立于本次 13 目录扫描）：

- `04_inductor/npu/10_NPU_Inductor_Backend_Analysis` → `10_npu_inductor_backend_analysis`（P4 已编号目录，仅改大小写）
- `05_codegen_backends/mlir/npu/NPU_MLIR_Backend_Technical_Analysis` → `npu_mlir_backend_technical_analysis`（豁免目录，仅改大小写，不加编号）
- `03_runtime_graphs/cuda/01_PyTorch_CUDA_Graphs_Complete_Guide` → `10_pytorch_cuda_graphs_complete_guide`：判段修正——该页是四种用法+实现原理的核心机制主线而非纯入门页，从段 0 移入段 1；同目录 `10_cudagraph_trees_warmup_record_and_replay_analysis` 顺延为 `11_...` 保持流水线顺序
- `01_theory/04_posttraining/29_kimi_k1.5_analysis` → `29_kimi_k1_5_analysis`（P5 Task 8 changelog 原文明确"点号未消除…属 P7 范围非本任务"，本次补齐）

**snake_case / 非法后缀修复清单**（编号时一并处理，含 deepseek 目录内 4 处非法后缀就近改名）：

| 旧名 | 新名 | 说明 |
|---|---|---|
| `mHC.md` | `25_mhc_analysis.md` | 大写+无后缀 |
| `Engram_Analysis.md` | `29_engram_analysis.md` | 驼峰大写 |
| `deepseek_math_v2.md` | `18_deepseek_math_v2_analysis.md` | 缺 `_analysis` 后缀 |
| `deepseek_v4_technical_deep_dive.md` | `26_deepseek_v4_technical_deepdive.md` | `_deep_dive`→`_deepdive` |
| `deepseek_v4_implementation_details.md` | `27_deepseek_v4_implementation_deepdive.md` | 非法后缀 `_details`→`_deepdive`（内容自称"实现级 Deep Dive"） |
| `deepseek_v4_architecture_diagrams.md` | `28_deepseek_v4_architecture_analysis.md` | 非法后缀 `_diagrams`→`_analysis` |
| `deepseek_v4_audit_report.md` | `30_deepseek_v4_audit_analysis.md` | 非法后缀 `_report`→`_analysis` |
| `kimi_k2.5_analysis.md` | `13_kimi_k2_5_analysis.md` | 点号消除 |
| `RL_Training_Inference_Precision_Analysis.md` | `20_rl_training_inference_precision_analysis.md` | 全大写 |
| `async_collective_tensor_deep_dive.md` | `21_async_collective_tensor_deepdive.md` | `_deep_dive`→`_deepdive` |
| `muon_sharded_hsdp_report.md` | `22_muon_sharded_hsdp_analysis.md` | 非法后缀 `_report`→`_analysis` |
| `distributed_optimizer_deep_dive.md` | `32_distributed_optimizer_deepdive.md` | `_deep_dive`→`_deepdive` |
| `megatron_moe_training_optimization_report.md` | `01_megatron_moe_training_optimization_analysis.md` | 非法后缀 `_report`→`_analysis` |
| `vllm_feature_optimizations_overview.md` | `01_vllm_feature_optimizations_guide.md` | 非法后缀 `_overview`→`_guide`（内容是"问题→flag→代码→深挖页"决策指南） |
| `triton_knowledge_map.md` | `triton_31_knowledge_guide.md` | 非法后缀 `_map`→`_guide`（同目录其余 7 篇均用 `_guide`） |
| `triton_00`–`triton_06` | `triton_01`/`triton_10-14`/`triton_30-31` | 课程序号规范化为段位（`triton_` 前缀保留，见下方判段说明） |

`_deep_dive`（5 篇）→`_deepdive` 全库清零；非 snake_case 全库清零（`find wiki -name "*.md" | grep -E "[A-Z]"` 为空）。

**遇到的一处误伤自查**：`mHC`→`25_mhc_analysis` 的全库替换脚本第一遍执行时，误将约 20 篇文件中作为
**技术缩写术语**裸出现的 "mHC"（如"该方案称为 mHC"）也替换成了文件名——因为原文件名恰好就是不带
后缀的裸词 "mHC"。发现后写了第二遍脚本：仅保护 `[[...]]` 链接span 内的替换结果，把 span 外的
"25_mhc_analysis" 字面量改回 "mHC"，回退 139 处误伤，链接本身不受影响（validate: checker broken=0、
pytest 77 通过）。其余各批次的旧文件名均已带 `_analysis`/`_guide`/`_report` 等后缀，词形唯一，未复现此问题。

**判段说明（内容实质优先于文件名后缀/既有体裁标签，同 P4 Task 9.5 / P5 Task 8 规程）**：

- `triton/` 目录的原课程序号 `triton_00`–`triton_06` 是计划书明文交办的特例处置：00（L0 地基）
  →段 0；01-05（L1 会写×3/L2 会调/L3 会debug）→段 1；06（L4 会优化，profiling 方法论）→段 3；
  未编号的 `triton_knowledge_map`（总纲/自测/资源）判定为方法论参考，同入段 3（31），并顺带补上
  述后缀修复。
- `02_train_frameworks/megatron-lm/megatron_pp_parallelism_analysis`（根目录内）：与
  `megatron-lm/15_megatron_pp_schedulers_analysis` 存在 spec §3.4 记载的三页合并待办（本次未执行,
  见 2026-07-29 设计文档 §3.4）——该合并未落地，页面仍按现状原地编号（`20_`），不属本任务范围。
- `05_gpu_kernel` 根：`cuda_execution_model_guide`/`operator_optimization_guide` 是 P6 Task 4 归一
  后的两篇"权威页"，判入段 1（核心机制主线）而非段 3；`gpu_kernel_guide` 内容覆盖面最广（执行层级
  /内存/Tensor Core/torch.compile/NPU 差异全景），判入段 0 作全域入口。
- `02_train_frameworks/mindspeed`：四大类特性(并行/通算掩盖/内存/昇腾亲和)判入段 1（index 原文
  称"四大类特性"为并列体裁）；仅 CP 深挖一篇（index 原文自称"并行·CP 深挖"）判入段 2。

**校验**：`python tools/check_links.py`：pages=373，broken=0（9 个 commit 均独立核验，ambiguous=70/
bare_index=70 为既有基线，P7 Task 8 处理范围）；`pytest tools/ -q`：77 passed（9 个 commit 均独立
核验，含 labs demo_manifest 契约测试）。

---

## 2026-07-31：知识库结构整改 P6 Task 5 —— NPU 三页划界 + 中重叠七组扫尾

**Type**: Structure Reorg / Dedup（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.5/§3.6；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 5）

### 一、npu/ 三页 vs 上游新页划界

`02_compile_stack/04_inductor/npu/` 的 `20_npu_lowering_guide.md`(884)/`22_npu_fusion_passes_deepdive.md`(425)/`30_npu_vs_upstream_fusion_passes.md`(287) 逐节核对 vs 上游 `10_fx_lowering_to_inductor_ir_analysis.md`（原 C17，GraphLowering/call_function 决策树/lowering 注册 API）与 `03_graph_ir_and_passes/` pass 页（`21_fx_graph_editing_primitives_and_invariants_analysis.md` FX 改图原语与不变量、`12_graph_effects_alias_mutation_and_order_analysis.md` functionalization）：

- `20_npu_lowering_guide.md`(884→869) §1.1「什么是 Lowering」的通用 FX→IR 管线示意图（与 §1-§5 讲的是同一套 `GraphLowering` 机制）收缩为一句 + 链接指向 `10_fx_lowering_to_inductor_ir_analysis`；§1.2 起的 torch_npu monkey-patch 策略、§2-§9（架构对比表、fallback 分流、专有 IR 节点、配置体系、v2.7.1 源码复核）**全部核实为 NPU 特有内容，未删改**。
- `22_npu_fusion_passes_deepdive.md`(425→428) §7「改图操作原语与 pass 通用原理」核对：`replace_all_uses_with`/`erase_node`/insertion point 语义及事务化改图状态机是 PyTorch FX **通用**设计，已在 §7 顶部新增划界声明指向 `21_fx_graph_editing_primitives_and_invariants_analysis`（原语通用语义）与 `12_graph_effects_alias_mutation_and_order_analysis`（post-grad 图 functionalization 基础）；§7.1 原语表『代表用处』列（`ascend_graph_pass.py` file:line）、§7.3 `view_fold_pass` 逐 pass 走查（含 DAG 扇出 mermaid）、§7.4 三条贯穿原理判定为 **NPU 应用侧独有内容，全部保留**（该表/走查在上游页无对应物——是"用这些通用原语具体怎么改某个 NPU pass"而非重讲原语本身）。
- `30_npu_vs_upstream_fusion_passes.md`(287→288) 核实为**已是范本**：全篇以 file:line 逐层对照上游三阶段 pass（§3.1-3.3 已链 `30_pre_grad_passes_guide`/`31_joint_graph_passes_guide`/`32_post_grad_passes_guide`），无需收缩；§3.5 补一句 + 链接 `10_fx_lowering_to_inductor_ir_analysis`（lowering/decomposition 通用机制侧）。
- **双向链回补**：`10_fx_lowering_to_inductor_ir_analysis`、`21_fx_graph_editing_primitives_and_invariants_analysis` 此前均未反向链接 npu/ 三页，补 Related Pages 各一条。

三页合计 1596→1585 行；无独有内容删除，全部核实后原样保留。

### 二、中重叠 7 组现状核查（spec §3.6 残余）

逐组 grep 现状，只补缺失的双向链，已处理的记录出处：

| 组 | 现状 | 处置 |
|---|---|---|
| vLLM compilation ↔ `03_runtime_graphs` | 正向链（vLLM→`10_pytorch_cuda_graphs_complete_guide`/`11_torch_compile_npugraphs_deepdive`）P3 已补；反向链缺失 | 补 2 条反向链（CUDA 侧 + NPU 侧回指 vLLM 分段 CUDA Graph 应用实例） |
| vLLM IR/fusion ↔ pass 页 | 原声明仍在：`25_vllm_ir_and_fusion_passes_analysis` 已链 `22_pattern_expression_and_matcher_engine_analysis`/`24_graph_pass_pipeline_ordering_and_fixpoint_analysis`/`32_post_grad_passes_guide`；`24_...§14` 跨框架对照表已含 vLLM/sglang/npu 三个代表页双向链 | 已完成，无需改动 |
| sglang ↔ vllm | 健康范本核实：两页头部即互相声明"对照面"，`24_...§14` 表格双向收录 | 已完成，无需改动 |
| TIM 分层 | `26_tim_causal_chain_analysis` 头部四环因果链声明完整，与 `25_on_policy_off_policy_staleness_analysis` §7 边界区分明确 | 已完成于 P5 |
| operator_optimization ↔ kernel 页 | 六页归一（Roofline/执行模型双权威 + NPU 段划界） | 已完成于本 P6 Task 4（commit `dbaa37e`） |
| D05 ↔ sandbox/infra | `01_posttraining_infra_mechanism_analysis` §4/§7 与 `11_rl_sandbox_design_analysis`/`12_rl_infra_efficiency_analysis` 逐句对照、独有内容迁移、三方划界声明均已生效并核实链接有效 | 已完成于 P5 Task 4 |
| megatron_precision_cudagraph ↔ Guide | 核查发现**未补链**（P3 遗留待办未落地）：`23_megatron_precision_cudagraph_fusion_analysis.md` Related Pages 无任何指向 `10_pytorch_cuda_graphs_complete_guide` 的链接 | 补双向链（训练框架应用实例 ↔ CUDA Graph 通用机制权威页） |

**验收**：`python tools/check_links.py` pages=373、broken=0（ambiguous=70 不变，属既有裸 index 基线，P7 Task 8 范围）；`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 完成（高重叠清零）

**Type**: Structure Reorg（设计 §3.5/§3.6；P0-P7 的第七段）

- **Ring Attention/CP 四写归一**：新建理论权威页（674 行，骨架来源逐节标注、公式口径冲突并列披露含一处源内不一致 [!contradiction]）；四框架页收缩为实现差异（DSv4 特有零删减经审查回补后达成）。
- **横向页矩阵化**：通信掩盖页 271→124（纯对比矩阵+扩 MindSpeed 列）；分布式优化器横向页收缩；**Megatron 优化器三页 1118→984 合并为单页**；FSDP 四页补分工声明。
- **Roofline/执行模型归一**：六页 2046→1922，执行模型权威=10_cuda_execution_model_guide，Roofline 权威=11_operator_optimization_guide §2；昇腾页各留划界。
- **npu 三页对上游划界**；中重叠七组全部补齐双向链（含 P3 承诺未落地的一处）。
- 至此设计盘点的 **13 组高重叠全部清零**。全程 broken=0；wiki 374→373 页。

## 2026-07-31：知识库结构整改 P6 Task 4 —— Roofline / GPU 执行模型归一

**Type**: Structure Reorg / Dedup（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.5；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 4）

`05_gpu_kernel/` 内 Roofline 讲了 ~4 遍、GPU 执行模型讲了 ~3 遍的六页归一，定两个权威：

- **执行模型权威 = `10_cuda_execution_model_guide.md`**（280→282 行）：核实其 Grid/Block/Warp/Thread/SM 讲解已完整覆盖 `01_gpu_kernel_guide.md` §01 与 `triton_01_gpu_essentials_guide.md` §2 直觉一的全部内容，无独有段需吸收；页头新增"本页地位"权威声明。
- **Roofline 权威 = `11_operator_optimization_guide.md` §2**（834→760 行，含新增 §2.5）：吸收 `triton_01_gpu_essentials_guide.md` §3 的独有内容——用 Triton 官方 `01/02/03-*.py` benchmark 公式逐项手算 AI 的完整推导（向量加法/融合 softmax/矩阵乘三例，含源码行号引用），逐字迁入新增 §2.5；§2 头新增权威声明。核实 `triton_30_optimization_profiling_guide.md` §1 与 `01_gpu_kernel_guide.md` 均无 Roofline 独有内容（前者的流程图与优化闭环已被 11_operator_optimization_guide §3/§8 覆盖，后者仅 §10 诊断清单一句提及）。
- **四页收缩为指针**：`01_gpu_kernel_guide.md`(303→299 行) §01 执行层级模型（含 Grid/Tiling 关系一节，该点已由 `20_cuda_gemm_kernel_analysis` §1 更深覆盖，改指该页）收缩为一句 + 双链接；§02 内存层级核实为无重复的唯一详解，未改动。`triton_01_gpu_essentials_guide.md`(119→90 行) §2 直觉三 + 原 §3 Demo 收缩为一段结论 + 链接（保留"一眼判别法"一句作为课程起点的独立可读锚点，§1/直觉一/直觉二/§3 分工表/§4 动手验证均未改，学习路线连贯性验证通过）。`triton_30_optimization_profiling_guide.md`(297→276 行) §1 的 roofline 循环流程图 + 长引述收缩为一句 + 链接，保留"优化杠杆速查表"（页内导航，非概念复述）。
- **§6 昇腾段 vs `22_ascend_kernel_execution_model_analysis.md` 判定**：后者独有内容（四类单元显式缓冲链完整图示、CUDA/Ascend 逐项对位表、GEMM 四层结构对照、片上缓冲预算账本、非 GEMM 算子按类分派表、FlashAttention Cube-Vector 融合、训练层三条主线）远超 50%，判定**各留 + 划界**（非合并）：两页互加"与 XX 的划界"声明——`11_operator_optimization_guide.md` §6.1/§6.2（834 行版本内约 149→95 行，含 §6 全节）的 AI Core 结构图/存储层次图/CopyIn-Compute-CopyOut 完整代码收缩为摘要 + 指向该深度页，保留独有的 Tiling 约束数值推导（L0A 容量→M 维上限 512）、DataCopy 32-byte 对齐要求、§6.3 GPU 经验迁移 checklist 与 AICPU/host CPU fallback 辨析（后两者深度页未覆盖，全部原样保留）；`22_ascend_kernel_execution_model_analysis.md`(213→215 行) 页头新增划界声明。
- **索引**：`05_gpu_kernel/index.md` 补 `11_operator_optimization_guide` 页面列表行（此前缺失，未入索引）；`10_cuda_execution_model_guide`/`01_gpu_kernel_guide`/`triton_00`/`triton_06`/`22_ascend_kernel_execution_model_analysis` 的 Related Pages 互补权威页链接。
- **净效果**：六页合计 2046→1922 行（−124，新增 sourced 内容的同时净收缩）；无内容丢失，独有段全部逐字保留或迁移。
- **验收**：`python tools/check_links.py` pages=373、broken=0（ambiguous=70 为既有裸 index 基线，不属本次范围）；`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 Task 3（二）—— Megatron 分布式优化器三页合并 + 横向页矩阵化

**Type**: Structure Reorg / Dedup（计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 3 第二部分）

- **三页合并为一**：`megatron-lm/megatron_ddp_optimizer_analysis.md`(446)、`megatron-lm/megatron_optimizer_internals_analysis.md`(227)逐字并入 `megatron-lm/16_megatron_distributed_optimizer_analysis.md`(原 445→合并后 984 行),文件名保留最通用者。以 `megatron_ddp_optimizer_analysis` 的 ZeRO 0-3 阶梯框架(§0-§5,含「阶段①-④」命名,供全库既有引用免改)为骨架;`megatron_optimizer_internals_analysis` 的优化器类层次/混合精度/step 五步/Loss Scaling/梯度裁剪/LR 调度整体并入为新 §6-§11;本页原稿(通信组定义/FP8-FP4/CPU offload/三种 FSDP 实现对比/Layer-Wise+Muon 集成)保留为 §2.8-§2.11、§12-§14,真正重复的段落(通信量 P 记号复述、类继承结构的简化版)删除、仅保留 2026-06-16 增量更新的独有更正。三页内部原有的跨页 `[[wiki link]]` 互指(如"详见 xxx §A.7 的更新")已改写为同页 §N 引用。
- **两个 torchtitan FSDP 四页补分工声明**(不合并,`02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis.md`/`25_torchtitan_simple_fsdp_analysis.md`/`20_torchtitan_fsdp_prefetch_overlap_memory_analysis.md`/`21_torchtitan_hsdp_backward_overlap_analysis.md` 各页头补一句"四页分工"声明,标杆篇/深挖伴篇/HSDP 展开篇/编译器路径替代方案互指)。
- **横向页收缩**:`32_distributed_optimizer_deepdive.md`(194→185 行)§一 的 ZeRO-1 因果链完整推导(与合并后的 megatron 页 §阶段② 重复)收缩为一段 + 指针,保留 §二(梯度累积 K 倍通信量表)、§三(FSDP2/Megatron/MindSpeed 三方对比矩阵,核心矩阵型内容)、§四(MindSpeed param 临时化,全库唯一出处,未改)、§五/§六(Adam vs Muon 内存与系统影响,通用理论、未在任何子页找到对应,未改)、§七(选型决策树)原样保留;补 §六.3 指向合并后 megatron 页 §14.1 的 Muon 具体解法指针;补齐全页此前缺失的 `## Related Pages` 区块(6 条)。
- **入链改写**:全库 12 个引用 `megatron_ddp_optimizer_analysis`/`megatron_optimizer_internals_analysis` 的文件(含 `megatron-lm/index.md` 的系列计数 18→16、`torchtitan/index.md`、`19_megatron_dist_checkpointing_analysis.md`、`17_megatron_parallelism_orchestration_analysis.md`、`15_megatron_pp_schedulers_analysis.md`、`12_megatron_tp_analysis.md`、`27_megatron_tp_fsdp_resharding_supplements_analysis.md`、`23_megatron_precision_cudagraph_fusion_analysis.md`、`28_megatron_training_stability_observability_analysis.md`、`11_torchtitan_fsdp_analysis.md`、本 changelog 历史条目的链接目标)全部改指 `16_megatron_distributed_optimizer_analysis`,原两处独立 Related Pages 行合并去重;`megatron-lm/index.md` 移除两行系列表格行(补一句"已并入"说明)。
- **验收**:`python tools/check_links.py` pages=373(375−2)、broken=0;`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 Task 3（一）—— 通信掩盖横向页收缩为跨框架矩阵

**Type**: Structure Reorg / Dedup（计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 3 第一部分）

- `02_train_frameworks/30_comm_compute_overlap_analysis.md`(271→124 行)由"基于 Megatron/torchtitan 两框架源码分析"的机制正文页,收缩为 **Megatron-LM/torchtitan/MindSpeed 三框架计算通信掩盖跨维度对比矩阵页**:与三个框架的通信掩盖权威机制页([[20_megatron_comm_overlap_analysis]] 740 行、[[24_torchtitan_comm_optimizations_overlap_analysis]] 170 行、[[11_mindspeed_comm_overlap_analysis]] 461 行)逐节核对,机制正文找到子页对应即收缩为一句 + 链接;子页缺失的机制段逐字下沉(注明来源):
  - `20_megatron_comm_overlap_analysis.md` 新增 §5.8(Shared Expert 独立 stream 状态机,`moe_shared_expert_overlap`)、§5.3 补 Layer→5 子节点拆解/`stream_acquire_context()`/镜像层配对三段、§5.6 补 DeepEP/HybridEP 硬件后端速查表(图片迁至 `megatron-lm/assets/`)。
  - `14_torchtitan_pp_analysis.md` §7.2/§7.3 补 I/W 拆分"非按模型结构"要点与 `OVERLAP_F_B` 源码 + "非真并发"澄清。
  - `24_torchtitan_comm_optimizations_overlap_analysis.md` §3.3 补 Graph Trainer FSDP AG/RS 重排 pass 一句。
  - 新增 §四"MindSpeed 掩盖机制概览"(此前该页完全未提及 MindSpeed)。
- 保留三子页均无的合成视图:概念分层图(§一)、combined_1f1b vs ZBV/DualPipeV 架构差异分析(新 §三,含 MindSpeed fb-overlap+DualPipeV 定位)、Sub-Layer 级掩盖可达性矩阵(§四,补 MindSpeed 列)、框架差异总结表(§五,补 MindSpeed 行)。
- `31_comm_compute_fusion_guide.md` 与本页头各补一句"融合"(单 kernel)vs"掩盖"(独立调度重叠)边界声明,互指。
- **验收**:`python tools/check_links.py` pages=375、broken=0;`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 Task 2 —— CP/Ring Attention 归一

**Type**: Structure Reorg / Dedup（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.5；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 2）

- **新建** `01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md`（655 行）：抽取四份框架 CP 分析页（`13_megatron_cp_analysis` 391、`13_torchtitan_cp_analysis` 345、`20_mindspeed_context_parallel_analysis` 420、`35_deepseek_v4_context_parallel_analysis` 853，合计 2009 行）里重复讲的通用机制——CP 动机（attention `O(S²)` 墙）、与 TP/PP/DP/EP 的组合关系、序列切分（朴素连续切分 + 折叠/头尾配对负载均衡定量证明 + PTRR 任意稀疏掩码均衡 + RoPE 切分不变量）、因果 mask 三分支裁剪、四种通信调度（Ring P2P + online-softmax、All-gather 双缓冲、Ulysses 头维换轴、分层混合 N 级分组构造）、通信量代数统一对比、Dynamic CP 通用机制。逐段择"最深最完整版本"逐字为骨架并注明来源：MindSpeed 提供负载均衡定量证明 + RoPE 不变量 + 因果三分支裁剪 + online-softmax 公式 + Ulysses 全套机制 + Hybrid 量化论证；torchtitan 提供 Ring 主循环伪代码 + 通信掩盖时序 + 反向双环原理 + PTRR；DeepSeek-V4 提供 Native CP AllGather 代码 + 分层分组构造代码 + 通信量统一公式；Megatron-LM 提供 CP 动机 + 并行组合关系。其余版本的独有补充/口径差异（如 Ulysses 通信量"2 次"vs"4 次"计数粒度、通信量公式是否显式含 `/TP` 因子）逐段并注，不强行合并掩盖。
- **四页收缩**（通用段替换为一句定位 + 链接，页头补划界声明，各自实现差异/源码走读/性能数据/配置全保留）：
  - `13_megatron_cp_analysis.md` 391→131 行：保留 `cp_comm_type` 四选一配置接口、TE 透传架构、选型决策树、Dynamic CP 的 dispatcher 兼容/CUDA Graph 守卫等 Megatron 特有源码细节。
  - `13_torchtitan_cp_analysis.md` 345→181 行：保留 SDPA-ring 与 FlexAttention-allgather 双路径架构（torchtitan 独有）、DTensor dispatcher 接线、`functional collectives`/`AsyncCollectiveTensor` 异步实现（不手写 CUDA stream）。
  - `20_mindspeed_context_parallel_analysis.md` 420→345 行：保留五算法运行期分派脊柱、Ring **双环**（outer/inner window + 双 dKV 反向环）、**Adaptive CP**（调度驱动 + rank 重映射）、**KV-cache CP**（显存换反向通信）——后三项四框架中仅 MindSpeed 独有。
  - `35_deepseek_v4_context_parallel_analysis.md` 853→613 行：只裁掉与理论页字面重复的 Native CP AllGather 源码 walkthrough、Hierarchical CP 分组构造代码、标准 attention 通信量公式；MLA 降低 CP 通信量 ~128 倍的推导、CSA/HCA 压缩注意力与 CP 交互的论文↔代码 gap 审计（本页核心贡献）、RoPE 的 CP 感知、TE `cp_stream` 双缓冲机制、Dynamic CP 对 MLA 的不支持、CP 与 EP 带宽竞争等 DSv4 特有内容零删减。
- **索引与入链**：`06_distributed_parallelism/index.md` 补条目（页面列表 + 建议阅读顺序 + 显存通信总账 CP 行 + 关联域指向四份框架页）；`megatron-lm/index.md`、`torchtitan/index.md`、`mindspeed/index.md` 三处描述同步指向理论页；四页 Related Pages 与理论页 Related Pages 双向互链。
- **验收**：`python tools/check_links.py` pages=375（374+1）、broken=0、orphans=0；`ambiguous`/`bare_index` 69→70（新页 Related Pages 末尾沿用同目录 8 个既有页面的裸 `[[index]]` 惯例，与本次改动前基线持平 +1，属 P7 Task 8 全库裸 index 清零范围，非本次引入的新问题）。`python -m pytest tools/ -q` 77 passed。

## 2026-07-31：P6 Task 2 逐句复核修复（六项）

**Type**: Correction（对上一条目 P6 Task 2 归一结果的逐句审查，`git show 70d096e:...` 逐字取回原文核对）

复核发现归一/收缩过程中丢失或失真的六处，逐项修复（不改上一条历史记录，按追加惯例新增本条）：

1. **A1 回补**：旧 `35_deepseek_v4_context_parallel_analysis.md` §2.4.2 的 All-gather"关键缺陷"半句——KV buffer 显存代价公式（原 `2×S×B×Hₖ×D`）在收缩时全库丢失。补回理论页 §6.2「显存代价」新增段（All-gather 机制的通用属性，非 DSv4 专属）。
2. **A2 回补**：旧 DSv4 §2.4.4 的 a2a+p2p 三级递进数据流（Level 1 Pair A2A(2-GPU) → Level 2 Quad A2A(4-GPU，建立在 Level 1 之上) → Level 3 跨节点 P2P 环）及"NVLink 承担大部分通信量"的关键优势说明，收缩后两处均只剩两级描述。逐字补回理论页 §8.2 末尾新增「执行序」段。
3. **虚假指针修正**：`35_deepseek_v4_context_parallel_analysis.md:565` 与理论页 §11 声称 `resolve_cp_group`/`PackedSeqParams` 源码"已并入 `13_megatron_cp_analysis.md` §3"，但收缩时该源码级细节（`packed_seq_params.py:23-24`/`:69`、`transformer_engine.py:1798`、#5215 修复 `:1886`、`GPTModel`/`GatedDeltaNet`/MTP 消费者清单）被误删且全库无处收留。采用方案 (a)：逐字恢复进 `13_megatron_cp_analysis.md` §3.1「机制（源码）」，两处指针随之变为如实表述（391→131→135 行）。
4. **理论页 §9 补充口径差异第三条**：核实 DSv4 与 MindSpeed 的 Ulysses/a2a 通信量公式在扣除已知的 TP 因子、fwd/bwd 口径两项调整后仍残留 **cp 倍**未归因差异（Ring 行同样两项调整后可精确抵消至 2 倍，Ulysses/a2a 行不能）——新增 `[!contradiction]` callout 如实披露，不假装已解释，提示读者两页公式不可直接换算。
5. **符号消歧**：§9 开头补充说明本节 $h$ 指头数，与 §0 全局记号表的 $h$（隐藏维度）是两个不同的量（本节 $h\times d$ 才对应 §0 的 $h$）。
6. **引用来源订正**：理论页 §4.1 "从朴素 $cp\cdot$(全块)降到约一半"一句的引用来源从 `20_mindspeed_context_parallel_analysis.md §4.3` 订正为 `§4.3/§4.4`（三分支裁剪机制本体在 §4.3，但该精确量化措辞的原始出处是 §4.4 通信量代数节的 `[!tip]` 优化点 callout）。

**验收**：`python tools/check_links.py` pages=375、broken=0（不变）；`python -m pytest tools/ -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P5 完成（后训练三域整合收官）

**Type**: Structure Reorg（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；P0-P7 的第六段）

- **`03_posttraining/` 纵向域解散**：D01-D12 按主题分发进功能树（理论算法→`01_theory/04_posttraining/`，框架与 infra→`02_engineering/04_posttrain_frameworks/`，K3 案例→`moonshot_kimi/`）；D00 阅读路线降为 `courses/posttraining_frontier.md` 纯导读页。
- **GRPO 三写归一**：D02 定位为算法演进权威页；grpo/dapo/gspo 论文页瘦身为"论文档案"（元数据+实验+特有公式）；verl 页保工程锚点——审查中回补 J_GRPO 的 KL 组合项等三处符号级细节。
- **verl 双基线整合**：D07（`983cb0f`）入主 verl/ 为端到端主链，ray_trainer 保留为 legacy 深潜页；**源码核实发现 `trainer.use_v1` 默认翻转**（旧教学主链在新基线非默认路径），三处 [!contradiction] 双记；其余 8 篇挂基线横幅。
- **错位页归位**：RL_PPO_Loss（实现分析）→框架域；20_batch_invariance_guide→训练可靠性域。
- **三目录分段编号**：01_theory/04_posttraining（17 页）、04_posttrain_frameworks 根（9 页）、verl/（10 页）按 0/1/2/3 段位编号，485 处链接改写。
- 全程 broken=0；wiki 375→374 页（净 −2 +1 课程页）。

## 2026-07-31：知识库结构整改 P5 Task 8（三目录分段编号）

**Type**: Naming Convention（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §5；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 8；P4 Task 9.5 同款规程）

对 P5 已重构定型的三个目录施行 §5 分段编号（文件名两位数字前缀 = 段位：段 0(01-09)入门导览、段 1(10-19)核心机制主线（按流水线/学习顺序）、段 2(20-29)深潜/专题、段 3(30-39)方法论/对照/工程实践；`index.md` 不编号）：`01_theory/04_posttraining/`、`02_engineering/04_posttrain_frameworks/`（根）、`02_engineering/04_posttrain_frameworks/verl/`。每目录先在 index.md 定段位表，再 `git mv`，全库改链，checker=0。判段按内容实质而非文件名后缀/既有体裁标签。`moonshot_kimi/`、`07_training_reliability/` 本次不编号（P7 全库推广时处理）。

### 段位表 1：`01_theory/04_posttraining/`（17 篇内容页）

| 段 | 编号区间 | 页面 |
|---|---|---|
| 0 | 01 | posttraining_frontier_map_analysis |
| 1 | 10–13 | instructgpt_rlhf_analysis、ppo_analysis、dpo_analysis、reasoning_rl_algorithm_evolution_analysis |
| 2 | 20–29 | grpo_analysis、dapo_analysis、gspo_analysis、rloo_analysis、agentic_rl_algorithm_analysis、on_policy_off_policy_staleness_analysis、tim_causal_chain_analysis、vapo_analysis、rlhf_foundations_analysis、kimi_k1.5_analysis |
| 3 | 30–31 | preference_optimization_analysis、reward_hacking_defense_analysis |

段 1 按学习序排列——InstructGPT 三步 RLHF 基础 → PPO-Clip → DPO → Reasoning RL 算法演进权威页（`13`，P5 Task 3 已定为 GRPO/DAPO/GSPO/SAO 谱系统一权威页）。段 2 恰好填满 10 格：GRPO/DAPO/GSPO/RLOO 四篇论文档案（Task 3 已瘦身为元数据/原始实验数字，公式与动机指向 `13`）+ agentic RL/staleness/TIM 因果链三个专题 + VAPO/RLHF 多论文综合/Kimi K1.5 三篇档案页。

### 段位表 2：`02_engineering/04_posttrain_frameworks/`（根，9 篇内容页；`verl/` 子目录见表 3）

| 段 | 编号区间 | 页面 |
|---|---|---|
| 0 | 01 | posttraining_infra_mechanism_analysis |
| 1 | 10–12 | rl_ppo_loss_and_grpo_analysis、rl_sandbox_design_analysis、rl_infra_efficiency_analysis |
| 2 | 20–22 | slime_architecture_analysis、areal_async_architecture_analysis、roll_strategy_and_ascend_analysis |
| 3 | 30–31 | rl_framework_comparison、cuda_ascend_posttraining_stack_comparison |

### 段位表 3：`verl/`（10 篇内容页）

| 段 | 编号区间 | 页面 |
|---|---|---|
| 0 | 01–02 | verl_architecture_overview_analysis、verl_quickstart_guide |
| 1 | 10–15 | verl_end_to_end_iteration_analysis、verl_single_controller_analysis、verl_dataproto_analysis、verl_workers_engine_analysis、verl_rollout_resharding_analysis、verl_rl_algorithms_analysis |
| 2 | 20 | verl_ray_trainer_analysis |
| 3 | 30 | verl_optimization_analysis |

段 1 按 index.md 既有「五条平面」表的管线顺序排列（入口/驱动当前基线主链 → 控制面 → 数据面 → 计算面 → 生成面 → 算法面）。

**判段说明**（内容实质优先于文件名后缀/既有体裁标签，同 P4 Task 9.5 规程）：

- `posttraining_infra_mechanism_analysis`（D05）：`_analysis` 后缀但内容是 control/data/weight 三平面模型这一域级总览，与 `posttraining_frontier_map_analysis` 在理论域的段 0 定位同构，判入段 0；与之相邻的 `rl_framework_comparison`（明确 comparison 类）分列段 3，两者不会同段竞争。
- `preference_optimization_analysis`：`_analysis` 后缀但内容是 DPO/IPO/SimPO/ORPO/KTO/MODPO 横向对比表，按内容判入段 3（对照类）而非段 1/2。
- `reward_hacking_defense_analysis`：内容是环境/penalty/inoculation prompting/agentic safety 四层防御体系，属方法论/工程实践，判入段 3。
- `rlhf_foundations_analysis`：`_analysis` 后缀但结构是 ReMax/Weak-to-Strong/RM Overoptimization/RigorLLM 等多篇独立论文的综合档案，与 grpo/dapo/gspo/rloo 四篇论文档案同类，判入段 2 而非段 1。
- `verl_ray_trainer_analysis`：verl/index.md「深挖实现」分组的一员，但该页自身层次列与页头横幅均明确自称 `8a694930` **legacy 深潜**（当前默认主链已切至 `verl_end_to_end_iteration_analysis` 的 `983cb0f`），故不与同组其余四篇（single_controller/dataproto/workers_engine/rollout_resharding）一起入段 1 主线，单独判入段 2。
- `verl_rl_algorithms_analysis`：虽在 verl/index.md 旧「算法与优化」分组与 `verl_optimization_analysis` 并列，但按「五条平面」表它是管线终点的算法面，判入段 1 主线；`verl_optimization_analysis` 是横切性能/显存方法论，判入段 3。

### 执行

`git mv` 36 个文件（17+9+10，均为「两位数字前缀 + 原基名」，不改文件名其余部分——含 `kimi_k1.5_analysis.md` 的点号未消除，naming §5 命名统一属 P7 范围非本任务）。全库改写裸基名 `[[wikilink]]` 链接 **485 处**（53 个文件，含 `courses/posttraining_frontier.md` 全部 D01–D12 阅读序链接、三目录自身 index.md 与页内互链、`wiki/index.md` 快速导航两行）。`wiki/changelog.md` 中 **6 处历史条目共 20 个 wikilink**（2026-05-24 交叉引用行、2026-06-22 verl 系列创建条目 9 处、2026-06-24 GLM-5 校验行、2026-07-25 TIM 首篇创建行、2026-07-27 D01 创建行，以及本 P5 序列内 Task 2 自身条目 1 处）按"历史不回写"惯例降级为反引号 + 去向说明；Task 3/4/5/7 自身条目中的相关链接此前已用反引号包裹（元描述文本），未被本次渲染判定为活链接，无需处理。

**顺带修订（Task 7 changelog 条目审查移交的两处）**：pages 口径句"计划文本写'375−2=373'"改为"按本次执行前的推算 375−2（未计新增课程页 +1）"，不改历史行结构，只修正误引；六级能力门槛压缩措辞补一句"压缩保留主干，个别子弹与前置/任务两列未 1:1 保留，实质由链接的深潜页承载"。

**验收**：`tools/check_links.py` broken=0、orphans=0（pages=374，纯改名不增减文件数）；`ambiguous`=69、`bare_index`=69（与 Task 7 基线持平，本次未引入）。`python -m pytest -q` 77 passed。

**自查**：三张段位表在编号前均先读取各页正文首屏/已有 index 描述定段（详见上方「判段说明」五条），非凭文件名猜测；`python tools/check_links.py --json` 全量 JSON 核对 broken 列表为空；链接重写脚本按 `check_links.py` 同款 fence/inline-code 豁免逻辑逐行处理，避免误改示例代码块内的字面 `[[...]]` 文本；`grep -rn` 全库确认无 markdown 式 `](...)` 链接指向本次改名的 36 个文件（本库仅用 Obsidian `[[wikilink]]`，无需额外处理）。

---

## 2026-07-31：知识库结构整改 P5 Task 7（课程页化，解散 03_posttraining 域）

**Type**: Course-page Consolidation + Directory Removal + Index Rebuild（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2/§6；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 7）

**新建** `wiki/courses/posttraining_frontier.md`：纯导读页，结构参照 `courses/torch_compile_end_to_end.md`（角色横幅 → 这门课是什么 → 阅读路线 → 六级能力门槛 → 与功能树的关系 → Related Pages）。从 `03_posttraining/00_posttraining_source_reading_guide.md`（D00，301 行）吸收阅读路线骨架（三原则、最短闭环 mermaid 流程图、"只想先抓主干"短路线）与六级能力门槛（L1–L6，整体保留为表，每级压缩为核心问题 + 验收物两列；压缩保留主干，个别子弹与前置/任务两列未 1:1 保留，实质由链接的深潜页承载）；从 `03_posttraining/index.md` 吸收 S00–S05 阶段叙述，作为阅读路线的六个分段小节（S00 基线与地图…S05 综合验收）；阅读序按 D01→D12 的新位置排列（链接用现基名，Task 8 分段编号后再改一轮，本次不编号）；另从 `03_posttraining/index.md` §5「四个框架的研究分工」吸收一张精简表（verl/slime/AReaL/ROLL 研究角色对照），放在 S03/S04 阅读分组之后。

**D00 独有内容台账**（先核实再删，逐节判定）：

| D00 章节 | 处置 | 理由 |
|---|---|---|
| §0 怎样使用这条路线、§1 阅读顺序表 + 最短闭环 + 抓主干短路线 | 吸收入课程页 | 阅读路线骨架，课程页核心内容 |
| §2 六级能力门槛 L1–L6 | 吸收入课程页（压缩为表） | 门槛定义本身是导读性内容 |
| §3 论文阅读方法（五问法 + 两个笔记模板） | 不迁移，无单独落地 | 通用研究方法论，不含后训练领域独有事实；与本库 `.claude/skills/source-faithful-analysis` 方法论同类，非本域专属 |
| §4 源码阅读方法（owner 追踪六步 + 六问表 + 证据记录模板 Repository/Branch/Commit/…） | 不迁移，无单独落地 | 同上；证据记录模板已被 spec §6.3「代码分析页头钉基线」规则取代 |
| §5 工业实现四级"支持"口径（P1 接口/P2 功能/P3 正确性/P4 性能） | 不迁移，无单独落地 | **核实结论**：已独立复现于 `rl_framework_comparison.md`（D06）§3「四级支持证据」（几乎同一张表，本域权威定义）；`cuda_ascend_posttraining_stack_comparison.md`（D11）§13 进一步实例化为 CUDA→Ascend 专用的 M1–M4 迁移验收；两页均不因 D00 删除而失去定义来源 |
| §6 学习过程实践题（按 S00–S05 阶段） | 不迁移，无单独落地 | 绑定即将解散的 S00–S05 阶段标签，为个人学习进度工具，非可复用领域事实 |
| §7 版本与复习节奏（快速变化页面 30 天复核；K3 `0797decb` 复核触发条件） | 不迁移，无单独落地 | CLAUDE.md 已有通用 staleness 复核规则；K3 复核触发条件是操作性元数据非域事实 |
| §8 最小源码路径与终局验收（各框架最小入口→调用链） | 不迁移，无单独落地 | **核实结论**：verl 入口（`main_ppo.py`→`RayPPOTrainer.fit`）已在 `verl_end_to_end_iteration_analysis`/`verl_ray_trainer_analysis` 逐行覆盖；slime（`train.py`→DataSource/Megatron/SGLang）、AReaL（`PPOTrainer`→workflow/staleness）、ROLL（RLVR/Agentic pipeline→Strategy/device mapping）均已在各自深挖页以真实 `file:line` 覆盖，细致度远超 D00 原一句话摘要 |

`03_posttraining/index.md` 除 S00–S05 阶段叙述外，§1「为什么建立统一领域」并入课程页「这门课是什么」；§4 D00–D12 顺序表、§6 既有知识入口均与 D01（`posttraining_frontier_map_analysis`）自身文档顺序表/既有知识复用规则重复，不重复迁移；§7 维护规则绑定即将解散的目录本身，作废不迁移。

**删除**：`git rm wiki/03_posttraining/00_posttraining_source_reading_guide.md wiki/03_posttraining/index.md`，`wiki/03_posttraining/` 目录随之清空自动移除。

**入链改写**：D00/`03_posttraining/index` 的全部活链接改指 `[[courses/posttraining_frontier]]`，涉及 `24_kimi_k3_posttraining_case_study_analysis.md`（阅读导航"回到 D00"）、`posttraining_frontier_map_analysis.md`（阅读导航 + 文档顺序表第 1 行 + Related Pages，3 处）、`cuda_ascend_posttraining_stack_comparison.md`（Related Pages）、`dapo_analysis.md`/`grpo_analysis.md`/`gspo_analysis.md`/`rl_infra_efficiency_analysis.md`/`rl_sandbox_design_analysis.md`/`verl/index.md`（各 1 处 Related Pages）。`wiki/changelog.md` 中 2026-07-27 历史条目的 2 处活链接（`03_posttraining/index`、`00_posttraining_source_reading_guide`）按"历史不回写"惯例降级为反引号 + 去向说明。

**`wiki/index.md` 重建**：删除"03 后训练纵向学习域"整段；courses 表新增 `posttraining_frontier` 行；快速导航"LLM 后训练前沿 D00–D05"/"D06–D12"两行合并改写为"D01–D06"/"D07–D12"（各 7/6 项，去掉已删除的 `03_posttraining/index`、`00_posttraining_source_reading_guide`，D01–D06 行首改指课程页）；"PPO/GRPO RL 训练"行沿用 Task 6 已完成的 `rl_ppo_loss_and_grpo_analysis` 改名。**页数重算**（覆盖全部因 P5 迁移产生偏差的行，不止 Task 7 直接改动的两域）：模型 56→57、Kimi 13→14（D12 迁入，Task 2）、后训练对齐 15→18（+D01/D02/D03/D04，−RL_PPO，Task 2/3/6）、后训练框架 14→21（+D05/D06/D07/D08/D09/D10/D11/RL_PPO，−batch_invariance，Task 2/4/5/6；verl 子计数 10→11）、训练可靠性 4→5（+batch_invariance，Task 6）；两处"统计于"日期改 2026-07-31。

**`01_theory/04_posttraining/index.md` 与 `04_posttrain_frameworks/index.md`**：两域此前在 Task 2/3/4/6 已逐批为迁入页补全条目 + 一句话定位（"后训练前沿整合"/"后训练框架源码对照"/"RL 算法源码实现"三张表），本次核对确认条目完整、无缺漏；标题/摘要行改为反映扩容后的真实覆盖范围（前者从"LLM 对齐与偏好优化"扩为"LLM 后训练算法理论"）；两处新增课程页入口段落（仿 `01_ai_frameworks/index.md` 既有惯例："阅读路线入口…不计入下表"）+ 关联域各补一行指向 `courses/posttraining_frontier`；日期改 2026-07-31。

**验收**：`tools/check_links.py` broken=0、orphans=0，`ambiguous`/`bare_index`=69（与基线持平，非本次引入）；pages=374。**pages 对账与计划口径的差异**：按本次执行前的推算 375−2（未计新增课程页 +1），但 Task 7 同时创建 1 篇新文件（课程页）与删除 2 篇（D00 + `03_posttraining/index`），净变化为 375+1−2=374，本报告按实际文件系统状态汇报 374，已在交付报告中显式披露此算术口径差异。`python -m pytest -q` 77 passed。

**自查**：`grep -rn "03_posttraining" wiki/` 除本 changelog 历史条目（已降级反引号）与新课程页/两域 index 的解释性反引号文本外 0 处活 `[[...]]` 链接命中；D00 §5/§8 的"不迁移"判定均先用 grep 核实目标内容已在 `rl_framework_comparison.md`/`cuda_ascend_posttraining_stack_comparison.md`/各框架深挖页独立存在，未凭印象判断。

---

## 2026-07-31：知识库结构整改 P5 Task 6（错位页归位）

**Type**: Misplaced-page Relocation（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 6）

两篇此前放错域的页面归位，均为纯搬运（正文零改动，仅补链接）：

- `git mv wiki/01_theory/04_posttraining/RL_PPO_Loss_and_GRPO_Analysis.md` → `wiki/02_engineering/04_posttrain_frameworks/rl_ppo_loss_and_grpo_analysis.md`（snake_case 化）。该页是 TorchTitan + vLLM 的 PPO Loss / GRPO 流程源码级实现分析，此前误放理论目录，实为框架工程分析。与 verl 域已有的 `verl_rl_algorithms_analysis.md`（同为源码级 PPO/GRPO loss 分析，框架为 verl core_algos 注册表）互补一句话双向链接。
- `git mv wiki/02_engineering/04_posttrain_frameworks/20_batch_invariance_guide.md` → `wiki/02_engineering/07_training_reliability/20_batch_invariance_guide.md`。该页讲批次不变性/确定性算子实现（源自 DeepSeek V4 报告 §3.3 + DeepGEMM 源码），主题属确定性/可靠性问题域而非后训练框架，与 `10_determinism_and_numerical_reliability_analysis.md` 问题 2（训推数值不一致 / batch 不变性）互为系统侧上游与算子层实现细化的关系，双向补链；`tools/batch_invariance_demo.py` 引用路径为仓库根相对路径文本，两个新旧目录深度相同（均为 `wiki/02_engineering/<domain>/`），无需改写。

**入链改写（裸基名）**：`RL_PPO_Loss_and_GRPO_Analysis` → `rl_ppo_loss_and_grpo_analysis`，涉及 `23_glm5_posttraining_deepdive.md`（2 处）、`tim_causal_chain_analysis.md`、`rl_infra_efficiency_analysis.md`、`rl_sandbox_design_analysis.md`、`10_determinism_and_numerical_reliability_analysis.md`、`07_training_reliability/index.md`、`wiki/index.md`。`20_batch_invariance_guide` 基名不变（同名文件仅换目录），裸基名链接天然不受影响；唯一一处路径限定链接 `01_ai_frameworks/index.md` 的 `[[04_posttrain_frameworks/20_batch_invariance_guide]]` 改为裸基名 `[[20_batch_invariance_guide]]`（同域内唯一同名文件，不存在歧义）。`wiki/changelog.md` 中 1 处 2026-05-24 历史活链接（`RL_PPO_Loss_and_GRPO_Analysis`）按"历史不回写"惯例降级为反引号 + 去向说明。

**索引同步**：`01_theory/04_posttraining/index.md` 移除 RL_PPO 行；`02_engineering/04_posttrain_frameworks/index.md` 「数值与确定性」小节（原仅 20_batch_invariance_guide 一行）改为「RL 算法源码实现」小节收纳新迁入的 `rl_ppo_loss_and_grpo_analysis`，并注明 20_batch_invariance_guide 去向；`07_training_reliability/index.md` 问题地图第 2 行「详见」列补 `[[20_batch_invariance_guide]]`，新增「第四篇：batch 不变性算子实现」小节介绍其归位背景与来源（独立于本域原 wanka 综述素材）。

**验收**：`tools/check_links.py` broken=0、orphans=0（pages=375，与基线一致，纯搬运不增删文件）；`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P5 Task 5（verl 端到端整合，双基线调和）

**Type**: Content Consolidation + Baseline Reconciliation（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 5）

`git mv wiki/03_posttraining/07_verl_end_to_end_iteration_analysis.md` → `wiki/02_engineering/04_posttrain_frameworks/verl/verl_end_to_end_iteration_analysis.md`（D07，基线 verl `983cb0f`，225 行）。verl 域此前已有 `verl_ray_trainer_analysis.md`（354 行，基线 `8a694930`）逐方法源码走读 `RayPPOTrainer.fit`；两页覆盖同一主题但基线不同、深度不同，本次逐节台账后**不合并、保留双页**并调和。

**与 `verl_ray_trainer_analysis.md` 逐节台账**：重叠区（fit 主循环、一步 PPO 字段流转、advantage/loss 数学）以 D07 为准，不回填 ray_trainer 的逐行细节（D07 §4/§5 的高层表已经是权威账本）。ray_trainer §2（Role 枚举/`ResourcePoolManager`/`init_workers` 的 `create_colocated_worker_cls`/`WorkerDict` colocate 机制）、§3–§6（fit() 逐行追踪、`_balance_batch`、时序图、dispatch 表）在 `983cb0f` 本地 checkout（`E:\97-codes\torch_parallel\verl`）核对后确认为**独有、且体量远超 40%**（全篇 354 行里仅"fit 主循环顺序/advantage 数学"这类主题级重叠，method 级源码走读几乎全部独有）——判定**保留残页**：`verl_ray_trainer_analysis.md` 不删除，作为 `8a694930` legacy 深潜companion 页保留;仅将其中直接回答 D07 §2 自身"colocate or disaggregate"清单的**核心机制结论**（Role 枚举、`need_critic`/`need_reference_policy` 判定、`create_colocated_worker_cls` 合并 WorkerDict 机制、`ref_in_actor`）逐字并入 D07 §2，全部按 `983cb0f` 重新核对行号（utils.py:27-107、single_controller/ray/base.py:185-233、ray_trainer.py:343-360,772-907，与 `8a694930` 相应位置逐一比对，多数行号完全一致，`fit()` 因 `983cb0f` 新增代码整体下移 21 行）。

**基线冲突（`[!contradiction]` 双记）**：核对 `main_ppo.py` 发现两基线间存在真实机制反转——`trainer.use_v1` 默认值从 `8a694930` 的 `false`（legacy `RayPPOTrainer.fit` 为默认执行路径）反转为 `983cb0f` 的 `true`（`config/ppo_trainer.yaml:201`→`:219`；`main_ppo.py:184-193` 默认改道 `TaskRunnerV1`/TransferQueue）。这不是页面撰写错误而是版本演进，按规程在 D07 §1、`verl_ray_trainer_analysis.md` §1 版本定位note、`verl/index.md`「HEAD 架构演进提示」三处以 `[!contradiction]` 双记：`RayPPOTrainer.fit`（本系列 9+1 篇文档共同的教学主链）在 `983cb0f` 已非默认路径，需显式 `trainer.use_v1=false` 才会执行；`TaskRunnerV1`/TransferQueue 路径本知识库尚无覆盖。`@deprecated` 装饰器本身两基线间未变（均在 `ray_trainer.py:285`）。

**D07 §3/§6 收缩（先验两专页覆盖，Task 4 同款流程）**：`verl_dataproto_analysis.md`（325 行）与 `verl_rollout_resharding_analysis.md`（347 行）均已全面覆盖 D07 §3/§6 的全部事实（前者到方法级，后者到 CUDA IPC bucket/CheckpointEngine 两条路径的机制级），未发现 D07 独有细节。§3（DataProto）57→70 行原文压缩为「容器契约表 + 四条不变量」+ `[[verl_dataproto_analysis]]` 链接；§6（权重刷新）保留原有 `983cb0f` 专属行号（`engine_workers.py:705-725,783-787`、`vllm_rollout.py:271-320,278`）改写为时序代码块，补 `[[verl_rollout_resharding_analysis]]` 链接。

**verl 域其余 8 篇（`verl_architecture_overview_analysis`/`verl_quickstart_guide`/`verl_single_controller_analysis`/`verl_dataproto_analysis`/`verl_workers_engine_analysis`/`verl_rollout_resharding_analysis`/`verl_rl_algorithms_analysis`/`verl_optimization_analysis`）页头加基线横幅**：``> [!note] 本页基线 verl `8a694930`；端到端迭代以 [[verl_end_to_end_iteration_analysis]]（基线 `983cb0f`）为准，两基线间机制差异以新基线页为先。``；`verl_ray_trainer_analysis.md`（D07 对手方，保留残页）同样加此横幅，另加上述 `[!contradiction]` 版本反转记录。

**verl/index.md 重建**：新增「端到端主链（当前基线）」表段承接 D07；「由浅入深三层」「深挖实现」「算法与优化」三表标注基线 `8a694930`；「五条平面」入口/驱动行、「经典 RL 数据流」承接句、Related Pages 均补 D07 双链接；「HEAD 架构演进提示」callout 补 `use_v1` 反转记录；页数 9→10；原对 D07 的 `[[03_posttraining/07_verl_end_to_end_iteration_analysis]]` 引用改裸基名 `[[verl_end_to_end_iteration_analysis]]`。`04_posttrain_frameworks/index.md` 子目录表 verl 行页数同步改「9 篇 8a694930 深潜 + 端到端主链页基线 983cb0f」。

**入链改写（裸基名）**：`03_posttraining/07_verl_end_to_end_iteration_analysis` → `verl_end_to_end_iteration_analysis`，涉及 `posttraining_frontier_map_analysis.md`、`cuda_ascend_posttraining_stack_comparison.md`、`posttraining_infra_mechanism_analysis.md`、`rl_framework_comparison.md`（阅读导航+正文两处）、`slime_architecture_analysis.md`（阅读导航+正文两处）、`00_posttraining_source_reading_guide.md`、`03_posttraining/index.md`、`wiki/index.md`。D07 自身阅读导航（`[[rl_framework_comparison|上一篇 D06]]` · `[[slime_architecture_analysis|下一篇 D08]]`）此前已是裸基名，无需改写。

**验收**：`tools/check_links.py` broken=0、orphans=0（pages=375，与基线一致，本次只搬运不新增/删除文件）；`python -m pytest -q` 77 passed。

**自查**：verl 本地 git checkout（`E:\97-codes\torch_parallel\verl`）同时含 `983cb0f24443f87b3d161fad318445130a620b0` 与 `8a694930` 两个 commit，本次全部行号引用与「机制未变/已反转」判断均用 `git show <sha>:<path>` 现场核对，未依赖两页文本互证兜底。ray_trainer 页保留判定基于逐节主题映射后的独有内容占比估算（远超 40% 阈值），非精确计数；D07 §2 新增内容与 D07 §3/§6 收缩前均逐一核对对应专页/源码覆盖,未发现事实丢失。`grep -rn '03_posttraining/07_verl_end_to_end_iteration_analysis' wiki/` 除 `changelog.md`（历史记录，规则豁免不回写）与 `verl/index.md` 的一处历史路径纯文本提及（非 wiki-link）外 0 处活链接命中。

---

## 2026-07-31：知识库结构整改 P5 Task 4（D05/D06 迁入收缩 + weight sync 三方划界）

**Type**: Content Consolidation + Boundary Linking（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 4）

`git mv wiki/03_posttraining/05_posttraining_infra_mechanism_analysis.md` → `wiki/02_engineering/04_posttrain_frameworks/posttraining_infra_mechanism_analysis.md`；`git mv wiki/03_posttraining/06_framework_comparison.md` → `wiki/02_engineering/04_posttrain_frameworks/rl_framework_comparison.md`（comparison 类型命名）。两文件标题保留 D05/D06 前缀（沿用 D01–D04 迁移惯例），页头阅读导航与内部互链改裸基名。

**D05 §7（Reward/Environment/Sandbox）逐句对照 `rl_sandbox_design_analysis.md`（190 行）**：两页实为不同来源（D05 引 Kimi K3 Technical Report `0797decb`；rl_sandbox 引 RollArt/ProRL/Anthropic），逐句核对后**未发现字面重复**——reward-as-service 契约字段与「需要防」六条威胁清单是 D05 独有的三平面接口定义，予以保留；K3 white-box harness 版本化段落与 Fork/Pause/Resume/Snapshot 语义段落判定为「生产 sandbox 实现细节」而非「三平面接口」，逐字回流至 `rl_sandbox_design_analysis.md` 新增 §2.1「K3 案例：harness 版本化与故障恢复语义」（同步补 `关键资料`/`入库日期`/`Related Pages`）。D05 §7 更名「...的接口视角」，末尾改为指向 §2.1 的一句链接。262 行 → 该节从 26 行压缩到约 17 行，无事实删除。

**D05 §4（数据面 backpressure）逐句对照 `rl_infra_efficiency_analysis.md`（301 行）**：同样未发现字面重复——buffer schema 与四层容量定义（并发/staleness/内存/状态）是 D05 独有的 data plane 接口定义，予以保留；AReaL `StalenessManager` 源码引用（`areal/infra/staleness_manager.py:80-112`）与 K3 cache-pressure-aware admission 信号组合段落判定为「准入控制机制实现」，逐字回流至 `rl_infra_efficiency_analysis.md` 新增「优化 6: Admission-aware Backpressure」（`## 2. 五个核心优化`→`## 2. 核心优化`，因新增第 6 项去掉"五个"；同步补 `关键资料`/`入库日期`/`Related Pages`）。D05 §4 末尾改为指向该节的一句链接。

**D06 §4.1（verl 控制面）压缩**：先验 `verl_architecture_overview_analysis.md`（229 行）覆盖情况——`RayPPOTrainer.fit` 主循环（其 §6）、`DataProto` 贯穿角色边界（其 §3）均逐字覆盖；"算法 registry 含 GRPO/GSPO/SAPO/CISPO/REINFORCE"中的 SAPO 一项在该页未直接出现，改核 `verl_rl_algorithms_analysis.md`（已含 SAPO 注册表条目）覆盖，判定为全覆盖。原「优点/代价」两段散文压缩为一个二行摘要表 + 双链接（`verl_architecture_overview_analysis` 架构总览 + `verl_rl_algorithms_analysis` registry 机制），4.2/4.3/4.4（slime/AReaL/ROLL）不动。反向在两篇 verl 页 Related Pages 补 D06 回链。

**weight sync 三方划界（只补链不合并，三页正文除本条声明外不动）**：`02_train_frameworks/megatron-lm/30_megatron_rl_posttraining_consistency_analysis.md`（207 行,Megatron 训练侧 refit/训推一致性）、`33_megatron_vllm_weight_sync_analysis.md`（182 行,verl 在 Megatron+vLLM 场景下的 Gather-Broadcast-Load 同步实现）与 D05 §6（weight publish 三平面协议）互相在页头/相应节插入同一句「三方分工」声明：D05=三平面机制视角（框架无关）；megatron 两页=训练侧/verl 具体实现；`verl_rollout_resharding_analysis`=verl 自身 resharding/3D-HybridEngine 实现（该页仅在声明句中提及，未改动）。三页 Related Pages 互补链接。

**入链改写**：`03_posttraining/05_posttraining_infra_mechanism_analysis`→`posttraining_infra_mechanism_analysis`、`03_posttraining/06_framework_comparison`→`rl_framework_comparison` 两个路径的全部 wiki-link 目标改为裸基名，涉及 `wiki/index.md`、`03_posttraining/index.md`、`00_posttraining_source_reading_guide.md`、`posttraining_frontier_map_analysis.md`、`on_policy_off_policy_staleness_analysis.md`、`agentic_rl_algorithm_analysis.md`、`dapo_analysis.md`、`24_kimi_k3_posttraining_case_study_analysis.md`、`07_verl_end_to_end_iteration_analysis.md`、`verl/index.md`、`slime_architecture_analysis.md`、`roll_strategy_and_ascend_analysis.md`、两文件自身的阅读导航与 Related Pages。`03_posttraining/index.md` D05/D06 两行链接目标随批量改写同步生效。`04_posttrain_frameworks/index.md` 新增「后训练框架源码对照」表两行（posttraining_infra_mechanism_analysis / rl_framework_comparison），迁入来源脚注 D08–D11 改为 D05–D11，`Coding RL Sandbox 与 Infra` 表两行补来源(Kimi K3/AReaL)与主题(K3 案例/admission-aware backpressure)。

**验收**：`tools/check_links.py` broken=0（pages=375，与基线一致）；`python -m pytest -q` 77 passed。

**自查**：D05 §4/§7、D06 §4.1 三处压缩均先做逐句/逐段对照，实测无字面重复，因此本次"收缩"不含事实删除——D05 两节的所有独有事实全部逐字迁移到承接页（rl_sandbox_design_analysis §2.1、rl_infra_efficiency_analysis「优化 6」）并补溯源，D06 §4.1 的收缩基于承接页已覆盖的验证结论（含 SAPO 单项二次核实）。weight sync 三方划界未改动任何一页的既有分析正文，只加边界声明与互链。`grep -rn '03_posttraining/05_posttraining_infra_mechanism_analysis\|03_posttraining/06_framework_comparison' wiki/` 0 处活链接命中。

---

## 2026-07-31：知识库结构整改 P5 Task 3（D02 演进权威页 + GRPO 三写归一）

**Type**: Content Consolidation（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 3）

`git mv wiki/03_posttraining/02_reasoning_rl_algorithm_evolution_analysis.md` → `wiki/01_theory/04_posttraining/reasoning_rl_algorithm_evolution_analysis.md`，定为 GRPO/DAPO/Dr.GRPO/GSPO/SAO 公式演进与工程语义的**统一权威页**（页头新增"定位"行）。§3.6 K3-MOPD 收缩为一句 + [[24_kimi_k3_posttraining_case_study_analysis|D12]] 链接：先核实 D12 §1/§2.2/§4 已逐字覆盖 §3.6 的 MOPD 公式（Eq. 15 与 §3.6 完全一致）、reasoning-effort 预算约束公式与"top-k 蒸馏无明确优势"结论，无独有事实需回流。205→187 行。

**GRPO/DAPO/GSPO 三篇论文页瘦身**（逐节台账，收缩前均在 D02 找到实际对应段；D02 未给出的公式变体一律保留）：

| 页面 | 收缩(D02 对应) | 保留(论文特有/D02 未给出的公式) | 行数 |
|---|---|---|---|
| `grpo_analysis` | Key Innovation 段落(D02 §3.1 $\hat A_i$ 公式)、GRPO Objective 的 clip-surrogate 结构(D02 §2)、Why GRPO Works Well 动机叙述(D02 §1/§3.1)、Practical Implementation 玩具伪代码(改指 verl `core_algos.py:268/1279` 真实代码锚点) | 论文元数据、KL 低方差无偏估计量(D02 未给)、DeepSeek-R1-Zero 训练配置表/涌现行为/性能、R1 全流程、GRPO vs DPO 对比表、Impact | 165→119 |
| `dapo_analysis` | Clip-Higher 与 Dynamic Sampling 两段动机叙述(公式与 D02 §2/§3.2 重复)、Relationship to Other Methods 表(与 D02 §4 重复) | 论文元数据、"30 分失败三症状"诊断段、Token-Level Loss 的 $J_{\mathrm{DAPO}}$/$J_{\mathrm{GRPO}}$ 显式求和公式(D02 无)、Overlong Reward Shaping 分段惩罚公式(D02 无)、DAPO Algorithm 伪代码、Training Configuration/Progressive Results 两张原始数字表、KL 移除动机、数据集细节、Key Insights | 187→157 |
| `gspo_analysis` | Problem 段 5 点缺陷列表压缩为一段(D02 §3.4)、Sequence-Level Ratio 公式与 GSPO Objective(D02 §3.4/§2 完全一致)、Key Difference 表与 Relationship to Other Methods 表(均与 D02 §4 重复) | 论文元数据、Gradient Comparison 的 $\nabla J$ 显式梯度公式(D02 无)、GSPO-token stop-gradient 变体公式(D02 无)、Clipping Range Difference 原始超参数表(3e-4/4e-4)、Empirical Results 原始实验数据、Why GSPO Matters | 133→88 |

**verl `verl_rl_algorithms_analysis.md` §3.2/§4.1/§4.2 数学部分收缩指 D02**（保留全部代码锚点、14 优势估计器/11 策略损失清单、注册表机制、config key 映射——spec 点名保护项）：GRPO 组内归一化公式（§3.2）、vanilla PPO clip 基础结构（§4.1，dual-clip 扩展因 D02 未给出而保留）、GSPO 序列比公式（§4.2，stop-grad 实现技巧因 D02 未给出而保留）分别收缩为指向 D02 §3.1/§2/§3.4 的一句话，`core_algos.py:xxx` 代码块与行号全部原样保留。§7"与 RL 文献的对应"由散文列表改写为"verl 选型→文献→D02 对应→论文页"表格（轻改，DAPO"在 verl 里不是新损失"的实现洞察保留为表内备注；表中特别标注 verl `sapo`(arXiv 2511.20347)与 D02 §3.5 的 SAO(arXiv 2607.07508)是不同算法，避免同名混淆）。389→382 行（净减 7 行：删除 3 处重复 LaTeX 展示块，新增 §7 对应表 + Related Pages 补链）。

**入链改写**：全库 12 处 `[[03_posttraining/02_reasoning_rl_algorithm_evolution_analysis...]]` 目标改为裸基名 `[[reasoning_rl_algorithm_evolution_analysis]]`，涉及 `wiki/index.md`、`03_posttraining/index.md`、`00_posttraining_source_reading_guide.md`、`agentic_rl_algorithm_analysis.md`(D03 阅读导航+Related Pages)、`posttraining_frontier_map_analysis.md`(D01 阅读导航+文档顺序表)、`on_policy_off_policy_staleness_analysis.md`(D04 Related Pages)、`24_kimi_k3_posttraining_case_study_analysis.md`(D12 Related Pages)。D02 自身页内指向 grpo/dapo/gspo 的 3 处路径限定链接改裸基名（同目录）。`01_theory/04_posttraining/dapo_analysis.md` 中指向仍在 `03_posttraining/` 的 D05 链接（Task 4 才迁移）保持路径限定不变。

**Index 更新**：`01_theory/04_posttraining/index.md` "GRPO 系列"表前新增定位说明（D02=权威页，四篇论文页=元数据/实验数字档案）；"后训练前沿整合"表插入 D02 行（D01→D02→D03→D04 顺序），迁入来源脚注补 D02。`03_posttraining/index.md` D02 行改指新路径（索引本身按计划保留到 Task 7 删除）。

**验收**：`tools/check_links.py` broken=0（pages=375，与 P4/Task 2 基线一致；ambiguous=69/bare_index=69 均为存量未变）；`python -m pytest -q` 77 passed。

**自查**：`grep -rn '\[\[03_posttraining/02_reasoning_rl_algorithm_evolution_analysis' wiki/` 0 处活链接命中（唯一残留的 3 处字符串是本条目自身的反引号内说明文字，非 `[[wikilink]]`，`check_links.py` 的行内代码豁免规则不计入 broken）；本次改动前该旧路径从未出现在此前的历史 changelog 条目中，无需"历史不回写"降级处理。

---

## 2026-07-31：知识库结构整改 P5 Task 2（后训练三域整合，纯迁移批 D01/D03/D04/D08–D12）

**Type**: Pure Migration（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 2）

`wiki/03_posttraining/` 解散工作的开局批次，把 8 篇纵向学习域文档 `git mv` 到功能树对应位置并去编号前缀：

| 旧路径 | 新路径 |
|---|---|
| `03_posttraining/01_posttraining_frontier_map_analysis.md` | `01_theory/04_posttraining/posttraining_frontier_map_analysis.md` |
| `03_posttraining/03_agentic_rl_algorithm_analysis.md` | `01_theory/04_posttraining/agentic_rl_algorithm_analysis.md` |
| `03_posttraining/04_on_policy_off_policy_staleness_analysis.md` | `01_theory/04_posttraining/on_policy_off_policy_staleness_analysis.md` |
| `03_posttraining/08_slime_architecture_analysis.md` | `02_engineering/04_posttrain_frameworks/slime_architecture_analysis.md` |
| `03_posttraining/09_areal_async_architecture_analysis.md` | `02_engineering/04_posttrain_frameworks/areal_async_architecture_analysis.md` |
| `03_posttraining/10_roll_strategy_and_ascend_analysis.md` | `02_engineering/04_posttrain_frameworks/roll_strategy_and_ascend_analysis.md` |
| `03_posttraining/11_cuda_ascend_posttraining_stack_comparison.md` | `02_engineering/04_posttrain_frameworks/cuda_ascend_posttraining_stack_comparison.md` |
| `03_posttraining/12_kimi_k3_posttraining_case_study_analysis.md` | `01_theory/01_models/moonshot_kimi/24_kimi_k3_posttraining_case_study_analysis.md` |

正文零改动（仅两处例外，见下）。**入链改写**：全库 109 处 `[[...]]` 目标从旧编号基名/`03_posttraining/NN_...` 路径限定形式改为新裸基名，涉及 26 个文件（含 `wiki/index.md`、`wiki/03_posttraining/` 内未迁移的 D00/D02/D05/D06/D07/index、moonshot_kimi 五篇 K3 页、`grpo_analysis`/`gspo_analysis`、`rl_infra_efficiency_analysis`/`rl_sandbox_design_analysis`/`verl/index`）；`03_posttraining/index.md` 与 `00_posttraining_source_reading_guide.md` 中指向这 8 篇的行同步改指新位置，但索引本身按计划保留到 Task 7 才删除。`wiki/changelog.md` 中 3 处写入当时的历史活链接（本文件之前记载 D01/D12 新增的条目）按"历史不回写"惯例降级为反引号 + 去向说明，不当作活链接维护。

**例外 1（spec 点名的良性分层，只补链不动正文）**：`on_policy_off_policy_staleness_analysis.md`（D04）§7 TIM 小节新增一句指向 `tim_causal_chain_analysis`（历史活链接，已于 2026-07-31 因 kb-reorg P5 Task 8 再编号为 [[26_tim_causal_chain_analysis]]，按"历史不回写"惯例降级为反引号）的速览提示；`tim_causal_chain_analysis.md` Related Pages 反向补一条指向 `on_policy_off_policy_staleness_analysis` 的链接，说明其覆盖"TIM 与 staleness/off-policy 关系"这一上层概念坐标。

**例外 2（三个承接目录 index 补条目）**：`01_theory/04_posttraining/index.md` 新增"后训练前沿整合"小节（3 条：posttraining_frontier_map/agentic_rl_algorithm/on_policy_off_policy_staleness）；`02_engineering/04_posttrain_frameworks/index.md` 新增"后训练框架源码对照"小节（4 条：slime/areal/roll/cuda_ascend_stack）；`01_theory/01_models/moonshot_kimi/index.md` 在既有 K3 报告行后新增一条专属条目指向 `24_kimi_k3_posttraining_case_study_analysis`。

**验收**：`tools/check_links.py` broken=0（pages=375，与 P4 收官基线一致）；`python -m pytest tools/ -q` 77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 10（课程页化 + 19 号目录解散 + 全 index 终校，P4 收官）

**Type**: Course-page Consolidation + Directory Removal + Index Audit（设计：
`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 10；
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §6 课程页规则）

**课程页化**：新建 `wiki/courses/torch_compile_end_to_end.md`（spec §6 定义的纯导读页——只含
阅读顺序、链接与一句话导读，不承载正文）：开篇(是什么/前置/三段式流水线速览) → 按目录顺序
(0 前置 eager_runtime → 1 Dynamo → 2 AOTAutograd → 3 Graph IR/Passes → 4 Inductor(内含五阶段
一览表) → 5 编译缓存 → 6 调试诊断 → 7 CUDA Graphs → 8 图导出/算子扩展 → 9 分布式原语 → 10
训练扩展收尾)、段位递进(0→1→2→3)排布的阅读路线表(链接全用编号后新基名，一句话导读复用各
目录自己 index.md 已有的表述) → 三条捷径 → labs 使用说明(六卷 demo 入口表 + 运行契约 + PASS/
BLOCKED/FAIL 语义) → 与功能树的关系说明(courses/ 是纯索引层，功能树为唯一权威)。

**删除清单**（导读/动机价值并入课程页后删除源文件，`git rm`）：

| 文件 | 行数 | 去向 |
|---|---|---|
| `19_torch_compile_end_to_end/00_torch_compile_end_to_end_index.md` | 221 | A-F 六卷学习路径故事线 → 课程页开篇 + 阅读路线表 |
| `19_torch_compile_end_to_end/00_pytorch_graph_series_index.md` | 302 | 卷 C Part I-IV 结构 → 课程页 §3 Graph IR / §4 Inductor 分节 |
| `19_torch_compile_end_to_end/01_graph_ir_motivation_and_taxonomy.md`（C01） | 372 | "为什么需要图 IR"动机段 → 课程页 §3 末尾导读级要点；逐项核对确认全篇技术细节（autograd Edge、FX Node opcode、AOT partition、Inductor lowering、Scheduler 工厂、torch.cond HOP、`Graph.lint`、复杂度分析、reverse-topological order）均已被 `03_graph_ir_and_passes`/`01_eager_runtime/05_autograd_engine` 对应深潜页独立引用同一批源码定位，无需逐字搬运 |
| `02_compile_stack/04_inductor/02_torch_compile_architecture.md`（overview 四写之一） | 151 | 五阶段全景表 → 课程页 §4 Inductor 小节；`04_inductor/index.md` 补一段简短 is-what/why 段落承接 overview 角色，不留空洞 |

`19_torch_compile_end_to_end/index.md`（孤儿页）随目录一并 `git rm -r` 删除，19 号目录清空。

**入链修复**：一次性脚本批量重写全库 127 处 `[[...]]` 目标（94 个文件），全部改指
`courses/torch_compile_end_to_end` 或 `02_compile_stack/04_inductor/index`；随后逐文件人工核查
消除批量替换产生的相邻重复条目（约 12 处，如同一 Related Pages 列表里两个旧索引坍缩成同一个
课程页链接）；`01_ai_frameworks/index.md` 删除"过渡期"19 号指向块，课程入口改指
`[[courses/torch_compile_end_to_end|torch.compile 端到端课程]]`；`wiki/index.md` 新增 courses
入口小节。changelog 历史条目中 4 处指向已删除页的活链接按"历史不回写"惯例降级为惰性反引号 +
去向说明（不计入上方 127 处）。

**pytest 侧修复**：`tools/labs_torch_compile/demo_manifest.json` 的 `c01` 条目 `page` 字段改指
`torch_compile_end_to_end.md`；`test_volume_demo_contract.py` 的 `_page_root()` 为 `volume=="C" and
page_id=="c01"` 特判路由到 `wiki/courses/`，末尾兜底分支从"回退到 19 号目录"改为
`raise AssertionError`（19 号目录已删除，理应不可达）；`CourseMarkdownContractTest._course_pages()`
的 `course_root` glob 目标同步改为 `wiki/courses`。labs README/`NATIVE_BACKEND_RUNBOOK.md` 对
`00_pytorch_graph_series_index` 的 2 处引用同步改指课程页。

**遗留小修**（上一任务审查移交的 5 处旧基名/层次标注遗留，均为反引号/prose 引用，未被
`check_links.py` 覆盖，故未在此前批量修复中被发现）：`04_inductor/index.md` 表格单元格 3 个旧
基名补前缀；`15_inductor_compile_fx_orchestration_analysis.md`"最后更新"注里 7 个旧基名补前缀/
改指现存后继页（`scheduler_analysis` 已随 Task 8 判重删除，改指其内容实际归宿
`13_scheduler_dependency_graph_fusion_and_ordering_analysis`）；`npu/30_npu_vs_upstream_fusion_passes.md`
的 `npu_compile_paths_overview.md` 引用补 `01_` 前缀；`npu/32_npu_debug_guide.md` 页头"层次"由
"quick start"改为"方法论/排查实践(段 3)"，与 Task 9.5 changelog 记载的编号判段结论对齐（该页头
此前遗漏同步）。

**全 index 终校**：`01_ai_frameworks/index.md` 五层表填平 Task 7/Task 4 完成后仍标"待填充"的
Graph IR/Passes 与调试诊断两格；`02_compile_stack/index.md`、`04_inductor/index.md`（Task 8 自查
点名重点通读，确认无遗留问题）及其余 26 个模块/层 index 逐一核对：脚本核实每个 index.md 均已
提及其目录下全部同级 `.md` 文件（0 处缺失）；`04_inductor/index.md`、`02_aot_autograd/index.md`、
`01_fx_export_extensibility/index.md` 三个当日实际编辑过内容的模块 index 补"最后更新"日期至
2026-07-30；`01_eager_runtime/index.md`/`02_compile_stack/index.md`/`04_export_and_distributed/index.md`
三个 Task 2 建立的架构层壳 index 确认按设计无日期字段，非遗漏。`wiki/index.md`：AI框架页数按
`find wiki/02_engineering/01_ai_frameworks -name "*.md" ! -name SUPERSEDED.md` 实数重算，
174→**150**（子域 TorchInductor 39→36、运行时图 10→12，均为 Task 3-9.5 期间累积漂移的一次性
对账，非本任务新增改动）；统计日期同步到 2026-07-30。

**自查披露**：`grep -rn "19_torch_compile_end_to_end" wiki/ --include="*.md" | grep -v changelog`
非零命中——8 处，均为 Task 3（A 卷迁移）遗留的历史出处脚注（形如"本节内容原属 P4 知识库整改
被删除的 A 卷回顾页(`19_torch_compile_end_to_end/aXX_....md`)"），在反引号内、非 `[[wikilink]]`、
明确注明"已删除"，起 changelog 同等的溯源作用；均早于本任务存在（`git show e5cc60a` 可验证），
本任务未新增此类残留，按惯例不追溯改写，留痕供控制者复核。

**P4 阶段小结**（Task 1-10 全阶段，本任务收官）：`01_ai_frameworks` 从 18 个平铺目录重组为 5
个架构层两级目录；`19_torch_compile_end_to_end`（63 篇/2.05 万行）经 A→E→B→D→C→F 六卷逐批
判重解散，独有内容验证后并入功能树对应页，课程导读价值最终归一到单一 `courses/` 纯导读页；
`checker` 全程 `broken=0`（Task 1 基线 pages=398 → 本任务终态 pages=375，净 -23 主要来自判重
删除的重复大文与本任务的 4 页课程化收口，非内容流失）；`pytest tools/` 全程 77 passed。

**校验**：`python tools/check_links.py`：pages=375，broken=0，ambiguous=69，bare_index=69，
orphans=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 9.5（目录内分段编号，用户追加需求）

**Type**: Naming Convention（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 9.5、
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §5）

对 `01_ai_frameworks` 下 19 个已定型模块目录（含 2 个硬件子目录）逐目录施行两位数字段位前缀命名：
段 0（01-09）入门/导览、段 1（10-19）核心机制主线（按该子系统执行流水线/依赖顺序排列）、段 2
（20-29）深潜/专题、段 3（30-39）方法论/对照/工程实践；`index.md` 不编号，仅在其中新增"段位与
阅读顺序"小节定段位表。纯改名，不改内容；跨 8 个 commit 分批完成，每批 checker broken=0 +
pytest 77 后提交：

| 目录 | 篇数 | 段位分布 | 编号区间 |
|---|---|---|---|
| `01_eager_runtime/01_tensor_and_storage` | 2 | 0段1/1段1 | 01,10 |
| `01_eager_runtime/02_dispatcher_and_device` | 5 | 0段1/1段2/2段2 | 01,10-11,20-21 |
| `01_eager_runtime/03_op_registration` | 0（未编号） | — | 顶层 0 篇内容页；`npu/` 3 篇低于 ≥4 递归阈值，两者均不编号 |
| `01_eager_runtime/04_aten_op_execution` | 2 | 0段1/1段1 | 01,10 |
| `01_eager_runtime/05_autograd_engine` | 3 | 0段1/1段1/2段1 | 01,10,20 |
| `01_eager_runtime/06_nn_module_system` | 2 | 0段1/1段1 | 01,10 |
| `01_eager_runtime/07_memory_amp_profiler` | 2 | 0段1/1段1 | 01,10 |
| `02_compile_stack/01_dynamo` | 14 | 0段1/1段9/2段3/3段1 | 01,10-18,20-22,30 |
| `02_compile_stack/02_aot_autograd` | 6 | 0段1/1段4/2段1 | 01,10-13,20 |
| `02_compile_stack/03_graph_ir_and_passes` | 12 | 1段6/2段6 | 10-15,20-25 |
| `02_compile_stack/04_inductor` | 23 | 0段2/1段6/2段9/3段6 | 01-02,10-15,20-28,30-35 |
| `02_compile_stack/04_inductor/npu` | 12 | 0段1/1段3/2段5/3段3 | 01,10-12,20-24,30-32 |
| `02_compile_stack/05_codegen_backends/mlir` | 4 | 0段1/1段2/3段1 | 01,10-11,30 |
| `02_compile_stack/05_codegen_backends/mlir/npu` | 0（未编号） | — | 2 篇低于 ≥4 递归阈值，不编号 |
| `02_compile_stack/06_compile_cache` | 4 | 1段4 | 10-13 |
| `02_compile_stack/07_debugging` | 10 | 1段10 | 10-19（见下方说明） |
| `03_runtime_graphs/cuda` | 3 | 0段1/1段1/2段1 | 01,10,20 |
| `03_runtime_graphs/npu` | 6 | 0段1/1段2/2段2/3段1 | 01,10-11,20-21,30 |
| `04_export_and_distributed/01_fx_export_extensibility` | 3 | 0段1/1段1/2段1 | 01,10,20 |
| `04_export_and_distributed/02_distributed_primitives` | 4 | 0段1/1段1/2段2 | 01,10,20-21 |
| `05_other_frameworks` | 1 | 1段1 | 10 |

共 118 篇内容页改名，全库改写裸基名 `[[wikilink]]` 链接约 1570 处（含 labs `demo_manifest.json` 的 `page`
字段与 `test_volume_demo_contract.py` 硬编码文件名字面量同步，涉及原 B/C/D/E/F 卷 page_id）。

**判段说明**（内容实质优先于文件名后缀/既有体裁标签）：

- `01_eager_runtime/04_aten_op_execution/adding_an_aten_operator_guide.md`：`_guide` 后缀但内容
  是 native_functions.yaml 速查实操，按内容实质入段 0（非默认的段 2/3）。
- `02_compile_stack/04_inductor/npu/npu_compile.md`、`.../npu_debug_guide.md`：目录旧三层
  （overview/quick start/deep dive）均标两者为 quick start；按内容实质改判——`npu_compile`
  是编译工作流/Autotune/精度校验等机制性叙述入段 1，`npu_debug_guide` 是排查方法论入段 3。
- `03_runtime_graphs/npu/npugraphs_make_graphed_callables_deep_dive.md`：旧体裁标 quick start，
  内容是窄 API 的六阶段实现级深挖，按内容实质改判段 2。
- `02_compile_stack/07_debugging`：十篇不做 quickstart/深潜/方法论层级切分——本目录的"核心机制
  主线"就是排查工作流本身，全部落段 1（10-19）恰好填满整段。编号顺序采用 index.md 原有的
  "建议阅读顺序"，而非旧卷内编号 D07/E01-E09 顺序：`14_compiled_artifact_lifecycle`（原 D07）
  页头仍标其原 D 卷前置为 `cudagraph_trees_...`、且被 `10_observability` 页头引用为"前置"，但
  在本目录实际教学顺序中排第 5 位（失败分层定位之后、minifier/bisector 工具之前）——是本次编号
  中唯一一处"页头前置字段"与"实际阅读顺序"不一致的目录，过程中发现后已改正（初次按 D07 在前
  的旧卷顺序命名，随即按 index.md 建议阅读顺序重排，同批提交前完成，无需回滚）。

**验证口径**：任务书给出的验证命令
`find wiki/02_engineering/01_ai_frameworks -name "*.md" ! -name "index.md" ! -path "*19_torch*" |
grep -v -E "/[0-9]{2}_"` 因按**完整路径**匹配（父目录如 `01_eager_runtime/`、`02_compile_stack/`
本身即带两位数字前缀），对本次未编号的 5 个文件（3 篇 `03_op_registration/npu/` + 2 篇
`05_codegen_backends/mlir/npu/`）不会显式列出——命令按此口径运行确为空。若改按**文件基名**匹配
（`basename | grep -v -E "^[0-9]{2}_"`），会精确列出这 5 个文件；它们是符合 spec §5"npu/cuda 子
目录页多（≥4 内容页）时递归适用，页少不编"规则的预期例外，非遗漏。

**校验**：`python tools/check_links.py`：pages=379，broken=0（8 个 commit 均独立核验）；
`pytest tools/ -q`：77 passed（8 个 commit 均独立核验）。

---

## 2026-07-30：知识库结构整改 P4 Task 9（19 号 F 卷分发，6 篇 1504 行）

**Type**: Volume Migration + Boundary Reconciliation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 9）

f01-f07（除已在 Task 6 随 D06 迁走的 f08）从 `19_torch_compile_end_to_end/` 分发进功能树，去
`fNN_` 前缀，六个子任务各一 commit：

| 篇 | 去向 | 处置 |
|---|---|---|
| f01（316 行）Compiled Autograd | `01_eager_runtime/05_autograd_engine/compiled_autograd_analysis.md` | 与 `autograd_engine_analysis.md`（481 行）互补划界：后者讲 eager 模式 `Engine::execute` 直接执行反向 DAG,本页讲同一引擎驱动 Python tracer 把运行时反向录制成 FX 图再编译——同一引擎的两种运行模式,无字面重复,两页页头各补显式 `[!note]` 分工声明 + 互指,eager 页新增 §12 |
| f02（226 行）Activation Checkpoint | `02_compile_stack/02_aot_autograd/activation_checkpoint_recompute_and_compile_analysis.md` | 与 C10（`saved_tensors_recompute_and_runtime_abi_analysis.md`,433 行）互指划界:本页站用户 API/策略层（`torch.utils.checkpoint`、Selective AC policy）,C10 站 partitioner 源码/runtime ABI 层,补齐 spec 附录 A 缺口 |
| f03（230 行）DDP Compile Boundaries + f04（221 行）FSDP/DTensor | `04_export_and_distributed/02_distributed_primitives/`（`ddp_compile_boundaries_and_optimizer_analysis.md`/`fsdp_dtensor_and_distributed_graphs_analysis.md`） | 与 `c10d_ddp_fsdp_dtensor_analysis.md`（433 行）三方互指划界:该页讲 DDP/FSDP/DTensor 原语本身（Reducer 分桶、FlatParameter shard/unshard、placement 传播）,零涉及 `torch.compile`;两篇讲原语与编译器相遇时新增的边界（DDPOptimizer bucket 切图、`use_orig_params=True`、Dynamo skip frame）;三页头各补 `[!note]`,索引新增"原语本身 vs compile 边界"小节 |
| f05（255 行）Custom Operator/Fake Kernel | `04_export_and_distributed/01_fx_export_extensibility/custom_operators_fake_kernels_and_decompositions_analysis.md` | 逐节判重 vs `fx_graph_export_and_custom_ops_analysis.md`（315 行）§7:该页 §7 是 FX/export/functorch 全景 deep dive 中约 40 行的 custom_op 注册概述（公开重导出入口）,本页是 13 节的"编译器边界契约"深度分析（fake kernel 正确性、mutation/version、decomposition/lowering/fallback 选择、失败定位分层）,独有内容占比远超 50%,判定**保留独立页**,不并入;仅在该页 §7 末补一条深度指路链接 |
| f06（260 行）Custom Backend/Device Integration | `01_eager_runtime/02_dispatcher_and_device/custom_backends_and_device_integration_analysis.md` | 三方划界 vs `privateuse1_device_integration_analysis.md`（278 行,eager/dispatcher 层设备接入,零涉及 compile,边界声明即可）+ `codegen_extension_guide.md`（247 行,Inductor codegen backend 注册实操指南,与本页 §5/§6/§8 有真实重叠）——**收缩重叠段**:§5/§6/§8 的 `register_backend_for_device`/`DeviceOpOverrides`/`BackendFeature` 具体注册细节改为指向该指南,保留本页独有的三层 backend 划分框架与 `DeviceInterface` |
| f07（292 行）AOTInductor 打包部署 | `02_compile_stack/04_inductor/aotinductor_packaging_and_deployment_analysis.md` | 纯平移,无判重需要 |

**B02 `use_aoti` todo 解答**（`backend_modes_options_stances_and_fullgraph_analysis.md` §14.2,
Task 5 遗留,原 `[!todo]`；**同日据 spec 复核修正**,见下方"复核修正"段）：读 F07 源码后对照
本地 pinned pytorch checkout 核实——`_TorchCompileAOTInductorWrapper.__call__` 最终调用与普通
JIT 编译相同的 `compile_fx(...)`（`torch/__init__.py:3016-3054` → 父类 `__call__`）；
`aoti_compile_and_package` 底层的 `compile_fx_aot`（`torch/_inductor/compile_fx.py:2221`）内部
同样调用 `compile_fx(...)`（`:2282-2284`）。两条路径在 `compile_fx` 内部汇合于同一段代码：
`V.aot_compilation` 为真时直接返回 `CompiledAOTI(filename=compiled_fn, device_type=graph.device_type)`
（`compile_fx.py:1849-1859`），是 `compile_fx_aot` 断言并解包 `.filename` 的同一个
`CompiledAOTI` 类型（`:2285-2289`）——两条路径汇合于同一套 AOT 代码生成/artifact 机制,但
`CompiledAOTI` 是否被真正装载成可调用 runner **不对称**：`__post_init__` 只有在
`config.enable_autograd_for_aot`（默认 `False`,`config.py:1696`）为真时才构造
`AOTIModelContainerRunner{Cuda,Xpu,Cpu}`（F07 §8 讲的同一个 `model_container_runner.cpp`
runner）并装进 `current_callable`；只有 use_aoti 路径会用 `config.patch("enable_autograd_for_aot", True)`
显式临时打开这个门（因为它必须马上把结果当 compiled callable 用），plain 的
export/package 路径不打这个 patch,`current_callable` 保持 `None`（但该路径本就只取
`.filename`，不依赖 `current_callable`）。差异还在前端捕获来源（Dynamo 运行时捕获 vs
`torch.export.export()` 产出的 `ExportedProgram`）与是否额外 `package_aoti` 打包成可跨进程
加载的 `.pt2` 归档。结论写入 B02 §14.2 note，双向加回链；旧 `[!todo]` 原文保留、标注已解答
（本文件"不回写活链接"惯例，`f07...` 引用降级为惰性反引号）。

**复核修正**（同日,spec 审查发现并修复,1 commit）：上一段与 B02 §14.2 note 最初表述为
"两条路径共享完全相同的 C ABI runner、`CompiledAOTI` 本身就是可直接调用对象"，忽略了
`enable_autograd_for_aot` 门控——已按上文改为不对称表述，并据此修正 F07/B02 两页互指
Related Pages 的措辞。另据复核，f06 迁移时 §5 收缩丢失了 `register_backend_for_device` 的
`device_custom_pass: CustomGraphModulePass`/`device_custom_config: ConfigModule` 两个可选
参数与其写入的独立全局注册表 `custom_backend_passes`/`custom_backend_codegen_configs`
事实（原引用 `common.py:389-418`/`:419-434` 经核精确）——已逐字取自 f06 迁移前版本
（`git show ffdff43:19_torch_compile_end_to_end/f06_...md` §5）落回
`codegen_extension_guide.md` §3 表格行 + §4 补充段；§6 被压缩掉的 `DeviceOpOverrides` 专项
方法名（`cpp_aoti_stream_guard`/`aoti_get_stream`/`tma_descriptor_helpers`/`cpp_scratch`,
`common.py:321-382`）也补回该指南 §5。校验：`checker` pages=379 broken=0；`pytest tools/ -q`
77 passed。

**入链修复**：六批共修复 20+ 处 inbound wikilink 到新基名（各篇自身前置/后续互链、course
index 表格逐行更新为新基名 + 迁移/划界说明、`00_torch_compile_end_to_end_index.md` §8）。
`docs/superpowers/specs/2026-07-28-torch-compile-end-to-end-learning-series-design.md` 是历史
设计规格（先例：B/C/D/E 迁移均未回写），本次同样不touch。

**labs 同步**（最后一个 commit 一并完成，先例见 Task 6 d8c1a5b）：`demo_manifest.json` 七个
F 卷 `page` 字段去前缀；`test_volume_demo_contract.py` 新增 `_F_PAGE_ROOTS`（f01-f08 共 8 个
page_id、7 个不同目录,f03/f04 同目录,含 Task 6 已迁的 f08）接入 `_page_root()`；
`test_call_chain_pages_have_source_walkthroughs` 的 f01 target_pages 条目改指新目录/新基名，
移除不再使用的 `course_root` 局部变量。

**校验**：`python tools/check_links.py`：pages=379（无净变化，纯移动+判重非删除），broken=0
（每个子 commit 独立核验）；`pytest tools/ -q`：77 passed（labs 同步 commit 后核验，中间过程
commit 未逐一跑 pytest，随 D 卷先例批量核验于最后一个 commit）。

---

## 2026-07-30：知识库结构整改 P4 Task 8 组 D 步骤 4（C21 vs codegen 碎片四页，收尾）

**Type**: Cross-link Reconciliation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 D）

C21（`codegen_kernel_mapping_autotuning_and_provenance_analysis.md`，505 行）定位为
codegen/kernel 映射/autotuning/provenance 的**总线页**（覆盖面广、单点纵深浅）。逐页核对
四篇碎片，均判定独有内容 >50%，**保留为专项，不合并**：

| 碎片页 | 独有度判定 | 依据 |
|---|---|---|
| `inductor_codegen_analysis.md`（283 行，含 Task 6 回补的 §7 CPU CodeGen） | 独有 | §7 CPU kernel 类体系（`CppKernel`/`CppVecKernel`/`CppTile2DKernel`）与向量化判定完全不被 C21 覆盖；§2.2 后端文件清单、§4.4 wrapper `IndentedBuffer` 段拆分、§4.3 `TritonKernel.codegen_kernel()` 内部结构均比 C21 对应段落更具体 |
| `inductor_gpu_kernel_dispatch_model.md`（98 行） | 独有 | `program_id→offset→index→mask` kernel 骨架、`IterationRanges` 树、`select_tiling` 打分算法、`GridExpr`/Y-Z 溢出族，全篇贯穿 NPU 对照，C21 §6"Loop codegen"仅 10 行概念性带过 |
| `inductor_reduction_codegen_deep_analysis.md`（89 行） | 独有 | persistent/looped/split/cooperative reduction 四种形态的具体代码生成与阈值，C21 未涉及 reduction codegen 细节 |
| `inductor_autotuning_analysis.md`（152 行，含 Task 6 回补的 §六 ChoiceCaller/TuningProcessPool，本任务组 D 步骤 2 新增 §七 CoordescTuner） | 独有 | `CachingAutotuner` 生命周期、config 启发式、`AttrsDescriptor`、Triton 编译链、NPU 对照表，C21 §8"两层autotuning"只有 20 行提及两层机制存在 |

**处置**：不做内容合并，只补齐此前缺失的双向交叉链接——C21 Related Pages 新增到
`inductor_gpu_kernel_dispatch_model`/`inductor_reduction_codegen_deep_analysis` 的链接
（此前只链 `inductor_codegen_analysis`/`inductor_autotuning_analysis`）；后两篇各自新增
回链 C21。四篇碎片页 Task 6 回补的独有内容（CPU codegen、ChoiceCaller/TuningProcessPool）
确认原样保留，未被本任务触碰或覆盖。

**校验**：`python tools/check_links.py`：pages=379（无净变化，纯加链接），broken=0；
`pytest tools/ -q`：77 passed。

**Task 8 组 D 全组小结**（步骤 1-4 共 4 commit）：C18-C21 从 `19_torch_compile_end_to_end`
纯移动至 `04_inductor/`；`PyTorch_Inductor_Technical_Analysis.md`（1699 行）与
`scheduler_analysis.md`（964 行）两个大型综合参考页判重后完全解体，独有内容分别落地到
`codegen_extension_guide`/`inductor_autotuning_analysis`/`post_grad_passes_guide`/C20 四页；
`inductor_memory_management_analysis.md`（278 行）与
`wrapper_execution_memory_allocation_and_reuse_analysis.md`（D05，228 行）并入 C19 成为
内存主题统一入口；C21 与四篇 codegen 碎片页确认分工清晰、只补链接。全组共删除 4 个页面
（1699+964+278+228 = 3169 行），净新增/改写约 1500 行精炼内容，checker 全程 broken=0。

**追记（复核披露，2026-07-30，不改上方历史行）**：事后复核发现上方步骤 2（8e6a6aa，解体
`PyTorch_Inductor_Technical_Analysis.md`/`scheduler_analysis.md`，本小结未单列 commit 但已
计入上面的删除统计）存在两处需要补救的问题，另有两处内容确认为已核实但归一时未落地：

1. **§9 整删决定推翻了 P3（97840f0）的隔离保留裁定**：97840f0 已明确裁定 addmulnorm 自定义
   融合规则教学"整节仅作历史材料保留"（unresolved-quarantine），8e6a6aa 却以"已自我标注为
   unverified/fictional"为由整删、未见零痕迹处置的说明。现补救：`codegen_extension_guide.md`
   新增 §11 隔离存根，记录该教学曾存在、§9.4 现有融合规则引用（`post_grad.py`/`mm_plus_mm.py`
   等）真实可核、教学主体未验证且含虚构文件名，并指路 `git show 6579658:.../PyTorch_Inductor_Technical_Analysis.md`
   查原文。
2. **§8"虚构"定性亦有夸大**：8e6a6aa 的 commit message 把整个 §8 后端扩展教学描述为
   "uncited/fictional examples"，但复核发现 §8.2 核心数据结构一节确有真实引用
   （`torch/_inductor/codegen/common.py:313`，`DeviceCodegen`/`device_codegens` 的真实定义
   行）；"虚构"定性对 §8 整节而言不准确，已在上述 §11 存根中一并披露口径偏差。
3. **C20（`scheduler_dependency_graph_fusion_and_ordering_analysis.md`）"逐字并入"表述不准
   确**：§18/§19 原标注"从 `scheduler_analysis.md` §7/§9 逐字并入"，实际该页解体时省略了
   §7.3 的 `can_fuse` 合法性判定流程图、§4 的 11 类核心类结构类图，§8.1 的 `min_order`/
   `max_order` O(1) 粗筛 rationale 也未落地——均非逐字。现已回补三处（§1.1 新增类图、§18.3
   新增流程图并保留 P3 当年对该图"单一阈值流程"误导提示的修正注、原 §5 补 `min_order`/
   `max_order` 说明），三处新回补内容保留原文 `L<line>` 引用形式；措辞相应改为"改写并入
   （示意图与部分代码曾省略，经复核回补）"。系统性被剥离的其余 file:line 引用不做全量恢复。
4. **两处已核实但未落地的内容，本次一并补上**：`saved_tensors_recompute_and_runtime_abi_analysis.md`
   （C10）§11.1"重放 view"句后加 `[!todo]` 指出该机制由 `gen_alias_from_base`
   （`functional_utils.py:315`，`runtime_wrappers.py` 六处调用，均已按当前 pinned checkout
   核验）实现，原 `aotautograd_analysis` §10.2 曾展开、归一时未落地；
   `buffer_liveness_memory_planning_and_reuse_analysis.md`（C19）§18.1 回补两个被丢的
   `output_code.py:785-813`/`:817-840` 定位符（对照已删除的 D05 原文位置核实）。

校验：`python tools/check_links.py` pages=379，broken=0；`pytest tools/ -q` 77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 D 步骤 3（C19+D05 内存归一）

**Type**: Redundancy Merge（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 D）

C19（`buffer_liveness_memory_planning_and_reuse_analysis.md`，编译期 realize/last-use/reuse
权威页）确立为内存主题骨架，吸收两页独有内容后二者删除：

- **`inductor_memory_management_analysis.md`（278 行）**：C19 此前完全不覆盖运行期
  `CUDACachingAllocator`（层 2）与 CUDA Graphs `cudagraph_trees` 私有池（层 3）——**独有，
  逐字改写并入 C19 新增 §16**（三层总览 + 层 2 段大小档位 + 层 3 树结构/共享私有池/地址
  稳定/checkpoint/graph partition 集成）；其 §2.6 池初始化大小的完整推导 + 真实
  `test_memory_planning.py` 实例，深于 C19 原有 §7-8 对同一批类的引用式带过——**独有，
  并入 C19 新增 §17**。该页 §2.1-2.5（编译期规划本身）与 C19 §1-15 重叠，未重复搬运。
- **`wrapper_execution_memory_allocation_and_reuse_analysis.md`（D05，228 行）**：boxed
  calling convention（`_BoxedCallable`/`CompiledFxGraph.__call__`）与通信 buffer 独立池
  两个事实 C19 完全未提——**独有，并入 C19 新增 §18.1/§18.2**；`AllocateLine.plan` 的
  完整 7 步决策序列比 C19 §6 的概念性描述更程序化——**并入 C19 新增 §18.3**。该页 §1-3/§9-14
  （boxed wrapper 宏观框架、liveness/reuse 三层区分、复杂度、常见误解）与 C19 §1-2/§6/§13
  重叠，未重复搬运。

**`inductor_memory_allocation_guide.md`（209 行）判定**：>50% 独有（分配器选型对照表、
`memory_stats`/snapshot 实测复现代码、§5 完整的内存越界/踩踏排查流程含
`compute-sanitizer` 工具链、§7 与外部报告的差异订正）——**保留**，6 处指向已删除
`inductor_memory_management_analysis` 的引用改指 C19 对应新章节（§16/§17）。

**入链修复**：C19 自身新增 3 处 Related Pages；`inductor_memory_management_analysis` 删除后
6 个外部文件改指；`wrapper_execution_memory_allocation_and_reuse_analysis` 删除后 7 个外部
文件改指（含 `04_inductor/index.md` 两行表格合并为一行、`00_torch_compile_end_to_end_index.md`
D05 行改指）；changelog 4 处历史活链接降级为反引号。

**配套改动**：`demo_manifest.json` 移除 d05 条目——其页面已并入 C19，而清单要求
`page` 字段全局唯一（C19 自己的 c19 条目已占用该文件名）；`test_volume_demo_contract.py`
的 `expected_ids`/条目计数（55→54）与 `_D_PAGE_ROOTS` 同步；原 d05 demo 用例
（`demo_d_artifact_runtime.py --case wrapper_memory_reuse`）仍可运行，C19 §18.3 末尾以
指路形式保留命令（C 卷页本就无需通过 manifest 强制"配套 Demo"小节）。

**校验**：`python tools/check_links.py`：pages 381→379，broken=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 C（C17 vs lowering_analysis 归一）

**Type**: Move + Redundancy Merge（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 C）

C17（435 行）`git mv` 为 `02_compile_stack/04_inductor/fx_lowering_to_inductor_ir_analysis.md`
（同目录内平移，标题保留原"17 ·"前缀）。

**vs `lowering_analysis.md`（448 行）判重**：两页体裁互补——C17 是"GraphLowering 怎样把
FX 解释成 IR"的机制权威页（Interpreter 状态模型、决策树式 `call_function` 选择、
realization 时机），`lowering_analysis` 是函数/API 级完整参考。用本地 pinned pytorch
checkout（`e8f97c1a6e...`）逐条核验后：

- §一/§二/§四/Call Chain/Data Flow/Key Design Decisions/Beginner Summary：与 C17 §1-§4
  概念重叠，且部分表述（"fallback 兜底保证编译永不失败"）已被 C17 §6 证伪，页内自身也已用
  `[!注]` 标注这些旧结论不成立并指回 C17——删除，不搬运。
- §2.2/2.2.1/2.2.2 完整注册 API 面（`register_pointwise`/`register_foreach_pointwise`/
  `add_needs_realized_inputs`/`add_layout_constraint`/`make_fallback` 等 C17 §4 未点名的
  API）+ 两个实操接入示例（纯 fallback 接入、复用已有 lowering 组合新 op）：**独有，
  改写并入 C17 新增 §4.3**，全部函数位置逐一核验现存（行号相对旧参考漂移 200-1500 行
  不等，以函数名定位为准，页内注明）。
- §三"Lowering 做了什么优化"完整八类目录（Pointwise 融合基础/View 零拷贝家族/Reduction
  优化含 Welford 与 OnlineSoftmax/常量折叠提升/智能 Fallback/Foreach 水平融合/量化 op
  融合/Layout 约束优化）：**独有，改写并入 C17 新增 §16**，逐函数核验存在
  （`make_pointwise`/`make_reduction`/`var_mean_welford_`/`promote_constants`/
  `maybe_layout_constraints` 等）。
- §五"自己做 Lowering 要关注的优化点"三级检查清单（基础/高优先级/进阶）+ Questions &
  Uncertainties（complex tensor 支持不完整、unbacked symbol slice/select 复杂、
  OnlineSoftmax 不支持 split reduction）：**独有，改写并入 C17 新增 §17**。

**入链修复**：C17 基名改名影响 12 个文件；`lowering_analysis` 删除后 14 个外部文件
（含 `PyTorch_Inductor_Technical_Analysis`、`torch_compile_architecture`、
`inductor_reduction_codegen_deep_analysis`、`npu_lowering_guide` 等）重定向到
[[10_fx_lowering_to_inductor_ir_analysis]]；changelog 1 处历史活链接降级为反引号
（另 1 处已是逐项反引号包裹，未受影响）。

**配套改动**：`demo_manifest.json` c17 `page` 字段与 `test_volume_demo_contract.py` 的
`_C_PAGE_ROOTS` 同步新增 `c17 → 04_inductor`。

**校验**：`python tools/check_links.py`：pages 384→383，broken=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 B（C09/C10 vs aotautograd_analysis 1460 行归一）

**Type**: Move + Redundancy Merge + New Specialist Page（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 B）

C09（555 行）`git mv` 为 `02_compile_stack/02_aot_autograd/aotautograd_joint_forward_backward_graphs_analysis.md`；
C10（491 行）`git mv` 为同目录 `saved_tensors_recompute_and_runtime_abi_analysis.md`。标题保留原
"09 ·"/"10 ·" 编号前缀（沿用 Task 5/7 惯例：文件名去前缀、标题不去）。

**vs `aotautograd_analysis.md`（1460 行 = 原 1127 + Task 3 补的 §13）逐节台账**：

| 节 | 判定 | 处置 |
|---|---|---|
| §1-§2 概览/工作流 | 与 C09 §1 自带 mermaid、页头「课程主线与本页分工」表完全重叠 | 删除，不搬运 |
| §3 Phase1 捕获+元数据（含 §3.3 InputAliasInfo/OutputAliasInfo/OutputType dataclass 字段列表） | 用本地 pinned pytorch checkout（`e8f97c1a6e...`）核验：字段基本准确但行号漂移 140-450 行；[[12_graph_effects_alias_mutation_and_order_analysis]]（C05）已用相同源码位置覆盖同一组 dataclass | 删除，不重复搬运 |
| §4/§8 functionalization 包装器链、子类解包 | 与 C05 已覆盖的 dedupe/synthetic-base wrapper 顺序重叠（C05 已引用 `runtime_wrappers.py:1586/1844` 等同一组行号） | 删除 |
| §5 Phase3 分区与编译 | 伪代码/无精确行号，被 C09 §5-§11 的逐行验证内容全面超越 | 删除 |
| §6 decomposition/常量折叠/DCE/pattern matching | 不属于 C09/C10 主题，分别已被 [[15_graph_normalization_decomposition_and_functionalization_analysis]]/`pattern_expression_and_matcher_engine_analysis`/`dead_code_topology_and_effect_order_analysis`（Task 7 新页）覆盖 | 删除 |
| §7 Phase5 运行时包装器链 + `_HANDLER_MAP` | 核验 `AOTDedupeWrapper`/`AOTSyntheticBaseWrapper` 已被 C05 覆盖；但 post-compile 链（`RuntimeWrapper:189`/`AOTDispatchSubclassWrapper:1406`/`FunctionalizedRngRuntimeWrapper:1212`/`AOTDispatchAutograd:3624`）与 `_HANDLER_MAP`/`OutputType→Handler` 派发表（`runtime_wrappers.py:320-345`）经核验后确认未被任何现存页覆盖 | **独有，逐字改写并入 C10 新增 §11.1** |
| §9 AOTConfig/ViewAndMutationMeta/AOTState 数据结构 | ViewAndMutationMeta 行号（446-475）与 C09 已有引用一致；AOTConfig/AOTState 字段无深层机制增量 | 删除 |
| §10.1 激活检查点/重计算 | 与 C10 §5-§8 min-cut 内容重叠 | 删除 |
| §10.2 视图重放优化（`gen_alias_from_base`） | 判定为 C05 alias territory，本任务范围外，暂未新落地（记为待核验缺口） | 未迁移，留 `[!todo]` 级别观察项（未写入正文，本条目记录） |
| §10.3/§10.4 静态输入优化/AOTAutogradCache | AOTAutogradCache 已有专页 [[11_aotautograd_cache_analysis]] | 删除 |
| §11.1/§11.2 输入别名/输出别名限制 | synthetic base 已被 C05 覆盖；输出别名 assert 为单行低价值 | 删除 |
| §11.3 自定义 autograd 函数检测 | 核验：`_is_result_of_custom_autograd_fn` 现已内联（非独立函数），逻辑见 `collect_metadata_analysis.py:479-485`，其对 `OutputType.custom_function_view` 分类的影响见 `:490-497`；未被任何现存页覆盖 | **独有，改写并入 C09 新增 §17** |
| §11.4 `aot_export` 元数据变异禁令 | 核验现行位置 `aot_autograd.py:651-657`（原引用漂移约 350 行）；未被覆盖 | **独有，并入 C09 新增 §17** |
| §12/附录 总结、术语表、参考资料 | 通用摘要，无独有事实 | 删除 |
| §13 ProxyTensor/FakeTensor（Task 3 从已删 A04 页迁入） | 本页此前称"未展开"；核验全部 13.1-13.11 引用行号在当前基线漂移 5-20 行内，内容仍准确 | **完整独立成页** `dispatch_modes_proxytensor_faketensor_analysis.md`，不塞进 C09（主题是 dispatch-mode 机制而非 joint 图提取本身），逐字保留 |

**vs `fx_graph_construction_and_transformation_analysis.md`（269 行，Task 7 瘦身后的 AOT 残留页）**：
§3.1（通用 Proxy/tracer 建 Node 路径）核验已被 [[10_fx_graph_core_data_model_analysis]] 覆盖（同引用
`torch/fx/proxy.py:600-635`），删除；§3.2-3.4（joint 构造、partition 提取、跨图 ABI）与 C09 §2-§11
重叠，删除；§4.1（min-cut 选择保存/重算，**计划点名"两轮审查确认为最深版本"**）核验独有事实
`activation_memory_budget=0/1` 两个边界的快速路径行为（`partitioners.py:3471-3480`，直接读源码确认：
`==0` 返回 `node_info.inputs`，`==1` 直接返回 `solve_min_cut` 结果不再进 knapsack）——**逐字改写并入
C10 §7 新增段**；§4.2-4.3 与 C10 §8-§10 重叠，删除；§10「残留提醒」两条中"saved tensor 不是跨图
Node 引用"已是 C10 §16 原文，删除；"bw 不是反向边图"与"阅读图时的四问"改写并入 C09 新增 §18；
§11 源码导航表全部行已是正文内联引用，删除独立表格。

**遗留两小项处理**：①该页原 §3.3 "有一处链接被 Task 7 规范化"——核实为历史记录性描述，指向已完成
的 Task 7 改动，不需要额外动作，此处记录确认。②"GraphNode 非独立节点类型"一句从
`26aeabb`（Task 6 commit，该 commit 时 fx_graph_construction 文件仍含 §2.1 全文）取回原文，逐字补入
[[10_fx_graph_core_data_model_analysis]] §2.2（C02）Node 定义段末尾，标注取回来源与日期。

**入链修复**：C09/C10 基名改名共影响 24 个文件的裸链接/路径限定链接（图系列两个 00 索引表格行、
自身互指、`graph_stage_boundaries_identity_and_provenance_analysis` 前置行、`aot_autograd_quickstart`、
`aotautograd_cache_analysis`、`memory_amp_profiler`/`12_activation_checkpointing_analysis` 等）；
`aotautograd_analysis` 删除后 15 个外部文件重定向（按引用内容精确改指 C09/C10/新
`dispatch_modes_proxytensor_faketensor_analysis` 三个目标之一，而非笼统指回同一处）；
`fx_graph_construction_and_transformation_analysis` 删除后 2 个外部文件改指；`02_aot_autograd/index.md`
两处表格重写；changelog 3 处历史活链接按"历史不回写"惯例降级为反引号+去向说明。

**配套改动**：`demo_manifest.json` c09/c10 `page` 字段与 `test_volume_demo_contract.py` 的
`_C_PAGE_ROOTS` 同步新增 `c09/c10 → 02_aot_autograd`。

**校验**：`python tools/check_links.py`：pages 385→384（净减 1，两页删除、一页新增），broken=0；
`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 A（C04 + 动态形状归一）

**Type**: Move + Redundancy Merge（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 A）

`19_torch_compile_end_to_end/04_symbolic_shapes_guards_and_graph_reuse.md`（415 行）`git mv`
到 `02_compile_stack/01_dynamo/symbolic_shapes_guards_and_graph_reuse_analysis.md`，作为符号
形状系统（ShapeEnv/SymNode/guard 生成/backed·unbacked 判定）的**概念权威页**。

**vs `dynamic_shapes_full_analysis.md`（460 行）判重**：两页覆盖同一主题但视角不同——C04 是
按机制分层的概念权威叙述，旧页是按 Dynamo→ShapeEnv→Guard→Inductor 链路组织的纵深稿。逐节核对
后确认旧页 §1-§4、§5.2-5.5 的大部分事实（EQUALS_MATCH、ShapeEnv 字段表、guard 三层生成、
automatic_dynamic_shapes 渐进策略、关键源码索引表）已被 C04 现有正文以不同措辞覆盖，真正独有
的三处逐字/改写并入：
1. `DimDynamic` 五值枚举（DYNAMIC/DUCK/STATIC/UNBACKED/INFER_STRIDE，`torch/fx/experimental/
   symbolic_shapes.py:1988`，据当前基线 `e8f97c1a6e...` 重新核验行号，原页引用的 1967 已漂移）
   ——并入 C04 §7 新增子节「维度分配策略：`DimDynamic`」。
2. `recompile_limit=8` 与 `EQUALS_MATCH`（`torch/_dynamo/guards.py:2772`，原引用 2638 已漂移）
   ——并入 C04 §2 静态特化缺点段。
3. `matmul` 端到端案例（Stage 0-5，用户代码→Dynamo 捕获→ShapeEnv 状态→guard 生成→Inductor
   codegen→第二次调用命中缓存）——改写为 C04 新增 §15，SizeArg/buffer_reuse_key 引用行号据
   当前基线重新核验（`common.py:286`、`wrapper.py:123`，原引用 291/100 已漂移）。
其余内容判定为同一事实的不同措辞，不重复搬运。旧页 mermaid 状态图（automatic dynamic 四次调用
时间线）与 C04 §8 结论重叠且与 b09 §9 的状态机图功能重复，判定为可省略的重复可视化，未搬运。

**与 b09/unbacked_symint/inductor_codegen_dynamic_shape 划界**：四页互相新增分工声明——
[[17_dynamic_shapes_generalization_and_fallback_analysis]]（b09）聚焦 Dynamo 侧自动泛化行为
（`frame_state`/`mark_dynamic`）；C04 是符号系统概念权威页；[[25_unbacked_symint_analysis]] 聚焦
unbacked 专项（`torch._check`/`guard_or_*`/size-oblivious）；[[24_inductor_codegen_dynamic_shape_analysis]]
聚焦符号如何流入 Inductor kernel/wrapper。四页页头/正文均补互链，不复述彼此机制细节。

**入链修复**：`dynamic_shapes_full_analysis` 删除后，18 处活链接改指新页——
`fx_graph_cache_analysis`、`dynamo_pgo_cache_analysis`（2 处）、`aotautograd_cache_analysis`、
`04_inductor/index`（表格行改为一行式说明指向 `01_dynamo/index`）、
`inductor_codegen_dynamic_shape_analysis`、`inductor_quickstart`（2 处）、
`npu_compile_paths_overview`、`npu_inductor_linearize_dynamic_shape_analysis`、
`torch_compile_architecture`（2 处）、`npu_inductor_optimization_analysis`（2 处），另 10 处
`04_symbolic_shapes_guards_and_graph_reuse` 裸链（图系列两个 00 索引表格行、C02/C03 前后篇
导航、b09、`06_compile_cache/index`）改指新基名。`wiki/changelog.md` 历史条目中 1 处活链接
（原 2026-07 中旬 NPU codegen 页条目）按"历史不回写"惯例降级为反引号 + 去向说明。

**配套改动**：`tools/labs_torch_compile/demo_manifest.json` 的 c04 `page` 字段与
`test_volume_demo_contract.py` 的 `_C_PAGE_ROOTS` 同步新增 `c04 → 01_dynamo`。

**校验**：`python tools/check_links.py`：pages 386→385，broken=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 7 组 4（control_flow_capture_analysis vs C06 收尾判重）

**Type**: Redundancy Review + Boundary Clarification（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7；Task 5 遗留收尾）

`02_compile_stack/01_dynamo/control_flow_capture_analysis.md`（204 行）vs
`00_torch_compile_end_to_end_index`（已于 P4 Task 10 删除，导读价值并入 [[courses/torch_compile_end_to_end]]）卷 B 的 b04 判重已在
Task 5 完成（零重叠）；本组补做 vs C06
（[[13_structured_outputs_higher_order_and_nested_graphs_analysis]]）的判重，Task 7 组 1-3
未涉及的最后一项遗留。

**判定**：两页体裁不同——本页讲**捕获前端**：Dynamo 字节码符号执行期间"控制流该不该
入图、走哪条路径"（`speculate_subgraph`/`generic_jump`/graph break，均是 `torch/_dynamo/`
内部机制，源码引用 `torch/_dynamo/variables/higher_order_ops.py`、
`torch/_dynamo/symbolic_convert.py`）；C06 讲**IR 层结构**：不论谁捕获（Dynamo/`make_fx`/
`torch.export`），outer/child GraphModule 的 ownership、pytree、DCE 递归边界怎样表达
（源码引用 `torch/_higher_order_ops/`、`torch/fx/`）。逐节核对确认本页 §1-§4（两条路径
框架、`speculate_subgraph` 四步机制、`cond` 深挖的 Dynamo 侧投机/checkpoint-rollback、
控制流 HOP 家族、原生 `if/for/while` 字节码分流、"trace 两支/编译两支/运行一支"三个
误解辨析、路径选型表）在 C06 中**零重叠**——C06 全篇未提及 `speculate_subgraph`、
`VariableTracker`、`generic_jump`、`symbolic_convert`、graph break 中任何一个术语。

**唯一重叠段**：本页 §2.4「下游:HOP 在编译后端的处理」的 `FakeTensorMode`/
`ProxyTorchDispatchMode` 两条 bullet（`cond.py:408/403`，讲 `trace_cond`/FakeTensor merge
机制）与 C06 §6 及其"源码跟读"§1-§3 讲的是**同一段源码**（locator 一致），但 C06 是完整
源码跟读、本页只是两行下游提及——收缩本页这两条为互指 C06，保留 §2.4 另外两条独有 bullet
（`py_functionalize_impl`/Inductor `Conditional` IR，C06 未涉及）。

**互链**：本页页头新增与 C06 的划界说明段；页尾 Related Pages 加一条指向 C06；C06 §6 开头
加一句指回本页（"Dynamo 如何决定该不该走 cond 是更早一层问题"），Related Pages 加一条
反向链接。本组是四组中改动量最小的一组，符合预期（该页"大概率保留，只补划界"）。

**校验**：`python tools/check_links.py`：pages 386→386，broken=0；`pytest tools/ -q`：
77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 7 组 2（pass 方法论归一：C13/C15/C16 吸收两旧页后删除）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

`04_inductor/fx_pass_optimization_methodology.md`（349 行）+
`04_inductor/torch_upstream_pass_deepdive.md`（232 行）逐节比对 [[22_pattern_expression_and_matcher_engine_analysis]]（C13）/
[[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]]（C15）/
[[25_graph_rewrite_legality_validation_and_complexity_analysis]]（C16）。判重结论：两旧页
的机制细节（PatternEntry 三型、序列化缓存、pre/joint/post_grad 三阶段执行顺序、
`b2b_gemm`/`fused_int_mm_mul` 等具体 pass 的门控与合法性）绝大部分已被 C13/C15/C16
以**更细粒度的当前基线 locator**覆盖（例如 C13 §15 复杂度已含旧页没有的 mutation-region
扫描成本、sort-key 比较成本项）；旧页 §3.1-3.4「pass 全集目录」与 `pre_grad_passes_guide`/
`joint_graph_passes_guide`/`post_grad_passes_guide`（各 800+ 行的 Inductor 阶段实操指南，
不属本目录范围）逐条重复，判定冗余不迁移。

**逐条核实并逐字迁移的独有事实**（迁移前用本地 pinned checkout `E:/97-codes/torch_parallel/p`
@ `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`——与 C13/C15/C16 页头基线完全一致——重新核验
了每条 locator，不是照搬旧页的 `9922478dffa` 基线数字；drift 处以本次核验结果为准）：

- **C13 新增两节**：`fwd_only` vs `joint_fwd_bwd` 两种 trace 模式（`pattern_matcher.py:2583/2613`，
  各自决定 pattern 匹配推理图还是训练图）；声明式 pattern vs 手工 `graph.find_nodes` 遍历
  两条并存改图路线 + "归一化 pass 为融合 pass 铺路"技巧。
- **C15 新增/扩写六处**：①pre-grad 的 `config.pattern_matcher` 门控（`config.py:290`）+
  `is_predispatch` 分岔到 `_run_pre_dispatch_passes`/`default_pass_list`
  （`pre_grad.py:200-224,353-361`）；②`lazy_init()` 在 pre_grad（`pre_grad.py:174-182`，
  imports `apply_gumbel_max_trick`/`efficient_conv_bn_eval`/`split_cat`）与 joint
  （`joint_graph.py:81-89`，imports `_pad_mm_init`/`_sfdp_init`/`_misc_patterns_init`）
  两处的具体注册内容；③post-grad 三桶 `pass_patterns[0]→[1]→[2]` 严格顺序
  （`post_grad.py:85-88`）+ auto_chunker 必须早于 pad_mm 的源码注释原文
  （`joint_graph.py:733-734`）+ `reinplace_inplaceable_ops` 具名尾部锚点
  （`post_grad.py:451-452`）；④`GraphTransformObserver` 完整机制新增专节——类定位
  （`torch/fx/passes/graph_transform_observer.py:22`）、`dynamo_timed(f"pass.{subsystem}.
  {passname}")` 计时格式、按名 `config.disabled_passes` 与按子系统 `CompilerBisector.
  disable_subsystem` 两级禁用（79-116）；⑤`GroupBatchFusionBase`/`GroupFusion`/
  `BatchFusion`/`register_fusion` 机制新增专节——与 §2.1 起的 `PatternExpr` 声明式匹配
  并存的另一套改图基础设施（`group_batch_fusion.py:101-159,1488,1615,1664,1677`）；
  ⑥六个"为什么"pass 设计检查框架，插入 §11 决策树之后并逐条标注在本系列的既有落点
  （不重开一套独立标准）。三则源码取证式脚注核实后**在当前基线仍然成立**，随对应
  driver 小节落地：`binary_folding.py` 实际归属 freezing 而非 `pre_grad_passes()`
  （`freezing_patterns.py:98-101,115`、`config.py:1670`）；`decompose_mem_bound_mm.py`
  只被 joint 借用 `check_device` 一个辅助函数，不是 joint pass 本身（`joint_graph.py:40,989`）；
  `b2b_gemm`/`micro_pipeline_tp` 是仅有的两个不经 `GraphTransformObserver` 包裹的
  post-grad pass（`post_grad.py:266-270`）；`check_shape_cuda_and_fused_int_mm_mul_enabled`
  全仓库零引用、确系孤儿函数，`config.decompose_mem_bound_mm` 名不副实——真正门控是
  `torch._C._has_mkldnn` + `post_grad_fusion_options` 字典 key（`post_grad.py:833-838,2048`、
  `config.py:1636`）。
- **C15 新增 §14「跨框架方法论对照」**（明确标注基线独立于本页，torch_npu/vLLM/SGLang
  三个基线不随 upstream 同步）：四家现状对照表 + "融合朝向谱系"一句话读法 + 两种本页
  未覆盖的下游落地形态（rewrite-existing-op、fallback/换手工算子）。这是本组唯一保留
  跨基线内容的一节，按计划指令"四家现状归纳...逐字并入对应 C 篇"执行，不因体裁不完全
  匹配而丢弃。
- **C16 §5 新增"三条可复用的安全判据"**：算子类别不变式、边局部改写+纯函数前提、单用户
  门槛判据——三条判据在 upstream/torch_npu/vLLM/SGLang 独立收敛，故上调为通用安全底线，
  vendor 专属举例（torch_npu `fold_cat`/`fold_squeeze`）未随之带入以保持 C16 纯 upstream
  机制页定位。**C16 §8 差异测试矩阵新增"收益"行**（pass 开/关 A/B、命中计数、端到端性能）。

**结构性冗余、判定不迁移**：§0-§0.8 八阶段"是什么/为什么放这里/适合做/不适合做"的教学
式 prose + 代码示例（C15 §1/§11 的表格化处理已覆盖同等结论，旧页的 pedagogical 铺陈
判定为表达形式而非独有事实）；序列化 pattern 缓存机制（C13 §13 已近逐句覆盖，含同一
`PYTORCH_GEN_PATTERNS` 环境变量）；三种 `PatternEntry`/lowering-pattern vs graph/
replacement-pattern 落地形态区分（C13 §11 locator 更细）；§3.1-3.4 pass 全集目录（与
三份 stage guide 冗余，非本目录职责）。

**两页删除**：`fx_pass_optimization_methodology.md`（349 行）、`torch_upstream_pass_
deepdive.md`（232 行）。**入链改指**：非 changelog 入链共 16 处（`dynamo_pass_methodology`/
`codegen_extension_guide`/`decomposition_passes_guide`/`joint_graph_passes_guide`/
`lowering_analysis`/`post_grad_passes_guide`/`pre_grad_passes_guide` 各 1 处一致模式的
Related Pages 单行，批量改指 C15；`04_inductor/index.md` 表格两行合并为一行更新描述；
`sglang/index.md` 两处 `../../` 相对路径链接顺带改为裸基名（分别指向 C13/C15，消除既有
`../` 违规）；`sglang_compilation_passes_analysis.md`/`25_vllm_ir_and_fusion_passes_analysis.md`
各 2-3 处按内容落点分别改指 C13（引擎机制）或 C15 §14（方法论/跨框架对照)）。changelog
历史条目（2026-07-22、2026-07-20 各一条）中的 3 处活链接按"历史不回写"惯例降级为反引号
+ 说明去向。

**校验**：`python tools/check_links.py`：pages 388→386（两页删除），broken=0；
`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 7 组 3（decomposition_passes_guide vs C08 判重，保留双页）

**Type**: Redundancy Review（no merge）（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

`04_inductor/decomposition_passes_guide.md`（163 行）逐节核对 vs
[[15_graph_normalization_decomposition_and_functionalization_analysis]]（C08，图规范化/Decomposition/
Functionalization 机制骨架）：guide 的 §1（decomposition table 结构与 mermaid）、§3（`register_
decomposition`/`decompositions`/`select_decomp_table`/`torch._decomp.get_decompositions`/
`compile_fx(decompositions=table)` API 表）、§4（注册并加入编译的可运行代码示例）、§5（写
decomposition 前必须回答的算子集收益/语义等价/动态形状三组问题，含 dtype promotion、整数
溢出、NaN/Inf、复数、低精度误差、RNG 顺序、高阶梯度等细项）、§7（六条验证清单）、§8（五条
反模式）在 C08 中**逐句核实均不存在**——C08 是捕获期机制解释（为什么/怎样发生），guide 是
面向开发者的 API 参考+可运行示例+checklist（如何写一个新 decomposition），两者体裁不同，
guide 独有内容占比远超 50% 判据。§2（适合/不适合做的理由）与 C08 §4（收益/代价）主题重叠
但角度不同（guide 是"该不该做"的决策框架，C08 是"是什么"的描述），判定为互补而非重复，
不删减。

**判定**：按计划判据（独有 >50% 时保留为 guide）保留双页，不合并、不删除。双向互链已存在
（guide 页头「课程分工」→ C08；C08 Related Pages → guide），本次只把 C08 → guide 的反向链接
补充一句说明区分二者体裁，避免后续误判为待归一的重复对。

**校验**：`python tools/check_links.py`：pages 388→388，broken=0（本组无内容迁移，仅一行
Related Pages 描述补充）。

---

## 2026-07-30：知识库结构整改 P4 Task 7 组 1（fx_graph_construction_and_transformation_analysis 判重归一）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

`02_compile_stack/02_aot_autograd/fx_graph_construction_and_transformation_analysis.md`（2026-07-23
综合报告快照，601 行）自带「阅读定位与迁移去向」节，按其自述逐节判重 vs Task 7 Step 1
迁入的 C02/03/05/12/13/14/16：

- **§2 FX 对象模型、边和图序**：逐段核对（`Graph`/`Node`/`GraphModule` 三对象表、
  `_update_args_kwargs`/`torch/csrc/fx/node.cpp` C++ 热点实现、`Graph.nodes` 链表遍历、
  `graph.lint()`、链表 vs 普通 list 的设计动机）均已存在于 [[10_fx_graph_core_data_model_analysis]]
  （locator 一致或仅个位数行号漂移，判定为 drift 非新事实）。唯一未被覆盖的一句独有澄清——
  "`users` 反向邻接"与"AOTAutograd backward `GraphModule`"是两个不同概念、不要混为一谈——
  逐字并入该页 §4「两种"反向"不要混为一谈」小节，并加一句到 `19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs`（历史活链接，该页已于 2026-07-30 移动为 [[11_aotautograd_joint_forward_backward_graphs_analysis]]，按"历史不回写"惯例降级为反引号）的互指。原 §2 删除。
- **§5-§6 PatternExpr/PatternMatcherPass**：`CallFunction`/`Arg`/`KeywordArg`/`Ignored`/
  `MultiOutputPattern` 语义表、候选桶注册与逆序匹配、三类 Entry（`LoweringPatternEntry`/
  `GraphPatternEntry`/`ReplacementPatternEntry`）、pre/joint/post 阶段收尾差异，均已存在于
  [[22_pattern_expression_and_matcher_engine_analysis]] 且粒度更细（如该页额外覆盖了旧页没有的
  serialized/precompiled pattern 缓存机制）。原 §5-§6 删除，无独有内容需要迁移。
- **§7-§8 DCE、拓扑与保序**：`Graph.eliminate_dead_code()` 判定条件、`Node.is_impure()`
  规则、DCE 安全前提、`stable_topological_sort` 的 `pending/ready/waiting/cursor` 状态机，
  均已存在于 [[23_dead_code_topology_and_effect_order_analysis]]（该页 §7 的算法级走读甚至更完整）。
  原 §7-§8 删除。
- **§9 复杂度分析（通用表 + §9.1 pattern 总体复杂度）**：已被 [[10_fx_graph_core_data_model_analysis]]
  §11、[[22_pattern_expression_and_matcher_engine_analysis]] §15（含旧页没有的 mutation-region 扫描
  成本 `M(v)`、非 `call_function` root 扫描成本 `H`、sort-key 比较成本 `Lcmp` 等更严格的界）、
  [[23_dead_code_topology_and_effect_order_analysis]] §11 分别覆盖，且新页复杂度分析普遍更严格
  （旧页多处用未加限定的 `O(N+E)`，新页显式标注仅在 arity/sort-key 有界时才成立）。§9.2
  （AOT 全链路复杂度，含 min-cut max-flow 项）核实已存在于
  `19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs` §14 与
  `10_saved_tensors_recompute_and_runtime_abi` §14 各自的复杂度节。原 §9（含 9.1/9.2）删除。
- **§10 不变量与排查清单**：§10.1 八条通用不变量已被 [[21_fx_graph_editing_primitives_and_invariants_analysis]]
  §14（10 项检查清单）覆盖；其中两条 AOT 特有提醒（"不把 bw 叫反向边图"、"不把 saved tensor
  理解为跨图 Node 引用"）保留在本页新 §10「残留提醒」。§10.2「阅读图时的四问」判定为跨
  FX 数据模型/pattern/AOT 三个语境的综合辨析工具，不属于单一目的地页，整节保留。
- **§11 关键源码导航**：通用行（Node/Graph/lint/DCE/Proxy tracing/pattern 相关）已核实存在于
  对应课程页各自的源码路径引用中；AOT 特有行（joint capture/partitioners/runtime_wrappers）
  与本页保留的 §3/§4 正文内联引用重复，压缩为一张 4 行 mini 表，标题改「AOT 特有部分」。

**属 AOTAutograd 特有、本组不处理**：§1 中 2/7 条（AOTAutograd 两张独立 GraphModule、
recompute 非特殊节点类型）、§3（joint→fw/bw 构造与跨图 ABI）、§4（recompute 选择/构图/reorder，
含 min-cut partition §4.1 细节）——按计划要求原样保留，留给 Task 8 与
`09_aotautograd_joint_forward_backward_graphs`/`10_saved_tensors_recompute_and_runtime_abi`
（C09/C10）归一；本组未改动这两节正文一字。

页面净变化：601 → 269 行；标题从「FX Graph 构图与改图机制 — AOTAutograd 正反向分图、
PatternMatcher、DCE 与保序」改为「AOTAutograd Joint Graph 构造与 Recompute — 正反向分图、
Saved-Tensor ABI 与 Min-Cut Partition」以反映瘦身后范围；文件名不变（`fx_graph_construction_
and_transformation_analysis.md`，Task 8 可能进一步处置）。

**入链改指**（9 处非 changelog 入链逐条核实内容落点后精确改）：
`aotautograd_analysis.md` 描述句删去已迁移的"复杂度"表述；`02_aot_autograd/index.md`
两处表格行更新为反映瘦身范围+互指新目录；`04_inductor/index.md`「FX Passes」表格行改指
[[02_compile_stack/03_graph_ir_and_passes/index]]；`torch_upstream_pass_deepdive.md`
Related Pages 一行改指同一索引；`fx_graph_export_and_custom_ops_analysis.md` Related Pages
一行改指同一索引（AOT 部分保留旁注）；`01_fx_export_extensibility/index.md` 两处（页面列表表格行 +
Related Pages 一行）同样改指该索引，注明 AOT 分图见 `02_aot_autograd/index`。

新目录 `03_graph_ir_and_passes/index.md` 补充「与旧页的关系」一节说明本次归一决策；
`02_aot_autograd/index.md` 新增一句互指该索引。

**校验**：`python tools/check_links.py`：pages 388→388，broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 7 Step 1（19 号 C 卷第一批迁入 03_graph_ir_and_passes，12 篇）

**Type**: Structure Reorg（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

**迁移**：`19_torch_compile_end_to_end/`下 C 卷 21 篇中的 12 篇（FX 数据模型/改图原语/pattern/DCE/保序部分：C02,03,05,06,07,08,11,12,13,14,15,16）`git mv` 到新建 `02_compile_stack/03_graph_ir_and_passes/`，去数字前缀规范重命名（新基名全库唯一，无冲突）；C 卷其余 9 篇（C01 动机、C04 动态形状、C09/C10 AOT joint/recompute、C17-C21 Inductor IR/Scheduler/Codegen）留 Task 8：

- `02_fx_graph_core_data_model.md` → `fx_graph_core_data_model_analysis.md`
- `03_graph_values_metadata_and_signatures.md` → `graph_values_metadata_and_signatures_analysis.md`
- `05_graph_effects_alias_mutation_and_order.md` → `graph_effects_alias_mutation_and_order_analysis.md`
- `06_structured_outputs_higher_order_and_nested_graphs.md` → `structured_outputs_higher_order_and_nested_graphs_analysis.md`
- `07_graph_capture_frontends_and_tracing.md` → `graph_capture_frontends_and_tracing_analysis.md`
- `08_graph_normalization_decomposition_and_functionalization.md` → `graph_normalization_decomposition_and_functionalization_analysis.md`
- `11_graph_stage_boundaries_identity_and_provenance.md` → `graph_stage_boundaries_identity_and_provenance_analysis.md`
- `12_fx_graph_editing_primitives_and_invariants.md` → `fx_graph_editing_primitives_and_invariants_analysis.md`
- `13_pattern_expression_and_matcher_engine.md` → `pattern_expression_and_matcher_engine_analysis.md`
- `14_dead_code_topology_and_effect_order.md` → `dead_code_topology_and_effect_order_analysis.md`
- `15_graph_pass_pipeline_ordering_and_fixpoint.md` → `graph_pass_pipeline_ordering_and_fixpoint_analysis.md`
- `16_graph_rewrite_legality_validation_and_complexity.md` → `graph_rewrite_legality_validation_and_complexity_analysis.md`

**入链修复**：42 个文件、179 处 `[[...]]` 目标随重命名替换（含裸基名与 `19_torch_compile_end_to_end/`-限定两种旧写法，后者一并去路径前缀改裸基名）；两个 00 系列索引（`00_pytorch_graph_series_index.md`、`00_torch_compile_end_to_end_index.md`）按 Task 6（D 卷）先例逐行替换链接目标，不做 B/E 式整段压缩（因 C 卷本次只分发一半，Part IV 与 C01/04/09/10 仍在原目录）；changelog.md 内唯一一处历史提及（反引号引用，非 `[[...]]` 活链接）按惯例不回写。

**Labs 同步**：`demo_manifest.json` 12 条 C 卷 `page` 字段（c02/c03/c05/c06/c07/c08/c11-c16）同步改名；`test_volume_demo_contract.py` 的 `_page_root()` 新增 `_C_PAGE_ROOTS` 字典分支（C 卷仅部分迁移，未列出的 c-id 仍回退旧目录，与 D 卷四散模式同构）；`CourseMarkdownContractTest._course_pages()` 同步纳入 `03_graph_ir_and_passes/*.md`。

新目录 `03_graph_ir_and_passes/index.md` 建成实质内容（替换占位）；`02_compile_stack/index.md`、`01_ai_frameworks` 相关导航行同步更新。

**校验**：`python tools/check_links.py`：pages 388→388（纯移动不改变页数），broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 6（19 号 D 卷分发,7 篇 1888 行）

**Type**: Redundancy Consolidation + Physical Move（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 6）

逐篇处置(每篇/每组一 commit,均去 `dNN_`/`f08_` 前缀):

- **D01**(335 行)→`02_compile_stack/04_inductor/inductor_compile_fx_orchestration_analysis.md`。与`inductor_compiler_pipeline_analysis.md`(921 行,原索引"脊柱文档")逐节判重:该页 §1-§7 逐阶段走读(Dynamo/AOTAutograd/Decomposition/FX Passes/Lowering/Scheduler/CodeGen)已被 04_inductor 目录各阶段专题页(`pre_grad_passes_guide`/`joint_graph_passes_guide`/`post_grad_passes_guide`/`decomposition_passes_guide`/`lowering_analysis`/`scheduler_analysis`/`inductor_codegen_analysis`)、01_dynamo、02_aot_autograd 目录各专题页更深入地覆盖(逐关键词核实,深度均超过原页对应节),判定为冗余,不强并;§0 全景图与§8 设计哲学/§9 文件速查表判定为独有综合价值,吸收为 D01 新增 §0(mermaid 全景图 + 各阶段深挖入口导航表)与 §15/§16。921 页删除,19 条入链逐条改指(9 条改指 D01 并按目标页语境调整描述文字,6 条因链接方本身就是该阶段权威页而判定自引用冗余后直接删除,2 条改指更贴切的具体目标——`decomposition_passes_guide`/`wrapper_execution_memory_allocation_and_reuse_analysis`,2 条改指`02_compile_stack/04_inductor/index`导航)。
- **D02**(315 行)→`02_compile_stack/02_aot_autograd/aot_runtime_wrappers_and_lazy_backward_compile_analysis.md`。纯平移,无判重对象(spec 缺口补齐)。
- **D03**(211 行)→`02_compile_stack/04_inductor/async_compile_workers_and_module_loading_analysis.md`。纯平移。
- **D04**(219 行)→内容改写为`02_compile_stack/06_compile_cache/index.md`的 overview 主体:D04 的 14 节源码级分析(七层 cache 对照表、FXGraphCache key 构造、guard-vs-key 双层策略、序列化边界、AOTAutograd cache 在 FXGraphCache 之上、失效非广播事件、不变量、复杂度、常见误解)置于 index 顶部作 §1-14,原 91 行导航(四层教学表、专题页表、生命周期阅读顺序、课程边界、审计边界)完整保留于后作 §15-19(未删减,含 PGO 行等 D04 未覆盖的互补内容)。D04 文件删除,12 条入链改指`02_compile_stack/06_compile_cache/index`(路径限定,因"index"歧义)。
- **D05**(224 行)→`02_compile_stack/04_inductor/wrapper_execution_memory_allocation_and_reuse_analysis.md`。纯平移;不做内存归一(留 Task 8 与 C19 一并),页头加互指说明,并与`inductor_memory_management_analysis`/`inductor_memory_allocation_guide`加双向回链。
- **D06**(320 行)+**F08**(322 行)→`03_runtime_graphs/cuda/`(`cudagraph_trees_warmup_record_and_replay_analysis.md`/`training_inference_cudagraph_and_freezing_analysis.md`)。与`PyTorch_CUDA_Graphs_Complete_Guide.md`"方式2"/"综合比较"节判重:该节内容为伪代码级使用示例与四种用法对比表,未引用`cudagraph_trees.py`/`freezing.py`任何一行;D06/F08 是源码行级机制分析(`CUDAGraphNode`/`CUDAGraphTreeManager`状态机、`freezing.py`变换链与所有权后果),独有内容占比 >>50%,按判定标准保留为专题页,双向加互指(Guide 方式2/综合比较节 + Related Pages 新增两条,`cuda/index.md`新增两行)。
- **D07**(264 行)→`02_compile_stack/07_debugging/compiled_artifact_lifecycle_and_runtime_failures_analysis.md`。纯平移;`07_debugging/index.md`补为第 10 篇(原"九篇"改"十篇"),插入 aotautograd 失败定位之后、minifier/bisector 之前(生命周期状态机为失败定位提供时序坐标),去除因迁入而冗余的旧"本卷前置卷 D"Related Pages 行。

**labs 同步**:`demo_manifest.json`7 条 D 卷 page 字段(d01-d07)去前缀更新,D04 字段改为`index.md`(因内容并入索引页);`test_volume_demo_contract.py`的`_page_root`新增`_D_PAGE_ROOTS`分支(D 卷四散至 04_inductor/02_aot_autograd/06_compile_cache/03_runtime_graphs::cuda/07_debugging 四个目录,不同于 B/E 整卷单目录迁移,故改按`page_id`路由)与 F08 特判;`CourseMarkdownContractTest.test_call_chain_pages_have_source_walkthroughs`的`target_pages`同步更新三个已迁移页面的根目录。

**校验**:`python tools/check_links.py`:pages 390→388(921 与 D04 两页删除,净 -2),broken=0;`pytest tools/ -q`:77 passed(labs 同步后恢复,过程中三个 manifest/call-chain 测试短暂失败,已在本任务收尾前修复)。

**追记(复核修正,2026-07-30,不改上方历史行)**:

(a) 上文 D01 条"2 条改指更贴切的具体目标——`decomposition_passes_guide`/`wrapper_execution_memory_allocation_and_reuse_analysis`"经复核有误:对照 d8c1a5b 实际 diff,该 commit 只改了 `decomposition_passes_guide.md` 一处(旧链 `[[inductor_compiler_pipeline_analysis]]` → `[[15_inductor_compile_fx_orchestration_analysis]]` — compile_fx 把 decomposition table 传给 AOTAutograd 的调用点);`wrapper_execution_memory_allocation_and_reuse_analysis.md` 在 d8c1a5b 中未被触碰(该页由更早的 01c1422 移入本目录,与本次改指无关)。实际应为"1 条改指更贴切的具体目标"。

(b) "19 条入链"为**页数**口径(`git grep -l`,d8c1a5b^ 下命中 19 个非 changelog 文件);若按**出现次数**(`git grep -o '\[\[inductor_compiler_pipeline_analysis'`,同一基线,排除 changelog.md 本身)计,实际为 **30** 次。两个数字口径不同,均属实,原文未注明口径,此处补注。

(c) "纯平移四篇"(D02/D03/D05/D07,各条独立标注"纯平移")中,**D05 并非字节对字节平移**:对照 01c1422 实际 diff,`wrapper_execution_memory_allocation_and_reuse_analysis.md` 净 +5 行(6 insertions/1 deletion)——除去 `d05_` 前缀迁移外,页头新增了一段与 `inductor_memory_management_analysis`/`inductor_memory_allocation_guide` 的互指说明(该 commit 消息本身已披露"页头加互指说明",但顶部"纯平移"表述未与此区分,此处复核标注更精确)。D02/D03/D07 复核 diffstat 确认为无额外插入的纯移动。

(d) 本次(kb-reorg P4 Task 6 复核修复)在 D01 判重时被跳过、未落地的 3 处原文事实,已按"逐字为基底+溯源"原则回补:`inductor_compiler_pipeline_analysis.md` §7.4(CPU CodeGen:`CppKernel`/`CppVecKernel`/`CppTile2DKernel` 类体系与向量化判定)→ [[20_inductor_codegen_analysis]] 新增 §7;§7.5(`ChoiceCaller`/`TritonTemplateCaller` 编译期算法选择基础设施、`TuningProcessPool` 子进程隔离)→ [[21_inductor_autotuning_analysis]] 新增 §六,并恢复一条指向 [[15_inductor_compile_fx_orchestration_analysis]] 的 Related Pages 链接(替代被删的旧链);§4.3.7 `dedup_reduce_scatters`(reduce_scatter 线性可加性融合)→ [[32_post_grad_passes_guide]] Pass 19-21 节补齐第 4 个成员。三处原文均无固定源码基线声明,回补时对照本地 pinned checkout(`E:/97-codes/torch_parallel/p`,`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`)核实所有行号引用,发现原文行号普遍漂移(如 `ir.py:5582` 实为 `ChoiceCaller` 所在 `ir.py:6185` 与 `TritonTemplateCaller` 所在 `select_algorithm.py:3347` 两处的合并粗写),均以 `[!correction]` 就地标注,不改写原文陈述本身。校验:`python tools/check_links.py` pages 388→388(纯编辑,broken=0);`pytest tools/ -q` 77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 5 spec 审查修复（bytecode_analysis 两算法补落点 + 加速数字弃置例外可见化）

**Type**: Redundancy Consolidation 修复（spec 审查发现 2 项：824c5a2 抽验通过、989080c 修剪无删失，另 2 项需修）

**修复 1(实质缺口)**：`git show d0b998f:...PyTorch_Dynamo_Technical_Analysis.md` §3.2「字节码分析技术」的 `remove_dead_code()`(活代码可达性,跟随跳转目标与异常表条目)与 `stacksize_analysis()`(≤100 轮定点迭代)两个算法此前全库无落点——97d19ce 的判重台账遗漏了这两个具体函数(只覆盖了架构层面的"死代码消除/栈大小分析"存在性,未展开算法机制)。按 B04 §14 吸收 a03 `bytecode_transformation.py` 内容的同款模式,在 [[12_instruction_translator_and_bytecode_state_machine_analysis]] 新增 §15,以原文算法主张为基底,对照本地 pinned 源码(`E:/97-codes/torch_parallel/p`,`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`)的 `torch/_dynamo/bytecode_analysis.py` 补全精确行号引用（原文无定位符,引用补全属允许的增强,非新造断言）：

- §15.1 `remove_dead_code()`(`torch/_dynamo/bytecode_analysis.py:69-125`)：`find_live_code`(`:74-89`)可达性遍历,含跳转目标空值断言(`:82-87`,原文简化版缺失)；Python 3.11+ 异常表 start/end 回填(`:93-123`,原文完全未覆盖)；调用点 `convert_frame.py:979`。
- §15.2 `stacksize_analysis()`(`:249-291`)：`StackSize`区间(`:224-247`)+ 三类传播边——顺序执行(`:266-270`)、跳转(`:271-276`)、异常表(`:277-281`,原文未覆盖)；`fixed_point`收敛判据(`:260-262`)与失败断言(`:286-289`,原文未覆盖)；结果写回 `co_stacksize`(`:290`，`bytecode_transformation.py:1875`)。
- [!correction] 原文简化伪代码的定点循环声明 `changed` 却从未赋值为 `True`，若照抄会导致收敛判断恒假、循环跑满 100 轮不提前退出；订正为当前源码的 `FixedPointBox` 判据，不迁移原文这段有缺陷的伪代码。
- §9"删除死bytecode和无意义跳转"补一句指向 §15 的前向指针。

**修复 2(例外可见化)**：97d19ce 判重台账里"§6.1 加速数字无源、与 e07 论点抵触、不落地"的例外此前只记录在 changelog，页面本身不可见。在 [[17_compile_latency_cache_and_steady_state_performance_analysis]] §1（"平均耗时没有诊断价值"论点处）补一句注，明示该旧页给出的"2-3x/4x/1.5-2x"数字已按记录弃置、未作 `[!todo]` 保留。

**校验**：`python tools/check_links.py`：pages 390→390（纯编辑）,broken=0；`pytest tools/ -q`：77 passed（新增 §15 的 mermaid/list-marker/locator-length 质量门禁通过，所有引用跨度 ≤57 行，未触发 100 行上限）。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 4-5（control_flow_capture_analysis 判重结论 + dynamo_pass_methodology 与 B10 互链）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**Step 4 判重结论**：`control_flow_capture_analysis.md`(204 行,已在 01_dynamo/)vs B04(`instruction_translator_and_bytecode_state_machine_analysis`)——关键词核查（`generic_jump`/`FOR_ITER`/`speculate_subgraph`/`higher_order`/`torch.cond`/`CondHigherOrderVariable`/`POP_JUMP_IF`/`evaluate_expr`/`guard_bool`/`install_subgraph` 在 B04 全部零命中）确认**零重叠**：B04 讲字节码解释器的通用状态机（`run→step→dispatch`，以 `CALL` 为例），本页讲控制流两条路径的专题机制（HOP `speculate_subgraph`/`torch.cond` 深挖、原生 `if/for` 的 `generic_jump`/`FOR_ITER`），两页视角完全互补。按任务指示**不动本页内容**（Step 2 已做的入链机械修复保留）。与 C06 的判重按计划原文注明留 Task 7 收尾。

**Step 5 判重结论**：`dynamo_pass_methodology.md`(152 行,已在 01_dynamo/)vs B10(`backend_contract_and_custom_backend_analysis`)——两页主题重叠（都在讲 Dynamo backend 扩展）但**层次不同**：B10 是契约机制的源码级深挖（registry/gm 形态/example inputs/返回值语义/mode-options 传递/可选 context 接口/failure 分层，全部带 `file:line`），本页是"要不要在 Dynamo 做这件事"的开发决策指南（适合/不适合决策表、可运行注册代码示例、改图排错流程、离开 Dynamo 的路由表），两者是 CLAUDE.md Page Types 里 Entity(deep dive) vs Guide 的正常搭配,不构成重复共存。**收窄重叠表述**：§1 删除与 B10 §1 重复的契约机制复述,改为精简结论 + 指向 B10;§4 API 速查表后加指针注明源码级实现见 B10 §2/§7;§6 改图规则清单后加指针注明完整正确性清单见 B10 §13、IR 方言边界见 B10 §10。**新增互链**：两页此前互相都没有链接对方（各自 Related Pages 检索确认零命中）——本任务修复,双向加回链。

**校验**：`python tools/check_links.py`：pages 390→390（零删除、纯编辑）,broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 3（删除 torch_compile_source_analysis.md，593 行，独有内容并入 B01/B02）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**判重**：`04_inductor/torch_compile_source_analysis.md`（593 行，`torch.compile()` 函数体逐段源码解析）与 B01/B02 通读比对——**与 A 卷/PyTorch_Dynamo_Technical_Analysis 不同,本页确有实质独有内容**（关键词 grep 核实 `3.15`/`GIL_DISABLED`/`sysconfig`/`_log_api_usage_once`/`CompilerBisector`/`guard_filter_fn`/`use_aoti`/`is_exporting`/`_in_hop_compile`/`_TorchCompileAOTInductorWrapper`/`apply_mode`/`apply_options`/`list_mode_options` 在 B01/B02 及全库均无匹配或仅部分匹配），逐字迁入并对照本地 pinned pytorch checkout（`E:/97-codes/torch_parallel/p`）核验行号：

- §4.1 Python 版本兼容性检查（3.15+ 拒绝、free-threaded GIL<3.13.3 拒绝、`_log_api_usage_once` 遥测）→ [[10_torch_compile_api_and_first_call_lifecycle_analysis]] 新增 §13.1，核验定位 `torch/__init__.py:3267-3280`
- §4.6 `torch.export` 兼容性（`is_exporting()`/`_in_hop_compile()` 短路为 no-op）→ 同页新增 §13.2，核验定位 `torch/__init__.py:3350-3359`
- §4.4 CompilerBisector 二分调试的 API 入口钩子（`bisect_backend := CompilerBisector.get_backend()` 覆盖 backend + vLLM 自定义 backend 保护条件）→ [[22_backend_modes_options_stances_and_fullgraph_analysis]] 新增 §13，核验定位 `torch/__init__.py:3330-3342`；与 [[15_minifier_repro_and_compiler_bisector_analysis]] §7（bisector 内部二分算法）互补,双向加回链
- §4.5 特殊选项提取 + §4.7/§5 三个 wrapper 类的方法级实现（`_TorchCompileInductorWrapper.apply_mode/apply_options/get_compiler_config/reset`、CUDA<12.6 CUPTI workaround、`_TorchCompileAOTInductorWrapper` 子类的 `cpp_wrapper`/`aot_inductor.package`/`V.set_aot_compilation`、`_TorchCompileWrapper` 的 `lookup_backend`+kwargs 透传）→ 同页新增 §14，核验定位 `torch/__init__.py:2907-3096` 区间多段
- [!todo] §14.2 迁移时发现:`use_aoti=True` 是从 `torch.compile()` **JIT 入口**直接触发的 AOTInductor 打包路径,而 `f07_aotinductor_packaging_and_deployment_analysis`（2026-07-30 起随 kb-reorg P4 Task 9 迁入并改名为 [[28_aotinductor_packaging_and_deployment_analysis]]，本条历史记载不回写活链接）§2-§3 记录的公开入口 `aoti_compile_and_package` 明确要求 `ExportedProgram`(export 驱动、部署前离线完成)。两条路径是否共享下游产物、`use_aoti` 这条 JIT 捷径的运维定位,F07 尚未覆盖——本任务范围内未展开核实,双向加回链留待后续处理。**已于 kb-reorg P4 Task 9(2026-07-30)核实解答**：两者在 `compile_fx`/`CompiledAOTI` 层汇合于同一段代码，但 `CompiledAOTI` 装载成可调用 runner 这一步受 `enable_autograd_for_aot` 门控、并不对称（只有 `use_aoti` 路径强制打开），差异还在前端捕获来源（Dynamo 运行时捕获 vs `ExportedProgram`）与是否额外 `package_aoti` 打包成 `.pt2`；详见 [[22_backend_modes_options_stances_and_fullgraph_analysis]] §14.2 note。
- §6 编译模式对照表（mode/CUDA Graphs/Triton autotune 布尔矩阵）与 B02 §3 的同一组事实（散文体）重复,不迁移；§7 能力范围与限制、§8 使用示例是已覆盖机制的摘要/演示,不迁移；§2/§3/§4.2/§4.3 是与 B01/B02 完全重复的调用栈图与参数说明,不迁移。

**删除**：`git rm` 该页。**入链修复**：`04_inductor/dynamic_shapes_full_analysis.md` 改指 [[10_torch_compile_api_and_first_call_lifecycle_analysis]]；`04_inductor/index.md` 移除本页表格行,改为一行式说明指向 [[02_compile_stack/01_dynamo/index]]。

**校验**：`python tools/check_links.py`：pages 391→390，broken=0，orphans=0；`pytest tools/ -q`：77 passed（含新增 §13/§14 的 mermaid/list-marker/locator-length 质量门禁）。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 2（删除 PyTorch_Dynamo_Technical_Analysis.md，2018 行）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**判重**：`02_compile_stack/01_dynamo/PyTorch_Dynamo_Technical_Analysis.md`（2018 行，早期"演示代码 + 简化伪源码"风格全景页）逐节通读（§1 概述、§2 核心架构 7 阶段、§3 技术实现细节、§4 典型示例代码、§5 核心模块架构与工作流程、§6 生态角色与应用、§7 覆盖度声明、总结）后与 B 卷十篇逐一核对，**全篇无独有事实需要迁移**：

- §2/§3/§5（eval-frame 拦截、字节码分析转换、符号执行、变量跟踪、Guard 系统、FX 图构建、后端编译 7 阶段机制，含模块依赖图与调用链图）均为 B01-B10 的低精度复述——本页用"简化伪代码"且无 `file:line` 定位符，B 卷十篇逐条给出精确源码定位。抽样核实：本页 §3.1 PEP 523 C 扩展入口链描述（`dynamo_custom_eval_frame_shim → dynamo__custom_eval_frame → set_eval_frame`）缺少 [[11_eval_frame_callback_and_code_cache_analysis]] §2 已给出的更完整更准确的三态协议（`None`/`False`/callable，定位 `torch/csrc/dynamo/eval_frame.c:518-533`/`:616-638`）；本页 §7 列的"未覆盖组件"（VariableBuilder、Source 系统、高阶操作）实际均已被 [[13_variable_tracker_source_and_python_object_model_analysis]]、[[21_control_flow_capture_analysis]] 覆盖。
- §4 典型示例代码分析（简单函数/nn.Module/动态形状/graph break/循环编译走读,均为未验证的插图式伪代码）被 `tools/labs_torch_compile/demo_b_dynamo_capture.py` 的十个真实可运行 case（compile_lifecycle/backend_modes_fullgraph/eval_frame_cache/bytecode_state_machine/variable_source_guards/output_graph_side_effects/guards_recompile/graph_break_resume/dynamic_shapes/custom_backend_contract）与 B 卷各篇「源码跟读」「配套 Demo」小节取代，不重复落地插图代码。
- §6.1"性能提升"给出的"典型模型 2-3 倍/Transformer 4 倍/小模型 1.5-2 倍"加速比**无源可查**（`raw/` 下仅 `dynamo.eddx`/`torch.compile.eddx` 图示源，非可引用的实测数据）；且与 [[17_compile_latency_cache_and_steady_state_performance_analysis]] §1 的核心论点（"平均耗时没有诊断价值，必须拆分测量场景"）直接抵触，判断为不应作为事实迁移的营销式泛化断言，不落地（不同于常规"无源可疑声明转 [!todo]"处理，因为该断言的方法论本身已被后继页的论点否定，保留只会误导读者）。
- §6.5"性能优化技术"（算子融合/内存布局/循环展开/并行化）实际描述的是 Inductor 层优化,不属于 Dynamo 机制,且与 B 卷主题不符,留待后续 Inductor 相关任务处理,本任务不落地。
- §5.1 模块依赖关系图（`torch._dynamo` 包结构树）经抽样核对本地 pinned pytorch checkout（`eval_frame.py`/`config.py`/`convert_frame.py`/`bytecode_analysis.py`/`bytecode_transformation.py`/`codegen.py`/`backends/`/`variables/` 均存在）大体准确，但属于可从源码树直接重建的编排性示意图（非独有事实），且 B 卷各篇「源码阅读顺序」「源码补充」小节已提供更具体的逐机制文件路径，不单独迁移。

**删除**：`git rm` 该页。**入链修复**（18 个外部文件的活链接，均为泛指性"参见 Dynamo 帧评估/字节码/Guard"式指针，按 Task 3 a05 先例改指 [[02_compile_stack/01_dynamo/index]]）：`pytorch_dispatcher_analysis`、`npu_operator_graph_eligibility_guide`（2 处）、`op_registration_pipeline_analysis`、`aotautograd_analysis`、`dynamic_shapes_full_analysis`、`inductor_compiler_pipeline_analysis`、`inductor_memory_management_analysis`、`torch_compile_source_analysis`、`unbacked_symint_analysis`、`torch_mlir_pass_pipeline_analysis`、`vllm/index`、`23_vllm_compilation_cudagraph_analysis`（2 处）、`25_vllm_ir_and_fusion_passes_analysis`、`wiki/index`；`dynamo_pgo_cache_analysis` 按其"VariableBuilder/guard 的宿主"原描述精确改指 [[13_variable_tracker_source_and_python_object_model_analysis]] + [[15_guards_cache_lookup_and_recompilation_analysis]] 两篇。域内三篇（`control_flow_capture_analysis`、`dynamo_pass_methodology`、`dynamo_quickstart`）的 Related Pages 按原描述拆成对应的具体 B 卷页链接（如"帧评估、字节码符号执行、Guard 与重编译"拆为三条精确链接），比泛指索引更精确。`wiki/changelog.md` 里 3 处写入当时的历史活链接（2026-06-30/2026-06-12 更早条目）按"历史不回写"惯例降级为惰性反引号 + 去向说明；另 2 处（2026-07-17 前后两条）本就是反引号包裹的非活链接，未受影响。

**索引重建**：`02_compile_stack/01_dynamo/index.md` 页面列表从 4 行重建为 13 行（quickstart + B01-B10 + control_flow 专题 + dynamo_pass_methodology development guide），移除 `PyTorch_Dynamo_Technical_Analysis` 行；页头摘要与最后更新同步。

**校验**：`python tools/check_links.py`：pages 392→391，broken=0，orphans=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 1（19 号 B 卷迁入 01_dynamo，10 篇 2653 行）

**Type**: Structure Reorg（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**迁移**：`19_torch_compile_end_to_end/b01-b10`（API 与首次编译生命周期、backend 参数/stance/fullgraph、eval-frame callback/code cache、字节码符号执行、VariableTracker/来源、OutputGraph/side effects、guard/cache/recompile、graph break/resume、动态形状泛化/fallback、backend contract，共 2653 行）`git mv` 到 `02_compile_stack/01_dynamo/`，去 `bNN_` 前缀规范重命名（新基名全库唯一，无冲突）：

- `b01_torch_compile_api_and_first_call_lifecycle_analysis.md` → `torch_compile_api_and_first_call_lifecycle_analysis.md`
- `b02_backend_modes_options_stances_and_fullgraph_analysis.md` → `backend_modes_options_stances_and_fullgraph_analysis.md`
- `b03_eval_frame_callback_and_code_cache_analysis.md` → `eval_frame_callback_and_code_cache_analysis.md`
- `b04_instruction_translator_and_bytecode_state_machine_analysis.md` → `instruction_translator_and_bytecode_state_machine_analysis.md`
- `b05_variable_tracker_source_and_python_object_model_analysis.md` → `variable_tracker_source_and_python_object_model_analysis.md`
- `b06_output_graph_side_effects_and_graph_emission_analysis.md` → `output_graph_side_effects_and_graph_emission_analysis.md`
- `b07_guards_cache_lookup_and_recompilation_analysis.md` → `guards_cache_lookup_and_recompilation_analysis.md`
- `b08_graph_break_resume_functions_and_partial_graphs_analysis.md` → `graph_break_resume_functions_and_partial_graphs_analysis.md`
- `b09_dynamic_shapes_generalization_and_fallback_analysis.md` → `dynamic_shapes_generalization_and_fallback_analysis.md`
- `b10_backend_contract_and_custom_backend_analysis.md` → `backend_contract_and_custom_backend_analysis.md`

**入链修复**：十篇互链（前置/后续 + Related Pages）随重命名全库替换（裸基名，新基名唯一）；`00_torch_compile_end_to_end_index.md` 卷 B 表按 Task 3/4 先例压缩为一行式指向（完整表格重建留 Task 10）；`00_pytorch_graph_series_index.md` 未引用卷 B，无需改动；`02_compile_stack/07_debugging/` 内 `dynamo_explain_and_graph_break_diagnosis_analysis.md`/`guard_failure_and_recompile_diagnosis_analysis.md`、`19_torch_compile_end_to_end/d04_compile_cache_hierarchy_keys_and_invalidation_analysis.md`/`f06_custom_backends_and_device_integration_analysis.md` 对十篇的引用改指新基名；`wiki/changelog.md` 里 Task 3 条目（本文件写入当时的历史记载，按本文件"不随后续迁移回写"惯例）中对 `b01`/`b03`/`b04` 的活链接降级为惰性反引号 + 说明新名，不当作活链接维护。

**Labs 同步**：`tools/labs_torch_compile/demo_manifest.json` 的 10 条 B 卷 `page` 字段同步改名（page_id/volume/script/case 不变，55 条不变）；`test_volume_demo_contract.py` 的 `_page_root(labs_root, volume)` 新增 B→`01_dynamo` 分支（Task 4 预留的同一模式）；`CourseMarkdownContractTest._course_pages()` 同步纳入 `01_dynamo/*.md`（含卷内既有的 quickstart/pass-methodology/control-flow/旧大文页，质量门禁全部通过）；`test_call_chain_pages_have_source_walkthroughs`/`test_b07_cache_miss_formula_contains_all_addends` 的 b04/b07 路径改指新家。

**校验**：`python tools/check_links.py`：pages 392→392（纯移动不改变页数），broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 4（19 号 E 卷迁入 07_debugging，9 篇 2391 行 + 吸收旧 debug 页）

**Type**: Structure Reorg + Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 4）

**迁移**：`19_torch_compile_end_to_end/e01-e09`（观测台账、Dynamo explain/graph break、guard failure/recompile、AOTAutograd/Inductor 失败分层定位、minifier/repro/bisector、正确性验证方法论、compile 延迟/cache/稳态性能、kernel/fusion/内存/硬件性能归因、生产上线/fallback/监控，共 2391 行）`git mv` 到 `02_compile_stack/07_debugging/`，去 `eNN_` 前缀规范重命名（新基名全库唯一，无冲突）：

- `e01_observability_logs_counters_and_artifact_map_analysis.md` → `observability_logs_counters_and_artifact_map_analysis.md`
- `e02_dynamo_explain_and_graph_break_diagnosis_analysis.md` → `dynamo_explain_and_graph_break_diagnosis_analysis.md`
- `e03_guard_failure_and_recompile_diagnosis_analysis.md` → `guard_failure_and_recompile_diagnosis_analysis.md`
- `e04_aotautograd_and_inductor_failure_localization_analysis.md` → `aotautograd_and_inductor_failure_localization_analysis.md`
- `e05_minifier_repro_and_compiler_bisector_analysis.md` → `minifier_repro_and_compiler_bisector_analysis.md`
- `e06_compiled_correctness_validation_methodology_analysis.md` → `compiled_correctness_validation_methodology_analysis.md`
- `e07_compile_latency_cache_and_steady_state_performance_analysis.md` → `compile_latency_cache_and_steady_state_performance_analysis.md`
- `e08_kernel_fusion_memory_and_hardware_performance_analysis.md` → `kernel_fusion_memory_and_hardware_performance_analysis.md`
- `e09_production_rollout_fallback_and_monitoring_analysis.md` → `production_rollout_fallback_and_monitoring_analysis.md`

**删除并吸收**：`02_compile_stack/04_inductor/Pytorch_Compile_Debug_Analysis.md`（558 行，E 卷的压缩前身）逐节判重后删除。机制性内容（Dynamo/FX/Guards/AOT/Inductor 各阶段原理与定位方法、通用决策树前 4 步）已被上述九篇更严谨的源码级分析取代，不重复落地；但该页提供的**可运行排查脚本**、**分布式专属决策分支**与**kernel/CUDA 层崩溃诊断**九篇均未覆盖，逐字迁入新建的 `02_compile_stack/07_debugging/index.md` 附录：`run_debug.sh`（环境变量一键启动）、`export_capture.py`/`capture_backend.py`（DIY GraphModule 捕获，与官方 `after_dynamo`/`after_aot` repro 生成器正交互补）、`collect_artifacts.sh`、`diff_rank_logs.sh`（多 rank 日志对比，九篇未提供任何等价工具）、决策树第 5 分支（仅分布式场景复现）、kernel/CUDA 崩溃关键词与修复清单（segfault/OOM/launch failed/arch mismatch，[[18_kernel_fusion_memory_and_hardware_performance_analysis]] 只覆盖性能不覆盖崩溃）、工程化使用流程、小技巧、upstream issue 附件清单（与 [[15_minifier_repro_and_compiler_bisector_analysis]] §11 的通用 repro 标准互补，不重复）。

**入链修复**：九篇互链（前置/后续 + Related Pages）随重命名全库替换（裸基名，新基名唯一）；`00_torch_compile_end_to_end_index.md` 卷 E 表按 Task 3 先例压缩为一行式指向（完整表格重建留 Task 10）；`b01/b02/b07/b08/b09/b10/d03/d05/d06/d07/f01/f03/f04/f07/f08` 对九篇的引用改指新基名；`04_inductor/index.md`、`inductor_memory_allocation_guide.md`、`inductor_quickstart.md`（3 处）、`npu/npu_debug_guide.md`、`torch_compile_architecture.md`（2 处）、`19_torch_compile_end_to_end/11_graph_stage_boundaries_identity_and_provenance.md` 对旧 debug 页的引用改指 `02_compile_stack/07_debugging/index`；changelog 历史条目（2026-07 更早的一条）按本文件「历史不回写」惯例改为惰性反引号 + 说明去向，不当作活链接维护。

**Labs 同步**：`tools/labs_torch_compile/demo_manifest.json` 的 9 条 E 卷 `page` 字段同步改名（page_id/volume/script/case 不变，55 条不变）；`test_volume_demo_contract.py` 新增 `_page_root(labs_root, volume)` 按卷解析页面目录（E→`07_debugging`，其余仍在 `19_torch_compile_end_to_end`，为 Task 5-9 逐卷迁移预留同一模式）；`CourseMarkdownContractTest._course_pages()` 同步纳入 `07_debugging/*.md`，保持 list-marker/mermaid/locator 质量门禁覆盖迁移后的九篇。

**校验**：`python tools/check_links.py`：pages 393→392（-1 为旧 debug 页删除，九篇是移动不减少），broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 3（19 号 A 卷删除，5 篇 1790 行）

**Type**: Redundancy Consolidation + Structure Reorg（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 3；P4 阶段首个编辑任务）

**删除**：`19_torch_compile_end_to_end/a01-a05`（Tensor/Storage/View、operator/schema/dispatcher/autograd、Python frame/code object/bytecode、dispatch mode/ProxyTensor/FakeTensor、eager-capture-compile-replay 成本模型，共 1790 行）。这五篇是"执行模型前置基础回顾"，判重发现约 60-70% 内容与 `01_eager_runtime` 各功能页重复，但**逐句核查后确认每篇均有编译器视角的独有分析**（不可盲信"回顾=重复"），已逐字迁入对应功能页新增小节，不是简单删除：

- a01 → [[01_eager_runtime/01_tensor_and_storage/10_tensor_impl_and_storage_analysis]] §13（differentiable view/DifferentiableViewMeta、mutation 状态机、复杂度记账、常见误解、view→Autograd→编译器的源码跟读）
- a02 → [[10_pytorch_dispatcher_analysis]] §12（ADInplaceOrView 分层、mutation 算子 rebase 样本、Dispatcher/Autograd Edge/FX data edge 三种"边"辨析）
- a03 → `b04_instruction_translator_and_bytecode_state_machine_analysis`（P4 Task 5 起更名为 [[12_instruction_translator_and_bytecode_state_machine_analysis]]） §14 + `b03_eval_frame_callback_and_code_cache_analysis`（P4 Task 5 起更名为 [[11_eval_frame_callback_and_code_cache_analysis]]） §13（`bytecode_transformation` 重组子系统、code object/frame/instruction 定义表、C-hook 与 ConvertFrame 边界）
- a04 → `aotautograd_analysis` §13（P4 Task 8 起独立成页 [[10_dispatch_modes_proxytensor_faketensor_analysis]]）（`__torch_function__`/`__torch_dispatch__`/ProxyTensor/FakeTensor 四层分工、`track_tensor_tree`、FakeTensorMode 状态、decomposition 落点、数据相关 operator 边界）
- a05 → `b01_torch_compile_api_and_first_call_lifecycle_analysis`（P4 Task 5 起更名为 [[10_torch_compile_api_and_first_call_lifecycle_analysis]]） §12 + [[17_compile_latency_cache_and_steady_state_performance_analysis]] §12-§16（七阶段成本词汇表、cache-entry 查找与 backend handoff 源码补充；参数化成本模型/break-even/四层 cache 对照表与 e07 既有的测量方法论合并，不重复落地两处）

**入链修复**：`f05`（a02/a04 引用改指 pytorch_dispatcher_analysis / aotautograd_analysis）、`b01`/`e07`/`d06`（a05 引用改指内容实际落点）；卷内 a01-a05 互链随整卷删除一并消失；`00_torch_compile_end_to_end_index.md` 的"卷 A"表按 Task 3 约定不做整体重排（留给 Task 10），仅去除失效行并加一行去向说明，避免 broken>0。

**Labs 同步**：`tools/labs_torch_compile/demo_manifest.json` 移除 a01-a05 五条 page 映射（60→55 条）；`test_volume_demo_contract.py` 同步更新 `expected_ids`/条目计数与 `test_call_chain_pages_have_source_walkthroughs` 的 target_pages；`demo_a_execution_model.py` 脚本与其两个 `VolumeABContractTest` 保留（脚本级机制验证，不依赖已删除的课程页）。

**校验**：迁移前逐条 grep+读段落核实独有性（而非凭印象判重）；迁移时对照本地 pinned PyTorch(`9922478`) 抽样核验源码定位符，发现 a05 一处 `output_graph.py` 引用行号漂移（原 A 卷标注的 `e8f97c1a...` 版本与当前 pinned 版本不一致），已重新定位到 `call_user_compiler`/`_call_user_compiler`/`BackendCompilerFailed` 的准确行号后再落地。`python tools/check_links.py`：pages 398→393，broken=0，orphans=0（基线持平）；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P3 完成（runtime_graphs 去重收官）

**Type**: Structure Reorg（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.3；P0-P7 的第四段）

- **06_graphs 目录 14 页 → 10 页**：删 cuda/README、npu/README（overview 三写归一）、CUDA_Graphs_Timing_Diagrams（时序图内联进 Complete_Guide 并以 mermaid 替换其 ASCII 重复图）、npugraphs_memory_reuse_analysis（Graph Tree 双写并入 torch_compile_npugraphs_deep_dive）；comparison 633→145 行只留真差异表。
- **每次合并均经逐句对抗审查**：三轮回补（8 组机制事实、6 组分析事实、硬件约束转 [!todo]）、一处推断性时序事件删除、一处"理论→实测"措辞漂移回退——详见本页下方各 Task 条目与 git 历史。
- CLAUDE.md「Never delete」条款同步为「Merge over coexist」（完整修订仍在 P7）。
- 全程 broken=0；本域净删约 2900 行重复内容。

## 2026-07-30：知识库结构整改 P3 Task 4(NPU Graph Tree 双写合并)

**Type**: Redundancy Consolidation(设计:`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`;P3 阶段最大编辑)

**合并**：`06_graphs/npu/npugraphs_memory_reuse_analysis.md`(1698 行)并入 `torch_compile_npugraphs_deep_dive.md`(2397→2714 行,净增 317)。两页讲同一套 NPU Graph Tree 机制,§三「NPU Graph Tree 核心机制」与被并页的「Graph Tree 机制」「内存复用策略」「@torch.compile 场景案例」约 800 行重叠;被并页内「关键代码解析(合并自 memory_management)」节为上次合并未合净的残留,一并处理。

**逐节处置**（被并页每个 `##`/`###` 节）：
- `目录`/`概述` — 纯框架文字,丢弃(信息已含于主干开篇)。
- `核心架构组件`(NPUGraph/NPUCachingAllocator/NPUGraphTreeManager/NPUGraphNode 四个 C++/Python 结构) — NPUGraph、PrivatePool 为独有,搬入新增 §3.5.1-3.5.2(表格化,无损);Manager/Node 类结构与主干 §2.5.1、§3.2 重叠,丢弃。
- `内存池管理`(MempoolId_t/生命周期/PrivatePool) — 独有,搬入 §3.5.1(含 capture_begin 池注册代码节选、Capture vs Replay 差异表、replay 可跨流回放的事实)。
- `Graph Tree 机制`(核心概念/树节点结构/路径管理与状态切换) — 核心概念的"树形峰值内存=max 而非 sum"公式独有,搬入 §3.1 补充段;树节点结构与主干重复,丢弃;路径管理与状态切换的 `_run`/`apply_checkpoint` 源码与主干 §3.2.2-3.2.3 重叠但含更细粒度行号引用,细粒度引用被 §3.6 新流程图吸收,原始代码块丢弃。
- `Capture 与 Replay 流程` — capture_begin/replay 完整 ACL 样板代码与主干 §2.7 概念重叠,丢弃;差异表与"可跨流回放"独有事实已搬入 §3.5.1。
- `内存复用策略`(Liveness/StorageWeakRefWrapper/Alias Detection) — 全部独有,搬入新增 §3.7.1-3.7.3(无损)。
- `详细案例分析:@torch.compile 场景` — 旗舰案例,与主干既有的"训练循环 A→B→C"场景不同(本例为 graph-break 分支场景);代码示例+完整时序图+效率对比搬入新增 §3.8(精简,舍弃与其自身内部重复的 4 张辅助图:Liveness T0-T3、Checkpoint/Restore 图解、Phase1-4 图解、总内存变化图,因其事实已被时序图/§3.6/§3.7 覆盖);两张拓扑图(代码执行流程图、Graph Tree 节点关系)舍弃,拓扑事实由时序图 Note 与案例文字保留。
- `关键源码解析`(Graph Tree 创建流程/add_function 流程/NPUGraphNode 内存管理) — 前两者与主干 §2.6、§3.2.2 逐字重复,丢弃;第三者的 `_record`/`run` 与主干 §2.7-2.8 重复丢弃,`__init__` 中父子 liveness 对比片段独有,搬入 §3.7.1。
- `内存复用可视化总结` — 与时序图/效率对比重复,除"内存复用效率对比"(独有的定量对比,搬入 §3.8)外丢弃。
- `关键代码解析(合并自 memory_management)`(TreeManager 生命周期/分配器检查点结构/Warmup 与静态输入/静态输入优化) — 全部独有:TreeManagerContainer 生命周期搬入新增 §3.4;BlockState/SegmentState/PrivatePoolState 搬入 §3.5.3;NPUWarmupNode.run 与主干 §3.2 warmup 概念重叠部分丢弃,`_use_npu_memory_pool_manager` 事实已隐含于既有描述;静态输入优化(`npugraph_managed_idxs` 等)搬入 §3.7.4。「使用建议补充」为通用泛泛建议,丢弃。
- `总结`(核心机制回顾/三分类/触发条件/最佳实践/待探索问题) — 前三者与新搬入内容重复,丢弃;最佳实践中"减少 graph break/稳定输入/避免频繁路径切换"三点折入 §3.8 结尾一句;`待探索问题`(stale_storages 恒空、clear_path_state 为空操作)独有,搬入 §3.6。
- `参考文档` — 源码文件列表并入主干 §六"核心实现分布在"(新增 NPUGraph.h/cpp、NPUCachingAllocator.cpp 两行);外部链接与主干无实质差异,丢弃。

**主干新增结构**：§三新增 3.4 TreeManagerContainer 单例生命周期、3.5 C++ 层数据结构(NPUGraph/PrivatePool/Checkpoint 快照)、3.6 Checkpoint 恢复流程与内存复用三分类、3.7 内存复用策略(Liveness/弱引用/别名检测)、3.8 案例分析(graph-break 分支完整生命周期);3.1 补充内存优化公式;3.2.3 加一句指向 §3.6 的前向引用。全部新增 mermaid(1 flowchart + 1 sequenceDiagram)经 `@mermaid-js/mermaid-cli` 实渲验证。

**收尾**：5 处入站 wikilink 改指(`12_activation_checkpointing_analysis`/`aclgraph_deep_analysis`/`aclgraph_multistream_rng_analysis`/`npugraphs_make_graphed_callables_deep_dive`/主干自身的 Related Pages 自链接一并清除);`06_graphs/npu/index.md` 删行+承接说明+日期 bump 至 2026-07-30。体量:两页合计 1698+317=2015 行变动,净删 1698-317=1381 行(≥1200 目标);主干净增 317 行(≤500 目标)。链接检查 broken 0→0、orphans 1→1(与本次改动无关的既有孤儿页)。

**追记(质量审查回补)**：上表 2714 行为本条目写入当时的统计;后续质量审查中对本次合并做了三处小修正回补(§3.6 措辞澄清、Related Pages 去重等),回补后终值为 **2809 行**(非 2714)。审查过程中还发现 §3.8 案例分析里有一处推断性事件描述并非源码可验证事实,已就地删除——记此一笔作为本次合并的自纠记录。

---

## 2026-07-30：知识库结构整改 P3 Task 5(deep_dive 内两处收缩 + 三项遗留修正)

**Type**: Redundancy Consolidation(设计:`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`;P3 阶段收尾编辑)

**A. §四「与 make_graphed_callables 的对比」收缩**：`torch_compile_npugraphs_deep_dive.md` §四(90 行)只留 4.1 功能对比表 + 到 [[20_npugraphs_make_graphed_callables_deepdive]] 的链接(90→18 行,净删 72)。逐段核对：4.2「实现对比」两幅 ASCII 流程图——`make_graphed_callables` 六步图被承接页「二、完整实现流程(六阶段)」(490-716 行)以远更详细的代码粒度完整覆盖，`torch.compile(backend="npugraphs")` 六步图被本文档自身 §1.3 流程图 + §2.3-2.8 完整覆盖(均原地保留未删)，故两图均丢弃不搬运；4.3「执行时序对比」sequenceDiagram 被本文档自身 §3.2.5(A→B→C 完整执行场景，含 generation/warmup/record/execute 状态转换)以更细粒度完整覆盖，同样丢弃不搬运。核对结论：§四无独有内容需要搬运，承接页 `npugraphs_make_graphed_callables_deep_dive.md` 本次未变动(668→668)。

**B. 附录 A「mode="reduce-overhead" 完整编译流程与双路径对比」收缩**：整改决策(spec §3.3)以 `aclgraph_deep_analysis.md` 为 reduce-overhead 捕获路径权威页。附录 A(759 行)逐段核对：
- 一、mode 参数澄清(1.1 mode→config 表、1.2 mode/backend 关系) — 承接页原无 mode 参数说明,独有,搬入承接页新增 §1.5「mode 参数与两条路径的触发关系」。
- 二、整体架构对比(2.1/2.2 双路径 ASCII+mermaid 架构图) — 路径 B 部分与本文档自身 §1-2 完全重叠丢弃;路径 A 部分(Wrapper→compile_fx→AOT→Codegen→cudagraph_post_compile)的阶段划分与源码定位独有,浓缩为 §4.4 的 Phase 表(不逐字保留原 ASCII/mermaid 图,信息已被表格等价表达,避免图文双份冗余)。
- 三、路径 A 完整调用链路(3.1-3.7,各 Phase 完整代码清单) — 逐 Phase 源码位置(文件+行号)独有,浓缩进 §4.4 Phase 表;3.6 npugraphify monkey-patch 入口代码与承接页 §二·差异1 现有代码逐字重复,丢弃、原地加引用。
- 四、路径 A 完整调用栈 / 五、路径 B 调用栈(对比参考) — 与三、八两节内容重复(调用链路的另一种呈现),丢弃。
- 六、关键代码路径差异分析(6.1 编译产物对比/6.2 录制内容差异/6.3 回退差异) — 独有,浓缩进 §4.4 对比表(录制内容/NPU 利用率/编译耗时/回退行为/典型定位五维)。
- 七、Wrapper 类对比(7.1/7.2) — 独有,并入新增 §1.5。
- 八、执行时序对比(8.1/8.2 两幅 sequenceDiagram) — 与三、四节的 Phase 文字描述重复(可视化重述而非新事实),且承接页已有 §1.3 通用捕获/回放时序图,丢弃不搬运。
- 九、性能特性对比(9.1 综合对比表/9.2 适用场景) — 独有,浓缩进 §4.4 对比表 + 选型清单。
- 十、max-autotune 额外优化 — 独有,并入 §1.5 mode 表备注。
- 十一、对比总结(11.1 一句话/11.2 决策流程图/11.3 代码示例/11.4 文件索引) — 11.1/11.2 改写为 §4.4 结尾的一句话总结+选型清单;11.3 代码示例与本文档自身 §5.2 最佳实践重复,丢弃;11.4 文件索引表独有,搬入 §4.4。

附录 A 本体替换为一段摘要(路径 A/B 核心差异一段话 + 链接到 [[10_aclgraph_deep_analysis]] §1.5/§4.4，759→12 行，净删 747)。

**承接页接收清单**：
- `npugraphs_make_graphed_callables_deep_dive.md`：净增 0 行(668→668，§四无独有内容)。
- `aclgraph_deep_analysis.md`：净增 63 行(569→632)，新增 §1.5「mode 参数与两条路径的触发关系」(并入「一、路径概述」)与 §4.4「与 backend="npugraphs" 路径(路径 B)的对比」(并入「四、这条路径为什么会存在」)，均为小节融入而非尾部堆贴；同步合并 Related Pages 中两行重复的 `[[11_torch_compile_npugraphs_deepdive]]` 为一行(见下 C.1)。

**C. 三项遗留小修正**(上一任务质量审查发现)：
1. `aclgraph_deep_analysis.md` Related Pages 两行 `[[11_torch_compile_npugraphs_deepdive]]` 合并为一行，注释合并为"NPU Graphs 与 torch.compile 集成深度分析；§3.4-3.8 内存管理与复用；reduce_overhead vs npugraphs 对比"。
2. `torch_compile_npugraphs_deep_dive.md` §3.6 加一句区分 checkpoint 恢复三步机制与分类表 dead 行 `_npu_npuCachingAllocator_raw_delete` 为两层释放，不引入新机制断言。
3. 本 changelog 上条(P3 Task 4)追记回补后终值 2809 行(非 2714)与 §3.8 自纠记录(即上方"追记"段)。

**体量**：`torch_compile_npugraphs_deep_dive.md` 2809→1990 行(净删 819 = §四 -72 + 附录A -747)；`aclgraph_deep_analysis.md` 569→632 行(净增 63)；`npugraphs_make_graphed_callables_deep_dive.md` 668→668 行(不变)。三页合计 4046→3290，净删 756 行。链接检查 broken 0→0、ambiguous 70→70、bare_index 70→70、orphans 1→1(与本次改动无关，四项均与改动前基线一致)。全部新增/删除的 mermaid 块经 fence 计数校验闭合平衡(无需重渲，本次未新增 mermaid 内容，只有整块删除)。

**追记(复核回补)**：上方 B 段"六、…独有,浓缩进 §4.4 对比表"与"九、…独有,浓缩进 §4.4 对比表 + 选型清单"两处表述不够精确——复核发现浓缩过程连带丢了 6 组分析性事实(Inductor 各编译阶段"对性能的影响"列、6.3 三类回退场景的路径 A 特有行为、7.1 Wrapper 的 options 处理/reset 行为两行、7.2 `backend="npugraphs"` 时 `mode` 参数的 TypeError footgun、9.1 首次执行延迟与内存占用两个对比维度、9.2 路径 A/B 的具体选型场景列表)，已在后续 commit 中以表行/脚注形式逐字回补至 `aclgraph_deep_analysis.md` §1.5/§4.4；更准确的表述应为"部分浓缩,若干维度经复核回补"。另，上方"三、"段对附录 §3.6 的删除依据也不完整：其 `npugraphify` 函数体代码不仅与 `aclgraph_deep_analysis.md` §二·差异1 的 monkey-patch 注册代码重复，也与 `torch_compile_npugraphs_deep_dive.md` 自身 §2.4.1「npugraphify 函数（utils/_graph_tree.py）」的函数体重复，是双重重复而非仅与差异1重复。

---

## 2026-07-29：知识库结构整改 P0–P2(工具、快速止血、图源入库)

**Type**: Structure Reorg(设计:`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`;这是 P0–P7 七阶段的前三段,后续 P3+ 将做内容去重与目录重组)

- **新增链接检查器** `tools/check_links.py`(+8 项测试):broken/ambiguous/裸 index/孤儿页四类检查,`--strict` 做阶段门禁;基线与终值存档于 `docs/research/2026-07-29-linkcheck-*.json`。
- **坏链清零**:broken 138 → 0。其中 113 处为指向已删审计产物的 `[[correction_report]]` 标注(删除时甄别出 14 处含实质技术警示的,以纯文本注回补);其余 24 处为陈旧目录名遗留、相对路径深度错、示例文本误解析等,逐处改指现路径/转义/登记 Knowledge Gap。
- **工作产物清理**:`docs/audits/`(55MB,107 文件)移出工作区并 gitignore;`torch_compile_debug/` 调试残留删除;`raw/_ingest/` 施工单与 wanka 源汇编稿归位 `docs/research/`;重复 docx 经核验后删除;demo 脚本迁 `tools/`。
- **图表可再生**:`.html2md` 中 24 个图表源 html 与 8 个渲染脚本入库 `tools/figs/`、`tools/html2md/`(修正六个脚本的硬编码路径深度);端到端验证 html 与 PNG 再生均字节一致。
- **文档除假**:README 与 `wiki/index.md` 重写/修正(删幻影目录树、全表页数按实数重算并注明统计口径);本 changelog 头部新增"按写入当时状态记载、不随迁移回写"政策。

---

## 2026-07-29：`torch.compile` A→F 六卷 CUDA-first 配套 Demo

**Type**: Executable Teaching Labs + Evidence Contract（固定实现审计仍以 PyTorch
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` 为准；当前本机为 PyTorch `2.9.1+cpu`，
不把 CPU、FakeTensor、generated code 或源码可达性冒充 CUDA 实测。）

- **建立六个卷级入口与 45 个 case**：新增统一 harness，以及 A/B/C/D/E/F 六个入口。
  A 覆盖执行模型，B 覆盖 Dynamo，C 编排原有 21 篇深度 Labs，D 覆盖 artifact/cache/runtime，
  E 覆盖诊断、复现、正确性与性能，F 覆盖 compiled autograd、checkpoint、distributed、
  custom op/backend 与 AOTInductor。
- **固定运行与失败合同**：统一支持 `--list --json`、可重复 `--case`、`--device`、`--seed`
  和隔离的 `--output-dir`；`PASS/BLOCKED/FAIL` 分别使用退出码 `0/3/2`，CLI 合同错误为
  `4`。能力缺失在 case 正文前阻断，异常和子进程非零退出不会被吞成成功。
- **形成 60 页唯一映射**：`labs/demo_manifest.json` 把 A01–F08 全部正文映射到真实入口和
  case；A/B/D/E/F 的 39 篇正文在最终 `Related Pages` 前新增可复制命令、证据字段与
  CUDA 边界。C01–C21 保留原有逐篇 Lab，由 C 卷入口分主题编排，不复制或弱化原证据。
- **真实预检结果**：本机实际通过 A 的 4 个 CPU case、B 的 10 个 case、C 的 6 个编排
  case、D 的 3 个 CPU case、E 的 7 个 CPU case、F 的 4 个 CPU case；CUDA/Triton、
  native compiler、Linux distributed 和多卡路径按声明返回 `BLOCKED`。修复了
  after-Dynamo repro 误传 `make_fx` 图的问题，并把 Windows Gloo 不可运行边界改为
  声明式 Linux gate。
- **验证与审计闭合**：Labs 合同由 42 项增至 63 项并全部通过；审计工具 90/90 通过；
  六入口 subprocess JSON 清单、60 页映射、Markdown 回链、CLI 退出码、C 子进程证据传播
  均有自动测试。统一课程账本重建为 6,483/6,483 个 decisions，validation error 为 0；
  `[S]/[R]/[I]/[M]/[B]` 为 1,287/366/3,675/19/43。
- **保留验收边界**：当前没有 CUDA receipt，因此 GPU kernel 数、显存峰值、加速比、
  autotune winner、CUDAGraph replay、多卡 FSDP/DTensor 与 AOTI package load 仍未升级为
  `PASS`，必须在目标 CUDA/Linux 环境重新运行对应 case。

---

## 2026-07-28：`torch.compile` 端到端 A→F 课程补齐

**Type**: Source-faithful Learning Series + Evidence Closure（固定源码基线为 PyTorch
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`；本阶段只完善原理和源码链路，不新增 demo，
不把 CPU 环境观察外推为 native/CUDA 实测。）

- **新增课程域 `19_torch_compile_end_to_end/00_torch_compile_end_to_end_index`**（该目录已于
  P4 Task 10 整体解散删除，导读价值并入 [[courses/torch_compile_end_to_end]]）：
  形成 A→F 六卷编号路径；新写 A01–A05、B01–B10、D01–D07、E01–E09、F01–F08 共
  39 篇正文，并将 `00_pytorch_graph_series_index`
  的 C01–C21 正文、Labs 和证据资产实体并入同一目录；旧目录 16 已删除，原 01–21
  文件顺序和内容身份继续保留。
- **从 eager 贯通生产运行**：卷 A 建立 Tensor/storage/layout、dispatcher/autograd、
  Python frame、dispatch modes 和成本模型；卷 B 追踪 `torch.compile`、eval-frame、
  bytecode、VariableTracker/Source、OutputGraph、guards、graph break、dynamic shape 与
  backend；卷 D 衔接 `compile_fx`、AOT runtime wrapper、异步编译、cache、wrapper memory、
  CUDAGraph Trees 与 artifact lifecycle。
- **补齐工程验收闭环**：卷 E 按 capture/AOT/lowering/codegen/load/runtime 分层组织日志、
  explain、guard/recompile、minifier/bisector、正确性、冷启动/热缓存/稳态、fusion/memory/
  hardware 和 production rollout；明确“能编译”“能加载”“数值正确”“性能获益”是四个门槛。
- **补齐高级边界**：卷 F 区分 Compiled Autograd 与 AOTAutograd，解释 activation
  checkpoint 与 AOT recompute 的叠加、DDP/FSDP/DTensor 的图和 rank state、custom op
  完整编译契约、Dynamo backend 与 Inductor device backend、AOTInductor 的 PT2/C ABI/
  constant ownership，以及 training/inference/freezing/CUDA Graph 的正交组合。
- **纠正历史路径和关键误解**：minifier 当前入口核准为
  `torch/_functorch/fx_minifier.py`；AOTInductor package 入口核准为
  `torch/_inductor/__init__.py` 与 `torch/_inductor/package/package.py`。明确 JIT cache
  不是部署 ABI、freezing 不是 `eval()`、CUDA Graph static input 首先约束地址而非值、
  partitioned forward 的 saved activation 不能全部当作 backward static input。
- **统一课程与证据闭合**：端到端 manifest 已逐篇纳入 C00–C21，包含
  61 个编号入口/正文和 1 个 C00 支持索引；6,312 个 claim units 全部有决定，其中
  `[S]/[R]/[I]/[M]/[B]` 分别为 1,287/366/3,552/19/43，另有 1,045 个非断言；
  validation error 为 0。卷 C 自身账本也已随实体迁移重建并复核。
- **导航闭合**：框架总索引、Dynamo、AOTAutograd、Inductor、runtime memory、FX/export/
  extensibility、distributed primitives、C 卷图编译主线与 compile cache
  均已建立课程回链。当前主机无 MSVC/CUDA/Triton；未新增或伪造 native runtime receipt。

---

## 2026-07-28（二）：K3 开源栈源码审计 + 稳定性专题横切

**Type**: Source Audit（代码）+ Deep Dive + Cross-Document Correction（把 K3 随发布开放的仓库从“报告项目级说法”推进到源码级，并把散落各页的稳定性机制横切成一页。）

- **新增 [[01_theory/01_models/moonshot_kimi/27_moonep_analysis]]**：固定 `MoonshotAI/MoonEP@0f385f03`（2026-07-28，MIT）做 file:line 级审计。拆出在线规划的五步算法（全局直方图 → `balance` 守恒 → 贪心配额 → 逐专家分配 → 精确落点 + dedup），并从算法本身推出 README 只给结论的那条约束——贪心“一次填满最空接收方”⇒ 每 rank 至多从一个远端 home group 接收 ⇒ 训练时 `B=E/R` 足够。记录工程形态（几乎全部 kernel 用 CUTLASS Python DSL 写，C++ 仅 60 行 pybind + VMM/IPC/multicast；planner 是单个 cooperative kernel，计划全程不下 GPU）、梯度回收闭环、以及源码注释里写明的被否决方案（远端写清零 vs 本地清零的 NVLink 预算权衡，`grad_reduce.py:24-30`）。
- **新增 [[01_theory/01_models/moonshot_kimi/26_kimi_k3_open_source_stack_analysis]]**：按 GitHub API 逐仓核对 `created_at`/`pushed_at`，把仓库分成三类（生产栈组件 / K3 自己写的作品 / 评测配套）。**更正一条流传口径**：FlashKDA 建于 2026-04-20、最新 commit `d2ff19a`（2026-05-26），**并非随 K3 新开源、也未为 K3 更新**；真正落在发布窗口的新仓是 MoonEP、AgentENV、minitriton、nano-kpu、PerceptionBench。补齐 AgentENV 的 overlaybd 与 memory ballooning（报告未提）、minitriton 与 nano-kpu 的免责声明与“综合估算而非 P&R”口径。
- **新增 [[01_theory/01_models/moonshot_kimi/25_kimi_k3_stability_analysis]]**：按“哪条轴会失稳”重组报告 §2.2/§2.3/§2.4/§2.5/§3.2/§3.3/§3.4/§4.1.2/§4.1.4 与 Appendix C，得出主线——**K3 拒绝的每个替代方案都是“用质量或超参换稳定”，被采纳的机制几乎都同时提升质量或降低开销**（aux loss、sign 更新、BIP、hard clamp、无界 SwiGLU、SigLIP 初始化、WSD 七处取舍指向同一方向）。
- **修正 [[01_theory/01_models/moonshot_kimi/23_kimi_k3_infra_deepdive]] §2.1**：原记“Per-Head Muon 如何与 K2 MuonClip 组合仍未知”过窄；报告 §3.3（p.11）明写三者并用（Per-Head Muon + K2 weight-clipping + QB），并给出 cosine + 1% warmup、weight decay 0.1、8k→64k 预训练。“未知”收窄到联合消融与 clip 阈值。§5 事实边界表的 MoonEP 行标为已兑现、AgentENV 行补仓库口径。
- **补齐 [[01_theory/01_models/moonshot_kimi/22_kimi_k3_architecture_deepdive]] 两处缺失的“为什么”**：MoonViT-V2 从零训练的**首要动机是训练稳定性**（SigLIP 初始化的 MoonViT-3D 梯度范数持续偏高且频繁 spike，Fig. 6），且视觉质量持平；Block AttnRes 在 K3 的精确配置为 8 块 × 12 层、计入 embedding 共 9 块，开销从 `O(Ld)` 降到 `O(Nd)`。
- **`D12`（`03_posttraining/12_kimi_k3_posttraining_case_study_analysis`，历史活链接，已于 2026-07-31 因 kb-reorg P5 迁移为 [[24_kimi_k3_posttraining_case_study_analysis]]，按"历史不回写"惯例降级为反引号）新增 §9.1**：AgentENV 仓库侧口径，并标注与报告延迟数字（133/49 ms vs `<100`/`<50 ms`）的口径差异，证据等级升至“可下载实现 + README 自报”，未升 P2/P3。
- **补写 [[01_theory/01_models/moonshot_kimi/22_kimi_k3_architecture_deepdive]] §六 SiTU（2026-07-28 追加）**：把原来 5 行的条目扩成完整一节，并新增自绘四联图 `assets/kimi_k3_fig_situ_range.png/.svg`（按报告 §2.3.2 与 Appendix B 公式数值绘制，非复制 Fig. 4）。补出报告 Appendix B 的设计目标原文（bound the SwiGLU product **without discarding the characteristic shape of Swish**——保住原点近线性与消失负尾）、只 cap 线性因子而保留 sigmoid 的理由、Eq. 18 的一阶等价与 Eq. 19 的界，以及 hard clamp 被否决的原因（饱和边界外梯度归零）。**值域主结果**：预激活不变；门支 `(−0.2785,+∞) → (−0.2698, 4)`（下确界只动 3%，cap 实际只作用于正半轴）；up 支 `ℝ → ±25`；输出 `ℝ → (−100,100)`。另加三条本库推算：四角点显示 ±100 两端都由门支饱和到 4 驱动、门支负半轴对输出量级贡献上限仅 6.74；cap 保留线性值的比例只依赖 `z/β`（0.25β→98.0%、1β→76.2%、2β→48.2%），据此说明 `β₁=4` vs `β₂=25` 意味着门被管得比 up 严约 6 倍（取值理由报告未给，标 [推断]）。
- **证据边界（显眼的缺席）**：K2 有“15.5T tokens 零 loss spike”，**K3 报告没有等价陈述**——无训练 loss 曲线、无 spike 计数、无容错章节；全文唯一 spike 级实测证据是 Fig. 6 的 MoonViT 梯度范数对比。因此“K3 稳定性机制更系统”可说，“K3 训练更稳定”不可说。

---

## 2026-07-28：Kimi K3 技术报告回填后训练统一学习域

**Type**: Source Ingest + Industrial Case Study + Cross-Document Correction（固定官方报告 `0797decb`，将算法、trajectory、environment、Infra 与部署精度放回同一个 `wiki/03_posttraining/` 闭环。）

- **原始来源**：新增 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_Technical_Report_2026-07-28.md`，SHA-256 `fd6ee35c07766a5eb6104235f1b407e4329f969e3482b8c42937c7b5f2b3efe1`；来源台账补 §4.1、§4.2、§5.3 与 Appendix F 的精确定位。
- **新增 D12 `03_posttraining/12_kimi_k3_posttraining_case_study_analysis`**（历史活链接，已于 2026-07-31 因 kb-reorg P5 迁移为 [[24_kimi_k3_posttraining_case_study_analysis]]，按"历史不回写"惯例降级为反引号）：串起 SFT → 九个 domain/effort 专家 → MOPD，澄清 partial rollout 保留 prompt 内 $K$ group，并分析 white-box environment、XTML preserved thinking、MXFP4/MXFP8 QAT、draft model、external KV pool 与 AgentENV。
- **回填统一主线**：更新 D00–D05 与 D11；把 K3 作为项目级工业案例，而不是没有训练源码证据的“第五个开源框架”；D00 与领域/全局索引扩展为 D00–D12 连续编号。
- **修正事实边界**：量化 scheme 一致只消除该维度 TIM；K3 partial rollout 不是 fully async；Figure 8、MOPD、GRM、external KV 与 AgentENV 均保留未披露超参数、消融或运行条件。
- **K3 旧页同步**：总览、架构、Infra 与 Moonshot 索引从“报告/权重待发布”更新为 2026-07-28 固定报告，并把后训练机制统一链接到 D12。

---

## 2026-07-28：PyTorch 图编译系列源码级重构与审计闭合

**Type**: Source-faithful Refactor + Final Audit（固定源码基线为 PyTorch
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`；本轮冻结演示扩展，集中完成原理与源码机制说明。）

- **建立编号化学习主线**：形成 `00` 总索引与 `01–21` 正文，按“图语义基础 → 捕获与
  AOT 正反向构图 → 安全改图与 PatternMatcher → Inductor IR/Scheduler/Codegen”
  顺序组织；总索引新增源码阅读方法与各阶段 driver 入口。
- **补强源码机制**：所有正文均从概念解释推进到真实调用链、读写状态、不变量、
  consumer、设计取舍和参数化复杂度；重点闭合 FX use-def、AOT fw/bw runtime ABI、
  saved tensor/recompute、PatternExpr 匹配、DCE/稳定拓扑、lowering/realization、
  Scheduler dependency/fusion 以及 codegen/autotune/cache 边界。
- **纠正关键误解**：fw/bw 是两张独立 FX Graph，没有跨图 Node 边；recompute 是
  partitioner 向 bw fresh graph 复制节点；PatternMatcher 按注册 root 候选索引匹配；
  DCE 必须结合副作用判定；Scheduler topo 与 fusion candidate 复杂度均按真实实现参数化，
  不再笼统写成“逐 pattern 扫整图”或“必然二次”。
- **课程证据闭合**：3134/3134 个课程 claim decisions，0 个 validation error；
  916 个固定源码证据、366 个当前环境实测、1284 个带已验证父结论的推论，41 个环境阻塞
  保持显式，不把 generated-only 或 blocked 能力升级成 native runtime 事实。
- **历史材料隔离闭合**：28 篇旧页完整保留；2190/2190 个历史 claim 有处置，
  91 个纠正，2099 个未证实结论全部 `retain-quarantined`；新课程未导入任何 unresolved
  history claim。
- **验证结果**：审计工具 90 项测试与课程合同 42 项测试均通过，21/21 个既有
  runtime producer receipts 成功；本机缺少 MSVC/CUDA/Triton，原生 C++ kernel 与
  CUDA/Triton autotune 仍保持 `BLOCKED`，没有扩展演示 demo。

---

## 2026-07-27：完成 LLM 后训练 D02–D11 算法、Infra、框架源码与 CUDA–Ascend 深挖

**Type**: Deep Dive + Source Audit（在统一 `wiki/03_posttraining/` 中完成 S01–S05，固定论文版本和四框架 commit，贯通算法统计语义、在线数据、系统与硬件。）

- **算法与数据语义**：新增 D02–D04，比较 GRPO、DAPO、Dr. GRPO、GSPO、SAO，定义 Agentic trajectory/reward/credit schema，并严格拆分 system async、policy lag、off-policy 与 training–inference mismatch。
- **Infra 与主框架**：新增 D05–D07，以 control/data/weight 三平面建立工业机制模型；完成四框架矩阵，并从 verl `main_ppo.py` 追踪 `RayPPOTrainer.fit`、`DataProto`、advantage、actor update 与 rollout weight refresh。
- **三个源码对照**：新增 D08–D10，固定 slime `aaf5c20`、AReaL `b23fa6c`、ROLL `370cb24`；澄清 TransferQueue 不属于 slime core，分别追踪 slime async producer、AReaL freshness/weight services、ROLL Strategy/device mapping/NPU platform。
- **CUDA–Ascend**：新增 D11，按 device、collective、train/rollout、weight sync、kernel、dynamic shape、profiling 和版本矩阵建立 M1–M4 迁移验收；纳入 2026-07-27 的 vLLM-Ascend、SGLang NPU、MindSpeed 与 HCCL 公开进展。
- **证据与导航**：新增 `docs/research/2026-07-27-posttraining-source-ledger.md`；D00、D01、领域索引和全局索引更新为 D00–D11 顺序入口；旧 GRPO/DAPO/GSPO/verl/infra/sandbox 页面增加统一领域回链。

---

## 2026-07-27：建立 LLM 后训练统一纵向学习域与 S00 前沿快照

**Type**: Research Baseline + Learning Route（将 Reasoning RL、Agentic/Coding RL、Infra、工业框架源码与 CUDA/Ascend 映射纳入同一研究闭环。）

- **新增统一入口**：建立 `03_posttraining/index`（历史活链接，该域已于 2026-07-31 因 kb-reorg P5 解散并删除，S00–S05 阶段叙述并入 [[courses/posttraining_frontier]]，按"历史不回写"惯例降级为反引号），后续 D00–D11 新研究统一写入 `wiki/03_posttraining/`；旧理论与工程页面保持原位，通过链接复用，不再把 RL 算法和 Infra 分散承载。
- **新增 D00 `00_posttraining_source_reading_guide`**（历史活链接，该页已于 2026-07-31 因 kb-reorg P5 解散删除，阅读路线骨架 + 六级能力门槛并入 [[courses/posttraining_frontier]]，按"历史不回写"惯例降级为反引号）：按 D00 → D11 固定推荐阅读顺序，定义 S00–S05 六个研究阶段、六级可验证能力门槛，以及论文、源码、工业“支持等级”和 CUDA→Ascend 适配的阅读方法。
- **新增 D01 `01_posttraining_frontier_map_analysis`**（历史活链接，已于 2026-07-31 因 kb-reorg P5 迁移为 `posttraining_frontier_map_analysis`、同日 Task 8 再编号为 [[01_posttraining_frontier_map_analysis]]，按"历史不回写"惯例降级为反引号）：以优化粒度、on-policy/freshness、训练—推理一致性和 Agentic 环境四组张力组织前沿地图；固定 verl `983cb0f`、slime `aaf5c20`、AReaL `b23fa6c`、ROLL `370cb24` 的 2026-07-27 源码快照。
- **研究分工**：verl 作为主基线，slime 作为性能/前沿对照，AReaL 作为 fully async/Agentic 对照，ROLL 作为多后端、异构和 Ascend 专项；不使用脱离模型、硬件、配置和 freshness 条件的总榜。

---

## 2026-07-26：图编译知识体系重构复核——课程主线通过，历史无损迁移仍未验收

**Type**: Source-faithful Refactor + Design Conformance Review（不删除旧页；固定源码审计基线为 PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`。本机 Lab 使用 PyTorch `2.9.1+cpu`、`torch.version.git_version=5811a8d7da873dd699ff6687092c225caffcf1bb`，两条基线分开记录。）

- **重审 `00_pytorch_graph_series_index`**（该页已于 P4 Task 10 删除，导读价值并入
  [[courses/torch_compile_end_to_end]]）**与 21 篇专题**：Part I 建立图 IR、FX 数据模型、值/metadata/signature、符号形状、effect/alias/mutation、结构化输出与 higher-order graph；Part II 解释捕获、规范化、AOTAutograd joint→fw/bw、saved tensor/recompute ABI 与跨阶段 provenance；Part III 解释 FX 改图原语、`PatternExpr`、DCE/稳定拓扑排序、pass 顺序/fixpoint、合法性与复杂度；Part IV 贯通 FX lowering、Inductor IR、buffer/liveness、Scheduler 与 codegen/autotune。21 篇正文的 323 个完整 repository-relative `file:line` 定位在固定 checkout 上全部路径存在且行号有效。
- **可执行证据升级**：Lab 目录现有 18 个机制/贯穿脚本和 1 个 9-test 自动合同入口；合同覆盖四种捕获、FX 不变量、effect/alias/DCE、AOT joint/fw/bw/recompute、PatternMatcher、pass/topology、Part III rewrite、Part IV IR/Scheduler/provenance 与统一模型 bundle。AOT Lab 现用同次 partition 的 lab-only origin token 建立 joint→fw/bw 精确 old-to-new 映射，并把 saved-slot fw value/bw placeholder 绑定到同一 joint origin；artifact manifest 明确整体 continuity 为 `partial`，不再把独立前端捕获和独立 backend 捕获写成一条单次编译链。本轮复跑 `Ran 9 tests`，结果 `OK`；审计工具 8 项测试也全部通过。
- **Part III 贯穿改写**：`add(matmul(x, weight), bias) → addmm(bias, x, weight)` 仅在 rank、dtype、shape/无 broadcast 等合法性成立时执行；数值、一阶梯度、`gradcheck`、shape、alias relation、输入 mutation relation、非法 broadcast 拒绝、失败原子性与第二次运行零改动均有 assertion。
- **Part IV 证据边界**：真实执行 GraphLowering、Scheduler、dependency/fusion/reorder、external matmul 与 `eigvals` fallback；custom lowering 到达 `ComputedBuffer`；生成 wrapper/C++ source 并完成 Scheduler→FX→Python provenance join。当前 Windows CPU 缺 MSVC `cl` 且无 CUDA，因此 native pointwise/reduction kernel、真实 fusion 性能、物理 allocator peak 与 Triton autotune 没有实测；mock/no-op 捕获的 codegen 产物明确标记为 generated-not-executed，Scheduler group 数也不再误写成 native kernel 数。
- **迁移与导航**：28/28 篇 manifest 页面均有“页面角色与审计状态”说明并保留全文；其中 19 篇旧主干页面不再整页 blanket deprecated，另 9 篇 runtime/checkpoint/cache/memory 专题补齐独有职责、基线和未闭合边界。综合报告增加旧节→新系列去向表；同步更新框架总索引、AOTAutograd、Inductor、FX/export、runtime memory 和 compile cache 六个索引，并补齐相关回链。
- **历史审计 gate 仍打开**：manifest 已扩为 28 页、2,514 条 inventory records、2,022 条 heading/code/locator ledger rows，围栏不平衡为 0；但仍有 832 个 `TBD` destination、1,041 个 unresolved-like row（1,030 `unresolved`、1 `not_semantically_audited`、10 `needs_manual_resolution`）、0 个真实 destination anchor。故当前新课程可作为固定基线下的已验证主线使用，但整个重构不能宣称满足“历史资料逐结构单元无损迁移”或 authoritative 发布门槛。完整结论见 `docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`。

---

## 2026-07-25：训推不一致（TIM）因果链——第 1 块首页与 raw 摄入清单

**Type**: Source Ingestion + Deep Dive（严格路线：所有断言带一手来源与 §/Fig/Table/Eq 级定位符；未核实项显式标注，不做推测补齐。）

- **新增 `tim_causal_chain_analysis`**（515 行，历史活链接，已于 2026-07-31 因 kb-reorg P5 Task 8 再编号为 [[26_tim_causal_chain_analysis]]，按"历史不回写"惯例降级为反引号）：打通本库此前断掉的一环——「kernel 非确定性 → logprob 偏差 → 重要性比方差放大 → 训练崩溃」。上游（浮点非确定性、batch 不变性）已由 [[10_determinism_and_numerical_reliability_analysis]] 覆盖，下游（loss spike 治理）已由 [[12_training_dynamics_stability_analysis]] 覆盖，本页补中间两环与算法侧修法全谱。
- **归因框架**：采用 Qwen 2512.01374 §2.4 的二因子分解（训推数值分歧 × 策略陈旧度）作为全页坐标系，据此把 PPO clip、TIS/TRM/ALP/MIPU、batch-invariant kernel、TBIK、FP16、MoE 路由回放归位到各自作用的因子上。
- **六类病因逐条落定位符**：引擎间 kernel 差异、batch 触发不同 tiling、**TP size 改变累加顺序**（本库此前未覆盖）、BF16 尾数不足、MoE 路由分歧、量化 rollout。其中 TBIK Table 1/2 的对照实验证明「只做 batch 不变，对跨 TP 数值发散几乎无效」（Llama-3.1-8B 上 BIO 的 27.54 甚至高于 BF16 的 26.48）。
- **崩溃形态学**：首次在本库区分 recomputation 与 bypass 两条路径的不同崩溃曲线（多阶段 vs 单阶段、是否伴随 loss spike），并记录一条对工程有直接影响的发现——**K1/K3 KL 对 recomputation 型崩溃是盲的**（前 700 步几乎平坦而 reward 已在退化），而 recomputation 正是 verl 等框架的常见默认路径。
- **确定性税汇总**：batch-invariant GEMM 194 vs cuBLAS 527 TFLOPS；单个确定性请求混入 11 请求批次使整批吞吐掉 56%；TBIK 端到端 22%–63%；vLLM+TorchTitan bitwise 一致 RL run 慢 2.4×。并指明三篇系统侧论文**无一测量 RL 闭环端到端代价**，列为头号 open question。
- **三处 `> [!contradiction]` 标注**：① TIM 根因是精度（2510.26788）还是优化（2602.01826）——两篇互相点名，本页据 $C\cdot T^2$ 的乘积结构论证二者数学上不互斥；② MoE 的 Routing Replay 是否必要——GSPO §5.3 主张可取代 vs 同作者组 2512.01374 §4.4 回到「必需」，并如实标注后者**未点名**前者；③ 确定性是否必须付性能代价——实测派 vs 2606.00279，本页指出后者目标是可审计而非不变性，且「无性能代价」这一标题级主张零测量。
- **两处 raw 原文核实**（直接读 `raw/` 中已有 PDF）：DeepSeek-V4 §3.3 的 dual-kernel strategy 抵消的是**放弃 split-KV 后 decoding attention 的 wave-quantization 损失**，不是 matmul 固定 tiling 的损失（matmul 侧是 DeepGEMM 端到端替换 + 放弃 split-k 后另做优化）；GSPO §5.3 的 Routing Replay 表述与 10% 专家漂移数字逐句核对。
- **更新 [[01_theory/04_posttraining/index]]**：新增「训推一致性（TIM）与 RL 稳定性」小节；**首次建立 Knowledge Gaps 节**，记录 9 条确认无一手来源的缺口（崩溃阈值、尾部刻画、逐位置增长曲线、极端 token 频率、重尾→熵坍塌因果、RL 闭环确定性税、VeXact 开销、RL 阶段路由坍塌、vLLM logprobs RFC）；标注三条旧目录结构下的失效链接并给出替代。
- **新增 `raw/_ingest/INGEST_MANIFEST_block1_tim.md` 与 `fetch_block1_tim_sources.ps1`**：29 项已核实来源（26 篇 PDF + 3 份官方文档）的摄入清单与下载脚本，含目标路径、arXiv ID、一句话定位、保真度标注，以及 7 项需浏览器手动存档的博客/issue。**容器侧 arXiv 被代理阻断，PDF 需在本机执行脚本落盘。**
- **Mermaid 三块**：均按 CLAUDE.md 清单逐条自检，并用 mermaid-cli 实渲通过。

---

## 2026-07-23：FX Graph 构图、AOT 正反向分图与 PatternMatcher 改图机制报告

**Type**: Deep Dive（按 PyTorch `ea5655fcebf` 固定基线核对 FX、AOTAutograd 与 Inductor 源码。）

- **新增 `fx_graph_construction_and_transformation_analysis`**（历史活链接，该页已于 2026-07-30 判重删除，AOT 特有内容并入 [[11_aotautograd_joint_forward_backward_graphs_analysis]]/[[12_saved_tensors_recompute_and_runtime_abi_analysis]]，FX 数据模型部分此前已并入 [[10_fx_graph_core_data_model_analysis]]，按"历史不回写"惯例降级为反引号）：统一解释 `Graph` 侵入式双向链表、`Node.args/kwargs`、`_input_nodes/users` 与查找辅助表，区分图序、依赖边和 `GraphModule.forward` 生成代码。
- **补全 AOTAutograd 正反向构图**：从 joint function tracing、partition 分类与子图抽取，解释 fw 额外输出 → runtime context → bw placeholder 的跨图 ABI；明确 fw/bw 无对象级 Node 边。
- **补全 recompute 机制**：说明 min-cut 保存/重算选择、普通 forward 节点复制进 bw，以及 backward recompute reorder 如何缩短临时值生命周期。
- **补全 PatternMatcher、DCE 与保序**：覆盖 PatternExpr 子类的图场景、候选桶与逆序匹配、三类 PatternEntry、mutation/stream 边界、dead node 定义、稳定拓扑排序、lint/recompile 检查点。
- **复杂度模型**：给出构图、候选检索、匹配/替换、DCE、lint、稳定拓扑排序、fw/bw 抽取、min-cut 与全 pass 管线的参数化复杂度；同步更新 AOTAutograd、Inductor、FX/export 索引与双向链接。

---

## 2026-07-22：Torch Compile 八阶段 Pass 开发方法论与 Dynamo/Codegen 缺口补齐

**Type**: Deep Dive + Correction（按 PyTorch `9922478dffa` 固定基线复核并补齐 pass 放置方法、关键 API、注册示例与阶段注意事项。）

- **方法论升级**：重写 `fx_pass_optimization_methodology`（2026-07-30 起并入 [[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]]，详见该日期 changelog 条目）的权威主干，完整覆盖 Dynamo → Pre-Grad → AOT/Decomposition → Joint → Post-Grad → Lowering → Scheduler → Codegen；每阶段均回答“是什么、为什么、适合做什么、为什么不放相邻阶段”，新增选择表、放置规则、六问设计法和验证矩阵。
- **补齐三块缺失内容**：新增 [[30_dynamo_pass_methodology]]（backend callable/`register_backend` 边界）、[[33_decomposition_passes_guide]]（decomp table、AOT 注入位置、注册/选择方法）、[[34_codegen_extension_guide]]（`BaseScheduling` + Wrapper + `DeviceOpOverrides` + `register_backend_for_device`）。
- **三阶段 Pass 指南纠错**：[[30_pre_grad_passes_guide]] 订正 non-functional/non-normalized IR、真实执行顺序、`pre_grad_custom_pass(Graph)->None` 和缺失 `PatternMatcherPass` import；[[31_joint_graph_passes_guide]] 订正 `pass_patterns` 所属模块、两轮顺序、Graph hook 契约和“空 hook 确保加载”错误；[[32_post_grad_passes_guide]] 补真实全流程、三轮 pattern、inference-aware hook、通信 bucketing 与 reinplace 尾部不变量。
- **Lowering/Scheduler/Codegen 纠错**：`lowering_analysis`（历史活链接，该页已于 2026-07-30 判重并入 [[10_fx_lowering_to_inductor_ir_analysis]]，按"历史不回写"惯例降级为反引号）订正“Post-Grad 在 Lowering 之后”的错误顺序并补 `register_lowering`/fallback API 示例；`scheduler_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]，按"历史不回写"惯例降级为反引号）明确真实接口是 `_pre/_post_fusion_custom_pass(list[BaseSchedulerNode]) -> list[...]`，将 `GraphLowering`/`node.fusable` 旧示例标为 deprecated；[[20_inductor_codegen_analysis]] 更新固定基线入口并链接完整扩展指南。
- **动态形状方法修正**：把“遇到 SymInt 一律跳过”改为“符号恒等或 ShapeEnv/guard 可证明则支持，无法证明才拒绝”。同步更新 Dynamo/Inductor 索引与交叉链接。

---

## 2026-07-22：CUDA GEMM / 非 GEMM / Ascend 算子三篇生产级 Kernel 资料入库

**Type**: Source Ingestion（按 `llm-knowledge` 的 raw→wiki→index→backlink→changelog 流程导入用户提供的三份 HTML。）

- **原始资料归档**：新增 `raw/02_engineering/05_gpu_kernel/{cuda_gemm_final,cuda_nonmatmul_kernels_final,ascend_kernels}.html`，保持下载文件字节不变；SHA-256 分别为 `56f589…f85b0`、`ba1ce1…db02`、`1a6b9c…f7f0`。
- **新增 [[20_cuda_gemm_kernel_analysis]]**：以 SM80 / A100 代表性配置串起 Grid→CTA→Warp→MMA、M/N 空间切块与 K 时间归约、`cp.async` 完成语义、每线程约 232 寄存器账本、shared-memory epilogue 与生产级 kernel 骨架。
- **新增 [[21_cuda_nonmatmul_kernels_analysis]]**：以 roofline + 五类数据依赖为统一分类，覆盖 elementwise、reduction、norm、FlashAttention、stencil、scan、gather/scatter/sort，明确 shape 会让同一算子跨 compute-/memory-/latency-bound 阵营。
- **新增 [[22_ascend_kernel_execution_model_analysis]]**：把 CUDA 两篇映射到 DaVinci AI Core 的 Cube/Vector/Scalar/MTE、GM→L1→L0→UB 显式缓冲链、Queue 双缓冲、FixPipe，以及 compute / memory / communication 三条训练优化主线；明确其为平台对照材料、非官方文档。
- **可核验性与图形**：三页逐章附 raw HTML 行号范围和快照哈希；19 个内嵌 SVG 已渲染为 PNG。更新 GPU Kernel、工程与总索引，并向 [[01_gpu_kernel_guide]]、[[10_cuda_execution_model_guide]]、[[triton_12_matmul_guide]]、[[11_operator_optimization_guide]]、[[13_mindspeed_ascend_affinity_analysis]]、[[21_npu_inductor_optimization_analysis]] 补回链；未发现需标记的既有内容冲突。

---

## 2026-07-20（四次更新）：[[22_npu_fusion_passes_deepdive]] §5 后端级融合大幅展开——加「决策链 + 代价模型」

**Type**: Update（应用户「§5 后端 pass 需展开:当前后端优化都做了哪些?怎么建模选择最终的融合方式?」。1 路 source-audit agent 核 scheduler/select_algorithm/tiling 决策链 + 本人抽验 7 处载荷 file:line。）

- **§5 从 3 小节（CATLASS/DVM/can_fuse）扩为 8 小节**：新增 §5.0 决策链（mermaid：后端路由→合法性→排序→收益实测→tiling / GEMM 分支 autotune→EVG）、§5.1 后端路由（`choose_node_backend` + `TORCHINDUCTOR_NPU_BACKEND`）、**§5.3 融合收益建模（核心）**、§5.4 GEMM 实现+epilogue autotune、§5.5 tiling 编译期穷举、§5.7 四类建模范式小结。原 can_fuse/CATLASS/DVM 归位重编号。
- **核心回答「怎么建模选最终融合」**：三段式=**启发式定合法性 + 实测定收益/实现 + 编译期穷举定形状**。真正的实测代价模型只有两处——③收益 `speedup_by_fusion` 编译 fused/unfused 真机实测 `ms_fused < ms1+ms2`（`scheduler.py:541`）；④GEMM 多模板 `finalize_as_caller` 选最快实现（`:440-441`，同时决定融不融与选哪个实现，真机 AICore profiling `do_batch_profiling`）。tiling 靠 UB 公式 `max_numel_threshold=ub_size//ptr//dtype` 编译期穷举（`tile_generator.py:47-48`）。
- **两处源码校正**：① `score_fusion`/`score_fusion_memory` **NPU 未覆写**（全库无 `def score_fusion`），NPU 只改邻近门 `are_long_distant_nodes`（64→20，仅 A5）——不能说「NPU 自定义融合打分」；② AKG 在本 baseline **实际停用**（mfusion 路径警告 "not supported currently"），不写成活跃后端。两个实测开关（`CATLASS_EPILOGUE_FUSION`、`TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING`）默认均关。
- 7 处载荷 file:line 已抽验（`scheduler.py:541/440-441/202-203`、`tile_generator.py:47-48`、`npu_combined_scheduling.py:40-43`、score_fusion 无覆写、`are_long_distant_nodes:39`）；新 mermaid 过本库规范校验。
- **追加（应用户「default 路径下走 CATLASS 分支怎么判？直接白名单吗」）**：§5.1 补「CATLASS 分支三时刻路由链」——**不是直接白名单**。① lowering 时 `tuned_mm` 过外层 3 门（连续性/非零/`use_catlass_template`）+ 内层 6 门（白名单 `mm/addmm/bmm` + size 阈值 + 非 ROCm + dtype + `use_max_autotune` + backend/库），过了才 `add_catlass_gemm_choices` 展开一批候选（`kernel/mm.py:79-135`、`utils.py:236-265`、`gemm_template.py:189,227-247`）；② autotune 真机实测选中，胜出才产 `CATLASSTemplateBuffer`；③ scheduler `is_catlass_template` 仅类型 dispatch（`catlass_scheduling.py:70-73`）。结论：白名单只是资格门之一，最终判据是 autotune 实测；无 max-autotune 时根本不生成 CATLASS 候选。所有 file:line 本人实读。

---

## 2026-07-20（三次更新）：工业界 FX Pass 全景——上游全集 + vLLM/SGLang 现状 + 开发方法论（4 页，3 路并发 agent）

**Type**: Deep Dive（应用户「是否有 upstream pass 分析全集?补一个 torch_upstream_pass_deepdive;vllm/sglang 是否也有大量 pass?并发总结;并归纳 pass 优化开发方法论」。3 路并发 source-audit agent 逐仓核源：upstream `9922478dffa`、vLLM `97a98006b0`、SGLang `d6ef68881e`。）

- **新增 `torch_upstream_pass_deepdive`**（04_inductor，2026-07-30 起判重删除、机制层并入 [[22_pattern_expression_and_matcher_engine_analysis]]/[[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]]，详见该日期 changelog 条目）：上游 Inductor pass「全集 + 机制」总纲，补上三份 stage 指南缺的**机制层**——`PatternMatcherPass` 声明式引擎（声明→trace→匹配→改写）、三种 `PatternEntry`（lowering/graph/replacement）、`fwd_only`/`joint_fwd_bwd` 两种 trace、序列化 pattern 缓存（30 个 `_sfdp`）、三阶段驱动器 + `*_custom_*_pass` 钩子 + `GraphTransformObserver`，及 pre/joint/post_grad 全集目录。6 处载荷 `file:line` 已实读复核（`PatternMatcherPass.apply:2352` + 跨 mutation/stream 护栏 2400/2409、`LoweringPatternEntry.apply:1156`↔`graph.py:1372-1376` passthrough、序列化缓存 import 1973-1981、post_grad 三桶 85-89、`_sfdp_init:1487`）。
- **新增 [[sglang_compilation_passes_analysis]]** + 新建 `sglang/` 目录与 index：**核心发现——SGLang `srt/compilation/` 是 vLLM piecewise-cudagraph 管线的近逐文件 fork，但融合 pass 被整个抽空，真实图重写 pass 数=0**；唯一的 `FixFunctionalizationPass.__call__` 是 no-op（`fix_functionalization.py:34-37` 只 `count+=1`，`pass_manager.py:47-51` `configure()` 只挂它、`passes` 恒空）——两处均本人实读复核。含两条 compile 路径、split_ops 切图、CUDA/NPU/XPU piecewise backend 差异。
- **更新 [[25_vllm_ir_and_fusion_passes_analysis]]**：新增 §3.5「Pass 全家福 + 三大融合维度」——回答「vLLM 有没有大量 pass：有，约 23 个」（16 融合 + 6 IR/utility + 1 pre-grad），建在 torch `pattern_matcher` 上，朝**厂商 kernel**（FlashInfer/cutlass/symm_mem/AITER）融合，三维度 upstream 没有：**集合通信 / 量化 / KV-cache 写入**；补 pre-grad 钩子 + `compile_range` 门控 + 三种 pattern 注册形态。
- **新增 `fx_pass_optimization_methodology`**（04_inductor，2026-07-30 起判重删除、内容并入 [[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]] §14 等处，详见该日期 changelog 条目）：跨四家归纳的 pass 开发方法论——四家现状对照表、四个决策问题（在哪做 Q1 / 匹配什么 Q2 / 怎么落地 Q3 / 怎么保证对 Q4）、**融合朝向谱系**（codegen→厂商库→预融合 kernel，解释 vLLM 写十几个而 SGLang 一个不写）、工业界工程护栏（uuid=源码 hash / 可 bisect / 门控分层 / pattern 对 custom_ops 鲁棒 / 序列化缓存）、开发 checklist、反模式（搬框架≠搬融合、pass 会腐化、跨类别改写会错）。
- **联动**：04_inductor/index 加两页；03_infer_frameworks/index 加 SGLang 子框架行；各页 Related 交叉链接。

---

## 2026-07-20（二次更新）：[[22_npu_fusion_passes_deepdive]] 新增 §7「改图操作原语与 pass 通用原理（机制总纲）」

**Type**: Update（应用户「补充这个改图的操作，以及 pass 主要操作的原理是什么」——承接对 `view_fold_pass` 的连续追问：多个 view 如何变一个、如何保证等价、DAG 扇出/多后继怎么处理。）

- **新增 §7（机制总纲）**：把 26 个 pass 共用的改图机制抽出来单讲。§7.1 **FX 改图操作原语表**（`replace_input_with` 边局部 / `replace_all_uses_with` 全局 / `call_function` / `inserting_before` / `erase_node` / `propagate_fake_tensor` / `eliminate_dead_code`，全文 132 处调用，每行带代表 `file:line`）+ 数据模型（`node.args` 与 `node.meta["val"]` FakeTensor）；§7.2 **pass 四步通用套路**（定位→改写[指针重接 | 造等价新子图]→维护 meta→DCE 清理）；§7.3 **`view_fold_pass` 全走查**（t0/t1/t2 拓扑序传递塌缩 + A/B/C/D DAG 扇出的边局部安全 + 等价性论证）；§7.4 **三条贯穿原理**（纯函数⇒边局部安全 & 单用户门槛判据、拓扑序⇒一趟塌缩、等价来自算子类别不变式、meta 一等公民 + 静态 shape 门槛）。
- **核心结论**：「多变一」= 指针重接让末端算子直连源头、中间节点变孤儿再 DCE，**不是生成合并算子**；view_fold **不需要单用户前提**（边局部改写对 DAG 扇出天然安全），而 `fold_cat`/`fold_squeeze` 因会改动前驱本身才查单用户（`:287`/`:580`）。
- §1 加「机制总纲建议先读 §7」前向指针；§3 `view_fold_pass` 表行加「全走查见 §7.3」。§7.1 全部 `file:line` 经 grep/直读复核。
- **补图**：§7.3 增 before/after mermaid 图直观展示 `replace_input_with` 的**边局部**改写——处理 B 时只把 B 的入边从 A 改指 x（橙色），`C→A`/`D→A` 两条边纹丝不动、A（绿色）因 D 仍引用而存活；对照 `replace_all_uses_with` 的全局替换。图经本库 mermaid 规范校验（无嵌套定界符/管道标签合规）。

---

## 2026-07-20：新增 [[22_npu_fusion_passes_deepdive]] —— 自定义融合 Pass 逐个深挖（场景·问题·优化·效果）

**Type**: Deep Dive（应用户反馈「[[30_npu_vs_upstream_fusion_passes]] 对每个 pass 的 why/场景/效果讲得不够——需要具体代码场景、为什么这么优化、优化带来什么效果」。三路并发 source-audit agent 逐函数读 `ascend_graph_pass.py`(2548 行)全部 26 个 pass + helper。）

- **新增 [[22_npu_fusion_passes_deepdive]]**：对 26 个自定义 pass（4 PRE + 22 POST）+ 3 个后端融合机制（CATLASS EVG epilogue / DVM 图级分区融合 / `NPUTritonScheduling.can_fuse`）逐个给「**触发场景（含 before 代码）→ 待优化问题 → 优化机制（after）→ 效果**」四拍，每条带 `file:line`。重点展开 9 个「真·融合」pass：`masked_add_compose`（互补掩码相加→单 where）、`bool_cast_mul_to_where`（bool cast×→where）、`sign_diff_hamming_fuse`（符号位汉明距离 6 算子链→gt/gt/ne/sum）、`batch_embedding_fusion`（N 段 embed+reduce→单次 reshape→embedding→reduce）、`cat_to_view`/`repeat_to_expand`（cat/repeat→零拷贝 view/expand/roll）等。
- **效果口径诚实边界**（核心方法论）：逐函数 grep 确认**全文除 CATLASS `catlass_epilogue_fusion_counter` 外无任何计数器/benchmark**，故所有效果均为**结构性**（少 N kernel / 少一次拷贝 / 转零拷贝 / int32 索引），非实测加速；多个 pass 带**中文 docstring 直述动机**（标「原文」，可信度最高），而源文件**无硬件注释**，凡「因为达芬奇/UB/i64」因果均标 [硬件推断]；`fold_four_op`/`fold_where` 经 `get_binary_fold_result` 会留一个 clone、非零成本。
- **校正**：`fusion_attention_v3_pass` 是 **基础版 `npu_fusion_attention.default` → `v3.default`**（本 baseline 无 `_v2`），非「v2→v3」——同步订正 [[30_npu_vs_upstream_fusion_passes]] §6 措辞。
- **联动**：[[30_npu_vs_upstream_fusion_passes]] §3.4 加深挖页指针、Related 补链；更新 [[04_inductor/npu/index]] deep dive 表。

---

## 2026-07-17（五次更新）：新增 [[30_npu_vs_upstream_fusion_passes]] —— torch_npu vs 上游 Inductor 融合 Pass 全流程对照

**Type**: Deep Dive（应用户「总结为一个新页，从 FX pass 到后端全流程比较 torch_npu 与 upstream 融合 pass 的差异、谁有谁无及原因，稽核源码」。逐行核验两套 checkout：torch_npu `b3c8a815b`(v2.7.1) + upstream `9922478dffa`(main)，三路并发 source-audit agent + 主体自审。）

- **新增 [[30_npu_vs_upstream_fusion_passes]]**：主线「torch_npu 不重写上游 pass 管线，而是三处介入 + 重活下沉后端」。逐层对照 pre_grad / joint_graph / post_grad / lowering-decomp / 后端 scheduler；三张「谁有谁无谁不同」总表；`is_gpu`/`GPU_TYPES` 门控如何决定上游 pass 在 NPU 上跑不跑（`patch_is_gpu` 追加 `"npu"`，但硬编码 `.is_cuda` 的 pad_mm/b2b_gemm/decompose_mm 仍不跑）；26 个 `ascend_custom_passes` 全清单（4 PRE + 22 POST，仅推理）；根因收敛到 Cube 专用单元 / ACLNN 手工库 / 达芬奇布局约束 / 集成方式四条。
- **源码级校正（source-wins）**：① `patch_pattern_mm_plus_mm` 是**删除**上游 `mm_plus_mm`（注释「torch_npu does not support」）而非添加——纠正 [[01_npu_compile_paths_overview]] §2.5 旧记法；② fallback 实测 **932**(340+592) 而非旧「约 963」；③ persistent reduction 内置后端是**阈值门控非恒关**（恒关的是实验性 Linearize 后端）；④ [[21_npu_inductor_optimization_analysis]] §12.4 自定义 pass 清单已过时（`unfold_dual_reduction_pass` 不在此 head，新增 `sign_diff_hamming_fuse_pass`/`batch_embedding_fusion_pass` 等）；⑤ `pattern_match/npu_fusion_attention_graph.py` 在此 checkout **未接线**（无生产代码 import），真正生效的 attention 改写是 v2→v3 的 `fusion_attention_v3_pass`。
- **联动**：更新 [[04_inductor/npu/index]] deep dive 表；为 [[01_npu_compile_paths_overview]] 与 [[21_npu_inductor_optimization_analysis]] 补 `## Related Pages` 反向链接。

---

## 2026-07-17（四次更新）：GDN/KDA 线性注意力从公式到训推融合 Kernel 的完整闭环

**Type**: Deep Dive（承接用户连续追问：QKVABZ、$a/b/z$ 设计、RNN 中的 $t$、chunk size 与 $t$ 的关系、chunk 数学等价性、仿射状态是否必须保序，以及当前训练/Prefill/Decode kernel 融合。）

- **新增 [[20_gdn_kda_linear_attention_analysis]]**：从 $x_t\rightarrow q,k,v,a,b,z$ 开始，逐步解释 raw $a\rightarrow g=\log\alpha$、$b\rightarrow\beta$、$z$ 输出门的职责分离；给出 GDN 标量 decay 与 KDA 逐通道 decay 的五步递推、统一仿射式 $S_t=A_tS_{t{-}1}+B_t$、$C=3$ 展开，以及 chunk 摘要 $(P,R)$ 的保序结合复合。明确纠正“chunk 状态矩阵直接相乘”和“有结合律即可乱序”两个误解。
- **新增 [[21_gdn_kda_kernel_implementation_analysis]]**：训练侧固定 FLA `ccb0ff944cbf`，拆解 autograd chunk forward/backward、gate+cumsum、KKT+solve-tril+W/U、状态 scan、输出和反向重算；推理侧固定 SGLang `7824903417b7`，拆解 QKVABZ 投影融合、Prefill $C=64$ chunk pipeline、Decode fused recurrent 五步、GDN packed-decode 与 speculative verify。明确 SGLang 是推理基线，不用其 forward-only 代码冒充训练反向。
- **原始来源与联动**：新增 raw 快照 `Gated_Delta_Networks-2412.06464v3.pdf`；修正 [[12_kimi_linear_analysis]] KDA 公式中 $S_{{-}1}$ 的下标笔误为 $S_{t{-}1}$；更新 Moonshot/Kimi 与模型总索引，并为 [[22_kimi_k3_architecture_deepdive]] 补充双向入口。
- **后续补充：TND/THD packed 输入**：[[21_gdn_kda_kernel_implementation_analysis]] 新增 §八，明确 TND/THD 的 $T=\sum_iL_i$、外层 batch=1 与 `cu_seqlens` 状态边界；逐段追踪 Megatron-LM `dev@232c478d43ce` 的 `T×1×D → per-sequence CP→HP → 1×T×N×d → boundary-aware short conv/chunk GDN → per-sequence HP→CP`，并解释为何每条 packed 序列必须独立重置 RNN 状态、chunk 不能跨 pack 边界，以及当前 batch、CP 对齐、FLA 与 inference 限制。
- **TND 代码级追踪**：同页 §8.6 补充最小等价伪代码，并把边界隔离落实到三层实现：Megatron `_unpack_sequence` 与逐序列 CP↔HP、FLA causal-conv 的 `bos/eos + boundary_check`、GDN Triton state kernel 的 `N×head` program grid 与每序列独立 chunk-state 槽。

---

## 2026-07-17(三次更新): [[23_kimi_k3_infra_deepdive]] 新增 §四「负载建模」—— 逐模块 roofline 记账与 bound 判定

**Type**: Update(应用户「探讨引入 KDA 后各部分的负载瓶颈与 bound;训练侧按 8K→64K→256K→1M 逐模块计算说明;不压缩合并入库」)

- **推理侧(§4.1-4.4)**:decode 逐部件字节/FLOP 记账——KDA 状态 R/W(~2 FLOPs/B 带宽 bound 但噪声级,Kimi-Linear 模板 40MB/token)、Gated MLA KV 扫描(1M=1.15GB/层/序列,**与论文实测 TPOT 1.84ms@1M 交叉验证吻合**)、LatentMoE 权重扫描(全量 1.49TB/步 ≈ 4.8ms 地板/64 卡,MXFP4 砍半的对象)、EP a2a 同步延迟(~120 次/步,supernode 的真正含义);prefill 三段迁移:冷算 170 PFLOPs@1M → FlashKDA 的"小矩阵效率" bound → 命中 >90% 后变"恢复带宽" bound(50-200ms vs 冷算三分钟)。
- **训练侧(§4.5-4.7)**:8 个模块逐个"公式→代数→判 bound"——专家 GEMM t≈2500/专家 compute bound(与推理相反);EP a2a 16ms vs 27ms/层的重叠余量(无 LatentMoE 则 64ms 反超成主 bound);MLA 注意力 ∝L(256K 交叉、1M 占 78%,373ms vs 27ms/层);KDA 常数但 bwd 状态检查点 32GB/层/序列;AttnRes 1M 显存税 8.6GB/卡;**Per-Head Muon 的 NS 正交化代价 ∝min²×max,按头切片砍 64×**(整矩阵 11 TFLOPs vs 64 片共 0.17 TFLOPs);PP 气泡 (p−1)/m 随 m∝1/L 崩塌 → 拓扑必须 PP→CP 重排;激活显存 1TB/序列@1M 决定 CP 下限。含四档 L 的 binding constraint 矩阵与两个反直觉结论(a2a 随 L 变轻;训推瓶颈迁移方向相反,3:1 同时钉住两头)。
- 全部模型维度假设(d=8192、48+16 层、d_lat=2048、激活 50B)与硬件锚点(H200/H20/H800 公开规格)显式标注 [推断];§五清单新增 6 行负载建模敏感参数,报告发布后代入公式即可整体刷新。原 §四顺延为 §五。
- 后续补充:§4.5 账本总纲下新增「记账单位说明」——"常数/∝L"均指每 token 成本;GEMM 每 token 恒定(按序列算总量仍 ∝L),注意力是唯一 token 两两交互、每 token ∝L 的模块;并注明两个二阶效应(GEMM 利用率、MoE t 判据)与口径稳健性检查。

---

## 2026-07-17(二次更新): K3 结构页吸收 GDN→KDA 单 token 数据流图;总体结构图恢复为官方原图

**Type**: Update(应用户「吸收 gdn-qkvabz-dataflow.svg 并详细解释;总体结构图直接复用原报告的,不要自绘」)

- **[[22_kimi_k3_architecture_deepdive]] 新增 §2.3「单 token 数据流:q/k/v/a/b/z 六路信号如何各司其职」**:融合投影切六路信号的分工表、状态更新五步拆解("误差修正"读法,并给出与 §2.2 闭式公式的代数等价展开)、GDN→KDA 需改的三处(每 head 标量门→Diag 逐通道门、投影/卷积布局、输出门定位),源码定位补齐(`fla/ops/gated_delta_rule/naive.py:31,54` vs `fla/ops/kda/naive.py:30-31` 等);原 §2.3/2.4 顺延为 §2.4/2.5。图源 `assets/kimi_k3_fig_gdn_qkvabz_dataflow.{svg,png}`(图内标注实现基线 SGLang main@78249034,以图注为准)。
- **官方架构图原图可渲染了**:从博客 CSS bundle(`78620784116c2822.css`)抽出全部 89 条 `BlockAttnRes-module` 规则,与官方内嵌 SVG 拼装成自包含 wrapper `assets/kimi_k3_official_arch_render.html`,按暗色主题 2× 渲染为 `assets/kimi_k3_official_arch.png`;[[14_kimi_k3_analysis]] 的总体结构图改用该官方原图,自绘重绘版 `kimi_k3_arch_redrawn.{svg,png}` 删除。原图证实:每个子层是常规残差 (+) 与 AttnRes (w,α) 取回并存,最终 Output 前还有一次 (w,α) 聚合;MoE/KDA 放大面板中的低秩梯形投影清晰可见。

---

## 2026-07-17: 新增 Kimi K3 收录(三页)—— 首个开源 3T 级模型的发布报告 + 结构变化 + 训推 infra

**Type**: Ingest(应用户「kimi3 发布了,总结模型报告 + 结合开源源码分析结构变化点与训推 infra」。**关键事实：K3 权重承诺于 2026-07-27 前发布，完整技术报告尚无明确发布日期**——当前“报告”实体 = 官方发布博客，已快照落 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.{txt,html}`；结构机制的源码证据来自官方声明 K3 所基于的组件仓库，均已实际克隆/打开核验)

- **新页**:[[14_kimi_k3_analysis]](发布总结:2.8T/896选16/1M 上下文,33 项基准全表与口径脚注,官方自报"位列 Fable 5 与 GPT 5.6 Sol 之后",preserved thinking history 等限制)、[[22_kimi_k3_architecture_deepdive]](六大变化点各按动机→机制→证据→为何不选替代:KDA 3:1 混合、Gated MLA+NoPE、AttnRes、Stable LatentMoE+Quantile Balancing、SiTU、2.8T/1M/视频)、[[23_kimi_k3_infra_deepdive]](Per-Head Muon、静态 shape 全平衡 EP、INT4→MXFP4/MXFP8 QAT 演进、Mooncake >90% 命中、KDA prefix caching 进 vLLM 的 PR 链 #27654/#42406、FlashKDA CUTLASS kernel、64+ 卡超节点账)。
- **可读性修订**：结合 K3 官方 Tech Blog（当前公开“报告”）重写三页的主线与长段落，明确区分官方事实、组件证据和推断；将 KDA、AttnRes、Per-Head Muon、MXFP4 容量与 Mooncake 成本公式改为独立公式块并补齐符号说明；价格统一写作 `USD/MTok`，避免美元符号与 Markdown LaTeX 定界符冲突。同时订正发布时间口径：7 月 27 日是权重发布期限，不是技术报告的官方承诺日期。
- **源码核验基线**:Kimi-Linear @`8c1d85e` + HF 48B config/modeling @`e1df551a` + fla @`b328e7c`(KDA 公式↔`fla/ops/kda/naive.py:59-63` 逐行对照;27 层实际 20 KDA:7 MLA;MLA `assert use_nope`);Attention-Residuals @`85e2231`(仓库仅 README+论文,伪代码 README.md:52-91;arXiv 2603.15031v1 全套消融);FlashKDA @`d2ff19a`(CHUNK=16 双 kernel,H20 1.85–2.31×)。
- **图**:9 张入 `moonshot_kimi/assets/`——官方基准图 2 张(PNG 原件)+ 官方内嵌架构 SVG 原件(aria-label 即 "Block Attention Residuals architecture diagram")+ 按官方风格重绘架构主图 + KDA/AttnRes/LatentMoE/KDA-prefix-cache/Mooncake/MXFP4-QAT 自绘深色 SVG(均 2× 渲染 PNG,已逐张目检)。
- **索引联动**:[[moonshot_kimi/index]] 家族表/时间线/论文索引增 K3 与 AttnRes 行,知识缺口标注 AttnRes 已覆盖;[[01_theory/01_models/index]] Kimi 段增 4 行(含补录 12_kimi_linear_analysis);[[12_kimi_linear_analysis]]/[[13_kimi_k2_5_analysis]]/[[11_kimi_k2_analysis]] 增后继回链。
- **待回填**：激活参数、层数、SiTU/Quantile Balancing/Per-Head Muon 精确定义与训练规模；三页中所有 `[推断]` 项须在完整技术报告发布后核对（报告日期未定，缺口清单见 [[23_kimi_k3_infra_deepdive]] §4）。

---

## 2026-07-16: 新增 [[21_aclgraph_multistream_rng_analysis]] —— ACLGraph 多流依赖与 graph-safe RNG

**Type**: Ingest / codebase deep dive（应用户连续咨询，将“多流入图、`wait_stream`/Event、遗漏 join、通信流场景、随机数入图与算子适配”去重后落库）

- **基线**：`torch_npu v2.7.1@b3c8a815`、其记录的 `op-plugin@6ef73e399`、PyTorch `main@2b460d01`。当前 op-plugin 工作树仍是较旧 `b7a17be...`，算子引用统一从固定 git 对象核验。
- **核心结论**：多流不是多个图 capture 后合并，而是 Event Record/Wait 把 side stream 纳入同一个 `model_ri_` 并在 capture end 前返回主流；`wait_stream` 只建立“源流调用点之前 → 目标流调用点之后”的局部单向偏序。RNG 也不是 CPU 完全不可用，而是 replay 不重跑 host 取 seed 逻辑，故用 device seed/offset tensor + intragraph offset + wholegraph increment 保证每次 replay 推进 Philox counter。
- **源码展开**：记录 `NPUGraph` 单模型 capture/replay、Stream/Event fork/join、HCCL 独立流及 allocator `recordStream`；以 native dropout 串起 secondary stream 与 Tensor RNG API；按固定 op-plugin 源码区分明确支持、部分重载/SoC 分支和仍走 `philox_engine_inputs()` 的路径。
- **测试审计**：保留 torch_npu 已有多流、RNG functional/distribution 测试证据，同时标出测试清单与固定 op-plugin 中 `bernoulli_` 等旧标量路径的版本/dispatch 张力；补充逐 overload、连续 replay、自定义 generator、无 join 负例、多副流 RNG 与 ACLGraph+HCCL 验收矩阵。
- **去重联动**：原 [[10_aclgraph_deep_analysis]] 只保留总览并链接专题；[[01_aclgraph]]、[[npu_operator_graph_eligibility_guide]] 与 NPU Graphs index 增加反链，不复制机制正文。
- **二次扩写**：按用户反馈恢复对话中的机制细节：显式 Event 与 `wait_stream` 的差别、capture/replay 逐阶段时序、三流间接加入与逐级返回、HCCL compute→comm→compute 双向依赖、数值化 Philox offset 演算、counter 计算边界、dropout 捕获/重放时间线，以及可直接转为用例的 RNG/多流验收模板。专题页由 312 行扩至 465 行，仍低于单页 500 行拆分线。

---

## 2026-07-16: 新增 [[inkling_analysis]] — Thinking Machines Inkling (975B-A41B) 收录,含 thinking_machines 新目录

**Type**: Ingest(应用户「今天 Thinking Machines 发布开源模型,分析并落库」。**关键事实: 无正式技术报告/论文**——arXiv 无;"technical report" = 官方公告 + HF 模型卡 + config.json,已全部落 `raw/01_theory/01_models/thinking_machines/`)

- **新源文件(raw/)**: `Inkling_config.json`(架构 ground truth,HF `thinkingmachines/Inkling`)、`Inkling_HF_model_card.md`(完整基准表 + Apache 2.0 许可 + 安全评估)、`Inkling_official_announcement_2026-07-15.html`(训练配方 + 设计哲学)。
- **新页**: [[inkling_analysis]](主线: 不抄 DeepSeek 作业的多模态开源 MoE,赌可定制底座而非榜首)+ [[thinking_machines/index]];[[01_theory/01_models/index]] 增 Thinking Machines 段。
- **本库对 config 核验的架构差异化要点**(全部带行号):
  1. **抛 RoPE**: 学习式相对位置编码(`d_rel=16, rel_extent=1024`)+ >128K logit 缩放外推到 1M。
  2. **抛 MLA**: 滑窗/全局 5:1 交错(`local_layer_ids` 55 层 + 11 全局),**非对称 KV 头**(全局 8 / 滑窗 16,窗口 512)。
  3. **SConv**(核 4)+ **encoder-free 四模态**(vision hMLP patchify 40×40 / audio 离散 dMel 16 级)+ **8 层 MTP**(vs DeepSeek/Hy3 的 1)。
  4. 路由沿用 sigmoid 免辅助损失,但 `route_scale=8.0` 异常大;Muon+Adam 混合训练 + muP。
- **影响力判断**(§五,已与事实分离): 前 OpenAI CTO 首发即开源(Apache 2.0)的象征意义;Tinker 微调变现的商业模式创新;抗审查/校准的差异化卡位;encoder-free 在视觉基准上已见代价(MMMU Pro 落后)。
- **联动反链**: [[hy3_analysis]](保守 vs 差异化对照)、[[12_deepseek_v3_analysis]](选择性继承)、[[13_kimi_k2_5_analysis]](多模态路线对照 + K2.5 合成数据冷启动)、[[21_hw_friendly_llm_codesign_analysis]](NVFP4 部署)。
- **校验**: config 行号逐一对 raw/ 核对;基准表从 HF 模型卡原表摘录(非二手);mermaid 结构图按库规范自查(首行 flowchart TB、英文 id、subgraph 标题无 `[]|`、标签无裸定界符)通过;无技术报告故"为什么"部分推断已显式标注。页 <300 行。

---

## 2026-07-15: 新增 [[21_torch_npu_upstream_adaptation_analysis]] — Ascend out-of-tree 适配与 PyTorch upstream 差异全景

**Type**: Ingest / codebase comparison（应用户“结合工作区 torch_npu 和 PyTorch 源码梳理 Ascend 适配主要点与上游差异”。基线：`torch_npu v2.7.1@b3c8a815`、其记录的 `op-plugin@6ef73e399`、PyTorch `main@2b460d01`，均按本地源码逐点核实行号）

- **中心结论**：torch_npu 应拆成三层评价——PrivateUse1/autoload/Guard/Hooks/Allocator/Dispatcher/AMP/c10d/Dynamo/Inductor registry 等**标准插件面**；ACL/CANN、私有 format、ACLNN、HCCL、Ascend codegen/ACLGraph 等**硬件实现面**；改写 Dynamo rule/Variable、Inductor lowering/scheduler/wrapper、cudagraph tree、distributed/FSDP 等**兼容补丁面**。前两层不是“落后”，第三层才是升级风险主来源。
- **口径订正**：旧页“Ascend 三路径 vs 社区统一 Triton”已加 `[!deprecated]`；当前 upstream CUDA/XPU 也采用 combined scheduling 混合多 codegen，差异应看公开注册接口与私有 patch 的边界。
- **upstream 新方向**：核实 AcceleratorHooks/`torch.accelerator`、实验性 Python PrivateUse1 hooks/guard、Inductor custom pass/config 与 device-module 自动 codegen 注册、`CUDAGraphPolicy`、distributed backend entry point、OpenReg 可执行规格；据此给出 P0-P2 收敛路线。
- **现场约束**：torch_npu 严格 pin PyTorch 2.7.1，而对照为 upstream main，页内明确区分版本差异与硬件差异；op-plugin 工作树未停在 gitlink commit，正文只引用 git 对象中可核验的记录版本配置。
- **联动**：更新 `01_dispatcher_and_device/index`、AI framework 总索引；为 [[11_privateuse1_device_integration_analysis]] 增反链；[[01_npu_compile_paths_overview]] 增过时口径提示与反链。

---

## 2026-07-15: 新增 [[21_hw_friendly_llm_codesign_analysis]] — NVIDIA 硬件友好 LLM 设计指南(软硬协同,系列第一篇)

**Type**: Ingest(应用户「总结该 blog 并加入知识库」。源 = NVIDIA Developer Blog 2026-07-10,HTML 快照已存 `raw/01_theory/06_distributed_parallelism/`;正文经本地 HTML→文本提取逐行核验,7 条 Guideline、公式、Table 1/2、Fig. 4/5/7/9 图注均照原文录入,未依赖 WebFetch 小模型转述——两次转述在"对齐 128/256/512"表述上确有出入,以原文为准)

- **落位裁定**: 本库 `05_inference` 域定义为 CoT/RAG/Agent(推理=reasoning,见 01_theory/index),故本页落 **06_distributed_parallelism**(兄弟页 EP/PP/TP 直接对应博客 §五/§六);raw 同步新建 `06_distributed_parallelism/` 目录。
- **页面要点**: 主线"模型超参(H/H'/L/对齐/精度)= 部署性能参数";roofline 记账 + Table 2 小 K 反例(H'=512 全程 memory-bound)+ Fig. 4 阈值(80% 吞吐需 K>3072/N>2560,GB300+NVFP4)+ tile 量化(128/256/512)+ NVFP4 双层缩放(16 值 micro-block E4M3 + FP32)+ 宽 EP 的 GEMM-M 公式 + CPP/Helix。§7 单列**立场评注**(NVIDIA 硬件本位、阈值不可跨硬件搬运)并对照 [[hy3_analysis]] 专家维 1536 与 K>3072 阈值的张力。
- **联动**: 06 index 页面列表增行、日期 bump;[[14_expert_parallel_analysis]] / [[15_pipeline_parallel_analysis]] Related 区各加反链;mermaid 决策图按库规范自查通过。

---

## 2026-07-14 (二): 新增 [[stem_sparse_attention_analysis]] — 混元 Stem 免训练稀疏注意力 (arXiv 2603.06274v1)

**Type**: Ingest(应用户提问「Hy3 提出的推理 attn 结构是什么」——Phase 5 按需生长。溯源链: 中文报道 → 腾讯技术工程公众号(2026-06-26,经 53AI/搜狐转载)→ **论文本体 PDF 已下载入 raw/ 并逐节核读** p1-7)

- **定性**: Stem 是**推理服务栈的免训练 prefill 优化插件,不在 Hy3 开源权重内**(`modeling_hy_v3.py` 纯稠密 GQA,README 部署配方无 Stem)——回答了"推理的 attn 结构"与开源模型结构的边界问题。
- **机制**(全部对论文核验): TPD 预算按 query 位置从 k_start 线性衰减到 μ·k_start(Eq. 3, μ=0.7),理论依据是因果信息流的递归误差放大(Eq. 1/Fig. 2/Fig. 3);OAM 选块度量 = QK^T + 0.2·max(0, log‖V‖₂)(Eq. 7),由最小化稀疏-稠密输出重构误差推导(Eq. 5-6);块大小 128、恒保 4 初始+4 局部块、下限 54 块(§3.1/Alg. 1)。
- **证据**: LongBench 25-31% 预算近稠密(Table 2)、RULER 25% 预算稀疏方法最高分(Table 4)、可叠加 DeepSeek-V3.2 DSA/MiniCPM-4.1 再压 15-18% 预算(Table 3)、等预算消融 TPD +2~3.4 分(Table 5)、128K H20 prefill 1540→420ms=3.7×(Fig. 1/§3.3)。
- **口径修正**: 新闻稿"3.6 倍首字延迟"与论文 Fig. 1 的 3.7× 不一致,页内以论文为准并注明;"ICML-26 接收"仅新闻口径,PDF 页眉仍为 Preprint,两说并存;HPC-BSA 算子 ~3× 加速为公众号口径(论文用 MIT-BSA),已标二手。
- **联动**: [[hy3_analysis]] §2.4 增服务栈指针 callout + Related 反链;tencent_hunyuan/index 与 01_models/index 增行;mermaid 流程图按库规范自查通过。

---

## 2026-07-14: 新增 [[hy3_analysis]] — 腾讯混元 Hy3 (295B-A21B) 收录,含 tencent_hunyuan 新目录

**Type**: Ingest(应用户「增加最新的 Hy3 technical report」。**关键事实: 截至 2026-07-14 Hy3 无 arXiv 正式论文**——arXiv API 与 HF papers 索引均核验无果;"technical report" 实体 = 模型卡 + 开源工件 + 官方博客/新闻稿,已全部落入 `raw/01_theory/01_models/tencent_hunyuan/`)

- **新源文件(raw/)**: GitHub README EN/CN(@ `8a12d9af87c6`, 2026-07-06)、`config.json`(HF `tencent/Hy3`)、`chat_template.jinja`、transformers `modeling_hy_v3.py`(@ `295cee3e1d00`)、官方榜图 benchmark.png / benchmark-appendix.png(约 40 基准 × 11 模型全量矩阵)。
- **新页**: [[hy3_analysis]](主线: 架构冻结、全靠后训练的性价比 Agent 模型)+ [[tencent_hunyuan/index]];[[01_theory/01_models/index]] 增 Hunyuan 段;[[12_deepseek_v3_analysis]] Related 区加反向链接。
- **本库独立核验的三个关键发现**:
  1. **preview 与正式版 `config.json` 逐字段完全一致**(实测 diff)——三个月提升纯来自后训练,榜单增量(DeepSWE 0.9→28.0、USAMO 37.3→72.0 等)构成一组罕见的"纯后训练 ablation"。
  2. **路由 = DeepSeek-V3 免辅助损失配方原样采用**(sigmoid L310 + 选择期偏置 L312-313 + 原始分加权),非标处仅 `router_scaling_factor: 2.826`;注意力为 GQA+QK-Norm(保守派,未用 MLA)。
  3. **[!contradiction] 已挂**: preview 官方博客称"differentiated expert size + P-Penalty Loss",但开源工件为均匀 1536 专家(config + `HYV3Experts` 单张量存储 L324-333)——按工件优先原则记录两说。
- **校验**: 全部 config/modeling/chat-template 行号逐一对 raw/ 内文件核对;榜图数字从 PNG 原图读取(非二手转述);mermaid 结构图按库规范逐条自查(首行 `flowchart TB`、英文 id、标签无裸 `[]()|`、subgraph 标题合规、单独 `end`)通过;API 定价仅有二手来源,已标"存疑待核"。页 <300 行。

---

## 2026-07-07: [[longcat_2_analysis]] 订正 ScMoE 结构描述 + 新增 §5.5「计算-通信重叠调度（SBO / 训练双 chunk）」

**Type**: Update（应用户「2.0 模型结构介绍有点问题、并行策略设置未介绍」。源忠实——窗口拓扑据 SGLang `sglang-longcat-pr/longcat_flash.py:429-461` 逐行核验；SBO 阶段级调度据 LongCat-Flash 技术报告 arXiv 2509.01322 §5 并明确标注来源与推断边界）

- **订正结构描述的「fork 点」易错点**：`clone` 发生在 `attn0`(MLA₁) **之后**（`longcat_flash.py:449`），故 Fig-1 图注与「一句话读结构」原先把 `attn0` 计入「与 MoE 并行的稠密链」不准——已改为 **attn0 是 fork 前的共享前置**，可掩盖 MoE 通信的**重叠窗口 = 稠密FFN₁ + MLA-attn1 + 稠密FFN₂**（不含 attn0）。Fig-1 PNG 本身画法正确（attn0 在 fork 之上），**无需重绘**，仅订正文字；§2.3 note 补「窗口要说精确」段。
- **新增 §5.5**：ScMoE 短路只建立**数据依赖上的自由度**，窗口内部「怎么切/谁盖谁」是调度层选择、训练与推理各一套——
  - **推理 SBO 四阶段**（含 ASCII 图）：① MLA₁ → ② 稠密FFN₁+**MLA₂.QKV 投影段**掩 dispatch → ③ **MoE GEMM 裸露**（靠 wide EP/EP128 压薄）→ ④ **MLA₂.核心+输出投影段**+稠密FFN₂ 掩 combine。**命名消歧 callout**：SBO 的「Attn0/Attn1」= MLA₂ 的两个 phase，≠ 层内两个 MLA 块（`self_attn[0/1]`）。列出相对「稠密FFN 掩 dispatch、MLA₂ 掩 combine」粗说法的三处精确修正。
  - **训练 token 维双 chunk 互掩**（Flash 报告 §2.2「token 维细粒度切分并发」）：两 chunk 与 dense FFN、彼此 overlap，**连 GEMM 也参与掩盖**——与推理「GEMM 裸露」相反。
- **联动更新**：§5.2「ScMoE 调度」bullet 改指 §5.5；页头「维度」补「并行与重叠调度」；[[longcat_flash_analysis]] §五 SBO bullet 补四阶段/双 chunk 摘要并交叉链到本页 §5.5。
- **校验**：两处 ASCII 图为纯 `code` 块（非 mermaid，无渲染风险）；`longcat_flash.py:433/449/456/460/467-492` 定位符逐一对源核对；页仍 <500 行。

---

## 2026-07-06: [[25_unbacked_symint_analysis]] 增补 §10 —— unbacked 处理的最新进展（`guard_size_oblivious` → 显式 size-oblivious 原语族）

**Type**: Update（应用户「根据最新技术更新知识库」。源忠实——全部新断言据 pinned pytorch checkout `torch/fx/experimental/symbolic_shapes.py` 逐个核验签名/行号/docstring/`__all__` 导出，只扩展不删除）

- **§7 API 表补 5 条**（均带全路径）：`statically_known_false`、`guard_or_true`、`optimization_hint(x, fallback)`、`sym_and`/`sym_or`；并加一行指针指向 §10。
- **新增 §10「从 `guard_size_oblivious` 到显式 size-oblivious 推理原语」**：
  - **旧机制** `guard_size_oblivious`（`:534`）——对 size-like unbacked 隐式临时设值域 `[2,Inf]`，docstring 自承 "we may diverge in behavior"，隐式/难推理。
  - **新原语族**（逐个核验行号）：`guard_or_false`(`:1573`)/`guard_or_true`(`:1580`)/`statically_known_true`(`:1648`)/`statically_known_false`(`:1621`)/`optimization_hint`(`:155`)/`sym_and`(`:1672`)/`sym_or`(`:1698`)，均在 `__all__` 导出；三档分工（静态保守 / 有默认分支 / 仅优化不影响正确性）。
  - **迁移规模实测**（当前 checkout `torch/`）：`guard_or_false|true` **~366 处/44 文件**（decomp、`_refs`、`_meta_registrations`、Inductor `lowering/ir`、**DTensor** `_view_ops.py` 单文件 ~38 处）vs `guard_size_oblivious` **~18 处/9 文件** —— 数量级反转，佐证「显式 guard_or_* 已成默认范式、旧接口沦为残留」，并呼应 DTensor+dynamic shape 正被改造为 unbacked-safe。
  - **选型决策 mermaid 图** + 「是否必须解决 unbacked」取舍（graph break 是合法逃生口；仅 `fullgraph=True`/export/AOTInductor/整图 CUDA Graph 穿过数据相关 op 时才必须）。
- **校验**：新 API 的签名/行号/`__all__` 全部对 pinned checkout 源码核对；mermaid 块按本库规范逐条自查（首行 `flowchart TD`、英文 id、矩形/菱形标签无裸 `[]()`、连线标签无引号/括号/`|`）通过；页头「最后更新」改 2026-07-06 并注明增补范围与定位符基准；§10 内部锚点因 heading 含 en-dash 会被 GitHub slugger 吞成 `20252026`，已改 heading 为 ASCII 连字符使 `#...2025-2026...` 锚点可解析。页仍 <500 行。

---

## 2026-07-06: [[07_training_reliability/index]] 簇补 7 张机制图示（提升可读性）

**Type**: Enrich（应用户「三类文章基于原始 technical report 补图示、提升可读性」。据原文机制 + 其引用的技术报告绘制）

- 手绘 HTML+CSS/SVG → 本库无头 Edge 2× 渲染 **7 图**（源 `.html2md/figs/training_reliability_figs.html`，gitignored）：
  - **确定性&数值页**（[[10_determinism_and_numerical_reliability_analysis]]）：`tr_det_fig1` 浮点非确定性五层来源 · `tr_det_fig2` 长链累加病(顺序 O(n·ε) 吞位,BF16 Σ1000×1.0=256)与药(树形 O(log n·ε) + DeepSeek FP8 两级累加) · `tr_det_fig3` SDC 四层检测 + Gemini split-phase 确定性重放闭环。
  - **容错&恢复页**（[[11_fault_tolerance_and_recovery_analysis]]）：`tr_ft_fig1` 恢复粒度坐标系(8 环恢复链路 + Job/Pod/进程内/Step 各级砍环时间轴) · `tr_ft_fig2` hang 症状/病灶空间分离 + Flight Recorder seq 对账定位。
  - **训练动力学页**（[[12_training_dynamics_stability_analysis]]）：`tr_dyn_fig1` spike/NaN 排查决策树(确定性重放为核心分岔) · `tr_dyn_fig2` spike 治理四层防线(架构/优化器/数据/运维)。
- 各图嵌入对应小节（背景 / 如何发现 / 排查思路 / 解决方案）。**校验**：7 张 PNG 逐张实渲肉眼核对（恢复链路时间轴、seq 对账暗框、决策树双通道分支均正常，无溢出/无裸定界符）；图片用标准 `![](assets/*.png)`（SVG 渲染，非 mermaid）。

---

## 2026-07-06: 新建 [[07_training_reliability/index]] 簇 —— 摄入《万卡训练确定性与可靠性深度分析》(9 问题域·多来源综述)

**Type**: Ingest（应用户「把这份基于 LongCat 衍生的稳定性训练文档吸收到知识库」。源忠实——二手综述的结构化摄入，机制/数字/命令/代码忠实原文，交叉链到已有一手页）

**源**：用户提供的多来源综述 `raw/02_engineering/wanka_determinism_reliability_deep_analysis.md`（已存 raw、747 行），综合 Gemini 1.0/2.5 · Llama 3 · ByteRobust(SOSP'25) · MegaScale(NSDI'24) · Aegis(NSDI'25) · C4(HPCA'25) · DeepSeek-V3(ISCA'25) · Thinking Machines「Defeating Nondeterminism」· Anthropic postmortem · 华为 CloudMatrix · 美团 LongCat-2.0 博客 + Megatron-LM/NVRx/torch_npu 代码。

- **新建 `02_engineering/07_training_reliability/` 簇（index + 3 内容页）**，按原文四部分/9 问题拆解：
  - [[07_training_reliability/index]]（**coordinator 手写的 exemplar**）：问题地图（9 问题×两主线）+「确定性是故障定界的地基」主线 + 趋势与开放问题（原文第四部分）+ 与本库已有页的交叉表。
  - [[10_determinism_and_numerical_reliability_analysis]]（问题 1-4）：浮点非确定性五层来源（atomicAdd/split-K/通信规约树/MoE 排序/框架随机性）、batch 不变性与 RL 确定性（Thinking Machines、Anthropic top-k 事故、TIS）、低精度长链累加（pairwise/树形、FP32 main_grad、DeepSeek FP8 两级累加/DeepGEMM、Kahan）、SDC 四层检测体系（压测/统计/ABFT+DP hash/确定性重放）。
  - [[11_fault_tolerance_and_recovery_analysis]]（问题 5-8）：goodput/ETTR + 五级恢复坐标系（Job/Pod/Node/进程/Step + 算子链路级）+ 各家术语对照（华为 MindIO TFT 的 TTP/UCE/ARF、NVRx in-process restart、torchft、Gemini slice 弹性…）、hang/straggler（flight recorder/栈聚类/straggler 打分）、Checkpoint（异步+原子提交/本地分层/临终/数据回放）、网络链路（PFC 风暴/ECMP hash/链路级快恢/流量工程）。
  - [[12_training_dynamics_stability_analysis]]（问题 9）：loss spike/NaN 四类根因、分层监控+前兆指标、排查决策树、四层防线（QK-Norm/z-loss/soft-capping/EGS、MuonClip/AdaGC/ZClip、数据指纹、运维自动化）、2026 前沿（Muon 路线共识、DeepSeek-V4 Anticipatory Routing/mHC、Kimi K2.5 与 GLM-5 的 RL 稳定性与问题 2 合流）。
- **并行 writer-agent 契约**：3 内容页由 3 个 subagent 并行写，各读 raw 指定行段（Part1/2/3），严格「不加源外事实、保留全部数字/env-var/代码/出处、只用给定交叉链、无 mermaid」，结构化回报。

**整合**：[[02_engineering/index]] 子领域表加 07 行；[[./index|总索引]] 目录树加 07、工程域表加「训练可靠性 4」行、按主题查找加一行。**校验**：3 页 grep 确认关键 env-var/数字/机制在位（CUBLAS_WORKSPACE_CONFIG、TORCH_NCCL_TRACE_BUFFER_SIZE、五级坐标系、MuonClip/Anticipatory Routing/129.3 MWh 等）；4 页全部 `[[链接]]` 机械核对**零死链**；4 页 grep 确认**零 mermaid**（全用 ASCII/代码/表）；抽读 determinism 页头+§1 核对源忠实与房风格。

---

## 2026-07-06: 新建 [[longcat_flash_analysis]] —— 摄入 LongCat-Flash（560B/27B MoE，ScMoE + 零计算专家首创）

**Type**: Ingest（应用户「把 LongCat-Flash 也摄入知识库」。LongCat-2.0 的架构前身，源忠实 + 抓本质）

**源（source-faithful）**：LongCat-Flash Technical Report **arXiv 2509.01322v1**（2025-09-01）+ released `meituan-longcat/LongCat-Flash-Chat/config.json`（含 `modeling_longcat_flash.py`）。config 逐字段核对；报告按 §/Eq./Table 定位（§2.1 零计算专家 Eq.1-5、§2.2 ScMoE Fig.4、§2.3 方差对齐 Eq.6-8、§2.4 MLA/MTP、§3.1 超参迁移 Table1 + 模型生长、§3.2 稳定性、Table2/3 评测）。

- **新建深挖页 [[longcat_flash_analysis]]**（主线「用零计算专家 + ScMoE 短路把 560B 的激活压到 ~27B」）：
  - **架构**：MLA(64h, q_lora1536/kv_lora512/nope128+rope64/v128) · **零计算专家**（512 FFN + **256 identity**、top-12、激活 18.6–31.3B 动态、PID 控偏置 + 设备级均衡损失）· **ScMoE 短路**（前块稠密 FFN ∥ 当前 MoE dispatch/combine 通信、TPOT 较 DeepSeek-V3 ↓~50%、质量中性 Fig.4）· MTP（单稠密头、接受率 >90%）。
  - **缩放/稳定性**：μP 超参迁移(s=8, 代理宽 768, Table1) · 模型生长初始化(14→28, r=2) · Router 稳定(Rg<0.1 + PID) · hidden z-loss(Eq.10) · Adam eps 1e-16 · 确定性 + SDC 检测。
  - **预训练**：20T tokens/30 天/98.48% 可用率；三段课程(通用→STEM&code 70%→长上下文 8k→32k 80B→128k 20B)；13-gram + BGE-m3>0.9 去污染。
  - **Infra/推理**：SBO(NVLink TP ∥ RDMA EP)；**H800 >100 TPS、$0.7/M**；投机解码。
  - **Agentic**：τ²-Bench 67.7 / VitaBench 24.30(30+ 工具、60+ 轮)为长板；Base/Chat 评测表(Table2/3)。
  - **§八 Flash→2.0 演进对照表**：Flash(560B/27B·28 层·MLA 全注意力·512+256 专家·128K·H800) → 2.0(1.6T/48B·38 层·MLA+**LSA**·768+128·**N-gram**·1M·**国产 ASIC**)。
  - [!correction] 订正本人先前假设：**Flash 亦用 MLA**（非 MHA/GQA）——config `attention_method:"MLA"` 坐实；ScMoE/零计算专家为两代共享、SGLang `longcat_flash.py` 同一份代码。

**整合**：[[meituan_longcat/index]] §一家族表 Flash 行改为已摄入、§五缺口更新；[[01_theory/01_models/index]] LongCat 区加 Flash 行；[[./index|总索引]] 模型 30→31、LongCat 子域 2→3、导航加 [[longcat_flash_analysis]]；[[longcat_2_analysis]] Related 加前身回链。**校验**：9 个交叉链接目标本会话/前序 grep 核对存在；纯文本+表+ASCII，无 mermaid；数值带 arXiv §/Eq./Table 或 config 定位。

---

## 2026-07-06: 用开源推理代码升级 [[longcat_2_analysis]] 架构描述 + 3 张代码级模型结构图

**Type**: Enrich（应用户「longcat2.0 开源了推理代码，结合推理代码完善模型结构描述 + 详细绘制模型结构图，每步数据流用 SVG→PNG」。源升级：**codebase 一手 > 博客二手**）

**源（source-faithful, codebase）**：官方 2026-07 放出 `config.json` + 194 分片权重（HF `meituan-longcat/LongCat-2.0`，`GIT_LFS_SKIP_SMUDGE` clone）；GPU 推理经 SGLang **PR #30042** @ `HarryWu99/sglang@c6c36d9`。逐文件核对 `longcat_flash.py`（模型/ScMoE/MoE，1093 行）、`n_gram_embedding.py`（:134-175）、`nsa_indexer.py`（SI/CLI，:493/:539-559）、`config.json`（60 行逐字段）、README（SGLang 部署 + LSA/N-gram 说明）。

- **3 张代码级结构图**（手绘 HTML+CSS/SVG → 本库无头 Edge 2× 渲染；源 `.html2md/figs/longcat2_architecture.html`）：`assets/longcat2_arch_fig1.png`（整体前向 + 单层 ScMoE 短路放大）、`fig2`（N-gram Embedding 数据流）、`fig3`（MLA + LSA 数据流）。逐张实渲肉眼核对。
- **§1.1 核心参数表全部落实**（每条带 config.json 行号或 `文件:行`）：38 层 · hidden 8192 · 64 heads · MLA(q_lora 1536/kv_lora 512/nope128+rope64/v128) · 稠密 FFN 12288 · **768 路由 + 128 零计算(identity)专家 / top-12** · 专家 FFN 2048 · vocab 163840 · RMSNorm/SiLU · RoPE-YaRN(factor 120) · N-gram 16 路哈希 · LSA index_topk 2048 / local 1024 / init 16 / cli_factor 2 · MTP 3-step。
- **§1.2 换成 3 图 + 代码级结构总览**；**§2 三处订正**（`[!important]`）：① 注意力实为 **MLA**（LSA = MLA 骨干 + DSA 式索引器，非全新注意力）；② 层是 **ScMoE 短路**（2×(MLA+稠密FFN) ∥ MoE，`longcat_flash.py:449-460`），非「注意力→MoE」常规块；③ **零计算专家确有其事**（`zero_expert_num:128 identity`），激活参数随 token 动态。
- **§2.1/2.2/2.3 加「代码补充」**：LSA 的 SI（force-keep 16 sink+1024 local = 50%，印证官方图）/ CLI（cli_factor=2、缓存索引集印证 Q4）/ HI（SGLang 未实现）；N-gram（16 路多项式哈希→查表→投影→**mean**）；ScMoE（`LongcatFlashMoE` 768+128、top-12、`zero_experts_compute_triton`）。
- **[!contradiction] 订正**：§9.1 zero-compute experts 由「未证实」→**代码定案为常设机制**、二手「33–56B 区间」方向可信；新增「注意力实为 MLA」订正。§6 补**推理 FP8**（`LongCat-2.0-FP8` + bf16 KV，`longcat_flash.py:697-808`），训练精度仍未披露。§9.2 结构项**全部划除**（已回填 §1.1），仅留训练侧未披露。
- **家族 index**：§四 开源状态**翻篇**（未放出 → 已开源：权重 + config + SGLang 推理码）；§一/§二/§三 补 38 层 / MLA / ScMoE 短路 / 推理 FP8 / 128 零计算专家。

**整合**：改动集中在 [[longcat_2_analysis]]、[[meituan_longcat/index]]，新增 3 图。**校验**：3 PNG 实渲核对；图片用标准 `![](assets/*.png)`（非 mermaid）；config.json 行号与 `longcat_flash.py`/`nsa_indexer.py`/`n_gram_embedding.py` 关键行本会话开文件核对；SGLang 建模码不入本库（仅引 `file:line`）。

---

## 2026-07-03: [[35_inductor_memory_allocation_guide]] 新增 §5「内存越界/踩踏排查」— 补第二份社区材料的缺口(纠错版)

**Type**: Enrich（应用户"第二份专家社区材料的第三部分——内存踩踏检测——库里缺,补上,一定忠于事实"。经 gap 分析:材料前两部分已被 `inductor_memory_management_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[12_buffer_liveness_memory_planning_and_reuse_analysis]]，按"历史不回写"惯例降级为反引号）覆盖且更准,仅第三部分是真缺口;材料本身有错,按源码收敛后再入库）

**源（source-faithful）**：pytorch @ `5f6df46744a` 逐行核对——`config.py:232-237`（`size_asserts`/`nan_asserts`/`scalar_asserts` 默认值）、`codegen/wrapper.py:1793-1827`（`assert_size_stride` 生成 + 延迟到首个 kernel 前）、`ir.py:7817/7845/9152`、`utils.py:161`（`GPU_ALIGN_BYTES=16`）、`codegen/triton.py:5458`（`mask=xindex<xnumel`）、`codegen/common.py:2789`（核内 NaN 断言）。

- **guide 新增 §5 内存越界/踩踏排查**（原 §5/§6 顺延为 §6/§7）：§5.1 自动核内置防护（Triton `mask` + `assert_size_stride`/`assert_alignment`/scalar·nan_asserts,多为默认 ON）· §5.2 真正越界来源（自定义算子/手写 Triton/错误 stride·offset/unbacked symint）· §5.3 工具（`compute-sanitizer` 取代 cuda-memcheck + `CUDA_LAUNCH_BLOCKING`）· §5.4 排查步骤。
- [!correction] **订正该社区材料**（源 > 材料，均已核实）：① 「Inductor 对越界无内置保护」**错**——mask + size/alignment/scalar 断言多为默认 ON;② 「规划池用 `_cuda_beginAllocateToPool` 申请」**错**——该 API 全库仅在 `cudagraph_trees.py`（CUDA Graphs 私有池）,规划池是普通 `empty_strided`;③ `cuda-memcheck` 已被 `compute-sanitizer` 取代。**材料对的部分**（保留）：16 字节对齐确有其事（`GPU_ALIGN_BYTES=16`）。前两部分（池初始化/复用）不收录——深挖页已覆盖且更源忠实。

**整合**：[[04_inductor/index]] guide 条目补「越界/踩踏排查」;guide Related 增 [[23_inductor_gpu_kernel_dispatch_model]]/`Pytorch_Compile_Debug_Analysis`(该页已于 P4 Task 4 判重删除，内容并入 [[02_compile_stack/07_debugging/index]])/[[25_unbacked_symint_analysis]] 回链。**校验**：所有 `file:line`/常量值本会话开文件核对;新增段无 mermaid。

---

## 2026-07-03: 深化 [[longcat_2_analysis]] —— 补官方原图(LSA/N-gram/MOPD) + 读图问答 + 开源状态 + 完整大纲审计 + MOPD 订正

**Type**: Enrich（应用户连续追问：架构更细解读 / 是否有开源代码 / 放原图并结合图说明「LSA 复用是缓存还是重算」/ 完善缺失内容。源忠实——官方架构图与图注为一手证据，纠正二手误读。提交 `305dac1`）

- **补 3 张官方原图**（美团 S3 CDN 下载 SVG → 本库无头 Edge 管线 2× 转 PNG、白底，与既有页一致；SVG 原图并存作 source）：`assets/lsa_overview.{svg,png}` · `ngram_embedding_overview.{svg,png}` · `mopd_overview.{svg,png}`。渲染工具 `.html2md/svg2png.mjs`（gitignored，不入库）。
- **LSA §2.1 读图深化**：据 LSA 图确证结构——Full KV 分 **Streaming(绿)→Contiguous KV(~50% 预算)** 与 **Non-Streaming(黄)→Block Indexer→Token Indexer 两级 top-k→Non-Contiguous KV(~50% 预算)**，右 **Reuse Layer 无索引器**、标注 "Directly Reusing the Indices from the Owner Layer"。新增「读图问答」回答四问：① streaming token = sink + 近窗连续段（约半预算），非纯滑窗；② 层次化 = 块→token 两级选择（共享参数）；③ CLI = Owner 算一次、多 Reuse 层复用；④ **复用是缓存非重算**（Reuse 层结构上无索引器；缓存的是 top-k 索引集合而非注意力结果，每层仍算自己的 Attn；证据「amortize indexing cost」+ MTP「reusing the index set」）。LSA 动机补为「定点修 DSA Lightning Indexer 的**输出不连续 + 二次方打分**两短板」。
- **N-gram §2.2 读图**：据图补机制——当前位置取 2/3/4/5-gram，各过 Hash+Embedding+Projection（多张哈希表）再与 Base Embedding 相加；动机补「MoE 稀疏度已过甜点区(~97%)、挪 135B 到 N-gram 收益远超标准专家」。
- **[!contradiction] MOPD 订正（源 > 二手，超越本 changelog 2026-07-02 条目的「多目标策略分布」）**：官方架构图副标题为 **"Multi-Teacher On-Policy Distill(ation)"（多教师在线策略蒸馏）**，据此订正早期二手误读「Multi-Objective Policy Distribution」。三组 teacher 原子能力据图列全（Agent: Tool Use/API Parsing/Self-Correction；Reasoning: Multi-Hop/STEM/**Adaptive Computation**；Interaction: 指令遵循/人类对齐/幻觉抑制）。
- **家族 index 补「四、开源状态」**：已核实 **LongCat-2.0 仓库 main 仅 README+LICENSE(MIT)+figures、无 config.json/建模代码/权重**（HF 下载 0、weights coming soon）；**架构前身 LongCat-Flash-Chat 完全开源（79K+ 下载，含 ScMoE / zero-compute experts）**，是当前唯一可读参考实现；2.0 新增的 LSA / N-gram / MOPD 无开源代码。
- **完整大纲审计补缺**：据博客全章节大纲补 §5.2「推理·模型专属优化」（absorb computation / pipelining indexer / KVP / ScMoE 调度）、§5.4 weight prefetch、§8「官方能力演示 3 场景（Codebase Migration / Agentic & Research / Content Generation）」；§9.2 审计确认 layers/dims/heads/vocab/activation/norm/RoPE/数据配比/tokenizer/吞吐 **全部 not stated**（非漏读，已在页内声明）。

**整合**：改动集中在 [[longcat_2_analysis]] 与 [[meituan_longcat/index]]。**校验**：3 张 PNG 均实渲肉眼核对（LSA/N-gram/MOPD 内容正确）；图片用标准 `![](assets/*.png)`（非 mermaid，无定界符风险）；§5 重编号后 §9 未动、页内/index 的「§9」引用仍有效；无新增死链。

---

## 2026-07-02: 新建 [[meituan_longcat/index]] + [[longcat_2_analysis]] —— 美团 LongCat-2.0（1.6T/48B MoE，国产 ASIC 全栈）

**Type**: Ingest（应用户「分析 longcat 2.0 的模型结构、训练、AI infra、低精度、稳定性、效果，录入知识库」。源忠实 + 抓本质）

**源（source-faithful）**：官方 Tech Blog `longcat.chat/blog/longcat-2.0`（**JS 渲染 SPA**，直取仅得标题）→ 经**渲染代理提取 + 三源交叉核对**（渲染博客文本 · HF/GitHub `README.md` · DeepWiki 镜像），并与二手报道对比去伪。Baseline = 访问日期 **2026-07-02**；权重/config.json「coming soon」、正式技术报告未见 → 页头与 §9 已标注保真度与未披露项，待 raw 源到位回填精确基线。

- **新建 family index [[meituan_longcat/index]]**：LongCat 家族（LongCat-Flash 前身 → LongCat-2.0）总览 + 一页速览 + 与 GLM-5/DeepSeek-V3/V4/Kimi-K2 的稀疏注意力/优化器/低精度/硬件定位对照表 + 知识缺口。
- **新建深挖页 [[longcat_2_analysis]]**（主线「在国产 AI ASIC 上把 1.6T MoE 推到近前沿 Agentic Coding」）：
  - **架构**：LSA 稀疏注意力三正交索引（SI 硬件对齐连续访问 / CLI 跨相邻层复用+跨层蒸馏 / HI 块级粗筛→token 细选 training-free）；**N-gram Embedding 135B, n=5**（与 MoE 正交稀疏维扩参、空间约 100×、<10% 预算、降大 batch 解码 I/O）；**ScMoE**（per-core 显式控制→dense/MoE 分支全并行）；**MTP 3-step**（第 2/3 步复用第 1 步 LSA 索引）。
  - **预训练**：>35T tokens；**Muon 大规模**（TP 适配 + DP 状态去冗 + 对称矩阵乘 kernel）；数百亿 token **原生 1M**（all-gather CP 扩到 512+）。
  - **后训练**：**MOPD** 多目标策略分布，融合 **Agent/Reasoning/Interaction** 三组 teacher expert 群蒸馏。
  - **AI Infra**：**6D 并行 = 5D(TP/CP/EP/DP/PP) + EMBP**（专并行 135B N-gram）；superpod ≤48 机 all-to-all + 跨 pod RoCE（+30%）+ 总体 +35% 吞吐；推理 **PD 分离**（prefill CPP+Attention SP / decode KVP+EP128，KV 走 200Gbps 网卡）；**Super Kernels + L2 预取 + EPLB**。
  - **低精度**：[!contradiction] **博客不讲 FP8/FP4 量化**——其「精度」叙事是国产 ASIC 上的**数值可靠性/确定性**（确定性算子覆盖 Embedding/FA/LSA/MoE + 二叉树分段累加降 FP 误差 + 对齐高精度基线验证）。与 DeepSeek-V3(FP8)/GLM-5(INT4 QAT) 是不同侧面。
  - **稳定性**：>35T tokens **零回滚/无不可恢复 spike**；bit-flip 检测 + 端到端自动故障识别/流量切换/恢复。
  - **效果**：全评测表（LongCat-2.0 vs Gemini 3.1 Pro / GPT-5.5 / Claude Opus 4.6/4.7/4.8）——**SWE-bench Pro 59.5 > GPT-5.5 58.6**、Terminal-Bench 2.1 70.8、GPQA-diamond 88.9；整体落后 Claude Opus 4.8。
  - **§9 源忠实修正**：[!contradiction] 二手报道称「动态激活 33–56B / zero-compute experts」——博客只提训练期 padding→zero-expert（省显存），激活即 ~48B，疑似把 LongCat-Flash 机制张冠李戴；[!contradiction] 训练算力 README「加速器·小时」vs 博客渲染「天」24× 分歧，FLOPs 粗算支持「小时」。

**整合**：[[01_theory/01_models/index]] 新增「LongCat / Meituan」家族区；[[./index|总索引]]模型行 28→30、加「LongCat (美团)」子行与「按主题查找」条目、更新日期至 2026-07-02；两新页与 [[01_glm_5_analysis]]/[[12_deepseek_v3_analysis]]/[[13_deepseek_v4_analysis]]/[[11_kimi_k2_analysis]]/[[11_muon_analysis]]/[[14_expert_parallel_analysis]] 等互链。**校验**：全用 ASIC——图表用 **ASCII**（与 GLM-5/Kimi 同系列风格，零 mermaid 定界符风险）；跨链目标经 grep 核对；因源为渲染提取，数值保真度与未披露项已在页头/§9 显式声明，不臆造未披露量。

---

## 2026-07-01: 新建 [[06_distributed_parallelism/index]] 分布式并行原理簇 —— 原语→DP→TP/SP/CP→EP→PP→ZeRO 全景（理论层）

**Type**: New（应用户"在 01_theory 加分布式并行原理解读，从分布式原语→TP→EP→PP→ZeRO 等基本概念；演示图用 SVG→PNG"。抓本质 + 引擎无关的原理层，与已有工程页分工）

**定位**：新建理论簇 `01_theory/06_distributed_parallelism/`，**原理（principle）层、引擎无关**——只讲「为什么这么切、代价函数长什么样、为什么不选替代」，两根主线贯穿全簇：**$\alpha$-$\beta$ 通信代价模型** + **显存账本（参数/梯度/优化器态/激活）**；「源码怎么实现」一律交叉链接到 [[02_engineering/index]] 已有的源级页（`[[15_distributed_primitives/index]]`、[[megatron-lm/index]]、[[torchtitan/index]] 等），不重复。填补「理论层无分布式并行原理页」的空白。

- **新增 index + 6 内容页**：
  - [[10_collectives_analysis]] — 六大原语语义、$\alpha$-$\beta(-\gamma)$ 模型、核心恒等式 **all-reduce = reduce-scatter + all-gather**、ring 每卡搬运 $\frac{2(N{-}1)}{N}M$ 的带宽最优性、ring vs tree、all-to-all/p2p 代价（全簇「代价词汇表」）。
  - [[11_data_parallel_analysis]] — DP：复制模型/切数据、all-reduce 梯度的等价性、通信 $\propto\Psi$ 与 batch/卡数无关、$\Psi\times16$ 显存账本（引出 ZeRO）、分桶重叠 + 梯度累积。
  - [[12_zero_fsdp_analysis]] — ZeRO 1/2/3 逐级切优化器态/梯度/参数、通信 vs DP 增量（1/2 免费、3 多 ~50% AG）、ZeRO-3 = FSDP 的 unshard→compute→reshard。
  - [[13_tensor_sequence_parallel_analysis]] — TP（Megatron 列切→行切 + f/g 共轭算子、每层 4 次 all-reduce、只敢机内）、SP（拆 all-reduce 为 RS+AG，零额外通信换激活显存）、CP（ring-attention 交换 KV 攻长序列）。
  - [[14_expert_parallel_analysis]] — EP：路由 + 两次 all-to-all（分发/回收）、负载不均与容量因子、分层 a2a。
  - [[15_pipeline_parallel_analysis]] — PP：microbatching、气泡率 $(P{-}1)/(m{+}P{-}1)$、GPipe vs 1F1B（同气泡、显存 $\propto m$ vs $\propto P$）vs interleaved（真降气泡）、zero-bubble。
- **演示图 9 张 SVG→PNG**（手绘 HTML+SVG，走 `.html2md/render_figs.mjs` 无头 Edge 2× 截图）：六原语语义、ring all-reduce 分解、DP 数据流、Megatron 列/行切+f/g、TP+SP 激活切分、ring-attention、EP 三段 a2a+负载不均、GPipe/1F1B 甘特气泡对比、ZeRO 0/1/2/3 显存分区、N 维正交布局（DP2×PP2×TP4）。原理演示图统一走 SVG（按用户约定：代码调用/类/逻辑图才用 mermaid）。

**工具改动**：`render_figs.mjs` 加 `FIGS_OUT` 环境变量支持自定义输出目录（默认仍指 GLM assets，向后兼容），本簇渲染到 `06_distributed_parallelism/assets/`。

**整合**：[[01_theory/index]] 子领域表加「06 分布式并行原理」一行；`[[15_distributed_primitives/index]]` 与 [[06_auto_parallel/index]] 各加回链（理论↔实现互指）。**校验**：9 张 PNG 逐张实渲肉眼核对（SVG 经 Edge 所见即所得，天然规避 mermaid 定界符坑）；本簇内 `[[链接]]` 与指向工程页的跨域链接经 grep/文件核对存在。

---

## Related Pages

- [[changelog/2026_q2_and_earlier|2026-Q2 及更早变更日志归档]]
- [[01_theory/index|理论研究]]
- [[02_engineering/index|工程实现]]
