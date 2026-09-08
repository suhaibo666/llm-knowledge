# vLLM 三批最终整合：旧02迁移守恒独立审阅

> 编号说明（2026-09-08 后续整理）：本文记录交付过程，正文中的裸编号保留当时口径；文件名、链接路径已同步为连续编号后的当前值，当前导航以 vLLM 域索引为准。其中“旧02”始终指已合并删除的系统设计原则页。

日期：2026-09-08。审阅角色：只读迁移审阅者；没有修改 wiki、共享导航、源码或图，没有派生 agent。源码基线仍为 `199cb9b964822e59ab9b58d88e7be31eb419a2ae`。

## 结论

**PASS：旧02的解释内容已完成实质迁移，可以删除旧正文。** 未发现需要保留旧02才能避免的信息缺口，也不需要在03前再造一篇设计原则前置页。删除应与当前导航、相关链接和基线状态同步收尾；这属于删除动作本身的必要整合，并非内容迁移未完成。

03确实接住了旧02§1的总体动机：§1用A生成中/B长输入/C新到达说明静态batch与请求独占的不足；§3.2以两步预算4重放动态成员和进度；§4.3把token/request/encoder/KV联合考虑与按需分页连起来；§4.5说明每步准备成本和状态/输入分离；§4.6说明模型、硬件与快路径组合约束。旧02中更精细的三预算字段、指标及推断限制分别保留在11、12、27，既没有丢失，也没有把03再写成术语前置课。

一个非阻断的用语澄清建议登记在后文：16§9的“NONE eager”可明确为“无graph的正常模型调用，仍可能使用compiled callable”，使两轴措辞与23完全一致。它不构成保留旧02的理由。

## 逐项守恒矩阵

| 旧02内容 | 当前正文实质承接 | 结论/当前语义 |
|---|---|---|
| header主问题：只优化一次forward不足；动态请求/进度/KV/快路径 | 03§1、§3.2–3.4、§4.3–4.6 | 主问题保留且进入请求例子；没有要求先读02 |
| §1不永久划分prefill/decode、每步重新分配 | 03§3.2；11§1–3 | 统一目标进度差额仍成立；chunked/prefix/spec是同一调度账的不同输入 |
| §1调度token上限、输入token上限、request slot与KV容量 | 11§1、§3.1；12§3.2 | 当前字段与单位更清楚：max_num_scheduled_tokens与max_num_batched_tokens不同，max_num_seqs是请求容量，KV另由pool决定。没有照抄旧02把三种字段与三类资源逐一对应的含混句 |
| §1吞吐/一步长度/排队/重算牵制 | 03§1、§4.3；11§1末、§7 | 保留成本因果，并标为分析而非测量 |
| §1running/waiting/KV/preempt/TTFT/ITL共同观察 | 11§7完整信号表；27§3.1–3.3 | 并非只靠GPU利用率；指标不是根因充分证明 |
| §1四条“瓶颈→替代→支点→成本”图 | 03§1/4.3/4.5/4.6；11§1；12§1；16§1/3/4；23§1/3 | 图的四条信息链分别有正文与更具体演算承接，不需要保留原同构四框图 |
| §2.1静态batch不能及时复用空席位 | 03§1及两步图；11§1 | 直接说明新请求下一step加入；推断标签保留 |
| §2.2 token/input/longprefill/model length与KV联合准入 | 11§2–5 | 预算、request上限、encoder、KV顺序均保留，且有具体数值 |
| §2.2 victim抢占、撤回本步token/encoder预算、重试 | 11§4 | L3/H2占5可用块，撤销L:1→归还3块→给H第3块可重放；不是通用异常事务 |
| §2.3三断言、配置拒绝、长请求不可准入 | 11§2/3/7 | token/input非负、总token和running上限；maxbatch<maxseq及关闭chunked时maxbatch<maxlen拒绝保留 |
| §2.3抢占computed归零、回waiting、累计preempt和重算成本 | 11§4/6 | 当前细化prefix恢复不必重算全部、非swap；原5可用块测试例完整强化 |
| §3.1 KV随computed/new/lookahead增长并maxlen截断 | 12§1/3.2 | 按需slot和尾块碎片保留；连续最大预留的替代成本明确是分析推断 |
| §3.2有限BlockPool、请求表与hash/free queue共用对象 | 12§1–2 | active/cache双身份、共享/回收/驱逐同一容量完整保留 |
| §3.2需求/free/reserved/watermark与失败前安全滑窗释放 | 12§3.1–3.2；11§4–5 | 新安全进度9−inflight2=7例具体化；分配失败可以已有安全回收，不误称纯无副作用 |
| §3.3 refcount/private可写/null/hash与释放优先级 | 12§2、§4 | 正确条件为非null/ref1/无hash；零引用可缓存也可驱逐，删除hash不强拆活跃引用 |
| §3.3 append-only重复块、末token logits重算、对齐可能扩大重算 | 12§2.1/4开头 | 普通表不为去重重写；保留重复占块成本。当前partial CoW能改私有tail，因此收窄“永远append-only”而非机械保留过时绝对说法 |
| §4.1连续batch相邻重合、Python全量重建与CPU间隙 | 03§4.5；16§1/3/4；15输入组织 | 差量原因保留，提供具体ragged四数组和稳定行映射 |
| §4.2Engine先填在途队列、不能立刻得到真实结果 | 03§3.4；10§4 | FIFO计划/future配对与等待分支独立解释，非一次future即全部完成 |
| §4.2AsyncScheduler placeholders和stale不改已重置计数 | 11§6；03§5.3 | 旧“stale不改计数”仍保留；当前普通stale可交付与drop-mode丢弃分别解释，避免旧基线一律忽略误读 |
| §4.2永久row/投影/staged提交 | 16§1–5 | 实际row3/1→idx1/3→token3，staged/group fused和完成边界均落实 |
| §4.3pinned race/barrier成本/生命周期隔离 | 15§6；16§4 | 输入事件和输出事件区分；MRV2普通CPU源/UVA快照/长期base分开；max(2,n)池、无逐槽event前提明示 |
| §4.3显式async失败/auto关闭/pooling默认关闭 | 16§8.2 | 当前executor/spec/disablepadded/ROCmDeepEPHTDBO矩阵已逐项核对；与runner选择独立 |
| §5.1所有组合一条快路径、compile/full强耦合问题 | 03§4.6；14§3；23§1/3 | 以同为3token的decode/mixed及compile/graph两轴例承接 |
| §5.2能力声明/统一reasons/显式指定拒绝、auto优先有效 | 14§3.1–3.2 | 当前connector基类默认true，不能保留旧02“所有语义默认拒绝”的泛化；旁路注入/backend_per_kind额外边界明确 |
| §5.2runtime full/piecewise/NONE、backend限制 | 23§1/3/7–9；16§9 | 两级miss及拒绝/降级条件分别保留，不把所有不兼容当自动fallback |
| §5.3显式blocksize排除高优先backend、不能保证最快 | 14§3.2/5 | 警告与allocator/kernel粒度成本明确，能力政策不等于实测最优 |
| §5.3capture时间/显存/OOM与尺寸约束 | 23§3.1/6.3/9 | 原默认混合模式成本已保留并具体化；当前capture upper limit与Mamba容量门有来源 |
| §6四机制可独立配置，共享token/KV/inflight边界 | 11§7末；23§9末；03§4.3–4.6 | 有明确耦合因果，不变成“四开关必须同时打开” |
| §6四项症状与不能推出结论 | 11§7完整覆盖；23§8/9；27定义 | waiting≠kernel慢、KV/preempt≠必须关prefix、ITL不足定位、piecewise/NONE≠compile全坏均保留 |
| Related原03/11/12/16/14/23/27 | 当前03导航与相关页、11/12/16/23之间关系链 | 删除旧02不切断这些机制的可达性；最终链接检查负责机械验证 |

## 本轮只读证据与检查边界

完整读了旧02全部正文及图；03从主问题到Related全文；11各承接章节（§1–8含抢占例、stale、指标表）、12§1–4及承接段、23§1–3/6/8–9，以及14能力选择、10队列、27指标段。15/16是本审阅者本批已逐源写作并由另一协调者正式复核PASS的正文，本轮按旧02迁移条件复读其相应内容；不把本人的先前作者检查谎称为新一轮非作者全文认证。

三批报告的旧02迁移栏、承接页记录和验收状态均已对照；报告只是审阅索引，结论来自当前正文，不以“报告写PASS”代替迁移检查。

本轮重新打开当前源码的承重边界：

- `vllm/config/scheduler.py`三预算字段及注释：scheduled可小于batched，请求数另受max_num_seqs约束。
- `vllm/v1/core/block_pool.py::BlockPool.is_block_writable/free_blocks`：私有条件及零ref回队列；与12一致。
- `vllm/v1/engine/core.py::EngineCore.step_with_batch_queue`：appendleft/pop配对，先填队列再等最早结果；与10承接旧02异步动机一致。
- `vllm/config/vllm.py`当前async显式和auto完整分支：pooling只默认关闭，spec与disablepadded/executor/ROCm条件；与16一致。
- `vllm/v1/attention/backend.py::AttentionBackend.supports_kv_connector`：当前默认true，14已经纠正旧02的泛化。
- `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model`的NONE分支：调用self.model，graph mode自身不设置skip_compiled；对应下述用语建议。

没有执行模型/GPU/分布式/全库门禁，没有改源码。内容迁移审阅PASS不替代最后一次links/build检查，也不把教学例推成性能结果。

## 最小整合动作与非阻断建议

1. 删除旧02正文；移除当前索引中的旧02迁移对照行/旧基线状态，调整总页数和统一基线。历史三批报告可保留“当批未删除”的历史事实，最终整合报告说明现在完成即可，不要篡改历史执行范围。
2. 03初读时仍有旧02 Related和“相邻页面分批重构中”句；本轮末次只读已经看到协调者将其改为Scheduler关系与统一基线说明。该最新改动符合迁移结论，无需再增加前置设计原则段落。
3. 最终搜索仍须区分活跃wiki链接与历史研究记录，不能因历史报告出现旧文件名就恢复一个空壳旧页。删页后重跑现有共享门禁。
4. 非阻断措辞建议：16§9 replay表的“NONE eager”可改成“NONE正常调用模型，仍可能进入compiled callable”；相邻句如说eager fallback，也可限定“不使用CUDA Graph”。23已经清楚区分compile轴和graph轴，真实runner分支self.model仍可能包含编译层。当前16是沿用源码局部Eager注释，不是旧02迁移漏项，因此不阻断删页。

最终判定：**旧02内容迁移PASS；允许在完成导航/状态同步的同一整合动作中删除。** 等待协调者提供最终共享文件diff，再做第二次只读复核。

复读追加：协调者已将16§9改为“NONE 正常调用模型，仍可能经过 compiled callable”（正文实际为正常ASCII NONE），本审阅者重新读取该表确认两轴措辞与23一致；03/05活跃旧02导航均已清除。上述非阻断建议已关闭。

## 最终共享差量复核（删除后）

**PASS，无阻断 finding。** 以 `/tmp/vllm-final-start/` 为整合前快照，逐一阅读实际diff；本轮仍未编辑wiki、共享文件或源码。

- **归属**：快照中的变化只有旧02删除、03/05导航及过渡状态、16一行NONE澄清、域/父/总索引、父技术栈、changelog、radar。其余已交vLLM正文、SGLang/投机专题/Mooncake快照无新增变化。父技术栈代码核验接受独立作者与协调者的三组源码PASS，本复核只确认六个原落点/原导航不丢失、其他框架观察仍为2026-08-18，SGLang学习路径已限定现有编译Pass，标准四行header没有把其他框架冒充成本次全量重核。
- **删页/历史**：旧02当前文件确已不存在；读取 `18ef114f9d220c49e9a54a7eaed0dff53bee0c5a:<原路径>` 的字节与整合前快照完全相同，最终报告给出的历史追溯有效。全wiki去掉代码块/inline code后，没有指向旧02的活跃wikilink；历史changelog仅把旧链接改成代码，事实文字未重写。
- **计数/唯一入口**：实际盘点为25篇vLLM正文、26个域内Markdown（含index），推理框架递归34个Markdown。25篇正文在域索引表中逐篇恰好一行。域25+index、父25+index、总索引26/34完全一致，不重排01/03/04/…编号。
- **基线/radar条件**：域内25篇与父技术栈共26篇页头均含完整199cb9 SHA，没有残余旧基线页。YAML语义diff只有vLLM的checkout与kb_baseline；剔除vLLM记录后，其余repos、vendors、arxiv整个对象与快照相等。checkout解析为 `/Users/suhaibo/97-llm/vllm`，现场 `git rev-parse HEAD` 为完整199cb9，`git status --porcelain`为空。因此本次repository-wide vLLM radar更新条件成立，未波及其他仓库。
- **changelog并行归属**：快照后另有Megatron 36新条目，属于并行工作；本审阅者不将它计入vLLM交付，也未删除/改动。机械比较确认去掉新增vLLM条目、该并行新条目并还原旧链接代码转义后，旧changelog与快照逐字相等。历史保存通过。
- **最终报告**：`docs/research/2026-09-08-vllm-final-integration.md`的三批24+03=25统计、旧02去向、五类入口、父级范围、已知覆盖缺口和未实跑限制与当前事实一致；持久审阅报告链接目标存在。门禁章节目前明确待填，本次PASS只认证共享整合差量，不提前声称全库/构建/渲染已通过。

结论：删除条件已完成，最终共享文件可进入协调者统一门禁；无需再次修改正文或新增迁移存根。
