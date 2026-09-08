# vLLM 第三批：系统专题交付与迁移记录

> 编号说明（2026-09-08 后续整理）：本文记录交付过程，正文中的裸编号保留当时口径；文件名、链接路径已同步为连续编号后的当前值，当前导航以 vLLM 域索引为准。其中“旧02”始终指已合并删除的系统设计原则页。

日期：2026-09-08。本文记录第三批 G/H 的页面边界、守恒迁移、证据限制与验收；正文权威仍是 `wiki/02_engineering/03_infer_frameworks/vllm/`。

## 范围与基线

用户指定第三批：G 编译算子（23/24/25），H 规模扩展（17/22/26/28/29），共八篇现有正文。路径与编号保持；不新增 LoRA 或 MoE/EPLB 专题。本批不删除旧02，不改前序批次正文；最终全域索引、旧页删除和共享 changelog 合并留给计划的最终整合阶段。

附件的 `6b110badbb22d3f66c7218b71138f13b7a6b3419` 已被工作区03、域索引及第一批交付记录明确替换为 `199cb9b964822e59ab9b58d88e7be31eb419a2ae`。本批继承已确立的目标，实际打开该提交的源码和测试重新核验八篇，非仅改页头。提交时间换算 UTC 为 2026-09-07 00:54:32；源码位置 `/Users/suhaibo/97-llm/vllm`，开始时 HEAD 精确匹配且工作树干净。没有更新、切换或修改源码仓库，没有改变全域 radar baseline。

开始时知识库含第一批及其他领域未提交改动，本批只拥有上述八篇与两个本批报告。只读它们作为导航上下文，不把其他作者正文当源码证据。执行过程中第二批04/13/18等亦继续变化；本批未覆盖这些并行改动，最终统计须按交付时工作树重新计算。

## 页面合同与交付状态

| 页 | 主要读者问题 | 拥有范围与相邻边界 | 状态 |
|---|---|---|---|
| 17 Serving控制面 | 两个API/两个副本如何启动、ready、路由及退出？ | 进程与服务生命周期；协议04、Engine10、collective22、统计27 | 独立复核PASS |
| 22 分布式执行 | 部署目标如何落实为切分、通信、重建与同步？ | TP/PP/DP/EP/CP、DBO、EPLB；局部算子24、在线更新29 | 独立复核PASS |
| 23 编译与CUDA Graph | 同一动态batch如何编译与重放、付出什么代价？ | compile/cache/capture/dispatch；IR变换25、kernel24、runner15/16 | 独立复核PASS |
| 24 融合算子 | 融合改变哪些数值步骤、内存流量和兼容条件？ | residual/RMSNorm/quant与MoE局部算子/provider/workspace；并行通信22 | 独立复核PASS |
| 25 IR与Pass | 图变换为何正确、哪些alias与形状使它不能做？ | IR语义/functionalization/fusion/lowering；生命周期23、设备实现24 | 独立复核PASS |
| 26 跨实例KV | producer的状态如何成为consumer可计算且可回收的状态？ | transferable groups、直连/store、lease/完成/失败；本地cache12、权重版本29 | 独立复核PASS |
| 28 插件 | 扩展如何在正确进程、正确时机对请求可用？ | 各插件ABI、选择、初始化、LoRA resolver；模型13、runner15/16、观测27 | 独立复核PASS |
| 29 在线更新 | R在旧版本已有输出时如何换版、何时能继续？ | pause/sleep/session/原位写/version/cache/draft与失败；collective22、KV26 | 独立复核PASS |

每次写作只修改一篇正文；同一工作包逐篇交付。非作者 reviewer 不改正文，按五项 rubric 开源码复核并查看渲染图，结果单独记录在 `docs/research/2026-09-08-vllm-system-topics-review.md`。

## 已交页面的旧内容守恒

### 17

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| §1 五类owner/四类入口/headless/API默认解析 | §1–2保留，新增双API双DP实例与模式表 |
| §2 分层设计与替代 | §2保留，启动调用图改为分阶段完成表 |
| §3 动态端点、worker/Core/coordinator/client/app ready、barrier | §3保留；补pipe/sentinel同时就绪；hash检查限定coordinated MoE DP |
| §4 stats/inflight/KV penalty/abort/背压 | §4保留，增加精确算分；纠正前端已存在可选admission，传输和DP选择器仍不提供硬cap |
| §5 watchdog/进程故障/FT | §5保留；补单API与backend限制及health边界 |
| §6 deadline/drain/abort | §6保留；ROCm与supervisor清理宽限区别于请求退出预算 |
| §7–8冲突/排查/发展与P2C TODO | §7保留并区分实现与推断；原六个Related目标保留 |

### 22

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| §1六轴动机、选型与通信代价 | §1–3保留；新增TP bias只加一次得305、PCP七token恢复索引、DCP两路径加权得8 |
| §2–3逻辑轴/进程/时间、world size、八rank例与GroupCoordinator | §4保留；补split-group、dense无EP、elastic分支 |
| §4–5Executor/worker/backend/output rank与PP+TP方法链 | §5保留；纠正connector aggregator和懒接收完成，跟到采样与Scheduler提交 |
| collective三不变量、DP dummy反例 | §5保留 |
| §6DBO全组决策/先yield/events/workspace/prefix否决 | §7保留；旧MRV2全不支持改为窄条件支持且graph NONE，PCP+DBO仍拒绝 |
| §7症状排查与原Related | §8及末尾保留，原七目标不丢失 |

新增§6以逻辑expert与物理slot区分EPLB负载记录、同步和异步权重搬迁、map发布与consumed确认；它补齐既有EP owner的完成合同，局部MoE计算仍归24，sharded_rdt互斥接29。

### 23

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| §1 编译图/执行图双职责、启动预编译 | §1/4/5保留；新增三token普通实例 |
| §2五owner、§3两轴/兼容求交/breakable | §2–3保留；抽象调用图换成输入→编译区间/捕获容量→有效输出原理图 |
| §4 guards/ranges/splitting | §4保留并核验当前配置 |
| §5 hash/AOT/rank缓存 | §5保留；纠正任何变化必然换key的过度概括 |
| §6 地址/两种descriptor/pool | §6保留，补LoRA预映射、num_ubatches、offloader与profile清理 |
| §7两层miss与回退 | §7保留；graph NONE仍可compile，首次capture受guard而非任意miss即capture |
| §8失效与恢复推断、§9诊断/冲突/成本、§10有锚点方向 | §8–10保留；所有旧Related目标保留；末尾新增稳定读码路线 |

### 24

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| §1收益账本与双benchmark | §1.2后保留；前置H=4的residual/RMSNorm/INT8实例与中间态流量账 |
| §2三层选择/native过滤/平台默认/opaque边界 | §2保留 |
| §3语义、CUDA/AITER/Oink guard、调用链、测量边界 | §1/3保留，补weight=None与ROCm高维；纠正fused add+norm等于单次遍历的概括 |
| §4 modular/monolithic、oracle、fallback、workspace、async/deferred完成 | §4.3–4.6保留，增加四个top-k slot数值重放；纠正Triton GEMM1/3实际复用workspace2，当前M_chunk=M_full |
| §5选择顺序与四类fallback | §5保留，补LoRA提前分支和humming未量化auto例外 |
| §6副作用/ABI/workspace/UUID/cache、§7有锚点方向 | §6–7保留；§8稳定读码路线；原五个Related保留 |

`FusedMoEConfig.swiglu_limit` 注释声称通用支持检查会过滤clamp，但当前通用 `is_supported_config` 未读取它；正文分别说明Triton activation已实现和外部family未证明，不将注释当作统一保障。

### 25

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| stable dialect、替代方案、四层合同 | §1–2保留；新增residual数值实例 |
| default/maybe_inplace、schema/fake/provider validation | §2–3保留并核源码 |
| donation、later-user拒绝、eager UB区别 | §3保留；新增donated/non-donated值与storage重放 |
| clone layout、graph input、unknown HOP、无一般alias证明 | §3保留；纠正纯读clone不要求donation，unknown HOP不等于直接保留 |
| pipeline完整阶段表与具体顺序 | §4.2–4.3保留；旧流程图替换为add/RMS/reshape双输出变换 |
| SP、AsyncTP与临时residual slice | §4.4新增rank/shape图，明确residual分片合同 |
| quant、KV dependency、guards、final defunctionalization | §5保留，补output buffer与scale变换 |
| fake lowering、UUID、测试、TODO与原Related | §6–8保留并增九项读码路线；原五目标全留 |

当前限制明确列出：SplitCoalescing不比较dim；clone pass缺metadata时并不自动保守；UE8M0 packed scale测试skip不代表安全fallback；Oink拒绝weight=None与lowering测试统一断言存在条件性冲突；attention quant的旧FP8注释已窄于NVFP4注册实现。均是静态核验结论，不修改源码或宣称已设备复现。

### 26

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| §1分离收益、延迟模型、FIFO、双侧connector | §1/3保留，加入12-token/3-block贯穿案例 |
| §2配置、三层身份、兼容性 | §2保留；纠正hash包含全部几何的误述，区分运行期异构映射 |
| §3transferable groups、HMA、SW/SSM | §3保留，明确transfer tuple index |
| §4四ready与七跳闭环 | §4保留，补partial-tail CoW、在途容量预留、V2 hooks、聚合及末token重算 |
| §5NIXL pull/push与MoRIIO | §5分路径保留；纠正threshold分支、push worker会合；MoRIIO图区分WRITE本地ACK与两级乱序缓冲 |
| §6Mooncake key/job ref/load errors | §6保留；补StoreLayout委托，纠正job完成意味着删除远端对象的概括 |
| §7heartbeat/lease/clock | §7保留，明确默认30/5秒及反向复用不受该heartbeat路径覆盖 |
| §8失败策略与两侧清理 | §4/7保留；默认fail、HMA invalid-block TODO、失败完成不等于成功 |
| §9容量/兼容/后台keep-alive、原Related | §2/4/8保留，补版本/reset与29接缝；原六目标全留 |

独立复核修正已完成并通过复读：将泛化Scheduler恢复算例（仅上报invalid block41，computed截到4）与当前NIXL整组handle失败（本例40/41/42均无效，computed到0）分开，避免错误外推。WRITE的完成ACK由P本地生成，不额外等待D回ACK；四图分别重放pull、push、MoRIIO和store的完成合同。

### 28

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| §1 discovery/import区别、进程范围、import-time副作用 | §1保留；前置管理接口实例 |
| §2各ABI与选择冻结点 | §2表保留；生命周期总图改为同一URL在API/render的不同结果 |
| §3.1 allowlist空/未设/entry-point名与实例name注释冲突 | §3.1保留并按当前loader验证 |
| §3.2 general guard先置位、各首次消费点、partial mutation | §3.2保留；稳定路线列全消费锚点 |
| §3.3 platform唯一性/惰性/重复探测、IO优先级与错误 | §3.3保留；补显式CPU提前返回 |
| §3.4 endpoint两阶段、render None、冲突和无teardown | §3.4保留；独立复核纠正“client尚未建立”为“尚未向插件注入”；重复路由描述限定vLLM顺序与上游文档风险，不冒称外部框架匹配实现 |
| §3.5 filesystem/HF注册、前端快照、名称锁、400/404、覆盖顺序 | §3.5保留；补test-lora实例与提交原理图，resolve自身抛错不在add_lora捕获范围 |
| §4错误矩阵与五项部署验证 | §4保留；新增§3.6 stat logger类型/构造与27接缝；全部旧Related目标保留 |

### 29

| 原位置与独有内容 | 当前去向与核验 |
|---|---|
| §1可见性动机/替代、§2四owner四种完成 | §1–2保留；前置R已输出“上海”的step-41→42实例 |
| §3 abort/wait/keep、clear_cache/stale、sleep/wake | §3保留；补当前按token份额丢弃stale，普通抢占与reset区别；DP resume另见§7.5 |
| §4控制/worker/version失败全图 | §4同一R与两worker序列图，依旧体现非原子提交、partial writes、完成/标签分离；main/draft差异留§5–7，sleep路径留§3 |
| §5 init/start/typed metadata、dense/IPC/sparse/RDT、layerwise稳定storage | §5保留；纠正sparse“最少buffering”为完整checkpoint形状NaN暂存，补最小W变换图、loader边界、RDT拒EPLB与drain队列/stream/producer信号 |
| §6 finish/version/普通reload与transfer收尾差别 | §6保留，无版本checksum/CAS/2PC/snapshot保证 |
| §7 request/KV/MM/encoder/draft/graph/LoRA | §7保留，补connector None/False与Core忽略bool的可见冲突、跨实例KV非自动版本隔离、Marlin辅助地址及DP暂停共识 |
| §8失败清理/不能回滚/三条操作不变量 | §8保留；稳定读码路线放§9；全部七个Related目标保留，正文新增23/26/17链接 |

## 旧02与前序批次交接

| 旧内容或前序要求 | 本批去向 | 边界与状态 |
|---|---|---|
| 旧02§5 graph双轴、兼容条件、降级成本与启动内存 | 23§1/3/6/9 | G拥有的部分已核验；Attention能力选择仍14，async兼容仍15/16 |
| 旧02§6四机制耦合、图回退不代表编译整体失效、指标不充分定位 | 23§8–9，导航11/12/16/27 | 编译侧已核验；其他owner由D/E核验，不以调优表替代机制论证 |
| 第一批23应解释enforce_eager诊断 | 23§1/8 | 当前同时禁compile/graph；成功运行不足以单独定位是哪层问题 |
| 第一批22应解释通信/hang拓扑 | 22§4–5 | 拓扑与完成合同已解释，实际多机运行排除 |
| 旧02§1–4调度/分页/异步具体内容 | 原02及D/E队列 | 本批未删；删除前须最终协调者逐项确认 |
| LoRA与MoE/EPLB候选独立分页 | 28/29与22/24原有分工 | 不新增专题；先把已找到组合限制写入当前owner |

## 组合场景核对

- 编译/Runner：graph模式、compile模式分开；23只解释微批/LoRA descriptor接缝，不承诺无上限LoRA组合。
- Serving/观测：17与第一批27共同区分alive、health和恢复完成；不把frontend admission泛化为每个transport的容量保证。
- EPLB/在线更新：29已核验sharded_rdt静态计划明确拒绝EPLB；22已补专家搬迁与map发布，24只拥有局部MoE数值计算。
- KV/在线更新：29已核验版本标签不触发跨实例失效；26已分别解释具体connector/store完成与回收，未实现reset的外部存储不能被当成已清空。
- LoRA/插件：28名称锁仅当前frontend；resolver找到文件、engine加载、frontend映射发布分开。resolver接入不构成同名adapter热换版或分布式cache一致性保证。

## 覆盖缺口与候选分页判断

这里区分正文解释、仅给接缝、尚缺验证和本批排除；不是整仓功能树认证。

| 项目 | 覆盖状态 | 证据与后续归属 |
|---|---|---|
| graph/compile双轴、动态容量与失效 | 已解释 | 23，旧02编译部分有逐项去向 |
| IR改写、donation/alias与SP重建 | 已解释 | 25；未实现一般alias证明，具体数值/scale未覆盖项已标出 |
| 跨实例KV四数据面及失败/引用收尾 | 已解释 | 26；远端发现、目标容量、数据有效与解除持有分开，HMA错误报告缺口明确 |
| 插件各ABI、endpoint API/render结果、LoRA resolver提交 | 已解释 | 28，源码与mock测试接线；数值执行15/16 |
| 旧请求跨权重版本、sparse真实暂存、RDT/EPLB互斥、缓存清理限制 | 已解释 | 29；26已接续具体跨实例协议及reset前置条件 |
| 各模型LoRA包装、adapter slot与低秩计算 | 本批仅接缝 | 13/15/16归E；28/29不重复写模型或Runner算法 |
| 多API同名LoRA热换版与旧KV安全 | 本批缺少端到端保证 | 实际打开 `OpenAIServingModels.unload_lora_adapter` 只删除前端映射；`kv_cache_utils._gen_lora_extra_hash_keys` 只返回LoRA名称。不能从这两个接口推导全worker卸载或内容版本隔离；最终协调者可在12/28现有边界补热换版反例与测试要求，无需先新建专题 |
| MoE局部计算与EPLB布局/发布 | 已解释 | 24的四slot数值过程与22的逻辑/物理映射、搬迁和提交分别拥有；全部外部专家库实现不在本批核验范围 |
| 外部transport内部故障恢复与跨实例原子换版 | 本批排除 | 26/29仅证明vLLM交接接口；依赖库内部、部署编排需独立运行证据 |
| GPU性能收益、真实多机hang恢复 | 本批排除 | 示例是静态推演，未开展测量或故障注入 |

候选分页结论：目前找到的缺口集中在现有owner之间的组合保证，而不是已经批准的全新独立机制。保留LoRA的13/15/16/28/29分工与MoE/EPLB的22/24分工，不擅自增加页；若后续需要完整多API热换版专题，应先给出边界与验证合同再走规划审批。

## 证据限制与最终整合待办

正文列出的源码与测试均以实际打开为准；本批没有执行GPU模型、基准测试、NCCL/RDMA通信、Ray/Rust服务、在线换权重或外部插件下载。数值与请求例子标明为教学推演；读取mock测试合同不等于真实设备验收。

本批八篇均已完成，非作者独立复核 **8/8 PASS**；逐篇五项rubric、正式抽查的三个源码锚点、hop-walk、20张最终渲染图及23/26/28首轮最小修正记录，见 [独立复核报告](2026-09-08-vllm-system-topics-review.md)。25的非阻断措辞建议也已采纳。

最终机械验收：

| 检查 | 实际结果 |
|---|---|
| `check_links.py --strict` | 448页；broken、ambiguous、bare_index、stale_section、orphans均0 |
| `check_math.py --changed --strict` | 37个Markdown文件，0错误/0警告 |
| `check_markdown.py --changed --strict` | 37个Markdown文件，0错误/0警告 |
| `check_assets.py --changed --strict` | 37个Markdown文件，0错误/0警告 |
| `mkdocs_site.cli build --changed` | 构建成功，33条改动route；broken links、missing anchors/assets/legacy routes均0；scoped orphan检查跳过，全库links检查已确认0 |
| `mathjax-corpus.mjs --pages <本批八篇>` | 八篇均检查；三篇含公式，合计10个公式实际渲染PASS |
| 本批图示 | 20张Mermaid图已实际渲染并由作者与非作者查看；最后修订的26图0/2已重新打开 |
| 基线/读码路径/空白检查 | 八篇均固定199cb9完整SHA；显式path:line残留0；提取到的源码文件路径均存在；本批正文diff空白检查通过 |
| 源码仓库 | HEAD仍为199cb9完整SHA，工作树干净；未修改源码 |

37个changed Markdown和33条route包括工作区其他批次/领域的并行改动，这是执行时工具实际范围，不能表述为本批拥有的文件数。本批拥有八篇正文和两份报告，没有新增正文、改编号或提交git commit。

交付时vLLM目录实际有26篇编号正文，按页头统计为21篇199cb9、5篇6b110bad；该数字含并行第二批的当前进度，最终协调者仍须在整合时重新计算，不能将其冻结为后续全域状态。旧02仍在。本批未更新共享索引/changelog；最终整合需核对D/E/F迁移、处理本报告登记的组合验证缺口，然后统一决定旧页删除与导航更新。本报告不宣称全域重构完成。
