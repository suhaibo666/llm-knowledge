# slime 系列专家意见整改记录（2026-09-10）

本记录是整改与验收台账，机制正文只在 [slime 功能树](../../wiki/02_engineering/04_posttrain_frameworks/slime/index.md)。输入是用户提供的 `slime_series_audit_20260910.md` 与 `slime_series_fix_plan_20260910.md`；附件作为待核实的建议，没有被当成高于用户请求、`CLAUDE.md` 或源码的指令。

## 范围与基线

- 更新原有 20 篇，新增 26 多模态、27 评估、28 SFT：共 23 篇内容页，加索引 24 个 Markdown 文件。
- slime 固定 `THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`，提交日期 2026-08-12。读取持久 detached worktree `/Users/suhaibo/97-llm/slime-681b3adc`。
- 用户原 checkout `/Users/suhaibo/97-llm/slime` 保持 `4c193f1f37509cca70f0e88807a9305b70f63f4e`，工作区仍干净；没有修改源代码。
- vime 新增段落读取临时 checkout `/tmp/slime-audit-vime` 的 `vllm-project/vime@8144096e3f4fb0fb670c37b8f2d84015f7e92320`。只复核本轮新增范围，不把原页其余事实全部记为重新核验。
- 未推进 v0.3.2；后续提交只作为明确分隔的历史备注。初次交付保留工作区改动；用户随后明确授权提交并推送。

## 整改覆盖

| 页面 | 本轮已落地范围 |
|---|---|
| 01、02 | 三类参数来源、协议载荷、driver/生命周期、role 与服务 YAML、派生约束、解析/失败入口；去除重复时序 |
| 10、11 | 同步发布频率、one-stage async 与 fully-async 叠加；release/save/recreate；full-disk 版本责任、六元组、环境、端口实例、Ray 对象身份 |
| 12、14 | Sample 状态/有效长度/生成与元数据契约；DP schedule 规范归属、静态拒绝/动态拆分、tag 切换、checkpoint/offload/ref/optimizer |
| 15、17 | GAE/dual clip/空梯度连接、rejection 原分母；TIS/MIS 配置与指标、routing 四态与两个开关、MoE 对齐与确定性 route kernels |
| 13、18、19 | 请求/奖励/过滤/oversampling/partial/fully-async；健康检查默认值矛盾、恢复与故障注入；Forge/sleep/hooks/buffer 扩展及 backend 义务 |
| 16、21、22 | Direct 的准确使用范围、bucket 与精度 hook；MTP 层号/PP 依赖/draft 边界；KV-QAT、recipes、UE8M0 条件与 INT4 扩展 |
| 20、24 | teacher server 的队列/请求/更新协议；agent harness/parser/session、共享轨迹去重与 drift 分支、第三方 reward 边界 |
| 23、25 | provider/spec、双向模型 registry、GLU 布局；vime Bridge/迁移入口/证据判据、有界队列与超量消费差异 |
| 26（新增） | 图像记录→processor→双表示→训练特征→视觉注入→packed mRoPE；geo3k 单/多轮边界与源码接口冲突 |
| 27（新增） | 配置优先级、多数据集复制/排序/日志、共享服务评估时序；默认未消费字段、空集与异构 passrate 边界 |
| 28（新增） | 离线对话→Sample→assistant loss mask→SFT loss；真实模板算例、长度过滤/截断/无 assistant 边界 |
| 30、31 | 调优与诊断表绑定实际指标/参数；修正选 rank、fully-async、zero_std/repetition 等口径 |

同步/异步时序归 10，磁盘版本归 11，rollout 分母归 12，DP schedule 归 14，权重提交归 16，TIS/MIS 归 17，恢复链归 18。相关页收缩为定位结论与链接；其余必要上下文保留，不机械删去跨页连接。

## 源码核实后修正的建议

| 附件建议或旧结论 | 实际处理 |
|---|---|
| Direct 只供 tensor updater | 完整 HF saver 也构造 Direct，16 明确两种用途；NCCL/delta 保持独立实现 |
| `--tis-*` 都是核心 CLI | MIS 主要是 custom YAML 属性；17 将 CLI 与配置字段分开 |
| 第三种 OPD teacher 等于第三个 `opd_type` | 核心 enum 仍是两种；独立 teacher server 的返回协议需要适配，20 分开说明 |
| 后续 scale 提交无条件强制 UE8M0 | 实际新增可选条件，22 保留当前基线与后续分支区别 |
| `--eval-datasets` | 此 CLI 不存在；它是配置解析后的内部属性，02/27 修正 |
| 所有“逻辑 rollout”改为 rollout round | `Sample.rollout_id` 的逻辑执行可包含多个片段，不能与一次 generate 批次混同；索引术语分别定义 |
| 索引继续扩写覆盖矩阵和机制正文 | 按本库宪法改为单一页面入口表，新增主题直接体现在 20/26/27/28 行，避免第二套内容权威 |
| 附件最后要求一次提交 | 初次交付时用户未要求提交，因此先交付可审阅改动；后续按用户明确授权提交并推送 |

独立复核另关闭：全局 exactly-once 过度保证、full-disk 图中 publish/pause 顺序、MIS 图缺配置前提、FP8 字节式重复计数、MTP 的 vp_stage 兼容前提、CI simulate_crash 杀服务子进程而非必然杀 Ray actor、parser 异常范围、自定义 generate 与默认请求协议的边界。

两项源码问题已明确标注，未悄悄替源码“修成正确叙述”：26 的非训练 observation 传入 log_probs 与 Sample 契约冲突；25 的一次 drain 后按 target 截断会丢掉超额完成组。它们是文档所审基线的行为，不是本轮改动代码后的测试结果。

## 独立复审与图形验收

| 审查单元 | 独立复审结果 |
|---|---|
| 01/02/10/11/27 | 集成者对照 driver、arguments、placement、actor_group、eval 生成/日志进行改动范围审查；修正 10 的重复消费保证、delta 版本表述和 11 六元组中 num_new 的类型；通过 |
| 12/14/15/17/28 | 由非作者复审；修复 mask 不变量、CP/DP 数值重建图、MoE forward/backward/128-row padding 图、MIS 前提；通过 |
| 13/18/19/24/30/31 | 由非作者复审；九项已发现问题修复并复验；通过 |
| 16/20/21/22/23/26 | 由非作者复审；六项准确性/图示问题修复并复验；通过 |
| 25 | 由非作者对本轮 diff 复审；Bridge、配置、队列反例及新版图通过 |

55 个 Mermaid block 使用仓库现有 Mermaid 与 Chromium 实际渲染，无解析失败。新增/改动图实际打开检查文字、边界与箭头；FP8 block、TP2 GLU、CP2×DP2、mRoPE、MIS、队列及共享轨迹实例能从图中复述处理前后状态。部分旧架构图较宽，网站保留图形缩放查看；本轮没有把全部既有图重设计。

## 检查结果与限制

| 检查 | 最终结果 |
|---|---|
| `check_links.py --strict` | 全库 451 页；broken/ambiguous/bare_index/stale_section/orphans 全 0 |
| `check_math.py --changed --strict` | 28 个改动 Markdown 文件；0 错误、0 警告 |
| `check_markdown.py --changed --strict` | 同上；0 错误、0 警告 |
| `check_assets.py --changed --strict` | 同上；0 错误、0 警告 |
| `check_locators.py --dir .../slime` | 原生命令 errors=0、warnings=2、env=23；须结合下述实际 checkout 补充审计解读，不能冒称全引用通过 |
| 按实际 checkout 补充 locator 审计 | 1073 pass，0 missing/out-of-range/ambiguous/region-sized；3 unresolved_repo、28 unverifiable 均为保留 SGLang 引用 |
| `mkdocs_site.cli build --changed` | 27 个网站路由构建成功；broken_links/missing_anchors/missing_assets/missing_legacy_routes 全 0；scoped orphan 检查按工具设计跳过，全库孤页由 links gate 验证 |
| `mathjax-corpus.mjs --pages <slime 24页>` | 24 页检查，14 页中的 163 个公式实际渲染通过 |
| Mermaid 实际渲染 | 55 图，0 解析失败；新增/修正图已完成独立视觉复验 |
| `git diff --check` | 通过 |


- 本轮不声称进行了 GPU/Megatron/SGLang 训练或多机 E2E；配置和实例以固定源码与仓内测试定义核实。
- 16/21/23 保留的 SGLang 跨仓引用缺本地对应 checkout；页头已注明没有重新核验依赖内部结论。
- 原生 locator 命令受 watchlist 的旧相对 checkout 配置影响；补充审计只在内存映射实际 slime/vime checkout，没有改 watchlist。1073 条可检查的 legacy locator 通过，0 missing/out-of-range/ambiguous/region-sized；剩余 3 个 SGLang 页面环境提示与 28 条未验证引用不算通过。
- 对 `.github` / `.buildkite` 引用使用完整固定提交文件链接；legacy 检查器的 `lstrip("./")` 会误去掉目录前导点，故没有用它误报的 `path:line` 标签。实际目标文件存在，URL 保留原行区间；未改检查器。
- 独立复审的通过范围是本轮修改及承重边界，不是对整套框架所有配置组合的运行担保。
