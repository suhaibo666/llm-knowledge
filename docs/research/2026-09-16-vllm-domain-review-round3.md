# vLLM 01–26 第三轮验收（2026-09-16）

对象是第二轮验收（[recheck](2026-09-16-vllm-domain-review-recheck.md)）之后的第二次修复：工作区 +2104/−729，含 26 页正文、index、changelog，以及新增的 `courses/torch_compile_end_to_end.md` 与推理技术栈页的两处补链（均未提交）。四个与写作无关的评审者逐条验收上一轮的开放项，并把本轮新增文字当新稿重新核验；协调者复核了全部阻塞项与新发现的关键事实。

- **源码**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，只读，HEAD 未移动。
- **明细**：[`2026-09-16-vllm-domain-review-round3/`](2026-09-16-vllm-domain-review-round3/)。

## 0. 结论

| 维度 | 第二轮 | 本轮 |
|---|---|---|
| 阻塞（B1 站点构建） | `mkdocs build --changed` 因三处锚点失败 | **已修**：构建通过（35 页，broken_links / missing_anchors / missing_assets / missing_legacy_routes 全 0，协调者亲自复现） |
| 第二轮 10 条新 P1（B2） | — | **全部修复**并经源码核验 |
| 02 架构 item 5/8（B3） | 未动 | item 8 **通过**；item 5 因本轮新增文字的三条新错误**仍不通过** |
| 重复解释（第二轮唯一变差项） | 变成三份 | **完全收口**：每个概念单一 owner，推出方只剩链接；删文逐条做了保留核对，仅一处数值算例全域丢失（P2） |
| 域级 | 27 处带锚点链接、124 处跨页 `§` 引用 | **全部命中**，无悬空；index 新增 11 行覆盖边界表；课程页补链符合课程层规则 |
| 新 P1 | 10 条 | **9 条**（02×3、07、12、18、22、24、index 各 1） |
| 页面判定 | PASS/ACCEPT 17 / 条件 4 / REJECT 3 | **PASS 或 ACCEPT 20 篇**；条件通过 5 篇（07、12、18、22、24）；**REJECT 1 篇（02）** |

T0 四项全绿；改动页 25 个公式 MathJax 渲染 PASS；`git diff --check` 无空白错误。

## 1. 本轮做对的部分

- **B1 三处锚点**已改正（24→14 §4.3、25→13 §3.1、25→26 §6），域级评审者重扫 27 处带锚点链接，目标标题全部逐字存在，渲染后 24/24 条唯一 href 可解析。
- **B2 十条全修**，且改法是对的而不是绕开：
  - 20 §5.3/§13 的 AITER 链改成 `…_impl` → backend `do_qk_norm_rope_kvcache_update` → `_aiter_ops.py::rocm_aiter_ops.do_qk_norm_rope_kvcache_update` → `fused_qk_norm_rope_and_cache` → AITER kernel；§13 阅读路线锚点也跟着改了；
  - 20 §3.5 的 lowering UUID 口径与 `lowering_pass.py` 及 21 一致（遍历全部注册 op 的全部 provider，`selected_impls` 只进日志）；
  - 24 补出前端与 runner 两个进入点、`lru_cache` 共享、两种失败面与双调用树；
  - 17 §6.1 与 09 §2.8 统一为“凡挂 `quant_method` 的模块参数都无条件补进 loaded set”，并指明 owner 是 09；
  - 18 三处改为 `DPEngineCoreProc`，并注明基类没有这两个入口、与 23 §6.3 一致；
  - 19 图 1 改为 `post_update` 写 `RequestState.last_sampled_tokens / all_token_ids`；图 2 与 §6.5 改为“resolve 不在首次编译之前且跑两次”，与 12 §2.9、11 §2.9 一致；§4.3/§9.1 的 `compile_sizes` 约束限定到构造 `CudagraphDispatcher` 的 MRV1 与两个 drafter；
  - 12 的 Mamba `align` 记账改成 hash 粒度并标注该说法来自字段 docstring，`max_concurrent_batches` 四行规则与 `config/vllm.py` 一致。
- **重复解释收口**（第二轮唯一变差的一项）：stale/drop 归 07 §8.3（06 §5.1 压成一段不变量、删图并把编号 5→4 重排干净）；完成信号表归 06 §6.1；`finished_req_ids` 生产与“不得 `clear()`”归 07 §8.1.2；async 解析轴归 06 §4.1（07 §2.1 改为指过去）；Mamba checkpoint 判定归 08 §5.1.3。评审者逐条核对被删内容是否在新 owner 处重现，只有 stale 排空的 2→1→0 数值算例全域消失，记 P2。
- **域级可见性**：index 新增 11 行覆盖边界表，把规划待批的缺口对读者显式化；`courses/torch_compile_end_to_end.md` 只加了阅读顺序、三条链接和一句定位，并写明不重复本课内容，符合 CLAUDE.md 的课程层规则；技术栈页学习路径插入 02。
- 08 的六个重写单元在改写后全部保住（准入上限两个调用点、`create_connector` 的硬 `ValueError`、Mamba 判定去重），07 的 11 项 P2 全部修掉。

## 2. 仍需修（9 条新 P1，全部是一句到一格的改动）

| # | 页 §节 | 问题 | 源码事实 |
|---|---|---|---|
| 1 | 02 §2.4 调用树 | 把 `AsyncLLM._run_output_handler` 挂在 `_add_request` 之下、`check_admission` 之后 | `add_request` 在 `_add_request` **之前**调它（async_llm.py:465），且 `__init__`（:186-190）在有事件循环时已提前启动；父节点、顺序、注解三处都错（协调者已复核） |
| 2 | 02 §5.5 | 称完整 render → token-in → derender 往返需要两个进程 | render/derender router 不按 `render` 任务门控：`factories.py` 在 env 门之后无条件纳入，`app_state.py` 对任何 generate 服务都建三个 handler；一个 `vllm serve` 加 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1` 即可 |
| 3 | 02 §5.3 | `VLLM_RUST_FRONTEND_PATH` 的报错条件写反 | `auto` 且找不到二进制才抛 `FileNotFoundError`；显式路径原样返回、不校验（envs.py:581-608） |
| 4 | 07 §8.4 | “delay 与否不是 connector 直接要求的” | `_free_request` 有 `delay_free_blocks |= connector_delay_free_blocks`（scheduler.py:2538，来源是 connector 自己的 `request_finished()`）；WAITING_FOR_REMOTE_KVS 只是另一个来源（协调者已复核） |
| 5 | 12 §3.2 worker 调用树 | LoRA 激活排在 FULL/ubatch/PIECEWISE forward 分支**之后** | `gpu/model_runner.py:1661-1668` 在 `preprocess_state` 之后、forward 之前调用，且是唯一调用点（协调者已复核）；顺带补回 `+--` 标记 |
| 6 | 18 §4.3、§13 | 把 `worker_type="new"` 归给 `EngineCore._eep_scale_up_before_kv_init` | 该方法在基类是 `raise NotImplementedError`（core.py:1023），实体在 `DPEngineCoreProc`（core.py:2397）——与本轮刚修好的 #5 是同一类归属错误（协调者已复核） |
| 7 | 24 §3.1、§6.1 | 举例“插件导入失败会在 MRV2 部署的首个采样请求上炸” | 装了 `vllm.logits_processors` entry point 或给了 `--logits-processors`，`use_v2_model_runner` 就返回 False（强制 V2 则配置校验硬失败），该组合不存在；页内下一句也自相矛盾。改用 MRV1 + 投机解码举例 |
| 8 | 22 §14.4 | 表中仍有一格把“P/D 实例拓扑”写 `→ 13`，而同表下一行与 §2、§14、Related Pages 都已改成“本域暂无 owner，已提交规划” | 13 对 P/D proxy 仍 0 命中 |
| 9 | index 覆盖边界表 | 列“投机解码下的 `logprob_token_ids`”为无 owner | 14 §3.5 明写“这里是它的 owner”，16 §9.1 指过去 |

第 8、9 两条与第二轮 B1 是同一种失误：一次“如实化”改动没有覆盖到全部出现位置。

02 的架构 item 5（场景清单）不通过，原因正是第 1–3 条；item 7（调用树）不通过，原因是第 1 条。其余六项架构检查通过。

## 3. 仍按规划待批（不算失败）

“调用 API / 使用特性”的使用说明 owner（01 覆盖缺口、03 的分类）、P/D proxy 路由、配置解析总链、Ray DP actor 生命周期与 external_launcher 监督、tool/reasoning parser、pooling 执行、非 CUDA 平台、gRPC/Rust frontend、`n>1`/beam search。本轮已把这批缺口写进 index 覆盖边界表，读者可见；23 §6.2 与 13 §4.2 对 Ray 监督改成了“如实登记无 owner”，这是正确处理。

## 4. 一处需要更正的记录

`wiki/changelog.md` 写中文标题锚点“落在页首”。实际不是：wikilink 改写器会在每个标题前输出 `<a name="…">` 别名锚点（协调者在 `site/` 产物中验证过），unicode 与 ASCII 两种形式都能跳转。mkdocs 自带校验只看标题 id，因此会打两条 INFO，仓库自己的 `missing_anchors` 计数已把别名算进去、报 0。这句记录应改掉，以免后人去“修”本来可用的链接。

## 5. 建议收口顺序

1. 上表 9 条，各一句到一格；其中 02 的三条修完即可清掉 item 5/7。
2. 各组新增 P2 约 30 条（明细在分组报告）：优先 06 §4.2 图里把队列项写成二元组（源码是三元组）、08 §5.3 把 `prefix_cache_retention_interval` 说成保留时长、24 §2 的异常类型仍写 `AttributeError`（源码 `serial_utils.py::run_method` 抛 `NotImplementedError`）、19 图 1 残留的 `num_tokens` 来源边、11/12 §1.4–1.5 把“未自定义 Scheduler”当成 async 解析条件。
3. changelog 那句锚点记录。
4. 修完再跑 T0 四项 + `mkdocs_site.cli build --changed`（这一轮证明了带锚点链接与标题改名必须靠构建把关）。

本轮仍为静态读码：未运行 vLLM、GPU、跨机传输或故障注入；除本报告与明细归档外未改 wiki 页面，未提交或推送。
