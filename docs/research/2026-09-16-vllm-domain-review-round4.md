# vLLM 01–26 第四轮验收（2026-09-16）

对象是第三轮验收（[round3](2026-09-16-vllm-domain-review-round3.md)）之后的收尾修复：9 条 P1 加约 30 条 P2。两个未参与写作的评审者按页分工验收，并把本轮新增文字当新稿核验；其中一人用 `git apply` 复原第三轮页面状态后再与工作区对比，把本轮改动精确隔离为 44 行新增，逐行回源码核对。协调者独立复核了全部 9 条 P1 与本轮两条新发现。

- **源码**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，只读，HEAD 未移动。
- **明细**：[`2026-09-16-vllm-domain-review-round4/`](2026-09-16-vllm-domain-review-round4/)。

## 0. 结论

| 维度 | 第三轮 | 本轮 |
|---|---|---|
| 第三轮 9 条 P1 | — | **全部修复**，协调者逐条对源码复核 |
| 02 架构检查 | item 5、7 不通过 | **八项全通过**，02 判 PASS |
| 第三轮约 30 条 P2 | — | FIXED 37 条 / NOT_FIXED 8 条（均 P2/P3，不阻塞） |
| 新 P0 | 0 | **0** |
| 新 P1 | 9 | **2**（均为一词/一处引用） |
| 页面判定 | PASS/ACCEPT 20，条件 5，REJECT 1 | **PASS 或 ACCEPT 24 篇**；条件通过 1 篇（25）+ index；01、03 仍待规划裁决 |
| 门禁 | 全绿 + 构建通过 | T0 四项全绿（453 页五项为 0；27 文件 0 错 0 警）、`build --changed` 通过（35 页五项为 0）、`git diff --check` 无空白错误、改动页公式渲染 PASS |
| 锚点与引用 | 27 处锚点全中 | **34 处带锚点链接全部逐字命中**；页面内 82 处跨页 `§` 引用无悬空（index 有 1 处，见下） |

## 1. 仍需修（2 条 P1，各一处）

| # | 位置 | 问题 | 源码事实 |
|---|---|---|---|
| 1 | 25 §13 源码路线（L365） | 写 `EngineCore._has_global_unfinished_reqs` | 该方法只存在于 `DPEngineCoreProc`（`vllm/v1/engine/core.py:2244`，类自 2015 行起；`coordinator.py:41` 的注释也这样称呼）。这是连续第三轮出现的同一类“基类 vs DP 子类”归属错误 |
| 2 | index 覆盖边界表（L17） | 引用 `03 §9` | 03 只有 §1–§6，输入路径缺口登记在 §6（L291）。这一句是本轮为修第三轮 P2 新写的；裸 `§` 引用不被 `check_links` 覆盖，会直接漏到线上 |

## 2. 值得顺手修的 P2（不阻塞）

- **11 §1.4 残句**：前半句已改对（“该解析不读 `scheduler_cls` 也不读 PP”），后面留着旧文“`async_scheduling=None` 仍解析成 True，因此使用 `AsyncScheduler`、容量 2”。两句自相矛盾，且对 MRV1 + async + PP>1 不成立（`max_concurrent_batches` 返回 `p`）。删掉后半句即可——这是本轮唯一残留的事实性缺陷。
- **07 §8.4**：“delay 有两个独立来源”仍少算一支，`_connector_finished` 返回的是 `delay_free or partial_tail_delay`（`scheduler.py:2847`，协调者已复核）。
- **06 §4.1**：新增的 `model_executed` 段把本节结论句吞进去了，“因此……缺省是 AsyncScheduler + 容量 2”读起来像从 EC producer 那点推出来的。事实无误，是段落切分问题。
- **6 处“标签写子节、锚点落父节”**：17:193、18:352、07:411、07:492、12:229、16:394。检查器看不到这一类。
- 其余：20 §5.2 的 HIP in-place 措辞（`forward_hip` 的 `use_aiter` 分支是独立实现，不是复用 `forward_cuda`）、20 §5.3 的 `pairOffset = 4 / 2` 边标签、20 两张新图未用房内配色类、22 L1019 页级 `→ 23`、26 未写明 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 实际默认为 1（`envs.py` 的类型桩写 `False`，是个真陷阱）、26 §1.1 新拥有 `Executor.get_class` 表而 18 仍在四处自行枚举“六分支”。

## 3. 本轮确认修好的关键项

- **02**：§2.4 调用树的 `_run_output_handler` 改为 `add_request` 的直接子节点并注明 `__init__` 已提前启动，三个 `_add_request` 子节点与 `async_llm.py:492-512` 一致；§5.5 改为“一个普通 `vllm serve` 加 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1` 即同时暴露三段”，三态门与 `ValueError`/info-return 分支与 `factories.py:66-100` 一致；§5.3 的 Rust 路径改为“`auto` 找不到才抛 `FileNotFoundError`，显式路径不校验”。item 5 与 item 7 的失败原因随之消除。
- **07 §8.4**：改为“delay 由 `_free_request` 求或”，四条新表述均成立；§8.3 补回了第三轮全域丢失的 2→1→0 排空算例。
- **12 §3.2**：LoRA 激活移到 forward 之前，且是唯一调用点。
- **18**：`_eep_scale_up_before_kv_init` 等归属改为 `DPEngineCoreProc` 并注明基类没有实现。
- **24**：举例改为 MRV1 + 投机解码（runner 早返回、前端仍加载），并写明“装了该组插件就阻断 MRV2”。
- **22 §14.4** 两处单元格与 **index** 覆盖边界表一致化；index 还进一步标出每条缺口“由哪些页面登记”，评审者逐行核对无一行与页面冲突（tool/reasoning parser、`n>1`/beam search 确为全域 0 登记）。
- **06 图 2** 的队列项改为三元组，与 `batch_queue.appendleft((future, scheduler_output, exec_future))` 一致（协调者已复核）。

## 4. 范围之外

工作区同时含另一条工作线的改动：新增未跟踪页 `wiki/02_engineering/04_posttrain_frameworks/23_dora_multi_version_rollout_analysis.md` 及其 assets，并在 `raw/README.md`、posttraining frontier map、posttrain index、verl 22 各加一行登记。本轮未评审这些内容。

## 5. 建议

先修 §1 的两条，再顺手处理 §2 的 11 §1.4 残句与 07 §8.4 的一支来源；之后重跑 T0 四项加 `mkdocs_site.cli build --changed` 即可收口。01、03 的使用说明归属仍等规划裁决，index 已把它登记为已知缺口。

本轮仍为静态读码：未运行 vLLM、GPU、跨机传输或故障注入；除本报告与明细归档外未改 wiki 页面，未提交或推送。
