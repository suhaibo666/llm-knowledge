# vLLM 24–26 分类重构与独立审读

本轮重构现有 24（扩展插件系统）、25（在线权重更新）、26（MultiprocExecutor RPC）三页，不增加页面、不改变功能树归属。用户指定具体功能页参考 [[02_engineering/02_train_frameworks/megatron-lm/12_megatron_tp_analysis|Megatron-LM 12]] 的解释顺序；非单一功能则按知识库机制分析画像组织。因此最终分类为：24 是“接口—注册—生命周期”机制分析，25 是功能分析，26 是并发与分布式机制分析。

源码冻结为本机 `/Users/suhaibo/97-llm/vllm` 的 `vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，`main`，提交时间 2026-09-07（Asia/Shanghai）。源码 checkout 只读；开始和结束时都只有既存未跟踪目录 `artifacts/`，本轮没有检查、修改、提交或推送源码。

## 页面合同与承接关系

| 页面 | 文档画像 | 主解释路径 | 前后承接 |
|---|---|---|---|
| 24 扩展插件系统 | 机制分析 | Endpoint 真实例子 → 公共发现底座 → 五类 ABI 与不同冻结点 → LoRA 运行时二阶段 → 所有权、调用树、失败与部署门 | 从 23 的运行期观测退到启动期扩展；把真正运行时模型状态变更交给 25 |
| 25 在线权重更新 | 功能分析 | `step-41 → step-42` 请求例 → pause/session/finish/version/resume → Trainer/Worker 双侧 factory → 四类数据通路 → 派生状态、失败和配置 | 从 24 的“代码进入进程”转到“权重安全变更”；把 collective 的进程内实现交给 26 |
| 26 MultiprocExecutor RPC | 机制分析 | TP=2、PP=2 的 rank 例 → READY → 广播/唯一回复/全员回复 → SHM ring 背压 → Future FIFO → shutdown | 下钻 25 的 Worker fan-out；结尾回到 Engine、Runner、分布式与可观测性 |

24、26 没有套用具体特性的固定章节模板，而是按机制画像回答“状态归谁、何时冻结、消息怎样流动、失败在哪里可见”。25 按范文的语义顺序先给问题和可跟随的数值/请求例，再逐步解释处理、实现、变体、系统接缝、成本与适用边界；没有复制范文标题。

## 实质订正与内容守恒

### 24 扩展插件系统

- 用仓库真实 `DummyAdminEndpointPlugin` 重放 `/v1/admin/scheduler_config`：API client 得到 `cfg-a/cfg-b` 后返回 200，Render 的 `engine_client=None` 返回 503；Phase B 对未经过 Phase A 的 bare `State` 安全 no-op。
- 区分 entry-point 元数据名称与 `EndpointPlugin.name` docstring 的漂移；Endpoint factory 可以隔离，`attach_router` / `init_state` 异常则直接进入服务启动失败。
- 校正 Platform 选择：显式 CPU 快路；两个以上 OOT 报错；唯一 OOT 优先；无 OOT 才检查多个 builtin；选中 factory 在探测后还会再次调用。
- General 的 once 是每进程一次“尝试”，不是全部署成功事务；恢复 CLI `AsyncEngineArgs.add_cli_args` 的首次消费点与跨进程可重入要求。
- IO Processor 是 Pooling serving processor 构造并常驻复用的 online/offline parse、pre-process、post-process 能力，不是 per-request 实例或 Worker 能力。
- Stat Logger 区分 per-engine `(vllm_config, engine_index)` 与 aggregate `(vllm_config, engine_indexes)` 构造；LoRA 记录同名覆盖、serving 构造时快照、单 frontend 专名锁、有序 fallback、无效 filesystem 目录和远程 resolver 安全门。
- 保留“无统一 teardown/rollback”及 FastAPI/Starlette 路由匹配属于外部语义的证据边界。

### 25 在线权重更新

- 用同一请求 R、版本 `step-41 → step-42` 与 `W=(1,2) → (1,9)` 贯穿 pause、原位写入、finish、标签发布和 resume；区分 `abort`、`wait`、`keep` 以及 sleep 的资源门语义。
- 把“完成”拆成 update 返回、Worker finish、Executor 全员成功回复和版本标签发布四层；明确标签不是 checksum、epoch 或 request snapshot，也没有参数 rollback。
- 补齐独立的 `WeightTransferEngineFactory` 与 `WeightTransferTrainerFactory`。两侧以同名 backend 和 typed init info 配对，但各自持有接收/发送实例。
- 四类路径分别说明：dense NCCL 的 packed/unpacked tensor broadcast；IPC 的逐参数或 packed buffer handles 与物理 GPU UUID 导入；sparse NCCL 的 NaN 同形展开；sharded RDT 的本地 slice、后台 scatter/按需 quant 与 finish drain。
- 纠正 dense update info 不校验 handle 数；handle 数是 IPC 合同。补 `ipc_handles_pickled` 的 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` 门和 pickle 信任边界。
- 区分 Multiproc 的首失败早退/未 drain 与 Ray 的 `ray.get(refs)` 收集；两者都没有事务 rollback。补强制 KV reset 遇远程传输仍持 block 时可抛 `RuntimeError` 的反向失败路径。
- 保留 target/draft、LoRA、KV/prefix/MM/encoder、DP pause 共识、CUDA Graph storage 与外部 KV store 的系统边界。

### 26 MultiprocExecutor RPC

- 用 TP=2、PP=2、PCP=1 复算 `output_rank=2`，说明所有 rank 都执行，而唯一回复只决定父进程等谁；aggregator 改为全员回复。
- 区分 READY pipe、death pipe、广播队列和每 Worker 响应队列；解释为什么必须先创建全员、并行等 READY、再按固定顺序完成队列握手。
- 明确普通 unique-reply 中非 output rank 的 Python 异常不会直接进入当前 RPC response；all-reply 只在成功路径形成全回复屏障，首个失败会令父进程停止 drain 后续队列。
- 更正共享内存容量：广播输入默认 `10 × 16 MiB`，每 Worker 响应默认 `10 × 24 MiB`；四 Worker 示例的 payload 区约 1120 MiB，尚未计元数据。
- 新增 ring 槽闭环图，区分“大消息占槽后走 socket”和“ring 满后等待最慢 reader”；反序列化失败可以在 `finally` 释放 read flag，消费者代码不在该保证内。
- 收紧 writer 卡死边界：collective deadline 只传给 response `dequeue`，broadcast `enqueue` 没有 timeout；queue shutdown/Worker monitor 不能证明唤醒已阻塞的 `acquire_write`，只有更上层终止或重建父进程可界定该路径。
- 新增 Future A/B/C 图，重放“先等 C 也必须依次 drain A、B、C”，说明无 request ID 时的 FIFO 对账不变量。

重构前后的实质主题、旧有纠正、配置、失败边界和 Related Pages 目的地均逐项对照。审阅过程中发现的遗漏都补回正文，没有把旧内容静默删除，也没有把内容转交给不存在的页面。

## 独立审读

独立审阅者 `/root/review_24_26` 未修改任何文件，按基础 rubric、功能/机制专项画像、源码 spot-check、重写守恒和跨页一致性完成两轮检查。首轮退回 Endpoint ABI 与真实例、Platform 选择、IO 生命周期、Trainer 双侧协议、NCCL/IPC 区分、队列容量、unique/all reply 失败可见性、writer shutdown 证明边界等问题；协调者逐项回源码修正后，审阅者刷新最新文件给出最终结论。

| 页面 | Beat-2 | Hop-walk | Delete-code | 图示触发与重放 | 源码抽查 | 专项画像 | Verdict |
|---|---|---|---|---|---|---|---|
| 24 扩展插件系统 | PASS | PASS | PASS | timing、coupled-planes；PASS | 3/3 | 机制分析 PASS | **PASS** |
| 25 在线权重更新 | PASS | PASS | PASS | transform、layout、timing、coupled-planes；PASS | 3/3 | 功能分析 PASS | **PASS** |
| 26 MultiprocExecutor RPC | PASS | PASS | PASS | layout、timing、coupled-planes；PASS | 3/3 | 机制分析 PASS | **PASS** |

交叉页结论也是 **PASS**：24 的启动扩展面、25 的运行时状态变更、26 的本地 collective 控制通道层层下钻，所有权、完成点和失败语义没有互相冲突。三页 Related Pages 均为 3–7 条，并逐条解释关系。

主要 spot-check 包括：

- 24：`EndpointPlugin` 与 endpoint tests；`resolve_current_platform_cls_qualname`；`OpenAIServingModels.resolve_lora`、resolver registry 与内置 filesystem/HF resolver。
- 25：Facade 端到端顺序测试；`gpu_worker.py` 的 start/update/finish；双 factory 与 NCCL、IPC、sparse、sharded RDT 的 Worker/Trainer 实现。
- 26：`_get_output_rank`、`collective_rpc`、`_execute_worker_rpc`；`MessageQueue` / `ShmRingBuffer` 的读写与 shutdown；`FutureWrapper`、Worker monitor 与终止流程。

## 图与机械验收

三页分别有 3、3、4 幅 Mermaid，共 10 幅。所有图用仓库随附 Mermaid runtime 解析；协调者在本地构建页面中逐图查看，重点复看 Endpoint 200/503 分支、四后端 `W` 数据路径、ring 槽复用闭环与 Future A/B/C drain。将最初过宽而缩成细线的 Endpoint、ring、Future 图改为纵向布局后重新解析、构建和查看；最终图无语法失败、横向溢出或泄漏 wikilink。

最终集成门禁：

| 执行 | 结果 |
|---|---|
| `.venv/bin/python tools/check_links.py --strict` | 452 页；broken、ambiguous、bare_index、stale_section、orphans 均为 0 |
| `.venv/bin/python tools/check_markdown.py --changed --strict` | 10 个改动 Markdown，0 error / 0 warning |
| `.venv/bin/python tools/check_assets.py --changed --strict` | 10 个改动 Markdown，0 error / 0 warning |
| `.venv/bin/python tools/check_math.py --changed --strict` | 10 个改动 Markdown，0 error / 0 warning |
| `node --test tools/mkdocs-site/mermaid-corpus.test.mjs` | 全库 538 个 Mermaid 块解析通过 |
| `.venv/bin/python -m tools.mkdocs_site.cli build --changed` | 构建成功；8 个变更 wiki 路由的链接、锚点、资源、旧路由均为 0 问题；scoped orphans 按工具合同跳过，全库链接检查已覆盖 |
| `git diff --check` | 无空白错误 |

changed-file 数量包含工作区内既有的 21–23、index、changelog 等并行改动，不能当作本轮独占文件数。构建输出的 Material/MkDocs 2.0 上游提示不是页面验证失败；仓库既有 `.venv` 已包含所需依赖，本轮没有安装依赖。

本轮只做冻结源码、测试与文档的静态分析，没有运行模型、GPU、NCCL/CUDA IPC、Ray/NIXL/RDT、多节点或故障注入测试，也不宣称传输性能或恢复行为已做设备验收。
