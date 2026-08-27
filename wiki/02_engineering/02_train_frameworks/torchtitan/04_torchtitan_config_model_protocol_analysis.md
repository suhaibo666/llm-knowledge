---
title: "TorchTitan 控制面：full configuration、Configurable 与模型协议"
---

# TorchTitan 控制面：full configuration、Configurable 与模型协议

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页论点**：当前 TorchTitan 的配置系统不是“把 TOML 改写成 Python”，而是把一份完整训练作业变成一棵**可构造、可遍历、可在构建前替换节点**的对象图。`Trainer.Config` 负责作业级组合，每个组件的嵌套 `Config` 自己拥有构造权，`ModelSpec` 暂时桥接模型结构与训练回调，`Module`/`ShardingConfig` 再把构造树延伸到参数初始化和前后向布局边界。它用类型和组合能力换掉了平铺配置、全局工厂与不断膨胀的 CLI，但代价是 Python 配置不再是纯声明数据，序列化、第三方扩展和模型/Trainer 解耦仍未完全收口。
>
> 本页只负责“配置如何成为可运行组件图”。数据迭代与恢复语义分别见 [[02_torchtitan_data_pipeline_grain_analysis]]、[[03_torchtitan_checkpoint_state_recovery_analysis]]；并行 mesh 与布局执行见 [[10_torchtitan_parallel_dims_analysis]]、[[16_torchtitan_spmd_types_analysis]]。

---

## 1. Overview：控制面是一棵带 owner 的配置树

旧心智模型通常把训练配置看成一个参数字典：解析 TOML/CLI 后，Trainer 根据枚举或字符串分支创建模型、优化器和数据加载器。这在组件少时直观，但会同时制造三个问题：配置的字段归全局 `JobConfig` 所有，组件拿到的权限远大于自身需要；每加一种实现都要修改中央 factory；模型结构与训练开关只能靠后置的 `update_from_config()` 再同步。

TorchTitan 当前选择了一条对象图路线。2026-02-18 的提交 `9810191bb` 在提交说明中把旧系统的缺陷概括为：TOML/自定义 `JobConfig` 缺乏类型安全，配置空间平铺，所有组件接收整份配置，实验扩展依赖 `custom_config_module` 与 `_merge_configs`，而 `ModelArgs` 又脱离 `TrainSpec`。当前源码把这些职责拆成四层：

| 层 | 当前所有者 | 解决的问题 | 不负责什么 |
|---|---|---|---|
| 作业选择 | `ConfigManager` | `--module/--config` 定位完整配置函数，CLI 只覆盖其字段 | 不知道组件怎样构造 |
| 作业组合 | `Trainer.Config` | 聚合 tokenizer、data、optimizer、checkpoint、loss、model 等配置 | 不直接实例化组件 |
| 节点构造 | `Configurable.Config` | `Config` 与 owner class 绑定，`build()` 生成该节点 | 不决定全局训练顺序 |
| 模型/布局协议 | `ModelSpec`、`BaseModel`、`Module`、`ShardingConfig` | 连接模型 config、回调、输入边界、状态初始化与布局 | 不替代 mesh/rank 预算 |

一条最小调用链是：

```text
--module llama3 --config llama3_debugmodel
                 |
                 v
ConfigManager._load_config() -> config_registry.llama3_debugmodel()
                 |                         |
                 |                         v
                 |                  完整 Trainer.Config 树
                 v
          tyro CLI override -> config.build() -> Trainer(config)
                                            |
                                            +-> update model config
                                            +-> apply node overrides
                                            +-> model_config.build()
                                            +-> component_config.build(runtime objects...)
```

`train.py` 只解析配置并调用 `config.build()`，再选择 seed checkpoint 或 `trainer.train()`（`torchtitan/train.py:28`、`torchtitan/train.py:43`、`torchtitan/train.py:46`）。Llama3 的 debug config 返回的不是零散 patch，而是一份包含 loss、optimizer、training、dataloader、parallelism、checkpoint 和 validator 的完整 `Trainer.Config`（`torchtitan/models/llama3/config_registry.py:38`、`torchtitan/models/llama3/config_registry.py:41`、`torchtitan/models/llama3/config_registry.py:61`）。

---

## 2. Full configuration：为什么选择完整 Python 作业，而不是 TOML + 大 CLI

### ① 背景/问题

平铺 TOML 的明显优势是可读、可 diff、易被外部系统生成；但组件图开始包含“选择哪个具体 Config class”“根据模型词表生成 loss config”“构造每层不同的 module config”等动作后，纯数据文件要么引入一套自定义表达式语言，要么把真正的组合逻辑重新塞回中央 registry。旧式大 CLI 还有另一种扩散：每个新字段都要决定是否公开为 flag，CLI 表面逐渐等同于整个内部对象模型。

### ② 为什么这么设计

**选中的路线**是让 config registry 函数直接返回完整、已选定具体类型的 `Trainer.Config`；**明显替代方案**是保留 TOML 作为主配置，再用 CLI/后处理 patch 它。决定性标准是：作业配置需要表达带类型的组件组合和普通 Python 计算，而不只是标量覆盖。`ConfigManager` 因此只负责两级选择：`--module` 找模块，`--config` 找可调用的 registry 函数；两者都必填（`torchtitan/config/manager.py:48`、`torchtitan/config/manager.py:85`、`torchtitan/config/manager.py:89`）。

短模块名按 models、experiments、RL examples 三个命名空间查找，第三方也可传完整模块路径（`torchtitan/config/manager.py:104`、`torchtitan/config/manager.py:125`）。加载函数返回对象后，tyro 以该对象的具体 class 为 schema，并让 CLI 覆盖 registry 默认值（`torchtitan/config/manager.py:34`、`torchtitan/config/manager.py:40`）。所以优先级是“CLI > 完整 config function”，不是把多个半成品 config 深度合并。

### ③ 实现思路与细节

`Trainer.Config` 是作业的组合根。它聚合 profiler、metrics、tokenizer、dataloader、optimizer、scheduler、parallelism、checkpoint、AC、compile、validator、override 和 loss；每一项尽量由对应组件自己的 Config class 表达（`torchtitan/trainer.py:65`、`torchtitan/trainer.py:86`、`torchtitan/trainer.py:94`、`torchtitan/trainer.py:100`、`torchtitan/trainer.py:102`、`torchtitan/trainer.py:110`）。`model_spec` 被 tyro 抑制，只允许 registry 以 Python 对象设置，因为它含 callable 和 state-dict adapter，而不是稳定的 CLI 数据（`torchtitan/trainer.py:72`、`torchtitan/trainer.py:74`）。

当前仓库甚至用测试冻结 CLI 字段集合：测试文件的目的就是“Ensure no more flags are added to CLI”，并显式维护 `_FROZEN_CLI_OPTIONS`（`tests/unit_tests/cpu/test_no_new_cli_options.py:7`、`tests/unit_tests/cpu/test_no_new_cli_options.py:18`）。这证明“完整 config 为主、CLI 不继续扩张”是当前维护约束，不只是文档建议。旧 `_merge_configs()` 仍在，但运行时发出弃用警告并要求改用 Config subclass + registry（`torchtitan/config/manager.py:161`、`torchtitan/config/manager.py:166`）。

### ④ 约束/代价/失败边界

- Python config 能执行任意代码，也因此比 TOML 更难做静态审计和跨语言生成；提交 `9810191bb` 明确承认其声明性和直接可读性较差，贡献者学习成本更高。
- CLI 目前仍能覆盖大量既有嵌套字段；“冻结”不是“已删除”。不能把 2026-02-25 的 RFC 提交 `c82675eb6` 中“移除大部分 CLI”的计划写成当前事实。
- `--module` 的导入会执行 Python 模块顶层代码；这是一种受信任代码扩展机制，不是无副作用的数据加载器。
- registry 函数必须返回与 tyro 可处理 schema 相符的对象。函数不存在或不可调用会在进入分布式初始化前失败（`torchtitan/config/manager.py:143`、`torchtitan/config/manager.py:153`）。

### ⑤ 发展趋势（有源码锚点的推断）

RFC `c82675eb6` 与当前冻结 CLI 测试共同指向“更少临时覆盖、更多完整作业配置”，但 HEAD 仍保留全部冻结字段。因此只能推断新功能更倾向先落入 registry/config subclass，不能声称 CLI 已进入确定的删除时间表。

---

## 3. Configurable owner：为什么组件自己拥有构造权

### ① 背景/问题

当 tokenizer、dataloader、optimizer、loss 和模型都有多个实现时，中央 factory 常写成 `if backend == ...`。它把所有构造签名和依赖聚到一处：新组件必须修改中央枚举；组件需要 runtime 对象时又容易把 device、mesh、tokenizer 混进可序列化配置。另一个直觉方案是把 class 直接存在每个 config 字段里，但这仍不能约束 Config 与 class 的构造协议。

### ② 为什么这么设计

**选中的路线**是每个 `Configurable` 定义嵌套 `Config`，由 `__init_subclass__` 自动写入 owner，统一通过 `config.build()` 构造；**替代方案**是中央 factory/后端枚举。决定性标准是让“配置 schema、默认值和构造目标”一起归组件所有，同时仍允许 Trainer 注入不应进入配置的 runtime 对象。源码把契约写成：外层类继承 `Configurable`，嵌套 Config 使用 `@dataclass(kw_only=True, slots=True)`，构造函数接受自身 config（`torchtitan/config/configurable.py:18`、`torchtitan/config/configurable.py:21`）。

### ③ 实现思路与细节

`__init_subclass__` 检查 `slots` 和所有 init 字段的 keyword-only 属性，然后把 `config_cls._owner` 设为外层 class（`torchtitan/config/configurable.py:162`、`torchtitan/config/configurable.py:167`、`torchtitan/config/configurable.py:173`、`torchtitan/config/configurable.py:179`）。`build()` 不直接传原 config，而用 `dataclasses.replace(self)` 复制后交给 owner，减少实例侧修改污染 registry 原对象的风险（`torchtitan/config/configurable.py:134`、`torchtitan/config/configurable.py:149`）。

构造有两种模式：纯配置节点调用 `owner(config=copy)`；依赖运行时状态的节点额外接受 kwargs。Trainer 构建 tokenizer 时注入资产路径，构建 dataloader 时再注入 DP rank/size、tokenizer、上下文长度和 token budget（`torchtitan/trainer.py:551`、`torchtitan/trainer.py:555`、`torchtitan/trainer.py:556`）。如果 runtime kwargs 与 config 字段同名，`build()` 立即拒绝，迫使调用方明确某个值究竟属于可复现配置还是本次运行环境（`torchtitan/config/configurable.py:153`、`torchtitan/config/configurable.py:155`）。

### ④ 约束/代价/失败边界

- `Config` 没有 owner 时 `build()` 抛 `NotImplementedError`；抽象 base config 因而不能被误当成可运行实现（`torchtitan/config/configurable.py:144`）。
- `slots=True` 与 keyword-only 是硬约束，不是风格建议；它们减少意外字段和位置参数漂移，但使某些 dataclass 继承/反射写法更繁琐。
- owner 绑定发生在 class 定义时；动态换实现应换掉 Config 节点，而不是在已构造实例上改 `_owner`。
- `build()` 复制的是 dataclass 的浅层结构；嵌套可变对象仍应由配置作者避免共享突变。源码没有承诺任意深拷贝。

---

## 4. 树遍历与 override：为什么替换发生在 build 之前

### ① 背景/问题

完整配置解决了作业组合，却没有自动解决实验注入：用户可能只想把第 10–19 层的 MoE 或某个 RoPE 换成第三方实现。修改每个 registry 函数会复制大量配置；在模型实例构建后 monkey-patch 又太晚，因为 meta init、参数初始化和 sharding contract 已经绑定旧 module。

### ② 为什么这么设计

**选中的路线**是遍历 Config 树，以 config class + FQN glob 定位节点，并在任何组件 `build()` 之前替换；**替代方案**是修改 registry 或替换已构造 `nn.Module`。决定性标准是替换必须同时改变构造目标、参数 schema 和后续 sharding 行为，而不是只改 forward。override 模块的顶层契约明确覆盖 model、optimizer、loss、dataloader 等全部 `Configurable`，并强调“config construction 后、任何 build 前”应用（`torchtitan/config/override.py:8`、`torchtitan/config/override.py:14`、`torchtitan/config/override.py:17`）。

### ③ 实现思路与细节

`Config.traverse()` 产生 `(fqn, config, parent, field_name)`，FQN 与将来 module tree 对齐；它会递归 dataclass 字段和 list，并保留 parent/索引以原位替换（`torchtitan/config/configurable.py:75`、`torchtitan/config/configurable.py:80`、`torchtitan/config/configurable.py:113`、`torchtitan/config/configurable.py:121`）。默认命中目标 class 后停止下钻，`recurse=True` 才继续遍历命中子树（`torchtitan/config/configurable.py:95`）。

override 用 import path 精确激活 factory，可选 FQN glob 和 `exact=True`；冲突按具体节点而不是按 class 判断，因此两个替换只要作用于不相交 FQN 就可共存（`torchtitan/config/override.py:14`、`torchtitan/config/override.py:19`、`torchtitan/config/override.py:25`）。Trainer 先让模型 config 吸收训练作业设置，再对整棵 config tree 应用 override，最后才在 meta device 上 build 模型（`torchtitan/trainer.py:334`、`torchtitan/trainer.py:342`、`torchtitan/trainer.py:347`、`torchtitan/trainer.py:353`）。

`ModelSpec` 不是 `Configurable.Config`，因此它自己实现 `traverse()`，只把内部 model config 暴露给树遍历，刻意不暴露 parallelize/pipelining 等 callable 字段（`torchtitan/protocols/model_spec.py:49`、`torchtitan/protocols/model_spec.py:52`、`torchtitan/protocols/model_spec.py:57`）。这是 model config 能被全树 override 命中的桥，而不是通用对象反射。

### ④ 约束/代价/失败边界

- FQN 依赖 config tree 与 module tree 的结构约定；列表重排会改变 `layers.10...` 的选择结果。针对层号的 override 应被视为结构耦合。
- 默认匹配 target 的子类；只适用于精确实现的替换必须用 `exact=True`，否则会吞掉更具体的配置（`torchtitan/config/override.py:188`、`torchtitan/config/override.py:197`）。
- override factory 返回的是 Config，不是任意对象；装饰器显式拒绝 `ModelSpec` 或普通 class 作为 target（`torchtitan/config/override.py:220`、`torchtitan/config/override.py:223`）。
- model override 排在 `update_from_config()` 之后，因为旧节点先被写入 sharding config；这也暴露了当前封装缺口：如果 replacement 没有正确继承/迁移这些字段，替换仍可能在后续 parallelize 时失败。

---

## 5. ModelSpec 与 Module：为什么“模型配置”还不是完整模型协议

### ① 背景/问题

一个模型不仅有层数和维度，还要提供 parallelize、pipeline 切分、checkpoint 适配、输入预处理、初始化和 sharding 边界。把这些都塞进 Trainer 会让每个模型新增分支；只把一个 `nn.Module` class 注册进去，又无法携带模型专属回调和 config tree。

### ② 为什么这么设计

**当前选中的路线**是用 `ModelSpec` 捆绑“已经选定的模型 Config + 少量模型级 callable/adapter”，再要求模型树遵守 `BaseModel/Module` 协议；**明显替代方案**是一个全局 TrainSpec/ModelArgs 或 Trainer 内的模型分支。决定性标准是模型结构可以独立构造和遍历，而 Trainer 只依赖稳定协议。`ModelSpec` 当前包含 name、flavor、model config、parallelize、pipelining、post-optimizer hook 和 state-dict adapter（`torchtitan/protocols/model_spec.py:31`、`torchtitan/protocols/model_spec.py:33`、`torchtitan/protocols/model_spec.py:36`、`torchtitan/protocols/model_spec.py:44`）。

### ③ 实现思路与细节

`BaseModel` 本身是 `Module = nn.Module + Configurable`，模型由其 Config tree 构建（`torchtitan/protocols/model.py:36`、`torchtitan/protocols/model.py:39`）。`Module.Config.build()` 还把 param initializer 和 sharding config 写回实例；之所以在 Config 层完成，是因为 Module 不定义统一 `__init__`，以兼容 `class Foo(nn.SomeModule, Module)` 的菱形继承（`torchtitan/protocols/module.py:58`、`torchtitan/protocols/module.py:63`、`tests/unit_tests/cpu/test_module.py:103`、`tests/unit_tests/cpu/test_module.py:110`）。

Trainer 在 meta device 上构建后，立即遍历所有 submodule，拒绝不满足 Module 协议的节点（`torchtitan/trainer.py:353`、`torchtitan/trainer.py:359`、`torchtitan/protocols/model.py:81`、`torchtitan/protocols/model.py:91`）。状态初始化则递归 Module child；带参数却既没有 `param_init` 又没有 `reset_parameters()` 的节点会明确失败，而不是留下未初始化内存（`torchtitan/protocols/module.py:74`、`torchtitan/protocols/module.py:89`、`torchtitan/protocols/module.py:156`、`torchtitan/protocols/module.py:180`）。

### ④ 约束/代价/失败边界

- `ModelSpec` 是过渡接口：源码已有 TODO 要弃用它并把字段移动到 model/trainer config（`torchtitan/protocols/model_spec.py:31`）。所以第三方不应把它当永久稳定 ABI。
- callable 让配置无法完整 round-trip。`Trainer.Config.to_dict()` 对 `model_spec` 只保存 name、flavor 和 model config，明确跳过 callable（`torchtitan/trainer.py:198`、`torchtitan/trainer.py:201`、`torchtitan/trainer.py:203`）。导出的 JSON 是观测快照，不是保证可重建的作业定义。
- Config 树与 Module 树的 FQN 对齐是一项协议，不由类型系统完全证明；动态创建未在 config 中出现的子模块会削弱 override 与 sharding 的可追踪性。
- 普通 `nn.Module` 不能混入标准模型树；确需例外的模型必须重写 verify 行为，否则 Trainer 会在并行化前失败（`torchtitan/protocols/model.py:88`、`torchtitan/protocols/model.py:95`）。

### ⑤ 发展趋势（有源码锚点的推断）

`ModelSpec` 的弃用 TODO 与 `BaseModel.Config.update_from_config()` 的“violates encapsulation” TODO 同时存在（`torchtitan/protocols/model.py:108`）。据此可以推断职责还会继续向 model config / 外部 config pass 收敛；但 HEAD 仍先执行 `update_from_config()`，不能把它描述为已由纯构造树取代。

---

## 6. 输入与 sharding boundary：为什么构造成功仍不代表可训练

### ① 背景/问题

配置树只能保证对象能被创建，不能保证 dataloader batch 与模型 forward 对得上，也不能保证参数和激活在 TP/CP/EP mesh 上拥有合法布局。一个“只实现 forward 的模型”可能单卡运行，却在标准 Trainer、validator 或组合并行下失效。

### ② 为什么这么设计

**选中的路线**是把 batch 适配放在模型的 `preprocess_inputs()`，把状态和激活的布局契约放在每个 Module 的 `ShardingConfig`；**替代方案**是 Trainer 按模型名分支预处理，或由 parallelize 函数临时 patch 每个 forward。决定性标准是输入语义和模块边界都属于模型结构，必须与 config/module tree 同位置演进。

### ③ 实现思路与细节

标准 Trainer 把 labels 合入 batch 后调用第一个 model part 的 `preprocess_inputs()`，传入 `ParallelDims` 与 parallelism config；返回 `(inputs, labels, extra_kwargs)` 再进入统一 forward/backward（`torchtitan/trainer.py:688`、`torchtitan/trainer.py:690`、`torchtitan/trainer.py:693`、`torchtitan/trainer.py:701`）。BaseModel 的默认实现直接抛错，因为 mask、CP shard 和标签拆分都不能有通用默认值（`torchtitan/protocols/model.py:57`、`torchtitan/protocols/model.py:66`、`torchtitan/protocols/model.py:72`、`torchtitan/protocols/model.py:77`）。

`ShardingConfig` 分开描述 state、input source/destination、output source/destination 和可选 local-map boundary；source/destination 成对表达“进入边界时是什么布局、需要变成什么布局”（`torchtitan/protocols/sharding.py:59`、`torchtitan/protocols/sharding.py:70`、`torchtitan/protocols/sharding.py:108`）。`Module.parallelize()` 递归子树，先分布状态，再把 forward 包成“输入 redistribution → 可选 local region → 原 forward → 输出 redistribution”（`torchtitan/protocols/module.py:244`、`torchtitan/protocols/module.py:266`、`torchtitan/protocols/module.py:279`、`torchtitan/protocols/module.py:285`）。

### ④ 约束/代价/失败边界

- `preprocess_inputs()` 只属于标准 trainer/validator 路径；Flux 等 bespoke pipeline 可以不调用它，源码明确保留这一例外（`torchtitan/protocols/model.py:72`、`torchtitan/protocols/model.py:74`）。
- sharding layout 必须为实际 mesh 的每个轴声明 placement；缺轴直接抛 `ValueError`，多声明的轴才会被忽略（`torchtitan/protocols/sharding.py:120`、`torchtitan/protocols/sharding.py:126`、`torchtitan/protocols/sharding.py:148`）。
- size-1 轴上的 Shard/Partial 当前归一成 Replicate，以避开 DTensor 严格 placement 规则；源码 TODO 表明这是一项过渡兼容，计划待 FlexShard 替代 fully_shard 后清理（`torchtitan/protocols/sharding.py:130`、`torchtitan/protocols/sharding.py:137`）。
- Module 禁止重复 parallelize；第二次调用会失败，防止 forward wrapper 和 state distribution 被安装两遍（`torchtitan/protocols/module.py:259`、`torchtitan/protocols/module.py:260`）。

---

## 7. 版本纠偏与接入清单

| 旧心智模型 | 当前事实 | 证据/原因 |
|---|---|---|
| TOML 是主配置，Python 只做扩展 | registry 函数返回完整 `Trainer.Config`，CLI 覆盖其默认值 | `torchtitan/config/manager.py:23`、`torchtitan/config/manager.py:158` |
| 所有实现由中央 factory 构造 | 每个 Config 自动绑定 owner，并由 `build()` 构造 | `torchtitan/config/configurable.py:134`、`torchtitan/config/configurable.py:179` |
| `_merge_configs()` 是推荐扩展点 | 已弃用，推荐 Config subclass + registry | `torchtitan/config/manager.py:161`、`torchtitan/config/manager.py:166` |
| model config 只是超参数字典 | 它是可构造、可遍历的 Module config tree，并携带 init/sharding 节点 | `torchtitan/protocols/model.py:101`、`torchtitan/protocols/module.py:58` |
| 替换 kernel 应 patch 已构建模型 | override 在 build 前按 Config class/FQN 换节点 | `torchtitan/config/override.py:14`、`torchtitan/trainer.py:342` |
| 保存的 config JSON 可完整复现作业 | callable 被省略或转 repr，ModelSpec 只序列化结构字段 | `torchtitan/config/configurable.py:60`、`torchtitan/trainer.py:203` |
| 实现 `forward()` 就能接标准 Trainer | 还要满足 Module tree、`preprocess_inputs()`、init 与 sharding contract | `torchtitan/protocols/model.py:57`、`torchtitan/protocols/model.py:81` |

接入一个新模型/组件时，应按以下顺序验证：

1. 组件定义 `@dataclass(kw_only=True, slots=True)` 的嵌套 Config，runtime 对象不要伪装成配置字段。
2. 模型结构用 Config tree 表达；每层/每模块 Config FQN 应与最终 Module FQN 可解释地对应。
3. registry 返回完整作业，而不是依赖大量启动时 patch；模型对象由 `model_spec` 程序化设置。
4. 若使用标准训练路径，实现 `preprocess_inputs()` 并明确 labels、mask、CP/SPMD 注解的边界。
5. 所有标准子模块遵守 Module 协议，参数拥有 `param_init` 或合法 `reset_parameters()`。
6. 为 state/input/output 写完整 sharding contract，并在目标 mesh 上验证所有轴 placement。
7. 第三方实现优先以 build 前 override 接入；用 class + FQN 限定作用域，避免实例 monkey-patch。
8. 把 `to_dict()` 结果视为日志与排障材料；真正可复现定义仍是 Python config 代码、版本基线和 CLI 覆盖三者的组合。

> [!important] 证据边界
> “带 owner 的配置树”“对象图控制面”是本页对 `Trainer.Config`、`Configurable.Config`、traverse/build 链路的知识库抽象，不是上游公开术语。源码直接保证的是 owner wiring、构建顺序、树遍历和协议检查；“它们共同形成控制面”是据此作出的机制归纳。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 配置、训练循环、并行和实验子系统的总入口。
- [[01_torchtitan_trainer_quickstart]] —— 从启动命令进入 Trainer 生命周期，适合先建立全局调用顺序。
- [[02_torchtitan_data_pipeline_grain_analysis]] —— dataloader Config 接收 runtime token budget 后如何生成可恢复 iterator graph。
- [[03_torchtitan_checkpoint_state_recovery_analysis]] —— Configurable checkpointer 怎样把组件状态图映射为恢复与导出语义。
- [[10_torchtitan_parallel_dims_analysis]] —— parallelism config 怎样变成多张 storage/value mesh。
- [[16_torchtitan_spmd_types_analysis]] —— ShardingConfig 的逻辑轴如何在 Module boundary 执行为 DTensor redistribution。
- [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] —— GraphTrainer 如何消费同一模型协议并暴露当前兼容边界。
