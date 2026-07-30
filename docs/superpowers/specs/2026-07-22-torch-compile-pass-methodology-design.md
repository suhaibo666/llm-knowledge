# Torch Compile Pass 方法论与阶段指南重构设计

> **设计日期**：2026-07-22
> **知识库基线**：`llm-knowledge@2cfdeebf2a64e3081884106c8a7c2f85ae45e659`
> **PyTorch 源码核验基线**：`pytorch@9922478dffa`（main，2026-07-20 已在本地 checkout 可读取）

## 1. 目标

本次重构同时解决两个问题：

1. 建立完整的 Torch Compile 优化选层方法论，逐阶段回答“是什么、为什么、适合做什么、为什么不放在相邻阶段”。
2. 校验并补强现有阶段指南，纠正错误示例，说明真实主要 Pass、设计动机、执行顺序、关键 API、注册方式和工程注意事项。

最终读者应能先依据方法论选择优化层次，再进入对应阶段指南，完成一个与固定 PyTorch 源码基线一致的最小 Pass 或扩展接入。

## 2. 范围

完整阶段范围为：

1. Dynamo / 原始 FX 图捕获与 Backend 交接
2. Pre-Grad FX Pass
3. Joint Graph Pass
4. Post-Grad FX Pass
5. Decomposition
6. Lowering
7. Scheduler / Fusion
8. Codegen / Kernel / Wrapper

本次不重写各页面已有的全部 Pass 逐项解释；保留已有有效内容，重点校验主干、纠正错误、补齐缺失的方法论和开发入口。NPU、vLLM、SGLang 页面作为对照证据和交叉引用，不在本次逐页重写范围内。

## 3. 文档架构

采用“一个总方法论页 + 多个阶段实战页”的结构。

### 3.1 总方法论页

重构：

- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/fx_pass_optimization_methodology.md`

该页从“FX Pass 方法论”提升为“完整 Torch Compile 优化选层方法论”。每个阶段统一回答：

1. **是什么**：所在位置、输入输出 IR、可见范围。
2. **为什么**：该阶段存在的核心矛盾或瓶颈。
3. **信息边界**：保留了什么、已经丢失什么、建立了哪些不变量。
4. **适合做什么**：典型优化类型和代表案例。
5. **不适合做什么**：风险、缺少的信息或过早/过晚的代价。
6. **为什么不放相邻阶段**：明确与前后阶段的取舍。

总页提供跨阶段决策树和对照矩阵，不重复阶段页的大段 API 说明。

### 3.2 已有阶段指南

校验并补强：

- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/pre_grad_passes_guide.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/joint_graph_passes_guide.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/post_grad_passes_guide.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/lowering_analysis.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/scheduler_analysis.md`

每页新增或重构统一的“阶段方法论与开发指南”部分：

1. 阶段定位与选择理由。
2. 当前源码真实执行的主要 Pass 或扩展机制。
3. 每个主干 Pass 的作用、为什么放在本阶段、关键门控和顺序依赖。
4. 关键 API 速查：签名、作用、输入输出、稳定性等级。
5. 内建注册路径与下游扩展钩子的区别。
6. 一个与源码一致的最小注册/接入示例。
7. 正确性、别名与 Mutation、FakeTensor、动态形状、缓存失效、可观测性和性能验证清单。

现有错误不静默覆盖：按仓库规则用 `> [!deprecated]` 或 `> [!contradiction]` 标出旧写法，再给出核验后的正确写法和源码定位。

### 3.3 新增缺失阶段指南

新增：

- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/01_dynamo/dynamo_pass_methodology.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/decomposition_passes_guide.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/codegen_extension_guide.md`

三个新页面分别解决：

- **Dynamo**：说明 Dynamo 的主要职责是捕获、分图、Guard 和 Backend 交接；给出自定义 Backend 接收并变换 FX Graph 的最小例子，同时解释设备特定融合通常不应放在此层。
- **Decomposition**：说明“保持复合算子完整”与“拆成可融合基础算子”的核心取舍；介绍 decomposition table、注册 API、条件化分解和最小示例。
- **Codegen**：明确该阶段通常不是注册 FX 图 Pass，而是兑现 Scheduler 的融合决策；介绍 `BaseScheduling`、`codegen_node`、`codegen_template`、Backend 注册、Kernel Template 与 Wrapper 扩展，并给出最小扩展骨架。

`inductor_codegen_analysis.md` 保留为机制分析页，只补充指向开发指南的导航，避免复制整套内容。

## 4. API 说明标准

每个阶段的 API 表至少包含：

| 字段 | 要求 |
|---|---|
| API | 当前基线中的真实符号名 |
| 作用 | 它解决的具体问题，不只复述函数名 |
| 输入/输出 | 关键对象类型与是否原地修改 |
| 使用位置 | 内建注册、配置钩子、Backend 扩展或测试用途 |
| 稳定性 | 公开 API、半公开扩展点或 `torch._inductor` 内部原型接口 |
| 注意事项 | 顺序、缓存、动态形状、Mutation、元数据等约束 |

所有最小示例必须区分：

- **可直接使用的下游扩展示例**：通过 config hook、自定义 Backend 或正式注册入口接入。
- **修改 PyTorch 上游源码的内建示例**：注册到内部 `pass_dict`、lowering 表或 Backend 实现；明确其不是稳定公开 API。

## 5. 已知错误修订清单

至少修订以下已确认问题，并在实施时继续扫描同类错误：

1. Pre-Grad 指南使用 `joint_custom_pre_pass` 观察 Pre-Grad 图；正确入口应为 `pre_grad_custom_pass`。
2. Pre-Grad 示例使用 `PatternMatcherPass` 却未导入。
3. Pre-Grad 页面把 IR 同时写成 functional 与未 functionalized，需统一为“未函数化、未规范化”。
4. Joint 指南把 `pass_patterns` 从错误模块导入，并以无关的 custom hook 作为“确保加载”手段。
5. Post-Grad 指南的 Pattern 注册导入路径、OpOverload target 和自定义 Graph Pass 输入类型需对照源码核验。
6. Scheduler 页面同时出现 `_pre_fusion_custom_pass(list[BaseSchedulerNode])` 与不存在的 `pre_fusion_custom_pass(GraphLowering)` 两套冲突写法；以固定基线源码为准保留前者，标明 Scheduler IR 是原型内部接口。
7. 方法论页把“SymInt 一律跳过”泛化到所有实现；改为按阶段和具体 Pass 区分“加符号 Guard”与“保守跳过”。

## 6. 索引与知识图谱集成

更新：

- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/01_dynamo/index.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/index.md`
- `wiki/02_engineering/01_ai_frameworks/02_compile_stack/04_inductor/inductor_codegen_analysis.md`
- `wiki/changelog.md`

每个新增页面必须包含 `## Related Pages`，并与总方法论页、流水线总览和相邻阶段页面建立双向 Obsidian 链接。

## 7. 源码忠实性与版本策略

1. 所有新增非平凡断言先定位并读取 `pytorch@9922478dffa` 对应源码。
2. 引用格式统一为仓库相对源码路径与行号，例如 `torch/_inductor/fx_passes/pre_grad.py:287`。
3. 页面头部记录源码 baseline 和本次更新日期。
4. 发现旧页面与固定基线冲突时，以源码为准并显式标注修订，不把推断写成源码事实。
5. 对内部 API 明确提示版本漂移风险，尤其是 `torch._inductor`、Scheduler IR 和 PatternMatcher 内部表。

## 8. 验证标准

完成条件为：

1. 八阶段均在总方法论页拥有“是什么 / 为什么 / 适合 / 不适合 / 相邻阶段取舍”。
2. Pre、Joint、Post、Decomposition、Lowering、Scheduler、Dynamo、Codegen 各自拥有关键 API 表和接入示例。
3. 已知错误修订清单逐项关闭。
4. 新增及修改页面的 `[[wiki links]]` 无悬空目标。
5. 所有新增 Mermaid 图按仓库规则人工复核，并在可用时实际渲染。
6. Markdown 通过 `git diff --check`，无尾随空白和冲突标记。
7. 所有新增代码块进行语法级检查；内部伪代码必须显式标记，不能伪装成可直接运行示例。
8. `git diff --cached --name-only` 只包含本次设计和知识库修改，不包含用户已有的 `cuda_nonmatmul_kernels_analysis.md` 修改。
9. 最终 commit 成功并推送到 `origin/main`。

## 9. 提交策略

设计说明单独提交。实施完成后使用一个聚合文档提交，提交信息聚焦 Torch Compile Pass 方法论与阶段指南；推送前再次确认远端跟踪分支与暂存文件范围。用户已有的无关工作区修改始终保留且不暂存。
