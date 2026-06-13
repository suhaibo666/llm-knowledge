# 07 · op-plugin 算子接入(NPU)— 目录索引

> torch_npu 的「算子供给侧」:op-plugin 配置与分类、yaml→codegen→dispatcher 的注册链路与生效时机、以及算子能否「入图」的判别。
> 核验基准:torch_npu **v2.7.1.post5**;op-plugin `all_version = v2.1 ~ v2.10`
> 最后更新: 2026-06-13

---

## 这一域回答什么

| 问题 | 看哪篇 |
|------|--------|
| `op_plugin/config/` 每个文件、字段什么意思?怎么看一条配置就分类(原生/自定义、aclop/aclnn、手写/结构化、版本)? | [[op_plugin_config_and_classification_guide]] |
| 上层 `torch.abs` / `torch_npu.npu_xxx` 怎么调用到 op_plugin?注册胶水自动生成吗?何时生效? | [[op_registration_pipeline_analysis]] |
| 一个算子能不能「入图」?卡在 dynamo / inductor+triton / aclgraph 哪一关?怎么验证? | [[npu_operator_graph_eligibility_guide]] |

---

## 页面列表

| 页面 | 类型 | 核心主题 |
|------|------|---------|
| [[op_plugin_config_and_classification_guide]] | Guide | config 五文件字段;official/custom/symint/quant 分组;acl_op(aclop) vs op_api(aclnn);gen_opapi 结构化 vs 手写;四维分类速查表 |
| [[op_registration_pipeline_analysis]] | Analysis | 两段 codegen 串联;生成产物(RegisterNPU.cpp / CustomRegisterSchema.cpp / custom_ops.py);TORCH_LIBRARY 静态初始化「库加载即注册」;编译→加载→运行期时间线;两条完整调用链 |
| [[npu_operator_graph_eligibility_guide]] | Guide | 入图四路线;非 torchair 三关递进(dynamo meta / inductor lowering+fallback / aclgraph aclnn-only);每关判别命令;三关速查表 |

---

## 一图概览:从 yaml 到入图

```mermaid
graph TD
    Y["op_plugin_functions.yaml<br/>(配置与分类)"] -->|两段 codegen| R["dispatcher 注册<br/>(import 时静态初始化生效)"]
    R --> E["eager 可用<br/>op_plugin::xxx → aclnn/aclop"]
    E -->|torch.compile| G["入图判别<br/>dynamo → inductor+triton → aclgraph"]
    style Y fill:#fff3cd
    style G fill:#cfe2ff
```

---

## 关联域

- [[01_dispatcher_and_device/index]] — Dispatcher(注册目标)
- [[04_inductor/npu/index]] — NPU Inductor(第二关 lowering/fallback 的实现)
- [[06_graphs/npu/index]] — NPU Graphs / ACLGraph(第三关 capture 的实现)
- [[07_op_registration/index]] — 算子接入总索引
- [[01_ai_frameworks/index]] — 本域总索引
