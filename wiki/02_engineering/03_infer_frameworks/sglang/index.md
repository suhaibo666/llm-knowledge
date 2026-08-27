---
title: "SGLang — 目录索引"
---

# SGLang — 目录索引

> SGLang 推理框架源码级分析。首篇聚焦其 torch.compile / 编译 pass 体系。
> 最后更新: 2026-07-20

---

## deep dive

| 页面 | 核心主题 |
|------|---------|
| [[sglang_compilation_passes_analysis]] | **编译 Pass 与 torch.compile 适配**:`srt/compilation/` 是 vLLM piecewise-cudagraph 管线的近逐文件 fork,但**融合 pass 被整个抽空**——唯一的 `FixFunctionalizationPass` 是 no-op,真实图重写 pass 数=0;含两条 compile 路径、split_ops 切图、CUDA/NPU/XPU piecewise backend 差异、与 vLLM 血缘对照 |

---

## 关联域

- [[../vllm/index]] — vLLM(SGLang compile 层的血缘来源)
- [[../index]] — 推理框架总索引
- [[22_pattern_expression_and_matcher_engine_analysis]] — 上游 Inductor PatternMatcher 引擎基座
- [[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]] — pass 开发方法论(含跨框架对照)
