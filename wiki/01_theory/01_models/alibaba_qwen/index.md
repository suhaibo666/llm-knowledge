---
title: "Qwen / 阿里巴巴技术路线总览"
---

# Qwen / 阿里巴巴技术路线总览

> Qwen 模型家族的报告分析与开放权重审计入口。当前优先覆盖 2026-08 首次开放 Max 级权重的 Qwen3.8，以及同系列的 27B 紧凑档与 Qwen4 架构预览 Flash-Next。
> 最后更新：2026-08-27

---

## 分析页

| 段位 | 页面 | 来源基线 | 核心主题 | 状态 |
|---|---|---|---|---|
| 1 | [[10_qwen3_8_analysis]] | 官方博客 2026-08-03 + `Qwen3.8-2.4T-A95B@207bd685` | 2.4T/95B、3:1 Gated DeltaNet/Attention、512 选 10+1 MoE、真实工作 RL、Max endpoint 与开放 checkpoint 边界、自定义许可证 | ✅ 已摄入 |
| 1 | [[11_qwen3_8_27b_analysis]] | 模型卡 + `Qwen3.8-27B@1d4bf0f2` | 27.8B **稠密** VL、保留 3:1 混合注意力、1M 靠 YaRN 外推、思考控制三件套、基准口径折扣 | ✅ 已摄入 |
| 1 | [[12_qwen3_8_flash_next_analysis]] | 模型卡 + `Qwen3.8-Flash-Next@f5d08274` | **Qwen4 架构预览**：QSA 微块稀疏、Gated Residual、20M 条 n-gram 嵌入（51B）、6B 激活越级 | ✅ 已摄入 |
| 2 | [[20_qwen3_8_flash_next_architecture_deepdive]] | 技术报告 §2（2026-08-26，28 页 PDF） | GDN 门控 delta 规则、QSA 两阶段 CPT 训练与 IndexShare 对比、GR 五条消融与跨层路径分析、n-gram 放置与词表缩放 | ✅ 已摄入 |
| 2 | [[21_qwen3_8_flash_next_optimization_deepdive]] | 技术报告 §3–§4 | Muon 权重归属与拆分融合参数、Canzona、超参缩放律与取消 batch-size warmup、稳定性压力测试、三方基座评测 | ✅ 已摄入 |

## 模型状态

| 模型 | 开放状态 | 一句话定位 |
|---|---|---|
| `Qwen3.8-2.4T-A95B` | BF16 + FP8 权重已开放 | text-only、强制 thinking、原生 262K；Max 级 backbone 的可部署 checkpoint |
| `Qwen3.8-Max` | 托管 API | 基于开放 checkpoint，另含视觉输入、non-thinking、默认 1M 上下文与官方工具 |
| `Qwen3.8-27B` | BF16 权重已开放 | 27.8B 稠密、原生 VL、thinking 可关；`qwen3_5` 谱系的最小可部署切片 |
| `Qwen3.8-Flash-Next` | BF16 权重已开放 | 180B 落盘（125B 主干 + 51B n-gram + 4B MTP）、6B 激活；`qwen4_exp` 谱系 |

## 原始来源

- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Max_blog_2026-08-03.txt` — 官方发布博客正文快照。
- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_2_4T_A95B_model_card_207bd685.md` — 固定 revision 模型卡。
- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Max_LICENSE_207bd685.txt` — Qwen3.8-Max 自定义许可证。
- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_27B_model_card_1d4bf0f2.md` + `Qwen3_8_27B_config_1d4bf0f2.json`。
- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Flash_Next_model_card_f5d08274.md` + `Qwen3_8_Flash_Next_config_f5d08274.json`。
- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Flash_Next_tech_report.md` — 技术报告的**来源索引页**（链接 + 元数据 + 章节定位表 + 关键外部引用核实）。按库约定 `raw/` 不分发第三方论文原文。
- 同目录还保存 Qwen、Qwen2、Qwen2.5、Qwen-VL/Audio 与 Qwen3-Omni 的历史 PDF，尚未逐篇摄入。

## 知识缺口

- ~~Qwen3.8-Flash-Next 的技术报告尚未摄入~~ —— **已于 2026-08-27 摄入**，见上表两篇深挖页。报告自身遗留的问题（3:1 比例仍无消融、新缩放律函数形式未公开、80× n-gram 词表的选定依据未说明等）登记在两篇深挖页的末节。
- **官方博客**（`qwen.ai/blog?id=qwen3.8-flash-next`）仍未摄入，但大概率是报告的科普版，优先级低。
- Qwen3.8 其余档位（2.4T-A95B、27B）尚无独立 arXiv/PDF 训练报告；预训练数据/token、优化器、训练 Infra、RL 算法与关键消融均未披露。
- 历史 Qwen/Qwen2/Qwen2.5/Qwen3-Omni PDF 已在 `raw/`，但尚未形成家族演进分析页。
- 开放权重目前没有与托管 Max endpoint 分离的统一第三方 benchmark 复测。

## 关联域

- [[01_theory/01_models/index|模型架构与模型家族总索引]] — 上级模型域入口。
- [[14_kimi_k3_analysis]] — 同期 2T+ 开放权重模型对照。
- [[24_agentic_rl_algorithm_analysis]] — 真实工作 RL 的 trajectory、reward 与 harness 工程契约。
