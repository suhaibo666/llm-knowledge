# Qwen / 阿里巴巴技术路线总览

> Qwen 模型家族的报告分析与开放权重审计入口。当前优先覆盖 2026-08 首次开放 Max 级权重的 Qwen3.8。
> 最后更新：2026-08-13

---

## 分析页

| 段位 | 页面 | 来源基线 | 核心主题 | 状态 |
|---|---|---|---|---|
| 1 | [[10_qwen3_8_analysis]] | 官方博客 2026-08-03 + `Qwen3.8-2.4T-A95B@207bd685` | 2.4T/95B、3:1 Gated DeltaNet/Attention、512 选 10+1 MoE、真实工作 RL、Max endpoint 与开放 checkpoint 边界、自定义许可证 | ✅ 已摄入 |

## 模型状态

| 模型 | 开放状态 | 一句话定位 |
|---|---|---|
| `Qwen3.8-2.4T-A95B` | BF16 + FP8 权重已开放 | text-only、强制 thinking、原生 262K；Max 级 backbone 的可部署 checkpoint |
| `Qwen3.8-Max` | 托管 API | 基于开放 checkpoint，另含视觉输入、non-thinking、默认 1M 上下文与官方工具 |

## 原始来源

- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Max_blog_2026-08-03.txt` — 官方发布博客正文快照。
- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_2_4T_A95B_model_card_207bd685.md` — 固定 revision 模型卡。
- `raw/01_theory/01_models/alibaba_qwen/Qwen3_8_Max_LICENSE_207bd685.txt` — Qwen3.8-Max 自定义许可证。
- 同目录还保存 Qwen、Qwen2、Qwen2.5、Qwen-VL/Audio 与 Qwen3-Omni 的历史 PDF，尚未逐篇摄入。

## 知识缺口

- Qwen3.8 尚无独立 arXiv/PDF 训练报告；预训练数据/token、优化器、训练 Infra、RL 算法与关键消融均未披露。
- 历史 Qwen/Qwen2/Qwen2.5/Qwen3-Omni PDF 已在 `raw/`，但尚未形成家族演进分析页。
- 开放权重目前没有与托管 Max endpoint 分离的统一第三方 benchmark 复测。

## 关联域

- [[01_theory/01_models/index|模型架构与模型家族总索引]] — 上级模型域入口。
- [[14_kimi_k3_analysis]] — 同期 2T+ 开放权重模型对照。
- [[24_agentic_rl_algorithm_analysis]] — 真实工作 RL 的 trajectory、reward 与 harness 工程契约。
