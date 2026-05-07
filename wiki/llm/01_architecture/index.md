# LLM 基础架构 — 目录索引

> 覆盖 Transformer 架构、缩放定律、记忆机制等基础论文分析
> 最后更新: 2026-05-07

---

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[attention_is_all_you_need_analysis]] | Attention Is All You Need (1706.03762) | Transformer 架构, 缩放点积注意力, 多头注意力, 位置编码 |
| [[scaling_laws_analysis]] | Scaling Laws for Neural Language Models (2001.08361) | 幂律缩放, 计算最优训练, 关键批量大小 |
| [[long_context_scaling_law_analysis]] | Long Context Scaling Law (2503.04725) | 交互信息缩放, L2M 条件, Transformer vs SSM 长上下文能力 |
| [[Engram_Analysis]] | Engram — DeepSeek Memory Architecture | 记忆稀疏性, 条件记忆, N-gram 哈希查找, U 型缩放 |
| [[moba_analysis]] | MoBA — Mixture of Block Attention | 块级稀疏注意力, MoE 风格路由 |
| [[kimi_linear_analysis]] | Kimi Linear Attention | 线性注意力机制分析 |

---

## 关联域

- [[../02_training/index]] — 训练技术（优化器, 精度）
- [[../06_infra/megatron-lm/index]] — Megatron-LM 分布式训练
- [[../../torch_compile/index]] — torch.compile 编译优化
