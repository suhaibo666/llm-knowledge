# GLM-5V-Turbo: 原生多模态 Agent 基础模型

> **论文**: GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents
> **作者**: GLM-5V-Turbo Team (Z.ai & Tsinghua University)
> **arXiv**: 2604.26752 (2026-04)
> **开源**: https://github.com/zai-org/GLM-5

---

## 一、模型架构

### 1.1 整体设计

GLM-5V-Turbo 基于 GLM-5-Turbo 语言模型构建，引入原生多模态感知能力：

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   CogViT     │───▶│  Projection  │───▶│  GLM-5-Turbo │
│  视觉编码器   │    │   投影层      │    │  MoE LLM     │
│  (NaFlex)    │    │              │    │  (744B/40B)  │
└──────────────┘    └──────────────┘    └──────────────┘
       │                                    │
       ▼                                    ▼
  图像/视频                            文本/代码
  原生分辨率                            推理/工具
  输入处理                              调用能力
```

### 1.2 CogViT 视觉编码器

**两阶段预训练**：

#### Stage 1: 蒸馏式掩码图像建模

| 配置 | 值 |
|------|-----|
| 掩码比例 | 35% |
| 分辨率 | 224×224 |
| 教师模型 | SigLIP2 (语义) + DINOv3 (纹理) |
| 数据混合 | 80% 高质量自然图像 + 10% 指令跟随 + 10% 科学图像 |
| 优化器 | **Muon** + cosine decay |
| 稳定性 | **QK-Norm** 防止 logits 爆炸 |

#### Stage 2: 对比式图文预训练

| 升级 | 说明 |
|------|------|
| 分辨率 | 固定 224×224 → **NaFlex** 可变分辨率 |
| Batch Size | **64K**（SigLIP loss + 双向分布式实现） |
| 数据 | **80 亿**中英双语图文对 |

### 1.3 多模态多 Token 预测 (MMTP)

**三种方案对比**：

| 方案 | 描述 | 问题 |
|------|------|------|
| Option 1 | 直接传递视觉 embedding 到 MTP head | 视觉 embedding 分布与文本差异大，优化困难 |
| Option 2 | MTP head 输入掩码所有视觉 token | 退化为纯文本 MTP |
| **Option 3 (采用)** | 保留视觉位置信息，用共享 `<|image|>` token 替代 | **最佳平衡** |

**Option 3 优势**：
- 无需跨 pipeline 阶段传播视觉 embedding，**通信复杂度降低**
- 兼容 sequence parallelism 和 context parallelism
- 训练 loss 更低，收敛更稳定

---

## 二、训练方法

### 2.1 预训练数据混合

| 数据类型 | 说明 |
|---------|------|
| 纯文本 | World knowledge, code, math |
| 图文交错 | Caption, interleaved |
| OCR | 多语言、密集布局 |
| 知识 | 学术、百科 |
| GUI | 界面截图、交互 |
| 视频 | 时空理解 |
| 多模态工具使用 | 视觉工具调用 |
| 空间感知 | 2D/3D grounding |
| 多模态编码 | UI-to-code, SVG |

### 2.2 联合多模态 RL

**30+ 任务类别联合优化**：

| 能力层 | 任务 | 提升 (vs SFT) |
|--------|------|--------------|
| **感知** | 2D grounding (RefCOCO) | **+4.8%** |
| | PointBench | **+3.2%** |
| | 视频理解 (MVBench) | **+5.6%** |
| | 3D grounding (SUNRGBD) | **+7.7%** |
| | OCR (OCRBench) | **+4.2%** |
| | 图表理解 (CharXiv) | **+7.7%** |
| **推理** | STEM (MMMU, MathVista) | **+1.8%** |
| **Agent** | GUI Agent (OSWorld) | **+4.9%** |
| | Coding Agent (CC-Backend) | **+0.2%** |
| | 工具使用 (MMSearch) | **+3.5%** |

**关键发现**：
1. RL 跨域干扰弱于 SFT，多域可同时提升
2. 窄分布任务在多任务训练中更稳定
3. 推理模式可跨域迁移
4. 未覆盖能力可能下降 → RL 覆盖范围决定泛化边界

### 2.3 大规模多模态 RL 基础设施

**四维重新设计**：

1. **统一任务和奖励抽象**：
   - VLM RL Gym 统一接口
   - 独立奖励系统编排多验证器
   - 数据源标签实现源特定指标

2. **全流水线解耦与异步**：
   - Rollout 推理、奖励评估、batch 构建、权重传输解耦
   - 完成回调触发奖励计算（无需等待整个 batch）
   - 参考模型参数驻留 CPU，异步预取
   - 支持 early-abort 模式

3. **细粒度内存管理**：
   - ViT 和 projector 独立 recomputation + CPU offloading
   - 防止激活内存随图像数量线性增长

4. **拓扑感知分区与动态负载均衡**：
   - CP/TP 分区上移到 data-loading 阶段
   - 异步 all-to-all 精确分发
   - 大 Python 对象移出 GPU 通信路径（减少 ~7GB buffer）
   - Sequence length + ViT token count 联合 bin-packing

---

## 三、多模态 Agent 能力

### 3.1 多模态工具链

| 场景 | 工具 |
|------|------|
| 通用识别 | zai_recognize_plant/location/person |
| 多模态搜索 | zai_search_web_text/image/similar_images/scholar |
| 浏览器 | zai_load_image_from_url, zai_read_webpage |
| 图像处理 | crop, draw bounding boxes/points/geometry/3D boxes |
| Web 创建 | submit_plan, apply_edits, zai_generate_web_html |
| PPT 创建 | zai_generate_slide_html, zai_generate_outline_ppt |
| Deep Research | zai_dr_python/open_url_mm/visit_img/search/images_search |

### 3.2 外部 Agent 框架集成

- **Claude Code**：系统级协作，多模态终端环境导航
- **AutoClaw**：浏览器和 GUI 自动化，GLM-5V-Turbo 作为视觉语言控制器
- **OpenClaw**：开源个人 Agent 框架

### 3.3 ImageMining 基准

**"Think with image, deep search with image"**

| 维度 | 说明 |
|------|------|
| 数据量 | 217 测试用例 |
| 领域 | Social, Entertainment, Products, Places, Rich Text, Nature, Science |
| 推理类型 | Universal Recognition, Spatio-Temporal, Event, Text-based, Visual Search |
| 关键约束 | **Visual Jump**：中间推理必须涉及视觉转换 |

**GLM-5V-Turbo 得分**：**30.7**（对比 Claude Opus 4.6 和 Kimi K2.5 持平或超越）

---

## 四、性能基准

### 4.1 多模态编码

| Benchmark | GLM-5V-Turbo | Claude Opus 4.6 |
|-----------|-------------|-----------------|
| **Design2Code** | **94.8** | - |
| **Vision2Web** | **领先** | - |

### 4.2 多模态工具使用

| Benchmark | GLM-5V-Turbo | Claude Opus 4.6 | Kimi K2.5 |
|-----------|-------------|-----------------|-----------|
| **ImageMining** | **30.7** | - | - |
| **BrowseComp-VL** | **51.9** | - | - |
| **MMSearch** | **72.9** | - | - |
| **SimpleVQA** | **78.2** | - | - |

### 4.3 GUI Agent

| Benchmark | GLM-5V-Turbo | Claude Opus 4.5 |
|-----------|-------------|-----------------|
| **AndroidWorld** | **75.7** | - |
| **OSWorld** | **62.3** | ~66 |

### 4.4 纯文本编码（保持能力）

| Benchmark | GLM-5V-Turbo | GLM-5-Turbo |
|-----------|-------------|-------------|
| **CC-Backend** | **22.8** | - |
| **CC-Frontend** | **68.4** | - |
| **CC-RepoExploration** | **72.2** | - |

### 4.5 Claw Agent 框架

| Benchmark | GLM-5V-Turbo |
|-----------|-------------|
| **PinchBench** | 87.0 / 80.7 |
| **ClawEval** | 57.7 / 75.0 |
| **ZClawBench** | **57.6** |

---

## 五、开发经验总结

### 5.1 感知是基础

> 即使是最强的 VLM，细粒度感知和空间理解错误仍然常见，这些错误会传播到下游推理、决策和执行。

**代理任务**：
- 多模态编码（前端/SVG）要求捕捉布局、结构、相对位置
- Subject-specific 图像与 SVG 配对数据 → STEM 问题解决提升
- Grounding RL 训练 → GUI Agent 性能提升

### 5.2 分层优化

Agent 能力通过**分层优化**比端到端训练更有效：

```
GUI Agent 分层任务层次:
┌─────────────────────────────────────┐
│ 轨迹级动作预测 (最高层)              │
├─────────────────────────────────────┤
│ 单步动作预测                        │
├─────────────────────────────────────┤
│ GUI Grounding                       │
├─────────────────────────────────────┤
│ 元素感知 (最底层)                    │
└─────────────────────────────────────┘
```

**优势**：
- 低层任务更容易构建、标注和验证
- 低层能力未充分发展时，仅推高层任务往往无法获得可靠提升

### 5.3 端到端任务的规范与验证

端到端任务的价值取决于：
1. **清晰规范**：多源约束而非单一 prompt
2. **可靠验证**：workflow-based verification
3. **程序控制评估**：结构化验证过程

**Vision2Web 实践**：
- 任务基于 PRD、mockup、参考页面等多源规范
- 工作流验证：通过受控依赖步骤序列评估
- 分离功能正确性和视觉一致性评估

---

## 六、官方 Skills

| Skill 类型 | 名称 |
|-----------|------|
| **Native** | PDF-to-Web, PDF-to-PPT, Web Replication, PRD-to-App, Stock Analyst |
| **External Tool** | Image Captioning, Visual Grounding, Doc-based Writing, Resume Screening, Prompt Generation |
| **Specialized (GLM-OCR)** | General OCR, Table Recognition, Handwriting Recognition, Formula Recognition |
| **Specialized (GLM-Image)** | Image Generation |

---

## 七、剩余挑战

1. **Agentic 策略涌现**：如何让模型自发发现更好的策略
2. **长程多模态上下文管理**：多图像/视频的上下文窗口管理
3. **模型能力与 harness 设计的纠缠**：评估结果受框架设计影响

---

## Related Pages

- [[llm/overview]]
- [[zhipu_glm/glm_overview]]
- [[zhipu_glm/glm_5_analysis]]
- [[llm/07_multimodal]]
