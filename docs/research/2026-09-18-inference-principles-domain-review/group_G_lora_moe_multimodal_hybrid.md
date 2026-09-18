# G 组独立审阅：Multi-LoRA、MoE 推理、多模态推理、混合状态模型

> **审阅者**：独立审阅者（未参与撰写）。日期：2026-09-18。
> **页面**（均在 `wiki/01_theory/05_inference/`）：`25_multi_lora_serving_analysis.md`、`31_moe_inference_analysis.md`、`33_multimodal_inference_analysis.md`、`34_hybrid_state_inference_analysis.md`。
> **依据**：`SCRATCH/REVIEW_BRIEF.md`、计划 §6 T25/T31/T33/T34 与 §7、来源证据账、`page-review-rubric.md`、`paper.md`、`mechanism-analysis.md`。
> **结论速览**：25 **REJECT**（1 个 P1：LoRA v1 定位不存在或错节）；31、33、34 **PASS（带 P2）**。本组没有 P0。

## 1. 已打开的来源与复算内容

### 1.1 论文（按页首固定版本下载 PDF，生成分页文本；必要处用 pypdf 版面模式复核）

| 来源 | 版本 | 读过的位置 |
|---|---|---|
| LoRA | arXiv 2106.09685**v1**（2021-06-17） | §2；§3 “Our Method” 全部段落（Eq. (3)、初始化与缩放句、“No Additional Inference Latency”）；§3.1（含 “Practical Benefits and Limitations”）；§4 为 Related Works |
| LoRA（仅作对照） | arXiv 2106.09685v2（2021-10-16） | §4.1、§4.2，用于说明页面定位与两个版本都不符 |
| Punica | arXiv 2310.18547v1 | §2.2、§3、§4、Fig. 3–4、§5、§5.1–5.2、§6、§7.1 |
| S-LoRA | arXiv 2311.03285v1（2023-11-06） | §2 Eq. (1)–(2)、§3、§4.1、Fig. 1–2、§5.1–5.3、§6 开头 |
| DeepSeek-V3 | arXiv 2412.19437v1 | §2.1.2 “No Token-Dropping” 段、§3.4 导言、§3.4.1、§3.4.2（p.18–20） |
| LLaVA | arXiv 2304.08485**v2**（2023-12-11） | §4.1 Eq. (1)、Fig. 1、§4.2 |
| Flamingo | arXiv 2204.14198**v1**（2022-04-29） | §3 导言、§3.1.1、§3.1.2、§3.1.3、Fig. 3–6 |
| Mamba | arXiv 2312.00752**v1** | §1、§2 Eq. (1a)–(4)、§3.1、§3.2 Algorithm 1–2、§3.3.1–3.3.2、§3.4 Fig. 3 |
| Jamba | arXiv 2403.19887**v1**（2024-03-28） | §1、§2、Fig. 1、Table 1、§3.1、Fig. 2 |

### 1.2 源码（只读）

`/Users/suhaibo/97-llm/vllm`，`git rev-parse HEAD` = `199cb9b964822e59ab9b58d88e7be31eb419a2ae`，与 34 页 `源码基线` 一致。读过：`vllm/v1/kv_cache_interface.py::MambaSpec.max_memory_usage_bytes`、`MambaSpec.max_num_blocks_per_req`；`vllm/v1/core/single_type_kv_cache_manager.py::MambaManager.find_longest_cache_hit`、`MambaManager.reachable_block_mask`（`MambaManager.cache_blocks` 仅确认存在）；`vllm/v1/core/kv_cache_coordinator.py::HybridKVCacheCoordinator.find_longest_cache_hit`。

### 1.3 raw 来源索引

页首引用的 8 个 raw 路径全部存在。LoRA、DeepSeek-V3 是既有文件；Punica、S-LoRA、LLaVA、Flamingo、Jamba、Mamba 是本批新建的未跟踪文件。新文件的 arXiv 编号与版本都和页首一致。

### 1.4 复算（`SCRATCH/groupG_recompute.py`，仓库 `.venv` Python 3.13，纯 Python 计算）

- **25**：$X$ 为 3×2、$W=I_2$、$A$ 为 2×1、$B$ 为 1×2。逐行 shrink 结果为 1/1/2，expand 为 (0,2)/(1,−1)/(0,4)，$Y$ 为 (1,4)/(4,0)/(2,4)。行 2 若误用 $A_\alpha$，shrink 得 3。按组顺序堆叠为 `[(1,4),(2,4),(4,0)]`。每个 adapter 单独合并出 $W+sAB$ 也能逐行复现，但每个 adapter 需要一份合并副本。**全部与页面一致。**
- **31**：均衡路由的负载为 4/4，max 4，跨卡赋值 4 个（行 5、6、3、4）。偏斜路由为 6/2，max 6，跨卡 2 个。偏斜加 GPU1 上的 E0 副本后为 4/4，max 4，跨卡 0 个。**全部与页面一致。**
- **33**：LLaVA 式为 $1+6+2=9$ 个输入位置；Flamingo 式最少 3 个文本位置。**一致。**
- **34**：$s_1=1$、$s_2=3$、$s_3^{d}=5$、$s_3=4$；$\min(5,s+4)=5$ 的旧态为 $\{1,2,3,4,5\}$；$H_A=\{0,1,2\}$ 时，$H_R=\{0,2\}$ 得 2，$H_R=\{0,1\}$ 得 1。**一致。**

### 1.5 机械门禁（轻量）

- `check_links --strict`：482 页，broken、ambiguous、bare_index、stale_section、orphans 均为 0。
- `check_math`、`check_markdown`、`check_assets`（`--strict`，只查这 4 页）：均为 0 error、0 warning。
- 4 页没有 `path:line` 引用，不需要跑 `check_locators`。

## 2. 结论表

| page | beat2 | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|
| 25_multi_lora_serving_analysis | pass | pass | transform, layout | pass（图重放行身份，数值在表；P2） | LoRA 定位 0/2；Punica/S-LoRA 内容 5/5（Punica “Eq. 1” 定位不存在） | **REJECT** | 最小失败单元：页首 `:7` 与 §1 出处 `:14` 的 LoRA v1 定位（同类还有 `:22`、`:76`） |
| 31_moe_inference_analysis | pass（可加强，P2） | pass | layout, transform | pass（图未标负载数；P2） | 4/4 | PASS（带 P2） | DeepSeek-V3 数字与“已部署/探索中”限定逐项吻合 |
| 33_multimodal_inference_analysis | pass | pass | transform, layout | pass | 4/4 | PASS（带 P2） | 9 位置账成立；Flamingo 文本 self-KV 依赖图像宜明说 |
| 34_hybrid_state_inference_analysis | pass | pass | timing, coupled-planes | pass | 论文 4/4；源码 4/4 | PASS（带 P2） | 公式定位现已正确；§7 与工程页 08 重复（边界，P2） |

**beat2 依据**：
- 25：否决了“为每个 adapter 合并一份完整模型”，理由是复制大矩阵且失去跨 adapter 批处理；另有 LoRA 单 adapter 合并的对照，以及 $r\ll d$ 不代表免费的成本论证。
- 31：以“不改逻辑 Router、只改物理放置”为主线，副本成本账完整。
- 33：两种连接范式的缓存账由架构推出，明确占位数不等于工作量。
- 34：比较不存快照、存 $k$ 份快照和重算三条路，并说明 KV hash 不能替循环态作证。

**delete-code**：4 页除 Mermaid 外没有代码块，只读正文也能重建机制。

## 3. 发现

### 3.1 25 Multi-LoRA

- **P1｜已确认｜LoRA v1 定位不存在或错节**
  - **位置**：
    - `25_multi_lora_serving_analysis.md:7`：“（§3.1、Eq. 3、§3.2）”
    - `:14`：“LoRA 原论文对**单个已选 adapter**建议部署前合并 $W+\Delta W_j$ … [来源：LoRA §3.2；…]”
    - `:22`：“[来源：LoRA §3.1 Eq. 3；…]”
    - `:76`：“LoRA 原论文 §3.1 冻结基座并训练低秩矩阵”
  - **问题**：固定版本 v1 **没有 §3.2**。第 1 节“合并/共服”论点的出处按页面写法无法在 v1 找到。Eq. (3) 也不在 §3.1。
  - **证据**（2106.09685v1）：
    - §3 “Our Method” 正文（p.3）包含 Eq. (3) $h=W_0x+\Delta Wx=W_0x+BAx$，以及 “During training, $W_0$ is frozen”。
    - 同在 §3 的 “No Additional Inference Latency” 段写明：部署时显式计算并存储 $W=W_0+BA$；换任务时减去 $BA$ 再加上 $B'A'$；不增加推理延迟。
    - §3.1 是 “Applying LoRA to Transformer”（p.3–4）。其中 “Practical Benefits and Limitations” 段写明：若把 $A,B$ 吸收进 $W$，不同任务的输入难以在同一次前向中批处理。
    - 对照 v2：内容整体移到 §4.1/§4.2，也不是页面的 “§3.1/§3.2”。
    - 证据账 T25 行写的是 “2106.09685v1 §3”（这是对的），页面没有沿用。
  - **修复**：保留 v1 时，`:7`、`:14`、`:22` 改为 “§3（Eq. (3)、No Additional Inference Latency 段）；§3.1 Practical Benefits and Limitations 段”；`:76` 改为 “§3”。也可改钉 v2，写 §4.1/§4.2。v2 §4.2 多一句 “possible to not merge the weights and dynamically choose the LoRA modules … for samples in a batch”，正好支撑本页主题。
- **P2｜已确认｜Punica 没有 “Eq. 1”**
  - **位置**：`:22`，“Punica §2.2 Eq. 1”。
  - **证据**：2310.18547v1 全文没有编号公式（版面抽取已复核）。§2.2 只在行文中定义 $W\in\mathbb{R}^{h_1\times h_2}$、$A\in\mathbb{R}^{h_1\times r}$、$B\in\mathbb{R}^{r\times h_2}$、$W+AB$。§4 的分段矩阵式也没有编号。
  - **修复**：改为 “Punica §2.2（行内定义）、§4 分段式”。
- **P2｜已确认｜“同 rank” 的归因**
  - **位置**：`:72`，“Punica §4 为同 rank／segment 场景设计 SGMV”。
  - **证据**：Punica §4 只写按 LoRA 模型连续分段（Fig. 3：`Y[s[i]:s[i+1]] += X[s[i]:s[i+1]] @ W[i]`），没有谈 rank 是否相同。能支撑“同 rank”的是 S-LoRA §5.3：MBGMV “is modified from Punica … to support multiple ranks in a batch”。
  - **修复**：把“同 rank”改由 S-LoRA §5.3 支撑。
- **P2｜已确认｜缩放因子 $s_j$ 没有定义**
  - **位置**：`:22`，“$s_j$ 包括该 adapter 的缩放因子”。
  - **证据**：LoRA v1 §3 的原句是 “scale $\Delta Wx$ by $1/r$”（版面抽取确认）。$\alpha/r$ 是 v2 §4.1 才有的写法。
  - **修复**：写明 $s_j$ 从哪来（v1 为 $1/r$，常见实现为 $\alpha/r$，属于 adapter 配置），初学者否则无从得知。
- **P2｜已确认｜图只重放身份，没有数值；布局只是其中一种**
  - **位置**：`:28-40`。
  - **问题**：Mermaid 重放了 `1α 2β 3α` 的分组、散回和相加，但数值与 $\Delta Y$ 只在 `:56-60` 的表里。图中也没有直接标出不变量“行 2 只加 $\Delta Y_2^{\beta}$”。另外，图用“基座保持行序、LoRA 路散回”的布局；Punica §6 的做法是先把整批按 adapter 重排，基座与 LoRA 共用同一顺序，靠 segment 索引，每层不需要散回。
  - **修复**：在图节点标出 shrink/expand 的值或 “row2←β”；再用一句话说明另一种“整批重排”的布局。
- **P2｜已确认｜初学者用语与分组理由**
  - **位置**：`:26`、`:72`。
  - **问题**：MBGMM/MBGMV 没有展开全称（Multi-size Batched Gather Matrix-Matrix / Matrix-Vector Multiplication，见 S-LoRA §5.3）。分组的理由只写了“避免各行独立启动小算子”。
  - **修复**：补上全称；分组理由可补 Punica §3 的原文理由：同一 adapter 的行聚在一起，提高算子运算强度并用上 Tensor Core。

**已通过的要求**：页面没有把多个 adapter 写成同一个合并模型（`:14`、`:54`、`:62`）。前缀缓存身份包含 base/adapter 版本（`:74`）。Punica §5.2 的异步 H2D 与“加载完成后才入批”、S-LoRA §4.1 的主存存放与按批拉取、§5.1 Unified Paging、§5.2 预取，归因都准确。

### 3.2 31 MoE

DeepSeek-V3 v1 逐项核对全部吻合，**没有归因错误**：
- **prefill**：跨节点 IB 再到节点内 NVLink 的 all-to-all；EP32 的理由是“each expert processes a sufficiently large batch”；目标是让每卡 token 数近似相同；高负载专家按在线统计周期调整（“e.g., every 10 minutes”）；在节点内重排；32 个冗余专家；每 GPU 在原有 8 个之外另放 1 个；两个 micro-batch 重叠已部署；动态冗余标为 exploring，页面没有写错。
- **decode**：320 GPU、EP320、每 GPU 1 个专家、64 GPU 放冗余与共享专家、IB 点对点、周期选冗余集合但无需重排；两个 micro-batch 重叠与动态冗余均为 exploring；每专家 batch “usually within 256 tokens”，瓶颈是访存。

页面没有写最小部署单元（4 节点 32 卡 / 40 节点 320 卡）和 attention 的 TP4+SP、DP8/DP80。这是遗漏，不是错误，不单列发现。

- **P2｜已确认｜延迟式与正文不一致**
  - **位置**：`:61-66` 只对专家计算取 $\max_g$，$T_{\mathrm{dispatch}}$、$T_{\mathrm{combine}}$ 是标量；而 `:59` 正文说热门卡会同时放大输入和返回通信。
  - **修复**：通信两段也按最慢卡取值（例如三段分别取 $\max_g$），与正文一致。
- **P2｜已确认｜图没有标负载和副本代价**
  - **位置**：`:36` 的图规格说要“展示…两卡专家行数”，`:38-53` 的 Mermaid 只列了分配。
  - **修复**：标出 4/4、6/2、4/4，以及“GPU1 多存一份 E0 权重”的代价。
- **P2｜已确认｜记号与措辞**
  - `:80` 的 “$max_g$” 会渲染成斜体 m·a·x，应写 `$\max_g$`。
  - `:80` 的“通常小于 256”对应原文 “within 256”，应为“不超过 256”。
  - `:22`、`:80` 的 IB、NVLink 首次出现时没有给出全称（InfiniBand 等）。
- **P2｜已确认｜beat-2 可加强**
  - **位置**：`:72-74`。
  - **问题**：只隐含否决了“改 Router”，没有点名推理时最常见的另一条路：按 capacity 截断或丢 token。它会改变输出。
  - **证据**：DeepSeek-V3 v1 §2.1.2 “No Token-Dropping” 明言依靠部署策略保持推理负载均衡，推理时不丢 token。
  - **修复**：补一句，把“为何用副本而不丢 token”的判据写实。

### 3.3 33 多模态

已核对并成立的归因：
- LLaVA v2 §4.1：CLIP ViT-L/14 网格特征 $Z_v=g(X_v)$；可训练投影 $H_v=W\cdot Z_v$（Eq. (1)）；原文说得到 “a sequence of visual tokens”。
- Flamingo v1 §3.1.1：Resampler 把数量可变的特征映射为固定数量的输出（实践中为 64）；视频按 1 FPS 逐帧编码。
- Flamingo v1 §3.1.2：GATED XATTN-DENSE 插在冻结 LM 层之间，tanh 门控初值为 0。
- Flamingo v1 §3.1.3 与 Fig. 6：`<image>` 与 `<EOC>` 是插入的标记；每个文本 token 只 cross-attend 最近一张前图。

“9 个 self-KV 输入位置”成立，$y_1$ 的 KV 尚未写入，与 10 页一致。页面在 `:14`、`:37`、`:74` 都说明了“占位符数不等于 encoder 工作量”。前缀身份包含媒体内容摘要（`:72`）。

- **P2｜已确认｜Flamingo 文本 self-KV 依赖图像，页面没有明说**
  - **位置**：`:24`，“self-KV 主要对应文字序列”；`:72`，“对 cross-attention 模型，要分别核对文字 self-KV 和视觉记忆及其 mask”。
  - **证据**：Flamingo Fig. 5 伪代码中，门控 cross-attention 的输出先加进残差流，然后才进入冻结的 self-attention。§3.1.3 末段明言：“there is still a causal dependency on all previous images in the sequence via causal self-attention in the text decoder”。所以图之后文本位置的 self-KV 本身随图像内容变化。
  - **说明**：`:72` 首句已要求身份包含媒体摘要，并说“相同文字不意味着前缀 KV 相同”，结论并不错。但“分别核对”容易被读成文本 KV 可以只按文本匹配。
  - **修复**：补一句明确这种依赖。
- **P2｜已确认｜表列指代有误**
  - **位置**：`:27`，“第一列是从 LLaVA…”。
  - **问题**：实际指的是表的第 2 列（第 1 列是“阶段”）。
- **P2｜已确认｜Resampler 的设计理由可补**
  - **位置**：`:14`、`:22`。
  - **问题**：页面只写了“固定数量”。
  - **证据**：Flamingo §3.1.1 给出的理由是把输出重采样为固定的少量（64）个，“to significantly reduce the computational complexity of vision-text cross attention”。
  - **修复**：补上这个理由。它直接支撑页面 §5 关于变长媒体成本的论点：latent 数与分辨率、帧数无关，但 encoder 的工作量有关。

### 3.4 34 混合状态

已核对：
- Jamba v1 §1–2、Fig. 1：Transformer 与 Mamba 按 $a:m$ 混排；实现为 $l=8$、$1:7$、每 $e=2$ 层 MoE、$n=16$、$K=2$；KV cache 比普通 Transformer 小 8×。
- Mamba v1：
  - §2 Eq. (2a) $h_t=\bar A h_{t-1}+\bar B x_t$ 与 (2b) $y_t=Ch_t$；
  - §3.2 Algorithm 2 使 $\Delta,B,C$ 依赖输入；
  - §3.3.2 融合扫描不在 HBM 物化中间状态，反向时重算；
  - §3.4 Fig. 3 的 block 含 Conv。
- 证据账所说的“公式定位更正”现已正确。
- §7 的 4 条源码断言在 `199cb9b9` 全部成立：
  - `MambaSpec.max_memory_usage_bytes` 三档预算 $\lceil L/B\rceil+n_{\mathrm{spec}}$ / $2+n_{\mathrm{spec}}+n_{\mathrm{ckpt}}$ / $1+n_{\mathrm{spec}}$ 页，且 $L$ 取 `max_model_len`；
  - `MambaManager.find_longest_cache_hit` 从右向左找单个边界态；
  - `reachable_block_mask` 按可达边界或间隔稀疏保留快照（默认 None 时为密集保留）；
  - `HybridKVCacheCoordinator.find_longest_cache_hit` 做不动点迭代：任一组缩短就重查，最后把 full-attention 组截到共同长度。

- **P2｜已确认｜玩具的不可逆性不是真实原因**
  - **位置**：`:59`，“饱和更新的输出 5 可能来自旧态 1、2、3、4、5 … 真实 Mamba 不是这个饱和函数”。
  - **问题**：页面没有给出真实原因，读者容易以为 SSM 在数学上不可逆。实际上 Mamba 的对角 $\bar A_t=\exp(\Delta_t A)$ 在精确算术下可逆（已知 $x_t$ 就能重算 $\Delta_t,B_t$）。真实障碍有三：
    - $\Delta_t$ 大时 $\bar A_t$ 的元素趋近 0，求逆会放大舍入误差；
    - 短卷积状态是移位窗口，每前进一步就丢掉最旧的输入；
    - 引擎一般不实现逆更新。
  - **修复**：补一句上述真实原因。
- **P2｜已确认｜只讲了单 token 草稿**
  - **位置**：`:55-77`。
  - **问题**：现实中的 $k$ token 草稿会部分接受。在第 $j$ 个位置拒绝时，循环侧需要边界 $j$ 的状态，做法只有两种：一是在验证时保留逐步中间态，约多付 $k\,m_{\mathrm R}$，这正是 §7 预算里 $n_{\mathrm{spec}}$ 的来由；二是从已提交快照重放已接受的 token。页面 §6 的 $(k+1)m_{\mathrm R}$ 与 §7 的 $n_{\mathrm{spec}}$ 都没有和这一点连上。
  - **修复**：补一段，把投机回滚的真实成本讲清。
- **P2｜已确认｜边界与重复（契约漂移）**
  - **位置**：`:8` 的 `源码基线` 与 §7（`:91-97`）。
  - **问题**：§7 复述了工程页 `wiki/02_engineering/03_infer_frameworks/vllm/08_vllm_kv_cache_management_analysis.md` 已有的同一批字段级事实：`MambaSpec.max_memory_usage_bytes` 三档预算（08:582）、不动点协调（08:674）、快照保留（08:692）、`reachable_block_mask`（08:793）。这与证据账对 08 的去重决定（“12–14 讲跨引擎原理，不复制该页字段和配置矩阵”）精神相悖。页首也因此变成 5 行，不符合 4 行页首约定。
  - **说明**：内容当前正确，基线与 08 相同，并已标明是引擎特例，所以记为 P2。
  - **修复**：§7 收成一两句指向 08 §5.1.3/§5.3 的链接，删掉 `源码基线` 行。
- **P2｜已确认｜容量动机没有落地**
  - **位置**：`:13-29`。
  - **问题**：计划大纲第 3 项是“历史范围与容量”，页面只用 $m_{\mathrm A},m_{\mathrm R}$ 符号表达。
  - **证据**：Jamba §2 给出 “8x smaller KV cache”；Table 1 的 256K 上下文、16bit KV：Jamba 4GB、Mixtral 32GB、Llama-2 128GB。
  - **修复**：用一句带出处的数字说明混合模型为什么存在，初学者更容易进入。

## 4. 抽查的锚点

| 页 | 锚点 | 结果 |
|---|---|---|
| 25 | LoRA v1 “§3.2”（合并建议） | **不存在**：内容在 §3 “No Additional Inference Latency” 段 |
| 25 | LoRA v1 “§3.1 Eq. 3” | **错节**：Eq. (3) 在 §3 |
| 25 | Punica v1 §4 SGMV-shrink/expand、按 LoRA 连续分组（Fig. 3–4、§6） | 成立 |
| 25 | Punica v1 §5.2 按需加载 | 成立：异步 H2D，本次执行结束时权重已就绪，再加入批次 |
| 25 | S-LoRA v1 §2 Eq. (1)–(2)、§4.1、§5.1–5.3 | 成立 |
| 25 | Punica v1 “§2.2 Eq. 1” | **定位不存在**：内容成立，但没有编号公式 |
| 31 | DeepSeek-V3 v1 §3.4.1：IB→NVLink、EP32、冗余 32、8+1、约 10 分钟 | 成立 |
| 31 | DeepSeek-V3 v1 §3.4.2：320 卡、EP320、64 卡、IB 点对点、无需重排 | 成立 |
| 31 | DeepSeek-V3 v1 §3.4.1–3.4.2：动态冗余与 decode 重叠为 “exploring” | 成立 |
| 31 | DeepSeek-V3 v1 §3.4.2：“within 256 tokens”、访存瓶颈 | 成立（措辞见 P2） |
| 33 | LLaVA v2 §4.1 Eq. (1) | 成立 |
| 33 | Flamingo v1 §3.1.1 Resampler 固定输出、视频 1 FPS | 成立 |
| 33 | Flamingo v1 §3.1.2 门控 xattn 插在冻结层之间 | 成立 |
| 33 | Flamingo v1 §3.1.3、Fig. 6 最近前图掩码 | 成立 |
| 34 | Jamba v1 §2、Fig. 1 混排 | 成立 |
| 34 | Mamba v1 §2 Eq. (2a) | 成立 |
| 34 | Mamba v1 §3.2 Algorithm 2、§3.3.2 | 成立 |
| 34 | Mamba v1 §3.4 Fig. 3 含 Conv | 成立 |
| 34 | vLLM `MambaSpec.max_memory_usage_bytes` | 成立：三档页数与页面一致 |
| 34 | vLLM `MambaManager.find_longest_cache_hit` | 成立：从右向左找单个边界态 |
| 34 | vLLM `MambaManager.reachable_block_mask` | 成立：稀疏快照保留 |
| 34 | vLLM `HybridKVCacheCoordinator.find_longest_cache_hit` | 成立：不动点迭代，full 组截到共同长度 |

## 5. 给协调者的附注

- **brief 预期与 v1 原文不符（不是页面错误）**：LoRA v1 的缩放是 $1/r$，$\alpha/r$ 是 v2 才有的；v1 的 §4 是 Related Works，没有 §4.1。S-LoRA v1 的编号是 §4 批处理与调度、§5 内存与内核、§6 张量并行，页面用的就是这套编号，是正确的。
- **索引状态待更新**：`wiki/01_theory/05_inference/index.md:33` 把 25 标为“已完成，交叉审阅通过”，在 P1 修复前不准确。
- **Mermaid**：4 张 Mermaid 只读了源码，没有实际渲染；标签文字与正文数字已逐项对账。
