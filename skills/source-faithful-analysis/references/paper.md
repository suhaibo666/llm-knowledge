# Source-type pack — Research paper (arXiv / PDF)

The concrete edge of source-faithful analysis when the source is a **paper**. Read alongside the
SKILL.md core. The ground truth here is a passage/table in the paper — and, for an open-weights
model/system paper, the released artifact.

## Locator & baseline
- **Locator = `§N.M` / `pX` / `Table K` / `Fig. J` / `Eq. (n)`.** You **open the paper to that spot
  and read the passage before you cite it** — catch the conditions ("only with Muon", "at 128K", "on
  the private set"). Never cite from the abstract, a blog, a press release, or memory: they round,
  drop conditions, and conflate papers. A number lifted from the abstract that the body qualifies (or
  contradicts) is worse than no number.
- **Baseline = arXiv id + VERSION + date** (`v1` ≠ `v2` — ablations, numbers, even claims get revised
  between versions). Put it in every header.
- **Separate "the paper reports X (§4.2, Table 3)" from "this implies Y" (your inference).**

## Ingest the text in a citeable, splittable form (Phase 0)
1. **Confirm the paper:** WebFetch `https://arxiv.org/abs/<id>` → title, authors, **version**, date.
2. **Don't trust a truncated fetch.** WebFetch on the HTML/PDF runs a *small* model over converted
   markdown and **truncates long papers** (tables/tail sections silently drop). The Read tool renders
   a PDF only if poppler/`pdftoppm` is installed (often not, e.g. bare Windows).
3. **Robust path — download → page-markered text dump:**
   ```
   # mind the temp-path trap: Bash /tmp ≠ the Windows Read tool's path — write where both see it
   Invoke-WebRequest -Uri "https://arxiv.org/pdf/<id>v<v>" -OutFile "$env:TEMP\paper.pdf"   # or curl -L
   ```
   ```python
   import PyPDF2  # or: import pypdf as PyPDF2
   r = PyPDF2.PdfReader(r"<path>\paper.pdf"); out=[]
   for i,p in enumerate(r.pages):
       out.append(f"\n\n===== PAGE {i+1} =====\n"); out.append(p.extract_text())
   open(r"<path>\paper.txt","w",encoding="utf-8").write("".join(out)); print("pages:",len(r.pages))
   ```
   Then `Read`/`Grep` the `.txt`. The `===== PAGE N =====` markers let you **cite by page** and, in
   Phase 3, hand each agent a precise **page range** ("read PAGE 8–10"). Caveat: `extract_text()`
   mangles math/table layout — reason from surrounding prose and double-check any number pulled from a
   garbled table.

## Cross-check a model/system paper against the released artifact
The highest-value fidelity move for an **open-weights** paper: **the paper tells you the *why*; the
shipped weights tell you the *what*.** Prose rounds ("~80 layers"), describes an earlier config, or
omits exact dims.
- **Fetch ground truth:** WebFetch `https://huggingface.co/<org>/<model>/resolve/main/config.json`
  (returns the JSON); also the model card and the modeling code (`model_type`, attention/MoE modules).
- **Map config → claims** (MoE/latent-attention LLM): `hidden_size`/`num_hidden_layers`
  (+ `first_k_dense_replace`); `num_attention_heads`/`qk_nope_head_dim`/`qk_rope_head_dim`/`v_head_dim`;
  `kv_lora_rank`+RoPE = latent KV; `n_routed_experts`/`num_experts_per_tok`/`n_shared_experts`/
  `routed_scaling_factor`/`scoring_func`; `num_nextn_predict_layers` (MTP); `index_topk`/`index_n_heads`
  (sparse indexer); `vocab_size`/`max_position_embeddings`/`rope_theta`.
- **Rule:** released artifact = ground truth for the *number*; paper = ground truth for the
  *rationale*. Use config values in a **complete model-structure figure** and any exact-hyperparameter
  table; when they disagree, flag a `[!contradiction]` (figure follows the weights, prose keeps the
  paper's why). *Worked example (GLM-5):* `num_hidden_layers`=78 vs paper's "80"; "576-dim latent KV"
  = `kv_lora_rank` 512 + RoPE 64; "256 experts" = top-8 of 256 + 1 shared; `index_topk`=2048.

## Cross-check against the reference implementation
`config.json` 验的是**超参**，参考实现验的是**机制**——MTP 的第二头怎么接、router 的 aux-loss 挂在
哪一步、sparse indexer 的 top-k 到底在哪个轴上做。这些论文常常只给一段散文，只有代码能定案。所以
一篇 method/model 论文的分析，**至少要落 2 处「论文机制 ↔ 源码」对应**，每处带 `file:line`。

1. **找实现**：论文页脚或 Introduction 末尾的仓库链接 → 没有就用「标题 / 一作 + 方法名」搜 GitHub →
   再没有就找第三方复现（**必须标明是复现，不是官方**，并说明它可能与论文有偏差）。
2. **取代码**：本库已在父目录维护上游 checkout（Megatron-LM / vLLM / pytorch …），优先复用并记下其
   commit；不在其中的用 `git clone --depth 1 <url>` 到临时目录，同样记下 commit。
3. **对照**：贴代码 ≤30 行，标 `路径/文件.py:行号`，一句话说明它对应论文的哪个式子 / 哪一节。
4. **代码状态必须在页头写明**，四种情况都不能无声跳过：

   | 状态 | 页头怎么写 | 正文怎么处理 |
   |---|---|---|
   | ✅ 官方已发布 | `实现基线: <repo> @ <commit>` | 落 ≥2 处 机制↔`file:line` 对应 |
   | ⏳ 官方声明将发布 | `官方实现未发布(README 声明 <日期>)` | 只据论文写，标出哪些机制待代码确认 |
   | 🔁 仅第三方复现 | `第三方复现: <repo> @ <commit>(非官方)` | 可对照，但每处注明"复现实现，非作者代码" |
   | ❌ 无任何实现 | `无公开实现` | 明说：机制细节只有论文口径，未经代码验证 |

5. **冲突处理**：论文写的和实现不一致时——**实现是「是什么」的 ground truth，论文是「为什么」的
   ground truth**。用 `> [!contradiction]` 标出两边口径，机制描述跟实现走，动机与取舍跟论文走。
   （与上一节 config 的规则同源：数字跟权重，理由跟论文。）

## Paper-specific mechanism evidence

Apply the core analysis contract once per contribution. For a paper, the load-bearing evidence is:

- the section that states the bottleneck or gap;
- the rejected alternative and the criterion or ablation that selected this design;
- the mechanism expressed as math or a diagram, plus the result table **with its baseline column**;
- assumptions, costs, dataset/scale regime, caption conditions, and the Limitations section;
- for model papers, a complete structure view and exact hyperparameters derived from the released
  artifact rather than rounded prose.

Pull the deltas that argue; do not transcribe every cell. An outlook remains optional and needs the
anchored future-work/version evidence required by the core.

## Type-specific red flags
| If you catch yourself… | Do this instead |
|---|---|
| Citing a number from the abstract, a blog, or memory | Open the body to that `§/Table`, read the passage, cite *that* with its conditions. |
| Not recording the arXiv **version** | Pin id+version+date; `v1`≠`v2`. |
| No 约束 section — every number is an improvement | Mine the Limitations §, the caption conditions and the regime it was tested at; state what the design costs. |
| A 发展趋势 paragraph spun from your priors | Anchor it to the paper's future-work line (with §), a later version's change, or a beat-4 constraint — and mark it as inference. Otherwise drop it. |
| A results claim with no table / no baseline column | Reproduce the table with its baseline; numbers without a baseline argue nothing. |
| Guessing architecture hyperparameters from prose | Pull them from the released `config.json`; reconcile, flag paper-vs-weights gaps. |
| Citing a long paper from a truncated WebFetch | Download the PDF, extract a page-markered dump, cite by page. |
| A model-paper analysis with no complete structure figure | Draw the layer-stack + one zoomed block from the config. |
| A mechanism claim never checked against any implementation | Find the official repo (or state "no public implementation"); land ≥2 mechanism↔`file:line` correspondences. |
| Citing a third-party reproduction as if it were the authors' code | Label it a reproduction, pin its commit, and note where it may diverge from the paper. |
| The page never says whether code exists | Put the code status in the header — one of the four rows above. |
