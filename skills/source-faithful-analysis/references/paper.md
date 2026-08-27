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

## Essence checklist (Principle 2, for papers)
- **The thesis — the one main bet.** Lead with it.
- **The five beats per contribution** — the core's mandatory order, instantiated for a paper:
  1. **背景** — the bottleneck / failure / gap that forced this contribution. Usually in the § intro,
     the related-work gap, or the first paragraph of the method section — cite that spot.
  2. **为什么这么设计** — the route chosen **vs 被否掉的替代方案** the paper weighed it against, and
     the criterion that decided it. A paper analysis without the rejected alternatives is a press
     release. Look in the design-choice discussion and in the ablation the authors ran to justify it.
  3. **实现思路与细节** — 机制 (LaTeX or diagram) **plus the 证据**: the table/ablation reproduced
     **with its baseline column**. Pull the *deltas* that argue; don't transcribe every cell.
  4. **约束** — costs, assumptions, the scale/data regime it was shown at, what it doesn't do, when it
     breaks. Sources: the Limitations §, and the conditions buried in captions and footnotes ("only
     with Muon", "at 128K", "on the private set").
  5. **发展趋势** (optional) — the paper's own future-work line, or what a later version / follow-up
     changed. Cite the anchor and mark it as inference; no anchor → omit the beat.

## Doc structure (the paper variant of the template)
- **## 1. Overview** — 背景/问题 first (the bottleneck the paper opens on) → thesis + a
  contribution/results table (component → before → after → purpose → 出处 §/Table) + the headline
  pipeline figure. For a **model paper**, also a **complete
  model-structure figure** (layer-stack + zoomed block) + an exact-hyperparameter table from the
  released config.
- **## 2..N** — one section per contribution, **the five beats** (背景 → 为什么/替代方案 → 机制+证据 → 约束 → 趋势可选).
- **## Related / Cross-references**.

## Type-specific red flags
| If you catch yourself… | Do this instead |
|---|---|
| Citing a number from the abstract, a blog, or memory | Open the body to that `§/Table`, read the passage, cite *that* with its conditions. |
| Not recording the arXiv **version** | Pin id+version+date; `v1`≠`v2`. |
| Paraphrasing the abstract / listing contributions flatly | Rewrite as the five beats: 背景 → 为什么（+被否掉的替代）→ 机制+消融 → 约束. |
| No 约束 section — every number is an improvement | Mine the Limitations §, the caption conditions and the regime it was tested at; state what the design costs. |
| A 发展趋势 paragraph spun from your priors | Anchor it to the paper's future-work line (with §), a later version's change, or a beat-4 constraint — and mark it as inference. Otherwise drop it. |
| A results claim with no table / no baseline column | Reproduce the table with its baseline; numbers without a baseline argue nothing. |
| Guessing architecture hyperparameters from prose | Pull them from the released `config.json`; reconcile, flag paper-vs-weights gaps. |
| Citing a long paper from a truncated WebFetch | Download the PDF, extract a page-markered dump, cite by page. |
| A model-paper analysis with no complete structure figure | Draw the layer-stack + one zoomed block from the config. |
