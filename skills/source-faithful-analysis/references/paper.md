# Source-type pack — Research paper (arXiv / PDF)

The evidence-acquisition pack for a **paper**. Read it after `source-fidelity.md` and alongside the
selected document profile. The ground truth here is a passage/table in the paper — and, for an
open-weights model/system paper, the released artifact. This pack does not define page shape.

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
`config.json` verifies **hyperparameters**; the reference implementation verifies **mechanisms**:
how a second MTP head connects, where a router attaches auxiliary loss, or which axis a sparse
indexer selects along. A method/model-paper analysis with available implementation evidence must
include **at least two paper-mechanism ↔ code correspondences** at stable `path::qualified.symbol`
anchors on a frozen implementation baseline. Line numbers are optional for exact excerpts.

1. **Find the implementation:** follow the repository link in the paper; otherwise search by title,
   author and method. If only a third-party reproduction exists, label it **unofficial** and explain
   that it may differ from the paper.
2. **Obtain the code:** prefer the host's existing sibling checkout and record its commit. If absent,
   clone to a temporary directory and record the commit. Apply the codebase pack's frozen-baseline
   safeguards; never move an approved checkout during evidence collection.
3. **Map mechanisms:** if an excerpt helps, keep it within 30 lines, identify
   `path/file.py::qualified.symbol`, and explain which equation or section it implements. Add tight
   line ranges only when verifying exact text.
4. **State implementation status in the header.** Cover the applicable case; localize the labels:

   | Status | Header content | Body treatment |
   |---|---|---|
   | Official implementation released | `Implementation baseline: <repo> @ <commit>` | At least two mechanism ↔ stable-symbol correspondences |
   | Official release announced but unavailable | `Official implementation unreleased (README announcement <date>)` | Use the paper; identify mechanisms awaiting code confirmation |
   | Third-party reproduction only | `Third-party reproduction: <repo> @ <commit> (unofficial)` | Compare with explicit unofficial attribution at each correspondence |
   | No implementation found | `No public implementation` | State that mechanism details follow the paper and are not code-verified |

5. **Handle conflicts:** for the released behavior, implementation establishes what executes; the
   paper supplies stated motivation and tradeoffs. Preserve both accounts with
   `> [!contradiction]`; distinguish the paper's method from the shipped variant rather than
   silently conflating them. This follows the config cross-check: shipped numbers come from the
   pinned artifact, and paper rationale stays attributed to the paper.

## Paper-specific mechanism evidence

Apply the selected document profile once per owned contribution. For a paper, the load-bearing
evidence is:

- the section that states the bottleneck or gap;
- the rejected alternative and the criterion or ablation that selected this design;
- the mechanism expressed as math or a diagram, plus the result table **with its baseline column**;
- assumptions, costs, dataset/scale regime, caption conditions, and the Limitations section;
- for model papers, a complete structure view and exact hyperparameters derived from the released
artifact rather than rounded prose.

Pull the deltas that argue; do not transcribe every cell. An outlook remains optional and needs the
anchored future-work/version evidence required by the selected profile.

## Type-specific red flags
| If you catch yourself… | Do this instead |
|---|---|
| Citing a number from the abstract, a blog, or memory | Open the body to that `§/Table`, read the passage, cite *that* with its conditions. |
| Not recording the arXiv **version** | Pin id+version+date; `v1`≠`v2`. |
| No constraints discussion — every number is an improvement | Mine the Limitations §, the caption conditions and the regime it was tested at; state what the design costs. |
| An outlook paragraph spun from your priors | Anchor it to the paper's future-work line (with §), a later version's change, or a beat-4 constraint — and mark it as inference. Otherwise drop it. |
| A results claim with no table / no baseline column | Reproduce the table with its baseline; numbers without a baseline argue nothing. |
| Guessing architecture hyperparameters from prose | Pull them from the released `config.json`; reconcile, flag paper-vs-weights gaps. |
| Citing a long paper from a truncated WebFetch | Download the PDF, extract a page-markered dump, cite by page. |
| A model-paper analysis with no complete structure figure | Draw the layer-stack + one zoomed block from the config. |
| A mechanism claim never checked against any implementation | Find the official repo (or state "no public implementation"); land ≥2 mechanism↔stable-symbol correspondences. |
| Citing a third-party reproduction as if it were the authors' code | Label it a reproduction, pin its commit, and note where it may diverge from the paper. |
| The page never says whether code exists | Put the code status in the header — one of the four rows above. |
