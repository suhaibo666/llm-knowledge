# Directory Restructure — Design Spec

> 2026-05-08 | Status: approved

## Motivation

Current structure has 3 problems:
1. **Naming inconsistency**: raw uses `06_moe_and_distributed` while wiki uses `06_infra`; raw uses `09_pytorch` while wiki uses `torch_compile`
2. **Numbered flat layout is fragile**: 11 top-level numbered dirs in raw, but wiki splits into `llm/` + `torch_compile/` — no consistent mapping
3. **Mixed concerns**: Theory papers (architecture, alignment) and engineering docs (framework diagrams) share the same numbering scheme

## Target Structure

Two top-level domains: `01_theory/` and `02_engineering/`. Numbered from 01 within each.

```
raw/  &  wiki/                   # Mirror structure, same names
├── 01_theory/
│   ├── 01_models/               # Architecture papers + model family technical reports
│   ├── 02_pretraining/          # Scaling laws, optimizers, precision, initialization
│   ├── 03_sft/                  # SFT + LoRA/PEFT (merged per Option C)
│   ├── 04_posttraining/         # RLHF, DPO, GRPO, PPO, reward modeling
│   └── 05_inference/            # CoT, RAG, agents, verification, test-time compute
└── 02_engineering/
    ├── 01_ai_frameworks/        # PyTorch compile stack (Dynamo, Inductor, CUDA Graphs)
    ├── 02_train_frameworks/     # Megatron-LM, distributed training, MindFormers
    ├── 03_infer_frameworks/     # vLLM, TRT-LLM, Mooncake, inference serving
    └── 04_posttrain_frameworks/ # RLHF training infra (reserved, empty)
```

## Category Mapping (Old → New)

| Old raw/ | Old wiki/ | New |
|----------|-----------|-----|
| 01_architecture | llm/01_architecture | 01_theory/01_models |
| 05_model_families/ | llm/05_model_families/ | 01_theory/01_models/ (vendor subdirs) |
| 07_multimodal | llm/07_multimodal | 01_theory/01_models (folded in) |
| 02_training | llm/02_training | 01_theory/02_pretraining |
| — | — | 01_theory/03_sft (new, sparse) |
| 03_alignment | llm/03_alignment | 01_theory/04_posttraining |
| 04_reasoning_and_retrieval | llm/04_reasoning_and_retrieval | 01_theory/05_inference |
| 08_agents | llm/08_agents | 01_theory/05_inference (folded) |
| 09_pytorch | torch_compile | 02_engineering/01_ai_frameworks |
| 06_moe_and_distributed + 10_train_framework | llm/06_infra + llm/10_train_framework | 02_engineering/02_train_frameworks |
| 11_infer_framework | llm/11_infer_framework | 02_engineering/03_infer_frameworks |
| — | — | 02_engineering/04_posttrain_frameworks (new, empty) |

## Key Decisions

1. **Mirror naming**: raw/ and wiki/ use identical directory names — no more translation
2. **Drop old numbering (01-11)**: Replace with two-domain numbering (01_theory/01-05, 02_engineering/01-04)
3. **SFT includes PEFT**: Low-rank adaptation treated as a technique under SFT, avoids a sparse standalone category
4. **Multimodal → 01_models**: Multimodal papers are model architecture variations, not a separate theory branch
5. **Agents → 05_inference**: Agents are fundamentally about inference-time reasoning and tool use

## What Changes

### New directories created
9 leaf category dirs × 2 (raw + wiki) = 18 new directories

### Files moved
- ~99 PDFs → new raw/ locations
- ~102 wiki .md files → new wiki/ locations

### Files rewritten
- All `index.md` files (new taxonomy, new links)
- `CLAUDE.md` (directory layout section)
- `README.md` (structure section)

### Link updates
All `[[wiki links]]` in ~100 .md files must be updated to reflect new paths.

## Risks

- **Breaking Obsidian links**: All internal `[[links]]` need path updates. Mitigation: use global search-replace after move.
- **Wiki link churn**: This is a one-time restructuring; future additions won't need renumbering.

## Out of Scope

- Re-categorizing individual papers (belongs to ingest workflow)
- Creating new wiki pages for sparse categories
- Deleting or merging existing wiki content
