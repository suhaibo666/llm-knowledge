# Attention Is All You Need — Analysis

**Source**: `raw/01_theory/01_models/Attention_Is_All_You_Need-1706.03762.md`
**Authors**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google Brain / Google Research)
**Published**: NIPS 2017 | arXiv:1706.03762

---

## Core Contribution

The Transformer — the first sequence transduction model based **entirely on self-attention**, dispensing with recurrence and convolutions entirely.

## Problem Motivation

RNNs (LSTM/GRU) were dominant for sequence modeling but have a fundamental flaw: **inherently sequential computation**. Hidden state h_t depends on h_{t-1}, precluding parallelization within training examples. This becomes a critical bottleneck at longer sequence lengths.

The Transformer solves this by replacing recurrence with attention, enabling:
- Full parallelization across sequence positions
- O(1) path length between any two positions
- Superior quality with significantly less training time

## Architecture

### Encoder-Decoder Structure

| Component | Layers | Sub-layers per Layer |
|-----------|--------|---------------------|
| Encoder | N=6 | Multi-head self-attention + Position-wise FFN |
| Decoder | N=6 | Masked self-attention + Encoder-decoder attention + Position-wise FFN |

All sub-layers use **residual connections + LayerNorm**: `LayerNorm(x + Sublayer(x))`. Model dimension `d_model = 512`.

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Why the scaling factor `1/sqrt(d_k)`?** For large d_k, dot products grow large in magnitude, pushing softmax into regions with extremely small gradients. The scaling counteracts this effect.

### Multi-Head Attention

Instead of one attention function with d_model dimensions, project Q, K, V h times with different learned linear projections to d_k, d_k, d_v dimensions, run attention in parallel, concatenate and project back.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

**Base model config**: h=8 heads, d_k=d_v=d_model/h=64. Total computational cost similar to single-head attention with full dimensionality.

**Three attention applications**:
1. **Encoder-decoder attention**: Q from decoder, K/V from encoder output
2. **Encoder self-attention**: Q, K, V all from previous encoder layer
3. **Decoder masked self-attention**: Q, K, V from decoder, with causal masking to prevent attending to future positions

### Position-wise Feed-Forward Networks

```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```

- Input/output dimension: d_model = 512
- Inner dimension: d_ff = 2048
- Same linear transformations across positions, different parameters per layer

### Positional Encoding

Since no recurrence/convolution, must inject position information. Uses sine/cosine functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why sinusoidal?** Allows model to easily learn to attend by relative positions — for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_{pos}. Also enables extrapolation to longer sequence lengths.

Learned positional embeddings produce nearly identical results.

### Embeddings

- Learned token embeddings for input/output
- Shared weight matrix between embedding layers and pre-softmax linear transformation
- Embedding weights multiplied by sqrt(d_model)

## Why Self-Attention (vs RNN/CNN)

| Layer Type | Complexity per Layer | Sequential Ops | Max Path Length |
|------------|---------------------|----------------|-----------------|
| Self-Attention | O(n^2 * d) | O(1) | O(1) |
| Recurrent | O(n * d^2) | O(n) | O(n) |
| Convolutional | O(k * n * d^2) | O(1) | O(log_k(n)) |

Self-attention wins on parallelization and path length. It's faster than recurrent when sequence length n < representation dimension d (usually the case with word-piece/byte-pair representations).

## Training Details

| Aspect | Config |
|--------|--------|
| Optimizer | Adam (beta1=0.9, beta2=0.98, eps=1e-9) |
| LR Schedule | Linear warmup (4000 steps) then inverse sqrt decay |
| LR Formula | `d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))` |
| Dropout | 0.1 on sub-layer outputs + embeddings |
| Label Smoothing | epsilon_ls = 0.1 |
| Base Model | 100K steps, ~12 hours on 8 P100 GPUs |
| Big Model | 300K steps, ~3.5 days on 8 P100 GPUs |

## Results

### Machine Translation

| Model | EN-DE BLEU | EN-FR BLEU | Training Cost (FLOPs) |
|-------|-----------|-----------|----------------------|
| GNMT + RL | 24.6 | 39.9 | 2.3e19 / 1.4e20 |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7e19 / 1.2e21 |
| **Transformer (base)** | **27.3** | **38.1** | **3.3e18** |
| **Transformer (big)** | **28.4** | **41.8** | **2.3e19** |

Transformer (big) outperforms all previous best results (including ensembles) on EN-DE by 2+ BLEU, at a fraction of the training cost.

### Ablation Studies

| Variation | Impact |
|-----------|--------|
| Single attention head | -0.9 BLEU |
| Too many heads (32) | -0.4 BLEU |
| Reducing d_k | Hurts quality |
| No dropout | Severe overfitting |
| Learned vs sinusoidal PE | Nearly identical |

### English Constituency Parsing

Transformer generalizes well beyond translation. On WSJ parsing with only 40K sentences, it outperforms BerkeleyParser — a task where RNN seq2seq models previously struggled in small-data regimes.

## Key Insights

1. **Attention alone is sufficient** — no recurrence or convolution needed for state-of-the-art sequence modeling
2. **Multi-head is critical** — single head loses the ability to attend to different representation subspaces
3. **Scaling factor matters** — without 1/sqrt(d_k), large d_k causes gradient vanishing in softmax
4. **Positional encoding choice is flexible** — sinusoidal vs learned makes little difference
5. **Label smoothing hurts perplexity but improves BLEU** — model learns to be more unsure, which generalizes better
6. **Attention heads learn interpretable behaviors** — some heads learn syntactic relationships, anaphora resolution, long-distance dependencies

## Impact

This paper is the foundation of all modern LLMs. Every subsequent model (GPT, BERT, T5, PaLM, LLaMA, DeepSeek, Qwen, etc.) builds on the Transformer architecture. The key innovation — self-attention as a complete replacement for recurrence — enabled the scaling laws that drive the current AI revolution.

## Related Pages

- [[01_theory/index]]
- [[11_deepseek_v2_analysis]] — MLA (Multi-head Latent Attention) is a direct descendant
- [[29_engram_analysis]] — extends attention with memory mechanisms
- [[02_engineering/01_ai_frameworks/index]] — attention kernel optimization via torch.compile
