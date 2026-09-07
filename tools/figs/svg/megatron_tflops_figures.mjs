// 32_megatron_tflops_analysis.md 的三张图。
//
// 图 1：**原理图** —— FLOPs 账本。把共用算例（h=512、4 层、8 头、SwiGLU ffn=1536、词表 4096，
//       THD 一个 batch 里 4 条真实长度 768/512/384/256 的序列）逐项走过 QKV / 输出投影 /
//       core-attention / MLP-or-MoE / logits，每项都是「GEMM 形状 → ×2 FMA ×3 fwd/wgrad/dgrad →
//       乘 ΣLᵢ 或 ΣLᵢ²」，dense 与 MoE 两条 lane 并排，最后落到 num_floating_point_operations
//       上报的那一个数。每个数都由 transformer_flops 的 JS 复刻算出。
// 图 2：偏差方向 —— 同一算例下「上报值」与「真正执行的 GEMM」之比：(a) THD padding 在旧口径
//       batch×seq 与新口径真实 token 之间差多少；(b) MoE 容量丢弃 20% 时上报高估多少；
//       (c) 全量重计算时上报低估多少；(d) (b)(c) 同时发生时两者部分抵消。
// 图 3：DSA —— _dsa_sparse_core_scale 随长度加权均值 L̄ 的曲线（top-k=256）、每层 L² 系数在
//       plain-MLA / DSA 稀疏 core / indexer 打分之间的对比（阶数不降、只降系数）、以及
//       1×/2×/3× 倍率开关与 _num_dsa_indexer_layers 的付费层条带。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 是本页的 principle figure。要回答的问题是「上报的 TFLOPS 数是怎么从超参与两个统计量算出来的」。
// 布局：一张账本表，行 = 模型定义性 GEMM（QKV、输出投影、core attn、FFN、logits），列 =
//   「每 token / 每对的 GEMM 形状」→「×2×3 后的每层系数」→「乘的统计量」→「dense lane 本例值」
//   →「MoE lane 本例值」。FFN 一行是唯一分叉：dense 用 h×ffn×3，MoE 用 h×(moe_ffn×topk + shared)×3。
//   底部两行：token-linear 小计（× ΣLᵢ=1920）与 core 小计（× ΣLᵢ²=1,064,960），再合成上报值；
//   右下角一个盒子写吞吐公式 FLOPs / (t × world_size × 1e12)。
// 标注：ΣLᵢ 与 ΣLᵢ² 用 acc1（它们是本页论点的核心），MoE lane 与「不进公式」的项用 ghost/acc2。
//
// 图 2 布局：四组横条，每组两条 —— 上报值（acc1，长度固定为 1.0）与真正执行（neutral，按比例）。
//   右侧标 reported/executed 的比值与方向词。底部一行列出「从不计入」的工作（norm、softmax、
//   router top-k、通信），用 ghost。
//
// 图 3 布局：上排三栏 —— 左 = scale(L̄) 曲线（横轴对数 L̄，纵轴 0..1），标出 L̄≤k 的平台与本例 L̄、
//   L̄=4096 两个点；中 = 每层 L² 系数柱：plain-MLA、absorbed 稠密、DSA 稀疏 core（本例 L̄）、
//   DSA 稀疏 core（L̄=4096）、indexer 打分 1×、indexer 打分 3×；右 = 8 层 freq=4 offset=1 的付费层条带
//   （is_dsa_skip_topk_layer 的复刻逐层判定）。下排三个盒子：1×/2×/3× 倍率开关及其 detach/no_grad
//   依据、近似的三条边界（docstring 自述）、同一模型三种读法的整批数值。
//
// 用法：node tools/figs/svg/megatron_tflops_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例（页面 §2.1）
// ============================================================================

const EXAMPLE = Object.freeze({
  hidden: 512,
  layers: 4,
  heads: 8,
  ffn: 1536, // SwiGLU：h→2·ffn 与 ffn→h 叠起来是 3 个 h×ffn GEMM
  vocab: 4096, // padded_vocab_size
  seqLength: 1024, // 每个打包缓冲区的容量
  buffers: 2, // 本 batch 两个缓冲区（旧口径下的 batch_size）
  lengths: Object.freeze([768, 512, 384, 256]), // 四条真实序列
  // MoE lane：8 专家 top-2、moe_ffn_hidden_size=512、共享专家 512 —— 活跃宽度 512×2+512=1536，与 dense 相同
  moe: Object.freeze({ experts: 8, topk: 2, moeFfn: 512, shared: 512 }),
  dropFraction: 0.2, // 图 2(b)：容量丢弃掉 20% 的路由 token
  // MLA / DSA 扩展（图 3）：与 tests/unit_tests/test_num_floating_point_operations.py::_make_dsa_args 同尺寸
  mla: Object.freeze({ qLora: 128, kvLora: 64, qkHead: 48, rope: 16, vHead: 64 }),
  dsa: Object.freeze({ idxHeads: 4, idxHeadDim: 32, topk: 256, lossCoeff: 0.01 }),
  dsaShare: Object.freeze({ layers: 8, freq: 4, offset: 1 }), // _num_dsa_indexer_layers 的算例
  longSeq: 4096,
});

const TOK = EXAMPLE.lengths.reduce((a, b) => a + b, 0); // ΣLᵢ
const SQ = EXAMPLE.lengths.reduce((a, b) => a + b * b, 0); // ΣLᵢ²
const OLD_TOK = EXAMPLE.buffers * EXAMPLE.seqLength; // 旧口径 batch×seq
const OLD_SQ = EXAMPLE.buffers * EXAMPLE.seqLength * EXAMPLE.seqLength; // 旧口径 batch×seq²

// ============================================================================
// megatron/training/training.py 的 JS 复刻（基线 85902ef）
// 覆盖的分支：标准 MHA/GQA、plain MLA、DSA、dense MLP、MoE（路由 + 共享，无 moe_latent_size）、
// MTP、logits。未复刻：linear attention（GDN/KDA）、dsv4_hybrid、attention_output_gate、
// moe_latent_size、hybrid_flops —— 遇到即抛错，不静默算错。
// ============================================================================

function isDsaSkipTopkLayer(layerNumber, skipTopkOffset, topkFreq) {
  // megatron/core/transformer/experimental_attention_variant/dsa.py::is_dsa_skip_topk_layer
  if (layerNumber < 1) throw new Error(`layer_number must be 1-indexed and positive, got ${layerNumber}.`);
  if (skipTopkOffset < 0) throw new Error(`skip_topk_offset must be non-negative, got ${skipTopkOffset}.`);
  if (topkFreq < 1) throw new Error(`topk_freq must be positive, got ${topkFreq}.`);
  skipTopkOffset = Math.max(skipTopkOffset, 1);
  return (Math.max(layerNumber - skipTopkOffset, 0) % topkFreq) !== 0;
}

function numDsaIndexerLayers(numLayers, skipTopkOffset, topkFreq) {
  // training.py::_num_dsa_indexer_layers
  let n = 0;
  for (let l = 1; l <= numLayers; l += 1) {
    if (!isDsaSkipTopkLayer(l, skipTopkOffset || 0, topkFreq || 1)) n += 1;
  }
  return n;
}

function dsaSparseCoreScale(totalRealTokens, seqlenSquaredSum, topk) {
  // training.py::_dsa_sparse_core_scale
  if (!topk || totalRealTokens <= 0 || seqlenSquaredSum <= 0) return 1.0;
  const meanSeqlen = seqlenSquaredSum / totalRealTokens;
  const eff = Math.min(topk, Math.floor(meanSeqlen));
  const attended = eff * (1 - eff / (2 * meanSeqlen));
  return attended / (meanSeqlen / 2);
}

function dsaIndexerFlops({ hiddenSize, qLoraRank, nHeads, headDim, numIndexerLayers, indexerLossCoeff }) {
  // training.py::_dsa_indexer_flops
  if (numIndexerLayers <= 0) return [0, 0];
  if (qLoraRank == null) qLoraRank = hiddenSize;
  const indexDim = nHeads * headDim;
  const tokenLinear = numIndexerLayers * (qLoraRank * indexDim + hiddenSize * headDim + hiddenSize * nHeads);
  const core = (numIndexerLayers * indexDim) / 2;
  const fma = 2;
  const lossEnabled = (indexerLossCoeff || 0.0) > 0;
  return [(lossEnabled ? 2 : 1) * fma * tokenLinear, (lossEnabled ? 3 : 1) * fma * core];
}

function isHybridModel(args) {
  // megatron/training/utils/common_utils.py::is_hybrid_model
  return args.hybrid_layer_pattern != null;
}

function moeLayerCounts(a) {
  if (a.num_experts == null) {
    return { numDense: a.num_layers, numMoe: 0, topk: 0, lastIsMoe: 0 };
  }
  let pattern;
  if (Number.isInteger(a.moe_layer_freq)) {
    pattern = Array.from({ length: a.num_layers }, (_, i) => (i % a.moe_layer_freq === 0 ? 1 : 0));
  } else if (Array.isArray(a.moe_layer_freq)) {
    pattern = a.moe_layer_freq;
  } else {
    throw new Error('Illegal --moe-layer-freq argument provided!');
  }
  if (pattern.length !== a.num_layers) throw new Error('Invalid length of moe_layer_pattern');
  const numMoe = pattern.reduce((x, y) => x + y, 0);
  return { numDense: a.num_layers - numMoe, numMoe, topk: a.moe_router_topk, lastIsMoe: pattern[pattern.length - 1] };
}

/** 复刻 num_floating_point_operations（标准 Transformer 路径），返回总数与账本分项。 */
function numFloatingPointOperations(argsIn, batchSize, seqlenSquaredSumInBatch = null, totalRealTokensInBatch = null) {
  const a = { ...argsIn };
  if (isHybridModel(a)) throw new Error('port: hybrid_flops path not ported');
  if (seqlenSquaredSumInBatch == null) seqlenSquaredSumInBatch = batchSize * a.seq_length * a.seq_length;
  if (totalRealTokensInBatch == null) totalRealTokensInBatch = batchSize * a.seq_length;

  if (!a.group_query_attention) a.num_query_groups = a.num_attention_heads;
  let { numDense, numMoe, topk, lastIsMoe } = moeLayerCounts(a);
  let mtp = 0;
  let numLayers = a.num_layers;
  if (a.mtp_num_layers != null) {
    mtp = a.mtp_num_layers;
    numMoe += lastIsMoe * mtp;
    numDense += (1 - lastIsMoe) * mtp;
    numLayers = a.num_layers + mtp;
  }
  const moeFfn = a.moe_ffn_hidden_size != null ? a.moe_ffn_hidden_size : a.ffn_hidden_size;
  if (a.moe_latent_size != null) throw new Error('port: moe_latent_size not ported');
  const sharedFfn = a.moe_shared_expert_intermediate_size == null ? 0 : a.moe_shared_expert_intermediate_size;

  const FB = 3; // fwd + wgrad + dgrad
  const FMA = 2; // 2mnk
  const FFN = a.swiglu ? 3 : 2;

  let attnTerm;
  let attnCore;
  let attnRows;
  if (a.multi_latent_attention) {
    if (a.group_query_attention) throw new Error('assert not args.group_query_attention');
    if (a.attention_output_gate) throw new Error('port: attention_output_gate not ported');
    if (a.experimental_attention_variant === 'dsv4_hybrid') throw new Error('port: dsv4_hybrid not ported');
    const h = a.hidden_size;
    const nh = a.num_attention_heads;
    const qTerm =
      a.q_lora_rank == null
        ? h * nh * (a.qk_head_dim + a.qk_pos_emb_head_dim)
        : a.q_lora_rank * (h + nh * (a.qk_head_dim + a.qk_pos_emb_head_dim) + 1);
    const kvTerm = a.kv_lora_rank * (h + nh * (a.qk_head_dim + a.v_head_dim) + 1) + h * a.qk_pos_emb_head_dim;
    const oTerm = nh * a.v_head_dim * h;
    attnTerm = FB * FMA * (qTerm + kvTerm + oTerm);
    attnCore = FB * FMA * ((nh * (a.qk_head_dim + a.qk_pos_emb_head_dim)) / 2 + (nh * a.v_head_dim) / 2);
    attnRows = { qkv: FB * FMA * (qTerm + kvTerm), out: FB * FMA * oTerm };
  } else {
    if (a.attention_output_gate) throw new Error('port: attention_output_gate not ported');
    const qps = a.kv_channels * a.num_attention_heads;
    const kps = a.kv_channels * a.num_query_groups;
    const vps = a.kv_channels * a.num_query_groups;
    attnTerm = FB * FMA * (a.hidden_size * (qps + kps + vps) + qps * a.hidden_size);
    attnCore = ((FB * FMA * qps) / 2) * 2;
    attnRows = { qkv: FB * FMA * a.hidden_size * (qps + kps + vps), out: FB * FMA * qps * a.hidden_size };
  }

  let dsaExtraTerm = 0;
  let dsaExtraCore = 0;
  let dsa = null;
  const variant = a.experimental_attention_variant;
  if (variant === 'gdn' || variant === 'kda' || variant === 'gated_delta_net') {
    throw new Error('port: linear attention not ported');
  } else if (variant === 'dsv4_hybrid') {
    throw new Error('port: dsv4_hybrid not ported');
  } else if (variant === 'dsa') {
    const nh = a.num_attention_heads;
    const scale = dsaSparseCoreScale(totalRealTokensInBatch, seqlenSquaredSumInBatch, a.dsa_indexer_topk);
    const rawCore = (nh * (a.kv_lora_rank + a.qk_pos_emb_head_dim)) / 2 + (nh * a.kv_lora_rank) / 2;
    attnCore = FB * FMA * rawCore * scale;
    const numIndexerLayers = numDsaIndexerLayers(numLayers, a.dsa_indexer_skip_topk_offset, a.dsa_indexer_topk_freq);
    [dsaExtraTerm, dsaExtraCore] = dsaIndexerFlops({
      hiddenSize: a.hidden_size,
      qLoraRank: a.q_lora_rank,
      nHeads: a.dsa_indexer_n_heads,
      headDim: a.dsa_indexer_head_dim,
      numIndexerLayers,
      indexerLossCoeff: a.dsa_indexer_loss_coeff,
    });
    dsa = { scale, rawCore, numIndexerLayers, absorbedCoreDense: FB * FMA * rawCore };
  } else if (variant != null) {
    throw new Error(`port: unknown variant ${variant}`);
  }

  const selfAttnTerm = attnTerm * numLayers + dsaExtraTerm;
  const selfAttnCore = attnCore * numLayers + dsaExtraCore;

  const mlpDense = FB * FMA * a.hidden_size * (a.ffn_hidden_size * FFN) * numDense;
  const moeRouted = FB * FMA * a.hidden_size * (moeFfn * topk * FFN) * numMoe;
  const moeShared = FB * FMA * a.hidden_size * (sharedFfn * FFN) * numMoe;
  const mtpTerm = FB * FMA * mtp * (3 * a.hidden_size + 2 * a.hidden_size * a.hidden_size);
  const logits = FB * FMA * a.hidden_size * a.padded_vocab_size * (mtp + 1);

  // 与源码同一结合顺序：tok × ( FB·FMA·h·(ffn·FFN·nDense + moe_ffn·topk·FFN·nMoe + shared·FFN·nMoe) + attn + mtp + logit ) + sq × core
  const total =
    totalRealTokensInBatch *
      (FB * FMA * a.hidden_size * ((a.ffn_hidden_size * FFN) * numDense + moeFfn * topk * FFN * numMoe + sharedFfn * FFN * numMoe) +
        selfAttnTerm +
        mtpTerm +
        logits) +
    seqlenSquaredSumInBatch * selfAttnCore;

  return {
    total,
    tok: totalRealTokensInBatch,
    sq: seqlenSquaredSumInBatch,
    rows: {
      qkvPerLayer: attnRows.qkv,
      outPerLayer: attnRows.out,
      corePerLayer: attnCore,
      mlpDensePerLayer: FB * FMA * a.hidden_size * (a.ffn_hidden_size * FFN),
      moeRoutedPerLayer: FB * FMA * a.hidden_size * (moeFfn * topk * FFN),
      moeSharedPerLayer: FB * FMA * a.hidden_size * (sharedFfn * FFN),
      logits,
      mtpTerm,
      numLayers,
      numDense,
      numMoe,
      tokenLinear: FB * FMA * a.hidden_size * ((a.ffn_hidden_size * FFN) * numDense + moeFfn * topk * FFN * numMoe + sharedFfn * FFN * numMoe) + selfAttnTerm + mtpTerm + logits,
      core: selfAttnCore,
      mlpDense,
      moeRouted,
      moeShared,
      dsaExtraTerm,
      dsaExtraCore,
    },
    dsa,
  };
}

// ============================================================================
// 算例参数 → 各 lane 的 args
// ============================================================================

function denseArgs(over = {}) {
  return {
    num_layers: EXAMPLE.layers,
    hidden_size: EXAMPLE.hidden,
    num_attention_heads: EXAMPLE.heads,
    seq_length: EXAMPLE.seqLength,
    padded_vocab_size: EXAMPLE.vocab,
    swiglu: true,
    ffn_hidden_size: EXAMPLE.ffn,
    kv_channels: EXAMPLE.hidden / EXAMPLE.heads,
    group_query_attention: false,
    num_query_groups: EXAMPLE.heads,
    attention_output_gate: false,
    multi_latent_attention: false,
    num_experts: null,
    moe_layer_freq: 1,
    moe_router_topk: 0,
    moe_ffn_hidden_size: null,
    moe_latent_size: null,
    moe_shared_expert_intermediate_size: null,
    mtp_num_layers: null,
    experimental_attention_variant: null,
    q_lora_rank: null,
    qk_head_dim: null,
    qk_pos_emb_head_dim: null,
    kv_lora_rank: null,
    v_head_dim: null,
    hybrid_layer_pattern: null,
    dsa_indexer_n_heads: null,
    dsa_indexer_head_dim: null,
    dsa_indexer_topk: null,
    dsa_indexer_topk_freq: 1,
    dsa_indexer_skip_topk_offset: 0,
    dsa_indexer_loss_coeff: null,
    ...over,
  };
}
function moeArgs(over = {}) {
  return denseArgs({
    num_experts: EXAMPLE.moe.experts,
    moe_router_topk: EXAMPLE.moe.topk,
    moe_ffn_hidden_size: EXAMPLE.moe.moeFfn,
    moe_shared_expert_intermediate_size: EXAMPLE.moe.shared,
    ...over,
  });
}
function mlaArgs(over = {}) {
  return denseArgs({
    multi_latent_attention: true,
    q_lora_rank: EXAMPLE.mla.qLora,
    kv_lora_rank: EXAMPLE.mla.kvLora,
    qk_head_dim: EXAMPLE.mla.qkHead,
    qk_pos_emb_head_dim: EXAMPLE.mla.rope,
    v_head_dim: EXAMPLE.mla.vHead,
    ...over,
  });
}
function dsaArgs(over = {}) {
  return mlaArgs({
    experimental_attention_variant: 'dsa',
    dsa_indexer_n_heads: EXAMPLE.dsa.idxHeads,
    dsa_indexer_head_dim: EXAMPLE.dsa.idxHeadDim,
    dsa_indexer_topk: EXAMPLE.dsa.topk,
    dsa_indexer_loss_coeff: EXAMPLE.dsa.lossCoeff,
    ...over,
  });
}

// ============================================================================
// 图 1 / 图 2 的数
// ============================================================================

const DENSE_THD = numFloatingPointOperations(denseArgs(), EXAMPLE.buffers, SQ, TOK);
const DENSE_OLD = numFloatingPointOperations(denseArgs(), EXAMPLE.buffers); // 旧口径：batch×seq 默认
const MOE_THD = numFloatingPointOperations(moeArgs(), EXAMPLE.buffers, SQ, TOK);
const MOE_OLD = numFloatingPointOperations(moeArgs(), EXAMPLE.buffers);

if (DENSE_THD.total !== MOE_THD.total) {
  throw new Error('算例前提被推翻：活跃宽度相同的 dense 与 MoE lane 上报值应完全相等');
}
if (DENSE_OLD.tok !== OLD_TOK || DENSE_OLD.sq !== OLD_SQ) {
  throw new Error('旧口径默认值与 batch×seq / batch×seq² 不一致');
}

// 图 2：偏差账本（都相对 MoE lane 的上报值）
const routedReported = MOE_THD.rows.moeRouted * TOK; // 路由专家那部分的上报 FLOPs
const droppedFlops = EXAMPLE.dropFraction * routedReported;
const BIAS = Object.freeze({
  reported: MOE_THD.total,
  // (a) THD padding：旧口径 vs 新口径；真正执行的就是新口径（varlen attention 只算真 token）
  oldAccounting: MOE_OLD.total,
  oldOverNew: MOE_OLD.total / MOE_THD.total,
  tokenLinearRatio: OLD_TOK / TOK,
  coreRatio: OLD_SQ / SQ,
  // (b) 容量丢弃：上报不变，执行少了被丢 token 的路由专家 GEMM
  routedShare: routedReported / MOE_THD.total,
  executedDrop: MOE_THD.total - droppedFlops,
  dropRatio: MOE_THD.total / (MOE_THD.total - droppedFlops),
  // (c) 全量重计算：前向再跑一遍，执行 = 4/3 × 上报
  executedRecompute: (MOE_THD.total * 4) / 3,
  recomputeRatio: 3 / 4,
  // (d) 同时发生
  executedBoth: ((MOE_THD.total - droppedFlops) * 4) / 3,
  bothRatio: MOE_THD.total / (((MOE_THD.total - droppedFlops) * 4) / 3),
});

// ============================================================================
// 图 3 的数
// ============================================================================

const MLA_THD = numFloatingPointOperations(mlaArgs(), EXAMPLE.buffers, SQ, TOK);
const DSA_ON = numFloatingPointOperations(dsaArgs(), EXAMPLE.buffers, SQ, TOK);
const DSA_OFF = numFloatingPointOperations(dsaArgs({ dsa_indexer_loss_coeff: null }), EXAMPLE.buffers, SQ, TOK);
const DSA_LONG = numFloatingPointOperations(dsaArgs({ seq_length: EXAMPLE.longSeq }), 1);
const DSA_SHARE = numFloatingPointOperations(
  dsaArgs({ num_layers: EXAMPLE.dsaShare.layers, dsa_indexer_topk_freq: EXAMPLE.dsaShare.freq, dsa_indexer_skip_topk_offset: EXAMPLE.dsaShare.offset }),
  EXAMPLE.buffers,
);

const MEAN_L = SQ / TOK; // 长度加权均值
const DSA = Object.freeze({
  meanL: MEAN_L,
  scaleExample: dsaSparseCoreScale(TOK, SQ, EXAMPLE.dsa.topk),
  scaleLong: dsaSparseCoreScale(EXAMPLE.longSeq, EXAMPLE.longSeq * EXAMPLE.longSeq, EXAMPLE.dsa.topk),
  attendedExample: Math.min(EXAMPLE.dsa.topk, Math.floor(MEAN_L)) * (1 - Math.min(EXAMPLE.dsa.topk, Math.floor(MEAN_L)) / (2 * MEAN_L)),
  denseAttendedExample: MEAN_L / 2,
  // 每层 L² 系数（都乘 ΣLᵢ²）
  plainMlaCore: MLA_THD.rows.corePerLayer, // 3·2·(nh(qk+rope)/2 + nh·v/2)
  absorbedDenseCore: DSA_ON.dsa.absorbedCoreDense, // 3·2·(nh(kv_lora+rope)/2 + nh·kv_lora/2)
  dsaCoreExample: DSA_ON.rows.corePerLayer,
  dsaCoreLong: DSA_LONG.rows.corePerLayer,
  indexerCore1x: DSA_OFF.rows.dsaExtraCore / DSA_OFF.dsa.numIndexerLayers,
  indexerCore3x: DSA_ON.rows.dsaExtraCore / DSA_ON.dsa.numIndexerLayers,
  indexerToken1x: DSA_OFF.rows.dsaExtraTerm / DSA_OFF.dsa.numIndexerLayers,
  indexerToken2x: DSA_ON.rows.dsaExtraTerm / DSA_ON.dsa.numIndexerLayers,
  totalOn: DSA_ON.total,
  totalOff: DSA_OFF.total,
  totalMla: MLA_THD.total,
  shareLayers: numDsaIndexerLayers(EXAMPLE.dsaShare.layers, EXAMPLE.dsaShare.offset, EXAMPLE.dsaShare.freq),
  shareMask: Array.from({ length: EXAMPLE.dsaShare.layers }, (_, i) => !isDsaSkipTopkLayer(i + 1, EXAMPLE.dsaShare.offset, EXAMPLE.dsaShare.freq)),
  totalShare: DSA_SHARE.total,
});

if (DSA.shareLayers !== 2) throw new Error(`8 层 freq=4 offset=1 应有 2 个付费层，得到 ${DSA.shareLayers}`);
if (!(DSA.scaleLong < DSA.scaleExample && DSA.scaleExample < 1)) throw new Error('稀疏缩放应随 L̄ 单调下降');
if (!(DSA.indexerCore3x === 3 * DSA.indexerCore1x)) throw new Error('indexer 打分倍率应为 3×');
if (!(DSA.indexerToken2x === 2 * DSA.indexerToken1x)) throw new Error('indexer 投影倍率应为 2×');

// scale(L̄) 曲线采样点（对数轴）
const SCALE_CURVE = (() => {
  const pts = [];
  for (let e = 6; e <= 13; e += 0.125) {
    const L = 2 ** e;
    pts.push([L, dsaSparseCoreScale(L, L * L, EXAMPLE.dsa.topk)]);
  }
  return pts;
})();

// ============================================================================
// SVG 基础设施（与 megatron_stability_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  return value;
}

const sharedStyle = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .card{fill:#fff;stroke:#E4E7EC;stroke-width:1.2}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
  .cost{fill:none;stroke:#C3651F;stroke-width:2;marker-end:url(#arrowCost)}
  .edge{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:4 4}
  .ti{font-size:18px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .pt{font-size:14px;font-weight:700;fill:#2A313B}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#68717D}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .rank{font-size:11px;font-weight:700;fill:#5B6470}
  .cap{font-size:11px;fill:#747C88}
  .gl{fill:none;stroke:#C8CFDA;stroke-width:.9}
  .curve{fill:none;stroke:#2563EB;stroke-width:2}
  .trigline{fill:none;stroke:#C3651F;stroke-width:2;stroke-dasharray:7 4}
  .refline{fill:none;stroke:#C8CFDA;stroke-width:1;stroke-dasharray:3 4}
  .band{fill:#F5F7FA;stroke:none}
  .bar1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.2}
  .bar0{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .bar2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.2}
  .barg{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dot{fill:#C3651F;stroke:none}
`;

const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
    <marker id="arrowCost" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#C3651F"/></marker>
  </defs>`;

function rect(x, y, w, h, cls = 'neutral', radius = 8) {
  return `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"/>`;
}
function text(x, y, value, cls = 'tx', anchor = 'start') {
  return `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`;
}
function line(x1, y1, x2, y2, cls = 'edge') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}
function polyline(points, cls) {
  return `<path class="${cls}" d="M ${points.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ')}"/>`;
}
function infoBox(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 24;
  out.push(text(x + 12, y + 21, guard(title, 12, inner, `${where}/title`), 'tx'));
  out.push(line(x + 12, y + 29, x + w - 12, y + 29));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(text(x + 12, y + 47 + index * 16, guard(value, 10.5, inner, `${where}/L${index}`), lineCls));
  });
  return out.join('\n');
}
function header(w, title, subtitle) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} __H__" width="${w}" height="__H__" role="img">`,
    defs,
    `<style>${sharedStyle}</style>`,
    text(28, 32, title, 'ti'),
    text(28, 52, subtitle, 'su'),
  ];
}

const FONT = Object.freeze({ ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 });

function assertNoTextOverlap(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  const canvasW = Number(m[1]);
  const canvasH = Number(m[2]);
  const boxes = [];
  for (const t of svg.matchAll(/<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g)) {
    const [, cls, xs, ys, anchor, raw] = t;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      throw new Error(`${name}: 文字盒出画布 "${b.raw}"（${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} × ${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)}，画布 ${canvasW}×${canvasH}）`);
    }
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
    }
  }
  return svg;
}

const seal = (parts, w, h, name) => assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

const fx = (v, d = 1) => v.toFixed(d).replace(/\.0+$/, '');
const fmtInt = (v) => Math.round(v).toLocaleString('en-US'); // 194,280,161,280
const fmtSci = (v) => {
  const e = Math.floor(Math.log10(Math.abs(v)));
  return `${(v / 10 ** e).toFixed(3)}e${e}`; // 1.943e11
};
const pct = (r) => `${r >= 1 ? '+' : '−'}${fx(Math.abs(r - 1) * 100, 1)}%`;

// ============================================================================
// 图 1：FLOPs 账本
// ============================================================================

function renderLedger() {
  const W = 1272;
  const H = 646;
  const h = EXAMPLE.hidden;
  const ffn = EXAMPLE.ffn;
  const { moeFfn, topk, shared } = EXAMPLE.moe;
  const R = DENSE_THD.rows;
  const RM = MOE_THD.rows;
  const L = EXAMPLE.layers;

  const parts = header(
    W,
    '图 1　FLOPs 账本：上报值 = 模型定义性 GEMM 的 FMA 次数 × 2 × 3，只乘两个批级统计量',
    `算例：h=${h}、${L} 层、${EXAMPLE.heads} 头、SwiGLU ffn=${ffn}、词表 ${EXAMPLE.vocab}；THD batch 四条真实序列 ${EXAMPLE.lengths.join('/')} → ΣLᵢ=${fmtInt(TOK)}，ΣLᵢ²=${fmtInt(SQ)}`,
  );

  // 列布局
  const X0 = 28; // 行名
  const X1 = 196; // GEMM 形状
  const X2 = 540; // ×2×3 后每层系数
  const X3 = 704; // 乘的统计量
  const X4 = 790; // dense lane
  const X5 = 1030; // MoE lane
  const XE = 1244;
  const TY = 88;
  const ROW = 46;

  // 表头
  parts.push(rect(X0, TY - 20, XE - X0, 26, 'band', 4));
  parts.push(text(X0 + 8, TY - 3, '模型定义性 GEMM', 'rank'));
  parts.push(text(X1, TY - 3, '每 token / 每对 (query,key) 的形状', 'rank'));
  parts.push(text(X2, TY - 3, '×2 FMA ×3 → 每层系数', 'rank'));
  parts.push(text(X3, TY - 3, '乘的统计量', 'rank'));
  parts.push(text(X4, TY - 3, `dense lane（×${L} 层后 × 统计量）`, 'rank'));
  parts.push(text(X5, TY - 3, 'MoE lane', 'rank'));

  const rows = [
    {
      name: 'QKV 投影',
      shape: `h × (q+k+v) = ${h} × ${3 * h}`,
      coef: `${fmtInt(R.qkvPerLayer)}`,
      stat: 'ΣLᵢ',
      dense: fmtSci(R.qkvPerLayer * L * TOK),
      moe: '同左',
      cls: 'neutral',
    },
    {
      name: '输出投影',
      shape: `h × h = ${h} × ${h}`,
      coef: `${fmtInt(R.outPerLayer)}`,
      stat: 'ΣLᵢ',
      dense: fmtSci(R.outPerLayer * L * TOK),
      moe: '同左',
      cls: 'neutral',
    },
    {
      name: 'core attn：QKᵀ 与 AV',
      shape: `每对 2 × q_proj = 2 × ${h}，因果 /2`,
      coef: `${fmtInt(R.corePerLayer)}`,
      stat: 'ΣLᵢ²',
      dense: fmtSci(R.corePerLayer * L * SQ),
      moe: '同左',
      cls: 'acc1',
    },
    {
      name: 'FFN（唯一分叉）',
      shape: `dense: h×ffn×3 = ${h}×${ffn}×3`,
      coef: `${fmtInt(R.mlpDensePerLayer)}`,
      stat: 'ΣLᵢ',
      dense: fmtSci(R.mlpDense * TOK),
      moe: fmtSci((RM.moeRouted + RM.moeShared) * TOK),
      moeShape: `MoE: h×(moe_ffn×topk+shared)×3 = ${h}×(${moeFfn}×${topk}+${shared})×3`,
      cls: 'neutral',
    },
    {
      name: 'logits',
      shape: `h × vocab = ${h} × ${EXAMPLE.vocab}（只算一次）`,
      coef: `${fmtInt(R.logits)}`,
      stat: 'ΣLᵢ',
      dense: fmtSci(R.logits * TOK),
      moe: '同左',
      cls: 'neutral',
    },
  ];

  rows.forEach((r, i) => {
    const y = TY + 16 + i * ROW;
    parts.push(rect(X0, y - 14, XE - X0, ROW - 6, r.cls === 'acc1' ? 'acc1' : 'card', 6));
    parts.push(text(X0 + 8, y + 6, guard(r.name, 12, X1 - X0 - 16, `fig1/r${i}/name`), 'tx'));
    parts.push(text(X1, y + (r.moeShape ? -1 : 6), guard(r.shape, 10.5, X2 - X1 - 10, `fig1/r${i}/shape`), 'sm'));
    if (r.moeShape) parts.push(text(X1, y + 15, guard(r.moeShape, 10.5, X2 - X1 - 10 + 40, `fig1/r${i}/moe`), 'sm'));
    parts.push(text(X2 + 24, y + 6, guard(r.coef, 10.5, X3 - X2 - 30, `fig1/r${i}/coef`), 'dim'));
    parts.push(text(X3, y + 6, r.stat, r.stat === 'ΣLᵢ²' ? 'costtx' : 'dim'));
    parts.push(text(X4, y + 6, guard(r.dense, 10.5, X5 - X4 - 10, `fig1/r${i}/dense`), 'tx'));
    parts.push(text(X5, y + 6, guard(r.moe, 10.5, XE - X5 - 6, `fig1/r${i}/moe`), r.moe === '同左' ? 'sm' : 'tx'));
  });

  // 小计与合成
  const SY = TY + 16 + rows.length * ROW + 6;
  parts.push(line(X0, SY - 10, XE, SY - 10, 'gl'));
  parts.push(text(X0 + 8, SY + 8, 'token-linear 小计（每 token 系数 × ΣLᵢ）', 'tx'));
  parts.push(text(X2 - 120, SY + 8, guard(`${fmtInt(R.tokenLinear)} × ${fmtInt(TOK)}`, 10.5, 340, 'fig1/tl'), 'dim'));
  parts.push(text(X4, SY + 8, fmtSci(R.tokenLinear * TOK), 'tx'));
  parts.push(text(X5, SY + 8, fmtSci(RM.tokenLinear * TOK), 'tx'));
  parts.push(text(X0 + 8, SY + 30, 'core 小计（每对系数 × ΣLᵢ²）', 'tx'));
  parts.push(text(X2 - 120, SY + 30, guard(`${fmtInt(R.core)} × ${fmtInt(SQ)}`, 10.5, 340, 'fig1/core'), 'costtx'));
  parts.push(text(X4, SY + 30, fmtSci(R.core * SQ), 'tx'));
  parts.push(text(X5, SY + 30, fmtSci(RM.core * SQ), 'tx'));

  // 上报值
  const BY = SY + 52;
  parts.push(rect(X0, BY, 700, 58, 'acc1', 8));
  parts.push(text(X0 + 12, BY + 22, guard(`num_floating_point_operations = ${fmtInt(DENSE_THD.total)} FLOPs（${fmtSci(DENSE_THD.total)}）`, 12, 676, 'fig1/total'), 'tx'));
  parts.push(
    text(X0 + 12, BY + 44, guard(`dense lane 与 MoE lane 相等：活跃 FFN 宽度都是 ${moeFfn}×${topk}+${shared} = ${ffn}；公式只看活跃宽度，看不见 ${EXAMPLE.moe.experts} 个专家里谁被选中`, 10.5, 676, 'fig1/eq'), 'sm'),
  );
  parts.push(
    infoBox(
      748,
      BY - 6,
      496,
      64,
      '日志里的那个数',
      [
        'throughput per GPU (TFLOP/s/GPU) = 上报值 / (t_iter × world_size × 10¹²)',
        `本例 core 只占 ${fx((R.core * SQ) / DENSE_THD.total * 100, 1)}%：h=${h} 太小，ΣLᵢ² 项在长上下文才会主导`,
      ],
      'neutral',
      'fig1/thr',
    ),
  );

  // 不进公式的项
  const NY = BY + 74;
  parts.push(rect(X0, NY, XE - X0, 44, 'ghost', 6));
  parts.push(text(X0 + 12, NY + 18, '从不计入（源码只数矩阵乘）：LayerNorm / RMSNorm、softmax、激活函数、dropout、router 的 top-k 排序、所有集合通信、重计算再跑的那一遍前向', 'sm'));
  parts.push(text(X0 + 12, NY + 35, `统计量来源：BSHD 默认 batch×seq 与 batch×seq²（本例 ${fmtInt(OLD_TOK)} / ${fmtInt(OLD_SQ)}）；THD 由 cu_seqlens 累加或 sequence_packing_scheduler 直接给出`, 'sm'));

  return seal(parts, W, H, 'fig1-ledger');
}

// ============================================================================
// 图 2：偏差方向
// ============================================================================

function renderBias() {
  const W = 1272;
  const H = 560;
  const BX = 370;
  const BW = 480;
  const BY0 = 96;
  const GROUP = 88;
  const BH = 22;

  const parts = header(
    W,
    '图 2　上报值对真正执行的 GEMM 的偏差：每种效应各推向哪边、推多少',
    `同一算例的 MoE lane，上报值固定 = ${fmtSci(BIAS.reported)} FLOPs；条长 = 该口径下的 FLOPs / 上报值`,
  );

  const maxRatio = 4 / 3;
  const sx = (ratio) => BX + (ratio / maxRatio) * BW;

  const groups = [
    {
      title: '(a) THD padding：旧口径 vs 新口径',
      sub: `旧 ${fmtInt(OLD_TOK)} token / ${fmtInt(OLD_SQ)} 对 → 新 ${fmtInt(TOK)} / ${fmtInt(SQ)}`,
      bars: [
        { label: '旧口径 batch×seq（已修掉）', ratio: BIAS.oldOverNew, cls: 'bar2' },
        { label: '新口径 = 真正执行', ratio: 1, cls: 'bar1' },
      ],
      note: `旧口径高 ${pct(BIAS.oldOverNew)}：token-linear ×${fx(BIAS.tokenLinearRatio, 3)}，core ×${fx(BIAS.coreRatio, 3)}`,
      noteCls: 'costtx',
    },
    {
      title: `(b) MoE 容量丢弃 ${fx(EXAMPLE.dropFraction * 100, 0)}% 的路由 token`,
      sub: `路由专家占上报 ${fx(BIAS.routedShare * 100, 1)}%；公式无 capacity 项`,
      bars: [
        { label: '上报值（不变）', ratio: 1, cls: 'bar1' },
        { label: '真正执行', ratio: BIAS.executedDrop / BIAS.reported, cls: 'bar0' },
      ],
      note: `上报 / 执行 = ${fx(BIAS.dropRatio, 3)}（高估 ${pct(BIAS.dropRatio)}）`,
      noteCls: 'costtx',
    },
    {
      title: '(c) 全量重计算：前向再跑一遍',
      sub: '公式里 3× 是常量，没有 recompute 项',
      bars: [
        { label: '上报值（不变）', ratio: 1, cls: 'bar1' },
        { label: '真正执行 = 4/3 × 上报', ratio: BIAS.executedRecompute / BIAS.reported, cls: 'bar0' },
      ],
      note: `上报 / 执行 = ${fx(BIAS.recomputeRatio, 2)}（低估 ${pct(BIAS.recomputeRatio)}）`,
      noteCls: 'dim',
    },
    {
      title: '(d) (b) 与 (c) 同时发生',
      sub: '方向相反的两个偏差在同一份日志里部分抵消',
      bars: [
        { label: '上报值（不变）', ratio: 1, cls: 'bar1' },
        { label: '真正执行', ratio: BIAS.executedBoth / BIAS.reported, cls: 'bar0' },
      ],
      note: `上报 / 执行 = ${fx(BIAS.bothRatio, 3)}（低估 ${pct(BIAS.bothRatio)}）`,
      noteCls: 'dim',
    },
  ];

  // 参考线
  [0.5, 0.75, 1, 4 / 3].forEach((r) => {
    parts.push(line(sx(r), BY0 - 14, sx(r), BY0 + groups.length * GROUP - 20, r === 1 ? 'trigline' : 'refline'));
    parts.push(text(sx(r), BY0 - 20, r === 4 / 3 ? '4/3' : String(r), 'sm', 'middle'));
  });

  groups.forEach((g, gi) => {
    const y = BY0 + gi * GROUP;
    parts.push(text(28, y + 6, guard(g.title, 12, BX - 40, `fig2/g${gi}/t`), 'tx'));
    parts.push(text(28, y + 22, guard(g.sub, 10.5, BX - 40, `fig2/g${gi}/s`), 'sm'));
    g.bars.forEach((b, bi) => {
      const by = y - 4 + bi * (BH + 6);
      parts.push(rect(BX, by, sx(b.ratio) - BX, BH, b.cls, 3));
      parts.push(text(BX + 6, by + 15, guard(b.label, 10.5, 200, `fig2/g${gi}/b${bi}`), 'sm'));
      parts.push(text(sx(b.ratio) + 6, by + 15, fx(b.ratio, 3), 'sm'));
    });
    parts.push(text(BX + BW + 40, y + 12, guard(g.note, 10.5, 360, `fig2/g${gi}/n`), g.noteCls));
  });

  const FY = BY0 + groups.length * GROUP - 6;
  parts.push(
    infoBox(
      28,
      FY,
      600,
      96,
      '方向表',
      [
        '精确：dense、dropless MoE、THD 真实长度（varlen attention 只算真 token）',
        '高估：MoE 容量丢弃（被丢的路由 GEMM 仍被计入）',
        '低估：任何形式的重计算；以及从不计入的 norm / softmax / 通信',
        '因此它是「等效吞吐」：假定被丢 token 也按当前速度处理、重跑的前向不算数',
      ],
      'neutral',
      'fig2/dir',
    ),
  );
  parts.push(
    infoBox(
      648,
      FY,
      596,
      96,
      '为什么 (a) 已经不在偏差表里',
      [
        `token-linear 乘 total_real_tokens_in_batch，core 乘 seqlen_squared_sum_in_batch，`,
        'docstring 承诺 neither kind of padding shows up in the reported FLOPs；',
        '但这条前提是有人把真实 cu_seqlens 喂给累加器 —— 见正文 §5.1 的接线边界。',
        '(b) 没有对应的 TODO，是当前唯一留下的高估来源。',
      ],
      'ghost',
      'fig2/why',
    ),
  );

  return seal(parts, W, H, 'fig2-bias');
}

// ============================================================================
// 图 3：DSA 稀疏缩放与 indexer
// ============================================================================

function renderDsa() {
  const W = 1272;
  const H = 600;
  const k = EXAMPLE.dsa.topk;

  const parts = header(
    W,
    '图 3　DSA：稀疏 core 的缩放在长度加权均值处求值；indexer 的打分仍是 L²，只是系数小',
    `top-k=${k}；本例 L̄ = ΣLᵢ²/ΣLᵢ = ${fmtInt(SQ)}/${fmtInt(TOK)} = ${fx(DSA.meanL, 2)}；MLA q_lora=${EXAMPLE.mla.qLora} kv_lora=${EXAMPLE.mla.kvLora} qk=${EXAMPLE.mla.qkHead} rope=${EXAMPLE.mla.rope} v=${EXAMPLE.mla.vHead}；indexer ${EXAMPLE.dsa.idxHeads} 头 × ${EXAMPLE.dsa.idxHeadDim}`,
  );

  // ---- 左：scale(L̄) 曲线 ----
  const PX = 70;
  const PY = 100;
  const PW = 430;
  const PH = 240;
  const lo = 64;
  const hi = 8192;
  const lx = (L) => PX + ((Math.log2(L) - Math.log2(lo)) / (Math.log2(hi) - Math.log2(lo))) * PW;
  const ly = (sc) => PY + PH - sc * PH;

  parts.push(rect(PX, PY, PW, PH, 'panel', 4));
  [0.25, 0.5, 0.75, 1].forEach((sc) => {
    parts.push(line(PX, ly(sc), PX + PW, ly(sc), 'refline'));
    parts.push(text(PX - 6, ly(sc) + 4, String(sc), 'sm', 'end'));
  });
  [64, 256, 1024, 4096].forEach((L) => {
    parts.push(line(lx(L), PY + PH, lx(L), PY + PH + 5, 'gl'));
    parts.push(text(lx(L), PY + PH + 18, String(L), 'sm', 'middle'));
  });
  parts.push(text(PX + PW / 2, PY + PH + 36, 'L̄（对数轴）', 'rank', 'middle'));
  parts.push(text(PX, PY - 8, '_dsa_sparse_core_scale(L̄)：稀疏 core 对稠密因果 core 的比', 'rank'));
  parts.push(`<rect class="band" x="${PX}" y="${PY}" width="${(lx(k) - PX).toFixed(1)}" height="${PH}"/>`);
  parts.push(text(PX + 4, PY + 16, guard(`L̄ ≤ k=${k} → 1.0`, 10.5, lx(k) - PX, 'fig3/flat'), 'sm'));
  parts.push(polyline(SCALE_CURVE.map(([L, sc]) => [lx(L), ly(sc)]), 'curve'));

  const mark = (L, sc, label, dx, dy, anchor = 'start') => {
    parts.push(`<circle class="dot" cx="${lx(L).toFixed(1)}" cy="${ly(sc).toFixed(1)}" r="4"/>`);
    parts.push(text(lx(L) + dx, ly(sc) + dy, label, 'costtx', anchor));
  };
  mark(DSA.meanL, DSA.scaleExample, `本例 L̄=${fx(DSA.meanL, 1)} → ${fx(DSA.scaleExample, 3)}`, 8, -8);
  mark(EXAMPLE.longSeq, DSA.scaleLong, `L̄=${EXAMPLE.longSeq} → ${fx(DSA.scaleLong, 3)}`, -8, -10, 'end');
  parts.push(
    text(
      PX,
      PY + PH + 58,
      guard(`每 query 参与的 key：稀疏 e(1−e/2L̄)=${fx(DSA.attendedExample, 1)}，稠密 L̄/2=${fx(DSA.denseAttendedExample, 1)}，e=min(k,⌊L̄⌋)=${Math.min(k, Math.floor(DSA.meanL))}`, 10.5, 480, 'fig3/att'),
      'sm',
    ),
  );

  // ---- 中：每层 L² 系数柱 ----
  const CX = 580;
  const CY = 100;
  const CW = 400;
  const CH = 240;
  const bars = [
    { label: 'plain MLA，core', v: DSA.plainMlaCore, cls: 'barg' },
    { label: 'absorbed，稠密', v: DSA.absorbedDenseCore, cls: 'barg' },
    { label: `DSA core，L̄=${fx(DSA.meanL, 0)}`, v: DSA.dsaCoreExample, cls: 'bar1' },
    { label: `DSA core，L̄=${EXAMPLE.longSeq}`, v: DSA.dsaCoreLong, cls: 'bar1' },
    { label: 'indexer，打分 1×', v: DSA.indexerCore1x, cls: 'bar2' },
    { label: 'indexer，打分 3×', v: DSA.indexerCore3x, cls: 'bar2' },
  ];
  const vmax = Math.max(...bars.map((b) => b.v));
  const slot = CW / bars.length;
  parts.push(rect(CX, CY, CW, CH, 'panel', 4));
  parts.push(text(CX, CY - 8, '每层 L² 系数（都乘 ΣLᵢ²；含 ×2×3 或 indexer 自己的倍率）', 'rank'));
  bars.forEach((b, i) => {
    const bh = (b.v / vmax) * (CH - 40);
    const x = CX + slot * i + slot * 0.18;
    const y = CY + CH - 16 - bh;
    parts.push(rect(x, y, slot * 0.64, bh, b.cls, 3));
    parts.push(text(x + slot * 0.32, y - 6, fmtInt(b.v), 'dim', 'middle'));
    const words = b.label.split('，');
    words.forEach((wd, wi) => parts.push(text(x + slot * 0.32, CY + CH + 14 + wi * 13, guard(wd, 10.5, slot + 10, `fig3/bar${i}`), 'sm', 'middle')));
  });
  parts.push(
    text(CX, CY + CH + 58, guard(`阶数没降：打分每对 n×d/2 = ${EXAMPLE.dsa.idxHeads}×${EXAMPLE.dsa.idxHeadDim}/2，仍乘 ΣLᵢ²；降的只是系数`, 10.5, 420, 'fig3/order'), 'costtx'),
  );

  // ---- 右：付费层条带 ----
  const RX = 1010;
  const RW = 234;
  const SY = 100;
  parts.push(rect(RX, SY, RW, 240, 'neutral', 8));
  parts.push(text(RX + 12, SY + 21, guard('谁付 indexer 的钱', 12, RW - 24, 'fig3/share/t'), 'tx'));
  parts.push(line(RX + 12, SY + 29, RX + RW - 12, SY + 29));
  parts.push(text(RX + 12, SY + 46, guard(`${EXAMPLE.dsaShare.layers} 层，freq=${EXAMPLE.dsaShare.freq}，offset=${EXAMPLE.dsaShare.offset}`, 10.5, RW - 24, 'fig3/share/s'), 'sm'));
  const cw = (RW - 24) / EXAMPLE.dsaShare.layers;
  DSA.shareMask.forEach((pays, i) => {
    const x = RX + 12 + cw * i;
    parts.push(rect(x + 2, SY + 58, cw - 4, 28, pays ? 'acc1' : 'ghost', 3));
    parts.push(text(x + cw / 2, SY + 76, String(i + 1), pays ? 'dim' : 'sm', 'middle'));
  });
  const payers = DSA.shareMask.map((pays, i) => (pays ? i + 1 : null)).filter(Boolean);
  [
    [`付费层：${payers.join('、')} → ${DSA.shareLayers} 层`, 'dim'],
    ['判据：is_dsa_skip_topk_layer', 'sm'],
    ['(max(l−off,0) % freq) ≠ 0 即复用', 'sm'],
    ['其余层复用最近一次 top-k', 'sm'],
    ['MTP 层编号 layer+num_layers，', 'sm'],
    ['落在同一 1..num_layers 区间内', 'sm'],
    [`本例整批：${fmtSci(DSA.totalShare)}`, 'sm'],
  ].forEach(([t, cls], i) => parts.push(text(RX + 12, SY + 108 + i * 16, guard(t, 10.5, RW - 24, `fig3/share/L${i}`), cls)));

  // ---- 底部三个盒子 ----
  const FY = 420;
  parts.push(
    infoBox(
      28,
      FY,
      400,
      150,
      'indexer 不吃全局 3×',
      [
        'loss 关（默认 None）：投影 1×，打分 1×',
        'loss 开（coeff>0）：投影 2× = fwd+wgrad，打分 3× = fwd+dq+dk',
        `本例每层投影 ${fmtInt(DSA.indexerToken1x)} → ${fmtInt(DSA.indexerToken2x)}`,
        `本例每对打分 ${fmtInt(DSA.indexerCore1x)} → ${fmtInt(DSA.indexerCore3x)}`,
        `整批：关 ${fmtSci(DSA.totalOff)} → 开 ${fmtSci(DSA.totalOn)}`,
        '依据：x/qr 先 detach，整段包在 no_grad 或 enable_grad 里',
      ],
      'acc2',
      'fig3/mult',
    ),
  );
  parts.push(
    infoBox(
      448,
      FY,
      400,
      150,
      '近似的三条边界（源码 docstring 自述）',
      [
        '等长 batch 下精确：L̄ 就是真实长度',
        '（打包 THD 基准测试的常见情形）',
        'ragged batch 偏向长序列：L̄ 是长度加权均值，',
        '权重落在支配 attention 代价的长序列上',
        'L̄ ≤ top-k 坍缩为 1.0；离散精确均值的修正项',
        '应是 e−1 而非 e，差 O(1/L)，低于本估计精度',
      ],
      'neutral',
      'fig3/bounds',
    ),
  );
  parts.push(
    infoBox(
      868,
      FY,
      376,
      150,
      '同一模型三种读法（本例 THD batch）',
      [
        `plain MLA 分支（修复前落点）：${fmtSci(DSA.totalMla)}`,
        `DSA 分支，loss 开：${fmtSci(DSA.totalOn)}`,
        `DSA 分支，loss 关：${fmtSci(DSA.totalOff)}`,
        `L=${EXAMPLE.longSeq} 单序列：稀疏 core 只剩稠密的 ${fx(DSA.scaleLong * 100, 1)}%，`,
        '二次项由 indexer 打分接管',
      ],
      'ghost',
      'fig3/read',
    ),
  );

  return seal(parts, W, H, 'fig3-dsa');
}

// ============================================================================

const outputs = new Map([
  ['megatron_tflops_ledger.svg', renderLedger()],
  ['megatron_tflops_bias.svg', renderBias()],
  ['megatron_tflops_dsa.svg', renderDsa()],
]);

export {
  EXAMPLE, TOK, SQ, OLD_TOK, OLD_SQ,
  DENSE_THD, DENSE_OLD, MOE_THD, MOE_OLD, MLA_THD, DSA_ON, DSA_OFF, DSA_LONG, DSA_SHARE,
  BIAS, DSA, SCALE_CURVE,
  numFloatingPointOperations, dsaSparseCoreScale, dsaIndexerFlops, numDsaIndexerLayers, isDsaSkipTopkLayer,
  denseArgs, moeArgs, mlaArgs, dsaArgs, fmtInt, fmtSci, fx, pct, outputs,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir =
    process.argv[2] ?? join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of outputs) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`${name}  ${svg.length} bytes`);
  }
  console.log(`\n账本：dense THD=${fmtInt(DENSE_THD.total)} 旧口径=${fmtInt(DENSE_OLD.total)} MoE THD=${fmtInt(MOE_THD.total)}`);
  console.log(
    `偏差：旧/新=${fx(BIAS.oldOverNew, 4)} 丢弃=${fx(BIAS.dropRatio, 4)} 重计算=${fx(BIAS.recomputeRatio, 4)} 同时=${fx(BIAS.bothRatio, 4)} 路由占比=${fx(BIAS.routedShare, 4)}`,
  );
  console.log(
    `DSA：L̄=${fx(DSA.meanL, 4)} scale=${fx(DSA.scaleExample, 6)} scale(4096)=${fx(DSA.scaleLong, 6)} on=${fmtInt(DSA.totalOn)} off=${fmtInt(DSA.totalOff)} mla=${fmtInt(DSA.totalMla)} share=${fmtInt(DSA.totalShare)} long=${fmtInt(DSA_LONG.total)}`,
  );
}
