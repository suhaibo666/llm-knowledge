// 37_megatron_trtllm_export_analysis.md 的图。
//
// 图 1：**布局 / 变换图** —— 一个小 GPT 的训练态 state dict 怎样被离线重布局成 TensorRT-LLM 的
//       逐 rank 权重。逐句复刻冻结基线（NVIDIA/Megatron-LM@85902ef）里的规则：
//       single_device_trtllm_model_weights_converter.py::SingleDeviceTRTLLMModelWeightsConverter.
//       _convert_transformer_layer 的 QKV 解交错 + !3383 的 KV 复制（expand）+ gate/up 拆分 +
//       列并行 / 行并行切片（torch.chunk 语义）、::convert 的 vocab pad、::get_padded_vocab_size、
//       ::get_local_model_weights_per_gpu 的 PP 切层与 embedding / lm_head 的 _split；
//       distributed_trtllm_model_weights_converter.py::DistributedTRTLLMModelWeightsConverter.
//       _get_remove_vocab_padding 的去 pad 与 _convert_transformer_layer 的 ng // tp 重排；
//       trtllm_helper.py::TRTLLMHelper._load_scaling_factors / _add_scales_to_converter 的 FP8 scale 三跳。
//       两条规则来自外部 tensorrt_llm、本仓只以公开实现为契约（仓内测试用 mock 替代）：
//       _utils.pad_vocab_size(v, tp) = ceil(v / tp) × tp；Mapping.pp_layers(L) = 按 pp_size 均分。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「训练态布局与 TensorRT-LLM 逐 rank 布局差在哪，一次导出把每一行搬到了哪个 rank，
//   复制与 pad 各多付了多少」。算例：num_layers=4、hidden=8、heads=4、query groups=2（kv_channels=2）、
//   ffn=16（gated：linear_fc1 是 [2·ffn, hidden] 的 gate|up 融合）、vocab=10；目标 TP=4、PP=2。
//   TP=4 > kv heads=2 触发 KV 复制（rep = 2）。
//   面板 A：Megatron 训练布局 —— linear_qkv.weight 16 行按 query group 交错（每 group：q 行 × 2 头 + k 行 + v 行）；
//     linear_fc1.weight 32 行（前半 gate、后半 up）；embedding.word_embeddings 10 行。每格标 (头 / group / 行号)。
//   面板 B：TRT-LLM 全局布局 —— attention.qkv 变成 q|k|v 拼接（KV 复制后 k、v 各 4 份，acc1 标复制份）；
//     mlp.fc / mlp.gate 拆成两张；vocab pad 到 pad_vocab_size(10, 4) = 12（acc2 标 pad 行）。
//   面板 C：4 个 TP rank × 2 个 PP rank 的本地切片 —— 每个 rank 拿哪些行 / 列、哪几层；列并行（qkv、fc、gate、
//     embedding）切行、行并行（dense、proj）切列；acc1 = 复制的 KV 行，acc2 = pad 行；lm_head 不 pad、
//     torch.chunk 给出 3,3,3,1。右侧：每 rank 张量形状表（脚本算出）。
//   面板 D：FP8 时 scale 的三跳（extra_state → 提取 → 过滤 → 注入到输出字典），ghost = 被过滤的 extra_state；
//     右侧：分布式路径在同一算例上的两处失效（ng // tp = 0；divide(10, 4)）与 TP=2 时的去 pad 数字。
//   底部盒子：守恒账（Σ rank 本地 = 全局 + KV 复制 + norm 复制 + pad）、单设备路径的峰值、复刻范围。
//
// 用法：node tools/figs/svg/megatron_trtllm_export_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例
// ============================================================================

const CASE = Object.freeze({
  layers: 4,
  hidden: 8,
  heads: 4,
  groups: 2, // num_query_groups
  kvChannels: 2,
  ffn: 16, // gated：linear_fc1 = [2·ffn, hidden]
  vocab: 10,
  tp: 4, // ExportConfig.inference_tp_size
  pp: 2, // ExportConfig.inference_pp_size
  useParallelEmbedding: true,
  hasLmHead: true, // output_layer.weight 存在（未 tie）
  bytesPerElem: 2, // DataType.bfloat16
  makeVocabSizeDivisibleBy: 128, // 训练侧 calculate_padded_vocab_size 的默认基数（分布式 lane 用）
  distributedTp: 2, // 分布式 lane 可行的对照拓扑
});

// ============================================================================
// 复刻：torch / tensorrt_llm 的两条基础语义
// ============================================================================

// torch.chunk(n 个元素, k)：块大小 = ceil(n / k)，最后一块可短，块数可少于 k
function torchChunkSizes(n, k) {
  if (k === 1) return [n];
  const size = Math.ceil(n / k);
  const out = [];
  let left = n;
  while (left > 0) {
    out.push(Math.min(size, left));
    left -= Math.min(size, left);
  }
  return out;
}
function torchChunk(items, k) {
  const sizes = torchChunkSizes(items.length, k);
  const out = [];
  let at = 0;
  for (const s of sizes) {
    out.push(items.slice(at, at + s));
    at += s;
  }
  return out;
}

// 外部契约（tensorrt_llm._utils.pad_vocab_size 的公开实现；仓内测试用 mocker.patch 替代）
function padVocabSizeTrtllm(vocab, tp) {
  return Math.ceil(vocab / tp) * tp;
}
// 外部契约（tensorrt_llm.Mapping.pp_layers：num_layers 按 pp_size 均分，rank 取自己那段）
function ppLayers(numLayers, ppSize, ppRank) {
  const per = Math.floor(numLayers / ppSize);
  const out = [];
  for (let l = ppRank * per; l < (ppRank + 1) * per; l += 1) out.push(l);
  return out;
}
// 训练侧 megatron/training/vocab_utils.py::_calculate_padded_vocab_size_cached
function trainingPaddedVocab(vocab, divisibleBy, tp) {
  const multiple = divisibleBy * tp;
  return Math.ceil(vocab / multiple) * multiple;
}

// ============================================================================
// 面板 A：Megatron 训练布局的行标签
// ============================================================================

// linear_qkv.weight 的行序：每个 query group 内 [q × (heads/groups), k, v]，每项 kv_channels 行
// （attention.py 注释「q1 q2 k1 v1 | q3 q4 k2 v2」；exporter 按 [hidden, ng, q_num+2, hn] reshape）
function megatronQkvRows(c) {
  const qNum = c.heads / c.groups;
  const hn = c.kvChannels;
  const rows = [];
  for (let g = 0; g < c.groups; g += 1) {
    for (let slot = 0; slot < qNum + 2; slot += 1) {
      for (let r = 0; r < hn; r += 1) {
        const kind = slot < qNum ? 'q' : slot === qNum ? 'k' : 'v';
        const head = kind === 'q' ? g * qNum + slot : g;
        rows.push({ idx: rows.length, group: g, kind, head, r });
      }
    }
  }
  return rows;
}

// ============================================================================
// 复刻 SingleDeviceTRTLLMModelWeightsConverter._convert_transformer_layer 的四种重排
// ============================================================================

// attention_qkv_weight 分支：val.T → reshape(hidden, ng, q_num+2, hn) → split → [expand] → reshape → chunk → concat
// 返回每个 tp rank 的「源行索引」序列（q | k | v），以及复制份数与守卫
function singleDeviceQkv(c) {
  const qNum = c.heads / c.groups;
  const hn = c.kvChannels;
  const rows = megatronQkvRows(c);
  const at = (g, slot, r) => rows[g * (qNum + 2) * hn + slot * hn + r].idx;
  let rep = 1;
  if (c.groups < c.tp) {
    if (c.tp % c.groups !== 0) throw new Error('Number of query groups ... duplicate or split');
    rep = c.tp / c.groups;
  } else if (c.groups % c.tp !== 0) {
    throw new Error('Number of query groups ... duplicate or split');
  }
  // q_weight.reshape(hidden, -1)：(g, s, r) C 序展平
  const q = [];
  for (let g = 0; g < c.groups; g += 1) for (let s = 0; s < qNum; s += 1) for (let r = 0; r < hn; r += 1) q.push(at(g, s, r));
  // k_weight / v_weight：expand 到 rep 份后 (g, e, r) 展平
  const k = [];
  const v = [];
  for (let g = 0; g < c.groups; g += 1) for (let e = 0; e < rep; e += 1) for (let r = 0; r < hn; r += 1) {
    k.push(at(g, qNum, r));
    v.push(at(g, qNum + 1, r));
  }
  const qs = torchChunk(q, c.tp);
  const ks = torchChunk(k, c.tp);
  const vs = torchChunk(v, c.tp);
  const perRank = [];
  for (let i = 0; i < c.tp; i += 1) perRank.push({ q: qs[i], k: ks[i], v: vs[i] });
  return { rows, rep, perRank, rowsPerRank: perRank[0].q.length + perRank[0].k.length + perRank[0].v.length };
}

// mlp_fc_weight 分支（gated）：val.T → chunk(2, -1) → 前半 = TRT fc、后半 = TRT gate → 各 chunk(tp, -1)
function singleDeviceFc1(c) {
  const rowsAll = Array.from({ length: 2 * c.ffn }, (_, i) => i);
  const [fcHalf, gateHalf] = torchChunk(rowsAll, 2);
  return { fc: torchChunk(fcHalf, c.tp), gate: torchChunk(gateHalf, c.tp), fcRows: fcHalf, gateRows: gateHalf };
}

// attention_dense_weight / mlp_projection_weight 分支：val.T → chunk(tp, axis=0) → 再转置回来 = 切输入列
function singleDeviceRowParallel(inCols, tp) {
  return torchChunk(Array.from({ length: inCols }, (_, i) => i), tp);
}

// convert() 的 vocab pad（仅 use_parallel_embedding 且 vocab % tp ≠ 0）+ get_padded_vocab_size + _split
function singleDeviceVocab(c) {
  let embRows = c.vocab;
  if (c.useParallelEmbedding && c.vocab % c.tp !== 0) embRows = padVocabSizeTrtllm(c.vocab, c.tp);
  const vocabSizePadded = c.hasLmHead ? padVocabSizeTrtllm(embRows, c.tp) : embRows;
  const embPerRank = c.useParallelEmbedding
    ? torchChunk(Array.from({ length: embRows }, (_, i) => i), c.tp)
    : Array.from({ length: c.tp }, () => Array.from({ length: embRows }, (_, i) => i));
  // lm_head 从不 pad：_split 直接 torch.chunk
  const lmHeadPerRank = c.hasLmHead ? torchChunk(Array.from({ length: c.vocab }, (_, i) => i), c.tp) : null;
  return { embRows, padRows: embRows - c.vocab, vocabSizePadded, embPerRank, lmHeadPerRank, lmHeadChunkSizes: lmHeadPerRank ? lmHeadPerRank.map((x) => x.length) : null };
}

// get_local_model_weights_per_gpu：每个 (pp, tp) rank 的本地字典（形状 + 来源）
function localWeights(c, qkv, fc1, vocab) {
  const hn = c.kvChannels;
  const denseCols = singleDeviceRowParallel(c.heads * hn, c.tp);
  const projCols = singleDeviceRowParallel(c.ffn, c.tp);
  const ranks = [];
  for (let pp = 0; pp < c.pp; pp += 1) {
    const layers = ppLayers(c.layers, c.pp, pp);
    for (let tp = 0; tp < c.tp; tp += 1) {
      const r = qkv.perRank[tp];
      const entries = [];
      layers.forEach((l, local) => {
        entries.push({ name: `layers.${local}.input_layernorm.weight`, shape: [c.hidden], from: `layer ${l}`, replicated: true });
        entries.push({ name: `layers.${local}.attention.qkv.weight`, shape: [r.q.length + r.k.length + r.v.length, c.hidden], from: `layer ${l}` });
        entries.push({ name: `layers.${local}.attention.dense.weight`, shape: [c.hidden, denseCols[tp].length], from: `layer ${l}` });
        entries.push({ name: `layers.${local}.post_layernorm.weight`, shape: [c.hidden], from: `layer ${l}`, replicated: true });
        entries.push({ name: `layers.${local}.mlp.fc.weight`, shape: [fc1.fc[tp].length, c.hidden], from: `layer ${l}` });
        entries.push({ name: `layers.${local}.mlp.gate.weight`, shape: [fc1.gate[tp].length, c.hidden], from: `layer ${l}` });
        entries.push({ name: `layers.${local}.mlp.proj.weight`, shape: [c.hidden, projCols[tp].length], from: `layer ${l}` });
      });
      if (pp === 0) entries.push({ name: 'vocab_embedding.weight', shape: [vocab.embPerRank[tp].length, c.hidden], from: 'embedding' });
      if (pp === c.pp - 1) {
        if (vocab.lmHeadPerRank) entries.push({ name: 'lm_head.weight', shape: [vocab.lmHeadPerRank[tp].length, c.hidden], from: 'output_layer' });
        entries.push({ name: 'ln_f.weight', shape: [c.hidden], from: 'final_layernorm', replicated: true });
      }
      const numel = entries.reduce((acc, e) => acc + e.shape.reduce((a, b) => a * b, 1), 0);
      ranks.push({ pp, tp, layers, entries, numel, qkv: r, denseCols: denseCols[tp], fc: fc1.fc[tp], gate: fc1.gate[tp], projCols: projCols[tp], emb: pp === 0 ? vocab.embPerRank[tp] : null, lmHead: pp === c.pp - 1 && vocab.lmHeadPerRank ? vocab.lmHeadPerRank[tp] : null });
    }
  }
  return { ranks, denseCols, projCols };
}

// 守恒账：Σ rank 本地 = 训练态全局 + KV 复制 + norm 复制 + vocab pad
function conservation(c, qkv, vocab, local) {
  const hn = c.kvChannels;
  const perLayerGlobal = {
    qkv: c.groups * (c.heads / c.groups + 2) * hn * c.hidden,
    dense: c.hidden * c.heads * hn,
    fc1: 2 * c.ffn * c.hidden,
    fc2: c.hidden * c.ffn,
    norms: 2 * c.hidden,
  };
  const layerGlobal = Object.values(perLayerGlobal).reduce((a, b) => a + b, 0);
  const globalElems = layerGlobal * c.layers + c.vocab * c.hidden + (c.hasLmHead ? c.vocab * c.hidden : 0) + c.hidden;
  const kvDupExtra = (qkv.rep - 1) * c.groups * 2 * hn * c.hidden * c.layers; // 每层多出 (tp − ng)·2·hn·hidden
  const normReplicaExtra = (c.tp - 1) * (2 * c.hidden * c.layers + c.hidden); // layernorm 在每个 tp rank 都有一份
  const padExtra = vocab.padRows * c.hidden;
  const sumLocal = local.ranks.reduce((acc, r) => acc + r.numel, 0);
  if (sumLocal !== globalElems + kvDupExtra + normReplicaExtra + padExtra) {
    throw new Error(`守恒不成立：Σ本地 ${sumLocal} ≠ ${globalElems} + ${kvDupExtra} + ${normReplicaExtra} + ${padExtra}`);
  }
  // 单设备路径的峰值（上界）：调用者手里的输入 state dict 与 converter 的全量字典同时存在；
  // 全量字典 = 全部 .N.bin 切片 + 整张 padded embedding + 整张 lm_head + 一份 ln_f，
  // 逐 rank 字典是它的视图（torch.chunk 沿 dim 0 的块已 contiguous，_split 不复制），
  // 所以全量字典 = Σ rank 本地 − (tp − 1) 份 ln_f（Σ 本地里 ln_f 在末 PP 的每个 tp rank 各算了一份）
  const convertedElems = sumLocal - (c.tp - 1) * c.hidden;
  return {
    perLayerGlobal, layerGlobal, globalElems, kvDupExtra, normReplicaExtra, padExtra, sumLocal,
    convertedElems, peakElems: globalElems + convertedElems,
    globalBytes: globalElems * c.bytesPerElem, peakBytes: (globalElems + convertedElems) * c.bytesPerElem,
    kvDupBytes: kvDupExtra * c.bytesPerElem, padBytes: padExtra * c.bytesPerElem,
  };
}

// ============================================================================
// 复刻 DistributedTRTLLMModelWeightsConverter 在同一算例上的可行性与 TP=2 对照
// ============================================================================

function distributedLane(c, tp) {
  const hn = c.kvChannels;
  const qNum = c.heads / c.groups;
  const groupsPerRank = Math.floor(c.groups / tp); // 源码：self.num_kv_heads // self.inference_tp_size
  const qkvReshapeOk = groupsPerRank >= 1 && c.groups % tp === 0; // ng // tp = 0 时 reshape 形状非法
  const vocabDivisible = c.vocab % tp === 0; // VocabUtility.vocab_range_from_global_vocab_size → divide() 断言
  const trainingPadded = trainingPaddedVocab(c.vocab, c.makeVocabSizeDivisibleBy, tp);
  const localEmbRows = trainingPadded / tp;
  const unpaddedPerRank = vocabDivisible ? c.vocab / tp : null;
  const localQkvRows = qkvReshapeOk ? groupsPerRank * (qNum + 2) * hn : null;
  return { tp, groupsPerRank, qkvReshapeOk, vocabDivisible, trainingPadded, localEmbRows, unpaddedPerRank, localQkvRows, kvDuplication: false };
}

// ============================================================================
// 复刻 FP8 scale 三跳的计数（_load_scaling_factors / 入口过滤 / _add_scales_to_converter）
// ============================================================================

function fp8Hops(c, { gated = true, fp8Kvcache = true } = {}) {
  // 假设：TE spec 下每层 4 个 Linear（linear_qkv / linear_proj / linear_fc1 / linear_fc2）各带一个 _extra_state，
  // 外加 core_attention._extra_state 一个（被 `not key.endswith("core_attention._extra_state")` 跳过）
  const linearPerLayer = 4;
  const extraStatePerLayer = linearPerLayer + 1;
  const extraStateKeys = extraStatePerLayer * c.layers;
  const usedKeys = linearPerLayer * c.layers;
  const scalesPerLinear = 2; // activation_scaling_factor + weights_scaling_factor
  const gateCopiesPerLayer = gated ? 2 : 0; // ".mlp.fc" 的两条再复制一份给 ".mlp.gate"
  const injectedScales = (linearPerLayer * scalesPerLinear + gateCopiesPerLayer) * c.layers;
  const kvScales = fp8Kvcache ? c.layers : 0; // 每层所有 ".attention.qkv.weight*.bin" 键折叠成一个 kv_cache_scaling_factor
  return { extraStateKeys, usedKeys, filteredKeys: extraStateKeys, injectedScales, kvScales, total: injectedScales + kvScales };
}

// ============================================================================
// 运行算例
// ============================================================================

const QKV = singleDeviceQkv(CASE);
const FC1 = singleDeviceFc1(CASE);
const VOCAB = singleDeviceVocab(CASE);
const LOCAL = localWeights(CASE, QKV, FC1, VOCAB);
const CONS = conservation(CASE, QKV, VOCAB, LOCAL);
const DIST_SAME = distributedLane(CASE, CASE.tp);
const DIST_TP2 = distributedLane(CASE, CASE.distributedTp);
const FP8 = fp8Hops(CASE);

// 立论前提自检
{
  if (QKV.rep !== CASE.tp / CASE.groups) throw new Error('KV 复制份数应为 tp / ng');
  if (VOCAB.embRows !== padVocabSizeTrtllm(CASE.vocab, CASE.tp)) throw new Error('embedding 应 pad 到 pad_vocab_size');
  if (DIST_SAME.qkvReshapeOk || DIST_SAME.vocabDivisible) throw new Error('分布式 lane 在本算例应两处失效');
  if (!DIST_TP2.qkvReshapeOk || !DIST_TP2.vocabDivisible) throw new Error('TP=2 对照应可行');
}

if (process.env.TRTLLM_FIG_DEBUG) {
  console.log('QKV', JSON.stringify({ rep: QKV.rep, perRank: QKV.perRank }));
  console.log('FC1', JSON.stringify(FC1));
  console.log('VOCAB', JSON.stringify(VOCAB));
  console.log('LOCAL', JSON.stringify(LOCAL.ranks.map((r) => ({ pp: r.pp, tp: r.tp, layers: r.layers, numel: r.numel, entries: r.entries.map((e) => `${e.name}${JSON.stringify(e.shape)}`) }))));
  console.log('CONS', JSON.stringify(CONS));
  console.log('DIST', JSON.stringify({ same: DIST_SAME, tp2: DIST_TP2 }));
  console.log('FP8', JSON.stringify(FP8));
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_fsdp_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) =>
  String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) {
    throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  }
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
  .outline1{fill:none;stroke:#2563EB;stroke-width:1.8}
  .outline2{fill:none;stroke:#C3651F;stroke-width:1.8}
  .r0{fill:#EEF1F5;stroke:#AEB6C2;stroke-width:1}
  .r1{fill:#E3E8EF;stroke:#AEB6C2;stroke-width:1}
  .r2{fill:#D8DEE8;stroke:#AEB6C2;stroke-width:1}
  .r3{fill:#CDD5E1;stroke:#AEB6C2;stroke-width:1}
  .g0{fill:#F7F8FA;stroke:#AEB6C2;stroke-width:1}
  .g1{fill:#E9ECF1;stroke:#AEB6C2;stroke-width:1}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
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
  .sep{fill:none;stroke:#5B6470;stroke-width:1.6}
`;

const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
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

const FONT = Object.freeze({
  ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5,
  costtx: 10.5, rank: 11, cap: 11,
});

function assertNoTextOverlap(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  const canvasW = Number(m[1]);
  const canvasH = Number(m[2]);
  const boxes = [];
  for (const hit of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = hit;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      throw new Error(
        `${name}: 文字盒出画布 "${b.raw}"（${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} × ` +
          `${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)}，画布 ${canvasW}×${canvasH}）`,
      );
    }
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) {
        throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
      }
    }
  }
  return svg;
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

const shapeStr = (shape) => `(${shape.join(', ')})`;
const rangeStr = (ids) => (ids.length === 0 ? '∅' : ids.length === 1 ? `${ids[0]}` : `${ids[0]}–${ids[ids.length - 1]}`);

// ============================================================================
// 图 1：布局 / 变换
// ============================================================================

function renderLayout() {
  const W = 1272;
  const X0 = 28;
  const c = CASE;
  const hn = c.kvChannels;
  const parts = header(
    W,
    '图 1　TensorRT-LLM 导出的离线重布局：训练态 state dict → TRT-LLM 全局布局 → 逐 rank 切片',
    `算例：num_layers=${c.layers}、hidden=${c.hidden}、heads=${c.heads}、query groups=${c.groups}（kv_channels=${hn}）、ffn=${c.ffn}（gated）、vocab=${c.vocab}；目标 TP=${c.tp}、PP=${c.pp}；TP > kv heads 触发 KV 复制 rep=${QKV.rep}；bf16`,
  );

  // ---- 面板 A：Megatron 训练布局 ----
  const AY = 92;
  parts.push(text(X0, AY - 10, 'A　Megatron 训练布局（每格 = 权重的一行；linear_qkv 按 query group 交错，linear_fc1 是 gate|up 融合）', 'pt'));
  const CELL = 38;
  const CH = 30;
  const qkvRows = QKV.rows;
  parts.push(text(X0, AY + 12, `linear_qkv.weight ${shapeStr([qkvRows.length, c.hidden])}`, 'rank'));
  let ax = X0;
  const AY1 = AY + 18;
  qkvRows.forEach((row) => {
    const x = ax + row.idx * CELL;
    parts.push(rect(x, AY1, CELL - 2, CH, `g${row.group}`, 2));
    parts.push(text(x + (CELL - 2) / 2, AY1 + 12, `${row.kind}${row.head}`, row.kind === 'q' ? 'sm' : 'dim', 'middle'));
    parts.push(text(x + (CELL - 2) / 2, AY1 + 24, `r${row.idx}`, 'sm', 'middle'));
  });
  // group 括号
  const perGroup = (c.heads / c.groups + 2) * hn;
  for (let g = 0; g < c.groups; g += 1) {
    const x1 = ax + g * perGroup * CELL;
    parts.push(text(x1 + (perGroup * CELL) / 2 - 1, AY1 + CH + 13, `group ${g}：q${g * (c.heads / c.groups)}…q${(g + 1) * (c.heads / c.groups) - 1} | k${g} | v${g}`, 'sm', 'middle'));
  }
  ax += qkvRows.length * CELL + 40;
  // embedding
  parts.push(text(ax, AY + 12, `embedding.word_embeddings.weight ${shapeStr([c.vocab, c.hidden])}`, 'rank'));
  const EC = 30;
  for (let i = 0; i < c.vocab; i += 1) {
    const x = ax + i * EC;
    parts.push(rect(x, AY1, EC - 2, CH, 'neutral', 2));
    parts.push(text(x + (EC - 2) / 2, AY1 + 19, `t${i}`, 'sm', 'middle'));
  }
  parts.push(text(ax + (c.vocab * EC) / 2, AY1 + CH + 13, `${c.vocab} 行，训练态未按 TP 对齐`, 'sm', 'middle'));
  // fc1
  const AY2 = AY1 + CH + 26;
  parts.push(text(X0, AY2 + 12, `linear_fc1.weight ${shapeStr([2 * c.ffn, c.hidden])}：前 ${c.ffn} 行 = gate（激活的那一半，源码变量 x_glu），后 ${c.ffn} 行 = up（线性）`, 'rank'));
  const FC = 36;
  const AY3 = AY2 + 18;
  for (let i = 0; i < 2 * c.ffn; i += 1) {
    const x = X0 + i * FC;
    parts.push(rect(x, AY3, FC - 2, CH, i < c.ffn ? 'g0' : 'g1', 2));
    parts.push(text(x + (FC - 2) / 2, AY3 + 12, i < c.ffn ? 'gate' : 'up', 'sm', 'middle'));
    parts.push(text(x + (FC - 2) / 2, AY3 + 24, `r${i}`, 'sm', 'middle'));
  }
  const capA = `每层还有 linear_proj ${shapeStr([c.hidden, c.heads * hn])}、linear_fc2 ${shapeStr([c.hidden, c.ffn])} 与两个 layernorm ${shapeStr([c.hidden])}；每层 ${CONS.layerGlobal} 个元素，整模型 ${CONS.globalElems} 个（含 output_layer ${shapeStr([c.vocab, c.hidden])} 与 final_layernorm）`;
  parts.push(text(X0, AY3 + CH + 16, guard(capA, 11, W - 2 * X0, 'fig1/capA'), 'cap'));

  // ---- 面板 B：TRT-LLM 全局布局 ----
  const BY = AY3 + CH + 46;
  parts.push(text(X0, BY, `B　TRT-LLM 全局布局（single-device converter 的 trtllm_model_weights，尚未按 rank 切开）`, 'pt'));
  const BY1 = BY + 24;
  parts.push(text(X0, BY1 - 6, `attention.qkv：q | k | v 拼接，k、v 各 expand 成 rep=${QKV.rep} 份 → 每 rank ${QKV.rowsPerRank} 行`, 'rank'));
  const BQ = 34;
  let bx = X0;
  const BY2 = BY1 + 2;
  const globalQkv = [];
  QKV.perRank.forEach((r, tp) => r.q.forEach((idx) => globalQkv.push({ idx, kind: 'q', tp })));
  QKV.perRank.forEach((r, tp) => r.k.forEach((idx) => globalQkv.push({ idx, kind: 'k', tp })));
  QKV.perRank.forEach((r, tp) => r.v.forEach((idx) => globalQkv.push({ idx, kind: 'v', tp })));
  // 复制份：同一源行第二次及以后出现
  const seen = new Map();
  globalQkv.forEach((cell, i) => {
    const row = qkvRows[cell.idx];
    const dup = seen.has(cell.idx);
    seen.set(cell.idx, true);
    const x = bx + i * BQ + (cell.kind === 'k' ? 10 : cell.kind === 'v' ? 20 : 0);
    parts.push(rect(x, BY2, BQ - 2, CH, dup ? 'acc1' : `r${cell.tp}`, 2));
    parts.push(text(x + (BQ - 2) / 2, BY2 + 12, `${row.kind}${row.head}`, dup ? 'dim' : 'sm', 'middle'));
    parts.push(text(x + (BQ - 2) / 2, BY2 + 24, `r${row.idx}`, 'sm', 'middle'));
  });
  const qLen = QKV.perRank.reduce((a, r) => a + r.q.length, 0);
  const kLen = QKV.perRank.reduce((a, r) => a + r.k.length, 0);
  parts.push(text(bx + (qLen * BQ) / 2, BY2 + CH + 13, `q：${qLen} 行（${c.heads} 头）`, 'sm', 'middle'));
  parts.push(text(bx + qLen * BQ + 10 + (kLen * BQ) / 2, BY2 + CH + 13, `k：${kLen} 行（蓝 = 复制份）`, 'sm', 'middle'));
  parts.push(text(bx + (qLen + kLen) * BQ + 20 + (kLen * BQ) / 2, BY2 + CH + 13, `v：${kLen} 行`, 'sm', 'middle'));
  bx += globalQkv.length * BQ + 20 + 40;
  // vocab padded
  parts.push(text(bx, BY1 - 6, `vocab_embedding：pad_vocab_size(${c.vocab}, ${c.tp}) = ${VOCAB.embRows}`, 'rank'));
  const VC = 26;
  for (let i = 0; i < VOCAB.embRows; i += 1) {
    const x = bx + i * VC;
    const pad = i >= c.vocab;
    parts.push(rect(x, BY2, VC - 2, CH, pad ? 'acc2' : 'neutral', 2));
    parts.push(text(x + (VC - 2) / 2, BY2 + 19, pad ? '·' : `t${i}`, pad ? 'costtx' : 'sm', 'middle'));
  }
  parts.push(text(bx + (VOCAB.embRows * VC) / 2, BY2 + CH + 13, `橙 = ${VOCAB.padRows} 行零 pad；config.vocab_size = ${VOCAB.vocabSizePadded}`, 'sm', 'middle'));
  // fc / gate
  const BY3 = BY2 + CH + 36;
  parts.push(text(X0, BY3 - 6, `mlp.fc（= Megatron 前半 gate 行 ${rangeStr(FC1.fcRows)}）与 mlp.gate（= 后半 up 行 ${rangeStr(FC1.gateRows)}）拆成两张；TRT-LLM 侧 act(fc)·gate 是它的公开契约`, 'rank'));
  const BF = 36;
  FC1.fcRows.forEach((r, i) => {
    const x = X0 + i * BF;
    parts.push(rect(x, BY3 + 2, BF - 2, CH, 'g0', 2));
    parts.push(text(x + (BF - 2) / 2, BY3 + 14, 'fc', 'sm', 'middle'));
    parts.push(text(x + (BF - 2) / 2, BY3 + 26, `r${r}`, 'sm', 'middle'));
  });
  FC1.gateRows.forEach((r, i) => {
    const x = X0 + c.ffn * BF + 30 + i * BF;
    parts.push(rect(x, BY3 + 2, BF - 2, CH, 'g1', 2));
    parts.push(text(x + (BF - 2) / 2, BY3 + 14, 'gate', 'sm', 'middle'));
    parts.push(text(x + (BF - 2) / 2, BY3 + 26, `r${r}`, 'sm', 'middle'));
  });
  const capB = `解交错后 q 按头连续、k / v 按 kv 头连续；rep=${QKV.rep} 份复制多出 ${CONS.kvDupExtra} 个元素（${c.layers} 层），pad 多出 ${CONS.padExtra} 个；lm_head 在这一步不 pad`;
  parts.push(text(X0, BY3 + CH + 20, guard(capB, 11, W - 2 * X0, 'fig1/capB'), 'cap'));

  // ---- 面板 C：逐 rank 切片 ----
  const CY = BY3 + CH + 50;
  parts.push(text(X0, CY, `C　get_local_model_weights_per_gpu：${c.tp} 个 TP rank × ${c.pp} 个 PP rank 各拿什么（行号 = 面板 A 的源行；列号 = 输入列）`, 'pt'));
  const colW = 278;
  const rowH = 118;
  const CX = X0 + 92;
  const CY1 = CY + 10;
  for (let tp = 0; tp < c.tp; tp += 1) parts.push(text(CX + tp * colW + colW / 2 - 4, CY1 + 14, `tp rank ${tp}`, 'rank', 'middle'));
  for (let pp = 0; pp < c.pp; pp += 1) {
    const y = CY1 + 22 + pp * rowH;
    const layers = ppLayers(c.layers, c.pp, pp);
    parts.push(text(X0, y + 16, `pp rank ${pp}`, 'rank'));
    parts.push(text(X0, y + 30, `layers ${rangeStr(layers)}`, 'sm'));
    parts.push(text(X0, y + 44, `→ 本地 0–${layers.length - 1}`, 'sm'));
    for (let tp = 0; tp < c.tp; tp += 1) {
      const r = LOCAL.ranks[pp * c.tp + tp];
      const x = CX + tp * colW;
      parts.push(rect(x - 4, y, colW - 8, rowH - 8, `r${tp}`, 6));
      const lines = [
        { t: `qkv 行 q ${rangeStr(r.qkv.q)} | k ${rangeStr(r.qkv.k)} | v ${rangeStr(r.qkv.v)}`, cls: 'dim' },
        { t: `dense 列 ${rangeStr(r.denseCols)}　proj 列 ${rangeStr(r.projCols)}`, cls: 'sm' },
        { t: `fc 行 ${rangeStr(r.fc)}　gate 行 ${rangeStr(r.gate)}`, cls: 'sm' },
        { t: 'layernorm ×2：整份复制', cls: 'sm' },
      ];
      if (r.emb) lines.push({ t: `embedding 行 ${rangeStr(r.emb)}${r.emb.some((i) => i >= c.vocab) ? `（含 ${r.emb.filter((i) => i >= c.vocab).length} 行 pad）` : ''}`, cls: r.emb.some((i) => i >= c.vocab) ? 'costtx' : 'sm' });
      if (r.lmHead) lines.push({ t: `lm_head 行 ${rangeStr(r.lmHead)}（${r.lmHead.length} 行，不 pad）+ ln_f`, cls: r.lmHead.length !== r.lmHead.length ? 'sm' : 'sm' });
      lines.push({ t: `本地 ${r.numel} 个元素`, cls: 'rank' });
      lines.forEach((l, k) => parts.push(text(x + 4, y + 16 + k * 15, guard(l.t, 10.5, colW - 16, `fig1/C/${pp}/${tp}/${k}`), l.cls)));
    }
  }
  const CY2 = CY1 + 22 + c.pp * rowH + 4;
  const capC1 = `列并行（qkv / fc / gate / embedding）切行，行并行（dense / proj）切列；k、v 行在相邻两个 rank 上是同一份（蓝）；layernorm 每个 rank 整份`;
  const capC2 = `lm_head 走 torch.chunk(${c.vocab}, ${c.tp}) = ${VOCAB.lmHeadChunkSizes.join(',')}，与 config.vocab_size=${VOCAB.vocabSizePadded} 不一致，TRT-LLM 侧是否接受本仓不能证明`;
  parts.push(text(X0, CY2 + 10, guard(capC1, 11, W - 2 * X0, 'fig1/capC1'), 'cap'));
  parts.push(text(X0, CY2 + 26, guard(capC2, 11, W - 2 * X0, 'fig1/capC2'), 'cap'));

  // 每 rank 张量形状表
  const TY = CY2 + 44;
  parts.push(text(X0, TY, '每 rank 张量形状表（脚本按 _convert_transformer_layer / get_local_model_weights_per_gpu 算出）', 'rank'));
  const tcols = [X0, X0 + 200, X0 + 380, X0 + 560, X0 + 790, X0 + 1000];
  const theads = ['张量（TRT-LLM 名）', '训练态', 'TRT-LLM 全局', '每 rank 本地', '切分', '来源'];
  const TY1 = TY + 10;
  const trows = [
    ['attention.qkv.weight', shapeStr([qkvRows.length, c.hidden]), shapeStr([globalQkv.length, c.hidden]), shapeStr([QKV.rowsPerRank, c.hidden]), '列并行：切行', 'linear_qkv 解交错 + KV expand'],
    ['attention.dense.weight', shapeStr([c.hidden, c.heads * hn]), shapeStr([c.hidden, c.heads * hn]), shapeStr([c.hidden, LOCAL.denseCols[0].length]), '行并行：切列', 'linear_proj'],
    ['mlp.fc.weight / mlp.gate.weight', shapeStr([2 * c.ffn, c.hidden]), `${shapeStr([c.ffn, c.hidden])} × 2`, `${shapeStr([FC1.fc[0].length, c.hidden])} × 2`, '列并行：切行', 'linear_fc1 前半 / 后半'],
    ['mlp.proj.weight', shapeStr([c.hidden, c.ffn]), shapeStr([c.hidden, c.ffn]), shapeStr([c.hidden, LOCAL.projCols[0].length]), '行并行：切列', 'linear_fc2'],
    ['vocab_embedding.weight', shapeStr([c.vocab, c.hidden]), shapeStr([VOCAB.embRows, c.hidden]), shapeStr([VOCAB.embPerRank[0].length, c.hidden]), '切行（仅首 PP rank）', 'word_embeddings + pad'],
    ['lm_head.weight', shapeStr([c.vocab, c.hidden]), shapeStr([c.vocab, c.hidden]), `${VOCAB.lmHeadChunkSizes.map((n) => shapeStr([n, c.hidden])).join(' / ')}`, '切行，不 pad（仅末 PP rank）', 'output_layer'],
  ];
  parts.push(rect(X0 - 4, TY1, W - 2 * X0 + 8, 22 * (trows.length + 1) + 8, 'panel'));
  theads.forEach((h, i) => parts.push(text(tcols[i], TY1 + 17, h, 'rank')));
  parts.push(line(X0, TY1 + 23, W - X0 - 4, TY1 + 23));
  trows.forEach((r, k) => {
    const y = TY1 + 40 + k * 22;
    r.forEach((cell, i) => parts.push(text(tcols[i], y, guard(cell, 10.5, (tcols[i + 1] ?? W - X0) - tcols[i] - 8, `fig1/T/${k}/${i}`), i === 3 ? 'dim' : 'sm')));
  });

  // ---- 面板 D：FP8 三跳 + 分布式 lane ----
  const DY = TY1 + 22 * (trows.length + 1) + 34;
  parts.push(text(X0, DY, 'D　FP8 时 scale 的三跳（fp8_quantized=True；ghost = 被过滤的 extra_state）', 'pt'));
  const DX = X0;
  const gapX = 28;
  const bw = (W - 2 * X0 - 3 * gapX) / 4;
  const bh = 92;
  const DY1 = DY + 12;
  const boxes = [
    { title: '① state dict 里的 _extra_state', cls: 'ghost', lines: [`${FP8.extraStateKeys} 个键（${c.layers} 层 × 5，TE 模块各带一个）`, 'TE 序列化的 BytesIO', 'core_attention 的那个被跳过'] },
    { title: '② _load_scaling_factors', cls: 'neutral', lines: [`取 ${FP8.usedKeys} 个：改成 .weight 键再重命名`, 'torch.load → scale_fwd / scale_inv_fwd', '[0] 激活、[1] 权重'] },
    { title: '③ 入口过滤', cls: 'ghost', lines: [`删掉全部 ${FP8.filteredKeys} 个 extra_state 键`, 'rename 时再删 adapter_layer', '被过滤 ≠ 推理不需要'] },
    { title: '④ 注入输出字典', cls: 'acc1', lines: [`_cast_value：× scale_fwd 再转 e4m3`, `_add_scales_to_converter：${FP8.injectedScales} 个 scale`, `fp8_kvcache 再加 ${FP8.kvScales} 个 = 1.0`] },
  ];
  boxes.forEach((b, i) => {
    const x = DX + i * (bw + gapX);
    parts.push(infoBox(x, DY1, bw, bh, b.title, b.lines, b.cls, `fig1/D/${i}`));
    if (i < boxes.length - 1) parts.push(line(x + bw + 4, DY1 + bh / 2, x + bw + gapX - 6, DY1 + bh / 2, i === 2 ? 'main' : 'aux'));
  });
  // 分布式 lane：同一算例上的对照
  const EY = DY1 + bh + 26;
  parts.push(text(X0, EY, `E　分布式路径（on_device_distributed_conversion=True）在同一算例上`, 'pt'));
  const EY1 = EY + 12;
  const ew = (W - 2 * X0 - gapX) / 2;
  parts.push(
    infoBox(X0, EY1, ew, 76, `TP=${DIST_SAME.tp}（算例的目标拓扑）：两处失效`, [
      { text: `ng // tp = ${DIST_SAME.groupsPerRank} → qkv reshape(hidden, 0, q_num+2, hn) 非法；这条路不做 KV 复制`, cls: 'costtx' },
      { text: `divide(${c.vocab}, ${DIST_SAME.tp}) 断言失败：去 pad 后再切片要求 tokenizer vocab 整除 TP`, cls: 'costtx' },
    ], 'neutral', 'fig1/E/same'),
  );
  parts.push(
    infoBox(X0 + ew + gapX, EY1, ew, 76, `TP=${DIST_TP2.tp} 才可行：只改名 + 去训练 pad`, [
      `训练 pad ${DIST_TP2.trainingPadded} 行 → 每 rank ${DIST_TP2.localEmbRows} 行 → all_reduce 拼回 → 截到 ${c.vocab} → 每 rank ${DIST_TP2.unpaddedPerRank} 行`,
      `qkv 每 rank ${DIST_TP2.localQkvRows} 行：单 group 内 q|k|v 拼接，不跨 rank 重排`,
    ], 'neutral', 'fig1/E/tp2'),
  );

  // ---- 底部盒子（全宽、上下两块）----
  const OY = EY1 + 76 + 20;
  const OW = W - 2 * X0;
  parts.push(
    infoBox(X0, OY, OW, 126, '守恒账与代价（脚本算出）', [
      `Σ rank 本地 ${CONS.sumLocal} = 全局 ${CONS.globalElems} + KV 复制 ${CONS.kvDupExtra} + layernorm 复制 ${CONS.normReplicaExtra} + vocab pad ${CONS.padExtra}`,
      { text: `KV 复制每层多 (tp − ng)·2·hn·hidden = ${CONS.kvDupExtra / c.layers} 个元素；pad 多 ${CONS.padExtra} 个；bf16 各 ${CONS.kvDupBytes} B / ${CONS.padBytes} B`, cls: 'costtx' },
      `单设备路径峰值上界 = 输入 ${CONS.globalElems} + converter 全量字典 ${CONS.convertedElems} = ${CONS.peakElems} 个元素（${CONS.peakBytes} B）；逐 rank 字典是全量字典的视图`,
      `PP 切层：Mapping.pp_layers(${c.layers}) 按 pp_size=${c.pp} 均分，本地层号减去起点；embedding 只在首 PP rank，lm_head / ln_f 只在末 PP rank`,
      '分布式路径不复制 KV、不 pad，只改名 + 去训练 pad + 单 group 内 q|k|v 拼接',
    ], 'neutral', 'fig1/rules'),
  );
  const OY2 = OY + 126 + 14;
  parts.push(
    infoBox(X0, OY2, OW, 110, '本图复刻了什么 / 依赖边界', [
      '精确复刻：_convert_transformer_layer 的 qkv / fc / dense / proj 四种分支、_duplicate_kv_head 的 expand、convert 的 vocab pad、get_padded_vocab_size、get_local_model_weights_per_gpu 的 _split 与 PP 重编号',
      { text: '外部契约（本仓测试用 mocker.patch 替代）：tensorrt_llm._utils.pad_vocab_size = ceil(v / tp) · tp；tensorrt_llm.Mapping.pp_layers 按 pp_size 均分', cls: 'costtx' },
      '分布式 lane：_get_remove_vocab_padding 的 all_reduce 拼接 + VocabUtility 切片；训练侧 pad 用 vocab_utils 的 ceil(v / (128 · tp)) · 128 · tp',
      '简化：不画 bias、position_embedding、MoE 的 expert_split，也不画 layernorm1p 的 +1',
    ], 'neutral', 'fig1/scope'),
  );
  const H = OY2 + 110 + 20;
  return seal(parts, W, H, 'fig1-layout');
}

// ============================================================================
// 输出
// ============================================================================

const FIGURES = Object.freeze({
  'megatron_trtllm_export_layout.svg': renderLayout,
});

export {
  CASE, QKV, FC1, VOCAB, LOCAL, CONS, DIST_SAME, DIST_TP2, FP8,
  torchChunkSizes, torchChunk, padVocabSizeTrtllm, ppLayers, trainingPaddedVocab,
  megatronQkvRows, singleDeviceQkv, singleDeviceFc1, singleDeviceRowParallel, singleDeviceVocab,
  localWeights, conservation, distributedLane, fp8Hops, FIGURES,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir =
    process.argv[2] ??
    join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, render] of Object.entries(FIGURES)) {
    const svg = render();
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`wrote ${join(outDir, name)}`);
  }
}
