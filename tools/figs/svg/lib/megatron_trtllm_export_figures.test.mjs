// 锁住 TensorRT-LLM 导出图示的可执行契约：单设备 converter 的 QKV 解交错 + KV 复制、gate/up 拆分、
// vocab pad、逐 rank 切片、守恒账、分布式 lane 的两处失效与 FP8 scale 的三跳计数，全部由复刻自冻结基线
// （NVIDIA/Megatron-LM@85902ef）的规则算出，并与 37_megatron_trtllm_export_analysis.md 正文引用的
// 数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_trtllm_export_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CASE, QKV, FC1, VOCAB, LOCAL, CONS, DIST_SAME, DIST_TP2, FP8,
  torchChunkSizes, torchChunk, padVocabSizeTrtllm, ppLayers, trainingPaddedVocab,
  megatronQkvRows, singleDeviceQkv, singleDeviceFc1, singleDeviceRowParallel, singleDeviceVocab,
  localWeights, conservation, distributedLane, fp8Hops,
} from '../megatron_trtllm_export_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_trtllm_export_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '37_megatron_trtllm_export_analysis.md',
);

const NAME = 'megatron_trtllm_export_layout.svg';

const FONT = { ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 };

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function parseCanvas(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(m, `${name}: SVG 必须声明 viewBox`);
  return { w: Number(m[1]), h: Number(m[2]) };
}

// (d) 图元不出画布：每个 <rect> 都落在 viewBox 内
function assertInsideCanvas(svg, name) {
  const { w, h } = parseCanvas(svg, name);
  assert.equal(w, 1272, `${name}: viewBox 宽度须为 1272`);
  let rects = 0;
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    rects += 1;
    assert.ok(Number(x) >= -1 && Number(y) >= -1, `${name}: rect 起点越界 (${x}, ${y})`);
    assert.ok(Number(x) + Number(rw) <= w + 1, `${name}: rect 右边越界 ${Number(x) + Number(rw)} > ${w}`);
    assert.ok(Number(y) + Number(rh) <= h + 1, `${name}: rect 下边越界 ${Number(y) + Number(rh)} > ${h}`);
  }
  assert.ok(rects > 100, `${name}: 图元数量异常（${rects}）`);
}

// (d) 文本包围盒不重叠（与生成器同一套字号表与估宽规则，独立重算一遍）
function assertNoTextOverlap(svg, name) {
  const { w: canvasW, h: canvasH } = parseCanvas(svg, name);
  const boxes = [];
  for (const hit of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = hit;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const tw = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - tw / 2 : anchor === 'end' ? Number(xs) - tw : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w: tw, h: size * 1.06, raw });
  }
  assert.ok(boxes.length > 200, `${name}: 文本数量异常（${boxes.length}）`);
  for (const b of boxes) {
    assert.ok(
      b.x >= -1 && b.y >= -1 && b.x + b.w <= canvasW + 1 && b.y + b.h <= canvasH + 1,
      `${name}: 文字盒出画布 "${b.raw}"`,
    );
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      assert.ok(!(dx > 1 && dy > 1), `${name}: 文字重叠 "${a.raw}" × "${b.raw}"`);
    }
  }
}

// ============================================================================
// (a) 复刻器锁定：在共用算例上逐格断言
// ============================================================================

test('torch.chunk 语义：块大小 ceil(n / k)，最后一块可短，块数可少于 k', () => {
  assert.deepEqual(torchChunkSizes(10, 4), [3, 3, 3, 1]);
  assert.deepEqual(torchChunkSizes(9, 4), [3, 3, 3]); // 第四个 rank 取不到块 → _split 的 [idx] 会 IndexError
  assert.deepEqual(torchChunkSizes(16, 4), [4, 4, 4, 4]);
  assert.deepEqual(torchChunkSizes(7, 1), [7]);
  assert.deepEqual(torchChunk([0, 1, 2, 3, 4], 2), [[0, 1, 2], [3, 4]]);
});

test('外部契约复刻：pad_vocab_size 与 Mapping.pp_layers、训练侧 padded vocab', () => {
  assert.equal(padVocabSizeTrtllm(10, 4), 12);
  assert.equal(padVocabSizeTrtllm(12, 4), 12);
  assert.equal(padVocabSizeTrtllm(100, 2), 100);
  assert.deepEqual(ppLayers(4, 2, 0), [0, 1]);
  assert.deepEqual(ppLayers(4, 2, 1), [2, 3]);
  assert.equal(trainingPaddedVocab(10, 128, 4), 512);
  assert.equal(trainingPaddedVocab(10, 128, 2), 256);
  assert.equal(trainingPaddedVocab(256, 128, 2), 256); // test_trtllm_distributed_gpu_converter 的 _VOCAB_SIZE
});

test('面板 A：linear_qkv 的行按 query group 交错（每 group：q × 2 头 + k + v，各 kv_channels 行）', () => {
  const rows = megatronQkvRows(CASE);
  assert.equal(rows.length, 16);
  const kinds = rows.map((r) => `${r.kind}${r.head}`);
  assert.deepEqual(kinds, [
    'q0', 'q0', 'q1', 'q1', 'k0', 'k0', 'v0', 'v0',
    'q2', 'q2', 'q3', 'q3', 'k1', 'k1', 'v1', 'v1',
  ]);
});

test('!3383 KV 复制：TP=4 > ng=2 时 rep=2，k / v 每 rank 一份、相邻两个 rank 同源（test_num_kv_heads_less_than_tp_size_valid）', () => {
  assert.equal(QKV.rep, 2);
  assert.equal(QKV.rowsPerRank, 6);
  assert.deepEqual(QKV.perRank, [
    { q: [0, 1], k: [4, 5], v: [6, 7] },
    { q: [2, 3], k: [4, 5], v: [6, 7] },
    { q: [8, 9], k: [12, 13], v: [14, 15] },
    { q: [10, 11], k: [12, 13], v: [14, 15] },
  ]);
  // ng=3、TP=4：tp % ng ≠ 0 → raise（test_num_kv_heads_less_than_tp_size_invalid）
  assert.throws(() => singleDeviceQkv({ ...CASE, heads: 6, groups: 3 }), /duplicate or split/);
  // ng=5 ≥ TP=4 但 ng % tp ≠ 0 → raise（test_num_kv_heads_greater_equal_tp_size_invalid）
  assert.throws(() => singleDeviceQkv({ ...CASE, heads: 10, groups: 5 }), /duplicate or split/);
  // ng=8 ≥ TP=4：不复制，每 rank 2 个 group（test_num_kv_heads_greater_equal_tp_size_valid）
  const wide = singleDeviceQkv({ ...CASE, heads: 8, groups: 8 });
  assert.equal(wide.rep, 1);
  assert.equal(wide.perRank[0].k.length, 4); // 2 group × hn 2
  assert.equal(new Set(wide.perRank.flatMap((r) => r.k)).size, 16); // 无复制：k 行两两不同
});

test('面板 B：gated linear_fc1 的前半给 TRT fc、后半给 TRT gate，再各按 TP 切行', () => {
  assert.deepEqual(FC1.fcRows, [...Array(16).keys()]);
  assert.deepEqual(FC1.gateRows, [...Array(16).keys()].map((i) => i + 16));
  assert.deepEqual(FC1.fc[1], [4, 5, 6, 7]);
  assert.deepEqual(FC1.gate[3], [28, 29, 30, 31]);
  assert.deepEqual(singleDeviceRowParallel(8, 4), [[0, 1], [2, 3], [4, 5], [6, 7]]);
  assert.deepEqual(singleDeviceRowParallel(16, 4)[2], [8, 9, 10, 11]);
});

test('vocab：embedding pad 到 pad_vocab_size(10, 4) = 12，lm_head 从不 pad、走 torch.chunk → 3,3,3,1', () => {
  assert.equal(VOCAB.embRows, 12);
  assert.equal(VOCAB.padRows, 2);
  assert.equal(VOCAB.vocabSizePadded, 12);
  assert.deepEqual(VOCAB.embPerRank, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]);
  assert.deepEqual(VOCAB.lmHeadChunkSizes, [3, 3, 3, 1]);
  // use_parallel_embedding=False：convert 不 pad，但 get_padded_vocab_size 仍因 lm_head 存在而报 12
  const noPar = singleDeviceVocab({ ...CASE, useParallelEmbedding: false });
  assert.equal(noPar.embRows, 10);
  assert.equal(noPar.vocabSizePadded, 12);
  assert.equal(noPar.embPerRank[3].length, 10); // 每个 rank 整张
  // 共享 embedding（无 output_layer）：config.vocab_size = embedding 行数，不再 pad
  const tied = singleDeviceVocab({ ...CASE, hasLmHead: false });
  assert.equal(tied.vocabSizePadded, 12);
  assert.equal(tied.lmHeadPerRank, null);
  const tiedNoPar = singleDeviceVocab({ ...CASE, hasLmHead: false, useParallelEmbedding: false });
  assert.equal(tiedNoPar.vocabSizePadded, 10);
});

test('面板 C：每个 (pp, tp) rank 的本地形状与元素数', () => {
  assert.equal(LOCAL.ranks.length, 8);
  const numels = LOCAL.ranks.map((r) => r.numel);
  assert.deepEqual(numels, [376, 376, 376, 376, 384, 384, 384, 368]);
  const r00 = LOCAL.ranks[0];
  assert.deepEqual(r00.layers, [0, 1]);
  const shape = (rank, name) => rank.entries.find((e) => e.name === name).shape;
  assert.deepEqual(shape(r00, 'layers.0.attention.qkv.weight'), [6, 8]);
  assert.deepEqual(shape(r00, 'layers.0.attention.dense.weight'), [8, 2]);
  assert.deepEqual(shape(r00, 'layers.0.mlp.fc.weight'), [4, 8]);
  assert.deepEqual(shape(r00, 'layers.0.mlp.gate.weight'), [4, 8]);
  assert.deepEqual(shape(r00, 'layers.0.mlp.proj.weight'), [8, 4]);
  assert.deepEqual(shape(r00, 'vocab_embedding.weight'), [3, 8]);
  assert.ok(!r00.entries.some((e) => e.name === 'lm_head.weight'), '首 PP rank 没有 lm_head');
  const r13 = LOCAL.ranks[7];
  assert.deepEqual(r13.layers, [2, 3]);
  assert.deepEqual(shape(r13, 'lm_head.weight'), [1, 8]);
  assert.deepEqual(shape(r13, 'ln_f.weight'), [8]);
  assert.ok(!r13.entries.some((e) => e.name === 'vocab_embedding.weight'), '末 PP rank 没有 embedding');
  // 本地层号从 0 重编号：末 PP rank 的字典里没有 layers.2 / layers.3
  assert.ok(r13.entries.every((e) => !/^layers\.[23]\./.test(e.name)));
});

test('守恒账：Σ rank 本地 = 全局 + KV 复制 + layernorm 复制 + vocab pad；峰值上界与字节', () => {
  assert.equal(CONS.layerGlobal, 592);
  assert.equal(CONS.globalElems, 2536);
  assert.equal(CONS.kvDupExtra, 256);
  assert.equal(CONS.kvDupExtra / CASE.layers, 64);
  assert.equal(CONS.normReplicaExtra, 216);
  assert.equal(CONS.padExtra, 16);
  assert.equal(CONS.sumLocal, 3024);
  assert.equal(CONS.sumLocal, CONS.globalElems + CONS.kvDupExtra + CONS.normReplicaExtra + CONS.padExtra);
  assert.equal(CONS.convertedElems, 3000);
  assert.equal(CONS.peakElems, 5536);
  assert.equal(CONS.globalBytes, 5072);
  assert.equal(CONS.peakBytes, 11072);
  assert.equal(CONS.kvDupBytes, 512);
  assert.equal(CONS.padBytes, 32);
  // 换一个不触发复制也不 pad 的拓扑：TP=2、vocab=10 → 仍 pad；TP=2、vocab=12 → 不 pad、不复制
  const c2 = { ...CASE, tp: 2, vocab: 12 };
  const q2 = singleDeviceQkv(c2);
  const v2 = singleDeviceVocab(c2);
  const l2 = localWeights(c2, q2, singleDeviceFc1(c2), v2);
  const k2 = conservation(c2, q2, v2, l2);
  assert.equal(q2.rep, 1);
  assert.equal(k2.kvDupExtra, 0);
  assert.equal(k2.padExtra, 0);
});

test('分布式 lane：同一算例 TP=4 两处失效，TP=2 只改名 + 去训练 pad', () => {
  assert.equal(DIST_SAME.groupsPerRank, 0);
  assert.equal(DIST_SAME.qkvReshapeOk, false);
  assert.equal(DIST_SAME.vocabDivisible, false);
  assert.equal(DIST_SAME.trainingPadded, 512);
  assert.equal(DIST_SAME.localEmbRows, 128);
  assert.equal(DIST_TP2.groupsPerRank, 1);
  assert.equal(DIST_TP2.trainingPadded, 256);
  assert.equal(DIST_TP2.localEmbRows, 128);
  assert.equal(DIST_TP2.unpaddedPerRank, 5);
  assert.equal(DIST_TP2.localQkvRows, 8);
  assert.equal(DIST_TP2.kvDuplication, false);
  // test_trtllm_distributed_gpu_converter：hidden 64、heads 2、ng 2、TP 2、vocab 256 → qkv 96 行、embedding 128 行
  const t = distributedLane({ ...CASE, hidden: 64, heads: 2, groups: 2, kvChannels: 32, vocab: 256, makeVocabSizeDivisibleBy: 128 }, 2);
  assert.equal(t.localQkvRows, 96);
  assert.equal(t.unpaddedPerRank, 128);
});

test('FP8 三跳计数：20 个 extra_state → 取 16 → 全删 → 注入 40 + 4 个 scale（非 gated 时每层 8 个，对齐 test_single_device_fp8 的 SCALING_FACTORS）', () => {
  assert.equal(FP8.extraStateKeys, 20);
  assert.equal(FP8.usedKeys, 16);
  assert.equal(FP8.filteredKeys, 20);
  assert.equal(FP8.injectedScales, 40);
  assert.equal(FP8.kvScales, 4);
  assert.equal(FP8.total, 44);
  const plain = fp8Hops(CASE, { gated: false, fp8Kvcache: false });
  assert.equal(plain.injectedScales / CASE.layers, 8);
  assert.equal(plain.kvScales, 0);
});

// ============================================================================
// (b) 生成器产出与仓内资产一致，且图元不出画布、文本不重叠
// ============================================================================

test('生成器产出图 1，图上的关键量与算例一致，且与仓内 SVG 逐字节相同', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-trtllm-export-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const svg = await readFile(join(outputDir, NAME), 'utf8');
  assert.ok(svg.includes(`rep=${QKV.rep}`) && svg.includes(`每 rank ${QKV.rowsPerRank} 行`));
  assert.ok(svg.includes(`pad_vocab_size(${CASE.vocab}, ${CASE.tp}) = ${VOCAB.embRows}`));
  assert.ok(svg.includes(`config.vocab_size = ${VOCAB.vocabSizePadded}`));
  assert.ok(svg.includes(`torch.chunk(${CASE.vocab}, ${CASE.tp}) = ${VOCAB.lmHeadChunkSizes.join(',')}`));
  for (const r of LOCAL.ranks) assert.ok(svg.includes(`本地 ${r.numel} 个元素`), `图上缺 rank (${r.pp}, ${r.tp}) 的元素数`);
  assert.ok(svg.includes(`Σ rank 本地 ${CONS.sumLocal} = 全局 ${CONS.globalElems} + KV 复制 ${CONS.kvDupExtra} + layernorm 复制 ${CONS.normReplicaExtra} + vocab pad ${CONS.padExtra}`));
  assert.ok(svg.includes(`= ${CONS.peakElems} 个元素（${CONS.peakBytes} B）`));
  assert.ok(svg.includes(`ng // tp = ${DIST_SAME.groupsPerRank}`) && svg.includes(`divide(${CASE.vocab}, ${DIST_SAME.tp})`));
  assert.ok(svg.includes(`每 rank ${DIST_TP2.unpaddedPerRank} 行`) && svg.includes(`qkv 每 rank ${DIST_TP2.localQkvRows} 行`));
  assert.ok(svg.includes(`${FP8.extraStateKeys} 个键`) && svg.includes(`${FP8.injectedScales} 个 scale`) && svg.includes(`再加 ${FP8.kvScales} 个`));
  // 复制的 k / v 格（acc1）= 每层 (tp − ng)·2 个 kv 头 × hn 行 = 8 格；pad 格（acc2）= 2
  assert.equal((svg.match(/class="acc1" x=/g) ?? []).length, (CASE.tp - CASE.groups) * 2 * CASE.kvChannels + 1, 'acc1 = 复制的 k/v 行 + 面板 D 的注入盒');
  assert.equal((svg.match(/>·</g) ?? []).length, VOCAB.padRows, 'pad 格数');
  assert.ok(!svg.includes('[['), '图上不允许漏出 wikilink 语法');

  assertInsideCanvas(svg, NAME);
  assertNoTextOverlap(svg, NAME);

  const tracked = await readFile(join(assetDir, NAME), 'utf8');
  assert.equal(tracked, svg, `${NAME} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_trtllm_export_figures.mjs`);
});

// ============================================================================
// (c) 页面正文引用的数值与图上一致
// ============================================================================

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  // 共用算例
  assert.ok(page.includes(`num_layers = ${CASE.layers}`) && page.includes(`hidden_size = ${CASE.hidden}`));
  assert.ok(page.includes(`num_attention_heads = ${CASE.heads}`) && page.includes(`num_query_groups = ${CASE.groups}`));
  assert.ok(page.includes(`kv_channels = ${CASE.kvChannels}`) && page.includes(`ffn_hidden_size = ${CASE.ffn}`));
  assert.ok(page.includes(`vocab = ${CASE.vocab}`));
  assert.ok(page.includes(`inference_tp_size = ${CASE.tp}`) && page.includes(`inference_pp_size = ${CASE.pp}`));

  // QKV 解交错与 KV 复制
  assert.ok(page.includes(`rep = ${QKV.rep}`), 'rep');
  assert.ok(page.includes(`每 rank ${QKV.rowsPerRank} 行`), 'qkv 每 rank 行数');
  assert.ok(page.includes(`(${16}, ${CASE.hidden})`) && page.includes(`(${QKV.rowsPerRank * CASE.tp}, ${CASE.hidden})`) && page.includes(`(${QKV.rowsPerRank}, ${CASE.hidden})`), 'qkv 三种形状');
  assert.ok(page.includes('q 0–1 | k 4–5 | v 6–7') && page.includes('q 2–3 | k 4–5 | v 6–7'), 'rank 0 / rank 1 共享同一份 k、v');

  // gate / up、dense / proj
  assert.ok(page.includes(`(${FC1.fc[0].length}, ${CASE.hidden})`) && page.includes(`(${CASE.hidden}, ${LOCAL.projCols[0].length})`) && page.includes(`(${CASE.hidden}, ${LOCAL.denseCols[0].length})`));

  // vocab pad 与 lm_head
  assert.ok(page.includes(`pad_vocab_size(${CASE.vocab}, ${CASE.tp}) = ${VOCAB.embRows}`), 'pad_vocab_size');
  assert.ok(page.includes(`${VOCAB.padRows} 行零`), 'pad 行数');
  assert.ok(page.includes(`config.vocab_size = ${VOCAB.vocabSizePadded}`), 'config.vocab_size');
  assert.ok(page.includes(`${VOCAB.lmHeadChunkSizes.join(',')}`), 'lm_head chunk');
  assert.ok(page.includes(`(${VOCAB.embPerRank[0].length}, ${CASE.hidden})`), 'embedding 每 rank');

  // 逐 rank 元素数与守恒账
  assert.ok(page.includes(`${LOCAL.ranks[0].numel} 个元素`) && page.includes(`${LOCAL.ranks[4].numel} 个元素`) && page.includes(`${LOCAL.ranks[7].numel} 个元素`), '三种本地元素数');
  assert.ok(page.includes(`${CONS.layerGlobal} 个元素`) && page.includes(`${CONS.globalElems} 个`), '每层 / 整模型元素数');
  assert.ok(page.includes(`${CONS.sumLocal}`) && page.includes(`KV 复制 ${CONS.kvDupExtra}`) && page.includes(`layernorm 复制 ${CONS.normReplicaExtra}`) && page.includes(`pad ${CONS.padExtra}`), '守恒账四项');
  assert.ok(page.includes(`${CONS.kvDupExtra / CASE.layers} 个元素`), '每层 KV 复制元素数');
  assert.ok(page.includes(`${CONS.kvDupBytes} B`) && page.includes(`${CONS.padBytes} B`), 'KV 复制 / pad 字节');
  assert.ok(page.includes(`${CONS.peakElems} 个元素`) && page.includes(`${CONS.peakBytes} B`) && page.includes(`${CONS.globalBytes} B`), '峰值上界');

  // 分布式 lane
  assert.ok(page.includes(`ng // tp = ${DIST_SAME.groupsPerRank}`) && page.includes(`divide(${CASE.vocab}, ${DIST_SAME.tp})`), '两处失效');
  assert.ok(page.includes(`${DIST_SAME.trainingPadded}`) && page.includes(`${DIST_TP2.trainingPadded}`), '训练侧 pad');
  assert.ok(page.includes(`每 rank ${DIST_TP2.localEmbRows} 行`) && page.includes(`每 rank ${DIST_TP2.unpaddedPerRank} 行`) && page.includes(`每 rank ${DIST_TP2.localQkvRows} 行`), 'TP=2 对照');

  // FP8 三跳
  assert.ok(page.includes(`${FP8.extraStateKeys} 个`) && page.includes(`${FP8.usedKeys} 个`) && page.includes(`${FP8.injectedScales} 个`) && page.includes(`${FP8.kvScales} 个`), 'FP8 计数');

  assert.ok(page.includes(`assets/${NAME}`), `正文没有引用 ${NAME}`);
  assert.ok(!/[A-Za-z_/]+\.py:\d+/.test(page), '正文不得含 path:line 引用');
  assert.ok(!/`:\d+/.test(page), '正文不得含 `:123 式行号引用');
  assert.ok(!page.includes('../'), '正文不得含 ../ 相对链接');
});
