// 锁住 DSv4 单卡执行面图示的可执行契约：CSA 索引规则、参数账本、kernel 分派谓词与 Sinkhorn
// 收敛，全部由复刻自冻结基线（NVIDIA/Megatron-LM@85902ef）的规则算出，并与
// 34_deepseek_v4_tensor_parallel_analysis.md 正文引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_dsv4_tp_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, FIG1, FIG2, FIG3, FIG3_LAYERS, MHC, SINKHORN,
  windowTopkIdxs, compressTopkIdxs, compressorRows, indexerTopkWithOffset, rowsPerQuery,
  buildCuSeqlensKvFull, catPerSegmentOrder, compressTopkIdxsThdRow, ledgerFor, mhcLedger,
  useFusedDsaKernels, dispatch, sinkhornIterations, sinkhornDeviation, fmt, pct,
} from '../megatron_dsv4_tp_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_dsv4_tp_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '34_deepseek_v4_tensor_parallel_analysis.md',
);

const NAMES = [
  'megatron_dsv4_tp_csa_index.svg',
  'megatron_dsv4_tp_param_ledger.svg',
  'megatron_dsv4_tp_dispatch.svg',
];

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

function assertInsideCanvas(svg, name) {
  const { w, h } = parseCanvas(svg, name);
  assert.equal(w, 1272, `${name}: viewBox 宽度须为 1272`);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    assert.ok(Number(x) >= -2, `${name}: rect 左边越界 x=${x}`);
    assert.ok(Number(y) >= -2, `${name}: rect 上边越界 y=${y}`);
    assert.ok(Number(x) + Number(rw) <= w + 2, `${name}: rect 右边越界 ${x}+${rw} > ${w}`);
    assert.ok(Number(y) + Number(rh) <= h + 2, `${name}: rect 下边越界 ${y}+${rh} > ${h}`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w, `${name}: text x=${x} 越界`);
    assert.ok(Number(y) >= 0 && Number(y) <= h, `${name}: text y=${y} 越界`);
  }
}

// 文字盒之间不重叠、文字盒不出画布（越界不等于不重叠，两条都要）
function assertNoTextOverlap(svg, name) {
  const { w: canvasW, h: canvasH } = parseCanvas(svg, name);
  const boxes = [];
  for (const m of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const tw = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - tw / 2 : anchor === 'end' ? Number(xs) - tw : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w: tw, h: size * 1.06, raw });
  }
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
// (a) 复刻器对源码语义的锁定
// ============================================================================

test('滑窗索引：_get_window_topk_idxs_cached 的 clamp 与 -1（TestGetWindowTopkIdxs）', () => {
  const win = windowTopkIdxs(4, 16);
  assert.deepEqual(win[0], [0, -1, -1, -1], '第一个 query 只能看自己');
  assert.deepEqual(win[1], [0, 1, -1, -1]);
  assert.deepEqual(win[3], [0, 1, 2, 3], 'q3 起滑窗填满');
  assert.deepEqual(win[5], [2, 3, 4, 5]);
  assert.deepEqual(win[15], [12, 13, 14, 15]);
  // 窗口大于序列：全部有效位都 < seqlen（test_window_larger_than_seqlen）
  for (const row of windowTopkIdxs(8, 3)) for (const v of row) assert.ok(v < 3);
  assert.equal(FIG1.firstFullWindow, 3);
});

test('压缩索引：_get_compress_topk_idxs_cached 的因果可见集、offset 与 Compressor 的 sq < ratio → None', () => {
  assert.equal(compressorRows(3, 4), 0, 'TestCompressor::test_compressor_too_short_input');
  assert.equal(compressorRows(16, 4), 4);
  assert.equal(compressorRows(4096, 128), 32);
  assert.equal(compressorRows(64, 128), 0, 'H 层在 S < 128 时没有压缩行');
  const comp = compressTopkIdxs(4, 16, 16);
  assert.deepEqual(comp[2], [-1, -1, -1, -1], 'q2 一个压缩行都看不见');
  assert.deepEqual(comp[3], [16, -1, -1, -1], 'q3 起看得见 c0，且已加 offset=sq');
  assert.deepEqual(comp[15], [16, 17, 18, 19]);
  assert.equal(FIG1.firstCompVisible, 3);
  assert.equal(FIG1.nComp, 4);
  assert.equal(FIG1.offset, 16);
});

test('indexer top-k：causal mask → top-k → -inf→-1 → 校验 → +offset（图 1 的三行）', () => {
  const t = FIG1.topkIdxs;
  assert.deepEqual(t[1], [0, 1, -1, -1, -1, -1]);
  assert.deepEqual(t[5], [2, 3, 4, 5, 16, -1], 'q5 只看得见 c0，第二个 top-k 槽位是 -1');
  assert.deepEqual(t[15], [12, 13, 14, 15, 19, 17], '四个压缩行里选两个');
  assert.equal(FIG1.invalidTotal, 16, '全图 16 个 -1 槽位');
  assert.equal(FIG1.firstTopkSaturated, 11, 'q11 起可见压缩行 > top-k');
  assert.equal(FIG1.kvFullRows, 20);
  // 校验规则：选中的下标永远 < 该 query 的可见压缩行数
  t.forEach((row, i) => {
    const visible = Math.floor((i + 1) / 4);
    for (const v of row.slice(4)) if (v >= 0) assert.ok(v - 16 < visible, `q${i} 选中了不可见的压缩行 ${v}`);
  });
  // 无效分数（全 -inf）→ 全 -1
  assert.deepEqual(indexerTopkWithOffset([[1, 2, 3]], 4, 2, 10)[0], [-1, -1]);
});

test('规模面板：每 query 最多访问的 KV 行数（window=128, top-k=512）', () => {
  const s4k = FIG1.scale.find((s) => s.S === 4096);
  const s64k = FIG1.scale.find((s) => s.S === 65536);
  assert.deepEqual([s4k.dense, s4k.W, s4k.C, s4k.Cdense, s4k.H, s4k.comp4, s4k.comp128], [4096, 128, 640, 1152, 160, 1024, 32]);
  assert.deepEqual([s64k.dense, s64k.W, s64k.C, s64k.Cdense, s64k.H, s64k.comp4, s64k.comp128], [65536, 128, 640, 16512, 640, 16384, 512]);
  assert.equal(rowsPerQuery('C', 1 << 20, CFG), 640, 'C 层与 S 无关');
});

test('THD 拼接：build_cu_seqlens_kv_full / cat_per_segment / get_compress_topk_idxs_thd（两段 [8, 6]）', () => {
  const { cuKv, cuComp, cuFull, order, rows } = FIG1.thd;
  assert.deepEqual(cuKv, [0, 8, 14]);
  assert.deepEqual(cuComp, [0, 2, 3]);
  assert.deepEqual(cuFull, [0, 10, 17]);
  assert.deepEqual(buildCuSeqlensKvFull(cuKv, cuComp), cuFull);
  assert.deepEqual(order.slice(8, 10), ['c0', 'c1']);
  assert.deepEqual(order[16], 'c2');
  assert.equal(order.length, 17);
  assert.deepEqual(catPerSegmentOrder([0, 5], [0, 0]).order, ['t0', 't1', 't2', 't3', 't4'], '无压缩行时只有 kv');
  assert.deepEqual(rows[0].row, [8, 9], '段 0 末 query：压缩局部下标偏移 = seqlen_kv = 8');
  assert.deepEqual(rows[1].row, [6, -1], '段 1 只有一个压缩行');
  assert.deepEqual(compressTopkIdxsThdRow(2, 0, 4, cuKv, cuComp, 2), [-1, -1]);
});

test('参数账本：形状按冻结源码的构造调用相乘（gpt3_mcore_te_tp1_pp2_dsv4_hybrid_fused 算例）', () => {
  const C = FIG2.C;
  const shape = (path) => C.rows.find((r) => r.path === path).shape;
  assert.deepEqual(shape('linear_q_down_proj'), [192, 512]);
  assert.deepEqual(shape('linear_q_up_proj'), [128, 192]);
  assert.deepEqual(shape('linear_kv_proj'), [16, 512]);
  assert.deepEqual(shape('linear_proj'), [512, 8192]);
  assert.deepEqual(shape('linear_o_group_proj'), [8192, 16]);
  assert.deepEqual(shape('core.indexer.linear_wq_b'), [8192, 192]);
  assert.deepEqual(shape('core.indexer.linear_weights_proj'), [64, 512]);
  assert.deepEqual(shape('core.compressor.linear_wkv'), [32, 512]);
  assert.deepEqual(shape('core.compressor.ape'), [4, 32]);
  assert.deepEqual(shape('core.indexer.compressor.linear_wkv'), [256, 512]);
  assert.deepEqual(shape('core.indexer.compressor.ape'), [4, 256]);
  assert.equal(C.total, 6358504);
  assert.equal(C.proj, 4194304);
  assert.equal(pct(C.proj, C.total), 66);
  assert.equal(C.rows.find((r) => r.path === 'core.indexer.linear_wq_b').n, 1572864);
  assert.deepEqual([C.interface, C.interfaceTensors], [4227072, 3]);
  assert.deepEqual([C.duplicated, C.duplicatedTensors], [1998848, 7]);
  assert.deepEqual([C.fp32Params, C.fp32Tensors], [1160, 3]);
  assert.deepEqual([C.bf16OutParams, C.bf16OutTensors], [327680, 5]);
  assert.equal(FIG2.H.total, 4475112);
  assert.equal(FIG2.W.total, 4456664);
  assert.equal(FIG2.Cdense.total, 4489576);
  assert.equal(FIG2.H.rows.find((r) => r.path === 'core.compressor.linear_wkv').shape[0], 16, 'ratio 128 coff=1');
  assert.equal(FIG2.derivedQkHeadDim, 8, '__post_init__ 把 qk_head_dim / kv_lora_rank 改写成 v_head_dim - qk_pos_emb_head_dim');
  assert.equal(FIG2.flashC.total, 126098624);
  assert.equal(FIG2.flashC.proj, 33554432);
  assert.equal(pct(FIG2.flashC.proj, FIG2.flashC.total), 27);
  assert.equal(FIG2.flashC.rows.find((r) => r.path === 'core.indexer.linear_wq_b').n, 8388608);
  assert.equal(FIG2.flashC.bf16OutParams, 10747904);
  assert.throws(() => ledgerFor({ ...CFG, heads: 1, vHeadDim: 12 }, 4), /o_groups/, 'heads × v_head_dim 不能被 o_groups 整除');
  assert.deepEqual([MHC.out, MHC.mappingProj, MHC.mappingParams, MHC.alpha, MHC.bias], [24, [24, 2048], 49152, 3, 24]);
  assert.equal(mhcLedger(4096, 4).mappingParams, 24 * 4 * 4096);
});

test('分派谓词：use_fused_dsa_kernels 与 forward 的四条分支（图 3 矩阵）', () => {
  assert.equal(useFusedDsaKernels('unfused', 'cudnn'), false, 'attention_backend=unfused 关掉一切');
  assert.equal(useFusedDsaKernels('fused', 'none'), false);
  assert.equal(useFusedDsaKernels('fused', 'cudnn'), true);
  assert.equal(useFusedDsaKernels('auto', 'tilelang'), true);
  assert.throws(() => useFusedDsaKernels('fused', 'flash'), /dsa_kernel_backend/);
  const byKey = Object.fromEntries(FIG3_LAYERS.map((L, i) => [L.key, FIG3.cells[i].map((c) => c.path)]));
  assert.deepEqual(byKey.W, ['unfused·A', 'unfused·A', 'A', 'A']);
  assert.deepEqual(byKey.C, ['unfused·B', 'unfused·C', 'B', 'C']);
  assert.deepEqual(byKey.Cd, ['unfused·A', 'unfused·A', 'A', 'A']);
  assert.deepEqual(byKey.H, ['unfused·A', 'unfused·A', 'A', 'A']);
  assert.equal(FIG3.distinct.length, 6);
  assert.equal(FIG3.lossCells, 2);
  assert.equal(FIG3.unfusedOverride.path, 'unfused·B', '功能测试配置的重复键落到 unfused');
  assert.equal(FIG3.hcaShort.path, 'A');
  assert.match(FIG3.hcaShort.kernels, /^window/, 'S=64 的 H 层退化成纯滑窗');
  assert.equal(dispatch({ ratio: 4, dense: false, attentionBackend: 'fused', dsaKernelBackend: 'cudnn', training: true, seqlen: 3 }).path, 'A', 'sq < ratio 时 Compressor 返回 None');
});

test('Sinkhorn：_sinkhorn_iterations 先列归一，再 (行归一, 列归一)×(k−1)', () => {
  const t = Object.fromEntries(SINKHORN.trace.map((x) => [x.k, x]));
  assert.equal(t[1].rowDev.toExponential(2), '3.49e-2');
  assert.equal(t[1].colDev.toExponential(2), '2.09e-6', '第一步就是列归一，列和先对齐');
  assert.equal(t[5].rowDev.toExponential(2), '1.61e-3');
  assert.equal(t[20].rowDev.toExponential(2), '1.02e-6');
  assert.equal(t[20].colDev.toExponential(2), '1.00e-6');
  const M = sinkhornIterations(SINKHORN.logits, 20);
  const { rowDev, colDev } = sinkhornDeviation(M);
  assert.ok(rowDev < 1e-4 && colDev < 1e-4);
  for (const row of M) for (const v of row) assert.ok(v > 0 && v < 1);
  // 单位矩阵的 logits 放大后趋近置换矩阵
  const sharp = sinkhornIterations([[10, 0, 0], [0, 10, 0], [0, 0, 10]], 20);
  assert.ok(sharp[0][0] > 0.99 && sharp[1][1] > 0.99);
});

// ============================================================================
// (b)+(d) 重新生成与仓库 .svg 一致；图元不出画布、文字不重叠
// ============================================================================

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-dsv4-tp-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [index, ledger, dispatchSvg] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1 ----
  for (const s of ['topk_idxs[q1] =', 'topk_idxs[q5] =', 'topk_idxs[q15] =', 'offset = sq = 16',
    `cu_kv_full=[${FIG1.thd.cuFull.join(', ')}]`, `${fmt(16512)}`, `${fmt(1152)}`]) {
    assert.ok(index.includes(s), `图 1 缺 ${s}`);
  }
  assert.ok(index.includes(`共 ${FIG1.invalidTotal} 个 −1 槽位`), '图 1 缺 −1 计数');

  // ---- 图 2 ----
  for (const s of [fmt(FIG2.C.total), fmt(FIG2.C.proj), fmt(FIG2.C.bf16OutParams), fmt(FIG2.flashC.total), fmt(MHC.mappingParams), 'duplicated', 'mark_keep_in_fp32']) {
    assert.ok(ledger.includes(s), `图 2 缺 ${s}`);
  }

  // ---- 图 3 ----
  for (const s of ['unfused·B', 'unfused·C', '3.49e-2', '1.02e-6', 'attention_backend=unfused']) {
    assert.ok(dispatchSvg.includes(s), `图 3 缺 ${s}`);
  }

  const all = [index, ledger, dispatchSvg];
  NAMES.forEach((name, i) => {
    assert.ok(!all[i].includes('[['), `${name}: 图上不允许漏出 wikilink 语法`);
    assertInsideCanvas(all[i], name);
    assertNoTextOverlap(all[i], name);
  });

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      all[i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_dsv4_tp_figures.mjs`,
    );
  });
});

// ============================================================================
// (c) 页面正文引用的数值与图上一致
// ============================================================================

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  // 共用算例
  assert.ok(page.includes('S=16、window=4、ratio=4、top-k=2'));
  assert.ok(page.includes('`[0,4,128,4,128,4]`'));

  // 图 1：三行 topk_idxs、-1 计数、饱和点、规模面板、THD 拼接
  assert.ok(page.includes('`topk_idxs[q1] = [0, 1, -1, -1 | -1, -1]`'));
  assert.ok(page.includes('`topk_idxs[q5] = [2, 3, 4, 5 | 16, -1]`'));
  assert.ok(page.includes('`topk_idxs[q15] = [12, 13, 14, 15 | 19, 17]`'));
  assert.ok(page.includes(`共 ${FIG1.invalidTotal} 个 −1 槽位`));
  assert.ok(page.includes(`q${FIG1.firstFullWindow} 起滑窗填满`) && page.includes(`q${FIG1.firstTopkSaturated} 起可见压缩行 > top-k`));
  const s4k = FIG1.scale.find((s) => s.S === 4096);
  const s64k = FIG1.scale.find((s) => s.S === 65536);
  assert.ok(page.includes(`W ${s4k.W}、C ${s4k.C}（压缩行 ${fmt(s4k.comp4)} > top-k）、C dense ${fmt(s4k.Cdense)}、H ${s4k.H}`));
  assert.ok(page.includes(`C 仍是 ${s64k.C}、C dense 涨到 ${fmt(s64k.Cdense)}、H ${s64k.H}（压缩行 ${s64k.comp128}）`));
  assert.ok(page.includes(`钉死在 ${s4k.C} 行`));
  assert.ok(page.includes(`\`cu_seqlens_kv=[${FIG1.thd.cuKv.join(', ')}]\``));
  assert.ok(page.includes(`\`cu_seqlens_compressed=[${FIG1.thd.cuComp.join(', ')}]\``));
  assert.ok(page.includes(`\`cu_seqlens_kv_full=[${FIG1.thd.cuFull.join(', ')}]\``));
  assert.ok(page.includes(`\`[${FIG1.thd.rows[0].row.join(', ')}]\``) && page.includes(`\`[${FIG1.thd.rows[1].row.join(', ')}]\``));

  // 图 2：形状与合计
  for (const s of ['`[192, 512]`', '`[128, 192]`', '`[16, 512]`', '`[512, 8192]`', '`[8192, 16]`', '`linear_wq_b [8192, 192]`', '`linear_weights_proj [64, 512]`', '`[32, 512]`', '`ape [4, 32]`', '`[256, 512]`', '`ape [4, 256]`']) {
    assert.ok(page.includes(s), `正文缺形状 ${s}`);
  }
  const C = FIG2.C;
  assert.ok(page.includes(`C 层 ${fmt(C.total)} 个参数`));
  assert.ok(page.includes(`\`linear_proj\` ${fmt(C.proj)}（${pct(C.proj, C.total)}%）`));
  assert.ok(page.includes(`\`indexer.linear_wq_b\` ${fmt(1572864)}（${pct(1572864, C.total)}%）`));
  assert.ok(page.includes(`Column/Row 接口投影 ${C.interfaceTensors} 个张量 ${fmt(C.interface)} 参数`));
  assert.ok(page.includes(`duplicated ${C.duplicatedTensors} 个张量 ${fmt(C.duplicated)} 参数`));
  assert.ok(page.includes(`fp32 保持 ${C.fp32Tensors} 个张量 ${fmt(C.fp32Params)} 参数`));
  assert.ok(page.includes(`FP8 上下文外 BF16 ${C.bf16OutTensors} 个张量 ${fmt(C.bf16OutParams)} 参数`));
  assert.ok(page.includes(`H 层 ${fmt(FIG2.H.total)}`) && page.includes(`W 层 ${fmt(FIG2.W.total)}`) && page.includes(`C 层 ${fmt(FIG2.Cdense.total)}`));
  assert.ok(page.includes(`C 层 ${fmt(FIG2.flashC.total)} 个参数，\`linear_proj\` ${fmt(FIG2.flashC.proj)}（${pct(FIG2.flashC.proj, FIG2.flashC.total)}%），\`wq_b\` ${fmt(8388608)}`));
  assert.ok(page.includes(`BF16 参数 ${fmt(FIG2.flashC.bf16OutParams)}`));
  assert.ok(page.includes(`\`mapping_proj [${MHC.mappingProj.join(', ')}]\` = ${fmt(MHC.mappingParams)} 参数`));
  assert.ok(page.includes(`${fmt(C.bf16OutParams)} 个参数留在 FP8 之外`));
  assert.ok(page.includes(`= ${FIG2.derivedQkHeadDim}，配置里写的 16 与 64 不生效`));

  // 图 3：分派矩阵与 Sinkhorn
  assert.ok(page.includes(`共 ${FIG3.distinct.length} 种路径`));
  assert.ok(page.includes(`带 indexer loss 的格子共 ${FIG3.lossCells} 个`));
  assert.ok(page.includes('也走 unfused·B'));
  assert.ok(page.includes('S=64 的 cudnn·train 退化成纯滑窗的 Path A'));
  const t = Object.fromEntries(SINKHORN.trace.map((x) => [x.k, x]));
  assert.ok(page.includes(`1 次迭代后 max|行和 − 1| 是 ${t[1].rowDev.toExponential(2)}、max|列和 − 1| 是 ${t[1].colDev.toExponential(2)}`));
  assert.ok(page.includes(`5 次后行偏差 ${t[5].rowDev.toExponential(2)}`));
  assert.ok(page.includes(`20 次（配方值）后 ${t[20].rowDev.toExponential(2)} 与 ${t[20].colDev.toExponential(2)}`));

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
  assert.ok(!/[A-Za-z_/]+\.py:\d+/.test(page), '正文不得含 path:line 引用');
});
