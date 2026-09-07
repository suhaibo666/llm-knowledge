// 锁住 DSv4 THD CP 图示的可执行契约：边界宽度、定容压紧、rank-major 行号映射、索引降级、
// 前向 / 反向的集合通信顺序、真实配方的通信字节，全部由复刻自冻结基线
// （NVIDIA/Megatron-LM@85902ef）的规则算出，并与
// 35_deepseek_v4_context_parallel_analysis.md 正文引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_dsv4_cp_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, RECIPE, LAYOUT, DEMO, COST16, SWEEP, CP_LIMIT, fmtMiB,
  FUSED_BACKWARD_INTERNAL, BRANCH_ORDER, LAYER_BACKWARD,
  dComp, dWindow, compactCapacity, thdCpPositionIds, buildCpIndexerLayout, compactForward,
  seqToRankRow, buildAttentionIndices, visibleCompressed, demoTopk, replayBranchOrderTest,
  recipeCost, cumsum,
} from '../megatron_dsv4_cp_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_dsv4_cp_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '35_deepseek_v4_context_parallel_analysis.md',
);

const NAMES = [
  'megatron_dsv4_cp_layout.svg',
  'megatron_dsv4_cp_timing.svg',
  'megatron_dsv4_cp_cost.svg',
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

test('d_comp / d_window / c_cap 的整数规则（cp_utils.py）', () => {
  assert.equal(dComp(4), 8, 'ratio 4 → d_comp 8（overlap 压缩器要上一组）');
  assert.equal(dComp(128), 128);
  assert.equal(dComp(1), 0);
  assert.equal(dWindow(4, 4), 8);
  assert.equal(dWindow(128, 4), 128, '配方里 window 128 压过 d_comp 8');
  assert.equal(dWindow(128, 128), 128);
  assert.equal(compactCapacity(16, 4), 8, '(16+8)//4=6 对齐到 8');
  assert.equal(compactCapacity(4096, 4), 1032, '(4096+8)//4=1026 对齐到 8');
  assert.equal(compactCapacity(4096, 128), 33, '(4096+128)//128=33，ratio 128 的对齐是 1');
});

test('seq_to_rank_row：锁定 test_prepare_cp_compressor_input_builds_rank_row_map 的算例', () => {
  // 单条 32 token 序列，CP2，ratio 4，c_cap=8：rank1 的首个可见逻辑行是 2，所以它拥有的
  // 组 4..7 落在槽 2..5 → 物理行 10..13
  const { map, firstLogicalRows } = seqToRankRow({
    cu: [0, 32], cuComp: [0, 8], lLocal: 16, cpSize: 2, ratio: 4, cCap: 8,
  });
  assert.deepEqual(map, [0, 1, 2, 3, 10, 11, 12, 13]);
  assert.deepEqual(firstLogicalRows, [0, 2]);
});

test('_build_cp_indexer_layout：锁定 test_compute_cp_indexer_topk_passes_offsets_without_repacking_k', () => {
  const layout = buildCpIndexerLayout([0, 5, 13, 20], [0, 1, 3, 4], 7, 8);
  assert.deepEqual(layout.cuQTopk, [0, 0, 6, 8, 8]);
  assert.deepEqual(layout.cuKTopk, [0, 1, 3, 4, 4]);
  assert.deepEqual(layout.qCausalOffsets, [0, 2, 0, 0]);
});

test('_thd_cp_position_ids：越界的边界行映射到位置 0（test_apply_thd_cp_local_rope_maps_invalid_boundary_rows_to_position_zero）', () => {
  const positions = thdCpPositionIds([0, 4, 12], -2, 16);
  assert.deepEqual(positions, [0, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0]);
});

test('压紧内核的组枚举与所有权规则（test_composed_cp_layout_maps_every_index_and_gradient_to_its_source）', () => {
  const L = LAYOUT;
  assert.deepEqual(L.cu, [0, 18, 32]);
  assert.deepEqual(L.cuComp, [0, 4, 7]);
  assert.equal(L.lLocal, 16);
  const r0 = L.ranks[0];
  const r1 = L.ranks[1];
  assert.deepEqual(r0.compact.slots.map((s) => s.comp), [0, 1, 2, 3, -1, -1, -1, -1]);
  assert.deepEqual(r1.compact.slots.map((s) => s.comp), [2, 3, 0, 1, 2, -1, -1, -1]);
  assert.deepEqual(r1.compact.slots.map((s) => s.seq), [0, 0, 1, 1, 1, -1, -1, -1]);
  // rank1 的槽 0/1 全部来自边界（全局 8..15），槽 2 起来自本地
  assert.ok(r1.compact.slots[0].tokens.every((t) => t.from === 'boundary'));
  assert.ok(r1.compact.slots[1].tokens.every((t) => t.from === 'boundary'));
  assert.ok(r1.compact.slots[2].tokens.every((t) => t.from === 'local'));
  assert.deepEqual(r1.compact.slots[0].tokens.map((t) => t.src), [8, 9, 10, 11]);
  // 所有权 = 组末 token 所在 rank，等价于 composed 测试的 (first_token + ratio − 1) // local_rows
  assert.deepEqual(r1.owned, ['dup', 'dup', 'owned', 'owned', 'owned', 'pad', 'pad', 'pad']);
  for (const r of L.ranks) {
    r.compact.slots.forEach((s, i) => {
      if (s.comp < 0) return;
      const firstToken = L.cu[s.seq] + s.comp * CFG.ratio;
      const owner = Math.floor((firstToken + CFG.ratio - 1) / L.lLocal);
      assert.equal(r.owned[i] === 'owned', owner === r.rank);
    });
  }
  assert.deepEqual(L.rankRow.map, [0, 1, 2, 3, 10, 11, 12, -1]);
  assert.deepEqual([L.realCompressedRows, L.dupRows, L.padRows, L.droppedTail, L.gatheredRows], [7, 2, 7, 4, 16]);
  // 被映射到的物理行永远不是重算槽或 pad 槽
  for (const phys of L.rankRow.map) {
    if (phys < 0) continue;
    const r = L.ranks[Math.floor(phys / L.cCap)];
    assert.equal(r.owned[phys % L.cCap], 'owned');
  }
});

test('build_attention_indices 三种 index_mode 的降级（_build_attention_indices_kernel）', () => {
  const L = LAYOUT;
  const r1 = L.ranks[1];
  assert.equal(r1.selected.compressedBase, 24);
  const byQ = Object.fromEntries(DEMO.map((d) => [d.q, d]));
  assert.deepEqual(byQ[16].idx, [5, 6, 7, 8, 27, 26]);
  assert.equal(byQ[16].length, 6);
  assert.deepEqual(byQ[21].idx, [10, 11, 12, 13, 34, -1]);
  assert.equal(byQ[21].length, 5);
  assert.deepEqual(byQ[29].idx, [18, 19, 20, 21, 36, 35]);
  assert.equal(byQ[29].length, 6);
  // 可见块数 = min((pos+1)//ratio, 本序列压缩行数)
  assert.deepEqual(DEMO.map((d) => d.visible), [4, 1, 3]);
  assert.equal(visibleCompressed(L.cu, L.cuComp, 17, 4).visible, 4, 'seq0 尾巴 token 也能看全部 4 组');
  // index_mode 1：全部可见块，window 在前
  const all16 = r1.allVisible.rows[0];
  assert.deepEqual(all16.idx.slice(0, all16.length), [5, 6, 7, 8, 24, 25, 26, 27]);
  // index_mode 2：压缩块在前、window 在后，并返回 rank-major 行
  const loss16 = r1.lossLayout.rows[0];
  assert.deepEqual(loss16.idx, [27, 26, 5, 6, 7, 8]);
  assert.deepEqual(loss16.rankMajor, [3, 2]);
  // 序列首 token：window 只有自己，压缩块为空
  const q18 = r1.selected.rows[2];
  assert.deepEqual(q18.idx.slice(0, q18.length), [10]);
  // 跨序列不串：q21 的 window 不含 seq0 的 token
  assert.ok(byQ[21].windowGlobal.every((g) => g >= 18));
  // 单条本地 query 的 top-k 契约：不足补 -1
  assert.deepEqual(demoTopk(L.cu, L.cuComp, 16, 16, 4, 2)[5], [0, -1]);
});

test('反向顺序：锁定 TestCPCommunicationOverlap 的两组事件序列', () => {
  assert.deepEqual(replayBranchOrderTest(), [
    'fused', 'q_weight_branch', 'wait_compressed_kv', 'attention_kv_compressor', 'wait_indexer', 'indexer_compressor',
  ]);
  assert.deepEqual(BRANCH_ORDER, replayBranchOrderTest());
  assert.deepEqual(FUSED_BACKWARD_INTERNAL.slice(0, 3), ['sparse_attention_backward', 'launch_compressed_kv', 'launch_indexer']);
  const labels = LAYER_BACKWARD.events.map((e) => e.label);
  const at = (l) => labels.indexOf(l);
  assert.ok(at('稀疏注意力反向') < at('RS 压缩 KV 发起'));
  assert.ok(at('RS 压缩 KV 发起') < at('RS Indexer-K 发起'));
  assert.ok(at('Indexer Q / weights 投影反向') < at('wait RS-KV'), 'wait 之前先跑不依赖 RS 的分支');
  assert.ok(at('wait RS-KV') < at('attention-KV compressor 反向'));
  assert.ok(at('attention-KV compressor 反向') < at('wait RS-K'));
  assert.ok(at('wait RS-K') < at('Indexer-K compressor 反向'));
  assert.ok(at('Indexer-K compressor 反向') < at('压紧反向：散射到 local + boundary'));
  assert.equal(labels[labels.length - 1], 'P2P 边界梯度回送（阻塞）');
});

test('真实配方 THD64K / CP16 的通信字节', () => {
  assert.equal(COST16.lLocal, 4096);
  assert.equal(COST16.dw, 128);
  assert.equal(COST16.boundaryBytes, 1048576);
  const r4 = COST16.perRatio[4];
  const r128 = COST16.perRatio[128];
  assert.deepEqual([r4.cCap, r4.gatheredRows, r4.realRows, r4.padRows], [1032, 16512, 16384, 128]);
  assert.deepEqual([r128.cCap, r128.gatheredRows, r128.realRows, r128.padRows], [33, 528, 512, 16]);
  assert.equal(r4.kRecv, 15 * 1032 * 128 * 2);
  assert.equal(r4.kvRecv, 15 * 1032 * 512 * 2);
  assert.equal(r128.kRecv, 0, 'ratio 128 层没有 indexer');
  assert.equal(COST16.naiveRecv, 15 * 4096 * 512 * 2);
  assert.deepEqual([fmtMiB(COST16.boundaryBytes), fmtMiB(r4.kRecv), fmtMiB(r4.kvRecv), fmtMiB(r128.kvRecv), fmtMiB(COST16.naiveRecv)],
    ['1.00', '3.78', '15.12', '0.48', '60.00']);
  assert.equal(fmtMiB(r4.kRecv + r4.kvRecv), '18.90');
  // CP 扫描：接收量单调饱和，本地行按 1/CP 缩
  for (let i = 1; i < SWEEP.length; i += 1) {
    assert.ok(SWEEP[i].recvBytes > SWEEP[i - 1].recvBytes);
    assert.equal(SWEEP[i].lLocal * 2, SWEEP[i - 1].lLocal);
  }
  assert.deepEqual(SWEEP.map((s) => s.cp), [...RECIPE.cpSweep]);
  assert.deepEqual([fmtMiB(SWEEP[0].recvBytes), fmtMiB(SWEEP[3].recvBytes), fmtMiB(SWEEP[6].recvBytes)], ['10.01', '18.90', '21.08']);
  assert.equal(CP_LIMIT, 512, 'local_rows >= d_window：65536 / 128');
  assert.equal(recipeCost(RECIPE, 512).lLocal, 128);
});

// ============================================================================
// (b)+(d) 重新生成与仓库 .svg 一致；图元不出画布、文字不重叠
// ============================================================================

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-dsv4-cp-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [layout, timing, cost] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );
  const L = LAYOUT;

  // ---- 图 1 ----
  assert.ok(layout.includes(`seq_to_rank_row = [${L.rankRow.map.join(', ')}]`));
  assert.ok(layout.includes(`c_cap=${L.cCap}`));
  assert.ok(layout.includes(`compressed_base = d_window + l_local = ${L.ranks[1].selected.compressedBase}`));
  for (const d of DEMO) {
    assert.ok(layout.includes(`[${d.windowIdx.join(', ')}]`), `图 1 缺 query ${d.q} 的 window 下标`);
    assert.ok(layout.includes(`[${d.compIdx.join(', ')}]`), `图 1 缺 query ${d.q} 的压缩下标`);
  }
  assert.ok(layout.includes(`其中真实压缩行 ${L.realCompressedRows}、重算槽 ${L.dupRows}、容量补齐 ${L.padRows}`));
  assert.equal((layout.match(/>pad</g) ?? []).length, L.padRows * 2, 'pad 槽在面板 B 与面板 C 各画一次');
  assert.ok(!layout.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 2 ----
  for (const s of ['_forward_thd_cp', 'FusedCSAIndexerSparseAttnFromTopkFunc.backward', 'gather_from_sequence_parallel_region', 'AG-K·在飞', 'AG-KV·在飞', 'RS-KV·在飞', 'RS-K·在飞']) {
    assert.ok(timing.includes(s), `图 2 缺 ${s}`);
  }
  assert.ok(timing.includes(`${FUSED_BACKWARD_INTERNAL.slice(0, 3).join(' → ')}`));
  assert.ok(timing.includes(`${BRANCH_ORDER.slice(0, 3).join(' → ')}`));
  assert.ok(timing.includes('AG-K：AG Indexer-K，发起于 4 之后、wait 于 7 之前'));
  assert.ok(timing.includes('AG-KV：AG 压缩 KV，发起于 5 之后、wait 于 9 之前'));
  assert.ok(timing.includes('RS-KV：RS 压缩 KV 发起，发起于 1 之后、wait 于 4 之前'));
  assert.ok(timing.includes('RS-K：RS Indexer-K 发起，发起于 1 之后、wait 于 6 之前'));

  // ---- 图 3 ----
  const r4 = COST16.perRatio[4];
  assert.ok(cost.includes(`c_cap=${r4.cCap}，gather ${r4.gatheredRows} 行`));
  assert.ok(cost.includes(`c_cap=${COST16.perRatio[128].cCap}，gather ${COST16.perRatio[128].gatheredRows} 行`));
  for (const v of [fmtMiB(COST16.boundaryBytes), fmtMiB(r4.kRecv), fmtMiB(r4.kvRecv), fmtMiB(COST16.naiveRecv)]) {
    assert.ok(cost.includes(`>${v}<`), `图 3 缺柱顶数值 ${v}`);
  }
  assert.ok(cost.includes(`CP ≤ ${CP_LIMIT} 的硬上限`));

  const all = [layout, timing, cost];
  NAMES.forEach((name, i) => {
    assertInsideCanvas(all[i], name);
    assertNoTextOverlap(all[i], name);
  });

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      all[i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_dsv4_cp_figures.mjs`,
    );
  });
});

// ============================================================================
// (c) 页面正文引用的数值与图上一致
// ============================================================================

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');
  const L = LAYOUT;
  const r1 = L.ranks[1];

  // 共用算例
  assert.ok(page.includes(`CP=${CFG.cpSize}`));
  assert.ok(page.includes(`${L.total} 个 token`) && page.includes(`${CFG.seqLens.join('+')}`));
  assert.ok(page.includes(`csa_window_size=${CFG.windowSize}`));
  assert.ok(page.includes(`d_comp=${L.dc}`) && page.includes(`d_window=${L.dw}`));
  assert.ok(page.includes(`c_cap=${L.cCap}`));
  assert.ok(page.includes(`top-k=${CFG.topk}`));
  assert.ok(page.includes(`l_local=${L.lLocal}`));

  // 图 1
  assert.ok(page.includes(`seq_to_rank_row = [${L.rankRow.map.join(', ')}]`));
  assert.ok(page.includes(`compressed_base = ${r1.selected.compressedBase}`));
  const byQ = Object.fromEntries(DEMO.map((d) => [d.q, d]));
  assert.ok(page.includes(`[${byQ[16].idx.join(', ')}]`), 'query 16 的最终下标');
  assert.ok(page.includes(`[${byQ[29].idx.join(', ')}]`), 'query 29 的最终下标');
  assert.ok(page.includes(`[${byQ[21].idx.slice(0, byQ[21].length).join(', ')}]`) && page.includes(`topk_length=${byQ[21].length}`), 'query 21 只有 5 个有效下标');
  assert.ok(page.includes(`真实压缩行 ${L.realCompressedRows}`) && page.includes(`重算槽 ${L.dupRows}`) && page.includes(`容量补齐 ${L.padRows}`));
  assert.ok(page.includes(`${L.gatheredRows} 行缓冲`));
  assert.ok(page.includes(`共 ${L.dw + L.lLocal + L.gatheredRows} 行`), 'kv_full_thd 的总行数');
  assert.ok(page.includes(`尾巴共 ${L.droppedTail} 个 token`));
  assert.ok(page.includes(`[${L.rankRow.firstLogicalRows.join(', ')}]`), '每个 rank 的首个可见逻辑行');

  // 图 2
  assert.ok(page.includes(FUSED_BACKWARD_INTERNAL.slice(0, 3).join(' → ')));
  assert.ok(page.includes(BRANCH_ORDER.join(' → ')));

  // 图 3
  const r4 = COST16.perRatio[4];
  const r128 = COST16.perRatio[128];
  assert.ok(page.includes(`c_cap=${r4.cCap}`) && page.includes(`c_cap=${r128.cCap}`));
  assert.ok(page.includes(`${r4.gatheredRows} 行`) && page.includes(`补齐 ${r4.padRows} 行`));
  for (const v of [fmtMiB(COST16.boundaryBytes), fmtMiB(r4.kRecv), fmtMiB(r4.kvRecv), fmtMiB(r128.kvRecv), fmtMiB(COST16.naiveRecv), fmtMiB(r4.kRecv + r4.kvRecv)]) {
    assert.ok(page.includes(`${v} MiB`), `正文缺 ${v} MiB`);
  }
  assert.ok(page.includes(`${fmtMiB(SWEEP[0].recvBytes)} MiB`) && page.includes(`${fmtMiB(SWEEP[6].recvBytes)} MiB`), 'CP 扫描两端');
  assert.ok(page.includes(`CP ≤ ${CP_LIMIT}`));
  assert.ok(page.includes(`l_local=${COST16.lLocal}`));

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
  assert.ok(!/[A-Za-z_/]+\.py:\d+/.test(page), '正文不得含 path:line 引用');
});
