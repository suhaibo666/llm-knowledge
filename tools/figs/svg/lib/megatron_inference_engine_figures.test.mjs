// 锁住推理引擎图示的可执行契约：块级 KV cache 的分配/共享/驱逐、三条批处理路径的时序、
// CUDA graph 尺寸枚举，全部由复刻自冻结基线的算法算出，并与
// 31_megatron_inference_engine_analysis.md 正文引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_inference_engine_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, INVARIANT, NAIVE, PROMPT_TOKENS_PER_WAVE, FIG1_RZ, FIG1_LRU, FIG2, FIG3,
  KVBlockAllocator, EngineSim, computeBlockHashes, calculateCudaGraphTokenCounts,
  decodeGraphTokenCounts, paddingProfile,
} from '../megatron_inference_engine_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_inference_engine_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '31_megatron_inference_engine_analysis.md',
);

const NAMES = [
  'megatron_inference_kv_blocks.svg',
  'megatron_inference_batching.svg',
  'megatron_inference_cuda_graphs.svg',
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

test('块池不变量：usable = total−1，active = total−paused−1，dummy 是最后一块', () => {
  assert.equal(INVARIANT.totalCount, CFG.totalCount);
  assert.equal(INVARIANT.totalAvail, CFG.totalCount - 1);
  assert.equal(INVARIANT.activeCount, CFG.totalCount - CFG.pausedCount - 1);
  assert.equal(INVARIANT.dummyBlockIdx, CFG.totalCount - 1);
  const ctx = { activeRows: () => [], pausedRows: () => [], prefixCacheLruClock: 0 };
  // paused_count 必须严格小于 total_count − 1，否则 assert active_count >= 1
  assert.throws(() => new KVBlockAllocator(ctx, 4, 3), /active_count/);
  const kv = new KVBlockAllocator(ctx, 6, 1);
  // 块 id 是栈：从 block_bag 顶端弹出，dummy（最后一块）永远不被分配
  assert.deepEqual(kv.allocateMemoryBlocks(2), [3, 4]);
  assert.equal(kv.getTotalUsed(), 2);
  assert.equal(kv.allocateMemoryBlocks(4), null, '只剩 3 块可用');
  kv.releaseMemoryBlocks([3, 4]);
  assert.equal(kv.totalAvail, 5);
});

test('prefix caching：ref_zero 归零即回池，lru 留作 cached 并按 timestamp 驱逐', () => {
  const ctx = { activeRows: () => [], pausedRows: () => [], prefixCacheLruClock: 0 };
  const rz = new KVBlockAllocator(ctx, 6, 1, true, 'ref_zero');
  const ids = rz.allocateMemoryBlocks(2);
  rz.registerKvBlockHashes(ids, ['h0', 'h1']);
  assert.equal(rz.kvHashToBlockId.get('h0'), ids[0]);
  rz.releaseMemoryBlocks(ids);
  assert.equal(rz.kvHashToBlockId.size, 0, 'ref_zero：ref 归零即 deregister');
  assert.equal(rz.totalAvail, 5);
  assert.equal(rz.isMemoryAvailable(6), false, 'ref_zero 没有 evictable 这一层');

  const lru = new KVBlockAllocator(ctx, 6, 1, true, 'lru');
  const a = lru.allocateMemoryBlocks(2);
  lru.registerKvBlockHashes(a, ['h0', 'h1']);
  ctx.prefixCacheLruClock = 5;
  const b = lru.allocateMemoryBlocks(2);
  lru.registerKvBlockHashes(b, ['h2', 'h3']);
  lru.releaseMemoryBlocks([...a, ...b]);
  assert.equal(lru.totalAvail, 1, 'lru：登记过 hash 的块不回池');
  assert.equal(lru.getEvictableBlockCount(), 4);
  assert.equal(lru.isMemoryAvailable(3), true, 'free + evictable 一起算');
  const got = lru.allocateMemoryBlocks(3);
  assert.equal(lru.evictions, 2, '缺 2 块，驱逐 2 块');
  assert.ok(!lru.kvHashToBlockId.has('h0') && !lru.kvHashToBlockId.has('h1'), '最旧 timestamp 先出');
  assert.ok(lru.kvHashToBlockId.has('h2') && lru.kvHashToBlockId.has('h3'));
  assert.equal(got.length, 3);
});

test('hash 父链：同前缀同 hash，前缀不同则之后全部不同', () => {
  const h1 = computeBlockHashes([1, 2, 3, 4, 5, 6, 7, 8], 4);
  const h2 = computeBlockHashes([1, 2, 3, 4, 9, 9, 9, 9], 4);
  const h3 = computeBlockHashes([0, 2, 3, 4, 5, 6, 7, 8], 4);
  assert.equal(h1.length, 2);
  assert.equal(h1[0], h2[0]);
  assert.notEqual(h1[1], h2[1]);
  assert.notEqual(h1[1], h3[1], '第一块不同则第二块的父链也不同');
  assert.deepEqual(computeBlockHashes([1, 2, 3], 4), [], '不满一块没有 hash');
});

test('图 1：两波请求在 ref_zero / lru 下的命中、跳过与推迟', () => {
  const rz1 = FIG1_RZ.wave1;
  const rz2 = FIG1_RZ.wave2;
  const lru1 = FIG1_LRU.wave1;
  const lru2 = FIG1_LRU.wave2;
  assert.equal(PROMPT_TOKENS_PER_WAVE, 31);
  // 第一波：两种策略行为一致（还没有东西可以跨波复用）
  for (const w of [rz1, lru1]) {
    assert.equal(w.steps, 5);
    assert.equal(w.hits, 2, 'R1 命中 R0 的块 A，R3 命中 R1 的 A+B');
    assert.equal(w.blocksMatched, 3);
    assert.equal(w.tokensSkipped, 12);
    assert.equal(w.waits, 3, 'R1/R3 在 s1 推迟，R3 在 s2 再推迟一次');
    assert.equal(w.evictions, 0);
    assert.equal(w.peak, 6);
  }
  assert.equal(rz1.cachedAfter, 0, 'ref_zero 结束时没有 cached 块');
  assert.equal(lru1.cachedAfter, 3, 'lru 结束时 A、B、R3 的第三块留作 cached');
  // 第二波：ref_zero 重演第一波，lru 全部命中且无推迟
  assert.deepEqual(
    [rz2.hits, rz2.blocksMatched, rz2.tokensSkipped, rz2.waits],
    [rz1.hits, rz1.blocksMatched, rz1.tokensSkipped, rz1.waits],
  );
  assert.deepEqual([lru2.hits, lru2.blocksMatched, lru2.tokensSkipped, lru2.waits], [3, 6, 20, 0]);
  assert.equal(lru2.steps, 5);
  assert.equal(lru2.evictions, 0, '池够用就不驱逐');
  assert.equal(FIG1_RZ.wave2Start, 5);
  assert.equal(FIG1_LRU.wave2Start, 5);
  // lru 第二波四条同一步进入
  const s6 = FIG1_LRU.sim.trace[5];
  assert.equal(s6.active.length, 4);
  assert.equal(s6.stepTokens, 11, '2 + 2 + 3 + 4：R3 被钳到至少留 2 个 token');
  // 朴素预留的对照
  assert.deepEqual([NAIVE.perReq, NAIVE.reserved, NAIVE.usable, NAIVE.concurrent, NAIVE.usedBlocks, NAIVE.wastePct], [4, 16, 11, 2, 11, 31]);
});

test('图 1：块表可以跨越不连续的物理块，dummy 永远空着', () => {
  for (const sim of [FIG1_RZ.sim, FIG1_LRU.sim]) {
    for (const snap of sim.trace) {
      assert.equal(snap.blocks[sim.kv.dummyBlockIdx].owners.length, 0);
    }
    const r3 = sim.finished.find((r) => r.id === 'R3');
    const table = r3.blockTable.filter((b) => b !== undefined);
    const sorted = [...table].sort((a, b) => a - b);
    assert.ok(sorted.some((b, i) => i > 0 && b - sorted[i - 1] !== 1), 'R3 的块表不是物理连续的');
  }
});

test('图 2：定长批 / 连续批处理 / chunked 的步数、空转与 R3 进入时刻', () => {
  const st = FIG2.static;
  const co = FIG2.continuous;
  const ch = FIG2.chunked;
  assert.deepEqual([st.steps, st.idle, st.slotSteps, st.maxStepTokens, st.r3Admitted, st.walked], [11, 11, 33, 12, 10, 8]);
  assert.deepEqual([co.steps, co.idle, co.slotSteps, co.maxStepTokens, co.r3Admitted], [9, 15, 27, 12, 8]);
  assert.deepEqual([ch.steps, ch.idle, ch.slotSteps, ch.maxStepTokens, ch.r3Admitted], [6, 4, 18, 12, 4]);
  // 连续批处理与 chunked 每步 token 都不超预算；定长批第一步只 prefill 到最短 prompt
  for (const run of [co, ch]) run.columns.forEach((c) => assert.ok(c.stepTokens <= CFG.fig2MaxTokens));
  assert.equal(st.columns[0].stepTokens, 3 * 3);
  // chunk 切法：R1 先 6 再 4；R3 先 10（不留 1 token 尾巴）再 2
  const chunks = (id) => ch.columns.flatMap((c) => c.cells.filter((x) => x && x.id === id && x.kind === 'chunk').map((x) => x.tokens));
  assert.deepEqual(chunks('R1'), [6, 4]);
  assert.deepEqual(chunks('R3'), [10, 2]);
});

test('图 2：非 chunked 模式下队头装不下就整体停下（FIFO 硬规则）', () => {
  const sim = new EngineSim({
    blockSize: 4, totalCount: 64, pausedCount: 0, maxRequests: 3, maxTokens: 12,
    prefixCaching: false, policy: 'ref_zero', chunked: false,
  });
  for (const r of CFG.requests) sim.submit(r, '');
  sim.runStep();
  assert.deepEqual(sim.active.map((r) => r.id), ['R0'], 'R1 装不下，R2 被队头拦住');
  const chunked = new EngineSim({
    blockSize: 4, totalCount: 64, pausedCount: 0, maxRequests: 3, maxTokens: 12,
    prefixCaching: false, policy: 'ref_zero', chunked: true,
  });
  for (const r of CFG.requests) chunked.submit(r, '');
  chunked.runStep();
  assert.deepEqual(chunked.active.map((r) => r.id), ['R0', 'R1']);
  assert.equal(chunked.chunkedPrefillRequestId, 'R1');
  assert.equal(chunked.waiting[0].id, 'R1', 'chunked 请求钉在队头');
});

test('图 3：EXPONENTIAL 与 LINEAR 的枚举尺寸、图数与 padding', () => {
  assert.deepEqual(FIG3.expDefault.sizes, [64, 32, 16, 8, 4, 2]);
  assert.deepEqual(FIG3.expAuto.sizes, [64, 32, 16, 8, 4, 2]);
  assert.deepEqual(FIG3.linDefault.sizes, [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4]);
  assert.deepEqual(FIG3.linAuto.sizes, [64, 56, 48, 40, 32, 24, 16, 8, 4, 2]);
  assert.deepEqual([FIG3.expDefault.count, FIG3.expDefault.worstRelPct, FIG3.expDefault.meanRelPct], [6, 100, 35]);
  assert.deepEqual([FIG3.linDefault.count, FIG3.linDefault.worstRelPct, FIG3.linDefault.meanRelPct], [16, 300, 13]);
  assert.deepEqual([FIG3.expLarge.count, FIG3.expLarge.worstRelPct], [10, 100]);
  assert.deepEqual([FIG3.linLarge.count, FIG3.linLarge.worstRelPct, FIG3.linLarge.worst.n, FIG3.linLarge.worst.chosen], [16, 6300, 1, 64]);
  // docstring 的例子：tp=1，8 张图，max 128 → [128, 64, ..., 1]
  assert.deepEqual(calculateCudaGraphTokenCounts(1, 8, 128, 'exponential'), [128, 64, 32, 16, 8, 4, 2, 1]);
  // 自动模式补的最小尺寸：tp=4，spec=0 时 4 与 8 必须在
  const auto = decodeGraphTokenCounts(calculateCudaGraphTokenCounts(4, -1, 64, 'exponential'), 4, 0, 64, true);
  assert.ok(auto.includes(4) && auto.includes(8));
  // match_graph_config：选「≥ n」的最小图
  const p = paddingProfile([64, 32, 16], 64);
  assert.deepEqual(p.rows[16], { n: 17, chosen: 32, pad: 15 });
  assert.equal(p.rows[0].chosen, 16);
});

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-inference-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [kv, batching, graphs] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1 ----
  assert.ok(kv.includes(`total_count=${INVARIANT.totalCount} → total_avail=${INVARIANT.totalAvail}`));
  assert.ok(kv.includes(`active_count=${INVARIANT.activeCount}`));
  assert.ok(kv.includes(`块 ${INVARIANT.dummyBlockIdx}·dummy`));
  assert.ok(kv.includes(`跳过 ${FIG1_RZ.wave1.tokensSkipped}/${PROMPT_TOKENS_PER_WAVE} 个 prompt token`));
  assert.ok(kv.includes(`lru 第二波：命中 ${FIG1_LRU.wave2.hits} 条、匹配 ${FIG1_LRU.wave2.blocksMatched} 块、跳过 ${FIG1_LRU.wave2.tokensSkipped} 个`));
  assert.ok(kv.includes(`预留量的 ${NAIVE.wastePct}% 被浪费`));
  assert.equal((kv.match(/>ref 3</g) ?? []).length, 1, 'lru 第二波里块 A 被三条请求共享');
  const cachedCells = FIG1_LRU.sim.trace.reduce(
    (n, snap) => n + snap.blocks.filter((b) => b.owners.length === 0 && b.cached).length, 0,
  );
  assert.equal(cachedCells, 17, 'lru 面板里 ref=0 仍留在 hash 表的格数');
  assert.equal((kv.match(/>cached</g) ?? []).length, cachedCells, 'cached 格数与仿真一致');
  assert.ok(!kv.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 2 ----
  for (const s of ['schedule_non_chunked_prefill', 'schedule_chunked_prefill', 'TokenOverflowError']) {
    assert.ok(batching.includes(s), `图 2 缺 ${s}`);
  }
  assert.ok(batching.includes(`>${FIG2.static.idle}/${FIG2.static.slotSteps}<`));
  assert.ok(batching.includes(`>${FIG2.continuous.idle}/${FIG2.continuous.slotSteps}<`));
  assert.ok(batching.includes(`>${FIG2.chunked.idle}/${FIG2.chunked.slotSteps}<`));
  assert.ok(batching.includes(`共 ${FIG2.static.walked} 槽·步`));
  assert.equal((batching.match(/>R1·p</g) ?? []).length, 6, 'R1 逐 token 走 prompt 的格数');
  assert.ok(batching.includes('>R3·c10<') && batching.includes('>R3·c2<'));

  // ---- 图 3 ----
  assert.ok(graphs.includes(`枚举尺寸：[${FIG3.expDefault.sizes.join(', ')}]`));
  assert.ok(graphs.includes(`枚举尺寸：[${FIG3.linDefault.sizes.join(', ')}]`));
  assert.ok(graphs.includes(`图数 ${FIG3.expDefault.count}；最坏相对 padding ${FIG3.expDefault.worstRelPct}%`));
  assert.ok(graphs.includes(`图数 ${FIG3.linDefault.count}；最坏相对 padding ${FIG3.linDefault.worstRelPct}%`));
  assert.ok(graphs.includes(`LINEAR：${FIG3.linLarge.count} 张图，最坏 ${FIG3.linLarge.worstRelPct}%`));
  assert.ok(graphs.includes('match_graph_config'));

  const all = [kv, batching, graphs];
  NAMES.forEach((name, i) => {
    assertInsideCanvas(all[i], name);
    assertNoTextOverlap(all[i], name);
  });

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      all[i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_inference_engine_figures.mjs`,
    );
  });
});

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  // 共用算例
  assert.ok(page.includes(`block_size_tokens=${CFG.blockSize}`));
  assert.ok(page.includes('R0(6→3)、R1(10→2)、R2(3→5)、R3(12→2)'));
  assert.ok(page.includes(`total_count=${INVARIANT.totalCount}`));
  assert.ok(page.includes(`paused_count=${INVARIANT.pausedCount}`));

  // 图 1
  assert.ok(page.includes(`可用块 = ${INVARIANT.totalAvail}`), '不变量 usable = total−1');
  assert.ok(page.includes(`active_count = ${INVARIANT.activeCount}`), '不变量 active = total−paused−1');
  assert.ok(page.includes(`dummy_block_idx = ${INVARIANT.dummyBlockIdx}`));
  assert.ok(page.includes(`峰值只占 ${FIG1_RZ.wave1.peak} 块`));
  assert.ok(page.includes(`跳过 ${FIG1_RZ.wave1.tokensSkipped}/${PROMPT_TOKENS_PER_WAVE} 个 prompt token`));
  assert.ok(page.includes(`推迟 ${FIG1_RZ.wave1.waits} 次`));
  assert.ok(page.includes(`留下 ${FIG1_LRU.wave1.cachedAfter} 个 cached 块`));
  assert.ok(page.includes(`命中 ${FIG1_LRU.wave2.hits} 条、匹配 ${FIG1_LRU.wave2.blocksMatched} 块、跳过 ${FIG1_LRU.wave2.tokensSkipped} 个`));
  assert.ok(page.includes(`预留 ${NAIVE.reserved} 块`) && page.includes(`最多 ${NAIVE.concurrent} 条`) && page.includes(`${NAIVE.wastePct}%`));

  // 图 2
  const st = FIG2.static;
  const co = FIG2.continuous;
  const ch = FIG2.chunked;
  assert.ok(page.includes(`max_requests=${CFG.fig2MaxRequests}`) && page.includes(`max_tokens=${CFG.fig2MaxTokens}`));
  assert.ok(page.includes(`${st.steps} 步`) && page.includes(`${st.idle}/${st.slotSteps}`) && page.includes(`s${st.r3Admitted}`));
  assert.ok(page.includes(`${co.steps} 步`) && page.includes(`${co.idle}/${co.slotSteps}`) && page.includes(`s${co.r3Admitted}`));
  assert.ok(page.includes(`${ch.steps} 步`) && page.includes(`${ch.idle}/${ch.slotSteps}`) && page.includes(`s${ch.r3Admitted}`));
  assert.ok(page.includes(`${st.walked} 个槽·步`), '逐 token 走 prompt 的槽·步数');

  // 图 3
  assert.ok(page.includes(`[${FIG3.expDefault.sizes.join(', ')}]`));
  assert.ok(page.includes(`${FIG3.expDefault.count} 张图`) && page.includes(`${FIG3.linDefault.count} 张图`));
  assert.ok(page.includes(`最坏相对 padding ${FIG3.expDefault.worstRelPct}%`));
  assert.ok(page.includes(`${FIG3.linDefault.worstRelPct}%`));
  assert.ok(page.includes(`${FIG3.linLarge.worstRelPct}%`), '规模放大后 LINEAR 的最坏 padding');
  assert.ok(page.includes(`${FIG3.expLarge.count} 张图`));
  assert.ok(page.includes(`cuda_graph_max_tokens=${CFG.cudaGraphMaxTokens}`) && page.includes(`tp_size=${CFG.tpSize}`));

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
});
