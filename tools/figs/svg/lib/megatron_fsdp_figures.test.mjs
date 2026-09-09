// 锁住 Megatron-FSDP 图示的可执行契约：两代布局算法在 docstring 算例与差异算例上的逐格结果、
// hook 状态机仿真的 makespan 与峰值、llama3-8b 配方一个 unit 的参数量与字节账，全部由复刻自
// 冻结基线（NVIDIA/Megatron-LM@85902ef）的规则算出，并与 36_megatron_fsdp_analysis.md 正文
// 引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_fsdp_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  LAYOUT_CASE, DIFF_CASE, ORDER_CASE, SIM, RECIPE, DTYPES,
  LAYOUT, DIFF, ORDER, CONJUGATES,
  LANE_V1, LANE_V1_SAME_COMM, LANE_V2, LANE_RECOMPUTE, LANE_RECOMPUTE_NO_PREFETCH, AG_COMM,
  COST, UNIT, MODEL_PARAMS, fmtMiB, fmtGiB,
  numel, lcmChunkSizeFactor, v1BuildIndex, v2BuildLayout, localShapes, perParameterShard0,
  simulateV1, simulateV2, residentBytes, commBytes,
} from '../megatron_fsdp_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_fsdp_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '36_megatron_fsdp_analysis.md',
);

const NAMES = [
  'megatron_fsdp_layout.svg',
  'megatron_fsdp_timeline.svg',
  'megatron_fsdp_cost.svg',
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

test('GlobalLayout.build docstring 算例（test_compute_layout_fills_lcm_padding_gaps）', () => {
  assert.equal(LAYOUT.chunk, 12, 'LCM(6,4,4,2,6) = 12');
  assert.deepEqual(LAYOUT.v2.offsets, [0, 12, 32, 28, 48]);
  assert.equal(LAYOUT.v2.size, 60);
  // v1 build_data_parallel_buffer_index 在同一算例上给出同一组 offsets
  assert.deepEqual(LAYOUT.v1.offsets, [0, 12, 32, 28, 48]);
  assert.equal(LAYOUT.v1.size, 60);
  // 测试文件里逐 rank 锁定的本地形状
  const expected = [
    [[2, 6], [0, 4], [0, 4], [0, 2], [0, 6]],
    [[0, 6], [3, 4], [0, 4], [0, 2], [0, 6]],
    [[0, 6], [1, 4], [1, 4], [1, 2], [0, 6]],
    [[0, 6], [0, 4], [3, 4], [0, 2], [0, 6]],
    [[0, 6], [0, 4], [0, 4], [0, 2], [1, 6]],
  ];
  assert.deepEqual(LAYOUT.ranks.map((r) => r.local), expected);
  assert.deepEqual(LAYOUT.classes, ['regular', 'regular', 'regular', 'fragment', 'fragment']);
  assert.deepEqual(CONJUGATES, [[1, 2]], 'P1 余 4 与 P2 余 4 配对');
  assert.equal(LAYOUT.totalNumel, 52);
  assert.equal(LAYOUT.padCells, 8);
  // 每个 rank 恰好 12 个元素；本地元素数 = 12 − pad
  assert.deepEqual(
    LAYOUT.ranks.map((r) => r.local.reduce((acc, s) => acc + numel(s), 0)),
    [12, 12, 10, 12, 6],
  );
});

test('v1 / v2 在碎片对齐与装填顺序上的差异（test_dbuffer_layout_aligns_fragment_offsets_to_rows）', () => {
  assert.deepEqual(DIFF.v2.offsets, [0, 18]);
  assert.equal(DIFF.v2.size, 24);
  assert.deepEqual(DIFF.v1.offsets, [0, 16], 'v1 直接放在 gap_offset');
  assert.equal(DIFF.v1.size, 24);
  // v2 __post_init__ 的行宽对齐：18 % 6 == 0，16 % 6 != 0
  assert.equal(DIFF.v2.offsets[1] % 6, 0);
  assert.notEqual(DIFF.v1.offsets[1] % 6, 0);
  // 顺序小例：v1 按注册顺序填 gap（(1,2) 先），v2 大碎片优先 + 行宽对齐，导致 (1,2) 掉到尾部
  assert.deepEqual(ORDER.v1.offsets, [0, 16, 18]);
  assert.equal(ORDER.v1.size, 24);
  assert.deepEqual(ORDER.v2.offsets, [0, 24, 18]);
  assert.equal(ORDER.v2.size, 48);
  // test_dbuffer_layout_pads_to_lcm_times_dp_size_and_fills_gaps：(5,4)+(2,6)+(3,) DP=2 → (0,24,20)，size 48
  const t = v2BuildLayout([[5, 4], [2, 6], [3]], 2);
  assert.deepEqual(t.offsets, [0, 24, 20]);
  assert.equal(t.size, 48);
  const t1 = v1BuildIndex([[5, 4], [2, 6], [3]], 2, lcmChunkSizeFactor([[5, 4], [2, 6], [3]]));
  assert.deepEqual(t1.offsets, [0, 24, 20]);
  // no_shard 不 pad
  assert.equal(v1BuildIndex(LAYOUT_CASE.shapes, LAYOUT_CASE.dp, 12, true).size, 54);
  // get_local_tensor 的整行约束：故意给一个劈开行的布局要抛错
  assert.throws(() => localShapes([[4, 4]], [2], 24, 2), /split mid-row/);
});

test('FSDP2 式逐参数 Shard(0) 对照按 DTensor 公开契约推演', () => {
  const f = perParameterShard0(LAYOUT_CASE.shapes, LAYOUT_CASE.dp);
  assert.deepEqual(f.map((x) => x.perRank), [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]);
  assert.deepEqual(f.map((x) => x.emptyRanks), [3, 1, 1, 4, 4]);
});

test('hook 状态机仿真：v1 前向只暴露 AG0，反向只暴露 AG(L−1) 与最后一个 RS', () => {
  assert.equal(LANE_V1.fwdEnd, SIM.a + SIM.L * SIM.c);
  assert.equal(LANE_V1.fwdEnd, 18);
  assert.equal(LANE_V1.bwdEnd, 54);
  assert.equal(LANE_V1.peakParams, SIM.slots, '峰值在途参数桶 = 持久池槽数');
  assert.equal(LANE_V1.peakGrads, SIM.slots);
  const byId = Object.fromEntries(LANE_V1.tasks.map((t) => [t.id, t]));
  // 前向：AG(i+1) 在 F(i−1) 结束后才发起（all_gather_stream.wait_stream(current)）
  assert.equal(byId.AGf2.start, byId.C0.end);
  assert.equal(byId.AGf3.start, byId.C1.end);
  // 反向：AG(i−1) 在 pre_backward(i) 发起，等 B(i+1) 结束
  assert.equal(byId.AGb1.start, byId.B3.end);
  assert.equal(byId.AGb0.start, byId.B2.end);
  // RS(i) 紧跟 B(i)，与 B(i−1) 重叠
  assert.equal(byId.RS3.start, byId.B3.end);
  assert.ok(byId.RS3.end <= byId.B2.end);
  assert.equal(LANE_V1.bwdEnd, byId.RS0.end, 'finish_grad_sync 等最后一个 RS');
  // 槽数 3 时预取深度 2，前向 makespan 不变（AG 已被掩盖），峰值在途变 3
  const three = simulateV1({ ...SIM, slots: 3, separateAgComm: true });
  assert.equal(three.fwdEnd, LANE_V1.fwdEnd);
  assert.equal(three.peakParams, 3);
});

test('hook 状态机仿真：v2 前向靠 CPU run-ahead 重叠，反向完全串行', () => {
  assert.equal(LANE_V2.fwdEnd, SIM.s + SIM.a + SIM.L * SIM.c);
  assert.equal(LANE_V2.fwdEnd, 19);
  assert.equal(LANE_V2.bwdEnd - LANE_V2.fwdEnd, SIM.L * (SIM.a + 2 * SIM.c + SIM.r));
  assert.equal(LANE_V2.bwdEnd, 67);
  assert.equal(LANE_V2.peakParams, 2);
  assert.equal(LANE_V2.delayedQueueMax, 1);
  const byId = Object.fromEntries(LANE_V2.tasks.map((t) => [t.id, t]));
  // AG(i) 在 drain 释放 i−2 之后（wait_event(F(i−2))），且与 F(i−1) 重叠
  assert.equal(byId.AGf2.start, Math.max(byId.C0.end, byId.S2.end));
  assert.ok(byId.AGf2.start < byId.C1.end && byId.AGf2.end > byId.C1.start);
  // 反向 AG(L−1) 排在 root drain（wait_event(F(L−1)））之后
  assert.equal(byId.AGb3.start, byId.C3.end);
  // RS 在 compute 流上：B(i) → RS(i) → AG(i−1) → B(i−1)
  assert.equal(byId.RS3.lane, 'compute');
  assert.equal(byId.AGb2.start, byId.RS3.end);
  assert.equal(byId.B2.start, byId.AGb2.end);
});

test('独立 AG 进程组的阈值：共用 communicator 在 a ≥ 2c−r+1 时开始拉长反向', () => {
  assert.deepEqual(AG_COMM, { sameCommFirstLag: 7, separateFirstLag: 9, expectedSame: 7, expectedSeparate: 9 });
  assert.equal(LANE_V1_SAME_COMM.bwdEnd, LANE_V1.bwdEnd, '本例 a=2 时两者相同');
  const lag = simulateV1({ ...SIM, a: 7, separateAgComm: false });
  const noLag = simulateV1({ ...SIM, a: 7, separateAgComm: true });
  assert.ok(lag.bwdEnd > noLag.bwdEnd);
});

test('MXFP8 全量重计算：prefetch_recompute_forward_weights 的三连发把暴露的 rowwise AG 藏起来', () => {
  assert.equal(LANE_RECOMPUTE.bwdEnd, 72);
  assert.equal(LANE_RECOMPUTE_NO_PREFETCH.bwdEnd, 84);
  assert.equal(LANE_RECOMPUTE_NO_PREFETCH.bwdEnd - LANE_RECOMPUTE.bwdEnd, 12);
  assert.equal(LANE_RECOMPUTE.peakRowwise, 2);
  assert.equal(LANE_RECOMPUTE.peakColwise, 2);
  const byId = Object.fromEntries(LANE_RECOMPUTE.tasks.map((t) => [t.id, t]));
  // 三连发顺序：AGc(3) → AGr(3) → AGr(2) → AGc(2)
  assert.ok(byId.AGb3.end <= byId.AGr3.start && byId.AGr3.end <= byId.AGr2.start && byId.AGr2.end <= byId.AGb2.start);
  // 重算 R(i) 用 rowwise 桶，B(i) 用 columnwise 桶
  assert.equal(byId.R3.start, byId.AGr3.end);
  assert.ok(byId.B3.start >= byId.AGb3.end && byId.B3.start >= byId.R3.end);
});

test('llama3-8b 配方：一个 TransformerLayer unit 的参数量与字节账', () => {
  assert.equal(UNIT.q, 4096 * 4096);
  assert.equal(UNIT.kv, 2 * 4096 * 8 * 128);
  assert.equal(UNIT.fc1, 2 * 14336 * 4096);
  assert.equal(UNIT.total, 218112000);
  assert.equal(MODEL_PARAMS, 218112000 * 32 + 2 * 128256 * 4096);
  assert.equal(fmtMiB(COST.unitBytes.bf16), '416.0');
  assert.equal(fmtMiB(COST.unitBytes.fp8), '208.0');
  assert.deepEqual(COST.resident.bf16.map((r) => r.total), [16, 5.5, 3.75, 2, 2, 0.875]);
  assert.deepEqual(COST.resident.fp8.map((r) => r.total), [15, 4.5, 2.75, 1.875, 1.875, 0.75]);
  // HFSDP 公式 Opt/(D·O) + (G+W)/D（docs megatron_fsdp.md "Understanding Hybrid-FSDP"）
  const hf = residentBytes(DTYPES.bf16, { strategy: 'optim_grads_params', D: 8, O: 4, outer: 'optim' });
  assert.equal(hf.total, (4 + 8) / 32 + (2 + 2) / 8);
  assert.deepEqual(COST.transient.map((t) => [t.buffers, fmtMiB(t.bf16), fmtMiB(t.fp8), fmtMiB(t.mxfp8)]),
    [[2, '832.0', '416.0', '832.0'], [3, '1248.0', '624.0', '1248.0']]);
  const comm = COST.comm.bf16;
  assert.equal(comm[3].perMicrobatch, (2 * 2 + 2) * (7 / 8));
  assert.equal(comm[3].total.toFixed(2), '84.00');
  assert.equal(comm[4].perCycle.toFixed(3), '0.375');
  assert.equal(comm[5].perCycle.toFixed(3), '0.375');
  assert.equal(comm[0].total.toFixed(2), '3.50');
  assert.equal(comm[2].total.toFixed(2), '29.75');
  assert.equal(fmtGiB(MODEL_PARAMS * 2), '14.96');
  assert.equal(fmtGiB(MODEL_PARAMS * 16), '119.66');
  // 每 microbatch 通信只在 optim_grads / optim_grads_params 出现
  assert.equal(commBytes(DTYPES.bf16, { strategy: 'optim', D: 8, microbatches: 16 }).perMicrobatch, 0);
});

// ============================================================================
// (b)+(d) 重新生成与仓库 .svg 一致；图元不出画布、文字不重叠
// ============================================================================

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-fsdp-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [layout, timeline, cost] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1 ----
  assert.ok(layout.includes(`offsets = (${LAYOUT.v2.offsets.join(', ')})`));
  assert.ok(layout.includes(`size = pad(…, DP × LCM) = ${LAYOUT.v2.size}`));
  assert.ok(layout.includes(`P1 offset ${DIFF.v1.offsets[1]}`) && layout.includes(`P1 offset ${DIFF.v2.offsets[1]}`));
  assert.ok(layout.includes(`v1 size ${ORDER.v1.size} vs v2 ${ORDER.v2.size}`));
  assert.ok(layout.includes(`${LAYOUT.shapes.length} 个参数 = ${LAYOUT.shapes.length} 次 collective`));
  const diffPad = DIFF.v2.size - DIFF.shapes.reduce((acc, s) => acc + numel(s), 0);
  assert.equal((layout.match(/>·</g) ?? []).length, LAYOUT.padCells + 2 * diffPad, 'pad 格：面板 B 8 格 + 面板 E 两行各 2 格');
  assert.ok(!layout.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 2 ----
  assert.ok(timeline.includes(`前向 ${LANE_V1.fwdEnd}，反向结束 ${LANE_V1.bwdEnd}`));
  assert.ok(timeline.includes(`前向 ${LANE_V2.fwdEnd}，反向结束 ${LANE_V2.bwdEnd}`));
  assert.ok(timeline.includes(`反向结束 ${LANE_RECOMPUTE.bwdEnd}（不开预取则 ${LANE_RECOMPUTE_NO_PREFETCH.bwdEnd}）`));
  assert.ok(timeline.includes(`a ≥ ${AG_COMM.sameCommFirstLag}`) && timeline.includes(`a ≥ ${AG_COMM.separateFirstLag}`));
  for (const s of ['_pre_forward_param_unshard', 'RegisterFSDPBackwardFunction', 'sync_model_weight_from_main_weight', 'drain', 'PRE_BACKWARD', 'READY_TO_USE', 'PRESERVED']) {
    assert.ok(timeline.includes(s), `图 2 缺 ${s}`);
  }

  // ---- 图 3 ----
  assert.ok(cost.includes(`unit = ${UNIT.total.toLocaleString('en-US')} 参数`));
  assert.ok(cost.includes(`bf16 ${fmtMiB(COST.unitBytes.bf16)} MiB，fp8 ${fmtMiB(COST.unitBytes.fp8)} MiB`));
  for (const v of ['16', '5.500', '3.750', '0.875', '15', '4.500', '2.750', '1.875', '0.750']) {
    assert.ok(cost.includes(`>${v}<`), `图 3 缺柱顶数值 ${v}`);
  }
  assert.ok(cost.includes(`${fmtMiB(COST.transient[0].bf16)} MiB`) && cost.includes(`${fmtMiB(COST.transient[1].bf16)} MiB`));
  assert.ok(cost.includes(`= ${COST.comm.bf16[3].perMicrobatch.toFixed(2)} B`));

  const all = [layout, timeline, cost];
  NAMES.forEach((name, i) => {
    assertInsideCanvas(all[i], name);
    assertNoTextOverlap(all[i], name);
  });

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      all[i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_fsdp_figures.mjs`,
    );
  });
});

// ============================================================================
// (c) 页面正文引用的数值与图上一致
// ============================================================================

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  // 共用算例（布局）
  assert.ok(page.includes(`DP=${LAYOUT_CASE.dp}`));
  assert.ok(page.includes(`LCM(6, 4, 4, 2, 6) = ${LAYOUT.chunk}`));
  assert.ok(page.includes(`(${LAYOUT.v2.offsets.join(', ')})`));
  assert.ok(page.includes(`${LAYOUT.v2.size} 个元素`) && page.includes(`每 rank ${LAYOUT.shard} 个`));
  assert.ok(page.includes('(3, 4)') && page.includes('(1, 4)') && page.includes('(1, 2)'), 'rank1 / rank2 的本地形状');
  assert.ok(page.includes(`共 ${LAYOUT.totalNumel} 个元素`) && page.includes(`pad ${LAYOUT.padCells} 格`));
  assert.ok(page.includes(`offset ${DIFF.v1.offsets[1]}`) && page.includes(`offset ${DIFF.v2.offsets[1]}`), 'v1 / v2 差异小例');
  assert.ok(page.includes(`v1 给出 ${ORDER.v1.size} 个元素`) && page.includes(`v2 给出 ${ORDER.v2.size} 个`), '顺序小例');
  assert.ok(page.includes('4 个 rank 为空'), 'FSDP2 对照：P3 只有 1 行');

  // 仿真
  assert.ok(page.includes(`L=${SIM.L}`) && page.includes(`c=${SIM.c}`) && page.includes(`a=${SIM.a}`) && page.includes(`r=${SIM.r}`) && page.includes(`s=${SIM.s}`));
  assert.ok(page.includes(`makespan ${LANE_V1.fwdEnd}`) && page.includes(`反向结束 ${LANE_V1.bwdEnd}`), 'v1 makespan');
  assert.ok(page.includes(`前向 ${LANE_V2.fwdEnd}`) && page.includes(`反向结束 ${LANE_V2.bwdEnd}`), 'v2 makespan');
  assert.ok(page.includes(`${LANE_RECOMPUTE_NO_PREFETCH.bwdEnd} 缩到 ${LANE_RECOMPUTE.bwdEnd}`), '重算预取');
  assert.ok(page.includes(`a ≥ ${AG_COMM.sameCommFirstLag}`) && page.includes(`a ≥ ${AG_COMM.separateFirstLag}`), '独立 AG 组阈值');
  assert.ok(page.includes(`峰值在途参数桶 ${LANE_V1.peakParams}`), 'v1 峰值');
  assert.ok(page.includes(`< 3`), 'v2 峰值上界（测试）');

  // 字节账
  assert.ok(page.includes(UNIT.total.toLocaleString('en-US')));
  assert.ok(page.includes(`${fmtMiB(COST.unitBytes.bf16)} MiB`) && page.includes(`${fmtMiB(COST.unitBytes.fp8)} MiB`));
  assert.ok(page.includes(`${(MODEL_PARAMS / 1e9).toFixed(2)}B`) || page.includes(`${(MODEL_PARAMS / 1e9).toFixed(2)} B 参数`));
  for (const v of [16, 5.5, 3.75, 2, 0.875]) assert.ok(page.includes(`${v} B`), `正文缺常驻 ${v} B`);
  assert.ok(page.includes(`${fmtGiB(MODEL_PARAMS * 2)} GiB`) && page.includes(`${fmtGiB(MODEL_PARAMS * 16)} GiB`));
  assert.ok(page.includes(`${fmtMiB(COST.transient[0].bf16)} MiB`) && page.includes(`${fmtMiB(COST.transient[1].bf16)} MiB`));
  assert.ok(page.includes(`${COST.comm.bf16[3].perMicrobatch.toFixed(2)} B`) && page.includes(`${COST.comm.bf16[3].total.toFixed(2)} B`));
  assert.ok(page.includes(`${COST.comm.bf16[4].perCycle.toFixed(3)} B`));
  assert.ok(page.includes(`${COST.comm.bf16[0].total.toFixed(2)} B`) && page.includes(`${COST.comm.bf16[2].total.toFixed(2)} B`));
  assert.ok(page.includes(`D=${RECIPE.dp}`) && page.includes(`O=${RECIPE.hsdpOuter}`) && page.includes(`${RECIPE.microbatches} 个 microbatch`));

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
  assert.ok(!/[A-Za-z_/]+\.py:\d+/.test(page), '正文不得含 path:line 引用');
  assert.ok(!page.includes('../'), '正文不得含 ../ 相对链接');
});

// ============================================================================
// (a′) 业务链：unit → 桶 → 预取预算 / RS 队列（页面 §2.6）
// ============================================================================

import { UNIT_SHAPES, UNIT_BUCKET, PREFETCH, v1UnitBucket, prefetchPlan } from '../megatron_fsdp_figures.mjs';

test('llama3-8b unit 的 chunk_size_factor、pad 与预取深度按源码规则算出', async () => {
  assert.equal(UNIT_SHAPES.length, 6);
  assert.equal(UNIT_BUCKET.chunk, 28672, 'LCM(14336, 4096)');
  assert.equal(UNIT_BUCKET.elems, 218_112_000);
  assert.equal(UNIT_BUCKET.padded, 218_136_576, 'pad 到 DP × chunk = 229,376 的倍数');
  assert.equal(PREFETCH.commUnit, 1_000_000_000, '2 × unit < 1e9 时取下限 1e9');
  assert.equal(PREFETCH.agBudget, 500_000_000);
  assert.equal(PREFETCH.prefetchUnitsNoPool, 3, '不开持久池时预取 3 个 unit');
  assert.equal(PREFETCH.rsQueueUnits, 5, '梯度桶最多 5 个 unit 在途才等');
  // 规则的边界：unit 大到 2 × unit > 1e9 时预算随 unit 变，预取深度回到 1
  const big = prefetchPlan({ unitElems: 600_000_000, paddedUnitElems: 600_000_000 });
  assert.equal(big.commUnit, 1_200_000_000);
  assert.equal(big.prefetchUnitsNoPool, 1);
  // Step 3 合并规则：行宽整除时不抬 LCM
  assert.equal(v1UnitBucket([[8, 4], [2, 2]], 2).chunk, 4);

  const page = await readFile(pagePath, 'utf8');
  assert.ok(page.includes(`LCM(14336, 4096) = ${UNIT_BUCKET.chunk}`), '正文缺 chunk_size_factor');
  assert.ok(page.includes(UNIT_BUCKET.padded.toLocaleString('en-US')), '正文缺 pad 后的桶大小');
  assert.ok(page.includes(`预取 ${PREFETCH.prefetchUnitsNoPool} 个 unit`), '正文缺无持久池的预取深度');
  assert.ok(page.includes(`${PREFETCH.rsQueueUnits} 个 unit 的梯度桶`), '正文缺 RS 队列深度');
});
