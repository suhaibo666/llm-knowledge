import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { replay, planes, draw, drawPlanes, fmt, TOPK, R, S, K, EPN, CAP } from './moonep_planning_figures.mjs';

const page = readFileSync(new URL('../../../wiki/01_theory/01_models/moonshot_kimi/27_moonep_analysis.md', import.meta.url), 'utf8');
const asset = name => readFileSync(new URL(`../../../wiki/01_theory/01_models/moonshot_kimi/assets/${name}`, import.meta.url), 'utf8');

test('planner invariants hold on the worked example', () => {
  const r = replay();
  assert.equal(r.balance0.reduce((a, b) => a + b, 0), 0);
  for (const rk of r.ranks) {
    assert.equal(rk.total, CAP, `rank ${rk.d} does not receive exactly CAP tokens`);
    assert.ok(rk.etc.filter(e => e >= 0).length <= EPN);
  }
  // 每个接收 rank 在 z 的列上至多一个非零项：迁入只来自一个 home group
  for (let u = 0; u < R; u++) assert.ok(r.z.filter(row => row[u] > 0).length <= 1);
  assert.ok(r.rounds.length <= R - 1);
  for (let e = 0; e < r.alloc.length; e++) assert.equal(r.alloc[e].reduce((a, b) => a + b, 0), r.expertCount[e]);
  // 本例专门包含一次过冲：发送方翻负后再当接收方
  assert.ok(r.rounds.some(x => x.after[x.h] < 0));
  assert.ok(r.moved > r.lowerBound);
});

test('page §2.1 inputs, Step 1 and Step 2 match the replay', () => {
  const r = replay();
  TOPK.forEach((row, i) => assert.ok(page.includes(`| src${i} | \`${fmt(row)}\` |`), `topk src${i} drifted`));
  assert.ok(page.includes(`\`expert_count = ${fmt(r.expertCount)}\``));
  assert.ok(page.includes(`\`group_tokens = ${fmt(r.groupTokens)}\``));
  assert.ok(page.includes(`\`balance = ${fmt(r.balance0)}\``));
  r.rounds.forEach((x, i) => {
    const row = `| ${i + 1} | ${x.h} | ${x.u} | ${x.move} | \`${fmt(x.after)}\` |`;
    assert.ok(page.includes(row), `Step 2 round ${i + 1} drifted: ${row}`);
  });
  assert.ok(page.includes(`总迁移量 ${r.moved}，比下界 \`Σ 正 balance = ${r.lowerBound}\``));
});

test('page Step 3–5 match the replay', () => {
  const r = replay();
  for (const t of r.takes) assert.ok(page.includes(`\`z[${t.h},${t.d}]=${t.take}\``), `take z[${t.h},${t.d}] drifted`);
  for (const rk of r.ranks) {
    const cell = gs => gs.map(g => (g.cnt > 0 ? `e${g.eid}×${g.cnt}` : '空')).join('、');
    const row = `| rank ${rk.d} | ${cell(rk.groups.slice(0, EPN))} | ${cell(rk.groups.slice(EPN))} | \`${fmt(rk.etc)}\` | \`${fmt(rk.cu)}\` |`;
    assert.ok(page.includes(row), `Step 4 row drifted: ${row}`);
  }
  const copiesOfRank0 = r.ranks.flatMap(rk => rk.etc).filter(e => e >= 0 && Math.floor(e / EPN) === 0).length;
  assert.ok(page.includes(`\`remote_stats\` 第二项为 ${copiesOfRank0}`));
  assert.ok(page.includes(`\`alloc_cumsum[e0] = ${fmt(r.allocCumsum[0])}\``));
  const [a, b] = r.dst[1].filter(x => x.e === 0);
  assert.ok(page.includes(`src1 的 2 个是 ${a.g}、${b.g}`));
  assert.ok(page.includes(`rank ${a.dest} 第 ${a.row} 行（\`dst = ${a.raw}\`）`));
  assert.ok(page.includes(`\`dst = ${b.dest}·${r.NvS} + ${b.row} = ${b.raw}\``));
  assert.ok(page.includes(`src0 最终的 \`dst = ${fmt(r.canon[0])}\``));
  // Step 5 的逐 token 表：每个 source rank 哪些 token 含 e0，以及 e0 列直方图的前缀和
  TOPK.forEach((row, s) => {
    const toks = Array.from({ length: S }, (_, t) => row.slice(t * K, t * K + K));
    const has = toks.flatMap((pair, t) => (pair.includes(0) ? [`t${t}`] : []));
    const line = `| src${s} | ${toks.map(pair => `(${pair.join(', ')})`).join(' | ')} | ${has.join('、')}，共 ${has.length} 个 |`;
    assert.ok(page.includes(line), `Step 5 token table row drifted: ${line}`);
  });
  const e0Counts = r.tpe.map(row => row[0]);
  const e0Starts = e0Counts.map((_, s) => e0Counts.slice(0, s).reduce((a, b) => a + b, 0));
  assert.ok(page.includes(`e0 这一列是 \`${fmt(e0Counts)}\`，前缀和 \`${fmt(e0Starts)}\``));
  // Step 5 的逐条目表：e0 的全局序号 → 来源、目标 rank、目标行、dst
  const e0 = r.dst.flatMap((arr, src) => arr.map((x, i) => ({ ...x, src, tok: Math.floor(i / K) })).filter(x => x.e === 0)).sort((x, y) => x.g - y.g);
  for (const [label, key] of [['e0 的全局序号', x => x.g], ['来自', x => `src${x.src} t${x.tok}`], ['目标 rank', x => x.dest], ['目标行', x => x.row], ['`dst`', x => x.raw]]) {
    const line = `| ${label} | ${e0.map(key).join(' | ')} |`;
    assert.ok(page.includes(line), `Step 5 table row drifted: ${line}`);
  }
  // §2.2.2 用同一例子说明全局序号只靠直方图前缀和
  assert.ok(page.includes(`\`tpe_cumsum\` 在 e0 这一列是 \`${fmt(r.tpe.map((_, s) => r.tpe.slice(0, s + 1).reduce((p, row) => p + row[0], 0)))}\``));
  const r4 = replay(4);
  assert.ok(page.includes(`= ${r4.NvS}\``), 'token_padding=4 NvS drifted');
  assert.ok(page.includes(`rank 3 的 \`cu_seqlens = ${fmt(r4.ranks[3].cu)}\``));
});

test('page §2.1.2–2.1.3 data plane and weight plane match the replay', () => {
  const p = planes();
  p.ents.forEach((row, s) => {
    const line = `| src${s} | ${row.join(' | ')} | ${row.reduce((a, b) => a + b, 0)} |`;
    assert.ok(page.includes(line), `pair matrix row src${s} drifted: ${line}`);
  });
  const recv = Array.from({ length: R }, (_, d) => p.ents.reduce((s, row) => s + row[d], 0));
  assert.deepEqual(recv, Array(R).fill(CAP));
  assert.ok(page.includes(`| 接收合计 | ${recv.join(' | ')} | |`));
  const how = { local: '本地写', nvlink: 'NVLink 直写' };
  for (const x of p.landed) {
    const line = `| ${x.row} | e${x.e} | src${x.src} token ${x.tok}，k=${x.k} | ${x.dup ? `epilogue 复制行 ${x.primaryRow}` : how[x.kind]} |`;
    assert.ok(page.includes(line), `rank 0 row ${x.row} drifted: ${line}`);
  }
  assert.ok(page.includes(`dispatch 写出的 hidden 行依次是 ${p.written.join('、')}`));
  assert.ok(page.includes(`跨 NVLink 到达的依次是 ${p.viaNvlink.join('、')}`));
  const pullText = p.pulls.map(q => `src${q.src} ${q.src === 0 ? '本地' : '远端'}读行 ${q.rows.join('、')}`).join('，');
  assert.ok(page.includes(pullText), `combine pulls drifted: ${pullText}`);
  assert.ok(page.includes(`共有 ${p.pushes.length} 份复制`));
  const pushText = p.pushes.map(q => `rank ${q.owner} 把 e${q.e} 推到 rank ${q.dest} 的槽 ${q.slot}`).join('，');
  assert.ok(page.includes(pushText), `prefetch pushes drifted: ${pushText}`);
});

test('checked-in SVGs are current and contain real text', () => {
  for (const [name, fn] of [['moonep_planning_example.svg', draw], ['moonep_dataplane_example.svg', drawPlanes]]) {
    const svg = fn();
    assert.equal(asset(name).trim(), svg.trim(), `${name} is stale; rerun the generator`);
    assert.ok(svg.includes('<text') && !svg.includes('<foreignObject'));
    assert.ok(page.includes(`](assets/${name})`), `${name} is not referenced by the page`);
  }
  const r = replay(), p = planes();
  for (const rk of r.ranks) assert.ok(draw().includes(fmt(rk.cu)), `SVG cu_seqlens rank ${rk.d} drifted`);
  assert.ok(draw().includes(`dst = 3·NvS + 3 = ${r.dst[1].filter(x => x.e === 0)[1].raw}`));
  assert.ok(drawPlanes().includes(`跨 NVLink 到达   ${p.viaNvlink.join(' / ')}`));
});
