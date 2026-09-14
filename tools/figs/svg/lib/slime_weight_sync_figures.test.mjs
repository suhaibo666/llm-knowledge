// 锁住 slime 权重同步原理图的可执行契约：图上每个数字都由同一份 CFG 经源码算法的复现推导，
// 并且必须与 16_slime_weight_sync_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_weight_sync_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, PLANES, bucketSequential, expertRouting, model, pullReplay } from '../slime_weight_sync_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_weight_sync_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '16_slime_weight_sync_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_weight_sync_planes.svg');

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

function assertInsideCanvas(svg) {
  const { w, h } = viewBox(svg);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    assert.ok(Number(x) >= 0 && Number(y) >= 0, `rect 左上越界 ${x},${y}`);
    assert.ok(Number(x) + Number(rw) <= w && Number(y) + Number(rh) <= h, `rect 右下越界 ${x}+${rw},${y}+${rh}`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w && Number(y) >= 0 && Number(y) <= h, `text 越界 ${x},${y}`);
  }
}

test('共同前半段与分桶复现 all_gather_param / convert_qwen2_to_hf / 装桶规则', () => {
  const m = model();
  assert.deepEqual(m.fc1.shards, [['g0', 'g1', 'u0', 'u1'], ['g2', 'g3', 'u2', 'u3']]);
  assert.deepEqual(m.fc1.gathered, ['g0', 'g1', 'g2', 'g3', 'u0', 'u1', 'u2', 'u3']);
  assert.deepEqual(m.fc1.naiveGate, ['g0', 'g1', 'u0', 'u1'], '不重排时 gate_proj 混进 up 行');
  assert.deepEqual(m.fc1.gate, ['g0', 'g1', 'g2', 'g3']);
  assert.deepEqual(m.fc1.infer[4][0], ['g0', 'u0']);
  assert.deepEqual(m.fc1.directCopyOk, { 1: false, 2: true, 4: false }, '只有推理 TP=2 时直接拷贝恰好相等');
  assert.deepEqual(m.buckets.nonExpert, [[300], [260, 100]]);
  assert.deepEqual(m.buckets.big, [[600]]);
  assert.deepEqual(m.buckets.expert.map((b) => [b.local.length, b.gathered]), [[1, 400], [1, 400], [1, 400], [1, 400]]);
  // 已有内容且会超限才换桶：空桶时超阈值的参数照样放进去
  assert.deepEqual(bucketSequential([700, 10], 512), [[700], [10]]);
});

test('NCCL、共卡划分与 MoE 定向路由复现源码', () => {
  const m = model();
  assert.deepEqual(m.nccl, { world: 7, rankOffsets: [1, 3] });
  assert.equal(m.coloc.prefixCount, 2);
  assert.deepEqual(m.coloc.gatherGroups, [{ ranks: [0, 1], src: 0 }, { ranks: [2, 3], src: 2 }]);
  assert.deepEqual(m.coloc.suffix, { world: 3, rankOffsets: [1] });
  assert.deepEqual(m.coloc.pauseTargets, [0, 1], '越界 engine 不在 pause/flush 名单里');
  assert.deepEqual(m.moe.targets, [[0, 2], [1, 3]]);
  assert.deepEqual(m.moe.owners, [[0, 1], [2, 3], [4, 5], [6, 7]]);
  assert.equal(m.moe.sends, 12);
  assert.equal(m.moe.genericDeliveries, 24);
  assert.deepEqual(m.moe.routedPerRank, [4, 4, 4, 4]);
  assert.deepEqual(m.moe.batches, [[0, 1, 2, 3], [4, 5, 6, 7]]);
  // 与 tests/test_expert_routing.py 的单包超阈值规则一致
  assert.throws(() => expertRouting({ ...CFG.moe, bufferBundles: 0.5 }), /exceeds/);
});

test('增量字节与 pull 版本逻辑复现 _encode_delta / overwrite_encode / local_checkpoint.pull', () => {
  const m = model();
  const hex = (bs) => bs.map((b) => b.toString(16).toUpperCase().padStart(2, '0')).join(' ');
  assert.equal(hex(m.delta.oldB), '80 3F 00 3F 00 C0 80 3E');
  assert.equal(hex(m.delta.newB), '80 3F 02 3F 00 C0 81 3E');
  assert.equal(m.delta.changed, 2);
  assert.equal(m.delta.density, 0.25);
  assert.equal(m.delta.xorLen, 8);
  assert.equal(m.delta.overwriteLen, 14);
  assert.ok(m.delta.xorTwiceRevertsToOld, 'xor 是对合');
  assert.ok(m.delta.overwriteTwiceStaysNew, 'overwrite 幂等');
  assert.deepEqual(m.pulls.fresh.ops, ['reset base', 'apply v1', 'apply v2']);
  assert.deepEqual(m.pulls.atOne.ops, ['apply v2']);
  assert.deepEqual(m.pulls.staleDelta, { ops: [], localAfter: 57 }, '残留标记让低版本 pull 什么也不做');
  assert.deepEqual(m.pulls.staleFull, { ops: [], localAfter: 57 });
  assert.deepEqual(m.pulls.staleFullCatchUp.ops, ['reset v58']);
  assert.deepEqual(pullReplay({ applied: 3, target: 5, isDelta: (v) => v !== 4 }).ops, ['reset v4', 'apply v5']);
});

test('在线路径的搬运落在暂停窗口里，磁盘路径的搬运在暂停之前', () => {
  const firstPause = (ps) => ps.findIndex(([, p]) => p);
  const idx = (ps, needle) => ps.findIndex(([l]) => l.includes(needle));
  assert.ok(idx(PLANES.nccl, '广播') > firstPause(PLANES.nccl));
  assert.ok(idx(PLANES.ipc, 'IPC') > firstPause(PLANES.ipc));
  assert.ok(idx(PLANES.disk, '写 HF') < firstPause(PLANES.disk));
  assert.ok(idx(PLANES.delta, 'pull') < firstPause(PLANES.delta));
  assert.ok(idx(PLANES.disk, 'reload') > firstPause(PLANES.disk));
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-weight-sync-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_weight_sync_planes.svg'), 'utf8');
    assertInsideCanvas(svg);
    for (const needle of ['2 + 4 + 1 = 7', 'world 3', 'P2P 发送 12 份', 'EP 广播 24 份', 'batch0 = e0–e3', '4 + 2×4 + 2 = 14', '25%', 'weight_v000001/', '服务暂停窗口']) {
      assert.ok(svg.includes(needle), `SVG 必须出现 ${needle}`);
    }
    assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
    const tracked = await readFile(trackedSvg, 'utf8');
    assert.equal(tracked, svg, '已跟踪的 SVG 必须由当前生成器重新生成');
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const hex = (bs) => bs.map((b) => b.toString(16).toUpperCase().padStart(2, '0')).join(' ');
  for (const needle of [
    m.fc1.gathered.join(' '),
    m.fc1.naiveGate.join(' '),
    m.fc1.infer[4][0].join(' '),
    `[${m.buckets.nonExpert[0].join('+')}] [${m.buckets.nonExpert[1].join('+')}]`,
    `EP 聚合后 ${m.buckets.expert[0].gathered} MiB`,
    `world = ${CFG.ncclEngineGpuCounts.join(' + ')} + 1 = ${m.nccl.world}`,
    `rank_offset ${m.nccl.rankOffsets[0]} 与 ${m.nccl.rankOffsets[1]}`,
    `world ${m.coloc.suffix.world}`,
    `{${m.coloc.gatherGroups[0].ranks.join(',')}}、{${m.coloc.gatherGroups[1].ranks.join(',')}}`,
    `P2P ${m.moe.sends} 份`,
    `共 ${m.moe.genericDeliveries} 份`,
    `batch0 = e${m.moe.batches[0][0]}–e${m.moe.batches[0][3]}`,
    `batch1 = e${m.moe.batches[1][0]}–e${m.moe.batches[1][3]}`,
    hex(m.delta.oldB),
    hex(m.delta.newB),
    `${m.delta.changed}/${m.delta.total} 字节`,
    `${m.delta.density * 100}%`,
    `4 + ${m.delta.changed}×4 + ${m.delta.changed} = ${m.delta.overwriteLen}`,
    `xor 差分 ${m.delta.xorLen} 字节`,
    '依次应用 v1、v2',
    '只应用 v2',
    '标记（如 57）',
    '`pull(1)` 什么也不做',
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_weight_sync_planes.svg'), '正文必须引用原理图');
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留 path:line 链接');
});
