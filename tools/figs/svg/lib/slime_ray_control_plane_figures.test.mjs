// 锁住 slime Ray 控制面布局图的可执行契约：图上每个数字都由同一份 CFG 经
// 源码算法的复现推导，并且必须与 11_slime_ray_control_plane_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_ray_control_plane_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, model } from '../slime_ray_control_plane_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_ray_control_plane_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '11_slime_ray_control_plane_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_ray_control_plane_layout.svg');

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

test('布局模型复现源码算法', () => {
  const m = model();
  assert.deepEqual([m.colocate.pgGpus, m.colocate.rolloutOffset], [4, 0]);
  assert.deepEqual([m.disaggregate.pgGpus, m.disaggregate.rolloutOffset], [8, 4]);
  // 排序：节点 A（.1）先于节点 B（.2），同节点按 gpu id
  assert.deepEqual(m.disaggregate.sorted.map((b) => `${b.ip.split('.').pop()}:${b.gpu}`),
    ['1:0', '1:1', '1:2', '1:3', '2:0', '2:1', '2:2', '2:3']);
  assert.deepEqual(m.colocate.sorted.map((b) => b.index), [1, 3, 0, 2]);
  // trainer rank r → 逻辑 r
  assert.deepEqual(m.colocate.trainers.map((t) => t.logical), [0, 1, 2, 3]);
  // engine i → 逻辑 offset + 2i，base_gpu_id 取该槽位 physical id
  assert.deepEqual(m.colocate.engines.map((e) => [e.logical, e.baseGpuId]), [[0, 0], [2, 2]]);
  assert.deepEqual(m.disaggregate.engines.map((e) => [e.logical, e.baseGpuId]), [[4, 0], [6, 2]]);
  assert.equal(m.colocate.needsOffload, true);
  assert.equal(m.disaggregate.needsOffload, false);
  // 端口：server/nccl 先分完，dist 再分，各预留 31
  assert.equal(m.ports.reserve, 30 + CFG.sglangDpSize);
  assert.deepEqual(m.ports.rows.map((r) => [r.server, r.nccl, r.dist, r.distEnd]),
    [[15000, 15001, 15004, 15034], [15002, 15003, 15035, 15065]]);
  assert.equal(m.ports.cursor, 15066);
});

test('生成器产出布局图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-ray-cp-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_ray_control_plane_layout.svg'), 'utf8');
    assertInsideCanvas(svg);
    assert.match(svg, /max\(4, 4\) = 4/);
    assert.match(svg, /4 \+ 4 = 8/);
    assert.match(svg, /rollout offset = 4/);
    assert.match(svg, /engine 1 · 0\.2 · base_gpu_id=2/);
    assert.match(svg, /needs_offload=True/);
    assert.match(svg, /needs_offload=False/);
    assert.match(svg, /server 15000 · nccl 15001/);
    assert.match(svg, /预留 15035–15065/);
    assert.match(svg, /cursor 终值 15066/);
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
  for (const needle of [
    `max(${CFG.actorNumGpusPerNode}, ${CFG.rolloutNumGpus}) = ${m.colocate.pgGpus}`,
    `${CFG.actorNumGpusPerNode} + ${CFG.rolloutNumGpus} = ${m.disaggregate.pgGpus}`,
    `offset = ${m.disaggregate.rolloutOffset}`,
    `逻辑 ${m.disaggregate.engines[1].logical}`,
    `base_gpu_id=${m.disaggregate.engines[1].baseGpuId}`,
    `${m.ports.rows[0].server}`, `${m.ports.rows[0].nccl}`, `${m.ports.rows[0].dist}`,
    `${m.ports.rows[1].dist}–${m.ports.rows[1].distEnd}`,
    `${m.ports.cursor}`,
    `${CFG.trainerGpuDecl}`, `${CFG.engineGpuDecl}`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_ray_control_plane_layout.svg'), '正文必须引用布局图');
});
