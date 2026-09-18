// 锁住 slime rollout 接收循环与变体数据面两张原理图的可执行契约：图上每个数字都由同一份 CFG 经源码算法的复现推导，
// 并且必须与 13_slime_sglang_rollout_engine_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_rollout_admission_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, D_TOKENS_AT_ABORT, OUTPUTS, STREAM, abortPlan, accumulate, checkRewardNonzeroStd, checkRewardNonzeroStdWithFallback,
  model, semaphoreCapacity, shouldDrop, streamChunks,
} from '../slime_rollout_admission_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_rollout_admission_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '13_slime_sglang_rollout_engine_analysis.md');

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

function rects(svg, cls) {
  return [...svg.matchAll(/<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)" rx="[\d.]+" class="([\w]+)"\/>/g)]
    .filter((m) => !cls || m[5] === cls)
    .map((m) => ({ x: Number(m[1]), y: Number(m[2]), w: Number(m[3]), h: Number(m[4]) }));
}

function texts(svg) {
  return [...svg.matchAll(/<text x="(-?[\d.]+)" y="(-?[\d.]+)" class="(\w+)" text-anchor="(\w+)">([^<]*)<\/text>/g)]
    .map((m) => ({ x: Number(m[1]), y: Number(m[2]), cls: m[3], anchor: m[4], s: m[5] }));
}

// 粗略字宽：CJK 与全角符号按 1 em，其余按 0.56 em（只用于检查面板越界，不代替渲染目检）
const FONT = { ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, cap: 11.5, mono: 11, tag: 10.5, tagx: 10.5 };
function estimateWidth(t) {
  const size = FONT[t.cls] ?? 12;
  const decoded = t.s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  let w = 0;
  for (const ch of decoded) w += /[⺀-鿿＀-￯　-〿]/.test(ch) ? size : size * 0.56;
  return w;
}

function assertGeometry(svg) {
  const { w, h } = viewBox(svg);
  for (const r of rects(svg)) {
    assert.ok(r.x >= 0 && r.y >= 0, `rect 左上越界 ${r.x},${r.y}`);
    assert.ok(r.x + r.w <= w && r.y + r.h <= h, `rect 右下越界 ${r.x}+${r.w},${r.y}+${r.h}`);
  }
  const panels = rects(svg, 'panel');
  const lastPanelBottom = Math.max(...panels.map((p) => p.y + p.h));
  for (const t of texts(svg)) {
    assert.ok(t.x >= 0 && t.x <= w && t.y >= 0 && t.y <= h, `text 越界 ${t.s}`);
    const width = estimateWidth(t);
    const left = t.anchor === 'middle' ? t.x - width / 2 : t.anchor === 'end' ? t.x - width : t.x;
    assert.ok(left >= 0 && left + width <= w + 1, `text 横向越出画布：${t.s}`);
    const host = panels.find((p) => t.x >= p.x && t.x <= p.x + p.w && t.y > p.y && t.y <= p.y + p.h);
    if (host) {
      // 文字底线必须在面板底边之上，右端不得跨进相邻面板
      assert.ok(t.y <= host.y + host.h - 4, `text 压在面板底边上：${t.s}`);
      assert.ok(left + width <= host.x + host.w + 1, `text 横穿出所在面板：${t.s}`);
    } else if (t.y > panels[0].y) {
      // 面板外的文字（页脚）不得与任何面板重叠
      assert.ok(t.y - (FONT[t.cls] ?? 12) >= lastPanelBottom + 4, `面板外文字压到面板：${t.s}`);
    }
  }
}

test('动态过滤与丢弃规则复现源码', () => {
  assert.equal(checkRewardNonzeroStd([1, 1, 1, 1]).keep, false);
  assert.equal(checkRewardNonzeroStd([1, 1, 1, 1]).reason, 'zero_std_1');
  assert.equal(checkRewardNonzeroStd([1, 0, 0, 1]).keep, true);
  const zero = checkRewardNonzeroStdWithFallback([0, 0, 0, 0]);
  assert.equal(shouldDrop(zero, { remainingBatchSize: 3, targetDataSize: 2 }), true);
  assert.equal(shouldDrop(zero, { remainingBatchSize: 2, targetDataSize: 2 }), false);
  assert.equal(shouldDrop(checkRewardNonzeroStd([0, 0, 0, 0]), { remainingBatchSize: 2, targetDataSize: 2 }), true);
  // http_utils.get_rollout_num_engines：rollout_num_gpus <= 0 时 engine 数为 0，信号量容量为 0
  assert.deepEqual(semaphoreCapacity({ ...CFG, rolloutNumGpus: 0 }), { engines: 0, capacity: 0 });
});

test('abort 决策复现 sglang_rollout.abort 的两种模式与回收判定', () => {
  const server = abortPlan({ inFlight: [{ group: 'D', tokensAtAbort: [6, 4, 0, 8] }] });
  assert.deepEqual([server.cancellableTasks, server.activeServerGenerations, server.serverAbort], [0, 4, true]);
  assert.equal(server.groups[0].recycled, true);
  const request = abortPlan({ inFlight: [{ group: 'D', tokensAtAbort: [6, 4, 0, 8] }], abortMode: 'request' });
  assert.deepEqual([request.cancellableTasks, request.activeServerGenerations, request.serverAbort], [4, 0, false]);
  // tests/test_streaming_rollout.py::test_partial_abort_buffers_and_resumes_only_aborted_siblings：[empty] 与 [terminal] 不回收
  assert.equal(abortPlan({ inFlight: [{ group: 'X', tokensAtAbort: [0, 0, 0, 0] }] }).groups[0].recycled, false);
  assert.equal(abortPlan({ inFlight: [{ group: 'X', tokensAtAbort: [2, 2, 2, 2], completed: true }] }).groups[0].recycled, false);
  assert.equal(abortPlan({ inFlight: [{ group: 'X', tokensAtAbort: [6, 4, 0, 8] }], partialRollout: false }).groups[0].recycled, false);
});

test('流式累加器复现 SGLangStreamAccumulator 的两种格式与长度校验', () => {
  const cum = streamChunks({ mode: 'cumulative', interval: 2, totalTokens: 6, firstTokenId: 11 });
  const inc = streamChunks({ mode: 'incremental', interval: 2, totalTokens: 6, firstTokenId: 11 });
  assert.deepEqual(cum.map((c) => c.wireLen), [2, 4, 6]);
  assert.deepEqual(inc.map((c) => c.wireLen), [2, 2, 2]);
  assert.deepEqual(cum.map((c) => c.reported), [2, 4, 6]);
  assert.deepEqual(inc.map((c) => c.reported), [2, 4, 6]);
  assert.deepEqual(accumulate('cumulative', cum).tokens, [11, 12, 13, 14, 15, 16]);
  assert.deepEqual(accumulate('incremental', inc).tokens, [11, 12, 13, 14, 15, 16]);
  // 配置与服务端格式不一致：第 2 个 chunk 抛错（test_stream_accumulator_rejects_output_incompatible_with_configured_mode）
  assert.throws(() => accumulate('incremental', cum), /incremental streaming output has inconsistent/);
  assert.throws(() => accumulate('cumulative', inc), /cumulative streaming output has inconsistent/);
  // 不整除时最后一个 chunk 收尾
  assert.deepEqual(streamChunks({ mode: 'incremental', interval: 4, totalTokens: 6, firstTokenId: 1 }).map((c) => c.reported), [4, 6]);
});

test('接收循环、流式与 fully-async 模型复现源码算法', () => {
  const m = model();
  assert.deepEqual(m.semaphore, { engines: 2, capacity: 1024 });
  assert.deepEqual(m.main.all, ['B', 'A', 'C']);
  assert.deepEqual(m.main.data, ['A', 'C']);
  assert.deepEqual(m.main.inFlight, ['D']);
  assert.deepEqual(m.main.recycled, ['D']);
  assert.equal(m.main.plan.activeServerGenerations, 4);
  assert.deepEqual(m.main.plan.groups[0].samples.map((s) => s.responseLength), [...D_TOKENS_AT_ABORT]);
  assert.equal(m.dNoTokens.groups[0].recycled, false);
  assert.equal(m.dCompleted.groups[0].recycled, false);
  assert.deepEqual(m.main.log.map((l) => l.remaining), [4, 3, 3, 3, 3]);
  assert.deepEqual(m.main.log.map((l) => l.accepted), [0, 0, 1, 2, 2]);
  assert.deepEqual(m.main.log.map((l) => l.pendings), [4, 3, 2, 1, 0]);
  assert.deepEqual(m.groupsMain.D.indices, [12, 13, 14, 15]);
  // 连续拒绝：remaining 4 → 3 → 2 → 1，再补一整波 → 5
  assert.deepEqual(m.cascade.log.slice(0, 5).map((l) => l.remaining), [4, 3, 2, 1, 5]);
  assert.deepEqual(m.cascade.waves.map((w) => w.wave), [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']]);
  assert.deepEqual(m.cascade.data, ['D', 'E']);
  assert.equal(m.cascade.inFlight.length, 3);
  // fallback：remaining 2 ≤ 2 时保留零方差的 C
  assert.equal(m.fallback.log[2].remaining, 2);
  assert.equal(m.fallback.log[3].action, 'accept (keep_when_insufficient)');
  assert.deepEqual(m.fallback.data, ['C', 'D']);
  assert.equal(m.fallback.waves.length, 1);
  // 流式：样本 12 三个 chunk，累计 12 对、增量 6 对；request 模式取消 4 个 task，不走服务端排空
  assert.equal(STREAM.streamInterval, 2);
  assert.deepEqual(m.stream.cumulative.updates.map((u) => u.responseLength), [2, 4, 6]);
  assert.deepEqual([m.stream.cumulativePairs, m.stream.incrementalPairs], [12, 6]);
  assert.deepEqual([m.stream.abort.cancellableTasks, m.stream.abort.activeServerGenerations], [4, 0]);
  assert.equal(m.stream.abort.groups[0].recycled, true);
  // fully-async：generate 在途时边等边取，队列不积压；generate(1) 返回后无消费者，t6 队列 [E F] 关闸；
  // t7 pause 让在途 G 回 buffer；t8 generate(2) 取走 E F，补位先取回 G（不推进游标）再取 H
  const fa = m.fullyAsync;
  assert.equal(fa.cap, 2);
  assert.deepEqual(fa.collected[0], ['A', 'B']);
  assert.deepEqual(fa.collected[1], ['C', 'D']);
  assert.deepEqual(fa.collected[2], ['E', 'F']);
  assert.ok(fa.ticks.slice(0, 5).every((t) => t.consumerActive && t.queue.length === 0 && !t.gateClosed), 'generate 在途时队列不积压');
  assert.deepEqual(fa.ticks[2].started, [1]);
  assert.equal(fa.ticks[4].drained[0].complete, true);
  assert.equal(fa.ticks[5].consumerActive, false);
  assert.deepEqual(fa.ticks[6].queue, ['E', 'F']);
  assert.equal(fa.ticks[6].gateClosed, true);
  assert.deepEqual(fa.ticks[6].topped, []);
  assert.deepEqual(fa.ticks[6].active, ['G']);
  assert.deepEqual(fa.ticks[7].requeued, ['G']);
  assert.deepEqual(fa.ticks[7].topped, [], '闸门仍关，G 暂不补回');
  assert.equal(fa.ticks.findIndex((t) => t.gateClosed), 6, '闸门只在没有 generate 在途时关上');
  assert.deepEqual(fa.ticks[8].fromBuffer, ['G']);
  assert.deepEqual(fa.ticks[8].topped, ['G', 'H']);
  assert.equal(fa.ticks[8].cursor, fa.ticks[7].cursor + 1, '取自 buffer 的 G 不推进游标，只有 H 推进');
  assert.equal(fa.ticks[8].resumed, true);
});

test('生成器产出两张原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-admission-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const admission = await readFile(join(outputDir, OUTPUTS.admission), 'utf8');
    const variants = await readFile(join(outputDir, OUTPUTS.variants), 'utf8');
    for (const svg of [admission, variants]) {
      assertGeometry(svg);
      assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
      assert.match(svg, /THUDM\/slime@4c193f1f3750/);
    }
    assert.match(admission, /512 × 2 engines = 1024/);
    assert.match(admission, /B 被拒：remaining 4 → 3/);
    assert.match(admission, /remaining 3 → 2 → 1/);
    assert.match(admission, /remaining=2 ≤ 2/);
    assert.match(admission, /active_server_generations = 4/);
    assert.match(admission, /response_length = 6 4 0 8/);
    assert.match(admission, /四条都是 0 → 不回收/);
    assert.match(variants, /共传 12 对 logprob，增量模式 6 对/);
    assert.match(variants, /cancellable_tasks = 4/);
    assert.match(variants, /active_server_generations = 0/);
    assert.match(variants, /r0←B/);
    assert.match(variants, /r2←E F/);
    assert.match(variants, /队列 \[E F\] 达闸门/);
    assert.match(variants, /G↺ H/);
    assert.match(variants, /PauseGenerationReqInput\.mode 默认 &quot;abort&quot;|PauseGenerationReqInput\.mode 默认 "abort"/);
    for (const [name, svg] of [[OUTPUTS.admission, admission], [OUTPUTS.variants, variants]]) {
      const tracked = await readFile(join(slimeDir, 'assets', name), 'utf8');
      assert.equal(tracked, svg, `已跟踪的 ${name} 必须由当前生成器重新生成`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const c = m.semaphore;
  const fa = m.fullyAsync;
  for (const needle of [
    `${CFG.sglangServerConcurrency} × ${c.engines} = ${c.capacity}`,
    `${CFG.overSamplingBatchSize * CFG.nSamplesPerPrompt} 个请求`,
    `、${m.groupsMain.D.indices[0]}–${m.groupsMain.D.indices.at(-1)}）`,
    `| t1 | B 完成，reward \`${m.groupsMain.B.rewards.join(' ')}\` 零方差 → 拒绝 | ${m.main.log[0].remaining} → ${m.main.log[1].remaining} |`,
    `| t3 | C 完成 → 入选，达到目标 | ${m.main.log[3].remaining} | ${m.main.log[3].accepted} | ${m.main.log[3].pendings} |`,
    `入选的 ${m.main.data.join('、')}`,
    `\`active_server_generations = ${m.main.plan.activeServerGenerations}\``,
    `已生成 ${D_TOKENS_AT_ABORT.join('、')} 个 token`,
    `从 ${m.cascade.log[0].remaining} 降到 ${m.cascade.log[1].remaining}、${m.cascade.log[2].remaining}、${m.cascade.log[3].remaining} 后，一次再补 ${CFG.overSamplingBatchSize} 组`,
    `随后 remaining 为 ${m.cascade.log[4].remaining}`,
    `--sglang-stream-interval ${STREAM.streamInterval}`,
    `分别带 ${m.stream.cumulativeChunks.map((ch) => ch.wireLen).join('、')} 对`,
    `${m.stream.cumulativeChunks.length} 个 chunk 共传 ${m.stream.cumulativePairs} 对 logprob，增量模式 ${m.stream.incrementalPairs} 对`,
    `依次为 ${m.stream.cumulative.updates.map((u) => u.responseLength).join('、')}`,
    `\`cancellable_tasks = ${m.stream.abort.cancellableTasks}\``,
    `样本 12 保留 ${m.stream.cumulative.tokens.length} 个 token`,
    `池容量 ${fa.cap}`,
    `队列 \`[${fa.ticks[6].queue.join(' ')}]\` 达闸门`,
    `\`[${fa.collected[2].join(' ')}]\` 留待下轮`,
    `在途 ${fa.ticks[7].requeued.join('、')} 以 ABORTED 回 buffer`,
    `补位先从 buffer 取回 ${fa.ticks[8].fromBuffer.join('、')}`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes(`assets/${OUTPUTS.admission}`), '正文必须引用接收循环原理图');
  assert.ok(page.includes(`assets/${OUTPUTS.variants}`), '正文必须引用变体数据面原理图');
  assert.ok(page.includes('#### 2.3.3 ABORTED 回队分支何时可达'), '页 10 链接的稳定标题必须存在');
});
