// 图 1：slime 默认 rollout 接收循环（`generate_rollout_async`）怎样用 over-sampling、FIRST_COMPLETED、
// 动态过滤与 abort 从四个候选 prompt group 得到两个训练 group，abort 时在途组 D 怎样被判定回收。
// 图 2：同一在途组 D 在两条变体数据面上的回放——流式内层调用（`generate_streaming`，abort_mode="request"）
// 逐 chunk 落到 Sample；fully-async 后台池（`AsyncRolloutWorker._loop`）跨轮保持在途组，
// 权重更新的 pause（mode=abort）让在途组以 ABORTED 回 buffer。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；
// 依赖侧：sgl-project/sglang@0b3bb0cbe318（v0.5.15.post1 上游源码，静态阅读）+ slime docker/patch/latest/*.patch。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 图 1 要讲清楚：
//  1. 接收单位是整组：一组 n 条 Sample 全部生成并打完 reward 后才进入 FIRST_COMPLETED 的接收判断。
//  2. remaining_batch_size 决定何时再补一整波；len(data) 达到 rollout_batch_size 即停止并 abort。
//  3. abort 按生成函数的 abort_mode 分两路：默认 generate 计入 active_server_generations，走
//     GET /workers → abort_all → /v1/loads；服务端以 finish_reason=abort 返回，Sample 变 ABORTED；
//     partial 下只回收"至少一个 ABORTED 且 response_length>0"的组。
//  4. 两条过滤变体：连续拒绝触发整波补采；with_fallback 在 remaining ≤ target 时保留零方差组。
// 布局：上方面板画四组时间线、三条计数器与 t4 abort 细节条；下方面板左侧两条变体条，右侧轮末处理注释。
//
// 图 2 要讲清楚：
//  1. 流式：同一 D 组的样本 12，stream_interval=2；累计模式每个 chunk 携带到目前为止的全部 logprob 对，
//     增量模式只带新增；SGLangStreamAccumulator 两种模式都只追加新 token；abort 时 request 模式
//     直接取消 task，不发 abort_all、不查 /v1/loads，Sample 保留最后观测到的前缀；非流式对照在
//     abort JSON 返回前 Sample 上没有 token。
//  2. fully-async：池容量与 qsize 闸门都是 2；generate 在途时边等边按缺口取，队列不积压；generate(1) 返回后、
//     update_weights 结束前没有消费者，完成组积压到闸门；pause_generation（上游默认 mode=abort）让在途 G 以
//     finish_reason=abort 返回 → ABORTED → 回 buffer；generate(2) 取走积压后补位先从 buffer 取回 G（不推进游标）。
//     依赖侧（上游 / 补丁）与 slime 源码分开标注。
// acc1（蓝）标入选、决定性判定与恢复路径；acc2（橙）标拒绝、abort、回收与代价。
//
// 用法：node tools/figs/svg/slime_rollout_admission_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  rolloutBatchSize: 2,
  overSamplingBatchSize: 4,
  nSamplesPerPrompt: 4,
  sglangServerConcurrency: 512,
  rolloutNumGpus: 4,
  rolloutNumGpusPerEngine: 2,
  asyncPool: 2,
});

// 示例假设：abort 生效时在途组 D（样本 12–15）各条请求在服务端已生成的 token 数。
export const D_TOKENS_AT_ABORT = Object.freeze([6, 4, 0, 8]);

// 流式示例：stream_interval 取 2（示例取值；上游 ServerArgs.stream_interval 默认 1），样本 12 的 token id 从 11 起。
export const STREAM = Object.freeze({ streamInterval: 2, firstTokenId: 11 });

// slime/utils/http_utils.py::get_rollout_num_engines 与 GenerateState.__init__ 的信号量容量
export function semaphoreCapacity(cfg = CFG) {
  const engines = cfg.rolloutNumGpus <= 0 ? 0 : Math.max(1, Math.floor(cfg.rolloutNumGpus / cfg.rolloutNumGpusPerEngine));
  return { engines, capacity: cfg.sglangServerConcurrency * engines };
}

// slime/rollout/filter_hub/dynamic_sampling_filters.py::check_reward_nonzero_std（float64 std > 1e-6）
export function checkRewardNonzeroStd(rewards) {
  const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  const variance = rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / (rewards.length - 1); // torch.std 默认无偏
  const keep = Math.sqrt(variance) > 1e-6;
  return { keep, reason: keep ? null : `zero_std_${Math.round(rewards[0] * 10) / 10}`, keepWhenInsufficient: false };
}

export function checkRewardNonzeroStdWithFallback(rewards) {
  return { ...checkRewardNonzeroStd(rewards), keepWhenInsufficient: true };
}

// slime/rollout/filter_hub/base_types.py::should_drop_dynamic_filter_output
export function shouldDrop(output, { remainingBatchSize, targetDataSize }) {
  if (output.keep) return false;
  if (output.keepWhenInsufficient && remainingBatchSize <= targetDataSize) return false;
  return true;
}

// slime/rollout/sglang_rollout.py::abort 的决策复现。
// inFlight：[{ group, tokensAtAbort: [...] , completed?: bool }]；abortMode 取自生成函数的 abort_mode 属性。
// - request 模式：_run_request_abortable_generate 登记的 task 被 cancel，CancelledError 被吞掉并标 ABORTED；
// - 其他：_run_server_abort_generate 计入 active_server_generations，>0 才 GET /workers 并 abort_servers_until_idle；
//   服务端以 finish_reason=abort 返回，Sample._apply_meta_info 把状态置 ABORTED。
// - partial 下只回收 any(status == ABORTED 且 response_length > 0) 的组。
export function abortPlan({ inFlight, abortMode = null, partialRollout = true }) {
  const request = abortMode === 'request';
  const running = inFlight.filter((g) => !g.completed).flatMap((g) => g.tokensAtAbort);
  const cancellableTasks = request ? running.length : 0;
  const activeServerGenerations = request ? 0 : running.length;
  const serverAbort = activeServerGenerations > 0;
  const groups = inFlight.map((g) => {
    const samples = g.tokensAtAbort.map((n) => ({ status: g.completed ? 'COMPLETED' : 'ABORTED', responseLength: n }));
    const recycled = partialRollout && samples.some((s) => s.status === 'ABORTED' && s.responseLength > 0);
    return { group: g.group, samples, recycled };
  });
  return { cancellableTasks, activeServerGenerations, serverAbort, groups };
}

// slime/rollout/sglang_rollout.py::generate_rollout_async + abort 的接收循环复现。
// `events` 是 group 完成顺序；每个 group 携带其 n 条 Sample 的 reward。
export function simulateAdmission({ cfg = CFG, groups, events, filter = checkRewardNonzeroStd, partialRollout = true, tokensAtAbort = {} }) {
  const target = cfg.rolloutBatchSize;
  const state = { remaining: 0, pendings: new Set(), aborted: false };
  const data = []; const all = []; const log = []; const waves = [];
  const names = Object.keys(groups);
  let submitCursor = 0;
  let tick = 0;
  const eventQueue = [...events];
  while (data.length < target) {
    while (state.remaining < target) {
      const wave = names.slice(submitCursor, submitCursor + cfg.overSamplingBatchSize);
      if (wave.length !== cfg.overSamplingBatchSize) throw new Error('示例事件不够再补一整波');
      submitCursor += wave.length;
      wave.forEach((g) => state.pendings.add(g));
      state.remaining += wave.length;
      waves.push({ tick, wave: [...wave] });
      log.push({ tick, action: `submit ${wave.join(' ')}`, remaining: state.remaining, accepted: data.length, pendings: state.pendings.size });
    }
    const next = eventQueue.find((g) => state.pendings.has(g));
    if (next === undefined) throw new Error('没有可完成的 pending group');
    eventQueue.splice(eventQueue.indexOf(next), 1);
    tick += 1;
    state.pendings.delete(next);
    all.push(next);
    const output = filter(groups[next].rewards);
    if (shouldDrop(output, { remainingBatchSize: state.remaining, targetDataSize: target })) {
      state.remaining -= 1;
      log.push({ tick, group: next, action: 'drop', reason: output.reason, remaining: state.remaining, accepted: data.length, pendings: state.pendings.size });
      continue;
    }
    if (data.length < target) {
      data.push(next);
      log.push({ tick, group: next, action: output.keep ? 'accept' : 'accept (keep_when_insufficient)', remaining: state.remaining, accepted: data.length, pendings: state.pendings.size });
    } else {
      log.push({ tick, group: next, action: 'discard (target already full; not stored back)', remaining: state.remaining, accepted: data.length, pendings: state.pendings.size });
    }
  }
  tick += 1;
  state.aborted = true;
  const inFlight = [...state.pendings];
  const plan = abortPlan({
    inFlight: inFlight.map((g) => ({ group: g, tokensAtAbort: tokensAtAbort[g] ?? Array(cfg.nSamplesPerPrompt).fill(1) })),
    partialRollout,
  });
  const recycled = plan.groups.filter((g) => g.recycled).map((g) => g.group);
  log.push({ tick, action: `abort: wait ${inFlight.length} pending`, inFlight, recycled, remaining: state.remaining, accepted: data.length, pendings: 0 });
  const sorted = [...data].sort((a, b) => groups[a].indices[0] - groups[b].indices[0]);
  return { target, data: sorted, all, inFlight, recycled, plan, waves, log, finalRemaining: state.remaining };
}

// 模拟 SGLang 的两种流式输出格式：累计模式每个 chunk 的 output_token_logprobs 是到目前为止的全表，
// 增量模式只含本段；两者的 output_token_logprobs_length 都是累计长度（上游 TokenizerManager.add_logprob_to_meta_info）。
export function streamChunks({ mode, interval, totalTokens, firstTokenId }) {
  const tokens = Array.from({ length: totalTokens }, (_, i) => firstTokenId + i);
  const chunks = [];
  let prev = 0;
  for (let end = Math.min(interval, totalTokens); prev < totalTokens; end = Math.min(end + interval, totalTokens)) {
    const wire = mode === 'cumulative' ? tokens.slice(0, end) : tokens.slice(prev, end);
    chunks.push({ wire, wireLen: wire.length, reported: end });
    prev = end;
  }
  return chunks;
}

// slime/rollout/streaming_utils.py::SGLangStreamAccumulator.add 的长度校验与追加规则
export function accumulate(mode, chunks) {
  let outputLength = 0;
  const tokens = [];
  const updates = [];
  for (const c of chunks) {
    const expected = mode === 'cumulative' ? c.wire.length : outputLength + c.wire.length;
    if (expected !== c.reported) {
      throw new Error(`SGLang ${mode} streaming output has inconsistent output_token_logprobs_length`);
    }
    if (c.reported < outputLength) throw new Error('SGLang cumulative streaming output length decreased');
    const fresh = mode === 'cumulative' ? c.wire.slice(outputLength) : c.wire;
    tokens.push(...fresh);
    outputLength = c.reported;
    updates.push({ newTokens: [...fresh], responseLength: tokens.length });
  }
  return { tokens, updates };
}

// slime/rollout/fully_async_rollout.py::AsyncRolloutWorker._loop / _make_done_cb / _generate_rollout_async 的 tick 复现，
// 加上 driver 的 update_weights：pause 时在途请求以 finish_reason=abort 返回，done-callback 把含 ABORTED 的组
// 交回 data_buffer.add_samples；RolloutDataSourceWithBuffer.get_samples 先从 buffer（pop_first）取，取自 buffer 时不推进游标。
// 每个 tick 的顺序：pause → 完成入队 → 消费者 drain（_generate_rollout_async 每 0.05 s 轮询，先于 1 s 一次的补位）→ 补位。
// consumers[i] = { rollout, startTick }：generate(rollout) 从 startTick 起在途，取够 target 组即返回。
export function simulateFullyAsync({ cfg = CFG, newGroups, completions, consumers, pauseAt = null, resumeAt = null, lastTick, target = CFG.rolloutBatchSize }) {
  const cap = cfg.asyncPool;
  let active = [];
  const queue = []; const buffer = []; const ticks = [];
  let cursor = 0;
  const collected = Object.fromEntries(consumers.map((c) => [c.rollout, []]));
  for (let t = 0; t <= lastTick; t += 1) {
    const paused = pauseAt !== null && t >= pauseAt && (resumeAt === null || t < resumeAt);
    // 1. pause(mode=abort)：在途请求结束为 ABORTED，callback 回 buffer，不进输出队列
    let requeued = [];
    if (t === pauseAt) {
      requeued = [...active];
      buffer.push(...requeued);
      active = [];
    }
    // 2. 本 tick 正常完成的组由 done-callback 入队（无界队列）
    const done = completions[t] ?? [];
    for (const g of done) { active.splice(active.indexOf(g), 1); queue.push(g); }
    // 3. 在途的 generate 调用按缺口 drain
    const drained = [];
    const consumerActive = consumers.some((c) => t >= c.startTick && collected[c.rollout].length < target);
    for (const c of consumers) {
      if (t < c.startTick || collected[c.rollout].length >= target) continue;
      const got = queue.splice(0, target - collected[c.rollout].length);
      collected[c.rollout].push(...got);
      if (got.length) drained.push({ rollout: c.rollout, groups: got, complete: collected[c.rollout].length >= target });
    }
    // 4. 补位：池未满且 qsize < cap；先取 buffer（不推进游标），再按游标取新组
    const topped = []; const fromBuffer = [];
    while (active.length < cap && queue.length < cap) {
      let g;
      if (buffer.length) { g = buffer.shift(); fromBuffer.push(g); } else if (cursor < newGroups.length) { g = newGroups[cursor]; cursor += 1; } else break;
      active.push(g); topped.push(g);
    }
    const started = consumers.filter((c) => c.startTick === t).map((c) => c.rollout);
    ticks.push({
      tick: t, done: [...done], requeued, topped, fromBuffer, cursor, active: [...active], queue: [...queue], buffer: [...buffer],
      gateClosed: queue.length >= cap, paused, drained, consumerActive, started, resumed: t === resumeAt,
    });
  }
  return { cap, ticks, collected, queueLeft: [...queue] };
}

export function model(cfg = CFG) {
  const n = cfg.nSamplesPerPrompt;
  const mk = (name, first, rewards) => [name, { indices: Array.from({ length: n }, (_, i) => first + i), rewards }];
  const groupsMain = Object.fromEntries([
    mk('A', 0, [1, 0, 0, 1]), mk('B', 4, [1, 1, 1, 1]), mk('C', 8, [0, 0, 1, 0]), mk('D', 12, [1, 0, 1, 1]),
  ]);
  const main = simulateAdmission({ cfg, groups: groupsMain, events: ['B', 'A', 'C', 'D'], tokensAtAbort: { D: [...D_TOKENS_AT_ABORT] } });
  // 反例：同一 D 若四条都没拿到 token，或在 abort 前恰好整组完成，都不回 buffer
  const dNoTokens = abortPlan({ inFlight: [{ group: 'D', tokensAtAbort: [0, 0, 0, 0] }] });
  const dCompleted = abortPlan({ inFlight: [{ group: 'D', tokensAtAbort: [...D_TOKENS_AT_ABORT], completed: true }] });

  const groupsCascade = Object.fromEntries([
    mk('A', 0, [0, 0, 0, 0]), mk('B', 4, [1, 1, 1, 1]), mk('C', 8, [1, 1, 1, 1]), mk('D', 12, [1, 0, 1, 1]),
    mk('E', 16, [0, 1, 0, 0]), mk('F', 20, [1, 1, 0, 0]), mk('G', 24, [1, 0, 0, 0]), mk('H', 28, [0, 0, 1, 1]),
  ]);
  const cascade = simulateAdmission({ cfg, groups: groupsCascade, events: ['B', 'A', 'C', 'D', 'E', 'F', 'G', 'H'] });
  const fallback = simulateAdmission({ cfg, groups: groupsCascade, events: ['B', 'A', 'C', 'D', 'E', 'F', 'G', 'H'], filter: checkRewardNonzeroStdWithFallback });

  // 流式：同一 D 组、样本 12，abort 前观测到的 token 数取 D_TOKENS_AT_ABORT[0]
  const s12 = D_TOKENS_AT_ABORT[0];
  const cumulativeChunks = streamChunks({ mode: 'cumulative', interval: STREAM.streamInterval, totalTokens: s12, firstTokenId: STREAM.firstTokenId });
  const incrementalChunks = streamChunks({ mode: 'incremental', interval: STREAM.streamInterval, totalTokens: s12, firstTokenId: STREAM.firstTokenId });
  const stream = {
    cumulativeChunks,
    incrementalChunks,
    cumulative: accumulate('cumulative', cumulativeChunks),
    incremental: accumulate('incremental', incrementalChunks),
    cumulativePairs: cumulativeChunks.reduce((a, c) => a + c.wireLen, 0),
    incrementalPairs: incrementalChunks.reduce((a, c) => a + c.wireLen, 0),
    abort: abortPlan({ inFlight: [{ group: 'D', tokensAtAbort: [...D_TOKENS_AT_ABORT] }], abortMode: 'request' }),
  };

  // driver（train_async.py::train，--update-weights-interval 1）：generate(0) 在 t0 提交；取够即返回并立刻提交 generate(1)；
  // generate(1) 取够后 driver 进入训练 / 更新，到 update_weights 结束、提交 generate(2) 之前没有消费者。
  const fullyAsync = simulateFullyAsync({
    cfg,
    newGroups: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    completions: { 1: ['A'], 2: ['B'], 3: ['C'], 4: ['D'], 5: ['E'], 6: ['F'] },
    consumers: [{ rollout: 0, startTick: 0 }, { rollout: 1, startTick: 2 }, { rollout: 2, startTick: 8 }],
    pauseAt: 7,
    resumeAt: 8,
    lastTick: 8,
  });
  return { cfg, semaphore: semaphoreCapacity(cfg), groupsMain, main, dNoTokens, dCompleted, cascade, fallback, stream, fullyAsync };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .tag{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:10.5px;font-weight:700;fill:#2563EB}
  .tagx{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:10.5px;font-weight:700;fill:#8A919C}
  .mono{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:11px;fill:#38414D}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.1;stroke-dasharray:4 3}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .bar{fill:#E8ECF2;stroke:#AEB6C2;stroke-width:1}
  .tick{stroke:#D9DEE7;stroke-width:1;stroke-dasharray:3 3}
`;

function canvas(W, H, title, desc) {
  const o = [];
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  const rect = (x, y, w, h, cls = 'neutral', r = 7) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const line = (x1, y1, x2, y2, cls = 'tick') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  const done = () => { o.push('</svg>'); return o.join('\n'); };
  return { rect, text, line, done };
}

const BASELINE = 'THUDM/slime@4c193f1f3750';

export function renderAdmission(m) {
  const W = 1180; const H = 858;
  const { rect, text, line, done } = canvas(W, H,
    'slime rollout 接收循环：四个候选组如何得到两个训练组，abort 时在途组怎样回收',
    '上方时间线展示 over-sampling 提交、FIRST_COMPLETED 接收、动态过滤拒绝、目标满后按 abort_mode 走服务端 abort 与 partial 回收判定，以及 remaining_batch_size 与 len(data) 两个计数器；下方展示连续拒绝触发补采与 with_fallback 兜底。');
  const c = m.cfg;

  text(24, 34, '默认接收循环：四个候选组、两个计数器、一次 abort', 'ti');
  text(24, 56, `rollout_batch_size=${c.rolloutBatchSize} · over_sampling_batch_size=${c.overSamplingBatchSize} · n_samples_per_prompt=${c.nSamplesPerPrompt} · 信号量 = sglang_server_concurrency ${c.sglangServerConcurrency} × ${m.semaphore.engines} engines = ${m.semaphore.capacity}（本例 ${c.overSamplingBatchSize * c.nSamplesPerPrompt} 个请求不受限）`, 'su');

  // ---- 面板 1：默认循环时间线 ----
  const y1 = 76; const p1H = 430;
  rect(24, y1, 1132, p1H, 'panel', 10);
  text(40, y1 + 24, `generate_rollout_async：接收单位是整组（${c.nSamplesPerPrompt} 条 Sample 全部生成并打完 reward），FIRST_COMPLETED 取最先完成的组`, 'pt');
  const main = m.main;
  const tx0 = 170; const colW = 122; const rowH = 34; const rowsY = y1 + 60;
  const groupNames = Object.keys(m.groupsMain);
  main.log.forEach((l, i) => {
    const x = tx0 + i * colW;
    text(x + colW / 2, rowsY - 12, `t${l.tick}`, 'sm', 'middle');
    line(x, rowsY - 4, x, rowsY + groupNames.length * rowH + 4);
  });
  const doneCol = {}; main.log.forEach((l, i) => { if (l.group) doneCol[l.group] = i; });
  groupNames.forEach((g, r) => {
    const y = rowsY + r * rowH;
    const idx = m.groupsMain[g].indices;
    text(tx0 - 10, y + 15, `${g} · 样本 ${idx[0]}–${idx.at(-1)}`, 'sm', 'end');
    text(tx0 - 10, y + 28, `reward ${m.groupsMain[g].rewards.join(' ')}`, 'sm', 'end');
    const endCol = doneCol[g] ?? main.log.length - 1;
    const decision = main.log[endCol]?.group === g ? main.log[endCol].action : 'in flight → abort';
    const cls = decision === 'accept' ? 'acc1' : 'acc2';
    rect(tx0 + 4, y + 4, (endCol + 1) * colW - 8, 22, 'bar', 4);
    rect(tx0 + endCol * colW + 8, y + 4, colW - 16, 22, cls, 4);
    const label = decision === 'accept' ? '完成 → 通过' : decision === 'drop' ? '完成 → 拒绝' : main.recycled.includes(g) ? '在途 → 回收' : '在途 → 丢弃';
    text(tx0 + endCol * colW + colW / 2, y + 19, label, 'sm', 'middle');
  });
  const cy = rowsY + groupNames.length * rowH + 16;
  const counters = [
    ['remaining_batch_size', main.log.map((l) => l.remaining)],
    ['len(data)', main.log.map((l) => l.accepted)],
    ['pendings', main.log.map((l) => l.pendings)],
  ];
  counters.forEach((cnt, r) => {
    const y = cy + r * 26;
    text(tx0 - 10, y + 16, cnt[0], 'mono', 'end');
    cnt[1].forEach((v, i) => {
      const x = tx0 + i * colW;
      const key = (r === 0 && i > 0 && v < cnt[1][i - 1]) || (r === 1 && i > 0 && v > cnt[1][i - 1]);
      rect(x + 8, y, colW - 16, 22, key ? 'acc1' : 'cell', 4);
      text(x + colW / 2, y + 15, v, 'mono', 'middle');
    });
  });
  const ey = cy + 3 * 26 + 10;
  main.log.forEach((l, i) => {
    const x = tx0 + i * colW;
    const lines = l.action.startsWith('submit') ? ['提交一整波', `${c.overSamplingBatchSize} 组 → pendings`]
      : l.action === 'drop' ? ['zero-std 拒绝', 'remaining −1']
        : l.action === 'accept' ? ['入选', 'len(data) +1']
          : ['目标满 → abort', `等 ${l.inFlight.length} 个 pending`];
    lines.forEach((s, k) => text(x + colW / 2, ey + 12 + k * 14, s, 'sm', 'middle'));
  });

  const rx = 810; const ry0 = y1 + 40; const rw = 330;
  rect(rx, ry0, rw, 176, 'ghost', 8);
  text(rx + 12, ry0 + 20, '两个计数器的分工', 'pt');
  text(rx + 12, ry0 + 40, 'remaining_batch_size：已提交且未被拒的候选组；', 'tx');
  text(rx + 12, ry0 + 57, `低于 target ${c.rolloutBatchSize} 时再补一整波 ${c.overSamplingBatchSize} 组，不只补缺口。`, 'tx');
  text(rx + 12, ry0 + 77, `len(data)：已入选组数；达到 ${c.rolloutBatchSize} 即退出并 abort。`, 'tx');
  text(rx + 12, ry0 + 97, `B 被拒：remaining ${main.log[0].remaining} → ${main.log[1].remaining}，仍 ≥ ${c.rolloutBatchSize}，不补采。`, 'tx');
  text(rx + 12, ry0 + 117, `入选 ${main.data.join('、')}，按 group[0].index 排序后交训练。`, 'tx');
  text(rx + 12, ry0 + 137, '超出目标的已完成组不训练也不回收（源码 NOTE）。', 'tx');
  text(rx + 12, ry0 + 160, `信号量 ${m.semaphore.capacity} 按请求数限流，不按 token 或 KV 预算。`, 'cap');
  rect(rx, ry0 + 188, rw, 96, 'acc2', 8);
  text(rx + 12, ry0 + 208, 'abort 按生成函数的 abort_mode 分两路：', 'tx');
  text(rx + 12, ry0 + 225, '默认 generate：对默认 router 全部 worker 发', 'tx');
  text(rx + 12, ry0 + 242, 'abort_all，查 /v1/loads 到 0（查询异常只告警）；', 'tx');
  text(rx + 12, ry0 + 259, 'abort_mode="request"：取消 task，不查负载。', 'tx');
  text(rx + 12, ry0 + 276, '两路都在等本地 pendings 返回后才回收。', 'tx');

  // t4 abort 细节条（整宽）
  const plan = main.plan;
  const d = plan.groups.find((g) => g.group === 'D');
  const sy = y1 + 340;
  rect(40, sy, 1100, 78, 'acc2', 8);
  const lastCol = main.log.length - 1;
  text(56, sy + 20, `t${main.log[lastCol].tick} abort 细节（默认 generate，未设 abort_mode）：active_server_generations = ${plan.activeServerGenerations} → GET /workers → 每个 worker 发 abort_all 并轮询 /v1/loads 到 0`, 'tx');
  text(56, sy + 40, `D 的 ${d.samples.length} 条 /generate 以 finish_reason=abort 返回 → Sample._apply_meta_info 置 ABORTED；response_length = ${d.samples.map((s) => s.responseLength).join(' ')}（示例假设）`, 'tx');
  text(56, sy + 60, `partial 回收判定 any(ABORTED 且 response_length > 0) = ${d.recycled ? '是' : '否'} → D 整组回 buffer；四条都是 0 → ${m.dNoTokens.groups[0].recycled ? '回收' : '不回收'}；abort 前已整组完成 → ${m.dCompleted.groups[0].recycled ? '回收' : '不回收'}`, 'tx');

  // ---- 面板 2：两条变体 ----
  const y2 = y1 + p1H + 14; const p2H = 300;
  rect(24, y2, 1132, p2H, 'panel', 10);
  text(40, y2 + 24, '同一循环的两条过滤变体（8 个候选组，A B C 均为零方差）', 'pt');
  const variantRow = (yy, title, sim, notes) => {
    text(40, yy, title, 'tx');
    const cw = 70;
    sim.log.forEach((l, i) => {
      const x = 40 + i * (cw + 4);
      const cls = l.action === 'drop' ? 'acc2' : l.action.startsWith('accept') ? 'acc1' : l.action.startsWith('submit') ? 'cell' : 'ghost';
      rect(x, yy + 8, cw, 46, cls, 4);
      const head = l.action.startsWith('submit') ? `+${c.overSamplingBatchSize}` : l.action.startsWith('abort') ? 'abort' : `${l.group}`;
      const tag = l.action === 'drop' ? '拒' : l.action === 'accept' ? '入' : l.action.startsWith('accept') ? '兜底入' : '';
      text(x + cw / 2, yy + 24, `${head} ${tag}`.trim(), 'sm', 'middle');
      text(x + cw / 2, yy + 40, `r=${l.remaining} d=${l.accepted}`, 'mono', 'middle');
    });
    notes.forEach((nt, k) => text(40, yy + 72 + k * 16, nt, 'cap'));
  };
  const cas = m.cascade;
  const secondWave = cas.waves[1];
  variantRow(y2 + 50, `① check_reward_nonzero_std：连续拒绝 → remaining 低于 ${c.rolloutBatchSize} → 再提交一整波`, cas, [
    `B、A、C 依次被拒，remaining ${cas.log[1].remaining} → ${cas.log[2].remaining} → ${cas.log[3].remaining}；t${secondWave.tick} 再提交 ${secondWave.wave.join(' ')}（+${c.overSamplingBatchSize}）`,
    `随后 ${cas.data.join('、')} 入选；abort 时 ${cas.inFlight.length} 组在途，代价是多付一整波生成与 RM。`,
  ]);
  const fb = m.fallback;
  variantRow(y2 + 168, '② check_reward_nonzero_std_with_fallback：remaining ≤ target 时保留零方差组', fb, [
    `B、A 被拒后 remaining=${fb.log[2].remaining} ≤ ${c.rolloutBatchSize}，C 虽零方差仍入选（keep_when_insufficient）`,
    `${fb.data.join('、')} 凑满，不再补采，abort 时 ${fb.inFlight.length} 组在途；代价是送进一个无梯度信号的组。`,
  ]);
  const nx = 690; const nw = 450;
  rect(nx, y2 + 44, nw, 214, 'ghost', 8);
  text(nx + 12, y2 + 64, 'abort 之后的轮末处理', 'pt');
  text(nx + 12, y2 + 86, 'state.reset() 清空 remaining、pendings、aborted、', 'tx');
  text(nx + 12, y2 + 103, 'cancellable_tasks、active_server_generations。', 'tx');
  text(nx + 12, y2 + 125, '入选组按 index 排序 → rollout_sample_filter_path 原地处理；', 'tx');
  text(nx + 12, y2 + 142, 'rollout_all_samples_process_path 看到全部已完成组', 'tx');
  text(nx + 12, y2 + 159, '（含被拒者，不含 abort 时的在途组）。', 'tx');
  text(nx + 12, y2 + 181, '未接 dynamic_sampling_filter_path 时 call_dynamic_filter', 'tx');
  text(nx + 12, y2 + 198, '恒返回 keep=True，循环退化为"取够先完成的组"。', 'tx');
  text(nx + 12, y2 + 220, '回收组由 generate_rollout 交 data_source.add_samples，', 'tx');
  text(nx + 12, y2 + 237, '下一轮 buffer 先取（pop_first）。', 'tx');
  text(40, y2 + 286, '蓝：入选与决定性判定；橙：拒绝、abort 与回收。灰条为在途时间；方块内 r/d 为事件后的 remaining_batch_size / len(data)。', 'cap');

  text(24, H - 14, `源码基线：${BASELINE} · 复现 generate_rollout_async / abort / should_drop_dynamic_filter_output / check_reward_nonzero_std(_with_fallback)`, 'su');
  return done();
}

export function renderVariants(m) {
  const W = 1180; const H = 942;
  const { rect, text, done } = canvas(W, H,
    'slime rollout 变体数据面：流式内层调用与 fully-async 后台池的同例回放',
    '上方面板回放在途组 D 的样本 12 在流式内层调用中逐 chunk 落到 Sample，比较累计与增量两种 SGLang 流式格式、request 模式 abort 与非流式对照；下方面板回放 fully-async 后台池的补位、qsize 闸门、按缺口 drain，以及权重更新 pause 让在途组以 ABORTED 回 buffer 的路径，并标出 slime 源码、上游 SGLang 与补丁的边界。');
  const c = m.cfg; const s = m.stream; const fa = m.fullyAsync;

  text(24, 34, '同一在途组的两条变体数据面：流式内层调用与 fully-async 后台池', 'ti');
  text(24, 56, `沿用图 1 的在途组 D（样本 12–15，abort 时服务端已生成 ${D_TOKENS_AT_ABORT.join(' ')} 个 token）；stream_interval=${STREAM.streamInterval}（示例取值，上游默认 1）；fully-async 池容量 ${fa.cap}`, 'su');

  // ---- 面板 A：流式 ----
  const yA = 76; const hA = 340;
  rect(24, yA, 1132, hA, 'panel', 10);
  text(40, yA + 24, 'A. --custom-generate-function-path …generate_streaming：样本 12 的逐 chunk 回放（abort_mode="request"）', 'pt');
  const lx = 262; const colW = 116; const top = yA + 44; const rh = 34;
  const nChunks = s.cumulativeChunks.length;
  const heads = [...s.cumulativeChunks.map((_, i) => `chunk ${i + 1}`), 't4 abort'];
  heads.forEach((h, i) => text(lx + i * colW + colW / 2, top + 14, h, 'sm', 'middle'));
  const rows = [
    ['wire · 累计模式（默认）', s.cumulativeChunks.map((ch) => `${ch.wireLen} 对 · len ${ch.reported}`), '—', 'cell'],
    ['wire · 增量模式', s.incrementalChunks.map((ch) => `${ch.wireLen} 对 · len ${ch.reported}`), '—', 'cell'],
    ['Accumulator 追加的新 token', s.cumulative.updates.map((u) => u.newTokens.join(' ')), '—', 'cell'],
    ['流式 Sample.response_length', s.cumulative.updates.map((u) => `${u.responseLength}`), `${s.cumulative.tokens.length} · ABORTED`, 'acc1'],
    ['非流式 generate 对照', s.cumulative.updates.map(() => '0'), `${D_TOKENS_AT_ABORT[0]} · ABORTED`, 'acc2'],
  ];
  rows.forEach(([label, cells, last, lastCls], r) => {
    const y = top + 24 + r * rh;
    text(lx - 10, y + 18, label, 'tx', 'end');
    [...cells, last].forEach((v, i) => {
      const cls = i === nChunks ? lastCls : 'cell';
      rect(lx + i * colW + 4, y, colW - 8, 26, cls, 4);
      text(lx + i * colW + colW / 2, y + 17, v, 'mono', 'middle');
    });
  });
  const capY = top + 24 + rows.length * rh + 18;
  text(40, capY, `累计模式 ${nChunks} 个 chunk 共传 ${s.cumulativePairs} 对 logprob，增量模式 ${s.incrementalPairs} 对；两种模式 output_token_logprobs_length 都是累计长度，Accumulator 只追加新增部分。`, 'cap');
  text(40, capY + 18, '--sglang-incremental-streaming-output 必须与服务端一致：配置与实际格式不符时第 2 个 chunk 的长度校验抛 ValueError。', 'cap');
  text(40, capY + 36, '非流式对照：abort JSON 返回前 Sample 上没有 token；返回时带回服务端已生成的全部 token，finish_reason=abort → ABORTED。', 'cap');
  text(40, capY + 54, '流式的 partial 回收点是最后一个已观测 chunk；abort 生效与下一个 chunk 之间服务端新生成的 token 不在 Sample 上。', 'cap');

  const bx = 850; const bw = 290;
  rect(bx, yA + 44, bw, 136, 'acc1', 8);
  text(bx + 12, yA + 64, 'abort()：request 模式', 'pt');
  text(bx + 12, yA + 84, `cancellable_tasks = ${s.abort.cancellableTasks} → task.cancel()`, 'tx');
  text(bx + 12, yA + 101, `active_server_generations = ${s.abort.activeServerGenerations}`, 'tx');
  text(bx + 12, yA + 118, '→ 不 GET /workers、不发 abort_all', 'tx');
  text(bx + 12, yA + 135, 'CancelledError 被吞 → ABORTED，保留前缀', 'tx');
  text(bx + 12, yA + 152, `D 组 any(ABORTED 且 len>0) → ${s.abort.groups[0].recycled ? '回 buffer' : '不回收'}`, 'tx');
  text(bx + 12, yA + 169, '整轮不再查 /v1/loads 空闲证据', 'tx');
  rect(bx, yA + 190, bw, 106, 'acc2', 8);
  text(bx + 12, yA + 210, '代价与边界', 'pt');
  text(bx + 12, yA + 229, '终止 chunk 才带 top-p ids 与 routed experts', 'tx');
  text(bx + 12, yA + 246, '→ 取消后 Sample 上缺这两类元数据', 'tx');
  text(bx + 12, yA + 263, 'client.stream 直连，不走 _post 的 60 次重试', 'tx');
  text(bx + 12, yA + 280, '流结束缺 finish_reason → RuntimeError', 'tx');

  // ---- 面板 B：fully-async ----
  const yB = yA + hA + 14; const hB = 480;
  rect(24, yB, 1132, hB, 'panel', 10);
  text(40, yB + 24, `B. fully-async 后台池：池容量 ${fa.cap}、qsize 闸门 ${fa.cap}、按缺口 drain；driver 的 update_weights 让在途组以 ABORTED 回 buffer`, 'pt');
  const th = ['tick', 'driver / 服务端事件', '完成入队', '回 buffer', '补位', 'active', 'queue', '闸门', 'drain'];
  const cws = [40, 196, 70, 70, 62, 66, 70, 44, 104];
  let hx = 40;
  th.forEach((h, i) => { text(hx + cws[i] / 2, yB + 52, h, 'sm', 'middle'); hx += cws[i]; });
  const eventOf = (t) => {
    if (t.requeued.length) return 'update_weights → pause(abort)';
    if (t.resumed) return `continue → 提交 generate(${t.started[0]})`;
    if (t.tick === 0) return `提交 generate(${t.started[0]})，建池补满`;
    const dr = t.drained.find((x) => x.complete);
    if (dr) return t.started.length ? `generate(${dr.rollout}) 返回 → 提交 (${t.started[0]})` : `generate(${dr.rollout}) 取够返回`;
    if (t.drained.length) return `generate(${t.drained[0].rollout}) 边等边取`;
    if (!t.consumerActive) return t.gateClosed ? '无消费者，qsize 达闸门' : '无消费者，完成组积压';
    return '—';
  };
  fa.ticks.forEach((t, r) => {
    const yy = yB + 62 + r * 38;
    let x = 40;
    const drainTxt = t.drained.length ? t.drained.map((dd) => `r${dd.rollout}←${dd.groups.join(' ')}`).join(' ') : '—';
    const topped = t.topped.length ? t.topped.map((g) => (t.fromBuffer.includes(g) ? `${g}↺` : g)).join(' ') + (t.paused ? '（等）' : '') : '—';
    const cells = [
      `t${t.tick}`, eventOf(t), t.done.length ? t.done.join(' ') : '—', t.requeued.length ? t.requeued.join(' ') : '—', topped,
      t.active.length ? t.active.join(' ') : '—', `[${t.queue.join(' ')}]`, t.gateClosed ? '关' : '开', drainTxt,
    ];
    cells.forEach((v, i) => {
      let cls = 'cell';
      if (i === 7 && t.gateClosed) cls = 'acc2';
      if (i === 3 && t.requeued.length) cls = 'acc2';
      if (i === 1 && t.requeued.length) cls = 'acc2';
      if (i === 8 && t.drained.some((dd) => dd.complete)) cls = 'acc1';
      if (i === 4 && t.fromBuffer.length) cls = 'acc1';
      rect(x, yy, cws[i] - 4, 30, cls, 4);
      text(x + (cws[i] - 4) / 2, yy + 19, v, i === 1 ? 'sm' : 'mono', 'middle');
      x += cws[i];
    });
  });
  const tableEnd = yB + 62 + fa.ticks.length * 38;
  const gate = fa.ticks.find((x) => x.gateClosed);
  const pause = fa.ticks.find((x) => x.requeued.length);
  const back = fa.ticks.find((x) => x.fromBuffer.length);
  text(40, tableEnd + 18, `generate 在途时边等边取，队列不积压；t${gate.tick} 无消费者时队列 [${gate.queue.join(' ')}] 达闸门，只停补位，在途 ${gate.active.join(' ')} 照常运行。`, 'cap');
  text(40, tableEnd + 36, `t${pause.tick} pause 让在途 ${pause.requeued.join(' ')} 回 buffer；t${back.tick} generate(2) 取走 [${gate.queue.join(' ')}] 后补位：${back.fromBuffer.join(' ')}↺ 取自 buffer、不推进游标，其后按游标取新组。`, 'cap');
  const fx = 790; const fw = 350;
  rect(fx, yB + 40, fw, 226, 'ghost', 8);
  text(fx + 12, yB + 60, `t${fa.ticks.find((x) => x.requeued.length).tick} 的 ABORTED 从哪里来（默认 generate）`, 'pt');
  const chain = [
    ['slime', 'SGLangEngine.pause_generation：POST {}'],
    ['上游', 'PauseGenerationReqInput.mode 默认 "abort"'],
    ['上游', 'pause 循环 abort_request(abort_all=True)'],
    ['上游', '在途请求以 finish_reason=abort 正常返回'],
    ['slime', 'Sample._apply_meta_info：case "abort"'],
    ['slime', 'generate_and_rm 跳过 RM，组内有 ABORTED'],
    ['slime', '_make_done_cb → data_buffer.add_samples'],
  ];
  chain.forEach(([tag, body], k) => {
    const yy = yB + 70 + k * 26;
    rect(fx + 10, yy, fw - 20, 22, tag === '上游' ? 'dep' : 'cell', 4);
    text(fx + 18, yy + 15, tag, tag === '上游' ? 'tagx' : 'tag');
    text(fx + 58, yy + 15, body, 'mono');
  });
  rect(fx, yB + 276, fw, 84, 'acc2', 8);
  text(fx + 12, yB + 296, 'fully-async 从不调用 sglang_rollout.abort，', 'tx');
  text(fx + 12, yB + 313, 'GenerateState.aborted 恒为 False；没有 pause、', 'tx');
  text(fx + 12, yB + 330, '外部 /abort_request，也没有服务端以 abort', 'tx');
  text(fx + 12, yB + 347, '结束的请求时，回 buffer 分支到不了。', 'tx');
  text(40, yB + hB - 12, '实线框：slime 源码；虚线框「上游」：sgl-project/sglang v0.5.15.post1 源码（静态阅读，slime 补丁未改这段语义）。蓝：取够 / 恢复；橙：闸门、pause 与回 buffer。', 'cap');

  text(24, H - 14, `源码基线：${BASELINE} · 复现 SGLangStreamAccumulator.add / abort / AsyncRolloutWorker._loop / _make_done_cb；上游 sglang@0b3bb0cbe318`, 'su');
  return done();
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
export const OUTPUTS = Object.freeze({
  admission: 'slime_rollout_admission_timeline.svg',
  variants: 'slime_rollout_variant_planes.svg',
});
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  const m = model();
  writeFileSync(join(outputDir, OUTPUTS.admission), `${renderAdmission(m)}\n`, 'utf8');
  writeFileSync(join(outputDir, OUTPUTS.variants), `${renderVariants(m)}\n`, 'utf8');
}
