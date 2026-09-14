// 图：slime 默认 rollout 接收循环（`generate_rollout_async`）怎样用 over-sampling、FIRST_COMPLETED、
// 动态过滤与 abort 从四个候选 prompt group 得到两个训练 group；以及 fully-async 生产者
// （`AsyncRolloutWorker._loop`）的池容量、qsize 闸门与按缺口 drain。
// 源码基线：THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：
//  1. 接收单位是整组：一组内 n 条 Sample 全部生成并打完 reward 后，才进入 FIRST_COMPLETED 的接收判断。
//  2. 两个计数器分工：remaining_batch_size 统计"已提交且尚未被拒"的候选组，决定何时再补一整波
//     over_sampling_batch_size；len(data) 统计已入选组，达到 rollout_batch_size 即停止并 abort。
//  3. abort 是粗粒度排空：对默认 router 全部 worker 发 abort_all，再等 pending task 返回；
//     只有开 partial_rollout 时在途组才交回 DataSource，多余的已完成组既不训练也不回收。
//  4. fully-async 的池容量与 qsize 闸门是两个不同的上限：闸门只阻止补位，不阻止在途任务完成入队。
//
// 布局：上方一个大面板画默认循环的时间线（四个候选组各一行，右侧决策说明，下方三条计数器）；
// 下方左侧两条变体条（连续拒绝触发补采波次 / with_fallback 在 remaining ≤ target 时保留零方差组），
// 下方右侧 fully-async 的 tick 表。acc1 标入选组与决定性判定，acc2 标被拒、abort 与回收。
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

// slime/utils/http_utils.py::get_rollout_num_engines 与 GenerateState.__init__ 的信号量容量
export function semaphoreCapacity(cfg = CFG) {
  const engines = Math.max(1, Math.floor(cfg.rolloutNumGpus / cfg.rolloutNumGpusPerEngine));
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

// slime/rollout/sglang_rollout.py::generate_rollout_async + abort 的接收循环复现。
// `events` 是 group 完成顺序；每个 group 携带其 n 条 Sample 的 reward。
export function simulateAdmission({ cfg = CFG, groups, events, filter = checkRewardNonzeroStd, partialRollout = true }) {
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
  const recycled = partialRollout ? inFlight : [];
  log.push({ tick, action: `abort: abort_all → /v1/loads → wait ${inFlight.length} pending`, inFlight, recycled, remaining: state.remaining, accepted: data.length, pendings: 0 });
  const sorted = [...data].sort((a, b) => groups[a].indices[0] - groups[b].indices[0]);
  return { target, data: sorted, all, inFlight, recycled, waves, log, finalRemaining: state.remaining };
}

// slime/rollout/fully_async_rollout.py::AsyncRolloutWorker._loop / _make_done_cb / _generate_rollout_async 的 tick 复现。
// `completions[tick]` 是该 tick 完成的 group；`drainAt` 是消费者取数的 tick。
export function simulateFullyAsync({ cfg = CFG, groupNames, completions, drainAt, target = CFG.rolloutBatchSize }) {
  const cap = cfg.asyncPool;
  const active = []; const queue = []; const ticks = [];
  let next = 0; let drained = null;
  for (let t = 0; t <= drainAt; t += 1) {
    // 1. reap：本 tick 完成的 group 由 done-callback 入队（无界队列，永不阻塞 loop）
    const done = completions[t] ?? [];
    for (const g of done) { active.splice(active.indexOf(g), 1); queue.push(g); }
    // 2. top-up：池未满且 qsize < cap 才补位
    const topped = [];
    while (active.length < cap && queue.length < cap && next < groupNames.length) { active.push(groupNames[next]); topped.push(groupNames[next]); next += 1; }
    // 3. 消费者在 drainAt 只取 target - collected 个
    if (t === drainAt) drained = queue.splice(0, target);
    ticks.push({ tick: t, done: [...done], topped, active: [...active], queue: [...queue], gateClosed: queue.length >= cap, drained: t === drainAt ? [...drained] : null });
  }
  return { cap, ticks, drained, queueLeft: [...queue] };
}

export function model(cfg = CFG) {
  const n = cfg.nSamplesPerPrompt;
  const mk = (name, first, rewards) => [name, { indices: Array.from({ length: n }, (_, i) => first + i), rewards }];
  const groupsMain = Object.fromEntries([
    mk('A', 0, [1, 0, 0, 1]), mk('B', 4, [1, 1, 1, 1]), mk('C', 8, [0, 0, 1, 0]), mk('D', 12, [1, 0, 1, 1]),
  ]);
  const main = simulateAdmission({ cfg, groups: groupsMain, events: ['B', 'A', 'C', 'D'] });

  const groupsCascade = Object.fromEntries([
    mk('A', 0, [0, 0, 0, 0]), mk('B', 4, [1, 1, 1, 1]), mk('C', 8, [1, 1, 1, 1]), mk('D', 12, [1, 0, 1, 1]),
    mk('E', 16, [0, 1, 0, 0]), mk('F', 20, [1, 1, 0, 0]), mk('G', 24, [1, 0, 0, 0]), mk('H', 28, [0, 0, 1, 1]),
  ]);
  const cascade = simulateAdmission({ cfg, groups: groupsCascade, events: ['B', 'A', 'C', 'D', 'E', 'F', 'G', 'H'] });
  const fallback = simulateAdmission({ cfg, groups: groupsCascade, events: ['B', 'A', 'C', 'D', 'E', 'F', 'G', 'H'], filter: checkRewardNonzeroStdWithFallback });

  const fullyAsync = simulateFullyAsync({
    cfg, groupNames: ['A', 'B', 'C', 'D', 'E'], completions: { 1: ['A'], 2: ['B'], 3: ['C'] }, drainAt: 3,
  });
  return { cfg, semaphore: semaphoreCapacity(cfg), groupsMain, main, cascade, fallback, fullyAsync };
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
  .mono{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:11px;fill:#38414D}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .bar{fill:#E8ECF2;stroke:#AEB6C2;stroke-width:1}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
  .tick{stroke:#D9DEE7;stroke-width:1;stroke-dasharray:3 3}
`;

function render(m) {
  const W = 1180; const H = 900;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 7) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const line = (x1, y1, x2, y2, cls = 'tick') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  const c = m.cfg;

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime rollout 接收循环：四个候选组如何得到两个训练组，以及 fully-async 的池与队列</title>');
  o.push('<desc id="desc">上方时间线展示 over-sampling 提交、FIRST_COMPLETED 接收、动态过滤拒绝、目标满后 abort 与 partial 回收，以及 remaining_batch_size 与 len(data) 两个计数器；下方展示连续拒绝触发补采、with_fallback 兜底与 fully-async 的池容量、qsize 闸门和按缺口 drain。</desc>');
  o.push(`<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>`);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '默认接收循环：四个候选组、两个计数器、一次 abort', 'ti');
  text(24, 56, `rollout_batch_size=${c.rolloutBatchSize} · over_sampling_batch_size=${c.overSamplingBatchSize} · n_samples_per_prompt=${c.nSamplesPerPrompt} · 信号量 = sglang_server_concurrency ${c.sglangServerConcurrency} × ${m.semaphore.engines} engines = ${m.semaphore.capacity}（本例 ${c.overSamplingBatchSize * c.nSamplesPerPrompt} 个请求不受限）`, 'su');

  // ---- 面板 1：默认循环时间线 ----
  const y1 = 76; const p1H = 400;
  rect(24, y1, 1132, p1H, 'panel', 10);
  text(40, y1 + 24, `generate_rollout_async：接收单位是整组（${c.nSamplesPerPrompt} 条 Sample 全部生成并打完 reward），FIRST_COMPLETED 取最先完成的组`, 'pt');
  const main = m.main;
  const tx0 = 160; const colW = 122; const rowH = 34; const rowsY = y1 + 60;
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
    text(tx0 + endCol * colW + colW / 2, y + 19, decision === 'accept' ? '完成 → 通过' : decision === 'drop' ? '完成 → 拒绝' : '在途 → abort', 'sm', 'middle');
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
  const rx = 800; const ry0 = y1 + 40;
  rect(rx, ry0, 340, 196, 'ghost', 8);
  text(rx + 12, ry0 + 20, '两个计数器的分工', 'pt');
  text(rx + 12, ry0 + 40, 'remaining_batch_size：已提交且未被拒的候选组数；', 'tx');
  text(rx + 12, ry0 + 57, `低于 target ${c.rolloutBatchSize} 时再补一整波 ${c.overSamplingBatchSize} 组，不是只补缺口。`, 'tx');
  text(rx + 12, ry0 + 77, `len(data)：已入选组数；达到 ${c.rolloutBatchSize} 即退出并 abort。`, 'tx');
  text(rx + 12, ry0 + 97, `B 被拒：remaining ${main.log[0].remaining} → ${main.log[1].remaining}，仍 ≥ ${c.rolloutBatchSize}，不补采。`, 'tx');
  text(rx + 12, ry0 + 117, `入选 ${main.data.join('、')}，按 group[0].index 排序后交训练。`, 'tx');
  text(rx + 12, ry0 + 137, 'D 在途：仅 partial_rollout 开时整组回 buffer，', 'tx');
  text(rx + 12, ry0 + 154, '否则丢弃；有 response 的 Sample 记 start_rollout_id。', 'tx');
  text(rx + 12, ry0 + 178, `信号量 ${m.semaphore.capacity} 按请求数限流，不按 token 或 KV 预算。`, 'cap');
  rect(rx, ry0 + 208, 340, 78, 'acc2', 8);
  text(rx + 12, ry0 + 228, 'abort 不是精确取消：对默认 router 全部 worker', 'tx');
  text(rx + 12, ry0 + 245, '发 abort_all，再查 /v1/loads 直到 0；', 'tx');
  text(rx + 12, ry0 + 262, 'load 查询异常只告警并返回，不证明已排空。', 'tx');
  text(rx + 12, ry0 + 279, '超出目标的已完成组不训练也不回收（源码 NOTE）。', 'tx');

  // ---- 面板 2：两条变体 ----
  const y2 = y1 + p1H + 14; const p2W = 640; const p2H = 396;
  rect(24, y2, p2W, p2H, 'panel', 10);
  text(40, y2 + 24, '同一循环的两条变体（8 个候选组，A B C 均为零方差）', 'pt');
  const variantRow = (yy, title, sim, notes) => {
    text(40, yy, title, 'tx');
    const stepsX = 40; const cw = 70;
    sim.log.forEach((l, i) => {
      const x = stepsX + i * (cw + 4);
      const cls = l.action === 'drop' ? 'acc2' : l.action.startsWith('accept') ? 'acc1' : l.action.startsWith('submit') ? 'cell' : 'ghost';
      rect(x, yy + 8, cw, 46, cls, 4);
      const head = l.action.startsWith('submit') ? `+${c.overSamplingBatchSize}` : l.action.startsWith('abort') ? 'abort' : `${l.group}`;
      const tag = l.action === 'drop' ? '拒' : l.action === 'accept' ? '入' : l.action.startsWith('accept') ? '兜底入' : '';
      text(x + cw / 2, yy + 24, `${head} ${tag}`.trim(), 'sm', 'middle');
      text(x + cw / 2, yy + 40, `r=${l.remaining} d=${l.accepted}`, 'mono', 'middle');
    });
    notes.forEach((n, k) => text(40, yy + 72 + k * 16, n, 'cap'));
  };
  const cas = m.cascade;
  const secondWave = cas.waves[1];
  variantRow(y2 + 50, `① check_reward_nonzero_std：连续拒绝 → remaining 低于 ${c.rolloutBatchSize} → 再提交一整波`, cas, [
    `B、A、C 依次被拒，remaining ${cas.log[1].remaining} → ${cas.log[2].remaining} → ${cas.log[3].remaining}；t${secondWave.tick} 再提交 ${secondWave.wave.join(' ')}（+${c.overSamplingBatchSize}）`,
    `随后 ${cas.data.join('、')} 入选；abort 时 ${cas.inFlight.length} 组在途，代价是多付一整波生成与 RM。`,
  ]);
  const fb = m.fallback;
  variantRow(y2 + 168, `② check_reward_nonzero_std_with_fallback：remaining ≤ target 时保留零方差组`, fb, [
    `B、A 被拒后 remaining=${fb.log[2].remaining} ≤ ${c.rolloutBatchSize}，C 虽零方差仍入选（keep_when_insufficient）`,
    `${fb.data.join('、')} 凑满，不再补采，abort 时 ${fb.inFlight.length} 组在途；代价是送进一个无梯度信号的组。`,
  ]);
  text(40, y2 + 292, 'abort 后按 index 排序 → rollout_sample_filter_path 原地改 remove_sample', 'cap');
  text(40, y2 + 310, '→ rollout_all_samples_process_path 看到全部已完成组（含被拒者，不含在途回收者）。', 'cap');
  text(40, y2 + 334, '未接 dynamic_sampling_filter_path 时 call_dynamic_filter 恒返回 keep=True，', 'cap');
  text(40, y2 + 352, '循环退化为"取够 rollout_batch_size 个先完成的组"。', 'cap');
  text(40, y2 + 376, '蓝：入选与决定性判定；橙：拒绝、abort 与回收。灰条为在途时间；柱内 r/d 为事件后的 remaining_batch_size / len(data)。', 'cap');

  // ---- 面板 3：fully-async ----
  const x3 = 24 + p2W + 14; const w3 = 1156 - x3;
  rect(x3, y2, w3, p2H, 'panel', 10);
  const fa = m.fullyAsync;
  text(x3 + 16, y2 + 24, `fully-async 生产者：池容量 ${fa.cap}，qsize 闸门 ${fa.cap}，drain 限额 ${c.rolloutBatchSize}`, 'pt');
  const th = ['tick', 'callback 入队', '补位', 'active', 'queue', '闸门'];
  const cws = [40, 92, 60, 74, 96, 60];
  let cx = x3 + 16;
  th.forEach((h, i) => { text(cx + cws[i] / 2, y2 + 50, h, 'sm', 'middle'); cx += cws[i]; });
  fa.ticks.forEach((t, r) => {
    const yy = y2 + 60 + r * 40;
    let x = x3 + 16;
    const cells = [
      `t${t.tick}`, t.done.length ? t.done.join(' ') : '—', t.topped.length ? t.topped.join(' ') : '—',
      t.active.length ? t.active.join(' ') : '—', t.queue.length ? `[${t.queue.join(' ')}]` : '[]', t.gateClosed ? '关' : '开',
    ];
    cells.forEach((v, i) => {
      const cls = i === 5 && t.gateClosed ? 'acc2' : i === 4 && t.drained ? 'acc1' : 'cell';
      rect(x, yy, cws[i] - 4, 32, cls, 4);
      text(x + (cws[i] - 4) / 2, yy + 20, v, 'mono', 'middle');
      x += cws[i];
    });
  });
  const fy = y2 + 60 + fa.ticks.length * 40 + 12;
  rect(x3 + 16, fy, w3 - 32, 64, 'acc1', 8);
  text(x3 + 28, fy + 20, `t${fa.ticks.at(-1).tick} 消费者线程 get_completed_groups(limit=${c.rolloutBatchSize}) → ${fa.drained.join(' ')}`, 'tx');
  text(x3 + 28, fy + 38, `队列留下 [${fa.queueLeft.join(' ')}] 给下一轮（"queue stays warm"）`, 'tx');
  text(x3 + 28, fy + 56, `t2 闸门关：qsize ${fa.ticks[2].queue.length} ≥ ${fa.cap} 停止补位，但在途 ${fa.ticks[2].active.join(' ')} 仍可完成入队`, 'tx');
  text(x3 + 16, fy + 90, '队列无界：done-callback 在 loop 线程内 put，有界队列满时会冻住全部在途生成。', 'cap');
  text(x3 + 16, fy + 108, '含 ABORTED 成员的组回 data_buffer.add_samples，不进队列；异常 task 只记日志。', 'cap');
  text(x3 + 16, fy + 126, '默认动态过滤、轮末 sample filter 与 all-samples 钩子都在被替换的循环里，此处不运行。', 'cap');

  text(24, 890, '源码基线：THUDM/slime@681b3adca541 · 复现 generate_rollout_async / abort / should_drop_dynamic_filter_output / check_reward_nonzero_std(_with_fallback) / AsyncRolloutWorker._loop', 'su');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_rollout_admission_timeline.svg'), `${render(model())}\n`, 'utf8');
}
