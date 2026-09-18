// 图：同一组阶段时长在 slime 两个 driver 上的关键路径——train.py 串行一轮与 train_async.py 重叠一轮，
// 以及 actor 进程内 Timer 产出的 perf/train_wait_time 里哪些段有默认键、哪些只剩残差。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：
//  1. train.py 每轮串行 generate → train → save → update_weights，一轮等于各段之和；
//     --update-weights-interval 在该入口不被读取，每轮都发布。
//  2. train_async.py 在轮首提交下一轮 generate、同时训练当前批；发布前 ray.get 下一轮 future（fence）。
//     driver 等待只是两条臂的差值，已经包含在 max 里：周期 ≈ max(rollout + data, train + save) + publish。
//     等在哪个调用上取决于本轮是否保存：RolloutManager 是同步 Ray actor，保存轮的
//     ray.get(rollout_manager.save.remote(i)) 排在已提交的 generate(i+1) 之后，fence 随即返回；
//     不保存的轮次才在 fence 上等。再单独加一项等待就把同一段时间算了两次。
//  3. actor 的 inverse_timer("train_wait") / timer("train") 把相邻两次训练块之间的全部墙钟记为
//     perf/train_wait_time；同一条 perf 记录里 data_preprocess、save_model、update_weights 有自己的键，
//     剩下的残差（同步：rollout + 转换；异步：driver 在 rollout_manager.save 或 fence 上的等待）没有默认键。
//
// 布局：① train.py 两轮、② train_async.py 三轮，共用一条时间轴，每个面板四条泳道
// （RolloutManager.generate / actor.train / update_weights / actor Timer），② 另加周期标尺；
// ③ 把第 1 轮的 train_wait 窗口按时间顺序拆段并标出每段的键。acc1 标关键路径与周期，acc2 标无键的等待。
//
// 用法：node tools/figs/svg/slime_rollout_cycle_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入（时间单位抽象，只看相对长短） ----------------
export const CFG = Object.freeze({
  T: Object.freeze({ rollout: 6, data: 1, pre: 1, train: 3, save: 1, publish: 1 }),
  saveInterval: 1,
  rolloutGlobalDataset: true, // --disable-rollout-global-dataset 是 store_false，默认开启
  drawRounds: Object.freeze({ sync: 2, async: 3 }),
  statRounds: 6,
});

// 同一段在 perf 记录里的键；null 表示没有默认键（只能从残差推出或自建计时）。
export const KEY = Object.freeze({
  rollout: 'perf/rollout_time',
  data: null,
  pre: 'perf/data_preprocess_time',
  train: 'perf/train_time',
  save: 'perf/save_model_time',
  publish: 'perf/update_weights_time',
  rmsave: null,
  fence: null,
});

// slime/utils/misc.py::should_run_periodic_action（本例 num_rollout_per_epoch 取 None）
export function shouldRunPeriodic(rolloutId, interval, numRollout = null) {
  if (interval === null) return false;
  if (numRollout !== null && rolloutId === numRollout - 1) return true;
  return (rolloutId + 1) % interval === 0;
}

const seg = (kind, round, start, len, extra = {}) => ({ kind, round, start, end: start + len, ...extra });

// train.py::train 的主循环（offload/onload 在分离资源、未开 offload 时耗时取 0）。
// update_weights 每轮无条件调用：该入口不读取 --update-weights-interval。
export function simulateSync(cfg = CFG, rounds = cfg.drawRounds.sync, interval = 1) {
  void interval; // 刻意不使用：复现 train.py 忽略该参数
  const T = cfg.T;
  const lanes = { gen: [], actor: [], publish: [] };
  const batches = [];
  let t = 0;
  let version = 0; // 循环前的首推版本 v0
  for (let r = 0; r < rounds; r += 1) {
    lanes.gen.push(seg('rollout', r, t, T.rollout));
    lanes.gen.push(seg('data', r, t + T.rollout, T.data));
    batches.push({ round: r, genVersion: version, submit: t, ready: t + T.rollout + T.data });
    t += T.rollout + T.data;
    lanes.actor.push(seg('pre', r, t, T.pre));
    lanes.actor.push(seg('train', r, t + T.pre, T.train));
    t += T.pre + T.train;
    if (shouldRunPeriodic(r, cfg.saveInterval, rounds)) {
      lanes.actor.push(seg('save', r, t, T.save));
      t += T.save;
    }
    lanes.publish.push(seg('publish', r, t, T.publish, { version: r + 1 }));
    t += T.publish;
    version = r + 1;
  }
  return finish({ lanes, batches, end: t, serialGen: true });
}

// train_async.py::train 的主循环；RolloutManager 是串行 Ray actor，generate 按提交顺序执行。
export function simulateAsync(cfg = CFG, rounds = cfg.drawRounds.async, interval = 1) {
  const T = cfg.T;
  const lanes = { gen: [], actor: [], publish: [] };
  const batches = [];
  const tops = [];
  let managerFree = 0;
  let version = 0;
  const submit = (id, at) => {
    const start = Math.max(at, managerFree);
    const ready = start + T.rollout + T.data;
    managerFree = ready;
    lanes.gen.push(seg('rollout', id, start, T.rollout));
    lanes.gen.push(seg('data', id, start + T.rollout, T.data));
    const b = { round: id, genVersion: version, submit: start, ready };
    batches.push(b);
    return b;
  };
  let t = 0;
  let next = submit(0, 0);
  for (let r = 0; r < rounds; r += 1) {
    if (next !== null) t = Math.max(t, next.ready); // ray.get(rollout_data_next_future)
    tops.push(t);
    next = r + 1 < rounds ? submit(r + 1, t) : null; // 提前启动下一轮
    lanes.actor.push(seg('pre', r, t, T.pre));
    lanes.actor.push(seg('train', r, t + T.pre, T.train));
    t += T.pre + T.train;
    if (shouldRunPeriodic(r, cfg.saveInterval, rounds)) {
      lanes.actor.push(seg('save', r, t, T.save));
      t += T.save;
      // ray.get(rollout_manager.save.remote(r))：同步 actor 按提交顺序执行，排在 generate(r+1) 之后
      if (cfg.rolloutGlobalDataset && managerFree > t) {
        lanes.actor.push(seg('rmsave', r, t, managerFree - t));
        t = managerFree;
      }
    }
    if ((r + 1) % interval === 0) {
      if (next !== null && next.ready > t) {
        lanes.actor.push(seg('fence', r, t, next.ready - t)); // 发布前等 generation future
        t = next.ready;
      }
      next = null;
      lanes.publish.push(seg('publish', r, t, T.publish, { version: r + 1 }));
      t += T.publish;
      version = r + 1;
    }
  }
  return finish({ lanes, batches, end: t, tops, serialGen: false });
}

// actor 进程 Timer：init 结束时启动 train_wait；每个训练块 inverse_timer 结束 train_wait、
// timer("train") 计块长；块末 log_perf_data 取走同一窗口里累计的全部计时。
// 拆段只沿 driver 的串行路径：同步时生成也在路径上；异步时生成与训练并行，driver 只在 rm.save 或 fence 上等它。
function finish(run) {
  const trainBlocks = run.lanes.actor.filter((s) => s.kind === 'train');
  const all = [...(run.serialGen ? run.lanes.gen : []), ...run.lanes.actor, ...run.lanes.publish];
  run.timer = trainBlocks.map((b, i) => {
    const waitStart = i === 0 ? 0 : trainBlocks[i - 1].end;
    const inWindow = all.filter((s) => s.start >= waitStart && s.end <= b.start);
    const sum = (kind) => inWindow.filter((s) => s.kind === kind).reduce((a, s) => a + (s.end - s.start), 0);
    const keyed = { pre: sum('pre'), save: sum('save'), publish: sum('publish') };
    const trainWait = b.start - waitStart;
    const residual = trainWait - keyed.pre - keyed.save - keyed.publish;
    const pieces = inWindow.filter((s) => s.kind !== 'train').sort((a, c) => a.start - c.start);
    return {
      round: b.round,
      waitStart,
      trainWait,
      train: b.end - b.start,
      stepTime: trainWait + (b.end - b.start),
      keyed,
      residual,
      residualParts: { rollout: sum('rollout'), data: sum('data'), rmsave: sum('rmsave'), fence: sum('fence') },
      pieces,
    };
  });
  run.lags = run.batches
    .filter((b) => b.round < trainBlocks.length)
    .map((b) => b.round - b.genVersion);
  if (run.tops) run.cycles = run.tops.slice(1).map((x, i) => x - run.tops[i]);
  return run;
}

// 闭式：同步一轮之和；异步一轮 max(两条臂) + 发布；误读式把 driver 等待在 max 外再加一次。
export function closedForms(cfg = CFG) {
  const T = cfg.T;
  const genArm = T.rollout + T.data;
  const trainArm = T.pre + T.train + T.save;
  const driverWait = Math.max(0, genArm - trainArm);
  return {
    genArm,
    trainArm,
    sync: genArm + T.pre + T.train + T.save + T.publish,
    async: Math.max(genArm, trainArm) + T.publish,
    driverWait,
    oldBound: Math.max(genArm, T.pre + T.train) + driverWait + T.publish,
  };
}

export function model(cfg = CFG) {
  const r2 = (x) => Math.round(x * 1000) / 1000;
  const sync = simulateSync(cfg);
  const asyn = simulateAsync(cfg);
  const syncStat = simulateSync(cfg, cfg.statRounds, 2);
  const asyncStat1 = simulateAsync(cfg, cfg.statRounds, 1);
  const asyncStat2 = simulateAsync(cfg, cfg.statRounds, 2);
  const altSave = simulateAsync({ ...cfg, saveInterval: 2 }, 4, 1);
  const s1 = sync.timer[1];
  const a1 = asyn.timer[1];
  return {
    closed: closedForms(cfg),
    sync,
    async: asyn,
    row: {
      sync: { ...s1, ratio: r2(s1.trainWait / s1.stepTime) },
      async: { ...a1, ratio: r2(a1.trainWait / a1.stepTime) },
    },
    genIdle: {
      sync: s1.stepTime - (cfg.T.rollout + cfg.T.data),
      async: asyn.cycles[1] - (cfg.T.rollout + cfg.T.data),
    },
    interval: {
      syncSteps: syncStat.timer.slice(1).map((x) => x.stepTime),
      syncLags: syncStat.lags,
      async1Cycles: asyncStat1.cycles,
      async1Lags: asyncStat1.lags,
      async2Cycles: asyncStat2.cycles,
      async2Lags: asyncStat2.lags,
      // --save-interval 2：每轮 driver 等待落在哪个调用上、等多久
      altSaveWaits: altSave.lanes.actor.filter((x) => x.kind === 'rmsave' || x.kind === 'fence').map((x) => ({ round: x.round, kind: x.kind, len: x.end - x.start })),
      altSaveCycles: altSave.cycles,
    },
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function textWidth(s, size) {
  let w = 0;
  for (const ch of String(s)) w += /[\u2000-\u2bff\u2e80-\uffef]/.test(ch) ? size : size * 0.58;
  return w;
}

export const GEOM = Object.freeze({ W: 1180, H: 818, x0: 200, unit: 29, laneH: 22, laneGap: 32, barFont: 10.5 });
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .hatch{fill:url(#hatch);stroke:#C3651F;stroke-width:1.5}
  .axis{stroke:#AEB6C2;stroke-width:1}
  .tick{stroke:#D9DEE7;stroke-width:1}
  .ruler{fill:none;stroke:#2563EB;stroke-width:1.6}
`;

const CLASS = { rollout: 'neutral', data: 'ghost', pre: 'ghost', train: 'acc1', save: 'neutral', publish: 'neutral', rmsave: 'hatch', fence: 'hatch' };
const SHORT = { rollout: 'rollout', data: 'data', pre: 'pre', train: 'train', save: 'save', publish: 'pub', rmsave: '排队', fence: 'fence' };

function render(m, cfg = CFG, geom = GEOM) {
  const { W, H, x0, unit, laneH, laneGap, barFont } = geom;
  const T = cfg.T;
  const c = m.closed;
  const o = [];
  const labels = []; // 供测试检查：每个条内标签都放得下
  const X = (t) => x0 + t * unit;
  const rect = (x, y, w, h, cls = 'neutral', r = 4) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const bar = (t0, t1, y, cls, label) => {
    const x = X(t0);
    const w = (t1 - t0) * unit;
    rect(x, y, w, laneH, cls, 3);
    if (label) {
      text(x + w / 2, y + 15, label, 'sm', 'middle');
      labels.push({ label, width: w, need: textWidth(label, barFont) });
    }
  };
  const laneLabel = (y, s) => text(40, y + 15, s, 'sm');
  const axis = (y, tMax) => {
    o.push(`<line x1="${X(0)}" y1="${y}" x2="${X(tMax)}" y2="${y}" class="axis"/>`);
    for (let t = 0; t <= tMax; t += 2) {
      o.push(`<line x1="${X(t)}" y1="${y}" x2="${X(t)}" y2="${y + (t % 4 === 0 ? 6 : 3)}" class="axis"/>`);
      if (t % 4 === 0) text(X(t), y + 18, String(t), 'sm', 'middle');
    }
  };
  const genLabel = (run, s) => {
    const b = run.batches.find((x) => x.round === s.round);
    return `rollout D${s.round}·v${b.genVersion}`;
  };
  const lanesOf = (run, y) => {
    laneLabel(y, 'RolloutManager.generate');
    run.lanes.gen.forEach((s) => bar(s.start, s.end, y, CLASS[s.kind], s.kind === 'rollout' ? genLabel(run, s) : SHORT[s.kind]));
    laneLabel(y + laneGap, 'actor.train / save_model');
    run.lanes.actor.forEach((s) => {
      const lab = s.kind === 'train' ? `train D${s.round}` : s.kind === 'fence' || s.kind === 'rmsave' ? `${SHORT[s.kind]} ${s.end - s.start}` : SHORT[s.kind];
      bar(s.start, s.end, y + laneGap, CLASS[s.kind], lab);
    });
    laneLabel(y + 2 * laneGap, 'update_weights');
    run.lanes.publish.forEach((s) => bar(s.start, s.end, y + 2 * laneGap, 'neutral', SHORT.publish));
    laneLabel(y + 3 * laneGap, 'actor Timer');
    run.timer.forEach((row, i) => {
      bar(row.waitStart, row.waitStart + row.trainWait, y + 3 * laneGap, 'ghost', i === 0 ? 'train_wait' : `train_wait ${row.trainWait}`);
      bar(row.waitStart + row.trainWait, row.waitStart + row.stepTime, y + 3 * laneGap, 'neutral', 'train');
    });
  };

  const tMax = 32;
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime 训练闭环关键路径：同一组阶段时长在 train.py 与 train_async.py 上的周期、计时键与残差</title>');
  o.push('<desc id="desc">上面两块面板共用时间轴：train.py 每轮串行生成、训练、保存、发布，一轮 13；train_async.py 轮首提交下一轮生成并同时训练，保存轮的 DataSource 保存排在下一轮生成之后，一轮 max(7, 5) + 1 = 8，driver 等待 2 包含在 max 内。每块面板底部的 actor Timer 泳道给出 perf/train_wait_time 与 train 的窗口。第三块把第 1 轮的 train_wait 窗口拆段：data_preprocess、save_model、update_weights 有默认键，同步的 rollout 转换与异步的 driver 等待没有。</desc>');
  o.push('<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#FCF1E6"/><line x1="0" y1="0" x2="0" y2="6" stroke="#E7B98E" stroke-width="2"/></pattern></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '训练闭环关键路径：train.py 把各段相加，train_async.py 取两条臂的较大者再加发布', 'ti');
  text(24, 56, `示例时长：rollout ${T.rollout}、数据转换 ${T.data}、data_preprocess ${T.pre}、训练块 ${T.train}、save ${T.save}、update_weights ${T.publish}；每轮保存，--update-weights-interval 1；t=0 时首推已完成`, 'su');

  // ---------- ① train.py ----------
  const p1 = 72;
  rect(24, p1, 1132, 218, 'panel', 10);
  const d1 = m.sync.batches.find((b) => b.round === 1);
  text(40, p1 + 24, `① train.py：一轮 = ${T.rollout} + ${T.data} + ${T.pre} + ${T.train} + ${T.save} + ${T.publish} = ${c.sync}；训练批 D1 由 v${d1.genVersion} 生成，版本滞后 ${1 - d1.genVersion}`, 'pt');
  lanesOf(m.sync, p1 + 40);
  axis(p1 + 40 + 4 * laneGap - 4, tMax);
  const rs = m.row.sync;
  text(40, p1 + 204, `第 1 轮 perf：train_wait ${rs.trainWait} + train ${rs.train} = step_time ${rs.stepTime}，wait_time_ratio = ${rs.trainWait}/${rs.stepTime} = ${rs.ratio}；推理侧每轮空闲 ${m.genIdle.sync}`, 'sm');

  // ---------- ② train_async.py ----------
  const p2 = 304;
  rect(24, p2, 1132, 280, 'panel', 10);
  const ra = m.row.async;
  const d2 = m.async.batches.find((b) => b.round === 2);
  text(40, p2 + 24, `② train_async.py：一轮 = max(${T.rollout} + ${T.data}, ${T.pre} + ${T.train} + ${T.save}) + ${T.publish} = ${c.async}；driver 等待 ${c.driverWait} 已在 max 内，D2 由 v${d2.genVersion} 生成，滞后 ${2 - d2.genVersion}`, 'pt');
  lanesOf(m.async, p2 + 40);
  const ry = p2 + 40 + 4 * laneGap;
  laneLabel(ry, '轮首间隔');
  m.async.tops.forEach((t, i) => {
    if (i === 0) return;
    const a = X(m.async.tops[i - 1]);
    const b = X(t);
    o.push(`<path d="M${a} ${ry + 4} L${a} ${ry + 16} M${a} ${ry + 10} L${b} ${ry + 10} M${b} ${ry + 4} L${b} ${ry + 16}" class="ruler"/>`);
    text((a + b) / 2, ry + 6, `${t - m.async.tops[i - 1]}`, 'sm', 'middle');
  });
  axis(ry + 26, tMax);
  text(40, p2 + 230, `第 1 轮 perf：train_wait ${ra.trainWait} + train ${ra.train} = step_time ${ra.stepTime}，wait_time_ratio = ${ra.trainWait}/${ra.stepTime} = ${ra.ratio}；推理侧每轮空闲 ${m.genIdle.async}`, 'sm');
  const alt = m.interval.altSaveWaits;
  text(40, p2 + 248, `排队 = driver 在 ray.get(rollout_manager.save) 上等已提交的 generate，fence 随即返回；--save-interval 2 时不保存的轮次改在 fence 上等 ${alt[0].len}`, 'sm');
  text(40, p2 + 266, `max(${c.genArm}, ${T.pre + T.train}) + ${c.driverWait} + ${T.publish} = ${c.oldBound} 把等待在 max 外又算一次；interval 2 时轮首间隔 ${m.interval.async2Cycles.slice(0, 2).join('、')} 交替，滞后 ${m.interval.async2Lags.slice(1, 3).join('、')} 交替`, 'sm');

  // ---------- ③ 第 1 轮 train_wait 拆段 ----------
  const p3 = 598;
  const u3 = 60;
  const bx = 200;
  rect(24, p3, 1132, 176, 'panel', 10);
  text(40, p3 + 24, '③ 第 1 轮 perf/train_wait_time 按时间顺序拆段：有默认键的段与只剩残差的段', 'pt');
  const ledger = (row, y, name, tail) => {
    text(40, y + 15, name, 'tx');
    let x = bx;
    row.pieces.forEach((s) => {
      const w = (s.end - s.start) * u3;
      const cls = KEY[s.kind] === null ? (s.kind === 'fence' || s.kind === 'rmsave' ? 'hatch' : 'acc2') : s.kind === 'rollout' ? 'neutral' : 'ghost';
      rect(x, y, w, laneH, cls, 3);
      const lab = `${SHORT[s.kind]} ${s.end - s.start}`;
      text(x + w / 2, y + 15, lab, 'sm', 'middle');
      labels.push({ label: lab, width: w, need: textWidth(lab, barFont) });
      x += w;
    });
    text(x + 12, y + 15, tail, 'sm');
  };
  const syncTail = `= ${rs.trainWait}：有键 ${rs.keyed.pre + rs.keyed.save + rs.keyed.publish}，残差 ${rs.residual}（rollout_time ${rs.residualParts.rollout} + 转换 ${rs.residualParts.data}）`;
  const asyncTail = `= ${ra.trainWait}：有键 ${ra.keyed.pre + ra.keyed.save + ra.keyed.publish}，残差 ${ra.residual}（driver 等待：本轮在 rollout_manager.save 排队）`;
  ledger(rs, p3 + 44, 'train.py', syncTail);
  ledger(ra, p3 + 80, 'train_async.py', asyncTail);
  text(40, p3 + 128, `键：save → ${KEY.save}，pub → ${KEY.publish}，pre → ${KEY.pre}，rollout → ${KEY.rollout}（rollout 侧记录）`, 'sm');
  text(40, p3 + 146, 'data（样本转训练 dict、DP 切分、ray.put）与排队/fence 等待没有默认键；橙色段只能用 train_wait 减去同一条记录里的有键段得到，或在 driver 侧自建计时', 'sm');
  text(40, p3 + 164, 'update_weights 与 save_model 由 driver 在上一轮训练块之后调用，因此出现在下一条 perf 记录里；offload、整份磁盘 reload、eval、actor 权重 CPU 备份在本例取 0', 'sm');

  text(24, 798, '阅读顺序：①② 对齐同一时间轴看关键路径 → ② 的橙色等待是两臂之差 → ③ 同一条 perf 记录里减去有键段，剩下的才是 driver 侧等待。', 'cap');
  o.push('</svg>');
  return { svg: o.join('\n'), labels };
}

export function renderSvg(cfg = CFG) {
  return render(model(cfg), cfg);
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_rollout_cycle_timeline.svg'), `${renderSvg().svg}\n`, 'utf8');
}
