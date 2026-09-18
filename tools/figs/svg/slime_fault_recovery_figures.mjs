// 图 1（slime_fault_recovery_timeline.svg）：slime 按故障域局部恢复的两个决定性时刻：① 一次 CI 故障注入里，
// 监控线程在 generate 期间把坏 engine 整组标死，重建推迟到本轮训练之后的 update_weights；② 重启时 trainer
// checkpoint 的 iteration 决定下一轮 id，DataSource 游标按同一 id 读取，两次顺序写之间崩溃会让游标静默归零。
// 图 2（slime_metric_step_axes.svg）：同一批 rollout 在 train/step、rollout/step、eval/step 三个 step 键上的值，
// --wandb-always-use-train-step 的折算，以及 sync / async 入口下同一 step 值对应的生成权重与计时归属。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；Megatron tracker 语义核对 NVIDIA/Megatron-LM@1dcf0dafa884。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：检测只标死不重建；其余检查可忽略时检测上界 = interval + timeout，CI 注入固定等 interval + timeout + 5；
// 本轮 rollout 用剩余 engine 完成，重建与重新推送发生在训练之后的更新边界；E2E 的注入落在最后一轮，
// 重建的 engine 没有再服务请求。续训时 start = iteration + 1，DataSource 读 start − 1；共同恢复切点
// 是两份文件都在的最大 id。acc1 标恢复成立的路径，acc2 标失败边界与静默回退。
//
// 用法：node tools/figs/svg/slime_fault_recovery_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  engines: 4, // tests/test_sglang_config_mixed_offload_ft.py：actor 模型 4 个 1 卡 engine
  ft: { interval: 5, timeout: 10, firstWait: 0 }, // 该用例的健康检查参数
  defaults: { interval: 30, timeout: 30, firstWait: 0 }, // slime/utils/arguments.py 默认值
  doc: { interval: 10, timeout: 5, firstWait: 300 }, // docs/zh/advanced/fault-tolerance.md 写的默认值
  numRollout: 3,
  injectFrom: 2, // RolloutManager.generate：ci_test ∧ use_fault_tolerance ∧ rollout_id >= 2
  ckpt: { numRollout: 6, saveInterval: 2, crashAfterTrainerSave: 3 },
});

const range = (n) => Array.from({ length: n }, (_, i) => i);

// ---------------- 对源码算法的最小复现 ----------------

// slime/utils/misc.py::should_run_periodic_action
export function shouldRunPeriodic(rolloutId, interval, numRollout = null, perEpoch = null) {
  if (interval === null) return false;
  if (numRollout !== null && rolloutId === numRollout - 1) return true;
  const step = rolloutId + 1;
  return step % interval === 0 || (perEpoch !== null && step % perEpoch === 0);
}

// slime/utils/health_monitor.py::RolloutHealthMonitor._run_health_checks 逐个检查、查完一轮才等 interval：其余检查耗时可忽略时，
// 崩溃到标死不超过 interval + timeout；一般 ≤ interval + N·timeout（N 个 engine 各耗满 timeout）。
// slime/ray/rollout.py::RolloutManager._try_ci_fault_injection 固定等 interval + timeout + 5。
export const detectBound = (p) => p.interval + p.timeout;
export const detectGeneral = (p, n) => p.interval + n * p.timeout;
export const ciWait = (p) => p.interval + p.timeout + 5;

// train.py 主循环 + RolloutManager.generate 的注入 + actor.update_weights 的恢复
export function ftRun(cfg = CFG) {
  const rounds = [];
  let alive = cfg.engines;
  let pending = true;
  let dead = 0;
  for (let id = 0; id < cfg.numRollout; id += 1) {
    const r = { id, injected: false };
    if (pending && id >= cfg.injectFrom) {
      pending = false;
      r.injected = true;
      alive -= 1;
      dead += 1;
    }
    r.liveDuringRollout = alive;
    r.rebuiltAtUpdate = dead;
    alive += dead;
    dead = 0;
    rounds.push(r);
  }
  const last = rounds.map((r) => r.rebuiltAtUpdate > 0).lastIndexOf(true);
  return { rounds, roundsServedAfterRebuild: last < 0 ? null : cfg.numRollout - 1 - last };
}

// slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.init（start = iteration + 1）
// + slime/ray/placement_group.py::create_training_models（DataSource.load(start − 1)）
// + slime/rollout/data_source.py::RolloutDataSource.load（文件缺失只记日志）
export function restart({ trainerLatest, dsIds }) {
  const start = trainerLatest + 1;
  const loadId = start - 1;
  return { start, loadId, cursor: dsIds.includes(loadId) ? 'restored' : 'reset' };
}

export function ckptScenarios(cfg = CFG) {
  const c = cfg.ckpt;
  const saves = range(c.numRollout).filter((id) => shouldRunPeriodic(id, c.saveInterval, c.numRollout));
  const k = c.crashAfterTrainerSave;
  const prev = saves.filter((s) => s < k);
  const common = Math.max(...prev);
  return {
    saves,
    normal: restart({ trainerLatest: k, dsIds: [...prev, k] }),
    torn: restart({ trainerLatest: k, dsIds: prev }),
    common,
    fallback: restart({ trainerLatest: common, dsIds: prev }),
    asyncCase: restart({ trainerLatest: common, dsIds: [...prev, k] }),
  };
}

export function model(cfg = CFG) {
  return {
    bounds: ['ft', 'defaults', 'doc'].map((k) => ({ key: k, ...cfg[k], detect: detectBound(cfg[k]), general: detectGeneral(cfg[k], cfg.engines), wait: ciWait(cfg[k]) })),
    run: ftRun(cfg),
    ckpt: ckptScenarios(cfg),
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function textWidth(s, size) {
  let w = 0;
  for (const ch of String(s)) w += /[ -⯿⺀-￯]/.test(ch) ? size : size * 0.58;
  return w;
}
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
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4}
`;

function render(m, cfg = CFG) {
  const W = 1180;
  const H = 900;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 6) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2) => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="main"/>`);
  const row = (x, y, items) => {
    let cx = x;
    items.forEach(([lab, cls]) => {
      const w = textWidth(lab, 10.5) + 14;
      rect(cx, y, w, 24, cls || 'neutral', 4);
      text(cx + w / 2, y + 16, lab, 'sm', 'middle');
      cx += w + 5;
    });
    return cx;
  };

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime 故障域局部恢复：一次故障注入的检测与重建时间线，以及续训的共同恢复切点</title>');
  o.push('<desc id="desc">上半部分按轮画出 CI 故障注入：generate 期间监控线程把坏 engine 整组标死，本轮用剩余 engine 完成，重建与重新推送发生在训练之后的 update_weights，并对比三组健康检查参数的检测上界与 CI 等待时间。下半部分画出同步入口 train.py 在保存间隔 2 时的保存点与四种重启情形：两份文件都在、两次写之间崩溃导致 DataSource 游标静默归零、人工退回共同切点、异步保存尚未 finalize。</desc>');
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '故障按域恢复：检测在监控线程里，重建在更新边界上；续训的恢复点由两份顺序写的文件共同决定', 'ti');
  text(24, 56, `CI 故障注入 E2E：actor 模型 ${cfg.engines} 个 engine，interval ${cfg.ft.interval} s、timeout ${cfg.ft.timeout} s、first-wait ${cfg.ft.firstWait} s，num_rollout ${cfg.numRollout}，rollout_id ≥ ${cfg.injectFrom} 注入一次`, 'su');

  // ---------- ① 注入时间线 ----------
  const py = 72;
  rect(24, py, 1132, 430, 'panel', 10);
  text(40, py + 24, '① 一次故障注入：generate 里标死，训练之后的 update_weights 里重建并推送', 'pt');
  const r0 = m.run.rounds;
  const baseRow = (live) => [
    ['generate：resume 监控', 'neutral'],
    [`${live} 个 engine 生成`, 'neutral'],
    ['offload：pause 监控', 'ghost'],
    ['train', 'ghost'],
    ['onload_weights', 'ghost'],
  ];
  r0.forEach((r, i) => {
    const y = py + 48 + i * 70;
    text(40, y + 16, `rollout ${r.id}`, 'tx');
    let items;
    if (r.injected) {
      items = [
        ['generate：resume 监控', 'neutral'],
        ['注入 simulate_crash(e0)', 'acc2'],
        [`CI 等 ${m.bounds[0].wait} s`, 'acc2'],
        [`${r.liveDuringRollout} 个 engine 生成`, 'neutral'],
        ['offload：pause', 'ghost'],
        ['train', 'ghost'],
        ['onload_weights', 'ghost'],
        ['update_weights：recover e0 → connect → 推送', 'acc1'],
      ];
    } else {
      items = [...baseRow(r.liveDuringRollout), ['update_weights：recover（无死槽）', 'neutral']];
    }
    row(110, y, items);
  });
  const my = py + 48 + 2 * 70 + 34;
  text(110, my + 16, '监控线程', 'sm');
  const mEnd = row(170, my, [
    [`其余检查 + ≤ ${cfg.ft.interval} s 间隔`, 'ghost'],
    [`health_generate 失败（≤ ${cfg.ft.timeout} s）`, 'acc2'],
    ['shutdown + ray.kill 整个逻辑 engine → 槽位 None', 'acc2'],
  ]);
  text(170, my + 44, `检测上界 ${cfg.ft.interval} + ${cfg.ft.timeout} = ${m.bounds[0].detect} s（其余检查可忽略时；一般 ≤ ${cfg.ft.interval} + ${cfg.engines} × ${cfg.ft.timeout} = ${m.bounds[0].general} s）< CI 等待 ${cfg.ft.interval} + ${cfg.ft.timeout} + 5 = ${m.bounds[0].wait} s`, 'sm');
  text(170, my + 62, 'simulate_crash 只经 shutdown 注销 router worker 并杀掉 SGLang 子进程树；Ray actor 由监控随后 ray.kill', 'sm');
  rect(40, my + 76, 740, 40, 'acc2', 5);
  text(52, my + 92, `num_rollout = ${cfg.numRollout}：注入落在最后一轮，重建与推送之后再无 rollout，`, 'sm');
  text(52, my + 108, `重建的 e0 服务过的轮数 = ${m.run.roundsServedAfterRebuild}；用例证明降容完成与更新边界重建，不证明重建后能正常服务`, 'sm');
  void mEnd;

  const bx = 800;
  const by = my + 76;
  rect(bx, py + 256, 340, 160, 'neutral', 8);
  text(bx + 12, py + 276, '三组参数：检测上界与 CI 等待', 'tx');
  text(bx + 12, py + 296, '来源', 'sm');
  text(bx + 150, py + 296, '间隔 / 超时 / 首等', 'sm');
  text(bx + 300, py + 296, '上界', 'sm', 'end');
  text(bx + 330, py + 296, '等待', 'sm', 'end');
  const names = { ft: '故障注入 E2E', defaults: 'arguments.py 默认', doc: 'fault-tolerance.md' };
  m.bounds.forEach((b, i) => {
    const y = py + 318 + i * 24;
    rect(bx + 8, y - 15, 324, 21, b.key === 'doc' ? 'acc2' : 'ghost', 3);
    text(bx + 14, y, names[b.key], 'sm');
    text(bx + 150, y, `${b.interval} / ${b.timeout} / ${b.firstWait}`, 'sm');
    text(bx + 300, y, `${b.detect} s`, 'sm', 'end');
    text(bx + 330, y, `${b.wait} s`, 'sm', 'end');
  });
  text(bx + 12, py + 402, '文档与实现不一致，以实现为准；first-wait 默认 0', 'sm');
  void by;

  // ---------- ② 共同恢复切点 ----------
  const cy = 516;
  const k = m.ckpt;
  rect(24, cy, 1132, 344, 'panel', 10);
  text(40, cy + 24, `② 共同恢复切点（同步入口 train.py）：num_rollout ${cfg.ckpt.numRollout}、--save-interval ${cfg.ckpt.saveInterval} → 在 rollout ${k.saves.join('、')} 保存；先写 trainer，再写 DataSource`, 'pt');
  const gx = 110;
  range(cfg.ckpt.numRollout).forEach((id) => {
    const x = gx + id * 62;
    const saved = k.saves.includes(id);
    rect(x, cy + 42, 54, 26, saved ? 'acc1' : 'ghost', 4);
    text(x + 27, cy + 59, `id ${id}${saved ? ' 存' : ''}`, 'sm', 'middle');
  });
  text(40, cy + 59, '保存点', 'sm');
  text(gx + cfg.ckpt.numRollout * 62 + 10, cy + 59, 'should_run_periodic_action：(id+1) % interval == 0、epoch 边界，或最后一轮', 'sm');

  const scen = [
    ['正常：id 3 两份文件都在', `trainer 最新 iteration 3、DataSource 有 3`, `start_rollout_id = ${k.normal.start}，load(${k.normal.loadId})，游标恢复`, 'acc1'],
    ['撕裂：trainer 3 已写、DataSource 3 未写', `trainer 最新 iteration 3、DataSource 只有 ${k.saves.filter((s) => s < cfg.ckpt.crashAfterTrainerSave).join('、')}`, `start_rollout_id = ${k.torn.start}，load(${k.torn.loadId}) 找不到 → 只记日志，游标从 0 重来`, 'acc2'],
    ['人工退回共同切点', `--ckpt-step ${k.common}（两份文件都在的最大 id）`, `start_rollout_id = ${k.fallback.start}，load(${k.fallback.loadId})，游标恢复`, 'acc1'],
    ['异步保存：DataSource 3 已写、trainer 3 未 finalize', `tracker 仍指向 ${k.common}（finalize 前不写）`, `start_rollout_id = ${k.asyncCase.start}，load(${k.asyncCase.loadId})，两边一致`, 'neutral'],
  ];
  scen.forEach(([a, b, c, cls], i) => {
    const y = cy + 88 + i * 56;
    rect(40, y, 330, 44, 'neutral', 5);
    text(52, y + 18, a, 'tx');
    text(52, y + 36, b, 'sm');
    arrow(372, y + 22, 412, y + 22);
    rect(416, y, 724, 44, cls, 5);
    text(428, y + 26, c, 'tx');
  });
  text(40, cy + 326, '游标归零的后果：sample_offset、epoch、sample/group 计数都回到 0，重新从第 0 个 epoch 的开头取 prompt，身份编号与此前样本重复；buffer 里的 partial 样本本来就不进 checkpoint', 'sm');

  text(24, 880, '阅读顺序：① 标死与重建分属两个时刻，恢复只在下一次权重更新时发生 → ② 续训只看 trainer 的 iteration，DataSource 缺文件不拒绝启动。', 'cap');
  o.push('</svg>');
  return o.join('\n');
}

// ================= 图 2：指标 step 键与 x 轴 =================
// spec：上面板用同一组批次配置列出 rollout_id 0–3 在 train/step、rollout/step（默认与折算）上的值，
// acc1 标折算后与该批第一个 train/step 对齐的格子，acc2 标 gbs 不整除时的错位；下面板按 train.py 与
// train_async.py（update_weights_interval = 1）的调用顺序复现：生成该批的权重已训到哪一批、async 的
// generate 与哪次 train 并行、log_perf_data 取走的 update_weights 计时来自哪次调用。acc2 标 sync/async 的差异。
export const AXES = Object.freeze({
  rolloutBatchSize: 4,
  nSamplesPerPrompt: 8,
  gbsEven: 16,
  gbsUneven: 24,
  numRollout: 4,
});

const rolloutsPerBatch = (a) => a.rolloutBatchSize * a.nSamplesPerPrompt;
// slime/utils/dp_schedule.py::build_dp_schedule：num_steps = rollout 数 // global_batch_size，尾部不足一步的 rollout 丢弃
export const stepsPerRollout = (a, gbs) => Math.floor(rolloutsPerBatch(a) / gbs);
export const droppedRollouts = (a, gbs) => rolloutsPerBatch(a) - stepsPerRollout(a, gbs) * gbs;
// slime/backends/megatron_utils/model.py::train：train/step = rollout_id × num_steps_per_rollout + step_id
export const trainStepsOf = (a, gbs, id) => range(stepsPerRollout(a, gbs)).map((s) => id * stepsPerRollout(a, gbs) + s);
// slime/observability/metric_utils.py::compute_rollout_step
export const rolloutStepOf = (a, gbs, id, alwaysUseTrainStep) =>
  alwaysUseTrainStep ? Math.floor((id * rolloutsPerBatch(a)) / gbs) : id;

// train.py::train 与 train_async.py::train 的调用顺序复现。served = serving 权重已包含的训练批次数；
// 初始 update_weights 推送 served = 0。async 的 generate(k) 在提交时读 served：update_weights 之前总是先
// ray.get 在途 future，所以一次生成期间权重不变。Timer 只建模 update_weights：log_perf_data 在 train_actor 末尾取走并清零。
export function driverRun(mode, n, interval = 1) {
  let served = 0;
  let pendingTimers = ['init'];
  const trainedAtGenerate = [];
  const parallelTrain = [];
  const perfUpdateFrom = [];
  const logPerf = (k) => {
    perfUpdateFrom[k] = pendingTimers;
    pendingTimers = [];
  };
  const updateAfter = (k) => {
    pendingTimers = [...pendingTimers, k];
    served = k + 1;
  };
  const submit = (id, parallel) => {
    trainedAtGenerate[id] = served;
    parallelTrain[id] = parallel;
  };
  if (mode === 'sync') {
    for (let k = 0; k < n; k += 1) {
      submit(k, null);
      logPerf(k);
      updateAfter(k);
    }
  } else {
    submit(0, null);
    for (let i = 0; i < n; i += 1) {
      if (i + 1 < n) submit(i + 1, i);
      logPerf(i);
      if ((i + 1) % interval === 0) updateAfter(i);
    }
  }
  return { trainedAtGenerate, parallelTrain, perfUpdateFrom };
}

export function axesModel(a = AXES) {
  const ids = range(a.numRollout);
  return {
    ids,
    stepsEven: stepsPerRollout(a, a.gbsEven),
    stepsUneven: stepsPerRollout(a, a.gbsUneven),
    droppedUneven: droppedRollouts(a, a.gbsUneven),
    trainEven: ids.map((id) => trainStepsOf(a, a.gbsEven, id)),
    rolloutDefault: ids.map((id) => rolloutStepOf(a, a.gbsEven, id, false)),
    rolloutEven: ids.map((id) => rolloutStepOf(a, a.gbsEven, id, true)),
    trainUneven: ids.map((id) => trainStepsOf(a, a.gbsUneven, id)),
    rolloutUneven: ids.map((id) => rolloutStepOf(a, a.gbsUneven, id, true)),
    sync: driverRun('sync', a.numRollout),
    async1: driverRun('async', a.numRollout, 1),
  };
}

const trainedLabel = (n) => (n === 0 ? '无（初始权重）' : n === 1 ? '批 0' : `批 0–${n - 1}`);
const timerLabel = (from) => from.map((k) => (k === 'init' ? '初始推送' : `train ${k} 之后`)).join('、');

function renderAxes(m, a = AXES) {
  const W = 1180;
  const H = 668;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 5) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const labelX = 40;
  const colX = 440;
  const colW = 170;
  const grid = (y0, rows) => {
    text(labelX, y0 + 16, 'rollout_id', 'sm');
    m.ids.forEach((id, j) => text(colX + j * colW + (colW - 8) / 2, y0 + 16, String(id), 'pt', 'middle'));
    rows.forEach(([label, cells], i) => {
      const y = y0 + 30 + i * 32;
      text(labelX, y + 16, label, 'tx');
      cells.forEach(([value, cls], j) => {
        rect(colX + j * colW, y, colW - 8, 24, cls, 4);
        text(colX + j * colW + (colW - 8) / 2, y + 16, value, 'tx', 'middle');
      });
    });
    return y0 + 30 + rows.length * 32;
  };

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime 指标的三个 step 键：同一批 rollout 落在哪一步，以及 sync 与 async 下 step 值的含义</title>');
  o.push('<desc id="desc">上半部分用每批 32 条 rollout、global batch size 16 与 24 两种配置，列出 rollout_id 0 到 3 的 train/step、默认 rollout/step 与开启 --wandb-always-use-train-step 后折算的 rollout/step；整除时折算值等于该批第一个训练步，不整除时出现错位。下半部分复现 train.py 与 train_async.py 的调用顺序：生成每批数据的权重已包含哪些训练批次、async 生成与哪次训练并行、每一步上 update_weights 计时来自哪次调用。</desc>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '同一批 rollout 落在哪一步：三个 step 键、x 轴折算，以及 sync / async 下 step 值的含义', 'ti');
  text(24, 56, `示例：--rollout-batch-size ${a.rolloutBatchSize} × --n-samples-per-prompt ${a.nSamplesPerPrompt} = ${rolloutsPerBatch(a)} 条 rollout / 批；--global-batch-size ${a.gbsEven}（每批 ${m.stepsEven} 个优化步）与 ${a.gbsUneven}（每批 ${m.stepsUneven} 步，尾部 ${m.droppedUneven} 条丢弃）`, 'su');

  // ---------- ① step 值 ----------
  const p1 = 72;
  rect(24, p1, 1132, 284, 'panel', 10);
  text(40, p1 + 24, '① step 值：train/step 按优化步计，rollout/step 与 eval/step 按 rollout_id；开 --wandb-always-use-train-step 后折算', 'pt');
  const end1 = grid(p1 + 36, [
    [`train/step（gbs ${a.gbsEven}：每批 ${m.stepsEven} 步）`, m.trainEven.map((s) => [s.join('、'), 'neutral'])],
    ['rollout/step 默认（eval/step 同式）', m.rolloutDefault.map((s) => [String(s), 'neutral'])],
    [`rollout/step 折算（gbs ${a.gbsEven}）`, m.rolloutEven.map((s, j) => [String(s), s === m.trainEven[j][0] ? 'acc1' : 'acc2'])],
    [`train/step（gbs ${a.gbsUneven}：每批 ${m.stepsUneven} 步）`, m.trainUneven.map((s) => [s.join('、'), 'ghost'])],
    [`rollout/step 折算（gbs ${a.gbsUneven}）`, m.rolloutUneven.map((s, j) => [String(s), s === m.trainUneven[j][0] ? 'ghost' : 'acc2'])],
  ]);
  text(labelX, end1 + 22, `train/step = rollout_id × 每批步数 + step_id；每批步数 = ${rolloutsPerBatch(a)} // gbs（build_dp_schedule 按 rollout 切步，尾部不足一步的丢弃）`, 'sm');
  text(labelX, end1 + 40, `折算 rollout/step = rollout_id × ${a.rolloutBatchSize} × ${a.nSamplesPerPrompt} // gbs（compute_rollout_step）：gbs 整除 ${rolloutsPerBatch(a)} 时等于该批第一个 train/step；gbs ${a.gbsUneven} 时 rollout ${m.ids[m.ids.length - 1]} 折算成 ${m.rolloutUneven[m.rolloutUneven.length - 1]}，train/step 却是 ${m.trainUneven[m.trainUneven.length - 1][0]}`, 'sm');

  // ---------- ② sync / async ----------
  const p2 = p1 + 284 + 14;
  rect(24, p2, 1132, 252, 'panel', 10);
  text(40, p2 + 24, '② 同一个 rollout_id：生成它的权重已训到哪一批、async 生成与哪次训练并行、该步 perf 计时来自哪次调用', 'pt');
  const syncT = m.sync.trainedAtGenerate;
  const asyncT = m.async1.trainedAtGenerate;
  const end2 = grid(p2 + 36, [
    ['sync（train.py）：生成该批的权重已训', syncT.map((n) => [trainedLabel(n), 'neutral'])],
    ['async，interval 1（train_async.py）：同上', asyncT.map((n, j) => [trainedLabel(n), n === syncT[j] ? 'neutral' : 'acc2'])],
    ['async：generate 运行时 trainer 在做', m.async1.parallelTrain.map((k) => [k === null ? '空闲（循环前提交）' : `train ${k}`, k === null ? 'ghost' : 'acc2'])],
    ['两种入口：该步 perf/update_weights_time 来自', m.sync.perfUpdateFrom.map((from, j) => [timerLabel(from), timerLabel(from) === timerLabel(m.async1.perfUpdateFrom[j]) ? 'ghost' : 'acc2'])],
  ]);
  text(labelX, end2 + 22, 'sync 的 generate(k) 在 train k−1 与 update_weights 之后才开始；async 的 generate(k) 在 train k−1 之前提交，权重少一个已训批次，manager 侧 rollout/* 与 trainer 侧 train/* 交错到达', 'sm');
  text(labelX, end2 + 40, 'log_perf_data 在 train_actor 末尾取走并清零本进程 Timer：之后发生的 save_model、update_weights、sleep 计时记到下一个 rollout_id', 'sm');

  text(24, 650, '阅读顺序：① step 值只由 rollout_id 与批次配置决定 → ② 同一个 step 值在 sync 与 async 下对应不同的生成权重；x 轴是数据批次轴，不是策略版本轴。', 'cap');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_fault_recovery_timeline.svg'), `${render(model())}\n`, 'utf8');
  writeFileSync(join(outputDir, 'slime_metric_step_axes.svg'), `${renderAxes(axesModel())}\n`, 'utf8');
}
