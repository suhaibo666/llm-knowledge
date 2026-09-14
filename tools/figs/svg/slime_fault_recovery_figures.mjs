// 图：slime 按故障域局部恢复的两个决定性时刻：① 一次 CI 故障注入里，监控线程在 generate 期间把坏
// engine 整组标死，重建推迟到本轮训练之后的 update_weights；② 重启时 trainer checkpoint 的 iteration
// 决定下一轮 id，DataSource 游标按同一 id 读取，两次顺序写之间崩溃会让游标静默归零。
// 源码基线：THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9。
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
  defaults: { interval: 30, timeout: 30, firstWait: 0 }, // arguments.py 默认值
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

// health_monitor.py::_run_health_checks 逐个检查、查完一轮才等 interval：其余检查耗时可忽略时，
// 崩溃到标死不超过 interval + timeout；一般 ≤ interval + N·timeout（N 个 engine 各耗满 timeout）。
// RolloutManager._try_ci_fault_injection 固定等 interval + timeout + 5。
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

// actor.py::MegatronTrainRayActor.init（start = iteration + 1）+ placement_group.py::create_training_models
// （DataSource.load(start − 1)）+ data_source.py::RolloutDataSource.load（文件缺失只记日志）
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
  o.push('<desc id="desc">上半部分按轮画出 CI 故障注入：generate 期间监控线程把坏 engine 整组标死，本轮用剩余 engine 完成，重建与重新推送发生在训练之后的 update_weights，并对比三组健康检查参数的检测上界与 CI 等待时间。下半部分画出保存间隔 2 时的保存点与三种重启情形：两份文件都在、两次写之间崩溃导致 DataSource 游标静默归零、异步保存尚未 finalize。</desc>');
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
  text(40, cy + 24, `② 共同恢复切点：num_rollout ${cfg.ckpt.numRollout}、--save-interval ${cfg.ckpt.saveInterval} → 在 rollout ${k.saves.join('、')} 保存；先写 trainer，再写 DataSource`, 'pt');
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
    ['异步保存：DataSource 3 已写、trainer 3 未 finalize', `tracker 仍指向 ${k.common}（依赖侧：finalize 后才更新）`, `start_rollout_id = ${k.asyncCase.start}，load(${k.asyncCase.loadId})，两边一致`, 'neutral'],
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

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_fault_recovery_timeline.svg'), `${render(model())}\n`, 'utf8');
}
