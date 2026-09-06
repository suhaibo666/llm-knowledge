// 27_megatron_job_resilience_analysis.md 的四张图。
//
// 图 1：一次「跑不动了」的故障从发生到重新进入训练要走完的五段，每段的 owner、deadline
//       与失败表现；同时画出作业面与数值面的分界（数值面归 28 号页）。
// 图 2：NVRx 只记三个 section，而「哪个 section 现在允许更新阈值」由四道闸门决定 ——
//       闸门全部复刻自 ft_integration.py::_maybe_update_timeouts，逐场景算出结果。
// 图 3：**原理图** —— sniff test 的离群判据。同一组带宽测量，中位数+MAD+下界抓到两张坏卡，
//       均值±2σ 一张都抓不到（离群点自己把 σ 抬高了）；健康集群那一组则反过来证明下界的作用。
// 图 4：一次迭代里五条存档/退出路径的判定顺序，以及 saved_checkpoint 去重省下的那次全量存档。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 3 是本页的 principle figure，要回答「凭什么说 rank 37 有问题」。取一组 8 个 rank 的
// AllReduce 带宽读数，其中两个明显偏低。左右两个面板：
//   左 = 故障集群。画中位数竖线、±threshold 带（threshold = max(MAD, median×0.10)）、
//        以及作为对照的 mean±2σ 带（ghost）。被 MAD 判据抓到的两根条用 acc2，
//        mean±2σ 一根都没抓到 —— 这就是「离群点污染均值与标准差」的可视证据。
//   右 = 健康集群。threshold 仍由下界主导；把下界去掉（阈值退化成 MAD）会误报一根。
// 两个面板的 median / MAD / mean / σ / threshold / 命中集合全部由脚本算出，不手写。
//
// 图 2 布局：上半是一条迭代时间轴，画出 setup section、warmup 期（不开 section）、
// step section 逐次开合、checkpointing section；下半是四个场景 × 四行闸门的判定表，
// 每格由复刻的门控函数算出「更新 / 不更新」以及源码给出的理由。
//
// 图 1 布局：五段横排，上方一条窄带标「作业面（本页）」，下方一条 ghost 带标
// 「数值面（28 号页）」，说明同一次异常在两个平面上走不同的流程。
//
// 图 4 布局：左侧一条自上而下的判定阶梯（五个分支节点），右侧是一次迭代扫描的结果条：
// 同一段迭代下「有去重 / 无去重」的存档次数由同一段复刻代码算出。
//
// 用法：node tools/figs/svg/megatron_job_resilience_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  // gpu_sniff_test.py:45
  outlierMinDeviationFrac: 0.1,
  // 一组 8 rank 的 AllReduce 总线带宽读数（GB/s），两张卡明显偏低
  faulty: Object.freeze([152, 149, 151, 150, 153, 148, 92, 96]),
  healthy: Object.freeze([151, 149, 150, 152, 148, 150, 151, 149]),
  sigma: 2, // 被否掉的替代所用的 k·σ 判据
  // ft_integration.py:64 / :63
  minItersForStepTimeout: 16,
  numWarmupIters: 5, // --ft-num-warmup-iters 默认值
  // training.py::checkpoint_and_decide_exit 的扫描算例
  saveInterval: 500,
  nonPersistentSaveInterval: 100,
  exitInterval: 1000,
  scanIters: 1000,
});

// ============================================================================
// 复刻 gpu_sniff_test.py::_gather_and_check 的离群判据
// ============================================================================

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const stddev = (xs) => {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
};

// _gather_and_check：median / MAD / max(MAD, median×frac) / |x−median| > threshold
function madVerdict(vals, frac) {
  const med = median(vals);
  const mad = median(vals.map((v) => Math.abs(v - med)));
  const minDeviation = med * frac;
  const threshold = Math.max(mad, minDeviation);
  return {
    median: med,
    mad,
    minDeviation,
    threshold,
    outliers: vals.map((v, i) => [i, v]).filter(([, v]) => Math.abs(v - med) > threshold),
    // 没有下界时阈值退化成 MAD 本身
    outliersNoFloor: vals.map((v, i) => [i, v]).filter(([, v]) => Math.abs(v - med) > mad),
  };
}

// 被否掉的替代：均值 ± k·σ
function sigmaVerdict(vals, k) {
  const m = mean(vals);
  const sd = stddev(vals);
  return {
    mean: m,
    sd,
    threshold: k * sd,
    outliers: vals.map((v, i) => [i, v]).filter(([, v]) => Math.abs(v - m) > k * sd),
  };
}

const FAULTY_MAD = madVerdict(CFG.faulty, CFG.outlierMinDeviationFrac);
const FAULTY_SIGMA = sigmaVerdict(CFG.faulty, CFG.sigma);
const HEALTHY_MAD = madVerdict(CFG.healthy, CFG.outlierMinDeviationFrac);

if (FAULTY_MAD.outliers.length !== 2 || FAULTY_SIGMA.outliers.length !== 0) {
  throw new Error(
    `图 3 的立论前提被推翻：MAD 命中 ${FAULTY_MAD.outliers.length} 个、` +
      `±${CFG.sigma}σ 命中 ${FAULTY_SIGMA.outliers.length} 个，换一组算例`,
  );
}
if (HEALTHY_MAD.outliers.length !== 0 || HEALTHY_MAD.outliersNoFloor.length === 0) {
  throw new Error('图 3 右面板的立论前提被推翻：下界必须恰好挡住健康集群的噪声误报');
}

// ============================================================================
// 复刻 ft_integration.py::_maybe_update_timeouts 的四道闸门
// ============================================================================

function timeoutGates({ persistentLoaded, trIters, checkpoints, asyncCkpt, closing }) {
  const sections = [];
  if (persistentLoaded) sections.push('setup');
  if (trIters >= CFG.minItersForStepTimeout) sections.push('step');
  if (checkpoints > 0 && !asyncCkpt) sections.push('checkpointing');
  const outOfSection =
    closing && ['setup', 'step'].every((s) => sections.includes(s)) && checkpoints > 0;
  return { sections, outOfSection, willUpdate: sections.length > 0 || outOfSection };
}

const SCENARIOS = [
  {
    tag: '刚起步',
    state: { persistentLoaded: true, trIters: 8, checkpoints: 0, asyncCkpt: false, closing: false },
  },
  {
    tag: '稳态 + 同步存档',
    state: { persistentLoaded: true, trIters: 64, checkpoints: 3, asyncCkpt: false, closing: false },
  },
  {
    tag: '稳态 + 异步存档',
    state: { persistentLoaded: true, trIters: 64, checkpoints: 3, asyncCkpt: true, closing: false },
  },
  {
    tag: '从本地快照恢复',
    state: { persistentLoaded: false, trIters: 64, checkpoints: 3, asyncCkpt: false, closing: false },
  },
  {
    tag: '收尾（shutdown）',
    state: { persistentLoaded: true, trIters: 64, checkpoints: 3, asyncCkpt: true, closing: true },
  },
];
const GATE_RESULTS = SCENARIOS.map((s) => ({ ...s, out: timeoutGates(s.state) }));

// ============================================================================
// 复刻 training.py::checkpoint_and_decide_exit 的判定顺序与去重
// ============================================================================

function decideExit(iteration, opts) {
  const { save, signal, durationHit, dedup } = opts;
  const saves = [];
  if (signal) {
    if (save) saves.push('signal');
    return { saves, exit: true };
  }
  let savedCheckpoint = false;
  if (save && CFG.saveInterval && iteration % CFG.saveInterval === 0) {
    saves.push('periodic');
    savedCheckpoint = true;
  } else if (save && CFG.nonPersistentSaveInterval && iteration % CFG.nonPersistentSaveInterval === 0) {
    saves.push('non-persistent');
    savedCheckpoint = true;
  }
  if (durationHit) {
    if (save && (!dedup || !savedCheckpoint)) saves.push('duration');
    return { saves, exit: true };
  }
  if (CFG.exitInterval && iteration % CFG.exitInterval === 0) {
    if (save && (!dedup || !savedCheckpoint)) saves.push('iteration');
    return { saves, exit: true };
  }
  return { saves, exit: false };
}

function scanSaves(dedup) {
  let total = 0;
  for (let it = 1; it <= CFG.scanIters; it += 1) {
    total += decideExit(it, { save: true, signal: false, durationHit: false, dedup }).saves.length;
  }
  return total;
}
const SAVES_DEDUP = scanSaves(true);
const SAVES_NO_DEDUP = scanSaves(false);
const COLLIDE_ITER = CFG.exitInterval; // save-interval 与 exit-interval 同时命中的那一步

if (SAVES_NO_DEDUP - SAVES_DEDUP !== 1) {
  throw new Error(`图 4 的立论前提被推翻：扫描区间内去重只省下 ${SAVES_NO_DEDUP - SAVES_DEDUP} 次`);
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_comm_overlap_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) =>
  String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) {
    throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  }
  return value;
}

const sharedStyle = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .card{fill:#fff;stroke:#E4E7EC;stroke-width:1.2}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
  .cost{fill:none;stroke:#C3651F;stroke-width:2;marker-end:url(#arrowCost)}
  .edge{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:4 4}
  .ti{font-size:18px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .pt{font-size:14px;font-weight:700;fill:#2A313B}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#68717D}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .rank{font-size:11px;font-weight:700;fill:#5B6470}
  .cap{font-size:11px;fill:#747C88}
  .gl{fill:none;stroke:#C8CFDA;stroke-width:.9}
  .medline{fill:none;stroke:#2563EB;stroke-width:1.8}
  .band{fill:#EAF1FD;stroke:none;opacity:.55}
  .band2{fill:#F1F3F7;stroke:none}
  .refline{fill:none;stroke:#C8CFDA;stroke-width:1;stroke-dasharray:3 4}
`;

const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
    <marker id="arrowCost" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#C3651F"/></marker>
  </defs>`;

function rect(x, y, w, h, cls = 'neutral', radius = 8) {
  return `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"/>`;
}
function text(x, y, value, cls = 'tx', anchor = 'start') {
  return `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`;
}
function line(x1, y1, x2, y2, cls = 'edge') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}
function arrow(x1, y1, x2, y2, cls = 'main') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}
function infoBox(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 24;
  out.push(text(x + 12, y + 21, guard(title, 12, inner, `${where}/title`), 'tx'));
  out.push(line(x + 12, y + 29, x + w - 12, y + 29));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(
      text(x + 12, y + 47 + index * 16, guard(value, 10.5, inner, `${where}/L${index}`), lineCls),
    );
  });
  return out.join('\n');
}
function header(w, title, subtitle) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} __H__" width="${w}" height="__H__" role="img">`,
    defs,
    `<style>${sharedStyle}</style>`,
    text(28, 32, title, 'ti'),
    text(28, 52, subtitle, 'su'),
  ];
}

const FONT = Object.freeze({
  ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5,
  costtx: 10.5, rank: 11, cap: 11,
});

function assertNoTextOverlap(svg, name) {
  const { w: canvasW, h: canvasH } = (() => {
    const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
    return { w: Number(m[1]), h: Number(m[2]) };
  })();
  const boxes = [];
  for (const m of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      throw new Error(
        `${name}: 文字盒出画布 "${b.raw}"（${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} × ` +
          `${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)}，画布 ${canvasW}×${canvasH}）`,
      );
    }
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) {
        throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
      }
    }
  }
  return svg;
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

const fx = (v, d = 1) => v.toFixed(d).replace(/\.0+$/, '');

// ============================================================================
// 图 1：故障生命周期的五段
// ============================================================================

const STAGES = [
  {
    n: '① 观测',
    who: 'ft_integration 心跳分段',
    lines: ['setup / step / checkpointing', '三段各自记时长'],
    dl: '阈值由 NVRx 从观测反推',
    fail: '阈值失准 → 误报或迟报',
  },
  {
    n: '② 判定',
    who: 'NVRx rank monitor',
    lines: ['段内超时即判故障', '判定在依赖内部完成'],
    dl: 'section_timeouts',
    fail: '依赖缺失则整条不可用',
  },
  {
    n: '③ 中止',
    who: 'inprocess.Compose',
    lines: ['AbortTransformerEngine', 'AbortTorchDistributed', 'AbortCheckpoint'],
    dl: '各 abort 自带超时',
    fail: '通信域销毁本身可能挂住',
  },
  {
    n: '④ 清理',
    who: 'ThreadedFinalize',
    lines: ['destroy_global_state', 'destroy_rerun_state_machine'],
    dl: 'timeout = 10 s',
    fail: '超时即放弃清理',
    accent: true,
  },
  {
    n: '⑤ 重入',
    who: 'nvidia_resiliency_ext',
    lines: ['rank 重分配后再进 train', '载荷语义在依赖内部'],
    dl: 'wrapper 自有 timeout',
    fail: '残留全局状态污染下一轮',
  },
];

function renderLifecycle() {
  const W = 1272;
  const H = 460;
  const X0 = 28;
  const CW = 236;
  const GAP = 15;
  const cardX = (i) => X0 + i * (CW + GAP);
  const CARD_Y = 116;
  const CARD_H = 132;

  const parts = header(
    W,
    '图 1　「跑不动了」这一类故障要走完的五段，以及每段的 deadline',
    '同一次异常在两个平面上走不同流程：作业面（本页）问「还活着吗」，数值面（28 号页）问「算得对吗」',
  );

  parts.push(rect(X0, 74, W - 2 * X0, 30, 'acc1', 6));
  parts.push(text(X0 + 14, 94, '作业面：进程 / 通信域 / 硬件 —— 本页', 'tx'));

  STAGES.forEach((s, i) => {
    const x = cardX(i);
    parts.push(rect(x, CARD_Y, CW, CARD_H, s.accent ? 'acc2' : 'neutral'));
    parts.push(text(x + 14, CARD_Y + 24, guard(s.n, 12, 70, `fig1/n${i}`), 'tx'));
    parts.push(text(x + 14, CARD_Y + 42, guard(s.who, 10.5, CW - 28, `fig1/w${i}`), 'dim'));
    parts.push(line(x + 14, CARD_Y + 50, x + CW - 14, CARD_Y + 50));
    s.lines.forEach((l, k) =>
      parts.push(text(x + 14, CARD_Y + 68 + k * 15, guard(l, 10.5, CW - 28, `fig1/L${i}${k}`), 'sm')),
    );
    parts.push(
      text(x + 14, CARD_Y + 120, guard(`deadline：${s.dl}`, 10.5, CW - 28, `fig1/d${i}`), 'costtx'),
    );
    if (i < STAGES.length - 1) {
      parts.push(arrow(x + CW + 1, CARD_Y + CARD_H / 2, x + CW + GAP - 2, CARD_Y + CARD_H / 2));
    }
    // 失败表现另起一行，贴在卡片下方
    parts.push(
      text(x + 14, CARD_Y + CARD_H + 30, guard(s.fail, 10.5, CW - 20, `fig1/f${i}`), 'sm'),
    );
  });
  parts.push(text(X0, CARD_Y + CARD_H + 14, '这一段失败会怎样', 'rank'));

  parts.push(rect(X0, 306, W - 2 * X0, 30, 'ghost', 6));
  parts.push(
    text(X0 + 14, 326, '数值面：loss 可不可信 / SDC 归因 / 慢卡 —— 28 号页，本页不覆盖', 'sm'),
  );

  parts.push(
    infoBox(
      X0,
      350,
      608,
      92,
      '本页反复出现的同一条张力',
      [
        '要恢复就得先把坏掉的东西清理干净，而清理动作自己也可能挂住 ——',
        '销毁一个已经卡死的 NCCL 通信域就是最典型的一例。',
        '于是每一段都必须自带 deadline，第 ④ 段甚至允许「放弃清理」。',
      ],
      'neutral',
      'fig1/tension',
    ),
  );
  parts.push(
    infoBox(
      X0 + 628,
      350,
      588,
      92,
      '为了能清理干净，先做一件多余的事',
      [
        'maybe_force_nccl_backend_init 用一次无用的 all_reduce 逼 NCCL 走完惰性初始化。',
        '否则 destroy_process_group 终止不了已经在飞的 kernel，第 ④ 段就是假清理。',
        '同一模式还出现在自适应超时的样本下限与退出决定的集体通信上。',
      ],
      'acc1',
      'fig1/nccl',
    ),
  );

  return seal(parts, W, H, 'fig1-lifecycle');
}

// ============================================================================
// 图 2：三个 section 与四道阈值更新闸门
// ============================================================================

function renderSections() {
  const W = 1272;
  const H = 520;
  const TX = 150;
  const TW = 1000;
  const T_MAX = 100;
  const t = (v) => TX + (v / T_MAX) * TW;

  const parts = header(
    W,
    '图 2　NVRx 只记三个 section；「现在能不能更新阈值」由四道闸门决定',
    `warmup 期（前 ${CFG.numWarmupIters} 次迭代）根本不开 section，阈值也就不会被一次性开销带偏`,
  );

  // ---- 上半：一条迭代时间轴 ----
  const LY = 92;
  const LH = 26;
  const lanes = [
    {
      tag: 'setup',
      blocks: [[0, 16, 'start_section("setup")', 'acc1']],
    },
    {
      tag: 'step',
      blocks: [
        [16, 34, `warmup ×${CFG.numWarmupIters}：不开 section`, 'ghost'],
        [34, 46, 'step', 'acc1'],
        [48, 60, 'step', 'acc1'],
        [62, 68, 'step', 'acc1'],
        [80, 96, 'step（eval 也记这里）', 'acc1'],
      ],
    },
    {
      tag: 'checkpointing',
      blocks: [[68, 78, 'checkpointing', 'acc2']],
    },
  ];
  lanes.forEach((lane, i) => {
    const y = LY + i * (LH + 10);
    parts.push(text(28, y + 18, guard(lane.tag, 11, 116, `fig2/lane${i}`), 'rank'));
    lane.blocks.forEach(([a, b, label, cls], k) => {
      const x = t(a);
      const w = t(b) - t(a);
      parts.push(rect(x, y, w, LH, cls, 5));
      parts.push(text(x + w / 2, y + 17, guard(label, 10.5, w - 8, `fig2/b${i}${k}`), 'sm', 'middle'));
    });
  });
  const axisY = LY + 3 * (LH + 10) + 6;
  parts.push(line(TX, axisY, TX + TW, axisY, 'gl'));
  parts.push(text(TX, axisY + 18, '进程启动', 'sm'));
  parts.push(text(t(16), axisY + 18, 'setup 结束 / 首次训练步', 'sm', 'middle'));
  parts.push(text(TX + TW, axisY + 18, '训练时间', 'sm', 'end'));
  parts.push(
    text(
      t(16),
      LY - 10,
      guard('on_training_step_start 关掉 setup 段', 10.5, 260, 'fig2/note'),
      'dim',
      'middle',
    ),
  );

  // ---- 下半：闸门判定表 ----
  const GY = axisY + 46;
  const NAME_W = 176;
  const COL_W = (TW + TX - 28 - NAME_W) / GATE_RESULTS.length;
  const ROWS = ['setup', 'step', 'checkpointing', 'out-of-section'];

  parts.push(text(28, GY - 10, '阈值更新闸门（复刻 _maybe_update_timeouts）', 'pt'));
  GATE_RESULTS.forEach((g, c) => {
    const x = 28 + NAME_W + c * COL_W;
    parts.push(text(x + COL_W / 2, GY + 14, guard(g.tag, 10.5, COL_W - 8, `fig2/h${c}`), 'rank', 'middle'));
  });
  ROWS.forEach((row, r) => {
    const y = GY + 26 + r * 34;
    parts.push(text(28, y + 21, guard(row, 10.5, NAME_W - 8, `fig2/r${r}`), 'dim'));
    GATE_RESULTS.forEach((g, c) => {
      const x = 28 + NAME_W + c * COL_W;
      const ok = row === 'out-of-section' ? g.out.outOfSection : g.out.sections.includes(row);
      parts.push(rect(x + 4, y, COL_W - 8, 28, ok ? 'acc1' : 'ghost', 5));
      parts.push(
        text(x + COL_W / 2, y + 19, ok ? '更新' : '不更新', 'sm', 'middle'),
      );
    });
  });

  const noteY = GY + 26 + ROWS.length * 34 + 10;
  parts.push(
    infoBox(
      28,
      noteY,
      600,
      86,
      '每道闸门挡住的是一种把阈值算歪的方式',
      [
        `step：样本 < ${CFG.minItersForStepTimeout} 次迭代就不动阈值，头几步的一次性开销会把它抬得离谱。`,
        'setup：只有加载过持久 checkpoint 才算数 —— 内存里的本地快照读得太快，会低估 setup。',
        'checkpointing：异步存档下跨 run 波动过大，源码直接放弃更新这一段。',
      ],
      'neutral',
      'fig2/gates',
    ),
  );
  parts.push(
    infoBox(
      648,
      noteY,
      596,
      86,
      '依赖边界：阈值到底怎么算，不在本仓',
      [
        'Megatron 只决定「哪些 section 现在有资格更新」，然后调用',
        'rmon_cli.calculate_and_set_section_timeouts(...)，并把结果 json 落盘。',
        '反推公式、分位数、安全裕度都在 nvidia_resiliency_ext 内部，本页不作陈述。',
      ],
      'ghost',
      'fig2/dep',
    ),
  );

  return seal(parts, W, H, 'fig2-sections');
}

// ============================================================================
// 图 3：原理图 —— 离群判据
// ============================================================================

function sniffPanel(parts, { x, w, title, vals, mad, sigma, showSigma, showNoFloor, where }) {
  const ROW = 26;
  const top = 128;
  const lo = Math.min(...vals) * 0.92;
  const hi = Math.max(...vals) * 1.02;
  const bx = x + 58;
  const valCol = x + w - 34; // 数值固定一列，避免和中位数竖线互相压
  const bw = valCol - bx - 10;
  const sx = (v) => bx + ((v - lo) / (hi - lo)) * bw;
  const bottom = top + vals.length * ROW;

  parts.push(text(x, top - 56, guard(title, 12, w, `${where}/title`), 'tx'));

  // mean ± kσ 带（被否掉的替代，画在底层）
  if (showSigma) {
    const a = sx(Math.max(lo, sigma.mean - sigma.threshold));
    const b = sx(Math.min(hi, sigma.mean + sigma.threshold));
    parts.push(`<rect class="band2" x="${a}" y="${top - 10}" width="${b - a}" height="${bottom - top + 14}"/>`);
    parts.push(
      text(
        (a + b) / 2,
        top - 34,
        guard(`均值 ± ${CFG.sigma}σ = ${fx(sigma.mean, 1)} ± ${fx(sigma.threshold, 1)}：带太宽，一根都没判出`, 10.5, b - a + 120, `${where}/sig`),
        'sm',
        'middle',
      ),
    );
  }

  // median ± threshold 带
  const ma = sx(Math.max(lo, mad.median - mad.threshold));
  const mb = sx(Math.min(hi, mad.median + mad.threshold));
  parts.push(`<rect class="band" x="${ma}" y="${top - 6}" width="${mb - ma}" height="${bottom - top + 6}"/>`);
  parts.push(line(sx(mad.median), top - 6, sx(mad.median), bottom, 'medline'));
  parts.push(
    text(
      sx(mad.median),
      top - 14,
      guard(`median ${fx(mad.median, 1)} ± threshold ${fx(mad.threshold, 2)}`, 10.5, 240, `${where}/med`),
      'dim',
      'middle',
    ),
  );

  const outSet = new Set(mad.outliers.map(([i]) => i));
  const noFloorSet = new Set(mad.outliersNoFloor.map(([i]) => i));
  vals.forEach((v, i) => {
    const y = top + i * ROW;
    const flagged = outSet.has(i);
    const falsePositive = showNoFloor && !flagged && noFloorSet.has(i);
    parts.push(text(x, y + 15, `rank ${i}`, 'sm'));
    parts.push(
      rect(bx, y + 2, Math.max(sx(v) - bx, 2), ROW - 8, flagged ? 'acc2' : falsePositive ? 'ghost' : 'neutral', 4),
    );
    parts.push(
      text(valCol, y + 15, guard(String(v), 10.5, 32, `${where}/v${i}`), flagged ? 'costtx' : 'sm'),
    );
    if (falsePositive) {
      parts.push(text(valCol + 40, y + 15, guard('去掉下界会误报这根', 10.5, 140, `${where}/fp${i}`), 'costtx'));
    }
  });
  parts.push(
    text(bx, bottom + 16, guard(`横轴自 ${fx(lo, 1)} 起，不是 0 —— 本图比较的是与中位数的偏离`, 10.5, bw, `${where}/axis`), 'cap'),
  );
  return bottom + 20;
}

function renderSniff() {
  const W = 1272;
  const H = 520;

  const parts = header(
    W,
    '图 3　凭什么说这张卡有问题：中位数 + MAD + 下界，而不是均值 ± σ',
    `同一组 AllReduce 带宽读数（GB/s，${CFG.faulty.length} 个 rank 跑同一个合成负载，差异只可能来自硬件或链路）`,
  );

  const bottomL = sniffPanel(parts, {
    x: 28,
    w: 560,
    title: '故障集群：两张卡明显偏低',
    vals: CFG.faulty,
    mad: FAULTY_MAD,
    sigma: FAULTY_SIGMA,
    showSigma: true,
    showNoFloor: false,
    where: 'fig3/L',
  });
  const bottomR = sniffPanel(parts, {
    x: 660,
    w: 400,
    title: '健康集群：全部在噪声范围内',
    vals: CFG.healthy,
    mad: HEALTHY_MAD,
    sigma: sigmaVerdict(CFG.healthy, CFG.sigma),
    showSigma: false,
    showNoFloor: true,
    where: 'fig3/R',
  });

  const y = Math.max(bottomL, bottomR) + 24;
  parts.push(
    infoBox(
      28,
      y,
      608,
      120,
      `MAD 判据抓到 ${FAULTY_MAD.outliers.length} 张卡，均值 ± ${CFG.sigma}σ 抓到 ${FAULTY_SIGMA.outliers.length} 张`,
      [
        `median = ${fx(FAULTY_MAD.median)}，MAD = ${fx(FAULTY_MAD.mad)}，下界 = median × ${CFG.outlierMinDeviationFrac} = ${fx(FAULTY_MAD.minDeviation, 2)}`,
        `threshold = max(MAD, 下界) = ${fx(FAULTY_MAD.threshold, 2)} → 命中 rank ${FAULTY_MAD.outliers.map(([i]) => i).join(', ')}`,
        `对照：mean = ${fx(FAULTY_SIGMA.mean, 2)}（被两张坏卡拉低），σ = ${fx(FAULTY_SIGMA.sd, 2)}（被它们拉大）`,
        `${CFG.sigma}σ = ${fx(FAULTY_SIGMA.threshold, 2)} 反而比两张坏卡的偏离还大 —— 离群点把判据自己撑开了`,
      ],
      'acc2',
      'fig3/left',
    ),
  );
  parts.push(
    infoBox(
      656,
      y,
      588,
      120,
      '下界挡住的是反方向的错误：健康集群上的误报',
      [
        `median = ${fx(HEALTHY_MAD.median)}，MAD = ${fx(HEALTHY_MAD.mad)} —— 集群健康时 MAD 会非常小。`,
        `纯按 MAD 判（阈值 ${fx(HEALTHY_MAD.mad)}）会把 ${HEALTHY_MAD.outliersNoFloor.length} 根正常测量噪声报成离群；`,
        `加上下界后 threshold = ${fx(HEALTHY_MAD.threshold, 2)}，命中 ${HEALTHY_MAD.outliers.length} 根。`,
        '注：源码常量注释写的是「10% of mean」，实现取的是 median × 0.10，两者不一致。',
      ],
      'neutral',
      'fig3/right',
    ),
  );

  return seal(parts, W, H, 'fig3-sniff');
}

// ============================================================================
// 图 4：退出与存档路径
// ============================================================================

const EXIT_PATHS = [
  ['① 信号', 'any(signals_received()) —— all-gather 后全体一致', '存档后立即退出，不置 saved_checkpoint'],
  ['② 周期存档', 'iteration % save_interval == 0', '置 saved_checkpoint，不退出'],
  ['②′ 非持久存档', 'elif iteration % non_persistent_save_interval == 0', '与 ② 互斥；同样置 saved_checkpoint'],
  ['③ 时长', 'train_time > exit_duration_in_mins，经 MAX all-reduce', 'if save and not saved_checkpoint 才存'],
  ['④ 迭代 / 阶段切换', 'iteration % exit_interval == 0 或命中 phase_transition', '同上，去重后退出'],
];

function renderExit() {
  const W = 1272;
  const H = 452;
  const X0 = 28;
  const RW = 470;

  const parts = header(
    W,
    '图 4　一次迭代里的五条存档 / 退出路径，以及 saved_checkpoint 去重',
    `算例：save_interval=${CFG.saveInterval}、non_persistent=${CFG.nonPersistentSaveInterval}、exit_interval=${CFG.exitInterval}，扫描前 ${CFG.scanIters} 次迭代`,
  );

  EXIT_PATHS.forEach((p, i) => {
    const y = 86 + i * 62;
    const isCollide = i >= 3;
    parts.push(rect(X0, y, 720, 52, isCollide ? 'acc1' : 'neutral', 6));
    parts.push(text(X0 + 14, y + 21, guard(p[0], 12, 150, `fig4/n${i}`), 'tx'));
    parts.push(text(X0 + 160, y + 21, guard(p[1], 10.5, 546, `fig4/c${i}`), 'dim'));
    parts.push(text(X0 + 14, y + 41, guard(p[2], 10.5, 700, `fig4/e${i}`), 'sm'));
    if (i < EXIT_PATHS.length - 1) {
      parts.push(arrow(X0 + 360, y + 53, X0 + 360, y + 61, 'aux'));
    }
  });

  parts.push(
    infoBox(
      X0 + 748,
      86,
      RW,
      148,
      `碰撞点：第 ${COLLIDE_ITER} 次迭代`,
      [
        `${COLLIDE_ITER} % ${CFG.saveInterval} == 0 → ② 命中，saved_checkpoint = True`,
        `${COLLIDE_ITER} % ${CFG.exitInterval} == 0 → ④ 也命中`,
        `有去重：这一步存 1 次；无去重：存 2 次`,
        `扫描 ${CFG.scanIters} 次迭代，合计 ${SAVES_DEDUP} 次 vs ${SAVES_NO_DEDUP} 次`,
        '万卡尺度下一次全量存档是分钟级集体操作，重复即实打实的浪费',
      ],
      'acc2',
      'fig4/collide',
    ),
  );
  parts.push(
    infoBox(
      X0 + 748,
      250,
      RW,
      144,
      '为什么 ③ 要走一次 all-reduce',
      [
        '各 rank 的墙钟不完全一致，train_time 的比较结果因此可能不同。',
        '把布尔量放进 CUDA 张量做 MAX all-reduce，退出决定才是集体一致的。',
        '与 ① 的信号 all-gather 同一道理：判据必须由所有参与者算出同一答案，',
        '否则会出现「一部分 rank 已退出、另一部分还阻塞在集合通信上」的挂死。',
      ],
      'neutral',
      'fig4/why',
    ),
  );

  return seal(parts, W, H, 'fig4-exit');
}

// ============================================================================

const outputs = new Map([
  ['megatron_resilience_lifecycle.svg', renderLifecycle()],
  ['megatron_resilience_ft_sections.svg', renderSections()],
  ['megatron_resilience_sniff_outlier.svg', renderSniff()],
  ['megatron_resilience_exit_paths.svg', renderExit()],
]);

export {
  CFG, FAULTY_MAD, FAULTY_SIGMA, HEALTHY_MAD, GATE_RESULTS,
  SAVES_DEDUP, SAVES_NO_DEDUP, COLLIDE_ITER,
  madVerdict, sigmaVerdict, timeoutGates, decideExit, outputs,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir =
    process.argv[2] ??
    join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of outputs) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`${name}  ${svg.length} bytes`);
  }
  console.log(
    `\nsniff：median=${fx(FAULTY_MAD.median)} MAD=${fx(FAULTY_MAD.mad)} threshold=${fx(FAULTY_MAD.threshold, 2)} ` +
      `命中 ${FAULTY_MAD.outliers.map(([i]) => i).join(',')}；±${CFG.sigma}σ=${fx(FAULTY_SIGMA.threshold, 2)} 命中 ${FAULTY_SIGMA.outliers.length} 个`,
  );
  console.log(`exit：去重 ${SAVES_DEDUP} 次 vs 不去重 ${SAVES_NO_DEDUP} 次`);
}
