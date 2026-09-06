// 28_megatron_training_stability_observability_analysis.md 的三张图。
//
// 图 1：**原理图** —— `RerunStateMachine.is_unexpectedly_large` 的判据。前 N 个样本用来估计
//       最大值，之后按 max×threshold 触发；两条失效路径（窗口内混进尖峰、窗口后再不重估）
//       都由同一段复刻代码算出来，而不是嘴上说说。
// 图 2：梯度范数这一条轴上串了三道闸 —— 非有限值跳步、裁剪、超阈跳步，外加一条独立范数组
//       的旁路；哪个区间发生什么、谁参与判定，全部标在轴上。
// 图 3：`--calculate-per-token-loss` 下 aux-loss 的跨 rank 归一：闭式因子 local×group_size
//       在 THD padding / dynamic CP 让各 rank 有效 token 数不等时逐 rank 算错多少。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 是本页的 principle figure。取一条 loss 序列，画三条泳道：
//   上 = 数值本身（柱），采样窗口内的柱用 ghost，armed 之后用 neutral，被判尖峰的用 acc2；
//   中 = 当前 max 估计（阶梯线），窗口结束后变成一条水平线 —— 这正是「再不重估」的可视证据；
//   下 = 触发线 max×threshold。
// 右栏两个盒子分别复演两条失效路径：
//   (a) 把窗口内某个样本换成一次尖峰 → max 被抬高 → 后面真正的尖峰不再触发；
//   (b) 训练后期 loss 已降到 x，而触发线仍停在窗口期算出的值 → 需要多少倍的局部尖峰才触发。
// 两条路径的结论都是算出来的数字，不是形容词。
//
// 图 2 布局：一条横轴是 ‖g‖，用 clip_grad 与 grad_norm_skip_threshold 切成三段，
// 左端另挂一个「非有限」区（不在数轴上，用断口表示）。每段下面写这一步的结局。
// 下方另一条独立泳道画 mtp 组：自己的范数、自己的裁剪，且**不参与**超阈跳步判定。
//
// 图 3 布局：四个 rank 的有效 token 数条形；两行对照 ——「闭式：local×group」与
// 「实测：all_reduce(sum)」，逐 rank 标出比值。右侧再放一组等长 token 的对照，
// 说明闭式因子在均匀情形下是对的，被推翻的是它的前提而不是它的代数。
//
// 用法：node tools/figs/svg/megatron_stability_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  // rerun_state_machine.py::is_unexpectedly_large 的两个参数
  threshold: 10, // docstring 示例即 threshold=10
  numSamplesDefault: 100, // 源码默认 num_samples=100
  numSamples: 8, // 画图用的小窗口，正文会说明它替代的是默认值
  // 一条 loss 序列：前 8 个进采样窗口，之后进入 armed
  series: Object.freeze([
    2.10, 2.05, 2.02, 1.98, 1.95, 1.93, 1.90, 1.88,
    1.86, 1.84, 22.5, 1.83, 3.90, 1.81, 1.79, 1.78,
  ]),
  // 失效路径 (a)：窗口内混进一次尖峰
  pollutedIndex: 3,
  pollutedValue: 9.0,
  // 失效路径 (b)：训练后期的典型 loss
  lateLoss: 0.5,

  // 梯度闸门算例
  clipGrad: 1.0,
  gradNormSkipThreshold: 10.0,
  mainNorm: 3.4,
  mtpNorm: 18.6,

  // aux-loss 归一算例：一个 tp_cp 组内四个 rank 的有效 token 数（THD packing 后不等长）
  validTokens: Object.freeze([512, 384, 448, 256]),
  uniformTokens: Object.freeze([512, 512, 512, 512]),
});

// ============================================================================
// 复刻 rerun_state_machine.py::is_unexpectedly_large
// ============================================================================

function spikeDetector(series, { threshold, numSamples }) {
  let count = 0;
  let maxValue = 0;
  const trace = [];
  for (const raw of series) {
    const value = Math.abs(raw);
    if (Number.isNaN(value) || !Number.isFinite(value)) {
      // 源码显式忽略 NaN / Inf：它们由别的检查负责
      trace.push({ value: raw, phase: 'ignored', flagged: false, maxValue, trigger: null });
      continue;
    }
    if (count < numSamples) {
      count += 1;
      maxValue = Math.max(maxValue, value);
      trace.push({ value: raw, phase: 'sampling', flagged: false, maxValue, trigger: null });
    } else {
      const trigger = maxValue * threshold;
      trace.push({ value: raw, phase: 'armed', flagged: value >= trigger, maxValue, trigger });
    }
  }
  return { trace, maxValue, trigger: maxValue * threshold };
}

const BASE = spikeDetector(CFG.series, CFG);

// 失效路径 (a)：采样窗口里混进一次尖峰，max 估计被永久抬高
const polluted = CFG.series.map((v, i) => (i === CFG.pollutedIndex ? CFG.pollutedValue : v));
const POLLUTED = spikeDetector(polluted, CFG);

// 失效路径 (b)：触发线不随训练推进下移
const LATE_RATIO = BASE.trigger / CFG.lateLoss;

const BASE_FLAGS = BASE.trace.filter((t) => t.flagged).length;
const POLLUTED_FLAGS = POLLUTED.trace.filter((t) => t.flagged).length;

if (BASE_FLAGS !== 1) {
  throw new Error(`图 1 的立论前提被推翻：基准序列命中 ${BASE_FLAGS} 次尖峰，应为 1`);
}
if (POLLUTED_FLAGS !== 0) {
  throw new Error(`图 1 的失效路径 (a) 被推翻：污染后仍命中 ${POLLUTED_FLAGS} 次`);
}

// ============================================================================
// aux-loss 归一：闭式因子 vs 实测计数
// ============================================================================

function auxScale(tokens) {
  const measured = tokens.reduce((a, b) => a + b, 0); // all_reduce(SUM)
  const closed = tokens.map((t) => t * tokens.length); // local × group_size
  return { measured, closed, ratio: closed.map((c) => c / measured) };
}
const AUX_SKEWED = auxScale(CFG.validTokens);
const AUX_UNIFORM = auxScale(CFG.uniformTokens);

if (AUX_UNIFORM.ratio.some((r) => r !== 1)) {
  throw new Error('图 3 的对照被推翻：等长 token 下闭式因子应当与实测完全一致');
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
  .maxline{fill:none;stroke:#2563EB;stroke-width:2}
  .trigline{fill:none;stroke:#C3651F;stroke-width:2;stroke-dasharray:7 4}
  .refline{fill:none;stroke:#C8CFDA;stroke-width:1;stroke-dasharray:3 4}
  .band{fill:#F5F7FA;stroke:none}
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
function polyline(points, cls) {
  return `<path class="${cls}" d="M ${points.map(([x, y]) => `${x} ${y}`).join(' L ')}"/>`;
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
// 图 1：原理图 —— 尖峰判据的采样窗口与两条失效路径
// ============================================================================

function renderSpike() {
  const W = 1272;
  const H = 500;
  const PX = 92;
  const PW = 636;
  const PY = 100;
  const PH = 258;
  const RX = 812;
  const RW = 432;

  const n = CFG.series.length;
  const slot = PW / n;
  // 序列跨一个数量级（1.78 .. 22.5），线性轴会把正常步压成一条线，所以纵轴取对数
  const ylo = 1;
  const yhi = Math.max(BASE.trigger, POLLUTED.trigger, ...CFG.series) * 1.35;
  const yspan = Math.log10(yhi) - Math.log10(ylo);
  const sy = (v) => PY + PH - ((Math.log10(Math.max(v, ylo)) - Math.log10(ylo)) / yspan) * PH;
  const cx = (i) => PX + slot * (i + 0.5);

  const parts = header(
    W,
    '图 1　尖峰判据：前若干步只用来估最大值，之后才开始判',
    `is_unexpectedly_large(threshold=${CFG.threshold})；图示窗口取 ${CFG.numSamples} 个样本，源码默认 num_samples=${CFG.numSamplesDefault}；纵轴对数`,
  );

  // 采样窗口底色
  parts.push(`<rect class="band" x="${PX}" y="${PY}" width="${slot * CFG.numSamples}" height="${PH}"/>`);
  parts.push(
    text(
      PX + (slot * CFG.numSamples) / 2,
      PY - 10,
      guard(`采样窗口：只累计 max，不判定（${CFG.numSamples} 步）`, 10.5, slot * CFG.numSamples + 60, 'fig1/win'),
      'sm',
      'middle',
    ),
  );
  parts.push(
    text(
      PX + slot * CFG.numSamples + 8,
      PY - 10,
      guard('armed：开始按 max × threshold 判定', 10.5, PW - slot * CFG.numSamples - 10, 'fig1/armed'),
      'dim',
    ),
  );

  // 柱
  BASE.trace.forEach((t, i) => {
    const x = cx(i) - slot * 0.32;
    const w = slot * 0.64;
    const y = sy(t.value);
    const cls = t.flagged ? 'acc2' : t.phase === 'sampling' ? 'ghost' : 'neutral';
    parts.push(rect(x, y, w, PY + PH - y, cls, 3));
    if (t.flagged || i === 0 || t.value > 3) {
      parts.push(text(cx(i), y - 6, guard(String(t.value), 10.5, slot + 10, `fig1/v${i}`), t.flagged ? 'costtx' : 'sm', 'middle'));
    }
  });

  // max 估计阶梯：窗口内爬升，窗口后水平
  parts.push(polyline(BASE.trace.map((t, i) => [cx(i), sy(t.maxValue)]), 'maxline'));
  parts.push(text(PX + PW + 6, sy(BASE.maxValue) + 4, guard(`max=${BASE.maxValue}`, 10.5, 62, 'fig1/max'), 'dim'));

  // 触发线
  parts.push(line(PX + slot * CFG.numSamples, sy(BASE.trigger), PX + PW, sy(BASE.trigger), 'trigline'));
  parts.push(
    text(
      PX + PW + 6,
      sy(BASE.trigger) + 4,
      guard(`触发 ${fx(BASE.trigger, 1)}`, 10.5, 62, 'fig1/trig'),
      'costtx',
    ),
  );

  // 横轴
  parts.push(line(PX, PY + PH, PX + PW, PY + PH, 'gl'));
  BASE.trace.forEach((t, i) => {
    if (i % 2 === 0) parts.push(text(cx(i), PY + PH + 18, String(i), 'sm', 'middle'));
  });
  parts.push(text(PX + PW / 2, PY + PH + 38, '第 k 次 validate_result 调用', 'rank', 'middle'));
  parts.push(text(PX - 8, PY + 12, '|loss|（log）', 'rank', 'end'));

  parts.push(
    infoBox(
      RX,
      PY - 8,
      RW,
      140,
      `失效路径 (a)：窗口里混进一次尖峰，判据就废了`,
      [
        `把第 ${CFG.pollutedIndex} 个样本换成 ${CFG.pollutedValue}（窗口内的一次尖峰）：`,
        `max 由 ${BASE.maxValue} 抬到 ${POLLUTED.maxValue}，触发线由 ${fx(BASE.trigger, 1)} 抬到 ${fx(POLLUTED.trigger, 1)}。`,
        `同一条序列后面那次 ${CFG.series[10]} 的真尖峰，命中数从 ${BASE_FLAGS} 变成 ${POLLUTED_FLAGS}。`,
        '判据是相对量，而参照物本身来自训练最开始那几步。',
      ],
      'acc2',
      'fig1/a',
    ),
  );
  parts.push(
    infoBox(
      RX,
      PY + 146,
      RW,
      156,
      '失效路径 (b)：触发线不随训练推进下移',
      [
        '除非调用方显式传 resample=True，max 在窗口结束后永不重估。',
        `训练后期 loss 降到 ${CFG.lateLoss} 时，触发线仍停在 ${fx(BASE.trigger, 1)} ——`,
        `此时需要一次 ${fx(LATE_RATIO, 0)}× 于当前 loss 的尖峰才会被判出。`,
        '也就是说：越到后期，这条判据越迟钝。',
        '它挡的是「早期尺度上的数量级异常」，不是「相对当前水平的异常」。',
      ],
      'neutral',
      'fig1/b',
    ),
  );
  parts.push(
    infoBox(
      RX,
      PY + 310,
      RW,
      92,
      'NaN / Inf 走另一条路',
      [
        '源码在这里显式 return False，注释是 They should be checked separately：',
        '非有限值由 --check-for-nan-in-loss-and-grad 与优化器的溢出闸门负责，',
        '本判据只处理「有限但异常大」。',
      ],
      'ghost',
      'fig1/c',
    ),
  );

  return seal(parts, W, H, 'fig1-spike');
}

// ============================================================================
// 图 2：梯度范数轴上的四道闸
// ============================================================================

function renderGradGates() {
  const W = 1272;
  const H = 512;
  const AX = 300;
  const AW = 720;
  const AY = 150;

  // ‖g‖ 跨数量级，用对数轴才能让三段都有可读宽度；两个断点是 clip_grad 与 skip 阈值
  const lo = CFG.clipGrad / 10;
  const top = CFG.gradNormSkipThreshold * 4;
  const span = Math.log10(top) - Math.log10(lo);
  const sx = (v) => AX + ((Math.log10(v) - Math.log10(lo)) / span) * AW;

  const parts = header(
    W,
    '图 2　同一条梯度范数轴上串了三道闸与一条旁路，各自的结局并不相同',
    `算例：clip_grad=${CFG.clipGrad}，grad_norm_skip_threshold=${CFG.gradNormSkipThreshold}（默认 inf，即关闭）；横轴按 ‖g‖ 取对数`,
  );

  // 非有限区（不在数轴上）
  parts.push(rect(28, AY - 26, 236, 52, 'acc2', 6));
  parts.push(text(146, AY - 6, '非有限（inf / nan）', 'tx', 'middle'));
  parts.push(text(146, AY + 12, '闸①：整步丢弃，scaler 降档', 'sm', 'middle'));
  parts.push(text(276, AY + 4, '⋯', 'rank', 'middle'));

  // 数轴
  parts.push(line(AX, AY, AX + AW, AY, 'gl'));
  const marks = [
    [lo, String(lo)],
    [CFG.clipGrad, `clip_grad=${CFG.clipGrad}`],
    [CFG.gradNormSkipThreshold, `skip=${CFG.gradNormSkipThreshold}`],
  ];
  marks.forEach(([v, label]) => {
    parts.push(line(sx(v), AY - 10, sx(v), AY + 10, 'gl'));
    parts.push(text(sx(v), AY - 18, guard(label, 10.5, 150, 'fig2/mark'), 'dim', 'middle'));
  });
  parts.push(text(AX + AW, AY - 18, '‖g‖', 'rank', 'end'));

  const segs = [
    [lo, CFG.clipGrad, '照常更新', '不触任何闸', 'ghost'],
    [CFG.clipGrad, CFG.gradNormSkipThreshold, '闸②：等比缩到 clip_grad 后更新', 'g ← g·clip/‖g‖', 'acc1'],
    [CFG.gradNormSkipThreshold, top, '闸③：整步丢弃，参数不动', 'should_skip_update=True', 'acc2'],
  ];
  segs.forEach(([a, b, title, sub, cls], i) => {
    const x = sx(a);
    const w = sx(b) - sx(a);
    parts.push(rect(x + 2, AY + 22, w - 4, 56, cls, 5));
    parts.push(text(x + w / 2, AY + 44, guard(title, 10.5, w - 12, `fig2/s${i}`), 'sm', 'middle'));
    parts.push(text(x + w / 2, AY + 62, guard(sub, 10.5, w - 12, `fig2/u${i}`), 'sm', 'middle'));
  });

  // 本例的主范数落点
  parts.push(line(sx(CFG.mainNorm), AY + 22, sx(CFG.mainNorm), AY + 96, 'trigline'));
  parts.push(
    text(sx(CFG.mainNorm), AY + 112, guard(`主组 ‖g‖=${CFG.mainNorm} → 裁剪后更新`, 10.5, 220, 'fig2/main'), 'costtx', 'middle'),
  );

  // 独立范数组旁路
  const LY = AY + 148;
  parts.push(rect(AX, LY, AW, 56, 'neutral', 6));
  parts.push(text(AX + 12, LY + 22, "独立范数组 'mtp'：自己算范数、自己按同一个 clip_grad 裁剪", 'tx'));
  parts.push(
    text(
      AX + 12,
      LY + 42,
      guard(
        `本例 ‖g_mtp‖=${CFG.mtpNorm} > skip=${CFG.gradNormSkipThreshold}，但它不参与闸③ 的判定 —— 整步不会被丢`,
        10.5,
        AW - 24,
        'fig2/mtp',
      ),
      'sm',
    ),
  );
  parts.push(text(28, LY + 22, '旁路', 'rank'));
  parts.push(text(28, LY + 42, '（mtp_detach_heads=True 时）', 'sm'));

  parts.push(
    infoBox(
      28,
      LY + 76,
      600,
      120,
      '三道闸挡的不是同一类问题',
      [
        '闸①（非有限）：数值已经坏了，任何依赖 ‖g‖ 的计算都失去意义，必须先判。',
        '闸②（裁剪）：范数有限但偏大，缩放后这一步仍然有用 —— 不浪费前反向。',
        '闸③（超阈跳步）：范数有限却大到「缩放也不敢信」，宁可整步作废。',
        '判据差别在于「这一步的梯度还值不值得用」，而不是数值大小本身。',
      ],
      'neutral',
      'fig2/why',
    ),
  );
  parts.push(
    infoBox(
      648,
      LY + 76,
      596,
      120,
      '两条容易读错的边界',
      [
        `grad_norm_skip_threshold 默认 float('inf')，只有配置项、没有 CLI 开关，`,
        '且只在 ChainedOptimizer.step 这条路径上生效。',
        "启用 'mtp' 组后，日志里的 grad norm 只反映主组 —— 跨版本对比曲线时",
        '这是一次口径变化，不是模型变稳了。',
      ],
      'acc2',
      'fig2/edge',
    ),
  );

  return seal(parts, W, H, 'fig2-gates');
}

// ============================================================================
// 图 3：aux-loss 归一，闭式因子 vs 实测计数
// ============================================================================

function auxPanel(parts, { x, w, title, tokens, res, where }) {
  const ROW = 30;
  const top = 132;
  const bx = x + 60;
  const bw = w - 196;
  const maxTok = Math.max(...tokens);
  const sxx = (t) => (t / maxTok) * bw;

  parts.push(text(x, top - 44, guard(title, 12, w, `${where}/t`), 'tx'));
  parts.push(text(bx, top - 24, guard('有效 token 数', 10.5, 120, `${where}/h1`), 'rank'));
  parts.push(text(x + w - 136, top - 24, guard('闭式 / 实测 = 权重偏差', 10.5, 136, `${where}/h2`), 'rank'));

  tokens.forEach((t, i) => {
    const y = top + i * ROW;
    const ratio = res.ratio[i];
    const off = Math.abs(ratio - 1) > 1e-9;
    parts.push(text(x, y + 15, `rank ${i}`, 'sm'));
    parts.push(rect(bx, y + 2, sxx(t), ROW - 10, off ? 'acc2' : 'ghost', 4));
    parts.push(text(bx + 8, y + 15, guard(String(t), 10.5, 44, `${where}/v${i}`), 'sm'));
    parts.push(
      text(
        x + w - 136,
        y + 15,
        guard(`${res.closed[i]} / ${res.measured} = ${fx(ratio, 2)}×`, 10.5, 136, `${where}/r${i}`),
        off ? 'costtx' : 'sm',
      ),
    );
  });
  const bottom = top + tokens.length * ROW;
  parts.push(
    text(
      x,
      bottom + 18,
      guard(`all_reduce(SUM) = ${res.measured}；闭式 local×${tokens.length} 逐 rank 见右列`, 10.5, w, `${where}/sum`),
      'dim',
    ),
  );
  return bottom + 24;
}

function renderAuxScale() {
  const W = 1272;
  const H = 440;

  const parts = header(
    W,
    '图 3　aux-loss 的跨 rank 归一：闭式因子被前提推翻，不是被精度推翻',
    '--calculate-per-token-loss 下 finalize_model_grads 会按全局非 padding token 数整体相除，本图看的是分子怎么补',
  );

  const b1 = auxPanel(parts, {
    x: 28,
    w: 590,
    title: 'THD packing / dynamic CP：同组各 rank 有效 token 数不等',
    tokens: CFG.validTokens,
    res: AUX_SKEWED,
    where: 'fig3/L',
  });
  const b2 = auxPanel(parts, {
    x: 656,
    w: 590,
    title: '定长且无 padding：闭式因子与实测完全一致',
    tokens: CFG.uniformTokens,
    res: AUX_UNIFORM,
    where: 'fig3/R',
  });

  const y = Math.max(b1, b2) + 10;
  parts.push(
    infoBox(
      28,
      y,
      590,
      112,
      '闭式因子错在哪',
      [
        `local × group_size 假设「同组每个 rank 的有效 token 数相等」。`,
        `本例最重的 rank 被高估 ${fx(Math.max(...AUX_SKEWED.ratio), 2)}×，最轻的被低估到 ${fx(Math.min(...AUX_SKEWED.ratio), 2)}×，`,
        '于是同一个 aux loss 在不同 rank 上被赋予不同权重 —— 而它本该是一个全局量。',
        '注释原话：valid token counts can differ by rank/group.',
      ],
      'acc2',
      'fig3/left',
    ),
  );
  parts.push(
    infoBox(
      646,
      y,
      600,
      112,
      '代价：每个 aux-loss 域每步多一次 all_reduce',
      [
        '实测法把本域有效 token 数装进张量，沿 aux_loss_scale_reduce_groups 逐组求和。',
        '判据不是「实测更准」，而是闭式因子的前提在变长与动态 CP 下不再成立；',
        '宁可每步多付一次标量 all_reduce，也不留一个会静默算错的常数。',
        'reduce_group 缺失时直接 assert，不做猜测性回退。',
      ],
      'neutral',
      'fig3/right',
    ),
  );

  return seal(parts, W, H, 'fig3-aux');
}

// ============================================================================

const outputs = new Map([
  ['megatron_stability_spike_detector.svg', renderSpike()],
  ['megatron_stability_grad_gates.svg', renderGradGates()],
  ['megatron_stability_auxloss_scale.svg', renderAuxScale()],
]);

export {
  CFG, BASE, POLLUTED, LATE_RATIO, BASE_FLAGS, POLLUTED_FLAGS,
  AUX_SKEWED, AUX_UNIFORM, spikeDetector, auxScale, outputs,
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
    `\n尖峰判据：max=${BASE.maxValue} 触发=${fx(BASE.trigger, 1)} 命中 ${BASE_FLAGS}；` +
      `污染后 max=${POLLUTED.maxValue} 触发=${fx(POLLUTED.trigger, 1)} 命中 ${POLLUTED_FLAGS}；` +
      `后期需 ${fx(LATE_RATIO, 0)}× 当前 loss`,
  );
  console.log(
    `aux-loss：实测 ${AUX_SKEWED.measured}，闭式逐 rank ${AUX_SKEWED.closed.join('/')}，` +
      `比值 ${AUX_SKEWED.ratio.map((r) => fx(r, 2)).join('/')}`,
  );
}
