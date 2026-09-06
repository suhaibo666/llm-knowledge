// 26_megatron_optimizer_step_internals_deepdive.md 的四张图。
//
// 图 1：一次 mixed-precision step 的五步 —— 每一步交给下一步的是哪个张量、什么精度、谁持有，
//       以及溢出闸门把整步丢掉时，这条链在哪里断开。
// 图 2：**原理图** —— 为什么 master 必须是 fp32。同一个 Adam 更新在 bf16 master 与 fp32 master
//       上各跑 18 步，用真实的 bf16 / fp32 舍入算出来：bf16 master 一步都没动过。
// 图 3：18 bytes/param 从哪来 —— 四条 wrapper 各自持有哪几段，以及各自在哪个组上规约范数。
// 图 4：ChunkedOptimizerStateOffloader 的三条流 —— 预取、更新、回写怎样错开。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 2 是本页的 principle figure，要回答的问题是「fp32 master 到底换来了什么」。
// 取最小算例：一个初值 1.0 的 bf16 权重，Adam 稳态更新幅度 |Δ| = lr = 3.0e-4。
// bf16 在 1.0 附近的 ulp 是 2⁻⁷ = 7.8e-3（只有 7 位显式尾数），比一步更新大一个数量级，所以
// `w ← bf16(w + Δ)` 每步都被舍回 1.0 —— 权重永远不动。fp32 的 ulp 是 2⁻²³，
// 更新逐步累积；直到累积量越过半个 bf16 ulp，回拷才第一次让模型权重跳一档。
// 图上跳变发生在第几步不是手写的，是用真实的 IEEE 舍入算出来的。
// 两条泳道 + 一条回拷阶梯，右栏给出 ulp 数值与「无 master 时训练停滞」的结论。
//
// 图 1 布局：五个阶段卡横排，卡下面一条「交接物」轨道写出跨边界张量的 identity + dtype + owner；
// 第 ① 步下方引出代价色分支，画到「整步丢弃」的后果盒（参数不动 / scaler 计数 / scheduler 不推进）。
//
// 图 3 布局：四行堆叠条，横轴是 bytes/param（0..18 等比），每段标 dtype 与持有者；
// 右栏写该 wrapper 的 grad-stats 规约组。分片那一行的 12/dp 由 CFG.dp 算出。
//
// 图 4 布局：三条泳道（H2D / 更新 / D2H），resident 参数先算以覆盖首块预取，
// 之后每块「等当前 H2D → 预取下一块 → _step_subset → 当前 D2H」；
// 首次懒建 state 那一块额外画一条同步竖线。泳道内不允许时间交叠（串行资源）。
//
// 用法：node tools/figs/svg/megatron_optimizer_step_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  w0: 1.0, // 一个 bf16 权重的初值
  delta: 3.0e-4, // Adam 稳态每步更新幅度 ≈ lr
  steps: 18, // 图 2 复演的步数
  dp: 8, // DistributedOptimizer 的分片域大小
  chunkCount: 4, // 图 4 的 optimizer-state chunk 数
  lazyChunk: 1, // 首次懒建 state、需要额外 drain 的那一块
});

// ============================================================================
// 数值内核：真实的 IEEE 舍入，图 2 的每一个数都从这里算出来
// ============================================================================

const f32 = (x) => Math.fround(x);

const scratch = new DataView(new ArrayBuffer(4));

// round-to-nearest-even，把 fp32 截成 bfloat16
function bf16(x) {
  scratch.setFloat32(0, x);
  const bits = scratch.getUint32(0);
  const lsb = (bits >>> 16) & 1;
  scratch.setUint32(0, ((bits + 0x7fff + lsb) >>> 0) & 0xffff0000);
  return scratch.getFloat32(0);
}

// x 处的 ulp：把尾数最低位加 1，看值变了多少
function ulpF32(x) {
  scratch.setFloat32(0, x);
  const next = scratch.getUint32(0) + 1;
  scratch.setUint32(0, next);
  return scratch.getFloat32(0) - x;
}
const ulpBf16 = (x) => ulpF32(x) * 2 ** 16; // bf16 只有 7 位显式尾数，比 fp32 少 16 位

// 两条路径各跑 CFG.steps 步：
//   A) master 也是 bf16 —— 每步加完立刻舍回 bf16
//   B) master 是 fp32 —— 累积在 fp32 上，模型权重是每步回拷的 bf16 副本
function replay({ w0, delta, steps }) {
  const bf16Master = [bf16(w0)];
  const fp32Master = [f32(w0)];
  const modelCopy = [bf16(w0)];
  for (let i = 1; i <= steps; i += 1) {
    bf16Master.push(bf16(bf16Master[i - 1] + delta));
    fp32Master.push(f32(fp32Master[i - 1] + delta));
    modelCopy.push(bf16(fp32Master[i]));
  }
  const firstMove = modelCopy.findIndex((v) => v !== modelCopy[0]);
  return {
    bf16Master,
    fp32Master,
    modelCopy,
    firstMove, // 模型权重第一次真的动的那一步；-1 表示始终没动
    bf16MasterMoved: bf16Master.some((v) => v !== bf16Master[0]),
  };
}

const RUN = replay(CFG);
const ULP_BF16 = ulpBf16(CFG.w0);
const ULP_F32 = ulpF32(CFG.w0);

if (RUN.bf16MasterMoved) {
  throw new Error('图 2 的立论前提被推翻：bf16 master 竟然动了，重新选算例');
}
if (RUN.firstMove <= 0) {
  throw new Error('图 2 的立论前提被推翻：fp32 master 也没能推动模型权重');
}

// ============================================================================
// 显存账本：每一段字节数与持有者
// ============================================================================

const BYTES = Object.freeze({ bf16: 2, fp32: 4 });

const LEDGER = [
  {
    name: 'Float16OptimizerWithFloat16Params',
    tag: 'bf16 训练基准',
    segs: [
      ['权重 bf16', BYTES.bf16, 'neutral'],
      ['梯度 fp32', BYTES.fp32, 'neutral'],
      ['master fp32', BYTES.fp32, 'acc1'],
      ['m fp32', BYTES.fp32, 'acc1'],
      ['v fp32', BYTES.fp32, 'acc1'],
    ],
    group: 'model_parallel_group',
  },
  {
    name: '同上 + --grad-reduce-in-bf16',
    tag: '梯度降到 bf16',
    segs: [
      ['权重 bf16', BYTES.bf16, 'neutral'],
      ['梯度 bf16', BYTES.bf16, 'acc2'],
      ['master fp32', BYTES.fp32, 'acc1'],
      ['m fp32', BYTES.fp32, 'acc1'],
      ['v fp32', BYTES.fp32, 'acc1'],
    ],
    group: 'model_parallel_group',
  },
  {
    name: 'DistributedOptimizer',
    tag: `12 字节按 DP=${CFG.dp} 切`,
    segs: [
      ['权重 bf16', BYTES.bf16, 'neutral'],
      ['梯度 fp32', BYTES.fp32, 'neutral'],
      ['(master+m+v)/dp', (3 * BYTES.fp32) / CFG.dp, 'acc1'],
    ],
    group: 'intra_dist_opt_group',
  },
  {
    name: 'FP32Optimizer',
    tag: '模型本身就是 fp32',
    segs: [
      ['权重 fp32', BYTES.fp32, 'neutral'],
      ['梯度 fp32', BYTES.fp32, 'neutral'],
      ['m fp32', BYTES.fp32, 'acc1'],
      ['v fp32', BYTES.fp32, 'acc1'],
    ],
    group: 'model_parallel_group',
  },
];

const ledgerTotal = (row) => row.segs.reduce((sum, [, b]) => sum + b, 0);
const LEDGER_MAX = Math.max(...LEDGER.map(ledgerTotal));

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
  .plotline{fill:none;stroke:#2563EB;stroke-width:2.2}
  .plotflat{fill:none;stroke:#C3651F;stroke-width:2.2}
  .plotstep{fill:none;stroke:#5B6470;stroke-width:2;stroke-dasharray:6 4}
  .refline{fill:none;stroke:#C8CFDA;stroke-width:1;stroke-dasharray:3 4}
  .dot1{fill:#2563EB}
  .dot2{fill:#C3651F}
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

// 「越界不等于不重叠」：viewBox 断言挡不住"标注压在邻列文字上"。把每条 <text> 还原成
// 包围盒两两求交，版面错误在生成时就红；同时按还原出的盒子校验右/下边界。
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

// 一条泳道表达一个串行资源，时间交叠必须拆成两条。
function assertLaneSerial(blocks, where) {
  const sorted = [...blocks].sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < sorted.length; i += 1) {
    if (sorted[i][0] < sorted[i - 1][1]) {
      throw new Error(
        `${where}: 同一泳道上 "${sorted[i - 1][2]}" 与 "${sorted[i][2]}" 时间交叠 ` +
          `（${sorted[i - 1][0]}..${sorted[i - 1][1]} × ${sorted[i][0]}..${sorted[i][1]}）——拆成两条泳道`,
      );
    }
  }
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

const fmt = (x, digits) => Number(x.toPrecision(digits)).toString();
const sci = (x, digits = 3) => x.toExponential(digits).replace('e-', 'e−');

// ============================================================================
// 图 1：五步的交接物、精度与所有权
// ============================================================================

const STAGES = [
  {
    n: '①',
    title: 'prepare_grads',
    lines: ['_copy_model_grads_to_main_grads', '÷ loss scale，扫 inf/nan'],
    hand: 'main_param.grad',
    dtype: 'fp32',
    owner: 'optimizer',
  },
  {
    n: '闸门',
    title: 'found_inf_flag',
    lines: ['MAX all-reduce 到全组一致', '为真则整步丢弃'],
    hand: 'found_inf (1 元素)',
    dtype: 'fp32',
    owner: 'grad-stats 组',
    gate: true,
  },
  {
    n: '②',
    title: 'clip_grad_norm',
    lines: ['SUM all-reduce 求 ‖g‖', 'MTP 组另算另裁'],
    hand: 'total_norm',
    dtype: 'fp32',
    owner: 'grad-stats 组',
  },
  {
    n: '③④',
    title: 'step_with_ready_grads',
    lines: ['base optimizer 在 master 上更新', 'count_zeros 在其前，仅日志'],
    hand: 'main_param.data',
    dtype: 'fp32',
    owner: 'optimizer',
  },
  {
    n: '⑤',
    title: 'copy back',
    lines: ['_copy_main_params_to_model_params', '或写 DistOpt 的 param buffer'],
    hand: 'model_param.data',
    dtype: 'bf16',
    owner: '模型',
  },
];

function renderLifecycle() {
  const W = 1272;
  const H = 470;
  const X0 = 28;
  const CW = 224;
  const GAP = 18;
  const cardX = (i) => X0 + i * (CW + GAP);
  const CARD_Y = 84;
  const CARD_H = 96;
  const TRACK_Y = 214;
  const TRACK_H = 62;

  const parts = header(
    W,
    '图 1　一次 mixed-precision step：每一步把什么交给下一步',
    'MixedPrecisionOptimizer.step 的固定顺序；方框下方是跨边界的交接张量（identity · dtype · 持有者）',
  );

  STAGES.forEach((s, i) => {
    const x = cardX(i);
    parts.push(rect(x, CARD_Y, CW, CARD_H, s.gate ? 'acc2' : 'neutral'));
    parts.push(text(x + 14, CARD_Y + 24, s.n, 'rank'));
    parts.push(
      text(x + 46, CARD_Y + 24, guard(s.title, 12, CW - 60, `fig1/title${i}`), 'tx'),
    );
    parts.push(line(x + 14, CARD_Y + 34, x + CW - 14, CARD_Y + 34));
    s.lines.forEach((l, k) =>
      parts.push(text(x + 14, CARD_Y + 54 + k * 16, guard(l, 10.5, CW - 28, `fig1/L${i}${k}`), 'sm')),
    );
    if (i < STAGES.length - 1) {
      parts.push(arrow(x + CW + 2, CARD_Y + CARD_H / 2, x + CW + GAP - 3, CARD_Y + CARD_H / 2));
    }

    // 交接物轨道
    const cls = s.dtype === 'bf16' ? 'acc2' : 'ghost';
    parts.push(rect(x, TRACK_Y, CW, TRACK_H, cls, 6));
    parts.push(text(x + CW / 2, TRACK_Y + 22, guard(s.hand, 10.5, CW - 20, `fig1/h${i}`), 'dim', 'middle'));
    parts.push(
      text(
        x + CW / 2,
        TRACK_Y + 42,
        guard(`${s.dtype} · ${s.owner}`, 10.5, CW - 20, `fig1/o${i}`),
        'sm',
        'middle',
      ),
    );
  });

  parts.push(text(X0, TRACK_Y - 12, '交接物', 'rank'));

  // 闸门的代价分支
  const gateX = cardX(1) + CW / 2;
  parts.push(arrow(gateX, TRACK_Y + TRACK_H + 2, gateX, 322, 'cost'));
  parts.push(
    infoBox(
      X0,
      330,
      600,
      110,
      '闸门为真：整步丢弃，链条在这里断开',
      [
        'step() 立即 return (False, None, None) —— 参数一个字节都不改',
        'grad_scaler.update(True) 已在 prepare_grads 内先行记账',
        'train_step 跨 model-parallel 组做逻辑与，再决定 skipped_iter',
        'scheduler 不推进：跳过的这一步不消耗 LR/WD schedule',
      ],
      'acc2',
      'fig1/cost',
    ),
  );
  parts.push(
    infoBox(
      X0 + 620,
      330,
      624,
      110,
      '为什么闸门在裁剪之前，而不是之后',
      [
        '裁剪要先求全局范数，非有限梯度会把 ‖g‖ 直接污染成 inf/nan，',
        '缩放因子 clip_grad/‖g‖ 于是失去意义，还白付一次 SUM all-reduce。',
        '被否掉的替代：先裁剪再查 —— 判据不是省一次通信，而是裁剪本身',
        '在非有限输入上没有定义；闸门必须先于任何依赖 ‖g‖ 的计算。',
      ],
      'neutral',
      'fig1/why',
    ),
  );

  return seal(parts, W, H, 'fig1-lifecycle');
}

// ============================================================================
// 图 2：原理图 —— fp32 master 换来了什么
// ============================================================================

function renderMasterPrecision() {
  const W = 1272;
  const H = 500;
  const PX = 208; // 绘图区左
  const PW = 580;
  const PH = 118; // 两个面板同高、同刻度，比较才成立
  const PY_A = 92;
  const PY_B = 252;
  const RX = 820;
  const RW = 424;

  const lo = CFG.w0 - ULP_BF16 * 0.22;
  const hi = CFG.w0 + ULP_BF16 * 1.35;
  const sx = (step) => PX + (step / CFG.steps) * PW;
  const sy = (v, top) => top + PH - ((v - lo) / (hi - lo)) * PH;

  const parts = header(
    W,
    '图 2　为什么 master 必须是 fp32：同一串更新，两种 master，逐步复演',
    `w₀ = ${CFG.w0}，每步 Adam 更新 Δ = ${sci(CFG.delta, 1)}（≈ lr）。两个面板同一纵向刻度`,
  );

  const panels = [
    { top: PY_A, title: 'master 也是 bf16（被否掉的替代）', cls: 'ghost' },
    { top: PY_B, title: 'fp32 master + 每步回拷（当前实现）', cls: 'ghost' },
  ];
  panels.forEach((p, i) => {
    parts.push(rect(PX, p.top, PW, PH, p.cls, 4));
    parts.push(text(PX, p.top - 10, guard(p.title, 12, PW, `fig2/pt${i}`), 'tx'));
  });

  // 参考线只画在下面板：bf16 能表示的两档与舍入分界
  const refs = [
    [CFG.w0, `bf16 档位 ${fmt(CFG.w0, 9)}`],
    [CFG.w0 + ULP_BF16 / 2, '舍入分界 +½ ulp'],
    [CFG.w0 + ULP_BF16, `bf16 下一档 ${fmt(CFG.w0 + ULP_BF16, 9)}`],
  ];
  refs.forEach(([v, label]) => {
    parts.push(line(PX, sy(v, PY_B), PX + PW, sy(v, PY_B), 'refline'));
    parts.push(text(PX - 10, sy(v, PY_B) + 4, guard(label, 10.5, 190, 'fig2/ref'), 'sm', 'end'));
  });
  parts.push(line(PX, sy(CFG.w0, PY_A), PX + PW, sy(CFG.w0, PY_A), 'refline'));
  parts.push(
    text(PX - 10, sy(CFG.w0, PY_A) + 4, guard(`bf16 档位 ${fmt(CFG.w0, 9)}`, 10.5, 190, 'fig2/refA'), 'sm', 'end'),
  );

  // 上面板：bf16 master —— 每步加完就被舍回原档
  parts.push(polyline(RUN.bf16Master.map((v, i) => [sx(i), sy(v, PY_A)]), 'plotflat'));
  parts.push(
    `<circle class="dot2" cx="${sx(CFG.steps)}" cy="${sy(RUN.bf16Master[CFG.steps], PY_A)}" r="4.5"/>`,
  );
  parts.push(
    text(
      sx(CFG.steps) - 12,
      sy(RUN.bf16Master[CFG.steps], PY_A) - 12,
      guard(`${CFG.steps} 次加法之后仍是 w₀ —— 权重一步都没动`, 10.5, 280, 'fig2/flat'),
      'costtx',
      'end',
    ),
  );

  // 下面板：fp32 master 连续爬升，bf16 副本走阶梯
  parts.push(polyline(RUN.fp32Master.map((v, i) => [sx(i), sy(v, PY_B)]), 'plotline'));
  const stepPts = [];
  RUN.modelCopy.forEach((v, i) => {
    if (i > 0 && v !== RUN.modelCopy[i - 1]) stepPts.push([sx(i), sy(RUN.modelCopy[i - 1], PY_B)]);
    stepPts.push([sx(i), sy(v, PY_B)]);
  });
  parts.push(polyline(stepPts, 'plotstep'));

  const jx = sx(RUN.firstMove);
  parts.push(line(jx, PY_B, jx, PY_B + PH, 'gl'));
  parts.push(
    `<circle class="dot1" cx="${jx}" cy="${sy(RUN.modelCopy[RUN.firstMove], PY_B)}" r="4.5"/>`,
  );
  parts.push(
    text(
      jx - 10,
      sy(RUN.modelCopy[RUN.firstMove], PY_B) - 12,
      guard(`第 ${RUN.firstMove} 步：回拷第一次改变模型权重`, 10.5, 250, 'fig2/jump'),
      'dim',
      'end',
    ),
  );

  // 共享横轴
  const AX = PY_B + PH + 16;
  parts.push(line(PX, AX, PX + PW, AX, 'gl'));
  for (let s = 0; s <= CFG.steps; s += 3) {
    parts.push(text(sx(s), AX + 18, String(s), 'sm', 'middle'));
  }
  parts.push(text(PX + PW / 2, AX + 38, 'optimizer step', 'rank', 'middle'));

  const legend = [
    ['plotflat', 'bf16 master：w ← bf16(w + Δ)'],
    ['plotline', 'fp32 master：m ← fp32(m + Δ)'],
    ['plotstep', '模型权重 = bf16(m)，每步回拷'],
  ];
  legend.forEach(([cls, label], i) => {
    const ly = AX + 62 + i * 19;
    parts.push(line(PX, ly - 4, PX + 34, ly - 4, cls));
    parts.push(text(PX + 44, ly, guard(label, 10.5, 300, 'fig2/leg'), 'sm'));
  });

  parts.push(
    infoBox(
      RX,
      PY_A,
      RW,
      156,
      '决定这张图的两个数',
      [
        `bf16 在 w₀ 附近的 ulp = 2⁻⁷ = ${sci(ULP_BF16, 3)}`,
        `fp32 在 w₀ 附近的 ulp = 2⁻²³ = ${sci(ULP_F32, 3)}`,
        `一步更新 Δ = ${sci(CFG.delta, 1)}，只有 bf16 ulp 的 ${fmt(CFG.delta / ULP_BF16, 3)} 倍`,
        '所以 bf16(w₀+Δ) 舍回 w₀：加法整个丢失，训练停滞',
        `fp32 里 Δ 是 ulp 的 ${Math.round(CFG.delta / ULP_F32).toLocaleString('en-US')} 倍，逐步累积`,
      ],
      'neutral',
      'fig2/nums',
    ),
  );
  parts.push(
    infoBox(
      RX,
      PY_A + 172,
      RW,
      186,
      'master 换来的是「累积」，不是「精度」',
      [
        '模型权重仍然是 bf16，前反向也仍然在 bf16 上做。',
        `fp32 master 唯一改变的是：更新累积在一个 ulp 小 ${Math.round(ULP_BF16 / ULP_F32).toLocaleString('en-US')} 倍`,
        '的副本上，跨过半个 bf16 ulp 后才回写一次可见的变化。',
        `代价：master + m + v 三份 fp32，共 ${3 * BYTES.fp32} bytes/param（图 3）。`,
        '收益边界：Δ 若本来就大于 ½ bf16 ulp，这条链不产生差别；',
        '训练中后期 lr 衰减、更新变小，master 才真正开始决定收敛。',
      ],
      'acc1',
      'fig2/why',
    ),
  );

  return seal(parts, W, H, 'fig2-master');
}

// ============================================================================
// 图 3：18 bytes/param 的构成与四条 wrapper
// ============================================================================

function renderLedger() {
  const W = 1272;
  const H = 442;
  const LX = 28;
  const LW = 236; // 行名列
  const BX = 280; // 条形区左
  const BW = 516;
  const RX = 900;
  const RW = 344;
  const ROW_H = 54;
  const BAR_H = 30;
  const Y0 = 108;

  const bScale = (bytes) => (bytes / LEDGER_MAX) * BW;

  const parts = header(
    W,
    '图 3　18 bytes/param 从哪来：四条 wrapper 各自持有哪几段',
    `横轴是每个参数的常驻字节数，等比刻度到 ${LEDGER_MAX} B；蓝段归优化器，灰段归模型与 DDP buffer`,
  );

  // 刻度
  for (let b = 0; b <= LEDGER_MAX; b += 2) {
    const x = BX + bScale(b);
    parts.push(line(x, Y0 - 14, x, Y0 + LEDGER.length * ROW_H - 8, 'refline'));
    parts.push(text(x, Y0 - 20, String(b), 'sm', 'middle'));
  }
  parts.push(text(BX + BW / 2, Y0 - 40, 'bytes / param', 'rank', 'middle'));

  LEDGER.forEach((row, i) => {
    const y = Y0 + i * ROW_H;
    parts.push(text(LX, y + 14, guard(row.name, 12, LW, `fig3/n${i}`), 'tx'));
    parts.push(text(LX, y + 30, guard(row.tag, 10.5, LW, `fig3/t${i}`), 'sm'));

    let cursor = 0;
    row.segs.forEach(([label, bytes, cls], k) => {
      const x = BX + bScale(cursor);
      const w = bScale(bytes);
      parts.push(rect(x, y, w, BAR_H, cls, 4));
      // 窄段（例如按 DP 切过的 12 字节）放不下标签，改标到条形下方，不缩字号
      const fits = textWidth(label, 10.5) < w - 8;
      parts.push(
        fits
          ? text(x + w / 2, y + 20, guard(label, 10.5, w - 6, `fig3/s${i}${k}`), 'sm', 'middle')
          : text(x + w / 2, y + 44, guard(label, 10.5, 180, `fig3/s${i}${k}`), 'sm', 'middle'),
      );
      cursor += bytes;
    });
    const total = ledgerTotal(row);
    parts.push(
      text(BX + bScale(total) + 10, y + 20, guard(`合计 ${fmt(total, 4)} B`, 10.5, 96, `fig3/g${i}`), 'dim'),
    );
    parts.push(
      text(RX, y + 14, guard('grad-stats 规约组', 10.5, RW, `fig3/r${i}`), 'sm'),
    );
    parts.push(text(RX, y + 30, guard(row.group, 10.5, RW, `fig3/rg${i}`), 'dim'));
  });

  const bottom = Y0 + LEDGER.length * ROW_H + 12;
  parts.push(
    infoBox(
      LX,
      bottom,
      600,
      100,
      '这张图不覆盖的三段',
      [
        '激活与重计算、DDP bucket 的通信暂存、TE 的 user buffer 都不在其中。',
        `梯度那一段归 DDP grad buffer，不归优化器：bf16 训练默认强制 fp32 累加，`,
        '--grad-reduce-in-bf16 才把它降到 2 B，代价是跨 microbatch 累加损精度。',
      ],
      'ghost',
      'fig3/note',
    ),
  );
  parts.push(
    infoBox(
      LX + 620,
      bottom,
      624,
      100,
      '为什么分片那一行的规约组不一样',
      [
        'DistributedOptimizer 把 master/m/v 按 DP range 切开，本 rank 只持有一段；',
        `范数必须覆盖整个分片域，所以工厂注入 intra_dist_opt_group 而不是模型并行组。`,
        '非分片 wrapper 的 DP 梯度同步后各 rank 已相同，只需补模型并行方向。',
      ],
      'neutral',
      'fig3/grp',
    ),
  );

  return seal(parts, W, H, 'fig3-ledger');
}

// ============================================================================
// 图 4：chunked offload 的三条流
// ============================================================================

function renderOffload() {
  const W = 1272;
  const H = 372;
  const GX = 150;
  const GW = 1060;
  const LANE_H = 28;
  const T_MAX = 100;
  const tScale = (t) => GX + (t / T_MAX) * GW;

  // 时间刻度由 chunk 数算出：每块占一个等宽槽
  const slot = T_MAX / (CFG.chunkCount + 1);
  const h2d = [];
  const comp = [];
  const d2h = [];

  h2d.push([0, slot * 0.9, 'master + 块 0', 'acc1']);
  comp.push([slot * 0.15, slot * 0.95, '常驻参数更新', 'ghost']);
  for (let c = 0; c < CFG.chunkCount; c += 1) {
    const t0 = slot * (c + 1);
    comp.push([t0, t0 + slot * 0.62, `_step_subset(${c})`, 'neutral']);
    if (c + 1 < CFG.chunkCount) {
      h2d.push([t0, t0 + slot * 0.9, `预取块 ${c + 1}`, 'acc1']);
    }
    d2h.push([t0 + slot * 0.62, t0 + slot * 0.98, `回写块 ${c}`, 'acc2']);
  }

  assertLaneSerial(h2d, 'fig4/h2d');
  assertLaneSerial(comp, 'fig4/comp');
  assertLaneSerial(d2h, 'fig4/d2h');

  const parts = header(
    W,
    '图 4　ChunkedOptimizerStateOffloader：CPU 是 canonical，state 逐块短暂回到 GPU',
    `chunk_size_bytes 切出 ${CFG.chunkCount} 块；常驻参数先算，用来覆盖第一块的 H2D`,
  );

  const lanes = [
    { tag: 'H2D 流', blocks: h2d },
    { tag: '更新（主流）', blocks: comp },
    { tag: 'D2H 流', blocks: d2h },
  ];
  const Y0 = 92;
  lanes.forEach((lane, i) => {
    const y = Y0 + i * (LANE_H + 12);
    parts.push(text(28, y + 19, guard(lane.tag, 11, 112, `fig4/lane${i}`), 'rank'));
    lane.blocks.forEach(([t0, t1, label, cls], k) => {
      const x = tScale(t0);
      const w = tScale(t1) - tScale(t0);
      parts.push(rect(x, y, w, LANE_H, cls, 5));
      parts.push(
        text(x + w / 2, y + 18, guard(label, 10.5, w - 8, `fig4/b${i}${k}`), 'sm', 'middle'),
      );
    });
  });

  // 懒建 state 的一次性 drain
  const drainT = slot * (CFG.lazyChunk + 1) + slot * 0.98;
  const dx = tScale(drainT);
  parts.push(line(dx, Y0 - 8, dx, Y0 + 3 * (LANE_H + 12), 'gl'));
  parts.push(
    text(dx + 8, Y0 - 14, guard(`首次懒建 state：drain D2H 一次`, 10.5, 220, 'fig4/drain'), 'costtx'),
  );

  const bottom = Y0 + 3 * (LANE_H + 12) + 16;
  parts.push(
    infoBox(
      28,
      bottom,
      560,
      120,
      '与 HybridDeviceOptimizer 的分工',
      [
        'Hybrid：一部分参数的更新计算交给 CPU optimizer，用 CPU 算力换显存。',
        'Chunked：更新仍在 GPU，只让 state/master 在非使用期以 CPU 副本为准。',
        '判据不是"哪个更快"，而是要不要把 optimizer kernel 也搬走；',
        '两者由 optimizer_config 断言互斥，不能同开。',
      ],
      'neutral',
      'fig4/split',
    ),
  );
  parts.push(
    infoBox(
      608,
      bottom,
      636,
      120,
      '峰值界怎么守住',
      [
        `任一时刻 GPU 上最多驻留：常驻参数 + 当前块 + 正在预取的下一块。`,
        'chunk 以参数为不可切分原子装箱，超大单参数允许独占超限块。',
        '首次懒建 moment 的那一块不走复用池，必须同步一次再进下一块，',
        '否则两块的临时 state 会同时在场，峰值界失效。',
      ],
      'acc1',
      'fig4/peak',
    ),
  );

  return seal(parts, W, H, 'fig4-offload');
}

// ============================================================================

const outputs = new Map([
  ['megatron_optstep_lifecycle.svg', renderLifecycle()],
  ['megatron_optstep_master_precision.svg', renderMasterPrecision()],
  ['megatron_optstep_ledger.svg', renderLedger()],
  ['megatron_optstep_offload.svg', renderOffload()],
]);

export { CFG, RUN, ULP_BF16, ULP_F32, LEDGER, ledgerTotal, outputs, bf16, replay };

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
    `\n算例：w0=${CFG.w0} Δ=${sci(CFG.delta, 1)}；bf16 ulp=${sci(ULP_BF16, 3)}；` +
      `模型权重首次改变于第 ${RUN.firstMove} 步；bf16 master ${CFG.steps} 步未动=${!RUN.bf16MasterMoved}`,
  );
}
