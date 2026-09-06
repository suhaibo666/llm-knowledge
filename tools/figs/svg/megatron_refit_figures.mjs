// 30_megatron_rl_posttraining_consistency_analysis.md 的三张图。
//
// 图 1：**原理图** —— LCM tiling。训练 TP 与推理 TP 不同时，refit 怎样把权重的每一段
//       精确送到目标分片。微块划分、每块的源 rank 与本地偏移全部由复刻的
//       planner.py::_emit_lcm_block_ops 算出。
// 图 2：一次 refit 从公开 API 到目标权重可见的执行链，以及三类 writeback 与两次 synchronize。
// 图 3：MXFP8 的两种 scale 布局决定两条完全不同的写回路径，其中一条还带一个硬拒绝。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「布局重映射凭什么是精确的」。取一维长度 full_len，上下三条带：
//   上 = 源 TP 分片（src_world 段），下 = 目标 TP 分片（dst_world 段），
//   中 = LCM 微块（L 个，每块 unit 个元素）。微块按它的源 rank 上色，
//   目标分片行下方逐块标出「从哪个源 rank 的哪个本地偏移取」。
// 右栏给出 Ns / Nd / L / unit / cps / cpd 六个量与那条整除守卫。
// 要证的不变量：每个微块完整落在恰好一个源分片与恰好一个目标分片里 —— 这正是取 LCM 的原因。
//
// 图 2 布局：一条竖向执行链，右侧并排三列标出每一步的「谁在算 / 通信 / 同步点」；
// 底部两个盒子分别写三类 writeback 与「为什么必须写进持久 buffer」。
//
// 图 3 布局：左右两个面板，2D scale 逐片即时量化、1D swizzled scale 必须攒齐再量化；
// 右下角标出 convert_on_send=True 遇到 1D scale 时的 NotImplementedError。
//
// 用法：node tools/figs/svg/megatron_refit_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  fullLen: 24, // 权重在 TP 维上的全局长度（算例）
  srcWorld: 4, // 训练侧 TP
  dstWorld: 3, // 推理侧 TP
  srcStride: 1,
  dstStride: 1,
  // 图 3 的算例：一个 [rows, cols] 权重，2D scale 每行一个、1D swizzled 跨整张交织
  rows: 8,
  slices: 3, // 分几片到达
});

// ============================================================================
// 复刻 planner.py::_emit_lcm_block_ops（单块、src_block_offset = dst_block_offset = 0）
// ============================================================================

const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
const lcm = (a, b) => (a / gcd(a, b)) * b;

function emitLcmBlockOps({ fullLen, srcWorld, dstWorld, srcStride, dstStride, dstLocalRank }) {
  const Ns = srcWorld * Math.max(1, srcStride);
  const Nd = dstWorld * Math.max(1, dstStride);
  const L = lcm(Ns, Nd);
  if (fullLen % L !== 0) {
    throw new Error(`块长 ${fullLen} 不能被 LCM ${L} 整除（Ns=${Ns}, Nd=${Nd}）`);
  }
  const unit = fullLen / L;
  const cps = L / Ns;
  const cpd = L / Nd;
  const segSrc = cps * unit;
  const segDst = cpd * unit;

  const ops = [];
  for (let k = 0; k < Math.max(1, dstStride); k += 1) {
    const gDstSeg = dstLocalRank + k * dstWorld;
    for (let off = 0; off < cpd; off += 1) {
      const gMicro = gDstSeg * cpd + off;
      const sIdx = Math.trunc(gMicro / cps);
      const inSeg = gMicro % cps;
      const srcRank = sIdx % srcWorld;
      const srcLocalSegIdx = Math.trunc(sIdx / srcWorld);
      ops.push({
        srcRank,
        srcStart: srcLocalSegIdx * segSrc + inSeg * unit,
        dstStart: k * segDst + off * unit,
        unit,
        gMicro,
      });
    }
  }
  ops.sort((a, b) => a.dstStart - b.dstStart);
  return { ops, Ns, Nd, L, unit, cps, cpd, segSrc, segDst };
}

const PLAN = Array.from({ length: CFG.dstWorld }, (_, r) =>
  emitLcmBlockOps({ ...CFG, dstLocalRank: r }),
);
const META = PLAN[0];

// 立论：每个微块完整落在恰好一个源分片与恰好一个目标分片内
(() => {
  const covered = new Array(CFG.fullLen).fill(-1);
  PLAN.forEach((p, dstRank) => {
    for (const op of p.ops) {
      const globalStart = dstRank * META.segDst + op.dstStart;
      for (let i = 0; i < op.unit; i += 1) {
        if (covered[globalStart + i] !== -1) {
          throw new Error(`元素 ${globalStart + i} 被覆盖了两次`);
        }
        covered[globalStart + i] = dstRank;
        // 该元素在源侧的全局位置必须落在它声明的源分片内
        const srcGlobal = op.srcRank * META.segSrc + op.srcStart + i;
        if (Math.trunc(srcGlobal / META.segSrc) !== op.srcRank) {
          throw new Error(`微块跨过了源分片边界：元素 ${srcGlobal}`);
        }
        if (srcGlobal !== globalStart + i) {
          throw new Error(`布局重映射不是恒等：src ${srcGlobal} → dst ${globalStart + i}`);
        }
      }
    }
  });
  if (covered.some((c) => c === -1)) throw new Error('有元素没有被任何目标分片覆盖');
})();

// 每个目标 rank 从几个不同的源 rank 取数
const SRC_FANIN = PLAN.map((p) => new Set(p.ops.map((o) => o.srcRank)).size);

// ============================================================================
// 图 3 的算例：两种 scale 布局各自的量化时机
// ============================================================================

function scaleWriteback(kind, slices) {
  // 2D：每片到达即量化并写回；1D：先攒进 BF16 累积 buffer，攒齐才量化一次
  const events = [];
  let written = 0;
  for (let i = 0; i < slices; i += 1) {
    written += 1;
    if (kind === '2d') {
      events.push({ slice: i, quantized: true, action: '立即量化，写回 data/scale 的对应行' });
    } else if (written < slices) {
      events.push({ slice: i, quantized: false, action: 'copy 进 BF16 累积 buffer，先不量化' });
    } else {
      events.push({ slice: i, quantized: true, action: '攒齐 → 整张 from_bf16，一次写回 data + scale' });
    }
  }
  return events;
}
const WB_2D = scaleWriteback('2d', CFG.slices);
const WB_1D = scaleWriteback('1d', CFG.slices);
const QUANT_2D = WB_2D.filter((e) => e.quantized).length;
const QUANT_1D = WB_1D.filter((e) => e.quantized).length;

if (QUANT_2D !== CFG.slices || QUANT_1D !== 1) {
  throw new Error('图 3 的立论前提被推翻：两种 scale 的量化次数应为 N 与 1');
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
  .s0{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.1}
  .s1{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.1}
  .s2{fill:#EAF7EE;stroke:#2E8B57;stroke-width:1.1}
  .s3{fill:#F3EEFB;stroke:#6D4AAF;stroke-width:1.1}
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
  .sep{fill:none;stroke:#5B6470;stroke-width:1.6}
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

// ============================================================================
// 图 1：原理图 —— LCM tiling
// ============================================================================

function renderLcm() {
  const W = 1272;
  const H = 460;
  const X0 = 132;
  const BW = 700;
  const cell = BW / CFG.fullLen;
  const RX = 872;
  const RW = 372;

  const parts = header(
    W,
    '图 1　LCM tiling：训练 TP 与推理 TP 不同时，每一段权重去哪里',
    `算例：TP 维全长 ${CFG.fullLen}，训练侧 TP=${CFG.srcWorld}（每片 ${META.segSrc}），推理侧 TP=${CFG.dstWorld}（每片 ${META.segDst}）`,
  );

  // 上：源分片
  const Y_SRC = 104;
  parts.push(text(28, Y_SRC + 20, '训练侧分片', 'rank'));
  for (let r = 0; r < CFG.srcWorld; r += 1) {
    const x = X0 + r * META.segSrc * cell;
    parts.push(rect(x, Y_SRC, META.segSrc * cell, 32, `s${r % 4}`, 4));
    parts.push(text(x + (META.segSrc * cell) / 2, Y_SRC + 21, `src ${r}`, 'sm', 'middle'));
  }

  // 中：LCM 微块
  const Y_MICRO = 172;
  parts.push(text(28, Y_MICRO + 18, `LCM 微块 ×${META.L}`, 'rank'));
  parts.push(text(28, Y_MICRO + 34, `每块 ${META.unit} 个元素`, 'sm'));
  const microOwner = new Array(META.L).fill(-1);
  PLAN.forEach((p) => p.ops.forEach((o) => { microOwner[o.gMicro] = o.srcRank; }));
  for (let m = 0; m < META.L; m += 1) {
    const x = X0 + m * META.unit * cell;
    parts.push(rect(x, Y_MICRO, META.unit * cell - 1.5, 28, `s${microOwner[m] % 4}`, 3));
  }

  // 下：目标分片 + 逐块来源
  const Y_DST = 240;
  parts.push(text(28, Y_DST + 20, '推理侧分片', 'rank'));
  for (let r = 0; r < CFG.dstWorld; r += 1) {
    const x = X0 + r * META.segDst * cell;
    parts.push(rect(x, Y_DST, META.segDst * cell, 32, 'ghost', 4));
    parts.push(text(x + (META.segDst * cell) / 2, Y_DST + 21, `dst ${r}`, 'sm', 'middle'));
    // 该分片每个微块的来源
    PLAN[r].ops.forEach((o) => {
      const mx = X0 + (r * META.segDst + o.dstStart) * cell;
      parts.push(
        text(
          mx + (o.unit * cell) / 2,
          Y_DST + 48,
          guard(`s${o.srcRank}+${o.srcStart}`, 10.5, o.unit * cell + 8, 'fig1/op'),
          'dim',
          'middle',
        ),
      );
    });
  }
  parts.push(text(28, Y_DST + 48, '取自', 'rank'));

  // 全局刻度
  parts.push(line(X0, Y_DST + 62, X0 + BW, Y_DST + 62, 'gl'));
  for (let i = 0; i <= CFG.fullLen; i += META.unit * 2) {
    parts.push(text(X0 + i * cell, Y_DST + 78, String(i), 'sm', 'middle'));
  }
  parts.push(text(X0 + BW / 2, Y_DST + 96, 'TP 维上的全局下标', 'rank', 'middle'));

  parts.push(
    infoBox(
      RX,
      Y_SRC - 8,
      RW,
      172,
      '六个量决定整张图',
      [
        `Ns = src_world × src_stride = ${META.Ns}`,
        `Nd = dst_world × dst_stride = ${META.Nd}`,
        `L = lcm(Ns, Nd) = ${META.L}；unit = 全长 / L = ${META.unit}`,
        `cps = L / Ns = ${META.cps}（一个源分片含几个微块）`,
        `cpd = L / Nd = ${META.cpd}（一个目标分片含几个微块）`,
        `全长必须被 L 整除，否则直接 RuntimeError`,
      ],
      'neutral',
      'fig1/nums',
    ),
  );
  parts.push(
    infoBox(
      RX,
      Y_SRC + 180,
      RW,
      160,
      '取 LCM 就是为了这条不变量',
      [
        '每个微块完整落在恰好一个源分片、也恰好一个目标分片里，',
        '于是每段搬运都是「整块 slice → 整块 slice」，不必跨分片拼接。',
        `本例每个推理分片从 ${SRC_FANIN.join(' / ')} 个训练分片取数。`,
        '布局重映射本身是恒等的：搬完之后每个全局下标上的值不变，',
        '所以这一步不引入数值误差 —— 误差来自后面的量化（图 3）。',
      ],
      'acc1',
      'fig1/inv',
    ),
  );

  return seal(parts, W, H, 'fig1-lcm');
}

// ============================================================================
// 图 2：一次 refit 的执行链
// ============================================================================

const CHAIN = [
  ['swap_model_weights', '解析后端名 → CopyService；未给 transform 时从缓存 plan 取', ''],
  ['reshard_model_weights', '解包两侧 core，统一持久 buffer 的 dtype', 'dtype 严格：不齐会静默损坏'],
  ['_build_or_get_plan', '命中缓存则跳过抽元数据 / gather / rank 0 规划 / scatter', '缓存键含 rank offset'],
  ['execute_reshard_plan', '按名字索引参数与持久 buffer', ''],
  ['submit_send ×N', '需要时先 transform.prepare_send', ''],
  ['submit_recv ×N', '需要时先 transform.prepare_recv，并登记 writeback 类别', 'direct / transform / copy'],
  ['service.run()', '全部 op 提交之后才发起真正的传输', '第 1 次 synchronize + barrier'],
  ['writeback', 'direct 空操作 / finalize_recv / slice copy_()', ''],
  ['quantize_()', '需要整权重量化的 MXFP8 目标最后统一做一次', '第 2 次 synchronize'],
];

function renderChain() {
  const W = 1272;
  const H = 480;
  const X0 = 28;
  const LW = 780;
  const ROW = 40;
  const Y0 = 96;

  const parts = header(
    W,
    '图 2　一次 refit：从公开 API 到目标权重可见',
    'planner、transport 与 transform 是同一条调用链的三个阶段，不是三套平行机制',
  );

  CHAIN.forEach(([name, what, note], i) => {
    const y = Y0 + i * ROW;
    const hot = name === 'service.run()' || name === 'quantize_()';
    parts.push(rect(X0, y, LW, ROW - 6, hot ? 'acc1' : 'neutral', 5));
    parts.push(text(X0 + 12, y + 22, guard(name, 10.5, 174, `fig2/n${i}`), 'dim'));
    parts.push(text(X0 + 196, y + 22, guard(what, 10.5, 380, `fig2/w${i}`), 'sm'));
    if (note) parts.push(text(X0 + 588, y + 22, guard(note, 10.5, 180, `fig2/t${i}`), 'costtx'));
    if (i < CHAIN.length - 1) parts.push(arrow(X0 + 8, y + ROW - 6, X0 + 8, y + ROW - 1, 'aux'));
  });

  parts.push(
    infoBox(
      X0 + LW + 24,
      Y0,
      W - X0 * 2 - LW - 24,
      168,
      '三类 writeback',
      [
        'direct：接收 buffer 就是目标本身，空操作。',
        'transform：交给 transform.finalize_recv（MXFP8 走这条，见图 3）。',
        'copy：把接收 buffer 按 dst_slice 拷进目标张量。',
        '三类都在 service.run() 之后统一执行，而不是收到一片处理一片 ——',
        '因为所有 send/recv 必须先全部提交，集合通信才能成对匹配。',
      ],
      'neutral',
      'fig2/wb',
    ),
  );
  parts.push(
    infoBox(
      X0 + LW + 24,
      Y0 + 184,
      W - X0 * 2 - LW - 24,
      168,
      '为什么必须写进持久 buffer',
      [
        'MXFP8 变换用 .copy_() 直接写既有的持久 MXFP8Tensor buffer，',
        '目的是让 CUDA Graph 捕获的设备指针在多次 refit 之后仍然有效。',
        '这条约束顺带堵死了「落盘 checkpoint 再由推理侧重载」这条路：',
        '重建模型会换掉设备指针，而 RL 主循环每迭代都要 refit 一次。',
        '代价是 prepare_swap_model_weights 必须趁目标参数仍是 BF16 时调一次。',
      ],
      'acc2',
      'fig2/buf',
    ),
  );

  return seal(parts, W, H, 'fig2-chain');
}

// ============================================================================
// 图 3：两种 scale 布局，两条写回路径
// ============================================================================

function scalePanel(parts, { x, w, title, sub, events, cls, where }) {
  const top = 132;
  const ROW = 40;
  parts.push(text(x, top - 44, guard(title, 12, w, `${where}/t`), 'tx'));
  parts.push(text(x, top - 26, guard(sub, 10.5, w, `${where}/s`), 'sm'));
  events.forEach((e, i) => {
    const y = top + i * ROW;
    const isQuant = e.quantized;
    parts.push(rect(x, y, w, ROW - 8, isQuant ? cls : 'ghost', 5));
    parts.push(text(x + 12, y + 21, `第 ${e.slice + 1} 片到达`, 'rank'));
    parts.push(text(x + 108, y + 21, guard(e.action, 10.5, w - 120, `${where}/a${i}`), 'sm'));
    if (i < events.length - 1) parts.push(arrow(x + 14, y + ROW - 8, x + 14, y + ROW - 3, 'aux'));
  });
  return top + events.length * ROW;
}

function renderScale() {
  const W = 1272;
  const H = 428;

  const parts = header(
    W,
    '图 3　MXFP8 的 scale 布局决定写回路径：逐片量化，还是攒齐再量化',
    `算例：一个权重分 ${CFG.slices} 片到达接收端；2D scale 每行独立，1D swizzled scale 跨整张交织编码`,
  );

  const b1 = scalePanel(parts, {
    x: 28,
    w: 596,
    title: `2D scale：量化 ${QUANT_2D} 次`,
    sub: '每行 scale 只覆盖一行数据，逐行更新互不影响',
    events: WB_2D,
    cls: 'acc1',
    where: 'fig3/L',
  });
  const b2 = scalePanel(parts, {
    x: 652,
    w: 592,
    title: `1D swizzled scale：量化 ${QUANT_1D} 次`,
    sub: 'scale 跨整张权重交织，部分更新会破坏布局',
    events: WB_1D,
    cls: 'acc2',
    where: 'fig3/R',
  });

  const y = Math.max(b1, b2) + 12;
  parts.push(
    infoBox(
      28,
      y,
      596,
      104,
      '1D 那条路多付了什么',
      [
        '一个与整张权重同大的 BF16 累积 buffer，直到攒齐才释放。',
        '实现上让 prepare_recv 直接把接收 buffer 指到累积 buffer 的对应视图，',
        '省掉一次整参数量级的 BF16 分配 —— 只在切片非连续时才退回独立 buffer。',
        '攒齐的判据是累计元素数等于目标 numel，多一片少一片都直接 AssertionError。',
      ],
      'neutral',
      'fig3/cost',
    ),
  );
  parts.push(
    infoBox(
      652,
      y,
      592,
      104,
      '一条硬拒绝：1D scale 不支持发送端量化',
      [
        'convert_on_send=True 时线上传的是已量化的 data + scale 两个张量；',
        '但 1D swizzled scale 无法由「发送端各自量化的切片」拼回来，',
        'prepare_recv 因此在 buf.scale.ndim == 1 时直接 NotImplementedError，',
        '并要求改用 convert_on_send=False：传 BF16、在接收端汇齐后量化。',
      ],
      'acc2',
      'fig3/reject',
    ),
  );

  return seal(parts, W, H, 'fig3-scale');
}

// ============================================================================

const outputs = new Map([
  ['megatron_refit_lcm_tiling.svg', renderLcm()],
  ['megatron_refit_execution.svg', renderChain()],
  ['megatron_refit_mxfp8_scale.svg', renderScale()],
]);

export { CFG, PLAN, META, SRC_FANIN, WB_2D, WB_1D, QUANT_2D, QUANT_1D, emitLcmBlockOps, outputs };

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
    `\nLCM：Ns=${META.Ns} Nd=${META.Nd} L=${META.L} unit=${META.unit} ` +
      `cps=${META.cps} cpd=${META.cpd} segSrc=${META.segSrc} segDst=${META.segDst}；` +
      `各目标分片的源分片数 ${SRC_FANIN.join('/')}`,
  );
  console.log(`scale：2D 量化 ${QUANT_2D} 次，1D 量化 ${QUANT_1D} 次`);
}
