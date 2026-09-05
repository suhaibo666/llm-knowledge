// 图 1：同一块 FC1 输出经过四级激活融合时，Megatron 侧物化的中间张量、HBM 流量与
//       为反向保存的字节各是多少；下方一条附带的通信融合：vocab-parallel 交叉熵把两次
//       SUM all-reduce 并成一次。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要讲清楚：21 号页 §2 的"融合阶梯"每上一级到底省掉了什么、又多付了什么。
// 共用算例固定为 FC1 输出 y[T=4, 2H=8] bf16 与 bias[2H]，走 bias + GEGLU：
//   lane 0  eager      ：mlp.py 的 else 分支——加 bias、chunk、gelu、乘 gate 三个逐算子
//                       kernel，每个 kernel 的输入输出都落 HBM；autograd 按算子各自保存
//   lane 1  jit 融合    ：BiasGeGLUFunction 把整段交给 @jit_fuser 一个编译区域，
//                       只保存 (input, bias) 供反向；反向也是一个编译区域
//   lane 2  fp8 存储    ：SwiGLU 变体的 fp8_input_store 把保存的 input 转成 e4m3，1 B/元素
//   lane 3  加权变体    ：MoE 把 routing 权重 [T,1] 折进同一区域，省掉单独的乘法 kernel，
//                       代价是多保存 weights
// 每条 lane 五列：[输入] → [kernel / 编译区域] → [输出] → [为反向保存] → [本级付出]。
// 四条 lane 的字节、kernel 数全部由 solveExample() 从 T/H/dtype 算出，正文引用同一组数。
// 下方附带条：交叉熵融合——非融合版三次 all-reduce（max、predicted、sum_exp），
// 融合版把后两个拼成一个 [2T] 张量、两次 all-reduce；只标次数，不标物理字节。
// 强调色：acc1 = 每级新省下的部分（收益），acc2 = 每级新付出的部分（代价）。
//
// 硬规矩：图上每个数字都来自 solveExample()；每行文字过宽度守卫；渲染后做图元级文字
// 重叠断言。
//
// 用法：node tools/figs/svg/megatron_fusion_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 算例：T=4 个 token，H=4，bf16
// ============================================================================

export function solveExample() {
  const T = 4;
  const H = 4;
  const twoH = 2 * H;
  const bf16 = 2;
  const fp8 = 1;
  const fp32 = 4;
  const y = T * twoH * bf16; // FC1 输出 [T,2H]
  const bias = twoH * bf16; // bias [2H]
  const half = T * H * bf16; // [T,H] 一半
  const weights = T * fp32; // routing 权重 [T,1] fp32

  // lane 0：mlp.py else 分支的三个逐算子 kernel
  //   k1  tmp1 = y + bias              读 y,bias      写 tmp1[T,2H]
  //   k2  g    = gelu(x_glu)           读 half        写 half
  //   k3  out  = g * x_linear          读 half,half   写 half
  const eager = {
    kernels: 3,
    read: y + bias + half + half + half,
    write: y + half + half,
    materialized: y + half, // tmp1 与 gelu 输出：Megatron 侧显式出现的中间张量
    // 逐算子自动求导各自保存反向输入：gelu 保存 x_glu（tmp1 的 view，保活整块 tmp1），
    // 乘法保存两个操作数（gelu 输出 + x_linear 的 view）——依赖侧契约，见正文标注
    savedForBackward: y + half,
  };
  // lane 1：BiasGeGLUFunction.forward 只 save_for_backward(input, bias)
  const fused = { regions: 1, read: y + bias, write: half, materialized: 0, savedForBackward: y + bias };
  // lane 2：fp8_input_store —— 保存的 input 转 float8_e4m3fn
  const fp8Store = { savedForBackward: T * twoH * fp8 + bias, savedInputBytes: T * twoH * fp8 };
  // lane 3：weighted_bias_*_impl —— 权重折进区域；对照"区域外单独乘 probs"要多一个 kernel
  const weighted = {
    extraSaved: weights,
    savedForBackward: y + bias + weights,
    savedForBackwardFp8: T * twoH * fp8 + bias + weights,
    separateMulKernel: { read: half + weights, write: half },
  };
  // 通信融合：交叉熵。非融合三次 all-reduce（logits_max / predicted_logits / sum_exp_logits），
  // 融合版把后两者 cat 成 [2T] 一次 all-reduce。
  const crossEntropy = { unfusedAllReduce: 3, fusedAllReduce: 2, concatRows: 2 * T };
  return { T, H, twoH, bf16, fp8, fp32, y, bias, half, weights, eager, fused, fp8Store, weighted, crossEntropy };
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_comm_overlap_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}
function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  return value;
}

const sharedStyle = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
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
`;
const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
  </defs>`;

function rect(x, y, w, h, cls = 'neutral', radius = 8) {
  return `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"/>`;
}
function text(x, y, value, cls = 'tx', anchor = 'start') {
  return `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`;
}
function arrow(x1, y1, x2, y2, cls = 'main') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}
function box(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 20;
  out.push(text(x + 10, y + 19, guard(title, 12, inner, `${where}/title`), 'tx'));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(text(x + 10, y + 36 + index * 15, guard(value, 10.5, inner, `${where}/L${index}`), lineCls));
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
const FONT = Object.freeze({ ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 });

// 「越界不等于不重叠」：把每条 <text> 还原成包围盒两两求交，版面错误在生成时就红。
function assertNoTextOverlap(svg, name) {
  const boxes = [];
  for (const m of svg.matchAll(/<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g)) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
    }
  }
  return svg;
}
const seal = (parts, w, h, name) => assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

// ============================== 图 1 ==============================

export function renderLadderFigure() {
  const e = solveExample();
  const W = 1240;
  const p = header(
    W,
    '图 1　同一块 FC1 输出 y[T=4, 2H=8] bf16 走过四级激活融合，第四级把它折进 GEMM 链',
    '每上一级只改一件事：物化的中间张量、为反向保存的字节、或折进区域的乘法；数字全部按 T/H/dtype 算出。',
  );
  const laneX = 28;
  const laneW = W - 56;
  const cols = [
    { x: laneX + 140, w: 140, title: '输入' },
    { x: laneX + 296, w: 290, title: 'kernel / 编译区域' },
    { x: laneX + 602, w: 140, title: '输出' },
    { x: laneX + 758, w: 210, title: '为反向保存' },
    { x: laneX + 984, w: 200, title: '本级付出' },
  ];
  p.push(...cols.map((c) => text(c.x + c.w / 2, 78, c.title, 'rank', 'middle')));

  const lanes = [
    {
      id: 'eager',
      label: ['0  eager', 'mlp.py else 分支'],
      input: ['y[T,2H] ' + e.y + ' B', 'bias[2H] ' + e.bias + ' B'],
      kernel: {
        title: `${e.eager.kernels} 个逐算子 kernel`,
        lines: ['k1  tmp1 = y + bias  → HBM', 'k2  g = gelu(x_glu)   → HBM', 'k3  out = g * x_linear', `HBM 读 ${e.eager.read} B / 写 ${e.eager.write} B`],
        cls: 'neutral',
      },
      output: ['out[T,H] ' + e.half + ' B', `物化中间张量 ${e.eager.materialized} B`],
      saved: { title: `≈ ${e.eager.savedForBackward} B`, lines: ['tmp1 整块 + gelu 输出', '（PyTorch 逐算子保存，依赖侧契约）'], cls: 'acc2' },
      cost: { title: '两次 HBM 往返', lines: ['中间张量各写一次读一次', '3 次 launch'], cls: 'acc2' },
    },
    {
      id: 'jit',
      label: ['1  @jit_fuser', 'BiasGeGLUFunction'],
      input: ['y[T,2H] ' + e.y + ' B', 'bias[2H] ' + e.bias + ' B'],
      kernel: {
        title: `${e.fused.regions} 个编译区域（bias_geglu）`,
        lines: ['bias + chunk + gelu + 乘门 同一区域', '反向 bias_geglu_back 也是一个区域', `Megatron 侧 HBM 读 ${e.fused.read} B / 写 ${e.fused.write} B`],
        cls: 'acc1',
      },
      output: ['out[T,H] ' + e.half + ' B', `物化中间张量 ${e.fused.materialized} B`],
      saved: { title: `${e.fused.savedForBackward} B`, lines: ['save_for_backward(input, bias)', '反向重算 gelu，不存中间量'], cls: 'acc1' },
      cost: { title: '后端由 torch 决定', lines: ['nvFuser → Inductor 随版本换', '编译后端可能挂死 → 需总开关'], cls: 'acc2' },
    },
    {
      id: 'fp8',
      label: ['2  fp8 存储', 'fp8_input_store'],
      input: ['同上', '仅 SwiGLU 放行'],
      kernel: {
        title: '区域不变，只改保存',
        lines: ['input.to(float8_e4m3fn) 后再保存', '反向先 .to(ori_input_dtype) 还原'],
        cls: 'neutral',
      },
      output: ['out[T,H] ' + e.half + ' B', '反向输入已量化'],
      saved: { title: `${e.fp8Store.savedForBackward} B`, lines: [`input ${e.fp8Store.savedInputBytes} B（1 B/元素）+ bias ${e.bias} B`, `比第 1 级少 ${e.fused.savedForBackward - e.fp8Store.savedForBackward} B`], cls: 'acc1' },
      cost: { title: '精度换显存', lines: ['反向看到的是 e4m3 量化后的输入', '构造期只放行 SwiGLU'], cls: 'acc2' },
    },
    {
      id: 'weighted',
      label: ['3  weighted_*', 'MoE 路由权重'],
      input: ['同上', `w[T,1] fp32 ${e.weights} B`],
      kernel: {
        title: '乘权重折进同一区域',
        lines: ['out = act(y) * weights', `对照：区域外单独乘 = 多 1 个 kernel`, `（读 ${e.weighted.separateMulKernel.read} B / 写 ${e.weighted.separateMulKernel.write} B）`],
        cls: 'acc1',
      },
      output: ['out[T,H] ' + e.half + ' B', '权重梯度归约成 [T,1]'],
      saved: { title: `${e.weighted.savedForBackward} B（bf16）`, lines: [`多存 weights ${e.weighted.extraSaved} B`, `叠加 fp8 后 ${e.weighted.savedForBackwardFp8} B`], cls: 'acc2' },
      cost: { title: '只服务 MoE', lines: ['要求 per-token probs 已置换', '与 op-fuser 路径二选一'], cls: 'acc2' },
    },
    {
      id: 'opfuser',
      label: ['4  TE op-fuser', 'TEGroupedMLP 融合路径'],
      input: ['permuted_hidden', '计数留在 device'],
      kernel: {
        title: '一条 te.pytorch.ops.Sequential 链',
        lines: ['GroupedLinear → Scaled 激活 → GroupedLinear', 'Scaled 激活消费 permuted_probs', 'y 是否落 HBM 由 TE 决定（依赖边界）'],
        cls: 'ghost',
      },
      output: ['专家输出', '对齐段再 unpad'],
      saved: { title: 'TE 内部', lines: ['Megatron 只证明链与参数共享', '不叙述 TE 的保存策略'], cls: 'ghost' },
      cost: { title: '前置最多', lines: ['TE 2.14、专家 TP=1、环境变量', '无 GroupedTensor 则报错'], cls: 'acc2' },
    },
  ];

  let y = 92;
  const laneH = 108;
  for (const lane of lanes) {
    p.push(rect(laneX, y, laneW, laneH, 'panel', 10));
    p.push(text(laneX + 12, y + 24, guard(lane.label[0], 12, 130, `${lane.id}/label0`), 'pt'));
    p.push(text(laneX + 12, y + 42, guard(lane.label[1], 10.5, 130, `${lane.id}/label1`), 'sm'));
    p.push(box(cols[0].x, y + 10, cols[0].w, laneH - 20, lane.input[0], [lane.input[1]], 'ghost', `${lane.id}/input`));
    p.push(box(cols[1].x, y + 10, cols[1].w, laneH - 20, lane.kernel.title, lane.kernel.lines, lane.kernel.cls, `${lane.id}/kernel`));
    p.push(box(cols[2].x, y + 10, cols[2].w, laneH - 20, lane.output[0], [lane.output[1]], 'ghost', `${lane.id}/output`));
    p.push(box(cols[3].x, y + 10, cols[3].w, laneH - 20, lane.saved.title, lane.saved.lines, lane.saved.cls, `${lane.id}/saved`));
    p.push(box(cols[4].x, y + 10, cols[4].w, laneH - 20, lane.cost.title, lane.cost.lines, lane.cost.cls, `${lane.id}/cost`));
    const mid = y + laneH / 2;
    p.push(arrow(cols[0].x + cols[0].w, mid, cols[1].x - 2, mid, 'main'));
    p.push(arrow(cols[1].x + cols[1].w, mid, cols[2].x - 2, mid, 'main'));
    p.push(arrow(cols[2].x + cols[2].w, mid, cols[3].x - 2, mid, 'aux'));
    y += laneH + 10;
  }

  // 通信融合附带条
  y += 8;
  p.push(rect(laneX, y, laneW, 96, 'panel', 10));
  p.push(text(laneX + 12, y + 24, 'C  交叉熵通信融合', 'pt'));
  p.push(text(laneX + 12, y + 42, 'fused_cross_entropy.py', 'sm'));
  const ce = e.crossEntropy;
  p.push(box(cols[0].x, y + 10, 470, 76, `非融合：${ce.unfusedAllReduce} 次 all-reduce`, ['MAX(logits_max) → SUM(predicted_logits) → SUM(sum_exp_logits)', '每次都是一条独立的 TP 集合通信'], 'ghost', 'ce/unfused'));
  p.push(arrow(cols[0].x + 470, y + 48, cols[0].x + 470 + 22, y + 48, 'main'));
  p.push(box(cols[0].x + 494, y + 10, 340, 76, `融合：${ce.fusedAllReduce} 次 all-reduce`, ['MAX(logits_max) 不变', `cat(predicted, sum_exp) → 一个 [${ce.concatRows}] 张量一次 SUM`], 'acc1', 'ce/fused'));
  p.push(box(cols[0].x + 856, y + 10, cols[4].x + cols[4].w - (cols[0].x + 856), 76, '省的是一次集合通信的延迟', ['数学结果不变；结果 split 回两半', '@jit_fuser 只包住本地算术'], 'acc2', 'ce/cost'));
  y += 96 + 14;

  p.push(text(laneX, y + 8, `T=${e.T}, H=${e.H}, bf16=${e.bf16} B/元素, fp8=${e.fp8} B/元素；蓝框=本级新省下的部分，橙框=本级新付出的部分。`, 'cap'));
  return seal(p, W, y + 24, 'megatron_fusion_ladder.svg');
}

export function buildFigures() {
  return { 'megatron_fusion_ladder.svg': renderLadderFigure() };
}

const here = dirname(fileURLToPath(import.meta.url));
const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const outDir = process.argv[2] || join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of Object.entries(buildFigures())) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`wrote ${join(outDir, name)}`);
  }
}
