// 图 1：CP 到底怎么切序列 —— zigzag 首尾配对 vs 连续切，以及因果代价的均衡。
// 图 2：同一个例子在四种 cp_comm_type 上的本地计算 / 上线数据 / 重构 / 增量代价。
// 图 3：非标准 attention 两条数据面共用的第一步 —— 把 zigzag 还原成原序，以及轴切换的形状。
// 图 4：chunkwise / headwise / MambaContextParallel 三条非标准 CP 数据面的 lane 对照。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要讲清楚：为什么 Megatron 把序列切成 2c 块、让 rank r 拿第 r 块与第 2c-1-r 块，
// 而不是切成 c 段连续区间。三件事必须从图上直接读出来：
//   ① 切法：全局 S 个 token 切成 2c 块，rank r 取一早一晚两块，本地恰好 S/c 个 token，
//      c 个 rank 的并集重建完整序列（用一条 16 格的 token 条 × 4 行 rank 表达，
//      拥有的格子 acc1、别人的 ghost，右侧 .dim 标 chunk 下标与本地长度）。
//   ② 连续切为什么不行：画 S×S 因果 mask 网格，左侧 ribbon 标每行 query 属于哪个 rank，
//      右侧画 4 根条形，长度是**算出来的**每 rank 因果格数 —— 阶梯状 10/26/42/58。
//   ③ zigzag 为什么行：同一张网格换一种 ribbon，条形变成等长的 34/34/34/34 = 136/4。
//      两张网格共用同一个条形标尺，等长与阶梯的对比就是本图要证的结论。
//   ④ Megatron 自己的第二种模式：cp_partition_mode='contiguous' 用的正是 ② 的区间划分，
//      但它不是作为负载均衡选项提供的，而是被 THD + packing scheduler + variant 三重门控。
// 强调色：acc1 = zigzag / 均衡（收益），acc2 = 连续切的不均衡（代价），其余 neutral / ghost。
//
// 图 2 要讲清楚：同一个例子（c=4、S=16、每 rank 4 token、TP 后 a=8 head）在四种
// cp_comm_type 上分别怎么算、怎么传、怎么拼回来、各付什么代价，且每条 lane 单独可追。
// 布局：四条横向 lane，每条 lane 五列：
//   [Megatron 侧装配] ┊ [本地计算] [上线的数据] [重构与反向] [增量代价]
// 其中 ┊ 是 TE 边界：p2p / a2a / a2a+p2p 三条 lane 的后四列全部落在 TE 内核里，
// 用 ghost 底 + 虚线边界 + "TE 内核" 芯片标出，Megatron 源码只能证明第一列；
// all_gather 一条 lane 走 Megatron 原生 AttentionFuncionWithContextParallel，
// 五列全部可证，用 acc1 标出 —— 这条对比就是本页的 thesis。
// 图例带里放跨 lane 的可比数字（每 rank 每层的 mask 格数与理论下界），全部由 CFG 推导。
//
// 图 3 要讲清楚：为什么 §2.1 教的 zigzag 到了 conv / SSM / 线性 attention 面前必须先被拆掉，
// 以及拆掉之后张量的轴是怎么换的。两件事必须从图上直接读出来：
//   ① 置换本身：三条 16 格 token 条竖排 —— 全局原序 → CP 分片后按 rank 序拼起来的本地缓冲
//      （块序 0,7,1,6,2,5,3,4）→ `_undo_attention_load_balancing` 还原成 0..15。
//      两条之间画出 2c 根按 order 走的 gather 箭头，order 由 cp_size 现算，不手写。
//      不变量写在图上：conv 与 SSM 是递推，第 t 步吃第 t-1 步的状态，所以必须是全局原序。
//   ② 轴切换：把 [S/c, B, 通道] 画成"行=token、列=通道"的二维块，A2A 前后**面积相等**——
//      这正是 all-to-all 只搬不增的可视证明；中间那一步 repeat 组状态是唯一让面积变大的地方。
// 强调色：acc1 = 还原后的原序 / A2A 后的布局（收益），acc2 = 必须还原的理由与组状态复制（代价）。
//
// 图 4 要讲清楚：同一个 c=4、S=16 的例子在三条非标准 CP 数据面上分别怎么走，沿用图 2 的
// 五列语法（装配 / 本地计算 / 上线的数据 / 重构与反向 / 增量代价）以便两张图并排读。
// 边界芯片的语义与图 2 平行：chunkwise 把 CP 语义本身交给了 FLA 的 cp_context，
// 所以后四列不可证；headwise 与 Mamba 把 CP 收敛成 head 并行后交给一个**完全不知道 CP 存在**
// 的本地内核，因此 CP 数据面五列全部可证 —— 这条对比就是本节的 thesis。
//
// 硬规矩：图上每个数字（格数、块数、head-row 数、分层组的 rank 列表、通道宽度、置换下标）
// 都由 CFG 算出，不手写；每行文字过一遍宽度守卫，放不下直接抛错，杜绝裁字。
//
// 用法：node tools/figs/svg/megatron_cp_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  cp: 4, // context_parallel_size
  seq: 16, // S：全局序列长度
  heads: 8, // a：TP 切分后本 rank 的 query head 数
  batch: 1, // B：micro-batch size
  hierarchical: Object.freeze([2, 2]), // hierarchical_context_parallel_sizes
  tp: 4, // t：本例把 TP 度定死为 4，于是「TP 后每 rank 8 个 head」对三种层型同时成立
  // GDN / KDA：全部取 TransformerConfig 的字段默认值
  linear: Object.freeze({ keyHeads: 16, valueHeads: 32, convKernel: 4 }),
  // Mamba-2：mamba_* 字段默认值，外加一个全局 head 数
  mamba: Object.freeze({ numHeads: 32, numGroups: 8, headDim: 64, stateDim: 128 }),
});

const range = (start, count) => Array.from({ length: count }, (_, i) => start + i);
const sum = (values) => values.reduce((a, b) => a + b, 0);
const product = (values) => values.reduce((a, b) => a * b, 1);

// ---- 切分与因果代价：全部由 CFG 推导 ----
const chunks = 2 * CFG.cp;
const chunkLen = CFG.seq / chunks;
const perRank = CFG.seq / CFG.cp;
const causalTotal = (CFG.seq * (CFG.seq + 1)) / 2;
const balancedCells = causalTotal / CFG.cp;

// megatron/core/utils.py::_get_batch_on_this_cp_rank_per_sequence_balancing
// index[0] = cp_rank, index[1] = 2 * cp_size - cp_rank - 1
const zigzagChunks = (rank) => [rank, chunks - 1 - rank];
const zigzagTokens = (rank) =>
  zigzagChunks(rank).flatMap((chunk) => range(chunk * chunkLen, chunkLen));
// megatron/core/context_parallel_layout/routes.py::_build_thd_layout_segments (contiguous 分支)
const contiguousTokens = (rank) => range(rank * perRank, perRank);
const causalCells = (tokens) => sum(tokens.map((q) => q + 1));

const ranks = range(0, CFG.cp);
const zigzagOwner = new Map();
const contiguousOwner = new Map();
for (const rank of ranks) {
  for (const token of zigzagTokens(rank)) zigzagOwner.set(token, rank);
  for (const token of contiguousTokens(rank)) contiguousOwner.set(token, rank);
}
const zigzagCausal = ranks.map((rank) => causalCells(zigzagTokens(rank)));
const contiguousCausal = ranks.map((rank) => causalCells(contiguousTokens(rank)));
const contiguousRatio = Math.max(...contiguousCausal) / Math.min(...contiguousCausal);

// 因果块级裁剪：以 chunk 为粒度，(q_chunk, k_chunk) 满足 k_chunk <= q_chunk 才需要算
const liveBlocks = (rank) => sum(zigzagChunks(rank).map((chunk) => chunk + 1));
const blockCells = (rank) => liveBlocks(rank) * chunkLen * chunkLen;
const ringTotalBlocks = (perRank / chunkLen) * chunks;

// ---- 分层 CP 组：复刻 parallel_state.py::create_hierarchical_groups 的 einops 重排 ----
function hierarchicalLevels(groupRanks, sizes) {
  const levels = [];
  for (let level = 0; level < sizes.length; level += 1) {
    const u = product(sizes.slice(0, level));
    const s = sizes[level];
    const l = product(sizes.slice(level + 1));
    const rows = [];
    for (let li = 0; li < l; li += 1) {
      for (let ui = 0; ui < u; ui += 1) {
        rows.push(range(0, s).map((si) => groupRanks[li * s * u + si * u + ui]));
      }
    }
    levels.push(rows);
  }
  return levels;
}

const hierLevels = hierarchicalLevels(ranks, CFG.hierarchical);
const [hierLow, hierHigh] = CFG.hierarchical;
const hierSeqRows = perRank * hierLow;
const hierHeads = CFG.heads / hierLow;
const hierBlocksPerGroup = hierLevels[0].map((group) =>
  sum([...new Set(group.flatMap(zigzagChunks))].map((chunk) => chunk + 1)),
);
const hierBlocks = hierBlocksPerGroup[0];
const hierTotalBlocks = (hierSeqRows / chunkLen) * chunks;
const hierCommunicators = CFG.cp / hierLow + CFG.cp / hierHigh;
// 用花括号而不是方括号：SVG 标注里出现 [[..]] 会被 wiki 链接检查器当成 wikilink
const fmtGroups = (groups) => groups.map((group) => `{${group.join(',')}}`).join(' ');

// ---- 四种调度在同一个例子上的本地计算量（mask 格数 / rank / 层）----
const ringCells = CFG.heads * blockCells(0);
const allGatherCells = CFG.heads * perRank * CFG.seq;
const ulyssesHeads = CFG.heads / CFG.cp;
const a2aCells = ulyssesHeads * causalTotal;
const hierCells = hierHeads * hierBlocks * chunkLen * chunkLen;
const floorCells = (CFG.heads * causalTotal) / CFG.cp;
const allGatherOverhead = allGatherCells / floorCells;

// ---- 上线的数据量（head-row = 一个 token 行 × 一个 head）----
const localHeadRows = perRank * CFG.heads;
const ringWire = (CFG.cp - 1) * 2 * localHeadRows;
const allGatherWire = 2 * CFG.heads * (CFG.seq - perRank);
const allGatherCalls = 2 * CFG.heads; // heads_k_stride = 1，K 与 V 各一次
const a2aWire = (4 * localHeadRows * (CFG.cp - 1)) / CFG.cp;
const hierA2aWire = (4 * localHeadRows * (hierLow - 1)) / hierLow;
const hierRingWire = (hierHigh - 1) * 2 * hierSeqRows * hierHeads;
const probsCells = CFG.batch * CFG.heads * perRank * CFG.seq;

// ---- 非标准 attention 数据面（一）：zigzag 还原的置换 ----
// megatron/core/ssm/mamba_context_parallel.py::_all_to_all_cp2hp 沿 dim 0 按 rank 序拼接，
// 于是缓冲位置 2r / 2r+1 上坐的正是 rank r 的那两个 zigzag 块。
const a2aBufferChunks = ranks.flatMap((rank) => zigzagChunks(rank));
// ::_undo_attention_load_balancing 的 order（num_chunks_div_2 = cp_size）
const undoOrder = [
  ...range(0, CFG.cp).map((i) => 2 * i),
  ...range(0, CFG.cp).map((i) => chunks - 2 * i - 1),
];
// ::_redo_attention_load_balancing 的 order：order[::2]=range(c)，order[1::2]=reversed(range(c,2c))
const redoOrder = (() => {
  const order = new Array(chunks);
  range(0, CFG.cp).forEach((i) => {
    order[2 * i] = i;
    order[2 * i + 1] = chunks - 1 - i;
  });
  return order;
})();
const undoneChunks = undoOrder.map((slot) => a2aBufferChunks[slot]);
const bufferTokens = a2aBufferChunks.flatMap((chunk) => range(chunk * chunkLen, chunkLen));
// docstring 里那个 cp_size=3 的例子，用同一段代码复算而不是照抄
function loadBalancedIllustration(cpSize) {
  const total = 2 * cpSize;
  const buffer = range(0, cpSize).flatMap((rank) => [rank, total - 1 - rank]);
  const order = [
    ...range(0, cpSize).map((i) => 2 * i),
    ...range(0, cpSize).map((i) => total - 2 * i - 1),
  ];
  const label = (list) => list.map((chunk) => chunk + 1).join('');
  return { before: label(buffer), after: label(order.map((slot) => buffer[slot])) };
}
const docIllustration = loadBalancedIllustration(3);
// chunkwise 在入口把布局转成 contiguous：rank r 改持 [r·S/c, (r+1)·S/c)
const chunkwiseTokens = (rank) => contiguousTokens(rank);

// ---- 非标准 attention 数据面（二）：GDN / KDA 的两种 linear_cp_mode ----
// transformer_config.py::TransformerConfig.__post_init__
//   linear_head_parallel_size = tp；headwise 时再 *= cp
const keyHeadsLocalTp = CFG.linear.keyHeads / CFG.tp;
const valueHeadsLocalTp = CFG.linear.valueHeads / CFG.tp;
const headParallelChunkwise = CFG.tp;
const headParallelHeadwise = CFG.tp * CFG.cp;
const keyHeadsChunkwise = CFG.linear.keyHeads / headParallelChunkwise;
const keyHeadsHeadwise = CFG.linear.keyHeads / headParallelHeadwise;
const valueHeadsChunkwise = CFG.linear.valueHeads / headParallelChunkwise;
const valueHeadsHeadwise = CFG.linear.valueHeads / headParallelHeadwise;
// 同一份工作量的两种摆法：chunkwise 是「少 token × 多 head」，headwise 是「全 token × 少 head」
const keyRowsChunkwise = perRank * keyHeadsChunkwise;
const keyRowsHeadwise = CFG.seq * keyHeadsHeadwise;
const valueRowsChunkwise = perRank * valueHeadsChunkwise;
const valueRowsHeadwise = CFG.seq * valueHeadsHeadwise;
const convHalo = CFG.linear.convKernel - 1;

// ---- 非标准 attention 数据面（三）：MambaContextParallel ----
// mamba_context_parallel.py::MambaContextParallel._set_cp_params
const mambaHeadsLocalTp = CFG.mamba.numHeads / CFG.tp;
const mambaGroupsLocalTp = CFG.mamba.numGroups / CFG.tp;
const dInnerLocalTp = mambaHeadsLocalTp * CFG.mamba.headDim;
const mambaHeadsLocalTpcp = mambaHeadsLocalTp / CFG.cp;
const dInnerLocalTpcp = dInnerLocalTp / CFG.cp;
const groupRepeat = mambaGroupsLocalTp < CFG.cp ? CFG.cp / mambaGroupsLocalTp : 1;
const mambaGroupsLocalTpcp = mambaGroupsLocalTp < CFG.cp ? 1 : mambaGroupsLocalTp / CFG.cp;
// pre_conv_ssm 的 torch.split 五段：z, x, B, C, dt
const mambaSectionNames = ['z', 'x', 'B', 'C', 'dt'];
const groupWidth = mambaGroupsLocalTp * CFG.mamba.stateDim;
const sectionsIn = [dInnerLocalTp, dInnerLocalTp, groupWidth, groupWidth, mambaHeadsLocalTp];
const sectionsRepeated = [
  dInnerLocalTp,
  dInnerLocalTp,
  groupWidth * groupRepeat,
  groupWidth * groupRepeat,
  mambaHeadsLocalTp,
];
const sectionsOut = sectionsRepeated.map((width) => width / CFG.cp);
const widthIn = sum(sectionsIn);
const widthRepeated = sum(sectionsRepeated);
const widthOut = sum(sectionsOut);
const repeatInflation = widthRepeated / widthIn;
// conv1d_channels() = d_inner_local_tpcp + 2 * ngroups_local_tpcp * d_state
const mambaConvChannels = dInnerLocalTpcp + 2 * mambaGroupsLocalTpcp * CFG.mamba.stateDim;
const mambaConvChannelsCp1 = dInnerLocalTp + 2 * groupWidth;
// 每次 all-to-all 都只送走本地张量的 (c-1)/c，与 Ulysses 同一个分数
const wireFraction = (CFG.cp - 1) / CFG.cp;
const mambaPreWire = wireFraction * perRank * widthRepeated;
const mambaPostWire = wireFraction * CFG.seq * dInnerLocalTpcp;
const mambaWire = mambaPreWire + mambaPostWire;
const mambaCollectives = sectionsIn.length + 1;

const invariants = {
  perRank,
  chunkLen,
  balancedCells,
  floorCells,
  keyHeadsHeadwise,
  valueHeadsHeadwise,
  mambaHeadsLocalTpcp,
  mambaConvChannels,
  widthOut,
  mambaWire,
};
for (const [name, value] of Object.entries(invariants)) {
  if (!Number.isInteger(value)) throw new Error(`${name} must be integral, got ${value}`);
}
if (sum(zigzagCausal) !== causalTotal || sum(contiguousCausal) !== causalTotal) {
  throw new Error('两种切法的因果格数之和都必须等于完整下三角');
}
if (new Set(zigzagCausal).size !== 1) throw new Error('zigzag 必须让每个 rank 的因果格数相等');
if (new Set(hierBlocksPerGroup).size !== 1) throw new Error('分层低层组之间的块数必须相等');
if (a2aCells !== floorCells) throw new Error('Ulysses 换轴后应恰好落在理论下界');
if (ringWire !== allGatherWire) throw new Error('ring 与 all-gather 的 KV 流量在本例应相等');
// 置换的可执行契约：还原必须得到全局原序，redo 必须正好是 A2A 缓冲的块序，两者严格互逆
if (undoneChunks.some((chunk, index) => chunk !== index)) {
  throw new Error('_undo_attention_load_balancing 必须把缓冲还原成全局原序');
}
if (redoOrder.some((chunk, index) => chunk !== a2aBufferChunks[index])) {
  throw new Error('_redo 的 order 必须等于 A2A 缓冲的块序');
}
if (redoOrder.map((slot) => undoOrder[slot]).some((slot, index) => slot !== index)) {
  throw new Error('undo 与 redo 的置换必须互逆');
}
if (docIllustration.before !== '162534' || docIllustration.after !== '123456') {
  throw new Error('cp_size=3 的复算结果必须与源码 docstring 的例子一致');
}
// 换轴只搬不增：A2A 前后元素数必须相等，唯一的膨胀是组状态复制
if (perRank * widthRepeated !== CFG.seq * widthOut) {
  throw new Error('all-to-all 前后的元素数必须相等');
}
if (groupRepeat !== 1 && CFG.cp % mambaGroupsLocalTp !== 0) {
  throw new Error('ngroups < cp 时 cp 必须能被 ngroups 整除');
}
// chunkwise 与 headwise 摆的是同一份工作量，只是换了轴
if (keyRowsChunkwise !== keyRowsHeadwise || valueRowsChunkwise !== valueRowsHeadwise) {
  throw new Error('chunkwise 与 headwise 的本地 head-row 数应相等');
}

const esc = (value) =>
  String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// 宽度守卫：非 ASCII 一律按全宽估算（宁可高估，也不让文字被裁掉）
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
  .cell0{fill:#DCE9FB;stroke:#fff;stroke-width:.8}
  .cell1{fill:#9CC2F3;stroke:#fff;stroke-width:.8}
  .cellx{fill:#F5F7FA;stroke:#fff;stroke-width:.8}
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

// 带标题的左对齐信息框：每行都过宽度守卫
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

function chip(x, y, w, h, label, cls = 'ghost') {
  return [
    rect(x, y, w, h, cls, 6),
    text(x + w / 2, y + h / 2 + 4, guard(label, 10.5, w - 10, 'chip'), 'sm', 'middle'),
  ].join('\n');
}

// ============================== 图 1 ==============================

function tokenRow(x, y, cellW, cellH, owner, rank) {
  const out = [];
  for (let token = 0; token < CFG.seq; token += 1) {
    const mine = owner.get(token) === rank;
    out.push(rect(x + token * cellW, y, cellW - 2, cellH, mine ? 'acc1' : 'ghost', 4));
    if (mine) out.push(text(x + token * cellW + (cellW - 2) / 2, y + cellH / 2 + 4, token, 'dim', 'middle'));
  }
  return out.join('\n');
}

function causalGrid(x, y, cell, owner) {
  const out = [];
  for (let q = 0; q < CFG.seq; q += 1) {
    for (let k = 0; k < CFG.seq; k += 1) {
      const cls = k > q ? 'cellx' : owner.get(q) === owner.get(k) ? 'cell1' : 'cell0';
      out.push(`<rect class="${cls}" x="${x + k * cell}" y="${y + q * cell}" width="${cell}" height="${cell}"/>`);
    }
  }
  return out.join('\n');
}

function ownerRibbon(x, y, w, cell, owner) {
  const out = [];
  for (let q = 0; q < CFG.seq; q += 1) {
    out.push(rect(x, y + q * cell, w, cell, 'neutral', 0));
    out.push(text(x + w / 2, y + q * cell + cell / 2 + 4, `r${owner.get(q)}`, 'sm', 'middle'));
  }
  return out.join('\n');
}

function costBars(x, y, values, barMax, scaleMax, cls) {
  const out = [];
  values.forEach((value, rank) => {
    const width = Math.round((value / scaleMax) * barMax);
    const top = y + rank * 34;
    out.push(text(x - 8, top + 17, `rank ${rank}`, 'rank', 'end'));
    out.push(rect(x, top, width, 24, cls, 5));
    out.push(text(x + width + 10, top + 17, `${value} 格`, cls === 'acc2' ? 'costtx' : 'dim'));
  });
  return out.join('\n');
}

function renderSequencePartitionFigure() {
  const W = 1420;
  const H = 964;
  const p = [];
  p.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="上下文并行如何切序列：zigzag 首尾配对与连续切在因果掩码下的负载对照">`,
  );
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(0.5, 0.5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, 'CP 怎么切序列：2c 块首尾配对，让每个 rank 的因果代价相等', 'ti'));
  p.push(
    text(
      28,
      57,
      `示例：c=${CFG.cp} · S=${CFG.seq} · 2c=${chunks} 块 · 每块 ${chunkLen} token · 每 rank 本地 S/c=${perRank} token；蓝色为本 rank 拥有，橙色为不均衡代价`,
      'su',
    ),
  );

  // ---- 面板 ①：切法 ----
  p.push(rect(20, 78, W - 40, 250, 'panel', 12));
  p.push(text(40, 106, `① 切法：rank r 取第 r 块与第 ${chunks - 1}-r 块`, 'pt'));
  p.push(
    text(
      40,
      126,
      `megatron/core/utils.py::_get_batch_on_this_cp_rank_per_sequence_balancing 用 index_select 取 [cp_rank, 2·cp_size-cp_rank-1] 两块`,
      'su',
    ),
  );

  const cellW = 31;
  const stripX = 96;
  for (let chunk = 0; chunk < chunks; chunk += 1) {
    const cx = stripX + chunk * chunkLen * cellW;
    p.push(text(cx + (chunkLen * cellW - 2) / 2, 152, `块 ${chunk}`, 'sm', 'middle'));
    p.push(line(cx - 2, 158, cx - 2, 300));
  }
  p.push(line(stripX + chunks * chunkLen * cellW - 2, 158, stripX + chunks * chunkLen * cellW - 2, 300));

  ranks.forEach((rank) => {
    const y = 162 + rank * 34;
    p.push(text(84, y + 20, `rank ${rank}`, 'rank', 'end'));
    p.push(tokenRow(stripX, y, cellW, 28, zigzagOwner, rank));
    const [early, late] = zigzagChunks(rank);
    p.push(
      text(
        stripX + CFG.seq * cellW + 14,
        y + 20,
        `块 ${early} + 块 ${late} → 本地 ${perRank} token`,
        'dim',
      ),
    );
  });

  p.push(
    infoBox(
      888,
      156,
      256,
      152,
      '为什么一早一晚配对',
      [
        `第 r 块行号小、因果列少（便宜）`,
        `第 ${chunks - 1}-r 块行号大、因果列多（贵）`,
        '一便宜配一贵，和与 r 无关',
        `并集重建完整 S=${CFG.seq}，无重叠`,
      ],
      'acc1',
      'fig1/pair',
    ),
  );
  p.push(
    infoBox(
      1156,
      156,
      244,
      152,
      '硬约束',
      [
        `seq_length 必须被 2c=${chunks} 整除`,
        'validate_args 里是 assert',
        '没有自动 padding 兜底',
        { text: `attention_mask 沿 dim 2 同切`, cls: 'sm' },
      ],
      'neutral',
      'fig1/gate',
    ),
  );

  // ---- 面板 ② / ③：因果网格 ----
  const gridCell = 18;
  const panels = [
    {
      x: 20,
      title: '② 连续切：因果代价按 rank 递增',
      sub: `rank r 取 [r·S/c, (r+1)·S/c)；最贵/最便宜 = ${contiguousCausal[3]}/${contiguousCausal[0]} = ${contiguousRatio.toFixed(1)}×`,
      owner: contiguousOwner,
      values: contiguousCausal,
      bars: 'acc2',
      note: [
        `rank ${contiguousCausal.indexOf(Math.max(...contiguousCausal))} 一个人扛下 ${Math.max(...contiguousCausal)}/${causalTotal} 的因果格`,
        `rank 0 只有 ${Math.min(...contiguousCausal)} 格，算完只能等`,
        '整层被最慢的那张卡定速',
        `CP 度越大，${contiguousRatio.toFixed(1)}× 这个比值越离谱`,
        '这就是 zigzag 要解决的问题',
      ],
      noteCls: 'acc2',
    },
    {
      x: 720,
      title: `③ zigzag 切：每 rank 恰好 ${causalTotal}/${CFG.cp} = ${balancedCells}`,
      sub: `rank r 取第 r 块与第 ${chunks - 1}-r 块；四个 rank 的条形等长，这就是本图要证的结论`,
      owner: zigzagOwner,
      values: zigzagCausal,
      bars: 'acc1',
      note: [
        `块级裁剪后每 rank ${liveBlocks(0)}/${ringTotalBlocks} 个 ${chunkLen}×${chunkLen} 块`,
        `即 ${blockCells(0)} 格，其中 ${balancedCells} 格因果有效`,
        `${CFG.cp} 个 rank 的块数同样相等`,
        `多出的 ${blockCells(0) - balancedCells} 格是对角块的粒度浪费`,
        `${chunkLen}×${chunkLen} 块变大，这个浪费才显著`,
      ],
      noteCls: 'acc1',
    },
  ];

  const gridTop = 344;
  const panelW = 680;
  for (const panel of panels) {
    p.push(rect(panel.x, gridTop, panelW, 424, 'panel', 12));
    p.push(text(panel.x + 20, gridTop + 30, panel.title, 'pt'));
    p.push(text(panel.x + 20, gridTop + 50, panel.sub, 'su'));
    const gx = panel.x + 62;
    const gy = gridTop + 84;
    p.push(text(panel.x + 20, gy - 10, 'query ↓', 'sm'));
    p.push(text(gx, gy - 10, `key → （0..${CFG.seq - 1}）`, 'sm'));
    p.push(ownerRibbon(panel.x + 32, gy, 26, gridCell, panel.owner));
    p.push(causalGrid(gx, gy, gridCell, panel.owner));
    p.push(
      `<rect x="${gx - 0.5}" y="${gy - 0.5}" width="${CFG.seq * gridCell + 1}" height="${CFG.seq * gridCell + 1}" fill="none" stroke="#AEB6C2" stroke-width="1.2"/>`,
    );
    p.push(costBars(panel.x + 430, gy + 4, panel.values, 190, Math.max(...contiguousCausal), panel.bars));
    p.push(
      infoBox(
        panel.x + 366, gy + 152, 294, 136, '读数', panel.note, panel.noteCls, `fig1/note${panel.x}`,
      ),
    );
    p.push(
      text(
        panel.x + 20,
        gy + CFG.seq * gridCell + 26,
        `深蓝＝同 rank 的 query/key，浅蓝＝跨 rank 的因果格，灰＝被掩掉；合计 ${causalTotal} 格`,
        'cap',
      ),
    );
  }

  // ---- 面板 ④：Megatron 的两种 cp_partition_mode ----
  p.push(rect(20, 782, W - 40, 122, 'panel', 12));
  p.push(text(40, 810, '④ Megatron 里这两种切法的真实身份：cp_partition_mode', 'pt'));
  p.push(
    infoBox(
      40, 822, 420, 70, `zigzag（默认，③ 的切法）`,
      [
        '标准 attention 与 MLA 只接受这一种',
        'Attention.forward 入口把输入转到 zigzag',
      ],
      'acc1',
      'fig1/mode-zz',
    ),
  );
  p.push(
    infoBox(
      480, 822, 420, 70, 'contiguous（② 的区间划分）',
      [
        '不是作为负载均衡选项提供的',
        '为 DSv4 CSA 这类变体的布局要求而存在',
      ],
      'ghost',
      'fig1/mode-ct',
    ),
  );
  p.push(
    infoBox(
      920, 822, 460, 70, '三重门控（TransformerConfig.__post_init__）',
      [
        '需要 THD + packing scheduler；variant 限 dsv4_hybrid / gdn / kda',
        '反向地 zigzag + dsv4_hybrid 同样被 ValueError 拒',
      ],
      'neutral',
      'fig1/mode-gate',
    ),
  );
  p.push(arrow(900, 857, 920, 857, 'cost'));

  p.push(
    text(
      28,
      930,
      `不变量：每 rank 持有 S/c=${perRank} 个 token；c 个 rank 的并集无重无漏地重建 S=${CFG.seq}；zigzag 下每 rank 的因果格数恒为 ${balancedCells}。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      950,
      `因果负载均衡与块级裁剪的通用推导归理论页；本图只画 Megatron 实际执行的那次 index_select 与它的两种 cp_partition_mode。`,
      'cap',
    ),
  );
  p.push('</svg>');
  return p.join('\n');
}

// ============================== 图 2 ==============================

const lanes = [
  {
    name: '① p2p',
    kind: 'ring · KV 环传',
    owner: 'TransformerEngine',
    ownerCls: 'ghost',
    note: 'CLI 默认值',
    te: true,
    assemble: [
      'extra_kwargs 写入 "p2p"',
      `cp_group = ${CFG.cp} rank 的普通 CP 组`,
      'cp_global_ranks + 进程级 cp_stream',
      'TE ≥ 1.10.0 才写入这一项',
      { text: '低于该版本静默丢弃，无告警', cls: 'costtx' },
    ],
    compute: [
      `Q 不动：本 rank ${perRank} 行 query`,
      `第 i 步吃 rank (r-i) mod ${CFG.cp} 的 KV`,
      `共 ${CFG.cp} 步，每步一个 ${perRank}×${perRank} 分块`,
      { text: `块级裁剪 ${liveBlocks(0)}/${ringTotalBlocks} → ${ringCells} 格`, cls: 'dim' },
      `跳掉的 ${ringTotalBlocks - liveBlocks(0)} 块全是纯掩码块`,
    ],
    wire: [
      '每步 P2P 送一份 K/V 块给右邻',
      `块大小 ${perRank} 行 × ${CFG.heads} head`,
      { text: `${CFG.cp - 1} 次交换 ×2 = ${ringWire} head-row`, cls: 'costtx' },
      '异步，可与分块计算重叠',
      'docstring 只承诺到"可重叠"',
    ],
    rebuild: [
      'online-softmax 合并 m / ℓ / out',
      `${CFG.cp} 步累加成 ${perRank} 行本地输出`,
      '不需要物化全序列 KV',
      '反向多一根梯度环回传 dK/dV',
      'dQ 留本地，dK/dV 沿环累加',
    ],
    cost: [
      `${CFG.cp} 个同步点`,
      `峰值 probs 一个 ${perRank}×${perRank} 分块`,
      `每 head ${perRank} 个 LSE`,
      `不物化 ${perRank}×${CFG.seq} 概率矩阵`,
      '收益全在内核内的重叠',
    ],
  },
  {
    name: '② all_gather',
    kind: 'Megatron 原生 eager',
    owner: 'Megatron 可证',
    ownerCls: 'acc1',
    note: '原生路径被锁死为它',
    te: false,
    assemble: [
      'DotProductAttention.forward 分支',
      'AttentionFuncionWithContextParallel',
      'heads_k_stride 写死为 1',
      { text: '取全局静态组，不看注入组', cls: 'costtx' },
      'cp_comm_type 收下即丢弃',
    ],
    compute: [
      'to_zz_mask_attn_bias 重排 key 轴',
      `本 rank ${perRank} 行 q × 全序列 ${CFG.seq} 行 k`,
      'eager_attn_fwd 一次整块 softmax',
      { text: `无裁剪：${CFG.heads}×${perRank}×${CFG.seq} = ${allGatherCells} 格`, cls: 'costtx' },
      `是下界 ${floorCells} 的 ${allGatherOverhead.toFixed(1)} 倍`,
    ],
    wire: [
      `逐 head 两次 all_gather_into_tensor`,
      `每次收 (c-1)/c·S = ${CFG.seq - perRank} 行`,
      { text: `${allGatherCalls} 次 AG = ${allGatherWire} head-row`, cls: 'costtx' },
      'async_op=True，head 间可预取',
      'GQA 已 repeat_interleave，省不下',
    ],
    rebuild: [
      '不需要 online softmax',
      `逐 head cat 成 [${perRank}, B, ${CFG.heads}, d_h]`,
      '反向逐 head reduce_scatter_tensor',
      { text: 'RS 同步发起，无 async_op', cls: 'costtx' },
      'RS 与前向 AG 的 rank 序互逆',
    ],
    cost: [
      { text: `probs 全量保存 ${probsCells} 格/层`, cls: 'costtx' },
      `KV 物化回 ${CFG.seq} 行，本地只有 ${perRank} 行`,
      '静默丢弃 softmax sink 与滑窗',
      '不支持 packed seq 与 Dynamic CP',
      '定位是特性缺口时的回退',
    ],
  },
  {
    name: '③ a2a',
    kind: 'Ulysses · 换 head 轴',
    owner: 'TransformerEngine',
    ownerCls: 'ghost',
    note: 'Megatron 侧无整除断言',
    te: true,
    assemble: [
      'extra_kwargs 写入 "a2a"',
      'cp_group 仍是普通 CP 组',
      '分层组不参与',
      { text: 'Megatron 侧没有 head 整除断言', cls: 'costtx' },
      'head 不够分时行为由 TE 决定',
    ],
    compute: [
      `换轴后持全序列 ${CFG.seq} 行`,
      `每 rank 只留 a/c = ${ulyssesHeads} 个 head`,
      '对本地 head 做一次完整 attention',
      { text: `${ulyssesHeads}×${causalTotal} = ${a2aCells} 格`, cls: 'dim' },
      `正好落在下界 ${floorCells}，无块级浪费`,
    ],
    wire: [
      'Q / K / V 进场各一次换轴',
      'O 出场再一次，共 4 次 A2A',
      `每次发本地的 (c-1)/c`,
      { text: `4×${a2aWire / 4} = ${a2aWire} head-row`, cls: 'costtx' },
      `是 ring 的 ${(a2aWire / ringWire).toFixed(1)} 倍（MHA、c=${CFG.cp}）`,
    ],
    rebuild: [
      `本地直接得到 [${CFG.seq}, B, ${ulyssesHeads}, d_h]`,
      `出场 A2A 换回 [${perRank}, B, ${CFG.heads}, d_h]`,
      '不需要 online softmax',
      '反向把 scatter/gather 轴对调',
      '再做一次对称的 A2A',
    ],
    cost: [
      '4 次全连接换轴，都是同步点',
      `本地要放下完整 ${CFG.seq} 行的 QKV`,
      `要求 a 能被 c=${CFG.cp} 整除`,
      'GQA 下通信跟 a 走不跟 a_kv 走',
      'A2A 适合 NVLink 域内',
    ],
  },
  {
    name: '④ a2a+p2p',
    kind: '低层 A2A × 高层 P2P',
    owner: 'TransformerEngine',
    ownerCls: 'ghost',
    note: '要 TE ≥ 1.12.0',
    te: true,
    assemble: [
      `hierarchical sizes = [${CFG.hierarchical.join(', ')}]`,
      `低层组 ${fmtGroups(hierLevels[0])}`,
      `高层组 ${fmtGroups(hierLevels[1])}`,
      'cp_group 换成分层组列表',
      { text: '读全局 getter，忽略注入组', cls: 'costtx' },
    ],
    compute: [
      `低层 A2A 后持 ${hierSeqRows} 行 × ${hierHeads} head`,
      `高层环 ${hierHigh} 步补齐另外 ${hierSeqRows} 行 KV`,
      `块级裁剪 ${hierBlocks}/${hierTotalBlocks} 块`,
      { text: `${hierHeads}×${hierBlocks}×${chunkLen * chunkLen} = ${hierCells} 格`, cls: 'dim' },
      `${hierLevels[0].length} 个低层组的块数都是 ${hierBlocks}`,
    ],
    wire: [
      `低层 4 次 A2A = ${hierA2aWire} head-row`,
      `高层 ${hierHigh - 1} 次环形交换 ×2`,
      { text: `= ${hierRingWire} head-row 走跨节点`, cls: 'costtx' },
      '把大流量压进高带宽域',
      '两级各走各的物理链路',
    ],
    rebuild: [
      '高层环内 online-softmax 合并',
      `得到 [${hierSeqRows}, B, ${hierHeads}, d_h]`,
      `低层 A2A 换回 [${perRank}, B, ${CFG.heads}, d_h]`,
      '两级各自的反向叠加',
      '合并细节 Megatron 不可证',
    ],
    cost: [
      `每 CP 组多 ${hierCommunicators} 个子 communicator`,
      '乘积必须等于 CP size',
      '要 TE ≥ 1.12.0',
      '配错分层尺寸直接 assert 失败',
      '四条里可观测性最差的一条',
    ],
  },
];

function renderCommScheduleFigure() {
  const W = 1480;
  const laneH = 178;
  const laneStride = 188;
  const boxH = 135;
  const H = 138 + lanes.length * laneStride + 66;
  const p = [];
  p.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="四种 cp_comm_type 在同一个例子上的本地计算、通信、重构与代价对照">`,
  );
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(0.5, 0.5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, '四种 cp_comm_type：同一个例子，四条各自可追的路径', 'ti'));
  p.push(
    text(
      28,
      57,
      `沿用图 1 的例子：c=${CFG.cp} · S=${CFG.seq} · 每 rank ${perRank} token（块 r 与块 ${chunks - 1}-r）· TP 后 a=${CFG.heads} 个 query head · B=${CFG.batch}`,
      'su',
    ),
  );

  p.push(rect(20, 76, W - 40, 50, 'panel', 12));
  p.push(
    text(
      40,
      98,
      `每 rank 每层要算的 mask 格数：p2p ${ringCells} · all_gather ${allGatherCells} · a2a ${a2aCells} · a2a+p2p ${hierCells}；理论下界 a·S(S+1)/2/c = ${floorCells}`,
      'tx',
    ),
  );
  p.push(
    text(
      40,
      116,
      '虚线右侧＝TransformerEngine 内核内部，Megatron 源码不可证；只有 all_gather 一条 lane 五列全部由 Megatron 源码证明。',
      'su',
    ),
  );

  const colX = [196, 470, 718, 966, 1214];
  const colW = [252, 240, 240, 240, 232];
  const headings = ['Megatron 侧装配', '本地计算', '上线的数据', '重构与反向', '增量代价'];

  lanes.forEach((lane, index) => {
    const y = 138 + index * laneStride;
    p.push(rect(20, y, W - 40, laneH, 'panel', 12));
    p.push(text(36, y + 30, lane.name, 'pt'));
    p.push(text(36, y + 50, guard(lane.kind, 10.5, 150, 'fig2/kind'), 'sm'));
    p.push(chip(34, y + 60, 148, 24, lane.owner, lane.ownerCls));
    p.push(text(36, y + 106, guard(lane.note, 10.5, 150, 'fig2/note'), 'sm'));

    const boxes = [lane.assemble, lane.compute, lane.wire, lane.rebuild, lane.cost];
    boxes.forEach((lines, col) => {
      const provable = col === 0 || !lane.te;
      const cls = lane.te ? (col === 0 ? 'neutral' : 'ghost') : col === 0 ? 'acc1' : 'neutral';
      p.push(
        infoBox(
          colX[col],
          y + 20,
          colW[col],
          boxH,
          headings[col],
          lines,
          cls,
          `fig2/${lane.name}/${col}`,
        ),
      );
      if (!provable && col === boxes.length - 1) {
        p.push(
          text(colX[col] + colW[col], y + 170, '↑ 这四列全在 TE 内核里，Megatron 源码不可证', 'sm', 'end'),
        );
      }
    });

    // 边界芯片放在信息框上沿之上（芯片带 y+4..y+26，各列首行基线在 y+41），
    // 否则宽芯片会压住右侧「本地计算」列的行首字形。
    if (lane.te) {
      p.push(line(457, y + 30, 457, y + 155, 'edge'));
      p.push(chip(437, y + 4, 40, 22, 'TE', 'acc2'));
    } else {
      p.push(line(457, y + 30, 457, y + 155, 'edge'));
      p.push(chip(423, y + 4, 68, 22, '无 TE 边界', 'acc1'));
    }
    p.push(arrow(182, y + 87, 196, y + 87, 'main'));
  });

  p.push(
    text(
      28,
      H - 42,
      `同一份 KV 流量：p2p 的 ${CFG.cp - 1} 次环形交换与 all_gather 的 ${allGatherCalls} 次 AG 在本例都是 ${ringWire} head-row，差别在调度与显存，不在字节数。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      `TE 三条 lane 的步数与块数是按 cp_comm_type docstring 描述的分工在本例上的推算，不是 Megatron 源码事实；通信量代数归理论页。`,
      'cap',
    ),
  );
  p.push('</svg>');
  return p.join('\n');
}

// ============================== 图 3 ==============================

function tokenStrip(x, y, cellW, cellH, tokens, cls, labelCls) {
  const out = [];
  tokens.forEach((token, slot) => {
    out.push(rect(x + slot * cellW, y, cellW - 2, cellH, cls, 4));
    out.push(
      text(x + slot * cellW + (cellW - 2) / 2, y + cellH / 2 + 4, token, labelCls, 'middle'),
    );
  });
  return out.join('\n');
}

// 行=token、列=通道的二维块：A2A 前后面积相等这件事必须看得见，而不是只写在文字里
function channelBlock(x, y, rows, rowH, sections, chScale, sectionCls) {
  const out = [];
  const h = rows * rowH;
  let offset = 0;
  sections.forEach((width, index) => {
    const w = width * chScale;
    out.push(
      `<rect class="${sectionCls[index]}" x="${x + offset}" y="${y}" width="${w}" height="${h}"/>`,
    );
    offset += w;
  });
  for (let row = 1; row < rows; row += 1) {
    out.push(line(x, y + row * rowH, x + offset, y + row * rowH, 'gl'));
  }
  return { svg: out.join('\n'), w: offset, h };
}

function renderZigzagUndoFigure() {
  const W = 1440;
  const H = 1000;
  const p = [];
  p.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="非标准 attention 数据面共用的第一步：把 zigzag 还原成全局原序，以及 CP 到 head 并行的轴切换">`,
  );
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(0.5, 0.5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, '递推层拿到 CP 分片的第一件事：把 zigzag 拆回原序', 'ti'));
  p.push(
    text(
      28,
      57,
      `沿用图 1 与图 2 的例子：c=${CFG.cp} · S=${CFG.seq} · 2c=${chunks} 块 · 每块 ${chunkLen} token · t=${CFG.tp}；蓝色为还原后的原序，橙色为必须还原的理由与唯一的数据膨胀`,
      'su',
    ),
  );

  // ---- 面板 ①：置换 ----
  p.push(rect(20, 78, W - 40, 384, 'panel', 12));
  p.push(text(40, 106, '① 置换：缓冲块序 → 全局原序，order 由 cp_size 现算', 'pt'));
  p.push(
    text(
      40,
      126,
      'megatron/core/ssm/mamba_context_parallel.py::_undo_attention_load_balancing，被 MambaContextParallel.pre_conv_ssm 与 GDN 的 tensor_a2a_cp2hp 共用',
      'su',
    ),
  );

  const cellW = 40;
  const stripX = 156;
  const stripW = CFG.seq * cellW;
  const chunkCenter = (slot) => stripX + slot * chunkLen * cellW + (chunkLen * cellW - 2) / 2;

  // 第一条：全局原序
  for (let chunk = 0; chunk < chunks; chunk += 1) {
    p.push(text(chunkCenter(chunk), 150, `块 ${chunk}`, 'sm', 'middle'));
  }
  p.push(text(148, 178, '全局原序', 'rank', 'end'));
  p.push(tokenStrip(stripX, 158, cellW, 30, range(0, CFG.seq), 'ghost', 'sm'));

  // 第二条：CP 分片后按 rank 序拼起来的本地缓冲
  ranks.forEach((rank) => {
    const [early, late] = zigzagChunks(rank);
    const cx = stripX + rank * perRank * cellW + (perRank * cellW - 2) / 2;
    p.push(
      text(cx, 234, guard(`rank ${rank}：块 ${early},${late}`, 10.5, perRank * cellW - 8, 'fig3/r'), 'sm', 'middle'),
    );
  });
  p.push(text(148, 262, 'A2A 后缓冲', 'rank', 'end'));
  p.push(tokenStrip(stripX, 242, cellW, 30, bufferTokens, 'neutral', 'sm'));
  for (let rank = 0; rank <= CFG.cp; rank += 1) {
    p.push(line(stripX + rank * perRank * cellW - 2, 240, stripX + rank * perRank * cellW - 2, 276));
  }

  // 置换箭头：out[j] = in[order[j]]
  undoOrder.forEach((slot, target) => {
    // 分级即公式：前 c 个目标由 order 的 2i 半段供给（main），后 c 个由 num_chunks-2i-1 半段供给（aux）
    p.push(arrow(chunkCenter(slot), 280, chunkCenter(target), 336, target < CFG.cp ? 'main' : 'aux'));
  });

  p.push(text(148, 364, '还原后', 'rank', 'end'));
  p.push(tokenStrip(stripX, 344, cellW, 30, undoneChunks.flatMap((chunk) => range(chunk * chunkLen, chunkLen)), 'acc1', 'dim'));

  p.push(
    text(
      40,
      404,
      `order = {${undoOrder.join(',')}}：前 c 项取偶数位、后 c 项取奇数位倒序，恰好把 {${a2aBufferChunks.join(',')}} 拉直成 {${undoneChunks.join(',')}}。`,
      'cap',
    ),
  );
  p.push(
    text(
      40,
      424,
      `token 层面：${bufferTokens.slice(0, 8).join(',')},… → 0,1,2,…,${CFG.seq - 1}。THD 打包时这一步换成 tex.thd_get_partitioned_indices 逐 rank 取下标，语义相同。`,
      'cap',
    ),
  );

  p.push(
    infoBox(
      826, 148, 288, 146, 'order 怎么算出来',
      [
        `num_chunks_div_2 = cp_size = ${CFG.cp}`,
        `num_chunks = ${chunks}，torch.chunk 沿 dim 0`,
        `前半 2i → {${undoOrder.slice(0, CFG.cp).join(',')}}`,
        `后半 ${chunks}-2i-1 → {${undoOrder.slice(CFG.cp).join(',')}}`,
        '图上每根箭头都由这条式子画出',
      ],
      'neutral',
      'fig3/order',
    ),
  );
  p.push(
    infoBox(
      1130, 148, 290, 146, '为什么非还原不可',
      [
        'conv 与 SSM 都是递推：第 t 步',
        '吃第 t-1 步的状态',
        `缓冲里 ${bufferTokens[1]} 之后直接跳到 ${bufferTokens[2]}`,
        'attention 不在乎顺序，递推在乎',
        'zigzag 的均衡收益在这里作废',
      ],
      'acc2',
      'fig3/why',
    ),
  );
  p.push(
    infoBox(
      826, 306, 288, 146, 'docstring 的 cp_size=3 例子',
      [
        `源码原话：converts ${docIllustration.before}`,
        `to ${docIllustration.after}`,
        '本图用同一段代码复算 cp_size=3',
        `得到 ${docIllustration.before} → ${docIllustration.after}，与之一致`,
        `本例 c=${CFG.cp} 的结果就是上面那条`,
      ],
      'ghost',
      'fig3/doc',
    ),
  );
  p.push(
    infoBox(
      1130, 306, 290, 146, '回程严格互逆',
      [
        '_redo_attention_load_balancing',
        `order = {${redoOrder.join(',')}}`,
        '正好等于 A2A 缓冲的块序',
        '两个置换复合后是恒等，已在生成器里验算',
        'post_conv_ssm 用它把输出摆回去',
      ],
      'neutral',
      'fig3/redo',
    ),
  );

  // ---- 面板 ②：轴切换 ----
  p.push(rect(20, 474, W - 40, 330, 'panel', 12));
  p.push(text(40, 502, '② 轴切换：CP 分片 → head / 通道分片，面积不变', 'pt'));
  p.push(
    text(
      40,
      522,
      `MambaContextParallel.pre_conv_ssm 的五段 torch.split（${mambaSectionNames.join(' / ')}）各走一次 _all_to_all_cp2hp；行=token，列=本 rank 的通道`,
      'su',
    ),
  );

  const rowH = 12;
  const chScale = 0.16;
  const sectionCls = ['neutral', 'neutral', 'acc2', 'acc2', 'ghost'];
  const blockA = channelBlock(60, 560, perRank, rowH, sectionsIn, chScale, sectionCls);
  p.push(blockA.svg);
  p.push(text(60, 550, `in_proj 输出（本地）`, 'sm'));
  p.push(text(60, 560 + blockA.h + 18, `[${perRank}, B, ${widthIn}]`, 'dim'));
  p.push(text(60, 560 + blockA.h + 34, `${mambaSectionNames.map((name, i) => `${name} ${sectionsIn[i]}`).join(' · ')}`, 'sm'));

  p.push(arrow(60 + blockA.w + 10, 584, 60 + blockA.w + 46, 584, 'cost'));
  p.push(text(60 + blockA.w + 28, 574, `repeat ×${groupRepeat}`, 'costtx', 'middle'));

  const bx = 60 + blockA.w + 58;
  const blockB = channelBlock(bx, 560, perRank, rowH, sectionsRepeated, chScale, sectionCls);
  p.push(blockB.svg);
  p.push(text(bx, 550, '组状态复制之后', 'sm'));
  p.push(text(bx, 560 + blockB.h + 18, `[${perRank}, B, ${widthRepeated}]`, 'costtx'));
  p.push(
    text(bx, 560 + blockB.h + 34, `B 与 C 各由 ${groupWidth} 撑到 ${groupWidth * groupRepeat}`, 'sm'),
  );

  p.push(arrow(bx + blockB.w + 10, 584, bx + blockB.w + 46, 584, 'main'));
  p.push(text(bx + blockB.w + 28, 574, `${sectionsIn.length} 次 A2A`, 'dim', 'middle'));

  const cx = bx + blockB.w + 58;
  const blockC = channelBlock(cx, 560, CFG.seq, rowH, sectionsOut, chScale, sectionCls);
  p.push(blockC.svg);
  p.push(
    `<rect x="${cx - 3}" y="557" width="${blockC.w + 6}" height="${blockC.h + 6}" fill="none" stroke="#2563EB" stroke-width="1.6" rx="3"/>`,
  );
  p.push(text(cx - 3, 550, '换轴之后', 'sm'));
  p.push(text(cx - 3, 560 + blockC.h + 18, `[${CFG.seq}, B, ${widthOut}]`, 'dim'));

  p.push(
    infoBox(
      880, 546, 260, 118, '面积不变 = 只搬不增',
      [
        `${perRank}×${widthRepeated} = ${CFG.seq}×${widthOut} = ${perRank * widthRepeated}`,
        'all-to-all 换轴不复制元素',
        `序列 ×${CFG.cp}，通道 ÷${CFG.cp}`,
        '这一步之后内核不知道 CP 存在',
      ],
      'acc1',
      'fig3/area',
    ),
  );
  p.push(
    infoBox(
      1156, 546, 264, 118, '唯一的膨胀：组状态复制',
      [
        `ngroups/t = ${mambaGroupsLocalTp} < c = ${CFG.cp}`,
        `→ 复制 ${groupRepeat} 份，ngroups per rank = ${mambaGroupsLocalTpcp}`,
        `通道 ${widthIn} → ${widthRepeated}，+${((repeatInflation - 1) * 100).toFixed(1)}%`,
        `ngroups/t ≥ c 时 repeat 为 1，不膨胀`,
      ],
      'acc2',
      'fig3/repeat',
    ),
  );
  p.push(
    infoBox(
      60, 686, 640, 100, '为什么是五次 A2A 而不是一次',
      [
        `all_to_all_sp2hp 沿最后一维等分 ${CFG.cp} 份，切点必须落在每一段自己的 head 边界上`,
        `五段宽度并不相同（${sectionsRepeated.join(' / ')}），整体一次 A2A 的切点会横切段边界`,
        'GDN 换了个做法：先按 split_sections 预置换 head 维，一次不分段 A2A 就等价于逐段 A2A',
      ],
      'neutral',
      'fig3/why5',
    ),
  );
  p.push(
    infoBox(
      880, 676, 540, 110, '参数不通信：在前向里按 CP rank 切片，梯度自己回到整份参数',
      [
        `dt_bias / A_log：[${mambaHeadsLocalTp}] → [${mambaHeadsLocalTpcp}]；conv1d 权重 [${mambaConvChannelsCp1}, 1, ${CFG.linear.convKernel}] → [${mambaConvChannels}, 1, ${CFG.linear.convKernel}]`,
        `conv1d_channels() = d_inner/tpcp + 2·ngroups_tpcp·d_state = ${dInnerLocalTpcp} + 2×${mambaGroupsLocalTpcp}×${CFG.mamba.stateDim} = ${mambaConvChannels}`,
        `组状态被复制时 B / C 的参数切片在 CP rank 间重叠：起点是 (cp_rank // ${groupRepeat}) × ${mambaGroupsLocalTpcp * CFG.mamba.stateDim}`,
      ],
      'neutral',
      'fig3/param',
    ),
  );
  p.push(
    text(
      40,
      794,
      `本例取 t=${CFG.tp}、mamba_num_heads=${CFG.mamba.numHeads}、mamba_num_groups=${CFG.mamba.numGroups}、mamba_head_dim=${CFG.mamba.headDim}、mamba_state_dim=${CFG.mamba.stateDim}（后三项是字段默认值）。`,
      'cap',
    ),
  );

  // ---- 面板 ③：守卫 ----
  p.push(rect(20, 816, W - 40, 124, 'panel', 12));
  p.push(text(40, 842, '③ 这条路自己的守卫（MambaContextParallel._set_cp_params，三条 assert）', 'pt'));
  p.push(
    infoBox(
      40, 852, 440, 76, `nheads/t 必须被 c 整除`,
      [
        `本例 ${mambaHeadsLocalTp} % ${CFG.cp} = 0，每 rank ${mambaHeadsLocalTpcp} 个 head`,
        `源码注释：cp_size 的上界就是 nheads // tp_size`,
      ],
      'neutral',
      'fig3/g1',
    ),
  );
  p.push(
    infoBox(
      496, 852, 440, 76, `ngroups/t 与 c 必须一方整除另一方`,
      [
        `本例 ${mambaGroupsLocalTp} < ${CFG.cp} 且 ${CFG.cp} % ${mambaGroupsLocalTp} = 0 → 走复制分支`,
        `另一支要求 ngroups/t 能被 c 整除，此时不复制`,
      ],
      'acc2',
      'fig3/g2',
    ),
  );
  p.push(
    infoBox(
      952, 852, 448, 76, '没有布局守卫',
      [
        '这条路无条件假定输入是 zigzag，不读 cp_partition_mode',
        'GDN headwise 相反：非 zigzag 直接 ValueError',
      ],
      'neutral',
      'fig3/g3',
    ),
  );

  p.push(
    text(
      28,
      962,
      `不变量：还原后的 token 轴必须是全局原序 0..${CFG.seq - 1}；A2A 前后元素数恒为 ${perRank * widthRepeated}；undo 与 redo 的置换复合为恒等。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      982,
      `本图只画 Megatron 自己执行的置换与换轴；conv / SSM 内核本身（mamba_ssm 与 FLA 的 Triton kernel）不在本页可证范围内。`,
      'cap',
    ),
  );
  p.push('</svg>');
  return p.join('\n');
}

// ============================== 图 4 ==============================

const planeLanes = [
  {
    name: '① chunkwise',
    kind: 'linear_cp_mode 默认值',
    owner: 'FLA cp_context',
    ownerCls: 'ghost',
    note: 'GDN / KDA',
    external: true,
    assemble: [
      'linear_cp_mode = "chunkwise"',
      { text: '没有 CLI flag，只能改 config', cls: 'costtx' },
      'cp_group_chunkwise = 整个 CP 组',
      '入口把布局转成 contiguous',
      `build_cp_context(cu_seqlens, 组, K=${CFG.linear.convKernel})`,
    ],
    compute: [
      `序列仍分片：本 rank ${perRank} 个 token`,
      `保留全部 ${keyHeadsChunkwise} 个 k 头 / ${valueHeadsChunkwise} 个 v 头`,
      { text: `本地 ${perRank}×${keyHeadsChunkwise} = ${keyRowsChunkwise} k head-row`, cls: 'dim' },
      'chunk_gated_delta_rule + conv1d',
      '状态只存在 chunk 边界（FLA 内）',
    ],
    wire: [
      '入口 1 次布局 A2A：zigzag→contiguous',
      '出口 1 次转回',
      { text: `conv 每段还要 K-1 = ${convHalo} 行左邻上下文`, cls: 'costtx' },
      '跨 rank 的状态与 halo 交换在 FLA 内',
      'Megatron 只证明传了什么进去',
    ],
    rebuild: [
      'back_to_input_converter 转回原布局',
      '两次转换都是 A2A，反向自带转置',
      '没有手写 backward',
      '链上状态的梯度归 FLA',
      'recompute_gdn 可整段重算换显存',
    ],
    cost: [
      `不要求 head 数被 c=${CFG.cp} 整除`,
      { text: 'SBHD 且 batch>1 直接 ValueError', cls: 'costtx' },
      '与 gdn_conv_pad_alignment 互斥',
      'contiguous 全局配置下这两次归零',
      '代价换来的是不物化全序列状态',
    ],
  },
  {
    name: '② headwise',
    kind: 'linear_cp_mode 另一取值',
    owner: 'Megatron 可证',
    ownerCls: 'acc1',
    note: 'Ulysses 式换轴',
    external: false,
    assemble: [
      'linear_cp_mode = "headwise"',
      'cp_group_headwise = 整个 CP 组',
      { text: '非 zigzag 入口直接 ValueError', cls: 'costtx' },
      '与 cp_partition_mode=contiguous 互斥',
      '复用 mamba_context_parallel 的原语',
    ],
    compute: [
      `换轴后持全序列 ${CFG.seq} 个 token`,
      `每 rank 只留 ${keyHeadsHeadwise} 个 k 头 / ${valueHeadsHeadwise} 个 v 头`,
      { text: `本地 ${CFG.seq}×${keyHeadsHeadwise} = ${keyRowsHeadwise} k head-row，与 ① 相同`, cls: 'dim' },
      '内核当成单卡问题跑，不知道 CP 存在',
      { text: 'docstring 自评 memory-heavy', cls: 'costtx' },
    ],
    wire: [
      '进场 1 次不分段 A2A',
      '先按 split_sections 预置换 head 维',
      `再 _undo_attention_load_balancing`,
      '出场 1 次对称 A2A',
      { text: `每次搬本地的 (c-1)/c = ${CFG.cp - 1}/${CFG.cp}`, cls: 'costtx' },
    ],
    rebuild: [
      `a2a_hp_to_cp 换回 [${perRank}, B, ·]`,
      'THD 路径用 thd_cp_a2a_inv 逆置换',
      '_AllToAll 自带反向（转置 A2A）',
      '参数按 CP rank 切片，梯度经 dp_cp',
      '没有手写 backward',
    ],
    cost: [
      `linear_num_key_heads % (t·c) == 0`,
      { text: `本例 ${CFG.linear.keyHeads} % ${headParallelHeadwise} = 0，每 rank ${keyHeadsHeadwise} 个 k 头`, cls: 'dim' },
      '构造期另断言静态 cp 能整除',
      { text: '全序列递推状态是它的主要代价', cls: 'costtx' },
      '所以默认值不是它',
    ],
  },
  {
    name: '③ Mamba CP',
    kind: 'MambaContextParallel',
    owner: 'Megatron 可证',
    ownerCls: 'acc1',
    note: '没有模式字符串',
    external: false,
    assemble: [
      'MambaMixer.__init__ 无条件构造',
      `cp_size == 1 时整条链恒等`,
      'Dynamic CP: set_context_parallel_group',
      { text: '不读 cp_partition_mode，假定 zigzag', cls: 'costtx' },
      '选择由层型决定，不由配置决定',
    ],
    compute: [
      `换轴后持全序列 ${CFG.seq} 个 token`,
      `每 rank ${mambaHeadsLocalTpcp}/${mambaHeadsLocalTp} 个 SSM head`,
      { text: `conv 通道 ${dInnerLocalTpcp}+2×${mambaGroupsLocalTpcp}×${CFG.mamba.stateDim} = ${mambaConvChannels}`, cls: 'dim' },
      'mamba_split_conv1d_scan_combined',
      '内核同样完全不知道 CP 存在',
    ],
    wire: [
      `pre_conv_ssm ${sectionsIn.length} 次 A2A（${mambaSectionNames.join(' ')}）`,
      'post_conv_ssm 1 次换回',
      { text: `合计 ${mambaWire} 通道-行 / ${mambaCollectives} 次`, cls: 'costtx' },
      `每次同样搬本地的 ${CFG.cp - 1}/${CFG.cp}`,
      `五段宽度不同，不能合成一次`,
    ],
    rebuild: [
      '_redo_attention_load_balancing 复原',
      '与 undo 严格互逆（已验算）',
      '全程由 autograd 原语组合而成',
      '参数在前向切片，梯度回整份参数',
      '没有手写 backward',
    ],
    cost: [
      { text: `组状态复制 ${widthIn} → ${widthRepeated} 通道`, cls: 'costtx' },
      '这是全路径上唯一的数据膨胀',
      `nheads/t 与 ngroups/t 各一条整除断言`,
      '有单测：test_mamba_context_parallel',
      '非 mem-eff 路径多两次 post_conv_ssm',
    ],
  },
];

function renderNonstandardPlanesFigure() {
  const W = 1480;
  const laneH = 178;
  const laneStride = 188;
  const boxH = 135;
  const H = 138 + planeLanes.length * laneStride + 66;
  const p = [];
  p.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="chunkwise、headwise 与 MambaContextParallel 三条非标准 attention CP 数据面在同一个例子上的对照">`,
  );
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(0.5, 0.5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, '非标准 attention 的三条 CP 数据面：cp_comm_type 一个字也管不到', 'ti'));
  p.push(
    text(
      28,
      57,
      `沿用图 2 的五列语法与同一个例子：c=${CFG.cp} · S=${CFG.seq} · t=${CFG.tp} · 每 rank ${perRank} token；linear_num_key_heads=${CFG.linear.keyHeads}、mamba_num_heads=${CFG.mamba.numHeads} 取字段默认值`,
      'su',
    ),
  );

  p.push(rect(20, 76, W - 40, 50, 'panel', 12));
  p.push(
    text(
      40,
      98,
      `同一份工作量的两种摆法：chunkwise 是 ${perRank} token × ${keyHeadsChunkwise} 头 = ${keyRowsChunkwise} k head-row，headwise 是 ${CFG.seq} token × ${keyHeadsHeadwise} 头 = ${keyRowsHeadwise} k head-row —— 相等，差别只在轴与递推状态的形状。`,
      'tx',
    ),
  );
  p.push(
    text(
      40,
      116,
      '虚线右侧＝外部内核内部。三条 lane 的内核数学都在外部库里，但只有 chunkwise 把 CP 语义本身也交了出去；另外两条把 CP 收敛成 head 并行后交给一个 CP 无关的本地内核。',
      'su',
    ),
  );

  const colX = [196, 470, 718, 966, 1214];
  const colW = [252, 240, 240, 240, 232];
  const headings = ['Megatron 侧装配', '本地计算', '上线的数据', '重构与反向', '增量代价'];

  planeLanes.forEach((lane, index) => {
    const y = 138 + index * laneStride;
    p.push(rect(20, y, W - 40, laneH, 'panel', 12));
    p.push(text(36, y + 30, lane.name, 'pt'));
    p.push(text(36, y + 50, guard(lane.kind, 10.5, 150, 'fig4/kind'), 'sm'));
    p.push(chip(34, y + 60, 148, 24, lane.owner, lane.ownerCls));
    p.push(text(36, y + 106, guard(lane.note, 10.5, 150, 'fig4/note'), 'sm'));

    const boxes = [lane.assemble, lane.compute, lane.wire, lane.rebuild, lane.cost];
    boxes.forEach((lines, col) => {
      const cls = lane.external ? (col === 0 ? 'neutral' : 'ghost') : col === 0 ? 'acc1' : 'neutral';
      p.push(
        infoBox(colX[col], y + 20, colW[col], boxH, headings[col], lines, cls, `fig4/${lane.name}/${col}`),
      );
      if (lane.external && col === boxes.length - 1) {
        p.push(
          text(colX[col] + colW[col], y + 170, '↑ 这四列全在 FLA 内核里，Megatron 源码不可证', 'sm', 'end'),
        );
      }
    });

    // 芯片带 y+4..y+26，各列首行基线在 y+41：不压住右侧列的行首字形
    p.push(line(457, y + 30, 457, y + 155, 'edge'));
    if (lane.external) {
      p.push(chip(431, y + 4, 52, 22, 'FLA', 'acc2'));
    } else {
      p.push(chip(415, y + 4, 84, 22, '无 CP 边界', 'acc1'));
    }
    p.push(arrow(182, y + 87, 196, y + 87, 'main'));
  });

  p.push(
    text(
      28,
      H - 42,
      `三条 lane 都要先做图 3 那次还原：headwise 与 Mamba 用 _undo_attention_load_balancing 收成全局原序，chunkwise 用 cp_partition_mode 转成 contiguous 但仍保持分片。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      `cp_comm_type 的四条 lane 见图 2；那四条与这三条的取值互不影响，混合模型必须两个轴同时配。`,
      'cap',
    ),
  );
  p.push('</svg>');
  return p.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(
  here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets',
);
const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
mkdirSync(outputDir, { recursive: true });

const outputs = new Map([
  ['megatron_cp_sequence_partition.svg', renderSequencePartitionFigure()],
  ['megatron_cp_comm_schedules.svg', renderCommScheduleFigure()],
  ['megatron_cp_zigzag_undo.svg', renderZigzagUndoFigure()],
  ['megatron_cp_nonstandard_planes.svg', renderNonstandardPlanesFigure()],
]);

for (const [name, svg] of outputs) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
console.log([...outputs.keys()].join('\n'));
