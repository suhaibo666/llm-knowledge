// Megatron Distributed Optimizer / DP state sharding 的四张机制图。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md）----
// 图 1 用 N=16、DP=4、p=[2,7) 展示 equal shard 与 parameter boundary 不对齐，
// 并从同一交集推导 gbuf_world / bucket / local / param 四套坐标。
//
// 图 2 沿一个 bucket 的真实完成边界复演 backward hook → ready gate → RS →
// finish_grad_sync → local update → model shard copy → parameter AG → next-forward
// pre-hook；下半部分对照同步、unaligned、aligned 与 optimizer-step overlap 的
// 真实 dispatch owner 和 consumer wait。zero_grad 只清梯度，不拥有 AG dispatch。
//
// 图 3 沿用同一组 q/p/r 参数，对照 native DistributedOptimizer、Torch FSDP2、
// Megatron-FSDP，以及 LayerWise 的 compact-decoupled / padded-layout 两条 live path。
// 后两条分别复演 AR→whole-param owner→variable-size AG 与
// RS→whole-param owner→buffer AG；Megatron-FSDP 内部状态机仍由 page 36 拥有。
//
// 图 4 从同一个 q|p|r|pad buffer 与 rank1=[4,8) 检查点出发，独立复演普通
// all-reduce、标准 reduce-scatter、custom FP32 accumulation RS 和 multi-instance
// HSDP。每条 lane 都给 local compute、owner/data 移动、通信、重建、同步点和成本。
// 校准补注：N=16 是隔离等分/交集原语的输入，省略 native builder 的64/128对齐；
// padding仅通信、不进入optimizer。四图均在可见图注标出这一层级；LayerWise 13→256
// 保持真实padding计算。FSDP行明确是依赖/邻页契约对照，custom单bucket限定overlap。
//
// 用法：node tools/figs/svg/megatron_distributed_optimizer_figures.mjs [output-directory]

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pagePath = join(
  scriptDir, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '16_megatron_distributed_optimizer_analysis.md',
);
const markdownSource = readFileSync(pagePath, 'utf8');
const contractMatch = markdownSource.match(
  /<!-- distopt-figure-contract:start -->([\s\S]*?)<!-- distopt-figure-contract:end -->/,
);
if (contractMatch === null) throw new Error('Markdown is missing distopt-figure-contract block');
const figureContract = contractMatch[1].trim();
const figureContractSha256 = createHash('sha256').update(figureContract).digest('hex');

const CFG = Object.freeze({ elements: 16, dp: 4, wireBytesPerElement: 2 });
const HSDP = Object.freeze({ totalDp: 8, instances: 2 });
const PARAMS = Object.freeze([
  { name: 'q', start: 0, end: 2 },
  { name: 'p', start: 2, end: 7 },
  { name: 'r', start: 7, end: 13 },
  { name: 'pad', start: 13, end: 16 },
]);
const LAYERWISE_PARAMS = Object.freeze(PARAMS
  .filter((param) => param.name !== 'pad')
  .map((param) => Object.freeze({ name: param.name, elements: param.end - param.start })));
const layerWiseLogicalElements = LAYERWISE_PARAMS.reduce((sum, param) => sum + param.elements, 0);
const pingPongRanks = [...Array(CFG.dp).keys(), ...Array(CFG.dp).keys()].map(
  (rank, index) => (index < CFG.dp ? rank : CFG.dp - 1 - (index - CFG.dp)),
);
const layerWiseDecoupledOwners = Array.from({ length: CFG.dp }, () => []);
[...LAYERWISE_PARAMS]
  .sort((a, b) => a.elements - b.elements || a.name.localeCompare(b.name))
  .forEach((param, index) => layerWiseDecoupledOwners[pingPongRanks[index]].push(param));

const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
const lcm = (a, b) => a * b / gcd(a, b);
const padTo = (value, divisor) => Math.ceil(value / divisor) * divisor;
const padParamStart = (value) => padTo(value, 64);
const bucketEndDivisor = lcm(CFG.dp, 128);
const layerWiseShardDivisor = lcm(64, bucketEndDivisor / CFG.dp);
const layerWiseLayoutOwners = Array.from({ length: CFG.dp }, () => []);
const layerWiseLayoutLoads = Array(CFG.dp).fill(0);
[...LAYERWISE_PARAMS]
  .sort((a, b) => b.elements - a.elements || a.name.localeCompare(b.name))
  .forEach((param) => {
    const owner = layerWiseLayoutLoads.indexOf(Math.min(...layerWiseLayoutLoads));
    layerWiseLayoutOwners[owner].push(param);
    layerWiseLayoutLoads[owner] = padParamStart(layerWiseLayoutLoads[owner]) + param.elements;
  });
const layerWisePaddedShardElements = padTo(
  Math.max(...layerWiseLayoutLoads), layerWiseShardDivisor,
);
const layerWiseLayoutElements = CFG.dp * layerWisePaddedShardElements;
const layerWisePaddingElements = layerWiseLayoutElements - layerWiseLogicalElements;
const layerWiseRawBytes = layerWiseLogicalElements * CFG.wireBytesPerElement;
const layerWiseDecoupledArBytes = 2 * (CFG.dp - 1) * layerWiseRawBytes / CFG.dp;
const layerWiseLayoutBytes = layerWiseLayoutElements * CFG.wireBytesPerElement;
const layerWiseLayoutRsBytes = (CFG.dp - 1) * layerWiseLayoutBytes / CFG.dp;
const layerWiseLayoutAgBytes = layerWiseLayoutRsBytes;
if (layerWiseLogicalElements !== 13) throw new Error('LayerWise logical example drifted');
if (layerWiseShardDivisor !== 64 || layerWiseLayoutElements !== 256) {
  throw new Error('LayerWise padded-layout example drifted');
}
const shardSize = CFG.elements / CFG.dp;
if (!Number.isInteger(shardSize)) throw new Error('buffer must divide evenly across DP ranks');
const fullPayloadBytes = CFG.elements * CFG.wireBytesPerElement;
const ringAllReduceBytes = 2 * (CFG.dp - 1) * fullPayloadBytes / CFG.dp;
const ringReduceScatterBytes = (CFG.dp - 1) * fullPayloadBytes / CFG.dp;
const fp32LocalSumBytes = shardSize * 4;
const intraSize = HSDP.totalDp / HSDP.instances;
if (!Number.isInteger(intraSize) || intraSize !== CFG.dp) {
  throw new Error('HSDP example must preserve the same four-way flat-buffer shard');
}
const hsdpIntraRsBytes = (intraSize - 1) * fullPayloadBytes / intraSize;
const hsdpInterArBytes = (
  2 * (HSDP.instances - 1) * (fullPayloadBytes / intraSize) / HSDP.instances
);
const hsdpParamAgBytes = hsdpIntraRsBytes;
const hsdpTotalBytes = hsdpIntraRsBytes + hsdpInterArBytes + hsdpParamAgBytes;
const hsdpForceIntraArBytes = ringAllReduceBytes;
const hsdpForceTotalBytes = hsdpForceIntraArBytes + hsdpInterArBytes + hsdpParamAgBytes;

const ranges = Array.from({ length: CFG.dp }, (_, rank) => ({
  rank,
  start: rank * shardSize,
  end: (rank + 1) * shardSize,
}));
const pParam = PARAMS.find((param) => param.name === 'p');
const pIntersections = ranges.map((range) => {
  const worldStart = Math.max(range.start, pParam.start);
  const worldEnd = Math.min(range.end, pParam.end);
  if (worldEnd <= worldStart) return null;
  return {
    rank: range.rank,
    worldStart,
    worldEnd,
    localStart: worldStart - range.start,
    localEnd: worldEnd - range.start,
    paramStart: worldStart - pParam.start,
    paramEnd: worldEnd - pParam.start,
  };
});
if (pIntersections.filter(Boolean).length !== 2) throw new Error('p must cross exactly two shards');
if (pIntersections[0].paramStart !== 0 || pIntersections[0].paramEnd !== 2) throw new Error('rank0 p range drifted');
if (pIntersections[1].paramStart !== 2 || pIntersections[1].paramEnd !== 5) throw new Error('rank1 p range drifted');

const esc = (value) => String(value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

const style = `
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
  .ti{font-size:18px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .pt{font-size:14px;font-weight:700;fill:#2A313B}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#68717D}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .rank{font-size:11px;font-weight:700;fill:#5B6470}
  .cap{font-size:11px;fill:#747C88}
  .cellq{fill:#EEF1F5;stroke:#fff}
  .cellp{fill:#9CC2F3;stroke:#fff}
  .cellr{fill:#DCE9FB;stroke:#fff}
  .cellpad{fill:#F1E2D2;stroke:#fff}
`;
const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
    <marker id="arrowCost" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#C3651F"/></marker>
  </defs>`;

const rect = (x, y, w, h, cls = 'neutral', radius = 8) => (
  `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"/>`
);
const text = (x, y, value, cls = 'tx', anchor = 'start') => (
  `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`
);
const arrow = (x1, y1, x2, y2, cls = 'main') => (
  `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`
);
function box(lines, x, y, w, h, cls = 'neutral') {
  const out = [rect(x, y, w, h, cls)];
  const gap = 17;
  const start = y + h / 2 - ((lines.length - 1) * gap) / 2 + 4;
  lines.forEach((line, index) => out.push(text(
    x + w / 2, start + index * gap, line, index === 0 ? 'tx' : 'sm', 'middle',
  )));
  return out.join('\n');
}
function svgStart(width, height, aria) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" role="img" aria-label="${esc(aria)}">\n<style>${style}</style>${defs}`;
}

function paramAt(index) {
  return PARAMS.find((param) => index >= param.start && index < param.end);
}

function renderFlatRanges() {
  const W = 1460, H = 800, p = [];
  p.push(svgStart(W, H, '长度 16 的连续 buffer 按四个 DP rank 等分，参数 p 从世界位置 2 到 7 跨越 rank0 和 rank1 shard，并映射到四套坐标'));
  p.push(rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 38, '连续 buffer 的等分边界可以切进一个 parameter', 'ti'));
  p.push(text(28, 60, `N=${CFG.elements} · d=${CFG.dp} · shard=${shardSize} · p world [${pParam.start}, ${pParam.end})；蓝色 p 跨过 world index ${shardSize}`, 'su'));

  const x0 = 72, y0 = 132, cellW = 72, cellH = 62;
  p.push(text(x0, 104, 'param_data / grad_data 的统一 world index', 'pt'));
  for (let index = 0; index < CFG.elements; index += 1) {
    const param = paramAt(index);
    p.push(rect(x0 + index * cellW, y0, cellW - 2, cellH, `cell${param.name}`, 3));
    p.push(text(x0 + index * cellW + (cellW - 2) / 2, y0 + 25, `${param.name}[${index - param.start}]`, 'tx', 'middle'));
    p.push(text(x0 + index * cellW + (cellW - 2) / 2, y0 + 47, `w${index}`, 'sm', 'middle'));
  }
  ranges.forEach((range) => {
    const x = x0 + range.start * cellW;
    const w = shardSize * cellW - 2;
    p.push(`<path d="M ${x} ${y0 + cellH + 13} v 10 h ${w} v -10" fill="none" stroke="#2563EB" stroke-width="1.5"/>`);
    p.push(text(x + w / 2, y0 + cellH + 42, `rank${range.rank} world [${range.start}, ${range.end})`, 'dim', 'middle'));
  });

  p.push(rect(52, 270, 1356, 124, 'panel', 12));
  p.push(text(72, 302, 'parameter boundary ≠ shard boundary', 'pt'));
  p.push(text(72, 326, 'p 的 5 个元素不会整体归一张卡：rank0 更新 p[0,2)，rank1 更新 p[2,5)。rank2/3 对 p 没有交集。', 'tx'));
  p.push(box(['rank0: p[0, 2)', 'world [2,4)'], 72, 344, 214, 40, 'acc1'));
  p.push(box(['rank1: p[2, 5)', 'world [4,7)'], 304, 344, 214, 40, 'acc1'));
  p.push(box(['rank2 / rank3', 'p intersection = ∅'], 536, 344, 214, 40, 'ghost'));
  p.push(box(['为什么这样切', 'RS receive 与 state/update', '都能直接读连续 local view'], 1128, 292, 252, 80, 'neutral'));

  p.push(text(52, 438, '同一交集的四套 range（bucket offset = 0）', 'pt'));
  const cards = [
    {
      x: 52,
      title: 'rank0 拿到 p 的前 2 个元素',
      lines: ['gbuf_world       [2,4)', 'gbuf_world_in_bucket [2,4)', 'gbuf_local       [2,4)', 'param            [0,2)'],
    },
    {
      x: 730,
      title: 'rank1 拿到 p 的后 3 个元素',
      lines: ['gbuf_world       [4,7)', 'gbuf_world_in_bucket [4,7)', 'gbuf_local       [0,3)', 'param            [2,5)'],
    },
  ];
  cards.forEach((card) => {
    p.push(rect(card.x, 466, 628, 214, 'panel', 12));
    p.push(text(card.x + 20, 496, card.title, 'pt'));
    card.lines.forEach((line, index) => p.push(text(card.x + 28, 530 + index * 30, line, index === 3 ? 'dim' : 'tx')));
  });
  p.push(box(['存储成本', 'N 必须可被 d 整除', '不够则 padding'], 52, 706, 284, 62, 'acc2'));
  p.push(box(['映射成本', '同一 shard 要维护 world / bucket / local / param 坐标'], 356, 706, 482, 62, 'neutral'));
  p.push(box(['正确性收益', 'collective、main/model copy、optimizer update 共用同一连续 range'], 858, 706, 550, 62, 'acc1'));
  p.push(text(28, 788, '等分原语示例：省略 native builder 的64/128对齐；pad只通信、不更新。真实小参数布局另算，见LayerWise 13→256。', 'cap'));
  p.push('</svg>');
  return `${p.join('\n')}\n`;
}

function flowRow(p, y, nodes) {
  nodes.forEach((node, index) => {
    p.push(box(node.lines, node.x, y, node.w, node.h || 84, node.cls));
    if (index < nodes.length - 1) {
      const next = nodes[index + 1];
      p.push(arrow(node.x + node.w, y + (node.h || 84) / 2, next.x, y + (next.h || 84) / 2, node.arrow || 'main'));
    }
  });
}

function renderLifecycle() {
  const W = 1500, H = 1254, p = [];
  p.push(svgStart(W, H, 'Distributed Optimizer 从反向梯度 ready，经 reduce-scatter 和本地更新，再 all-gather 到下一次 forward 可见参数的完整生命周期'));
  p.push(rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 38, '一个 bucket 的训练闭环：RS 只交付 update shard，AG 才交付 next-forward 参数', 'ti'));
  p.push(text(28, 60, 'N=16、d=4 等分原语示例，省略native 64/128对齐；蓝=可消费，橙=等待；dispatch 不等于完成', 'su'));

  p.push(rect(22, 88, 1456, 326, 'panel', 12));
  p.push(text(42, 120, '① forward / backward：从 full parameter 到 local reduced gradient shard', 'pt'));
  flowRow(p, 150, [
    { x: 42, w: 184, lines: ['full P → forward → loss', 'backward autograd hook', 'grad → main_grad view'], cls: 'neutral' },
    { x: 258, w: 194, lines: ['ready-count gate', '最终 microbatch', 'bucket 全参数 ready'], cls: 'neutral' },
    { x: 484, w: 200, lines: ['start_grad_sync', 'reduce-scatter (RS)', '可异步 dispatch'], cls: 'acc2' },
    { x: 716, w: 194, lines: ['反向继续算', 'RS handle 可能 in flight', '不能交给 optimizer'], cls: 'ghost' },
    { x: 942, w: 206, lines: ['finish_grad_sync', 'finalizer wait handle', '补齐 copy-back'], cls: 'acc2' },
    { x: 1180, w: 250, lines: ['可消费 local shard', 'rank r 只更新其中真实参数', '每 rank 接收4槽，pad不更新'], cls: 'acc1' },
  ]);
  p.push(text(42, 270, '普通 DDP 对照：相同 gate 后执行 all-reduce，完成时每 rank 都有完整已规约 gradient；DistOpt 的 RS 改变 update input 的 owner。', 'sm'));
  p.push(text(42, 294, '若 num_distributed_optimizer_instances > 1：先在 intra-instance RS，再跨 instance 对等价 local range AR；两段都完成才可更新。', 'sm'));
  p.push(box(['完成信号', 'start_grad_sync = 已发射', 'finish_grad_sync = local shard 可消费'], 42, 326, 420, 66, 'acc1'));
  p.push(box(['失败边界', '少一个 ready parameter / 错误 no_sync 边界', '会在完成检查或 handle 状态上暴露'], 484, 326, 452, 66, 'acc2'));
  p.push(box(['重叠收益', '只可能隐藏 RS 等待', '不减少 collective 的逻辑 payload'], 958, 326, 472, 66, 'neutral'));

  p.push(rect(22, 438, 1456, 166, 'panel', 12));
  p.push(text(42, 470, '② optimizer handoff：只更新 range owner，再把新 model shard 写回 param buffer', 'pt'));
  flowRow(p, 498, [
    { x: 54, w: 222, lines: ['training schedule 返回', 'finalize_model_grads 已完成', 'local grad shard ready'], cls: 'acc1' },
    { x: 326, w: 218, lines: ['optimizer.step', 'local update', 'main/state 只处理 owned range'], cls: 'neutral' },
    { x: 594, w: 224, lines: ['main → model shard', 'dtype/copy 细节由 page 26', '这里只跟踪可见性'], cls: 'neutral' },
    { x: 868, w: 234, lines: ['新参数仍只有 local shard', '不能宣称所有 module 可见', '下一步必须同步'], cls: 'acc2' },
    { x: 1152, w: 260, lines: ['选择 parameter AG 时机', '同步：step 后', '重叠：下一轮按 bucket'], cls: 'acc1' },
  ]);

  p.push(rect(22, 628, 1456, 558, 'panel', 12));
  p.push(text(42, 660, '③ parameter AG：四条 dispatch / wait 路径必须分清 owner', 'pt'));
  p.push(text(42, 680, '每行都从更新后的 local model shard 出发；dispatch 只产生 in-flight 工作，consumer-visible 仍以 finish/wait 为界。', 'su'));

  flowRow(p, 696, [
    { x: 42, w: 300, h: 76, lines: ['SYNC（无 overlap）', 'local model shard 已更新', 'parameter AG 尚未发生'], cls: 'acc1' },
    { x: 404, w: 300, h: 76, lines: ['step_with_ready_grads', 'overlap_param_gather=False', '调用 start_param_sync'], cls: 'neutral' },
    { x: 766, w: 300, h: 76, lines: ['同步参数 AG', 'all-gather (AG)', '调用返回时完成'], cls: 'acc2' },
    { x: 1128, w: 300, h: 76, lines: ['full param_data', 'next forward 可直接读取', 'consumer-visible'], cls: 'acc1' },
  ]);

  flowRow(p, 808, [
    { x: 42, w: 300, h: 76, lines: ['UNALIGNED OVERLAP', 'step 返回；没有发 AG', 'align_param_gather=False'], cls: 'acc1' },
    { x: 404, w: 300, h: 76, lines: ['next-forward pre-hook', '_finish_param_sync_for_bucket_group', 'consumer 请求当前 bucket'], cls: 'neutral' },
    { x: 766, w: 300, h: 76, lines: ['finish_param_sync', 'lazy dispatch current AG', '随后 wait 当前 handle'], cls: 'acc2' },
    { x: 1128, w: 300, h: 76, lines: ['module 读取 full params', '可再 dispatch next bucket', 'compute(B0) ∥ AG(B1)'], cls: 'acc1' },
  ]);

  flowRow(p, 920, [
    { x: 42, w: 300, h: 76, lines: ['ALIGNED OVERLAP', 'pipeline schedule 拥有节拍', 'align_param_gather=True'], cls: 'acc1' },
    { x: 404, w: 300, h: 76, lines: ['param_sync_func', '调用 chunk.start_param_sync', '在目标 model chunk 之前'], cls: 'neutral' },
    { x: 766, w: 300, h: 76, lines: ['early async AG dispatch', 'handle in flight', '不隐式 dispatch next bucket'], cls: 'acc2' },
    { x: 1128, w: 300, h: 76, lines: ['forward pre-hook', 'finish_param_sync wait', '随后 module 读取 full P'], cls: 'acc1' },
  ]);

  flowRow(p, 1032, [
    { x: 42, w: 300, h: 76, lines: ['OPTIMIZER-STEP OVERLAP', 'ChainedOptimizer._step', 'child 0 step 成功'], cls: 'acc1' },
    { x: 404, w: 300, h: 76, lines: ['start_param_sync', 'force_dispatch=True', '强制发起首个 AG'], cls: 'acc2' },
    { x: 766, w: 300, h: 76, lines: ['later child step compute', '可覆盖 in-flight AG', 'dispatch ≠ visibility'], cls: 'neutral' },
    { x: 1128, w: 300, h: 76, lines: ['forward pre-hook waits', 'finish_param_sync', '随后 full P 可见'], cls: 'acc1' },
  ]);
  p.push(text(42, 1136, 'zero_grad_buffer / optimizer.zero_grad 只清 gradient，不是 parameter AG dispatch owner。', 'costtx'));
  p.push(text(42, 1159, '收益取决于 pre-hook 前的覆盖窗口；任何异步分支都没有减少 collective payload。', 'sm'));

  p.push(text(28, 1216, '全链完成信号：local reduced grad ready → local update 完成 → 对应 parameter AG 完成 → module view 才看见新权重。', 'cap'));
  p.push(text(28, 1238, 'reset_param_sync_dispatch_state 会拒绝旧 AG 仍 in flight 的状态；不存在把已发 collective 回滚的事务语义。', 'cap'));
  p.push('</svg>');
  return `${p.join('\n')}\n`;
}

function pathPanel(p, y, spec) {
  p.push(rect(22, y, 1456, 220, 'panel', 12));
  p.push(text(42, y + 30, spec.title, 'pt'));
  p.push(text(42, y + 50, spec.subtitle, 'su'));
  const xs = [42, 326, 614, 902];
  spec.steps.forEach((step, index) => {
    p.push(box(step.lines, xs[index], y + 76, 246, 92, step.cls));
    if (index < spec.steps.length - 1) p.push(arrow(xs[index] + 246, y + 122, xs[index + 1], y + 122, index === 1 ? 'cost' : 'main'));
  });
  p.push(box(spec.cost, 1190, y + 74, 264, 96, 'acc2'));
  p.push(text(42, y + 196, spec.footer, 'sm'));
}

function strategyLane(p, y, spec) {
  const xs = [42, 318, 594, 870, 1146];
  const width = 246;
  p.push(`<g data-strategy="${esc(spec.id)}">`);
  spec.cells.forEach((cell, index) => {
    p.push(box(cell.lines, xs[index], y, width, spec.height, cell.cls));
    if (index < spec.cells.length - 1) {
      p.push(arrow(
        xs[index] + width,
        y + spec.height / 2,
        xs[index + 1],
        y + spec.height / 2,
        cell.arrow || 'main',
      ));
    }
  });
  p.push('</g>');
}

function renderLivePaths() {
  const W = 1500, H = 1660, p = [];
  p.push(svgStart(W, H, '同一组参数在 native DistributedOptimizer、Torch FSDP2、Megatron-FSDP 与 LayerWise 两种 layout 中的常驻所有权、梯度归属、更新交接与参数重建对照'));
  p.push(`<metadata id="markdown-contract" data-contract-sha256="${figureContractSha256}">16_megatron_distributed_optimizer_analysis.md::distopt-figure-contract</metadata>`);
  p.push(rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 38, 'live paths：range-sharded families + LayerWise 的两种 whole-parameter layout', 'ti'));
  p.push(text(28, 60, '沿用 d=4 与同一 q(2), p(5), r(6)；各 path 会重新决定 buffer padding 与 owner，不能只看 use_distributed_optimizer 布尔值', 'su'));

  pathPanel(p, 84, {
    title: '① native DistributedOptimizer：persistent full parameter，state/update 按 range 分片',
    subtitle: '连续 DDP param/grad buffer 是 storage owner；bucket group 是 RS/AG communication owner。',
    steps: [
      { lines: ['常驻 model parameter', '[p0 | p1 | p2 | p3]', '每 DP rank forward 可见完整'], cls: 'acc1' },
      { lines: ['backward RS', 'local grad shard g_r', 'optimizer state/update owner = r'], cls: 'neutral' },
      { lines: ['local update + copy', '只改本 rank range', '其他 range 此刻仍旧'], cls: 'neutral' },
      { lines: ['bucket parameter AG', '同步或 next-forward overlap', '恢复完整新 param view'], cls: 'acc2' },
    ],
    cost: ['HBM / 通信', 'model param 保持完整副本', 'main/state 约按 d 分摊', '每轮 RS + AG 同阶 payload'],
    footer: 'N=16 为等分原语示例，省略 native builder 的64/128对齐；padding只通信、不更新；真实layout须另算。',
  });

  pathPanel(p, 318, {
    title: '② Torch FSDP2：PyTorch fully_shard 管理 module 的 transient full unit',
    subtitle: '依赖API契约对照，未核验PyTorch内核：Megatron只证明module选择、DeviceMesh与reshard_after_forward handoff。',
    steps: [
      { lines: ['persistent parameter shards', '[p0 | p1 | p2 | p3]', 'rank r 常驻 pr'], cls: 'acc1' },
      { lines: ['forward AG / unshard', 'F on transient full unit', '按策略保留或 reshard'], cls: 'acc2' },
      { lines: ['pre-backward AG / unshard', '若 forward 后已 reshard', 'B → gradient reduce-scatter'], cls: 'acc2' },
      { lines: ['sharded grad → local optimizer', 'update rank-r parameter/state', 'updated parameter shard'], cls: 'acc1' },
    ],
    cost: ['HBM / 通信', 'persistent param 约按 d 分片', 'F/B 各自可能需要 AG', 'gradient RS + transient unit'],
    footer: '完成：sharded gradient 被 optimizer 消费并产出 updated shard；下次 forward 再从该 owner 集合 unshard。实现 owner 是 PyTorch FSDP2。',
  });

  p.push(rect(22, 552, 1456, 438, 'panel', 12));
  p.push(text(42, 582, '③ Megatron-FSDP：三种 strategy 必须分别走到 next-forward visibility', 'pt'));
  p.push(text(42, 604, '邻页契约对照：no_shard复制P/G/O；下列三种策略分片程度来自config声明，内部hook/完成时序的证据owner是page 36。', 'su'));
  const headers = [
    ['常驻 owner / HBM', 165],
    ['compute 时参数可见性', 441],
    ['gradient sync / owner', 717],
    ['optimizer handoff / update', 993],
    ['next-forward visibility', 1269],
  ];
  headers.forEach(([label, x]) => p.push(text(x, 630, label, 'rank', 'middle')));

  strategyLane(p, 642, {
    id: 'optim',
    height: 88,
    cells: [
      { lines: ['optim', 'P,G buffer full', 'main/O ≈ 1/d'], cls: 'acc1' },
      { lines: ['F/B: full P', 'no compute-time param AG'], cls: 'neutral' },
      { lines: ['gradient RS', 'local g_r', '其余 G buffer 位非全规约值'], cls: 'acc2' },
      { lines: ['local main update', 'parameter AG', 'main shards → full model P'], cls: 'neutral' },
      { lines: ['next-F: full updated P', '每 rank 直接消费'], cls: 'acc1' },
    ],
  });

  strategyLane(p, 744, {
    id: 'optim_grads',
    height: 88,
    cells: [
      { lines: ['optim_grads', 'P full', 'G,main/O ≈ 1/d'], cls: 'acc1' },
      { lines: ['F/B: full P', 'no compute-time param AG'], cls: 'neutral' },
      { lines: ['gradient RS', 'persistent g_r', 'rank r 是 grad owner'], cls: 'acc2' },
      { lines: ['local main update', 'parameter AG', 'main shards → full model P'], cls: 'neutral' },
      { lines: ['next-F: full updated P', '每 rank 直接消费'], cls: 'acc1' },
    ],
  });

  strategyLane(p, 846, {
    id: 'optim_grads_params',
    height: 104,
    cells: [
      { lines: ['optim_grads_params', 'P,G,main/O ≈ 1/d', '+ transient full unit'], cls: 'acc1' },
      { lines: ['forward AG → F', 'post-F reshard', '只留 p_r'], cls: 'acc2' },
      { lines: ['pre-backward AG → B', 'gradient RS', '得到 g_r'], cls: 'acc2' },
      { lines: ['local update keeps p_r', 'updated owner shard', '无 post-step full P 常驻'], cls: 'neutral' },
      { lines: ['next-F AG', 'full updated unit', '再进入 compute'], cls: 'acc1' },
    ],
  });
  p.push(text(42, 974, '成本口径：optim/optim_grads 每周期都以 RS 交付 local g_r，并以参数 AG 把 main shards 写成完整 model P；optim_grads_params 另在 unit 的 F/B 使用边界物化 full P。', 'sm'));

  const decoupledOwnerText = layerWiseDecoupledOwners.map((owned, rank) => (
    `r${rank}:${owned.length ? owned.map((param) => `${param.name}(${param.elements})`).join('+') : '∅'}`
  ));
  const layoutOwnerText = layerWiseLayoutOwners.map((owned, rank) => (
    `r${rank}:${owned.length ? owned.map((param) => `${param.name}(${param.elements})`).join('+') : '∅'}`
  ));

  p.push(rect(22, 1010, 1456, 518, 'panel', 12));
  p.push(text(42, 1040, '④ LayerWiseDistributedOptimizer：同一 logical parameters，两种 live layout 不能并入 ordinary non-DistOpt', 'pt'));
  p.push(text(42, 1062, 'q/p/r 都由一个 rank 以 whole tensor 更新；差异在 gradient collective、persistent padding 与 parameter reconstruction transport。', 'su'));

  p.push('<g data-plane="layerwise-decoupled">');
  p.push(text(42, 1090, 'LayerWise decoupled / variable-size path（默认）', 'dim'));
  dataPlaneCells(p, 1104, [
    { lines: ['same q(2), p(5), r(6)', `compact grad buffer Nraw=${layerWiseLogicalElements}`, 'no equal-shard padding'], cls: 'acc1' },
    { lines: ['F on full logical model P', 'B writes compact gbuf G', 'effective use_distributed_optimizer=False', `gradient all-reduce ≈ ${layerWiseDecoupledArBytes} B/rank`], cls: 'acc2' },
    { lines: ['ping-pong whole-param owner', decoupledOwnerText.slice(0, 2).join(' · '), decoupledOwnerText.slice(2).join(' · ')], cls: 'neutral' },
    { lines: ['whole-param owner update', 'main/state only for owned tensors', 'non-owner copies are stale'], cls: 'neutral' },
    { lines: ['variable-size allgather_params', 'owned sizes [2,5,6,0]', 'sync return / pre-hook consumer wait'], cls: 'acc2' },
  ], 94);
  p.push(text(42, 1220, 'HBM: full logical model P + full compact grad_data；optimizer main/state 按 whole params 分配；同步 gather 暂存 logical payload，overlap path 复用 idle grad_data。', 'sm'));
  p.push(text(42, 1242, '约束/选择：num_distributed_optimizer_instances = 1；默认避开 persistent padding，并可走受限 FP8 gather；代价是 full-gradient AR、uneven AG 与 copy-back。', 'cap'));
  p.push('</g>');

  p.push('<g data-plane="layerwise-layout">');
  p.push(text(42, 1282, 'LayerWise layout / buffer AG path（显式 opt-in）', 'dim'));
  dataPlaneCells(p, 1296, [
    { lines: ['same q(2), p(5), r(6)', 'LPT keeps every tensor whole', layoutOwnerText.slice(0, 2).join(' · '), layoutOwnerText.slice(2).join(' · ')], cls: 'acc1' },
    { lines: ['padded equal-shard layout', `${layerWisePaddedShardElements} slots/rank → Nlayout=${layerWiseLayoutElements}`, `padding=${layerWisePaddingElements} over raw ${layerWiseLogicalElements}`], cls: 'acc2' },
    { lines: ['effective use_distributed_optimizer=True', 'gradient reduce-scatter (not AR)', `bf16 example ≈ ${layerWiseLayoutRsBytes} B/rank`], cls: 'acc2' },
    { lines: ['whole-param owner update', 'rank receives one 64-slot shard', 'only resident whole tensors step'], cls: 'neutral' },
    { lines: ['buffer AG', 'buffer all-gather reconstruction', `all_gather_into_tensor ≈ ${layerWiseLayoutAgBytes} B/rank`], cls: 'acc1' },
  ], 100);
  p.push(text(42, 1418, `HBM: param_data 与 grad_data 各多 ${layerWisePaddingElements} slots（dtype 各自计）；固定等长 RS+AG=${layerWiseLayoutRsBytes + layerWiseLayoutAgBytes} B/rank in this bf16 toy。`, 'sm'));
  p.push(text(42, 1440, '约束/选择：num_distributed_optimizer_instances = 1；FP8/FP4 param gather 与 optimizer-step overlap 不支持；只在接受 padding 且需要固定 buffer collective 时评估。', 'cap'));
  p.push(text(42, 1462, `padding 来源：64-element param-start alignment；shard divisor=lcm(64, lcm(d,128)/d)=${layerWiseShardDivisor}；high-busbw flag 还会纳入 2^16。`, 'cap'));
  p.push('</g>');

  p.push(rect(22, 1548, 1456, 68, 'neutral', 10));
  p.push(text(42, 1576, 'HSDP：只把 d 换成 inner shard group；outer group 复制等价 local shard，并增加跨 instance 对齐，不是新的 ZeRO stage。', 'dim'));
  p.push(text(42, 1600, '图给算法所有权与逻辑 collective；实际字节、峰值和吞吐取决于 unit/bucket、dtype、拓扑、prefetch 与覆盖窗口，所有路径均无固定加速比。', 'cap'));
  p.push(text(1450, 1644, '无固定加速比', 'costtx', 'end'));
  p.push('</svg>');
  return `${p.join('\n')}\n`;
}

function dataPlaneCells(p, y, cells, height = 88) {
  const xs = [42, 336, 630, 924, 1218];
  const width = 250;
  cells.forEach((cell, index) => {
    p.push(box(cell.lines, xs[index], y, width, height, cell.cls));
    if (index < cells.length - 1) {
      p.push(arrow(xs[index] + width, y + height / 2, xs[index + 1], y + height / 2, cell.arrow || 'main'));
    }
  });
}

function renderGradientDataPlanes() {
  const W = 1600, H = 1492, p = [];
  const rankOneRange = ranges[1];
  const rankOneLabels = PARAMS
    .map((param) => {
      const start = Math.max(param.start, rankOneRange.start);
      const end = Math.min(param.end, rankOneRange.end);
      if (end <= start || param.name === 'pad') return null;
      if (end - start === 1) return `${param.name}[${start - param.start}]`;
      return `${param.name}[${start - param.start}:${end - param.start}]`;
    })
    .filter(Boolean)
    .join(' + ');
  const instances = Array.from({ length: HSDP.instances }, (_, instance) => {
    const start = instance * intraSize;
    return `I${instance}={${Array.from({ length: intraSize }, (_, slot) => start + slot).join(',')}}`;
  });
  const interGroups = Array.from({ length: intraSize }, (_, slot) => (
    `{${Array.from({ length: HSDP.instances }, (_, instance) => instance * intraSize + slot).join(',')}}`
  ));

  p.push(svgStart(
    W,
    H,
    '同一个 N=16 flat buffer 从 forward 与 backward-ready 梯度出发，对照普通 all-reduce、标准 reduce-scatter、custom FP32 accumulation reduce-scatter 和 multi-instance HSDP',
  ));
  p.push(`<metadata id="markdown-contract" data-contract-sha256="${figureContractSha256}">16_megatron_distributed_optimizer_analysis.md::distopt-figure-contract</metadata>`);
  p.push(rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 38, '四条 gradient data plane：同一 flat buffer、同一 rank-1 owner 检查点', 'ti'));
  p.push(text(28, 60, `N=${CFG.elements} · d=${CFG.dp} · q[0,2) | p[2,7) | r[7,13) | pad[13,16) · wire=bf16 (${CFG.wireBytesPerElement} B/element)`, 'su'));

  p.push(rect(22, 82, 1556, 126, 'panel', 12));
  p.push(text(42, 110, '共享前置：forward → loss → gradient-ready / backward handoff', 'pt'));
  flowRow(p, 126, [
    { x: 42, w: 250, h: 62, lines: ['full parameter view', 'q | p | r | pad'], cls: 'acc1' },
    { x: 336, w: 250, h: 62, lines: ['forward local compute', 'F(P, batch_s) → loss_s'], cls: 'neutral' },
    { x: 630, w: 250, h: 62, lines: ['backward local compute', `G^(s), full N=${CFG.elements}`], cls: 'neutral' },
    { x: 924, w: 250, h: 62, lines: ['gradient-ready gate', 'final microbatch + full bucket'], cls: 'acc2' },
    { x: 1218, w: 250, h: 62, lines: ['start_grad_sync', '选择下方一条 lane'], cls: 'acc2' },
  ]);

  p.push('<g data-plane="ordinary-all-reduce">');
  p.push(rect(22, 228, 1556, 222, 'panel', 12));
  p.push(text(42, 258, 'A. 普通 all-reduce：只有 plain non-LayerWise optimizer 把 group-full gradient 直接用于 replica-full update', 'pt'));
  p.push(text(42, 278, 'gate: plain non-LayerWise optimizer 的 use_distributed_optimizer=False；或 DistOpt force_all_reduce。LayerWise 有独立 live lanes。', 'su'));
  dataPlaneCells(p, 294, [
    { lines: ['one AR group inputs', 'plain non-LayerWise: whole DP', 'force+k>1: one instance'], cls: 'neutral' },
    { lines: ['all-reduce', 'SUM or AVG', `full payload M=${fullPayloadBytes} B`], cls: 'acc2' },
    { lines: ['group-local full grad', 'global only if k=1', 'k>1 remains partial'], cls: 'acc1' },
    { lines: ['update ownership splits', 'plain non-LayerWise optimizer: full update', 'DistOpt: local shard only'], cls: 'neutral' },
    { lines: ['next parameter visibility', 'plain path: full P local', 'DistOpt: intra parameter AG'], cls: 'acc1' },
  ]);
  p.push(text(42, 404, `同步点：同步调用返回，或 finish_grad_sync 等 handle。理想 ring：2(d−1)M/d = ${ringAllReduceBytes} B/rank。`, 'dim'));
  p.push(text(42, 427, `force_all_reduce+k>1：intra full AR ${hsdpForceIntraArBytes} B + inter local-shard AR ${hsdpInterArBytes} B + AG ${hsdpParamAgBytes} B = ${hsdpForceTotalBytes} B/rank；所有 DP rank 都不持有全局 full gradient。`, 'sm'));
  p.push('</g>');

  p.push('<g data-plane="standard-reduce-scatter">');
  p.push(rect(22, 468, 1556, 238, 'panel', 12));
  p.push(text(42, 498, 'B. 标准 reduce-scatter：gradient owner 分片，step 后 parameter all-gather 重建', 'pt'));
  p.push(text(42, 518, 'gate: Distributed Optimizer + custom FP32 accumulation off；同一个 rank1 range [4,8)。', 'su'));
  dataPlaneCells(p, 534, [
    { lines: ['4 × full contribution', 'q | p | r | pad', 'input buffer N=16'], cls: 'neutral' },
    { lines: ['reduce-scatter', 'intra group d=4', `network ≈ ${ringReduceScatterBytes} B/rank`], cls: 'acc2' },
    { lines: ['rank1 local owner', '[4,8), length 4', rankOneLabels], cls: 'acc1' },
    { lines: ['local compute', 'update main/state shard', 'state ≈ P/4'], cls: 'neutral' },
    { lines: ['parameter all-gather', '4 shards → full P', 'wait before consumer'], cls: 'acc2' },
  ]);
  p.push(text(42, 646, `同步/重建：finish_grad_sync 交付 local grad；AG 另需 ${ringReduceScatterBytes} B/rank，RS+AG=${ringReduceScatterBytes * 2} B。`, 'dim'));
  p.push(text(42, 681, 'owner/data：RS 后非 local grad_data 不可解释为 full reduced gradient；next-forward visibility 以 AG wait 为准。', 'sm'));
  p.push('</g>');

  p.push('<g data-plane="custom-fp32-accumulation-reduce-scatter">');
  p.push(rect(22, 724, 1556, 318, 'panel', 12));
  p.push(text(42, 754, 'C. custom FP32 accumulation RS：lower-precision all_to_all_single + owner-local FP32 sum', 'pt'));
  p.push(text(42, 774, '同例 rank1：C_s→1 = G^(s)[4:8)；通信结束不等于 custom reduction 已完成。', 'su'));
  dataPlaneCells(p, 792, [
    { lines: ['rank-specific chunks', 'C_0→1 … C_3→1', '4 elements/source'], cls: 'neutral' },
    { lines: ['all_to_all_single', 'lower-precision wire', `network ≈ ${ringReduceScatterBytes} B/rank`], cls: 'acc2' },
    { lines: ['A1 full-size temp', '[C_0→1 | … | C_3→1]', `N=16 = ${fullPayloadBytes} B bf16`], cls: 'acc2' },
    { lines: ['custom handle.wait', 'view 4×4; FP32 accumulation', `local sum temp=${fp32LocalSumBytes} B`], cls: 'neutral' },
    { lines: ['downcast / copy', `output [4,8) = ${rankOneLabels}`, 'local update → AG'], cls: 'acc1' },
  ], 94);
  p.push(rect(42, 908, 462, 94, 'acc2', 8));
  p.push(text(60, 934, '同步与生命周期', 'costtx'));
  p.push(text(60, 956, 'successor dispatch 先 drain 已发 predecessor', 'sm'));
  p.push(text(60, 978, 'overlap 由 custom handle.wait 完成 A2A + sum + copy', 'sm'));
  p.push(rect(526, 908, 462, 94, 'neutral', 8));
  p.push(text(544, 934, 'hard constraints', 'dim'));
  p.push(text(544, 956, 'SUM · N%d=0 · overlap: one bucket / bucket group', 'sm'));
  p.push(text(544, 978, 'num_distributed_optimizer_instances = 1', 'sm'));
  p.push(rect(1010, 908, 458, 94, 'acc1', 8));
  p.push(text(1028, 934, '增量成本与重建', 'dim'));
  p.push(text(1028, 956, `+ full LP temp M (${fullPayloadBytes} B) + FP32 N/d (${fp32LocalSumBytes} B)`, 'sm'));
  p.push(text(1028, 978, `例：+${fullPayloadBytes + fp32LocalSumBytes} B temp；随后 parameter AG ${ringReduceScatterBytes} B`, 'sm'));
  p.push(text(42, 1024, 'precision owner：wire/output 仍为 lower precision；只有 owner-local reduction 在 FP32。', 'cap'));
  p.push('</g>');

  p.push('<g data-plane="multi-instance-hsdp">');
  p.push(rect(22, 1060, 1556, 384, 'panel', 12));
  p.push(text(42, 1090, `D. HSDP / num_distributed_optimizer_instances=${HSDP.instances}：intra RS + inter local-shard all-reduce`, 'pt'));
  p.push(text(42, 1110, `same N=${CFG.elements} buffer · total D=${HSDP.totalDp} · k=${HSDP.instances} instances · intra size s=${intraSize}`, 'su'));
  dataPlaneCells(p, 1128, [
    { lines: ['group topology', instances[0], instances[1]], cls: 'neutral' },
    { lines: ['intra reduce-scatter', 'I0 与 I1 各自 RS', `${hsdpIntraRsBytes} B/rank`], cls: 'acc2' },
    { lines: ['inter instance AR', 'same local slot', interGroups.join(' ')], cls: 'acc2' },
    { lines: ['replicated owner', 'rank1 & rank5 own [4,8)', `inter traffic=${hsdpInterArBytes} B/rank`], cls: 'acc1' },
    { lines: ['local update + intra AG', '每 instance 重建 full P', `AG=${hsdpParamAgBytes} B/rank`], cls: 'neutral' },
  ], 98);
  p.push(rect(42, 1250, 462, 126, 'acc2', 8));
  p.push(text(60, 1276, '同步/latency', 'costtx'));
  p.push(text(60, 1298, 'dense bucket-group collection: 1 shared stream', 'sm'));
  p.push(text(60, 1320, 'expert bucket-group collection: another shared stream', 'sm'));
  p.push(text(60, 1342, 'not one stream per bucket group', 'sm'));
  p.push(text(60, 1364, 'RS→inter AR async_op=False；finish: compute wait_stream', 'sm'));
  p.push(rect(526, 1250, 462, 126, 'neutral', 8));
  p.push(text(544, 1276, 'HBM / owner trade-off', 'dim'));
  p.push(text(544, 1298, 'main/state ≈ P/s=P/4，而非 full-D 的 P/8', 'sm'));
  p.push(text(544, 1320, 'owner/state 在两个 instance 各复制一份', 'sm'));
  p.push(text(544, 1342, '跨 instance 只规约 local shard M/s=8 B payload', 'sm'));
  p.push(text(544, 1364, '选择：fast intra / slow inter，或接受复制换拓扑', 'sm'));
  p.push(rect(1010, 1250, 458, 126, 'acc1', 8));
  p.push(text(1028, 1276, '字节/选择', 'dim'));
  p.push(text(1028, 1298, `${hsdpIntraRsBytes} B intra RS + ${hsdpInterArBytes} B inter AR + ${hsdpParamAgBytes} B AG = ${hsdpTotalBytes} B/rank`, 'sm'));
  p.push(text(1028, 1320, `${hsdpForceIntraArBytes} B forced intra AR + ${hsdpInterArBytes} B inter AR + ${hsdpParamAgBytes} B AG = ${hsdpForceTotalBytes} B/rank`, 'sm'));
  p.push(text(1028, 1342, 'forced AR 后只有 local shard 经 inter AR 成为 global', 'sm'));
  p.push(text(1028, 1364, '这是拓扑推导，不是源码吞吐保证', 'sm'));
  p.push(text(42, 1404, 'force_all_reduce+k>1：full gradient 只在 intra-instance group 内成立；跨 instance 只 all-reduce local shard。', 'cap'));
  p.push(text(42, 1426, '限制：custom FP32 accumulation RS 与 multi-instance HSDP 不可同时启用。', 'cap'));
  p.push('</g>');

  p.push(text(28, 1474, 'N=16为等分原语示例，省略native 64/128对齐；pad只通信、不更新；字节为bf16算法估算，不是NCCL实测。', 'cap'));
  p.push('</svg>');
  return `${p.join('\n')}\n`;
}

const outputDir = process.argv[2] || join(
  scriptDir, '..', '..', '..', 'wiki', '02_engineering',
  '02_train_frameworks', 'megatron-lm', 'assets',
);
mkdirSync(outputDir, { recursive: true });
writeFileSync(join(outputDir, 'megatron_distopt_flat_buffer_ranges.svg'), renderFlatRanges(), 'utf8');
writeFileSync(join(outputDir, 'megatron_distopt_rs_update_ag.svg'), renderLifecycle(), 'utf8');
writeFileSync(join(outputDir, 'megatron_distopt_live_paths.svg'), renderLivePaths(), 'utf8');
writeFileSync(join(outputDir, 'megatron_distopt_fp32_rs_hsdp.svg'), renderGradientDataPlanes(), 'utf8');
