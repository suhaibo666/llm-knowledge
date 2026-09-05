// Megatron EP 的三张机制图。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md）----
// 图 1 用同一个 T_global=4、T_local=2/rank、E=4、top-k=2、EP=2 算例复演：
// token 初始 owner → 8 条 route edge → expert owner 上的本地加权计算 → 回收到原
// token；底部补齐 backward
// 的反向所有权链。图中所有 route 数、remote 数和 per-expert 计数都由 ROUTES 推导。
//
// 图 2 不换算例，逐行比较 AllGather、AllToAll、Flex 的训练一级选择；底部单列
// 正交推理 sibling 轴：InferenceMode.is_active() 决定是否从保留的训练 dispatcher
// 切到 inference_moe_token_dispatcher_type={nccl,nvls} 选择的实例，并标出 30/31 owner。
//
// 图 3 把 Flex 的四个二级 backend 拆成独立 lane：每条都交代 MCore 本地重排、
// 依赖边界、expert-major 布局、combine/backward 对偶、同步点和增量资源。DeepEP
// 内部只采用冻结 DeepEP 基线能证实的事实；HybridEP 与 NCCL-EP 不越过未冻结依赖。
//
// 用法：node tools/figs/svg/megatron_ep_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({ tokensGlobal: 4, experts: 4, topk: 2, ep: 2, tp: 1 });
const TOKENS = Object.freeze([
  { id: 't0', owner: 0 },
  { id: 't1', owner: 0 },
  { id: 't2', owner: 1 },
  { id: 't3', owner: 1 },
]);
const ROUTES = Object.freeze([
  { token: 't0', expert: 0, probability: 'p0,0' },
  { token: 't0', expert: 3, probability: 'p0,3' },
  { token: 't1', expert: 1, probability: 'p1,1' },
  { token: 't1', expert: 2, probability: 'p1,2' },
  { token: 't2', expert: 2, probability: 'p2,2' },
  { token: 't2', expert: 3, probability: 'p2,3' },
  { token: 't3', expert: 0, probability: 'p3,0' },
  { token: 't3', expert: 1, probability: 'p3,1' },
]);

const expertOwner = (expert) => Math.floor(expert / (CFG.experts / CFG.ep));
const tokenOwner = (token) => TOKENS.find((entry) => entry.id === token).owner;
const perExpert = Array.from({ length: CFG.experts }, (_, expert) => (
  ROUTES.filter((route) => route.expert === expert).map((route) => route.token)
));
const remoteRoutes = ROUTES.filter((route) => tokenOwner(route.token) !== expertOwner(route.expert));
const receivedUniqueByRank = Array.from({ length: CFG.ep }, (_, rank) => (
  [...new Set(ROUTES.filter((route) => expertOwner(route.expert) === rank).map((route) => route.token))]
));
const routeEdgesByRank = Array.from({ length: CFG.ep }, (_, rank) => (
  ROUTES.filter((route) => expertOwner(route.expert) === rank)
));
const originTokensByRank = Array.from({ length: CFG.ep }, (_, rank) => (
  TOKENS.filter((token) => token.owner === rank)
));
const localTokenCounts = originTokensByRank.map((tokens) => tokens.length);
const tokensLocalPerRank = localTokenCounts[0];
const expertCapacity = (factor) => Math.ceil(
  (tokensLocalPerRank * CFG.topk / CFG.experts) * factor,
);
const capacityAtHalf = expertCapacity(0.5);
const capacityAtOnePointFive = expertCapacity(1.5);
const localEdgesPerExpert = originTokensByRank.map((tokens) => (
  Array.from({ length: CFG.experts }, (_, expert) => ROUTES.filter((route) => (
    tokens.some((token) => token.id === route.token) && route.expert === expert
  )).length)
));
const droppedEdgesAtHalf = localEdgesPerExpert.flat().reduce(
  (total, count) => total + Math.max(0, count - capacityAtHalf), 0,
);
const slotsPerOwnedExpertAfterA2A = capacityAtOnePointFive * CFG.ep * CFG.tp;
const realSlotsPerOwnedExpert = perExpert[0].length;
const zeroSlotsPerOwnedExpert = slotsPerOwnedExpertAfterA2A - realSlotsPerOwnedExpert;
const remoteRankCopies = TOKENS.flatMap((token) => {
  const destinationRanks = new Set(
    ROUTES.filter((route) => route.token === token.id).map((route) => expertOwner(route.expert)),
  );
  destinationRanks.delete(token.owner);
  return [...destinationRanks].map((rank) => `${token.id}:r${token.owner}->r${rank}`);
});
const FLEX_BACKENDS = Object.freeze(['DeepEP', 'DeepEPv2', 'HybridEP', 'NCCL-EP']);
const INFERENCE_SIBLING = Object.freeze({
  field: 'inference_moe_token_dispatcher_type',
  values: Object.freeze(['nccl', 'nvls']),
  classes: Object.freeze(['NCCLAllGatherDispatcher', 'NVLSAllGatherVDispatcher']),
  selector: 'InferenceMode.is_active',
  owners: Object.freeze([
    '30_megatron_rl_posttraining_consistency_analysis',
    '31_megatron_inference_engine_analysis',
  ]),
});
const FIGURE_CONTRACT = Object.freeze({
  tokensGlobal: CFG.tokensGlobal,
  tokensLocalPerRank,
  experts: CFG.experts,
  topk: CFG.topk,
  ep: CFG.ep,
  tp: CFG.tp,
  edges: ROUTES.length,
  remoteEdges: remoteRoutes.length,
  remoteUniqueRankCopies: remoteRankCopies.length,
  capacityAtHalf,
  droppedEdgesAtHalf,
  capacityAtOnePointFive,
  slotsPerOwnedExpertAfterA2A,
  realSlotsPerOwnedExpert,
  zeroSlotsPerOwnedExpert,
  backends: FLEX_BACKENDS,
  inferenceSibling: INFERENCE_SIBLING,
});

if (TOKENS.length !== CFG.tokensGlobal) throw new Error('token table must match T_global');
if (new Set(localTokenCounts).size !== 1 || tokensLocalPerRank * CFG.ep !== CFG.tokensGlobal) {
  throw new Error('example must have T_local=2 on every rank and sum to T_global');
}
if (ROUTES.length !== CFG.tokensGlobal * CFG.topk) {
  throw new Error('route edge count must be T_global*topk');
}
if (remoteRoutes.length !== 4) throw new Error(`expected 4 remote routes, got ${remoteRoutes.length}`);
if (remoteRankCopies.length !== 3) throw new Error(`expected 3 remote rank copies, got ${remoteRankCopies.length}`);
if (perExpert.some((tokens) => tokens.length !== 2)) throw new Error('example must route 2 tokens to every expert');
if (originTokensByRank.some((tokens) => tokens.length !== CFG.tokensGlobal / CFG.ep)) {
  throw new Error('example must have balanced origin tokens');
}
if (capacityAtHalf !== 1 || droppedEdgesAtHalf !== 0) {
  throw new Error(`expected C(0.5)=1 and 0 dropped edges, got ${capacityAtHalf}/${droppedEdgesAtHalf}`);
}
if (capacityAtOnePointFive !== 2) {
  throw new Error(`expected C(1.5)=2, got ${capacityAtOnePointFive}`);
}
if (slotsPerOwnedExpertAfterA2A !== 4 || realSlotsPerOwnedExpert !== 2 || zeroSlotsPerOwnedExpert !== 2) {
  throw new Error('padded AllToAll example must yield 4 slots = 2 real + 2 zero per owned expert');
}

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
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" role="img" aria-label="${esc(aria)}">\n<metadata id="megatron-ep-figure-contract">${esc(JSON.stringify(FIGURE_CONTRACT))}</metadata>\n<style>${style}</style>${defs}`;
}

function renderRouteFigure() {
  const W = 1440, H = 850, p = [];
  p.push(svgStart(W, H, 'EP token 路由训练闭环：全局四个 token、每 rank 两个本地 token，展开为八条专家边，在本地专家路径消费权重后回到原 token，反向按相反所有权链传播'));
  p.push(rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 38, 'EP 的最小原语：token 所有权只在专家计算期间改变', 'ti'));
  p.push(text(28, 60, `T_global=${CFG.tokensGlobal} · T_local=${tokensLocalPerRank}/rank · E=${CFG.experts} · top-k=${CFG.topk} · EP=${CFG.ep} · TP=${CFG.tp}；r0=[t0,t1]，r1=[t2,t3]`, 'su'));

  const columns = [
    { x: 24, w: 236, title: '① 原 token owner', sub: '每 rank 的序列/样本局部布局' },
    { x: 284, w: 300, title: '② router 展开 route edge', sub: `${ROUTES.length} 条 route edge，其中 ${remoteRoutes.length} 条跨 rank` },
    { x: 608, w: 404, title: '③ expert owner 本地计算', sub: '按本地 expert 连续分段，最忙段决定尾部' },
    { x: 1036, w: 380, title: '④ combine 回原位置', sub: '逆通信 + 反置换 + 累计已加权 edge' },
  ];
  columns.forEach((column) => {
    p.push(rect(column.x, 88, column.w, 552, 'panel', 12));
    p.push(text(column.x + 16, 118, column.title, 'pt'));
    p.push(text(column.x + 16, 138, column.sub, 'su'));
  });

  p.push(box(['EP rank 0', '原布局：[t0, t1]'], 48, 174, 188, 92, 'acc1'));
  p.push(box(['EP rank 1', '原布局：[t2, t3]'], 48, 338, 188, 92, 'acc1'));
  p.push(text(48, 474, '正确性不变量', 'rank'));
  p.push(text(48, 495, '输出与 dX 必须回到', 'sm'));
  p.push(text(48, 512, '这两个初始 token owner。', 'sm'));

  TOKENS.forEach((token, tokenIndex) => {
    const routes = ROUTES.filter((route) => route.token === token.id);
    const y = 164 + tokenIndex * 108;
    p.push(box([
      `${token.id} → e${routes[0].expert} (${routes[0].probability})`,
      `${token.id} → e${routes[1].expert} (${routes[1].probability})`,
      `origin = rank${token.owner}`,
    ], 312, y, 244, 82, tokenIndex < 2 ? 'neutral' : 'ghost'));
  });

  [0, 1].forEach((rank) => {
    const y = rank === 0 ? 164 : 394;
    p.push(rect(630, y, 360, 202, rank === 0 ? 'acc1' : 'neutral', 10));
    p.push(text(650, y + 28, `EP rank ${rank}：experts ${rank * 2}, ${rank * 2 + 1}`, 'pt'));
    [rank * 2, rank * 2 + 1].forEach((expert, index) => {
      const tokens = perExpert[expert];
      p.push(box([
        `expert ${expert}: [${tokens.join(', ')}]`,
        `local MLP 消费 p → [p·f${expert}(${tokens[0]}), p·f${expert}(${tokens[1]})]`,
        `tokens_per_expert = ${tokens.length}`,
      ], 650, y + 48 + index * 72, 320, 62, index === 0 ? 'ghost' : 'neutral'));
    });
  });

  p.push(box(['rank0 输出布局', 'y0, y1'], 1064, 172, 160, 72, 'acc1'));
  p.push(box(['rank1 输出布局', 'y2, y3'], 1064, 332, 160, 72, 'acc1'));
  p.push(box([
    '每个 token 的语义',
    'y_i = Σ_e p_i,e f_e(t_i)',
    'combine 不再重复乘 p',
  ], 1244, 236, 148, 116, 'neutral'));
  p.push(box([
    '本例 route / capacity 可核算',
    `${ROUTES.length} 条 edge；${remoteRoutes.length} 条跨 rank`,
    `f=.5 → C=${capacityAtHalf}；丢 ${droppedEdgesAtHalf} 条真 edge`,
    `f=1.5 → C=${capacityAtOnePointFive}`,
    `A2A 后每本地 expert ${slotsPerOwnedExpertAfterA2A} slots`,
    `= ${realSlotsPerOwnedExpert} 真 + ${zeroSlotsPerOwnedExpert} 零`,
  ], 1064, 438, 328, 144, 'acc2'));

  p.push(arrow(236, 220, 312, 205));
  p.push(arrow(236, 384, 312, 421));
  p.push(arrow(556, 300, 630, 266));
  p.push(arrow(556, 408, 630, 496));
  p.push(arrow(990, 266, 1064, 208));
  p.push(arrow(990, 496, 1064, 368));

  p.push(rect(24, 664, W - 48, 134, 'panel', 12));
  p.push(text(44, 694, '⑤ backward：同一 ownership 图反向走，不是“本地 MLP 算完就结束”', 'pt'));
  p.push(text(44, 716, 'dY → combine⁻¹ → expert backward（穿过 p）→ dispatch⁻¹ → dX', 'dim'));
  p.push(text(1384, 716, 'local expert dW 随后才交给 EDP / optimizer', 'sm', 'end'));
  p.push(box(['dY', '原 token owner'], 54, 731, 146, 48, 'acc1'));
  p.push(arrow(200, 755, 276, 755));
  p.push(box(['combine⁻¹', 'grad 发回 expert owner'], 276, 725, 210, 60, 'neutral'));
  p.push(arrow(486, 755, 562, 755));
  p.push(box(['expert backward', 'local dW + route dX'], 562, 725, 210, 60, 'neutral'));
  p.push(arrow(772, 755, 848, 755));
  p.push(box(['dispatch⁻¹', 'grad 回原 token owner'], 848, 725, 210, 60, 'neutral'));
  p.push(arrow(1058, 755, 1134, 755));
  p.push(box(['dX', '恢复初始布局'], 1134, 731, 146, 48, 'acc1'));
  p.push(text(28, 826, '读图结论：EP 分 expert 参数；route weight 在本地 expert 路径消费，combine 回送并累计已加权结果，最终恢复初始 token owner。', 'cap'));
  p.push('</svg>');
  return `${p.join('\n')}\n`;
}

function variantPanel(p, y, spec) {
  p.push(rect(22, y, 1456, 260, 'panel', 12));
  p.push(text(42, y + 30, spec.title, 'pt'));
  p.push(text(42, y + 50, spec.subtitle, 'su'));
  const xs = [42, 320, 608, 896];
  spec.steps.forEach((step, index) => {
    p.push(box(step.lines, xs[index], y + 78, 244, 98, step.cls));
    if (index < spec.steps.length - 1) p.push(arrow(xs[index] + 244, y + 127, xs[index + 1], y + 127, index === 0 ? 'cost' : 'main'));
  });
  p.push(box(spec.cost, 1184, y + 76, 270, 102, 'acc2'));
  p.push(text(42, y + 207, spec.backward, 'dim'));
  p.push(text(42, y + 229, spec.boundary, 'sm'));
}

function renderVariantsFigure() {
  const W = 1500, H = 1300, p = [];
  p.push(svgStart(W, H, '同一组全局四 token、每 rank 两 token 的路由在 AllGather、AllToAll 与 Flex 训练 dispatcher 中的数据布局、通信、恢复、反向对偶与成本对照，并单列 nccl 与 nvls 的正交推理 sibling 选择边界'));
  p.push(rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 38, '三种 dispatcher：模型语义相同，发送 buffer 与通信账不同', 'ti'));
  p.push(text(28, 60, `T_global=${CFG.tokensGlobal} · T_local=${tokensLocalPerRank}/rank；r0=[t0,t1]，r1=[t2,t3]；top-${CFG.topk} 共 ${ROUTES.length} edge、${remoteRoutes.length} remote edge；TP=${CFG.tp}`, 'su'));

  variantPanel(p, 84, {
    title: '① AllGather：先复制所有原 token 与整张路线表，再在每个 rank 筛本地专家',
    subtitle: '不计算逐目的地 split；用 TP×EP collective 的全局可见性换简单控制面。',
    steps: [
      { lines: ['原 owner buffer', 'r0:[t0,t1] · r1:[t2,t3]', '每 rank map/probs: [2,4]'], cls: 'ghost' },
      { lines: ['AG hidden + map + probs', '每 rank 看见 [t0,t1,t2,t3]', '聚合 map/probs: [4,4]'], cls: 'acc2' },
      { lines: ['筛本地 expert edge', 'r0: e0[t0,t3], e1[t1,t3]', 'r1: e2[t1,t2], e3[t0,t2]'], cls: 'acc1' },
      { lines: ['local MLP → unpermute', 'RS 累加并切回 origin', 'r0:[y0,y1] · r1:[y2,y3]'], cls: 'neutral' },
    ],
    cost: ['增量成本', `每 rank 从 T_local=${tokensLocalPerRank}`, `物化 T_global=${CFG.tokensGlobal}`, '并携带全局 map/probs', '无 input/output splits'],
    backward: 'Backward 对偶：dispatch AGᵀ = RS；combine RSᵀ = AG；mapping 反向回收 top-k edge。',
    boundary: '适用性是 README 的小 EP / 大 top-k 包络，不是通用性能保证；variable sequence lengths 与 packing 有 guard。',
  });

  variantPanel(p, 358, {
    title: '② AllToAll：发送前展开 route copy，只把每条 edge 送给 expert owner',
    subtitle: '通信量贴近实际 route；代价是变长 split、DtoH/sync 与两层排列。',
    steps: [
      { lines: ['permute 后发送 buffer', `每 rank ${tokensLocalPerRank * CFG.topk} 条 route copy`, `全局 K = ${CFG.tokensGlobal}·${CFG.topk} = ${ROUTES.length}`], cls: 'ghost' },
      { lines: ['EP variable-size A2A', 'input_splits = [2, 2]', '本例跨 rank edge = 4'], cls: 'acc2' },
      { lines: ['收件 + 本地二次排列', '按 e0/e1 或 e2/e3 连续', 'tokens_per_expert=[2,2]'], cls: 'acc1' },
      { lines: ['local MLP 消费 route weight', 'reverse A2A 交换 splits', 'unpermute 累计已加权 edge'], cls: 'neutral' },
    ],
    cost: ['增量成本', '4 条 remote route payload/方向', 'split materialize + sync', 'permute + local re-sort'],
    backward: 'Backward 对偶：combine A2A 把 dY 送回 expert；dispatch A2A 再把 route dX 送回 token owner。',
    boundary: '这里的 [2,2] 只是平衡算例；实际 split 随 routing_map 变化，最大收件量同时决定网络与 expert compute 尾部。',
  });

  variantPanel(p, 632, {
    title: '③ Flex：manager 融合数据面，但仍须满足同一个 ownership contract',
    subtitle: 'MCore 二级选择 DeepEP / DeepEPv2 / HybridEP / NCCL-EP；四条独立 lane 见下一图。',
    steps: [
      { lines: ['route metadata', '[T_local,world,E_local]', '本例 [2,2,2] 视图'], cls: 'ghost' },
      { lines: ['manager.dispatch', '不是 backend-defined 单一路径', '四个 manager 各自成 lane'], cls: 'acc2' },
      { lines: ['per-expert local segment', '本例仍是每 expert 2 条', 'local expert compute'], cls: 'acc1' },
      { lines: ['local MLP 先消费 p', 'manager.combine 逆向搬运', '累计后恢复 [T_local,H]'], cls: 'neutral' },
    ],
    cost: ['增量成本', 'workspace / sync 按 manager 分账', '仍受 route skew 约束', '无固定加速比声明'],
    backward: 'Backward 契约：manager autograd 必须完成 combine⁻¹ 与 dispatch⁻¹；融合名称不缩短正确性边界。',
    boundary: 'backend、TP×EP、capacity/padding 与安装版本由配置 guard 决定；本图只承诺 MCore 可观察的输入输出。',
  });

  p.push(rect(22, 906, 1456, 264, 'panel', 12));
  p.push(text(42, 936, '④ 正交推理 sibling 轴：不是训练 dispatcher 的第四值', 'pt'));
  p.push(text(42, 958, '仅在 transformer_impl=inference_optimized 且 EP>1 时构造；inference_moe_token_dispatcher_type ∈ {nccl,nvls}；训练 dispatcher 同时保留。', 'su'));
  p.push(box(['MoELayer.forward', 'InferenceMode.is_active()'], 42, 994, 250, 64, 'ghost'));
  p.push(arrow(292, 1026, 324, 1026, 'main'));
  p.push(box(['inference_moe_token_dispatcher_type', '∈ {nccl,nvls}'], 324, 994, 370, 64, 'acc2'));
  p.push(`<path data-inference-branch="nccl" class="main" d="M 694 1026 H 750 V 1004 H 800"/>`);
  p.push(`<path data-inference-branch="nvls" class="main" d="M 694 1026 H 750 V 1076 H 800"/>`);
  p.push(box(['nccl', 'NCCLAllGatherDispatcher'], 800, 976, 500, 56, 'acc1'));
  p.push(box(['nvls', 'NVLSAllGatherVDispatcher'], 800, 1048, 500, 56, 'acc1'));
  p.push(text(42, 1128, '选择边界：active=false → 训练 {allgather,alltoall,flex}；active=true → 推理实例。不是训练 dispatcher 的第四值。', 'dim'));
  p.push(text(42, 1150, '机制深挖 rehome → 30_megatron_rl_posttraining_consistency_analysis；InferenceMode 生命周期 → 31_megatron_inference_engine_analysis', 'sm'));

  p.push(rect(22, 1186, 1456, 92, 'neutral', 10));
  p.push(text(42, 1214, `共同不变量：T_global=${CFG.tokensGlobal}，每 rank T_local=${tokensLocalPerRank}；${ROUTES.length} edge、每 expert ${realSlotsPerOwnedExpert} 条真 edge，输出回原 owner。`, 'dim'));
  p.push(text(42, 1237, `本地 capacity：f=.5 → C=${capacityAtHalf}/不丢边；f=1.5 + pad → C=${capacityAtOnePointFive}，A2A 后每本地 expert ${slotsPerOwnedExpertAfterA2A} slots=${realSlotsPerOwnedExpert} 真+${zeroSlotsPerOwnedExpert} 零。`, 'cap'));
  p.push(text(42, 1258, '变化的是复制、split、同步与 workspace；图中为逻辑元素数而非物理链路字节或实测时延。', 'cap'));
  p.push('</svg>');
  return `${p.join('\n')}\n`;
}

function flexLane(p, y, spec) {
  p.push(rect(22, y, 1556, 250, 'panel', 12));
  p.push(text(42, y + 30, spec.title, 'pt'));
  p.push(text(42, y + 50, spec.subtitle, 'su'));
  const xs = [42, 294, 546, 798, 1050];
  const widths = [230, 230, 230, 230, 506];
  spec.steps.forEach((step, index) => {
    p.push(box(step.lines, xs[index], y + 70, widths[index], 112, step.cls));
    if (index < spec.steps.length - 1) {
      p.push(arrow(xs[index] + widths[index], y + 126, xs[index + 1], y + 126, index === 0 ? 'cost' : 'main'));
    }
  });
  p.push(text(42, y + 211, spec.backward, 'dim'));
  p.push(text(42, y + 233, spec.boundary, 'sm'));
}

function renderFlexBackendsFigure() {
  const W = 1600, H = 1248, p = [];
  p.push(svgStart(W, H, 'Flex 四后端独立数据面：DeepEP、DeepEPv2、HybridEP、NCCL-EP 对同一全局四 token、每 rank 两 token 路由执行分发、专家计算、合并和反向对偶'));
  p.push(rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, 'Flex 二级选择不是一个黑盒：四个 manager 的数据布局与付款点', 'ti'));
  p.push(text(28, 58, `T_global=${CFG.tokensGlobal} · T_local=${tokensLocalPerRank}/rank；每 rank hidden=[${tokensLocalPerRank},H]、router=[${tokensLocalPerRank},${CFG.experts}]；${ROUTES.length} edge / ${remoteRoutes.length} remote / ${remoteRankCopies.length} remote rank-copy`, 'su'));

  flexLane(p, 76, {
    title: '① DeepEP（moe_flex_dispatcher_backend=deepep）',
    subtitle: '冻结依赖 DeepEP@af9a040；v1 Buffer 先按目标 rank 去重传 token，MCore 再展开成本地 expert-major edge。',
    steps: [
      { lines: ['router → MCore top-k', `r0/r1 各 [${originTokensByRank[0].length},H]`, `indices/weights: [${originTokensByRank[0].length},${CFG.topk}]`, `逻辑 edge = ${ROUTES.length}`], cls: 'ghost' },
      { lines: ['MCore ↔ DeepEP 边界', 'Buffer.dispatch', '每目标 rank 一份 token', `remote rank-copy = ${remoteRankCopies.length}`], cls: 'acc2' },
      { lines: ['MCore 局部再 permute', `r0 收唯一 ${receivedUniqueByRank[0].join(',')}`, `→ e0/e1 共 ${routeEdgesByRank[0].length} edge`, `r1 对称为 ${routeEdgesByRank[1].length} edge`], cls: 'acc1' },
      { lines: ['expert-major + local MLP', `e0:[${perExpert[0].join(',')}]`, `e1:[${perExpert[1].join(',')}]`, 'TEGrouped/Sequential 在此消费 p'], cls: 'neutral' },
      { lines: ['combine / 账本', 'MCore unpermute 已累计加权 edge', 'Buffer.combine(x, handle)', 'forward 不传 route weights', 'layout/buffer + 默认 20 SM', '精确收件数与 layout 带同步'], cls: 'acc2' },
    ],
    backward: 'Backward：combineᵀ 按 handle dispatch dY；expert 反向穿过 p 得 p·dY 与 dp；dispatchᵀ 再 combine dX/dp 回 origin。',
    boundary: '依赖边界：MCore 负责 local expert-major 重排；DeepEP@af9a040 负责 v1 NVLink/RDMA 搬运。其 internode kernel 对同一目标 RDMA 域只发送一份 token。',
  });

  flexLane(p, 334, {
    title: '② DeepEPv2（moe_flex_dispatcher_backend=deepepv2）',
    subtitle: '冻结依赖 DeepEP@af9a040；ElasticBuffer 的非展开收件布局之后，仍由 MCore 做第二次 per-expert permute。',
    steps: [
      { lines: ['router → MCore top-k', `同一 [${originTokensByRank[0].length},H] / rank`, `indices/weights: [${originTokensByRank[0].length},${CFG.topk}]`, `逻辑 edge = ${ROUTES.length}`], cls: 'ghost' },
      { lines: ['MCore ↔ DeepEPv2 边界', 'ElasticBuffer.dispatch', 'do_expand=false', '自动或显式 SM/QP'], cls: 'acc2' },
      { lines: ['MCore 局部再 permute', 'non-expanded recv token', `→ 本 rank ${routeEdgesByRank[1].length} expert edge`, `tokens_per_expert=[${perExpert.slice(2).map((tokens) => tokens.length).join(',')}]`], cls: 'acc1' },
      { lines: ['expert-major + local MLP', `e2:[${perExpert[2].join(',')}]`, `e3:[${perExpert[3].join(',')}]`, '本地消费 p 后再 unpermute'], cls: 'neutral' },
      { lines: ['combine / 账本', 'Elastic combine 累计已加权输出', '全局缓存/按需放大 buffer', 'ranks≤1024 · experts≤2048 · local≤256', 'float32 probs；num_sms=0 自动', '默认精确计数有 CPU sync'], cls: 'acc2' },
    ],
    backward: 'Backward：Elastic combine 的反向调 dispatch；Elastic dispatch 的反向调 combine；相同 handle 恢复 origin 与 route-weight gradient。',
    boundary: '依赖边界：MCore 管 local rearrangement；DeepEP@af9a040 管 ElasticBuffer。源码发布测量只适用于其指定机器/shape，不是本例加速保证。',
  });

  flexLane(p, 592, {
    title: '③ HybridEP（moe_flex_dispatcher_backend=hybridep）',
    subtitle: 'dispatch_with_permute 直接返回 expert-major；与 NCCL-EP 一样不调用 MCore 二次 local permute，但这里由 HybridEP 融合通信与两级 permute。',
    steps: [
      { lines: ['router → MCore map', `同一 [${originTokensByRank[0].length},H] / rank`, 'routing_map/probs', `逻辑 edge = ${ROUTES.length}`], cls: 'ghost' },
      { lines: ['MCore ↔ HybridEP 边界', 'dispatch_with_permute', '通信 + 两级 permute', '实现依赖未在 main 冻结'], cls: 'acc2' },
      { lines: ['直接 expert-major', `r0: e0/e1 各 ${perExpert[0].length}`, `r1: e2/e3 各 ${perExpert[2].length}`, '无需 MCore 二次 permute'], cls: 'acc1' },
      { lines: ['local expert MLP', `有效 edge 共 ${ROUTES.length}`, 'rank 超预算：依赖内 drop', 'handle flag → over_budget'], cls: 'neutral' },
      { lines: ['恢复 / overflow', `handle 恢复 [${originTokensByRank[0].length},H]`, '常规 PagedStashRunner：整步重跑', '清 rank cap/stash → dropless', '仅已捕获 TE whole-MoE 图', 'RuntimeError，禁止动态 fallback'], cls: 'acc2' },
    ],
    backward: 'Backward：先完成含 drop 的首轮 F/B 并汇总 over_budget；常规整步清梯度后 dropless 重跑，成功轮才定义最终梯度。',
    boundary: '依赖边界：MCore 只证 wrapper、flag 与 rerun；DeepEP@af9a040 main 不含 HybridEPBuffer，不臆测其 hop。动态计数仍可能 DtoH。',
  });

  flexLane(p, 850, {
    title: '④ NCCL-EP（moe_flex_dispatcher_backend=ncclep）',
    subtitle: 'TE ep_dispatch 直接返回 expert-major packed buffer；同样不调用 MCore 二次 local permute，只做 valid narrow / capacity 恢复。',
    steps: [
      { lines: ['router → MCore top-k', `同一 [${originTokensByRank[0].length},H] / rank`, `indices/weights: [${originTokensByRank[0].length},${CFG.topk}]`, `逻辑 edge = ${ROUTES.length}`], cls: 'ghost' },
      { lines: ['MCore ↔ TE 边界', 'ep_dispatch(buffer,…)', 'collective bootstrap + barrier', 'TE 实现未纳入冻结源'], cls: 'acc2' },
      { lines: ['直接 expert-major', '不调用 MCore 二次 local permute', `有效段每 expert = ${perExpert[0].length}`, 'dynamic narrow / static slack'], cls: 'acc1' },
      { lines: ['local fused MLP 消费 p', `dynamic: 本 rank ${routeEdgesByRank[0].length} edge`, 'static: 固定 capacity view', '输出已加权再扩回 buffer'], cls: 'neutral' },
      { lines: ['combine / 账本', `ep_combine(buffer,…) → [${originTokensByRank[0].length},H]`, '动态 count 有 DtoH .item()', 'token ceiling 按 64 对齐', 'budget 再按 expert alignment', '每轮 fresh EpBuffer'], cls: 'acc2' },
    ],
    backward: 'Backward：MCore 保留 TE autograd 边界；梯度必须由 ep_combine 回 expert、ep_dispatch 回 origin，但未冻结 TE 源，内部 transpose 不作实现断言。',
    boundary: '依赖边界：static 需 SM100+、TE fused grouped-MLP/op-fuser 与环境开关；symmetric-memory 仍 NotImplemented；rank capacity overflow 是硬失败。',
  });

  p.push(rect(22, 1116, 1556, 92, 'neutral', 10));
  p.push(text(42, 1144, `共同不变量：T_global=${CFG.tokensGlobal}、T_local=${tokensLocalPerRank}/rank；local router [${tokensLocalPerRank},${CFG.experts}] → dispatch → expert-major → 加权 MLP → combine → origin [${tokensLocalPerRank},H]。`, 'dim'));
  p.push(text(42, 1167, `capacity 同例：f=.5 → C=${capacityAtHalf}/丢 ${droppedEdgesAtHalf} 真边；f=1.5 + pad → C=${capacityAtOnePointFive}，A2A 后每本地 expert ${slotsPerOwnedExpertAfterA2A}=${realSlotsPerOwnedExpert} 真+${zeroSlotsPerOwnedExpert} 零 slots。`, 'cap'));
  p.push(text(42, 1190, 'HybridEP 超预算是 drop+over_budget+常规 dropless rerun；NCCL-EP 依赖侧 hard trap。蓝=依赖，橙=通信/同步/资源。', 'cap'));
  p.push('</svg>');
  return `${p.join('\n')}\n`;
}

const outputDir = process.argv[2] || join(
  dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'wiki', '02_engineering',
  '02_train_frameworks', 'megatron-lm', 'assets',
);
mkdirSync(outputDir, { recursive: true });
writeFileSync(join(outputDir, 'megatron_ep_route_compute_combine.svg'), renderRouteFigure(), 'utf8');
writeFileSync(join(outputDir, 'megatron_ep_dispatcher_variants.svg'), renderVariantsFigure(), 'utf8');
writeFileSync(join(outputDir, 'megatron_ep_flex_backends.svg'), renderFlexBackendsFigure(), 'utf8');
