// 图 1：同一个 TransformerLayer 类，两份后端 spec 装出两张不同拓扑的插槽表。
// 图 2：MoE router 的 top-k 选择算法——一个 token 从 logits 走到 (probs, routing_map)。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 `megatron_model_spec_slot_topology.svg`
// 要回答的**一个**问题：同一份 config 换一个 transformer_impl 字符串，为什么装出来的
// 不是「同样的层换了算子」，而是**槽的数量都变了**——而 checkpoint 却仍然一致？
//
// 布局（左右两条泳道，同一张 12 槽表逐行对齐）：
// - 每条泳道自上而下：config 进 → get_backend(transformer_impl) → provider 实例
//   （带 fuse_layernorm_and_linear() 的返回值徽标）→ TransformerLayerSubmodules 的 12 行槽。
// - 12 行槽按 dataclass 的声明顺序（也是 TransformerLayer.__init__ 的构造顺序）排。
//   非哨兵槽画成实框（.acc1 = 本图机制色），哨兵槽画成 .ghost 并写出它到底是
//   IdentityOp 还是 IdentityFuncOp。
// - 左泳道 local：input_layernorm 与 pre_mlp_layernorm 是**独立的实框**；
//   self_attention 槽内嵌 SelfAttentionSubmodules 三行，linear_qkv = ColumnParallelLinear，
//   norm 不在里面。非哨兵计数 = 6。
// - 右泳道 transformer_engine：**同一个 TransformerLayer 类**，但两个 norm 槽退化成
//   .ghost 的 IdentityOp，norm 以一枚 LN 小片**画进** linear_qkv / linear_fc1 的框内——
//   证明 fuse_layernorm_and_linear() 改的是槽的拓扑，不只是算子选型。非哨兵计数 = 4。
// - 两个计数都由 SLOTS 表算出来，不许手写，图才不会和正文的 6 / 4 打架。
// - 底部 checkpoint band：local 的原始键 --(.aux 虚线, 标注 sharded_state_dict_keys_map
//   的两条重命名)--> 中间的**规范键** <--(.aux 虚线, "TE 布局即规范名，无需重命名")-- TE 的原生键。
//   这条虚线是本图的治理不变量：两条泳道模块树不同形，却写出同一份键。
// - 代价（.acc2，全图仅此一处橙色）：decoder-only 下 pre_cross_attn_layernorm /
//   cross_attention / cross_attn_bda 三个槽恒为哨兵，_forward_attention 仍无条件调用。
//   这三行在两条泳道上各打一枚 .acc2 tick，底部一条 .acc2 说明带解释。
//
// 图 2 `megatron_moe_router_routing.svg`
// 要回答的**一个**问题：expert bias 既然要改变「谁被选中」，为什么它不会污染
// 被选中专家的权重？——即 aux-loss-free 负载均衡为什么不改变模型的函数。
//
// 布局（四个面板，主线自上而下，同一个 token t0 贯穿）：
// ① 打分：logits(t0) 8 格 → sigmoid → scores 8 格。
// ② 选择：scores + expert_bias = scores_for_routing → 每组取 top-(topk/group_topk) 求和
//    得 group score → 选中 group_topk 组、其余置 -inf → top-k → top_indices。
//    右侧 .ghost 对照泳道：expert_bias 全零时选中的是另一组专家（反事实，证明 bias 真的在起作用）。
// ③ 加权：用 top_indices 去 **未加 bias 的 scores** 上 gather → 归一化 → probs。
//    这里是本图的不变量（.acc1）：bias 进选择、出权重，torch.gather 的对象是 scores 不是
//    scores_for_routing，所以负载均衡不进入前向函数值。
// ④ 输出与回路：scatter 成 [num_tokens, num_experts] 的 routing_probs / routing_map
//    （3 个 token × 8 专家的格子，t0 为主，t1/t2 让列和有意义）；routing_map.sum(dim=0)
//    累进 local_tokens_per_expert，.aux 虚线回到 ② 的 expert_bias——闭环。
//    右下 .acc2 记代价：稠密 [T,E] 张量、以及 bias 更新是 no_grad 的训练期副作用。
// - 边界：本图**停在** (probs, routing_map)。dispatch / expert / combine 属于页 14，不画。
// - 图上每个数字（sigmoid 值、group score、top_indices、probs、列和）都由下面的
//   routeToken() 现算，改 CFG2 的 logits 结论自动重算。
//
// 冻结基线：NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a（dev，2026-09-01）
// 用法：node tools/figs/svg/megatron_model_structure_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// 图 1 的数据源：TransformerLayerSubmodules 的 12 个模块槽（声明顺序 = 构造顺序），
// 以及 dense / 非 MLA / 无 mHC 下两个 spec 工厂各自填进去的类。
// 源码：megatron/core/transformer/transformer_layer.py::TransformerLayerSubmodules
//       megatron/core/models/gpt/gpt_layer_specs.py::get_gpt_layer_local_submodules
//       megatron/core/models/gpt/gpt_layer_specs.py::get_gpt_layer_with_transformer_engine_submodules
// ---------------------------------------------------------------------------

const SLOTS = Object.freeze([
  { name: 'input_layernorm', sentinel: 'IdentityOp', local: 'LNImpl', te: null, fusedInto: 'self_attention.linear_qkv' },
  { name: 'self_attention_hyper_connection', sentinel: 'IdentityOp', local: null, te: null },
  {
    name: 'self_attention', sentinel: 'IdentityOp', nested: 'attn',
    local: 'ModuleSpec(module=SelfAttention)', te: 'ModuleSpec(module=SelfAttention)',
  },
  { name: 'self_attn_bda', sentinel: 'IdentityFuncOp', local: 'get_bias_dropout_add', te: 'get_bias_dropout_add' },
  { name: 'pre_cross_attn_layernorm', sentinel: 'IdentityOp', local: null, te: null, dead: true },
  { name: 'cross_attention_hyper_connection', sentinel: 'IdentityOp', local: null, te: null },
  { name: 'cross_attention', sentinel: 'IdentityOp', local: null, te: null, dead: true },
  { name: 'cross_attn_bda', sentinel: 'IdentityFuncOp', local: null, te: null, dead: true },
  { name: 'pre_mlp_layernorm', sentinel: 'IdentityOp', local: 'LNImpl', te: null, fusedInto: 'mlp.linear_fc1' },
  { name: 'mlp_hyper_connection', sentinel: 'IdentityOp', local: null, te: null },
  {
    name: 'mlp', sentinel: 'IdentityOp', nested: 'mlp',
    local: 'partial(MLP.as_mlp_submodule)', te: 'partial(MLP.as_mlp_submodule)',
  },
  { name: 'mlp_bda', sentinel: 'IdentityFuncOp', local: 'get_bias_dropout_add', te: 'get_bias_dropout_add' },
]);

const NESTED = Object.freeze({
  attn: {
    table: 'SelfAttentionSubmodules',
    rows: [
      { slot: 'linear_qkv', local: 'ColumnParallelLinear', te: 'TELayerNormColumnParallelLinear', teFused: true },
      { slot: 'core_attention', local: 'DotProductAttention', te: 'TEDotProductAttention' },
      { slot: 'linear_proj', local: 'RowParallelLinear', te: 'TERowParallelLinear' },
    ],
  },
  mlp: {
    table: 'MLPSubmodules',
    rows: [
      { slot: 'linear_fc1', local: 'ColumnParallelLinear', te: 'TELayerNormColumnParallelLinear', teFused: true },
      { slot: 'linear_fc2', local: 'RowParallelLinear', te: 'TERowParallelLinear' },
    ],
  },
});

const LANES = Object.freeze([
  {
    key: 'local', impl: 'local', provider: 'LocalSpecProvider', fuse: false,
    title: '① transformer_impl = "local"',
    fc1Line: 'linear_fc1 = backend.column_parallel_linear()',
    rawKeys: ['input_layernorm.weight', 'pre_mlp_layernorm.weight'],
  },
  {
    key: 'te', impl: 'transformer_engine', provider: 'TESpecProvider', fuse: true,
    title: '② transformer_impl = "transformer_engine"',
    fc1Line: 'linear_fc1 = backend.column_parallel_layer_norm_linear()',
    rawKeys: ['self_attention.linear_qkv.layer_norm_weight', 'mlp.linear_fc1.layer_norm_weight'],
  },
]);

// sharded_state_dict_keys_map，local dense spec 实际携带的两条前缀重命名。
const RENAME_MAP = Object.freeze([
  ['input_layernorm.', 'self_attention.linear_qkv.layer_norm_'],
  ['pre_mlp_layernorm.', 'mlp.linear_fc1.layer_norm_'],
]);

// 规范键：TE 布局天然产出、也是 local 重命名后的目标；由 RENAME_MAP 推导，不手写。
const CANONICAL_KEYS = RENAME_MAP.map(([, to]) => `${to}weight`);

const countFilled = (lane) => SLOTS.filter((slot) => slot[lane] !== null).length;
const deadSlots = SLOTS.filter((slot) => slot.dead).map((slot) => slot.name);

const filledCount = Object.freeze({ local: countFilled('local'), te: countFilled('te') });

if (filledCount.local !== 6 || filledCount.te !== 4) {
  throw new Error(`slot occupancy drifted: local=${filledCount.local} te=${filledCount.te}`);
}
if (deadSlots.length !== 3) throw new Error(`dead slot count drifted: ${deadSlots.length}`);

// ---------------------------------------------------------------------------
// 图 2 的数据源：TopKRouter 的 sigmoid + expert bias + group-limited top-k 路径。
// 源码：megatron/core/transformer/moe/router.py::TopKRouter.routing / _apply_expert_bias
//       megatron/core/transformer/moe/moe_utils.py::topk_routing_with_score_function
//       megatron/core/transformer/moe/moe_utils.py::group_limited_topk
// ---------------------------------------------------------------------------

const CFG2 = Object.freeze({
  numExperts: 8,
  topk: 2,
  numGroups: 4,
  groupTopk: 2,
  scoreFunction: 'sigmoid',
  // 每个 token 一行 gating logits，形状 [num_tokens, num_experts]。
  logits: [
    [2.0, -1.0, 1.2, -0.5, 1.0, 0.3, -2.0, -1.5],
    [-0.8, 1.5, 0.9, -0.3, 0.2, 1.1, -1.2, 0.6],
    [0.4, -0.6, -1.1, 1.3, -0.2, 0.1, 1.6, -0.9],
  ],
  // aux-loss-free 的动态偏置，训练中由 local_tokens_per_expert 反馈更新。
  expertBias: [0, 0, -0.2, 0, 0.3, 0, 0, 0],
});

const expertsPerGroup = CFG2.numExperts / CFG2.numGroups;
const scorePerGroup = Math.floor(CFG2.topk / CFG2.groupTopk); // topk // group_topk

for (const [name, value] of Object.entries({ expertsPerGroup })) {
  if (!Number.isInteger(value)) throw new Error(`${name} must be integral, got ${value}`);
}
if (scorePerGroup < 1) throw new Error('topk // group_topk must be >= 1');

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function scoreFn(logit) {
  if (CFG2.scoreFunction === 'sigmoid') return sigmoid(logit);
  throw new Error(`unsupported score_function ${CFG2.scoreFunction}`);
}

// group_limited_topk：组内取 top-(topk//group_topk) 求和作为组分，选 group_topk 组，
// 其余组的专家置 -inf，再在剩下的候选里取 top-k。
function groupLimitedTopk(routingScores) {
  const groupScores = [];
  for (let g = 0; g < CFG2.numGroups; g += 1) {
    const slice = routingScores.slice(g * expertsPerGroup, (g + 1) * expertsPerGroup);
    const top = [...slice].sort((a, b) => b - a).slice(0, scorePerGroup);
    groupScores.push(top.reduce((a, b) => a + b, 0));
  }
  const selectedGroups = groupScores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .slice(0, CFG2.groupTopk)
    .map((entry) => entry.index)
    .sort((a, b) => a - b);
  const masked = routingScores.map(
    (score, e) => (selectedGroups.includes(Math.floor(e / expertsPerGroup)) ? score : -Infinity),
  );
  const topIndices = masked
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .slice(0, CFG2.topk)
    .map((entry) => entry.index);
  return { groupScores, selectedGroups, masked, topIndices };
}

// 一个 token 走完 routing：打分 → 加 bias 选择 → 回到无 bias 的 scores 上 gather → 归一化。
function routeToken(logits, expertBias) {
  const scores = logits.map(scoreFn);
  const routingScores = scores.map((score, e) => score + expertBias[e]);
  const selection = groupLimitedTopk(routingScores);
  const gathered = selection.topIndices.map((e) => scores[e]); // 注意：gather 的是 scores
  const total = gathered.reduce((a, b) => a + b, 0);
  const probs = CFG2.topk > 1 ? gathered.map((v) => v / (total + 1e-20)) : gathered;
  const routingProbs = new Array(CFG2.numExperts).fill(0);
  const routingMap = new Array(CFG2.numExperts).fill(false);
  selection.topIndices.forEach((e, i) => {
    routingProbs[e] = probs[i];
    routingMap[e] = true;
  });
  return { scores, routingScores, ...selection, gathered, total, probs, routingProbs, routingMap };
}

const zeroBias = new Array(CFG2.numExperts).fill(0);
const routed = CFG2.logits.map((row) => routeToken(row, CFG2.expertBias));
const counterfactual = routeToken(CFG2.logits[0], zeroBias);
const t0 = routed[0];

// routing_map.sum(dim=0)：本步累进到 local_tokens_per_expert 的每专家 token 数。
const tokensPerExpert = Array.from(
  { length: CFG2.numExperts },
  (_, e) => routed.reduce((sum, r) => sum + (r.routingMap[e] ? 1 : 0), 0),
);

if (tokensPerExpert.reduce((a, b) => a + b, 0) !== CFG2.topk * routed.length) {
  throw new Error('routing_map column sums disagree with topk × num_tokens');
}

const esc = (value) => String(value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

const f3 = (value) => (Number.isFinite(value) ? value.toFixed(3) : '−inf');

// 与 tools/figs/svg/megatron_tp_figures.mjs 共用同一组 class 名与色值，不另起调色板。
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
  .ti{font-size:18px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .pt{font-size:14px;font-weight:700;fill:#2A313B}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#68717D}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .rank{font-size:11px;font-weight:700;fill:#5B6470}
  .cap{font-size:11px;fill:#747C88}
  .cell0{fill:#DCE9FB;stroke:#fff;stroke-width:.8}
  .cell1{fill:#9CC2F3;stroke:#fff;stroke-width:.8}
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

function arrow(x1, y1, x2, y2, cls = 'main') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}

function elbow(points, cls = 'aux') {
  const d = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x} ${y}`).join(' ');
  return `<path class="${cls}" d="${d}"/>`;
}

function box(lines, x, y, w, h, cls = 'neutral') {
  const out = [rect(x, y, w, h, cls)];
  const gap = 16;
  const start = y + h / 2 - ((lines.length - 1) * gap) / 2 + 4;
  lines.forEach((line, index) => out.push(text(x + w / 2, start + index * gap, line, index === 0 ? 'tx' : 'sm', 'middle')));
  return out.join('\n');
}

function chip(x, y, w, h, label, cls) {
  return `${rect(x, y, w, h, cls, 4)}\n${text(x + w / 2, y + h / 2 + 3.5, label, 'sm', 'middle')}`;
}

// ---------------------------------------------------------------------------
// 图 1
// ---------------------------------------------------------------------------

const SLOT_ROW_H = Object.freeze({ plain: 34, attn: 118, mlp: 94 });
const SLOT_GAP = 5;

function slotRowHeight(slot) {
  if (slot.nested === 'attn') return SLOT_ROW_H.attn;
  if (slot.nested === 'mlp') return SLOT_ROW_H.mlp;
  return SLOT_ROW_H.plain;
}

function slotRowOffsets() {
  const offsets = [];
  let cursor = 0;
  for (const slot of SLOTS) {
    offsets.push(cursor);
    cursor += slotRowHeight(slot) + SLOT_GAP;
  }
  return { offsets, total: cursor - SLOT_GAP };
}

function renderNestedRows(lane, slot, bx, by, bw) {
  const out = [];
  const nested = NESTED[slot.nested];
  out.push(text(bx + 14, by + 19, `${nested.table}（嵌套第二张插槽表）`, 'sm'));
  nested.rows.forEach((row, index) => {
    const ry = by + 28 + index * 25;
    out.push(text(bx + 14, ry + 13, row.slot, 'sm'));
    let cx = bx + 118;
    const fused = lane.fuse && row.teFused;
    if (fused) {
      out.push(chip(cx, ry, 30, 18, 'LN', 'acc1'));
      cx += 33;
    }
    out.push(chip(cx, ry, 42, 18, 'GEMM', 'neutral'));
    cx += 50;
    out.push(text(cx, ry + 13, lane.fuse ? row.te : row.local, 'dim'));
  });
  if (lane.fuse) {
    out.push(text(bx + bw - 14, by + 19, 'norm 已在框内', 'dim', 'end'));
  }
  return out.join('\n');
}

function renderLane(lane, x, y, w, h, rows) {
  const out = [rect(x, y, w, h, 'panel', 12)];
  out.push(text(x + 20, y + 30, lane.title, 'pt'));

  const cx = x + w / 2;
  out.push(box(
    ['config', `transformer_impl = "${lane.impl}"`, 'num_experts=None · qk_layernorm=False'],
    x + 20, y + 46, w - 40, 58, 'neutral',
  ));
  out.push(arrow(cx, y + 104, cx, y + 118));
  out.push(box(['get_backend(transformer_impl)'], x + 20, y + 118, w - 40, 34, 'neutral'));
  out.push(arrow(cx, y + 152, cx, y + 166));
  out.push(box(
    [lane.provider, `fuse_layernorm_and_linear() → ${lane.fuse ? 'True' : 'False'}`, lane.fc1Line],
    x + 20, y + 166, w - 40, 66, lane.fuse ? 'acc1' : 'neutral',
  ));
  out.push(arrow(cx, y + 232, cx, y + 246));

  out.push(text(x + 20, y + 264, 'TransformerLayerSubmodules —— 12 个模块槽 + 1 张 sharded_state_dict_keys_map', 'tx'));
  out.push(rect(x + w - 152, y + 250, 132, 22, lane.fuse ? 'acc1' : 'neutral', 6));
  out.push(text(x + w - 86, y + 265, `非哨兵槽 ${filledCount[lane.key]} / 12`, 'sm', 'middle'));

  const rowTop = y + 282;
  const nameX = x + 18;
  const boxX = x + 186;
  const boxW = w - 186 - 20 - 18 - 8;

  SLOTS.forEach((slot, index) => {
    const ry = rowTop + rows.offsets[index];
    const rh = slotRowHeight(slot);
    const filled = slot[lane.key];
    out.push(text(nameX, ry + (slot.nested ? 20 : rh / 2 + 4), slot.name, filled ? 'tx' : 'sm'));
    if (filled === null) {
      out.push(rect(boxX, ry, boxW, rh, 'ghost'));
      const fused = lane.fuse && slot.fusedInto;
      const label = fused
        ? `${slot.sentinel}（哨兵）—— norm 已融进 ${slot.fusedInto}`
        : `${slot.sentinel}（哨兵：此槽未填）`;
      out.push(text(boxX + 14, ry + rh / 2 + 4, label, fused ? 'dim' : 'sm'));
    } else {
      out.push(rect(boxX, ry, boxW, rh, 'acc1'));
      out.push(text(boxX + 14, ry + (slot.nested ? 16 : rh / 2 + 4), filled, slot.nested ? 'tx' : 'dim'));
      if (slot.nested) out.push(renderNestedRows(lane, slot, boxX, ry + 14, boxW));
    }
    if (slot.dead) {
      out.push(rect(x + w - 32, ry + 4, 10, rh - 8, 'acc2', 3));
    }
  });

  return { svg: out.join('\n'), rowTop, boxX, boxW };
}

function renderSlotTopologyFigure() {
  const rows = slotRowOffsets();
  const laneW = 708;
  const laneY = 84;
  const laneH = 282 + rows.total + 22;
  const W = 1480;
  const bandY = laneY + laneH + 20;
  const bandH = 62 + RENAME_MAP.length * 66 + 4;
  const costY = bandY + bandH + 16;
  const costH = 56;
  const H = costY + costH + 52;

  const p = [];
  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="local 与 transformer engine 两份 spec 装出的插槽表拓扑对照及其共同的 checkpoint 键">`);
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, '同一个 TransformerLayer 类，两份后端 spec：改变的是槽的拓扑，不只是算子', 'ti'));
  p.push(text(28, 57, `dense · 非 MLA · 无 hyper-connection：非哨兵槽 local=${filledCount.local}、transformer_engine=${filledCount.te}，差的正是被融进后继 linear 的两个 norm 槽`, 'su'));

  const lanes = [
    renderLane(LANES[0], 20, laneY, laneW, laneH, rows),
    renderLane(LANES[1], 20 + laneW + 24, laneY, laneW, laneH, rows),
  ];
  p.push(lanes[0].svg, lanes[1].svg);

  // checkpoint 不变量带
  p.push(rect(20, bandY, W - 40, bandH, 'panel', 12));
  p.push(text(40, bandY + 28, '③ 治理不变量：两条泳道模块树不同形，却写出同一份 checkpoint 键', 'pt'));
  p.push(text(40, bandY + 47, 'TransformerLayerSubmodules 的第 13 个字段 sharded_state_dict_keys_map 在 TransformerLayer.sharded_state_dict 里经 apply_prefix_mapping 就地改键', 'su'));

  RENAME_MAP.forEach(([from, to], index) => {
    const ry = bandY + 62 + index * 66;
    p.push(rect(40, ry, 360, 56, 'neutral', 6));
    p.push(text(54, ry + 20, 'local 原始键（模块树路径）', 'sm'));
    p.push(text(54, ry + 40, LANES[0].rawKeys[index], 'dim'));
    p.push(arrow(400, ry + 28, 468, ry + 28, 'aux'));
    p.push(text(434, ry + 21, '重命名', 'sm', 'middle'));
    p.push(rect(470, ry, 540, 56, 'acc1', 6));
    p.push(text(484, ry + 18, '规范键（写盘即规范名）', 'sm'));
    p.push(text(484, ry + 34, CANONICAL_KEYS[index], 'dim'));
    p.push(text(484, ry + 50, `map: "${from}" → "${to}"`, 'sm'));
    p.push(arrow(1078, ry + 28, 1012, ry + 28, 'aux'));
    p.push(text(1045, ry + 21, '无需映射', 'sm', 'middle'));
    p.push(rect(1080, ry, 360, 56, 'neutral', 6));
    p.push(text(1094, ry + 20, 'transformer_engine 原生键', 'sm'));
    p.push(text(1094, ry + 40, LANES[1].rawKeys[index], 'dim'));
  });

  // 代价带
  p.push(rect(20, costY, W - 40, costH, 'acc2', 10));
  p.push(text(40, costY + 23, `代价：${deadSlots.length} 个恒空槽被无条件调用`, 'costtx'));
  p.push(text(40, costY + 42, `decoder-only 下 ${deadSlots.join(' / ')} 在两条泳道上都是哨兵（右侧橙条），_forward_attention 仍每层每次前向调用它们`, 'sm'));

  p.push(text(28, H - 26, '蓝色：本图机制（被填的槽、被融进 linear 的 norm、规范键）；橙色：代价；灰底：哨兵与不参与本图立论的部分。', 'cap'));
  p.push(text(28, H - 8, '两条泳道的 12 行按 dataclass 声明顺序逐行对齐——同一行的差异就是「一个布尔量改变了槽的拓扑」。', 'cap'));
  p.push('</svg>');
  return p.join('\n');
}

// ---------------------------------------------------------------------------
// 图 2
// ---------------------------------------------------------------------------

const CELL_W = 86;
const CELL_H = 34;

function expertStrip(x, y, values, opts = {}) {
  const { highlight = [], ghost = [], formatter = f3 } = opts;
  const out = [];
  values.forEach((value, e) => {
    const cx = x + e * (CELL_W + 4);
    let cls = 'neutral';
    if (ghost.includes(e)) cls = 'ghost';
    if (highlight.includes(e)) cls = 'acc1';
    out.push(rect(cx, y, CELL_W, CELL_H, cls, 5));
    out.push(text(cx + CELL_W / 2, y + 15, `e${e}`, 'sm', 'middle'));
    out.push(text(cx + CELL_W / 2, y + 28, formatter(value), highlight.includes(e) ? 'dim' : 'sm', 'middle'));
  });
  return out.join('\n');
}

const stripWidth = CFG2.numExperts * (CELL_W + 4) - 4;

function groupBoxes(x, y, groupScores, selected) {
  const out = [];
  const gw = expertsPerGroup * (CELL_W + 4) - 4;
  groupScores.forEach((score, g) => {
    const gx = x + g * (CELL_W + 4) * expertsPerGroup;
    const on = selected.includes(g);
    out.push(rect(gx, y, gw, 38, on ? 'acc1' : 'ghost', 5));
    out.push(text(gx + gw / 2, y + 16, `group ${g} = {e${g * expertsPerGroup}, e${g * expertsPerGroup + 1}}`, 'sm', 'middle'));
    out.push(text(gx + gw / 2, y + 31, `组分 ${f3(score)}${on ? '  ✓选中' : ''}`, on ? 'dim' : 'sm', 'middle'));
  });
  return out.join('\n');
}

function renderRouterFigure() {
  const W = 1480;
  const p = [];
  const sx = 148;
  const noteX = 896;
  const noteW = 1440 - noteX;

  const p1y = 84;
  const p1h = 176;
  const p2y = p1y + p1h + 16;
  const p2h = 370;
  const wy = p2y + p2h + 16;
  const ph = 340;
  const costY = wy + ph + 16;
  const costH = 62;
  const H = costY + costH + 90;

  const cfGroups = counterfactual.selectedGroups.join(', ');
  const biasGroups = t0.selectedGroups.join(', ');

  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="MoE router 把一个 token 的 logits 经打分 专家偏置 分组 top-k 变成 probs 与 routing map">`);
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, 'MoE router：一个 token 从 logits 到 (probs, routing_map)', 'ti'));
  p.push(text(28, 57, `示例：num_moe_experts=${CFG2.numExperts} · moe_router_topk=${CFG2.topk} · moe_router_num_groups=${CFG2.numGroups} · moe_router_group_topk=${CFG2.groupTopk} · score_function=${CFG2.scoreFunction} · 开启 moe_router_enable_expert_bias`, 'su'));

  // ① 打分
  p.push(rect(20, p1y, W - 40, p1h, 'panel', 12));
  p.push(text(40, p1y + 28, '① 打分：Router.gating 的线性投影结果过 score function', 'pt'));
  p.push(text(40, p1y + 62, 'logits(t0)', 'rank'));
  p.push(expertStrip(sx, p1y + 46, CFG2.logits[0]));
  p.push(arrow(sx + stripWidth / 2, p1y + 80, sx + stripWidth / 2, p1y + 102));
  p.push(text(sx + stripWidth / 2 + 12, p1y + 96, 'scores = sigmoid(logits.float())', 'sm'));
  p.push(text(40, p1y + 124, 'scores', 'rank'));
  p.push(expertStrip(sx, p1y + 108, t0.scores));
  p.push(rect(noteX, p1y + 46, noteW, 96, 'neutral', 8));
  p.push(text(noteX + 16, p1y + 68, 'Router.gating', 'tx'));
  p.push(text(noteX + 16, p1y + 88, 'weight 是一个 [num_moe_experts, hidden_size] 的 fp32 参数', 'sm'));
  p.push(text(noteX + 16, p1y + 106, 'moe_router_dtype 可把这一段计算抬到 fp32 / fp64', 'sm'));
  p.push(text(noteX + 16, p1y + 124, `logits 进 routing 后 view 成 [num_tokens, ${CFG2.numExperts}]；本例 num_tokens=${CFG2.logits.length}`, 'sm'));

  // ② 选择
  p.push(rect(20, p2y, W - 40, p2h, 'panel', 12));
  p.push(text(40, p2y + 28, '② 选择：expert bias 只在这一段生效', 'pt'));
  p.push(text(40, p2y + 62, 'expert_bias', 'rank'));
  p.push(expertStrip(sx, p2y + 46, CFG2.expertBias, {
    highlight: CFG2.expertBias.map((v, e) => (v !== 0 ? e : -1)).filter((e) => e >= 0),
    formatter: (v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2)),
  }));
  p.push(arrow(sx + stripWidth / 2, p2y + 80, sx + stripWidth / 2, p2y + 102));
  p.push(text(sx + stripWidth / 2 + 12, p2y + 96, 'scores_for_routing = scores + expert_bias.float()', 'sm'));
  p.push(text(40, p2y + 124, 'scores_for_routing', 'rank'));
  p.push(expertStrip(sx, p2y + 108, t0.routingScores, { highlight: t0.topIndices }));
  p.push(arrow(sx + stripWidth / 2, p2y + 142, sx + stripWidth / 2, p2y + 164));
  p.push(text(sx + stripWidth / 2 + 12, p2y + 158, `组分 = 每组 top-${scorePerGroup} 之和；取分数最高的 group_topk=${CFG2.groupTopk} 组`, 'sm'));
  p.push(text(40, p2y + 190, 'group_limited_topk', 'rank'));
  p.push(groupBoxes(sx, p2y + 170, t0.groupScores, t0.selectedGroups));
  p.push(arrow(sx + stripWidth / 2, p2y + 208, sx + stripWidth / 2, p2y + 230));
  p.push(text(sx + stripWidth / 2 + 12, p2y + 224, '未选中组 masked_fill(-inf)，再在候选里 torch.topk', 'sm'));
  p.push(text(40, p2y + 254, 'masked', 'rank'));
  p.push(expertStrip(sx, p2y + 238, t0.masked, {
    highlight: t0.topIndices,
    ghost: t0.masked.map((v, e) => (Number.isFinite(v) ? -1 : e)).filter((e) => e >= 0),
  }));
  p.push(rect(sx, p2y + 288, stripWidth, 66, 'acc1', 8));
  p.push(text(sx + 16, p2y + 310, `top_indices = [${t0.topIndices.join(', ')}]`, 'tx'));
  p.push(text(sx + 16, p2y + 329, `即专家 ${t0.topIndices.map((e) => `e${e}`).join(' 与 ')}（torch.topk 按分数降序返回）`, 'sm'));
  p.push(text(sx + 16, p2y + 346, '② 到此结束：它只回答了「选了谁」，还没有回答「权重是多少」——那是 ③ 的事', 'sm'));

  // ② 的反事实对照泳道
  const cfY = p2y + 46;
  p.push(rect(noteX, cfY, noteW, 308, 'ghost', 10));
  p.push(text(noteX + 16, cfY + 26, '反事实对照：同一个 t0，expert_bias 全零', 'rank'));
  p.push(text(noteX + 16, cfY + 52, `组分排序随之改变 → 选中 group ${cfGroups}（而不是 ${biasGroups}）`, 'sm'));
  p.push(text(noteX + 16, cfY + 72, `top_indices = [${counterfactual.topIndices.join(', ')}]，即 ${counterfactual.topIndices.map((e) => `e${e}`).join(' 与 ')}`, 'sm'));
  p.push(text(noteX + 16, cfY + 104, `e${t0.topIndices[0]} 因 +${CFG2.expertBias[t0.topIndices[0]].toFixed(2)} 从落选升为首选`, 'dim'));
  p.push(text(noteX + 16, cfY + 124, `e2 因 ${CFG2.expertBias[2].toFixed(2)} 连同整个 group 1 被挤出候选`, 'dim'));
  p.push(text(noteX + 16, cfY + 156, '偏置同时改变了「选哪一组」和「选哪个专家」——', 'sm'));
  p.push(text(noteX + 16, cfY + 176, '它是 aux-loss-free 负载均衡的唯一杠杆。', 'sm'));
  p.push(text(noteX + 16, cfY + 208, '但它本身在 torch.no_grad 下更新，不进 loss，', 'sm'));
  p.push(text(noteX + 16, cfY + 228, '因此无法靠反向传播自我修正——只能靠 ④ 的计数回路。', 'sm'));
  p.push(text(noteX + 16, cfY + 260, '灰底 = 本图不走的路径；对照只为证明偏置真的在改变选择。', 'cap'));

  // ③ 加权
  p.push(rect(20, wy, 716, ph, 'panel', 12));
  p.push(text(40, wy + 28, '③ 加权：gather 回到没有 bias 的 scores', 'pt'));
  p.push(rect(40, wy + 48, 320, 58, 'acc1', 8));
  p.push(text(56, wy + 70, 'scores（无 bias）', 'tx'));
  p.push(text(56, wy + 92, t0.topIndices.map((e) => `e${e}: ${f3(t0.scores[e])}`).join('    '), 'dim'));
  p.push(rect(392, wy + 48, 324, 58, 'ghost', 8));
  p.push(text(408, wy + 70, 'scores_for_routing（含 bias）', 'tx'));
  p.push(text(408, wy + 92, t0.topIndices.map((e) => `e${e}: ${f3(t0.routingScores[e])}`).join('    '), 'sm'));
  p.push(arrow(200, wy + 106, 200, wy + 142));
  p.push(text(392, wy + 128, 'gather 不读这一行 —— 否则 bias 会进权重', 'costtx'));
  p.push(rect(40, wy + 142, 676, 52, 'acc1', 8));
  p.push(text(56, wy + 164, 'torch.gather(scores, 1, top_indices)', 'tx'));
  p.push(text(56, wy + 184, `= [${t0.gathered.map(f3).join(', ')}]`, 'dim'));
  p.push(arrow(378, wy + 194, 378, wy + 214));
  p.push(rect(40, wy + 214, 676, 56, 'neutral', 8));
  p.push(text(56, wy + 236, `probs = scores / (scores.sum(dim=-1) + 1e-20) = [${t0.probs.map(f3).join(', ')}]`, 'tx'));
  p.push(text(56, wy + 256, `分母 ${f3(t0.total)}；topk=${CFG2.topk} > 1 才做这一步归一化`, 'sm'));
  p.push(text(40, wy + 298, `不变量：e${t0.topIndices[0]} 的权重取 ${f3(t0.scores[t0.topIndices[0]])} 而非 ${f3(t0.routingScores[t0.topIndices[0]])}——偏置进选择、出权重。`, 'dim'));
  p.push(text(40, wy + 318, '因此负载均衡改变的是 token 的去向，不改变被选中专家在前向里的加权函数。', 'sm'));

  // ④ 输出与回路
  const qx = 752;
  const qw = 708;
  p.push(rect(qx, wy, qw, ph, 'panel', 12));
  p.push(text(qx + 20, wy + 28, '④ 输出：scatter 成两个稠密 [num_tokens, num_experts]', 'pt'));
  p.push(text(qx + 20, wy + 47, '着色 = routing_map（bool）；格内数值 = routing_probs 的非零项', 'su'));
  const gx = qx + 104;
  const gcw = 66;
  const gstep = gcw + 3;
  for (let e = 0; e < CFG2.numExperts; e += 1) {
    p.push(text(gx + e * gstep + gcw / 2, wy + 74, `e${e}`, 'sm', 'middle'));
  }
  routed.forEach((r, tIndex) => {
    const ry = wy + 82 + tIndex * 30;
    p.push(text(qx + 20, ry + 17, `t${tIndex}`, tIndex === 0 ? 'rank' : 'sm'));
    for (let e = 0; e < CFG2.numExperts; e += 1) {
      const on = r.routingMap[e];
      p.push(rect(gx + e * gstep, ry, gcw, 26, on ? 'acc1' : 'ghost', 4));
      p.push(text(gx + e * gstep + gcw / 2, ry + 17, on ? f3(r.routingProbs[e]) : '·', on ? 'dim' : 'sm', 'middle'));
    }
  });
  const sumY = wy + 82 + routed.length * 30 + 6;
  p.push(text(qx + 20, sumY + 17, 'Σ dim=0', 'sm'));
  for (let e = 0; e < CFG2.numExperts; e += 1) {
    p.push(rect(gx + e * gstep, sumY, gcw, 26, 'neutral', 4));
    p.push(text(gx + e * gstep + gcw / 2, sumY + 17, String(tokensPerExpert[e]), 'sm', 'middle'));
  }
  p.push(text(qx + 20, sumY + 52, 'local_tokens_per_expert += routing_map.sum(dim=0)   （torch.no_grad）', 'sm'));
  p.push(arrow(qx + qw / 2, sumY + 58, qx + qw / 2, sumY + 74, 'aux'));
  p.push(rect(qx + 20, sumY + 74, qw - 40, 28, 'ghost', 6));
  p.push(text(qx + 36, sumY + 93, '下一步据此更新 expert_bias → 回到 ②：全图唯一的反馈边', 'sm'));
  p.push(rect(qx + 20, sumY + 112, qw - 40, 50, 'neutral', 8));
  p.push(text(qx + 36, sumY + 134, '交接：(probs, routing_map)', 'tx'));
  p.push(text(qx + 36, sumY + 153, 'dispatch / expert / combine 属于 EP 专题页，本图到此为止', 'sm'));

  // 代价
  p.push(rect(20, costY, W - 40, costH, 'acc2', 10));
  p.push(text(40, costY + 24, '代价', 'costtx'));
  p.push(text(88, costY + 24, `两个输出都是稠密 [num_tokens, ${CFG2.numExperts}]：每 token 只有 ${CFG2.topk} 个非零，其余 ${CFG2.numExperts - CFG2.topk} 格是为下游 dispatcher 的固定形状付的显存与带宽。`, 'sm'));
  p.push(text(88, costY + 44, 'expert_bias 的更新是训练期副作用：它不进 loss，也因此不能靠反向传播自我修正，只能靠上面那条计数回路。', 'sm'));

  p.push(text(28, H - 62, 'group-limited routing 的意义：被 -inf 掉的不是「差专家」，而是「本 token 这一步不去访问的那些组」——组数常取为 EP 组数或节点数，用于压住跨节点扇出。', 'cap'));
  p.push(text(28, H - 42, '蓝色：本图机制（偏置、被选中的组与专家、无偏置的权重来源）；橙色：代价；灰底：反事实对照与未选中项。', 'cap'));
  p.push(text(28, H - 22, '所有数值由 CFG2 的 logits 与 expert_bias 现算：改一个 logit，组分、top_indices、probs 与列和一起重算。', 'cap'));
  p.push('</svg>');
  return p.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
mkdirSync(outputDir, { recursive: true });

const outputs = new Map([
  ['megatron_model_spec_slot_topology.svg', renderSlotTopologyFigure()],
  ['megatron_moe_router_routing.svg', renderRouterFigure()],
]);

for (const [name, svg] of outputs) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
console.log([...outputs.keys()].join('\n'));
