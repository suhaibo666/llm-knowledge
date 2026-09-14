// 图：一个已生成 token 的 logprob 怎样在训推两侧分叉、又被 slime 的重放与校正逐层收拢：
// ① top-p 支持集重放（含 TP 词表分片与工具 token 的空 nucleus）；② MoE 路由重放的 stage 与 cursor，
// 以及同一组 route 梯度按列顺序逐次 BF16 累加的差别；③ 同一组 train-old/rollout 比值走 vanilla TIS、
// ICEPOP 与示例 MIS 的权重与 mask。
// 源码基线：THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：权重相同也会因归一化域不同产生 δ；keep mask 把支持集外置 −inf 并写回目标 logit，
// 训练侧因此在 rollout 的 nucleus ∪ {y} 上重算；R3 的记录被 old-logprob 前向与训练前向各消费一遍，
// backward cursor 只在重算时前进；route 梯度必须按列顺序逐次舍入；校正函数改的是权重或 mask，
// 不是证据。acc1 标重放后的一致结果，acc2 标假差异与被拒绝的 token。
//
// 用法：node tools/figs/svg/slime_train_infer_consistency_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  logits: [2.0, 1.0, 0.0, -1.0], // 同一份权重下该位置的 logits（4 个 token 的词表）
  temperature: 1.0,
  nucleus: [0, 1], // rollout 记录的 top-p nucleus ids
  sampled: 1,
  tpSize: 2, // 训练侧词表 TP 分片
  toolTarget: 3, // 工具 token：loss_mask=0，nucleus 为空
  microBatches: 2,
  routeGrads: [1.0, 2 ** -8, 2 ** -8], // 一个 token 的 3 个 route 梯度，按 top-k 列顺序
  ratios: [0.5, 4.0], // 同一序列两个 token 的 exp(train_old − rollout)
  vanilla: { low: 0, high: 2 }, // --tis-clip-low / --tis-clip 默认值
  icepop: { low: 0.5, high: 2 }, // GLM-5 门禁传入的值
  mis: { low: 0.5, high: 2, veto: 1e-4 }, // examples/train_infer_mismatch_helper/mis.yaml
});

const range = (n) => Array.from({ length: n }, (_, i) => i);
const sum = (xs) => xs.reduce((a, x) => a + x, 0);
const logsumexp = (xs) => {
  const m = Math.max(...xs);
  return m + Math.log(sum(xs.map((x) => Math.exp(x - m))));
};

// ---------------- 对源码算法的最小复现 ----------------

// loss.py::get_log_probs_and_entropy（温度缩放）+ _build_topp_keep_mask +
// ppo_utils.py::_VocabParallelLogProbEntropy.forward（keep mask 置 −inf、目标 logit 写回、
// 词表分片上的 max / sum all-reduce）。
export function vocabParallelLogProb(logits, target, keep, tpSize, temperature = 1.0) {
  const z = logits.map((x) => x / temperature);
  const per = z.length / tpSize;
  const shards = range(tpSize).map((r) => {
    const ids = range(per).map((k) => r * per + k);
    const masked = ids.map((v) => (keep === null || keep.includes(v) || v === target ? z[v] : -Infinity));
    return { ids, masked, keptLocally: masked.some((x) => x > -Infinity) };
  });
  const gmax = Math.max(...shards.flatMap((s) => s.masked));
  const gsum = sum(shards.map((s) => sum(s.masked.map((x) => Math.exp(x - gmax)))));
  return { logProb: z[target] - gmax - Math.log(gsum), shards };
}

export function topPReplay(cfg = CFG) {
  const z = cfg.logits.map((x) => x / cfg.temperature);
  const full = z[cfg.sampled] - logsumexp(z);
  const nucleus = z[cfg.sampled] - logsumexp(cfg.nucleus.map((v) => z[v]));
  const tp = vocabParallelLogProb(cfg.logits, cfg.sampled, cfg.nucleus, cfg.tpSize, cfg.temperature);
  const tool = vocabParallelLogProb(cfg.logits, cfg.toolTarget, [], cfg.tpSize, cfg.temperature);
  return {
    full,
    nucleus,
    phantomDelta: Math.abs(full - nucleus),
    tpLogProb: tp.logProb,
    tpShards: tp.shards.map((s) => ({ ids: s.ids, keptLocally: s.keptLocally })),
    toolLogProb: tool.logProb,
  };
}

// slime/utils/routing_replay.py::RoutingReplay 的 record / pop_forward / pop_backward / clear_forward，
// 按 actor.py::train_actor 与 model.py::train_one_step 设置 ROUTING_REPLAY_STAGE 的顺序推进。
export function replayCursors({ microBatches, r3, recompute }) {
  const st = { recorded: 0, forward: 0, backward: 0 };
  const log = [];
  const snap = (what) => log.push({ what, ...st });
  if (r3) {
    for (let i = 0; i < microBatches; i += 1) st.recorded += 1;
    snap('fill_routing_replay');
  }
  snap('ref 前向 fallthrough');
  if (r3) {
    st.forward += microBatches;
    snap('old 前向 replay_forward');
    st.forward = 0;
    snap('clear_all_forward');
  } else {
    st.recorded += microBatches;
    snap('old 前向 record');
  }
  for (let i = 0; i < microBatches; i += 1) {
    st.forward += 1;
    if (recompute) st.backward += 1;
  }
  snap(recompute ? '训练前向 + 重算' : '训练前向（无重算）');
  return { final: { ...st }, log };
}

function toBf16(x) {
  const u = new Uint32Array(new Float32Array([x]).buffer)[0];
  const r = ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) << 16;
  return new Float32Array(new Uint32Array([r >>> 0]).buffer)[0];
}

// deterministic_route_kernels.py::_scatter_routes_backward_kernel：FP32 累加器每加一个 slot 就舍入到 BF16
export function orderedBf16Sum(values) {
  let acc = 0;
  for (const v of values) acc = toBf16(Math.fround(acc + v));
  return acc;
}
export function fp32SumThenBf16(values) {
  return toBf16(values.reduce((a, v) => Math.fround(a + v), 0));
}

// loss.py::vanilla_tis_function / icepop_function；examples/train_infer_mismatch_helper/mis.py::compute_mis_weights
export function corrections(cfg = CFG) {
  const rho = cfg.ratios;
  const ones = rho.map(() => 1);
  const vanilla = { w: rho.map((r) => Math.min(Math.max(r, cfg.vanilla.low), cfg.vanilla.high)), mask: ones };
  const icepop = { w: rho.map((r) => (r >= cfg.icepop.low && r <= cfg.icepop.high ? r : 0)), mask: ones };
  const { low, high } = cfg.mis;
  const truncate = { w: rho.map((r) => Math.min(r, high)), mask: ones };
  const clip = { w: rho.map((r) => Math.min(Math.max(r, low), high)), mask: ones };
  const maskMode = { w: [...rho], mask: rho.map((r) => (r >= low && r <= high ? 1 : 0)) };
  const logSum = sum(rho.map(Math.log));
  const sequence = { w: rho.map(() => Math.min(Math.exp(logSum), high)), mask: ones };
  const geometric = { w: rho.map(() => Math.exp(logSum / rho.length)), mask: ones };
  // mis.yaml：truncate → RS token（同一 log-ratio，界 [low, high]）→ veto → token 级 batch 归一
  const rsMask = rho.map((r) => (r >= low && r <= high ? 1 : 0));
  const veto = rho.some((r) => r < cfg.mis.veto) ? 0 : 1;
  const mean = sum(truncate.w) / truncate.w.length;
  const yaml = { w: truncate.w.map((w) => w / mean), mask: rsMask.map((m) => m * veto), batchMean: mean };
  return { vanilla, icepop, truncate, clip, maskMode, sequence, geometric, yaml };
}

export function model(cfg = CFG) {
  return {
    topP: topPReplay(cfg),
    r3: replayCursors({ microBatches: cfg.microBatches, r3: true, recompute: true }),
    r3NoRecompute: replayCursors({ microBatches: cfg.microBatches, r3: true, recompute: false }),
    r2: replayCursors({ microBatches: cfg.microBatches, r3: false, recompute: true }),
    accum: {
      columnOrder: orderedBf16Sum(cfg.routeGrads),
      reversed: orderedBf16Sum([...cfg.routeGrads].reverse()),
      fp32Once: fp32SumThenBf16(cfg.routeGrads),
    },
    corr: corrections(cfg),
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const f3 = (x) => (Object.is(x, -0) ? 0 : x).toFixed(3);
const fw = (x) => (Number.isInteger(x) ? String(x) : String(Number(x.toFixed(4))));
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .mono{font-size:11px;font-family:"SFMono-Regular",Menlo,Consolas,monospace;fill:#38414D}
  .cap{font-size:11.5px;fill:#5B6470}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .bar{fill:#AEB6C2}
  .bar1{fill:#2563EB}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
`;

function render(m, cfg = CFG) {
  const W = 1180;
  const H = 1052;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 6) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2) => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="main"/>`);

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime 训推一致性：一个 token 的 logprob 在支持集、路由与校正三处的重放</title>');
  o.push('<desc id="desc">三个面板：top-p 支持集重放把全词表 softmax 的假差异消掉，TP 分片与工具 token 的空 nucleus 仍得到一致结果；路由重放的记录与两个 cursor 如何被 old-logprob 前向、训练前向与重算消费，以及 route 梯度按列顺序逐次 BF16 累加的差别；同一组 train-old/rollout 比值在 vanilla TIS、ICEPOP 与示例 MIS 下得到的权重与 mask。</desc>');
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '同一份权重下，一个 token 的 logprob 在支持集、路由与校正三处怎样分叉、又怎样被重放收拢', 'ti');
  text(24, 56, `词表 4、logits [${cfg.logits.join(', ')}]、温度 ${cfg.temperature}、rollout nucleus {${cfg.nucleus.join(', ')}}、采到 y=${cfg.sampled} · 训练 TP=${cfg.tpSize} · 一个 router、${cfg.microBatches} 个 micro-batch · 比值 ρ=[${cfg.ratios.join(', ')}]`, 'su');

  // ---------- ① top-p ----------
  const t = m.topP;
  const py = 72;
  rect(24, py, 1132, 300, 'panel', 10);
  text(40, py + 24, '① L2 采样支持集：训练侧用 rollout 的 nucleus 重算，否则权重相同也有假差异', 'pt');
  const probsFull = cfg.logits.map((z) => Math.exp(z - logsumexp(cfg.logits)));
  const zS = cfg.nucleus.map((v) => cfg.logits[v]);
  const probsNuc = cfg.logits.map((z, v) => (cfg.nucleus.includes(v) ? Math.exp(z - logsumexp(zS)) : 0));
  const barBase = py + 190;
  text(40, py + 48, '全词表 softmax（不重放）', 'sm');
  text(330, py + 48, '在 nucleus 上归一（keep mask 重放）', 'sm');
  cfg.logits.forEach((z, v) => {
    const h1 = probsFull[v] * 150;
    const h2 = probsNuc[v] * 150;
    const x1 = 50 + v * 60;
    const x2 = 340 + v * 60;
    rect(x1, barBase - h1, 36, h1, v === cfg.sampled ? 'acc2' : 'ghost', 2);
    rect(x2, barBase - h2, 36, Math.max(h2, 0.01), v === cfg.sampled ? 'acc1' : cfg.nucleus.includes(v) ? 'neutral' : 'ghost', 2);
    text(x1 + 18, barBase + 16, `v=${v}`, 'sm', 'middle');
    text(x2 + 18, barBase + 16, `v=${v}`, 'sm', 'middle');
    text(x1 + 18, barBase - h1 - 6, probsFull[v].toFixed(2), 'sm', 'middle');
    text(x2 + 18, barBase - Math.max(h2, 0) - 6, probsNuc[v].toFixed(2), 'sm', 'middle');
  });
  text(40, barBase + 40, `log p(y=${cfg.sampled}) = ${f3(t.full)}`, 'tx');
  text(330, barBase + 40, `log q(y=${cfg.sampled}) = ${f3(t.nucleus)}`, 'tx');
  rect(40, barBase + 50, 250, 24, 'acc2', 4);
  text(50, barBase + 66, `假差异 δ = ${f3(t.phantomDelta)}（与 rollout 比）`, 'sm');
  rect(330, barBase + 50, 250, 24, 'acc1', 4);
  text(340, barBase + 66, 'δ = 0：与 rollout 的行为分布同域', 'sm');
  arrow(292, py + 120, 326, py + 120);

  const rx = 620;
  rect(rx, py + 40, 522, 118, 'neutral', 8);
  text(rx + 12, py + 60, `训练 TP=${cfg.tpSize}：每个 rank 只持有一段词表，keep mask 按本地段切`, 'tx');
  t.tpShards.forEach((s, r) => {
    const y = py + 72 + r * 30;
    rect(rx + 12, y, 498, 24, s.keptLocally ? 'cell' : 'ghost', 4);
    text(rx + 22, y + 16, `rank ${r} 持有 {${s.ids.join(', ')}}：${s.keptLocally ? '本地有保留项' : '本地整行被置 −inf（目标不在本段）'}`, 'sm');
  });
  text(rx + 12, py + 148, `max 与 sum 在 TP 组 all-reduce 后仍得 ${f3(t.tpLogProb)}`, 'sm');
  rect(rx, py + 168, 522, 96, 'neutral', 8);
  text(rx + 12, py + 188, `工具 token（loss_mask=0，nucleus 为空，目标 v=${cfg.toolTarget}）`, 'tx');
  text(rx + 12, py + 208, '空 span 让整行 keep=False；kernel 在 masked_fill 之后把目标 logit 写回，', 'sm');
  text(rx + 12, py + 224, `所以支持集只剩目标本身：训练 logprob = ${f3(t.toolLogProb)}，rollout 侧占位也是 0.0`, 'sm');
  text(rx + 12, py + 240, '实际重放的支持集是 S ∪ {y}；entropy 仍用未屏蔽的 logits', 'sm');
  text(rx + 12, py + 256, 'top-k 没有对应的 ids 字段，不在重放范围内', 'sm');

  // ---------- ② routing replay ----------
  const ry = 386;
  rect(24, ry, 1132, 318, 'panel', 10);
  text(40, ry + 24, '② L3 路由：记录被 old-logprob 前向与训练前向各消费一遍，后向 cursor 靠重算推进（本页推断）', 'pt');
  const rows = [
    ['R3（--use-rollout-routing-replay，开重算）', m.r3],
    ['R3，不开重算', m.r3NoRecompute],
    ['R2（只开 --use-routing-replay）', m.r2],
  ];
  rows.forEach(([lab, res], i) => {
    const y = ry + 40 + i * 70;
    text(40, y + 14, lab, 'tx');
    res.log.forEach((e, j) => {
      const x = 40 + j * 150;
      rect(x, y + 22, 140, 38, j === res.log.length - 1 ? 'acc1' : 'neutral', 5);
      text(x + 70, y + 37, e.what, 'sm', 'middle');
      text(x + 70, y + 53, `记录 ${e.recorded} · 前 ${e.forward} · 后 ${e.backward}`, 'mono', 'middle');
    });
  });
  text(40, ry + 262, `结束：R3 开重算为 前 ${m.r3.final.forward}/后 ${m.r3.final.backward}/记录 ${m.r3.final.recorded}；不开重算时后 cursor 停在 ${m.r3NoRecompute.final.backward}；`, 'sm');
  text(40, ry + 278, 'assert_all_consumed 要求两者都等于记录数，但冻结基线没有调用它；训练后 clear_all 清空记录', 'sm');
  text(40, ry + 294, '强制 id 仍从当前 scores gather 概率：离散路径固定，router 概率与梯度照常计算', 'sm');

  const ax = 800;
  rect(ax, ry + 40, 342, 262, 'neutral', 8);
  text(ax + 12, ry + 60, '反向平面：训练侧确定性 kernel 逐 slot 累加', 'tx');
  text(ax + 12, ry + 80, 'FP32 累加器每加一个 slot 舍入到 BF16（对照树内参考）', 'sm');
  const g = cfg.routeGrads;
  const lab = (xs) => xs.map((x) => (x === 1 ? '1' : '2⁻⁸')).join(', ');
  rect(ax + 12, ry + 92, 318, 40, 'acc1', 5);
  text(ax + 22, ry + 108, `列顺序 [${lab(g)}]`, 'sm');
  text(ax + 22, ry + 124, `1 + 2⁻⁸ 恰为半个 ULP，舍回 1 → 结果 ${String(m.accum.columnOrder)}`, 'sm');
  rect(ax + 12, ry + 140, 318, 40, 'acc2', 5);
  text(ax + 22, ry + 156, `反序 [${lab([...g].reverse())}]`, 'sm');
  text(ax + 22, ry + 172, `2⁻⁸ + 2⁻⁸ = 2⁻⁷ 先成形 → 结果 ${String(m.accum.reversed)}`, 'sm');
  rect(ax + 12, ry + 188, 318, 40, 'acc2', 5);
  text(ax + 22, ry + 204, '一次 FP32 求和再转 BF16', 'sm');
  text(ax + 22, ry + 220, `结果 ${String(m.accum.fp32Once)}：不能替代逐 slot 舍入`, 'sm');
  text(ax + 12, ry + 248, '前向另算：对齐桥让 Megatron 沿用 SGLang 无序', 'sm');
  text(ax + 12, ry + 264, 'top-k 的列顺序，owner 归约由 SGLang ep_gather', 'sm');
  text(ax + 12, ry + 280, '按 slot 顺序在 FP32 完成（依赖侧，本机未复核）', 'sm');

  // ---------- ③ corrections ----------
  const cy = 718;
  const c = m.corr;
  rect(24, cy, 1132, 282, 'panel', 10);
  text(40, cy + 24, `③ 校正：同一组 ρ=[${cfg.ratios.join(', ')}] 走不同函数，改的是权重或 mask，不是差异本身`, 'pt');
  const crow = [
    ['vanilla TIS（默认 hook）', `clamp 到 [${cfg.vanilla.low}, ${cfg.vanilla.high}]`, c.vanilla],
    ['ICEPOP（GLM-5 门禁）', `区间 [${cfg.icepop.low}, ${cfg.icepop.high}] 外置 0`, c.icepop],
    ['MIS truncate', `只截上界 ${cfg.mis.high}`, c.truncate],
    ['MIS clip', `截到 [${cfg.mis.low}, ${cfg.mis.high}]`, c.clip],
    ['MIS mask', '权重不变，区间外 mask 置 0', c.maskMode],
    ['MIS sequence', 'log 比之和 ln 2 → 整序列同权', c.sequence],
    ['MIS geometric', 'log 比均值 → 整序列同权', c.geometric],
    ['mis.yaml 默认', `truncate → RS [${cfg.mis.low}, ${cfg.mis.high}] → veto ${cfg.mis.veto} → 归一（均值 ${fw(c.yaml.batchMean)}）`, c.yaml],
  ];
  text(40, cy + 48, '函数', 'sm');
  text(260, cy + 48, '规则', 'sm');
  text(740, cy + 48, '权重', 'sm');
  text(900, cy + 48, 'mask', 'sm');
  crow.forEach(([a, b, r], i) => {
    const y = cy + 56 + i * 24;
    const rejected = r.mask.some((x) => x === 0);
    rect(36, y, 1106, 22, rejected ? 'acc2' : i % 2 ? 'ghost' : 'cell', 3);
    text(44, y + 15, a, 'sm');
    text(260, y + 15, b, 'sm');
    text(740, y + 15, `[${r.w.map(fw).join(', ')}]`, 'mono');
    text(900, y + 15, `[${r.mask.join(', ')}]`, 'mono');
  });
  text(40, cy + 270, '被拒绝的 token 只从分子删除，分母仍是原始 rollout_mask_sums（归约口径见 15 页）；use_tis 关闭时 MIS 只算指标、不产生权重', 'sm');

  text(24, 1022, '阅读顺序：① 先对齐归一化域 → ② 再固定离散路由与累加顺序 → ③ 前面都定位清楚之后，才用校正函数改估计量。', 'cap');
  text(24, 1042, '源码基线：THUDM/slime@681b3adca541 · 复现 _build_topp_keep_mask / _VocabParallelLogProbEntropy / RoutingReplay / _scatter_routes_backward_kernel / TIS 与 MIS 函数', 'su');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_train_infer_replay.svg'), `${render(model())}\n`, 'utf8');
}
