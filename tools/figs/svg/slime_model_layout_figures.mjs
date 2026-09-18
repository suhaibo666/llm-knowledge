// 图：slime 模型架构扩展里四个决定性布局规则，用同一个玩具层（hidden 8、4 个 query head / 2 个 KV group、
// head_dim 2、FFN 4、4 个 expert、4 层）与一个 packed batch（A 8 个 token、B 4 个 token）重放：
//   slime_model_layout_tp_fusion.svg  ① GLU linear_fc1 的 TP2 导入切分与导出重排、② 同一规则套到非 GLU fc1
//                                     时的静默错配、③ Qwen3-Next 门控 QKV 的 ×2 与按 KV group 交错
//   slime_model_layout_ep_cp.svg      ④ PP/EP 本地编号 → 全局名、⑤ HF 黑盒 GDN 前后的 CP 两段重组、
//                                     ⑥ 反向：SP 与 CP 两条 all-gather 的梯度账（前缀和分析模型）
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；
// Megatron 侧（依赖）：NVIDIA/Megatron-LM@1dcf0dafa884ad52ffb243625717a3471643e087（slime docker/Dockerfile 钉住）。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：导入 hf_to_megatron/common.py::_tensor_parallel_shard 与导出
// update_weight/common.py::all_gather_params_async 都只按参数名含 linear_fc1 套用 GLU 规则，不看
// partition_stride；GLU 下两者互逆且与 Megatron 的列并行一致，非 GLU 下往返恒等却让 fc1 行与 fc2 列错配。
// qwen3_next_hf_tensor 把每头 [q, gate] 交错的 q_proj 重排成按 KV group 连续的 [q…, gate…, k, v]，
// 于是 TP 按行均分时每个 rank 恰好拿整组。named_params_and_buffers 给 decoder 层号加 PP offset、
// expert 下标加 EP offset（MTP 层号不加）。hf_attention.py::HuggingfaceAttention.forward 在 CP>1 时
// 先 all-gather，再按“前段按 rank 顺序、后段按 rank 逆序”拼回原序列，GDN 之后按 chunk(2·CP) 取回两段。
// 反向面板用“输出 q 依赖同序列 p≤q 的输入”的前缀和代替 GDN（分析模型），对比取本片与 reduce-scatter。
// acc1 = 成立的规则，acc2 = 错误做法、失败边界与分析判断；每格只放短标签，解释在正文。
//
// 用法：node tools/figs/svg/slime_model_layout_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  hidden: 8,
  heads: 4,
  groups: 2,
  headDim: 2,
  ffn: 4,
  tp: 2,
  experts: 4,
  ep: 2,
  layers: 4,
  pp: 2,
  cp: 2,
  seqs: Object.freeze([
    ['a', 8],
    ['b', 4],
  ]),
  padMultiplier: 128, // --data-pad-size-multiplier 默认值；pad_size = TP × multiplier，本例 TP=1
  presetCount: 39, // ls scripts/models/*.sh（4c193f1f），全部直接或经 source 带 --swiglu
});

const range = (n) => Array.from({ length: n }, (_, i) => i);

// torch.chunk 语义：每块 ceil(len / n)，最后一块可更短，块数可少于 n
export function chunk(arr, n) {
  const size = Math.ceil(arr.length / n);
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

// ---------------- 对源码规则的最小复现 ----------------

// slime/backends/megatron_utils/hf_to_megatron/common.py::_tensor_parallel_shard（沿 partition_dim 的一维视图）
export function tpShard(name, units, size, rank, stride = 1) {
  if (size === 1) return units;
  if (name.includes('linear_fc1.weight') || name.includes('linear_fc1.bias')) {
    const [gate, up] = chunk(units, 2);
    return [...chunk(gate, size)[rank], ...chunk(up, size)[rank]];
  }
  const chunks = chunk(units, size * stride);
  const picked = [];
  for (let i = rank; i < chunks.length; i += size) picked.push(...chunks[i]);
  return picked;
}

// slime/backends/megatron_utils/update_weight/common.py::all_gather_params_async 第 3 阶段
export function gatherTp(name, partitions) {
  let parts = partitions;
  if (name.includes('linear_fc1.weight') || name.includes('linear_fc1.bias')) {
    const halves = parts.map((p) => chunk(p, 2));
    parts = [...halves.map((h) => h[0]), ...halves.map((h) => h[1])];
  }
  return parts.flat();
}

// megatron_to_hf/qwen2.py::convert_qwen2_to_hf：mlp.linear_fc1.weight → chunk(2) = gate_proj, up_proj
export const fc1ToHf = (full) => chunk(full, 2);

export function glu(cfg = CFG) {
  const name = 'decoder.layers.0.mlp.linear_fc1.weight';
  const hfGate = range(cfg.ffn).map((i) => `g${i}`);
  const hfUp = range(cfg.ffn).map((i) => `u${i}`);
  const full = [...hfGate, ...hfUp]; // hf_to_megatron/common.py::merge_gate_up
  const shards = range(cfg.tp).map((r) => tpShard(name, full, cfg.tp, r));
  const naive = shards.flat();
  const [naiveGate, naiveUp] = fc1ToHf(naive);
  const gathered = gatherTp(name, shards);
  const [gate, up] = fc1ToHf(gathered);
  return {
    hfGate,
    hfUp,
    full,
    shards,
    naive,
    naiveGate,
    naiveUp,
    gathered,
    gate,
    up,
    roundTrip: gate.join() === hfGate.join() && up.join() === hfUp.join(),
    naiveBroken: naiveGate.join() !== hfGate.join(),
  };
}

// 非 GLU 的 fc1 仍命中同名分支；fc2 走通用分支，按列（partition_dim 1）常规切
export function nonGlu(cfg = CFG) {
  const fc1 = range(cfg.ffn).map((i) => `a${i}`);
  const fc2 = range(cfg.ffn).map((i) => `c${i}`);
  const fc1Shards = range(cfg.tp).map((r) => tpShard('decoder.layers.0.mlp.linear_fc1.weight', fc1, cfg.tp, r));
  const fc2Shards = range(cfg.tp).map((r) => tpShard('decoder.layers.0.mlp.linear_fc2.weight', fc2, cfg.tp, r));
  const pairs = range(cfg.tp).map((r) => fc1Shards[r].map((a, i) => [a, fc2Shards[r][i]]));
  const mismatches = pairs.flat().filter(([a, c]) => a.slice(1) !== c.slice(1)).length;
  const exported = gatherTp('decoder.layers.0.mlp.linear_fc1.weight', fc1Shards);
  return {
    fc1,
    fc2,
    fc1Shards,
    fc2Shards,
    pairs,
    mismatches,
    exported,
    roundTrip: exported.join() === fc1.join(),
  };
}

// hf_to_megatron/qwen3_next.py::_qwen3_next_layer_tensor（linear_qkv 分支）与
// megatron_to_hf/qwen3_next.py::convert_qwen3_next_to_hf 的逆变换；以“头块”为单位，每块 head_dim 行
export function qkv(cfg = CFG) {
  const qpg = cfg.heads / cfg.groups;
  const hfQ = range(cfg.heads).flatMap((h) => [`q${h}`, `z${h}`]); // 每头 [query, gate]（HF/SGLang 契约）
  const hfK = range(cfg.groups).map((g) => `k${g}`);
  const hfV = range(cfg.groups).map((g) => `v${g}`);
  // q.reshape(groups, qpg, 2, hd).transpose(1, 2).flatten(1, 3)
  const qByGroup = range(cfg.groups).map((g) => {
    const heads = range(qpg).map((j) => [hfQ[2 * (g * qpg + j)], hfQ[2 * (g * qpg + j) + 1]]);
    return [...heads.map((h) => h[0]), ...heads.map((h) => h[1])];
  });
  const groupsLayout = range(cfg.groups).map((g) => [...qByGroup[g], hfK[g], hfV[g]]);
  const megatron = groupsLayout.flat();
  const blockRows = megatron.length * cfg.headDim;
  const shards = range(cfg.tp).map((r) => tpShard('decoder.layers.0.self_attention.linear_qkv.weight', megatron, cfg.tp, r));
  // param.view(groups, -1, hd, hidden) → split [2·qpg, 1, 1] → q.reshape(groups, 2, qpg, …).transpose(1, 2)
  const perGroup = megatron.length / cfg.groups;
  const exportQ = range(cfg.groups).flatMap((g) => {
    const row = megatron.slice(g * perGroup, (g + 1) * perGroup);
    const q = row.slice(0, 2 * qpg);
    const [queries, gates] = [q.slice(0, qpg), q.slice(qpg)];
    return range(qpg).flatMap((j) => [queries[j], gates[j]]);
  });
  const megatronSplit = {
    query: qpg * cfg.headDim,
    gate: qpg * cfg.headDim,
    key: cfg.headDim,
    value: cfg.headDim,
  }; // attention.py::SelfAttention.get_query_key_value_tensors（output_gate，依赖侧）
  return {
    qpg,
    hfQ,
    hfK,
    hfV,
    groupsLayout,
    megatron,
    rows: blockRows,
    hfQRows: hfQ.length * cfg.headDim,
    shards,
    shardRows: shards.map((s) => s.length * cfg.headDim),
    megatronSplit,
    exportQ,
    roundTrip: exportQ.join() === hfQ.join(),
    // hf_to_megatron/common.py::merge_qkv 期望 q 行数 = heads × head_dim
    genericExpectedQRows: cfg.heads * cfg.headDim,
  };
}

// update_weight/common.py::named_params_and_buffers：decoder 层号 + PP offset，expert 下标 + EP offset
export function naming(cfg = CFG) {
  const layersPerStage = cfg.layers / cfg.pp;
  const expertsPerRank = cfg.experts / cfg.ep;
  const ranks = [];
  for (let s = 0; s < cfg.pp; s += 1) {
    for (let e = 0; e < cfg.ep; e += 1) {
      const layerOffset = s * layersPerStage; // get_transformer_layer_offset（均匀切分）
      const expertOffset = (e * cfg.experts) / cfg.ep; // ep_rank * num_experts // ep_size
      const names = [];
      for (let l = 0; l < layersPerStage; l += 1) {
        for (let k = 0; k < expertsPerRank; k += 1) {
          names.push({
            local: `decoder.layers.${l}.mlp.experts.linear_fc1.weight${k}`,
            global: `decoder.layers.${l + layerOffset}.mlp.experts.linear_fc1.weight${k + expertOffset}`,
          });
        }
      }
      ranks.push({
        stage: s,
        epRank: e,
        layerOffset,
        expertOffset,
        localLayers: [0, layersPerStage - 1],
        localExperts: [0, expertsPerRank - 1],
        globalLayers: [layerOffset, layerOffset + layersPerStage - 1],
        globalExperts: [expertOffset, expertOffset + expertsPerRank - 1],
        names,
      });
    }
  }
  // hf_weight_iterator_direct.py::_get_megatron_local_param_infos：PP、EP 两轮交换后按名排序
  const union = [...new Set(ranks.flatMap((r) => r.names.map((n) => n.global)))].sort();
  const last = ranks[ranks.length - 1];
  const mtpLocal = 1;
  return {
    ranks,
    union,
    lastGlobal: last.names[last.names.length - 1].global,
    hfExample: `model.layers.${cfg.layers - 1}.mlp.experts.${cfg.experts - 1}`,
    mtp: {
      local: `mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight${mtpLocal}`,
      global: `mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight${mtpLocal + (1 * cfg.experts) / cfg.ep}`,
    },
  };
}

// cp_utils.py::slice_with_cp + data.py::get_batch（zigzag 分支）+ hf_attention.py::HuggingfaceAttention.forward
export function cp(cfg = CFG) {
  const C = cfg.cp;
  const seqs = cfg.seqs.map(([p, n]) => range(n).map((i) => `${p}${i}`));
  const local = range(C).map((r) => {
    const toks = seqs.flatMap((t) => {
      const cs = Math.ceil(t.length / (2 * C));
      const padded = [...t, ...Array(2 * C * cs - t.length).fill('pad')];
      return [...padded.slice(cs * r, cs * (r + 1)), ...padded.slice(cs * (2 * C - r - 1), cs * (2 * C - r))];
    });
    return toks;
  });
  const localLens = [0];
  seqs.forEach((t) => {
    const cs = Math.ceil(t.length / (2 * C));
    localLens.push(localLens[localLens.length - 1] + 2 * cs);
  });
  const padSize = 1 * cfg.padMultiplier; // TP=1
  const realLocal = localLens[localLens.length - 1];
  const pad = (padSize - (realLocal % padSize)) % padSize;
  const localCu = pad ? [...localLens, realLocal + pad] : localLens;
  const cuSeqlens = localCu.map((x) => x * C);
  // forward：前段按 rank 顺序、后段按 rank 逆序
  const reassembled = [];
  const segments = [];
  for (let i = 0; i < seqs.length; i += 1) {
    const lcu0 = localCu[i];
    const lcu1 = localCu[i + 1];
    const seqlen = cuSeqlens[i + 1] - cuSeqlens[i];
    const cs = Math.floor(Math.floor(seqlen / 2) / C);
    const front = range(C).map((r) => local[r].slice(lcu0, lcu0 + cs));
    const back = range(C)
      .map((r) => local[r].slice(lcu0 + cs, lcu1))
      .reverse();
    reassembled.push(...front.flat(), ...back.flat());
    segments.push({ seqlen, chunkSize: cs });
  }
  // output：每条序列 chunk(2·CP)，rank r 取第 r 与 2·CP−1−r 段
  const outputs = range(C).map((r) =>
    seqs.flatMap((_, i) => {
      const seq = reassembled.slice(cuSeqlens[i], cuSeqlens[i + 1]);
      const cs = chunk(seq, 2 * C);
      return [...cs[r], ...cs[2 * C - 1 - r]];
    }),
  );
  return {
    seqs,
    local,
    pad,
    padGlobal: pad * C,
    localCu,
    cuSeqlens,
    segments,
    reassembled,
    outputs,
    restoresOrder: reassembled.join() === seqs.flat().join(),
    outputsMatchInput: outputs.every((o, r) => o.join() === local[r].join()),
  };
}

// 反向梯度账（分析模型）：GDN 的输出 q 依赖同序列 p≤q 的输入，用前缀和 y_q = Σ_{p≤q} x_p、L = Σ_q y_q 代替。
// 每个 rank 的输出梯度 g_r 只在“它拿回的输出位置”上非零（SP 例外：scatter 的反向 all-gather 让各 rank 相同）；
// G_r = Jᵀ g_r。取本片 = G_r[本片]，reduce-scatter = Σ_k G_k[本片]。
export function grads(cfg = CFG) {
  const c = cp(cfg);
  const seq = c.seqs[0];
  const n = seq.length;
  const truth = range(n).map((p) => n - p);
  const owner = new Map();
  range(cfg.cp).forEach((r) => c.local[r].forEach((t) => owner.set(t, r)));
  const G = (positions) => range(n).map((p) => positions.filter((q) => q >= p).length);
  const cpPositions = range(cfg.cp).map((r) => range(n).filter((p) => owner.get(seq[p]) === r));
  const cpG = cpPositions.map((pos) => G(pos));
  const cpLocal = range(n).map((p) => cpG[owner.get(seq[p])][p]);
  const cpReduce = range(n).map((p) => cpG.reduce((s, g) => s + g[p], 0));
  const spG = range(cfg.tp).map(() => G(range(n))); // 反向 all-gather 后每个 TP rank 都是完整输出梯度
  const spLocal = range(n).map((p) => spG[0][p]);
  const spReduce = range(n).map((p) => spG.reduce((s, g) => s + g[p], 0));
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);
  return {
    tokens: seq,
    truth,
    spLocal,
    spReduce,
    cpLocal,
    cpReduce,
    totals: { truth: sum(truth), spLocal: sum(spLocal), spReduce: sum(spReduce), cpLocal: sum(cpLocal), cpReduce: sum(cpReduce) },
    cpPositions,
  };
}

export function model(cfg = CFG) {
  return { glu: glu(cfg), nonGlu: nonGlu(cfg), qkv: qkv(cfg), naming: naming(cfg), cp: cp(cfg), grads: grads(cfg) };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export const FONT = { ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, cap: 11.5, cl: 11 };
export function textWidth(s, size) {
  let w = 0;
  for (const ch of String(s)) w += /[\u2000-\u2bff\u2e80-\uffef]/.test(ch) ? size : size * 0.56;
  return w;
}
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .cl{font-size:11px;fill:#1F2430}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .h1{fill:#DCE7FB;stroke:#7DA2E8;stroke-width:1}
  .h2{fill:#F7E3CF;stroke:#D79A62;stroke-width:1}
  .h3{fill:#E6E9EE;stroke:#AEB6C2;stroke-width:1}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
`;

function canvas(W, H, title, desc) {
  const o = [];
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  const api = {
    o,
    rect: (x, y, w, h, cls = 'neutral', r = 6) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`),
    text: (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`),
    arrow: (x1, y1, x2, y2) => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="main"/>`),
    // 一排等宽格子；clsOf(label, i) 决定填色；返回右端 x
    cells: (x, y, labels, clsOf = () => 'h3', w = 34, h = 22, gap = 3) => {
      labels.forEach((lab, i) => {
        const cx = x + i * (w + gap);
        o.push(`<rect x="${cx}" y="${y}" width="${w}" height="${h}" rx="3" class="${clsOf(lab, i)}"/>`);
        o.push(`<text x="${cx + w / 2}" y="${y + h / 2 + 4}" class="cl" text-anchor="middle">${esc(lab)}</text>`);
      });
      return x + labels.length * (w + gap) - gap;
    },
    // 一个带文字的框，宽度按文字估算
    tag: (x, y, s, cls = 'neutral', size = 'sm') => {
      const w = Math.ceil(textWidth(s, FONT[size]) + 16);
      o.push(`<rect x="${x}" y="${y}" width="${w}" height="22" rx="4" class="${cls}"/>`);
      o.push(`<text x="${x + 8}" y="${y + 15}" class="${size}" text-anchor="start">${esc(s)}</text>`);
      return x + w;
    },
  };
  return api;
}

const gCls = (lab) => (lab.startsWith('g') ? 'h1' : lab.startsWith('u') ? 'h2' : 'h3');
const qCls = (lab) => (lab.startsWith('q') ? 'h1' : lab.startsWith('z') ? 'h2' : 'h3');
const bracket = (xs) => `[${xs.join(' ')}]`;

export function renderTpFusion(m = model(), cfg = CFG) {
  const W = 1180;
  const H = 1010;
  const c = canvas(
    W,
    H,
    'slime 权重布局规则一：GLU fc1 的 TP 切分与重排、非 GLU fc1 的静默错配、Qwen3-Next 门控 QKV 的按组交错',
    '玩具层 hidden 8、4 个 query head 与 2 个 KV group、head_dim 2、FFN 4，TP=2。面板一：HF gate_proj 与 up_proj 拼成 linear_fc1 后，导入按 gate、up 各自切半，导出先各自拆半再按全部 gate、全部 up 拼接；朴素按 rank 拼接会让 gate_proj 混入 up 行。面板二：同一名称规则套到非 GLU 的 fc1 时，fc1 行与按列常规切分的 fc2 在 rank 内错配，但导出重排又把它拼回原序，往返恒等。面板三：HF q_proj 每头 query 与 gate 交错，导入重排成按 KV group 连续的 query、gate、key、value，TP=2 时每个 rank 恰好持有整组，导出逆变换逐行还原。',
  );
  const { rect, text, cells, tag, arrow } = c;
  const g = m.glu;
  const ng = m.nonGlu;
  const q = m.qkv;

  text(24, 34, '同一层权重的三条布局规则：导入切分与导出重排必须配对，且只按参数名生效', 'ti');
  text(24, 56, `玩具层：hidden ${cfg.hidden}、${cfg.heads} 个 query head / ${cfg.groups} 个 KV group、head_dim ${cfg.headDim}、FFN ${cfg.ffn}；训练 TP=${cfg.tp}。格子里一个标签代表一整行（QKV 面板里一个标签代表 head_dim 行）`, 'su');

  const LX = 40;
  const CX = 250;

  // ---------- ① GLU ----------
  let y = 72;
  rect(24, y, 1132, 292, 'panel', 10);
  text(LX, y + 24, '① GLU linear_fc1：导入先把 gate、up 各自切半，导出先把每片拆成 gate、up 再分别拼', 'pt');
  text(LX, y + 62, 'HF gate_proj · up_proj', 'tx');
  let x = cells(CX, y + 46, g.hfGate, gCls);
  cells(x + 18, y + 46, g.hfUp, gCls);
  text(LX, y + 100, `merge_gate_up → ${g.full.length}×${cfg.hidden}`, 'tx');
  x = cells(CX, y + 84, g.full, gCls);
  text(x + 14, y + 100, 'Megatron fc1 的逻辑全量', 'sm');
  text(LX, y + 144, '_tensor_parallel_shard', 'tx');
  x = CX;
  g.shards.forEach((s, r) => {
    text(x, y + 123, `rank${r}`, 'sm');
    x = cells(x, y + 128, s, gCls) + 26;
  });
  text(x, y + 144, '每片 4×8：前半 gate、后半 up，与 Megatron 列并行一致', 'sm');
  text(LX, y + 184, '导出：按 rank 朴素拼接', 'tx');
  x = cells(CX, y + 168, g.naive, gCls);
  arrow(x + 8, y + 179, x + 38, y + 179);
  tag(x + 44, y + 168, `chunk(2) → gate_proj = ${bracket(g.naiveGate)}，混入 up 行`, 'acc2');
  text(LX, y + 222, 'all_gather_params_async', 'tx');
  x = cells(CX, y + 206, g.gathered, gCls);
  arrow(x + 8, y + 217, x + 38, y + 217);
  tag(x + 44, y + 206, `chunk(2) → gate_proj = ${bracket(g.gate)}，逐行还原`, 'acc1');
  text(LX, y + 262, '两处都只看参数名含 linear_fc1，不读 partition_stride（Megatron 对 GLU 设为 2）；family converter 再按 dim 0 对半拆成 gate_proj / up_proj', 'sm');
  text(LX, y + 280, '同一重排也用于 expert fc1：名字里的 linear_fc1 决定一切，expert 参数改用 expert-TP 组', 'sm');

  // ---------- ② 非 GLU ----------
  y = 376;
  rect(24, y, 1132, 250, 'panel', 10);
  text(LX, y + 24, '② 同一规则套到非 GLU 的 fc1：rank 内 fc1 行与 fc2 列错配，导出重排却把它拼回原序', 'pt');
  text(LX, y + 62, 'HF fc1 行 · fc2 列', 'tx');
  x = cells(CX, y + 46, ng.fc1, () => 'h1');
  cells(x + 18, y + 46, ng.fc2, () => 'h3');
  text(LX, y + 106, '导入后各 rank 持有', 'tx');
  x = CX;
  range(cfg.tp).forEach((r) => {
    text(x, y + 85, `rank${r}  fc1 · fc2`, 'sm');
    const x1 = cells(x, y + 90, ng.fc1Shards[r], () => 'h1');
    x = cells(x1 + 8, y + 90, ng.fc2Shards[r], () => 'h3') + 30;
  });
  text(x, y + 106, 'fc1 走同名 GLU 分支，fc2 按列常规切', 'sm');
  text(LX, y + 140, 'rank 内神经元配对', 'tx');
  x = CX;
  ng.pairs.flat().forEach(([a, cc]) => {
    x = tag(x, y + 124, `${a}↔${cc}`, a.slice(1) === cc.slice(1) ? 'ghost' : 'acc2') + 6;
  });
  tag(x + 14, y + 124, `${ng.mismatches}/${cfg.ffn} 个神经元错配，shape 全部正确`, 'acc2');
  text(LX, y + 178, '导出 all_gather_params_async', 'tx');
  x = cells(CX, y + 162, ng.exported, () => 'h1');
  tag(x + 20, y + 162, `= HF ${bracket(ng.fc1)}：往返恒等，round-trip 单测看不出`, 'acc2');
  text(LX, y + 216, '无守卫：Direct 路径只 assert partition_dim is not None（Megatron 默认 -1，恒真）；distributed 路径的 stride 断言也放行 stride=1 的 fc1', 'sm');
  text(LX, y + 234, `因此 linear_fc1 必须是 [gate; up] 融合；${cfg.presetCount} 个 scripts/models preset 都带 --swiglu（直接或经 source）`, 'sm');

  // ---------- ③ QKV ----------
  y = 638;
  rect(24, y, 1132, 350, 'panel', 10);
  text(LX, y + 24, '③ Qwen3-Next 门控 QKV：q_proj 每头 [query, gate] 交错，导入按 KV group 重排成连续的 [query…, gate…, key, value]', 'pt');
  text(LX, y + 62, `HF q_proj（${q.hfQRows} 行）`, 'tx');
  x = cells(CX, y + 46, q.hfQ, qCls);
  text(x + 18, y + 62, 'k_proj', 'sm');
  x = cells(x + 58, y + 46, q.hfK, qCls);
  text(x + 18, y + 62, 'v_proj', 'sm');
  cells(x + 58, y + 46, q.hfV, qCls);
  text(LX, y + 108, `linear_qkv（${q.rows}×${cfg.hidden}）`, 'tx');
  x = CX;
  q.groupsLayout.forEach((grp, gi) => {
    text(x, y + 88, `group ${gi}`, 'sm');
    x = cells(x, y + 92, grp, qCls) + 26;
  });
  text(x, y + 108, 'reshape(groups, qpg, 2, hd).transpose(1, 2)', 'sm');
  text(LX, y + 154, `TP${cfg.tp} 按行均分（stride 1）`, 'tx');
  x = CX;
  q.shards.forEach((s, r) => {
    text(x, y + 134, `rank${r}：${q.shardRows[r]} 行`, 'sm');
    x = cells(x, y + 138, s, qCls) + 26;
  });
  tag(x, y + 138, '每个 rank 恰好一整组，本地就能拆出 q、gate、k、v', 'acc1');
  text(LX, y + 200, 'Megatron 每组拆分（依赖侧）', 'tx');
  x = CX;
  [
    [`query ${q.megatronSplit.query} 行`, 'h1'],
    [`gate ${q.megatronSplit.gate} 行`, 'h2'],
    [`key ${q.megatronSplit.key} 行`, 'h3'],
    [`value ${q.megatronSplit.value} 行`, 'h3'],
  ].forEach(([s, cls]) => {
    x = tag(x, y + 184, s, cls) + 6;
  });
  text(x + 12, y + 200, 'SelfAttention.get_query_key_value_tensors（output_gate，Megatron@1dcf0daf）', 'sm');
  text(LX, y + 246, '导出逆变换', 'tx');
  x = cells(CX, y + 230, q.exportQ, qCls);
  tag(x + 20, y + 230, `q_proj = ${bracket(q.exportQ)}，与 HF 逐块相同`, 'acc1');
  text(LX, y + 290, `若误用不带门控的 merge_qkv：它期望 q 为 ${cfg.heads} × ${cfg.headDim} = ${q.genericExpectedQRows} 行，实得 ${q.hfQRows} 行，reshape 直接报错`, 'sm');
  text(LX, y + 308, 'group 数小于 TP 时，Megatron 前向先 all-gather 整个 qkv 输出再取本组（依赖侧），导入仍按行均分', 'sm');
  text(LX, y + 332, '颜色：蓝 = gate_proj 行 g 或 query 块 q；橙 = up_proj 行 u 或 attention gate 块 z；灰 = key、value 与非 GLU 行', 'cap');

  c.o.push('</svg>');
  return c.o.join('\n');
}

export function renderEpCp(m = model(), cfg = CFG) {
  const W = 1180;
  const H = 1000;
  const c = canvas(
    W,
    H,
    'slime 权重布局规则二：PP/EP 全局命名、HF 黑盒 GDN 的 CP 两段重组与反向梯度账',
    '面板四：4 层、4 个 expert，PP=2、EP=2 的四个 rank 各自把本地层号 0 到 1、expert 下标 0 到 1 加上 PP 与 EP offset，元数据交换后每个 rank 都看到同一份 16 个 expert fc1 名；MTP 层号不加 offset。面板五：CP=2 时两条 packed 序列按 zigzag 切给两个 rank，HF 包装层 all-gather 后按前段 rank 顺序、后段 rank 逆序拼回原序，GDN 之后按 chunk 取回各 rank 的两段。面板六：用前缀和代替 GDN 的因果依赖，比较 SP 与 CP 两条 all-gather 反向取本片与 reduce-scatter 的梯度，CP 取本片会丢掉其他 rank 输出对本片输入的梯度项。',
  );
  const { rect, text, cells, tag, arrow } = c;
  const nm = m.naming;
  const cpm = m.cp;
  const gr = m.grads;
  const LX = 40;
  const CX = 250;
  let x;

  text(24, 34, '身份与序列的两条布局规则：权重名按 PP/EP 恢复全局编号，黑盒模块在 CP 下先拼回整条序列', 'ti');
  text(24, 56, `玩具模型：${cfg.layers} 层、${cfg.experts} 个 expert，PP=${cfg.pp}、EP=${cfg.ep}；packed batch：a 序列 ${cfg.seqs[0][1]} 个 token、b 序列 ${cfg.seqs[1][1]} 个，CP=${cfg.cp}、TP=1、pad 倍数 ${cfg.padMultiplier}`, 'su');

  // ---------- ④ 命名 ----------
  let y = 72;
  rect(24, y, 1132, 262, 'panel', 10);
  text(LX, y + 24, '④ named_params_and_buffers：decoder 层号加 PP offset，expert 下标加 EP offset；导入与导出共用这份全局名', 'pt');
  const bw = 300;
  const bh = 78;
  nm.ranks.forEach((r, i) => {
    const bx = LX + (i % cfg.ep) * (bw + 14);
    const by = y + 42 + Math.floor(i / cfg.ep) * (bh + 12);
    rect(bx, by, bw, bh, 'neutral', 6);
    text(bx + 10, by + 18, `PP stage ${r.stage} · EP rank ${r.epRank}（offset 层 +${r.layerOffset}，expert +${r.expertOffset}）`, 'sm');
    text(bx + 10, by + 40, `本地 layers.${r.localLayers[0]}–${r.localLayers[1]} · weight${r.localExperts[0]}–${r.localExperts[1]}`, 'tx');
    text(bx + 10, by + 62, `全局 layers.${r.globalLayers[0]}–${r.globalLayers[1]} · weight${r.globalExperts[0]}–${r.globalExperts[1]}`, 'tx');
  });
  const rx = LX + 2 * (bw + 14) + 10;
  arrow(rx - 8, y + 130, rx + 22, y + 130);
  rect(rx + 30, y + 42, 1140 - rx - 30, 178, 'acc1', 6);
  text(rx + 42, y + 64, '_get_megatron_local_param_infos', 'tx');
  text(rx + 42, y + 86, `PP 与 EP 两轮交换元数据后排序：每个 rank 都是同一份 ${nm.union.length} 个 expert fc1 名`, 'sm');
  text(rx + 42, y + 106, '再逐项校验 name / shape / dtype 一致', 'sm');
  text(rx + 42, y + 128, '例：PP1·EP1 的本地 layers.1 · weight1', 'sm');
  text(rx + 42, y + 146, `→ ${nm.lastGlobal.replace('decoder.', '')}`, 'sm');
  text(rx + 42, y + 164, `→ ${nm.hfExample}.gate_proj / up_proj`, 'sm');
  text(rx + 42, y + 188, '导入同样按全局名去读 HF 的 experts.{全局下标}', 'sm');
  text(rx + 42, y + 208, '不在本 rank 的 expert 由 EP 组广播补齐', 'sm');
  tag(LX, y + 232, `MTP：EP rank 1 的 mtp.layers.0 … weight1 → weight${nm.mtp.global.match(/weight(\d+)$/)[1]}，层号仍为 0（不加 PP offset）`, 'acc2');

  // ---------- ⑤ CP 前向 ----------
  y = 346;
  rect(24, y, 1132, 318, 'panel', 10);
  text(LX, y + 24, '⑤ HuggingfaceAttention.forward：CP>1 时先 all-gather，再按“前段 rank 顺序、后段 rank 逆序”拼回原序列交给 GDN', 'pt');
  const tokCls = (t) => (t.startsWith('a') ? 'h1' : t.startsWith('b') ? 'h2' : 'h3');
  text(LX, y + 62, '全局 packed 序列', 'tx');
  x = cells(CX, y + 46, cpm.seqs[0], tokCls);
  x = cells(x + 14, y + 46, cpm.seqs[1], tokCls);
  text(x + 14, y + 62, 'slice_with_cp：每条补到 2·CP 的倍数，rank r 取第 r 与 2·CP−1−r 段', 'sm');
  range(cfg.cp).forEach((r) => {
    const ry = y + 84 + r * 30;
    text(LX, ry + 16, `rank${r} 本地`, 'tx');
    const xe = cells(CX, ry, cpm.local[r], tokCls);
    tag(xe + 10, ry, `pad 本地 ${cpm.pad}`, 'ghost');
  });
  text(CX + 410, y + 100, `cu_seqlens = ${bracket(cpm.cuSeqlens).replace(/ /g, ', ')}（本地 ${bracket(cpm.localCu).replace(/ /g, ', ')} × CP）`, 'sm');
  text(CX + 410, y + 130, `a 段 chunk_size ${cpm.segments[0].chunkSize}、b 段 ${cpm.segments[1].chunkSize}：seqlen // 2 // CP`, 'sm');
  text(LX, y + 176, '拼回原序', 'tx');
  x = cells(CX, y + 160, cpm.reassembled, tokCls);
  x = tag(x + 10, y + 160, `pad 全局 ${cpm.padGlobal}`, 'ghost');
  tag(x + 14, y + 160, 'GDN 按 cu_seqlens 逐序列递推，不跨 a、b', 'acc1');
  text(LX, y + 228, '输出取回', 'tx');
  x = CX;
  range(cfg.cp).forEach((r) => {
    text(x, y + 207, `rank${r}`, 'sm');
    x = cells(x, y + 212, cpm.outputs[r], tokCls) + 26;
  });
  tag(x, y + 212, '每条序列 chunk(2·CP)，取第 r 与 2·CP−1−r 段', 'acc1');
  text(LX, y + 262, '代价：每个 CP rank 都对全局全部位置（含 pad）跑一遍 GDN；SP 下先按 TP 组 gather、输出再 scatter，模块内部没有 TP 切分', 'sm');
  text(LX, y + 282, '假设：每条序列本地长度是 2 的倍数、全局长度可被 2·CP 整除（由 slice_with_cp 的补齐保证），wrapper 自身不校验', 'sm');
  text(LX, y + 302, 'allgather-CP（DSA 模式）是另一种连续等分布局，只允许 DSA 架构；本面板是默认 zigzag 布局', 'sm');

  // ---------- ⑥ 反向梯度账 ----------
  y = 676;
  rect(24, y, 1132, 304, 'panel', 10);
  text(LX, y + 24, '⑥ 反向：两条 all-gather 都“只取本片梯度”；SP 成立，CP 下会丢掉其他 rank 输出对本片输入的梯度（分析判断）', 'pt');
  text(LX, y + 46, '分析模型：用前缀和 y_q = Σ_{p≤q} x_p 代替 GDN 的因果依赖、损失 L = Σ_q y_q，只看 a 序列；未在 slime 的 GPU 训练中验证', 'sm');
  const colX = (i) => 520 + i * 60;
  text(LX, y + 76, '输入 token', 'tx');
  gr.tokens.forEach((t, i) => text(colX(i), y + 76, t, 'tx', 'middle'));
  text(colX(gr.tokens.length) + 10, y + 76, '合计', 'tx', 'middle');
  const rows = [
    ['真实 ∂L/∂x', gr.truth, gr.totals.truth, 'ghost'],
    [`SP（TP=${cfg.tp}）：scatter 反向 all-gather，各 rank 输出梯度相同 → 取本片`, gr.spLocal, gr.totals.spLocal, 'acc1'],
    [`SP（TP=${cfg.tp}）：若改用 reduce-scatter 求和`, gr.spReduce, gr.totals.spReduce, 'ghost'],
    [`CP（CP=${cfg.cp}）：输出只按段切片，各 rank 输出梯度不同 → 取本片`, gr.cpLocal, gr.totals.cpLocal, 'acc2'],
    [`CP（CP=${cfg.cp}）：若改用 reduce-scatter 求和`, gr.cpReduce, gr.totals.cpReduce, 'ghost'],
  ];
  rows.forEach(([label, vals, total, cls], ri) => {
    const ry = y + 90 + ri * 34;
    rect(LX - 6, ry, 1112, 28, cls, 4);
    text(LX + 4, ry + 19, label, 'tx');
    vals.forEach((v, i) => text(colX(i), ry + 19, v, 'tx', 'middle'));
    text(colX(vals.length) + 10, ry + 19, total, 'tx', 'middle');
  });
  const owners = gr.cpPositions.map((pos, r) => `rank${r} 持有 ${pos.map((p) => gr.tokens[p]).join(' ')}`).join('；');
  text(LX, y + 272, `${owners}。例：rank1 上 a2–a5 的损失对 a0 的梯度（4）在 rank1 算出后被丢弃`, 'sm');
  text(LX, y + 292, '源码注释的前提是“重复计算 → 梯度相同”；它对 SP 成立（输出 scatter 的反向是 all-gather），对 CP 的输出切片不成立', 'sm');

  c.o.push('</svg>');
  return c.o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
export const OUTPUTS = Object.freeze({
  tpFusion: 'slime_model_layout_tp_fusion.svg',
  epCp: 'slime_model_layout_ep_cp.svg',
});
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  const m = model();
  writeFileSync(join(outputDir, OUTPUTS.tpFusion), `${renderTpFusion(m)}\n`, 'utf8');
  writeFileSync(join(outputDir, OUTPUTS.epCp), `${renderEpCp(m)}\n`, 'utf8');
}
