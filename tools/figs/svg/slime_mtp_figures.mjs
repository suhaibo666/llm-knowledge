// 图：slime 在线 MTP 的三个平面各用一个最小例子复现：
//   ① 训练：一条 P=3、R=4 的样本经 get_batch 的 full_loss_masks 与 megatron.patch 的两次 roll 后，
//      MTP-1 实际计入哪些目标；同一位置的 MTP loss 反向在上游 1dcf0daf 与补丁后分别落到哪里；
//   ② 发布：PP=2 × EP=2、4 层 decoder + 1 层 MTP、8 个 expert 时 named_params_and_buffers 的改名规则、
//      GLM4-MoE 转换器的 HF 层号，以及三条 transport 把同一批张量交给哪个 SGLang runner；
//   ③ 观测：锁定镜像里 SpecInfo 读不到的两个键，以及键兼容时 spec_accept_rate 的按样本等权平均。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；NVIDIA/Megatron-LM@1dcf0dafa884 + slime
// docker/patch/latest/megatron.patch；sgl-project/sglang@0b3bb0cbe318（v0.5.15.post1）+ docker/patch/latest/sglang*.patch。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 图 1 要讲清楚：full_loss_masks 已经按"位置 i 预测 t[i+1]"对齐；补丁先把 tokens 与 mask 各 roll 一次，
// 循环里再各 roll 一次，于是 MTP-1 的标签是 t[i+2]、mask 是 full[i+2]，计入的目标整体早一位（含最后一个
// prompt token、漏最后一个 response token）；三处切断（decoder_input.detach、keep_graph=False、
// mtp_output_weight.detach）让 MTP loss 只回传到 MTP 层；CI 用全截断让 policy 梯度为 0 再断言。
// 图 2 要讲清楚：decoder 层号加 layer_offset、expert 下标加 expert_offset，MTP 层号保持从 0 开始，
// 因为转换器还要再加 num_layers；embedding 在 MTP 段有副本，PP 交换元数据时保留较小 src_rank；
// tensor 与 disk 入口会更新 draft，distributed 入口只更新 target。
// 图 3 要讲清楚：SGLang v0.5.15.post1 输出的键与 SpecInfo.add 读取的键不相交的两项让分子分母都是 0；
// 键兼容时 partial 两段先累加原始计数，再按样本求比值，最后等权平均。
// acc1 标源码成立的主路径，acc2 标代价、缺口与错位；依赖侧（Megatron 上游、SGLang）用虚线框。
//
// 用法：node tools/figs/svg/slime_mtp_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  // 图 1：一条样本，P 个 prompt token、R 个 response token，loss_mask 全 1；mtp_num_layers = 1
  sample: { P: 3, R: 4 },
  gradPos: 2, // 图 1 面板②取的位置 i
  // 图 2：最小并行布局（GLM4-MoE 结构，4 层 decoder、1 层 MTP、8 个 routed expert）
  layout: { numLayers: 4, pp: 2, ep: 2, numExperts: 8, mtpLayers: 1 },
  // 图 3：num_draft_tokens = 4（slime 文档的最小配置），每次 verify 提议 3 个 draft
  spec: {
    numDraftTokens: 4,
    samples: [
      { name: 'A', segments: [{ verify: 2, correct: 5, completion: 8 }] },
      {
        name: 'B',
        segments: [
          { verify: 8, correct: 4, completion: 13 },
          { verify: 12, correct: 8, completion: 21 },
        ],
      },
    ],
  },
});

const range = (n) => Array.from({ length: n }, (_, i) => i);

// ---------------- 对源码算法的最小复现 ----------------

// megatron/core/transformer/multi_token_prediction.py::_roll_tensor_packed_seq（CP=1）：
// 序列内左移一位，最后一位清零。null 表示被清零的位置（token id 0 / mask 0）。
export function roll(arr) {
  return [...arr.slice(1), null];
}

// slime/backends/megatron_utils/data.py::get_batch：F.pad(loss_mask, (prompt_length - 1, 1))
export function fullLossMask(P, R, respMask = null) {
  const m = respMask ?? Array(R).fill(1);
  return [...Array(P - 1).fill(0), ...m, 0];
}

export function alignment(cfg = CFG) {
  const { P, R } = cfg.sample;
  const T = P + R;
  const tokens = range(T); // token 身份用下标表示：t0..t(T-1)
  const full = fullLossMask(P, R);
  const nz = (v) => (v === null ? 0 : v);
  // 主 loss：loss.py::_build_shifted_tokens，位置 i 的目标是 t[i+1]
  const mainTarget = range(T).map((i) => (i + 1 < T ? i + 1 : null));
  // MTP-1 的输入 embedding：MultiTokenPredictionLayer._get_embeddings 把 input_ids roll 一次
  const mtpInput = roll(tokens);
  // gpt_model.py::_postprocess（megatron.patch）：mtp_labels = tokens，先 roll 一次，循环里再 roll 一次
  const mtpLabel = roll(roll(tokens));
  // loss_mask 同样先 roll 一次（补丁新增），循环里再 roll 一次
  const maskPatched = roll(roll(full)).map(nz);
  // 对照：只做循环里的一次 roll（上游 1dcf0daf 对"按标签对齐的 mask"的处理方式）
  const maskLoopOnly = roll(full).map(nz);
  const counted = range(T).filter((i) => maskPatched[i] === 1 && mtpLabel[i] !== null).map((i) => mtpLabel[i]);
  const countedLoopOnly = range(T).filter((i) => maskLoopOnly[i] === 1 && mtpLabel[i] !== null).map((i) => mtpLabel[i]);
  const response = range(R).map((k) => P + k);
  return {
    P,
    R,
    T,
    full,
    mainTarget,
    mtpInput,
    mtpLabel,
    maskPatched,
    maskLoopOnly,
    counted,
    countedLoopOnly,
    response,
    numTokens: maskPatched.reduce((a, b) => a + b, 0),
    numTokensLoopOnly: maskLoopOnly.reduce((a, b) => a + b, 0),
    extraPrompt: counted.filter((t) => t < P),
    missedResponse: response.filter((t) => !counted.includes(t)),
  };
}

// slime/backends/megatron_utils/update_weight/common.py::named_params_and_buffers 的改名规则
export function globalName(localName, { layerOffset, expertOffset }) {
  let m = localName.match(/^decoder\.layers\.(\d+)\.(.+)$/);
  if (m) {
    const layer = Number(m[1]) + layerOffset;
    const e = m[2].match(/^mlp\.experts\.(.+)\.(weight|bias)(\d+)$/);
    if (e) return `decoder.layers.${layer}.mlp.experts.${e[1]}.${e[2]}${Number(e[3]) + expertOffset}`;
    return `decoder.layers.${layer}.${m[2]}`;
  }
  m = localName.match(/^mtp\.layers\.(\d+)\.(.+)$/);
  if (m) {
    // "MTP layer indices start from 0"：层号不加 layer_offset，只有 expert 下标加 EP 偏移
    const e = m[2].match(/^transformer_layer\.mlp\.experts\.(.+)\.(weight|bias)(\d+)$/);
    if (e) return `mtp.layers.${m[1]}.transformer_layer.mlp.experts.${e[1]}.${e[2]}${Number(e[3]) + expertOffset}`;
    return localName;
  }
  return localName;
}

// slime/backends/megatron_utils/megatron_to_hf/glm4moe.py::convert_glm4moe_to_hf 中与本例相关的分支
export function hfNames(name, numLayers) {
  if (name === 'embedding.word_embeddings.weight') return ['model.embed_tokens.weight'];
  let m = name.match(/^mtp\.layers\.(\d+)\.(.+)$/);
  if (m) {
    const layer = Number(m[1]) + numLayers;
    const direct = { 'eh_proj.weight': 'eh_proj.weight', 'enorm.weight': 'enorm.weight', 'hnorm.weight': 'hnorm.weight', 'final_layernorm.weight': 'shared_head.norm.weight' };
    if (direct[m[2]]) return [`model.layers.${layer}.${direct[m[2]]}`];
    return hfNames(`decoder.layers.${layer}.${m[2].replace('transformer_layer.', '')}`, numLayers);
  }
  m = name.match(/^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc1\.weight(\d+)$/);
  if (m) return [`model.layers.${m[1]}.mlp.experts.${m[2]}.gate_proj.weight`, `model.layers.${m[1]}.mlp.experts.${m[2]}.up_proj.weight`];
  m = name.match(/^decoder\.layers\.(\d+)\.(.+)$/);
  if (m) return [`model.layers.${m[1]}.${m[2]}`];
  return [name];
}

export function rename(cfg = CFG) {
  const L = cfg.layout;
  const perStage = L.numLayers / L.pp;
  const perEp = L.numExperts / L.ep;
  const ranks = [];
  for (let s = 0; s < L.pp; s += 1) {
    for (let e = 0; e < L.ep; e += 1) {
      const rank = s * L.ep + e;
      // get_transformer_layer_offset（均分）与 ep_rank * num_experts // ep_size
      const layerOffset = s * perStage;
      const expertOffset = (e * L.numExperts) / L.ep;
      // megatron get_mtp_num_layers_to_build：无自定义 layout 时 MTP 只在最后一个 PP stage 构建
      const hasMtp = s === L.pp - 1;
      // GPTModel.__init__：pre_process 或 mtp_process 的 stage 构建 embedding
      const hasEmbedding = s === 0 || hasMtp;
      const locals = [];
      if (hasEmbedding) locals.push('embedding.word_embeddings.weight');
      range(perStage).forEach((l) => locals.push(`decoder.layers.${l}.mlp.experts.linear_fc1.weight${perEp - 1}`));
      if (hasMtp) {
        locals.push('mtp.layers.0.eh_proj.weight');
        locals.push(`mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight${perEp - 1}`);
      }
      ranks.push({
        rank,
        stage: s,
        epRank: e,
        layerOffset,
        expertOffset,
        hasMtp,
        names: locals.map((n) => ({ local: n, global: globalName(n, { layerOffset, expertOffset }) })),
      });
    }
  }
  const pick = (rank, local) => ranks[rank].names.find((n) => n.local === local);
  const lastRank = L.pp * L.ep - 1;
  // hf_weight_iterator_direct.py::_get_megatron_local_param_infos：在 PP group 内 all_gather 元数据，
  // 同名参数保留较小 src_rank。PP group = 同一 EP 位置、不同 PP stage 的 rank。
  const ppGroup = ranks.filter((k) => k.epRank === ranks[lastRank].epRank);
  const holders = ppGroup.filter((k) => k.names.some((n) => n.global === 'embedding.word_embeddings.weight')).map((k) => k.rank);
  const expertLocal = `decoder.layers.${perStage - 1}.mlp.experts.linear_fc1.weight${perEp - 1}`;
  const mtpExpertLocal = `mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight${perEp - 1}`;
  const traces = [
    { key: 'decoderExpert', rank: lastRank, ...pick(lastRank, expertLocal) },
    { key: 'mtpExpert', rank: lastRank, ...pick(lastRank, mtpExpertLocal) },
    { key: 'mtpProj', rank: lastRank, ...pick(lastRank, 'mtp.layers.0.eh_proj.weight') },
  ].map((t) => ({ ...t, hf: hfNames(t.global, L.numLayers) }));
  // 被否掉的写法：把 decoder 的 layer_offset 也加到 MTP 层号上
  const wrongLayer = ranks[lastRank].layerOffset;
  const wrongHf = hfNames(`mtp.layers.${wrongLayer}.eh_proj.weight`, L.numLayers)[0];
  const hfLayerCount = L.numLayers + L.mtpLayers; // HF checkpoint 中 model.layers.0 .. model.layers.(N+M-1)
  return {
    perStage,
    perEp,
    ranks,
    traces,
    wrongLayer,
    wrongHf,
    hfLayerCount,
    ppGroup: ppGroup.map((k) => k.rank),
    embeddingSrc: Math.min(...holders),
    embeddingHolders: holders,
  };
}

// SGLang tokenizer_manager.py::_calculate_spec_decoding_metrics 输出的键（v0.5.15.post1 上游；slime 补丁未改）
export function sglangMetaInfo(seg, numDraftTokens) {
  const proposed = seg.verify * (numDraftTokens - 1);
  return {
    finish_reason: { type: 'stop' },
    completion_tokens: seg.completion,
    spec_accept_rate: seg.correct / proposed,
    spec_accept_length: seg.completion / seg.verify,
    spec_num_correct_drafts: seg.correct,
    spec_num_proposed_drafts: proposed,
    spec_verify_ct: seg.verify,
    spec_accepted_drafts: seg.correct,
    spec_proposed_drafts: proposed,
  };
}

// slime/utils/types.py::Sample.SpecInfo（add / spec_accept_rate / spec_accept_length）
export function specInfo(metaInfos) {
  const s = { accept: 0, draft: 0, verify: 0, completion: 0 };
  metaInfos.forEach((mi) => {
    s.accept += mi.spec_accept_token_num ?? 0;
    s.draft += mi.spec_draft_token_num ?? 0;
    s.verify += mi.spec_verify_ct ?? 0;
    s.completion += mi.completion_tokens ?? 0;
  });
  return {
    ...s,
    rate: s.draft > 0 ? s.accept / s.draft : 0,
    length: s.verify > 0 ? s.completion / s.verify : 0,
  };
}

// slime/observability/rollout_metrics.py::_compute_spec_metrics：对样本等权平均
export const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

export function acceptMetrics(cfg = CFG) {
  const k = cfg.spec.numDraftTokens;
  const pinned = cfg.spec.samples.map((smp) => ({ name: smp.name, ...specInfo(smp.segments.map((g) => sglangMetaInfo(g, k))) }));
  // 键兼容的情形：假设服务端同时给出 slime 读取的两个键（docker/patch/v0.5.5.post1/sglang.patch 曾经这样补过）
  const compatible = cfg.spec.samples.map((smp) => ({
    name: smp.name,
    segments: smp.segments.map((g) => ({ ...g, proposed: g.verify * (k - 1) })),
    ...specInfo(
      smp.segments.map((g) => ({
        ...sglangMetaInfo(g, k),
        spec_accept_token_num: g.correct,
        spec_draft_token_num: g.verify * (k - 1),
      })),
    ),
  }));
  const pooledRate = compatible.reduce((a, s) => a + s.accept, 0) / compatible.reduce((a, s) => a + s.draft, 0);
  const pooledLength = compatible.reduce((a, s) => a + s.completion, 0) / compatible.reduce((a, s) => a + s.verify, 0);
  return {
    proposedPerVerify: k - 1,
    pinned,
    pinnedRate: mean(pinned.map((s) => s.rate)),
    pinnedLength: mean(pinned.map((s) => s.length)),
    compatible,
    meanRate: mean(compatible.map((s) => s.rate)),
    pooledRate,
    meanLength: mean(compatible.map((s) => s.length)),
    pooledLength,
  };
}

export function model(cfg = CFG) {
  return { align: alignment(cfg), rename: rename(cfg), accept: acceptMetrics(cfg) };
}

export const fmt = (x, d = 3) => x.toFixed(d);

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function textWidth(s, size, mono = false) {
  let w = 0;
  // CJK 与全角标点按 1em、箭头与 ✓✕ 等符号按 0.8em、拉丁字符按比例字体 0.58em（等宽 0.61em）估计
  for (const ch of String(s)) {
    if (/[\u2E80-\uFFEF]/.test(ch)) w += size;
    else if (/[\u2190-\u2BFF]/.test(ch)) w += size * 0.8;
    else w += size * (mono ? 0.61 : 0.58);
  }
  return w;
}
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .mo{font-family:"JetBrains Mono","Cascadia Mono",Consolas,monospace;font-size:10.5px;fill:#2A313B}
  .cap{font-size:11.5px;fill:#5B6470}
  .b{font-weight:700}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .cost{fill:none;stroke:#C3651F;stroke-width:2;marker-end:url(#arrowCost)}
  .fwd{fill:none;stroke:#8C95A3;stroke-width:1.3;marker-end:url(#arrowAux)}
  .cut{fill:none;stroke:#C3651F;stroke-width:2.2}
`;
const DEFS =
  '<defs>' +
  '<marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker>' +
  '<marker id="arrowCost" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#C3651F"/></marker>' +
  '<marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#8C95A3"/></marker>' +
  '</defs>';

function canvas(W, H, title, desc) {
  const o = [];
  const api = {
    o,
    rect: (x, y, w, h, cls = 'neutral', r = 6) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`),
    text: (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`),
    line: (x1, y1, x2, y2, cls = 'fwd') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`),
    path: (d, cls) => o.push(`<path d="${d}" class="${cls}"/>`),
    box: (x, y, w, h, lines, cls = 'neutral', textCls = 'sm', anchor = 'middle') => {
      api.rect(x, y, w, h, cls, 5);
      const lh = 15;
      const y0 = y + h / 2 - ((lines.length - 1) * lh) / 2 + 4;
      lines.forEach((ln, i) => {
        const [s, c] = Array.isArray(ln) ? ln : [ln, textCls];
        api.text(anchor === 'middle' ? x + w / 2 : x + 10, y0 + i * lh, s, c, anchor);
      });
    },
    // 剪刀记号：在线段中点画一个 ×
    scissor: (x, y) => {
      o.push(`<path d="M${x - 7} ${y - 7} L${x + 7} ${y + 7} M${x + 7} ${y - 7} L${x - 7} ${y + 7}" class="cut"/>`);
    },
  };
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push(DEFS);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  return api;
}

// ---------- 图 1：训练平面 ----------
export function renderTraining(m, cfg = CFG) {
  const a = m.align;
  const W = 1180;
  const H = 930;
  const c = canvas(
    W,
    H,
    'slime 在线 MTP 训练：一条样本的标签与 mask 对齐，以及 MTP loss 的梯度切断',
    `面板①以 P=${a.P}、R=${a.R} 的样本逐位置展示 full_loss_masks、主 loss 目标、MTP-1 的输入、补丁两次 roll 后的 mtp_labels 与 loss_mask，以及实际计入的目标 t2 到 t5 与对照的 t3 到 t6。面板②对比上游 Megatron-LM 1dcf0daf 与 slime megatron.patch 下位置 i=2 的 MTP loss 反向传播落点。面板③是 CI 门禁的前提与断言。`,
  );
  const { rect, text, line, box, scissor } = c;
  text(24, 34, 'MTP 是辅助目标：补丁把梯度关在 MTP 层里；两次 roll 让标签对齐 t[i+2]，mask 却因此早一位', 'ti');
  text(24, 56, `示例：一条样本 P = ${a.P} 个 prompt token、R = ${a.R} 个 response token（T = ${a.T}），loss_mask 全 1，--mtp-num-layers 1，CP = 1；token 用下标 t0…t${a.T - 1} 表示`, 'su');

  // ① 对齐网格
  const py = 72;
  rect(24, py, 1132, 382, 'panel', 10);
  text(40, py + 24, '① 同一条样本在 get_batch 与 megatron.patch 的 _postprocess 里怎样对齐（位置 i 的 MTP-1 看到 h_i 与 t[i+1]，预测 t[i+2]）', 'pt');
  const gx = 392;
  const cw = 80;
  const rh = 32;
  const labelX = 40;
  const top = py + 40;
  const tok = (v) => (v === null ? '0' : `t${v}`);
  text(labelX, top + 20, '位置 i', 'tx');
  range(a.T).forEach((i) => text(gx + i * cw + cw / 2, top + 20, String(i), 'tx', 'middle'));
  const rows = [
    {
      label: ['tokens（mtp_labels 的来源）', 'data.py::get_batch'],
      cells: range(a.T).map((i) => [tok(i), i < a.P ? 'ghost' : 'neutral']),
      note: `prompt t0–t${a.P - 1} · response t${a.P}–t${a.T - 1}`,
    },
    {
      label: ['full_loss_masks = F.pad(mask, (P−1, 1))', '已按"位置 i → t[i+1]"对齐'],
      cells: a.full.map((v) => [String(v), v ? 'neutral' : 'ghost']),
      note: '与主 loss 目标同一坐标',
    },
    {
      label: ['主 loss 目标 t[i+1]', 'loss.py::_build_shifted_tokens'],
      cells: a.mainTarget.map((v, i) => [v === null ? '—' : tok(v), a.full[i] ? 'neutral' : 'ghost']),
      note: `mask=1 处恰好是 t${a.P}–t${a.T - 1}`,
    },
    {
      label: ['MTP-1 输入 Emb(t[i+1])', '_get_embeddings：input_ids roll 1 次'],
      cells: a.mtpInput.map((v) => [tok(v), v === null ? 'ghost' : 'neutral']),
      note: '末位清零',
    },
    {
      label: ['mtp_labels：补丁预 roll + 循环 roll', '= t[i+2]'],
      cells: a.mtpLabel.map((v) => [tok(v), v === null ? 'ghost' : 'neutral']),
      note: '= MTP-1 的预测对象',
    },
    {
      label: ['loss_mask：补丁预 roll + 循环 roll', '= full[i+2]'],
      cells: a.maskPatched.map((v) => [String(v), v ? 'acc2' : 'ghost']),
      note: `num_tokens = ${a.numTokens}`,
    },
    {
      label: ['计入 MTP loss 的目标（补丁后实际）', 'label × mask'],
      cells: range(a.T).map((i) => {
        if (a.maskPatched[i] !== 1) {
          const lab = a.mtpLabel[i];
          if (lab !== null && a.missedResponse.includes(lab)) return [`${tok(lab)} 未计`, 'acc2'];
          return ['', 'ghost'];
        }
        const lab = a.mtpLabel[i];
        return [tok(lab), lab < a.P ? 'acc2' : 'acc1'];
      }),
      note: `t${a.counted[0]}–t${a.counted[a.counted.length - 1]}：多 t${a.extraPrompt[0]}、漏 t${a.missedResponse[0]}`,
    },
    {
      label: ['对照：mask 只随循环 roll 1 次', '= full[i+1]'],
      cells: range(a.T).map((i) => (a.maskLoopOnly[i] === 1 ? [tok(a.mtpLabel[i]), 'dep'] : ['', 'ghost'])),
      note: `t${a.countedLoopOnly[0]}–t${a.countedLoopOnly[a.countedLoopOnly.length - 1]} = response`,
    },
  ];
  rows.forEach((r, k) => {
    const y = top + 30 + k * (rh + 4);
    text(labelX, y + 13, r.label[0], 'tx');
    text(labelX, y + 27, r.label[1], 'sm');
    r.cells.forEach(([s, cls], i) => {
      rect(gx + i * cw + 3, y + 2, cw - 6, rh - 4, cls, 4);
      text(gx + i * cw + cw / 2, y + 20, s, 'mo', 'middle');
    });
    text(gx + a.T * cw + 10, y + 20, r.note, 'sm');
  });
  text(40, py + 368, '静态复算（分析判断，未运行）：mask 在 get_batch 已经左移过一位，补丁的预 roll 又移一位，计入集合整体早一位；P ≥ 3 时每条样本计数仍为 R，P = 2 时只剩 R − 1。', 'sm');

  // ② 梯度切断
  const gy = py + 392;
  rect(24, gy, 1132, 300, 'panel', 10);
  text(40, gy + 24, `② 位置 i = ${cfg.gradPos} 的 MTP loss（标签 t${a.mtpLabel[cfg.gradPos]}）反向传到哪里`, 'pt');
  const sub = (x0, title, patched) => {
    text(x0, gy + 50, title, 'tx');
    const bw = 160;
    const emb = { x: x0, y: gy + 64, w: bw, h: 44 };
    const trunk = { x: x0, y: gy + 124, w: bw, h: 44 };
    const head = { x: x0, y: gy + 184, w: bw, h: 44 };
    const mtp = { x: x0 + 225, y: gy + 94, w: 150, h: 60 };
    const loss = { x: x0 + 225, y: gy + 194, w: 150, h: 44 };
    box(emb.x, emb.y, emb.w, emb.h, [`共享 embedding：Emb(t${a.mtpInput[cfg.gradPos]})`, 'decoder_input'], 'neutral');
    box(trunk.x, trunk.y, trunk.w, trunk.h, [`主干 h${cfg.gradPos}`, 'torch.chunk 出来的 view'], 'neutral');
    box(head.x, head.y, head.w, head.h, ['输出头权重 W', '共享或 output_layer'], 'neutral');
    box(mtp.x, mtp.y, mtp.w, mtp.h, ['MTP 层参数', 'enorm/hnorm/eh_proj', 'transformer_layer/norm'], patched ? 'acc1' : 'neutral');
    box(loss.x, loss.y, loss.w, loss.h, [`CE(t${a.mtpLabel[cfg.gradPos]}) × mask`, 'MTPLossAutoScaler'], 'neutral');
    // 前向
    line(emb.x + emb.w, emb.y + 22, mtp.x, mtp.y + 16, 'fwd');
    line(trunk.x + trunk.w, trunk.y + 22, mtp.x, mtp.y + 44, 'fwd');
    line(mtp.x + mtp.w / 2, mtp.y + mtp.h, loss.x + loss.w / 2, loss.y, 'fwd');
    line(head.x + head.w, head.y + 22, loss.x, loss.y + 22, 'fwd');
    const note = { x: x0 + 390, y: gy + 64 };
    if (patched) {
      scissor((emb.x + emb.w + mtp.x) / 2, (emb.y + 22 + mtp.y + 16) / 2);
      scissor((trunk.x + trunk.w + mtp.x) / 2, (trunk.y + 22 + mtp.y + 44) / 2);
      scissor((head.x + head.w + loss.x) / 2, head.y + 22);
      text(note.x, note.y + 12, '✕ decoder_input', 'mo');
      text(note.x, note.y + 26, '   .detach()', 'mo');
      text(note.x, note.y + 58, '✕ make_viewless_tensor(', 'mo');
      text(note.x, note.y + 72, '   keep_graph=False)', 'mo');
      text(note.x, note.y + 86, '→ 新叶子，不连回主干', 'sm');
      text(note.x, note.y + 132, '✕ mtp_output_weight', 'mo');
      text(note.x, note.y + 146, '   .detach()', 'mo');
      box(x0, gy + 244, 540, 42, [['MTP loss 梯度只落在 MTP 层参数；主干、embedding、输出头只收 policy loss 的梯度', 'tx']], 'acc1');
      line(mtp.x + 20, loss.y, mtp.x + 20, mtp.y + mtp.h + 2, 'main');
    } else {
      text(note.x, note.y + 12, '不 detach', 'mo');
      text(note.x, note.y + 58, 'keep_graph=True', 'mo');
      text(note.x, note.y + 72, '（MakeViewlessTensor', 'sm');
      text(note.x, note.y + 86, '  反向原样透传）', 'sm');
      text(note.x, note.y + 132, 'weight=output_weight', 'mo');
      box(x0, gy + 244, 540, 42, [['MTP loss 梯度同时落到 MTP 层、主干、embedding 与输出头', 'tx']], 'acc2');
    }
  };
  sub(40, '上游 Megatron-LM@1dcf0daf（multi_token_prediction.py / gpt_model.py）', false);
  // 上游：反向箭头画成 acc2
  const ux = 40;
  line(ux + 225, gy + 110, ux + 162, gy + 88, 'cost');
  line(ux + 225, gy + 138, ux + 162, gy + 150, 'cost');
  line(ux + 225, gy + 222, ux + 162, gy + 210, 'cost');
  line(ux + 280, gy + 194, ux + 280, gy + 156, 'cost');
  sub(610, 'slime 镜像：1dcf0daf + docker/patch/latest/megatron.patch', true);

  // ③ CI 门禁
  const cy = gy + 310;
  rect(24, cy, 1132, 118, 'panel', 10);
  text(40, cy + 24, '③ CI 门禁 tests/test_mimo_7B_mtp_only_grad.py：先把 policy 梯度压成 0，再看剩下的梯度落在哪', 'pt');
  const steps = [
    ['--rollout-max-response-len 128', '全部 response 截断', 'neutral'],
    ['--rm-type deepscaler', '无 </think> → reward 0', 'neutral'],
    ['GRPO 组归一化', '组内全 0 → 0（分析判断）', 'neutral'],
    ['policy loss 梯度 = 0', 'kl / entropy 系数也为 0', 'neutral'],
    ['check_mtp_only_grad', '非 MTP 非零梯度数 = 0 且 MTP 的 > 0', 'acc1'],
  ];
  let sx = 40;
  steps.forEach(([l1, l2, cls], i) => {
    const w = [214, 196, 196, 196, 236][i];
    box(sx, cy + 38, w, 46, [[l1, 'mo'], [l2, 'sm']], cls);
    if (i < steps.length - 1) line(sx + w + 2, cy + 61, sx + w + 14, cy + 61, 'main');
    sx += w + 18;
  });
  text(40, cy + 104, '断言放在 optimizer.step() 之前、读 main_grad（没有则读 grad）；只要同时开 --ci-test 与 --enable-mtp-training 就会执行。上游无补丁时第一条断言应失败（分析判断，未运行）。', 'sm');

  text(24, H - 12, '阅读顺序：① 标签与 mask 的坐标 → ② 补丁的三处切断 → ③ CI 用"policy 梯度为 0"把两类梯度分开验证。', 'cap');
  c.o.push('</svg>');
  return c.o.join('\n');
}

// ---------- 图 2：发布平面 ----------
export function renderPublication(m, cfg = CFG) {
  const r = m.rename;
  const L = cfg.layout;
  const W = 1180;
  const H = 892;
  const c = canvas(
    W,
    H,
    'slime 在线 MTP 发布：PP 与 EP 下的全局改名，以及三条 transport 把同一批张量交给哪个 SGLang runner',
    `面板①在 PP=${L.pp}、EP=${L.ep}、${L.numLayers} 层 decoder、${L.mtpLayers} 层 MTP、${L.numExperts} 个 expert 的布局上展示每个 rank 的层号与 expert 偏移、三个参数的本地名到全局名到 HF 名、把 decoder 偏移错加到 MTP 的后果，以及 embedding 副本按较小 src_rank 去重。面板②展示 tensor、disk、distributed 三个入口在 SGLang v0.5.15.post1 加 slime 补丁下分别更新 target 与 draft 的情况。`,
  );
  const { rect, text, line, box } = c;
  text(24, 34, 'MTP 参数进了发布列表，但层号规则与 transport 各自决定 draft 是否真的拿到新权重', 'ti');
  text(24, 56, `示例：GLM4-MoE 结构，num_layers = ${L.numLayers}，mtp_num_layers = ${L.mtpLayers}，num_experts = ${L.numExperts}；训练 PP = ${L.pp}、EP = ${L.ep}，共 ${L.pp * L.ep} 个 rank；名字省略 module.module. 前缀`, 'su');

  // ① rank 网格
  const py = 72;
  rect(24, py, 1132, 472, 'panel', 10);
  text(40, py + 24, '① named_params_and_buffers：decoder 层号加 layer_offset、expert 下标加 expert_offset，MTP 层号保持从 0 开始', 'pt');
  const cellW = 250;
  const cellH = 118;
  const gx = 40;
  const gy = py + 60;
  text(gx + 110, gy - 8, `EP rank 0（offset ${r.ranks[0].expertOffset}）`, 'sm', 'middle');
  text(gx + 110 + cellW + 10, gy - 8, `EP rank 1（offset ${r.ranks[1].expertOffset}）`, 'sm', 'middle');
  r.ranks.forEach((k) => {
    const x = gx + k.epRank * (cellW + 10);
    const y = gy + k.stage * (cellH + 10);
    rect(x, y, cellW, cellH, k.hasMtp ? 'neutral' : 'ghost', 6);
    text(x + 10, y + 18, `rank ${k.rank}：PP ${k.stage} · EP ${k.epRank}`, 'tx');
    text(x + 10, y + 36, `layer_offset = ${k.layerOffset}，expert_offset = ${k.expertOffset}`, 'sm');
    const layers = range(r.perStage).map((l) => l + k.layerOffset);
    text(x + 10, y + 56, `decoder 本地 0–${r.perStage - 1} → 全局 ${layers[0]}–${layers[layers.length - 1]}`, 'sm');
    text(x + 10, y + 74, `expert 本地 0–${r.perEp - 1} → 全局 ${k.expertOffset}–${k.expertOffset + r.perEp - 1}`, 'sm');
    text(x + 10, y + 92, k.hasMtp ? 'mtp.layers.0 → 仍为 0，expert 加偏移' : '无 MTP（只在最后一个 PP stage 构建）', 'sm');
    text(x + 10, y + 110, k.stage === 0 || k.hasMtp ? 'embedding.word_embeddings.weight ✓' : '', 'sm');
  });
  text(gx, gy + 2 * (cellH + 10) + 24, `get_mtp_num_layers_to_build：无自定义 layout 时 MTP 只在最后一个 PP stage；`, 'sm');
  text(gx, gy + 2 * (cellH + 10) + 40, 'GPTModel.__init__：pre_process 或 mtp_process 的 stage 都建 embedding（Megatron 上游）', 'sm');

  // 右侧：三条追踪
  const tx = 570;
  let ty = gy - 4;
  const labels = {
    decoderExpert: `rank ${r.traces[0].rank} 的 decoder expert`,
    mtpExpert: `rank ${r.traces[1].rank} 的 MTP expert`,
    mtpProj: `rank ${r.traces[2].rank} 的 MTP eh_proj`,
  };
  r.traces.forEach((t) => {
    text(tx, ty + 12, labels[t.key], 'tx');
    box(tx, ty + 18, 572, 22, [[`本地  ${t.local}`, 'mo']], 'neutral', 'mo', 'start');
    box(tx, ty + 44, 572, 22, [[`全局  ${t.global}`, 'mo']], t.key === 'decoderExpert' ? 'neutral' : 'acc1', 'mo', 'start');
    box(tx, ty + 70, 572, 22, [[`HF    ${t.hf[0]}${t.hf.length > 1 ? ' …' : ''}`, 'mo']], 'neutral', 'mo', 'start');
    ty += 102;
  });
  box(tx, ty + 2, 572, 44, [[`若把 layer_offset = ${r.wrongLayer} 也加到 MTP：mtp.layers.${r.wrongLayer} → ${r.wrongHf.replace('.eh_proj.weight', '')}`, 'mo'], [`转换器还要再加 num_layers = ${L.numLayers}；HF checkpoint 只有 model.layers.0–${r.hfLayerCount - 1}`, 'sm']], 'acc2', 'mo', 'start');
  box(tx, ty + 54, 572, 44, [[`PP group {${r.ppGroup.join(', ')}}：embedding.word_embeddings.weight 在 rank ${r.embeddingHolders.join(' 与 ')} 同名`, 'mo'], [`rank ${r.embeddingHolders[r.embeddingHolders.length - 1]} 那份是 MTP 段副本；交换 PP 元数据时保留较小 src_rank = ${r.embeddingSrc}`, 'sm']], 'neutral', 'mo', 'start');

  // ② transport lanes
  const ly = py + 484;
  rect(24, ly, 1132, 308, 'panel', 10);
  text(40, ly + 24, '② 同一批 HF 张量到达哪个 runner（EAGLEWorkerV2；SGLang v0.5.15.post1 上游 + slime 补丁，均为读源码结论，未运行）', 'pt');
  const lanes = [
    {
      name: 'tensor',
      slime: ['UpdateWeightFromTensor', '--colocate 且 engine 在 actor GPU 区间内'],
      rpc: '/update_weights_from_tensor',
      sgl: ['SchedulerWeightUpdaterManager', '.update_weights_from_tensor → EAGLEWorkerV2 同名方法'],
      out: [['draft_runner ✓ → target ✓', 'acc1']],
    },
    {
      name: 'disk',
      slime: ['UpdateWeightFromDisk / …DiskDelta', 'full+disk；delta 先 /pull_weights（补丁）'],
      rpc: '/update_weights_from_disk',
      sgl: ['SchedulerWeightUpdaterManager', '.update_weights_from_disk → tp_worker，再 draft_worker'],
      out: [['target ✓ → draft ✓', 'acc1']],
    },
    {
      name: 'nccl',
      slime: ['UpdateWeightFromDistributed', '或 colocate 下越出 actor 区间的 engine'],
      rpc: '/update_weights_from_distributed',
      sgl: ['SchedulerWeightUpdaterManager', '.update_weights_from_distributed → 只调 tp_worker'],
      out: [['target ✓，draft 无调用 ✕', 'acc2']],
    },
  ];
  lanes.forEach((ln, i) => {
    const y = ly + 42 + i * 70;
    box(40, y, 260, 52, [[ln.slime[0], 'mo'], [ln.slime[1], 'sm']], 'neutral');
    line(302, y + 26, 330, y + 26, 'main');
    box(334, y + 10, 214, 32, [[ln.rpc, 'mo']], 'neutral');
    line(550, y + 26, 562, y + 26, 'main');
    box(566, y, 366, 52, [[ln.sgl[0], 'mo'], [ln.sgl[1], 'sm']], 'dep');
    line(934, y + 26, 946, y + 26, ln.name === 'nccl' ? 'cost' : 'main');
    box(950, y + 8, 192, 36, [[ln.out[0][0], 'tx']], ln.out[0][1]);
  });
  const ny = ly + 42 + 3 * 70;
  box(40, ny, 1102, 44, [['post_process_weights（sglang.patch 新增，只在 compressed-tensors 时调用）：先调 tp_worker；只有 draft_worker 有同名方法才转发，而 EAGLEWorkerV2 没有 → draft 侧跳过', 'sm'], ['--enable-multi-layer-eagle（MiMoV2、Step3p5 在 EAGLE 下自动打开）选 MultiLayerEagleWorkerV2：锁定源码里没有 update_weights_from_tensor，本图 tensor 行不覆盖它', 'sm']], 'ghost', 'sm', 'start');

  text(24, H - 12, '阅读顺序：① 名字与层号只决定"转换器能否对上 HF 名" → ② 名字对上以后，还要看 transport 入口是否把张量交给 draft runner。', 'cap');
  c.o.push('</svg>');
  return c.o.join('\n');
}

// ---------- 图 3：观测平面 ----------
export function renderAccept(m, cfg = CFG) {
  const s = m.accept;
  const W = 1180;
  const H = 650;
  const c = canvas(
    W,
    H,
    'slime spec_accept_rate：锁定镜像的键不兼容，以及键兼容时按样本等权平均的归约',
    `面板①列出 SGLang v0.5.15.post1 在请求结束时写入 meta_info 的投机指标键与 Sample.SpecInfo.add 读取的键，读不到的两个键让 rollout/spec_accept_rate 为 0。面板②用两条样本（B 由 partial rollout 的两段组成）复现 SpecInfo 累加原始计数、按样本求比值、再等权平均的过程，并与按 token 汇总的比值对照。`,
  );
  const { rect, text, line, box } = c;
  text(24, 34, 'spec_accept_rate 是"样本比值的等权平均"，而锁定镜像里它读到的分子分母都是 0', 'ti');
  text(24, 56, `示例：--sglang-speculative-num-draft-tokens ${cfg.spec.numDraftTokens}，SGLang 按 verify 次数 × (${cfg.spec.numDraftTokens} − 1) = 每次 ${s.proposedPerVerify} 个记提议 draft；两条样本，B 被 partial rollout 切成两段`, 'su');

  // ① 键对照
  const py = 72;
  rect(24, py, 1132, 232, 'panel', 10);
  text(40, py + 24, '① 锁定镜像：请求结束时 SGLang 写出的键 vs slime 读取的键', 'pt');
  const left = [
    ['spec_accept_rate / spec_accept_length', '单请求比值（slime 不读）'],
    ['spec_num_correct_drafts / spec_num_proposed_drafts', '原始计数'],
    ['spec_accepted_drafts / spec_proposed_drafts', '向后兼容别名'],
    ['spec_verify_ct', '原始计数'],
    ['completion_tokens', '原始计数'],
  ];
  text(40, py + 48, 'TokenizerManager._calculate_spec_decoding_metrics（上游；slime 补丁未改这段）', 'sm');
  left.forEach(([k, d], i) => box(40, py + 56 + i * 32, 470, 26, [[`${k}   ${d}`, 'mo']], 'dep', 'mo', 'start'));
  text(620, py + 48, 'slime/utils/types.py::Sample.SpecInfo.add：meta_info.get(key, 0)', 'sm');
  const right = [
    ['spec_accept_token_num', '无此键 → 0', 'acc2'],
    ['spec_draft_token_num', '无此键 → 0', 'acc2'],
    ['spec_verify_ct', '匹配', 'neutral'],
    ['completion_tokens', '匹配', 'neutral'],
  ];
  right.forEach(([k, d, cls], i) => box(620, py + 56 + i * 32, 300, 26, [[`${k}   ${d}`, 'mo']], cls, 'mo', 'start'));
  line(512, py + 56 + 3 * 32 + 13, 618, py + 56 + 2 * 32 + 13, 'main');
  line(512, py + 56 + 4 * 32 + 13, 618, py + 56 + 3 * 32 + 13, 'main');
  box(940, py + 56, 200, 122, [['每条样本', 'tx'], [`rate = 0 / 0 → ${fmt(s.pinned[0].rate, 1)}`, 'mo'], [`A length = ${fmt(s.pinned[0].length, 2)}`, 'mo'], [`B length = ${fmt(s.pinned[1].length, 2)}`, 'mo'], [`日志 rate = ${fmt(s.pinnedRate, 1)}`, 'mo']], 'acc2');
  text(40, py + 222, '这两个键只出现在旧的 docker/patch/v0.5.5.post1/sglang.patch；v0.5.6 起的补丁集与 latest 都不再提供。tests/test_sample.py 只用合成 meta_info 验证开关。', 'sm');

  // ② 归约
  const gy = py + 244;
  rect(24, gy, 1132, 300, 'panel', 10);
  text(40, gy + 24, '② 若键兼容：SpecInfo 先累加原始计数，再按样本求比值，_compute_spec_metrics 最后对样本等权平均', 'pt');
  const cols = [
    ['请求段', 150],
    ['spec_verify_ct  V', 150],
    ['被接受 draft  A', 150],
    [`提议 draft  D = V×${s.proposedPerVerify}`, 170],
    ['completion_tokens  C', 170],
    ['样本 rate = ΣA / ΣD', 160],
    ['样本 length = ΣC / ΣV', 150],
  ];
  const x0 = 40;
  const colX = [];
  let cx = x0;
  cols.forEach(([, w]) => {
    colX.push(cx);
    cx += w;
  });
  const hy = gy + 42;
  cols.forEach(([h, w], i) => {
    rect(colX[i], hy, w - 4, 26, 'ghost', 3);
    text(colX[i] + (w - 4) / 2, hy + 17, h, 'sm', 'middle');
  });
  let ry = hy + 32;
  const rowsOut = [];
  s.compatible.forEach((smp) => {
    smp.segments.forEach((g, j) => {
      rowsOut.push({ cells: [`${smp.name}${smp.segments.length > 1 ? ` 段 ${j + 1}` : ''}`, g.verify, g.correct, g.proposed, g.completion, '', ''], cls: 'neutral' });
    });
    if (smp.segments.length > 1) {
      rowsOut.push({ cells: [`${smp.name} 累加`, smp.verify, smp.accept, smp.draft, smp.completion, `${smp.accept}/${smp.draft} = ${fmt(smp.rate)}`, `${smp.completion}/${smp.verify} = ${fmt(smp.length, 2)}`], cls: 'acc1' });
    } else {
      rowsOut[rowsOut.length - 1].cells[5] = `${smp.accept}/${smp.draft} = ${fmt(smp.rate)}`;
      rowsOut[rowsOut.length - 1].cells[6] = `${smp.completion}/${smp.verify} = ${fmt(smp.length, 2)}`;
      rowsOut[rowsOut.length - 1].cls = 'acc1';
    }
  });
  rowsOut.forEach((row) => {
    row.cells.forEach((v, i) => {
      const w = cols[i][1];
      const cls = i >= 5 && v !== '' ? row.cls : 'neutral';
      rect(colX[i], ry, w - 4, 26, cls, 3);
      text(colX[i] + (w - 4) / 2, ry + 17, String(v), 'mo', 'middle');
    });
    ry += 32;
  });
  const [A, B] = s.compatible;
  box(40, ry + 10, 540, 58, [['slime 日志 rollout/spec_accept_rate（等权）', 'tx'], [`(${fmt(A.rate)} + ${fmt(B.rate)}) / 2 = ${fmt(s.meanRate)}`, 'mo'], [`rollout/spec_accept_length = (${fmt(A.length, 2)} + ${fmt(B.length, 2)}) / 2 = ${fmt(s.meanLength, 3)}`, 'mo']], 'acc1');
  box(600, ry + 10, 540, 58, [['对照：按 token 汇总（slime 不这样算）', 'tx'], [`(${A.accept} + ${B.accept}) / (${A.draft} + ${B.draft}) = ${fmt(s.pooledRate)}`, 'mo'], [`(${A.completion} + ${B.completion}) / (${A.verify} + ${B.verify}) = ${fmt(s.pooledLength, 3)}`, 'mo']], 'ghost');
  text(40, ry + 90, `短样本 A 与长样本 B 权重相同，所以等权 rate ${fmt(s.meanRate)} 高于按 token 的 ${fmt(s.pooledRate)}；length 含每次 verify 的 bonus token，不能当作"每次 verify 接受的 draft 数"。`, 'sm');

  text(24, H - 12, '阅读顺序：① 先确认键能读到 → ② 再按"样本等权"解读 rate 与 length；两者都不是端到端吞吐。', 'cap');
  c.o.push('</svg>');
  return c.o.join('\n');
}

export const OUTPUTS = Object.freeze({
  training: 'slime_mtp_training_isolation.svg',
  publication: 'slime_mtp_rename_publication.svg',
  accept: 'slime_mtp_accept_metrics.svg',
});

export function renderAll(cfg = CFG) {
  const m = model(cfg);
  return {
    [OUTPUTS.training]: renderTraining(m, cfg),
    [OUTPUTS.publication]: renderPublication(m, cfg),
    [OUTPUTS.accept]: renderAccept(m, cfg),
  };
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  for (const [name, svg] of Object.entries(renderAll())) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
}
