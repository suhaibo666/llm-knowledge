// 图 1：一个矩阵乘上的列并行 / 行并行。
// 图 2：TP 在 dense Transformer 训练中的前向、loss 与反向闭环。
// 图 3：Sequence Parallelism 在一对 Column→Row 区域中的前反向布局变换。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 图 1 要讲清楚：列并行复制 X、切 W 的输出维并保留 Y 分片；行并行消费 X 分片、
// 切 W 的输入维，最后把同形状 partial output 求和。上下两面板统一使用 tp=2，所有
// per-rank 尺寸从 CFG 推导，蓝色标本地保留的分片，橙色只标不可省的通信。
//
// 图 2 要讲清楚：前向从 token ids 经 vocab embedding、Transformer、LM head 到
// vocab-parallel CE 和逐 token loss；反向从 loss 回到 local logits grad、LM head dgrad AR、
// 每层 Column dgrad AR，最后到 embedding 的本地 vocab-shard wgrad。它是训练闭环，不只是模块列表。
//
// 图 3 要讲清楚：为什么沿 token/序列维分片，区域外每 rank 保存 S/t，
// 前向用 AG 恢复全 token 维、Row 出口用 RS 求和并重新分片；反向要由 Row
// mapping AG 梯度，Column backward 再做 saved-input AG 与 dgrad RS。
//
// 用法：node tools/figs/svg/megatron_tp_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  tp: 2, sequence: 8, hidden: 8, output: 12, ffn: 16, heads: 4, vocab: 16,
});
const derived = Object.freeze({
  sequencePerRank: CFG.sequence / CFG.tp,
  hiddenPerRank: CFG.hidden / CFG.tp,
  outputPerRank: CFG.output / CFG.tp,
  ffnPerRank: CFG.ffn / CFG.tp,
  gatedFfnPerRank: (2 * CFG.ffn) / CFG.tp,
  headsPerRank: CFG.heads / CFG.tp,
  vocabPerRank: CFG.vocab / CFG.tp,
});

for (const [name, value] of Object.entries(derived)) {
  if (!Number.isInteger(value)) throw new Error(`${name} must be integral, got ${value}`);
}

const esc = (value) => String(value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

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

function box(lines, x, y, w, h, cls = 'neutral') {
  const out = [rect(x, y, w, h, cls)];
  const gap = 16;
  const start = y + h / 2 - ((lines.length - 1) * gap) / 2 + 4;
  lines.forEach((line, index) => out.push(text(x + w / 2, start + index * gap, line, index === 0 ? 'tx' : 'sm', 'middle')));
  return out.join('\n');
}

function gemmBox(x, y, w, h) {
  return `<g data-role="gemm">${box(['GEMM'], x, y, w, h, 'neutral')}</g>`;
}

function fixedRow(out, nodes, y, h = 66, direction = 'lr') {
  nodes.forEach((node) => out.push(box(node.lines, node.x, y, node.w, h, node.cls)));
  for (let i = 0; i < nodes.length - 1; i += 1) {
    const left = nodes[i];
    const right = nodes[i + 1];
    if (direction === 'lr') out.push(arrow(left.x + left.w, y + h / 2, right.x, y + h / 2));
    else out.push(arrow(right.x, y + h / 2, left.x + left.w, y + h / 2));
  }
}

function tokenStrip(x, y, start, count, cls) {
  const out = [];
  for (let i = 0; i < count; i += 1) {
    out.push(rect(x + i * 38, y, 34, 30, cls, 5));
    out.push(text(x + i * 38 + 17, y + 20, `s${start + i}`, 'sm', 'middle'));
  }
  return out.join('\n');
}

function matrix(x, y, rows, cols, split, orientation) {
  const cell = Math.min(12, 104 / cols, 58 / rows);
  const out = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const shard = orientation === 'column' ? c >= split : r >= split;
      out.push(`<rect class="${shard ? 'cell1' : 'cell0'}" x="${x + c * cell}" y="${y + r * cell}" width="${cell}" height="${cell}"/>`);
    }
  }
  return { svg: out.join('\n'), width: cols * cell, height: rows * cell };
}

function renderMatrixFigure() {
  const W = 1320, H = 760, p = [];
  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="tp 等于 2 时列并行与行并行矩阵乘的分片和通信对照">`);
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, '同一个矩阵乘：列并行切输出维，行并行切输入维', 'ti'));
  p.push(text(28, 57, `示例参数：H=${CFG.hidden} · O=${CFG.output} · tp=${CFG.tp}；蓝色为 rank 本地分片，橙色为不可省的规约`, 'su'));

  const panels = [
    { y: 82, title: '① Column Parallel：复制输入，保留输出分片', sub: 'W = [W₀ | W₁]；每个 rank 独立得到输出列分片，默认不 all-gather。', kind: 'column' },
    { y: 402, title: '② Row Parallel：切分输入，求和 partial output', sub: 'W 按输入行切；每个 rank 先得到完整形状的 partial output，再 all-reduce 求和。', kind: 'row' },
  ];

  for (const panel of panels) {
    p.push(rect(20, panel.y, W - 40, 286, 'panel', 12));
    p.push(text(40, panel.y + 30, panel.title, 'pt'));
    p.push(text(40, panel.y + 50, panel.sub, 'su'));
    const isColumn = panel.kind === 'column';
    const master = matrix(61, panel.y + 82, CFG.hidden, CFG.output, isColumn ? derived.outputPerRank : derived.hiddenPerRank, panel.kind);
    p.push(master.svg);
    p.push(text(61 + master.width / 2, panel.y + 82 + master.height + 18, `逻辑权重 W: ${CFG.hidden}×${CFG.output}`, 'sm', 'middle'));

    const laneY = [panel.y + 78, panel.y + 176];
    laneY.forEach((y, rank) => {
      p.push(text(186, y + 46, `rank ${rank}`, 'rank'));
      if (isColumn) {
        p.push(box([`X: N×${CFG.hidden}`, '完整副本'], 235, y, 150, 40, 'ghost'));
        p.push(box([`W${rank === 0 ? '₀' : '₁'}: ${CFG.hidden}×${derived.outputPerRank}`, '输出维分片'], 235, y + 44, 150, 40, 'acc1'));
        p.push(arrow(385, y + 20, 420, y + 31));
        p.push(arrow(385, y + 64, 420, y + 53));
        p.push(gemmBox(420, y + 17, 96, 50));
        p.push(arrow(516, y + 42, 558, y + 42));
        p.push(box([`Y${rank === 0 ? '₀' : '₁'}: N×${derived.outputPerRank}`, '保持分片'], 558, y + 10, 170, 64, 'acc1'));
      } else {
        p.push(box([`X${rank === 0 ? '₀' : '₁'}: N×${derived.hiddenPerRank}`, '输入维分片'], 235, y, 150, 40, 'acc1'));
        p.push(box([`W${rank === 0 ? '₀' : '₁'}: ${derived.hiddenPerRank}×${CFG.output}`, '输入维分片'], 235, y + 44, 150, 40, 'acc1'));
        p.push(arrow(385, y + 20, 420, y + 31));
        p.push(arrow(385, y + 64, 420, y + 53));
        p.push(gemmBox(420, y + 17, 96, 50));
        p.push(arrow(516, y + 42, 558, y + 42));
        p.push(box([`Z${rank === 0 ? '₀' : '₁'}: N×${CFG.output}`, 'partial output'], 558, y + 10, 170, 64, 'neutral'));
      }
    });

    if (isColumn) {
      p.push(arrow(728, laneY[0] + 42, 820, panel.y + 126));
      p.push(arrow(728, laneY[1] + 42, 820, panel.y + 126));
      p.push(`<g data-role="column-default">${box(['默认：不 concat', 'Yᵣ 留在各自 rank', '直接交给 Row Parallel'], 820, panel.y + 88, 240, 76, 'acc1')}</g>`);
      p.push(arrow(1060, panel.y + 126, 1090, panel.y + 126, 'aux'));
      p.push(`<g data-role="column-optional">${box(['可选兼容出口', 'gather_output = true', 'AG → concat 完整 Y'], 1090, panel.y + 88, 190, 76, 'ghost')}</g>`);
    } else {
      p.push(arrow(728, laneY[0] + 42, 820, panel.y + 126, 'cost'));
      p.push(arrow(728, laneY[1] + 42, 820, panel.y + 126, 'cost'));
      p.push(box([`Y = Σ Zᵣ: N×${CFG.output}`, 'TP group 内 all-reduce'], 820, panel.y + 88, 240, 76, 'acc2'));
      p.push(box(['为什么必须求和', 'XW = X₀W₀ + X₁W₁', '各 rank 只算到一项'], 1090, panel.y + 88, 190, 76, 'neutral'));
      p.push(arrow(1060, panel.y + 126, 1090, panel.y + 126, 'aux'));
    }
  }

  p.push(text(28, 720, '逻辑公式使用 W 为 H×O；PyTorch 参数实际按 O×H 存储，因此 ColumnParallelLinear 在 dim 0 分片、RowParallelLinear 在 dim 1 分片。', 'cap'));
  p.push(text(28, 739, 'Column → 逐元素或按 head 的局部计算 → Row：中间分片不拼回，只在 Row 的区域出口规约。', 'cap'));
  p.push('</svg>');
  return p.join('\n');
}

function renderTransformerFigure() {
  const W = 1480, H = 1040, p = [];
  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Megatron TP 从 token 到 loss 再到 embedding 梯度的完整训练闭环">`);
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, 'TP 在 dense GPT 训练中的前向、Loss 与反向闭环', 'ti'));
  p.push(text(28, 57, `非 SP · MHA 基准 · tp=${CFG.tp} · heads/rank = ${derived.headsPerRank} · FFN/rank = ${derived.ffnPerRank} · vocab/rank = ${derived.vocabPerRank}`, 'su'));

  p.push(rect(20, 86, W - 40, 164, 'panel', 12));
  p.push(text(40, 116, '① 训练前向：token 到逐 token loss', 'pt'));
  fixedRow(p, [
    { x: 42, w: 120, lines: ['token ids', '每 rank 相同'], cls: 'ghost' },
    { x: 186, w: 190, lines: ['VocabParallelEmbedding', `vocab rows ${CFG.vocab}→${derived.vocabPerRank}/rank`, 'lookup 后 AR'], cls: 'acc1' },
    { x: 400, w: 178, lines: ['Transformer ×L', '下方展开单层'], cls: 'neutral' },
    { x: 602, w: 150, lines: ['LM head', 'Column Parallel'], cls: 'acc1' },
    { x: 776, w: 144, lines: ['分片 logits', `N×${derived.vocabPerRank}/rank`], cls: 'acc1' },
    { x: 944, w: 228, lines: ['VocabParallelCE', 'MAX AR + 2×SUM AR', '不聚合 N×V logits'], cls: 'acc2' },
    { x: 1196, w: 210, lines: ['逐 token loss', 'N 个标量', '各 TP rank 等价'], cls: 'neutral' },
  ], 146, 76);

  p.push(rect(20, 274, W - 40, 300, 'panel', 12));
  p.push(text(40, 304, '② Transformer layer 前向展开：两个 Column→local→Row 闭合区', 'pt'));
  p.push(text(48, 365, 'Attention', 'rank'));
  fixedRow(p, [
    { x: 136, w: 178, lines: ['Input LayerNorm', '完整 H，逐 token'], cls: 'neutral' },
    { x: 338, w: 178, lines: ['linear_qkv', 'Column Parallel'], cls: 'acc1' },
    { x: 540, w: 178, lines: ['Core Attention', `${derived.headsPerRank} heads/rank`], cls: 'neutral' },
    { x: 742, w: 178, lines: ['linear_proj', 'Row Parallel'], cls: 'acc1' },
    { x: 944, w: 178, lines: ['all-reduce', 'N×H partial sum'], cls: 'acc2' },
    { x: 1146, w: 220, lines: ['BDA + residual', 'N×H replicated'], cls: 'neutral' },
  ], 330, 70);
  p.push(text(48, 477, 'MLP', 'rank'));
  fixedRow(p, [
    { x: 136, w: 178, lines: ['Pre-MLP Norm', '完整 H，逐 token'], cls: 'neutral' },
    { x: 338, w: 178, lines: ['linear_fc1', `GeLU preact/rank = ${derived.ffnPerRank}`, `SwiGLU preact/rank = ${derived.gatedFfnPerRank}`], cls: 'acc1' },
    { x: 540, w: 178, lines: ['GeLU / SwiGLU', `activation/rank = ${derived.ffnPerRank}`], cls: 'neutral' },
    { x: 742, w: 178, lines: ['linear_fc2', 'Row Parallel'], cls: 'acc1' },
    { x: 944, w: 178, lines: ['all-reduce', 'N×H partial sum'], cls: 'acc2' },
    { x: 1146, w: 220, lines: ['BDA + residual', 'N×H replicated'], cls: 'neutral' },
  ], 442, 70);
  p.push(box(['通信账', '朴素 TP：每层前向 2 次规约', '反向 2 次 Column dgrad AR'], 1120, 520, 286, 42, 'ghost'));

  p.push(rect(20, 598, W - 40, 192, 'panel', 12));
  p.push(text(40, 628, '③ 训练反向：loss 梯度右向左返回参数分片', 'pt'));
  fixedRow(p, [
    { x: 42, w: 190, lines: ['Embedding wgrad local', '只更新命中的 vocab rows', '前向 AR 的 backward 为 identity'], cls: 'acc1' },
    { x: 266, w: 220, lines: ['Transformer backward', '每层 qkv/fc1 dgrad AR', 'Row dgrad 保持分片'], cls: 'neutral' },
    { x: 520, w: 190, lines: ['LM head dgrad AR', '求和各 vocab shard', '本地计算 weight grad'], cls: 'acc2' },
    { x: 744, w: 178, lines: ['sharded logits grad', `N×${derived.vocabPerRank}/rank`, '不聚合全词表'], cls: 'acc1' },
    { x: 956, w: 230, lines: ['CE backward：local logits grad', 'softmax shard 减 owner target', '无额外 TP collective'], cls: 'neutral' },
    { x: 1220, w: 186, lines: ['loss grad', 'N 个上游标量', '每 rank 同形'], cls: 'ghost' },
  ], 660, 88, 'rl');

  p.push(rect(20, 814, W - 40, 154, 'panel', 12));
  p.push(text(40, 844, '④ 词表边界的选择与代价', 'pt'));
  p.push(box(['Embedding 沿 vocab rows 切', '避免每 rank 保存整表', '代价：forward 1 AR，或 SP 下 1 RS'], 48, 866, 416, 82, 'neutral'));
  p.push(box(['LM head 沿 vocab outputs 切', '避免生成/聚合 N×V logits', '代价：backward hidden dgrad 1 AR'], 532, 866, 416, 82, 'neutral'));
  p.push(box(['CE 直接消费 vocab shards', '只恢复 max、target logit、exp sum', '代价：forward 3 次 N-元素规约'], 1016, 866, 416, 82, 'acc2'));
  p.push(text(28, 1004, '蓝色：本地分片计算；橙色：TP collective。图画逻辑边界；linear+CE 融合路径可不显式 materialize 分片 logits。', 'cap'));
  p.push(text(28, 1024, '基准路径中，Attention/MLP 的 Row 前向 AR 与 Column 反向 dgrad AR 构成每层 2+2 次规约。', 'cap'));
  p.push('</svg>');
  return p.join('\n');
}

function renderSequenceParallelFigure() {
  const W = 1480, H = 930, p = [];
  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Sequence Parallelism 沿序列维分片时 Column Row 区域的前反向布局和通信">`);
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, 'Sequence Parallelism：在 TP linear 区域之外切 token 维', 'ti'));
  p.push(text(28, 57, `示例：S=${CFG.sequence} · H=${CFG.hidden} · tp=${CFG.tp}；每 rank 保存 S/t=${derived.sequencePerRank} 个 token，linear 仍按 hidden/output 维 TP 切权重`, 'su'));

  p.push(rect(20, 82, W - 40, 142, 'panel', 12));
  p.push(text(40, 112, '① 为什么切序列维', 'pt'));
  p.push(text(50, 151, 'rank 0', 'rank'));
  p.push(tokenStrip(112, 130, 0, derived.sequencePerRank, 'acc1'));
  p.push(text(112 + derived.sequencePerRank * 38 + 12, 151, `rank local：${derived.sequencePerRank}×B×${CFG.hidden}`, 'dim'));
  p.push(text(50, 198, 'rank 1', 'rank'));
  p.push(tokenStrip(112, 177, derived.sequencePerRank, derived.sequencePerRank, 'ghost'));
  p.push(text(112 + derived.sequencePerRank * 38 + 12, 198, `rank local：${derived.sequencePerRank}×B×${CFG.hidden}`, 'dim'));
  p.push(box(['选 token/序列维', 'LayerNorm、dropout、residual 按 token 独立', '每份仍保留完整 H'], 640, 126, 360, 72, 'acc1'));
  p.push(box(['不选 hidden 维', 'LayerNorm mean/variance 会跨 rank', '非 GEMM 区域需额外规约'], 1044, 126, 360, 72, 'ghost'));

  p.push(rect(20, 246, W - 40, 208, 'panel', 12));
  p.push(text(40, 276, '② 前向：区域外分片，区域内恢复全 token 维', 'pt'));
  fixedRow(p, [
    { x: 40, w: 178, lines: [`rank local：${derived.sequencePerRank}×B×${CFG.hidden}`, 'LN / dropout / residual'], cls: 'acc1' },
    { x: 238, w: 158, lines: [`AG → ${CFG.sequence}×B×${CFG.hidden}`, '恢复 token 维'], cls: 'acc2' },
    { x: 416, w: 178, lines: ['Column Parallel', '本 rank 权重分片'], cls: 'acc1' },
    { x: 614, w: 168, lines: ['rank-local compute', 'head 或 FFN channels'], cls: 'neutral' },
    { x: 802, w: 178, lines: ['Row Parallel', '产生 N×H partial'], cls: 'acc1' },
    { x: 1000, w: 158, lines: [`RS → ${derived.sequencePerRank}×B×${CFG.hidden}`, '求和 + 分 token'], cls: 'acc2' },
    { x: 1178, w: 222, lines: [`rank local：${derived.sequencePerRank}×B×${CFG.hidden}`, 'BDA / next norm'], cls: 'acc1' },
  ], 316, 78);
  p.push(text(40, 428, '本地 GEMM 仍看到 S×B 个 token：SP 不再切 GEMM 计算，它切的是区域外需保存的非 GEMM 激活。', 'cap'));

  p.push(rect(20, 478, W - 40, 264, 'panel', 12));
  p.push(text(40, 508, '③ 反向：右向左恢复梯度，Column wgrad 另需 saved input', 'pt'));
  fixedRow(p, [
    { x: 40, w: 178, lines: [`rank-local input grad`, `${derived.sequencePerRank}×B×${CFG.hidden}`], cls: 'acc1' },
    { x: 238, w: 158, lines: ['dgrad RS', `RS → ${derived.sequencePerRank}×B×${CFG.hidden}`], cls: 'acc2' },
    { x: 416, w: 178, lines: ['Column dgrad GEMM', '各 rank 一份贡献'], cls: 'acc1' },
    { x: 614, w: 168, lines: ['rank-local backward', 'activation / attention'], cls: 'neutral' },
    { x: 802, w: 178, lines: ['Row dgrad GEMM', '返回 hidden shard'], cls: 'acc1' },
    { x: 1000, w: 178, lines: ['Row mapping AG', `${derived.sequencePerRank}×BH → ${CFG.sequence}×BH`], cls: 'acc2' },
    { x: 1198, w: 202, lines: ['rank-local output grad', `${derived.sequencePerRank}×B×${CFG.hidden}`], cls: 'ghost' },
  ], 548, 78, 'rl');
  p.push(box(['Column wgrad 支路', `saved-input AG：${derived.sequencePerRank}×BH → ${CFG.sequence}×BH`], 350, 654, 300, 58, 'acc2'));
  p.push(arrow(650, 683, 710, 683, 'aux'));
  p.push(box(['wgrad GEMM', '与 dgrad RS 可重叠'], 710, 654, 260, 58, 'neutral'));
  p.push(text(994, 684, 'saved-input AG 先 wait，才能消费完整 input 计算 wgrad', 'cap'));

  p.push(rect(20, 766, W - 40, 118, 'panel', 12));
  p.push(text(40, 796, '④ 收益与代价', 'pt'));
  p.push(box(['显存', '非 GEMM 激活约 ÷tp', '权重分片由 TP 本身提供'], 52, 812, 390, 56, 'acc1'));
  p.push(box(['计算', 'linear GEMM 不再 ÷tp', 'AG 后仍处理全 token 维'], 524, 812, 390, 56, 'neutral'));
  p.push(box(['通信 / 同步', '每对前向 AG+RS；反向 AG+AG+RS', '更多 launch 与等待点'], 996, 812, 390, 56, 'acc2'));
  p.push(text(28, 911, '实线是布局闭环；虚线是 Column wgrad 的 saved-input 支路。AG/RS 都在同一 TP group 内沿第一维执行。', 'cap'));
  p.push('</svg>');
  return p.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
mkdirSync(outputDir, { recursive: true });

const outputs = new Map([
  ['megatron_tp_matrix_partition.svg', renderMatrixFigure()],
  ['megatron_tp_transformer_layer.svg', renderTransformerFigure()],
  ['megatron_tp_sequence_parallel.svg', renderSequenceParallelFigure()],
]);

for (const [name, svg] of outputs) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
console.log([...outputs.keys()].join('\n'));
