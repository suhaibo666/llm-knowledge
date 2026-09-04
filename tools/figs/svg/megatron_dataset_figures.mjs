// 图 1：GPTDataset 的三级索引如何把一个整数下标翻译成一段跨文档 token 切片，
//        以及那条 S+ε 缓冲区如何变成 tokens / labels 两个重叠视图。
// 图 2：同一组文档、同一个 S 下，隐式打包与显式打包两条数据面的并排回放。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要讲清楚：`GPTDataset.__getitem__` 的决定性变换是"下标 → 跨文档 token 切片"，
// 而这一步全部由查表完成、没有任何扫描。
// 布局（自上而下三个面板）：
// - 面板 ①：三跳查表。shuffle_index[k] → j；sample_index[j] = (2,300)；
//   sample_index[j+1] = (5,24)；再由 document_index[2..5] 拿到 4 个文档号。
//   右侧 .zoom 标"三跳全是查表 → O(1) 随机读"。
// - 面板 ②：一条横向文档带，宽度按文档长度**按比例**画（740/260/300/880），
//   两条竖直切口分别落在 (2,300) 与 (5,24)。带内落在样本 j 里的部分用 acc1，
//   两侧邻居样本的部分用 ghost —— 一眼看出"文档是变长的、样本边界不认文档边界"。
//   带下四张卡片给出四段的区间与长度 440/260/300/25，右侧一张 acc1 卡片给出
//   Σ = 1025 = S+ε。所有数字由 CFG 复算 build_sample_idx 的循环得出。
// - 面板 ③：那条 S+ε 缓冲区。**只画一排格子**（index 0..4 与 1020..1024，中间省略），
//   下面用两条横条表示 tokens = text[:-1] 与 labels = text[1:] ——
//   两条横条错开正好一格，覆盖同一排格子，因此图上表达的是"一块缓冲区、两个重叠视图"，
//   不是两份拷贝。格子配色：两个视图共享的 [1,1024) 用 cell1，仅 tokens 的 0 用 cell0，
//   仅 labels 的 1024（即 ε）用 acc2。
// 图注给不变量：三索引复合成一次 O(1) 随机读；相邻样本步长 S、窗口 S+ε，
// 重叠恰好 ε 个 token —— 缓冲区第 1024 个 token 就是样本 j+1 的第 0 个。
//
// 图 2 要讲清楚：同一组文档、同一个 S，两条打包路径各自付什么、各自留下什么。
// 两条 lane 用同一个 px/token 比例尺，所以"切断"与"padding"两种浪费可以直接目测比较。
// 布局（自上而下三个面板）：
// - 面板 ⓪：共同输入 —— 4 篇文档（740/260/300/880，末尾已含预处理追加的 EOD）。
// - 面板 ①：隐式路径。token 流每 S 切一刀；样本 j 两侧画出被切走的半篇文档（ghost），
//   样本 j 内按 EOD 分成 4 段（440/260/300/24），EOD 位置用 acc2 虚线标出，
//   下方 position_ids 行显示 reset_position_ids 让每段从 0 重数，
//   右侧 .zoom 缩略图画出 reset_attention_mask 清左下块之后的 1024×1024 掩码块结构。
//   代价卡（acc2）：2 篇文档被切断；收益卡（acc1）：1024/1024 行都是真实 token。
// - 面板 ②：显式路径。同样 4 篇文档各自是一条完整样本，先按 g 对齐、再贪心装桶、
//   最后逐 microbatch 尾部对齐。两条 microbatch 横条里 acc1 是有效 token、
//   acc2 是死槽；下方逐条打印 cu_seqlens 与 cu_seqlens_padded；
//   右侧两张 .zoom 卡分别给出两级 padding 的逐项加法。
//   代价卡（acc2）：死槽 384/2560 = 15.0%；收益卡（acc1）：0 篇被切断。
// 图注给出这张图要证的那笔交易：切断 2 篇换 0% padding，或 0 篇被切换 15.0% 死槽。
//
// 硬要求：图上每个数字都由下面的 CFG 算出来，不手写。隐式一侧复算
// helpers.cpp::build_sample_idx 的 while 循环；显式一侧复算
// SFTDataset._calculate_padding_divisor → VarlenDataset.__getitem__ 的 g 对齐
// → DpBalancedScheduler.get_groups_and_subsamples 的贪心装桶
// → _pack_sequences 的两套 cumsum → pad_sequence_for_thd 的尾部对齐与 dummy 序列。
//
// 用法：node tools/figs/svg/megatron_dataset_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  // GPTDataset 侧
  sequenceLength: 1024,
  addExtraToken: 1,
  // document_index[2..5] 指向的四篇文档的 token 数（末尾已含预处理追加的 EOD）
  documentTokens: Object.freeze([740, 260, 300, 880]),
  firstDocumentSlot: 2,
  beginOffset: 300,
  // 显式打包侧的并行几何
  contextParallel: 4,
  sequenceParallel: 8,
  dataParallel: 1,
  dynamicContextParallel: false,
  maxSeqlenPerDpCpRank: 512,
  packedSeqAlignment: 64,
  microbatchGroupSizePerVpStage: 1,
});

// ---------------------------------------------------------------------------
// 1. 隐式路径：复算 helpers.cpp::build_sample_idx 的取样窗口
// ---------------------------------------------------------------------------

function buildSampleWindow() {
  const eps = CFG.addExtraToken;
  const parts = [];
  let remaining = CFG.sequenceLength + eps;
  let slot = 0;
  let offset = CFG.beginOffset;
  let endSlot = 0;
  let endOffset = 0;

  while (remaining !== 0) {
    const documentLength = CFG.documentTokens[slot] - offset;
    remaining -= documentLength;
    if (remaining <= 0) {
      const advance = remaining + documentLength - eps;
      parts.push({ slot, offset, length: advance + eps });
      endSlot = slot;
      endOffset = advance;
      remaining = 0;
    } else {
      parts.push({ slot, offset, length: documentLength });
      slot += 1;
      offset = 0;
      if (slot >= CFG.documentTokens.length) throw new Error('语料不足以切出一条完整样本');
    }
  }
  return { parts, endSlot, endOffset };
}

const sampleWindow = buildSampleWindow();
const windowTokens = sampleWindow.parts.reduce((sum, part) => sum + part.length, 0);
if (windowTokens !== CFG.sequenceLength + CFG.addExtraToken) {
  throw new Error(`窗口长度必须是 S+ε，得到 ${windowTokens}`);
}

const docStarts = CFG.documentTokens.reduce((acc, length) => {
  acc.push(acc[acc.length - 1] + length);
  return acc;
}, [0]);
const corpusTokens = docStarts[docStarts.length - 1];
const windowStartFlat = CFG.beginOffset;
const strideEndFlat = windowStartFlat + CFG.sequenceLength;

// 样本 j 内部按 EOD 切出的段（作用在长度为 S 的 tokens 上，ε 不在其中）
const eodSegments = (() => {
  const segments = [];
  let cursor = 0;
  sampleWindow.parts.forEach((part, index) => {
    const last = index === sampleWindow.parts.length - 1;
    const length = last ? CFG.sequenceLength - cursor : part.length;
    segments.push({ start: cursor, length, slot: part.slot, hasEod: !last });
    cursor += length;
  });
  return segments;
})();
const eodIndices = eodSegments.filter((s) => s.hasEod).map((s) => s.start + s.length - 1);
if (eodSegments.reduce((sum, s) => sum + s.length, 0) !== CFG.sequenceLength) {
  throw new Error('EOD 分段之和必须等于 S');
}

// ---------------------------------------------------------------------------
// 2. 显式路径：复算 g 对齐 → 贪心装桶 → 两套 cumsum → 尾部对齐
// ---------------------------------------------------------------------------

function paddingDivisor() {
  const cpPad = CFG.dynamicContextParallel
    ? CFG.dataParallel * CFG.contextParallel * 2
    : (CFG.contextParallel > 1 ? CFG.contextParallel * 2 : 1);
  const tpPad = CFG.sequenceParallel > 0 ? CFG.sequenceParallel : 1;
  return cpPad * tpPad;
}

const padGranularity = paddingDivisor();
// VarlenDataset 先补 EOD、再做 next-token 左移，因此一条 L token 的文档给出 L-1 的有效长度
const originalLengths = CFG.documentTokens.map((length) => length - 1);
const paddedLengths = originalLengths.map(
  (length) => Math.ceil(length / padGranularity) * padGranularity,
);

function greedyBuckets() {
  const limit = CFG.maxSeqlenPerDpCpRank * CFG.contextParallel;
  const buckets = [];
  let current = [];
  let sum = 0;
  paddedLengths.forEach((length, index) => {
    if (sum + length <= limit) {
      current.push(index);
      sum += length;
    } else {
      buckets.push(current);
      current = [index];
      sum = length;
    }
  });
  if (current.length > 0) buckets.push(current);
  const multiple = CFG.dataParallel * CFG.microbatchGroupSizePerVpStage;
  if (buckets.length % multiple !== 0) throw new Error('示例配置应恰好凑齐整数个 microbatch');
  return buckets;
}

function cumulative(values) {
  return values.reduce((acc, value) => {
    acc.push(acc[acc.length - 1] + value);
    return acc;
  }, [0]);
}

const microbatches = greedyBuckets().map((bucket) => {
  const originals = bucket.map((index) => originalLengths[index]);
  const padded = bucket.map((index) => paddedLengths[index]);
  const cuSeqlens = cumulative(originals);
  const cuSeqlensPadded = cumulative(padded);
  const globalActual = cuSeqlensPadded[cuSeqlensPadded.length - 1];
  if (globalActual % CFG.contextParallel !== 0) throw new Error('打包总长必须可被 cp 整除');
  const localActual = globalActual / CFG.contextParallel;
  const localTarget = Math.ceil(localActual / CFG.packedSeqAlignment) * CFG.packedSeqAlignment;
  const globalTarget = localTarget * CFG.contextParallel;
  const dummy = globalTarget - globalActual;
  if (dummy % (2 * CFG.contextParallel) !== 0) throw new Error('zigzag 要求 dummy 可被 2·cp 整除');
  const perSamplePad = padded.reduce((sum, value, i) => sum + (value - originals[i]), 0);
  return {
    members: bucket,
    originals,
    padded,
    // append_dummy_seq + 序列间已有物理空隙：dummy 的有效长度与物理长度都等于尾长
    cuSeqlens: [...cuSeqlens, cuSeqlens[cuSeqlens.length - 1] + dummy],
    cuSeqlensPadded: [...cuSeqlensPadded, globalTarget],
    maxSeqlen: Math.max(...padded, dummy),
    localActual,
    localTarget,
    globalActual,
    globalTarget,
    dummy,
    perSamplePad,
    deadSlots: perSamplePad + dummy,
  };
});

const explicitRows = microbatches.reduce((sum, mb) => sum + mb.globalTarget, 0);
const explicitReal = originalLengths.reduce((sum, value) => sum + value, 0);
const explicitDead = microbatches.reduce((sum, mb) => sum + mb.deadSlots, 0);
const explicitSamplePad = microbatches.reduce((sum, mb) => sum + mb.perSamplePad, 0);
const explicitTailPad = microbatches.reduce((sum, mb) => sum + mb.dummy, 0);
if (explicitReal + explicitDead !== explicitRows) throw new Error('显式路径的行数账不平');
const explicitWaste = (100 * explicitDead) / explicitRows;
// 被切断的文档：窗口起点落在文档中间的那篇，加上窗口终点落在文档中间的那篇
const splitDocuments = (CFG.beginOffset > 0 ? 1 : 0)
  + (sampleWindow.endOffset > 0 ? 1 : 0);

// ---------------------------------------------------------------------------
// 3. 绘图原语（class 名与色值沿用 megatron_tp_figures.mjs，不另起一套）
// ---------------------------------------------------------------------------

const esc = (value) => String(value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

const round = (value) => Math.round(value * 100) / 100;

const sharedStyle = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .card{fill:#fff;stroke:#E4E7EC;stroke-width:1.2}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .zoom{fill:#FBFCFE;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:4 3}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
  .cost{fill:none;stroke:#C3651F;stroke-width:2;marker-end:url(#arrowCost)}
  .cut{fill:none;stroke:#2563EB;stroke-width:2}
  .cutcost{fill:none;stroke:#C3651F;stroke-width:1.6;stroke-dasharray:4 3}
  .lead{fill:none;stroke:#AEB6C2;stroke-width:1.1}
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
  return `<rect class="${cls}" x="${round(x)}" y="${round(y)}" width="${round(w)}" height="${round(h)}" rx="${radius}"/>`;
}

function text(x, y, value, cls = 'tx', anchor = 'start') {
  return `<text class="${cls}" x="${round(x)}" y="${round(y)}" text-anchor="${anchor}">${esc(value)}</text>`;
}

function line(x1, y1, x2, y2, cls = 'lead') {
  return `<path class="${cls}" d="M ${round(x1)} ${round(y1)} L ${round(x2)} ${round(y2)}"/>`;
}

function arrow(x1, y1, x2, y2, cls = 'main') {
  return `<path class="${cls}" d="M ${round(x1)} ${round(y1)} L ${round(x2)} ${round(y2)}"/>`;
}

function box(lines, x, y, w, h, cls = 'neutral') {
  const out = [rect(x, y, w, h, cls)];
  const gap = 16;
  const start = y + h / 2 - ((lines.length - 1) * gap) / 2 + 4;
  lines.forEach((value, index) => out.push(
    text(x + w / 2, start + index * gap, value, index === 0 ? 'tx' : 'sm', 'middle'),
  ));
  return out.join('\n');
}

function zoomCard(lines, x, y, w, h) {
  const out = [rect(x, y, w, h, 'zoom', 10)];
  const gap = 19;
  const start = y + h / 2 - ((lines.length - 1) * gap) / 2 + 4;
  lines.forEach((value, index) => out.push(
    text(x + 14, start + index * gap, value, index === 0 ? 'tx' : 'sm'),
  ));
  return out.join('\n');
}

function fixedRow(out, nodes, y, h = 66) {
  nodes.forEach((node) => out.push(box(node.lines, node.x, y, node.w, h, node.cls)));
  for (let i = 0; i < nodes.length - 1; i += 1) {
    const left = nodes[i];
    const right = nodes[i + 1];
    out.push(arrow(left.x + left.w, y + h / 2, right.x, y + h / 2));
  }
}

// ---------------------------------------------------------------------------
// 图 1：三级索引与重叠视图
// ---------------------------------------------------------------------------

function renderSampleIndexFigure() {
  const W = 1400, H = 1012, p = [];
  const S = CFG.sequenceLength, eps = CFG.addExtraToken;
  const slotOf = (index) => CFG.firstDocumentSlot + index;

  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="GPTDataset 三级索引把一个下标翻译成跨文档 token 切片，并把 S 加 1 的缓冲区变成 tokens 与 labels 两个重叠视图">`);
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, '三级索引：把一个下标翻译成一段跨文档的 token 切片', 'ti'));
  p.push(text(28, 57, `S=${S} · ε=${eps} · sample_index[j]=(${slotOf(sampleWindow.parts[0].slot)}, ${CFG.beginOffset}) · sample_index[j+1]=(${slotOf(sampleWindow.endSlot)}, ${sampleWindow.endOffset}) · 文档长度 ${CFG.documentTokens.join(' / ')}`, 'su'));

  // 面板 ① 三跳查表
  p.push(rect(20, 82, W - 40, 150, 'panel', 12));
  p.push(text(40, 112, '① 三次查表，没有扫描', 'pt'));
  fixedRow(p, [
    { x: 48, w: 208, cls: 'neutral', lines: ['shuffle_index[k]', '决定喂入顺序', '→ 得到 j'] },
    { x: 296, w: 250, cls: 'acc1', lines: [`sample_index[j] = (${slotOf(sampleWindow.parts[0].slot)}, ${CFG.beginOffset})`, '(进入 document_index 的下标 i,', '该文档内的 token 偏移 o)'] },
    { x: 586, w: 250, cls: 'acc1', lines: [`sample_index[j+1] = (${slotOf(sampleWindow.endSlot)}, ${sampleWindow.endOffset})`, '下一条样本的起点', '两行正好夹住第 j 条样本'] },
    { x: 876, w: 230, cls: 'neutral', lines: [`document_index[${slotOf(0)}..${slotOf(CFG.documentTokens.length - 1)}]`, `${sampleWindow.parts.length} 个文档号`, '逐段 IndexedDataset.get'] },
  ], 132, 74);
  p.push(arrow(1106, 169, 1130, 169, 'aux'));
  p.push(zoomCard(['三跳全是查表', '不扫描 token 流 → O(1) 随机读', '索引建好即落盘，跨 rank 一致'], 1130, 130, 230, 78));

  // 面板 ② 文档带与两条切口
  const panelY = 248;
  p.push(rect(20, panelY, W - 40, 300, 'panel', 12));
  p.push(text(40, panelY + 30, '② 两条切口把连续 token 流切出一条样本', 'pt'));

  const bandX = 60, bandW = 1240, bandY = panelY + 82, bandH = 56;
  const k = bandW / corpusTokens;
  const px = (flat) => bandX + flat * k;

  CFG.documentTokens.forEach((length, index) => {
    const start = docStarts[index];
    const end = docStarts[index + 1];
    const pieces = [];
    if (start < windowStartFlat) pieces.push([start, Math.min(end, windowStartFlat), 'ghost']);
    const inStart = Math.max(start, windowStartFlat);
    const inEnd = Math.min(end, strideEndFlat);
    if (inEnd > inStart) pieces.push([inStart, inEnd, 'acc1']);
    if (end > strideEndFlat) pieces.push([Math.max(start, strideEndFlat), end, 'ghost']);
    pieces.forEach(([a, b, cls]) => p.push(rect(px(a), bandY, (b - a) * k, bandH, cls, 4)));
    p.push(text((px(start) + px(end)) / 2, bandY - 12, `document_index[${slotOf(index)}]`, 'sm', 'middle'));
    p.push(text((px(start) + px(end)) / 2, bandY + 34, `${length} token`, 'sm', 'middle'));
    if (index > 0) p.push(line(px(start), bandY - 6, px(start), bandY + bandH + 6));
  });

  p.push(line(px(windowStartFlat), bandY - 24, px(windowStartFlat), bandY + bandH + 28, 'cut'));
  p.push(line(px(strideEndFlat), bandY - 24, px(strideEndFlat), bandY + bandH + 28, 'cut'));
  p.push(text(px(windowStartFlat) + 8, bandY + bandH + 38, `切口 1：sample_index[j] = (${slotOf(0)}, ${CFG.beginOffset})`, 'dim'));
  p.push(text(px(strideEndFlat) + 8, bandY + bandH + 38, `切口 2：sample_index[j+1] = (${slotOf(sampleWindow.endSlot)}, ${sampleWindow.endOffset})`, 'dim'));

  const cardY = panelY + 190, cardW = 200, cardGap = 16;
  sampleWindow.parts.forEach((part, index) => {
    const to = part.offset + part.length;
    p.push(box([
      `段 ${index + 1}`,
      `document_index[${slotOf(part.slot)}][${part.offset}:${to})`,
      index === sampleWindow.parts.length - 1 ? `${part.length - eps}+ε = ${part.length} token` : `${part.length} token`,
    ], bandX + index * (cardW + cardGap), cardY, cardW, 68, 'neutral'));
  });
  p.push(box([
    `Σ = ${windowTokens} = S+ε`,
    'numpy.concatenate → (1025,) int64',
    '零拷贝只读视图 → 一次拷贝并加宽',
  ], bandX + 4 * (cardW + cardGap), cardY, 376, 68, 'acc1'));
  p.push(text(40, panelY + 274, `文档是变长的，样本边界不认文档边界：document_index[${slotOf(0)}] 的前 ${CFG.beginOffset} 个 token 属于样本 j−1，document_index[${slotOf(sampleWindow.endSlot)}] 的后 ${CFG.documentTokens[sampleWindow.endSlot] - sampleWindow.endOffset} 个属于样本 j+1。`, 'sm'));
  p.push(text(40, panelY + 290, `切口 2 同时是样本 j+1 的起点：样本 j 在这里多读 ε=${eps} 个 token，因此相邻两条样本在扁平流上重叠恰好 ${eps} 个 token。`, 'costtx'));

  // 面板 ③ 一块缓冲区，两个重叠视图
  const bufY = 568;
  p.push(rect(20, bufY, W - 40, 380, 'panel', 12));
  p.push(text(40, bufY + 30, '③ 一块缓冲区，两个错开一格的视图', 'pt'));

  const cellW = 56, cellH = 46, cellY = bufY + 62;
  const headIndices = [0, 1, 2, 3, 4];
  const tailIndices = [S - 4, S - 3, S - 2, S - 1, S];
  const headX = 200, gap = 84;
  const tailX = headX + headIndices.length * cellW + gap;
  const cellX = (index) => {
    const head = headIndices.indexOf(index);
    if (head >= 0) return headX + head * cellW;
    return tailX + tailIndices.indexOf(index) * cellW;
  };
  [...headIndices, ...tailIndices].forEach((index) => {
    const cls = index === 0 ? 'cell0' : (index === S ? 'acc2' : 'cell1');
    const x = cellX(index);
    p.push(`<rect class="${cls}" x="${round(x)}" y="${cellY}" width="${cellW}" height="${cellH}"/>`);
    p.push(text(x + cellW / 2, cellY - 10, `${index}`, 'sm', 'middle'));
  });
  p.push(text(headX + headIndices.length * cellW + gap / 2, cellY + cellH / 2 + 6, '⋯', 'pt', 'middle'));
  p.push(text(60, cellY + cellH / 2 + 5, 'text', 'tx'));
  p.push(text(60, cellY + cellH / 2 + 22, `(${windowTokens},) int64`, 'dim'));

  const tokensLeft = cellX(0), tokensRight = cellX(S - 1) + cellW;
  const labelsLeft = cellX(1), labelsRight = cellX(S) + cellW;
  p.push(rect(tokensLeft, cellY + cellH + 26, tokensRight - tokensLeft, 32, 'acc1', 6));
  p.push(text((tokensLeft + tokensRight) / 2, cellY + cellH + 47, `tokens = text[:-1]   (${S},)`, 'tx', 'middle'));
  p.push(rect(labelsLeft, cellY + cellH + 70, labelsRight - labelsLeft, 32, 'acc1', 6));
  p.push(text((labelsLeft + labelsRight) / 2, cellY + cellH + 91, `labels = text[1:]   (${S},)`, 'tx', 'middle'));
  p.push(line(cellX(S) + cellW / 2, cellY + cellH + 4, cellX(S) + cellW / 2, cellY + cellH + 70, 'cutcost'));

  p.push(zoomCard([
    '一块缓冲区，两个视图',
    `重叠区间 [1, ${S})：${S - 1} 个 token 共享同一段内存`,
    '.contiguous() 对一维 step-1 切片原样返回自身',
    'tokens[tokens == pad] = 0 直接改这块缓冲区',
    '（pad→0 幂等，所以重叠写入无害）',
  ], 900, cellY - 16, 440, 132));
  p.push(box(['仅 tokens', `下标 0`], 200, bufY + 268, 190, 48, 'neutral'));
  p.push(box(['两个视图共享', `下标 1 … ${S - 1}`], 406, bufY + 268, 250, 48, 'acc1'));
  p.push(box(['仅 labels（ε）', `下标 ${S}`], 672, bufY + 268, 210, 48, 'acc2'));
  p.push(text(900, bufY + 288, `第 ${S} 个 token 同时是样本 j+1 的第 0 个：`, 'sm'));
  p.push(text(900, bufY + 306, `sample_index[j+1] 指回 document_index[${slotOf(sampleWindow.endSlot)}] 偏移 ${sampleWindow.endOffset}。`, 'sm'));

  p.push(text(28, 970, `不变量：三索引复合成一次 O(1) 随机读——shuffle 定顺序、sample 定切口、document 定文档号，三跳之后才第一次碰 .bin。`, 'cap'));
  p.push(text(28, 990, `相邻样本在扁平流上以 S=${S} 为步长、S+ε=${windowTokens} 为窗口，重叠恰好 ε=${eps} 个 token；tokens 与 labels 是同一块缓冲区上错开一格的两个视图，不是两份拷贝。`, 'cap'));
  p.push('</svg>');
  return p.join('\n');
}

// ---------------------------------------------------------------------------
// 图 2：两条打包路径
// ---------------------------------------------------------------------------

function renderPackingPathsFigure() {
  const W = 1480, H = 1112, p = [];
  const S = CFG.sequenceLength;
  const slotOf = (index) => CFG.firstDocumentSlot + index;
  const PX = 0.6;

  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="同一组文档在隐式打包与显式打包两条数据面上的切断代价与 padding 代价对照">`);
  p.push(`<style>${sharedStyle}</style>${defs}`, rect(.5, .5, W - 1, H - 1, 'card', 14));
  p.push(text(28, 36, '同一组文档、同一个 S：隐式打包与显式打包的两条数据面', 'ti'));
  p.push(text(28, 57, `S=${S} · ε=${CFG.addExtraToken} · cp=${CFG.contextParallel} · sp(tp)=${CFG.sequenceParallel} → g=${padGranularity} · C=${CFG.maxSeqlenPerDpCpRank} · 尾部对齐 ${CFG.packedSeqAlignment} · 两条 lane 同一比例尺 ${PX} px/token`, 'su'));

  // 面板 ⓪ 共同输入
  p.push(rect(20, 82, W - 40, 116, 'panel', 12));
  p.push(text(40, 110, '⓪ 共同输入：同一组文档（长度已含预处理追加的末尾 EOD）', 'pt'));
  const inX = 60, inW = 1140, inY = 128, k0 = inW / corpusTokens;
  CFG.documentTokens.forEach((length, index) => {
    const x = inX + docStarts[index] * k0;
    p.push(rect(x, inY, length * k0, 42, 'neutral', 5));
    p.push(text(x + (length * k0) / 2, inY + 20, `document_index[${slotOf(index)}]`, 'sm', 'middle'));
    p.push(text(x + (length * k0) / 2, inY + 34, `${length} token`, 'sm', 'middle'));
  });
  p.push(text(inX + inW + 16, inY + 26, `Σ = ${corpusTokens} token`, 'dim'));
  p.push(text(40, 190, '下面两条 lane 用的就是这 4 篇，S 也相同；差别只在"谁决定一条训练样本的边界"。', 'sm'));

  // 面板 ① 隐式打包
  const iy = 210;
  p.push(rect(20, iy, W - 40, 396, 'panel', 12));
  p.push(text(40, iy + 30, '① 隐式打包（GPTDataset）：token 流每 S 个切一刀', 'pt'));

  const laneX = 64, laneY = iy + 62, laneH = 54;
  const preTokens = CFG.beginOffset;
  const postTokens = CFG.documentTokens[sampleWindow.endSlot] - sampleWindow.endOffset;
  const sampleX = laneX + preTokens * PX;
  const sampleW = S * PX;
  const postX = sampleX + sampleW;

  p.push(rect(laneX, laneY, preTokens * PX, laneH, 'ghost', 5));
  p.push(text(laneX + (preTokens * PX) / 2, laneY - 10, `document_index[${slotOf(0)}] 的前 ${preTokens} 个 → 样本 j−1`, 'costtx', 'middle'));
  p.push(rect(postX, laneY, postTokens * PX, laneH, 'ghost', 5));
  p.push(text(postX + (postTokens * PX) / 2, laneY - 10, `document_index[${slotOf(sampleWindow.endSlot)}] 的后 ${postTokens} 个 → 样本 j+1`, 'costtx', 'middle'));
  p.push(text(sampleX + sampleW / 2, laneY - 10, `样本 j：${S} 行`, 'rank', 'middle'));

  eodSegments.forEach((segment, index) => {
    const x = sampleX + segment.start * PX;
    const w = segment.length * PX;
    p.push(rect(x, laneY, w, laneH, 'acc1', 4));
    if (w > 46) {
      p.push(text(x + w / 2, laneY + 24, `document_index[${slotOf(segment.slot)}]`, 'sm', 'middle'));
      p.push(text(x + w / 2, laneY + 40, `${segment.length}`, 'sm', 'middle'));
    }
    if (index > 0) p.push(line(x, laneY, x, laneY + laneH));
  });
  eodIndices.forEach((eod) => {
    const x = sampleX + (eod + 1) * PX;
    p.push(line(x, laneY - 6, x, laneY + laneH + 18, 'cut'));
    p.push(text(x, laneY + laneH + 32, `EOD@${eod}`, 'dim', 'middle'));
  });
  p.push(line(sampleX, laneY - 22, sampleX, laneY + laneH + 8, 'cutcost'));
  p.push(line(postX, laneY - 22, postX, laneY + laneH + 8, 'cutcost'));

  const posY = iy + 158;
  p.push(text(laneX, posY + 26, 'position_ids', 'rank'));
  eodSegments.forEach((segment) => {
    const x = sampleX + segment.start * PX;
    const w = segment.length * PX;
    p.push(rect(x, posY, w, 38, 'ghost', 4));
    if (w > 46) p.push(text(x + w / 2, posY + 24, `0 … ${segment.length - 1}`, 'sm', 'middle'));
  });
  p.push(text(laneX, posY + 62, `reset_position_ids=True：每遇 EOD 重新计数，${eodSegments.length} 段各自从 0 开始。被切断的那半篇文档在这里成了独立的一段。`, 'sm'));

  const lastSegment = eodSegments[eodSegments.length - 1];
  const asideY = posY - 28;
  p.push(line(sampleX + (lastSegment.start + lastSegment.length / 2) * PX, laneY + laneH, 900, asideY + 46, 'lead'));
  p.push(zoomCard([
    `段 ${eodSegments.length}：document_index[${slotOf(lastSegment.slot)}][0:${lastSegment.length})`,
    `${lastSegment.length} 行，position_ids 0 … ${lastSegment.length - 1}`,
    '段内没有 EOD：这篇文档在这里被切断，',
    '剩下的 token 要到样本 j+1 才继续。',
  ], 900, asideY, 340, 92));

  const maskX = 1266, maskY = asideY, maskSize = 116, kk = maskSize / S;
  p.push(rect(maskX, maskY, maskSize, maskSize, 'ghost', 3));
  eodSegments.forEach((segment) => {
    p.push(`<rect class="cell1" x="${round(maskX + segment.start * kk)}" y="${round(maskY + segment.start * kk)}" width="${round(segment.length * kk)}" height="${round(segment.length * kk)}"/>`);
  });
  p.push(text(maskX, maskY + maskSize + 16, `attention_mask [1,${S},${S}]`, 'dim'));
  p.push(text(maskX, maskY + maskSize + 30, 'reset_attention_mask 清左下块', 'sm'));

  p.push(box([
    `代价：${splitDocuments} 篇文档被切断`,
    `document_index[${slotOf(0)}] 的前 ${preTokens} 个 token 在样本 j−1`,
    `document_index[${slotOf(sampleWindow.endSlot)}] 的后 ${postTokens} 个在样本 j+1`,
  ], laneX, iy + 300, 430, 76, 'acc2'));
  p.push(box([
    '收益：样本内 padding = 0',
    `${S} / ${S} 行都是真实 token`,
    '语料尾巴之外没有任何死槽',
  ], laneX + 456, iy + 300, 430, 76, 'acc1'));
  p.push(box([
    '隔离靠 EOD，不靠边界数组',
    `create_attention_mask=True 时物化 ${S}×${S} bool`,
    '任一 reset 开关打开都让 mask 缓存捷径失效',
  ], laneX + 912, iy + 300, 444, 76, 'neutral'));

  // 面板 ② 显式打包
  const ey = 618;
  p.push(rect(20, ey, W - 40, 418, 'panel', 12));
  p.push(text(40, ey + 30, '② 显式打包（VarlenDataset + DpBalancedScheduler）：整条样本不切断，改付两级 padding', 'pt'));

  const mbTops = [ey + 72, ey + 194];
  microbatches.forEach((mb, mbIndex) => {
    const top = mbTops[mbIndex];
    p.push(text(laneX, top - 10, `microbatch ${mbIndex} · ${mb.globalTarget} 行`, 'rank'));
    let cursor = 0;
    mb.members.forEach((member, index) => {
      const valid = mb.originals[index];
      const padded = mb.padded[index];
      const vx = laneX + cursor * PX;
      p.push(rect(vx, top, valid * PX, laneH, 'acc1', 4));
      p.push(rect(vx + valid * PX, top, (padded - valid) * PX, laneH, 'acc2', 0));
      if (valid * PX > 46) {
        p.push(text(vx + (valid * PX) / 2, top + 24, `document_index[${slotOf(member)}]`, 'sm', 'middle'));
        p.push(text(vx + (valid * PX) / 2, top + 40, `${valid} 有效`, 'sm', 'middle'));
      }
      cursor += padded;
    });
    p.push(rect(laneX + cursor * PX, top, mb.dummy * PX, laneH, 'acc2', 4));
    p.push(text(laneX + (cursor + mb.dummy / 2) * PX, top + 27, `尾 ${mb.dummy}`, 'costtx', 'middle'));
    p.push(text(laneX, top + laneH + 22, `cu_seqlens = [${mb.cuSeqlens.join(', ')}]`, 'dim'));
    p.push(text(laneX, top + laneH + 38, `cu_seqlens_padded = [${mb.cuSeqlensPadded.join(', ')}] · max_seqlen = ${mb.maxSeqlen}`, 'dim'));
  });

  p.push(zoomCard([
    `第一级：逐样本对齐到 g=${padGranularity}`,
    ...originalLengths.map((value, index) => `document_index[${slotOf(index)}]：${value} → ${paddedLengths[index]}（+${paddedLengths[index] - value}）`),
    `合计 +${explicitSamplePad} 行`,
  ], 1000, ey + 58, 420, 138));
  p.push(zoomCard([
    `第二级：逐 microbatch 尾部对齐 ${CFG.packedSeqAlignment}`,
    ...microbatches.map((mb, index) => `mb ${index}：CP 本地 ${mb.localActual} → ${mb.localTarget}，全局 ${mb.globalActual} → ${mb.globalTarget}（+${mb.dummy}）`),
    `合计 +${explicitTailPad} 行，各自作为一条 dummy 序列记进 cu_seqlens`,
  ], 1000, ey + 210, 420, 110));

  const limit = CFG.maxSeqlenPerDpCpRank * CFG.contextParallel;
  p.push(zoomCard([
    `贪心装桶：上限 C×cp = ${limit}`,
    `${microbatches[0].padded.join(' + ')} = ${microbatches[0].globalActual} ≤ ${limit}`,
    `再加 ${microbatches[1].padded[0]} → ${microbatches[0].globalActual + microbatches[1].padded[0]} > ${limit}，另起 microbatch`,
    '超限样本自成一组，整条样本从不拆开',
  ], 700, ey + 200, 286, 104));

  p.push(box([
    `代价：死槽 ${explicitDead} / ${explicitRows} = ${explicitWaste.toFixed(1)}%`,
    `逐样本 ${explicitSamplePad} 行 + 逐 microbatch 尾部 ${explicitTailPad} 行`,
    'padding_mask 标 True 的物理行',
  ], laneX, ey + 330, 430, 76, 'acc2'));
  p.push(box([
    '收益：0 篇文档被切断',
    `${CFG.documentTokens.length} 条样本整条留在 cu_seqlens 里`,
    'attention_mask 恒为 None，边界下推给 kernel',
  ], laneX + 456, ey + 330, 430, 76, 'acc1'));
  p.push(box([
    '两套边界坐标缺一不可',
    `cu_seqlens 末项 ${microbatches[0].cuSeqlens[microbatches[0].cuSeqlens.length - 1]} 把尾部 dummy 记成"有效"`,
    '死槽只由 padding_mask 承担，两者不能合并',
  ], laneX + 912, ey + 330, 444, 76, 'neutral'));

  p.push(text(28, 1060, `这张图要证的那笔交易：同一组 ${corpusTokens} token 的语料，隐式路径切断 ${splitDocuments} 篇文档换来样本内 0 padding；显式路径一篇不切，换来 ${explicitDead}/${explicitRows} = ${explicitWaste.toFixed(1)}% 的死槽。`, 'cap'));
  p.push(text(28, 1080, `蓝色是有效 token 与机制路径，橙色是必须付掉的代价；灰色是本条 lane 之外的邻居样本。两条 lane 的横向长度按同一比例尺画，可直接目测。`, 'cap'));
  p.push(text(28, 1100, `g=${padGranularity} 由 cp_pad=${CFG.contextParallel * 2} 乘 tp_pad=${CFG.sequenceParallel} 得出；装桶上限 ${CFG.maxSeqlenPerDpCpRank * CFG.contextParallel} = C×cp；尾部 dummy 长度必须能被 2×cp=${2 * CFG.contextParallel} 整除。`, 'cap'));
  p.push('</svg>');
  return p.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
mkdirSync(outputDir, { recursive: true });

const outputs = new Map([
  ['megatron_dataset_sample_index.svg', renderSampleIndexFigure()],
  ['megatron_dataset_packing_paths.svg', renderPackingPathsFigure()],
]);

for (const [name, svg] of outputs) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
console.log([...outputs.keys()].join('\n'));
