import { writeFileSync } from 'node:fs';

const out = new URL('./prefix_cache_identity_radix.svg', import.meta.url);
const C = {
  ink: '#172033', muted: '#475569', line: '#94a3b8', panel: '#f8fafc',
  blue: '#dbeafe', blueStroke: '#2563eb', orange: '#ffedd5', orangeStroke: '#ea580c',
  green: '#dcfce7', greenStroke: '#16a34a', red: '#fee2e2', redStroke: '#dc2626',
};
const esc = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const rect = (x, y, w, h, fill, stroke = C.line, r = 8, extra = '') =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const text = (x, y, value, size = 15, anchor = 'start', weight = 400, fill = C.ink) =>
  `<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" font-weight="${weight}" fill="${fill}">${esc(value)}</text>`;
const line = (x1, y1, x2, y2, stroke = C.line, width = 2, dash = '') =>
  `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${stroke}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''} marker-end="url(#arrow)"/>`;
const token = (x, y, w, label, fill, stroke) =>
  `${rect(x, y, w, 30, fill, stroke, 5)}${text(x + w / 2, y + 20, label, 13, 'middle', 600)}`;

const request = (y, label, pieces, hit, add) => {
  let x = 162;
  let body = `${rect(24, y, 1135, 49, '#ffffff', C.line, 8)}${text(42, y + 21, label, 15, 'start', 700)}${text(42, y + 39, `命中 ${hit}；新算 ${add}`, 12, 'start', 400, C.muted)}`;
  for (const p of pieces) {
    body += token(x, y + 9, p.w, p.label, p.fill, p.stroke);
    x += p.w + 8;
  }
  return body;
};

let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="835" viewBox="0 0 1180 835">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="${C.line}"/></marker></defs>
<style>text{font-family:-apple-system,BlinkMacSystemFont,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}.small{font-size:12px;fill:${C.muted}}</style>
${rect(12, 12, 1156, 58, C.panel, C.line, 12)}
${text(32, 40, '三条请求的教学输入：相同身份 I，按到达顺序查树', 19, 'start', 700)}
${text(32, 65, '蓝：已命中前缀；橙：本请求须新计算的 token。没有假定页大小或物理块编号。', 13, 'start', 400, C.muted)}
${request(82, '请求 A', [
  {w:170,label:'S0 S1 S2 S3',fill:C.orange,stroke:C.orangeStroke},
  {w:170,label:'H0 H1 H2 H3',fill:C.orange,stroke:C.orangeStroke},
  {w:105,label:'A0 A1',fill:C.orange,stroke:C.orangeStroke},
], '0', '10')}
${request(138, '请求 B', [
  {w:170,label:'S0 S1 S2 S3',fill:C.blue,stroke:C.blueStroke},
  {w:170,label:'H0 H1 H2 H3',fill:C.blue,stroke:C.blueStroke},
  {w:78,label:'B0',fill:C.orange,stroke:C.orangeStroke},
], '8', '1')}
${request(194, '请求 C', [
  {w:170,label:'S0 S1 S2 S3',fill:C.blue,stroke:C.blueStroke},
  {w:105,label:'C0 C1',fill:C.orange,stroke:C.orangeStroke},
], '4', '2')}

${rect(12, 276, 286, 251, C.panel, C.line, 12)}
${text(32, 305, '身份门（T14）', 18, 'start', 700)}
${text(32, 334, '可比较的输入：', 14, 'start', 600)}
${text(43, 360, '• token ID 前缀及其顺序', 13)}
${text(43, 385, '• 位置与注意力语义', 13)}
${text(43, 410, '• 模型 / adapter / 其他上下文', 13)}
${rect(32, 440, 242, 59, C.green, C.greenStroke, 8)}
${text(153, 464, '身份 I 一致', 15, 'middle', 700)}
${text(153, 485, '允许查找最长 token 前缀', 12, 'middle', 400, C.muted)}
${rect(12, 544, 286, 69, '#fff7ed', C.orangeStroke, 12)}
${text(32, 572, '任一身份条件改变', 15, 'start', 700, C.orangeStroke)}
${text(32, 595, '→ 隔离命名空间或 miss 后重算', 12, 'start', 400, C.muted)}

${rect(329, 276, 500, 337, C.panel, C.line, 12)}
${text(350, 305, '前缀树：索引内容身份与最长前缀（T14）', 18, 'start', 700)}
${rect(521, 326, 116, 36, '#ffffff', C.line, 8)}${text(579, 349, '根：身份 I', 14, 'middle', 700)}
${rect(499, 398, 160, 39, C.blue, C.blueStroke, 8)}${text(579, 422, 'S0…S3（长度 4）', 14, 'middle', 700)}
${rect(485, 474, 188, 39, C.blue, C.blueStroke, 8)}${text(579, 498, 'H0…H3（长度 8）', 14, 'middle', 700)}
${rect(391, 551, 103, 37, C.orange, C.orangeStroke, 8)}${text(442, 574, 'A0 A1', 14, 'middle', 700)}
${rect(529, 551, 102, 37, C.orange, C.orangeStroke, 8)}${text(580, 574, 'B0', 14, 'middle', 700)}
${rect(665, 551, 103, 37, C.orange, C.orangeStroke, 8)}${text(716, 574, 'C0 C1', 14, 'middle', 700)}
${line(579, 362, 579, 398)}${line(579, 437, 579, 474)}${line(579, 513, 442, 551)}${line(579, 513, 580, 551)}${line(579, 437, 716, 551)}
${text(579, 606, 'A、B 在长度 8 分叉；C 在长度 4 分叉', 12, 'middle', 400, C.muted)}

${rect(860, 276, 308, 337, C.panel, C.line, 12)}
${text(881, 305, '命中结果与物理层边界', 18, 'start', 700)}
${rect(882, 332, 263, 74, '#ffffff', C.blueStroke, 9)}
${text(1014, 358, '查找输出（T14）', 15, 'middle', 700)}
${text(1014, 382, '匹配节点 + 命中长度 m', 13, 'middle', 400, C.muted)}
${rect(882, 438, 263, 105, '#ffffff', C.line, 9)}
${text(1014, 466, '交给 T13 的物理映射层', 15, 'middle', 700)}
${text(1014, 490, '取得并保护对应 KV；续写时', 12, 'middle', 400, C.muted)}
${text(1014, 512, '按其块表、引用与 CoW 规则处理', 12, 'middle', 400, C.muted)}
${text(1014, 578, '本图不画物理块、引用计数或迁移', 12, 'middle', 400, C.muted)}
${line(298, 469, 499, 418, C.line, 2)}${line(829, 418, 882, 369, C.line, 2)}${line(1014, 406, 1014, 438, C.line, 2)}

${rect(12, 649, 1156, 163, C.panel, C.line, 12)}
${text(32, 679, '按图中到达顺序复算', 18, 'start', 700)}
${text(32, 711, 'A：m=0，新算 10；插入 S→H→A。', 15)}
${text(32, 740, 'B：m=8，新算 1；复用 S→H 并写入 B 分支。', 15)}
${text(32, 769, 'C：m=4，新算 2；复用 S 并写入 C 分支。总长度 25，新算 13，命中位置合计 12。', 15)}
${text(32, 796, '这是 token 位置计数，忽略 kernel、批处理、读带宽、淘汰与跨层传输成本。', 12, 'start', 400, C.muted)}
</svg>`;

writeFileSync(out, svg);
