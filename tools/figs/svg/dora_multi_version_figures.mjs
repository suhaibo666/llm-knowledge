// 23_dora_multi_version_rollout_analysis.md 的原理图：DORA 多版本流式训练时间线。
// 来源：Meituan LongCat, DORA (arXiv:2604.26256v2, 2026-07-20) §3.2 图 6 的思路重画；
// 论文无公开实现，图中的请求编号、长度与步边界是**示例输入**，不是论文数据。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要回答的问题：一条长尾轨迹为什么既不被丢弃、也不需要重 prefill，同时其它 DP group 还能
// 持续接新版本的请求。
// 布局：一条时间轴，上方 Trainer 行，下方四个 rollout DP group 各一行。
//  - Trainer 行：ghost 段 = 等 TBS 条轨迹凑齐并做经验准备；neutral 深字段 = 训练产出下一版本。
//    训练结束的时刻定义步边界 t1/t2/t3（竖向虚线）。
//  - DP 行：每个条块是一条请求，条块样式表示它所用的策略版本：
//    W1 = ghost，W2 = neutral，W3 = band2（浅灰底），W4 = acc1（最新版本，唯一主强调色）。
//  - 请求 4（DP0，W1）是长尾：acc2 描边；它跨三个训练步始终在 W1 下生成。
//  - t2 时编排器收回 DP1 换成 W3，DP1 上未完成的 W2 请求 10 连同 KV cache 迁到仍装着 W2 的 DP3：
//    用 main 箭头表示迁移；迁入后的条块长度 = 剩余 decode 长度，不变。
//  - 左下角一个 acc2 虚线框：同步训练下被请求 4 卡住的整段空转（对照）。
// 只用 acc1（最新版本/迁移主路径）与 acc2（长尾代价/同步空转）两种强调色。
//
// 用法：node tools/figs/svg/dora_multi_version_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 示例输入（不是论文数据） ----------------
// 时间单位为“token 时间”的示意刻度，t=0..400；步边界由训练结束时刻定义。
export const CFG = Objecteze({
  t0: 0,
  tEnd: 400,
  // 每步：等待收集 TBS 条的区间 + 训练区间；训练结束 = 下一版本可用
  steps: [
    { collectFrom: 0, trainFrom: 58, trainTo: 100, produces: 'W2' },
    { collectFrom: 100, trainFrom: 158, trainTo: 200, produces: 'W3' },
    { collectFrom: 200, trainFrom: 258, trainTo: 300, produces: 'W4' },
  ],
  // 每个 DP group 的请求：id、版本、开始、结束。migrateTo = 在 t 时刻迁往哪个 group。
  groups: {
    DP0: [
      { id: '1', v: 'W1', s: 0, e: 32 }, { id: '2', v: 'W1', s: 34, e: 58 }, { id: '3', v: 'W1', s: 60, e: 76 },
      { id: '4', v: 'W1', s: 78, e: 262, longTail: true },
      { id: '14', v: 'W4', s: 305, e: 400 },
    ],
    DP1: [
      { id: '5', v: 'W1', s: 0, e: 52 }, { id: '6', v: 'W1', s: 54, e: 100 },
      { id: '9', v: 'W2', s: 103, e: 166 }, { id: '10', v: 'W2', s: 168, e: 200, migrate: { at: 200, to: 'DP3', remaining: 46 } },
      { id: '13', v: 'W3', s: 203, e: 400 },
    ],
    DP2: [
      { id: '7', v: 'W1', s: 0, e: 96 },
      { id: '11', v: 'W2', s: 103, e: 145 }, { id: '12', v: 'W2', s: 147, e: 198 },
      { id: '15', v: 'W3', s: 203, e: 298 }, { id: '17', v: 'W4', s: 303, e: 400 },
    ],
    DP3: [
      { id: '8', v: 'W1', s: 0, e: 38 }, { id: '8b', v: 'W1', s: 40, e: 98 },
      { id: '19', v: 'W2', s: 103, e: 198 },
      // 请求 10 迁入：起点 = 迁移时刻，长度 = 剩余 decode，不变
      { id: '10', v: 'W2', s: 200, e: 246, migratedIn: true },
      { id: '16', v: 'W3', s: 250, e: 298 }, { id: '18', v: 'W4', s: 303, e: 400 },
    ],
  },
});

function Objecteze(o) { return Object.freeze(o); }

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

const style = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .band2{fill:#E9ECF2;stroke:#C4CAD4;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .train{fill:#5B6470;stroke:#5B6470}
  .tail{fill:#F5F7FA;stroke:#C3651F;stroke-width:1.8}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .cost{fill:none;stroke:#C3651F;stroke-width:1.4;stroke-dasharray:5 4}
  .refline{fill:none;stroke:#C8CFDA;stroke-width:1;stroke-dasharray:3 4}
  .axis{fill:none;stroke:#AEB6C2;stroke-width:1}
  .ti{font-size:17px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .tx{font-size:11.5px;fill:#38414D}
  .txw{font-size:11.5px;fill:#fff;font-weight:600}
  .sm{font-size:10.5px;fill:#68717D}
  .rank{font-size:12px;font-weight:700;fill:#5B6470}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .cap{font-size:11px;fill:#747C88}
`;
const defs = `<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker></defs>`;

const W = 980;
const L = 120; // 时间轴左缘
const R = 950;
const px = (t) => L + ((t - CFG.t0) / (CFG.tEnd - CFG.t0)) * (R - L);

function rect(x, y, w, h, cls, rx = 3) {
  return `<rect class="${cls}" x="${x.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="${h}" rx="${rx}"/>`;
}
function text(x, y, v, cls = 'tx', anchor = 'start') {
  return `<text class="${cls}" x="${x.toFixed(1)}" y="${y}" text-anchor="${anchor}">${esc(v)}</text>`;
}
const CLS = { W1: 'ghost', W2: 'neutral', W3: 'band2', W4: 'acc1' };

export function render() {
  const out = [];
  const H = 392;
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img">`);
  out.push(defs, `<style>${style}</style>`);
  out.push(text(24, 30, 'DORA 多版本流式训练时间线', 'ti'));
  out.push(text(24, 48, '每个 DP group 只装一个版本；一条轨迹从头到尾只用一个版本；换版本只发生在 group 之间（示例输入，非论文数据）', 'su'));

  const yTrain = 80, rowH = 30;
  const rows = ['DP0', 'DP1', 'DP2', 'DP3'];
  const yRow = (i) => 134 + i * 46;
  const yNote = yRow(rows.length - 1) + rowH + 24; // 注释行：迁移说明与同步对照
  const yAxis = yNote + 14;

  // 步边界虚线 + 刻度
  const marks = [{ t: CFG.t0, l: 't0' }];
  CFG.steps.forEach((s, i) => marks.push({ t: s.trainTo, l: `t${i + 1} 训练步 ${i + 1} 结束` }));
  marks.push({ t: CFG.tEnd, l: '' });
  for (const m of marks) {
    out.push(`<path class="refline" d="M ${px(m.t).toFixed(1)} ${yTrain - 8} L ${px(m.t).toFixed(1)} ${yAxis}"/>`);
    if (m.l) out.push(text(px(m.t), yAxis + 16, m.l, 'sm', 'middle'));
  }
  out.push(`<path class="axis" d="M ${L} ${yAxis} L ${R} ${yAxis}"/>`);

  // Trainer 行
  out.push(text(24, yTrain + 20, 'Trainer', 'rank'));
  CFG.steps.forEach((s, i) => {
    const x1 = px(s.collectFrom), x2 = px(s.trainFrom), x3 = px(s.trainTo);
    out.push(rect(x1, yTrain, x2 - x1, rowH, 'ghost'));
    out.push(text((x1 + x2) / 2, yTrain + 19, '收 TBS 条 + 经验准备', 'sm', 'middle'));
    out.push(rect(x2, yTrain, x3 - x2, rowH, 'train'));
    out.push(text((x2 + x3) / 2, yTrain + 19, `train → ${s.produces}`, 'txw', 'middle'));
    // 训练结束 → P2P 权重发往部分 DP group
    out.push(text(x3 + 4, yTrain + rowH + 14, 'P2P 权重 → 选中的 group', 'dim'));
  });
  const lastTo = CFG.steps[CFG.steps.length - 1].trainTo;
  out.push(rect(px(lastTo), yTrain, px(CFG.tEnd) - px(lastTo), rowH, 'ghost'));
  out.push(text((px(lastTo) + px(CFG.tEnd)) / 2, yTrain + 19, '收 TBS 条，含请求 4', 'sm', 'middle'));

  // DP 行
  rows.forEach((g, i) => {
    const y = yRow(i);
    out.push(text(24, y + 20, g, 'rank'));
    for (const r of CFG.groups[g]) {
      const x1 = px(r.s), x2 = px(r.e), w = x2 - x1;
      const cls = r.longTail ? 'tail' : CLS[r.v];
      out.push(rect(x1, y, w, rowH, cls));
      let label = r.id;
      if (r.longTail) label = `${r.id} · 长尾，始终在 W1 下生成，跨三个训练步`;
      else if (r.migratedIn) label = `${r.id} 迁入`;
      else if (w > 60) label = `${r.id} · ${r.v}`;
      const cls2 = r.longTail ? 'costtx' : 'tx';
      if (textWidth(label, 11.5) <= w - 6) out.push(text(x1 + w / 2, y + 19, label, cls2, 'middle'));
      else if (w >= 14) out.push(text(x1 + w / 2, y + 19, r.id, 'tx', 'middle'));
    }
  });

  // 迁移箭头：DP1 的请求 10 在 t2 迁到 DP3
  const mig = CFG.groups.DP1.find((r) => r.migrate);
  if (mig) {
    const x = px(mig.migrate.at);
    const y1 = yRow(1) + rowH, y2 = yRow(3);
    out.push(`<path class="main" d="M ${(x + 1).toFixed(1)} ${y1 + 1} L ${(x + 1).toFixed(1)} ${y2 - 3}"/>`);
    out.push(text(x + 20, yNote, 'DP1 收回换 W3；请求 10 带 KV cache 迁到 DP3 续跑，剩余长度不变', 'dim'));
  }

  // 同步训练对照框：t0 到请求 4 完成之前，整批都在等
  const tail = CFG.groups.DP0.find((r) => r.longTail);
  out.push(`<rect class="cost" x="${(L - 4).toFixed(1)}" y="${yRow(0) - 6}" width="${(px(tail.e) - L + 8).toFixed(1)}" height="${yRow(3) + rowH + 8 - (yRow(0) - 6)}" rx="4"/>`);
  out.push(text(L + 2, yNote, '同步训练下整批都在等请求 4 完成（对照）', 'costtx'));

  // 图例
  const ly = H - 14;
  const legend = [['ghost', 'W1 最旧'], ['neutral', 'W2'], ['band2', 'W3'], ['acc1', 'W4 最新'], ['tail', '长尾轨迹'], ['train', '训练']];
  let lx = L;
  for (const [cls, name] of legend) {
    out.push(rect(lx, ly - 11, 14, 12, cls, 2));
    out.push(text(lx + 19, ly, name, 'cap'));
    lx += 19 + textWidth(name, 11) + 22;
  }
  out.push('</svg>');
  return out.join('\n');
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir = process.argv[2] || join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'assets');
  mkdirSync(outDir, { recursive: true });
  const file = join(outDir, 'dora_multi_version_timeline.svg');
  writeFileSync(file, render());
  console.log(`wrote ${file}`);
}
