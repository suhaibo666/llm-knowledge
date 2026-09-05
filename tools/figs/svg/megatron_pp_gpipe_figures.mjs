// GPipe is a teaching baseline; 1F1B follows the pinned Megatron schedule.
// Spec: two equal-scale time × rank grids, same four stages and eight microbatches.
// Show F0/B0 through the loss boundary and back, plus peak pending records and messages.
// Compute all cells and counts from dependency simulation, never from drawn positions.
import { seq1f1b, simulate } from './lib/megatron_pp_sim.mjs';
import { pathToFileURL } from 'node:url';

export function model(pp = 4, m = 8) {
  if (!Number.isInteger(pp) || !Number.isInteger(m) || pp < 1 || m < pp) {
    throw new Error('Require integer m >= pp >= 1');
  }
  const gpipe = Array.from({ length: pp }, () => ({
    seq: [true, false].flatMap(f => Array.from({ length: m }, (_, mb) => ({ f, mb, c: 0 }))),
  }));
  return [gpipe, seq1f1b({ pp, m })].map((seq, i) => {
    const result = simulate(seq, { pp, vp: 1 });
    const peaks = seq.map(({ seq: ops }) => {
      let pending = 0, peak = 0;
      for (const op of ops) { pending += op.f ? 1 : -1; peak = Math.max(peak, pending); }
      return peak;
    });
    return { ...result, peaks, messages: 2 * m * (pp - 1), name: i ? '1F1B' : 'GPipe' };
  });
}

export function render() {
  const pp = 4, m = 8, panels = model(pp, m);
  const cell = 31, row = 30, left = 88, width = 1100, height = 510;
  const out = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="GPipe 与 1F1B：四个阶段、八条 microbatch 的前向到 loss、反向回传时序；总时长相同，峰值待反向记录不同">
<style>text{font-family:'Microsoft YaHei','Segoe UI',sans-serif;fill:#344054;font-size:12px}.title{font-size:19px;font-weight:700}.heading{font-size:15px;font-weight:700}.small{font-size:11px}.cell{fill:#fff;stroke:#cbd0d7}.bwd{fill:#f2f4f7;stroke:#cbd0d7}.focus{stroke:#2563eb;stroke-width:2}.bubble{fill:#fafafa;stroke:#eee}.note{fill:#fff7ed;stroke:#c3651f}.blue{fill:#2563eb}.label{text-anchor:middle;font-size:10px}</style>
<rect width="1100" height="510" fill="white"/>
<text class="title" x="24" y="30">GPipe → 1F1B：先减少激活保留量，气泡并未缩短</text>
<text x="24" y="53">P=4，m=8；F/B 各占 1 格。忽略通信与重计算耗时；每个 batch 只在全部反向后更新参数。</text>`];
  panels.forEach((p, index) => {
    const top = 99 + index * 196;
    out.push(`<text class="heading" x="24" y="${top - 16}">${p.name}：${index ? '预热后前后向交替，最后排空' : '先全部前向，再全部反向（教学对照）'}</text>`);
    p.rows.forEach((cells, r) => {
      const y = top + r * row;
      out.push(`<text x="24" y="${y + 19}">rank ${r}</text>`);
      cells.forEach((op, time) => {
        const x = left + time * cell;
        const cls = op ? `${op.f ? 'cell' : 'bwd'}${op.mb === 0 ? ' focus' : ''}` : 'bubble';
        out.push(`<rect class="${cls}" x="${x}" y="${y}" width="29" height="26" rx="2"/>`);
        if (op) out.push(`<text class="label" x="${x + 14.5}" y="${y + 17}">${op.f ? 'F' : 'B'}${op.mb}</text>`);
      });
    });
    out.push(`<text x="${left}" y="${top + pp * row + 15}">0</text><text x="${left + p.span * cell - 24}" y="${top + pp * row + 15}">${p.span} t_f</text>
<rect class="note" x="797" y="${top + 6}" width="276" height="104" rx="6"/>
<text x="811" y="${top + 30}">峰值待反向记录：[${p.peaks.join(',')}]</text>
<text x="811" y="${top + 54}">每 rank：计算 ${p.ops} 格，空转 ${p.bubble} 格</text>
<text x="811" y="${top + 78}">总时长 ${p.span} t_f；边界消息 ${p.messages} 条</text>
<text class="small" x="811" y="${top + 98}">记录数是依赖计数，不等于激活字节数。</text>`);
  });
  out.push(`<text class="blue" x="24" y="468">蓝框追踪 m0：F0 沿 rank 0→3 到达 loss，B0 沿 rank 3→0 回传梯度。</text>
<text x="24" y="491">空白格表示等待依赖；灰色 B 格表示反向。GPipe 保留全部 8 条记录，1F1B 及时消费最早的记录。</text></svg>`);
  return out.join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) console.log(render());
