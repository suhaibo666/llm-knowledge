// Megatron-LM PP 调度的离散事件仿真器。
//
// 算法逐条照抄自 NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a
// （与 wiki/02_engineering/02_train_frameworks/megatron-lm/15_megatron_pp_schedulers_analysis.md
//  的页头基线一致），每个函数标出对应的 file:line。
//
// 存在的意义：配图的格子由它解算，**图不可能和调度对不上**。
// 校验：本模块的输出已与该页 §③.3 / §⑤.3(a) 两张手工 ASCII 表逐格比对一致
// （pp=4, vp=2, m=8, N=4：makespan=38，每设备 32 op + 6 空泡）。

/** 标准 1F1B（调度器②）：warmup = pp-r-1，随后 1F1B 稳态，最后 cooldown。
 *  megatron/core/pipeline_parallel/schedules.py:906 */
export function seq1f1b({ pp, m }) {
  return Array.from({ length: pp }, (_, r) => {
    const w = pp - r - 1;
    const seq = [];
    for (let i = 0; i < w; i++) seq.push({ mb: i, c: 0, f: true });
    let fi = w, bi = 0;
    for (; fi < m; fi++) { seq.push({ mb: fi, c: 0, f: true }); seq.push({ mb: bi++, c: 0, f: false }); }
    while (bi < m) seq.push({ mb: bi++, c: 0, f: false });
    return { warmup: w, seq };
  });
}

/** 调度表：按 N 个 microbatch 一组，组内先排完 chunk0 再排 chunk1……
 *  megatron/core/pipeline_parallel/schedules.py:938-965 */
export function scheduleTable(m, vp, N) {
  const t = [];
  for (let g = 0; g < m; g += N) {
    const hi = g + N >= m ? m : g + N;          // 末组吸收余数 (:944-952)
    for (let c = 0; c < vp; c++) for (let i = g; i < hi; i++) t.push([i, c]);
  }
  return t;
}

/** 交错 1F1B / VPP（调度器③，含 combined-1F1B 的 warmup+1 宿主形态）。
 *  warmup   : schedules.py:915-919
 *  反向 chunk 反转 : schedules.py:1346-1351  model_chunk_id = vp - chunk - 1 */
export function seqVpp({ pp, m, vp, N, extraWarmup = 0 }) {
  const T = scheduleTable(m, vp, N);
  const tmb = T.map((x) => x[0]), tch = T.map((x) => x[1]);
  const total = T.length;                                    // = m * vp
  // 每个 chunk 的前向 microbatch 队列；反向按 FIFO 消费同一队列
  const fwdQ = Array.from({ length: vp }, (_, c) => T.filter((x) => x[1] === c).map((x) => x[0]));

  return Array.from({ length: pp }, (_, r) => {
    const w = (pp - r - 1) * 2 + (vp - 1) * N + extraWarmup;
    const seq = [], taken = Array(vp).fill(0);
    const fwd = (vid) => ({ mb: tmb[vid], c: tch[vid], f: true });
    const bwd = (bid) => { const c = vp - 1 - tch[bid]; return { mb: fwdQ[c][taken[c]++], c, f: false }; };
    for (let vid = 0; vid < w; vid++) seq.push(fwd(vid));
    for (let vid = w; vid < total; vid++) { seq.push(fwd(vid)); seq.push(bwd(vid - w)); }
    for (let bid = total - w; bid < total; bid++) seq.push(bwd(bid));
    return { warmup: w, seq };
  });
}

/** 依赖解算：每个 rank 按自己的固定序列推进，依赖未就绪就停顿（= 空泡）。
 *  前向 (mb,c) 在 rank r 依赖 rank r-1 的同 (mb,c)；r=0 且 c>0 时依赖末 rank 的 (mb,c-1)。
 *  反向 (mb,c) 在 rank r 依赖 rank r+1 的同 (mb,c)；末 rank 上 c<vp-1 依赖 rank0 的 (mb,c+1)，
 *  最深 chunk 的反向依赖该 mb 在末 rank 的最后一次前向。 */
export function simulate(ranks, { pp, vp }) {
  const done = {}, start = {};
  const K = (f, mb, c, r) => `${f ? 'F' : 'B'}${mb}.${c}@${r}`;
  const ptr = Array(pp).fill(0), free = Array(pp).fill(0);
  let rem = ranks.reduce((a, x) => a + x.seq.length, 0), guard = 0;
  while (rem > 0 && guard++ < 200000) {
    let progressed = false;
    for (let r = 0; r < pp; r++) {
      const p = ptr[r];
      if (p >= ranks[r].seq.length) continue;
      const { mb, c, f } = ranks[r].seq[p];
      let dep = free[r], k = null;
      if (f) {
        if (r > 0) k = K(true, mb, c, r - 1);
        else if (c > 0) k = K(true, mb, c - 1, pp - 1);
      } else if (r < pp - 1) k = K(false, mb, c, r + 1);
      else if (c < vp - 1) k = K(false, mb, c + 1, 0);
      else k = K(true, mb, vp - 1, pp - 1);
      if (k) { if (!(k in done)) continue; dep = Math.max(dep, done[k]); }
      const key = K(f, mb, c, r);
      start[key] = dep; done[key] = dep + 1; free[r] = dep + 1;
      ptr[r]++; rem--; progressed = true;
    }
    if (!progressed) break;
  }
  if (rem > 0) throw new Error(`调度死锁，剩余 ${rem} 个 op —— 依赖模型与源码不一致`);
  const span = Math.max(...Object.values(done));
  const rows = ranks.map((rk, r) => {
    const row = Array(span).fill(null);
    for (const op of rk.seq) row[start[K(op.f, op.mb, op.c, r)]] = op;
    return row;
  });
  return { span, rows, ops: ranks[0].seq.length, bubble: span - ranks[0].seq.length };
}
