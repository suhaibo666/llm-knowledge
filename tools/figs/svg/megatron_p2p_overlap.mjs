// 图：overlap_p2p_comm —— 把 vp× 的 P2P 通信移出关键路径
// 用于 wiki/02_engineering/02_train_frameworks/megatron-lm/15_megatron_pp_schedulers_analysis.md §④.3
//
// ---- spec（见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：VPP 默认走同步融合算子 send_forward_backward_recv_forward_backward
// （schedules.py:2036），稳态每一步"算 → 同步收发 → 再算"串成一条线，P2P 暴露在关键路径上；
// overlap_p2p_comm 改成异步 isend/irecv，只拿 wait handle 不立即等，在下一次计算**之后**
// 才 wait（schedules.py:1849 的四个回调），于是通信与计算在 GPU 上并行、P2P 被计算掩盖。
// 立论一句话：VPP 把气泡比降到 1/vp，但通信 ×vp 可能把省下的时间吐回去；本特性负责把它藏起来。
//
// 布局（上下两条，共用同一条时间轴）：
//   上「同步融合 P2P」：comp 流 F ▮ 干等 ▮ B ▮ 干等 ▮ …，干等段用斜纹（= 关键路径上的通信）
//   下「overlap_p2p_comm」：comp 流 F B F B 连续无缝；下方 comm 轨的 isend/irecv 条与计算并行
//   两条轴各画 makespan 尺规，长度差 = 省下的 N_comm·t_p2p
// 强调（两个）：上 acc2「每步多付 t_p2p」；下 acc1「P2P 移出关键路径」
// 底部标出 schedules.py:1849 的四个回调落在时间轴的哪个位置——图与源码同构
//
// ⚠️ 数据诚实性：本图**不是**仿真产物，本库没有该场景的真实带宽/时长测量。
//    - **序列**（谁先谁后、回调挂在哪一步）逐条取自源码，可核对；
//    - **时间比例**是下面 TF/TB/TP2P 三个显式参数，属于示意，图上已标注。
//    据 §④.4，t_p2p 的量级取决于跨机 IB 带宽；这里取 t_p2p = 0.375·t_f 仅为作图。

const TF = 4.0, TB = 4.0, TP2P = 1.5;   // 时间比例（示意，非实测）
const STEPS = 2;                         // 画 2 个稳态 1F1B 步：F B F B

/* 上：同步——计算与通信严格串行 */
const sync = [];
{
  let t = 0;
  for (let i = 0; i < STEPS; i++) {
    sync.push({ t, d: TF, k: 'f', label: `F${i}` });               t += TF;
    sync.push({ t, d: TP2P, k: 'stall', label: 'P2P' });            t += TP2P;
    sync.push({ t, d: TB, k: 'b', label: `B${i}` });                t += TB;
    sync.push({ t, d: TP2P, k: 'stall', label: 'P2P' });            t += TP2P;
  }
  sync.span = t;
}
/* 下：重叠——comp 连续，comm 与之并行 */
const ovComp = [], ovComm = [];
{
  let t = 0;
  for (let i = 0; i < STEPS; i++) {
    ovComp.push({ t, d: TF, k: 'f', label: `F${i}` });
    ovComm.push({ t: t + TF, d: TP2P, k: 'comm', label: 'isend/irecv 激活' });   // pp_post_forward
    t += TF;
    ovComp.push({ t, d: TB, k: 'b', label: `B${i}` });
    ovComm.push({ t: t + TB, d: TP2P, k: 'comm', label: 'isend/irecv 梯度' });   // pp_post_backward
    t += TB;
  }
  ovComp.span = t;
}
const saved = sync.span - ovComp.span;
const pct = ((saved / sync.span) * 100).toFixed(0);

/* ---------- 几何 ---------- */
const U = 46, BH = 30, PAD = 20, LAB = 74, NOTE = 176;
const axis = Math.max(sync.span, ovComp.span + TP2P);
const gridW = axis * U;
const W = PAD + LAB + gridW + 14 + NOTE + PAD;

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const body = [];
const put = (s) => body.push(s);

const bar = (x, y, w, h, cls, label, small) => {
  put(`  <rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="4"/>`);
  if (label) put(`  <text class="${small ? 'bls' : 'bl'}" x="${x + w / 2}" y="${y + h / 2 + 3.5}">${esc(label)}</text>`);
};

function track(y, name, items, cls) {
  put(`  <text class="lab" x="${PAD + LAB - 10}" y="${y + BH / 2 + 4}">${esc(name)}</text>`);
  put(`  <line class="base" x1="${PAD + LAB}" y1="${y + BH + 3}" x2="${PAD + LAB + gridW}" y2="${y + BH + 3}"/>`);
  for (const it of items) {
    bar(PAD + LAB + it.t * U, y, it.d * U - 2, BH, cls ? cls(it) : it.k, it.label, it.k === 'comm' || it.k === 'stall');
  }
}

function panel(y, title, sub, { comp, comm, span, tone, note }) {
  put(`  <text class="pt" x="${PAD}" y="${y}">${esc(title)}</text>`);
  put(`  <text class="ps" x="${PAD}" y="${y + 17}">${esc(sub)}</text>`);
  let ty = y + 30;
  track(ty, 'comp 流', comp);
  if (comm) { ty += BH + 14; track(ty, 'comm 流', comm); }
  const my = ty + BH + 18, len = span * U;
  put(`  <line class="rule" x1="${PAD + LAB}" y1="${my}" x2="${PAD + LAB + len}" y2="${my}"/>`);
  put(`  <line class="tick" x1="${PAD + LAB + len}" y1="${my - 5}" x2="${PAD + LAB + len}" y2="${my + 5}"/>`);
  put(`  <text class="mk" x="${PAD + LAB + len / 2}" y="${my + 14}">壁钟 = ${span.toFixed(1)} t（以 t_f = ${TF} t 为刻度）</text>`);
  const bx = PAD + LAB + gridW + 14, by = y + 34;
  put(`  <rect class="box-${tone}" x="${bx}" y="${by}" width="${NOTE}" height="${52}" rx="8"/>`);
  note.forEach((ln, i) => put(`  <text class="txt-${tone}" x="${bx + 11}" y="${by + 18 + i * 15}">${esc(ln)}</text>`));
  return my + 26;
}

let y = PAD + 60;
y = panel(y, '① VPP 默认：同步融合 P2P（send_forward_backward_recv_forward_backward）',
  '算完 → 同步收发 → 再算，串成一条线；斜纹段 = 计算流干等通信', {
    comp: sync, span: sync.span, tone: 'cost',
    note: ['P2P 在关键路径上', `每步多付 t_p2p`],
  });
y += 18;
y = panel(y, '② overlap_p2p_comm：异步 isend/irecv + 延迟 wait',
  'isend/irecv 只拿 handle，到下一次计算之后才 wait；comm 与 comp 在 GPU 上并行', {
    comp: ovComp, comm: ovComm, span: ovComp.span, tone: 'gain',
    note: ['P2P 移出关键路径', `省 ${saved.toFixed(1)} t（≈${pct}%）`],
  });

// 四个回调锚点：图与 schedules.py:1849 的稳态循环同构
const cb = [
  { t: 0, s: 'pp_pre_forward\nwait(上轮预取 recv)' },
  { t: TF, s: 'pp_post_forward\nisend 激活 + irecv 预取' },
  { t: TF + TB, s: 'pp_post_backward\nisend 梯度 + irecv 预取' },
];
cb.forEach(({ t, s }) => {
  const x = PAD + LAB + t * U;
  put(`  <line class="cb" x1="${x}" y1="${y - 4}" x2="${x}" y2="${y + 6}"/>`);
  s.split('\n').forEach((ln, i) =>
    put(`  <text class="cbt" x="${x + 4}" y="${y + 16 + i * 12}">${esc(ln)}</text>`));
});
y += 46;                 // 让过回调标注的两行文字（y+16 / y+28）
const H = y + 52;

const head = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
     aria-label="overlap_p2p_comm 前后对照：同步融合 P2P 时计算流每步干等一段通信，异步 isend/irecv 加延迟 wait 后计算流连续、通信被掩盖，壁钟从 ${sync.span} 降到 ${ovComp.span}">
  <style>
    text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
    .card{fill:#fff;stroke:#E4E7EC}
    .ti{font-size:15px;font-weight:700;fill:#1F2430}
    .su{font-size:11.5px;fill:#8A8F98}
    .pt{font-size:12.5px;font-weight:700;fill:#2A313B}
    .ps{font-size:11px;fill:#8A8F98}
    .lab{font-size:11px;fill:#5B6470;text-anchor:end}
    .bl{font-size:12px;font-weight:700;fill:#fff;text-anchor:middle}
    .bls{font-size:10px;font-weight:600;fill:#5B6470;text-anchor:middle}
    .f{fill:#3E7BD0}                                  /* 前向计算 */
    .b{fill:#3E9970}                                  /* 反向计算 */
    .stall{fill:url(#hatch);stroke:#C3651F;stroke-width:1.2}   /* 关键路径上的干等 */
    .comm{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.2}        /* 被掩盖的异步通信 */
    .base{stroke:#ECEFF3}
    .rule{stroke:#C7CCD3;stroke-dasharray:3 3}
    .tick{stroke:#9AA1AC;stroke-width:1.5}
    .mk{font-size:10px;fill:#9AA1AC;text-anchor:middle}
    .cb{stroke:#B6BCC4;stroke-width:1}
    .cbt{font-size:9.5px;fill:#8A8F98}
    .box-cost{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.4}
    .txt-cost{font-size:10.5px;font-weight:600;fill:#8A4A11}
    .box-gain{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.4}
    .txt-gain{font-size:10.5px;font-weight:600;fill:#173F87}
    .cap{font-size:11px;fill:#7A808A}
  </style>
  <defs>
    <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="6" height="6" fill="#FBF1E6"/>
      <rect width="3" height="6" fill="#EFD9BE"/>
    </pattern>
  </defs>
  <rect class="card" x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="14"/>
  <text class="ti" x="${PAD}" y="${PAD + 14}">overlap_p2p_comm：把 vp× 的 P2P 通信移出关键路径</text>
  <text class="su" x="${PAD}" y="${PAD + 32}">VPP 稳态的 2 个 1F1B 步 · 上下共用同一条时间轴 · 蓝 = 前向计算，绿 = 反向计算，橙斜纹 = 计算流干等，浅蓝框 = 异步 P2P</text>
  <text class="su" x="${PAD}" y="${PAD + 48}">序列取自 schedules.py:1849 的四个回调（可逐条核对）；时间比例为示意参数 t_f=t_b=${TF}t、t_p2p=${TP2P}t，非实测</text>`;

const tail = `  <text class="cap" x="${PAD}" y="${y + 14}">壁钟从 ${sync.span.toFixed(1)} t 降到 ${ovComp.span.toFixed(1)} t（省 ${pct}%）——省的是<tspan font-weight="700">暴露的通信</tspan>，不是气泡：气泡率仍是 (pp−1)/(m·vp)，与 VPP 相同。</text>
  <text class="cap" x="${PAD}" y="${y + 30}">代价：额外的 fwd/bwd_recv_buffer（首/末 rank 各 N−pp+1 份 chunk 激活，schedules.py:1668）；且必须 batch_p2p_comm = False（schedules.py:1180-1181 断言）。</text>
</svg>`;

console.log([head, ...body, tail].join('\n'));
