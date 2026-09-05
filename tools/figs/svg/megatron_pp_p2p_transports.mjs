// 同一 activation/gradient 在三条 P2P transport 和 NCCL/UCC backend 下的控制、等待与成本。
// Usage: node tools/figs/svg/megatron_pp_p2p_transports.mjs > .../assets/megatron_pp_p2p_transports.svg

const W = 1560, H = 970;
const out = [];
const put = (s) => out.push(s);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function lines(x, y, xs, cls = 'tx', gap = 17, anchor = 'start') {
  put(`<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${xs.map((v, i) => `<tspan x="${x}" dy="${i ? gap : 0}">${esc(v)}</tspan>`).join('')}</text>`);
}
function box(x, y, w, h, title, body, cls = 'neutral') {
  put(`<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/>`);
  put(`<text class="bt" x="${x + 14}" y="${y + 23}">${esc(title)}</text>`);
  lines(x + 14, y + 45, body, 'sm', 17);
}
function arrow(x1, y1, x2, y2, cls = 'arr') { put(`<path class="${cls}" d="M ${x1} ${y1} H ${x2}"/>`); }

put(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
 data-contract="tensor=A[S,B,H];gradient=G[S,B,H];shape_words=3;shape_bytes_per_direction=24;transports=ring_exchange,batch_isend_irecv,isend/irecv;backends=NCCL,UCC"
 aria-label="同一 A[S,B,H] activation 与 G[S,B,H] gradient 经 ring_exchange、batch_isend_irecv、独立 isend/irecv 的数据和控制流，并逐项比较等待、host device 同步、launch、SM 与 NCCL UCC 选择。">`);
put(`<style>text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}.card{fill:#fff;stroke:#dfe3e8}.panel{fill:#fafbfc;stroke:#dfe3e8}.title{font-size:19px;font-weight:700;fill:#202630}.sub{font-size:11px;fill:#737b87}.pt{font-size:14px;font-weight:700;fill:#303844}.neutral{fill:#fff;stroke:#b6bdc7}.blue{fill:#eaf1fd;stroke:#2563eb;stroke-width:1.4}.orange{fill:#fcf0e6;stroke:#c3651f;stroke-width:1.4}.bt{font-size:12px;font-weight:700;fill:#303844}.tx{font-size:11px;fill:#46505d}.sm{font-size:10.5px;fill:#606a76}.arr{fill:none;stroke:#2563eb;stroke-width:2;marker-end:url(#ab)}.grad-arr{fill:none;stroke:#c3651f;stroke-width:2;marker-end:url(#ao)}.hdr{font-size:10.5px;font-weight:700;fill:#606a76;text-anchor:middle}.foot{font-size:10.5px;fill:#606a76}</style>`);
put(`<defs><marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#2563eb"/></marker><marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#c3651f"/></marker></defs>`);
put(`<rect class="card" x=".5" y=".5" width="${W - 1}" height="${H - 1}" rx="14"/>`);
put(`<text class="title" x="28" y="38">P2P transport × backend：相同 payload，不同提交与完成语义</text>`);
put(`<text class="sub" x="28" y="60">统一例子：rank r 发送 activation A[S,B,H] 到 r+1；反向时 r+1 发送同形 gradient G[S,B,H] 回 r。蓝=forward data，橙=backward data。</text>`);

put(`<rect class="panel" x="20" y="80" width="1520" height="116" rx="11"/>`);
box(42, 105, 280, 64, 'rank r', ['持有 A；为 G 分配 recv buffer'], 'blue');
box(640, 105, 280, 64, '链路 / PP ProcessGroup', ['wire payload 每方向 Θ(S·B·H)'], 'neutral');
box(1238, 105, 280, 64, 'rank r+1', ['为 A 分配 recv；持有 G'], 'orange');
arrow(322, 125, 640); arrow(920, 125, 1238);
arrow(1238, 153, 920, 'grad-arr'); arrow(640, 153, 322, 'grad-arr');

put(`<rect class="panel" x="20" y="214" width="1520" height="414" rx="11"/>`);
put(`<text class="pt" x="42" y="244">① 三条 transport：每行都搬同一个 A/G；差别在 CPU 提交、request 所有权和 wait</text>`);
const cols = [42, 330, 665, 995, 1250];
['transport / 提交', '控制与数据流', 'wait / 完成边界', 'host ↔ device sync', '成本与适用'].forEach((h, i) => put(`<text class="hdr" x="${cols[i] + [130,150,145,110,130][i]}" y="272">${h}</text>`));
const rows = [
  {
    y: 286, name: 'ring_exchange', cls: 'blue',
    intro: ['单次 API', '四方向组合'],
    a: ['A/G 在同一调用表达', 'send/recv prev/next'],
    b: ['wrapper 返回 []；Megatron', '没有 request handle 可延后 wait'],
    c: ['不走 batch_p2p_sync；立即消费的', '完成保证属于 PyTorch/backend 边界'],
    d: ['1 个高层调用；wire Θ(n)', '要求 build 提供 ring_exchange'],
  },
  {
    y: 392, name: 'batch_isend_irecv', cls: 'orange',
    intro: ['最多 4 个 P2POp', '一次 batch call'],
    a: ['A/G 共用 P2POp list', '由 list 顺序描述方向'],
    b: ['_communicate 断言 wait_on_reqs', '逐个 req.wait() 后才返回'],
    c: ['batch_p2p_sync=true 且非 capture：', '额外 torch.cuda.synchronize()'],
    d: ['1 个 batch launch + ≤4 waits', '简单同步路径；不能供 VPP overlap'],
  },
  {
    y: 498, name: 'isend/irecv（独立）', cls: 'blue',
    intro: ['最多 4 次调用', '命名 request'],
    a: ['偶/奇 rank 反转提交顺序', '使 send/recv 与对端匹配'],
    b: ['wait_on_reqs=true：本调用等待', 'false：返回命名 handle 字典'],
    c: ['无 batch_p2p_sync 的全设备同步；', '消费 recv / 复用 send 前局部 wait'],
    d: ['launch/handle 最多 4；管理复杂', '唯一支持 overlap_p2p_comm 的路径'],
  },
];
for (const r of rows) {
  box(cols[0], r.y, 260, 84, r.name, r.intro, r.cls);
  box(cols[1], r.y, 305, 84, 'data/control', r.a, 'neutral');
  box(cols[2], r.y, 300, 84, 'completion', r.b, 'neutral');
  box(cols[3], r.y, 225, 84, 'sync', r.c, 'neutral');
  box(cols[4], r.y, 268, 84, 'trade-off', r.d, 'neutral');
}

put(`<rect class="panel" x="20" y="646" width="1520" height="134" rx="11"/>`);
put(`<text class="pt" x="42" y="676">② variable sequence shape：先走控制面，再分配 A/G 数据面</text>`);
box(42, 696, 300, 60, 'shape producer', ['每个方向 3 × int64 = 24 B'], 'blue');
box(390, 696, 350, 60, '_communicate_shapes', ['ring_exchange 或 batch_isend_irecv'], 'neutral');
box(788, 696, 350, 60, '完成与主机可见', ['batch: wait + cuda.synchronize；再 .tolist()'], 'orange');
box(1186, 696, 332, 60, 'allocate + data P2P', ['按收到的 [S,B,H] 创建 recv buffer'], 'blue');
arrow(342, 726, 390); arrow(740, 726, 788); arrow(1138, 726, 1186);

put(`<rect class="panel" x="20" y="798" width="1520" height="132" rx="11"/>`);
put(`<text class="pt" x="42" y="828">③ backend：transport API 之下的 PP ProcessGroup</text>`);
box(42, 846, 650, 68, 'NCCL', ['默认 GPU backend；PP-size=2 的 independent 特例可借 WORLD 分开方向', '通信仍使用 GPU backend/SM；沿用现有 NCCL options'], 'blue');
box(720, 846, 798, 68, 'UCC', ['源码以 IB 带宽 / zero-SM 为意图；拒绝 CUDA_DEVICE_MAX_CONNECTIONS=1', 'WORLD 是 NCCL，size=2 仍留 UCC group；需验证 UCC/UCX 环境与版本'], 'orange');
put(`<text class="foot" x="42" y="951">选择线：要 request overlap → independent isend/irecv；要同步且聚合提交 → batch_isend_irecv；只有在已验证 backend contract/build 时选 ring_exchange。暴露 PP/关键 stage 且需释放 SM 时再评估 UCC，否则以 NCCL 为基线。</text>`);
put(`</svg>`);
console.log(out.join('\n'));
