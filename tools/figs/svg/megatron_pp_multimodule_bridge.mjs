// MultiModuleProcessGroupCollection + MultiModulePipelineCommunicator + BridgeCommunicator。
//
// 图 4 spec：在原 DAG / forward-backward / fan-in-out 主线上锁住两个容易混淆的边界契约。
// - leader 左右非对称：source 的 per-DP replica rank group 取 group[-1]，destination 取 group[0]。
// - split metadata 有两种合法形式：len=peers 直接切；len>peers 且整除时，先把连续的逐样本
//   sizes 按 peer 聚合。图内用 [0,3,1,2] / 2 peers → [3,3] 回放，并标出长度/总和拒绝线。
// Usage: node tools/figs/svg/megatron_pp_multimodule_bridge.mjs > .../assets/megatron_pp_multimodule_bridge.svg

const W = 1580, H = 1060;
const splitSampleInput = [0, 3, 1, 2];
const splitSamplePeers = 2;
const samplesPerPeer = splitSampleInput.length / splitSamplePeers;
const splitSampleOutput = Array.from({ length: splitSamplePeers }, (_, peer) => (
  splitSampleInput
    .slice(peer * samplesPerPeer, (peer + 1) * samplesPerPeer)
    .reduce((sum, size) => sum + size, 0)
));
const out = [];
const put = (s) => out.push(s);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function lines(x, y, xs, cls = 'sm', gap = 17, anchor = 'start') {
  put(`<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${xs.map((v, i) => `<tspan x="${x}" dy="${i ? gap : 0}">${esc(v)}</tspan>`).join('')}</text>`);
}
function box(x, y, w, h, title, body = [], cls = 'neutral') {
  put(`<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/>`);
  put(`<text class="bt" x="${x + 13}" y="${y + 23}">${esc(title)}</text>`);
  lines(x + 13, y + 44, body, 'sm', 16);
}
function arrow(x1, y1, x2, y2, cls = 'arr') { put(`<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`); }

put(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
 data-contract="host_P=4;host_m=8;host_span=22;host_bubble=6;host_live=4,3,2,1;batch=8;fanin_src_dp=4;fanin_dst_dp=2;fanin_src_shard=2;fanin_dst_shard=4;fanout_src_dp=2;fanout_dst_dp=4;fanout_src_shard=4;fanout_dst_shard=2;bridge_backend=NCCL;bridge_cp=1;split_peer_form=direct;split_sample_form=grouped;split_sample_input=${splitSampleInput.join(',')};split_sample_peers=${splitSamplePeers};split_sample_output=${splitSampleOutput.join(',')};split_sum_rule=batch_dim;source_leader=group[-1];destination_leader=group[0]"
 aria-label="MultiModuleProcessGroupCollection 选择 non-interleaved 调度；模块 DAG 内用 P2PCommunicator，跨模块用 BridgeCommunicator。图示 source group[-1] 与 destination group[0] 的非对称 leader 选举、DP fan-in/fan-out、per-peer 与 grouped per-sample batch split metadata、边界 broadcast，以及 backward 的逆向交接。">`);
put(`<style>text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}.card{fill:#fff;stroke:#dfe3e8}.panel{fill:#fafbfc;stroke:#dfe3e8}.title{font-size:19px;font-weight:700;fill:#202630}.sub{font-size:11px;fill:#737b87}.pt{font-size:14px;font-weight:700;fill:#303844}.neutral{fill:#fff;stroke:#b6bdc7}.blue{fill:#eaf1fd;stroke:#2563eb;stroke-width:1.5}.orange{fill:#fcf0e6;stroke:#c3651f;stroke-width:1.5}.bt{font-size:12px;font-weight:700;fill:#303844}.sm{font-size:10.5px;fill:#5f6976}.arr{fill:none;stroke:#2563eb;stroke-width:2;marker-end:url(#ab)}.back{fill:none;stroke:#c3651f;stroke-width:2;marker-end:url(#ao)}.dash{fill:none;stroke:#2563eb;stroke-width:1.6;stroke-dasharray:5 4;marker-end:url(#ab)}.lab{font-size:10.5px;font-weight:700;fill:#173f87}.cost{font-size:10.5px;font-weight:700;fill:#8a4a11}.foot{font-size:10.5px;fill:#5f6976}</style>`);
put(`<defs><marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#2563eb"/></marker><marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#c3651f"/></marker></defs>`);
put(`<rect class="card" x=".5" y=".5" width="${W - 1}" height="${H - 1}" rx="14"/>`);
put(`<text class="title" x="28" y="38">Multi-Module Pipeline：module 内 P2P，module 边界 Bridge</text>`);
put(`<text class="sub" x="28" y="60">蓝 = forward / activation，橙 = backward / gradient。split/cat 只重分同一 DAG edge 的 DP shards；多个 incoming module edge 保持独立 dict key。</text>`);

put(`<rect class="panel" x="20" y="80" width="1540" height="194" rx="11"/>`);
put(`<text class="pt" x="42" y="110">① 选择点与 module DAG：每个有向 edge 各自拥有 BridgeCommunicator</text>`);
box(42, 132, 335, 112, 'MultiModuleProcessGroupCollection', ['per-rank: module name → ProcessGroupCollection', '可另标 language_model_module_name', 'get_forward_backward_func → non-interleaved'], 'blue');
box(472, 144, 184, 66, 'vision', ['V0 → V1（module 内 P2P）'], 'neutral');
box(472, 220, 184, 42, 'audio: A0', [], 'neutral');
box(830, 166, 224, 78, 'language_model', ['L0 → L1（module 内 P2P）', 'recv dict: vision, audio'], 'blue');
box(1210, 166, 190, 78, 'heads', ['text / auxiliary', '各 outgoing edge'], 'neutral');
arrow(656, 174, 830, 184); arrow(656, 242, 830, 222); arrow(1054, 205, 1210, 205);
put(`<text class="lab" x="700" y="157">bridge edge #1</text><text class="lab" x="700" y="246">bridge edge #2</text>`);
put(`<text class="cost" x="1420" y="194">DAG 必须无环</text><text class="sm" x="1420" y="214">total_stages = 加权最长路径</text>`);

put(`<rect class="panel" x="20" y="292" width="1540" height="224" rx="11"/>`);
put(`<text class="pt" x="42" y="322">② 一条 edge 的 forward → loss → backward 交接（module A → module B）</text>`);
box(42, 346, 250, 100, 'A: internal stage 0', ['recv_forward → forward_step', 'P2PCommunicator.send_forward'], 'neutral');
box(350, 346, 250, 100, 'A: last PP stage', ['forward output A_edge', 'source leader 位于此 boundary'], 'blue');
box(682, 346, 250, 100, 'BridgeCommunicator', ['leader-to-leader send/recv', 'shape → payload → boundary broadcast'], 'orange');
box(1014, 346, 250, 100, 'B: first PP stage', ['recv_forward 返回 {A: tensor}', 'downstream model 决定如何融合'], 'blue');
box(1322, 346, 196, 100, 'B: sink / loss', ['loss.backward()', '产生 edge gradient'], 'neutral');
arrow(292, 380, 350, 380); arrow(600, 380, 682, 380); arrow(932, 380, 1014, 380); arrow(1264, 380, 1322, 380);
arrow(1322, 420, 1264, 420, 'back'); arrow(1014, 420, 932, 420, 'back'); arrow(682, 420, 600, 420, 'back'); arrow(350, 420, 292, 420, 'back');
put(`<text class="lab" x="700" y="475">forward：module output dict 沿 DAG edge 向下游</text>`);
put(`<text class="cost" x="700" y="495">backward：backward_step_multimodule 独立 autograd，再把 input-grad dict 逆 edge 送回</text>`);

put(`<rect class="panel" x="20" y="534" width="1540" height="306" rx="11"/>`);
put(`<text class="pt" x="42" y="564">③ leader、fan-in / fan-out 与 batch 维重分布（统一 global batch B=8）</text>`);
box(42, 586, 450, 76, 'leader 选举（每个 DP replica）', ['src：module A 的 last PP boundary，per-DP group[-1]', 'dst：module B 的 first PP boundary，per-DP group[0]'], 'neutral');
box(520, 586, 480, 76, 'bridge / broadcast groups', ['bridge_pg 只含 src/dst leaders，硬编码 NCCL', 'dst leader 完成 cat 后向该 DP replica 的 TP×CP boundary broadcast'], 'orange');
box(1028, 586, 490, 76, '边界成员', ['非 leader 不参加跨-module send/recv，只参加 shape+tensor broadcast', 'Bridge 要求两侧 CP=1；TP 可不同'], 'blue');

box(42, 688, 700, 118, 'fan-in：src DP=4 → dst DP=2', ['forward：每个 src leader 持 B/4=2；dst recv 两份 → cat(batch) = B/2=4', 'backward：dst 把 grad B/2 split [2,2]，每个 src 收 B/4，再 boundary broadcast', 'metadata：len=peers 直接作 per-peer split sizes', `逐样本：[${splitSampleInput}] / ${splitSamplePeers} peers → 连续分组求和 [${splitSampleOutput}]`, '长度不合法或 sum ≠ batch dim → ValueError'], 'blue');
box(778, 688, 740, 118, 'fan-out：src DP=2 → dst DP=4', ['forward：每个 src 的 B/2=4 split [2,2]，两个 dst 各收 B/4=2，再 boundary broadcast', 'backward：两个 dst grad 各 B/4；src leader recv 两份 → cat(batch) = B/2=4', 'leader 数必须整除；不整除在 build_comm_map 处拒绝'], 'orange');
put(`<text class="foot" x="42" y="826">注意：上面的 cat 合并的是同一 module edge 的 DP shards；vision 与 audio 两条 incoming edge 仍是 {vision: tensor, audio: tensor}，由 B 的 forward 代码决定 concat/add/其它融合。</text>`);

put(`<rect class="panel" x="20" y="858" width="1540" height="166" rx="11"/>`);
put(`<text class="pt" x="42" y="888">④ 付款点与拒绝线</text>`);
box(42, 906, 350, 88, '通信增量', ['每条 DAG edge：shape exchange + leader P2P', '+ boundary shape/data broadcast；fan-in/out 多 peer'], 'orange');
box(414, 906, 350, 88, '内存 / 同步', ['split 产 contiguous shards，cat 产新 tensor', 'paired API batch_isend_irecv 后等待；init 全局 barrier'], 'neutral');
box(786, 906, 350, 88, '形状 / 调度边界', ['Bridge 原生 2D/3D；module 内 2D 临时升 3D', 'MultiModule 强制 variable_seq_lengths；不进入 VPP'], 'blue');
box(1158, 906, 360, 88, '实现 caveat', ['current_stage 取 first local module（源码 TODO）', '跨模块 group 构造属 orchestration owner'], 'neutral');
put(`<text class="foot" x="42" y="1017">Host 账（P=4,m=8，加权最长路径=4）：22t_f、bubble=6、live=[4,3,2,1]；Bridge 的 shape/leader/split-cat/broadcast 是增量，不会消掉这些 bubble。</text>`);
put(`<text class="foot" x="42" y="1042">陌生读者线：先找“这是 module 内还是 module edge？”；edge 再问 DP leader 比例是否整除、batch split/cat 后由谁 broadcast，最后沿橙色箭头确认 gradient 能回到原 source module。</text>`);
put(`</svg>`);
console.log(out.join('\n'));
