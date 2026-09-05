// 图 1：一个描述子如何把并行布局编码进 checkpoint —— TP4×DP2 存、TP2×DP1 载的同一张全局张量。
// 图 2：访问计数校验为什么必须恰好等于 1，以及缺口/重叠/免检三种结局。
// 图 3：fully-parallel save 的贪心分配 —— 四条排序键把写盘工作摊平。
// 图 4：fully-parallel load 的三条 exchange 数据面在同一个例子上的报文、空载与代价。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要讲清楚：ShardedTensor 是纯元数据的描述子，"换并行布局加载"不是一次数据搬运，
// 而是**存端与载端各自声明网格、由 DCP 求交**。四件事必须从图上直接读出来：
//   ① 存端声明：8 个 rank（TP=4 × DP=2）各自给出 (global_offset, local_shape,
//      axis_fragmentations, replica_id)。同一 TP 位置的两个 DP rank 描述子**完全相同**，
//      只有 replica_id 不同 —— 这正是 keep_only_main_replica 能去冗余的依据。
//   ② 盘上形态：写盘量是 4 片 × 16 B = 64 B，而不是 8 片 128 B；DP 副本被整片丢弃。
//   ③ 载端声明：2 个 rank（TP=2）各要 [4,4]，axis_fragmentations 变成 (2,1)。
//      载端的描述子里没有任何"上一次是 TP4"的信息 —— 它只描述自己想要什么。
//   ④ 求交发生在哪：存端 4 个 chunk 与载端 2 个 chunk 的**矩形求交**，每个交集是一条
//      ReadItem。这一步在 PyTorch DCP 内，Megatron 只负责把 chunk 清单递过去 ——
//      用 ghost 底 + 虚线边界 + "PyTorch DCP" 芯片标出依赖边界。
// 强调色：acc1 = 载端要的那一片（收益：换布局成立），acc2 = 被丢弃的 DP 副本（省下的代价）。
//
// 图 2 要讲清楚：validate_sharding_integrity 检的是**chunk 网格上的访问计数**，不是元素，
// 计数张量的形状就是 axis_fragmentations，下标是 global_offset // local_shape。三条 lane
// 用图 1 的同一个例子，共用同一张 4 格计数条：
//   ① 正确：4 个 main replica 各命中一格 → [1,1,1,1]。
//   ② 缺口：某个 rank 忘了把自己标成 main（replica_id 全非 0）→ 该格 0，报 gap。
//   ③ 重叠：两个 rank 都自认 main（例如 fully-parallel 分配被绕开）→ 该格 2，报 overlap。
// 右列放三件正文要用的事实：校验代价（all_gather_object 收全量元数据、只有 rank 0 检查）、
// ShardedObject 走另一条规则（unique_key 去重 + 数量必须等于 prod(global_shape)）、
// 以及**免检口子**：has_regular_grid=False 时校验直接返回、交给 DCP。
//
// 图 3 要讲清楚：贪心分配凭什么能把写盘摊平，以及四条排序键各自防住什么。
// 布局：上排 6 个分片卡（大小 + 覆盖 rank 集合）→ 中排排序后的执行顺序（标出是哪条键决定的）
//   → 下排 6 步分配轨迹（每步显示分配给谁、该 rank 的累计字节）
//   → 右下两组共标尺条形：无 wrapper（只有 replica_id=0 的 rank 写，全部落在一张卡上）
//     vs 有 wrapper（四张卡等量）。等长与单柱的对比就是本图要证的结论。
// 强调色：acc1 = 摊平后的结果（收益），acc2 = 无 wrapper 时那根独柱（代价）。
//
// 图 4 要讲清楚：同一份分配结果在三条 exchange 数据面上分别发几条报文、空载多少、贵在哪。
// 沿用 13 号页图 2 的五列 lane 语法（选择条件 / 报文构造 / 本例报文序列 / 空载与浪费 / 增量代价），
// 三条 lane 各自可追。共享的第一条规则单独提出来：coverage==1 的分片整条跳过交换。
//
// 硬规矩：图上每个数字（字节数、覆盖数、轮数、报文数、空载条数、交集矩形）都由 CFG 与
// 复刻的源码算法算出，不手写；每行文字过一遍宽度守卫，放不下直接抛错，杜绝裁字。
//
// 用法：node tools/figs/svg/megatron_dist_checkpointing_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  // 图 1/图 2 的例子：一张按行切的权重
  key: 'layer0.mlp.fc1.weight',
  rows: 8, // 全局张量行数（out_features）
  cols: 4, // 全局张量列数（in_features）
  bytes: 2, // bf16
  saveTp: 4,
  saveDp: 2,
  loadTp: 2,
  loadDp: 1,
  // 图 3/图 4 的例子：一个 4-rank 并行化组里待分配的六个分片
  group: 4,
  shards: Object.freeze([
    Object.freeze({ key: 'decoder.layers.0.mlp.fc1.weight', bytes: 8192, ranks: [0, 1, 2, 3] }),
    Object.freeze({ key: 'decoder.layers.0.mlp.fc2.weight', bytes: 8192, ranks: [0, 1, 2, 3] }),
    Object.freeze({ key: 'decoder.layers.0.self_attn.qkv.weight', bytes: 4096, ranks: [0, 1, 2, 3] }),
    Object.freeze({ key: 'decoder.layers.1.self_attn.qkv.weight', bytes: 4096, ranks: [0, 1, 2, 3] }),
    Object.freeze({ key: 'decoder.layers.0.mlp.experts.w1', bytes: 6144, ranks: [0, 1] }),
    Object.freeze({ key: 'embedding.word_embeddings.weight', bytes: 2048, ranks: [0, 1, 2, 3] }),
  ]),
});

const range = (start, count) => Array.from({ length: count }, (_, i) => start + i);
const sum = (values) => values.reduce((a, b) => a + b, 0);

// ============================================================================
// 复刻源码算法（图上的数字全部由这一段算出）
// ============================================================================

// megatron/core/dist_checkpointing/mapping.py::ShardedTensor.from_rank_offsets
// 沿 axis 0 切 `fragm` 份、本 rank 取第 `rankOffset` 块。
function shardOf(rankOffset, fragm, replicaId) {
  const localRows = CFG.rows / fragm;
  return {
    key: CFG.key,
    localShape: [localRows, CFG.cols],
    globalShape: [CFG.rows, CFG.cols],
    globalOffset: [rankOffset * localRows, 0],
    axisFragmentations: [fragm, 1],
    replicaId,
  };
}

// mapping.py::ShardedTensor.local_chunk_offset_in_global
const chunkOffset = (sh) =>
  sh.globalOffset.map((off, axis) => {
    const local = axis === 0 ? sh.localShape[0] : sh.localShape[1];
    if (off % local !== 0) throw new Error(`global_offset ${off} 不能被 local_shape ${local} 整除`);
    return off / local;
  });

// mapping.py::ShardedTensor.global_slice —— 本片在全局张量里的行区间
const rowSpan = (sh) => [sh.globalOffset[0], sh.globalOffset[0] + sh.localShape[0]];
const shardBytes = (sh) => sh.localShape[0] * sh.localShape[1] * CFG.bytes;

// 存端：TP=4 × DP=2，8 个 rank。rank = dp * saveTp + tp
const saveRanks = range(0, CFG.saveTp * CFG.saveDp).map((rank) => {
  const tp = rank % CFG.saveTp;
  const dp = Math.floor(rank / CFG.saveTp);
  return { rank, tp, dp, sh: shardOf(tp, CFG.saveTp, dp) };
});
// strategies/torch.py::_replace_state_dict_keys_with_sharded_keys（keep_only_main_replica=True）
// mapping.py::is_main_replica —— 整数 0 才是 main
const writtenShards = saveRanks.filter((r) => r.sh.replicaId === 0);
const droppedShards = saveRanks.filter((r) => r.sh.replicaId !== 0);
const writtenBytes = sum(writtenShards.map((r) => shardBytes(r.sh)));
const allShardBytes = sum(saveRanks.map((r) => shardBytes(r.sh)));

// 载端：TP=2 × DP=1
const loadRanks = range(0, CFG.loadTp * CFG.loadDp).map((rank) => ({
  rank,
  sh: shardOf(rank, CFG.loadTp, 0),
}));

// PyTorch DCP 的公开契约：create_read_items_for_chunk_list 对存/载两侧 chunk 求矩形交集。
// 本例只沿 axis 0 切，交集退化为行区间求交。
function readItems(loadShard) {
  const [wantLo, wantHi] = rowSpan(loadShard);
  const items = [];
  writtenShards.forEach(({ sh }, chunkIdx) => {
    const [haveLo, haveHi] = rowSpan(sh);
    const lo = Math.max(wantLo, haveLo);
    const hi = Math.min(wantHi, haveHi);
    if (hi > lo) items.push({ chunkIdx, rows: [lo, hi], bytes: (hi - lo) * CFG.cols * CFG.bytes });
  });
  return items;
}

// validation.py::_compute_shards_access —— 计数张量形状 = axis_fragmentations
function shardAccessCount(entries, fragm) {
  const counts = new Array(fragm).fill(0);
  for (const { sh } of entries) if (sh.replicaId === 0) counts[chunkOffset(sh)[0]] += 1;
  return counts;
}
const accessOk = shardAccessCount(saveRanks, CFG.saveTp);
// 缺口：tp=2 的两个 rank 都没把自己标成 main
const accessGap = shardAccessCount(
  saveRanks.map((r) => (r.tp === 2 ? { ...r, sh: { ...r.sh, replicaId: 1 } } : r)),
  CFG.saveTp,
);
// 重叠：tp=1 的 DP 副本也自认 main
const accessDup = shardAccessCount(
  saveRanks.map((r) => (r.tp === 1 ? { ...r, sh: { ...r.sh, replicaId: 0 } } : r)),
  CFG.saveTp,
);

// exchange_utils.py::distribute_shards_to_ranks —— 四条排序键 + 贪心取最闲 rank
function distributeShardsToRanks(shards, numRanks, crossGroup = new Set()) {
  const order = [...shards].sort((a, b) => {
    const ka = [crossGroup.has(a.key) ? 1 : 0, a.ranks.length, -a.bytes, a.key];
    const kb = [crossGroup.has(b.key) ? 1 : 0, b.ranks.length, -b.bytes, b.key];
    for (let i = 0; i < ka.length; i += 1) {
      if (ka[i] < kb[i]) return -1;
      if (ka[i] > kb[i]) return 1;
    }
    return 0;
  });
  const rankSizes = range(0, numRanks).map(() => 0);
  const trace = [];
  for (const shard of order) {
    // min((size, rank) for size, rank in rank_sizes if rank in shard_ranks)
    let pick = null;
    for (const rank of [...shard.ranks].sort((a, b) => a - b)) {
      if (pick === null || rankSizes[rank] < rankSizes[pick]) pick = rank;
    }
    rankSizes[pick] += shard.bytes;
    trace.push({ shard, rank: pick, after: [...rankSizes] });
  }
  return { order, trace, rankSizes };
}

const distribution = distributeShardsToRanks(CFG.shards, CFG.group);
const totalBytes = sum(CFG.shards.map((s) => s.bytes));
const balancedMax = Math.max(...distribution.rankSizes);
// 无 wrapper：底层策略只让 replica_id==0 的 rank 写，本组里就是组内 rank 0 全写
const naiveSizes = range(0, CFG.group).map((rank) => (rank === 0 ? totalBytes : 0));
const speedup = totalBytes / balancedMax;

// exchange_utils.py：coverage==1 的分片整条跳过交换
const exchanged = distribution.trace.filter((t) => t.shard.ranks.length > 1);
const skipped = distribution.trace.filter((t) => t.shard.ranks.length === 1);

// exchange_loaded_tensors_gather_rounds：按 dtype 分组 → shards_by_rank → zip_longest 转置成轮
const shardsByRank = range(0, CFG.group).map((rank) =>
  exchanged.filter((t) => t.rank === rank).map((t) => t.shard),
);
const roundCount = Math.max(...shardsByRank.map((s) => s.length));
const rounds = range(0, roundCount).map((r) =>
  shardsByRank.map((shards) => shards[r] ?? null), // fillvalue=None → torch.empty(0)
);
const emptySlots = sum(rounds.map((r) => r.filter((s) => s === null).length));
const roundSlots = roundCount * CFG.group;
// broadcast：每个待交换分片一条 broadcast
const broadcastMsgs = exchanged.length;

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_cp_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) =>
  String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// 宽度守卫：非 ASCII 一律按全宽估算（宁可高估，也不让文字被裁掉）
function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) {
    throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  }
  return value;
}

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
  .edge{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:4 4}
  .ti{font-size:18px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .pt{font-size:14px;font-weight:700;fill:#2A313B}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#68717D}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .rank{font-size:11px;font-weight:700;fill:#5B6470}
  .cap{font-size:11px;fill:#747C88}
  .gl{fill:none;stroke:#C8CFDA;stroke-width:.9}
  .cell0{fill:#DCE9FB;stroke:#fff;stroke-width:.8}
  .cell1{fill:#9CC2F3;stroke:#fff;stroke-width:.8}
  .cellx{fill:#F5F7FA;stroke:#fff;stroke-width:.8}
  .cellw{fill:#F6DFC8;stroke:#fff;stroke-width:.8}
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

function line(x1, y1, x2, y2, cls = 'edge') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}

function arrow(x1, y1, x2, y2, cls = 'main') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}

function infoBox(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 24;
  out.push(text(x + 12, y + 21, guard(title, 12, inner, `${where}/title`), 'tx'));
  out.push(line(x + 12, y + 29, x + w - 12, y + 29));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(
      text(x + 12, y + 47 + index * 16, guard(value, 10.5, inner, `${where}/L${index}`), lineCls),
    );
  });
  return out.join('\n');
}

function chip(x, y, w, h, label, cls = 'ghost') {
  return [
    rect(x, y, w, h, cls, 6),
    text(x + w / 2, y + h / 2 + 4, guard(label, 10.5, w - 10, 'chip'), 'sm', 'middle'),
  ].join('\n');
}

function header(w, title, subtitle) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} __H__" width="${w}" height="__H__" role="img">`,
    defs,
    `<style>${sharedStyle}</style>`,
    text(28, 32, title, 'ti'),
    text(28, 52, subtitle, 'su'),
  ];
}

// 字号表：assertNoTextOverlap 要按 class 还原每条文字的实际盒子
const FONT = Object.freeze({
  ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5,
  costtx: 10.5, rank: 11, cap: 11,
});

// 「越界不等于不重叠」：只校验 viewBox 挡不住"标注压在邻列文字上、把行首字形裁掉"。
// 这里把每条 <text> 还原成包围盒，两两求交 —— 版面错误在生成时就红，不靠肉眼。
//
// 反过来也一样：**锚点在画布内不等于字形在画布内**。测试里的 assertInsideCanvas 只看
// `x`/`y` 两个锚点坐标，一条右端超出画布的长标注照样通过。所以这里同时按还原出的盒子
// 校验右/下边界。
function assertNoTextOverlap(svg, name) {
  const { w: canvasW, h: canvasH } = (() => {
    const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
    return { w: Number(m[1]), h: Number(m[2]) };
  })();
  const boxes = [];
  for (const m of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      throw new Error(
        `${name}: 文字盒出画布 "${b.raw}"（${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} × ` +
          `${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)}，画布 ${canvasW}×${canvasH}）`,
      );
    }
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) {
        throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
      }
    }
  }
  return svg;
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

// ============================== 图 1 ==============================

// 全局 [rows, cols] 张量画成一张网格；band 决定每一行属于哪个 chunk
function tensorGrid(x, y, cell, bands) {
  const out = [];
  for (let r = 0; r < CFG.rows; r += 1) {
    const band = bands(r);
    for (let c = 0; c < CFG.cols; c += 1) {
      out.push(
        `<rect class="${band.cls}" x="${x + c * cell}" y="${y + r * cell}" width="${cell}" height="${cell}"/>`,
      );
    }
  }
  for (let r = 0; r <= CFG.rows; r += 1) out.push(line(x, y + r * cell, x + CFG.cols * cell, y + r * cell, 'gl'));
  for (let c = 0; c <= CFG.cols; c += 1) out.push(line(x + c * cell, y, x + c * cell, y + CFG.rows * cell, 'gl'));
  return out.join('\n');
}

function renderReshardFigure() {
  const W = 1180;
  const H = 690;
  const p = header(
    W,
    '图 1　同一张权重：TP4×DP2 存、TP2×DP1 载',
    `${CFG.key} 全局 [${CFG.rows}, ${CFG.cols}] bf16 = ${CFG.rows * CFG.cols * CFG.bytes} B。描述子只说"我是全局的哪一块"，两端都不知道对方的并行度。`,
  );

  // ---------- 行 1：存端 ----------
  p.push(rect(20, 70, W - 40, 208, 'panel'));
  p.push(text(36, 94, '① 存端：8 个 rank 各交一份纯元数据的描述子', 'pt'));
  p.push(
    text(
      36,
      112,
      `同一 TP 位置的两个 DP rank 描述子逐字段相同，只有 replica_id 不同 —— 去冗余的全部依据就是这一个字段。`,
      'sm',
    ),
  );

  const cardW = 186;
  const cardH = 66;
  saveRanks.forEach(({ rank, tp, dp, sh }) => {
    const x = 36 + tp * (cardW + 10);
    const y = 126 + dp * (cardH + 8);
    const main = sh.replicaId === 0;
    p.push(rect(x, y, cardW, cardH, main ? 'acc1' : 'acc2'));
    p.push(text(x + 10, y + 18, `rank ${rank}  (tp=${tp}, dp=${dp})`, 'rank'));
    p.push(
      text(x + 10, y + 35, guard(`off=(${sh.globalOffset.join(',')})  shape=(${sh.localShape.join(',')})`, 10.5, cardW - 20, 'fig1/save'), 'sm'),
    );
    p.push(
      text(
        x + 10,
        y + 52,
        `frag=(${sh.axisFragmentations.join(',')})  replica_id=${sh.replicaId}`,
        main ? 'dim' : 'costtx',
      ),
    );
  });

  p.push(text(816, 140, '全局张量与 chunk 网格', 'tx'));
  p.push(tensorGrid(820, 150, 12, (r) => ({ cls: Math.floor(r / 2) % 2 === 0 ? 'cell0' : 'cell1' })));
  for (let k = 0; k < CFG.saveTp; k += 1) {
    p.push(text(878, 150 + k * 24 + 16, `chunk ${k} = 行 ${k * 2}–${k * 2 + 1}`, 'sm'));
  }
  p.push(text(816, 262, 'axis_fragmentations=(4,1)：网格被切成 4×1 个 chunk', 'sm'));

  // ---------- 行 2：盘上 ----------
  p.push(rect(20, 292, W - 40, 122, 'panel'));
  p.push(text(36, 316, '② 盘上：只有 4 个 main replica 进入写盘计划', 'pt'));
  p.push(
    text(
      36,
      334,
      `keep_only_main_replica 默认 True：写盘量 ${writtenBytes} B 而不是 ${allShardBytes} B，省掉的正是 ${droppedShards.length} 份 DP 副本。`,
      'sm',
    ),
  );
  writtenShards.forEach(({ rank, sh }, idx) => {
    const x = 36 + idx * 196;
    p.push(rect(x, 346, 186, 56, 'acc1'));
    p.push(text(x + 10, 366, `chunk ${idx} ← rank ${rank}`, 'rank'));
    p.push(
      text(x + 10, 385, `offset=(${sh.globalOffset.join(',')}) size=(${sh.localShape.join(',')}) ${shardBytes(sh)} B`, 'sm'),
    );
  });
  p.push(rect(824, 346, 320, 56, 'acc2'));
  p.push(text(834, 366, `丢弃：rank ${droppedShards.map((r) => r.rank).join(', ')}（replica_id=1）`, 'rank'));
  p.push(text(834, 385, `省下 ${allShardBytes - writtenBytes} B 写盘与同等的 I/O 时间`, 'costtx'));

  // ---------- 行 3：载端 ----------
  p.push(rect(20, 428, W - 40, 200, 'panel'));
  p.push(text(36, 452, '③ 载端：换成 TP=2，两个 rank 各要 [4,4]', 'pt'));
  p.push(
    text(
      36,
      470,
      '载端描述子里没有任何"上次存的是 TP4"的信息；它只声明自己想要哪一块，求交由 DCP 完成。',
      'sm',
    ),
  );

  loadRanks.forEach(({ rank, sh }) => {
    const y = 486 + rank * 66;
    p.push(rect(36, y, 250, 56, 'acc1'));
    p.push(text(46, y + 20, `rank ${rank}  (tp=${rank})`, 'rank'));
    p.push(
      text(46, y + 39, `off=(${sh.globalOffset.join(',')}) shape=(${sh.localShape.join(',')}) frag=(${sh.axisFragmentations.join(',')})`, 'dim'),
    );

    const items = readItems(sh);
    items.forEach((item, i) => {
      const x = 400 + i * 214;
      p.push(rect(x, y, 204, 56, 'neutral'));
      p.push(text(x + 10, y + 20, `ReadItem ${i}：chunk ${item.chunkIdx}`, 'rank'));
      p.push(text(x + 10, y + 39, `全局行 ${item.rows[0]}–${item.rows[1] - 1}，${item.bytes} B`, 'sm'));
      p.push(arrow(286, y + 28, 394, y + 28, 'main'));
    });
    p.push(
      text(
        836,
        y + 33,
        `${items.length} 条 ReadItem 拼出 ${sh.localShape.join('×')} = ${shardBytes(sh)} B`,
        'dim',
      ),
    );
  });

  p.push(rect(1024, 486, 132, 122, 'ghost'));
  p.push(chip(1030, 480, 120, 22, 'PyTorch DCP', 'acc2'));
  p.push(text(1034, 522, '求交在这里：', 'sm'));
  p.push(text(1034, 538, 'create_read_items_', 'sm'));
  p.push(text(1034, 552, '  for_chunk_list', 'sm'));
  p.push(text(1034, 570, 'Megatron 只递', 'sm'));
  p.push(text(1034, 584, 'chunk 清单，', 'sm'));
  p.push(text(1034, 598, '不做这一步', 'sm'));

  p.push(
    text(
      28,
      H - 42,
      `成立的两条前置：len(local_shape)+prepend_axis_num == len(global_shape)，且每个轴上 global_offset % local_shape == 0（validate_metadata_integrity）。第二条就是"规则网格"的落点。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      `注意存端与载端的 global_shape 必须一致（[${CFG.rows}, ${CFG.cols}]）：MCoreLoadPlanner._validate_global_shapes 不匹配就抛 CheckpointingException，除非该张量声明 allow_shape_mismatch。`,
      'cap',
    ),
  );
  return seal(p, W, H, 'fig1');
}

// ============================== 图 2 ==============================

function accessLane(x, y, w, label, counts, verdict, verdictCls, notes) {
  const out = [];
  out.push(text(x, y, guard(label, 14, w, 'fig2/label'), 'pt'));
  const cell = 46;
  counts.forEach((count, idx) => {
    const cls = count === 1 ? 'acc1' : 'acc2';
    out.push(rect(x + idx * (cell + 8), y + 14, cell, cell, cls, 6));
    out.push(text(x + idx * (cell + 8) + cell / 2, y + 14 + cell / 2 + 6, count, count === 1 ? 'dim' : 'costtx', 'middle'));
    out.push(text(x + idx * (cell + 8) + cell / 2, y + 14 + cell + 15, `chunk ${idx}`, 'sm', 'middle'));
  });
  out.push(text(x, y + 14 + cell + 40, guard(verdict, 11, w, 'fig2/verdict'), verdictCls));
  notes.forEach((note, i) =>
    out.push(text(x, y + 14 + cell + 58 + i * 16, guard(note, 10.5, w, 'fig2/note'), 'sm')),
  );
  return out.join('\n');
}

function renderAccessFigure() {
  const W = 1180;
  const H = 470;
  const p = header(
    W,
    '图 2　访问计数校验：检的是 chunk 网格，不是元素',
    `计数张量形状 = axis_fragmentations = (${CFG.saveTp},1)，下标 = global_offset ÷ local_shape；只有 is_main_replica 的分片计数。判据是"每格恰好 1"。`,
  );

  p.push(rect(20, 70, 800, 316, 'panel'));
  p.push(
    accessLane(
      40,
      100,
      760,
      '① 正确：4 个 main replica 各命中一格',
      accessOk,
      'torch.all(cnt == 1) 成立 → 通过',
      'dim',
      [`4 份 DP 副本 replica_id=1 不计数；全局 ${CFG.rows} 行被无缺口、无重叠地覆盖一次。`],
    ),
  );
  p.push(
    accessLane(
      40,
      234,
      344,
      '② 缺口：chunk 2 的两个 rank 都没标成 main',
      accessGap,
      'cnt[2]=0 → CheckpointingException',
      'costtx',
      ['checkpoint 会缺全局行 4–5。', '挡的是"算错 replica_id / 漏声明"。'],
    ),
  );
  p.push(
    accessLane(
      436,
      234,
      344,
      '③ 重叠：chunk 1 的 DP 副本也自认 main',
      accessDup,
      'cnt[1]=2 → 同一条异常',
      'costtx',
      ['两个 rank 写同一段，谁后写谁赢。', '挡的是"分配被绕开或算重"。'],
    ),
  );

  p.push(
    infoBox(
      840,
      70,
      316,
      142,
      '校验自己的代价',
      [
        'determine_global_metadata 用 all_gather_object',
        '把每个 rank 的**全部**分片元数据收齐 ——',
        '元数据量随 world size 线性增长；',
        '随后 `if get_rank() != 0: return`，',
        '整个检查只在 global rank 0 上串行跑。',
      ],
      'acc2',
      'fig2/cost',
    ),
  );
  p.push(
    infoBox(
      840,
      224,
      316,
      146,
      '两个不走这条规则的口子',
      [
        'ShardedObject 走 _validate_objects_for_key：',
        'unique_key 不许重复，且数量必须等于',
        'prod(global_shape)。common_state 正是靠',
        'replica_id=get_rank() 让 rank 0 成为唯一 main。',
        '',
        'has_regular_grid=False（axis_fragmentations',
        '为 None）时函数直接 return —— 不均匀分片的',
        '校验被让给 DCP，本页这条网格判据不适用。',
      ],
      'ghost',
      'fig2/exempt',
    ),
  );

  p.push(
    text(
      28,
      H - 42,
      `save 与 load 都会跑这一步：save 侧由 save_preprocess 调用，fully-parallel save 在第一次算完分配后再跑一次（cached_distribution 为 None 时）。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      `它证明的是"分片声明自洽"，不是"盘上数据正确"。要校验落盘字节需要另一条开关：save/load 的 verify_integrity 会对全部文件多做一遍 SHA-256。`,
      'cap',
    ),
  );
  return seal(p, W, H, 'fig2');
}

// ============================== 图 3 ==============================

function barRow(x, y, value, scaleMax, barMax, cls, label, h = 18) {
  const width = Math.max(2, Math.round((value / scaleMax) * barMax));
  return [
    text(x - 8, y + h - 5, label, 'rank', 'end'),
    rect(x, y, width, h, cls, 5),
    text(x + width + 8, y + h - 5, `${value} B`, cls === 'acc2' ? 'costtx' : 'dim'),
  ].join('\n');
}

function renderGreedySaveFigure() {
  const W = 1180;
  const H = 700;
  const p = header(
    W,
    '图 3　fully-parallel save：四条排序键把写盘摊平',
    `并行化组 ${CFG.group} 个 rank、${CFG.shards.length} 个分片、合计 ${totalBytes} B。分配只交换元数据，不搬一个字节的数据。`,
  );

  // ---------- 上排：分片 ----------
  p.push(rect(20, 70, W - 40, 118, 'panel'));
  p.push(text(36, 94, '① 输入：每个分片的大小，以及它在组内被哪些 rank 持有（coverage）', 'pt'));
  CFG.shards.forEach((shard, idx) => {
    const x = 36 + idx * 186;
    const low = shard.ranks.length < CFG.group;
    p.push(rect(x, 106, 176, 64, low ? 'acc2' : 'neutral'));
    p.push(text(x + 10, 124, guard(shard.key.replace('decoder.layers.', 'L').replace('.weight', ''), 10.5, 156, 'fig3/key'), 'rank'));
    p.push(text(x + 10, 142, `${shard.bytes} B`, low ? 'costtx' : 'dim'));
    p.push(text(x + 10, 160, `coverage ${shard.ranks.length}：{${shard.ranks.join(',')}}`, 'sm'));
  });

  // ---------- 中排：排序 ----------
  p.push(rect(20, 202, W - 40, 150, 'panel'));
  p.push(text(36, 226, '② 排序键 (跨组, coverage, −size, shard_id)，从左到右依次比较', 'pt'));
  p.push(
    text(
      36,
      244,
      'coverage 低的先分（可选 rank 少，先占住），同 coverage 里大的先分（大块先摆才填得平），shard_id 兜底保证各 rank 算出同一个顺序。',
      'sm',
    ),
  );
  distribution.order.forEach((shard, idx) => {
    const x = 36 + idx * 186;
    const reason =
      idx === 0
        ? '键②：coverage 最低'
        : shard.bytes === distribution.order[idx - 1].bytes
          ? '键④：shard_id 兜底'
          : '键③：size 降序';
    p.push(rect(x, 260, 176, 72, idx === 0 ? 'acc2' : 'neutral'));
    p.push(text(x + 10, 278, `第 ${idx + 1} 个`, 'rank'));
    p.push(text(x + 10, 296, guard(shard.key.replace('decoder.layers.', 'L').replace('.weight', ''), 10.5, 156, 'fig3/order'), 'sm'));
    p.push(text(x + 10, 314, `${shard.bytes} B · cov ${shard.ranks.length}`, 'dim'));
    p.push(text(x + 10, 328, guard(reason, 10.5, 156, 'fig3/reason'), 'sm'));
  });

  // ---------- 下排左：分配轨迹 ----------
  p.push(rect(20, 366, 704, 244, 'panel'));
  p.push(text(36, 390, '③ 贪心：每步给"该分片可选 rank 里累计最少的那个"', 'pt'));
  p.push(text(36, 408, '取 min((size, rank))，size 相同时 rank 号小的赢 —— 顺序在所有 rank 上一致，无需再通信。', 'sm'));
  const colX = [44, 236, 320, 404, 488, 572];
  p.push(text(colX[0], 434, '分配的分片', 'rank'));
  range(0, CFG.group).forEach((rank) => p.push(text(colX[2 + rank], 434, `rank ${rank}`, 'rank', 'middle')));
  p.push(text(656, 434, '落到', 'rank'));
  distribution.trace.forEach((step, idx) => {
    const y = 452 + idx * 25;
    p.push(line(44, y + 5, 700, y + 5, 'gl'));
    p.push(text(colX[0], y, guard(step.shard.key.replace('decoder.layers.', 'L').replace('.weight', ''), 10.5, 184, 'fig3/trace'), 'sm'));
    step.after.forEach((bytes, rank) => {
      p.push(text(colX[2 + rank], y, bytes, rank === step.rank ? 'dim' : 'sm', 'middle'));
    });
    p.push(text(656, y, `rank ${step.rank}`, 'dim'));
  });

  // ---------- 下排右：结果对照 ----------
  p.push(rect(740, 366, W - 760, 258, 'panel'));
  p.push(text(756, 390, '④ 结果：同一把标尺下的写盘量', 'pt'));
  p.push(text(756, 410, '无 wrapper（底层策略只让 main replica 写）', 'sm'));
  naiveSizes.forEach((value, rank) => {
    p.push(barRow(830, 416 + rank * 24, value, totalBytes, 250, value > 0 ? 'acc2' : 'ghost', `rank ${rank}`));
  });
  p.push(text(756, 526, '有 wrapper（贪心分配后）', 'sm'));
  distribution.rankSizes.forEach((value, rank) => {
    p.push(barRow(830, 532 + rank * 20, value, totalBytes, 250, 'acc1', `rank ${rank}`, 16));
  });

  p.push(
    text(
      28,
      H - 62,
      `本例的关键路径从 ${totalBytes} B 降到 ${balancedMax} B，即 ${speedup.toFixed(1)}× —— 上限就是组大小 ${CFG.group}，因为写盘时间由最慢的那个 rank 决定。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 42,
      `在纯 DP 组里所有分片 coverage 相等，键② 不起作用、只剩键③ 在填平；coverage 出现差异要靠把组换成 ep_dp 这类跨内容的组。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      `代价：一次 all_gather_object 收全组元数据。do_cache_distribution 可以把它省成一次，但前提是每次调用的 state dict 结构完全相同。`,
      'cap',
    ),
  );
  return seal(p, W, H, 'fig3');
}

// ============================== 图 4 ==============================

function renderExchangeFigure() {
  const W = 1240;
  const H = 810;
  const p = header(
    W,
    '图 4　fully-parallel load 的三条 exchange 数据面',
    `沿用图 3 的分配结果：${exchanged.length} 个分片需要交换，${skipped.length} 个 coverage=1 的分片整条跳过。三条 lane 用同一个例子。`,
  );

  // 共享的第一条规则
  p.push(rect(20, 70, W - 40, 98, 'panel'));
  p.push(text(36, 94, '三条 lane 共用的第一条规则：coverage == 1 就不交换', 'pt'));
  p.push(
    text(
      36,
      112,
      'if len(all_ranks_for_shard[shard_id]) == 1: continue —— 只有加载它的那个 rank 需要它，直接留在本地。',
      'sm',
    ),
  );
  p.push(
    text(
      36,
      128,
      '源码同处留着 TODO：coverage > 1 时也可以改成 P2P 交换，"Currently handling this case saves most of the work though"。',
      'sm',
    ),
  );
  p.push(
    text(
      36,
      150,
      `本例 ${skipped.length === 0 ? '没有 coverage=1 的分片，六个全部进入交换' : `跳过 ${skipped.length} 个`}；余下 ${exchanged.length} 个按下面三种方式散开。`,
      'sm',
    ),
  );

  const headings = ['选择条件', '报文构造', '本例的报文序列', '空载与浪费', '增量代价'];
  const colX = [28, 250, 472, 762, 992];
  const colW = [212, 212, 280, 220, 220];
  const boxH = 160;

  const roundText = rounds.map(
    (r, i) =>
      `第 ${i + 1} 轮：${r
        .map((s) => (s === null ? '空' : `${s.bytes}B`))
        .join(' / ')}`,
  );

  const lanes = [
    {
      name: '① broadcast（默认）',
      pick: [
        'ckpt_fully_parallel_load_',
        '  exchange_algo="broadcast"',
        '',
        'FullyParallelLoadStrategy',
        'Wrapper 的构造默认值。',
      ],
      build: [
        '遍历 main_rank_for_shard，',
        '每个待交换分片一条 broadcast，',
        'src = 加载它的那个 rank。',
        '本地是 CPU 张量时不能 async_op',
        '要等拷回 CPU 才继续。',
      ],
      seq: [
        `${broadcastMsgs} 条 broadcast，报文大小逐条不同：`,
        exchanged.slice(0, 3).map((t) => `${t.shard.bytes}B←r${t.rank}`).join('  '),
        exchanged.slice(3).map((t) => `${t.shard.bytes}B←r${t.rank}`).join('  '),
        '',
        '每条都精确等于该分片的大小 ——',
        '没有为了对齐而补的空位。',
      ],
      waste: ['0 条空载。', '', '代价换到了报文条数上：', `${broadcastMsgs} 次 collective 启动，`, '每次都要全组同步一次。'],
      cost: [
        `上线 ${sum(exchanged.map((t) => t.shard.bytes))} B`,
        `分 ${broadcastMsgs} 条报文`,
        '',
        'docstring 自陈：',
        '"A reasonable tradeoff in terms',
        ' of performance and simplicity."',
      ],
      cls: 'acc1',
    },
    {
      name: '② gather_rounds',
      pick: [
        'exchange_algo="gather_rounds"',
        '',
        '先按 dtype 分组，组内把',
        'shards_by_rank 转置成轮 ——',
        '每轮一次 all_gather。',
      ],
      build: [
        'zip_longest(*shards_by_rank,',
        '  fillvalue=None)',
        '',
        '轮数 = 单个 rank 最多加载的',
        `分片数，本例 ${roundCount} 轮。`,
      ],
      seq: [
        `${roundCount} 次 all_gather（本例只有一种 dtype）：`,
        ...roundText,
        '',
        '每轮各 rank 各出一份，同时到齐。',
      ],
      waste: [
        `${roundSlots} 个槽位里 ${emptySlots} 个是空张量`,
        `（torch.empty(0)），空载率 ${((emptySlots / roundSlots) * 100).toFixed(0)}%。`,
        '',
        '分配按字节数摊平，不按分片**条数**',
        '摊平 —— 空载正是这个错配的产物。',
      ],
      cost: [
        `${roundCount} 次 collective 启动`,
        '（少于 ①）',
        '',
        'docstring 自陈这条短板：',
        '"might result in a lot of',
        ' almost empty all_gathers"',
      ],
      cls: 'acc2',
    },
    {
      name: '③ gather_object',
      pick: [
        'exchange_algo="gather_object"',
        '',
        '一次 all_gather_object 把',
        '整个 loaded_tensors 字典发出去，',
        '再 reduce 合并成一份。',
      ],
      build: [
        '张量随 Python 对象一起 pickle，',
        '走 host 内存，不是设备侧',
        'collective。',
        '',
        '合并后校验条数：对不上即报',
        '"Duplicate shard ids"。',
      ],
      seq: [
        '1 次 all_gather_object。',
        `载荷 = 全部 ${sum(exchanged.map((t) => t.shard.bytes))} B 加 pickle 开销。`,
        '',
        '报文条数最少，但每个字节都要',
        '经过一次序列化与一次反序列化。',
      ],
      waste: ['无空载槽位，', '浪费换到了 CPU 与 host 内存上。'],
      cost: [
        'docstring 明说用途：',
        '"can be used for debugging',
        ' purposes do to its simplistic',
        ' implementation. Shouldn\'t be',
        ' used if performance is',
        ' important."',
      ],
      cls: 'ghost',
    },
  ];

  lanes.forEach((lane, idx) => {
    const y = 192 + idx * (boxH + 24);
    p.push(text(28, y + 14, lane.name, 'pt'));
    const boxes = [lane.pick, lane.build, lane.seq, lane.waste, lane.cost];
    boxes.forEach((lines, col) => {
      const cls = col === 4 ? lane.cls : col === 0 ? 'ghost' : 'neutral';
      p.push(
        infoBox(colX[col], y + 22, colW[col], boxH, headings[col], lines, cls, `fig4/${idx}/${col}`),
      );
    });
  });

  p.push(
    text(
      28,
      H - 42,
      `三条路的结果完全相同（同一份 all_loaded_tensors），差别只在报文条数、空载与走不走 host。选型判据是组大小与分片条数分布，不是正确性。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      `交换之后 fill_in_deferred_sharded_tensors 才把张量填回 state dict；缺任何一个 shard_id 都直接抛 "Missing shards after fully parallel loading"。`,
      'cap',
    ),
  );
  return seal(p, W, H, 'fig4');
}

// ============================== 图 5 ==============================

// 异步存档的完成阶梯：每一级由哪个符号证明、训练进程在哪几级上真的被挡住。
// 语义来自 strategies/async_utils.py、strategies/filesystem_async.py、
// strategies/state_dict_saver.py、serialization.py 与 megatron/training/checkpointing.py。
const LADDER = Object.freeze([
  Object.freeze({
    stage: 'created',
    proof: 'TorchDistSaveShardedStrategy.async_save 返回 AsyncRequest',
    who: '训练进程',
    blocks: false,
    note: '此时一个字节都没动：async_fn / preload_fn / finalize_fns 只是被打包。',
  }),
  Object.freeze({
    stage: 'staged（D2H 完成）',
    proof: 'preload_fn = FileSystemWriterAsync.preload_tensors',
    who: '两种 caller 都挡住训练',
    blocks: true,
    note: 'Temporal：在训练进程内联跑完再 torch.cuda.synchronize()；Persistent：交给常驻 worker，但训练进程仍 preload_q.join() 等它。',
  }),
  Object.freeze({
    stage: 'submitted',
    proof: 'fork 出进程 / 把请求放进常驻 worker 的队列',
    who: '训练进程',
    blocks: false,
    note: 'Temporal 每次存档 fork 一个新进程；Persistent 复用一个 daemon worker。',
  }),
  Object.freeze({
    stage: 'written（本 rank 的字节落盘）',
    proof: 'write_preloaded_data_multithread',
    who: 'worker 内的多个线程',
    blocks: false,
    note: '是线程不是子进程 —— docstring 明说要能安全跑在 daemon 进程里；最后一个 bucket 在调用线程上跑。',
  }),
  Object.freeze({
    stage: 'finalized（全局元数据成文）',
    proof: 'save_state_dict_async_finalize + metadata_finalize_fn',
    who: '训练进程（下次迭代循环头）',
    blocks: false,
    note: '一次 gather + 一次 broadcast + rank 0 写 metadata.json + barrier。轮询本身还要一次 all-reduce。',
  }),
  Object.freeze({
    stage: 'visible（可被续训选中）',
    proof: 'iter_finalize_fn 写 tracker 文件',
    who: 'rank 0',
    blocks: false,
    note: '在这一步之前，checkpoint 目录已存在但 latest_checkpointed_iteration.txt 还没指向它。',
  }),
]);

function renderAsyncLadderFigure() {
  const W = 1240;
  const H = 620;
  const p = header(
    W,
    '图 5　异步存档的完成阶梯：训练进程到底在哪一级被挡住',
    '"异步"省掉的是写盘那一段，不是 D2H。六级完成语义各由一个符号证明，只有第二级真的阻塞训练进程。',
  );

  const rowH = 62;
  LADDER.forEach((step, i) => {
    const y = 84 + i * (rowH + 6);
    p.push(rect(28, y, 292, rowH, step.blocks ? 'acc2' : 'acc1'));
    p.push(text(40, y + 22, `${i + 1}. ${step.stage}`, 'rank'));
    p.push(text(40, y + 42, step.blocks ? '训练进程在此等待' : '训练进程不等待', step.blocks ? 'costtx' : 'dim'));

    p.push(rect(336, y, 386, rowH, 'neutral'));
    p.push(text(348, y + 22, '由谁证明', 'sm'));
    p.push(text(348, y + 42, guard(step.proof, 10.5, 362, `fig5/${i}/proof`), 'tx'));

    p.push(rect(738, y, 178, rowH, 'ghost'));
    p.push(text(750, y + 22, '执行者', 'sm'));
    p.push(text(750, y + 42, guard(step.who, 10.5, 154, `fig5/${i}/who`), 'sm'));

    if (i < LADDER.length - 1) p.push(arrow(174, y + rowH + 1, 174, y + rowH + 5, 'aux'));
  });

  p.push(
    infoBox(
      932,
      84,
      280,
      196,
      '两种 caller 的差别不在"挡不挡"',
      [
        'Temporal：每次存档 fork 一个进程，',
        'D2H 在训练进程内联；轮询用一次',
        '单整数 all-reduce（注释自陈"与',
        'barrier 同等开销"）。',
        '',
        'Persistent：一个常驻 daemon worker，',
        'D2H 移到 worker 里做 —— 但训练进程',
        '仍要 join 等它完成。省掉的是 fork，',
        '不是那次同步。',
      ],
      'neutral',
      'fig5/callers',
    ),
  );
  p.push(
    infoBox(
      932,
      292,
      280,
      196,
      '同步存档走同一条路',
      [
        'TorchDistSaveShardedStrategy.save 就是',
        'async_save(async_strategy="mcore") 后',
        '立刻 execute_sync()：替换 preload 结果、',
        '跑 async_fn、再 torch.distributed.barrier()',
        '（注释："This utility implements a sync',
        ' cp save. Hence the barrier."）。',
        '',
        '所以同步与异步的差别只是 finalize 何时',
        '发生，不是两套写盘实现。',
      ],
      'ghost',
      'fig5/sync',
    ),
  );

  p.push(
    text(
      28,
      H - 62,
      '触发第 5 级的只有三个调用点：训练迭代循环头的 maybe_finalize_async_save(blocking=False)，以及 train() / pretrain() 结尾各一次 blocking=True。',
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 42,
      '推论（本页归纳）：如果一个作业在两次存档之间迭代很少，第 5 级可能被下一次存档追上；源码用 assert 挡住这种错配 —— 全体 rank 的 call_idx 必须一致。',
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      '第 2 级是 async_save 唯一没能省掉的同步点，也是 use_persistent_ckpt_worker 与 async_ckpt_use_cpu_shm 这两个旋钮真正作用的地方。',
      'cap',
    ),
  );
  return seal(p, W, H, 'fig5');
}

// ============================================================================

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(
  here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets',
);
const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
mkdirSync(outputDir, { recursive: true });

const outputs = new Map([
  ['megatron_ckpt_reshard.svg', renderReshardFigure()],
  ['megatron_ckpt_access_grid.svg', renderAccessFigure()],
  ['megatron_ckpt_greedy_save.svg', renderGreedySaveFigure()],
  ['megatron_ckpt_exchange_algos.svg', renderExchangeFigure()],
  ['megatron_ckpt_async_ladder.svg', renderAsyncLadderFigure()],
]);

for (const [name, svg] of outputs) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
console.log([...outputs.keys()].join('\n'));

export { CFG, LADDER, distributeShardsToRanks, distribution, readItems, rounds, shardAccessCount };
