// 36_megatron_fsdp_analysis.md 的三张图。
//
// 图 1：**布局图** —— 一个 FSDP unit 的五个参数怎样被装进一条按 DP-LCM 网格切分的扁平桶。
//       同时复刻冻结基线里两个活的布局算法：v1
//       param_and_grad_buffer.py::build_data_parallel_buffer_index（regular / conjugate /
//       fragment 三段逻辑逐句照抄，含「sorted() 返回值没被接住」这一源码事实）与 v2
//       experimental/layout.py::GlobalLayout.build（fragment 按大小排序、碎片对齐到自身行宽），
//       在 GlobalLayout.build docstring 与 mfsdp_v2/test_dbuffer.py::
//       test_compute_layout_fills_lcm_padding_gaps 锁定的算例上跑，再跑一个能显示两者差异的小例
//       （test_dbuffer_layout_aligns_fragment_offsets_to_rows）。
// 图 2：**时序图** —— hook 状态机 + AG / RS 两条流。一个最小离散事件仿真（L 个 unit，前向 c、
//       反向 2c、AG a、RS r、v2 每 unit 一次 main→model 权重同步 s）按冻结源码的发起点与等待点
//       排任务：v1 megatron_fsdp.py::MegatronFSDP._register_fsdp_hooks 的四类 hook 与
//       AllGatherPipeline / GradReducePipeline 的预取、槽位与等待；v2 experimental/module.py::
//       FsdpModule.pre_forward / post_forward / pre_backward / post_backward 的
//       allgather_stream 与 delayed release；v1 在 MXFP8 全量重计算下 prefetch_recompute_forward_weights
//       的三连发。输出各 lane 的 makespan 与峰值在途 unit 数。
// 图 3：**代价图** —— 用 examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh 的形状算一个
//       TransformerLayer unit 的参数量，再按 param_and_grad_buffer.py::
//       ParamAndGradBuffer._init_each_parameter_group_buffers 的三个布尔量、_resolve_group_grad_dtype、
//       MixedPrecisionPolicy 默认值与 Adam 两个 fp32 状态（DistributedOptimizer 侧，已知假设）算四档 +
//       HSDP + HFSDP 的常驻字节 / 参数 / rank、双缓冲与三缓冲的瞬态字节，以及每 step 每 rank 的通信量。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「分片切在 unit 的扁平桶上时，一个参数在各 rank 上到底长什么样，以及 pad / 配对 /
//   碎片各落在哪」。算例：DP=5，P0=(2,6)、P1=(4,4)、P2=(4,4)、P3=(1,2)、P4=(1,6)；
//   chunk = LCM(6,4,4,2,6) = 12，size = 60，offsets = (0,12,32,28,48)。
//   面板 A：五个参数条（每条按行分格）+ 一把 LCM=12 的尺规；标出 regular / fragment 分类与
//     P1·P2 的 conjugate 配对（余数 4+4 ≤ 12）。
//   面板 B：60 格全局缓冲，按 5 个 rank 着色（每 rank 12 格），每格写参数名与行号；pad 用 ghost，
//     conjugate 配对的两段用 acc1 描边，填进 gap 的碎片用 acc2 描边。
//   面板 C：每 rank 本地形状表（由 DBuffer.get_local_tensor 的规则算出：owned range 必须整行）。
//   面板 D：对照 lane —— FSDP2 式逐参数 Shard(0)：每个参数各自按 dim-0 均分到 5 个 rank；
//     P3=(1,2) 只有 1 行，4 个 rank 为空；每个参数一次 collective。此 lane 按 DTensor 公开契约推演，
//     不是读过 FSDP2 源码。
//   面板 E：v1 vs v2 小例 (4,4)+(1,6)，DP=2：v2 把碎片对齐到 row=6 得 offset 18，v1 直接放在
//     gap_offset 16；两者 size 都是 24。
//   底部盒子：规则（LCM、pad、conjugate、fragment）、两条算法的差异清单、本图复刻了什么 / 简化了什么。
//
// 图 2 要回答「参数什么时候被 gather、什么时候被放掉、梯度什么时候被 reduce，以及两代数据面在
//   同一算例上各自把通信藏在哪」。参数：L=4，c=4，a=2，r=2，s=1（示意单位，不是测量值）。
//   每个 lane 三条泳道：compute / AG 流 / RS 流；块按仿真时间定位；橙块 = 暴露的等待或同步集合通信；
//   蓝块 = 被计算掩盖的通信。
//   lane ①  v1 前向 + 反向（持久池 fsdp_buffer_count=2，独立 AG 进程组）：_pre_forward_param_unshard
//     发 AG(i) 并预取 AG(i+1)，_post_forward 释放；反向 _pre_backward_param_unshard 发 AG(i) 并预取
//     AG(i−1)，RegisterFSDPBackwardFunction 触发 _post_backward_release_module，post-accumulate-grad
//     hook → _process_post_backward_gradients → GradReducePipeline.reduce_gradients；finish_grad_sync
//     等最后一个 RS。
//   lane ②  v2 前向 + 反向：FsdpModule.pre_forward 在 allgather_stream 上先 sync_model_weight_from_main_weight
//     再 allgather，current_stream.wait_stream(allgather_stream)；重叠来自 CPU run-ahead；post_forward
//     enqueue_release + drain_delayed_releases(target_length=1)，root 处 drain 到 0；反向无预取，
//     post_backward 按参数计数触发 reduce_gradients（同步 reduce_scatter_tensor，落在 compute 流）；
//     同一 communicator 上的集合通信按发起顺序串行，这是 test_overlaps_all_gather_and_compute docstring
//     所述「反向不重叠」的仿真化。
//   lane ③  v1 MXFP8 全量重计算：_root_pre_backward 把全部模块置 PRE_BACKWARD；开
//     prefetch_recompute_forward_weights 时 pre-backward 三连发（bwd 桶不预取、fwd 桶预取 i−1、
//     bwd 桶预取 i−1），重算前向的 _pre_forward_param_unshard 取消预取、_post_forward lazy release。
//   右下三个盒子：四态 TrainingState、四态 BucketStatus 各自在图上的位置；仿真结论
//     （makespan、峰值在途 unit 数、同一 communicator 时的 v1 makespan）。
//
// 图 3 要回答「四档 + HSDP / HFSDP 各自让每个参数在每个 rank 上常驻多少字节、瞬态又要多少、
//   每 step 搬多少字节」。形状：hidden 4096、ffn 14336、32 层、GQA 8 组、kv-channels 128；
//   一个 TransformerLayer unit = q/k/v/o + 3×MLP + 2 norm 的参数量（vocab 不进本账）。
//   左：六档常驻字节 / 参数 / rank 的堆叠柱（W 模型权重、G 主梯度、M 主权重、O 优化器状态），
//     bf16 参数一组、fp8 参数一组。
//   中：瞬态字节（MiB）：unit 的 bf16 / fp8 未分片桶 × 双缓冲 / 三缓冲，MXFP8 再加一份转置桶。
//   右：每 step 每 rank 的通信量（B / 参数）：每 microbatch 的 AG 前向 + AG 反向 + RS，
//     加上每优化周期一次的外层 AR（HSDP，仅最后一个 microbatch）或外层 RS + AG（HFSDP）。
//
// 用法：node tools/figs/svg/megatron_fsdp_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例
// ============================================================================

// 图 1：GlobalLayout.build docstring 与 test_compute_layout_fills_lcm_padding_gaps 锁定的算例
const LAYOUT_CASE = Object.freeze({
  dp: 5,
  shapes: Object.freeze([
    Object.freeze([2, 6]), // P0
    Object.freeze([4, 4]), // P1
    Object.freeze([4, 4]), // P2
    Object.freeze([1, 2]), // P3
    Object.freeze([1, 6]), // P4
  ]),
});

// 图 1 面板 E：test_dbuffer_layout_aligns_fragment_offsets_to_rows 的小例
const DIFF_CASE = Object.freeze({
  dp: 2,
  shapes: Object.freeze([Object.freeze([4, 4]), Object.freeze([1, 6])]),
});

// 正文引用的第三个小例：碎片装填顺序不同时两种算法给出不同的桶大小
const ORDER_CASE = Object.freeze({
  dp: 2,
  shapes: Object.freeze([Object.freeze([4, 4]), Object.freeze([1, 2]), Object.freeze([1, 6])]),
});

// 图 2：仿真参数（示意单位）
const SIM = Object.freeze({ L: 4, c: 4, a: 2, r: 2, s: 1, slots: 2 });

// 图 3：examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh 的形状
const RECIPE = Object.freeze({
  hidden: 4096,
  ffn: 14336,
  layers: 32,
  heads: 32,
  queryGroups: 8,
  kvChannels: 128,
  vocab: 128256,
  dp: 8, // 1 节点 8 GPU，TP=CP=PP=1
  microbatches: 16, // GBS 128 / MBS 1 / DP 8
  hsdpOuter: 4, // HSDP / HFSDP 算例延伸：D=8 内层、O=4 外层
});

// ============================================================================
// 复刻 v1 param_and_grad_buffer.py::build_data_parallel_buffer_index（只保留偏移与 pad）
// ============================================================================

const numel = (shape) => shape.reduce((acc, dim) => acc * dim, 1);
const rowSize = (shape) => numel(shape.slice(1));
const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
const lcm = (a, b) => (a / gcd(a, b)) * b;
const padTo = (value, multiple) => Math.ceil(value / multiple) * multiple;

function lcmChunkSizeFactor(shapes) {
  // _get_parameter_groups Step 3：不同 chunk_size_factor 取 math.lcm
  let chunk = 1;
  for (const shape of shapes) chunk = lcm(chunk, rowSize(shape));
  return chunk;
}

function v1BuildIndex(shapes, dpSize, chunkSizeFactor, noShard = false) {
  const chunk = chunkSizeFactor;
  const fragmentItems = [];
  const regularItems = [];
  shapes.forEach((item, id) => {
    if (numel(item) < chunk) fragmentItems.push([id, item]);
    else regularItems.push([id, item]);
  });
  // 源码：`sorted(fragment_items, key=...)` 的返回值没有被接住 —— 碎片保持注册顺序。
  const offsets = new Array(shapes.length).fill(-1);
  let dataIndex = 0;
  while (regularItems.length > 0) {
    const [itemId, item] = regularItems.shift();
    offsets[itemId] = dataIndex;
    const n = numel(item);
    if (n % chunk === 0) {
      dataIndex += n;
      continue;
    }
    let gapOffset = dataIndex + n;
    dataIndex += (Math.floor(n / chunk) + 1) * chunk;
    const remain = n % chunk;
    let space = chunk - remain;
    let found = null;
    for (const candidate of regularItems.slice()) {
      const rhsNumel = numel(candidate[1]);
      if (rhsNumel % chunk === 0) continue;
      const rhsRemain = rhsNumel % chunk;
      if (remain + rhsRemain <= chunk) {
        found = candidate;
        regularItems.splice(regularItems.indexOf(candidate), 1);
        break;
      }
    }
    if (found) {
      const [rhsId, rhs] = found;
      const rhsNumel = numel(rhs);
      const rhsRemain = rhsNumel % chunk;
      offsets[rhsId] = dataIndex - rhsRemain;
      space -= rhsRemain;
      dataIndex += Math.floor(rhsNumel / chunk) * chunk;
    }
    for (const fragment of fragmentItems.slice()) {
      const [fragId, frag] = fragment;
      const fragNumel = numel(frag);
      if (fragNumel > space) continue;
      offsets[fragId] = gapOffset;
      space -= fragNumel;
      gapOffset += fragNumel;
      fragmentItems.splice(fragmentItems.indexOf(fragment), 1);
    }
  }
  for (const [fragId, frag] of fragmentItems) {
    offsets[fragId] = dataIndex;
    dataIndex += numel(frag);
  }
  // _pad_if_needed：no_shard 不 pad，其余 pad 到 dp_world_size × chunk_size_factor
  const size = noShard ? dataIndex : padTo(dataIndex, dpSize * chunk);
  return { offsets, size, chunk };
}

// ============================================================================
// 复刻 v2 experimental/layout.py::GlobalLayout.build
// ============================================================================

function v2BuildLayout(shapes, dpSize) {
  if (dpSize <= 0) throw new Error('DP size must be positive');
  let chunk = 1;
  for (const shape of shapes) {
    const rs = rowSize(shape);
    if (rs <= 0) throw new Error('zero-sized non-leading dims');
    chunk = lcm(chunk, rs);
  }
  const offsets = new Array(shapes.length).fill(-1);
  const fragmentItems = [];
  const regularItems = [];
  shapes.forEach((shape, id) => {
    if (numel(shape) < chunk) fragmentItems.push([id, shape]);
    else regularItems.push([id, shape]);
  });
  // v2 真的排序：大碎片优先
  fragmentItems.sort((x, y) => numel(y[1]) - numel(x[1]));
  let next = 0;
  while (regularItems.length > 0) {
    const [tensorId, shape] = regularItems.shift();
    const n = numel(shape);
    offsets[tensorId] = next;
    if (n % chunk === 0) {
      next += n;
      continue;
    }
    let gapOffset = next + n;
    next += padTo(n, chunk);
    let fragmentGapEnd = next;
    const remainder = n % chunk;
    let conjugate = null;
    for (const candidate of regularItems.slice()) {
      const candNumel = numel(candidate[1]);
      const candRemainder = candNumel % chunk;
      if (candRemainder === 0) continue;
      if (remainder + candRemainder <= chunk) {
        conjugate = candidate;
        regularItems.splice(regularItems.indexOf(candidate), 1);
        break;
      }
    }
    if (conjugate) {
      const [conjId, conjShape] = conjugate;
      const conjNumel = numel(conjShape);
      const conjRemainder = conjNumel % chunk;
      const conjOffset = next - conjRemainder;
      offsets[conjId] = conjOffset;
      fragmentGapEnd = conjOffset;
      next += Math.floor(conjNumel / chunk) * chunk;
    }
    for (const fragment of fragmentItems.slice()) {
      const [fragId, fragShape] = fragment;
      const fragNumel = numel(fragShape);
      const aligned = padTo(gapOffset, rowSize(fragShape));
      if (aligned + fragNumel > fragmentGapEnd) continue;
      offsets[fragId] = aligned;
      gapOffset = aligned + fragNumel;
      fragmentItems.splice(fragmentItems.indexOf(fragment), 1);
    }
  }
  for (const [fragId, fragShape] of fragmentItems) {
    next = padTo(next, rowSize(fragShape));
    offsets[fragId] = next;
    next += numel(fragShape);
  }
  // __post_init__：每个偏移必须是自身行宽的倍数，且互不重叠
  offsets.forEach((start, id) => {
    if (start % rowSize(shapes[id]) !== 0) throw new Error(`tensor ${id} offset ${start} not row-aligned`);
  });
  return { offsets, size: padTo(next, chunk * dpSize), chunk };
}

// v1 与 v2 共用：每个 rank 的本地形状（DBuffer.get_local_tensor：owned range 必须整行）
function localShapes(shapes, offsets, size, dpSize) {
  const shard = size / dpSize;
  const ranks = [];
  for (let r = 0; r < dpSize; r += 1) {
    const bufStart = r * shard;
    const bufEnd = bufStart + shard;
    const local = shapes.map((shape, id) => {
      const start = offsets[id];
      const end = start + numel(shape);
      const overlapStart = Math.max(start, bufStart);
      const overlapEnd = Math.min(end, bufEnd);
      const rs = rowSize(shape);
      if (overlapStart >= overlapEnd) return [0, ...shape.slice(1)];
      const relOffset = overlapStart - start;
      const n = overlapEnd - overlapStart;
      if (relOffset % rs !== 0 || n % rs !== 0) throw new Error(`tensor ${id} split mid-row on rank ${r}`);
      return [n / rs, ...shape.slice(1)];
    });
    ranks.push({ rank: r, range: [bufStart, bufEnd], local });
  }
  return ranks;
}

// 每个元素属于哪个参数、第几行；-1 为 pad / gap
function cellMap(shapes, offsets, size) {
  const cells = new Array(size).fill(null).map(() => ({ param: -1, row: -1 }));
  shapes.forEach((shape, id) => {
    const rs = rowSize(shape);
    for (let k = 0; k < numel(shape); k += 1) {
      cells[offsets[id] + k] = { param: id, row: Math.floor(k / rs) };
    }
  });
  return cells;
}

// FSDP2 式逐参数 Shard(0)：按 DTensor 公开契约推演（dim-0 按 rank 数切成尽量均匀的块，
// 前面的 rank 先拿），不是读过 FSDP2 源码
function perParameterShard0(shapes, dpSize) {
  return shapes.map((shape) => {
    const rows = shape[0];
    const base = Math.floor(rows / dpSize);
    const extra = rows % dpSize;
    const perRank = [];
    for (let r = 0; r < dpSize; r += 1) perRank.push(base + (r < extra ? 1 : 0));
    return { rows, perRank, emptyRanks: perRank.filter((n) => n === 0).length };
  });
}

function runLayout(caseSpec) {
  const chunk = lcmChunkSizeFactor(caseSpec.shapes);
  const v1 = v1BuildIndex(caseSpec.shapes, caseSpec.dp, chunk);
  const v2 = v2BuildLayout(caseSpec.shapes, caseSpec.dp);
  const classes = caseSpec.shapes.map((shape) => (numel(shape) < chunk ? 'fragment' : 'regular'));
  const v2Local = localShapes(caseSpec.shapes, v2.offsets, v2.size, caseSpec.dp);
  return {
    dp: caseSpec.dp,
    shapes: caseSpec.shapes,
    chunk,
    v1,
    v2,
    classes,
    cells: cellMap(caseSpec.shapes, v2.offsets, v2.size),
    ranks: v2Local,
    shard: v2.size / caseSpec.dp,
    totalNumel: caseSpec.shapes.reduce((acc, shape) => acc + numel(shape), 0),
    padCells: v2.size - caseSpec.shapes.reduce((acc, shape) => acc + numel(shape), 0),
    fsdp2: perParameterShard0(caseSpec.shapes, caseSpec.dp),
  };
}

const LAYOUT = runLayout(LAYOUT_CASE);
const DIFF = runLayout(DIFF_CASE);
const ORDER = runLayout(ORDER_CASE);

// conjugate 配对：v2 里 P1 的余数 4 与 P2 的余数 4 装进同一个 chunk
function conjugatePairs(layout) {
  const { shapes, chunk, v2 } = layout;
  const pairs = [];
  shapes.forEach((shape, id) => {
    const n = numel(shape);
    if (n < chunk || n % chunk === 0) return;
    shapes.forEach((other, otherId) => {
      if (otherId <= id) return;
      const m = numel(other);
      if (m < chunk || m % chunk === 0) return;
      // P_other 的起点落在 P_id 的 pad 区间末尾
      const idEnd = v2.offsets[id] + n;
      const idPadEnd = padTo(idEnd, chunk);
      if (v2.offsets[otherId] + (m % chunk) === idPadEnd) pairs.push([id, otherId]);
    });
  });
  return pairs;
}

const CONJUGATES = conjugatePairs(LAYOUT);

// ============================================================================
// 图 2：最小离散事件仿真
//   任务按 CPU 发起顺序列出；start = max(所在流的上一个任务结束, 所在 communicator 的上一个
//   任务结束, 显式依赖结束)。流 = compute / ag / rs；communicator = main / ag（独立 AG 组时）。
// ============================================================================

function schedule(tasks) {
  const laneFree = {};
  const commFree = {};
  const end = {};
  const out = [];
  for (const t of tasks) {
    let start = t.earliest ?? 0;
    start = Math.max(start, laneFree[t.lane] ?? 0);
    if (t.comm) start = Math.max(start, commFree[t.comm] ?? 0);
    for (const dep of t.deps ?? []) {
      if (!(dep in end)) throw new Error(`task ${t.id} depends on unknown task ${dep}`);
      start = Math.max(start, end[dep]);
    }
    const finish = start + t.dur;
    laneFree[t.lane] = finish;
    if (t.comm) commFree[t.comm] = finish;
    end[t.id] = finish;
    out.push({ ...t, start, end: finish });
  }
  return out;
}

// 峰值在途：给每个 unit 一段「存储被占用」的时间区间，数最大重叠
function peakLive(intervals) {
  const events = [];
  for (const [s, e] of intervals) {
    events.push([s, 1]);
    events.push([e, -1]);
  }
  events.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  let live = 0;
  let peak = 0;
  for (const [, d] of events) {
    live += d;
    peak = Math.max(peak, live);
  }
  return peak;
}

// lane ①：v1，持久池 slots 个槽（预取深度 slots−1），独立 AG 进程组可选
function simulateV1({ L, c, a, r, slots, separateAgComm }) {
  const agComm = separateAgComm ? 'ag' : 'main';
  const tasks = [];
  const issued = new Set();
  const lastCompute = () => {
    for (let i = tasks.length - 1; i >= 0; i -= 1) if (tasks[i].lane === 'compute') return [tasks[i].id];
    return [];
  };
  // ---- 前向 ----
  for (let i = 0; i < L; i += 1) {
    // _pre_forward_param_unshard(i)：AG(i) + 预取 i+1..i+slots-1（all_gather_stream.wait_stream(current)）
    const deps = lastCompute();
    for (let k = i; k < Math.min(L, i + slots); k += 1) {
      if (issued.has(`AGf${k}`)) continue;
      issued.add(`AGf${k}`);
      tasks.push({ id: `AGf${k}`, label: `AG${k}`, lane: 'ag', comm: agComm, dur: a, deps, phase: 'fwd', unit: k, kind: 'ag' });
    }
    tasks.push({ id: `C${i}`, label: `F${i}`, lane: 'compute', dur: c, deps: [`AGf${i}`], phase: 'fwd', unit: i, kind: 'compute' });
    // _post_forward(i)：release_module_parameters(bwd=False) —— 槽位立即回收（主机侧）
  }
  const fwdEnd = Math.max(...schedule(tasks).map((t) => t.end));
  // ---- 反向 ----（_root_pre_backward 把所有模块置 PRE_BACKWARD；bf16 无转置桶，bwd 桶 = fwd 桶键）
  for (let i = L - 1; i >= 0; i -= 1) {
    const deps = lastCompute();
    for (let k = i; k > Math.max(-1, i - slots); k -= 1) {
      if (issued.has(`AGb${k}`)) continue;
      issued.add(`AGb${k}`);
      tasks.push({ id: `AGb${k}`, label: `AG${k}`, lane: 'ag', comm: agComm, dur: a, deps, phase: 'bwd', unit: k, kind: 'ag' });
    }
    // 梯度桶槽位：unit i 的 main_grad 桶复用 unit i+slots 的槽，get_main_grad 里先 _enforce_double_buffer_limit
    const gradSlotDeps = i + slots < L ? [`RS${i + slots}`] : [];
    tasks.push({ id: `B${i}`, label: `B${i}`, lane: 'compute', dur: 2 * c, deps: [`AGb${i}`, ...gradSlotDeps], phase: 'bwd', unit: i, kind: 'compute' });
    // RegisterFSDPBackwardFunction → _post_backward_release_module(i)；post-accumulate-grad hook →
    // _process_post_backward_gradients → reduce_gradients：rs_stream.wait_stream(current)
    tasks.push({ id: `RS${i}`, label: `RS${i}`, lane: 'rs', comm: 'main', dur: r, deps: [`B${i}`], phase: 'bwd', unit: i, kind: 'rs' });
  }
  const sched = schedule(tasks);
  const byId = Object.fromEntries(sched.map((t) => [t.id, t]));
  // 在途参数桶：从 AG 发起到释放（前向：F(i) 结束；反向：B(i) 结束）
  const paramIntervals = [];
  for (let i = 0; i < L; i += 1) {
    paramIntervals.push([byId[`AGf${i}`].start, byId[`C${i}`].end]);
    paramIntervals.push([byId[`AGb${i}`].start, byId[`B${i}`].end]);
  }
  const gradIntervals = [];
  for (let i = 0; i < L; i += 1) gradIntervals.push([byId[`B${i}`].start, byId[`RS${i}`].end]);
  return {
    name: separateAgComm ? 'v1' : 'v1-same-comm',
    tasks: sched,
    fwdEnd,
    bwdEnd: Math.max(...sched.map((t) => t.end)),
    peakParams: peakLive(paramIntervals),
    peakGrads: peakLive(gradIntervals),
    exposedAg: sched.filter((t) => t.kind === 'ag').filter((t) => {
      const consumer = byId[t.phase === 'fwd' ? `C${t.unit}` : `B${t.unit}`];
      return consumer.start === t.end && t.start === Math.max(...(t.deps.map((d) => byId[d].end).concat([0])));
    }).length,
  };
}

// lane ②：v2，单一 communicator，allgather_stream + delayed release，反向无预取
function simulateV2({ L, c, a, r, s }) {
  const tasks = [];
  // ---- 前向 ----
  for (let i = 0; i < L; i += 1) {
    // pre_forward(i)：drain_delayed_releases(target_length=1) 释放 i−2（allgather_stream.wait_event(F(i−2))）
    const drainDeps = i - 2 >= 0 ? [`C${i - 2}`] : [];
    // sync_model_weight_from_main_weight：每 microbatch 一次 cast + redistribute（TODO 挪到 optimizer post-step）
    tasks.push({ id: `S${i}`, label: 'S', lane: 'ag', dur: s, deps: drainDeps, phase: 'fwd', unit: i, kind: 'sync' });
    tasks.push({ id: `AGf${i}`, label: `AG${i}`, lane: 'ag', comm: 'main', dur: a, deps: [`S${i}`], phase: 'fwd', unit: i, kind: 'ag' });
    // current_stream.wait_stream(allgather_stream)
    tasks.push({ id: `C${i}`, label: `F${i}`, lane: 'compute', dur: c, deps: [`AGf${i}`], phase: 'fwd', unit: i, kind: 'compute' });
    // post_forward(i)：enqueue_release(i)；root 的 post_forward drain 到 0
  }
  const fwdEnd = Math.max(...schedule(tasks).map((t) => t.end));
  // ---- 反向 ----（pre_backward 无预取；reduce_gradients 的 reduce_scatter_tensor 在 compute 流上同步）
  for (let i = L - 1; i >= 0; i -= 1) {
    // root 的 post_forward drain 到 0：allgather_stream.wait_event(最后一个前向计算)；
    // 之后 pre_backward(i) 的 drain 到 1 释放 i+2（wait_event(B(i+2))）
    const drainDeps = i === L - 1 ? [`C${L - 1}`] : i + 2 < L ? [`B${i + 2}`] : [];
    tasks.push({ id: `AGb${i}`, label: `AG${i}`, lane: 'ag', comm: 'main', dur: a, deps: drainDeps, phase: 'bwd', unit: i, kind: 'ag' });
    tasks.push({ id: `B${i}`, label: `B${i}`, lane: 'compute', dur: 2 * c, deps: [`AGb${i}`], phase: 'bwd', unit: i, kind: 'compute' });
    tasks.push({ id: `RS${i}`, label: `RS${i}`, lane: 'compute', comm: 'main', dur: r, deps: [`B${i}`], phase: 'bwd', unit: i, kind: 'rs' });
  }
  const sched = schedule(tasks);
  const byId = Object.fromEntries(sched.map((t) => [t.id, t]));
  // 在途：从 AG 发起到 drain 释放（前向 unit i 在 pre_forward(i+2) 才释放，最后两个在 root post_forward 释放）
  const paramIntervals = [];
  for (let i = 0; i < L; i += 1) {
    const fwdRelease = i + 2 < L ? byId[`AGf${i + 2}`].start : fwdEnd;
    paramIntervals.push([byId[`AGf${i}`].start, Math.max(fwdRelease, byId[`C${i}`].end)]);
    const bwdRelease = i - 2 >= 0 ? byId[`AGb${i - 2}`].start : byId.RS0.end;
    paramIntervals.push([byId[`AGb${i}`].start, Math.max(bwdRelease, byId[`B${i}`].end)]);
  }
  return {
    name: 'v2',
    tasks: sched,
    fwdEnd,
    bwdEnd: Math.max(...sched.map((t) => t.end)),
    peakParams: peakLive(paramIntervals),
    delayedQueueMax: 1, // drain_delayed_releases(target_length=1)
  };
}

// lane ③：v1 MXFP8 全量重计算，fwd（rowwise）与 bwd（columnwise）各一套桶与池
function simulateRecompute({ L, c, a, r, slots, prefetchRecomputeForwardWeights }) {
  const tasks = [];
  const issued = new Set();
  const lastCompute = () => {
    for (let i = tasks.length - 1; i >= 0; i -= 1) if (tasks[i].lane === 'compute') return [tasks[i].id];
    return [];
  };
  const issue = (id, label, extra) => {
    if (issued.has(id)) return;
    issued.add(id);
    tasks.push({ id, label, lane: 'ag', comm: 'ag', dur: a, ...extra });
  };
  // ---- 前向：同 lane ① ----
  for (let i = 0; i < L; i += 1) {
    const deps = lastCompute();
    for (let k = i; k < Math.min(L, i + slots); k += 1) issue(`AGf${k}`, `AGr${k}`, { deps, phase: 'fwd', unit: k, kind: 'ag' });
    tasks.push({ id: `C${i}`, label: `F${i}`, lane: 'compute', dur: c, deps: [`AGf${i}`], phase: 'fwd', unit: i, kind: 'compute' });
  }
  const fwdEnd = Math.max(...schedule(tasks).map((t) => t.end));
  // ---- 反向 + 重算 ----
  for (let i = L - 1; i >= 0; i -= 1) {
    const deps = lastCompute();
    if (prefetchRecomputeForwardWeights) {
      // 三连发：bwd(i) 不预取；fwd(i) + 预取 fwd(i−1)；bwd(i) + 预取 bwd(i−1)
      issue(`AGb${i}`, `AGc${i}`, { deps, phase: 'bwd', unit: i, kind: 'ag' });
      issue(`AGr${i}`, `AGr${i}`, { deps, phase: 'bwd', unit: i, kind: 'ag' });
      if (i - 1 >= 0) issue(`AGr${i - 1}`, `AGr${i - 1}`, { deps, phase: 'bwd', unit: i - 1, kind: 'ag' });
      if (i - 1 >= 0) issue(`AGb${i - 1}`, `AGc${i - 1}`, { deps, phase: 'bwd', unit: i - 1, kind: 'ag' });
    } else {
      issue(`AGb${i}`, `AGc${i}`, { deps, phase: 'bwd', unit: i, kind: 'ag' });
      if (i - 1 >= 0) issue(`AGb${i - 1}`, `AGc${i - 1}`, { deps, phase: 'bwd', unit: i - 1, kind: 'ag' });
      // 重算前向的 _pre_forward_param_unshard：状态是 PRE_BACKWARD → 取消预取，只 gather rowwise(i)
      issue(`AGr${i}`, `AGr${i}`, { deps, phase: 'bwd', unit: i, kind: 'ag' });
    }
    const gradSlotDeps = i + slots < L ? [`RS${i + slots}`] : [];
    tasks.push({ id: `R${i}`, label: `R${i}`, lane: 'compute', dur: c, deps: [`AGr${i}`], phase: 'bwd', unit: i, kind: 'recompute' });
    // _post_forward 在 PRE_BACKWARD 下 lazy_release：fwd 桶等下一次 AG 分配时回收
    tasks.push({ id: `B${i}`, label: `B${i}`, lane: 'compute', dur: 2 * c, deps: [`AGb${i}`, `R${i}`, ...gradSlotDeps], phase: 'bwd', unit: i, kind: 'compute' });
    tasks.push({ id: `RS${i}`, label: `RS${i}`, lane: 'rs', comm: 'main', dur: r, deps: [`B${i}`], phase: 'bwd', unit: i, kind: 'rs' });
  }
  const sched = schedule(tasks);
  const byId = Object.fromEntries(sched.map((t) => [t.id, t]));
  const rowIntervals = [];
  const colIntervals = [];
  for (let i = 0; i < L; i += 1) {
    rowIntervals.push([byId[`AGf${i}`].start, byId[`C${i}`].end]);
    rowIntervals.push([byId[`AGr${i}`].start, byId[`B${i}`].end]); // lazy release：到 post-backward 才真正释放
    colIntervals.push([byId[`AGb${i}`].start, byId[`B${i}`].end]);
  }
  return {
    name: prefetchRecomputeForwardWeights ? 'recompute+prefetch' : 'recompute',
    tasks: sched,
    fwdEnd,
    bwdEnd: Math.max(...sched.map((t) => t.end)),
    peakRowwise: peakLive(rowIntervals),
    peakColwise: peakLive(colIntervals),
  };
}

const LANE_V1 = simulateV1({ ...SIM, separateAgComm: true });
const LANE_V1_SAME_COMM = simulateV1({ ...SIM, separateAgComm: false });
const LANE_V2 = simulateV2(SIM);
const LANE_RECOMPUTE = simulateRecompute({ ...SIM, prefetchRecomputeForwardWeights: true });
const LANE_RECOMPUTE_NO_PREFETCH = simulateRecompute({ ...SIM, prefetchRecomputeForwardWeights: false });

// 独立 AG 进程组的价值：把 AG 时长 a 从 1 扫到 2c+r，找出「共用 communicator 先于独立组开始拉长反向」的阈值
function agCommThreshold(base) {
  let sameCommFirstLag = null;
  let separateFirstLag = null;
  const ideal = (a) => a + base.L * 2 * base.c + base.r; // 只暴露 AG(L−1) 与最后一个 RS
  for (let a = 1; a <= 2 * base.c + base.r; a += 1) {
    const sep = simulateV1({ ...base, a, separateAgComm: true });
    const same = simulateV1({ ...base, a, separateAgComm: false });
    const bwdSep = sep.bwdEnd - sep.fwdEnd;
    const bwdSame = same.bwdEnd - same.fwdEnd;
    if (sameCommFirstLag === null && bwdSame > ideal(a)) sameCommFirstLag = a;
    if (separateFirstLag === null && bwdSep > ideal(a)) separateFirstLag = a;
  }
  return { sameCommFirstLag, separateFirstLag, expectedSame: 2 * base.c - base.r + 1, expectedSeparate: 2 * base.c + 1 };
}

const AG_COMM = agCommThreshold(SIM);
if (AG_COMM.sameCommFirstLag !== AG_COMM.expectedSame || AG_COMM.separateFirstLag !== AG_COMM.expectedSeparate) {
  throw new Error(`独立 AG 组阈值与解析式不符：${JSON.stringify(AG_COMM)}`);
}

// 立论前提：v1 前向的 AG 除了第一个都被掩盖；v2 前向同样；v2 反向串行
{
  const v1 = LANE_V1;
  if (v1.fwdEnd !== SIM.a + SIM.L * SIM.c) throw new Error('v1 前向 makespan 与 a + L·c 不符');
  const v2ById = Object.fromEntries(LANE_V2.tasks.map((t) => [t.id, t]));
  for (let i = SIM.L - 1; i > 0; i -= 1) {
    // B(i) → RS(i) → AG(i−1) → B(i−1) 首尾相接：同一 communicator 串行 + 数据依赖
    if (v2ById[`RS${i}`].start !== v2ById[`B${i}`].end) throw new Error('v2 RS 应紧跟 B');
    if (v2ById[`AGb${i - 1}`].start !== v2ById[`RS${i}`].end) throw new Error('v2 反向 AG 应排在 RS 之后');
    if (v2ById[`B${i - 1}`].start !== v2ById[`AGb${i - 1}`].end) throw new Error('v2 反向 B 应紧跟 AG');
  }
  if (LANE_RECOMPUTE.bwdEnd >= LANE_RECOMPUTE_NO_PREFETCH.bwdEnd) throw new Error('重算预取应缩短 makespan');
}

// ============================================================================
// 图 3：字节账
// ============================================================================

function unitParams(rec) {
  const q = rec.hidden * rec.hidden; // heads × kv_channels = 32 × 128 = 4096
  const kv = 2 * rec.hidden * rec.queryGroups * rec.kvChannels;
  const o = rec.hidden * rec.hidden;
  const fc1 = 2 * rec.ffn * rec.hidden; // swiglu：gate + up
  const fc2 = rec.hidden * rec.ffn;
  const norms = 2 * rec.hidden; // 两个 RMSNorm，只有 weight
  return { q, kv, o, fc1, fc2, norms, total: q + kv + o + fc1 + fc2 + norms };
}

const UNIT = unitParams(RECIPE);
const MODEL_PARAMS = UNIT.total * RECIPE.layers + 2 * RECIPE.vocab * RECIPE.hidden; // untied embedding + output
const MiB = 1024 * 1024;
const fmtMiB = (bytes) => (bytes / MiB).toFixed(1);
const fmtGiB = (bytes) => (bytes / (1024 * MiB)).toFixed(2);

// 每参数字节：W 计算权重、Wt 转置（仅 MXFP8 或 keep_fp8_transpose_cache）、M 主权重 fp32、
// G 主梯度（_resolve_group_grad_dtype：None → 参数 dtype；fp8 → bf16）、O Adam 两个 fp32 状态（假设）
const DTYPES = Object.freeze({
  bf16: Object.freeze({ name: 'bf16', W: 2, Wt: 0, M: 4, G: 2, O: 8 }),
  fp8: Object.freeze({ name: 'fp8', W: 1, Wt: 0, M: 4, G: 2, O: 8 }),
  mxfp8: Object.freeze({ name: 'mxfp8', W: 1, Wt: 1, M: 4, G: 2, O: 8 }),
});

// 常驻字节 / 参数 / rank：_init_each_parameter_group_buffers 的三个布尔量
function residentBytes(dt, { strategy, D, O = 1, outer = 'no_shard' }) {
  const parts = { W: 0, Wt: 0, G: 0, M: 0, O: 0 };
  if (strategy === 'no_shard') {
    parts.W = dt.W; parts.G = dt.G; parts.M = dt.M; parts.O = dt.O;
  } else if (strategy === 'optim') {
    parts.W = dt.W; parts.G = dt.G; parts.M = dt.M / D; parts.O = dt.O / D;
  } else if (strategy === 'optim_grads') {
    parts.W = dt.W; parts.G = dt.G / D; parts.M = dt.M / D; parts.O = dt.O / D;
  } else if (strategy === 'optim_grads_params') {
    if (outer === 'optim') {
      // HFSDP：权重与梯度按内层 D 切（helper buffer），主权重与优化器状态按 D×O 切
      parts.W = dt.W / D; parts.Wt = dt.Wt / D; parts.G = dt.G / D; parts.M = dt.M / (D * O); parts.O = dt.O / (D * O);
    } else {
      // FSDP / HSDP：全部按内层 D 切，外层复制
      parts.W = dt.W / D; parts.Wt = dt.Wt / D; parts.G = dt.G / D; parts.M = dt.M / D; parts.O = dt.O / D;
    }
  } else {
    throw new Error(`unknown strategy ${strategy}`);
  }
  return { ...parts, total: parts.W + parts.Wt + parts.G + parts.M + parts.O };
}

// 每 step 每 rank 通信字节 / 参数（接收量口径）：AG 每次 W(D−1)/D，RS 每次 G(D−1)/D，AR 取 2G(D−1)/D
function commBytes(dt, { strategy, D, O = 1, outer = 'no_shard', microbatches }) {
  const f = (D - 1) / D;
  let perMicrobatch = 0;
  let perCycle = 0;
  if (strategy === 'no_shard') {
    perCycle = 2 * dt.G * f; // 最后一个 microbatch 的 all-reduce
  } else if (strategy === 'optim') {
    perCycle = dt.G * f + dt.W * f; // 延迟 RS + start_param_sync 的 AG
  } else if (strategy === 'optim_grads') {
    perMicrobatch = dt.G * f; // 每次反向 RS
    perCycle = dt.W * f; // 优化周期一次 AG
  } else {
    perMicrobatch = 2 * (dt.W + dt.Wt) * f + dt.G * f; // 前向 AG + 反向 AG + RS
    if (O > 1) {
      const g = (O - 1) / O;
      if (outer === 'optim') {
        perCycle = ((dt.W + dt.Wt) / D) * g + (dt.G / D) * g; // HFSDP：外层 AG + 外层 RS
      } else {
        perCycle = 2 * (dt.G / D) * g; // HSDP：外层 AR，仅最后一个 microbatch
      }
    }
  }
  return { perMicrobatch, perCycle, total: perMicrobatch * microbatches + perCycle };
}

const CONFIGS = Object.freeze([
  { key: 'no_shard', label: 'no_shard', strategy: 'no_shard', D: RECIPE.dp },
  { key: 'optim', label: 'optim', strategy: 'optim', D: RECIPE.dp },
  { key: 'optim_grads', label: 'optim_grads', strategy: 'optim_grads', D: RECIPE.dp },
  { key: 'optim_grads_params', label: 'optim_grads_params', strategy: 'optim_grads_params', D: RECIPE.dp },
  { key: 'hsdp', label: `HSDP D=${RECIPE.dp}·O=${RECIPE.hsdpOuter}`, strategy: 'optim_grads_params', D: RECIPE.dp, O: RECIPE.hsdpOuter, outer: 'no_shard' },
  { key: 'hfsdp', label: `HFSDP D=${RECIPE.dp}·O=${RECIPE.hsdpOuter}`, strategy: 'optim_grads_params', D: RECIPE.dp, O: RECIPE.hsdpOuter, outer: 'optim' },
]);

const COST = Object.freeze({
  unit: UNIT,
  modelParams: MODEL_PARAMS,
  unitBytes: {
    bf16: UNIT.total * DTYPES.bf16.W,
    fp8: UNIT.total * DTYPES.fp8.W,
    mxfp8Transpose: UNIT.total * DTYPES.mxfp8.Wt,
    grad: UNIT.total * DTYPES.bf16.G,
  },
  resident: Object.fromEntries(
    Object.values(DTYPES).map((dt) => [dt.name, CONFIGS.map((cfg) => ({ ...cfg, ...residentBytes(dt, cfg) }))]),
  ),
  comm: Object.fromEntries(
    Object.values(DTYPES).map((dt) => [dt.name, CONFIGS.map((cfg) => ({ ...cfg, ...commBytes(dt, { ...cfg, microbatches: RECIPE.microbatches }) }))]),
  ),
  transient: [2, 3].map((n) => ({
    buffers: n,
    bf16: n * UNIT.total * DTYPES.bf16.W,
    fp8: n * UNIT.total * DTYPES.fp8.W,
    mxfp8: n * UNIT.total * (DTYPES.mxfp8.W + DTYPES.mxfp8.Wt),
    grad: n * UNIT.total * DTYPES.bf16.G,
  })),
});

if (process.env.MFSDP_FIG_DEBUG) {
  console.log('LAYOUT', JSON.stringify({ chunk: LAYOUT.chunk, v1: LAYOUT.v1, v2: LAYOUT.v2, ranks: LAYOUT.ranks.map((r) => r.local), conj: CONJUGATES }));
  console.log('DIFF', JSON.stringify({ v1: DIFF.v1, v2: DIFF.v2 }));
  console.log('ORDER', JSON.stringify({ v1: ORDER.v1, v2: ORDER.v2 }));
  for (const lane of [LANE_V1, LANE_V1_SAME_COMM, LANE_V2, LANE_RECOMPUTE, LANE_RECOMPUTE_NO_PREFETCH]) {
    console.log(lane.name, JSON.stringify({ fwd: lane.fwdEnd, bwd: lane.bwdEnd, peakParams: lane.peakParams, peakGrads: lane.peakGrads, row: lane.peakRowwise, col: lane.peakColwise }));
    console.log('  ', lane.tasks.map((t) => `${t.id}[${t.start},${t.end})`).join(' '));
  }
  console.log('COST', JSON.stringify(COST, null, 1));
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_dsv4_cp_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) =>
  String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

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
  .acc1fill{fill:#2563EB;stroke:#2563EB;stroke-width:1}
  .acc2fill{fill:#C3651F;stroke:#C3651F;stroke-width:1}
  .outline1{fill:none;stroke:#2563EB;stroke-width:1.8}
  .outline2{fill:none;stroke:#C3651F;stroke-width:1.8}
  .r0{fill:#EEF1F5;stroke:#AEB6C2;stroke-width:1}
  .r1{fill:#E3E8EF;stroke:#AEB6C2;stroke-width:1}
  .r2{fill:#D8DEE8;stroke:#AEB6C2;stroke-width:1}
  .r3{fill:#CDD5E1;stroke:#AEB6C2;stroke-width:1}
  .r4{fill:#C2CBDA;stroke:#AEB6C2;stroke-width:1}
  .bar{fill:#F4C9A3;stroke:#C3651F;stroke-width:.8}
  .bar1{fill:#CFE0FA;stroke:#2563EB;stroke-width:.8}
  .bar2{fill:#9CC2F3;stroke:#2563EB;stroke-width:.8}
  .bar3{fill:#5E97E4;stroke:#2563EB;stroke-width:.8}
  .bar4{fill:#E7EAEE;stroke:#AEB6C2;stroke-width:.8}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
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
  .sep{fill:none;stroke:#5B6470;stroke-width:1.6}
`;

const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
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
function header(w, title, subtitle) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} __H__" width="${w}" height="__H__" role="img">`,
    defs,
    `<style>${sharedStyle}</style>`,
    text(28, 32, title, 'ti'),
    text(28, 52, subtitle, 'su'),
  ];
}

const FONT = Object.freeze({
  ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5,
  costtx: 10.5, rank: 11, cap: 11,
});

function assertNoTextOverlap(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  const canvasW = Number(m[1]);
  const canvasH = Number(m[2]);
  const boxes = [];
  for (const hit of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = hit;
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

const shapeStr = (shape) => `(${shape.join(',')})`;

// ============================================================================
// 图 1：布局
// ============================================================================

function renderLayout() {
  const W = 1272;
  const X0 = 28;
  const L = LAYOUT;
  const parts = header(
    W,
    '图 1　Megatron-FSDP 的 DP-LCM 网格：一个 unit 的五个参数怎样装进按 rank 切分的扁平桶',
    `算例：DP=${L.dp}，P0=${shapeStr(L.shapes[0])} P1=${shapeStr(L.shapes[1])} P2=${shapeStr(L.shapes[2])} P3=${shapeStr(L.shapes[3])} P4=${shapeStr(L.shapes[4])}；chunk_size_factor = LCM(行宽) = ${L.chunk}，size = pad(…, DP × LCM) = ${L.v2.size}，每 rank ${L.shard} 个元素；v1 与 v2 在此算例上给出同一组 offsets`,
  );

  // ---- 面板 A：参数条 + LCM 尺规 ----
  const AY = 92;
  parts.push(text(X0, AY - 10, 'A　unit 内的五个参数（每格一个元素，按行分段）与 LCM 尺规', 'pt'));
  const CELL = 14;
  let ax = X0;
  const conjSet = new Set(CONJUGATES.flat());
  L.shapes.forEach((shape, id) => {
    const n = numel(shape);
    const rs = rowSize(shape);
    const cls = L.classes[id] === 'fragment' ? 'acc2' : conjSet.has(id) ? 'acc1' : 'neutral';
    parts.push(rect(ax, AY + 4, n * CELL, 18, cls, 3));
    for (let row = 1; row < shape[0]; row += 1) parts.push(line(ax + row * rs * CELL, AY + 4, ax + row * rs * CELL, AY + 22, 'gl'));
    parts.push(text(ax + (n * CELL) / 2, AY + 36 + (id % 2) * 14, `P${id} ${shapeStr(shape)} = ${n}${n % L.chunk === 0 ? '，整除 LCM' : n > L.chunk ? `，余 ${n % L.chunk}` : '，碎片'}`, L.classes[id] === 'fragment' ? 'costtx' : conjSet.has(id) ? 'dim' : 'sm', 'middle'));
    ax += n * CELL + 40;
  });
  // 尺规
  const RX = ax + 10;
  parts.push(rect(RX, AY + 4, L.chunk * CELL, 18, 'ghost', 3));
  for (let k = 1; k < L.chunk; k += 1) parts.push(line(RX + k * CELL, AY + 4, RX + k * CELL, AY + 22, 'gl'));
  parts.push(text(RX + (L.chunk * CELL) / 2, AY + 36 + (L.shapes.length % 2) * 14, `LCM = ${L.chunk}：DP 分片边界只落在它的倍数上`, 'sm', 'middle'));
  const capA = `蓝框 = conjugate 配对（P${CONJUGATES[0][0]} 余 ${numel(L.shapes[CONJUGATES[0][0]]) % L.chunk} + P${CONJUGATES[0][1]} 余 ${numel(L.shapes[CONJUGATES[0][1]]) % L.chunk} ≤ ${L.chunk}，装进同一个 chunk）；橙框 = 小于 LCM 的碎片，去填 regular 参数留下的 gap；共 ${L.totalNumel} 个元素，pad ${L.padCells} 格`;
  parts.push(text(X0, AY + 70, guard(capA, 11, W - 2 * X0, 'fig1/capA'), 'cap'));

  // ---- 面板 B：全局缓冲 ----
  const BY = AY + 92;
  parts.push(text(X0, BY, `B　全局扁平桶 ${L.v2.size} 格，按 ${L.dp} 个 rank 各 ${L.shard} 格切开：offsets = (${L.v2.offsets.join(', ')})`, 'pt'));
  const BC = (W - 2 * X0 - 4 * 8) / L.v2.size; // 每格宽度（每 rank 间留 8px）
  const BX = X0;
  const BYY = BY + 14;
  L.cells.forEach((cell, idx) => {
    const r = Math.floor(idx / L.shard);
    const x = BX + idx * BC + r * 8;
    let cls = `r${r}`;
    if (cell.param < 0) cls = 'ghost';
    parts.push(rect(x, BYY, BC - 1.5, 26, cls, 2));
    if (cell.param >= 0) {
      parts.push(text(x + (BC - 1.5) / 2, BYY + 11, `P${cell.param}`, 'sm', 'middle'));
      parts.push(text(x + (BC - 1.5) / 2, BYY + 22, `r${cell.row}`, 'sm', 'middle'));
    } else {
      parts.push(text(x + (BC - 1.5) / 2, BYY + 17, '·', 'sm', 'middle'));
    }
  });
  // 参数区间描边
  L.shapes.forEach((shape, id) => {
    const start = L.v2.offsets[id];
    const end = start + numel(shape);
    const r0 = Math.floor(start / L.shard);
    const r1 = Math.floor((end - 1) / L.shard);
    const x1 = BX + start * BC + r0 * 8;
    const x2 = BX + end * BC + r1 * 8 - 1.5;
    const cls = L.classes[id] === 'fragment' ? 'outline2' : conjSet.has(id) ? 'outline1' : null;
    if (cls) parts.push(`<rect class="${cls}" x="${x1 - 1.5}" y="${BYY - 3}" width="${x2 - x1 + 3}" height="32" rx="3"/>`);
  });
  // rank 标签
  for (let r = 0; r < L.dp; r += 1) {
    const x = BX + r * L.shard * BC + r * 8;
    parts.push(text(x + (L.shard * BC) / 2, BYY + 44, `rank${r}：全局 [${r * L.shard}, ${(r + 1) * L.shard})`, 'rank', 'middle'));
  }
  const capB = `每格写「参数 / 行号」，· = pad；rank 边界都是 ${L.chunk} 的倍数，所以没有任何一行被劈开。P${CONJUGATES[0][1]} 的余数段 [${L.v2.offsets[CONJUGATES[0][1]]}, ${L.v2.offsets[CONJUGATES[0][1]] + (numel(L.shapes[CONJUGATES[0][1]]) % L.chunk)}) 补进 P${CONJUGATES[0][0]} 的 pad；P3 填进 gap；P4 装不进 gap，落在尾部并再 pad 到 ${L.v2.size}`;
  parts.push(text(X0, BYY + 66, guard(capB, 11, W - 2 * X0, 'fig1/capB'), 'cap'));

  // ---- 面板 C：本地形状表 ----
  const CY = BYY + 90;
  parts.push(text(X0, CY, 'C　每 rank 的本地形状（get_local_tensor：owned range 必须整行，否则 RuntimeError；整参数在别的 rank 时本地为空）', 'pt'));
  const cols = [X0, X0 + 140, X0 + 250, X0 + 360, X0 + 470, X0 + 580, X0 + 690, X0 + 820];
  const headers = ['rank', ...L.shapes.map((s, id) => `P${id} ${shapeStr(s)}`), '本地元素数'];
  const TY = CY + 12;
  parts.push(rect(X0 - 4, TY, 900, 22 * (L.dp + 1) + 10, 'panel'));
  headers.forEach((h, i) => parts.push(text(cols[i], TY + 17, h, 'rank')));
  parts.push(line(X0, TY + 23, X0 + 890, TY + 23));
  L.ranks.forEach((r, k) => {
    const y = TY + 40 + k * 22;
    parts.push(text(cols[0], y, `rank${r.rank}  [${r.range[0]}, ${r.range[1]})`, 'sm'));
    r.local.forEach((shape, id) => {
      parts.push(text(cols[id + 1], y, shapeStr(shape), shape[0] === 0 ? 'sm' : 'dim'));
    });
    const localNumel = r.local.reduce((acc, s) => acc + numel(s), 0);
    parts.push(text(cols[6], y, `${localNumel} / ${L.shard}（pad ${L.shard - localNumel}）`, 'sm'));
  });

  // ---- 面板 D：FSDP2 式逐参数 Shard(0) 对照 ----
  const DX = X0 + 920;
  parts.push(text(DX, CY, 'D　对照：逐参数 Shard(0)', 'pt'));
  parts.push(rect(DX - 4, TY, W - X0 - DX + 4, 22 * (L.dp + 1) + 10, 'panel'));
  parts.push(text(DX, TY + 17, '参数', 'rank'));
  parts.push(text(DX + 80, TY + 17, '各 rank 行数', 'rank'));
  parts.push(text(DX + 200, TY + 17, '空 rank', 'rank'));
  parts.push(line(DX, TY + 23, W - X0 - 8, TY + 23));
  L.fsdp2.forEach((f, id) => {
    const y = TY + 40 + id * 22;
    parts.push(text(DX, y, `P${id} ${shapeStr(L.shapes[id])}`, 'sm'));
    parts.push(text(DX + 80, y, `[${f.perRank.join(',')}]`, f.emptyRanks > 0 ? 'costtx' : 'sm'));
    parts.push(text(DX + 200, y, String(f.emptyRanks), f.emptyRanks > 0 ? 'costtx' : 'sm'));
  });
  parts.push(text(DX, TY + 40 + L.dp * 22, `${L.shapes.length} 个参数 = ${L.shapes.length} 次 collective`, 'costtx'));
  const capC = `按 DTensor 公开契约推演（每个参数各自沿 dim-0 均分、前面的 rank 先拿），不是读过 FSDP2 源码；Megatron-FSDP 的做法是整个 unit 一次 collective，每 rank 恰好 ${L.shard} 个元素`;
  parts.push(text(X0, TY + 22 * (L.dp + 1) + 34, guard(capC, 11, W - 2 * X0, 'fig1/capC'), 'cap'));

  // ---- 面板 E：v1 vs v2 差异小例 ----
  const EY = TY + 22 * (L.dp + 1) + 58;
  const D = DIFF;
  parts.push(text(X0, EY, `E　v1 与 v2 的差异：${D.shapes.map(shapeStr).join(' + ')}，DP=${D.dp}，LCM=${D.chunk}，size 都是 ${D.v2.size}`, 'pt'));
  const EC = 22;
  const rowsE = [
    { name: 'v1 build_data_parallel_buffer_index', offsets: D.v1.offsets, cls: 'neutral' },
    { name: 'v2 GlobalLayout.build', offsets: D.v2.offsets, cls: 'acc1' },
  ];
  rowsE.forEach((row, k) => {
    const y = EY + 14 + k * 40;
    parts.push(text(X0, y + 16, row.name, 'rank'));
    const ex = X0 + 250;
    const cells = cellMap(D.shapes, row.offsets, D.v2.size);
    cells.forEach((cell, idx) => {
      const r = Math.floor(idx / (D.v2.size / D.dp));
      const x = ex + idx * EC + r * 8;
      parts.push(rect(x, y, EC - 1.5, 24, cell.param < 0 ? 'ghost' : cell.param === 1 ? (k === 1 ? 'acc1' : 'acc2') : `r${r}`, 2));
      parts.push(text(x + (EC - 1.5) / 2, y + 16, cell.param < 0 ? '·' : `P${cell.param}`, 'sm', 'middle'));
    });
    const p1 = row.offsets[1];
    parts.push(text(ex + D.v2.size * EC + 8 + 12, y + 16, `P1 offset ${p1}${p1 % rowSize(D.shapes[1]) === 0 ? '，是行宽 6 的倍数' : '，不是行宽 6 的倍数'}`, k === 1 ? 'dim' : 'costtx'));
  });
  const capE1 = `v1 把碎片直接放在 gap_offset=${D.v1.offsets[1]}；v2 先 pad 到自身行宽 ${rowSize(D.shapes[1])} 得 ${D.v2.offsets[1]}，GlobalLayout.__post_init__ 会拒绝任何 offset % row_size ≠ 0 的布局`;
  const capE2 = 'v1 的 sorted(fragment_items, …) 返回值没被接住，碎片按注册顺序填 gap；v2 用 fragment_items.sort(…) 让大碎片优先';
  parts.push(text(X0, EY + 14 + 2 * 40 + 8, guard(capE1, 11, W - 2 * X0, 'fig1/capE1'), 'cap'));
  parts.push(text(X0, EY + 14 + 2 * 40 + 24, guard(capE2, 11, W - 2 * X0, 'fig1/capE2'), 'cap'));

  // ---- 底部盒子 ----
  const OY = EY + 14 + 2 * 40 + 42;
  const OW = 600;
  const gap = W - 2 * X0 - 2 * OW;
  parts.push(
    infoBox(X0, OY, OW, 130, '本图算出的规则', [
      `chunk_size_factor = LCM(p.shape[1:].numel()) = ${L.chunk}；桶 size = pad(装填终点, DP × LCM) = ${L.v2.size}`,
      '整除 LCM 的参数直接顺排；大于 LCM 但不整除的参数按余数找 conjugate 装进同一个 chunk',
      '小于 LCM 的碎片先填 regular 参数留下的 gap，填不进的放尾部',
      `每 rank 拿全局 [rank × ${L.shard}, (rank+1) × ${L.shard})：整参数不在本 rank 时本地形状首维为 0`,
      '不做 pad 的唯一情况：no_shard（_pad_if_needed 直接返回）',
    ], 'neutral', 'fig1/rules'),
  );
  parts.push(
    infoBox(X0 + OW + gap, OY, OW, 130, '两代算法在本算例上的差别与复刻范围', [
      `docstring 算例：v1 offsets (${L.v1.offsets.join(', ')}) = v2 offsets (${L.v2.offsets.join(', ')})，size 相同`,
      { text: `差异小例：v1 P1 offset ${D.v1.offsets[1]} vs v2 ${D.v2.offsets[1]}；顺序小例 ${ORDER.shapes.map(shapeStr).join('+')}：v1 size ${ORDER.v1.size} vs v2 ${ORDER.v2.size}`, cls: 'costtx' },
      '精确复刻：v1 的 regular / conjugate / fragment 三段与 _pad_if_needed；v2 的排序、行宽对齐、校验',
      '简化：不画 bucket group 的 NCCL 合并与 TP 维（strided shard 交给 make_fsdp_dtensor）',
      '面板 D 的逐参数分片只是推演，用来对照「一个 unit 一次 collective」',
    ], 'neutral', 'fig1/scope'),
  );
  const H = OY + 130 + 20;
  return seal(parts, W, H, 'fig1-layout');
}

// ============================================================================
// 图 2：时序
// ============================================================================

function ganttPanel(parts, { x0, y0, w, title, sub, lane, tMax, where, laneNames }) {
  parts.push(text(x0, y0, guard(title, 14, w, `${where}/title`), 'pt'));
  const subLines = Array.isArray(sub) ? sub : [sub];
  subLines.forEach((s, k) => parts.push(text(x0, y0 + 16 + k * 14, guard(s, 10.5, w, `${where}/sub${k}`), 'sm')));
  const LH = 20;
  const rows = ['compute', 'ag', 'rs'];
  const labelW = 70;
  const px = (w - labelW) / tMax;
  const rowY = (r) => y0 + 14 + subLines.length * 14 + 14 + rows.indexOf(r) * (LH + 6);
  rows.forEach((r) => {
    parts.push(text(x0 + labelW - 8, rowY(r) + 14, laneNames[r], 'rank', 'end'));
    parts.push(line(x0 + labelW, rowY(r) + LH / 2, x0 + w, rowY(r) + LH / 2, 'gl'));
  });
  // 时间刻度
  for (let t = 0; t <= tMax; t += 4) {
    const x = x0 + labelW + t * px;
    parts.push(line(x, rowY('compute') - 2, x, rowY('rs') + LH + 2, 'gl'));
    parts.push(text(x, rowY('rs') + LH + 14, String(t), 'sm', 'middle'));
  }
  const byId = Object.fromEntries(lane.tasks.map((t) => [t.id, t]));
  lane.tasks.forEach((t) => {
    const x = x0 + labelW + t.start * px;
    const bw = (t.end - t.start) * px;
    let cls = 'neutral';
    if (t.kind === 'ag' || t.kind === 'rs') {
      // 暴露 = 消费者紧接着它开始且它自己无法更早开始（对 AG）；对 RS：最后一个 RS 决定 makespan
      const consumer = t.kind === 'ag' ? byId[t.phase === 'fwd' ? (t.label.startsWith('AGr') && t.phase === 'bwd' ? `R${t.unit}` : `C${t.unit}`) : (t.label.startsWith('AGr') ? `R${t.unit}` : `B${t.unit}`)] : null;
      const exposed = t.kind === 'ag'
        ? consumer && consumer.start === t.end && (t.lane === 'compute' || t.start === Math.max(0, ...t.deps.map((d) => byId[d].end)))
        : t.end === lane.bwdEnd || t.lane === 'compute';
      cls = exposed ? 'acc2' : 'acc1';
    } else if (t.kind === 'sync') {
      cls = 'ghost';
    } else if (t.kind === 'recompute') {
      cls = 'ghost';
    }
    parts.push(rect(x + 1, rowY(t.lane), Math.max(bw - 2, 2), LH, cls, 3));
    parts.push(text(x + bw / 2, rowY(t.lane) + 14, guard(t.label, 10.5, Math.max(bw - 2, 8), `${where}/${t.id}`), t.kind === 'compute' || t.kind === 'recompute' ? 'sm' : cls === 'acc2' ? 'costtx' : 'dim', 'middle'));
  });
  // 前向 / 反向分界
  const xf = x0 + labelW + lane.fwdEnd * px;
  parts.push(line(xf, rowY('compute') - 2, xf, rowY('rs') + LH + 2, 'sep'));
  parts.push(text(xf + 4, rowY('compute') - 4, `前向 makespan ${lane.fwdEnd}`, 'dim'));
  const xb = x0 + labelW + lane.bwdEnd * px;
  parts.push(text(xb - 4, rowY('compute') - 4, `反向结束 ${lane.bwdEnd}`, 'dim', 'end'));
  return rowY('rs') + LH + 26;
}

function renderTimeline() {
  const W = 1272;
  const X0 = 28;
  const PW = W - 2 * X0;
  const parts = header(
    W,
    '图 2　hook 状态机与两条流：v1、v2 与 MXFP8 全量重计算在同一算例上各把通信藏在哪',
    `L=${SIM.L} 个 unit，前向 c=${SIM.c}，反向 2c=${2 * SIM.c}，AG a=${SIM.a}，RS r=${SIM.r}，v2 权重同步 s=${SIM.s}，fsdp_buffer_count=${SIM.slots}；蓝 = 被掩盖的通信，橙 = 暴露的等待，灰 = 权重同步 / 重算；时长示意，发起点与等待点来自冻结源码`,
  );
  const tMax = Math.max(LANE_V1.bwdEnd, LANE_V2.bwdEnd, LANE_RECOMPUTE.bwdEnd, LANE_RECOMPUTE_NO_PREFETCH.bwdEnd);
  let y = 84;
  y = ganttPanel(parts, {
    x0: X0, y0: y, w: PW, tMax, where: 'fig2/v1',
    title: `① v1 MegatronFSDP（optim_grads_params，独立 AG 进程组）：前向 ${LANE_V1.fwdEnd}，反向结束 ${LANE_V1.bwdEnd}，峰值在途参数桶 ${LANE_V1.peakParams}、梯度桶 ${LANE_V1.peakGrads}`,
    sub: [
      '前向：_pre_forward_param_unshard 发 AG(i) 并预取 AG(i+1)（all_gather_stream.wait_stream 使预取排在上一个 F 之后），_post_forward 释放',
      '反向：_pre_backward_param_unshard 发 AG(i) 并预取 AG(i−1)；RegisterFSDPBackwardFunction 释放；post-accumulate-grad hook → reduce_gradients；finish_grad_sync 等最后一个 RS',
    ],
    lane: LANE_V1,
    laneNames: { compute: 'compute', ag: 'AG 流', rs: 'RS 流' },
  });
  y = ganttPanel(parts, {
    x0: X0, y0: y + 10, w: PW, tMax, where: 'fig2/v2',
    title: `② v2 experimental.fully_shard（Flat placements，单一 communicator）：前向 ${LANE_V2.fwdEnd}，反向结束 ${LANE_V2.bwdEnd}，峰值在途 ${LANE_V2.peakParams}（延迟释放队列 ≤ ${LANE_V2.delayedQueueMax}）`,
    sub: [
      '前向：pre_forward 在 allgather_stream 上先 S(i)=sync_model_weight_from_main_weight 再 AG(i)，current_stream.wait_stream；post_forward enqueue_release，下一次 unshard 时 drain 到 1，root 处 drain 到 0',
      '反向：pre_backward 无预取；post_backward 的 reduce_scatter_tensor 落在 compute 流；同一 communicator 上 RS(i) → AG(i−1) 串行，AG(i−1) 只能在 RS(i) 之后开始',
    ],
    lane: LANE_V2,
    laneNames: { compute: 'compute', ag: 'allgather', rs: '（无 RS 流）' },
  });
  y = ganttPanel(parts, {
    x0: X0, y0: y + 10, w: PW, tMax, where: 'fig2/rc',
    title: `③ v1 MXFP8 全量重计算 + prefetch_recompute_forward_weights：反向结束 ${LANE_RECOMPUTE.bwdEnd}（不开预取则 ${LANE_RECOMPUTE_NO_PREFETCH.bwdEnd}），峰值 rowwise 桶 ${LANE_RECOMPUTE.peakRowwise}、columnwise 桶 ${LANE_RECOMPUTE.peakColwise}`,
    sub: [
      '_root_pre_backward 把全部模块置 PRE_BACKWARD；pre-backward 三连发：AGc(i) 不预取 → AGr(i) 并预取 AGr(i−1) → AGc(i) 并预取 AGc(i−1)',
      '重算 R(i) 进 _pre_forward_param_unshard 时状态是 PRE_BACKWARD → 取消预取；_post_forward → lazy release，rowwise 桶到 post-backward 才真正回收',
    ],
    lane: LANE_RECOMPUTE,
    laneNames: { compute: 'compute', ag: 'AG 流', rs: 'RS 流' },
  });

  const BW = 600;
  const gap = PW - 2 * BW;
  const BY = y + 14;
  parts.push(
    infoBox(X0, BY, BW, 150, 'TrainingState（挂在每个子模块上；图上的位置）', [
      { text: 'FORWARD：_pre_forward_param_unshard 置，_post_forward 回 IDLE —— 覆盖 ① 的每个 F(i)', cls: 'dim' },
      'PRE_BACKWARD：_root_pre_backward 在 loss.backward 起点对全部模块置 —— ① ③ 的反向全程',
      '  → 重算前向 R(i) 进 _pre_forward_param_unshard 时取消预取；进 _post_forward 时 lazy_release',
      'POST_BACKWARD：枚举里定义，冻结基线里没有任何赋值点',
      'IDLE：_post_backward_release_module 把 unit 及其子模块置回 —— 每个 B(i) 的终点',
      'v2 没有这套状态：只有 Flat / Replicate / Partial 三种 placement 与一条延迟释放队列',
    ], 'neutral', 'fig2/ts'),
  );
  parts.push(
    infoBox(X0 + BW + gap, BY, BW, 150, 'BucketStatus（AllGatherPipeline 与 GradReducePipeline 各一份）', [
      { text: 'EMPTY → COMMUNICATING：async_bucket_gather 发起 AG —— 蓝 / 橙块的起点', cls: 'dim' },
      'COMMUNICATING → READY_TO_USE：wait_bucket_ready —— F(i) / B(i) 的起点',
      'READY_TO_USE → EMPTY：release_bucket —— F(i) / B(i) 的终点；lazy 时延到下一次分配',
      'PRESERVED：reset(preserve_non_fsdp_units=True) 给非 unit 桶；下一次 AG 原地刷新',
      '梯度桶：EMPTY → COMMUNICATING（RS 发起）→ EMPTY（free_up_grad_bucket）',
      `槽位：参数桶与梯度桶各 ${SIM.slots} 个持久槽，AG 预取深度 = 槽数 − 1`,
    ], 'neutral', 'fig2/bs'),
  );
  const RY = BY + 150 + 12;
  parts.push(
    infoBox(X0, RY, PW, 118, '仿真结论（脚本算出；时长是示意单位）', [
      { text: `v1 前向 makespan = a + L·c = ${LANE_V1.fwdEnd}，只有 AG0 暴露；反向结束 ${LANE_V1.bwdEnd}，只有 AG${SIM.L - 1} 与最后一个 RS 暴露；峰值在途参数桶 ${LANE_V1.peakParams}、梯度桶 ${LANE_V1.peakGrads}`, cls: 'dim' },
      { text: `v2 前向 ${LANE_V2.fwdEnd}（多一次 S）；反向 = L·(a + 2c + r) = ${LANE_V2.bwdEnd - LANE_V2.fwdEnd}，完全串行；峰值在途 ${LANE_V2.peakParams}，延迟释放队列 ≤ ${LANE_V2.delayedQueueMax}（test_forward_peak_memory_bounds_in_flight_child_all_gathers 的上界是 3）`, cls: 'costtx' },
      { text: `独立 AG 进程组：a ≥ ${AG_COMM.sameCommFirstLag}（= 2c − r + 1）时共用 communicator 开始拉长反向，独立组要到 a ≥ ${AG_COMM.separateFirstLag}（= 2c + 1）；本例 a=${SIM.a} 时两者同为 ${LANE_V1_SAME_COMM.bwdEnd}`, cls: 'costtx' },
      `重算预取把反向从 ${LANE_RECOMPUTE_NO_PREFETCH.bwdEnd} 缩到 ${LANE_RECOMPUTE.bwdEnd}，代价是 rowwise 与 columnwise 各 ${LANE_RECOMPUTE.peakRowwise} 个桶同时在途；重叠证据：v1 靠显式预取，v2 靠 CPU run-ahead（test_overlaps_all_gather_and_compute 的 profiler 断言）`,
    ], 'neutral', 'fig2/res'),
  );
  const H = RY + 118 + 20;
  return seal(parts, W, H, 'fig2-timeline');
}

// ============================================================================
// 图 3：代价
// ============================================================================

function renderCost() {
  const W = 1272;
  const X0 = 28;
  const c = COST;
  const parts = header(
    W,
    '图 3　每参数每 rank 的字节账：四档 + HSDP / HFSDP 的常驻、双缓冲的瞬态、每 step 的通信',
    `llama3_8b 配方：hidden ${RECIPE.hidden}、ffn ${RECIPE.ffn}、${RECIPE.layers} 层、GQA ${RECIPE.queryGroups}×${RECIPE.kvChannels}，DP=${RECIPE.dp}，${RECIPE.microbatches} microbatch/step；unit = ${c.unit.total.toLocaleString('en-US')} 参数（bf16 ${fmtMiB(c.unitBytes.bf16)} MiB，fp8 ${fmtMiB(c.unitBytes.fp8)} MiB）；HSDP / HFSDP 按 D=${RECIPE.dp}、O=${RECIPE.hsdpOuter} 延伸`,
  );

  // ---- 左：常驻堆叠柱 ----
  const LX = X0;
  const LY = 92;
  const LW = 640;
  parts.push(text(LX, LY, '常驻字节 / 参数 / rank（W 计算权重、G 主梯度、M fp32 主权重、O Adam 两个 fp32 状态）', 'pt'));
  const AXY = LY + 210;
  const AXH = 160;
  const maxVal = Math.max(...c.resident.bf16.map((r) => r.total));
  const scaleY = (v) => (v / maxVal) * AXH;
  parts.push(line(LX + 40, AXY, LX + LW, AXY, 'sep'));
  for (const tick of [0, 4, 8, 12, 16]) {
    const yy = AXY - scaleY(tick);
    parts.push(line(LX + 40, yy, LX + LW, yy, 'gl'));
    parts.push(text(LX + 36, yy + 4, `${tick} B`, 'sm', 'end'));
  }
  const groups = [
    { name: 'bf16 参数', rows: c.resident.bf16, cls: ['bar1', 'bar2', 'bar3', 'bar4'] },
    { name: 'fp8 参数（--fp8-param-gather）', rows: c.resident.fp8, cls: ['bar1', 'bar2', 'bar3', 'bar4'] },
  ];
  const gw = (LW - 60) / groups.length;
  groups.forEach((g, gi) => {
    const gx = LX + 50 + gi * gw;
    const bw = (gw - 20) / g.rows.length;
    g.rows.forEach((row, bi) => {
      const x = gx + bi * bw;
      let yTop = AXY;
      [['W', 'bar3'], ['G', 'bar2'], ['M', 'bar1'], ['O', 'bar4']].forEach(([key, cls]) => {
        const h = scaleY(row[key]);
        if (h > 0) {
          parts.push(`<rect class="${cls}" x="${(x + 3).toFixed(1)}" y="${(yTop - h).toFixed(1)}" width="${(bw - 6).toFixed(1)}" height="${h.toFixed(1)}"/>`);
          yTop -= h;
        }
      });
      parts.push(text(x + bw / 2, yTop - 5, row.total.toFixed(row.total % 1 === 0 ? 0 : 3), 'dim', 'middle'));
      const short = row.key === 'optim_grads_params' ? 'ogp' : row.key === 'optim_grads' ? 'og' : row.key === 'no_shard' ? 'none' : row.key;
      parts.push(text(x + bw / 2, AXY + 13, short, 'sm', 'middle'));
    });
    parts.push(text(gx + (gw - 20) / 2, AXY + 28, g.name, 'rank', 'middle'));
  });
  // 图例
  const legendY = AXY + 46;
  [['W', 'bar3'], ['G', 'bar2'], ['M', 'bar1'], ['O', 'bar4']].forEach(([key, cls], i) => {
    parts.push(`<rect class="${cls}" x="${LX + 50 + i * 120}" y="${legendY - 9}" width="12" height="12"/>`);
    parts.push(text(LX + 68 + i * 120, legendY + 1, { W: 'W 计算权重', G: 'G 主梯度', M: 'M 主权重 fp32', O: 'O Adam 状态' }[key], 'sm'));
  });
  const bf = c.resident.bf16;
  parts.push(
    infoBox(LX, legendY + 16, LW, 130, '怎么算的（none=no_shard，og=optim_grads，ogp=optim_grads_params）', [
      `no_shard：W+G+M+O 全复制 = ${bf[0].total} B；optim：W+G + (M+O)/D = ${bf[1].total} B；optim_grads：W + (G+M+O)/D = ${bf[2].total} B`,
      `optim_grads_params：(W+G+M+O)/D = ${bf[3].total} B；HSDP 外层复制 = ${bf[4].total} B；HFSDP：(W+G)/D + (M+O)/(D·O) = ${bf[5].total} B`,
      `G：main_grads_dtype=None → 参数 dtype（bf16 ${DTYPES.bf16.G} B），fp8 参数 → bf16；M：main_params_dtype 默认 fp32`,
      { text: `O = Adam 两个 fp32 状态（假设）；整模型 ${(c.modelParams / 1e9).toFixed(2)}B 参数：ogp 每 rank ${fmtGiB(c.modelParams * bf[3].total)} GiB，no_shard ${fmtGiB(c.modelParams * bf[0].total)} GiB`, cls: 'costtx' },
      'vocab 不进 unit 账，进整模型账；TP=1，没有 strided shard',
    ], 'neutral', 'fig3/how'),
  );

  // ---- 右上：瞬态 ----
  const RX = X0 + LW + 40;
  const RW = W - RX - X0;
  parts.push(text(RX, LY, '瞬态：一个 unit 的未分片桶 × 持久池槽数（MiB / rank）', 'pt'));
  const TY = LY + 12;
  const trows = [
    ['bf16 参数桶', c.transient.map((t) => t.bf16)],
    ['fp8 参数桶', c.transient.map((t) => t.fp8)],
    ['MXFP8 rowwise + columnwise', c.transient.map((t) => t.mxfp8)],
    ['bf16 梯度桶（grad_comm_dtype=None）', c.transient.map((t) => t.grad)],
  ];
  parts.push(rect(RX - 4, TY, RW + 4, 22 * (trows.length + 1) + 10, 'panel'));
  parts.push(text(RX, TY + 17, '桶', 'rank'));
  parts.push(text(RX + 260, TY + 17, `× ${c.transient[0].buffers}（默认）`, 'rank'));
  parts.push(text(RX + 400, TY + 17, `× ${c.transient[1].buffers}（1F1B EP overlap）`, 'rank'));
  parts.push(line(RX, TY + 23, RX + RW - 8, TY + 23));
  trows.forEach(([name, vals], k) => {
    const y = TY + 40 + k * 22;
    parts.push(text(RX, y, name, 'sm'));
    parts.push(text(RX + 260, y, `${fmtMiB(vals[0])} MiB`, 'dim'));
    parts.push(text(RX + 400, y, `${fmtMiB(vals[1])} MiB`, 'costtx'));
  });
  parts.push(text(RX, TY + 22 * (trows.length + 1) + 26, guard('参数桶与梯度桶各一个池；双缓冲 = 当前 unit + 预取的后继；HSDP 自定义 grad_comm_dtype 另加一份', 11, RW, 'fig3/capT'), 'cap'));

  // ---- 右下：通信 ----
  const CY = TY + 22 * (trows.length + 1) + 50;
  parts.push(text(RX, CY, `每 step 每 rank 接收字节 / 参数（bf16，${RECIPE.microbatches} 个 microbatch）`, 'pt'));
  const CAXY = CY + 200;
  const CAXH = 150;
  const comm = c.comm.bf16;
  const maxComm = Math.max(...comm.map((r) => r.total));
  const cScale = (v) => (v / maxComm) * CAXH;
  parts.push(line(RX + 40, CAXY, RX + RW, CAXY, 'sep'));
  const tickStep = maxComm > 100 ? 40 : 10;
  for (let tick = 0; tick <= maxComm; tick += tickStep) {
    const yy = CAXY - cScale(tick);
    parts.push(line(RX + 40, yy, RX + RW, yy, 'gl'));
    parts.push(text(RX + 36, yy + 4, `${tick}`, 'sm', 'end'));
  }
  const cbw = (RW - 50) / comm.length;
  comm.forEach((row, i) => {
    const x = RX + 45 + i * cbw;
    const hMb = cScale(row.perMicrobatch * RECIPE.microbatches);
    const hCy = cScale(row.perCycle);
    if (hMb > 0) parts.push(`<rect class="bar3" x="${(x + 4).toFixed(1)}" y="${(CAXY - hMb).toFixed(1)}" width="${(cbw - 8).toFixed(1)}" height="${hMb.toFixed(1)}"/>`);
    if (hCy > 0) parts.push(`<rect class="bar" x="${(x + 4).toFixed(1)}" y="${(CAXY - hMb - hCy).toFixed(1)}" width="${(cbw - 8).toFixed(1)}" height="${hCy.toFixed(1)}"/>`);
    parts.push(text(x + cbw / 2, CAXY - hMb - hCy - 5, row.total.toFixed(1), 'dim', 'middle'));
    const short = row.key === 'optim_grads_params' ? 'ogp' : row.key === 'optim_grads' ? 'og' : row.key === 'no_shard' ? 'none' : row.key;
    parts.push(text(x + cbw / 2, CAXY + 13, short, 'sm', 'middle'));
  });
  parts.push(`<rect class="bar3" x="${RX + 45}" y="${CAXY + 22}" width="12" height="12"/>`);
  parts.push(text(RX + 63, CAXY + 32, '每 microbatch：AG 前向 + AG 反向 + RS', 'sm'));
  parts.push(`<rect class="bar" x="${RX + 300}" y="${CAXY + 22}" width="12" height="12"/>`);
  parts.push(text(RX + 318, CAXY + 32, '每优化周期：延迟 RS / AG / 外层 AR 或 RS+AG', 'sm'));
  const ogp = comm[3];
  const hs = comm[4];
  const hf = comm[5];
  parts.push(text(RX, CAXY + 52, guard(`ogp 每 microbatch (2W+G)(D−1)/D = ${ogp.perMicrobatch.toFixed(2)} B`, 11, RW, 'fig3/capC'), 'cap'));
  parts.push(text(RX, CAXY + 68, guard(`HSDP 外层 AR 只在最后一个 microbatch：+${hs.perCycle.toFixed(3)} B；HFSDP 外层 RS+AG 只在优化周期：+${hf.perCycle.toFixed(3)} B`, 11, RW, 'fig3/capC2'), 'cap'));
  parts.push(text(RX, CAXY + 84, guard(`no_shard 每周期 AR = ${comm[0].total.toFixed(2)} B；optim 延迟 RS + AG = ${comm[1].total.toFixed(2)} B；optim_grads = ${comm[2].total.toFixed(2)} B`, 11, RW, 'fig3/capC3'), 'cap'));

  const H = Math.max(legendY + 16 + 130, CAXY + 84) + 24;
  return seal(parts, W, H, 'fig3-cost');
}

// ============================================================================

const outputs = new Map([
  ['megatron_fsdp_layout.svg', renderLayout()],
  ['megatron_fsdp_timeline.svg', renderTimeline()],
  ['megatron_fsdp_cost.svg', renderCost()],
]);

export {
  LAYOUT_CASE, DIFF_CASE, ORDER_CASE, SIM, RECIPE, DTYPES, CONFIGS,
  LAYOUT, DIFF, ORDER, CONJUGATES,
  LANE_V1, LANE_V1_SAME_COMM, LANE_V2, LANE_RECOMPUTE, LANE_RECOMPUTE_NO_PREFETCH, AG_COMM, agCommThreshold,
  COST, UNIT, MODEL_PARAMS, MiB, fmtMiB, fmtGiB,
  numel, rowSize, lcmChunkSizeFactor, v1BuildIndex, v2BuildLayout, localShapes, cellMap,
  perParameterShard0, runLayout, conjugatePairs, schedule, peakLive,
  simulateV1, simulateV2, simulateRecompute, unitParams, residentBytes, commBytes, outputs,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir =
    process.argv[2] ??
    join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of outputs) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`${name}  ${svg.length} bytes`);
  }
  console.log('\n图 1：', JSON.stringify({ chunk: LAYOUT.chunk, v1: LAYOUT.v1.offsets, v2: LAYOUT.v2.offsets, size: LAYOUT.v2.size, diff: [DIFF.v1.offsets, DIFF.v2.offsets], order: [ORDER.v1, ORDER.v2] }));
  for (const lane of [LANE_V1, LANE_V1_SAME_COMM, LANE_V2, LANE_RECOMPUTE, LANE_RECOMPUTE_NO_PREFETCH]) {
    console.log('图 2：', lane.name, JSON.stringify({ fwd: lane.fwdEnd, bwd: lane.bwdEnd, peakParams: lane.peakParams, peakGrads: lane.peakGrads, row: lane.peakRowwise, col: lane.peakColwise }));
  }
  console.log('图 3：', JSON.stringify({ unit: UNIT.total, model: MODEL_PARAMS, bf16: COST.resident.bf16.map((r) => [r.key, r.total]), comm: COST.comm.bf16.map((r) => [r.key, r.total.toFixed(2)]) }));
}

// ============================================================================
// 业务链：unit → 桶 → 预取预算 / RS 队列（页面 §2.6 引用的数值）
//   规则来自 param_and_grad_buffer.py::_get_parameter_groups Step 3 的 chunk_size_factor 合并、
//   build_data_parallel_buffer_index 的 pad、以及 megatron_fsdp.py::MegatronFSDP.
//   _init_fsdp_param_and_grad_buffer 里 suggested_communication_unit_size 的推导
//   （None → max(1e9, 每 unit 平均元素数 × 2)；AG 预取预算 = 它的一半；RS 队列容量 = 它）。
// ============================================================================

// llama3-8b 一个 TransformerLayer unit 在 TE 模块下的参数形状（qkv 与 gate/up 各为一个融合权重）
const UNIT_SHAPES = Object.freeze([
  Object.freeze([RECIPE.hidden + 2 * RECIPE.queryGroups * RECIPE.kvChannels, RECIPE.hidden]), // linear_qkv
  Object.freeze([RECIPE.hidden, RECIPE.hidden]), // linear_proj
  Object.freeze([2 * RECIPE.ffn, RECIPE.hidden]), // linear_fc1（swiglu gate + up）
  Object.freeze([RECIPE.hidden, RECIPE.ffn]), // linear_fc2
  Object.freeze([RECIPE.hidden]), // input layer_norm_weight
  Object.freeze([RECIPE.hidden]), // pre-mlp layer_norm_weight
]);

function v1UnitBucket(shapes, dp) {
  // Step 3：按 shape[1:].numel() 降序；能整除当前 chunk、或本身小于 chunk 的并入，否则取 LCM
  const sorted = [...shapes].sort((a, b) => rowSize(b) - rowSize(a));
  let chunk = rowSize(sorted[0]);
  for (const s of sorted) {
    const r = rowSize(s);
    if (r === chunk || (chunk % r === 0 && numel(s) % chunk === 0) || numel(s) < chunk) continue;
    chunk = lcm(chunk, r);
  }
  const elems = shapes.reduce((acc, s) => acc + numel(s), 0);
  return { chunk, elems, padded: padTo(elems, dp * chunk) };
}

function prefetchPlan({ unitElems, paddedUnitElems }) {
  const commUnit = Math.max(1_000_000_000, unitElems * 2); // 注释写 Cap，实际是下限
  const agBudget = Math.floor(commUnit / 2); // suggested_AG_prefetch_size
  // all_gather_params 的预取循环：已预取量 < 预算就再并入一个 bucket group（= 一个 unit）
  let prefetchUnitsNoPool = 0;
  while (prefetchUnitsNoPool * paddedUnitElems < agBudget) prefetchUnitsNoPool += 1;
  // wait_for_previous_grad_reduce：只有队列里的元素数已经 > 容量才弹出等待，再追加新桶
  const rsQueueUnits = Math.floor(commUnit / paddedUnitElems) + 1;
  return { commUnit, agBudget, prefetchUnitsNoPool, rsQueueUnits };
}

const UNIT_BUCKET = v1UnitBucket(UNIT_SHAPES, RECIPE.dp);
const PREFETCH = prefetchPlan({ unitElems: UNIT_BUCKET.elems, paddedUnitElems: UNIT_BUCKET.padded });
if (UNIT_BUCKET.elems !== UNIT.total) throw new Error('unit 形状表与字节账的参数量不一致');

export { UNIT_SHAPES, UNIT_BUCKET, PREFETCH, v1UnitBucket, prefetchPlan };
