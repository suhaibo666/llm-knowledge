// 31_megatron_inference_engine_analysis.md 的三张图。
//
// 图 1：**原理图** —— 块级 KV cache。四条请求在一个 12 块的物理池上逐步申请、增长、释放，
//       以及 prefix caching 下的 hash 命中、ref count 与 ref_zero / lru 两种驱逐策略。
//       块的分配、释放、驱逐全部由复刻的 kv_block_allocator.py::KVBlockAllocator 语义算出，
//       admission 由复刻的 dynamic_context.py::DynamicInferenceContext._compute_prefix_match /
//       check_availability / add_request / update_requests(pause → resume) 与
//       dynamic_engine.py 的 schedule_* 规则算出。
// 图 2：**时序图** —— 同一批请求在「定长批 / 连续批处理 / 连续批处理 + chunked prefill」三条
//       路径上的槽位占用、空转槽位数与每步 token 负载，由复刻的 admission 规则逐步算出；
//       定长批复刻 text_generation_controller.py::generate_all_output_tokens_static_batch 的
//       「prefill 到最短 prompt，然后整批每步推进 1 个位置」语义。
// 图 3：**布局图** —— CUDA graph 尺寸枚举。EXPONENTIAL 与 LINEAR 两种分布在同一
//       cuda_graph_max_tokens / tp_size 下各枚举出哪些尺寸、多少张图、最坏 padding 多少，
//       由复刻的 batch_dimensions_utils.py::CUDAGraphBatchDimensionBuilder 算出，并在
//       上界放大到 1024 时重算一遍，证明「指数分布的 padding 上界与规模无关」。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「块为什么是按需给的、共享凭什么安全、没分出去的块为什么是浪费」。
//   共用算例：block_size_tokens=4；四条请求 R0(6→3) R1(10→2) R2(3→5) R3(12→2)，
//   写法是 prompt 长度 → 生成长度；R0/R1/R3 共享前 4 个 token（一个 system prompt 块），
//   R1/R3 再共享接下来 4 个。物理池 total_count=12、paused_count=2（10 块的池在忠实复刻
//   resume_paused_requests 的三条上限后会触发 paused buffer 溢出驱逐，本图刻意不复刻那条路，
//   所以池给到 12 块，恰好让每次 pause 都在同一步被 resume）。
//   上半格：ref_zero 策略，行 = 物理块 0..11（11 是 dummy），列 = 引擎 step，格子按 owner 上色，
//   被多条请求共享的块画 acc1 边并写 ref 数，第一波跑完再把同样四条 prompt 提交一遍（GRPO 式）。
//   下半格：同一算例换成 lru 策略，释放后的块以 ghost「cached」留在池里，第二波直接命中。
//   右栏四个盒子：不变量（可用块 = total−1、active = total−paused−1、dummy 下标、块 id 是栈）、
//   两种策略在两波的命中/跳过/推迟/驱逐数、朴素「按 max_sequence_length 预留」的浪费比、
//   本图复刻了什么/简化了什么。底部一行：每步的 admission 事件（+进入 / ~推迟 / −完成）。
//
// 图 2 要回答「连续批处理解决了什么、又留下了什么给 chunked prefill」。
//   三个面板纵向排列，行 = 槽位（max_requests=3），列 = step，格子写 R?·P<n>（整段 prefill）、
//   R?·p（定长批里逐 token 走 prompt）、R?·c<n>（一个 chunk）、R?·d（decode）、R?·×（已完成
//   仍被计算）；每个面板最下一行写该 step 的 token 负载。
//   定长批：同进同出；连续批处理：完成即走、FIFO 队头不进则整体停；chunked：队头装不下就切一段。
//   右栏一张结算表：步数 / 空转槽·步 / 最大单步 token / R3 进入的 step，以及 max_tokens 为什么是硬预算。
//
// 图 3 要回答「为什么默认从线性改成指数」。上半左右两个面板（num_cuda_graphs=16 默认值）各一把
//   0..64 的尺子，刻度是枚举出的图尺寸；尺子下方为每个真实 batch（n=1..64）画一根 padding 柱
//   （选中的最小合适图 − n）。面板底部写图数、最坏相对 padding 及其发生的 n、平均相对 padding，
//   以及 num_cuda_graphs=-1 自动模式的同三个量。下半把上界放大到 1024 再画两把尺子：线性步长
//   随上界变粗、最小图变成 64，指数分布仍含 tp_size 这一档。
//
// 用法：node tools/figs/svg/megatron_inference_engine_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例
// ============================================================================

const SHARED_A = [100, 101, 102, 103]; // 所有请求共享的 system prompt 块
const SHARED_B = [200, 201, 202, 203]; // R1 / R3 再共享的第二块

const CFG = Object.freeze({
  blockSize: 4,
  requests: Object.freeze([
    { id: 'R0', prompt: [...SHARED_A, 300, 301], gen: 3 },
    { id: 'R1', prompt: [...SHARED_A, ...SHARED_B, 400, 401], gen: 2 },
    { id: 'R2', prompt: [500, 501, 502], gen: 5 },
    { id: 'R3', prompt: [...SHARED_A, ...SHARED_B, 600, 601, 602, 603], gen: 2 },
  ]),
  // 图 1：块池
  totalCount: 12,
  pausedCount: 2,
  fig1MaxRequests: 4,
  maxSequenceLength: 16, // 只用于「朴素定长预留」的对照
  // 图 2：admission 预算
  fig2MaxRequests: 3,
  fig2MaxTokens: 12,
  fig2TotalCount: 64,
  // 图 3：CUDA graph 枚举
  tpSize: 2,
  cudaGraphMaxTokens: 64, // = max_requests × (num_speculative_tokens + 1)，max_requests=64
  numSpeculativeTokens: 0,
  defaultNumCudaGraphs: 16, // InferenceSetupConfig.inference_dynamic_batching_num_cuda_graphs 默认值
  largeCudaGraphMaxTokens: 1024, // 规模对照：max_requests=1024 时的 decode 图上界
});

// ============================================================================
// 复刻 inference_request.py::compute_block_hashes_batched 的父链语义
// （SHA-256 换成确定性字符串键；等价关系一致：同前缀 → 同 hash，前缀不同 → hash 不同）
// ============================================================================

function computeBlockHashes(tokens, blockSize) {
  const n = Math.floor(tokens.length / blockSize);
  const hashes = [];
  let parent = 'root';
  for (let i = 0; i < n; i += 1) {
    const block = tokens.slice(i * blockSize, (i + 1) * blockSize).join(',');
    parent = `${parent}|${block}`;
    hashes.push(parent);
  }
  return hashes;
}

// ============================================================================
// 复刻 kv_block_allocator.py::KVBlockAllocator
// ============================================================================

class KVBlockAllocator {
  constructor(context, totalCount, pausedCount, enablePrefixCaching = false, policy = 'ref_zero') {
    this.context = context;
    this.enablePrefixCaching = enablePrefixCaching;
    this.policy = policy;
    this.totalCount = totalCount;
    this.totalAvail = totalCount - 1; // -1 给 dummy_block_idx
    this.pausedCount = pausedCount;
    this.activeCount = totalCount - pausedCount - 1;
    if (!(this.activeCount >= 1)) throw new Error('assert active_count >= 1 失败');
    this.dummyBlockIdx = totalCount - 1;
    this.blockBag = Array.from({ length: totalCount }, (_, i) => i);
    if (enablePrefixCaching) {
      this.blockHashes = new Array(totalCount).fill(-1);
      this.kvHashToBlockId = new Map();
      this.blockRefCounts = new Array(totalCount).fill(0);
      if (policy === 'lru') this.blockTimestamps = new Array(totalCount).fill(0);
    }
    this.evictions = 0;
  }

  getTotalUsed() {
    return this.totalCount - this.totalAvail - 1;
  }

  // get_active_used：无 prefix caching 时按各请求块数求和；有则按 unique block id 计数
  getActiveUsed() {
    const rows = this.context.activeRows();
    if (!this.enablePrefixCaching) return rows.reduce((s, r) => s + r.blockTable.length, 0);
    return new Set(rows.flatMap((r) => r.blockTable)).size;
  }

  getPausedUsed() {
    const rows = this.context.pausedRows();
    if (!this.enablePrefixCaching) return rows.reduce((s, r) => s + r.blockTable.length, 0);
    return new Set(rows.flatMap((r) => r.blockTable)).size;
  }

  getActiveAvail() {
    return this.activeCount - this.getActiveUsed();
  }

  getEvictableBlockCount() {
    let n = 0;
    for (let i = 0; i < this.totalCount; i += 1) {
      if (this.blockRefCounts[i] === 0 && this.blockHashes[i] !== -1) n += 1;
    }
    return n;
  }

  isMemoryAvailable(numBlocks) {
    if (this.totalAvail >= numBlocks) return true;
    if (!this.enablePrefixCaching) return false;
    if (this.policy === 'ref_zero') return false;
    return this.totalAvail + this.getEvictableBlockCount() >= numBlocks;
  }

  allocateMemoryBlocks(numBlocks) {
    if (this.totalAvail < numBlocks) {
      if (!this.enablePrefixCaching || this.policy === 'ref_zero') return null;
      if (!this.evictLruBlocks(numBlocks - this.totalAvail)) return null;
    }
    this.totalAvail -= numBlocks;
    const ids = this.blockBag.slice(this.totalAvail, this.totalAvail + numBlocks);
    if (ids.length !== numBlocks) throw new Error('assert num_blocks == block_ids.numel() 失败');
    if (this.enablePrefixCaching) {
      for (const b of ids) this.blockRefCounts[b] = 1;
      if (this.policy === 'lru') this.updateTimestamps(ids);
    }
    return ids;
  }

  releaseMemoryBlocks(blocks) {
    if (blocks.length === 0) return;
    if (this.enablePrefixCaching) {
      for (const b of blocks) this.blockRefCounts[b] -= 1;
      if (this.policy === 'ref_zero') {
        const zero = blocks.filter((b) => this.blockRefCounts[b] === 0);
        if (zero.length) this.deregisterBlocks(zero);
      } else {
        // 未登记 hash 的块（尾部不满一块）没有可复用的条目，直接回池，避免泄漏
        const unreg = blocks.filter((b) => this.blockRefCounts[b] === 0 && this.blockHashes[b] === -1);
        for (let i = 0; i < unreg.length; i += 1) this.blockBag[this.totalAvail + i] = unreg[i];
        this.totalAvail += unreg.length;
      }
    } else {
      for (let i = 0; i < blocks.length; i += 1) this.blockBag[this.totalAvail + i] = blocks[i];
      this.totalAvail += blocks.length;
    }
  }

  registerKvBlockHashes(blockIds, hashes) {
    for (let i = 0; i < blockIds.length; i += 1) {
      this.blockHashes[blockIds[i]] = hashes[i];
      this.kvHashToBlockId.set(hashes[i], blockIds[i]);
    }
  }

  deregisterBlocks(blockIds) {
    if (blockIds.length === 0) return;
    for (const b of blockIds) {
      const h = this.blockHashes[b];
      if (h !== -1 && this.kvHashToBlockId.get(h) === b) this.kvHashToBlockId.delete(h);
      this.blockHashes[b] = -1;
      this.blockRefCounts[b] = 0;
      if (this.policy === 'lru') this.blockTimestamps[b] = 0;
    }
    for (let i = 0; i < blockIds.length; i += 1) this.blockBag[this.totalAvail + i] = blockIds[i];
    this.totalAvail += blockIds.length;
  }

  updateTimestamps(blockIds) {
    if (this.policy !== 'lru' || blockIds.length === 0) return;
    for (const b of blockIds) this.blockTimestamps[b] = this.context.prefixCacheLruClock;
  }

  evictLruBlocks(numNeeded) {
    const cached = [];
    for (let i = 0; i < this.totalCount; i += 1) {
      if (this.blockRefCounts[i] === 0 && this.blockHashes[i] !== -1) cached.push(i);
    }
    if (cached.length < numNeeded) return false;
    cached.sort((a, b) => this.blockTimestamps[a] - this.blockTimestamps[b] || a - b);
    const victims = cached.slice(0, numNeeded);
    this.deregisterBlocks(victims);
    this.evictions += victims.length;
    return true;
  }
}

// ============================================================================
// 复刻 DynamicInferenceContext 的 admission / 块表 / 请求生命周期子集，
// 以及 dynamic_engine.py 的 schedule_non_chunked_prefill / schedule_chunked_prefill。
//
// 精确复刻：_compute_prefix_match、_find_kv_match_count、check_availability、add_request 的
//   块分配与 hash 登记、schedule_* 的 FIFO / pending-hash 推迟 / chunk 切分与「不留 1 token
//   尾巴」规则、update_requests 的完成释放与「末块已满则需新块」判定、resume 的 LIFO 顺序。
// 简化：不复刻 paused buffer 溢出后的 evict_overflow_paused_requests（本算例的 paused 用量
//   始终 ≤ paused_count，脚本对此断言）；不复刻 Mamba、投机解码、stop words、logprobs；
//   每步每条 decode 请求恰好生成 1 个 token；定长批面板按 legacy static engine 的
//   「同进同出 + 按最长 prompt 补齐」语义直接算，不经过上面的 allocator。
// ============================================================================

class EngineSim {
  constructor({ blockSize, totalCount, pausedCount, maxRequests, maxTokens, prefixCaching, policy, chunked }) {
    this.blockSize = blockSize;
    this.maxRequests = maxRequests;
    this.maxTokens = maxTokens;
    this.prefixCaching = prefixCaching;
    this.chunked = chunked;
    this.prefixCacheLruClock = 0;
    this.kv = new KVBlockAllocator(this, totalCount, pausedCount, prefixCaching, policy);
    this.active = []; // 顺序即 bookkeeping 顺序；chunked 请求永远在末尾
    this.paused = []; // LIFO 恢复
    this.waiting = [];
    this.chunkedPrefillRequestId = null;
    this.prefixCoordinationWaits = 0;
    this.prefixCacheHits = 0;
    this.prefixCacheBlocksMatched = 0;
    this.prefillTokensSkipped = 0;
    this.step = 0;
    this.finished = [];
    this.trace = []; // 每步快照
    this.peakBlocksUsed = 0;
  }

  activeRows() {
    return this.active;
  }
  pausedRows() {
    return this.paused;
  }
  get totalRequestCount() {
    return this.active.length + this.paused.length;
  }
  get activeTokenCount() {
    return this.active.reduce((s, r) => s + r.queryLength, 0);
  }

  submit(spec, idSuffix = '') {
    const req = {
      id: spec.id + idSuffix,
      base: spec.id,
      prompt: spec.prompt,
      gen: spec.gen,
      remaining: spec.prompt.slice(),
      finishedChunkTokenCount: 0,
      hashes: this.prefixCaching ? computeBlockHashes(spec.prompt, this.blockSize) : [],
      blockTable: [],
      generated: 0,
      queryLength: 0,
      lastOffset: 0,
      kvLen: 0,
      admittedStep: null,
      firstTokenStep: null,
      finishedStep: null,
      events: [],
    };
    this.waiting.push(req);
    return req;
  }

  // ---- DynamicInferenceContext._compute_prefix_match ----
  computePrefixMatch(req, chunkLen) {
    const bs = this.blockSize;
    const finished = req.finishedChunkTokenCount;
    const already = Math.ceil(finished / bs);
    const overall = Math.ceil((finished + chunkLen) / bs);
    if (!this.prefixCaching) {
      return { matched: [], fromPool: Math.max(0, overall - already), already, overall, skip: 0, effective: chunkLen };
    }
    const matched = this.findKvMatch(req, already, overall);
    const n = matched.length;
    const aligned = finished % bs === 0;
    let skip = n > 0 && aligned ? Math.min(n * bs, chunkLen - 1) : 0;
    if (chunkLen - skip < 2 && chunkLen >= 2) {
      const maxSkip = chunkLen - 2;
      skip = Math.floor(maxSkip / bs) * bs;
    }
    return { matched, fromPool: Math.max(0, overall - already - n), already, overall, skip, effective: chunkLen - skip };
  }

  // ---- DynamicInferenceContext._find_kv_match_count ----
  findKvMatch(req, start, endIn) {
    if (!this.prefixCaching || req.hashes.length === 0) return [];
    const end = Math.min(endIn, req.hashes.length);
    if (start >= end) return [];
    const hashes = req.hashes.slice(start, end);
    for (let i = hashes.length - 1; i >= 0; i -= 1) {
      if (this.kv.kvHashToBlockId.has(hashes[i])) {
        return hashes.slice(0, i + 1).map((h) => this.kv.kvHashToBlockId.get(h));
      }
    }
    return [];
  }

  // ---- DynamicInferenceContext.check_availability ----
  checkAvailability(req) {
    const requestCanBeAdded = this.totalRequestCount < this.maxRequests && this.paused.length === 0;
    const m = this.computePrefixMatch(req, req.remaining.length);
    const tokensCanBeAdded = this.activeTokenCount + m.effective <= this.maxTokens;
    const kvAvailable = this.kv.isMemoryAvailable(m.fromPool);
    return [requestCanBeAdded, tokensCanBeAdded, kvAvailable];
  }

  // ---- DynamicInferenceContext.add_request ----
  addRequest(req, chunkLenIn = null) {
    const bs = this.blockSize;
    const chunkLen = chunkLenIn ?? req.remaining.length;
    if (!(chunkLen > 0 && chunkLen <= req.remaining.length)) throw new Error('chunk 长度非法');
    const m = this.computePrefixMatch(req, chunkLen);
    if (m.matched.length > 0) {
      this.prefixCacheHits += 1;
      this.prefixCacheBlocksMatched += m.matched.length;
    }
    let newIds = [];
    if (m.fromPool > 0) {
      newIds = this.kv.allocateMemoryBlocks(m.fromPool);
      if (newIds === null) throw new Error(`BlockOverflowError(${req.id})`);
    }
    if (m.matched.length > 0) {
      for (const b of m.matched) this.kv.blockRefCounts[b] += 1;
      if (this.kv.policy === 'lru') this.kv.updateTimestamps(m.matched);
    }
    if (this.totalRequestCount >= this.maxRequests && this.chunkedPrefillRequestId !== req.id) {
      throw new Error(`RequestOverflowError(${req.id})`);
    }
    if (this.activeTokenCount + m.effective > this.maxTokens) throw new Error(`TokenOverflowError(${req.id})`);
    // 块表：matched 放在 [already, already+n)，新块接在其后
    for (let i = 0; i < m.matched.length; i += 1) req.blockTable[m.already + i] = m.matched[i];
    for (let i = 0; i < newIds.length; i += 1) req.blockTable[m.already + m.matched.length + i] = newIds[i];
    req.blockTable.length = m.overall;
    req.queryLength = m.effective;
    req.lastOffset = (chunkLen + req.finishedChunkTokenCount - 1) % bs;
    req.kvLen = req.finishedChunkTokenCount + chunkLen;
    this.prefillTokensSkipped += m.skip;
    // 登记本 chunk 补满的完整块的 hash（跳过 matched 块）
    if (this.prefixCaching && req.hashes.length) {
      const totalAfter = req.finishedChunkTokenCount + chunkLen;
      const numComplete = Math.floor(totalAfter / bs);
      const prevComplete = Math.floor(req.finishedChunkTokenCount / bs);
      const reg = (s, e) => {
        if (s >= e) return;
        this.kv.registerKvBlockHashes(req.blockTable.slice(s, e), req.hashes.slice(s, e));
      };
      reg(prevComplete, Math.min(m.already, numComplete));
      reg(m.already + m.matched.length, numComplete);
    }
    if (!this.active.includes(req)) this.active.push(req);
    if (req.admittedStep === null) req.admittedStep = this.step;
    req.events.push({ step: this.step, kind: chunkLenIn === null ? 'prefill' : 'chunk', tokens: m.effective, skip: m.skip });
    return m;
  }

  // ---- dynamic_engine.py::schedule_non_chunked_prefill ----
  scheduleNonChunked() {
    const pendingHashes = new Set();
    const pendingIds = [];
    while (this.waiting.length) {
      const req = this.waiting[0];
      if (this.prefixCaching && req.hashes.some((h) => pendingHashes.has(h))) {
        this.prefixCoordinationWaits += 1;
        pendingIds.push(this.waiting.shift());
        req.events.push({ step: this.step, kind: 'wait' });
        continue;
      }
      const [a, b, c] = this.checkAvailability(req);
      if (a && b && c) {
        if (this.prefixCaching) {
          for (const h of req.hashes) if (!this.kv.kvHashToBlockId.has(h)) pendingHashes.add(h);
        }
        this.addRequest(req);
        req.remaining = [];
        this.waiting.shift();
      } else {
        break;
      }
    }
    if (this.prefixCaching && pendingIds.length) this.waiting.unshift(...pendingIds);
  }

  // ---- dynamic_engine.py::schedule_chunked_prefill ----
  scheduleChunked() {
    const pendingHashes = new Set();
    const pendingIds = [];
    let canSchedule = true;
    while (this.waiting.length && canSchedule) {
      canSchedule = false;
      const req = this.waiting[0];
      const continuing = this.chunkedPrefillRequestId !== null;
      if (this.prefixCaching && !continuing && req.hashes.some((h) => pendingHashes.has(h))) {
        this.prefixCoordinationWaits += 1;
        pendingIds.push(this.waiting.shift());
        req.events.push({ step: this.step, kind: 'wait' });
        canSchedule = true;
        continue;
      }
      const remainingLen = req.remaining.length;
      const fully = this.activeTokenCount + remainingLen <= this.maxTokens;
      const partially = this.activeTokenCount < this.maxTokens;
      let [requestCanBeAdded, , kvAvailable] = this.checkAvailability(req);
      requestCanBeAdded = continuing || requestCanBeAdded;
      if (requestCanBeAdded && kvAvailable) {
        if (fully) {
          if (this.prefixCaching) {
            for (const h of req.hashes) if (!this.kv.kvHashToBlockId.has(h)) pendingHashes.add(h);
          }
          this.chunkedPrefillRequestId = null;
          this.addRequest(req);
          req.remaining = [];
          this.waiting.shift();
          canSchedule = true;
        } else if (partially) {
          if (this.prefixCaching) {
            for (const h of req.hashes) if (!this.kv.kvHashToBlockId.has(h)) pendingHashes.add(h);
          }
          let chunk = this.maxTokens - this.activeTokenCount;
          // 不给最后一个 chunk 留下正好 1 个 token（flash-attention issue 1537）
          if (remainingLen - chunk === 1) {
            if (chunk > 1) chunk -= 1;
            else break;
          }
          this.addRequest(req, chunk);
          this.chunkedPrefillRequestId = req.id;
          req.remaining = req.remaining.slice(chunk);
          req.finishedChunkTokenCount += chunk;
          // 还有 token 未 prefill：chunked 请求留在队头，本步不再继续扫描
        }
      }
    }
    if (this.prefixCaching && pendingIds.length) {
      if (this.chunkedPrefillRequestId !== null) {
        const head = this.waiting.shift();
        this.waiting.unshift(...pendingIds);
        this.waiting.unshift(head);
      } else {
        this.waiting.unshift(...pendingIds);
      }
    }
  }

  // ---- 一步：schedule → forward → update_requests（简化） ----
  runStep() {
    this.step += 1;
    if (this.chunked) this.scheduleChunked();
    else this.scheduleNonChunked();
    const stepTokens = this.activeTokenCount;
    const snapshot = {
      step: this.step,
      active: this.active.map((r) => {
        const last = r.events[r.events.length - 1];
        let kind = last.step === this.step ? last.kind : 'decode';
        // 最后一个 chunk 走的是「整段加入」分支，但对读者它仍是一个 chunk
        if (kind === 'prefill' && r.finishedChunkTokenCount > 0) kind = 'chunk';
        return { id: r.id, kind, tokens: r.queryLength };
      }),
      stepTokens,
    };
    // forward：每条非 chunked-continuing 的活跃请求采样一个 token
    const finishedNow = [];
    for (const r of this.active) {
      const isChunkedContinuing = this.chunkedPrefillRequestId === r.id;
      if (isChunkedContinuing) continue;
      r.generated += 1;
      if (r.firstTokenStep === null) r.firstTokenStep = this.step;
      if (r.generated >= r.gen) finishedNow.push(r);
    }
    // update_requests 步骤 4：完成即释放
    for (const r of finishedNow) {
      this.kv.releaseMemoryBlocks(r.blockTable.filter((b) => b !== undefined));
      r.finishedStep = this.step;
      r.events.push({ step: this.step, kind: 'finish' });
      this.finished.push(r);
    }
    this.active = this.active.filter((r) => !finishedNow.includes(r));
    // 步骤 5：末块已满的活跃请求需要新块 → 先暂停
    const newlyPaused = [];
    for (const r of this.active) {
      if (this.chunkedPrefillRequestId === r.id) continue;
      if (r.lastOffset >= this.blockSize - 1) newlyPaused.push(r);
    }
    this.active = this.active.filter((r) => !newlyPaused.includes(r));
    this.paused.push(...newlyPaused);
    // 步骤 6：resume_paused_requests —— LIFO 恢复。能恢复几条由三条上限共同决定：
    //   (a) 从右往左累加「该请求已持有的块数 + 是否需要新块」，累加和 ≤ active_avail；
    //   (b) 恢复条数 ≤ total_avail（新块只从 free pool 取，源码对此 assert，不走驱逐）；
    //   (c) 恢复后活跃数 ≤ min(max_requests, max_tokens // (spec+1))。
    // 本算例 spec=0，被暂停的请求恰好都是「末块已满」的，所以每条恢复都要 1 个新块。
    {
      const activeAvail = this.kv.getActiveAvail();
      let cum = 0;
      let resumable = 0;
      for (let i = this.paused.length - 1; i >= 0; i -= 1) {
        cum += this.paused[i].blockTable.length + 1;
        if (cum > activeAvail) break;
        resumable += 1;
      }
      resumable = Math.min(resumable, this.kv.totalAvail);
      const maxAllowedActive = Math.min(this.maxRequests, this.maxTokens);
      const activeNow = this.active.filter((r) => this.chunkedPrefillRequestId !== r.id).length;
      resumable = Math.min(resumable, Math.max(0, maxAllowedActive - activeNow));
      for (let k = 0; k < resumable; k += 1) {
        const r = this.paused.pop();
        const ids = this.kv.allocateMemoryBlocks(1);
        if (ids === null) throw new Error('assert num_new_blocks <= total_avail 失败');
        r.blockTable.push(ids[0]);
        r.events.push({ step: this.step, kind: 'grow', block: ids[0] });
        this.active.unshift(r);
      }
    }
    if (this.kv.getPausedUsed() > this.kv.pausedCount) {
      throw new Error('本算例不复刻 evict_overflow_paused_requests，但 paused 用量超过了 paused_count');
    }
    // 步骤 7：offset 前进、decode 的 query length 变为 1。
    // chunked 请求在两个 chunk 之间被「藏」在 total_request_count 之外：
    // 它的 token 不计入 active_token_count（源码把 active_token_count 重置为
    // active_request_count × 1，而 active_request_count 已把它减掉）。
    for (const r of this.active) {
      if (this.chunkedPrefillRequestId === r.id) {
        r.queryLength = 0;
        continue;
      }
      r.lastOffset = (r.lastOffset + 1) % this.blockSize;
      r.kvLen += 1;
      r.queryLength = 1;
    }
    for (const r of this.paused) {
      r.lastOffset = (r.lastOffset + 1) % this.blockSize;
      r.kvLen += 1;
      r.queryLength = 1;
    }
    this.prefixCacheLruClock += 1;
    this.peakBlocksUsed = Math.max(this.peakBlocksUsed, this.kv.getTotalUsed());
    snapshot.finished = finishedNow.map((r) => r.id);
    snapshot.paused = this.paused.map((r) => r.id);
    snapshot.blocks = this.blockOwners();
    snapshot.totalAvail = this.kv.totalAvail;
    this.trace.push(snapshot);
    return snapshot;
  }

  blockOwners() {
    const owners = Array.from({ length: this.kv.totalCount }, () => ({ owners: [], cached: false }));
    for (const r of [...this.active, ...this.paused]) {
      for (const b of r.blockTable) if (b !== undefined) owners[b].owners.push(r.base);
    }
    if (this.prefixCaching) {
      for (let b = 0; b < this.kv.totalCount; b += 1) {
        if (this.kv.blockRefCounts[b] === 0 && this.kv.blockHashes[b] !== -1) owners[b].cached = true;
      }
    }
    return owners;
  }

  hasUnfinished() {
    return this.active.length + this.paused.length + this.waiting.length > 0;
  }

  runUntilIdle(limit = 64) {
    while (this.hasUnfinished()) {
      if (this.step >= limit) throw new Error('仿真没有收敛');
      this.runStep();
    }
  }
}

// ============================================================================
// 图 1 的算例：两波请求，ref_zero 与 lru 各跑一遍
// ============================================================================

function runFig1(policy) {
  const sim = new EngineSim({
    blockSize: CFG.blockSize,
    totalCount: CFG.totalCount,
    pausedCount: CFG.pausedCount,
    maxRequests: CFG.fig1MaxRequests,
    maxTokens: Infinity,
    prefixCaching: true,
    policy,
    chunked: false,
  });
  for (const r of CFG.requests) sim.submit(r, '');
  sim.runUntilIdle();
  const wave1 = {
    steps: sim.step,
    hits: sim.prefixCacheHits,
    blocksMatched: sim.prefixCacheBlocksMatched,
    tokensSkipped: sim.prefillTokensSkipped,
    waits: sim.prefixCoordinationWaits,
    evictions: sim.kv.evictions,
    peak: sim.peakBlocksUsed,
    cachedAfter: sim.blockOwners().filter((o) => o.cached).length,
  };
  const wave2Start = sim.step;
  const before = { hits: sim.prefixCacheHits, blocks: sim.prefixCacheBlocksMatched, skipped: sim.prefillTokensSkipped, waits: sim.prefixCoordinationWaits, ev: sim.kv.evictions };
  for (const r of CFG.requests) sim.submit(r, "'");
  sim.runUntilIdle();
  const wave2 = {
    steps: sim.step - wave2Start,
    hits: sim.prefixCacheHits - before.hits,
    blocksMatched: sim.prefixCacheBlocksMatched - before.blocks,
    tokensSkipped: sim.prefillTokensSkipped - before.skipped,
    waits: sim.prefixCoordinationWaits - before.waits,
    evictions: sim.kv.evictions - before.ev,
  };
  return { sim, wave1, wave2, wave2Start };
}

const FIG1_RZ = runFig1('ref_zero');
const FIG1_LRU = runFig1('lru');
const PROMPT_TOKENS_PER_WAVE = CFG.requests.reduce((s, r) => s + r.prompt.length, 0);

// 朴素定长预留的对照：每条请求按 max_sequence_length 预留
const NAIVE = (() => {
  const perReq = Math.ceil(CFG.maxSequenceLength / CFG.blockSize);
  const reserved = perReq * CFG.requests.length;
  const usable = CFG.totalCount - 1;
  const concurrent = Math.floor(usable / perReq);
  // 实际 KV 落盘 token 数：prompt + gen − 1（最后一个采样 token 不写 KV）
  const usedBlocks = CFG.requests.reduce((s, r) => s + Math.ceil((r.prompt.length + r.gen - 1) / CFG.blockSize), 0);
  return { perReq, reserved, usable, concurrent, usedBlocks, wastePct: Math.round((1 - usedBlocks / reserved) * 100) };
})();

// 不变量
const INVARIANT = Object.freeze({
  totalCount: CFG.totalCount,
  totalAvail: CFG.totalCount - 1,
  dummyBlockIdx: CFG.totalCount - 1,
  pausedCount: CFG.pausedCount,
  activeCount: CFG.totalCount - CFG.pausedCount - 1,
});

// ============================================================================
// 图 2 的算例：三条路径
// ============================================================================

function runStaticBatch() {
  // legacy static engine（text_generation_controller.py::generate_all_output_tokens_static_batch）：
  //   按 max_batch_size 取批，整批共用一份 sampling_params；第一步只 prefill 到批内「最短」prompt，
  //   之后每步对整批每行推进 1 个位置——prompt 更长的请求在这些步里逐 token「走」完自己的 prompt，
  //   已经生成完的请求的那一行仍然被计算（is_generation_done 只用来忽略它的输出）；
  //   全部完成（early termination）或 context_end_position 达到 max_prompt + num_tokens_to_generate 时停。
  // 简化：算例里各请求生成长度不同，而整批只有一份 num_tokens_to_generate；这里按
  //   「设成批内最大生成长度、各请求在自己的长度处以 EOD 终止」处理，不改变每条请求的语义。
  const slots = CFG.fig2MaxRequests;
  const queue = CFG.requests.slice();
  const columns = [];
  let step = 0;
  let idle = 0; // 空槽 + 已完成但仍被计算的行
  let walked = 0; // prompt 被逐 token 走的槽·步
  let maxStepTokens = 0;
  let r3Admitted = null;
  while (queue.length) {
    const batch = queue.splice(0, slots);
    const minPrompt = Math.min(...batch.map((r) => r.prompt.length));
    const maxPrompt = Math.max(...batch.map((r) => r.prompt.length));
    const genBatch = Math.max(...batch.map((r) => r.gen));
    const maxSeq = maxPrompt + genBatch;
    const generated = new Map(batch.map((r) => [r.id, 0]));
    let contextEnd = 0;
    for (;;) {
      step += 1;
      const contextStart = contextEnd;
      contextEnd = contextStart === 0 ? minPrompt : contextStart + 1;
      const positions = contextEnd - contextStart;
      const col = { step, cells: [], stepTokens: 0 };
      for (let s = 0; s < slots; s += 1) {
        const r = batch[s];
        if (!r) {
          col.cells.push(null);
          idle += 1;
          continue;
        }
        col.stepTokens += positions; // 整批每行都被计算
        if (generated.get(r.id) >= r.gen) {
          col.cells.push({ id: r.id, kind: 'done', tokens: positions });
          idle += 1;
          continue;
        }
        if (contextStart === 0) {
          col.cells.push({ id: r.id, kind: 'prefill', tokens: positions });
          if (r.id === 'R3') r3Admitted = step;
        } else if (r.prompt.length > contextEnd) {
          col.cells.push({ id: r.id, kind: 'walk', tokens: 1 });
          walked += 1;
        } else {
          col.cells.push({ id: r.id, kind: 'decode', tokens: 1 });
        }
        // generation_started = prompt_length <= context_end_position
        if (r.prompt.length <= contextEnd) generated.set(r.id, generated.get(r.id) + 1);
      }
      maxStepTokens = Math.max(maxStepTokens, col.stepTokens);
      columns.push(col);
      const allDone = batch.every((r) => generated.get(r.id) >= r.gen);
      if (allDone || contextEnd + 1 > maxSeq) break;
    }
  }
  return { name: 'static', columns, steps: step, idle, walked, slotSteps: step * slots, maxStepTokens, r3Admitted };
}

function runDynamic(chunked) {
  const sim = new EngineSim({
    blockSize: CFG.blockSize,
    totalCount: CFG.fig2TotalCount,
    pausedCount: 0,
    maxRequests: CFG.fig2MaxRequests,
    maxTokens: CFG.fig2MaxTokens,
    prefixCaching: false,
    policy: 'ref_zero',
    chunked,
  });
  for (const r of CFG.requests) sim.submit(r, '');
  sim.runUntilIdle();
  const slots = CFG.fig2MaxRequests;
  const columns = [];
  let idle = 0;
  let maxStepTokens = 0;
  // 槽位分配：一个请求从进入到完成占同一个槽
  const slotOf = new Map();
  const freeAt = new Array(slots).fill(0);
  for (const snap of sim.trace) {
    const col = { step: snap.step, cells: new Array(slots).fill(null), stepTokens: snap.stepTokens };
    for (const a of snap.active) {
      if (!slotOf.has(a.id)) {
        const s = freeAt.findIndex((v) => v === 0);
        if (s < 0) throw new Error('槽位不够');
        slotOf.set(a.id, s);
        freeAt[s] = 1;
      }
      col.cells[slotOf.get(a.id)] = a;
    }
    for (const f of snap.finished) freeAt[slotOf.get(f)] = 0;
    idle += col.cells.filter((c) => c === null).length;
    maxStepTokens = Math.max(maxStepTokens, snap.stepTokens);
    columns.push(col);
  }
  const r3 = sim.finished.find((r) => r.base === 'R3');
  return { name: chunked ? 'chunked' : 'continuous', columns, steps: sim.step, idle, slotSteps: sim.step * slots, maxStepTokens, r3Admitted: r3.admittedStep, sim };
}

const FIG2 = Object.freeze({
  static: runStaticBatch(),
  continuous: runDynamic(false),
  chunked: runDynamic(true),
});

// ============================================================================
// 复刻 batch_dimensions_utils.py::CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts
// ============================================================================

const roundUpTo = (v, m) => Math.ceil(v / m) * m;

function calculateTokenCountsLinear(tpSize, numCudaGraphs, maxTokensIn) {
  const rounder = 2;
  if (numCudaGraphs === -1) {
    let sizes = [1, 2, 4];
    for (let s = 8; s < 256; s += 8) sizes.push(s);
    for (let s = 256; s <= maxTokensIn; s += 16) sizes.push(s);
    sizes = [...new Set(sizes.map((s) => roundUpTo(s, tpSize)))];
    sizes = sizes.filter((s) => s <= maxTokensIn);
    if (!sizes.length || sizes[sizes.length - 1] !== maxTokensIn) sizes.push(maxTokensIn);
    sizes.reverse();
    return sizes;
  }
  if (!(numCudaGraphs >= 1)) throw new Error('num_cuda_graphs must be >= 1');
  let step = maxTokensIn / numCudaGraphs;
  step = rounder * Math.ceil(Math.trunc(step) / rounder);
  step = roundUpTo(step, tpSize);
  step = Math.max(step, tpSize);
  const maxTokens = Math.floor(maxTokensIn / tpSize) * tpSize;
  if (numCudaGraphs === 1) return [maxTokens];
  const sizes = [];
  for (let s = step; s < maxTokens; s += step) sizes.push(s);
  if (!sizes.length || sizes[sizes.length - 1] !== maxTokens) sizes.push(maxTokens);
  sizes.reverse();
  return sizes;
}

function calculateCudaGraphTokenCounts(tpSize, numCudaGraphsIn, maxTokensIn, distribution) {
  if (distribution === 'linear') return calculateTokenCountsLinear(tpSize, numCudaGraphsIn, maxTokensIn);
  let numCudaGraphs = numCudaGraphsIn;
  if (numCudaGraphs === -1) {
    const HEADROOM = 2;
    const MIN_GRAPHS = 4;
    const numHalvings = Math.floor(Math.log2(Math.max(2, maxTokensIn)));
    numCudaGraphs = Math.max(MIN_GRAPHS, numHalvings + HEADROOM);
  }
  if (!(numCudaGraphs >= 1)) throw new Error('num_cuda_graphs must be >= 1');
  const rounder = 2;
  const maxTokens = Math.floor(maxTokensIn / tpSize) * tpSize;
  if (numCudaGraphs === 1) return [maxTokens];
  const sizes = new Set();
  let val = maxTokens;
  for (let i = 0; i < numCudaGraphs; i += 1) {
    let rounded = Math.max(rounder, Math.floor(val / rounder) * rounder);
    rounded = Math.ceil(rounded / tpSize) * tpSize;
    sizes.add(rounded);
    val = Math.floor(val / 2);
    if (val < 1) break;
  }
  sizes.add(maxTokens);
  sizes.add(tpSize);
  const counts = [...sizes].sort((a, b) => b - a);
  while (counts.length > numCudaGraphs) counts.splice(counts.length - 2, 1);
  if (!counts.includes(maxTokens)) throw new Error('assert cuda_graph_max_tokens in counts 失败');
  return counts;
}

const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
const lcm = (a, b) => (a / gcd(a, b)) * b;

// decode-only 图的 token_count（generate_cuda_graph_batch_dimensions_list 的 decode 分支）。
// num_cuda_graphs == -1 时源码还会把 lcm(spec+1, tp) × {1, 2} 两个最小尺寸补进去，
// 免得 TP 对齐与 (spec+1) 整除把 1、2 条请求的图冲掉。
function decodeGraphTokenCounts(sizesIn, tpSize, spec, maxRequests, autoSized = false) {
  let sizes = [...sizesIn];
  if (autoSized) {
    const minDecode = lcm(spec + 1, tpSize);
    const maxDecode = Math.min(Math.max(...sizes), maxRequests * (spec + 1));
    for (const m of [1, 2]) {
      const floor = minDecode * m;
      if (floor <= maxDecode && !sizes.includes(floor)) sizes.push(floor);
    }
    sizes = [...new Set(sizes)].sort((a, b) => b - a);
  }
  const out = new Set();
  for (const size of sizes) {
    const decodeReq = Math.min(Math.floor(size / (spec + 1)), maxRequests);
    const tokenCount = Math.floor((decodeReq * (spec + 1)) / tpSize) * tpSize;
    if (tokenCount > 0 && decodeReq > 0) out.add(tokenCount);
  }
  return [...out].sort((a, b) => b - a);
}

// match_graph_config：最小的「够用」图
function paddingProfile(graphSizes, maxTokens) {
  const asc = [...graphSizes].sort((a, b) => a - b);
  const rows = [];
  let worst = { n: 0, rel: -1, pad: 0 };
  let relSum = 0;
  for (let n = 1; n <= maxTokens; n += 1) {
    const chosen = asc.find((s) => s >= n);
    if (chosen === undefined) throw new Error(`n=${n} 没有可用图`);
    const pad = chosen - n;
    const rel = pad / n;
    relSum += rel;
    if (rel > worst.rel) worst = { n, rel, pad, chosen };
    rows.push({ n, chosen, pad });
  }
  return { rows, worst, meanRelPct: Math.round((relSum / maxTokens) * 100), worstRelPct: Math.round(worst.rel * 100) };
}

function cudaGraphCase(distribution, numCudaGraphs, maxTokens = CFG.cudaGraphMaxTokens) {
  const raw = calculateCudaGraphTokenCounts(CFG.tpSize, numCudaGraphs, maxTokens, distribution);
  const maxRequests = maxTokens / (CFG.numSpeculativeTokens + 1);
  const sizes = decodeGraphTokenCounts(raw, CFG.tpSize, CFG.numSpeculativeTokens, maxRequests, numCudaGraphs === -1);
  const profile = paddingProfile(sizes, maxTokens);
  return { distribution, numCudaGraphs, maxTokens, raw, sizes, count: sizes.length, ...profile };
}

const FIG3 = Object.freeze({
  expAuto: cudaGraphCase('exponential', -1),
  linAuto: cudaGraphCase('linear', -1),
  expDefault: cudaGraphCase('exponential', CFG.defaultNumCudaGraphs),
  linDefault: cudaGraphCase('linear', CFG.defaultNumCudaGraphs),
  // 规模放大后的对照：同样的默认 num_cuda_graphs=16，把 decode 上界推到 1024
  expLarge: cudaGraphCase('exponential', CFG.defaultNumCudaGraphs, CFG.largeCudaGraphMaxTokens),
  linLarge: cudaGraphCase('linear', CFG.defaultNumCudaGraphs, CFG.largeCudaGraphMaxTokens),
});

// 图 3 的立论：指数分布的最坏相对 padding 在两个规模上都有界（< 100%），线性分布
// 在规模放大后最坏相对 padding 随步长线性恶化。
if (FIG3.expDefault.worstRelPct > 100 || FIG3.expLarge.worstRelPct > 100) {
  throw new Error('图 3 的立论前提被推翻：EXPONENTIAL 的最坏相对 padding 应 ≤ 100%（docstring 的 ~2x）');
}
if (FIG3.linLarge.worstRelPct <= FIG3.linDefault.worstRelPct) {
  throw new Error('图 3 的立论前提被推翻：LINEAR 的最坏相对 padding 应随规模恶化');
}

if (process.env.INFERENCE_FIG_DEBUG) {
  const dump = (tag, r) => {
    console.log(tag, JSON.stringify({ wave1: r.wave1, wave2: r.wave2, start2: r.wave2Start, steps: r.sim.step }));
    for (const s of r.sim.trace) {
      console.log(`  s${s.step} tok=${s.stepTokens} active=${s.active.map((a) => `${a.id}:${a.kind}${a.tokens}`).join(',')} fin=${s.finished.join(',')} paused=${s.paused.join(',')} avail=${s.totalAvail}`);
      console.log('     blocks=' + s.blocks.map((b, i) => `${i}:${b.owners.join('/') || (b.cached ? 'c' : '-')}`).join(' '));
    }
  };
  dump('RZ', FIG1_RZ);
  dump('LRU', FIG1_LRU);
  console.log('NAIVE', JSON.stringify(NAIVE), 'INV', JSON.stringify(INVARIANT), 'promptTokens', PROMPT_TOKENS_PER_WAVE);
  for (const k of ['static', 'continuous', 'chunked']) {
    const r = FIG2[k];
    console.log(`FIG2 ${k}: steps=${r.steps} idle=${r.idle}/${r.slotSteps} maxTok=${r.maxStepTokens} r3=${r.r3Admitted}`);
    for (const c of r.columns) console.log(`  s${c.step} tok=${c.stepTokens} ${c.cells.map((x) => (x ? `${x.id}:${x.kind}${x.tokens}` : '·')).join(' | ')}`);
  }
  for (const k of Object.keys(FIG3)) {
    const c = FIG3[k];
    console.log(`FIG3 ${k}: max=${c.maxTokens} sizes=[${c.sizes.join(',')}] count=${c.count} worst=${c.worstRelPct}%@n=${c.worst.n}->${c.worst.chosen} mean=${c.meanRelPct}%`);
  }
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_refit_figures.mjs 同一套 token）
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
  .s0{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.1}
  .s1{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.1}
  .s2{fill:#EAF7EE;stroke:#2E8B57;stroke-width:1.1}
  .s3{fill:#F3EEFB;stroke:#6D4AAF;stroke-width:1.1}
  .shared{fill:#EAF1FD;stroke:#2563EB;stroke-width:2.2}
  .bar{fill:#F4C9A3;stroke:#C3651F;stroke-width:.8}
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
  .sep{fill:none;stroke:#5B6470;stroke-width:1.6}
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

const REQ_CLASS = { R0: 's0', R1: 's1', R2: 's2', R3: 's3' };

// ============================================================================
// 图 1：块级 KV cache
// ============================================================================

function blockGrid(parts, { x0, y0, sim, title, where }) {
  const COL = 64;
  const ROW = 24;
  const steps = sim.trace;
  parts.push(text(x0, y0 - 8, title, 'pt'));
  // 表头：step
  steps.forEach((s, i) => {
    parts.push(text(x0 + 96 + i * COL + COL / 2, y0 + 12, `s${s.step}`, 'rank', 'middle'));
  });
  const gridTop = y0 + 20;
  for (let b = 0; b < sim.kv.totalCount; b += 1) {
    const y = gridTop + b * ROW;
    const isDummy = b === sim.kv.dummyBlockIdx;
    parts.push(text(x0 + 88, y + 16, isDummy ? `块 ${b}·dummy` : `块 ${b}`, 'sm', 'end'));
    steps.forEach((s, i) => {
      const x = x0 + 96 + i * COL;
      const info = s.blocks[b];
      if (isDummy) {
        parts.push(rect(x + 2, y + 2, COL - 4, ROW - 4, 'ghost', 3));
        if (i === 0) parts.push(text(x + COL / 2, y + 16, '不分配', 'sm', 'middle'));
        return;
      }
      if (info.owners.length === 0 && !info.cached) {
        parts.push(rect(x + 2, y + 2, COL - 4, ROW - 4, 'ghost', 3));
        return;
      }
      if (info.owners.length === 0 && info.cached) {
        parts.push(rect(x + 2, y + 2, COL - 4, ROW - 4, 'ghost', 3));
        parts.push(text(x + COL / 2, y + 16, 'cached', 'sm', 'middle'));
        return;
      }
      const uniq = [...new Set(info.owners)];
      const shared = info.owners.length > 1;
      parts.push(rect(x + 2, y + 2, COL - 4, ROW - 4, shared ? 'shared' : REQ_CLASS[uniq[0]], 3));
      const label = shared ? `ref ${info.owners.length}` : uniq[0];
      parts.push(text(x + COL / 2, y + 16, guard(label, 10.5, COL - 6, `${where}/cell`), shared ? 'dim' : 'sm', 'middle'));
    });
  }
  // 事件行
  const evY = gridTop + sim.kv.totalCount * ROW + 16;
  parts.push(text(x0 + 88, evY, '事件', 'sm', 'end'));
  const allReqs = [...sim.finished];
  steps.forEach((s, i) => {
    const x = x0 + 96 + i * COL;
    const ins = [];
    const waits = [];
    for (const r of allReqs) {
      for (const e of r.events) {
        if (e.step !== s.step) continue;
        if (e.kind === 'prefill') ins.push(r.base);
        if (e.kind === 'wait') waits.push(r.base);
      }
    }
    const outs = s.finished.map((id) => id.replace("'", ''));
    const num = (ids) => ids.map((id) => id.replace('R', '')).sort().join(' ');
    const lines = [];
    if (ins.length) lines.push(`+${num(ins)}`);
    if (waits.length) lines.push(`~${num(waits)}`);
    if (outs.length) lines.push(`−${num(outs)}`);
    lines.forEach((l, k) => {
      parts.push(text(x + COL / 2, evY + k * 14, guard(l, 10.5, COL - 4, `${where}/ev`), k === 1 ? 'costtx' : 'sm', 'middle'));
    });
  });
  return evY + 3 * 14;
}

function renderKvBlocks() {
  const W = 1272;
  const X0 = 28;
  const parts = header(
    W,
    '图 1　块级 KV cache：块按需分配、共享按 ref 计数、没分出去的块就是浪费',
    `算例：block_size_tokens=${CFG.blockSize}，池 total_count=${CFG.totalCount}；R0(6→3) R1(10→2) R2(3→5) R3(12→2) 先后提交两波，R0/R1/R3 共享前 4 个 token，R1/R3 再共享 4 个`,
  );
  const rz = FIG1_RZ.sim;
  const lru = FIG1_LRU.sim;
  const bottom1 = blockGrid(parts, {
    x0: X0,
    y0: 96,
    sim: rz,
    title: `ref_zero（默认）：ref 归零立即回池，第二波从 s${FIG1_RZ.wave2Start + 1} 起`,
    where: 'fig1/rz',
  });
  const bottom2 = blockGrid(parts, {
    x0: X0,
    y0: bottom1 + 24,
    sim: lru,
    title: `lru：ref 归零后留在 hash 表里当 cached，第二波从 s${FIG1_LRU.wave2Start + 1} 起`,
    where: 'fig1/lru',
  });
  parts.push(text(X0, bottom2 + 8, '事件行只写请求编号：+ 进入 context　~ 前缀正由同批请求计算、推迟一步　− 完成并释放；ref n = 被 n 条请求共享', 'cap'));
  const H = Math.max(760, bottom2 + 24);

  const RX = 792;
  const RW = 452;
  const rz1 = FIG1_RZ.wave1;
  const rz2 = FIG1_RZ.wave2;
  const lru1 = FIG1_LRU.wave1;
  const lru2 = FIG1_LRU.wave2;
  parts.push(
    infoBox(
      RX,
      96,
      RW,
      150,
      '池的不变量（KVBlockAllocator.__init__）',
      [
        `total_count=${INVARIANT.totalCount} → total_avail=${INVARIANT.totalAvail}（−1 给 dummy_block_idx=${INVARIANT.dummyBlockIdx}）`,
        `paused_count=${INVARIANT.pausedCount} → active_count=${INVARIANT.activeCount}（assert active_count ≥ 1）`,
        `块 id 是一个栈：从 block_bag 顶端弹出，第一批拿到 ${INVARIANT.totalAvail - 1}、${INVARIANT.totalAvail - 2}…`,
        '块只在 prompt 进入与末块写满时分配；末块写满 → 先 pause 再在同一步 resume',
        `本算例 4 条并发峰值只占 ${rz1.peak} 块；请求的块表可跨越不连续的物理块`,
      ],
      'neutral',
      'fig1/inv',
    ),
  );
  parts.push(
    infoBox(
      RX,
      262,
      RW,
      196,
      '两种驱逐策略：同一算例第一波 / 第二波',
      [
        `ref_zero 第一波：命中 ${rz1.hits} 条、匹配 ${rz1.blocksMatched} 块、跳过 ${rz1.tokensSkipped}/${PROMPT_TOKENS_PER_WAVE} 个 prompt token、推迟 ${rz1.waits} 次`,
        `ref_zero 第二波：命中 ${rz2.hits} 条、匹配 ${rz2.blocksMatched} 块、跳过 ${rz2.tokensSkipped} 个、推迟 ${rz2.waits} 次，与第一波一样`,
        `lru 第一波结束时 ${lru1.cachedAfter} 块以 ref=0 留作 cached，不回池`,
        { text: `lru 第二波：命中 ${lru2.hits} 条、匹配 ${lru2.blocksMatched} 块、跳过 ${lru2.tokensSkipped} 个、推迟 ${lru2.waits} 次，四条同一步进入`, cls: 'dim' },
        `lru 驱逐 ${lru1.evictions + lru2.evictions} 次：池够用就不驱逐；不够时按 timestamp 最旧先出`,
        'ref_zero 只能靠「同一 step 内已在跑的请求」共享；lru 才跨波次复用',
        '代价：lru 的 free pool 变成两级（free + evictable），分配前要数 evictable',
        '硬规则：前缀正被同批请求计算的请求推迟到下一步（pending hash）',
      ],
      'acc1',
      'fig1/policy',
    ),
  );
  parts.push(
    infoBox(
      RX,
      474,
      RW,
      150,
      '被否掉的替代：按 max_sequence_length 连续预留',
      [
        `max_sequence_length=${CFG.maxSequenceLength} → 每请求预留 ${NAIVE.perReq} 块，4 条共 ${NAIVE.reserved} 块`,
        `池只有 ${NAIVE.usable} 块可用 → 同时最多跑 ${NAIVE.concurrent} 条，其余排队`,
        `四条请求真正写入 KV 的块数合计 ${NAIVE.usedBlocks}，预留量的 ${NAIVE.wastePct}% 被浪费`,
        { text: '还要求每条请求的块物理连续，长请求会因外部碎片进不来', cls: 'costtx' },
        '块级方案的代价：块表间接寻址 + 一块永远不可用（dummy）',
      ],
      'acc2',
      'fig1/naive',
    ),
  );
  parts.push(
    infoBox(
      RX,
      640,
      RW,
      100,
      '本图复刻了什么、简化了什么',
      [
        '精确：allocate / release / evict_lru / register_kv_block_hashes',
        '精确：_compute_prefix_match / check_availability / schedule_* / resume',
        '简化：不复刻 paused buffer 溢出驱逐、Mamba、投机解码；每步 1 token',
      ],
      'neutral',
      'fig1/scope',
    ),
  );
  return seal(parts, W, H, 'fig1-kv-blocks');
}

// ============================================================================
// 图 2：连续批处理 + chunked prefill 时间线
// ============================================================================

function lanePanel(parts, { x0, y0, run, title, sub, where }) {
  const COL = 52;
  const ROW = 24;
  const slots = CFG.fig2MaxRequests;
  parts.push(text(x0, y0, title, 'pt'));
  parts.push(text(x0, y0 + 16, sub, 'sm'));
  const top = y0 + 26;
  run.columns.forEach((c, i) => {
    parts.push(text(x0 + 70 + i * COL + COL / 2, top + 12, `s${c.step}`, 'rank', 'middle'));
  });
  for (let s = 0; s < slots; s += 1) {
    const y = top + 20 + s * ROW;
    parts.push(text(x0 + 62, y + 16, `槽 ${s}`, 'sm', 'end'));
    run.columns.forEach((c, i) => {
      const x = x0 + 70 + i * COL;
      const cell = c.cells[s];
      if (!cell) {
        parts.push(rect(x + 2, y + 2, COL - 4, ROW - 4, 'ghost', 3));
        return;
      }
      const base = cell.id.replace("'", '');
      const cls =
        cell.kind === 'decode' ? 'neutral'
          : cell.kind === 'chunk' ? 'acc1'
            : cell.kind === 'walk' ? 'acc2'
              : cell.kind === 'done' ? 'ghost'
                : REQ_CLASS[base];
      parts.push(rect(x + 2, y + 2, COL - 4, ROW - 4, cls, 3));
      const label =
        cell.kind === 'decode' ? `${base}·d`
          : cell.kind === 'chunk' ? `${base}·c${cell.tokens}`
            : cell.kind === 'walk' ? `${base}·p`
              : cell.kind === 'done' ? `${base}·×`
                : `${base}·P${cell.tokens}`;
      const tcls = cell.kind === 'chunk' ? 'dim' : cell.kind === 'walk' ? 'costtx' : 'sm';
      parts.push(text(x + COL / 2, y + 16, guard(label, 10.5, COL - 6, `${where}/cell`), tcls, 'middle'));
    });
  }
  const ty = top + 20 + slots * ROW + 14;
  parts.push(text(x0 + 62, ty, 'token', 'sm', 'end'));
  run.columns.forEach((c, i) => {
    const x = x0 + 70 + i * COL;
    const hot = c.stepTokens > CFG.fig2MaxTokens;
    parts.push(text(x + COL / 2, ty, String(c.stepTokens), hot ? 'costtx' : 'sm', 'middle'));
  });
  return ty + 12;
}

function renderBatching() {
  const W = 1272;
  const H = 600;
  const X0 = 28;
  const parts = header(
    W,
    '图 2　同一批请求：定长批、连续批处理、连续批处理 + chunked prefill',
    `算例：max_requests=${CFG.fig2MaxRequests} 个槽位，max_tokens=${CFG.fig2MaxTokens}（一步的 prefill 激活预算）；P<n> 整段 prefill，c<n> 一个 chunk，d decode`,
  );
  const st = FIG2.static;
  const co = FIG2.continuous;
  const ch = FIG2.chunked;
  let y = 92;
  y = lanePanel(parts, {
    x0: X0,
    y0: y,
    run: st,
    title: '① 定长批（legacy static engine 语义）',
    sub: '同进同出：整批共用一份 sampling_params；P 整段 prefill，p 逐 token 走 prompt，d decode，× 已完成仍被计算',
    where: 'fig2/static',
  });
  y = lanePanel(parts, {
    x0: X0,
    y0: y + 22,
    run: co,
    title: '② 连续批处理（schedule_non_chunked_prefill）',
    sub: '完成即离开、等待即补入；队头 prompt 装不进 max_tokens 就整体停下',
    where: 'fig2/cont',
  });
  y = lanePanel(parts, {
    x0: X0,
    y0: y + 22,
    run: ch,
    title: '③ 连续批处理 + chunked prefill（schedule_chunked_prefill）',
    sub: '队头装不下就切一段填满预算，chunked 请求钉在队头',
    where: 'fig2/chunk',
  });

  const RX = 700;
  const RW = 544;
  const widest = Math.max(st.columns.length, co.columns.length, ch.columns.length);
  if (X0 + 70 + widest * 52 > RX - 8) {
    throw new Error(`fig2: ${widest} 列的时间线会压到右栏（${X0 + 70 + widest * 52} > ${RX - 8}）`);
  }
  const rows = [
    ['路径', '步数', '空转槽·步', '最大单步 token', 'R3 进入'],
    ['① 定长批', st.steps, `${st.idle}/${st.slotSteps}`, st.maxStepTokens, `s${st.r3Admitted}`],
    ['② 连续批处理', co.steps, `${co.idle}/${co.slotSteps}`, co.maxStepTokens, `s${co.r3Admitted}`],
    ['③ + chunked', ch.steps, `${ch.idle}/${ch.slotSteps}`, ch.maxStepTokens, `s${ch.r3Admitted}`],
  ];
  const TY = 96;
  parts.push(rect(RX, TY, RW, 24 * rows.length + 16, 'neutral'));
  const colX = [RX + 12, RX + 150, RX + 220, RX + 330, RX + 460];
  rows.forEach((r, i) => {
    const yy = TY + 22 + i * 24;
    r.forEach((v, k) => {
      parts.push(text(colX[k], yy, guard(String(v), i === 0 ? 11 : 10.5, 120, 'fig2/tbl'), i === 0 ? 'rank' : k === 0 ? 'tx' : 'sm'));
    });
    if (i === 0) parts.push(line(RX + 12, yy + 7, RX + RW - 12, yy + 7));
  });
  const minPrompt = Math.min(...CFG.requests.slice(0, CFG.fig2MaxRequests).map((r) => r.prompt.length));
  parts.push(
    infoBox(
      RX,
      TY + 24 * rows.length + 32,
      RW,
      186,
      '读法',
      [
        `① 第一步只 prefill 到批内最短 prompt（${minPrompt} token），之后每步整批推进 1 个位置：`,
        { text: `   长 prompt 被逐 token「走」完（p 格，共 ${st.walked} 槽·步），完成的行仍被计算（× 格）`, cls: 'costtx' },
        `② 的空转最多：R1(10 token) 在 s1 装不进 max_tokens，队头 break 把 R2 也拦住；`,
        `   R3(12 token) 要等到没有别的活跃请求才能独占一步（s${co.r3Admitted}）`,
        { text: `③ 把 R1/R3 的 prefill 切成 chunk 塞进预算缝隙，每步 token ≤ ${CFG.fig2MaxTokens}，R3 在 s${ch.r3Admitted} 进入`, cls: 'dim' },
        `③ 的代价：chunked 请求在最后一个 chunk 之前不出 token，且一次只允许一条`,
        '连续批处理不是自动赢：它把「批」拆成槽位与 token 两个预算，队头 FIFO 是硬规则',
        '本图不复刻 KV 块、pause、Mamba、投机解码；每条 decode 请求每步恰好 1 token',
      ],
      'neutral',
      'fig2/notes',
    ),
  );
  parts.push(
    infoBox(
      RX,
      TY + 24 * rows.length + 234,
      RW,
      102,
      '为什么 max_tokens 是一步的硬预算',
      [
        'InferenceConfig.max_tokens：「primarily limited by prefill activation memory usage」',
        '非 chunked 模式下 prompt 长于 max_tokens 的请求在 _add_request 直接 FAILED',
        { text: '（TokenOverflowError，非暂态）；chunked 模式才允许它分段进入', cls: 'costtx' },
      ],
      'acc2',
      'fig2/budget',
    ),
  );
  return seal(parts, W, H, 'fig2-batching');
}

// ============================================================================
// 图 3：CUDA graph 尺寸枚举
// ============================================================================

function ruler(parts, { x0, y, w, sizes, maxTokens, where, labelEvery = 18 }) {
  const scale = w / maxTokens;
  parts.push(line(x0, y, x0 + maxTokens * scale, y, 'sep'));
  let lastLabelX = -Infinity;
  for (const s of [...sizes].sort((a, b) => a - b)) {
    const x = x0 + s * scale;
    parts.push(line(x, y - 6, x, y + 6, 'sep'));
    if (x - lastLabelX >= labelEvery) {
      parts.push(text(x, y - 10, guard(String(s), 10.5, 40, `${where}/tick`), 'dim', 'middle'));
      lastLabelX = x;
    }
  }
  return scale;
}

function graphPanel(parts, { x0, y0, w, cs, auto, title, where }) {
  const RULER_Y = y0 + 40;
  const RW = w - 40;
  parts.push(text(x0, y0 + 14, guard(title, 14, w, `${where}/title`), 'pt'));
  const scale = ruler(parts, { x0, y: RULER_Y, w: RW, sizes: cs.sizes, maxTokens: cs.maxTokens, where });
  parts.push(text(x0, RULER_Y + 20, '真实 batch n 与它被 padding 到的图（柱高 = 多算的 token）', 'sm'));
  const BAR_TOP = RULER_Y + 30;
  const BAR_H = 110;
  const maxPad = Math.max(...cs.rows.map((r) => r.pad), 1);
  for (const r of cs.rows) {
    const x = x0 + (r.n - 1) * scale;
    const h = (r.pad / maxPad) * BAR_H;
    if (h > 0) {
      parts.push(`<rect class="bar" x="${(x + 0.5).toFixed(1)}" y="${(BAR_TOP + BAR_H - h).toFixed(1)}" width="${Math.max(1, scale - 1).toFixed(1)}" height="${h.toFixed(1)}"/>`);
    }
  }
  parts.push(line(x0, BAR_TOP + BAR_H, x0 + RW, BAR_TOP + BAR_H, 'gl'));
  for (const n of [1, 16, 32, 48, 64]) {
    parts.push(text(x0 + (n - 0.5) * scale, BAR_TOP + BAR_H + 14, `n=${n}`, 'sm', 'middle'));
  }
  parts.push(text(x0 + RW, BAR_TOP - 4, `最高柱 = ${maxPad} token`, 'costtx', 'end'));
  const by = BAR_TOP + BAR_H + 34;
  parts.push(
    infoBox(
      x0,
      by,
      RW,
      118,
      `${cs.distribution.toUpperCase()}，num_cuda_graphs=${cs.numCudaGraphs}（InferenceSetupConfig 默认）`,
      [
        `枚举尺寸：[${[...cs.sizes].sort((a, b) => b - a).join(', ')}]`,
        `图数 ${cs.count}；最坏相对 padding ${cs.worstRelPct}%（n=${cs.worst.n} 被垫到 ${cs.worst.chosen}）`,
        `n=1..${cs.maxTokens} 的平均相对 padding ${cs.meanRelPct}%`,
        `num_cuda_graphs=-1（自动）时：图数 ${auto.count}，最坏 ${auto.worstRelPct}%，平均 ${auto.meanRelPct}%`,
      ],
      cs.distribution === 'exponential' ? 'acc1' : 'neutral',
      where,
    ),
  );
  return by + 118;
}

function renderCudaGraphs() {
  const W = 1272;
  const H = 640;
  const parts = header(
    W,
    '图 3　CUDA graph 尺寸枚举：指数递减用更少的图换「与规模无关」的 padding 上界',
    `算例：tp_size=${CFG.tpSize}，decode 图上界 cuda_graph_max_tokens = max_requests × (num_speculative_tokens+1) = ${CFG.cudaGraphMaxTokens}；每个尺寸各捕获一张 decode-only 图`,
  );
  const b1 = graphPanel(parts, { x0: 28, y0: 82, w: 616, cs: FIG3.expDefault, auto: FIG3.expAuto, title: 'EXPONENTIAL（默认）：从 max 逐次减半到 tp_size', where: 'fig3/exp' });
  const b2 = graphPanel(parts, { x0: 656, y0: 82, w: 616, cs: FIG3.linDefault, auto: FIG3.linAuto, title: 'LINEAR（旧行为）：等步长 max / num_cuda_graphs', where: 'fig3/lin' });

  // 规模放大后的对照
  const SY = Math.max(b1, b2) + 30;
  const el = FIG3.expLarge;
  const ll = FIG3.linLarge;
  parts.push(text(28, SY, `同样 num_cuda_graphs=${CFG.defaultNumCudaGraphs}，把上界放大到 ${CFG.largeCudaGraphMaxTokens}（max_requests=${CFG.largeCudaGraphMaxTokens}）`, 'pt'));
  parts.push(text(28, SY + 30, guard(`EXPONENTIAL：${el.count} 张图，最坏 ${el.worstRelPct}%（n=${el.worst.n}→${el.worst.chosen}），平均 ${el.meanRelPct}%`, 11, 300, 'fig3/el'), 'dim'));
  ruler(parts, { x0: 340, y: SY + 28, w: 904, sizes: el.sizes, maxTokens: el.maxTokens, where: 'fig3/elr', labelEvery: 26 });
  parts.push(text(28, SY + 62, guard(`LINEAR：${ll.count} 张图，最坏 ${ll.worstRelPct}%（n=${ll.worst.n}→${ll.worst.chosen}），平均 ${ll.meanRelPct}%`, 11, 300, 'fig3/ll'), 'costtx'));
  ruler(parts, { x0: 340, y: SY + 60, w: 904, sizes: ll.sizes, maxTokens: ll.maxTokens, where: 'fig3/llr', labelEvery: 26 });
  parts.push(
    text(
      28,
      SY + 90,
      `线性步长 = max / num_cuda_graphs 随上界线性变粗，最小图从 ${FIG3.linDefault.sizes[FIG3.linDefault.sizes.length - 1]} 变成 ${ll.sizes[ll.sizes.length - 1]}；指数分布永远含 tp_size=${CFG.tpSize} 这一档，最坏比只多算 1 倍`,
      'cap',
    ),
  );
  parts.push(
    text(
      28,
      SY + 108,
      '运行时按 match_graph_config 选「token_count ≥ 真实 n」的最小图；两种分布由 batch_dimensions_utils.py 的同一函数枚举，只是 sizing_distribution 不同',
      'cap',
    ),
  );
  return seal(parts, W, H, 'fig3-cuda-graphs');
}

// ============================================================================

const outputs = new Map([
  ['megatron_inference_kv_blocks.svg', renderKvBlocks()],
  ['megatron_inference_batching.svg', renderBatching()],
  ['megatron_inference_cuda_graphs.svg', renderCudaGraphs()],
]);

export {
  CFG, INVARIANT, NAIVE, PROMPT_TOKENS_PER_WAVE, FIG1_RZ, FIG1_LRU, FIG2, FIG3,
  KVBlockAllocator, EngineSim, computeBlockHashes, calculateCudaGraphTokenCounts,
  decodeGraphTokenCounts, paddingProfile, outputs,
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
  console.log('\n图 1 ref_zero：', JSON.stringify({ wave1: FIG1_RZ.wave1, wave2: FIG1_RZ.wave2, start2: FIG1_RZ.wave2Start }));
  console.log('图 1 lru：', JSON.stringify({ wave1: FIG1_LRU.wave1, wave2: FIG1_LRU.wave2, start2: FIG1_LRU.wave2Start }));
  console.log('图 1 naive：', JSON.stringify(NAIVE), ' invariant：', JSON.stringify(INVARIANT));
  for (const k of ['static', 'continuous', 'chunked']) {
    const r = FIG2[k];
    console.log(`图 2 ${k}：steps=${r.steps} idle=${r.idle}/${r.slotSteps} maxStepTokens=${r.maxStepTokens} r3=${r.r3Admitted}`);
  }
  for (const k of ['expAuto', 'linAuto', 'expDefault', 'linDefault']) {
    const c = FIG3[k];
    console.log(`图 3 ${k}：sizes=[${c.sizes.join(',')}] count=${c.count} worst=${c.worstRelPct}%@n=${c.worst.n}→${c.worst.chosen} mean=${c.meanRelPct}%`);
  }
}
