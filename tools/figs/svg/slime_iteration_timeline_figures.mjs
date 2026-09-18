// 图：同一个 4 轮例子分别放进 `train.py`（同步）、`train_async.py`（--update-weights-interval 1 与 2）
// 和 `train_async.py` + fully-async rollout 替换，逐轮给出 serving 版本、被消费数据的策略年龄、
// 发布栅栏、fully-async 的完成队列与池内积压，以及第 1 轮保存时 DataSource 游标相对已训练数据的位置。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：
//  1. 同步 driver 每轮 generate → train → save → publish，训练起点版本等于生成版本，年龄恒 0；
//  2. 异步 driver 先提交 generate(i+1) 再训练 i，发布前等这个 future，所以 interval=1 年龄为 1、
//     interval=2 年龄在 1 与 2 之间交替，发布次数减半；
//  3. RolloutManager 是同步 Ray actor，save(i) 排在 generate(i+1) 之后执行（按 Ray 串行契约推导），
//     异步入口在 rollout 1 保存的游标已经越过第 2 批 prompt；
//  4. fully-async 的批次边界是"完成队列里凑够 rollout_batch_size 组"：每次发布时在途组被中止、
//     回到 buffer 再接前缀续生成，队列里的余量留到下一轮消费，年龄可以超过 1；
//     保存时游标还领先队列、池与 buffer 里的组。
// 布局：上方面板三条 driver 泳道共用时间轴（RolloutManager 行 + driver/trainer 行），
// 下方面板 fully-async：RolloutManager 行、trainer 行和逐组甘特行，栅栏竖线贯穿。
// acc1 标生成与版本，acc2 标发布栅栏、中止后回队续生成与游标越界。
//
// 发布语义：slime 的 updater 向 /pause_generation 发 {}，SGLang v0.5.15.post1 的
// PauseGenerationReqInput.mode 默认 "abort"，TokenizerManager.pause_generation 循环 abort_all；
// 在途请求以 finish_reason=abort 返回已生成前缀，Sample 变 ABORTED，_make_done_cb 把整组交回
// data_buffer；RolloutDataSourceWithBuffer.get_samples 先从 buffer 取、不推进 sample_offset；
// 回队组重新排信号量，generate 复用 tokens 接前缀续生成（剩余时长不变）。结论与证据归 13 页 §2.3.3。
// 模型简化（分析假设，页面正文同样声明）：时长是示意输入，不比较吞吐；信号量按 Sample 先来先到放行，
// 这里按整组放行；中止回调与重新补位视为发生在发布开始的同一时刻。
//
// 用法：node tools/figs/svg/slime_iteration_timeline_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  numRollout: 4, // --num-rollout
  rolloutBatchSize: 2, // --rollout-batch-size（prompt 组）；--over-sampling-batch-size 缺省取同值
  saveInterval: 2, // --save-interval
  ticks: Object.freeze({ generate: 3, train: 2, publish: 1 }), // 示意时长（默认 rollout 一批 / 一次训练 / 一次发布）
  fully: Object.freeze({
    sglangServerConcurrency: 2, // --sglang-server-concurrency
    engines: 2, // get_rollout_num_engines
    nSamplesPerPrompt: 2, // --n-samples-per-prompt
    groupTicks: Object.freeze([3, 2, 2, 2, 2, 5, 4, 5, 2, 4, 3, 2]), // 每组生成时长（示意；按 gid 取）
  }),
});

export const LANES = Object.freeze([
  Object.freeze({ key: 'sync', driver: 'train.py', interval: null, rollout: 'default' }),
  Object.freeze({ key: 'async1', driver: 'train_async.py', interval: 1, rollout: 'default' }),
  Object.freeze({ key: 'async2', driver: 'train_async.py', interval: 2, rollout: 'default' }),
  Object.freeze({ key: 'fully', driver: 'train_async.py', interval: 1, rollout: 'fully' }),
]);

// ---------------- 对源码控制流的最小复现 ----------------

// slime/utils/misc.py::should_run_periodic_action
export function shouldRunPeriodic(rolloutId, interval, numRollout = null, perEpoch = null) {
  if (interval === null || interval === undefined) return false;
  if (numRollout !== null && rolloutId === numRollout - 1) return true;
  const step = rolloutId + 1;
  return step % interval === 0 || (perEpoch !== null && step % perEpoch === 0);
}

// fully_async_rollout.py::_get_global_worker：池 = sglang_server_concurrency × engine 数（组）；
// sglang_rollout.py::GenerateState.__init__：信号量同一个数但按 Sample 计，按 Sample 先来先到放行、
// 组与组可以交错。模型简化为整组放行：同时生成的组 = 池 / n。
export function fullyCapacity(cfg = CFG) {
  const pool = cfg.fully.sglangServerConcurrency * cfg.fully.engines;
  return { pool, semaphore: pool, generatingGroups: Math.floor(pool / cfg.fully.nSamplesPerPrompt) };
}

// 离散时间仿真。每个整数时刻 t 的顺序：
//   0) 上一段 [t-1,t) 生成完的组进入完成队列（done callback → output_queue.put）；
//   A) driver 与 RolloutManager 交替推进，直到 driver 在等待（ray.get）或睡眠（训练 / 发布）；
//      RolloutManager 是同步 actor：按提交顺序一次执行一个方法；
//   B) fully-async worker 每 1 个时刻轮询一次：active < 池 且 qsize < 池 时 get_samples(1) 补位，
//      buffer 里有回队组时先取它，不推进游标；
//   C) 信号量按入池先后放行等待的组（发布暂停期间新请求在服务端等待，模型里不放行）；
//   D) 推进 [t,t+1)：未暂停时在途组剩余时长减一，并记下这一段的 serving 版本。
//   发布开始时（mode=abort）：在途组全部中止，已生成的前缀保留，整组交回 buffer。
export function simulate(lane, cfg = CFG) {
  const R = cfg.numRollout;
  const B = cfg.rolloutBatchSize;
  const fully = lane.rollout === 'fully';
  const cap = fullyCapacity(cfg);
  let t = 0;
  let serving = null;
  let trainer = 0;
  let paused = false;
  let cursor = 0; // RolloutDataSource.sample_offset（从数据集取出的 prompt 组数）
  const publishes = [];
  const trains = [];
  const calls = []; // RolloutManager 上执行过的方法
  const rmQueue = [];
  let rmCur = null;
  const groups = [];
  const outputQueue = [];
  const buffer = []; // RolloutDataSourceWithBuffer.buffer
  let workerStarted = false;
  let seq = 0;

  const inPool = () => groups.filter((g) => g.state === 'waiting' || g.state === 'running');
  const generating = () => groups.filter((g) => g.state === 'running');

  function submit(kind, id) {
    const call = { kind, id, submitAt: t, startAt: null, endAt: null, done: false, batch: [] };
    rmQueue.push(call);
    calls.push(call);
    return call;
  }

  function* publish(tag) {
    // SGLangEngine.pause_generation 发 {} → 上游默认 mode="abort"：在途请求以 abort 结束，
    // _make_done_cb 见到 ABORTED 成员，整组 data_buffer.add_samples
    const aborted = generating();
    const p = { tag, startAt: t, aborted: aborted.map((g) => g.gid), waiting: inPool().filter((g) => g.state === 'waiting').map((g) => g.gid) };
    for (const g of aborted) {
      g.runs[g.runs.length - 1].endAt = t;
      g.aborts.push(t);
      g.state = 'buffer';
      buffer.push(g);
    }
    paused = true;
    yield { sleep: cfg.ticks.publish };
    paused = false;
    serving = serving === null ? 0 : trainer;
    p.endAt = t;
    p.version = serving;
    publishes.push(p);
  }

  function* trainAndSave(id, batchCall) {
    const tr = { id, startAt: t, from: trainer, batch: batchCall };
    yield { sleep: cfg.ticks.train };
    trainer += 1;
    tr.endAt = t;
    trains.push(tr);
    if (shouldRunPeriodic(id, cfg.saveInterval, R)) {
      // actor_model.save_model 视为瞬时；ray.get(rollout_manager.save.remote(id)) 要等 actor 队列
      const s = submit('save', id);
      yield { until: () => s.done };
    }
  }

  // train.py::train
  function* syncDriver() {
    yield* publish('init');
    for (let id = 0; id < R; id += 1) {
      const g = submit('generate', id);
      yield { until: () => g.done };
      yield* trainAndSave(id, g);
      yield* publish(id);
    }
  }

  // train_async.py::train
  function* asyncDriver() {
    yield* publish('init');
    let next = submit('generate', 0);
    let curr = null;
    for (let id = 0; id < R; id += 1) {
      if (next !== null) {
        const n = next;
        yield { until: () => n.done };
        curr = n;
      }
      if (id + 1 < R) next = submit('generate', id + 1);
      yield* trainAndSave(id, curr);
      if ((id + 1) % lane.interval === 0) {
        if (next !== null) {
          const n = next;
          yield { until: () => n.done };
          curr = n;
        }
        next = null;
        yield* publish(id);
      }
    }
  }

  function rmStep() {
    let progressed = false;
    for (;;) {
      if (rmCur === null) {
        if (rmQueue.length === 0) return progressed;
        rmCur = rmQueue.shift();
        rmCur.startAt = t;
        progressed = true;
        if (rmCur.kind === 'generate') {
          if (fully) {
            workerStarted = true; // _generate_rollout_async 首次调用创建全局 worker
          } else {
            // generate_rollout_async：over_sampling_batch_size = rollout_batch_size，开头一次取数
            cursor += B;
            rmCur.version = serving;
            rmCur.readyAt = t + cfg.ticks.generate;
          }
        }
      }
      if (rmCur.kind === 'save') {
        // RolloutDataSource.save 只写游标与 metadata
        rmCur.cursor = cursor;
        rmCur.queued = outputQueue.map((g) => g.gid);
        rmCur.pool = inPool().map((g) => g.gid);
        rmCur.buffered = buffer.map((g) => g.gid);
        rmCur.done = true;
        rmCur.endAt = t;
        rmCur = null;
        progressed = true;
        continue;
      }
      if (!fully) {
        if (t < rmCur.readyAt) return progressed;
        rmCur.batch = [{ gid: null, versions: [rmCur.version] }, { gid: null, versions: [rmCur.version] }].slice(0, B);
      } else {
        // _generate_rollout_async 执行期间每 0.05 s get_completed_groups(limit=target - len(collected))：
        // 有 generate 在执行时每个时刻都取，队列只在两次 generate 之间积压
        while (rmCur.batch.length < B && outputQueue.length > 0) {
          const g = outputQueue.shift();
          g.state = 'consumed';
          g.consumedBy = rmCur.id;
          g.consumedAt = t;
          rmCur.batch.push(g);
        }
        if (rmCur.batch.length < B) return progressed;
      }
      rmCur.done = true;
      rmCur.endAt = t;
      rmCur = null;
      progressed = true;
    }
  }

  const driver = lane.driver === 'train.py' ? syncDriver() : asyncDriver();
  let wait = driver.next().value;
  let sleepUntil = null;
  for (let guard = 0; ; guard += 1) {
    if (guard > 400) throw new Error('仿真没有收敛');
    // 0) 完成回调
    if (fully) {
      for (const g of generating()) {
        if (g.remaining === 0) {
          g.doneAt = t;
          g.runs[g.runs.length - 1].endAt = t;
          g.state = 'queue';
          outputQueue.push(g);
        }
      }
    }
    // A) driver ⇄ RolloutManager
    for (;;) {
      let moved = rmStep();
      if (wait !== undefined) {
        let ready = false;
        if (wait.sleep !== undefined) {
          if (sleepUntil === null) sleepUntil = t + wait.sleep;
          ready = t >= sleepUntil;
        } else {
          ready = wait.until();
        }
        if (ready) {
          sleepUntil = null;
          const r = driver.next();
          wait = r.done ? undefined : r.value;
          moved = true;
        }
      }
      if (!moved) break;
    }
    if (wait === undefined && rmCur === null && rmQueue.length === 0) break;
    if (fully && workerStarted) {
      // B) AsyncRolloutWorker._loop 补位
      while (inPool().length < cap.pool && outputQueue.length < cap.pool) {
        if (buffer.length > 0) {
          // pop_first：先取回队组，sample_offset 不动
          const g = buffer.shift();
          g.state = 'waiting';
          g.pulls.push(t);
          g.seq = seq++;
        } else {
          groups.push({ gid: groups.length, state: 'waiting', seq: seq++, pulls: [t], runs: [], aborts: [], doneAt: null, remaining: null, versions: [], consumedBy: null, consumedAt: null });
          cursor += 1;
        }
      }
      // C) 信号量（模型按整组、按入池先后放行）
      if (!paused) {
        let running = generating().length;
        for (const g of inPool().filter((x) => x.state === 'waiting').sort((a, b) => a.seq - b.seq)) {
          if (running >= cap.generatingGroups) break;
          g.state = 'running';
          g.runs.push({ startAt: t, endAt: null, versions: [] });
          if (g.remaining === null) {
            g.remaining = cfg.fully.groupTicks[g.gid];
            if (g.remaining === undefined) throw new Error(`groupTicks 不够：gid ${g.gid}`);
          }
          running += 1;
        }
      }
    }
    // D) 推进
    if (fully && !paused) {
      for (const g of generating()) {
        g.remaining -= 1;
        if (!g.versions.includes(serving)) g.versions.push(serving);
        const run = g.runs[g.runs.length - 1];
        if (!run.versions.includes(serving)) run.versions.push(serving);
      }
    }
    t += 1;
  }

  const trainRows = trains.map((tr) => {
    const batch = tr.batch.batch;
    const ages = batch.map((g) => tr.from - Math.min(...g.versions));
    return {
      id: tr.id,
      startAt: tr.startAt,
      endAt: tr.endAt,
      from: tr.from,
      gids: batch.map((g) => g.gid),
      versions: batch.map((g) => [...g.versions]),
      ages,
      maxAge: Math.max(...ages),
    };
  });
  const saveRows = calls
    .filter((c) => c.kind === 'save')
    .map((c) => {
      const trainedGroups = (c.id + 1) * B;
      return {
        id: c.id,
        submitAt: c.submitAt,
        execAt: c.startAt,
        cursor: c.cursor,
        trainedGroups,
        lead: c.cursor - trainedGroups,
        queued: c.queued,
        pool: c.pool,
        buffered: c.buffered,
        resumeStart: c.id + 1, // MegatronTrainRayActor.init：start_rollout_id = iteration + 1；load(start − 1)
      };
    });
  return {
    lane: lane.key,
    endAt: t,
    publishes: publishes.map((p) => ({ ...p })),
    publishCount: publishes.length,
    generates: calls.filter((c) => c.kind === 'generate').map((c) => ({ id: c.id, submitAt: c.submitAt, startAt: c.startAt, endAt: c.endAt, version: c.version ?? null })),
    trains: trainRows,
    saves: saveRows,
    groups: groups.map((g) => ({ gid: g.gid, state: g.state, pulls: [...g.pulls], runs: g.runs.map((r) => ({ ...r, versions: [...r.versions] })), aborts: [...g.aborts], doneAt: g.doneAt, versions: [...g.versions], consumedBy: g.consumedBy, consumedAt: g.consumedAt })),
    cursor,
  };
}

export function model(cfg = CFG) {
  const lanes = Object.fromEntries(LANES.map((lane) => [lane.key, simulate(lane, cfg)]));
  const B = cfg.rolloutBatchSize;
  const saveOf = (key, id) => lanes[key].saves.find((s) => s.id === id);
  const s1 = { sync: saveOf('sync', 1), async1: saveOf('async1', 1), async2: saveOf('async2', 1), fully: saveOf('fully', 1) };
  const f = lanes.fully;
  const fs1 = s1.fully;
  return {
    cfg,
    capacity: fullyCapacity(cfg),
    lanes,
    ages: Object.fromEntries(Object.entries(lanes).map(([k, v]) => [k, v.trains.map((tr) => tr.maxAge)])),
    publishCounts: Object.fromEntries(Object.entries(lanes).map(([k, v]) => [k, v.publishCount])),
    save1: s1,
    // 从 rollout 1 的 checkpoint 恢复：start_rollout_id = 2，DataSource.load(1) 把游标放回保存值，
    // generate(2) 从这个游标取数；已训练 (1+1)·B 组，游标之前却已取出 cursor 组。
    skippedOnResume: Object.fromEntries(Object.entries(s1).map(([k, s]) => [k, s.cursor - s.trainedGroups])),
    fullyLead: {
      nextBatch: B,
      queued: fs1.queued.length,
      pool: fs1.pool.length,
      buffered: fs1.buffered.length,
    },
    fullyAbortedAtPublish: f.publishes.map((p) => p.aborted.length),
    fullyResumedConsumed: f.trains.flatMap((tr) => tr.gids.filter((gid) => f.groups[gid].aborts.length > 0).map((gid) => ({ id: tr.id, gid, versions: [...f.groups[gid].versions] }))),
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function textWidth(s, size) {
  let w = 0;
  for (const ch of String(s)) w += /[ -⯿⺀-￯]/.test(ch) ? size : size * 0.58;
  return w;
}
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .xs{font-size:9.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .inv{font-size:10.5px;fill:#1F2430}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .wait{fill:url(#hatch);stroke:#C9CFD8;stroke-width:1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.4}
  .acc1b{fill:#CFE0FB;stroke:#2563EB;stroke-width:1.4}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.4}
  .fence{stroke:#C3651F;stroke-width:1.4;stroke-dasharray:4 3}
  .savel{stroke:#2A313B;stroke-width:1.2;stroke-dasharray:2 2}
  .grid{stroke:#E6E9EF;stroke-width:1}
  .qline{stroke:#8A93A0;stroke-width:1.6}
  .band{fill:#FCF1E6;stroke:none}
  .ab{font-size:11px;font-weight:700;fill:#C3651F}
`;

function render(m) {
  const cfg = m.cfg;
  const W = 1180;
  const X0 = 258;
  const PX = 33;
  const L = m.lanes;
  const maxT = Math.max(...Object.values(L).map((l) => l.endAt));
  const xt = (t) => X0 + t * PX;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 4) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const line = (x1, y1, x2, y2, cls) => o.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`);
  const vfmt = (vs) => vs.map((v) => `v${v}`).join('→');

  const laneTop = 122;
  const laneH = 104;
  const fixedKeys = ['sync', 'async1', 'async2'];
  const panel2Top = laneTop + fixedKeys.length * laneH + 34;
  const rowH = 17;
  const nGroups = L.fully.groups.length;
  const groupTop = panel2Top + 104;
  const panel2Bottom = groupTop + nGroups * rowH + 84;
  const H = panel2Bottom + 44;

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime 迭代时序：同一 4 轮例子在同步、异步 interval 1 与 2、fully-async rollout 下的版本、年龄、发布栅栏、积压与保存游标</title>');
  o.push('<desc id="desc">上方三条泳道共用时间轴，画 train.py 与 train_async.py 两种 interval 下 RolloutManager 的生成与保存、trainer 的训练与发布，标出每批数据的生成版本与训练时的策略年龄，以及 rollout 1 保存时的游标。下方画 fully-async rollout：逐组的等待、生成、发布开始时中止并回队、接前缀续生成、排队与被哪一轮消费，发布暂停带贯穿，保存时游标领先已训练数据的组成。</desc>');
  o.push('<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#F5F7FA"/><line x1="0" y1="0" x2="0" y2="6" stroke="#C9CFD8" stroke-width="2"/></pattern></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 32, '同一个 4 轮例子：阶段边界在四种执行方式下落在哪里', 'ti');
  text(24, 54, `num_rollout ${cfg.numRollout}，rollout_batch_size ${cfg.rolloutBatchSize} 组，save_interval ${cfg.saveInterval}；示意时长：一批生成 ${cfg.ticks.generate}、一次训练 ${cfg.ticks.train}、一次发布 ${cfg.ticks.publish}（只看顺序，不比较吞吐）`, 'su');
  text(24, 72, 'vk = 训练 k 次后发布的权重（v0 是启动首推）；龄 = 训练起点版本 − 这批数据最早 token 的 serving 版本', 'su');

  // 时间轴
  const axisY = 98;
  for (let tt = 0; tt <= maxT; tt += 1) {
    text(xt(tt), axisY, String(tt), 'xs', 'middle');
  }
  text(X0 - 12, axisY, 't', 'xs', 'end');

  // ---------- ① 三条 driver 泳道 ----------
  rect(16, laneTop - 14, W - 32, fixedKeys.length * laneH + 8, 'panel', 10);
  const laneTitle = { sync: ['train.py 同步', '每轮 generate → train → save → publish'], async1: ['train_async.py', '--update-weights-interval 1'], async2: ['train_async.py', '--update-weights-interval 2'] };
  fixedKeys.forEach((key, li) => {
    const lane = L[key];
    const y = laneTop + li * laneH;
    const rmY = y + 8;
    const trY = y + 40;
    text(28, y + 22, laneTitle[key][0], 'pt');
    text(28, y + 40, laneTitle[key][1], 'sm');
    text(X0 - 8, rmY + 15, 'RolloutManager', 'xs', 'end');
    text(X0 - 8, trY + 15, 'driver·trainer', 'xs', 'end');
    for (let tt = 0; tt <= maxT; tt += 1) line(xt(tt), rmY - 2, xt(tt), trY + 24, 'grid');
    lane.generates.forEach((g) => {
      rect(xt(g.startAt), rmY, (g.endAt - g.startAt) * PX, 22, 'acc1', 3);
      text(xt(g.startAt) + ((g.endAt - g.startAt) * PX) / 2, rmY + 15, `生成${g.id}·v${g.version}`, 'inv', 'middle');
    });
    lane.trains.forEach((tr) => {
      rect(xt(tr.startAt), trY, (tr.endAt - tr.startAt) * PX, 22, 'neutral', 3);
      text(xt(tr.startAt) + ((tr.endAt - tr.startAt) * PX) / 2, trY + 15, `训${tr.id} 龄${tr.maxAge}`, 'inv', 'middle');
    });
    lane.publishes.forEach((p) => {
      rect(xt(p.startAt), trY, (p.endAt - p.startAt) * PX, 22, 'acc2', 3);
      text(xt(p.startAt) + ((p.endAt - p.startAt) * PX) / 2, trY + 15, `v${p.version}`, 'inv', 'middle');
      line(xt(p.endAt), rmY - 4, xt(p.endAt), trY + 26, 'fence');
    });
    lane.saves.forEach((s) => {
      const sx = xt(s.execAt);
      line(sx, rmY - 4, sx, trY + 26, 'savel');
      rect(sx - 4, rmY + 7, 8, 8, s.lead > 0 ? 'acc2' : 'neutral', 1);
      text(sx + 7, rmY + 15, `存${s.id}`, 'xs');
    });
    const s1 = lane.saves.find((s) => s.id === 1);
    const ages = lane.trains.map((tr) => tr.maxAge).join('、');
    const cut = s1.lead > 0
      ? `存 1 在生成 2 之后执行：游标 ${s1.cursor} > 已训 ${s1.trainedGroups} 组，从 1 恢复跳过 ${s1.lead} 组`
      : `存 1：游标 ${s1.cursor} = 已训 ${s1.trainedGroups} 组`;
    text(xt(0), trY + 42, `年龄 ${ages} · 发布 ${lane.publishCount} 次（含启动首推）· ${cut}`, s1.lead > 0 ? 'tx' : 'sm');
  });

  // ---------- ② fully-async ----------
  const f = L.fully;
  const cap = m.capacity;
  rect(16, panel2Top - 14, W - 32, panel2Bottom - panel2Top + 14, 'panel', 10);
  text(28, panel2Top + 8, 'train_async.py（interval 1）+ fully-async rollout：批次边界 = 完成队列里凑够 2 组', 'pt');
  text(28, panel2Top + 26, `池 = concurrency ${cfg.fully.sglangServerConcurrency} × engine ${cfg.fully.engines} = ${cap.pool} 组；信号量 ${cap.semaphore} 条请求，每组 ${cfg.fully.nSamplesPerPrompt} 条 → 同时生成 ${cap.generatingGroups} 组（模型按整组放行）；qsize ≥ ${cap.pool} 停止补位`, 'sm');
  const rmY = panel2Top + 40;
  const trY = panel2Top + 70;
  text(X0 - 8, rmY + 15, 'RolloutManager', 'xs', 'end');
  text(X0 - 8, trY + 15, 'driver·trainer', 'xs', 'end');
  f.generates.forEach((g) => {
    const w = Math.max(1, g.endAt - g.startAt) * PX;
    rect(xt(g.startAt), rmY, w, 22, 'acc1', 3);
    text(xt(g.startAt) + w / 2, rmY + 15, `取${g.id}`, 'inv', 'middle');
  });
  f.trains.forEach((tr) => {
    rect(xt(tr.startAt), trY, (tr.endAt - tr.startAt) * PX, 22, 'neutral', 3);
    text(xt(tr.startAt) + ((tr.endAt - tr.startAt) * PX) / 2, trY + 15, `训${tr.id} 龄${tr.maxAge}`, 'inv', 'middle');
  });
  const gridBottom = groupTop + nGroups * rowH;
  for (let tt = 0; tt <= f.endAt; tt += 1) line(xt(tt), rmY - 2, xt(tt), gridBottom, 'grid');
  f.publishes.forEach((p) => {
    o.push(`<rect x="${xt(p.startAt)}" y="${groupTop}" width="${(p.endAt - p.startAt) * PX}" height="${nGroups * rowH}" class="band"/>`);
    rect(xt(p.startAt), trY, (p.endAt - p.startAt) * PX, 22, 'acc2', 3);
    text(xt(p.startAt) + ((p.endAt - p.startAt) * PX) / 2, trY + 15, `v${p.version}`, 'inv', 'middle');
    line(xt(p.startAt), rmY - 4, xt(p.startAt), gridBottom + 4, 'fence');
    line(xt(p.endAt), rmY - 4, xt(p.endAt), gridBottom + 4, 'fence');
  });
  f.groups.forEach((g) => {
    const y = groupTop + g.gid * rowH;
    text(X0 - 8, y + 12, `组 ${g.gid}`, 'xs', 'end');
    // 每次入池（首次取出或回队后重新取出）先等信号量，再生成一段；发布开始时中止
    g.pulls.forEach((pulledAt, k) => {
      const run = g.runs[k];
      const endWait = run ? run.startAt : g.doneAt ?? f.endAt;
      if (endWait > pulledAt) rect(xt(pulledAt), y + 2, (endWait - pulledAt) * PX, rowH - 4, 'wait', 2);
    });
    g.runs.forEach((run, k) => {
      const end = run.endAt ?? f.endAt;
      rect(xt(run.startAt), y + 2, (end - run.startAt) * PX, rowH - 4, k === 0 ? 'acc1b' : 'acc2', 2);
      if (run.versions.length) text(xt(run.startAt) + 4, y + 12, `${k === 0 ? '' : '续'}${vfmt(run.versions)}`, 'xs');
    });
    g.aborts.forEach((at) => text(xt(at), y + 13, '✕', 'ab', 'middle'));
    if (g.doneAt !== null) {
      const qEnd = g.consumedAt ?? f.endAt;
      if (qEnd > g.doneAt) line(xt(g.doneAt), y + rowH / 2, xt(qEnd), y + rowH / 2, 'qline');
      if (g.consumedBy !== null) text(xt(qEnd) + 4, y + 12, `→训${g.consumedBy}`, 'xs');
    }
  });
  const fs1 = f.saves.find((s) => s.id === 1);
  const sx = xt(fs1.execAt);
  line(sx, rmY - 4, sx, gridBottom + 4, 'savel');
  rect(sx - 4, rmY + 7, 8, 8, 'acc2', 1);
  text(sx + 7, rmY + 15, `存${fs1.id}`, 'xs');
  for (let tt = 0; tt <= f.endAt; tt += 1) text(xt(tt), gridBottom + 16, String(tt), 'xs', 'middle');
  const ly = gridBottom + 36;
  text(28, ly, `存 1（t=${fs1.execAt}）：游标 ${fs1.cursor} = 已训 ${fs1.trainedGroups} + 取${fs1.id + 1} 的 ${m.fullyLead.nextBatch} + 完成队列 ${m.fullyLead.queued} + 池内 ${m.fullyLead.pool} + buffer ${m.fullyLead.buffered}；这些都不进 checkpoint，从 1 恢复跳过 ${fs1.lead} 组`, 'tx');
  const legendY = ly + 22;
  let lx = 28;
  const legend = [
    ['wait', '已取出、等信号量'],
    ['acc1b', '生成中（标版本）'],
    ['abort', '发布开始时中止（mode=abort）'],
    ['acc2', '回队后接前缀续生成'],
  ];
  legend.forEach(([cls, lab]) => {
    if (cls === 'abort') text(lx + 11, legendY + 1, '✕', 'ab', 'middle');
    else rect(lx, legendY - 10, 22, 12, cls, 2);
    text(lx + 28, legendY, lab, 'sm');
    lx += 28 + textWidth(lab, 10.5) + 22;
  });
  line(lx, legendY - 4, lx + 22, legendY - 4, 'qline');
  text(lx + 28, legendY, '完成后在队列里等待', 'sm');
  lx += 28 + textWidth('完成后在队列里等待', 10.5) + 22;
  line(lx + 10, legendY - 12, lx + 10, legendY + 2, 'fence');
  text(lx + 20, legendY, '发布暂停起止', 'sm');

  text(24, H - 16, '阅读顺序：同步年龄恒 0 → 异步先提交下一次生成再训练 → interval 2 一半批次再老一拍 → fully-async 每次发布中止在途组、回队接前缀续生成，队列里的组继续变老', 'cap');
  o.push('</svg>');
  return o.join('\n');
}

export function renderSvg(cfg = CFG) {
  return `${render(model(cfg))}\n`;
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_iteration_timeline.svg'), renderSvg(), 'utf8');
}
