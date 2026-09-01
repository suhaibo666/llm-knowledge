// 锁住仿真器与页面的一致性：期望值直接取自
// wiki/02_engineering/02_train_frameworks/megatron-lm/15_megatron_pp_schedulers_analysis.md
// 的 §③.3 / §⑤.3(a) 两张 ASCII 调度表（该页基线 Megatron@71092579）。
//
// 这两张表是独立于本模块手工推演出来的，因此逐格相等构成互相印证：
// 一旦源码基线推进导致调度变化，这里会先红，提醒同时更新页面、图与基线声明。
//
// 运行: node --test tools/figs/svg/lib/megatron_pp_sim.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { seq1f1b, seqVpp, scheduleTable, simulate } from './megatron_pp_sim.mjs';

const CFG = { pp: 4, m: 8, vp: 2, N: 4 };
// 页面记号：小写 f/b = chunk0 前向/反向，大写 F/B = chunk1；`..` = 空泡
const label = (op) => (op === null ? '..' : (op.c === 0 ? (op.f ? 'f' : 'b') : (op.f ? 'F' : 'B')) + op.mb);
const render = (rows) => rows.map((r) => r.map(label).join(' '));

test('get_schedule_table 与页面 §③.3 的调度表一致', () => {
  const T = scheduleTable(CFG.m, CFG.vp, CFG.N);
  assert.equal(T.map((x) => x[0]).join(' '), '0 1 2 3 0 1 2 3 4 5 6 7 4 5 6 7');
  assert.equal(T.map((x) => x[1]).join(' '), '0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1');
});

test('VPP warmup = 2(pp-r-1)+(vp-1)N —— 页面记为 rank0..3 = 10/8/6/4', () => {
  assert.deepEqual(seqVpp(CFG).map((r) => r.warmup), [10, 8, 6, 4]);
});

test('combined-1F1B 宿主 warmup 各 +1（schedules.py:918-919）', () => {
  assert.deepEqual(seqVpp({ ...CFG, extraWarmup: 1 }).map((r) => r.warmup), [11, 9, 7, 5]);
});

test('纯 VPP 空间-时间图逐格复现页面 §③.3', () => {
  const sim = simulate(seqVpp(CFG), CFG);
  assert.equal(sim.span, 38);
  assert.equal(sim.ops, 32);
  assert.equal(sim.bubble, 6);
  assert.deepEqual(render(sim.rows), [
    'f0 f1 f2 f3 F0 F1 F2 F3 f4 f5 f6 B0 f7 B1 .. .. F4 B2 F5 B3 F6 b0 F7 b1 b2 b3 .. B4 .. B5 .. B6 .. B7 b4 b5 b6 b7',
    '.. f0 f1 f2 f3 F0 F1 F2 F3 f4 B0 f5 B1 f6 B2 f7 B3 F4 b0 F5 b1 F6 b2 F7 b3 .. B4 .. B5 .. B6 .. B7 b4 b5 b6 b7 ..',
    '.. .. f0 f1 f2 f3 F0 F1 F2 B0 F3 B1 f4 B2 f5 B3 f6 b0 f7 b1 F4 b2 F5 b3 F6 B4 F7 B5 .. B6 .. B7 b4 b5 b6 b7 .. ..',
    '.. .. .. f0 f1 f2 f3 F0 B0 F1 B1 F2 B2 F3 B3 f4 b0 f5 b1 f6 b2 f7 b3 F4 B4 F5 B5 F6 B6 F7 B7 b4 b5 b6 b7 .. .. ..',
  ]);
});

test('combined-1F1B 宿主形态逐格复现页面 §⑤.3(a)', () => {
  const sim = simulate(seqVpp({ ...CFG, extraWarmup: 1 }), CFG);
  assert.equal(sim.span, 38);          // 阶梯右移一格，makespan 与空泡数不变
  assert.equal(sim.bubble, 6);
  assert.equal(
    render(sim.rows)[0],
    'f0 f1 f2 f3 F0 F1 F2 F3 f4 f5 f6 f7 B0 .. .. F4 B1 F5 B2 F6 B3 F7 b0 b1 b2 .. b3 .. B4 .. B5 .. B6 B7 b4 b5 b6 b7',
  );
});

test('标准 1F1B（调度器②）：makespan 22 stage-op，每设备 16 op + 6 空泡', () => {
  const sim = simulate(seq1f1b(CFG), { pp: CFG.pp, vp: 1 });
  assert.equal(sim.span, 22);          // 页面 §②.3
  assert.equal(sim.ops, 16);
  assert.equal(sim.bubble, 6);
});

test('气泡率关系：VPP 的空泡/计算 = 标准 1F1B 的 1/vp', () => {
  const a = simulate(seq1f1b(CFG), { pp: CFG.pp, vp: 1 });
  const b = simulate(seqVpp(CFG), CFG);
  assert.equal(a.bubble / a.ops, 3 / 8);        // (pp-1)/m
  assert.equal(b.bubble / b.ops, 3 / 16);       // (pp-1)/(m·vp)
  assert.equal(b.bubble / b.ops, (a.bubble / a.ops) / CFG.vp);
});
