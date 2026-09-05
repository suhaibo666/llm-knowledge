// 锁住跨轴 overlap 图示的可执行契约：图 4 的每一格、每个触发列、每个命中 chunk 都由
// lib/megatron_pp_sim.mjs 的离散事件仿真加复刻的 schedules.py 触发条件算出，
// 且与 20_megatron_comm_overlap_analysis.md 正文引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_comm_overlap_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_comm_overlap_figures.mjs');
const trackedDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '20_megatron_comm_overlap_analysis.md',
);

const NAMES = [
  'megatron_overlap_pairs.svg',
  'megatron_overlap_tp_paths.svg',
  'megatron_overlap_pp_send.svg',
  'megatron_overlap_pp_dp_align.svg',
  'megatron_overlap_connection_budget.svg',
];

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

// 越界断言。图元对图元的**重叠**由生成器里的 assertNoTextOverlap 负责——两者缺一不可。
function assertInsideCanvas(svg, name) {
  const { w, h } = viewBox(svg);
  const rects = svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  );
  for (const [, x, y, rw, rh] of rects) {
    assert.ok(Number(x) >= -2, `${name}: rect 左边越界 x=${x}`);
    assert.ok(Number(y) >= -2, `${name}: rect 上边越界 y=${y}`);
    assert.ok(Number(x) + Number(rw) <= w + 2, `${name}: rect 右边越界 ${x}+${rw} > ${w}`);
    assert.ok(Number(y) + Number(rh) <= h + 2, `${name}: rect 下边越界 ${y}+${rh} > ${h}`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w, `${name}: text x=${x} 越界`);
    assert.ok(Number(y) >= 0 && Number(y) <= h, `${name}: text y=${y} 越界`);
  }
}

test('生成器同步产出五张跨轴 overlap 图', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-overlap-figures-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);

    const [pairs, tp, ppsend, align, budget] = await Promise.all(
      NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
    );

    // ---- 图 1：六条轴的掩盖对，每条两泳道 ----
    for (const axis of ['TP', 'CP', 'EP', 'PP', 'DP', 'FSDP']) {
      assert.ok(pairs.includes(`>${axis}<`), `轴 ${axis} 必须单独成一个面板`);
    }
    // 六个面板 = 六条通信泳道 + 六条计算泳道；图 1 每个面板都是两泳道
    assert.equal((pairs.match(/>通信</g) ?? []).length, 6, '六条通信泳道');
    assert.equal((pairs.match(/>计算</g) ?? []).length, 6, '六条计算泳道');
    // 掩盖对必须写出真实符号，不能是抽象的"通信"
    for (const pair of [
      'AG(input) → total_input', 'dgrad = grad_output @ weight', 'wgrad GEMM ＋ grad_bias',
      'AG(K,V head 0)', 'attn_fwd(head 0)',
      'combine_bwd', 'dispatch_fwd → dispatch_bwd', 'mlp_bwd → mlp_bwd_dw → mlp_fwd',
      'irecv(下一槽要吃的张量)', 'recv_prev_wait_handles.pop(0).wait()',
      'AG(bucket i+1)', 'fwd(bucket i)', 'RS(bucket j 的梯度)',
      'AG(unit i+3) … 直到额度用尽', 'wait_bucket_ready(i)',
    ]) {
      assert.ok(pairs.includes(pair), `图 1 必须画出 ${pair}`);
    }
    // 结论行：只有 EP 是横向借另一个 microbatch
    assert.match(pairs, /只有 EP 例外/);
    assert.match(pairs, /用另一个 microbatch 的计算去掩盖本 microbatch 的 A2A/);
    assert.match(pairs, /rely on CUDA_DEVICE_MAX_CONNECTIONS=1 保证下发顺序/);
    assert.match(pairs, /heads_k_stride 写死为 1/);
    assert.match(pairs, /suggested_communication_unit_size \/\/ 2/);

    // ---- 图 2：TP 的两套掩盖对 ----
    assert.ok(tp.includes('>列并行<') && tp.includes('>行并行<'), '两个层类型各一个面板');
    assert.match(tp, /TELayerNorm/);
    assert.match(tp, /TELinear/);
    assert.match(tp, /ub_overlap_ag/);
    assert.match(tp, /ub_overlap_rs/);
    assert.match(tp, /bulk AG\(输入\)/);
    assert.match(tp, /bulk RS\(dgrad 输出\)/);
    // 名字与 docstring 互换那处矛盾必须在图上
    assert.match(tp, /All-Gather overlap with Bprop activation/);
    assert.match(tp, /Reduce-Scatter overlap with Bprop weight/);
    assert.match(tp, /名字与描述正好对调，两者必有一处是错的/);
    // 三条静默降级
    assert.match(tp, /is_expert=True/);
    assert.match(tp, /无 warning/);
    assert.match(tp, /parallel_mode == "duplicated"/);
    // 依赖边界必须画出来
    assert.match(tp, /TransformerEngine/);
    assert.match(tp, /没有 event、没有 \.wait\(\)/);

    // ---- 图 3：PP 的 send 不对称 ----
    assert.ok(ppsend.includes('>前向槽<') && ppsend.includes('>反向槽<'));
    // send / recv 必须各占一条泳道——同一条泳道上画两段交叠通信会互相盖住
    assert.equal((ppsend.match(/>send</g) ?? []).length, 2, 'send 各自成一条泳道');
    assert.equal((ppsend.match(/>recv</g) ?? []).length, 2, 'recv 各自成一条泳道');
    assert.match(ppsend, /isend\(激活\)/);
    assert.match(ppsend, /isend\(梯度\)/);
    assert.match(ppsend, /send_next_wait_handle\.wait\(\) ← 当场/);
    assert.match(ppsend, /本槽不收口/);
    assert.match(ppsend, /deallocate_pipeline_outputs：core 默认 False，训练入口无条件设 True/);

    // ---- 图 4：align_grad_reduce 的真实触发列 ----
    // 仿真结果：makespan 38、每 rank 32 op、6 空泡（与 15 号页锁定的同一组数）
    assert.match(align, /pp=4, vp=2, m=8, N=4/);
    assert.match(align, /makespan=38，每 rank 32 个 op、6 个空泡/);
    // rank 0 两个 chunk 都命中，rank 1..3 只命中 chunk 1
    assert.match(align, />发出：chunk 1、chunk 0</);
    assert.equal((align.match(/>发出：chunk 1</g) ?? []).length, 3, 'rank 1..3 只发 chunk 1');
    // 漏网：rank 1..3 的 chunk 0 落进 cooldown
    assert.equal(
      (align.match(/chunk 0 未被调度器同步 → 进 cooldown 收尾循环/g) ?? []).length,
      3,
    );
    assert.equal((align.match(/>全部 chunk 都在对齐槽发出</g) ?? []).length, 1);
    assert.match(align, /rank r 能命中的最大 grad_sync_vmb 是 15−r/);
    // 参数侧命中表：vp=2 空窗，vp=3 命中 {2}，vp=4 命中 {2,3}
    assert.match(align, /vp = 2/);
    assert.match(align, /调度器一次都不预取/);
    assert.match(align, /窗口 1&lt;c&lt;2 为空/);
    assert.match(align, /预取 chunk 2</);
    assert.match(align, /预取 chunk 2、3</);
    assert.match(align, /窗口 1&lt;c&lt;3/);
    assert.match(align, /窗口 1&lt;c&lt;4/);
    // 网格本身：32 个 op × 4 rank + 空泡；F/B 标注必须存在
    assert.ok((align.match(/>F\d</g) ?? []).length >= 4 * 16, '前向格子数');
    assert.ok((align.match(/>B\d</g) ?? []).length >= 4 * 16, '反向格子数');
    // 命中格用 .mark 描边，且只有 5 个（rank0 两个 + rank1..3 各一个）
    assert.equal((align.match(/class="mark"/g) ?? []).length, 5);

    // ---- 图 5：CUDA_DEVICE_MAX_CONNECTIONS 的三方仲裁 ----
    assert.match(budget, /设成 1 买到什么：kernel 按调用顺序下发/);
    assert.match(budget, /necessary for a speedup but not for correctness/);
    assert.match(budget, /prevent the overlap of communication kernels with computation kernels/);
    assert.match(budget, /arch ≥ 10（Blackwell）/);
    for (const branch of ['① 同时开了 FSDP', '② 同时开了 EP combined overlap', '③ 都没开']) {
      assert.ok(budget.includes(branch), `分支 ${branch} 必须出现`);
    }
    assert.match(budget, /CONNECTIONS to 1 or 32, which/);
    assert.match(budget, /assert os\.environ\.get\(\.\.\.\) == "1"/);
    assert.match(budget, /large than one/); // 源码里的原文（含笔误），反方向硬门
    assert.match(budget, /requires CUDA_DEVICE_MAX_CONNECTIONS &gt; 1 or unset/);

    // ---- 已跟踪资产必须与生成器同步 ----
    const tracked = await Promise.all(
      NAMES.map((name) => readFile(join(trackedDir, name), 'utf8')),
    );
    [pairs, tp, ppsend, align, budget].forEach((svg, i) => {
      assert.equal(svg, tracked[i], `已跟踪的 ${NAMES[i]} 必须与生成器同步`);
    });

    for (const [name, svg] of [
      ['pairs', pairs], ['tp', tp], ['ppsend', ppsend], ['align', align], ['budget', budget],
    ]) {
      assert.match(svg, /^<svg[\s\S]*<\/svg>\s*$/);
      assert.doesNotMatch(svg, /undefined|NaN/);
      assert.doesNotMatch(svg, /\[\[[^\]]{3,}\]\]/, `${name}: wikilink 不许漏进标注`);
      assertInsideCanvas(svg, name);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('页面正文引用的数值与图上一致', async () => {
  const page = await readFile(pagePath, 'utf8');

  for (const name of NAMES) {
    assert.ok(
      new RegExp(`!\\[[^\\]]+\\]\\(assets/${name.replace('.', '\\.')}\\)`).test(page),
      `正文必须引用 ${name}`,
    );
  }
  assert.doesNotMatch(page, /<svg/);

  // 图上算出来的数字必须在正文里出现。
  // 自检：只改正文、不改图，这些断言必须变红。
  for (const needle of [
    // 图 1 —— 六条掩盖对，正文必须逐条讲到
    'grad_input = grad_output.matmul(weight)',
    'heads_k_stride',
    'kv_buffer_copy',
    'combine_bwd | dispatch_fwd->dispatch_bwd  | combine_fwd',
    'recv_prev_wait_handles',
    'register_grad_ready',
    'suggested_communication_unit_size // 2',
    'wait_bucket_ready',
    // 图 2 —— TP 两套路径与那处矛盾
    'TELayerNormColumnParallelLinear',
    'ub_overlap_rs_dgrad',
    '**All-Gather** overlap with Bprop **activation gradient** GEMM',
    '**Reduce-Scatter** overlap with Bprop **weight gradient** GEMM',
    'parallel_mode == "duplicated"',
    // 图 3 —— send 不对称
    'send_next_wait_handle.wait()',
    'send_prev_wait_handle',
    "kw_args['deallocate_pipeline_outputs'] = True",
    // 图 2 —— 全部来自仿真与复刻的触发条件
    '$pp=4$、$vp=2$、$m=8$、$N=4$',
    'makespan 38，每 rank 32 个 op、6 个空泡',
    '`vmb - rank` 必须落在 $[0, 16)$ 内',
    '最大 `grad_sync_vmb` 是 $15-r$',
    '`vmb = 11`（对应 chunk 1）与 `vmb = 15`（对应 chunk 0）',
    'rank 1、2、3 上 chunk 0 的梯度归约完全没被调度器发出',
    '$1 < c < 2$',
    '$1 < c < 3$',
    '$1 < c < 4$',
    '一次都不预取',
    // 图 3
    'necessary for a speedup but not for correctness',
    'prevent the overlap of communication kernels with computation kernels',
    'CUDA_DEVICE_MAX_CONNECTIONS to 1 or 32',
    'no longer exists since the Blackwell architecture',
    'CUDA_DEVICE_MAX_CONNECTIONS > 1 or unset',
  ]) {
    assert.ok(page.includes(needle), `正文必须引用图上的 ${needle}`);
  }
});

test('页面不再携带 path:line 引用', async () => {
  const page = await readFile(pagePath, 'utf8');
  const offenders = [...page.matchAll(/[\w/]+\.py:\d+/g)].map((m) => m[0]);
  assert.deepEqual(offenders, [], `正文仍有 path:line 引用：${offenders.join(', ')}`);
});
