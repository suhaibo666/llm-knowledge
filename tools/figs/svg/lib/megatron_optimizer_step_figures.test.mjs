// 锁住 optimizer-step 图示的可执行契约：四张图上的每个数字都必须由同一组 CFG 与真实的
// IEEE 舍入算出，且与 26_megatron_optimizer_step_internals_deepdive.md 正文引用的数值逐个对齐
// —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_optimizer_step_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, RUN, ULP_BF16, ULP_F32, LEDGER, ledgerTotal, bf16, replay } from '../megatron_optimizer_step_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_optimizer_step_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '26_megatron_optimizer_step_internals_deepdive.md',
);

const NAMES = [
  'megatron_optstep_lifecycle.svg',
  'megatron_optstep_master_precision.svg',
  'megatron_optstep_ledger.svg',
  'megatron_optstep_offload.svg',
];

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

// 越界等于裁字。图元对图元的重叠由生成器里的 assertNoTextOverlap 负责，这里只判越界。
function assertInsideCanvas(svg, name) {
  const { w, h } = viewBox(svg);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
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

// ============================================================================

test('数值内核：bf16 舍入与两条 master 路径的复演结论', () => {
  // bf16 只有 7 位显式尾数：1.0 附近的 ulp 是 2⁻⁷，不是 2⁻⁸
  assert.equal(ULP_BF16, 2 ** -7);
  assert.equal(ULP_F32, 2 ** -23);

  // 舍入分界恰在半个 ulp：低于它舍回原档，高于它进下一档
  assert.equal(bf16(1.0 + ULP_BF16 * 0.4), 1.0);
  assert.equal(bf16(1.0 + ULP_BF16 * 0.6), 1.0 + ULP_BF16);
  assert.equal(bf16(1.0 + CFG.delta), 1.0, '一步更新被 bf16 整个舍掉');

  // 图 2 的两条立论
  assert.equal(RUN.bf16MasterMoved, false, 'bf16 master 全程未动');
  assert.equal(RUN.firstMove, 14, 'fp32 master 第 14 步才推动模型权重');
  assert.equal(RUN.modelCopy[RUN.firstMove], 1.0 + ULP_BF16);
  assert.equal(RUN.modelCopy[RUN.firstMove - 1], CFG.w0);

  // 跳变步数不是常数，而是「半个 ulp / 每步更新」的取整 —— 换 Δ 应当同步改变
  assert.equal(RUN.firstMove, Math.ceil(ULP_BF16 / 2 / CFG.delta));
  const faster = replay({ w0: CFG.w0, delta: CFG.delta * 4, steps: CFG.steps });
  assert.ok(faster.firstMove < RUN.firstMove, 'Δ 变大，跳变必须提前');
});

test('字节账本：四行合计与分片行都由 CFG 推出', () => {
  assert.deepEqual(LEDGER.map(ledgerTotal), [18, 16, 6 + 12 / CFG.dp, 16]);
  assert.equal(ledgerTotal(LEDGER[2]), 7.5, 'DP=8 时分片路径 7.5 bytes/param');
  // 归优化器的那 12 字节 = master + m + v
  const optBytes = LEDGER[0].segs.filter(([, , cls]) => cls === 'acc1').reduce((s, [, b]) => s + b, 0);
  assert.equal(optBytes, 12);
});

test('生成器同步产出四张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-optstep-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [lifecycle, master, ledger, offload] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1：五步与交接物 ----
  for (const stage of [
    'prepare_grads',
    'found_inf_flag',
    'clip_grad_norm',
    'step_with_ready_grads',
    'copy back',
  ]) {
    assert.ok(lifecycle.includes(stage), `图 1 缺阶段 ${stage}`);
  }
  // 交接物必须写出 identity + dtype + 持有者，而不是抽象的「梯度」二字
  assert.ok(lifecycle.includes('main_param.grad'));
  assert.ok(lifecycle.includes('found_inf (1 元素)'));
  assert.ok(lifecycle.includes('model_param.data'));
  assert.equal((lifecycle.match(/fp32 · optimizer/g) ?? []).length, 2, 'main grad 与 master 两处');
  assert.ok(lifecycle.includes('bf16 · 模型'), '链条终点必须回到 bf16 模型参数');
  // 闸门的后果与「为什么在裁剪之前」两个盒子
  assert.ok(lifecycle.includes('return (False, None, None)'));
  assert.ok(lifecycle.includes('scheduler 不推进'));
  assert.ok(lifecycle.includes('被否掉的替代：先裁剪再查'));

  // ---- 图 2：原理图 ----
  assert.ok(master.includes('master 也是 bf16（被否掉的替代）'));
  assert.ok(master.includes('fp32 master + 每步回拷（当前实现）'));
  assert.ok(master.includes(`第 ${RUN.firstMove} 步：回拷第一次改变模型权重`));
  assert.ok(master.includes(`${CFG.steps} 次加法之后仍是 w₀`));
  assert.ok(master.includes(`bf16 下一档 ${CFG.w0 + ULP_BF16}`));
  // 三条曲线各一条 polyline，且回拷阶梯真的有一次跳变（两个不同的 y）
  const stepPath = master.match(/class="plotstep" d="([^"]+)"/);
  assert.ok(stepPath, '回拷阶梯必须存在');
  const ys = new Set(stepPath[1].match(/[\d.]+ ([\d.]+)/g).map((s) => s.split(' ')[1]));
  assert.equal(ys.size, 2, '阶梯恰好有两个高度：跳变前后各一个');
  // 上面板的 bf16 master 是一条平线：所有 y 相同
  const flatPath = master.match(/class="plotflat" d="([^"]+)"/);
  const flatYs = new Set(flatPath[1].match(/[\d.]+ ([\d.]+)/g).map((s) => s.split(' ')[1]));
  assert.equal(flatYs.size, 1, 'bf16 master 必须画成一条水平线');

  // ---- 图 3：字节账本 ----
  assert.ok(ledger.includes('合计 18 B'));
  assert.ok(ledger.includes(`合计 ${6 + 12 / CFG.dp} B`));
  assert.ok(ledger.includes(`12 字节按 DP=${CFG.dp} 切`));
  assert.equal((ledger.match(/model_parallel_group/g) ?? []).length, 3);
  assert.equal((ledger.match(/intra_dist_opt_group/g) ?? []).length, 2, '分片行 + 说明盒');
  assert.ok(ledger.includes('--grad-reduce-in-bf16'));
  assert.ok(!ledger.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 4：offload 三条流 ----
  for (const lane of ['H2D 流', '更新（主流）', 'D2H 流']) {
    assert.ok(offload.includes(lane), `图 4 缺泳道 ${lane}`);
  }
  assert.ok(offload.includes('常驻参数更新'), '常驻参数先算才能覆盖首块预取');
  for (let c = 0; c < CFG.chunkCount; c += 1) {
    assert.ok(offload.includes(`_step_subset(${c})`), `缺 chunk ${c} 的更新块`);
    assert.ok(offload.includes(`回写块 ${c}`), `缺 chunk ${c} 的 D2H`);
  }
  // 最后一块没有下一块可预取
  assert.equal(
    (offload.match(/预取块 /g) ?? []).length,
    CFG.chunkCount - 1,
    '预取块数应比 chunk 数少一',
  );
  assert.ok(offload.includes('drain D2H 一次'));

  for (const [name, svg] of NAMES.map((n, i) => [n, [lifecycle, master, ledger, offload][i]])) {
    assertInsideCanvas(svg, name);
  }

  // 仓库里跟踪的那份必须与刚生成的一致（改了脚本忘记重跑，这里就红）
  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      [lifecycle, master, ledger, offload][i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_optimizer_step_figures.mjs`,
    );
  });
});

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红 —— 所以它读的是 .md 本体，不是硬编码的常量。
  const page = await readFile(pagePath, 'utf8');

  assert.ok(page.includes(`跑 ${CFG.steps} 步`), '正文必须说明复演了多少步');
  assert.ok(page.includes(`第 ${RUN.firstMove} 步`), '正文的跳变步数必须与图一致');
  assert.ok(
    page.includes(`${ULP_BF16 * 1e3}\\times10^{-3}`),
    'bf16 ulp 的正文写法必须与算出来的值一致',
  );
  assert.ok(page.includes(String(CFG.w0 + ULP_BF16)), '正文必须给出 bf16 的下一档');
  assert.ok(
    page.includes(String(Number((CFG.delta / ULP_BF16).toPrecision(3)))),
    'Δ 与 bf16 ulp 的比值必须与图一致',
  );
  assert.ok(
    page.includes(String(Math.round(CFG.delta / ULP_F32))),
    'Δ 与 fp32 ulp 的比值必须与图一致',
  );
  assert.ok(page.includes(`${ledgerTotal(LEDGER[0])} bytes/param`), '18 字节账必须出现在正文');
  assert.ok(
    page.includes(`${ledgerTotal(LEDGER[2])} bytes/param`),
    'DP 分片后的字节数必须出现在正文',
  );
  assert.ok(page.includes(`$d=${CFG.dp}$`), '正文必须点明分片行用的 d');

  // 四张图都要真的被正文引用
  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
});
