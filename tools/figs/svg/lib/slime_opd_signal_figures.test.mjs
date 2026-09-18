// 锁住 slime OPD 信号原理图的可执行契约：图上每个数字都由同一份 CFG 经源码规则的复现推导，
// 并且必须与 20_slime_on_policy_distillation_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_opd_signal_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG,
  SVG_BASELINE,
  alignSglangTeacher,
  applyOpd,
  fneg,
  fsig,
  maskedWhiten,
  megatronResponsePositions,
  model,
  policyLossGrad,
  render,
  sliceLogProbWithCp,
  vec,
} from '../slime_opd_signal_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_opd_signal_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '20_slime_on_policy_distillation_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_opd_signal.svg');
const PAGE_BASELINE = 'THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e';

const close = (a, b, tol = 1e-9) => Math.abs(a - b) < tol;

test('对齐：丢首项 + 尾部裁剪与 Megatron 的 logits 位移落到同一组 response token', () => {
  const m = model();
  assert.equal(m.P, 2);
  assert.equal(m.R, 3);
  assert.deepEqual(m.sglang.dropped, [-1.2, -0.4, -2.1, -0.3]);
  assert.deepEqual(m.teacher, [-0.4, -2.1, -0.3]);
  assert.deepEqual(m.sglang.cropOnly, m.teacher, '对齐由尾部裁剪决定');
  assert.ok(m.sglang.tensorizable, '[1:] 去掉 None 后才能张量化');
  assert.ok(!alignSglangTeacher(CFG.teacherInputLogProbs, 3).cropOnly.includes(null));
  assert.deepEqual(
    m.positions.map((p) => [p.logit, p.target]),
    [
      [1, 2],
      [2, 3],
      [3, 4],
    ],
  );
  assert.deepEqual(m.positions.map((p) => m.tokens[p.target]), CFG.responseIds);
  // prompt 为空：丢首项后只剩 T−1 项，长度断言失败
  const emptyPrompt = alignSglangTeacher([null, -0.5, -0.6, -0.7], 4);
  assert.equal(emptyPrompt.cropped.length, 3);
  assert.throws(() => sliceLogProbWithCp(emptyPrompt.cropped, 4));
  assert.deepEqual(megatronResponsePositions(1, 4).map((p) => p.target), [1, 2, 3]);
});

test('注入：d̂ = student − teacher，Â = A − λ·d̂；学生项缺失时静默返回', () => {
  const m = model();
  assert.ok(m.reverseKl.every((d, i) => close(d, [-0.5, 0.8, 0.1][i])));
  assert.ok(m.pure.every((a, i) => close(a, [0.5, -0.8, -0.1][i])));
  assert.ok(m.mixed.every((a, i) => close(a, [1.0, -0.3, 0.4][i])));
  assert.deepEqual(applyOpd([0, 0], null, [1, 1], 1).advantages, [0, 0]);
  assert.throws(() => applyOpd([0], [-1], null, 1));
});

test('白化：纯蒸馏里 |λ| 被消去；clip 让越界 token 梯度为 0', () => {
  const m = model();
  assert.equal(vec(m.whitened, 3), '[+0.973, −1.025, +0.051]');
  assert.equal(vec(m.whitenedAlt, 3), vec(m.whitened, 3));
  const flipped = maskedWhiten(m.pure.map((a) => -a), [1, 1, 1], CFG.whitenEpsilon);
  assert.ok(flipped.every((v, i) => close(v, -m.whitened[i], 1e-6)), '负 λ 只翻转符号');
  assert.deepEqual(
    m.clipRows.map((r) => r.grads.map((g) => fneg(g))),
    [
      ['0.80', '0.00'],
      ['1.60', '0.00'],
    ],
  );
  assert.equal(m.clipLow.toFixed(2), '0.80');
  assert.ok(close(policyLossGrad(1.3, 0.5, 0.2), 0), 'Â>0 时 ρ > 1+ε 截断');
  assert.ok(close(policyLossGrad(1.1, 0.5, 0.2), -0.55), '信任域内 g = −ρÂ');
});

test('温度边界：同一 logits 学生 ÷τ 与未缩放 teacher 的差', () => {
  const t = model().temperature;
  assert.equal(fneg(t.studentTau, 3), '−0.314');
  assert.equal(fneg(t.teacherUnscaled, 3), '−0.408');
  assert.equal(fsig(t.gap, 3), '+0.094');
});

test('生成器产出原理图，且与已跟踪的 SVG 一致、几何不越界', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-opd-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_opd_signal.svg'), 'utf8');
    const m = model();
    for (const needle of [
      'None',
      '[−0.40, −2.10, −0.30]',
      '−0.90',
      '+0.80',
      '[+0.973, −1.025, +0.051]',
      'g = 1.60',
      'g = 0.00',
      '+0.094',
      '依赖侧',
      SVG_BASELINE,
    ]) {
      assert.ok(svg.includes(needle), `SVG 必须出现 ${needle}`);
    }
    assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');

    const { texts, rects, W, H } = render(m);
    for (const r of rects) {
      assert.ok(r.x >= 0 && r.y >= 0 && r.x + r.w <= W && r.y + r.h <= H, `rect 越界 ${JSON.stringify(r)}`);
    }
    for (const t of texts) {
      assert.ok(t.x0 >= 0 && t.x1 <= W && t.y <= H, `text 越界：${t.s}`);
    }
    // 文字估算包围盒互不重叠
    for (let i = 0; i < texts.length; i += 1) {
      for (let j = i + 1; j < texts.length; j += 1) {
        const a = texts[i];
        const b = texts[j];
        const overlapX = a.x0 < b.x1 && b.x0 < a.x1;
        const overlapY = a.y - a.size * 0.85 < b.y + b.size * 0.2 && b.y - b.size * 0.85 < a.y + a.size * 0.2;
        assert.ok(!(overlapX && overlapY), `文字重叠：「${a.s}」与「${b.s}」`);
      }
    }
    // 落在非面板框内的文字不越出框右缘
    const boxes = rects.filter((r) => r.cls !== 'panel' && r.w < W);
    for (const t of texts) {
      for (const b of boxes) {
        const inside = t.x0 + 1 >= b.x && t.x0 <= b.x + b.w && t.y >= b.y && t.y <= b.y + b.h;
        if (inside) assert.ok(t.x1 <= b.x + b.w + 0.5, `文字越出框：${t.s}`);
      }
    }

    const tracked = await readFile(trackedSvg, 'utf8');
    assert.equal(tracked, svg, '已跟踪的 SVG 必须由当前生成器重新生成');
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const t = m.temperature;
  for (const needle of [
    PAGE_BASELINE,
    `[None, ${m.sglang.dropped.map((v) => fneg(v)).join(', ')}]`,
    vec(m.sglang.dropped, 2, fneg),
    `\`teacher_log_probs = ${vec(m.teacher, 2, fneg)}\``,
    vec(m.student, 2, fneg),
    `$\\widehat d$ = \`${vec(m.reverseKl)}\``,
    `$\\widehat A$ = \`${vec(m.pure)}\``,
    `$\\widehat A$ = \`${vec(m.mixed)}\``,
    `${vec(m.pureAlt)}`,
    vec(m.whitened, 3),
    `为 ${fneg(m.clipRows[0].grads[0])}（$\\lambda=1$）或 ${fneg(m.clipRows[1].grads[0])}（$\\lambda=2$）`,
    `$\\rho=0.75<1-\\varepsilon=${m.clipLow.toFixed(2)}$`,
    `学生 ${fneg(t.studentTau, 3)}、SGLang teacher ${fneg(t.teacherUnscaled, 3)}`,
    `${fsig(t.gap, 3)}`,
    '--opd-teacher-ckpt-step',
    '--use-rollout-logprobs',
    '--keep-old-actor',
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_opd_signal.svg'), '正文必须引用原理图');
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留 path:line 链接');
  assert.doesNotMatch(page, /681b3adc/, '正文不再引用旧基线');
});
