// 锁住作业韧性图示的可执行契约：判据、闸门与去重全部由复刻的源码算法算出，
// 且与 27_megatron_job_resilience_analysis.md 正文引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_job_resilience_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, FAULTY_MAD, FAULTY_SIGMA, HEALTHY_MAD, GATE_RESULTS,
  SAVES_DEDUP, SAVES_NO_DEDUP, COLLIDE_ITER,
  madVerdict, sigmaVerdict, timeoutGates, decideExit,
} from '../megatron_job_resilience_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_job_resilience_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '27_megatron_job_resilience_analysis.md',
);

const NAMES = [
  'megatron_resilience_lifecycle.svg',
  'megatron_resilience_ft_sections.svg',
  'megatron_resilience_sniff_outlier.svg',
  'megatron_resilience_exit_paths.svg',
];

function assertInsideCanvas(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(m, `${name}: SVG 必须声明 viewBox`);
  const w = Number(m[1]);
  const h = Number(m[2]);
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

test('离群判据：MAD 抓到坏卡，均值±σ 抓不到；下界挡住健康集群的误报', () => {
  assert.equal(FAULTY_MAD.median, 149.5);
  assert.equal(FAULTY_MAD.mad, 2);
  assert.equal(FAULTY_MAD.minDeviation.toFixed(2), '14.95');
  assert.equal(FAULTY_MAD.threshold, FAULTY_MAD.minDeviation, 'threshold 由下界主导，不是 MAD');
  assert.ok(FAULTY_MAD.threshold > FAULTY_MAD.mad);
  assert.deepEqual(FAULTY_MAD.outliers.map(([i]) => i), [6, 7]);

  // 被否掉的替代：离群点把自己的判据撑开了
  assert.equal(Number(FAULTY_SIGMA.mean.toFixed(2)), 136.38);
  assert.equal(Number(FAULTY_SIGMA.sd.toFixed(2)), 24.53);
  assert.equal(Number(FAULTY_SIGMA.threshold.toFixed(2)), 49.06);
  assert.equal(FAULTY_SIGMA.outliers.length, 0, `±${CFG.sigma}σ 一张都判不出来`);
  // 两张坏卡相对均值的偏离都够不到 2σ
  for (const v of [92, 96]) {
    assert.ok(Math.abs(v - FAULTY_SIGMA.mean) < FAULTY_SIGMA.threshold);
  }

  // 健康集群：下界把噪声挡在外面，去掉下界就误报
  assert.equal(HEALTHY_MAD.median, 150);
  assert.equal(HEALTHY_MAD.mad, 1);
  assert.equal(HEALTHY_MAD.threshold, 15);
  assert.equal(HEALTHY_MAD.outliers.length, 0);
  assert.equal(HEALTHY_MAD.outliersNoFloor.length, 2, '纯 MAD 判会误报两根');

  // 判据本身不是常数拟合：把坏卡拉得更极端，MAD 判据依然命中同样两个
  const worse = madVerdict([152, 149, 151, 150, 153, 148, 40, 44], CFG.outlierMinDeviationFrac);
  assert.deepEqual(worse.outliers.map(([i]) => i), [6, 7]);
  // 而 σ 判据的阈值会跟着被撑得更大
  const worseSigma = sigmaVerdict([152, 149, 151, 150, 153, 148, 40, 44], CFG.sigma);
  assert.ok(worseSigma.threshold > FAULTY_SIGMA.threshold, '离群越极端，σ 阈值越宽');
});

test('阈值闸门：五个场景各自允许更新哪几段', () => {
  const byTag = Object.fromEntries(GATE_RESULTS.map((g) => [g.tag, g.out]));
  assert.deepEqual(byTag['刚起步'].sections, ['setup'], '样本不足 + 未存档');
  assert.deepEqual(byTag['稳态 + 同步存档'].sections, ['setup', 'step', 'checkpointing']);
  assert.deepEqual(byTag['稳态 + 异步存档'].sections, ['setup', 'step'], '异步存档下 ckpt 段不更新');
  assert.deepEqual(byTag['从本地快照恢复'].sections, ['step', 'checkpointing'], '本地快照不解锁 setup');
  assert.equal(byTag['收尾（shutdown）'].outOfSection, true);
  for (const tag of ['刚起步', '稳态 + 同步存档', '稳态 + 异步存档', '从本地快照恢复']) {
    assert.equal(byTag[tag].outOfSection, false, 'out-of-section 只在 shutdown 时更新');
  }
  // 样本下限确实是 16，不是随手写的数
  assert.equal(
    timeoutGates({ persistentLoaded: true, trIters: CFG.minItersForStepTimeout - 1, checkpoints: 1, asyncCkpt: false, closing: false })
      .sections.includes('step'),
    false,
  );
  assert.equal(
    timeoutGates({ persistentLoaded: true, trIters: CFG.minItersForStepTimeout, checkpoints: 1, asyncCkpt: false, closing: false })
      .sections.includes('step'),
    true,
  );
});

test('退出去重：碰撞点存 1 次而不是 2 次', () => {
  const collide = decideExit(COLLIDE_ITER, { save: true, signal: false, durationHit: false, dedup: true });
  assert.deepEqual(collide.saves, ['periodic'], '去重后只存一次');
  assert.equal(collide.exit, true);
  const noDedup = decideExit(COLLIDE_ITER, { save: true, signal: false, durationHit: false, dedup: false });
  assert.deepEqual(noDedup.saves, ['periodic', 'iteration'], '不去重会存两次');
  assert.equal(SAVES_NO_DEDUP - SAVES_DEDUP, 1);
  // ② 与 ②′ 是 elif：save_interval 命中时不再走非持久那条
  assert.deepEqual(
    decideExit(CFG.saveInterval, { save: true, signal: false, durationHit: false, dedup: true }).saves,
    ['periodic'],
  );
  assert.deepEqual(
    decideExit(CFG.nonPersistentSaveInterval, { save: true, signal: false, durationHit: false, dedup: true }).saves,
    ['non-persistent'],
  );
  // ① 信号路径不置 saved_checkpoint，直接退出
  assert.deepEqual(
    decideExit(COLLIDE_ITER, { save: true, signal: true, durationHit: false, dedup: true }).saves,
    ['signal'],
  );
});

test('生成器同步产出四张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-resilience-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [lifecycle, sections, sniff, exit] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1：五段与 deadline ----
  for (const stage of ['① 观测', '② 判定', '③ 中止', '④ 清理', '⑤ 重入']) {
    assert.ok(lifecycle.includes(stage), `图 1 缺 ${stage}`);
  }
  assert.ok(lifecycle.includes('timeout = 10 s'), '清理段的 deadline 必须画出来');
  assert.ok(lifecycle.includes('AbortTransformerEngine'));
  assert.ok(lifecycle.includes('maybe_force_nccl_backend_init'));
  // 两个平面都要出现，且数值面明确标成不覆盖
  assert.ok(lifecycle.includes('作业面：进程 / 通信域 / 硬件 —— 本页'));
  assert.ok(lifecycle.includes('本页不覆盖'));

  // ---- 图 2：section 与闸门 ----
  assert.ok(sections.includes(`warmup ×${CFG.numWarmupIters}：不开 section`));
  assert.ok(sections.includes('start_section("setup")'));
  assert.ok(sections.includes('step（eval 也记这里）'), 'eval 与训练共用 step 段');
  assert.ok(sections.includes('calculate_and_set_'), '依赖边界必须写在图上');
  // 判定表的格子数 = 场景数 × 4 行；「更新」的总数必须与复刻结果一致
  const expectUpdate = GATE_RESULTS.reduce(
    (n, g) => n + g.out.sections.length + (g.out.outOfSection ? 1 : 0),
    0,
  );
  assert.equal((sections.match(/>更新</g) ?? []).length, expectUpdate);
  assert.equal(
    (sections.match(/>不更新</g) ?? []).length,
    GATE_RESULTS.length * 4 - expectUpdate,
  );

  // ---- 图 3：原理图 ----
  assert.ok(sniff.includes(`median 149.5 ± threshold 14.95`));
  assert.ok(sniff.includes('带太宽，一根都没判出'));
  assert.ok(sniff.includes('去掉下界会误报这根'));
  for (const v of CFG.faulty) assert.ok(sniff.includes(`>${v}<`), `缺读数 ${v}`);
  // 被判离群的两根用代价色，其余不用
  assert.equal(
    (sniff.match(/<text class="costtx" x="\d+(?:\.\d+)?" y="\d+(?:\.\d+)?" text-anchor="start">9[26]<\/text>/g) ?? []).length,
    2,
    '恰好两根被标成离群',
  );
  assert.ok(sniff.includes('不是 0'), '截断的横轴必须自陈');

  // ---- 图 4：退出路径 ----
  for (const tag of ['① 信号', '② 周期存档', '②′ 非持久存档', '③ 时长', '④ 迭代 / 阶段切换']) {
    assert.ok(exit.includes(tag), `图 4 缺路径 ${tag}`);
  }
  assert.ok(exit.includes(`碰撞点：第 ${COLLIDE_ITER} 次迭代`));
  assert.ok(exit.includes(`合计 ${SAVES_DEDUP} 次 vs ${SAVES_NO_DEDUP} 次`));
  assert.ok(exit.includes('MAX all-reduce'));

  NAMES.forEach((name, i) => assertInsideCanvas([lifecycle, sections, sniff, exit][i], name));

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      [lifecycle, sections, sniff, exit][i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_job_resilience_figures.mjs`,
    );
  });
});

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  assert.ok(page.includes(`中位数 ${FAULTY_MAD.median}`), '正文必须给出故障组的中位数');
  assert.ok(page.includes(`MAD ${FAULTY_MAD.mad}`));
  assert.ok(page.includes(FAULTY_MAD.minDeviation.toFixed(2)), '下界数值必须出现在正文');
  assert.ok(page.includes(FAULTY_SIGMA.mean.toFixed(2)), '被拉低的均值必须出现在正文');
  assert.ok(page.includes(FAULTY_SIGMA.sd.toFixed(2)), '被拉大的标准差必须出现在正文');
  assert.ok(page.includes(FAULTY_SIGMA.threshold.toFixed(2)), '2σ 阈值必须出现在正文');
  assert.ok(page.includes(`中位数 ${HEALTHY_MAD.median}`), '健康组的中位数必须出现在正文');
  assert.ok(page.includes(`[${CFG.faulty.join(', ')}]`), '正文必须列出故障组读数');
  assert.ok(page.includes(`[${CFG.healthy.join(', ')}]`), '正文必须列出健康组读数');
  assert.ok(page.includes(`OUTLIER_MIN_DEVIATION_FRAC = ${CFG.outlierMinDeviationFrac.toFixed(2)}`));

  assert.ok(page.includes(`>= ${CFG.minItersForStepTimeout}`), '样本下限必须出现在正文');
  assert.ok(page.includes(`默认 ${CFG.numWarmupIters}`), 'warmup 次数必须出现在正文');

  assert.ok(page.includes(`save_interval=${CFG.saveInterval}`), '算例参数必须出现在正文');
  assert.ok(page.includes(`exit_interval=${CFG.exitInterval}`));
  assert.ok(page.includes(`第 ${COLLIDE_ITER} 次迭代`), '碰撞点必须出现在正文');
  assert.ok(
    page.includes(`${SAVES_DEDUP} 次对 ${SAVES_NO_DEDUP} 次`),
    '去重前后的存档次数必须与图一致',
  );

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
});
