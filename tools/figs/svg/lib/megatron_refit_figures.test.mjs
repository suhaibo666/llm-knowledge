// 锁住 refit 图示的可执行契约：LCM tiling 复刻自 planner.py::_emit_lcm_block_ops，
// 微块划分与逐块来源全部算出来，并与 30_megatron_rl_posttraining_consistency_analysis.md 正文对齐。
//
// 运行：node --test tools/figs/svg/lib/megatron_refit_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, PLAN, META, SRC_FANIN, WB_2D, WB_1D, QUANT_2D, QUANT_1D, emitLcmBlockOps,
} from '../megatron_refit_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_refit_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '30_megatron_rl_posttraining_consistency_analysis.md',
);

const NAMES = [
  'megatron_refit_lcm_tiling.svg',
  'megatron_refit_execution.svg',
  'megatron_refit_mxfp8_scale.svg',
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

test('LCM tiling：六个量与算例一致', () => {
  assert.equal(META.Ns, CFG.srcWorld * CFG.srcStride);
  assert.equal(META.Nd, CFG.dstWorld * CFG.dstStride);
  assert.equal(META.L, 12);
  assert.equal(META.unit, CFG.fullLen / META.L);
  assert.equal(META.cps, META.L / META.Ns);
  assert.equal(META.cpd, META.L / META.Nd);
  assert.equal(META.segSrc, CFG.fullLen / CFG.srcWorld, '一个源分片的长度');
  assert.equal(META.segDst, CFG.fullLen / CFG.dstWorld, '一个目标分片的长度');
  // 每个目标分片恰好收 cpd 个微块
  PLAN.forEach((p, r) => assert.equal(p.ops.length, META.cpd, `dst ${r} 的微块数`));
  assert.deepEqual(SRC_FANIN, [2, 2, 2], '本例每个推理分片各自从两个训练分片取数');
});

test('LCM tiling：搬运是恒等的，且每个微块不跨分片', () => {
  const covered = new Array(CFG.fullLen).fill(-1);
  PLAN.forEach((p, dstRank) => {
    for (const op of p.ops) {
      const dstGlobal = dstRank * META.segDst + op.dstStart;
      const srcGlobal = op.srcRank * META.segSrc + op.srcStart;
      assert.equal(srcGlobal, dstGlobal, '同一个全局下标：布局重映射必须是恒等的');
      // 整块落在一个源分片、一个目标分片内
      assert.equal(Math.trunc(srcGlobal / META.segSrc), op.srcRank);
      assert.equal(Math.trunc((srcGlobal + op.unit - 1) / META.segSrc), op.srcRank);
      assert.equal(Math.trunc(dstGlobal / META.segDst), dstRank);
      assert.equal(Math.trunc((dstGlobal + op.unit - 1) / META.segDst), dstRank);
      for (let i = 0; i < op.unit; i += 1) {
        assert.equal(covered[dstGlobal + i], -1, '不得重复覆盖');
        covered[dstGlobal + i] = dstRank;
      }
    }
  });
  assert.ok(covered.every((c) => c >= 0), '每个元素都必须被覆盖');
});

test('LCM tiling：整除守卫与其它 TP 组合', () => {
  // 不整除即拒绝，不做静默 padding
  assert.throws(
    () => emitLcmBlockOps({ ...CFG, fullLen: CFG.fullLen + 1, dstLocalRank: 0 }),
    /不能被 LCM/,
  );
  // TP 8 → TP 2：整除关系下每个目标分片正好吃满 4 个源分片
  const eight = emitLcmBlockOps({
    fullLen: 32, srcWorld: 8, dstWorld: 2, srcStride: 1, dstStride: 1, dstLocalRank: 0,
  });
  assert.equal(eight.L, 8, 'lcm(8,2)=8');
  assert.equal(eight.unit, 4);
  assert.equal(new Set(eight.ops.map((o) => o.srcRank)).size, 4);
  // 反向：TP 2 → TP 8，每个目标分片只从一个源分片取
  const two = emitLcmBlockOps({
    fullLen: 32, srcWorld: 2, dstWorld: 8, srcStride: 1, dstStride: 1, dstLocalRank: 5,
  });
  assert.equal(new Set(two.ops.map((o) => o.srcRank)).size, 1);
});

test('MXFP8：两种 scale 布局的量化次数', () => {
  assert.equal(QUANT_2D, CFG.slices, '2D scale 每片到达即量化');
  assert.equal(QUANT_1D, 1, '1D swizzled scale 只在攒齐后量化一次');
  assert.equal(WB_1D.filter((e) => !e.quantized).length, CFG.slices - 1);
  assert.equal(WB_2D.length, WB_1D.length, '两条路径经历同样多的到达事件');
});

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-refit-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [lcm, chain, scale] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1：原理图 ----
  assert.ok(lcm.includes(`L = lcm(Ns, Nd) = ${META.L}`));
  assert.ok(lcm.includes(`unit = 全长 / L = ${META.unit}`));
  assert.ok(lcm.includes(`cps = L / Ns = ${META.cps}`));
  assert.ok(lcm.includes(`cpd = L / Nd = ${META.cpd}`));
  assert.ok(lcm.includes('RuntimeError'), '整除守卫必须写在图上');
  for (let r = 0; r < CFG.srcWorld; r += 1) assert.ok(lcm.includes(`>src ${r}<`), `缺 src ${r}`);
  for (let r = 0; r < CFG.dstWorld; r += 1) assert.ok(lcm.includes(`>dst ${r}<`), `缺 dst ${r}`);
  // 逐块来源标注必须与复刻结果一一对应
  PLAN.forEach((p) => {
    p.ops.forEach((o) => {
      assert.ok(lcm.includes(`>s${o.srcRank}+${o.srcStart}<`), `缺来源标注 s${o.srcRank}+${o.srcStart}`);
    });
  });
  assert.ok(!lcm.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 2：执行链 ----
  for (const step of ['swap_model_weights', '_build_or_get_plan', 'execute_reshard_plan', 'service.run()', 'quantize_()']) {
    assert.ok(chain.includes(step), `图 2 缺 ${step}`);
  }
  assert.ok(chain.includes('第 1 次 synchronize + barrier'));
  assert.ok(chain.includes('第 2 次 synchronize'));
  assert.ok(chain.includes('direct / transform / copy'));
  assert.ok(chain.includes('CUDA Graph'));

  // ---- 图 3：scale ----
  assert.ok(scale.includes(`2D scale：量化 ${QUANT_2D} 次`));
  assert.ok(scale.includes(`1D swizzled scale：量化 ${QUANT_1D} 次`));
  assert.equal((scale.match(/立即量化，写回 data\/scale 的对应行/g) ?? []).length, CFG.slices);
  assert.equal((scale.match(/copy 进 BF16 累积 buffer，先不量化/g) ?? []).length, CFG.slices - 1);
  assert.ok(scale.includes('NotImplementedError'));

  NAMES.forEach((name, i) => assertInsideCanvas([lcm, chain, scale][i], name));

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      [lcm, chain, scale][i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_refit_figures.mjs`,
    );
  });
});

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  assert.ok(
    page.includes(`训练侧 TP=${CFG.srcWorld} 的权重要搬进推理侧 TP=${CFG.dstWorld}`),
    '正文必须点明算例的两侧 TP',
  );
  assert.ok(page.includes(`全局长度为 ${CFG.fullLen}`), '正文必须给出算例全长');
  assert.ok(page.includes(`$N_s=${META.Ns}$、$N_d=${META.Nd}$、$L=${META.L}$`), 'LCM 三量必须与图一致');
  assert.ok(page.includes(`\`unit\`$=${META.unit}$`), 'unit 必须与图一致');
  assert.ok(page.includes(`\\text{cps}=L/N_s=${META.cps}$ 个微块（共 ${META.segSrc} 个元素）`));
  assert.ok(page.includes(`\\text{cpd}=L/N_d=${META.cpd}$ 个微块（共 ${META.segDst} 个元素）`));
  assert.ok(
    page.includes(`每个推理分片各自从 ${SRC_FANIN[0]} 个训练分片取数`),
    '扇入必须与图一致',
  );
  assert.ok(page.includes(`本例 ${CFG.slices} 片到达即量化 ${QUANT_2D} 次`));
  assert.ok(page.includes(`本例 ${CFG.slices} 片到达但只量化 ${QUANT_1D} 次`));

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
});
