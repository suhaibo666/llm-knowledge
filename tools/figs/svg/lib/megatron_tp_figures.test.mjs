// 锁住 TP 图示的可执行契约：三张图必须由同一组配置推导出分片尺寸，
// 不能把正文里的 H/tp、heads/tp、h_ffn/tp 手写成彼此可能漂移的数字。
//
// 运行：node --test tools/figs/svg/lib/megatron_tp_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_tp_figures.mjs');
const trackedDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);

test('生成器同步产出矩阵、训练闭环和 SP 前反向图', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-tp-figures-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);

    const matrix = await readFile(join(outputDir, 'megatron_tp_matrix_partition.svg'), 'utf8');
    const layer = await readFile(join(outputDir, 'megatron_tp_transformer_layer.svg'), 'utf8');
    const sequence = await readFile(
      join(outputDir, 'megatron_tp_sequence_parallel.svg'), 'utf8',
    );

    assert.match(matrix, /H=8 · O=12 · tp=2/);
    assert.match(matrix, /W₀: 8×6/);
    assert.match(matrix, /X₀: N×4/);
    assert.match(matrix, /Σ Zᵣ/);
    assert.equal((matrix.match(/data-role="gemm"/g) ?? []).length, 4);
    assert.match(matrix, /data-role="column-default"[\s\S]*默认：不 concat/);
    assert.match(matrix, /data-role="column-optional"[\s\S]*gather_output = true/);

    assert.match(layer, /heads\/rank = 2/);
    assert.match(layer, /FFN\/rank = 8/);
    assert.match(layer, /vocab\/rank = 8/);
    assert.match(layer, /非 SP · MHA 基准/);
    assert.match(layer, /MHA 基准/);
    assert.match(layer, /GeLU preact\/rank = 8/);
    assert.match(layer, /SwiGLU preact\/rank = 16/);
    assert.match(layer, /activation\/rank = 8/);
    assert.match(layer, /每层前向 2 次规约/);
    assert.match(layer, /训练前向/);
    assert.match(layer, /分片 logits/);
    assert.match(layer, /MAX AR \+ 2×SUM AR/);
    assert.match(layer, /逐 token loss/);
    assert.match(layer, /训练反向/);
    assert.match(layer, /CE backward：local logits grad/);
    assert.match(layer, /LM head dgrad AR/);
    assert.match(layer, /Embedding wgrad local/);

    assert.match(sequence, /S=8 · H=8 · tp=2/);
    assert.match(sequence, /rank local：4×B×8/);
    assert.match(sequence, /AG → 8×B×8/);
    assert.match(sequence, /RS → 4×B×8/);
    assert.match(sequence, /Row mapping AG/);
    assert.match(sequence, /saved-input AG/);
    assert.match(sequence, /dgrad RS/);
    assert.match(sequence, /每对前向 AG\+RS；反向 AG\+AG\+RS/);

    const trackedMatrix = await readFile(
      join(trackedDir, 'megatron_tp_matrix_partition.svg'), 'utf8',
    );
    const trackedLayer = await readFile(
      join(trackedDir, 'megatron_tp_transformer_layer.svg'), 'utf8',
    );
    const trackedSequence = await readFile(
      join(trackedDir, 'megatron_tp_sequence_parallel.svg'), 'utf8',
    );
    assert.equal(matrix, trackedMatrix, '已跟踪的矩阵图必须与生成器同步');
    assert.equal(layer, trackedLayer, '已跟踪的 Transformer 图必须与生成器同步');
    assert.equal(sequence, trackedSequence, '已跟踪的 SP 图必须与生成器同步');

    for (const svg of [matrix, layer, sequence]) {
      assert.match(svg, /^<svg[\s\S]*<\/svg>\s*$/);
      assert.doesNotMatch(svg, /undefined|NaN/);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});
