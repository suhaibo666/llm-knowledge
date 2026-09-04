import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { test } from 'node:test';

const here = path.dirname(fileURLToPath(import.meta.url));
const generator = path.join(here, 'megatron_architecture.mjs');
const committedAsset = path.resolve(
  here,
  '../../../wiki/02_engineering/02_train_frameworks/megatron-lm/assets/megatron_architecture.svg',
);

function renderArchitecture() {
  return spawnSync(process.execPath, [generator], { encoding: 'utf8' });
}

function approximateTextWidth(value, fontSize, latinFactor) {
  return [...value].reduce((width, character) => {
    const isWide = /[\u2e80-\u9fff\uac00-\ud7af]/u.test(character);
    return width + fontSize * (isWide ? 1 : latinFactor);
  }, 0);
}

test('生成器输出按依赖方向排列的七层 Megatron 架构 SVG', () => {
  const result = renderArchitecture();
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^<svg[\s>]/);
  assert.match(result.stdout, /role="img"/);
  assert.match(result.stdout, /aria-label="Megatron-LM 静态软件分层与能力框图"/);

  const layers = [
    '01 场景入口层',
    '02 应用编排层',
    '03 MCore 可组合组件层',
    '04 MCore 分布式执行层',
    '05 加速适配层',
    '06 基础框架层',
    '07 设备与通信基础设施层',
  ];
  let previous = -1;
  for (const layer of layers) {
    const current = result.stdout.indexOf(layer);
    assert.ok(current > previous, `${layer} 缺失或层序错误`);
    previous = current;
  }
});

test('能力框图保留主代码栈、Lite 旁路和外部生态三种边界', () => {
  const result = renderArchitecture();
  assert.equal(result.status, 0, result.stderr);

  for (const label of [
    'megatron/training',
    'train_rl.py',
    'models · transformer · ssm',
    'datasets · tokenizers · inference',
    'parallel_state · TP · PP · CP · EP',
    'distributed · optimizer · dist-ckpt',
    '并行算子、时序与重叠',
    'extensions · fusions · quantization',
    'CUDA Graph · backend hooks',
    'PyTorch · torch.distributed',
    'CUDA · NCCL · NVIDIA GPU',
    'Megatron Lite 独立实验纵切',
    'native models · runtime · primitives',
    '外部生态适配',
    'Energon · ModelOpt · TRT-LLM · NVRx',
  ]) {
    assert.ok(result.stdout.includes(label), `缺少能力或边界：${label}`);
  }

  assert.match(result.stdout, /实线：主依赖/);
  assert.match(result.stdout, /虚线：共享或适配/);
  assert.doesNotMatch(result.stdout, /(?:NaN|undefined)/);
  assert.match(result.stdout, /<\/svg>\s*$/);
});

test('仓库内 SVG 与生成器输出保持一致', () => {
  const result = renderArchitecture();
  assert.equal(result.status, 0, result.stderr);

  const normalize = (value) => value.replaceAll('\r\n', '\n').trimEnd();
  assert.equal(normalize(readFileSync(committedAsset, 'utf8')), normalize(result.stdout));
});

test('层头和能力块文本不超过各自的几何预算', () => {
  const result = renderArchitecture();
  assert.equal(result.status, 0, result.stderr);

  for (const group of result.stdout.matchAll(/<g data-layer-index="\d+">([\s\S]*?)<\/g>/g)) {
    const headerWidth = Number(group[1].match(/class="layer-head"[^>]*width="([\d.]+)"/)?.[1]);
    assert.ok(Number.isFinite(headerWidth));
    for (const label of group[1].matchAll(/class="(layer-title|layer-summary)"[^>]*>([^<]+)<\/text>/g)) {
      const fontSize = label[1] === 'layer-title' ? 14 : 10.5;
      assert.ok(
        approximateTextWidth(label[2], fontSize, label[1] === 'layer-title' ? 0.56 : 0.47) <=
          headerWidth - 16,
        `层头文本超宽：${label[2]}`,
      );
    }
  }

  const boxPattern = /<rect class="cap-box"[^>]*width="([\d.]+)"[^>]*\/>\s*<text class="cap-title"[^>]*>([^<]+)<\/text>\s*<text class="cap-code"[^>]*>([^<]+)<\/text>/g;
  const boxes = [...result.stdout.matchAll(boxPattern)];
  assert.ok(boxes.length > 0, '未找到能力块');
  for (const [, rawWidth, title, code] of boxes) {
    const budget = Number(rawWidth) - 18;
    assert.ok(approximateTextWidth(title, 12.5, 0.56) <= budget, `能力标题超宽：${title}`);
    assert.ok(approximateTextWidth(code, 9.5, 0.59) <= budget, `代码标签超宽：${code}`);
  }
});
