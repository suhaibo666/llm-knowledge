// 锁住模型结构页两张图的可执行契约。
//
// 图 1：插槽占用数（local=6 / transformer_engine=4）必须由 SLOTS 表算出来，而不是手写；
//       两条泳道必须落到同一份规范 checkpoint 键上——这是正文 §2.3.1 与 §4.1 的立论。
// 图 2：一个 token 走完 routing 的每个中间量（sigmoid 分数、组分、top_indices、gather 回
//       无偏置 scores、probs、routing_map 列和）必须与正文 §2.3.3 引用的数值逐个一致；
//       其中「gather 的是 scores 而不是 scores_for_routing」是本页的不变量，必须锁死。
//
// 源码基线 NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a（dev，2026-09-01）
// 推进基线导致插槽表或 router 行为变化时，本测试先红——提醒同时更新页面、图与基线声明。
//
// 运行：node --test tools/figs/svg/lib/megatron_model_structure_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_model_structure_figures.mjs');
const trackedDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '10_megatron_model_structure_analysis.md');

// —— 以下两块由 megatron_cp_figures.test.mjs 移植：画布越界与图/正文漂移的机械判定 ——

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

// 没有任何图元允许溢出画布 —— 溢出等于裁字，肉眼过一遍不可靠，这里机械判定
function assertInsideCanvas(svg, name) {
  const { w, h } = viewBox(svg);
  const rects = svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  );
  for (const [, x, y, rw, rh] of rects) {
    assert.ok(Number(x) >= -1, `${name}: rect 左边越界 x=${x}`);
    assert.ok(Number(y) >= -1, `${name}: rect 上边越界 y=${y}`);
    assert.ok(Number(x) + Number(rw) <= w + 1, `${name}: rect 右边越界 ${x}+${rw} > ${w}`);
    assert.ok(Number(y) + Number(rh) <= h + 1, `${name}: rect 下边越界 ${y}+${rh} > ${h}`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w, `${name}: text x=${x} 越界`);
    assert.ok(Number(y) >= 0 && Number(y) <= h, `${name}: text y=${y} 越界`);
  }
}


const SLOT_NAMES = [
  'input_layernorm',
  'self_attention_hyper_connection',
  'self_attention',
  'self_attn_bda',
  'pre_cross_attn_layernorm',
  'cross_attention_hyper_connection',
  'cross_attention',
  'cross_attn_bda',
  'pre_mlp_layernorm',
  'mlp_hyper_connection',
  'mlp',
  'mlp_bda',
];

test('生成器同步产出插槽拓扑图与 MoE router 路由图', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-model-structure-figures-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);

    const slots = await readFile(join(outputDir, 'megatron_model_spec_slot_topology.svg'), 'utf8');
    const router = await readFile(join(outputDir, 'megatron_moe_router_routing.svg'), 'utf8');

    // --- 图 1：槽的拓扑 ---
    // 12 个槽名一个不少，且按 dataclass 声明顺序出现在两条泳道里。
    for (const name of SLOT_NAMES) {
      assert.equal(
        (slots.match(new RegExp(`>${name}<`, 'g')) ?? []).length, 2,
        `槽 ${name} 应在两条泳道各出现一次`,
      );
    }
    // 占用计数由 SLOTS 算出，正文引用的 6 / 4 必须与图一致。
    assert.match(slots, /非哨兵槽 6 \/ 12/);
    assert.match(slots, /非哨兵槽 4 \/ 12/);
    assert.match(slots, /非哨兵槽 local=6、transformer_engine=4/);
    // 决定拓扑的那个布尔量，及它落到 linear_fc1 上的两条分支。
    assert.match(slots, /LocalSpecProvider/);
    assert.match(slots, /fuse_layernorm_and_linear\(\) → False/);
    assert.match(slots, /linear_fc1 = backend\.column_parallel_linear\(\)/);
    assert.match(slots, /TESpecProvider/);
    assert.match(slots, /fuse_layernorm_and_linear\(\) → True/);
    assert.match(slots, /linear_fc1 = backend\.column_parallel_layer_norm_linear\(\)/);
    // norm 被融进后继 linear：TE 侧两个 norm 槽退化为哨兵。
    assert.match(slots, /norm 已融进 self_attention\.linear_qkv/);
    assert.match(slots, /norm 已融进 mlp\.linear_fc1/);
    assert.match(slots, /TELayerNormColumnParallelLinear/);
    assert.match(slots, /ColumnParallelLinear/);
    assert.match(slots, /IdentityFuncOp/);
    // 治理不变量：两条泳道写出同一份规范键。
    assert.match(slots, /"input_layernorm\." → "self_attention\.linear_qkv\.layer_norm_"/);
    assert.match(slots, /"pre_mlp_layernorm\." → "mlp\.linear_fc1\.layer_norm_"/);
    assert.equal(
      (slots.match(/self_attention\.linear_qkv\.layer_norm_weight/g) ?? []).length, 2,
      '规范键必须同时出现在规范列与 TE 原生列',
    );
    assert.equal(
      (slots.match(/mlp\.linear_fc1\.layer_norm_weight/g) ?? []).length, 2,
      '规范键必须同时出现在规范列与 TE 原生列',
    );
    // 代价：三个恒空槽。
    assert.match(slots, /代价：3 个恒空槽被无条件调用/);
    assert.match(slots, /pre_cross_attn_layernorm \/ cross_attention \/ cross_attn_bda/);

    // --- 图 2：router 的一次完整路由 ---
    assert.match(router, /num_moe_experts=8 · moe_router_topk=2 · moe_router_num_groups=4 · moe_router_group_topk=2/);
    assert.match(router, /scores = sigmoid\(logits\.float\(\)\)/);
    // t0 的打分：sigmoid(2.0)=0.881、sigmoid(1.2)=0.769、sigmoid(1.0)=0.731。
    assert.match(router, /0\.881/);
    assert.match(router, /0\.769/);
    assert.match(router, /0\.731/);
    // 偏置只加在选择用的分数上：e4 由 0.731 抬到 1.031。
    assert.match(router, /scores_for_routing = scores \+ expert_bias\.float\(\)/);
    assert.match(router, /1\.031/);
    // 组分与被选中的组（group 0 与 group 2）。
    assert.match(router, /组分 0\.881 {2}✓选中/);
    assert.match(router, /组分 1\.031 {2}✓选中/);
    assert.match(router, /组分 0\.569/);
    assert.match(router, /组分 0\.182/);
    assert.match(router, /−inf/);
    // 选择结果，以及去掉偏置后的反事实结果。
    assert.match(router, /top_indices = \[4, 0\]/);
    assert.match(router, /top_indices = \[0, 2\]，即 e0 与 e2/);
    assert.match(router, /选中 group 0, 1（而不是 0, 2）/);
    // 本页不变量：gather 的对象是 scores，不是 scores_for_routing。
    assert.match(router, /torch\.gather\(scores, 1, top_indices\)/);
    assert.match(router, /= \[0\.731, 0\.881\]/);
    assert.match(router, /gather 不读这一行/);
    assert.match(router, /不变量：e4 的权重取 0\.731 而非 1\.031/);
    // 归一化后的 probs 与其分母。
    assert.match(router, /\[0\.454, 0\.546\]/);
    assert.match(router, /分母 1\.612/);
    // routing_map 的列和 = 下一轮偏置更新的输入；3 token × topk=2 共 6 个非零。
    assert.match(router, /local_tokens_per_expert \+= routing_map\.sum\(dim=0\)/);
    for (const prob of ['0.546', '0.454', '0.598', '0.402', '0.486', '0.514']) {
      assert.match(router, new RegExp(prob.replace('.', '\\.')), `routing_probs 应含 ${prob}`);
    }
    // 边界：本图停在 (probs, routing_map)，不画 dispatch。
    assert.match(router, /交接：\(probs, routing_map\)/);
    assert.doesNotMatch(router, /token_dispatcher|permute|MoEAlltoAll/);

    // --- 与仓库里已跟踪的资产一致 ---
    const trackedSlots = await readFile(
      join(trackedDir, 'megatron_model_spec_slot_topology.svg'), 'utf8',
    );
    const trackedRouter = await readFile(
      join(trackedDir, 'megatron_moe_router_routing.svg'), 'utf8',
    );
    assert.equal(slots, trackedSlots, '已跟踪的插槽拓扑图必须与生成器同步');
    assert.equal(router, trackedRouter, '已跟踪的 router 图必须与生成器同步');

    for (const svg of [slots, router]) {
      assert.match(svg, /^<svg[\s\S]*<\/svg>\s*$/);
      assert.doesNotMatch(svg, /undefined|NaN/);
      assert.doesNotMatch(svg, /\[\[/, '标注里不得混入 wiki 链接语法');
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('插槽占用与 router 的中间量由数据推导，改一个输入会一起变', async () => {
  // 生成器内部对占用计数和列和都有断言；这里再从外部证明它确实是「算出来的」：
  // 把 SLOTS/CFG2 换掉会让生成器直接抛错，而不是安静地画出一张与正文不符的图。
  const source = await readFile(generator, 'utf8');
  assert.match(source, /slot occupancy drifted/);
  assert.match(source, /dead slot count drifted/);
  assert.match(source, /routing_map column sums disagree with topk × num_tokens/);
  assert.doesNotMatch(source, /非哨兵槽 6 \/ 12/, '占用数必须来自 filledCount，不得硬编码进标注');
});

test('页面正文引用图，且图上的数值在正文中出现', async () => {
  const page = await readFile(pagePath, 'utf8');

  // 两张图都必须以外链 svg 的方式被正文引用，且不许内联 <svg>（内联会被 docs-site 打坏）
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_model_spec_slot_topology\.svg\)/);
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_moe_router_routing\.svg\)/);
  assert.doesNotMatch(page, /<svg/);

  // 图上算出来的关键量必须在正文里出现，防止图与正文各自漂移
  for (const needle of [
    'top_indices',
    'routing_map',
    'sharded_state_dict_keys_map',
  ]) {
    assert.ok(page.includes(needle), `正文必须引用图上的 ${needle}`);
  }
});

test('两张图都不越界', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'fig-bounds-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    for (const name of ['megatron_model_spec_slot_topology', 'megatron_moe_router_routing']) {
      const svg = await readFile(join(outputDir, `${name}.svg`), 'utf8');
      assertInsideCanvas(svg, name);
      assert.doesNotMatch(svg, /\[\[[^\]]{3,}\]\]/, `${name}: wikilink 不许漏进标注`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});
