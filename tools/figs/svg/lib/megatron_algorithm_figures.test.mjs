// 锁住 14/15/16 三页算法图的可执行契约。
//
// 每张图都必须由生成器产出，并把同一个具体算例的 ownership、前反向、通信与成本
// 写进最终 SVG；测试对用户可见结果做断言，并校验已跟踪资产没有漂移。
//
// 运行：node --test tools/figs/svg/lib/megatron_algorithm_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const scriptDir = join(here, '..');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pageDir = dirname(assetDir);

function run(script, args = []) {
  return spawnSync(process.execPath, [join(scriptDir, script), ...args], { encoding: 'utf8' });
}

async function assertTracked(name, generated) {
  const tracked = await readFile(join(assetDir, name), 'utf8');
  assert.equal(normalizeEol(generated), normalizeEol(tracked), `${name} 必须与生成器同步`);
}

test('EP 图复演 route 闭环、三种 dispatcher 和四种 Flex backend 数据面', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-ep-figures-'));
  try {
    const result = run('megatron_ep_figures.mjs', [outputDir]);
    assert.equal(result.status, 0, result.stderr || result.stdout);

    const route = await readFile(join(outputDir, 'megatron_ep_route_compute_combine.svg'), 'utf8');
    assert.match(route, /T_global=4 · T_local=2\/rank · E=4 · top-k=2 · EP=2/);
    assert.match(route, /8 条 route edge/);
    assert.match(route, /4 条跨 rank(?: edge)?/);
    assert.match(route, /t3 → e0/);
    assert.match(route, /y_i = Σ_e p_i,e f_e\(t_i\)/);
    assert.match(route, /dY → combine⁻¹ → expert backward[^<]*→ dispatch⁻¹ → dX/);

    const variants = await readFile(join(outputDir, 'megatron_ep_dispatcher_variants.svg'), 'utf8');
    assert.match(variants, /AllGather/);
    assert.match(variants, /AllToAll/);
    assert.match(variants, /Flex/);
    assert.match(variants, /input_splits = \[2, 2\]/);
    assert.match(variants, /reverse A2A/);

    const flexBackends = await readFile(
      join(outputDir, 'megatron_ep_flex_backends.svg'), 'utf8',
    );
    for (const backend of ['DeepEP', 'DeepEPv2', 'HybridEP', 'NCCL-EP']) {
      assert.match(flexBackends, new RegExp(backend));
    }
    assert.doesNotMatch(flexBackends, /backend-defined/);

    await assertTracked('megatron_ep_route_compute_combine.svg', route);
    await assertTracked('megatron_ep_dispatcher_variants.svg', variants);
    await assertTracked('megatron_ep_flex_backends.svg', flexBackends);
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('PP 主调度图显式给出 PP=1 控制流，且不伪装成与 PP=4 同壁钟比较', async () => {
  const result = run('megatron_vpp_vs_1f1b.mjs');
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /PP=1 \/ no-pipeline 控制流/);
  assert.match(result.stdout, /F0 → B0 → F1 → B1/);
  assert.match(result.stdout, /控制流参考，不与下方 t_f 共用壁钟/);
  await assertTracked('megatron_pp_vpp_vs_1f1b.svg', result.stdout);
});

test('combined-1F1B 图复演相邻 microbatch 的前反向共调度及 EP A2A 覆盖窗口', async () => {
  const result = run('megatron_combined_1f1b.mjs');
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /Phase 0：F\(m0\)/);
  assert.match(result.stdout, /Phase 1：B\(m0\) ∥ F\(m1\)/);
  assert.match(result.stdout, /Phase 4：B\(m3\)/);
  assert.match(result.stdout, /EP A2A/);
  assert.match(result.stdout, /attention\/MLP compute/);
  assert.match(result.stdout, /extra warmup = 1/);
  assert.match(result.stdout, /checkpoint_activations_microbatch = None/);
  await assertTracked('megatron_pp_combined_1f1b.svg', result.stdout);
});

test('PP 图独立复演 multi-module bridge 和三种 P2P transport/两种 backend', async () => {
  const multiModule = run('megatron_pp_multimodule_bridge.mjs');
  assert.equal(multiModule.status, 0, multiModule.stderr || multiModule.stdout);
  for (const term of [
    'MultiModuleProcessGroupCollection', 'BridgeCommunicator', 'fan-in', 'fan-out',
    'split', 'cat', 'forward', 'backward',
  ]) {
    assert.match(multiModule.stdout, new RegExp(term));
  }
  await assertTracked('megatron_pp_multimodule_bridge.svg', multiModule.stdout);

  const transports = run('megatron_pp_p2p_transports.mjs');
  assert.equal(transports.status, 0, transports.stderr || transports.stdout);
  for (const term of [
    'ring_exchange', 'batch_isend_irecv', 'isend', 'irecv', 'NCCL', 'UCC', 'wait',
  ]) {
    assert.match(transports.stdout, new RegExp(term));
  }
  await assertTracked('megatron_pp_p2p_transports.svg', transports.stdout);
});

test('DistOpt 图复演跨参数边界 range、RS-update-AG 可见性与三条实现路径', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-distopt-figures-'));
  try {
    const result = run('megatron_distributed_optimizer_figures.mjs', [outputDir]);
    assert.equal(result.status, 0, result.stderr || result.stdout);

    const ranges = await readFile(join(outputDir, 'megatron_distopt_flat_buffer_ranges.svg'), 'utf8');
    assert.match(ranges, /N=16 · d=4 · shard=4/);
    assert.match(ranges, /p world \[2, 7\)/);
    assert.match(ranges, /rank0: p\[0, 2\)/);
    assert.match(ranges, /rank1: p\[2, 5\)/);
    assert.match(ranges, /parameter boundary ≠ shard boundary/);

    const lifecycle = await readFile(join(outputDir, 'megatron_distopt_rs_update_ag.svg'), 'utf8');
    assert.match(lifecycle, /finish_grad_sync/);
    assert.match(lifecycle, /reduce-scatter \(RS\)/);
    assert.match(lifecycle, /local update/);
    assert.match(lifecycle, /all-gather \(AG\)/);
    assert.match(lifecycle, /forward pre-hook/);
    assert.match(lifecycle, /同步参数 AG/);
    assert.match(lifecycle, /overlap_param_gather/);

    const paths = await readFile(join(outputDir, 'megatron_distopt_live_paths.svg'), 'utf8');
    assert.match(paths, /native DistributedOptimizer/);
    assert.match(paths, /Torch FSDP2/);
    assert.match(paths, /Megatron-FSDP/);
    assert.match(paths, /persistent full parameter/);
    assert.match(paths, /transient full unit/);
    assert.match(paths, /pre-backward AG \/ unshard/);
    assert.match(paths, /gradient reduce-scatter/);
    assert.match(paths, /sharded grad → local optimizer/);
    assert.match(paths, /依赖API契约对照，未核验PyTorch内核/);
    assert.match(paths, /邻页契约对照：no_shard复制P\/G\/O/);
    assert.match(paths, /updated owner shard/);
    assert.match(
      paths,
      /<g data-strategy="optim">[\s\S]*?P,G buffer full[\s\S]*?main\/O ≈ 1\/d[\s\S]*?F\/B: full P[\s\S]*?no compute-time param AG[\s\S]*?gradient RS[\s\S]*?local g_r[\s\S]*?local main update[\s\S]*?parameter AG[\s\S]*?next-F: full updated P[\s\S]*?<\/g>/,
    );
    assert.match(
      paths,
      /<g data-strategy="optim_grads">[\s\S]*?P full[\s\S]*?G,main\/O ≈ 1\/d[\s\S]*?F\/B: full P[\s\S]*?no compute-time param AG[\s\S]*?gradient RS[\s\S]*?persistent g_r[\s\S]*?local main update[\s\S]*?parameter AG[\s\S]*?next-F: full updated P[\s\S]*?<\/g>/,
    );
    assert.match(
      paths,
      /<g data-strategy="optim_grads_params">[\s\S]*?P,G,main\/O ≈ 1\/d[\s\S]*?transient full unit[\s\S]*?forward AG → F[\s\S]*?post-F reshard[\s\S]*?pre-backward AG → B[\s\S]*?gradient RS[\s\S]*?local update keeps p_r[\s\S]*?next-F AG[\s\S]*?full updated unit[\s\S]*?<\/g>/,
    );
    assert.match(paths, /无固定加速比/);

    const gradPlanes = await readFile(
      join(outputDir, 'megatron_distopt_fp32_rs_hsdp.svg'), 'utf8',
    );
    for (const term of [
      'all-reduce', 'reduce-scatter', 'FP32 accumulation', 'all_to_all_single', 'HSDP',
    ]) {
      assert.match(gradPlanes, new RegExp(term));
    }

    await assertTracked('megatron_distopt_flat_buffer_ranges.svg', ranges);
    await assertTracked('megatron_distopt_rs_update_ag.svg', lifecycle);
    await assertTracked('megatron_distopt_live_paths.svg', paths);
    await assertTracked('megatron_distopt_fp32_rs_hsdp.svg', gradPlanes);
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('14–16 页面实际嵌入算法图，并在正文复演图中的同一组数字', async () => {
  const ep = await readFile(join(pageDir, '14_megatron_ep_analysis.md'), 'utf8');
  assert.match(ep, /assets\/megatron_ep_route_compute_combine\.svg/);
  assert.match(ep, /assets\/megatron_ep_dispatcher_variants\.svg/);
  assert.match(ep, /assets\/megatron_ep_flex_backends\.svg/);
  for (const term of ['_DeepepManager', '_DeepepV2Manager', '_HybridEPManager', '_NCCLEPManager']) {
    assert.match(ep, new RegExp(term));
  }
  assert.match(ep, /T_\{\\mathrm\{local\}\}=2/);
  assert.match(ep, /K=T_\{\\mathrm\{global\}\}\\cdot k=4\\cdot2=8/);
  assert.match(ep, /K_\{\\mathrm\{remote\}\}=4/);
  assert.match(ep, /input_splits=\[2,2\]/);

  const pp = await readFile(join(pageDir, '15_megatron_pp_schedulers_analysis.md'), 'utf8');
  assert.match(pp, /assets\/megatron_pp_vpp_vs_1f1b\.svg/);
  assert.match(pp, /assets\/megatron_pp_p2p_overlap\.svg/);
  assert.match(pp, /assets\/megatron_pp_combined_1f1b\.svg/);
  assert.match(pp, /assets\/megatron_pp_multimodule_bridge\.svg/);
  assert.match(pp, /assets\/megatron_pp_p2p_transports\.svg/);
  for (const term of [
    'BridgeCommunicator', 'ring_exchange', 'batch_isend_irecv', 'UCC',
  ]) {
    assert.match(pp, new RegExp(term));
  }
  assert.match(pp, /22t_f/);
  assert.match(pp, /19t_f/);
  assert.match(pp, /combine_bwd\(m0\)/);
  assert.match(pp, /combine_fwd\(m1\)/);

  const distopt = await readFile(
    join(pageDir, '16_megatron_distributed_optimizer_analysis.md'), 'utf8',
  );
  assert.match(distopt, /assets\/megatron_distopt_flat_buffer_ranges\.svg/);
  assert.match(distopt, /assets\/megatron_distopt_rs_update_ag\.svg/);
  assert.match(distopt, /assets\/megatron_distopt_live_paths\.svg/);
  assert.match(distopt, /assets\/megatron_distopt_fp32_rs_hsdp\.svg/);
  assert.match(distopt, /reduce_scatter_with_fp32_accumulation/);
  assert.match(distopt, /all_to_all_single/);
  assert.match(distopt, /LayerWiseDistributedOptimizer/);
  assert.match(distopt, /gbuf_world=\[2,4\)/);
  assert.match(distopt, /finish_grad_sync/);
  assert.match(distopt, /forward pre-hook/);
  // 16 拥有 wrapper 选择/交接契约；FSDP 内部 hook 的唯一 owner 是 36。
  // 同时读两页，防止重构只删除细节而没有真实落到被指向的 owner。
  assert.match(distopt, /Torch FSDP2[^\n]*已发布契约/);
  assert.match(distopt, /hook[^\n]*内部状态机[^\n]*36_megatron_fsdp_analysis/);
  for (const strategy of ['no_shard', 'optim', 'optim_grads', 'optim_grads_params']) {
    assert.ok(distopt.includes(`\`${strategy}\``), `16 必须保留 ${strategy} 的选择契约`);
  }
  const fsdp = await readFile(join(pageDir, '36_megatron_fsdp_analysis.md'), 'utf8');
  assert.match(fsdp, /`PRE_BACKWARD`/);
  assert.match(fsdp, /pre-backward[^\n]*AG 本 unit/);
  for (const strategy of ['optim', 'optim_grads', 'optim_grads_params']) {
    assert.ok(fsdp.includes(strategy), `36 必须承接 ${strategy} 的内部机制`);
  }
});

// Git checkouts may use CRLF; compare SVG content with only line endings normalized.
function normalizeEol(text) { return text.replace(/\r\n/g, '\n'); }
