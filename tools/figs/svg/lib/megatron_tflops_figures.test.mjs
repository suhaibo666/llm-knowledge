// 锁住 TFLOPS 图示的可执行契约：
//   1. JS 复刻的 num_floating_point_operations 必须与冻结基线 85902ef 上 ast 抽出来、
//      不依赖 torch 直接执行的 Python 原函数逐位一致（参考值内嵌；设置 MEGATRON_LM_FROZEN_ROOT
//      指向冻结检出时会现场重跑 Python 再比一次）；
//   2. 三张图上的每个数都由同一份复刻代码算出，与仓库里跟踪的 .svg 一致；
//   3. 32_megatron_tflops_analysis.md 正文引用的数值逐个出现（只改正文、不改图，这里必须红）；
//   4. 图元不出画布、文字不互相压盖。
//
// 运行：node --test tools/figs/svg/lib/megatron_tflops_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  EXAMPLE, TOK, SQ, OLD_TOK, OLD_SQ,
  DENSE_THD, DENSE_OLD, MOE_THD, MOE_OLD, MLA_THD, DSA_ON, DSA_OFF, DSA_LONG, DSA_SHARE,
  BIAS, DSA,
  numFloatingPointOperations, dsaSparseCoreScale, dsaIndexerFlops, numDsaIndexerLayers, isDsaSkipTopkLayer,
  denseArgs, moeArgs, mlaArgs, dsaArgs, fmtInt, fmtSci, fx, pct,
} from '../megatron_tflops_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_tflops_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm',
  '32_megatron_tflops_analysis.md',
);

const NAMES = ['megatron_tflops_ledger.svg', 'megatron_tflops_bias.svg', 'megatron_tflops_dsa.svg'];

// ============================================================================
// Python 参考值：在冻结基线上用 ast 抽出 num_floating_point_operations 及其 4 个模块级 helper、
// is_dsa_skip_topk_layer、is_hybrid_model、is_linear_attention_variant / is_gated_delta_net_variant，
// 在只有 math 的命名空间里执行（无 torch / 无 megatron import）。产生方式：
//   python3 xcheck_flops.py <frozen-worktree@85902ef599ea4eb06ada7567a479c524b605767a> out.json
// 脚本正文见下方 XCHECK_PY；输出原样内嵌于此。
// ============================================================================

const PY_REFERENCE = Object.freeze({
  dense_bshd: 219043332096.0,
  dense_thd: 194280161280.0,
  moe_bshd: 219043332096.0,
  moe_thd: 194280161280.0,
  mla_thd: 168619868160.0,
  dsa_thd_loss_on: 168692461883.07693,
  dsa_thd_loss_off: 167067169083.07693,
  dsa_bshd_L4096: 387941662720.0,
  dsa_share_freq4_off1: 333583482880.0,
  mtp1_dense_thd: 267027087360.0,
  gqa2_thd: 176160768000.0,
  sparse_scale_example: 0.7100591715976331,
  sparse_scale_L4096: 0.12109375,
  indexer_layers_8_f4_o1: 2,
});

const XCHECK_PY = String.raw`
"""Torch-free cross-run of the frozen num_floating_point_operations (see test header)."""
import ast, json, math, sys
from types import SimpleNamespace
from typing import Optional, Tuple

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None


def grab(path, names):
    src = open(f"{ROOT}/{path}", encoding="utf-8").read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(out)
    assert not missing, missing
    return out


ns = {"math": math, "Optional": Optional, "Tuple": Tuple}
srcs = {}
srcs.update(grab("megatron/core/transformer/experimental_attention_variant/dsa.py", ["is_dsa_skip_topk_layer"]))
srcs.update(grab("megatron/training/utils/common_utils.py", ["is_hybrid_model"]))
srcs.update(grab(
    "megatron/core/models/gpt/experimental_attention_variant_module_specs.py",
    ["is_gated_delta_net_variant", "is_linear_attention_variant"],
))
ns["GDN_ATTENTION_VARIANTS"] = ("gdn", "kda")
ns["_DEPRECATED_ATTENTION_VARIANT_ALIASES"] = {"gated_delta_net": "gdn"}
srcs.update(grab(
    "megatron/training/training.py",
    ["_dsa_sparse_core_scale", "_dsa_indexer_flops", "_num_dsa_indexer_layers",
     "_dsv4_hybrid_self_attention_flops", "num_floating_point_operations"],
))
for name, code in srcs.items():
    exec(compile(code, name, "exec"), ns)

f = ns["num_floating_point_operations"]


def base_args(**kw):
    a = SimpleNamespace(
        num_layers=4, hidden_size=512, num_attention_heads=8, seq_length=1024,
        padded_vocab_size=4096, swiglu=True, ffn_hidden_size=1536, kv_channels=64,
        group_query_attention=False, num_query_groups=8, attention_output_gate=False,
        gated_attention_proj_granularity="elementwise", multi_latent_attention=False,
        num_experts=None, moe_layer_freq=1, moe_router_topk=0, moe_ffn_hidden_size=None,
        moe_latent_size=None, moe_shared_expert_intermediate_size=None, mtp_num_layers=None,
        experimental_attention_variant=None, linear_attention_freq=None,
        linear_key_head_dim=None, linear_value_head_dim=None, linear_num_key_heads=None,
        linear_num_value_heads=None, linear_conv_kernel_dim=None,
        q_lora_rank=None, qk_head_dim=None, qk_pos_emb_head_dim=None, kv_lora_rank=None,
        v_head_dim=None, hybrid_layer_pattern=None,
        dsa_indexer_n_heads=None, dsa_indexer_head_dim=None, dsa_indexer_topk=None,
        dsa_indexer_topk_freq=1, dsa_indexer_skip_topk_offset=0, dsa_indexer_loss_coeff=None,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def moe_args(**kw):
    return base_args(num_experts=8, moe_router_topk=2, moe_ffn_hidden_size=512,
                     moe_shared_expert_intermediate_size=512, **kw)


def dsa_args(loss=0.01, **kw):
    return base_args(multi_latent_attention=True, q_lora_rank=128, kv_lora_rank=64,
                     qk_head_dim=48, qk_pos_emb_head_dim=16, v_head_dim=64,
                     experimental_attention_variant="dsa", dsa_indexer_n_heads=4,
                     dsa_indexer_head_dim=32, dsa_indexer_topk=256,
                     dsa_indexer_loss_coeff=loss, **kw)


def mla_args(**kw):
    return base_args(multi_latent_attention=True, q_lora_rank=128, kv_lora_rank=64,
                     qk_head_dim=48, qk_pos_emb_head_dim=16, v_head_dim=64, **kw)


LENS = [768, 512, 384, 256]
TOK = sum(LENS)
SQ = sum(L * L for L in LENS)
B = 2

cases = {
    "dense_bshd": f(base_args(), B),
    "dense_thd": f(base_args(), B, seqlen_squared_sum_in_batch=SQ, total_real_tokens_in_batch=TOK),
    "moe_bshd": f(moe_args(), B),
    "moe_thd": f(moe_args(), B, seqlen_squared_sum_in_batch=SQ, total_real_tokens_in_batch=TOK),
    "mla_thd": f(mla_args(), B, seqlen_squared_sum_in_batch=SQ, total_real_tokens_in_batch=TOK),
    "dsa_thd_loss_on": f(dsa_args(0.01), B, seqlen_squared_sum_in_batch=SQ, total_real_tokens_in_batch=TOK),
    "dsa_thd_loss_off": f(dsa_args(None), B, seqlen_squared_sum_in_batch=SQ, total_real_tokens_in_batch=TOK),
    "dsa_bshd_L4096": f(dsa_args(0.01, seq_length=4096), 1),
    "dsa_share_freq4_off1": f(dsa_args(0.01, num_layers=8, dsa_indexer_topk_freq=4, dsa_indexer_skip_topk_offset=1), B),
    "mtp1_dense_thd": f(base_args(mtp_num_layers=1), B, seqlen_squared_sum_in_batch=SQ, total_real_tokens_in_batch=TOK),
    "gqa2_thd": f(base_args(group_query_attention=True, num_query_groups=2), B, seqlen_squared_sum_in_batch=SQ, total_real_tokens_in_batch=TOK),
    "sparse_scale_example": ns["_dsa_sparse_core_scale"](TOK, SQ, 256),
    "sparse_scale_L4096": ns["_dsa_sparse_core_scale"](4096, 4096 * 4096, 256),
    "indexer_layers_8_f4_o1": ns["_num_dsa_indexer_layers"](8, 1, 4),
}
print(json.dumps(cases, indent=1))
if OUT:
    json.dump(cases, open(OUT, "w"), indent=1)
`;

/** 与 PY_REFERENCE 同名的 14 个用例，全部走 JS 复刻。 */
function portCases() {
  const B = EXAMPLE.buffers;
  return {
    dense_bshd: numFloatingPointOperations(denseArgs(), B).total,
    dense_thd: numFloatingPointOperations(denseArgs(), B, SQ, TOK).total,
    moe_bshd: numFloatingPointOperations(moeArgs(), B).total,
    moe_thd: numFloatingPointOperations(moeArgs(), B, SQ, TOK).total,
    mla_thd: numFloatingPointOperations(mlaArgs(), B, SQ, TOK).total,
    dsa_thd_loss_on: numFloatingPointOperations(dsaArgs(), B, SQ, TOK).total,
    dsa_thd_loss_off: numFloatingPointOperations(dsaArgs({ dsa_indexer_loss_coeff: null }), B, SQ, TOK).total,
    dsa_bshd_L4096: numFloatingPointOperations(dsaArgs({ seq_length: 4096 }), 1).total,
    dsa_share_freq4_off1: numFloatingPointOperations(
      dsaArgs({ num_layers: 8, dsa_indexer_topk_freq: 4, dsa_indexer_skip_topk_offset: 1 }), B,
    ).total,
    mtp1_dense_thd: numFloatingPointOperations(denseArgs({ mtp_num_layers: 1 }), B, SQ, TOK).total,
    gqa2_thd: numFloatingPointOperations(
      denseArgs({ group_query_attention: true, num_query_groups: 2 }), B, SQ, TOK,
    ).total,
    sparse_scale_example: dsaSparseCoreScale(TOK, SQ, 256),
    sparse_scale_L4096: dsaSparseCoreScale(4096, 4096 * 4096, 256),
    indexer_layers_8_f4_o1: numDsaIndexerLayers(8, 1, 4),
  };
}

// ============================================================================
// 版面断言（与生成器内的 assertNoTextOverlap 独立实现一遍，免得生成器改坏了测试跟着坏）
// ============================================================================

const FONT = Object.freeze({ ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 });

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function viewBox(svg) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(m, 'SVG 必须声明 viewBox');
  return { w: Number(m[1]), h: Number(m[2]) };
}

function textBoxes(svg) {
  const boxes = [];
  for (const m of svg.matchAll(/<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g)) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const text = raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
    const w = textWidth(text, size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, text });
  }
  return boxes;
}

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
  for (const [, d] of svg.matchAll(/<path[^>]*? d="([^"]+)"/g)) {
    for (const [, xs, ys] of d.matchAll(/(-?[\d.]+) (-?[\d.]+)/g)) {
      assert.ok(Number(xs) >= -2 && Number(xs) <= w + 2, `${name}: path x=${xs} 越界`);
      assert.ok(Number(ys) >= -2 && Number(ys) <= h + 2, `${name}: path y=${ys} 越界`);
    }
  }
  for (const [, cx, cy] of svg.matchAll(/<circle[^>]*?cx="(-?[\d.]+)" cy="(-?[\d.]+)"/g)) {
    assert.ok(Number(cx) >= 0 && Number(cx) <= w && Number(cy) >= 0 && Number(cy) <= h, `${name}: circle 越界`);
  }
  for (const b of textBoxes(svg)) {
    assert.ok(b.x >= -1 && b.y >= -1 && b.x + b.w <= w + 1 && b.y + b.h <= h + 1, `${name}: 文字盒出画布 "${b.text}"`);
  }
}

function assertNoTextOverlap(svg, name) {
  const boxes = textBoxes(svg);
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      assert.ok(!(dx > 1 && dy > 1), `${name}: 文字重叠 "${a.text}" × "${b.text}"`);
    }
  }
}

// ============================================================================

test('JS 复刻与冻结基线 Python 原函数逐位一致（内嵌参考值）', () => {
  const got = portCases();
  for (const [name, ref] of Object.entries(PY_REFERENCE)) {
    assert.ok(name in got, `缺用例 ${name}`);
    assert.equal(got[name], ref, `${name}: JS=${got[name]} Python=${ref}`);
  }
  // 复刻明确拒绝、而非静默算错的分支
  assert.throws(() => numFloatingPointOperations(denseArgs({ hybrid_layer_pattern: 'M-M-' }), 2), /hybrid_flops/);
  assert.throws(() => numFloatingPointOperations(moeArgs({ moe_latent_size: 256 }), 2), /moe_latent_size/);
  assert.throws(() => numFloatingPointOperations(denseArgs({ experimental_attention_variant: 'gdn' }), 2), /linear attention/);
  assert.throws(() => numFloatingPointOperations(mlaArgs({ experimental_attention_variant: 'dsv4_hybrid' }), 2), /dsv4_hybrid/);
  assert.throws(() => numFloatingPointOperations(denseArgs({ attention_output_gate: true }), 2), /attention_output_gate/);
  assert.throws(() => numFloatingPointOperations(mlaArgs({ group_query_attention: true }), 2), /group_query_attention/);
  assert.throws(() => numFloatingPointOperations(moeArgs({ moe_layer_freq: 'x' }), 2), /moe-layer-freq/);
  assert.throws(() => numFloatingPointOperations(moeArgs({ moe_layer_freq: [1, 0] }), 2), /length of moe_layer_pattern/);
});

test('可选：MEGATRON_LM_FROZEN_ROOT 指向冻结检出时现场重跑 Python 交叉核对', async (t) => {
  const root = process.env.MEGATRON_LM_FROZEN_ROOT;
  if (!root || !existsSync(join(root, 'megatron', 'training', 'training.py'))) {
    t.skip('未设置 MEGATRON_LM_FROZEN_ROOT（应指向 NVIDIA/Megatron-LM@85902ef 的检出）');
    return;
  }
  const head = spawnSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).stdout.trim();
  assert.equal(head, '85902ef599ea4eb06ada7567a479c524b605767a', '检出不在冻结基线上');
  const dir = await mkdtemp(join(tmpdir(), 'megatron-tflops-xcheck-'));
  const script = join(dir, 'xcheck_flops.py');
  const out = join(dir, 'out.json');
  await writeFile(script, XCHECK_PY, 'utf8');
  const run = spawnSync('python3', [script, root, out], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  const live = JSON.parse(await readFile(out, 'utf8'));
  assert.deepEqual(live, { ...PY_REFERENCE }, '现场 Python 输出与内嵌参考值不一致：基线变了，先更新 PY_REFERENCE');
  const got = portCases();
  for (const [name, ref] of Object.entries(live)) assert.equal(got[name], ref, name);
});

test('共用算例的账本结构：token-linear × ΣLᵢ + core × ΣLᵢ² 恰好合成上报值', () => {
  assert.equal(TOK, 1920);
  assert.equal(SQ, 1064960);
  assert.equal(OLD_TOK, 2048);
  assert.equal(OLD_SQ, 2097152);
  const R = DENSE_THD.rows;
  // 每层系数：×2 FMA ×3 fwd/wgrad/dgrad 之后
  assert.equal(R.qkvPerLayer, 6 * EXAMPLE.hidden * 3 * EXAMPLE.hidden);
  assert.equal(R.outPerLayer, 6 * EXAMPLE.hidden * EXAMPLE.hidden);
  assert.equal(R.corePerLayer, 6 * EXAMPLE.hidden); // 因果 /2 与 QKᵀ、AV 的 ×2 抵消
  assert.equal(R.mlpDensePerLayer, 6 * EXAMPLE.hidden * EXAMPLE.ffn * 3);
  assert.equal(R.logits, 6 * EXAMPLE.hidden * EXAMPLE.vocab);
  const tokenLinear = (R.qkvPerLayer + R.outPerLayer + R.mlpDensePerLayer) * EXAMPLE.layers + R.logits;
  assert.equal(R.tokenLinear, tokenLinear);
  assert.equal(R.core, R.corePerLayer * EXAMPLE.layers);
  assert.equal(DENSE_THD.total, tokenLinear * TOK + R.core * SQ);
  // BSHD 默认值就是旧口径：batch×seq 与 batch×seq²
  assert.equal(DENSE_OLD.total, tokenLinear * OLD_TOK + R.core * OLD_SQ);
  // dense 与 MoE lane 活跃宽度相同 → 上报值相同：公式只看活跃宽度
  assert.equal(EXAMPLE.moe.moeFfn * EXAMPLE.moe.topk + EXAMPLE.moe.shared, EXAMPLE.ffn);
  assert.equal(MOE_THD.total, DENSE_THD.total);
  assert.equal(MOE_OLD.total, DENSE_OLD.total);
  // 图 1 右下角引用的 core 占比
  assert.equal(fx((R.core * SQ) / DENSE_THD.total * 100, 1), '6.7');
});

test('偏差账本：四个方向的比值都由同一上报值推出', () => {
  assert.equal(fx(BIAS.oldOverNew, 4), '1.1275');
  assert.equal(pct(BIAS.oldOverNew), '+12.7%');
  assert.equal(fx(BIAS.tokenLinearRatio, 3), '1.067');
  assert.equal(fx(BIAS.coreRatio, 3), '1.969');
  assert.equal(fx(BIAS.routedShare * 100, 1), '37.3');
  assert.equal(fx(BIAS.dropRatio, 3), '1.081');
  assert.equal(pct(BIAS.dropRatio), '+8.1%');
  assert.equal(BIAS.recomputeRatio, 0.75);
  assert.equal(pct(BIAS.recomputeRatio), '−25%');
  assert.equal(fx(BIAS.bothRatio, 3), '0.810');
  assert.equal(pct(BIAS.bothRatio), '−19%');
  // 丢弃只影响路由专家那一份，dense lane 的偏差与丢弃比例无关
  assert.ok(BIAS.dropRatio > 1 && BIAS.recomputeRatio < 1 && BIAS.bothRatio < 1);
  assert.ok(BIAS.bothRatio > BIAS.recomputeRatio, '两个方向相反的偏差应部分抵消');
});

test('DSA：稀疏缩放、indexer 倍率与付费层都由复刻的 helper 算出', () => {
  assert.equal(fx(DSA.meanL, 2), '554.67');
  assert.equal(fx(DSA.scaleExample, 3), '0.710');
  assert.equal(fx(DSA.scaleLong, 3), '0.121');
  assert.equal(fx(DSA.attendedExample, 1), '196.9');
  assert.equal(fx(DSA.denseAttendedExample, 1), '277.3');
  // 三条退化边界
  assert.equal(dsaSparseCoreScale(0, 0, 2048), 1.0);
  assert.equal(dsaSparseCoreScale(512, 512 * 4096, null), 1.0);
  assert.equal(dsaSparseCoreScale(256, 256 * 256, 256), 1.0, 'L̄ ≤ top-k 时坍缩为 1.0');
  for (let e = 9; e <= 13; e += 1) {
    const L = 2 ** e;
    assert.ok(dsaSparseCoreScale(L, L * L, 256) < dsaSparseCoreScale(L / 2, (L * L) / 4, 256), '随 L̄ 单调下降');
  }
  // 每层 L² 系数
  assert.equal(DSA.plainMlaCore, 3072);
  assert.equal(DSA.absorbedDenseCore, 3456);
  assert.equal(Math.round(DSA.dsaCoreExample), 2454);
  assert.equal(Math.round(DSA.dsaCoreLong), 419);
  assert.equal(DSA.indexerCore1x, 128);
  assert.equal(DSA.indexerCore3x, 384);
  assert.equal(DSA.indexerToken1x, 69632);
  assert.equal(DSA.indexerToken2x, 139264);
  assert.deepEqual(dsaIndexerFlops({ hiddenSize: 512, qLoraRank: 128, nHeads: 4, headDim: 32, numIndexerLayers: 0, indexerLossCoeff: 0.01 }), [0, 0]);
  // 付费层：8 层 freq=4 offset=1 → 1、5
  assert.equal(DSA.shareLayers, 2);
  assert.deepEqual(DSA.shareMask, [true, false, false, false, true, false, false, false]);
  assert.equal(numDsaIndexerLayers(8, 0, 1), 8, 'freq=1 时每层都算');
  assert.throws(() => isDsaSkipTopkLayer(0, 0, 1), /1-indexed/);
  assert.throws(() => isDsaSkipTopkLayer(1, -1, 1), /non-negative/);
  assert.throws(() => isDsaSkipTopkLayer(1, 0, 0), /positive/);
  // 三种读法
  assert.equal(fmtSci(DSA.totalMla), '1.686e11');
  assert.equal(fmtSci(DSA.totalOn), '1.687e11');
  assert.equal(fmtSci(DSA.totalOff), '1.671e11');
  assert.equal(fmtSci(DSA.totalShare), '3.336e11');
  assert.equal(DSA_LONG.total, PY_REFERENCE.dsa_bshd_L4096);
  assert.equal(DSA_SHARE.total, PY_REFERENCE.dsa_share_freq4_off1);
  assert.ok(DSA_ON.total > DSA_OFF.total, 'loss 开必须比关贵');
  assert.equal(MLA_THD.total, PY_REFERENCE.mla_thd);
});

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-tflops-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  const [ledger, bias, dsa] = await Promise.all(NAMES.map((name) => readFile(join(outputDir, name), 'utf8')));

  // ---- 图 1：账本 ----
  assert.ok(ledger.includes(`ΣLᵢ=${fmtInt(TOK)}`));
  assert.ok(ledger.includes(`ΣLᵢ²=${fmtInt(SQ)}`));
  for (const v of [DENSE_THD.rows.qkvPerLayer, DENSE_THD.rows.outPerLayer, DENSE_THD.rows.corePerLayer, DENSE_THD.rows.mlpDensePerLayer, DENSE_THD.rows.logits]) {
    assert.ok(ledger.includes(`>${fmtInt(v)}<`), `账本缺每层系数 ${fmtInt(v)}`);
  }
  assert.ok(ledger.includes(`${fmtInt(DENSE_THD.rows.tokenLinear)} × ${fmtInt(TOK)}`));
  assert.ok(ledger.includes(`${fmtInt(DENSE_THD.rows.core)} × ${fmtInt(SQ)}`));
  assert.ok(ledger.includes(`num_floating_point_operations = ${fmtInt(DENSE_THD.total)} FLOPs`));
  assert.ok(ledger.includes('throughput per GPU (TFLOP/s/GPU)'));
  assert.ok(ledger.includes('world_size'));
  assert.ok(ledger.includes(`本例 ${fmtInt(OLD_TOK)} / ${fmtInt(OLD_SQ)}`));
  assert.ok(!ledger.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 2：偏差 ----
  assert.ok(bias.includes(`旧口径高 ${pct(BIAS.oldOverNew)}`));
  assert.ok(bias.includes(`高估 ${pct(BIAS.dropRatio)}`));
  assert.ok(bias.includes(`低估 ${pct(BIAS.recomputeRatio)}`));
  assert.ok(bias.includes(`低估 ${pct(BIAS.bothRatio)}`));
  assert.ok(bias.includes(`路由专家占上报 ${fx(BIAS.routedShare * 100, 1)}%`));
  assert.ok(bias.includes('neither kind of padding shows up in the reported FLOPs'));

  // ---- 图 3：DSA ----
  assert.ok(dsa.includes(`本例 L̄=${fx(DSA.meanL, 1)} → ${fx(DSA.scaleExample, 3)}`));
  assert.ok(dsa.includes(`L̄=${EXAMPLE.longSeq} → ${fx(DSA.scaleLong, 3)}`));
  for (const v of [DSA.plainMlaCore, DSA.absorbedDenseCore, DSA.dsaCoreExample, DSA.dsaCoreLong, DSA.indexerCore1x, DSA.indexerCore3x]) {
    assert.ok(dsa.includes(`>${fmtInt(v)}<`), `图 3 缺柱值 ${fmtInt(v)}`);
  }
  assert.ok(dsa.includes(`付费层：1、5 → ${DSA.shareLayers} 层`));
  assert.ok(dsa.includes('is_dsa_skip_topk_layer'));
  assert.ok(dsa.includes('投影 2× = fwd+wgrad，打分 3× = fwd+dq+dk'));
  assert.ok(dsa.includes('e−1 而非 e'));

  const all = [ledger, bias, dsa];
  NAMES.forEach((name, i) => {
    assertInsideCanvas(all[i], name);
    assertNoTextOverlap(all[i], name);
  });

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(tracked[i], all[i], `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_tflops_figures.mjs`);
  });
});

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红 —— 所以它读的是 .md 本体。
  const page = await readFile(pagePath, 'utf8');
  const R = DENSE_THD.rows;

  // §2.1 共用算例与两个统计量
  assert.ok(page.includes(EXAMPLE.lengths.join('/')), '正文必须列出四条真实长度');
  assert.ok(page.includes(`${fmtInt(TOK)}`) && page.includes(`${fmtInt(SQ)}`));
  assert.ok(page.includes(`${fmtInt(OLD_TOK)}`) && page.includes(`${fmtInt(OLD_SQ)}`));
  // §2.2 账本系数与上报值
  for (const v of [R.qkvPerLayer, R.outPerLayer, R.corePerLayer, R.mlpDensePerLayer, R.logits, R.tokenLinear, R.core]) {
    assert.ok(page.includes(fmtInt(v)), `正文缺账本系数 ${fmtInt(v)}`);
  }
  assert.ok(page.includes(fmtInt(DENSE_THD.total)), '正文必须给出上报值');
  assert.ok(page.includes(fmtInt(DENSE_OLD.total)), '正文必须给出旧口径值');
  assert.ok(page.includes(`${fx((R.core * SQ) / DENSE_THD.total * 100, 1)}%`), 'core 占比');
  // §2.7 偏差
  assert.ok(page.includes(pct(BIAS.oldOverNew)));
  assert.ok(page.includes(`${fx(BIAS.tokenLinearRatio, 3)}`) && page.includes(`${fx(BIAS.coreRatio, 3)}`));
  assert.ok(page.includes(pct(BIAS.dropRatio)));
  assert.ok(page.includes(`${fx(BIAS.routedShare * 100, 1)}%`));
  assert.ok(page.includes(pct(BIAS.recomputeRatio)));
  assert.ok(page.includes(pct(BIAS.bothRatio)));
  assert.ok(page.includes(`${fx(EXAMPLE.dropFraction * 100, 0)}%`));
  // §2.5 DSA
  assert.ok(page.includes(fx(DSA.meanL, 2)));
  assert.ok(page.includes(fx(DSA.scaleExample, 3)));
  assert.ok(page.includes(fx(DSA.scaleLong, 3)));
  assert.ok(page.includes(fx(DSA.attendedExample, 1)) && page.includes(fx(DSA.denseAttendedExample, 1)));
  for (const v of [DSA.plainMlaCore, DSA.absorbedDenseCore, Math.round(DSA.dsaCoreExample), Math.round(DSA.dsaCoreLong), DSA.indexerCore1x, DSA.indexerCore3x, DSA.indexerToken1x, DSA.indexerToken2x]) {
    assert.ok(page.includes(fmtInt(v)), `正文缺 DSA 系数 ${fmtInt(v)}`);
  }
  assert.ok(page.includes(fmtSci(DSA.totalOn)) && page.includes(fmtSci(DSA.totalOff)) && page.includes(fmtSci(DSA.totalMla)));
  assert.ok(page.includes(`top-k=${EXAMPLE.dsa.topk}`) || page.includes(`k=${EXAMPLE.dsa.topk}`));
  assert.ok(page.includes(`freq=${EXAMPLE.dsaShare.freq}`) && page.includes(`offset=${EXAMPLE.dsaShare.offset}`));
  assert.ok(page.includes('1 与 5') || page.includes('1、5'), '付费层必须点名');

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
});
