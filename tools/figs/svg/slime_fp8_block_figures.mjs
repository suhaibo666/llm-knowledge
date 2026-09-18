// 图：同一个 256×256 权重怎样变成低精度 weight + scale。
// ① FP8：tests/test_block_fp8_zero_block.py 的输入（W[0,0]=1.0，其余为 0，block 128×128）逐块求 absmax 与 scale，
//    全零块靠下限避开 0/0；再在第 4 块放一个小值，分开 blockwise、per-tensor 与 UE8M0 三种 scale 规则；
// ② INT4：同一输入按行分组（group=128），训练侧 fake-QAT 的 STE 前向/反向与 rollout 打包对同一组数给出 q、scale 与 int32 字。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；
// fake-QAT 来自 slime 的 docker/patch/latest/megatron.patch（打在 NVIDIA/Megatron-LM@1dcf0dafa884 上）；
// UE8M0 列复现上游 sgl-project/sglang@0b3bb0cbe318（v0.5.15.post1）python/sglang/srt/layers/quantization/fp8_utils.py
// 的 per_block_cast_to_fp8 / ceil_to_ue8m0（slime 补丁未改该文件，依赖侧）。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 图 1 要讲清楚：scale 的粒度决定一个元素能被量化到多细；只有一个非零元素的测试输入分不开 blockwise 与 per-tensor，
// 它说明的是分块布局与零块下限（去掉下限就是 0/0=NaN）；加一个 2⁻²⁰ 的小值后，per-tensor 因全张量 amax 把它舍成 0，
// blockwise 用本块 scale 精确保留，UE8M0 把 scale 向上取 2 的幂并有 1e-4 的 amax 下限。acc1 标 slime 默认分块结果，
// acc2 标 NaN 反例与被舍掉的元素，依赖侧 UE8M0 用虚线框。
// 图 2 要讲清楚：INT4 group 是“每行按列切组”，STE 前向做 round/clamp、反向直通；rollout 打包用同一公式得到同一个 q，
// 但 weight_scale 以权重 dtype（BF16）存储；q+8 的 nibble 怎样拼成 int32；以及 direct 转换器默认非对称这一冲突。
//
// 用法：node tools/figs/svg/slime_fp8_block_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  shape: [256, 256], // tests/test_block_fp8_zero_block.py::test_block_fp8_all_zero_block_has_no_nan
  block: [128, 128], // converter.block_fp8(weight, (128, 128))
  testEntries: Object.freeze([{ r: 0, c: 0, v: 1.0 }]), // weight[0, 0] = 1.0
  extraEntry: Object.freeze({ r: 128, c: 128, v: 2 ** -20 }), // 作图补充：不在测试里
  fp8Max: 448, // torch.finfo(torch.float8_e4m3fn).max
  toolEps: 1e-12, // tools/convert_hf_to_fp8.py::block_fp8 / tensor_fp8 的 clamp(min=1e-12)
  tritonEps: 1e-10, // kernels/fp8_kernel.py::blockwise_cast_to_fp8_triton 传入的 eps
  ue8m0Floor: 1e-4, // sglang fp8_utils.py::per_block_cast_to_fp8 的 clamp(1e-4)
  int4: Object.freeze({ group: 128, qMax: 7, scaleFloor: 1e-5 }), // megatron.patch::_FakeInt4QuantizationSTE / fake_int4_quant_cuda.cu
});

const f32 = Math.fround;
const range = (n) => Array.from({ length: n }, (_, i) => i);

// ---------------- 数值格式 ----------------

// float32 → bfloat16：保留 1 位符号、8 位指数、7 位尾数，低 16 位按 round-half-even 舍入
export function toBf16(x) {
  const u = new Uint32Array(new Float32Array([x]).buffer)[0];
  const r = ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) << 16;
  return new Float32Array(new Uint32Array([r >>> 0]).buffer)[0];
}

// float8_e4m3fn 的全部非负可表示值：e=0 为次正规 m·2⁻⁹，e=1..15 为 (8+m)·2^(e−10)，e=15,m=7 是 NaN
export const E4M3_VALUES = (() => {
  const out = [];
  for (let e = 0; e < 16; e += 1) {
    for (let m = 0; m < 8; m += 1) {
      if (e === 15 && m === 7) continue;
      out.push({ code: e * 8 + m, value: e === 0 ? m * 2 ** -9 : (8 + m) * 2 ** (e - 10) });
    }
  }
  return out;
})();

// 向 float8_e4m3fn 转换：就近舍入，恰在中点时取尾数码为偶数的一侧（torch 实测同此规则）
export function castE4m3(x) {
  if (Number.isNaN(x)) return NaN;
  const a = Math.abs(x);
  let best = E4M3_VALUES[0];
  for (const cand of E4M3_VALUES) {
    const d = Math.abs(cand.value - a);
    const bd = Math.abs(best.value - a);
    if (d < bd || (d === bd && cand.code % 2 === 0 && best.code % 2 === 1)) best = cand;
  }
  return Math.sign(x) * best.value || 0;
}

// sglang fp8_utils.py::ceil_to_ue8m0：取 float32 指数，尾数非零则指数 +1，结果是 2 的幂
export function ceilUe8m0(x) {
  const bits = new Uint32Array(new Float32Array([Math.abs(x)]).buffer)[0];
  let exp = (bits >>> 23) & 0xff;
  if ((bits & 0x7fffff) !== 0) exp += 1;
  exp = Math.min(Math.max(exp, 1), 254);
  return 2 ** (exp - 127);
}

// 四舍六入五成双（torch.round 与 rintf 的默认舍入）
export function roundHalfEven(x) {
  const fl = Math.floor(x);
  const d = x - fl;
  if (d < 0.5) return fl;
  if (d > 0.5) return fl + 1;
  return fl % 2 === 0 ? fl : fl + 1;
}

// ---------------- 对源码算法的最小复现 ----------------

const valueAt = (entries, r, c) => (entries.find((e) => e.r === r && e.c === c) || { v: 0 }).v;

// 把稀疏输入切成 ceil(M/B)×ceil(N/B) 块，记录每块的 BF16 absmax 与非零元素
export function tiles(entries, cfg = CFG) {
  const [M, N] = cfg.shape;
  const [bm, bn] = cfg.block;
  const rows = Math.ceil(M / bm);
  const cols = Math.ceil(N / bn);
  return range(rows).map((i) =>
    range(cols).map((j) => {
      const inside = entries.filter((e) => Math.floor(e.r / bm) === i && Math.floor(e.c / bn) === j);
      return { i, j, amax: toBf16(Math.max(0, ...inside.map((e) => Math.abs(e.v)))), entries: inside };
    }),
  );
}

// tools/convert_hf_to_fp8.py::block_fp8：BF16 上 clamp(min=1e-12) → 转 FP32 除以 448 → (W/scale).clamp(±448) → E4M3
export function blockFp8Tool(entries, { clamp = true } = {}, cfg = CFG) {
  const floor = toBf16(f32(cfg.toolEps));
  return tiles(entries, cfg).map((row) =>
    row.map((t) => {
      const clamped = clamp ? Math.max(t.amax, floor) : t.amax;
      const scale = f32(clamped / cfg.fp8Max);
      const q = (v) => {
        const ratio = f32(v / scale); // 0/0 → NaN
        return castE4m3(Number.isNaN(ratio) ? NaN : Math.min(Math.max(ratio, -cfg.fp8Max), cfg.fp8Max));
      };
      return { ...t, clamped, scale, q, nanCount: Number.isNaN(q(0)) ? cfg.block[0] * cfg.block[1] - t.entries.length : 0 };
    }),
  );
}

// kernels/fp8_kernel.py::_blockwise_cast_to_fp8_triton：FP32 上 max(absmax, eps) / 448，y = clamp(x · (1/x_s))
export function blockFp8Triton(entries, cfg = CFG) {
  return tiles(entries, cfg).map((row) =>
    row.map((t) => {
      const scale = f32(Math.max(t.amax, f32(cfg.tritonEps)) / cfg.fp8Max);
      const inv = f32(1 / scale);
      return { ...t, scale, q: (v) => castE4m3(Math.min(Math.max(f32(v * inv), -cfg.fp8Max), cfg.fp8Max)) };
    }),
  );
}

// tools/convert_hf_to_fp8.py::tensor_fp8（与 quantizer_fp8.py::_quantize_param 无 block 分支同式）：全张量一个 scale
export function tensorFp8(entries, cfg = CFG) {
  const amax = toBf16(Math.max(0, ...entries.map((e) => Math.abs(e.v))));
  const scale = f32(Math.max(amax, toBf16(f32(cfg.toolEps))) / cfg.fp8Max);
  return { scale, q: (v) => castE4m3(Math.min(Math.max(f32(v / scale), -cfg.fp8Max), cfg.fp8Max)) };
}

// sglang fp8_utils.py::per_block_cast_to_fp8（quant_weight_ue8m0 调用）：amax.clamp(1e-4) → ceil_to_ue8m0(amax/448) → x·(1/sf)
export function ue8m0Blocks(entries, cfg = CFG) {
  return tiles(entries, cfg).map((row) =>
    row.map((t) => {
      const sf = ceilUe8m0(f32(Math.max(t.amax, f32(cfg.ue8m0Floor)) / cfg.fp8Max));
      return { ...t, sf, q: (v) => castE4m3(f32(v * f32(1 / sf))) };
    }),
  );
}

// megatron.patch::_FakeInt4QuantizationSTE.forward：块 (1, group)；scale = (max/7).clamp(1e-5)；
// q = round(W/scale).clamp(−7, 7)；输出 q·scale 转回 BF16。backward 原样返回 grad_output。
export function int4Ste(entries, cfg = CFG) {
  const { group, qMax, scaleFloor } = cfg.int4;
  const [, N] = cfg.shape;
  const rowsUsed = [...new Set([0, ...entries.map((e) => e.r)])].sort((a, b) => a - b);
  return rowsUsed.map((r) => ({
    r,
    groups: range(Math.ceil(N / group)).map((g) => {
      const inside = entries.filter((e) => e.r === r && Math.floor(e.c / group) === g);
      const max = f32(Math.max(0, ...inside.map((e) => Math.abs(e.v))));
      const raw = f32(max / qMax);
      const scale = Math.max(raw, f32(scaleFloor));
      const ratio = (v) => f32(v / scale);
      const q = (v) => Math.min(Math.max(roundHalfEven(ratio(v)), -qMax), qMax);
      return { g, max, raw, scale, entries: inside, ratio, q, out: (v) => toBf16(f32(q(v) * scale)) };
    }),
  }));
}

// fake_int4_quant_cuda.cu 对称分支（scale = max(block_max · (1/7), 1e-5)，val = rint(val/scale)，out_scale 取输入 dtype）
// + quantizer_compressed_tensors.py::pack_layer / pack_to_int32（q + 8 → uint8 → 每 8 个 nibble 左移 4k 求和成 int32）
export function int4Pack(entries, cfg = CFG) {
  const { group, scaleFloor } = cfg.int4;
  const [M, N] = cfg.shape;
  const packFactor = 32 / 4;
  const rows = int4Ste(entries, cfg).map((row) => ({
    r: row.r,
    groups: row.groups.map((gr) => {
      const scale = Math.max(f32(gr.max * f32(1 / 7)), f32(scaleFloor));
      const q = (v) => roundHalfEven(f32(v / scale));
      return { g: gr.g, scale, storedScale: toBf16(scale), q };
    }),
  }));
  const qAt = (r, c) => {
    const row = rows.find((x) => x.r === r);
    return row ? row.groups[Math.floor(c / group)].q(valueAt(entries, r, c)) : 0;
  };
  const word = (r, k) => {
    let acc = 0;
    for (let j = 0; j < packFactor; j += 1) acc += (qAt(r, k * packFactor + j) + 8) * 2 ** (4 * j);
    return acc % 2 ** 32;
  };
  const bytes = {
    packed: M * (N / packFactor) * 4, // int32
    scale: M * Math.ceil(N / group) * 2, // 与 BF16 权重同 dtype
    shape: 2 * 4, // torch.tensor(param.shape, dtype=int32)
  };
  return { rows, word, qAt, bytes, total: bytes.packed + bytes.scale + bytes.shape };
}

const hex32 = (u) => `0x${(u >>> 0).toString(16).toUpperCase().padStart(8, '0')}`;

export function model(cfg = CFG) {
  const test = [...cfg.testEntries];
  const ext = [...cfg.testEntries, cfg.extraEntry];
  const x = cfg.extraEntry;
  const tool = blockFp8Tool(test, {}, cfg);
  const noClamp = blockFp8Tool(test, { clamp: false }, cfg);
  const tritonTest = blockFp8Triton(test, cfg);
  const tensorTest = tensorFp8(test, cfg);
  const toolExt = blockFp8Tool(ext, {}, cfg);
  const tritonExt = blockFp8Triton(ext, cfg);
  const tensorExt = tensorFp8(ext, cfg);
  const ueExt = ue8m0Blocks(ext, cfg);
  const [M, N] = cfg.shape;
  const tI = Math.floor(x.r / cfg.block[0]);
  const tJ = Math.floor(x.c / cfg.block[1]);
  const agree = test.every((e) => tool[Math.floor(e.r / 128)][Math.floor(e.c / 128)].q(e.v) === tensorTest.q(e.v)) && tensorTest.q(0) === 0;
  const ste = int4Ste(ext, cfg);
  const pack = int4Pack(ext, cfg);
  const steRow0 = ste.find((r) => r.r === 0).groups[0];
  const steZero = ste.find((r) => r.r === 0).groups[1];
  const steX = ste.find((r) => r.r === x.r).groups[Math.floor(x.c / cfg.int4.group)];
  const packRow0 = pack.rows.find((r) => r.r === 0).groups[0];
  return {
    cfg,
    fp8: {
      bf16Bytes: M * N * 2,
      weightBytes: M * N,
      blockScaleBytes: tool.length * tool[0].length * 4,
      tensorScaleBytes: 4,
      scaleGrid: [tool.length, tool[0].length],
      toolFloor: toBf16(f32(cfg.toolEps)),
      test: {
        scale00: tool[0][0].scale,
        q00: tool[0][0].q(1.0),
        zeroScale: tool[0][1].scale,
        zeroQ: tool[0][1].q(0),
        zeroTiles: tool.flat().filter((t) => t.amax === 0).length,
        nanCount: noClamp.flat().reduce((a, t) => a + t.nanCount, 0),
        unclampedZeroScale: noClamp[0][1].scale,
        allScalesPositive: tool.flat().every((t) => t.scale > 0),
        tritonZeroScale: tritonTest[0][1].scale,
        tensorScale: tensorTest.scale,
        tensorAgrees: agree,
      },
      ext: {
        value: x.v,
        blockScale: toolExt[tI][tJ].scale,
        blockQ: toolExt[tI][tJ].q(x.v),
        blockDeq: f32(toolExt[tI][tJ].q(x.v) * toolExt[tI][tJ].scale),
        tritonQ: tritonExt[tI][tJ].q(x.v),
        tensorScale: tensorExt.scale,
        tensorRatio: f32(x.v / tensorExt.scale),
        zeroThreshold: 2 ** -10,
        tensorQ: tensorExt.q(x.v),
        ueSf00: ueExt[0][0].sf,
        ueQ00: ueExt[0][0].q(1.0),
        ueSfZero: ueExt[0][1].sf,
        ueSfX: ueExt[tI][tJ].sf,
        ueQX: ueExt[tI][tJ].q(x.v),
        ueDeqX: f32(ueExt[tI][tJ].q(x.v) * ueExt[tI][tJ].sf),
      },
    },
    int4: {
      groupsPerRow: Math.ceil(N / cfg.int4.group),
      scaleShape: [M, Math.ceil(N / cfg.int4.group)],
      row0: { scale: steRow0.scale, q: steRow0.q(1.0), out: steRow0.out(1.0) },
      zero: { scale: steZero.scale, q: steZero.q(0) },
      x: { raw: steX.raw, scale: steX.scale, ratio: steX.ratio(x.v), q: steX.q(x.v), out: steX.out(x.v) },
      packQ00: pack.qAt(0, 0),
      packQX: pack.qAt(x.r, x.c),
      storedScale00: packRow0.storedScale,
      nibble00: pack.qAt(0, 0) + 8,
      nibbleZero: pack.qAt(0, 1) + 8,
      word0: hex32(pack.word(0, 0)),
      wordsPerRow: N / 8,
      bytes: pack.bytes,
      total: pack.total,
    },
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export const sci = (x, d = 4) => {
  if (x === 0) return '0';
  const [m, e] = x.toExponential(d).split('e');
  return `${m}e${Number(e)}`;
};
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export const pow2 = (x) => `2${String(Math.log2(x)).split('').map((ch) => SUP[ch]).join('')}`;

const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .tb{font-size:12px;font-weight:700;fill:#1F2430}
  .sm{font-size:10.5px;fill:#5B6470}
  .mono{font-size:11px;font-family:"SFMono-Regular",Menlo,Consolas,monospace;fill:#38414D}
  .cap{font-size:11.5px;fill:#5B6470}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .dot1{fill:#2563EB}
  .dot2{fill:#C3651F}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;marker-end:url(#arrowAux)}
`;

function canvas(W, H, title, desc) {
  const o = [];
  const api = {
    o,
    rect: (x, y, w, h, cls = 'neutral', r = 6) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`),
    text: (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`),
    arrow: (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`),
    dot: (x, y, cls = 'dot1') => o.push(`<circle cx="${x}" cy="${y}" r="4" class="${cls}"/>`),
  };
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  return api;
}

// 2×2 块网格：lines(i, j) 返回该块内的若干行文字，cls(i, j) 返回块样式
function grid(api, x, y, tw, th, lines, cls) {
  for (let i = 0; i < 2; i += 1) {
    for (let j = 0; j < 2; j += 1) {
      const bx = x + j * tw;
      const by = y + i * th;
      api.rect(bx, by, tw, th, cls(i, j), 0);
      const ls = lines(i, j);
      const top = by + th / 2 - ((ls.length - 1) * 15) / 2 + 4;
      ls.forEach((s, k) => api.text(bx + tw / 2, top + k * 15, s, k === 0 ? 'tb' : 'sm', 'middle'));
    }
  }
}

function renderFp8(m) {
  const cfg = m.cfg;
  const F = m.fp8;
  const T = F.test;
  const E = F.ext;
  const W = 1180;
  const H = 936;
  const api = canvas(
    W,
    H,
    'slime FP8 分块量化：一个 256×256 权重的块、scale 与零块边界',
    '两个面板。①取 tests/test_block_fp8_zero_block.py 的输入，逐块求 absmax 与 scale，说明全零块为什么需要下限、去掉下限会写出多少 NaN，以及只有一个非零元素时 blockwise 与 per-tensor 得到同一组量化值；②在第四块再放一个 2 的负 20 次方，比较 blockwise、per-tensor 与 SGLang UE8M0 三种 scale 规则对这个小值的量化结果与字节数。',
  );
  const { rect, text, arrow, dot } = api;
  const s4 = (v) => sci(v, 4);
  text(24, 34, '同一个 256×256 权重：scale 的粒度决定一个元素能被量化到多细', 'ti');
  text(24, 56, `输入取自 tests/test_block_fp8_zero_block.py（W[0,0]=1.0，其余为 0，block ${cfg.block.join('×')}）· E4M3 最大值 ${cfg.fp8Max} · 面板 ② 另放 W[128,128]=${pow2(E.value)}（作图补充，不在测试里）`, 'su');

  // ---------- ① ----------
  const py = 72;
  rect(24, py, 1132, 340, 'panel', 10);
  text(40, py + 26, '① 测试输入：每块一个 absmax 与 scale；全零块靠下限避开 0/0', 'pt');
  const gx = 56;
  const gy = py + 84;
  const tw = 104;
  text(gx, py + 52, `输入 W：BF16 256×256，${F.bf16Bytes} 字节`, 'tx');
  text(gx, gy - 8, '列 0', 'sm', 'middle');
  text(gx + tw, gy - 8, '128', 'sm', 'middle');
  text(gx + 2 * tw, gy - 8, '256', 'sm', 'middle');
  grid(api, gx, gy, tw, tw, (i, j) => [`T${i}${j}`, i === 0 && j === 0 ? 'amax 1.0' : 'amax 0'], (i, j) => (i === 0 && j === 0 ? 'cell' : 'ghost'));
  dot(gx + 8, gy + 8);
  text(gx + 16, gy + 30, 'W[0,0]=1.0', 'sm');
  text(gx - 6, gy + tw + 4, '128', 'sm', 'end');
  text(gx - 6, gy + 2 * tw + 4, '256', 'sm', 'end');

  arrow(gx + 2 * tw + 10, gy + tw, gx + 2 * tw + 50, gy + tw);
  const sx = 324;
  text(sx, py + 52, '逐块 scale（tools/convert_hf_to_fp8.py::block_fp8）', 'tx');
  grid(
    api,
    sx,
    gy,
    176,
    tw,
    (i, j) =>
      i === 0 && j === 0
        ? ['T00：amax 1.0', `s = 1/${cfg.fp8Max}`, `= ${s4(T.scale00)}`, `Q[0,0] = ${T.q00}`]
        : [`T${i}${j}：amax 0 → 下限`, `BF16(1e-12) = ${s4(F.toolFloor)}`, `s = ${s4(T.zeroScale)}`, `Q = ${T.zeroQ}`],
    (i, j) => (i === 0 && j === 0 ? 'acc1' : 'cell'),
  );
  text(sx, gy + 2 * tw + 16, `scale 形状 ceil(256/128)×ceil(256/128) = ${F.scaleGrid.join('×')}`, 'sm');

  const rx = 716;
  rect(rx, py + 44, 424, 96, 'acc1', 8);
  text(rx + 12, py + 64, '输出（与单测断言一致）', 'tb');
  text(rx + 12, py + 84, `weight：FP8 256×256，${F.weightBytes} 字节；Q[0,0]=${T.q00}，其余 0`, 'tx');
  text(rx + 12, py + 102, `weight_scale_inv：FP32 ${F.scaleGrid.join('×')}，${F.blockScaleBytes} 字节`, 'tx');
  text(rx + 12, py + 120, `无 NaN/Inf、全部 s > 0：${T.allScalesPositive ? '成立' : '不成立'}；零块反量化 Q·s = 0`, 'tx');
  rect(rx, py + 150, 424, 74, 'acc2', 8);
  text(rx + 12, py + 170, '去掉下限（单测 docstring 描述的修复前行为）', 'tb');
  text(rx + 12, py + 190, `零块 s = 0/${cfg.fp8Max} = ${T.unclampedZeroScale} → Q = 0/0 = NaN`, 'tx');
  text(rx + 12, py + 208, `${T.zeroTiles} 块 × 128×128 = ${T.nanCount} 个 NaN 写进 checkpoint`, 'tx');
  rect(rx, py + 234, 424, 74, 'neutral', 8);
  text(rx + 12, py + 254, '同一输入的另两种实现', 'tb');
  text(rx + 12, py + 274, `per-tensor：s = ${s4(T.tensorScale)}（1 个，${F.tensorScaleBytes} 字节）；Q 与分块逐元素${T.tensorAgrees ? '相同' : '不同'}`, 'tx');
  text(rx + 12, py + 292, `在线 Triton：零块 s = 1e-10/${cfg.fp8Max} = ${s4(T.tritonZeroScale)}（单测不覆盖）`, 'tx');
  text(40, py + 330, '只有一个非零元素时，这个输入只能说明分块布局与零块下限，分不开 blockwise 与 per-tensor——所以面板 ② 再加一个小值。', 'cap');

  // ---------- ② ----------
  const qy = 426;
  rect(24, qy, 1132, 452, 'panel', 10);
  text(40, qy + 26, `② 在 T11 放 W[128,128] = ${pow2(E.value)} ≈ ${s4(E.value)}：三种 scale 规则分开`, 'pt');
  const ix = 48;
  const iy = qy + 76;
  const it = 72;
  text(ix, qy + 54, '输入 W（两个非零元素）', 'tx');
  grid(api, ix, iy, it, it, (i, j) => [`T${i}${j}`], (i, j) => ((i === 0 && j === 0) || (i === 1 && j === 1) ? 'cell' : 'ghost'));
  dot(ix + 8, iy + 8);
  dot(ix + it + 8, iy + it + 8, 'dot2');
  text(ix, iy + 2 * it + 20, 'W[0,0] = 1.0', 'sm');
  text(ix, iy + 2 * it + 36, `W[128,128] = ${pow2(E.value)}`, 'sm');
  text(ix, iy + 2 * it + 52, '比值 1 : 1048576', 'sm');

  const lanes = [
    { x: 216, title: 'blockwise（slime 默认）', sub: 'block_fp8 / blockwise_cast_to_fp8_triton', cls: 'acc1' },
    { x: 526, title: 'per-tensor（slime）', sub: 'tensor_fp8 / _quantize_param 无 block', cls: 'acc2' },
    { x: 836, title: 'UE8M0（SGLang 上游 helper）', sub: 'per_block_cast_to_fp8：force / runtime 分支', cls: 'dep' },
  ];
  lanes.forEach((ln) => {
    rect(ln.x, qy + 44, 300, 44, ln.cls, 8);
    text(ln.x + 12, qy + 62, ln.title, 'tb');
    text(ln.x + 12, qy + 79, ln.sub, 'sm');
  });
  arrow(ix + 2 * it + 12, iy + it, lanes[0].x - 8, iy + it);

  const cw = 150;
  const ch = 62;
  const gyy = qy + 104;
  // lane a
  grid(
    api,
    lanes[0].x,
    gyy,
    cw,
    ch,
    (i, j) => {
      if (i === 0 && j === 0) return ['T00', `s = ${s4(T.scale00)}`];
      if (i === 1 && j === 1) return ['T11', `s = ${pow2(E.value)}/448`, `= ${sci(E.blockScale, 3)}`];
      return [`T${i}${j}`, `s = ${s4(T.zeroScale)}`];
    },
    (i, j) => (i === 1 && j === 1 ? 'acc1' : 'cell'),
  );
  // lane b
  rect(lanes[1].x, gyy, 2 * cw, 2 * ch, 'cell', 0);
  text(lanes[1].x + cw, gyy + ch - 6, `全张量 s = ${s4(E.tensorScale)}`, 'tb', 'middle');
  text(lanes[1].x + cw, gyy + ch + 12, '由 W[0,0]=1.0 决定，4 块共用', 'sm', 'middle');
  // lane c
  grid(
    api,
    lanes[2].x,
    gyy,
    cw,
    ch,
    (i, j) => {
      if (i === 0 && j === 0) return ['T00', `sf = ceil₂(1/448) = ${pow2(E.ueSf00)}`];
      if (i === 1 && j === 1) return ['T11', 'amax 升到 1e-4', `sf = ${pow2(E.ueSfX)}`];
      return [`T${i}${j}`, `sf = ${pow2(E.ueSfZero)}`];
    },
    (i, j) => (i === 1 && j === 1 ? 'dep' : 'cell'),
  );

  const ry = gyy + 2 * ch + 14;
  const res = [
    [lanes[0], 'acc1', ['T11 的量化结果', `Q = ${E.blockQ}（Triton 同为 ${E.tritonQ}）`, `Q·s = ${s4(E.blockDeq)}，误差 0`]],
    [lanes[1], 'acc2', ['T11 的量化结果', `W/s = ${s4(E.tensorRatio)} ≤ 2⁻¹⁰（最小次正规数 2⁻⁹ 之半）`, `Q = ${E.tensorQ} → Q·s = 0，误差 −100%`]],
    [lanes[2], 'dep', ['两块的量化结果', `T00：Q[0,0] = ${E.ueQ00}（不是 448）`, `T11：Q = ${E.ueQX} → Q·sf = ${s4(E.ueDeqX)}`]],
  ];
  res.forEach(([ln, cls, ls]) => {
    rect(ln.x, ry, 300, 74, cls, 8);
    ls.forEach((s, k) => text(ln.x + 12, ry + 20 + k * 20, s, k === 0 ? 'tb' : 'tx'));
  });

  const ly = ry + 90;
  rect(216, ly, 920, 110, 'neutral', 8);
  text(228, ly + 20, '载荷与代价（只计张量字节，不含名字、dtype、shape 元数据）', 'tb');
  text(228, ly + 42, `BF16 原权重 ${F.bf16Bytes} 字节 → FP8 weight ${F.weightBytes} 字节 + scale：blockwise ${F.blockScaleBytes}（FP32 ${F.scaleGrid.join('×')}）/ per-tensor ${F.tensorScaleBytes} / UE8M0 打包前 ${F.blockScaleBytes}`, 'tx');
  text(228, ly + 62, 'per-tensor：|W| ≤ 2⁻¹⁰·s 的元素舍成 0，s 由全张量最大值决定；blockwise 把这个阈值降到本块 amax/448·2⁻¹⁰，代价是 scale 张量随块数增长', 'tx');
  text(228, ly + 82, 'UE8M0：scale 向上取 2 的幂，amax/sf 落在 (224, 448]，并有 1e-4 的 amax 下限；Blackwell 打包后的 scale 布局由 DeepGEMM 决定，本图不计算', 'tx');
  text(228, ly + 100, 'blockwise 与 per-tensor 两列复现 slime 源码；UE8M0 列复现上游 SGLang v0.5.15.post1 源码（依赖侧，slime 补丁未改）', 'sm');

  text(24, 902, '阅读顺序：① 先看一个块怎样得到 scale、零块为什么要下限 → ② 再用同一张量上的小值看 scale 粒度怎样决定舍入。', 'cap');
  text(24, 922, '源码基线：THUDM/slime@4c193f1f37 · 复现 tools/convert_hf_to_fp8.py::block_fp8 / tensor_fp8、kernels/fp8_kernel.py::_blockwise_cast_to_fp8_triton；UE8M0 列为 sgl-project/sglang@0b3bb0cbe318', 'su');
  api.o.push('</svg>');
  return api.o.join('\n');
}

function renderInt4(m) {
  const cfg = m.cfg;
  const I = m.int4;
  const x = cfg.extraEntry;
  const W = 1180;
  const H = 690;
  const api = canvas(
    W,
    H,
    'slime INT4 group 量化：训练侧 fake-QAT 与 rollout 打包对同一输入的映射',
    '三个面板。①同一个 256×256 输入按行、每 128 列分一组，scale 形状为 256×2，给出每组的 max、scale 与整数 q；②megatron.patch 中 _FakeInt4QuantizationSTE 的前向 round/clamp 与反向直通；③rollout 侧 pack_layer 用同一公式得到 q，把 scale 按 BF16 存储，并把 q+8 的 nibble 拼成 int32，给出字节数与 direct 转换器默认非对称的冲突。',
  );
  const { rect, text, arrow, dot } = api;
  text(24, 34, `同一个输入在 INT4 group=${cfg.int4.group} 下：训练侧 fake-QAT 与 rollout 打包各算一遍`, 'ti');
  text(24, 56, `输入同图 ②：W[0,0]=1.0，W[${x.r},${x.c}]=${pow2(x.v)}，其余为 0 · 块 (1, ${cfg.int4.group})：每行按列切组 · 对称 q ∈ [−${cfg.int4.qMax}, ${cfg.int4.qMax}] · scale 下限 1e-5`, 'su');

  // ---------- ① ----------
  const py = 72;
  rect(24, py, 1132, 226, 'panel', 10);
  text(40, py + 26, `① 分组：256 列切成 ${I.groupsPerRow} 组，scale 形状 ${I.scaleShape.join('×')}（每行每组一个）`, 'pt');
  const sx = 118;
  const gw = 500;
  const rows = [
    { r: 0, y: py + 48, cells: [
      ['acc1', ['G0 列 [0,128)：max 1.0', `s = 1/7，q = round(7) = ${I.row0.q}`, `前向 q·s = ${I.row0.out}`]],
      ['cell', ['G1 列 [128,256)：max 0', 's = 下限 1e-5', `q = ${I.zero.q}`]],
    ] },
    { r: x.r, y: py + 128, cells: [
      ['cell', ['G0 列 [0,128)：max 0', 's = 下限 1e-5', `q = ${I.zero.q}`]],
      ['acc2', [`G1：max ${pow2(x.v)}，max/7 = ${sci(I.x.raw, 3)} < 1e-5`, `s = 1e-5，W/s = ${I.x.ratio.toFixed(4)}`, `q = ${I.x.q} → 前向 ${I.x.out}（小值被舍掉）`]],
    ] },
  ];
  rows.forEach((row) => {
    text(sx - 12, row.y + 40, `行 ${row.r}`, 'tb', 'end');
    row.cells.forEach(([cls, ls], g) => {
      const bx = sx + g * gw;
      rect(bx, row.y, gw, 70, cls, 0);
      ls.forEach((s, k) => text(bx + 14, row.y + 20 + k * 19, s, k === 0 ? 'tb' : 'tx'));
    });
  });
  dot(sx + 6, py + 54);
  dot(sx + gw + 6, py + 134, 'dot2');
  text(40, py + 216, '其余 254 行全零：每组 s 取下限、q = 0。块 (1, group) 要求 group·1 是 32 的倍数（fake_int4_quant_cuda 的 TORCH_CHECK）。', 'cap');

  // ---------- ② ----------
  const qy = 312;
  rect(24, qy, 552, 312, 'panel', 10);
  text(40, qy + 26, '② 训练侧 fake-QAT（megatron.patch）', 'pt');
  rect(40, qy + 42, 520, 58, 'neutral', 8);
  text(52, qy + 62, '只包在 TEGroupedLinear._get_weight_tensors 外：MoE experts 的权重', 'tx');
  text(52, qy + 82, '开关 OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1，group 读 …_GROUP_SIZE', 'tx');
  rect(40, qy + 112, 520, 78, 'acc1', 8);
  text(52, qy + 132, '前向：_FakeInt4QuantizationSTE.forward', 'tb');
  text(52, qy + 152, 's = (max|W| / 7).clamp(1e-5)，Ŵ = clamp(round(W/s), −7, 7) · s', 'mono');
  text(52, qy + 172, `本例 Ŵ[0,0] = ${I.row0.out}，Ŵ[${x.r},${x.c}] = ${I.x.out}；s 在 FP32 中参与重建，结果转回 BF16`, 'tx');
  arrow(300, qy + 192, 300, qy + 208);
  rect(40, qy + 210, 520, 78, 'acc2', 8);
  text(52, qy + 230, '反向：_FakeInt4QuantizationSTE.backward', 'tb');
  text(52, qy + 250, '∂L/∂W = ∂L/∂Ŵ（round 与 clamp 不求导，梯度直通）', 'mono');
  text(52, qy + 270, `W[${x.r},${x.c}] 前向被舍成 0，梯度仍原样流回；optimizer 更新的是 BF16 主权重`, 'tx');

  // ---------- ③ ----------
  const px = 590;
  rect(px, qy, 566, 312, 'panel', 10);
  text(px + 16, qy + 26, '③ rollout 打包（quantizer_compressed_tensors.py::pack_layer）', 'pt');
  rect(px + 16, qy + 42, 534, 58, 'neutral', 8);
  text(px + 28, qy + 62, `kernel 对称分支：s = max(max·(1/7), 1e-5)，q = rint(W/s) → q[0,0] = ${I.packQ00}，q[${x.r},${x.c}] = ${I.packQX}`, 'tx');
  text(px + 28, qy + 82, `weight_scale 取权重 dtype：BF16(1/7) = ${I.storedScale00}，不是 FP32 的 1/7`, 'tx');
  rect(px + 16, qy + 112, 534, 78, 'acc1', 8);
  text(px + 28, qy + 132, 'pack_to_int32：q + 8 ∈ [1, 15]，每 8 个 nibble 左移 4k 求和成一个 int32', 'tb');
  text(px + 28, qy + 152, `行 0 的前 8 列 nibble = [${I.nibble00}, ${Array(7).fill(I.nibbleZero).join(', ')}]`, 'mono');
  text(px + 28, qy + 172, `→ 第 1 个字位模式 ${I.word0}；每行 ${I.wordsPerRow} 个字`, 'mono');
  rect(px + 16, qy + 202, 534, 44, 'neutral', 8);
  text(px + 28, qy + 222, `weight_packed int32 256×${I.wordsPerRow} = ${I.bytes.packed} · weight_scale BF16 ${I.scaleShape.join('×')} = ${I.bytes.scale}`, 'tx');
  text(px + 28, qy + 239, `weight_shape int32×2 = ${I.bytes.shape} → 共 ${I.total} 字节（BF16 原权重 ${m.fp8.bf16Bytes}）`, 'tx');
  rect(px + 16, qy + 256, 534, 44, 'acc2', 8);
  text(px + 28, qy + 276, 'convert_hf_to_int4_direct.py 默认非对称（--is-symmetric 未给）：', 'tx');
  text(px + 28, qy + 293, 's = (max−min)/15，另存 weight_zero_point——与 STE 的对称 q_max=7 不是同一映射', 'tx');

  text(24, 654, '阅读顺序：① 先确定每个元素属于哪一组、得到哪个 q → ② 训练前向看到的是 q·s，反向直通 → ③ rollout 收到的是 q 的 nibble 与 BF16 scale。', 'cap');
  text(24, 674, '源码基线：THUDM/slime@4c193f1f37 · 复现 docker/patch/latest/megatron.patch::_FakeInt4QuantizationSTE、kernels/int4_qat/fake_int4_quant_cuda.cu 对称分支、pack_layer / pack_to_int32', 'su');
  api.o.push('</svg>');
  return api.o.join('\n');
}

export const OUTPUTS = Object.freeze({ fp8: 'slime_fp8_block_scales.svg', int4: 'slime_fp8_block_int4_groups.svg' });

export function renderAll(m = model()) {
  return { [OUTPUTS.fp8]: renderFp8(m), [OUTPUTS.int4]: renderInt4(m) };
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  for (const [name, svg] of Object.entries(renderAll())) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
}
