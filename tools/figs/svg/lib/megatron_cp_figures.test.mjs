// 锁住 CP 图示的可执行契约：四张图上的每个数字都必须由同一组 CFG 推导，
// 且与 13_megatron_cp_analysis.md 正文引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_cp_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_cp_figures.mjs');
const trackedDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '13_megatron_cp_analysis.md',
);

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

test('生成器同步产出四张 CP 图', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-cp-figures-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);

    const partition = await readFile(join(outputDir, 'megatron_cp_sequence_partition.svg'), 'utf8');
    const schedules = await readFile(join(outputDir, 'megatron_cp_comm_schedules.svg'), 'utf8');
    const undo = await readFile(join(outputDir, 'megatron_cp_zigzag_undo.svg'), 'utf8');
    const planes = await readFile(join(outputDir, 'megatron_cp_nonstandard_planes.svg'), 'utf8');

    // ---- 图 1：切法与因果代价 ----
    assert.match(partition, /c=4 · S=16 · 2c=8 块 · 每块 2 token · 每 rank 本地 S\/c=4 token/);
    assert.match(partition, /rank r 取第 r 块与第 7-r 块/);
    assert.match(partition, /_get_batch_on_this_cp_rank_per_sequence_balancing/);
    assert.match(partition, /\[cp_rank, 2·cp_size-cp_rank-1\]/);
    // zigzag 的四对块：rank r -> (r, 7-r)
    for (const [rank, early, late] of [[0, 0, 7], [1, 1, 6], [2, 2, 5], [3, 3, 4]]) {
      assert.match(
        partition,
        new RegExp(`块 ${early} \\+ 块 ${late} → 本地 4 token`),
        `rank ${rank} 的配对必须出现在图上`,
      );
    }
    // 连续切的阶梯与 zigzag 的等长：这四组数字正文逐个引用
    for (const value of [10, 26, 42, 58]) {
      assert.match(partition, new RegExp(`${value} 格`), `连续切的 ${value} 必须出现`);
    }
    assert.equal((partition.match(/>34 格</g) ?? []).length, 4, 'zigzag 必须四个 rank 都是 34 格');
    assert.match(partition, /最贵\/最便宜 = 58\/10 = 5\.8×/);
    assert.match(partition, /每 rank 恰好 136\/4 = 34/);
    assert.match(partition, /块级裁剪后每 rank 9\/16 个 2×2 块/);
    assert.match(partition, /即 36 格，其中 34 格因果有效/);
    assert.match(partition, /多出的 2 格是对角块的粒度浪费/);
    assert.match(partition, /seq_length 必须被 2c=8 整除/);
    // cp_partition_mode 两种取值与门控
    assert.match(partition, /zigzag（默认，③ 的切法）/);
    assert.match(partition, /contiguous（② 的区间划分）/);
    assert.match(partition, /variant 限 dsv4_hybrid \/ gdn \/ kda/);
    assert.match(partition, /每 rank 持有 S\/c=4 个 token/);

    // ---- 图 2：四条 lane ----
    assert.match(schedules, /c=4 · S=16 · 每 rank 4 token（块 r 与块 7-r）· TP 后 a=8 个 query head/);
    assert.match(
      schedules,
      /p2p 288 · all_gather 512 · a2a 272 · a2a\+p2p 288；理论下界 a·S\(S\+1\)\/2\/c = 272/,
    );
    for (const lane of ['① p2p', '② all_gather', '③ a2a', '④ a2a\\+p2p']) {
      assert.match(schedules, new RegExp(lane), `lane ${lane} 必须单独出现`);
    }
    // 每条 lane 的五列都在
    for (const heading of ['Megatron 侧装配', '本地计算', '上线的数据', '重构与反向', '增量代价']) {
      assert.equal(
        (schedules.match(new RegExp(`>${heading}<`, 'g')) ?? []).length,
        4,
        `${heading} 必须四条 lane 各一份`,
      );
    }
    // TE / 原生边界：三条 ghost lane + 一条 Megatron 可证 lane
    assert.equal((schedules.match(/>TransformerEngine</g) ?? []).length, 3);
    assert.equal((schedules.match(/>Megatron 可证</g) ?? []).length, 1);
    assert.equal((schedules.match(/>无 TE 边界</g) ?? []).length, 1);
    assert.equal(
      (schedules.match(/↑ 这四列全在 TE 内核里，Megatron 源码不可证/g) ?? []).length,
      3,
    );

    // p2p lane
    assert.match(schedules, /第 i 步吃 rank \(r-i\) mod 4 的 KV/);
    assert.match(schedules, /块级裁剪 9\/16 → 288 格/);
    assert.match(schedules, /3 次交换 ×2 = 192 head-row/);
    assert.match(schedules, /online-softmax 合并 m \/ ℓ \/ out/);
    // all_gather lane（唯一 Megatron 可证的一条）
    assert.match(schedules, /AttentionFuncionWithContextParallel/);
    assert.match(schedules, /heads_k_stride 写死为 1/);
    assert.match(schedules, /无裁剪：8×4×16 = 512 格/);
    assert.match(schedules, /是下界 272 的 1\.9 倍/);
    assert.match(schedules, /16 次 AG = 192 head-row/);
    assert.match(schedules, /probs 全量保存 512 格\/层/);
    assert.match(schedules, /RS 同步发起，无 async_op/);
    // a2a lane
    assert.match(schedules, /每 rank 只留 a\/c = 2 个 head/);
    assert.match(schedules, /2×136 = 272 格/);
    assert.match(schedules, /4×24 = 96 head-row/);
    assert.match(schedules, /是 ring 的 0\.5 倍（MHA、c=4）/);
    // a2a+p2p lane：分层组由 einops 重排推出
    assert.match(schedules, /hierarchical sizes = \[2, 2\]/);
    assert.match(schedules, /低层组 \{0,1\} \{2,3\}/);
    assert.match(schedules, /高层组 \{0,2\} \{1,3\}/);
    assert.match(schedules, /低层 A2A 后持 8 行 × 4 head/);
    assert.match(schedules, /块级裁剪 18\/32 块/);
    assert.match(schedules, /4×18×4 = 288 格/);
    assert.match(schedules, /低层 4 次 A2A = 64 head-row/);
    assert.match(schedules, /= 64 head-row 走跨节点/);
    assert.match(schedules, /每 CP 组多 4 个子 communicator/);
    // 两种调度在本例上 KV 流量相同，这是图注要证的结论
    assert.match(schedules, /在本例都是 192 head-row/);

    // ---- 图 3：zigzag 还原的置换与轴切换 ----
    // 置换本身：缓冲块序、undo 的 order、还原结果、redo 的 order，四组数字互相锁死
    assert.match(undo, /c=4 · S=16 · 2c=8 块 · 每块 2 token · t=4/);
    assert.match(undo, /_undo_attention_load_balancing/);
    for (const [rank, early, late] of [[0, 0, 7], [1, 1, 6], [2, 2, 5], [3, 3, 4]]) {
      assert.match(
        undo,
        new RegExp(`rank ${rank}：块 ${early},${late}`),
        `rank ${rank} 在 A2A 缓冲里的两块必须标出`,
      );
    }
    assert.match(undo, /前半 2i → \{0,2,4,6\}/);
    assert.match(undo, /后半 8-2i-1 → \{7,5,3,1\}/);
    assert.match(undo, /order = \{0,2,4,6,7,5,3,1\}/);
    assert.match(undo, /恰好把 \{0,7,1,6,2,5,3,4\} 拉直成 \{0,1,2,3,4,5,6,7\}/);
    assert.match(undo, /order = \{0,7,1,6,2,5,3,4\}/); // _redo，等于缓冲块序
    assert.match(undo, /token 层面：0,1,14,15,2,3,12,13,… → 0,1,2,…,15/);
    // docstring 的 cp_size=3 例子必须由同一段代码复算出来
    assert.match(undo, /converts 162534/);
    assert.match(undo, /得到 162534 → 123456，与之一致/);
    // 轴切换的形状与"只搬不增"
    assert.match(undo, /\[4, B, 1544\]/);
    assert.match(undo, /\[4, B, 2056\]/);
    assert.match(undo, /\[16, B, 514\]/);
    assert.match(undo, /z 512 · x 512 · B 256 · C 256 · dt 8/);
    assert.match(undo, /4×2056 = 16×514 = 8224/);
    assert.match(undo, /通道 1544 → 2056，\+33\.2%/);
    assert.match(undo, /ngroups\/t = 2 &lt; c = 4/); // esc() 会把 < 转义成实体
    assert.match(undo, /→ 复制 2 份，ngroups per rank = 1/);
    // 参数切片与三条守卫
    assert.match(undo, /dt_bias \/ A_log：\[8\] → \[2\]/);
    assert.match(undo, /conv1d 权重 \[1024, 1, 4\] → \[384, 1, 4\]/);
    assert.match(undo, /= 128 \+ 2×1×128 = 384/);
    assert.match(undo, /本例 8 % 4 = 0，每 rank 2 个 head/);
    assert.match(undo, /本例 2 &lt; 4 且 4 % 2 = 0 → 走复制分支/);
    assert.match(undo, /五段宽度并不相同（512 \/ 512 \/ 512 \/ 512 \/ 8）/);

    // ---- 图 4：三条非标准数据面 ----
    assert.match(planes, /c=4 · S=16 · t=4 · 每 rank 4 token/);
    assert.match(planes, /linear_num_key_heads=16、mamba_num_heads=32 取字段默认值/);
    for (const lane of ['① chunkwise', '② headwise', '③ Mamba CP']) {
      assert.match(planes, new RegExp(lane), `lane ${lane} 必须单独出现`);
    }
    // 沿用图 2 的五列语法，两张图才能并排读
    for (const heading of ['Megatron 侧装配', '本地计算', '上线的数据', '重构与反向', '增量代价']) {
      assert.equal(
        (planes.match(new RegExp(`>${heading}<`, 'g')) ?? []).length,
        3,
        `${heading} 必须三条 lane 各一份`,
      );
    }
    // 边界语义：只有 chunkwise 把 CP 语义交给了外部内核
    assert.equal((planes.match(/>FLA cp_context</g) ?? []).length, 1);
    assert.equal((planes.match(/>Megatron 可证</g) ?? []).length, 2);
    assert.equal((planes.match(/>无 CP 边界</g) ?? []).length, 2);
    assert.equal((planes.match(/>FLA</g) ?? []).length, 1);
    assert.equal(
      (planes.match(/↑ 这四列全在 FLA 内核里，Megatron 源码不可证/g) ?? []).length,
      1,
    );
    // 同一份工作量的两种摆法 —— 本节的核心恒等式
    assert.match(planes, /chunkwise 是 4 token × 4 头 = 16 k head-row/);
    assert.match(planes, /headwise 是 16 token × 1 头 = 16 k head-row/);
    // chunkwise lane
    assert.match(planes, /没有 CLI flag，只能改 config/);
    assert.match(planes, /保留全部 4 个 k 头 \/ 8 个 v 头/);
    assert.match(planes, /本地 4×4 = 16 k head-row/);
    assert.match(planes, /conv 每段还要 K-1 = 3 行左邻上下文/);
    assert.match(planes, /不要求 head 数被 c=4 整除/);
    // headwise lane
    assert.match(planes, /非 zigzag 入口直接 ValueError/);
    assert.match(planes, /每 rank 只留 1 个 k 头 \/ 2 个 v 头/);
    assert.match(planes, /本地 16×1 = 16 k head-row，与 ① 相同/);
    assert.match(planes, /本例 16 % 16 = 0，每 rank 1 个 k 头/);
    assert.match(planes, /每次搬本地的 \(c-1\)\/c = 3\/4/);
    // Mamba lane
    assert.match(planes, /每 rank 2\/8 个 SSM head/);
    assert.match(planes, /conv 通道 128\+2×1×128 = 384/);
    assert.match(planes, /pre_conv_ssm 5 次 A2A（z x B C dt）/);
    assert.match(planes, /合计 7704 通道-行 \/ 6 次/);
    assert.match(planes, /组状态复制 1544 → 2056 通道/);

    // ---- 已跟踪资产必须与生成器同步 ----
    const tracked = Object.fromEntries(
      await Promise.all(
        [
          'megatron_cp_sequence_partition.svg',
          'megatron_cp_comm_schedules.svg',
          'megatron_cp_zigzag_undo.svg',
          'megatron_cp_nonstandard_planes.svg',
        ].map(async (name) => [name, await readFile(join(trackedDir, name), 'utf8')]),
      ),
    );
    assert.equal(partition, tracked['megatron_cp_sequence_partition.svg'], '已跟踪的切分图必须与生成器同步');
    assert.equal(schedules, tracked['megatron_cp_comm_schedules.svg'], '已跟踪的调度对照图必须与生成器同步');
    assert.equal(undo, tracked['megatron_cp_zigzag_undo.svg'], '已跟踪的还原置换图必须与生成器同步');
    assert.equal(planes, tracked['megatron_cp_nonstandard_planes.svg'], '已跟踪的三数据面图必须与生成器同步');

    for (const [name, svg] of [
      ['partition', partition],
      ['schedules', schedules],
      ['undo', undo],
      ['planes', planes],
    ]) {
      assert.match(svg, /^<svg[\s\S]*<\/svg>\s*$/);
      assert.doesNotMatch(svg, /undefined|NaN/);
      assert.doesNotMatch(svg, /\[\[[^\]]{3,}\]\]/, `${name}: wikilink 不许漏进标注`);
      assertInsideCanvas(svg, name);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('页面正文引用的数值与图上一致', async () => {
  const page = await readFile(pagePath, 'utf8');

  // 四张图都被正文以外链 svg 的方式引用，且不许内联 <svg>
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_cp_sequence_partition\.svg\)/);
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_cp_comm_schedules\.svg\)/);
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_cp_zigzag_undo\.svg\)/);
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_cp_nonstandard_planes\.svg\)/);
  assert.doesNotMatch(page, /<svg/);

  // 图上算出来的数字必须在正文里出现，防止图与正文各自漂移
  for (const needle of [
    '10 / 26 / 42 / 58', // 连续切的阶梯
    '5.8', // 不均衡比
    '$136/4=34$', // zigzag 的均衡值
    '9/16', // 块级裁剪
    '288', // ring 的本地格数
    '512', // 原生 all-gather 的本地格数
    '272', // Ulysses 与理论下界
    '192 head-row', // ring 与 all-gather 的同量 KV 流量
    '96 head-row', // Ulysses
    '64 head-row', // 分层两级各自
    '{0,1} 与 {2,3}', // 分层低层组
    '{0,2} 与 {1,3}', // 分层高层组
    // ---- 图 3 / 图 4：两条非标准数据面（§2.5）----
    '**0, 2, 4, 6, 7, 5, 3, 1**', // _undo 的 order
    '**0, 7, 1, 6, 2, 5, 3, 4**', // _redo 的 order，等于 A2A 缓冲块序
    '`162534` → `123456`', // docstring 的 cp_size=3 例子
    '0,1,14,15,\\;2,3,12,13,\\;4,5,10,11,\\;6,7,8,9', // 缓冲的 token 序
    '$4\\times4=16$', // chunkwise 的本地 k head-row
    '$16\\times1=16$', // headwise 的同一个数
    '$4\\times8=32=16\\times2$', // v 头同理
    '$16/4=4$', // chunkwise 每 rank 的 k 头数
    '$16/16=1$', // headwise 每 rank 的 k 头数
    '**1544**', // Mamba 五段合计通道
    '**2056**', // 组状态复制之后
    '$[16,B,514]$', // 换轴之后
    '$4\\times2056=16\\times514=8224$', // 只搬不增
    '$+33.2\\%$', // 复制带来的膨胀
    '$128+2\\times1\\times128=384$', // conv1d_channels()
    '$[1024,1,4]$', // conv1d 权重的 cp1 形状
    '$[384,1,4]$', // 切片之后
    '$K-1=3$', // causal-conv 的左邻上下文
  ]) {
    assert.ok(page.includes(needle), `正文必须引用图上的 ${needle}`);
  }
});
