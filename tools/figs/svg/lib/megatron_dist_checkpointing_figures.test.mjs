// 锁住 dist-checkpointing 图示的可执行契约：五张图上的每个数字都必须由同一组 CFG 与
// 复刻的源码算法推导，且与 19_megatron_dist_checkpointing_analysis.md 正文引用的数值逐个对齐
// —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_dist_checkpointing_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_dist_checkpointing_figures.mjs');
const trackedDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '19_megatron_dist_checkpointing_analysis.md',
);

const NAMES = [
  'megatron_ckpt_reshard.svg',
  'megatron_ckpt_access_grid.svg',
  'megatron_ckpt_greedy_save.svg',
  'megatron_ckpt_exchange_algos.svg',
  'megatron_ckpt_async_ladder.svg',
];

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

// 没有任何图元允许溢出画布 —— 溢出等于裁字，肉眼过一遍不可靠，这里机械判定。
// 注意：图元对图元的**重叠**由生成器里的 assertNoTextOverlap 负责，越界不等于不重叠。
function assertInsideCanvas(svg, name) {
  const { w, h } = viewBox(svg);
  const rects = svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  );
  for (const [, x, y, rw, rh] of rects) {
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

test('生成器同步产出五张 checkpoint 图', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-ckpt-figures-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);

    const [reshard, access, greedy, exchange, ladder] = await Promise.all(
      NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
    );

    // ---- 图 1：TP4×DP2 存、TP2×DP1 载 ----
    assert.match(reshard, /全局 \[8, 4\] bf16 = 64 B/);
    // 存端 8 个 rank 的描述子：同一 tp 位置两份，只有 replica_id 不同
    for (const tp of [0, 1, 2, 3]) {
      assert.match(
        reshard,
        new RegExp(`off=\\(${tp * 2},0\\)  shape=\\(2,4\\)`),
        `tp=${tp} 的存端描述子必须出现`,
      );
    }
    assert.equal((reshard.match(/replica_id=0/g) ?? []).length, 4, '四个 main replica');
    // 四张 DP 副本卡 + 「丢弃：rank 4, 5, 6, 7（replica_id=1）」那一条标注
    assert.equal((reshard.match(/replica_id=1/g) ?? []).length, 5, '四个 DP 副本加一条丢弃标注');
    assert.match(reshard, /丢弃：rank 4, 5, 6, 7（replica_id=1）/);
    assert.match(reshard, /写盘量 64 B 而不是 128 B/);
    assert.match(reshard, /省下 64 B 写盘与同等的 I\/O 时间/);
    // 载端两个 rank，各自 2 条 ReadItem 拼出 4×4
    assert.match(reshard, /off=\(0,0\) shape=\(4,4\) frag=\(2,1\)/);
    assert.match(reshard, /off=\(4,0\) shape=\(4,4\) frag=\(2,1\)/);
    assert.equal((reshard.match(/2 条 ReadItem 拼出 4×4 = 32 B/g) ?? []).length, 2);
    assert.match(reshard, /全局行 0–1，16 B/);
    assert.match(reshard, /全局行 6–7，16 B/);
    // 依赖边界必须画出来
    assert.match(reshard, /PyTorch DCP/);
    assert.match(reshard, /create_read_items_/);
    assert.match(reshard, /axis_fragmentations=\(4,1\)/);

    // ---- 图 2：访问计数的三种结局 ----
    assert.match(access, /axis_fragmentations = \(4,1\)/);
    assert.match(access, /torch.all\(cnt == 1\) 成立 → 通过/);
    assert.match(access, /cnt\[2\]=0 → CheckpointingException/);
    assert.match(access, /cnt\[1\]=2 → 同一条异常/);
    // 三条 lane 共 12 格，计数值分别是 [1,1,1,1] / [1,1,0,1] / [1,2,1,1]
    assert.equal((access.match(/>1<\/text>/g) ?? []).length, 4 + 3 + 3, '计数为 1 的格子总数');
    assert.equal((access.match(/>0<\/text>/g) ?? []).length, 1, '缺口 lane 恰好一格为 0');
    assert.equal((access.match(/>2<\/text>/g) ?? []).length, 1, '重叠 lane 恰好一格为 2');
    // 两个免检口子
    assert.match(access, /has_regular_grid=False/);
    assert.match(access, /prod\(global_shape\)/);
    assert.match(access, /all_gather_object/);

    // ---- 图 3：贪心分配 ----
    assert.match(greedy, /并行化组 4 个 rank、6 个分片、合计 32768 B/);
    assert.match(greedy, /coverage 2：\{0,1\}/);
    // 排序键的三种理由都要出现
    assert.match(greedy, /键②：coverage 最低/);
    assert.match(greedy, /键③：size 降序/);
    assert.match(greedy, /键④：shard_id 兜底/);
    // 摊平结果：四根条形各标 8192 B（x=901 是 barRow 的标注列）
    assert.equal(
      (greedy.match(/<text class="dim" x="901"[^>]*>8192 B<\/text>/g) ?? []).length,
      4,
      '四个 rank 各 8192 B',
    );
    // 分配轨迹的最后一行同样是四个 8192（累计字节）
    assert.match(greedy, /<text class="dim" x="320" y="577"[^>]*>8192<\/text>/);
    assert.match(greedy, />32768 B</, '无 wrapper 那根独柱');
    assert.match(greedy, /从 32768 B 降到 8192 B，即 4\.0×/);
    assert.match(greedy, /上限就是组大小 4/);
    assert.match(greedy, /纯 DP 组里所有分片 coverage 相等，键② 不起作用/);

    // ---- 图 4：三条 exchange 数据面 ----
    assert.match(exchange, /6 个分片需要交换，0 个 coverage=1 的分片整条跳过/);
    assert.match(exchange, /6 条 broadcast，报文大小逐条不同/);
    assert.match(exchange, /2 次 all_gather（本例只有一种 dtype）/);
    assert.match(exchange, /8 个槽位里 2 个是空张量/);
    assert.match(exchange, /空载率 25%/);
    assert.match(exchange, /1 次 all_gather_object/);
    for (const lane of ['① broadcast（默认）', '② gather_rounds', '③ gather_object']) {
      assert.ok(exchange.includes(lane), `lane ${lane} 必须单独出现`);
    }
    // 三条 docstring 自陈，逐条锁住
    assert.match(exchange, /A reasonable tradeoff in/);
    assert.match(exchange, /almost empty all_gathers/);
    assert.match(exchange, /can be used for debugging/);
    // 共享的第一条规则与它的 TODO
    assert.match(exchange, /len\(all_ranks_for_shard\[shard_id\]\) == 1/);
    assert.match(exchange, /Currently handling this case saves most of the work though/);

    // ---- 图 5：异步完成阶梯 ----
    for (const stage of [
      '1. created', '2. staged（D2H 完成）', '3. submitted',
      '4. written（本 rank 的字节落盘）', '5. finalized（全局元数据成文）',
      '6. visible（可被续训选中）',
    ]) {
      assert.ok(ladder.includes(stage), `阶梯 ${stage} 必须出现`);
    }
    // 只有第 2 级阻塞训练进程 —— 这是本图的结论
    assert.equal((ladder.match(/>训练进程在此等待</g) ?? []).length, 1);
    assert.equal((ladder.match(/>训练进程不等待</g) ?? []).length, 5);
    assert.match(ladder, /preload_tensors/);
    assert.match(ladder, /write_preloaded_data_multithread/);
    assert.match(ladder, /iter_finalize_fn 写 tracker 文件/);
    assert.match(ladder, /Hence the barrier/);

    // ---- 已跟踪资产必须与生成器同步 ----
    const tracked = await Promise.all(
      NAMES.map((name) => readFile(join(trackedDir, name), 'utf8')),
    );
    [reshard, access, greedy, exchange, ladder].forEach((svg, i) => {
      assert.equal(svg, tracked[i], `已跟踪的 ${NAMES[i]} 必须与生成器同步`);
    });

    for (const [name, svg] of [
      ['reshard', reshard], ['access', access], ['greedy', greedy],
      ['exchange', exchange], ['ladder', ladder],
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

  // 五张图都被正文以外链 svg 的方式引用，且不许内联 <svg>
  for (const name of NAMES) {
    assert.ok(
      new RegExp(`!\\[[^\\]]+\\]\\(assets/${name.replace('.', '\\.')}\\)`).test(page),
      `正文必须引用 ${name}`,
    );
  }
  assert.doesNotMatch(page, /<svg/);

  // 图上算出来的数字必须在正文里出现，防止图与正文各自漂移。
  // 自检：只改正文、不改图，这些断言必须变红。
  for (const needle of [
    // 图 1
    '$8\\times4\\times2=64$ B',
    '`local_shape=(2,4)`',
    '`axis_fragmentations=(4,1)`',
    '`local_shape=(4,4)`',
    '`axis_fragmentations=(2,1)`',
    '4 个 chunk、共 **64 B**，而不是 8 片 128 B',
    '各贡献 16 B，拼出 $4\\times4=32$ B',
    'create_read_items_for_chunk_list',
    // 图 2
    '`[1,1,1,1]`',
    'prod(global_shape)',
    'has_regular_grid',
    // 图 3
    '合计 32768 B',
    '**8192 / 8192 / 8192 / 8192**',
    '$32768/4$',
    '从 32768 B 降到 8192 B，即 4.0×',
    '6144 B',
    '纯 DP 组里键 ② 不起作用',
    // 图 4
    '**6 条** broadcast',
    '**2 轮**',
    '**2 个是 `torch.empty(0)`**，空载率 25%',
    'almost empty all_gathers',
    // 图 5
    '**六级**',
    'preload_tensors',
    'iter_finalize_fn',
    'Hence the barrier',
  ]) {
    assert.ok(page.includes(needle), `正文必须引用图上的 ${needle}`);
  }
});

test('页面不再携带 path:line 引用', async () => {
  const page = await readFile(pagePath, 'utf8');
  // 配置契约表里的 `:NNN` 行号列是本库既定格式，单独放行；正文里不许再出现 path.py:NNN
  const offenders = [...page.matchAll(/[\w/]+\.py:\d+/g)].map((m) => m[0]);
  assert.deepEqual(offenders, [], `正文仍有 path:line 引用：${offenders.join(', ')}`);
});
