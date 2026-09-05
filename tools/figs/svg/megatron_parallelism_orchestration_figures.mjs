// 图spec（先于实现）：同一world=16的R5，从两种order中的dense/expert形状到名单。
// 图1每条lane左侧是order、sizes、mask、stride，中部是4个完整组（R5所在组蓝色），
// 右侧是R5的PP名单。默认与alternative各两条，证明dense/expert共享PP但不共享DP。
// 底部从名单指向create_group/new_group，再到当前rank的handle；虚线标PyTorch边界。
// 图2两条lane对应HyperCommGrid base/expert view：反序reshape的轴→moveaxis后的轴→
// 同样四组名单→_pgs键与PGC字段；底部shared pp汇成同一handle，重复创建为橙色失败路径。
// 图3三个派生lane复用R5：hierarchical CP、dynamic DP×CP、partial DP划分；
// 画真实父名单、选择规则、R5逐级名单与新增group族，不把collective内部混入编排图。
// 数字与名单均从CFG/解算器生成；回归测试读取真实页面的例子表。
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const CFG = { world: 16, rank: 5, tp: 2, cp: 2, pp: 2, etp: 1, ep: 4, instances: 2 };
const product = a => a.reduce((p, v) => p * v, 1);
const range = n => Array.from({ length: n }, (_, i) => i);
export const fmt = a => '[' + a.join(',') + ']';
export const own = groups => groups.find(g => g.includes(CFG.rank));
export function orthogonal(sizes, mask, offset = 0) {
  let p = 1;
  const strides = sizes.map(s => { const q = p; p *= s; return q; });
  const coordinates = (index, selected) => {
    let step = 1;
    return sizes.map((s, i) => {
      if (!selected(i)) return 0;
      const c = Math.floor(index / step) % s; step *= s; return c;
    });
  };
  const inner = c => c.reduce((n, x, i) => n + x * strides[i], offset);
  const width = product(sizes.filter((_, i) => mask[i]));
  const groups = range(p / width).map(g => range(width).map(r =>
    inner(coordinates(g, i => !mask[i]).map((c, i) => c + coordinates(r, j => mask[j])[i]))));
  return { sizes, mask, strides, groups, width };
}
export function rankCase(alternative, expert) {
  const names = alternative ? ['tp','cp','ep','pp','dp'] : ['tp','cp','ep','dp','pp'];
  const values = expert
    ? {tp: CFG.etp, cp: 1, ep: CFG.ep, pp: CFG.pp, dp: CFG.world / (CFG.etp * CFG.ep * CFG.pp)}
    : {tp: CFG.tp, cp: CFG.cp, ep: 1, pp: CFG.pp, dp: CFG.world / (CFG.tp * CFG.cp * CFG.pp)};
  const sizes = names.map(n => values[n]);
  const token = expert ? ['ep'] : ['dp','cp'];
  return { alternative, expert, names, token, ...orthogonal(sizes, names.map(n => token.includes(n))),
    pp: orthogonal(sizes, names.map(n => n === 'pp')).groups };
}
// 独立模拟numpy的row-major reshape + moveaxis；不调用orthogonal求答案。
export function gridEnum(shape, names, requested, offset = 0) {
  const input = names.toReversed();
  const ordered = input.filter(n => requested.includes(n));
  const axes = [...input.filter(n => !ordered.includes(n)), ...ordered];
  const sizeByName = Object.fromEntries(names.map((n, i) => [n, shape[i]]));
  const rowStrides = axisNames => axisNames.map((_, i) => product(axisNames.slice(i + 1).map(n => sizeByName[n])));
  const outStride = rowStrides(axes), inStride = rowStrides(input);
  const flat = range(product(shape)).map(i => {
    const coords = Object.fromEntries(axes.map((n, j) => [n, Math.floor(i / outStride[j]) % sizeByName[n]]));
    return offset + input.reduce((v, n, j) => v + coords[n] * inStride[j], 0);
  });
  const width = product(ordered.map(n => sizeByName[n]));
  return { shape, names, input, axes, ordered, sourceAxes: ordered.map(n => input.indexOf(n)),
    targetAxes: range(ordered.length).map(i => names.length - ordered.length + i),
    groups: range(flat.length / width).map(i => flat.slice(i * width, (i + 1) * width)) };
}
export function hierarchical(ranks, sizes) {
  if (product(sizes) !== ranks.length) throw new Error('hierarchy product mismatch');
  return sizes.map((s, i) => {
    const u = product(sizes.slice(0, i)), l = product(sizes.slice(i + 1));
    return range(l).flatMap(outer => range(u).map(inner => range(s).map(x => ranks[(outer * s + x) * u + inner])));
  });
}
export const cases = [false, true].flatMap(a => [false, true].map(e => rankCase(a,e)));
export const grids = [gridEnum([2,2,2,2], ['tp','cp','dp','pp'], ['cp','dp']),
  gridEnum([1,4,2,2], ['expt_tp','ep','expt_dp','pp'], ['ep'])];
export function derived() {
  const c = cases[0], parent = own(c.groups);
  const e = cases[1], exptParent = own(orthogonal(e.sizes, e.names.map(n=>n==='dp')).groups);
  const cp = own(orthogonal(c.sizes, c.names.map(n => n === 'cp')).groups);
  const sizes = range(Math.floor(Math.log2(parent.length))).map(i => 2 ** i);
  const dynamic = sizes.map(s => range(parent.length / s).map(i => parent.slice(i * s, (i+1) * s)));
  return { parent, cp, exptParent, hcp: hierarchical(cp, [1,CFG.cp]), dynamic,
    partial: [range(CFG.instances).map(i=>parent.slice(i*parent.length/CFG.instances,(i+1)*parent.length/CFG.instances))],
    exptPartial: hierarchical(exptParent, [exptParent.length / CFG.instances, CFG.instances]) };
}
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const style = `<style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#263445;font-size:16px}.title{font-size:23px;font-weight:700}.dim{font-size:14px;fill:#526174}.cap{font-size:16px}.mono{font-family:Consolas,monospace;font-size:16px}.neutral{fill:#fff;stroke:#aeb9c6}.ghost{fill:#f5f7fa;stroke:#dae0e7}.acc1{fill:#e8f1fb;stroke:#2f6fba;stroke-width:2}.acc2{fill:#fff1e5;stroke:#c77b31;stroke-width:2}.main{stroke:#2f6fba;stroke-width:2.2;fill:none;marker-end:url(#arrow)}.aux{stroke:#9aa8b7;stroke-width:1.4;fill:none}.boundary{stroke:#9aa8b7;stroke-width:1.4;stroke-dasharray:6 5}</style>`;
const textAt = (x,y,s,cls='') => `<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`;
const box = (x,y,w,h,cls='neutral') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" class="${cls}"/>`;
const arrow = (x,y,x2,y2) => `<path d="M${x},${y} L${x2},${y2}" class="main"/>`;
const start = (h,title) => `<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="${h}" viewBox="0 0 1180 ${h}">${style}<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2f6fba"/></marker></defs><rect width="1180" height="${h}" fill="white"/>${textAt(25,36,title,'title')}`;
function groupRows(groups, x,y,w=286) {
  return groups.map((g,i) => box(x,y+i*35,w,30,g.includes(CFG.rank)?'acc1':'ghost') + textAt(x+12,y+21+i*35,fmt(g),'mono')).join('');
}
export function rankFigure() {
  let s = start(984,`同一 R${CFG.rank}：两种 order × dense / expert 分解`);
  s += textAt(25,65,`输入 R0…R${CFG.world-1}；TP=${CFG.tp} CP=${CFG.cp} PP=${CFG.pp}；ETP=${CFG.etp} EP=${CFG.ep}；offset=0`,'dim');
  s += textAt(25,97,'order → sizes / mask / stride')+textAt(474,97,'枚举全部成员名单')+textAt(852,97,`R${CFG.rank} 的 PP 组`);
  cases.forEach((c,i) => {
    const y = 116+i*174;
    s += box(18,y-7,1143,164,'ghost');
    s += textAt(32,y+18,`${c.alternative?'alternative':'default'} · ${c.expert?'expert':'dense'}`);
    s += textAt(32,y+45,c.names.join('-'),'mono');
    s += textAt(32,y+74,`sizes ${fmt(c.sizes)}   mask ${c.mask.map(Number).join('')}`,'mono');
    s += textAt(32,y+103,`stride ${fmt(c.strides)}`,'mono');
    s += textAt(32,y+133,`${c.token.join('-')}：组内变这些轴，其余轴固定`,'dim');
    s += arrow(427,y+67,462,y+67)+groupRows(c.groups,476,y+3);
    s += textAt(852,y+21,`另求 mask=${c.names.map(n=>Number(n==='pp')).join('')}`,'dim')+box(836,y+35,297,67,'acc1');
    s += textAt(856,y+62,fmt(own(c.pp)),'mono')+textAt(856,y+87,'dense / expert 完全相等','dim');
  });
  const c=cases[0], coord=c.sizes.map((a,i)=>Math.floor(CFG.rank/c.strides[i])%a);
  const selected=c.names.map((n,i)=>i).filter(i=>c.mask[i]);
  const fixed=c.names.map((n,i)=>i).filter(i=>!c.mask[i]);
  s+=textAt(28,837,`default dense：固定 ${fixed.map(i=>`${c.names[i]}=${coord[i]}`).join(' ')}；组内 ${selected.map(i=>`${c.names[i]}=${coord[i]}`).join(' ')}`,'mono');
  s+=textAt(28,866,`decompose → r = ${coord.map((v,i)=>`${v}×${c.strides[i]}`).join(' + ')} = ${CFG.rank}；mask只让组内坐标变化`,'mono');
  s += textAt(28,914,'每个 rank 按相同顺序遍历名单','cap')+arrow(298,908,346,908);
  s += textAt(360,914,'create_group → new_group','mono');
  s += `<line x1="704" y1="891" x2="704" y2="971" class="boundary"/>`;
  s += arrow(656,908,733,908)+textAt(750,914,'PyTorch：返回当前 rank 的 handle','cap');
  s += textAt(29,957,'不变量：每族名单覆盖全部 rank 一次；名单相等不保证不同建组调用的 handle 相同。','cap');
  return s+'</svg>';
}
export function gridFigure() {
  let s = start(736,'HyperCommGrid：反序数组 + moveaxis，把所选轴变成一行');
  s += textAt(25,68,`同一 R0…R${CFG.world-1} / R${CFG.rank}；沿用 default 布局；蓝色行是本 rank 接收的组`,'dim');
  grids.forEach((g,i) => {
    const y = 92+i*225;
    s += box(18,y,1143,211,'ghost');
    s += textAt(31,y+28,i===0?'base view：请求 cp + dp':'expert view：请求 ep');
    s += textAt(31,y+58,`shape ${fmt(g.shape)}`,'mono');
    s += textAt(31,y+87,g.input.join(' · '),'mono');
    s += textAt(31,y+111,'arange → reshape，左轴慢 / 右轴快','dim');
    s += textAt(31,y+146,`source ${fmt(g.sourceAxes)} → target ${fmt(g.targetAxes)}`,'mono');
    s += textAt(31,y+175,`moveaxis → ${g.axes.join(' · ')}`,'mono');
    s += arrow(499,y+100,539,y+100)+groupRows(g.groups,555,y+43,245);
    s += textAt(555,y+28,'reshape → 每行一个组','dim');
    s += arrow(813,y+100,843,y+100)+box(858,y+60,285,95,'acc1');
    s += textAt(873,y+87,i===0?'key = dp-cp':"key = (expert, ep)",'mono');
    s += textAt(873,y+117,i===0?'PGC.dp_cp':'PGC.ep','mono');
    s += textAt(873,y+142,`R${CFG.rank} ← ${fmt(own(g.groups))}`,'mono');
  });
  s += textAt(28,564,'shared_dims = pp','mono')+arrow(251,557,282,557)+textAt(301,564,`两套枚举均为 R${CFG.rank} → ${fmt(own(cases[0].pp))}`,'mono');
  s += arrow(770,557,805,557)+textAt(825,564,'一个 _pgs[pp] handle','mono');
  s += box(22,589,1136,48,'acc2')+textAt(38,618,'先 create_pg；get_pg 只取句柄。再建同键 → KeyError；destroy 按对象身份去重。','cap');
  s += textAt(28,676,'rank_enum','mono')+arrow(142,670,200,670)+textAt(218,676,'PyTorch new_subgroups_by_enumeration','mono');
  s += `<line x1="194" y1="650" x2="194" y2="718" class="boundary"/><line x1="654" y1="650" x2="654" y2="718" class="boundary"/>`;
  s += arrow(667,670,723,670)+textAt(742,676,'handle → _pgs[key] → PGC','mono');
  s += textAt(219,710,'依赖边界：NCCL内部启动不在本图证据内','dim');
  return s+'</svg>';
}
export function derivedFigure() {
  const d=derived();
  let s=start(535,`父组之后的派生规则：继续跟踪 R${CFG.rank}`);
  const lanes = [
    ['hierarchical CP',`CP ${fmt(d.cp)} / levels ${fmt([1,CFG.cp])}`,d.hcp.map(g=>fmt(own(g))).join(' → '),'每一级单独建组；乘积必须等于 CP'],
    ['dynamic DP×CP',`DP×CP ${fmt(d.parent)} / min=1`,d.dynamic.map(g=>fmt(own(g))).concat(fmt(d.parent)).join(' → '),'幂次连续切片；完整父组由 getter 回退'],
    ['partial DP',`DP×CP ${fmt(d.parent)} / instances=${CFG.instances}`,d.partial.map(g=>fmt(own(g))).join(' + '),'仅切 intra；inter 复用下方 expert 组'],
    ['partial expert DP',`EDP ${fmt(d.exptParent)} / instances=${CFG.instances}`,d.exptPartial.map(g=>fmt(own(g))).join(' + '),'各层新组；inter 由 dense / expert 共享，可启用 dp_replica']
  ];
  lanes.forEach((l,i)=>{const y=64+i*111;
    s+=box(20,y,1139,100,'ghost')+textAt(35,y+29,l[0])+textAt(35,y+59,l[1],'mono');
    s+=arrow(492,y+43,532,y+43)+box(552,y+15,585,42,'acc1')+textAt(568,y+43,l[2],'mono')+textAt(558,y+80,l[3],'dim');
  });
  s+=textAt(28,523,'只构造成员域；CP交换、梯度规约、参数聚合的执行与等待分别由13/16页拥有。','cap');
  return s+'</svg>';
}
export function generate(outDir) {
  mkdirSync(outDir,{recursive:true});
  for(const [name,fn] of [['ranks',rankFigure],['grid',gridFigure],['derived',derivedFigure]])
    writeFileSync(join(outDir,`megatron_orchestration_${name}.svg`),fn());
}
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1])
  generate(process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)),'../../../wiki/02_engineering/02_train_frameworks/megatron-lm/assets'));
