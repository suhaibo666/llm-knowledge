import {writeFileSync} from 'node:fs';

const out = new URL('./paged_attention_block_refcount.svg', import.meta.url);
const stage = [
  {x: 26, title: '① 建立 7 个已处理位置', sub: 'A: T_A = [7, 1]'},
  {x: 322, title: '② 分叉 B，共享历史', sub: 'A, B: T = [7, 1]'},
  {x: 618, title: '③ A 续写触发 CoW', sub: 'A: [7, 3]；B: [7, 1]'},
  {x: 914, title: '④ A 结束，B 仍引用', sub: 'B: T_B = [7, 1]'},
];
const C = {ink:'#0f172a', line:'#64748b', light:'#e2e8f0', blue:'#dbeafe', blueStroke:'#2563eb', orange:'#ffedd5', orangeStroke:'#ea580c', ghost:'#f8fafc', green:'#dcfce7', greenStroke:'#15803d'};
const esc = s => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text = (x,y,s,cls='label',anchor='start') => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;
const box = (x,y,w,h,fill=C.ghost,stroke=C.light,rx=7) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}"/>`;
function request(x,y,name,table,active=true) {
  if (!active) {
    return `${box(x,y,248,57,C.ghost,C.line)}${text(x+10,y+21,name+' 已结束','label')}${text(x+10,y+42,'无活跃块表；A 的引用已归还','small')}`;
  }
  const cells = table.map((v,i) => {
    const fill = v === '3' ? C.orange : C.blue;
    const stroke = v === '3' ? C.orangeStroke : C.blueStroke;
    return `${box(x+64+i*38,y+20,34,26,fill,stroke,4)}${text(x+81+i*38,y+38,v,'small','middle')}`;
  }).join('');
  return `${box(x,y,248,57,'#ffffff',C.line)}${text(x+10,y+20,name,'label')}${text(x+10,y+40,'逻辑块表','tiny')}${cells}`;
}
function pool(x,y,items) {
  const rows = items.map((item,i) => {
    const yy=y+31+i*47;
    const fill=item.ref===0?C.green:item.id==='3'?C.orange:C.blue;
    const stroke=item.ref===0?C.greenStroke:item.id==='3'?C.orangeStroke:C.blueStroke;
    return `${box(x,yy,246,38,fill,stroke)}${text(x+10,yy+16,'P'+item.id,'label')}${text(x+56,yy+16,item.kv,'tiny')}${text(x+10,yy+31,item.note,'tiny')}${text(x+229,yy+23,'ref '+item.ref,'small','end')}`;
  }).join('');
  return `${box(x-8,y,262,32,C.ghost,C.line)}${text(x+8,y+21,'物理块池','label')}${rows}`;
}
const stages = stage.map(s => `${text(s.x+128,28,s.title,'stage','middle')}${text(s.x+128,47,s.sub,'sub','middle')}`).join('');
const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1188" height="500" viewBox="0 0 1188 500" role="img" aria-labelledby="title desc">
<title id="title">PagedAttention 块表、写时复制和引用变化</title>
<desc id="desc">块长四，展示 A 的七个已处理位置被 B 分叉共享，A 写入共享尾块时复制到 P3，A 结束后 P3 无引用但字节仍在池中。</desc>
<style>
  .stage {font: 700 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:${C.ink}}
  .sub {font: 11px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:${C.line}}
  .label {font: 600 12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:${C.ink}}
  .small {font: 11px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:${C.ink}}
  .tiny {font: 9px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:${C.line}}
  .arrow {stroke:${C.line};stroke-width:1.4;fill:none;marker-end:url(#arrow)}
  .main {stroke:${C.blueStroke};stroke-width:2;fill:none;marker-end:url(#arrowBlue)}
  .cost {stroke:${C.orangeStroke};stroke-width:2;fill:none;marker-end:url(#arrowOrange)}
</style>
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="${C.line}"/></marker>
  <marker id="arrowBlue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="${C.blueStroke}"/></marker>
  <marker id="arrowOrange" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="${C.orangeStroke}"/></marker>
</defs>
<rect width="1188" height="500" fill="white"/>
${stages}
<line x1="296" y1="25" x2="312" y2="25" class="arrow"/><line x1="592" y1="25" x2="608" y2="25" class="arrow"/><line x1="888" y1="25" x2="904" y2="25" class="arrow"/>
${request(26,76,'请求 A',['7','1'])}
${pool(26,151,[{id:'7',kv:'位置 0–3',note:'完整块',ref:1},{id:'1',kv:'位置 4–6',note:'尾块：3 / 4',ref:1}])}
${request(322,76,'请求 A',['7','1'])}${request(322,140,'请求 B',['7','1'])}
${pool(322,215,[{id:'7',kv:'位置 0–3',note:'两个请求共享',ref:2},{id:'1',kv:'位置 4–6',note:'两个请求共享尾块',ref:2}])}
${request(618,76,'请求 A',['7','3'])}${request(618,140,'请求 B',['7','1'])}
${pool(618,215,[{id:'7',kv:'位置 0–3',note:'共享完整块',ref:2},{id:'1',kv:'位置 4–6 + b7',note:'B 独占写入 offset 3',ref:1},{id:'3',kv:'位置 4–6 复制 + a7',note:'A 的私有尾块',ref:1}])}
${request(914,76,'请求 A',['7','3'],false)}${request(914,140,'请求 B',['7','1'])}
${pool(914,215,[{id:'7',kv:'位置 0–3',note:'B 仍持有',ref:1},{id:'1',kv:'位置 4–6 + b7',note:'B 仍持有',ref:1},{id:'3',kv:'A 的旧 KV 字节仍在',note:'无引用，可领取后覆盖',ref:0}])}
${box(26,390,1136,21,C.ghost,C.light)}
${text(42,405,'蓝色：共享或仍被引用；橙色：A 的 CoW 私有尾块；绿色：旧字节尚在但 ref=0，可领取后覆盖。','tiny')}
${box(26,420,1136,57,C.ghost,C.light)}
${text(42,443,'块长 B = 4；位置 t 的逻辑块 = floor(t / 4)，offset = t mod 4。','small')}
${text(42,463,'绿色表示“字节尚在池中、但没有请求引用，下一次分配可覆盖”，不表示它仍可由内容匹配自动命中。','small')}
</svg>`;
writeFileSync(out, svg);
