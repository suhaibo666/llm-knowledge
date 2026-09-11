// Source baseline: vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae.
// Conceptual layout, not an observed GPU allocation. Figure specification:
// Show two layers and three symbolic block columns (6,7,8). Layer-major rows
// order whole layers; block-major rows order each block's layer pages. Highlight
// block 7 in blue in both. Zoom into layer1/block7/offset8/head0: D=128 K then V,
// orange marks element d=5 and D+d=133. Caption exposes stride dependence.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const out=path.join(root,'wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_kv_slot_address_layout.svg');
const D=128,e=2,block=7,offset=8,B=16,H=1,d=5;
const slot=block*B+offset,k=((block*B+offset)*H*2*D+d)*e,v=k+D*e;
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const t=(x,y,s,c='body')=>`<text x="${x}" y="${y}" class="${c}">${esc(s)}</text>`;
const box=(x,y,w,h,c)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${c}"/>`;
let s=[`<svg id="kv-address-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 555" role="img" aria-labelledby="kv-title kv-desc"><title id="kv-title">同一 block id 的跨层视图与 K/V 元素地址</title><desc id="kv-desc">两种共享 backing 排列的局部切片。普通 FlashAttention 每层 view 为块、head、state、内容。K/V 在内容轴相邻，层间连续性由布局决定。</desc><style>#kv-address-svg text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#0f172a}#kv-address-svg .title{font-size:20px;font-weight:600}#kv-address-svg .body{font-size:16px}#kv-address-svg .small{font-size:14px;fill:#475569}#kv-address-svg .neutral{fill:#fff;stroke:#64748b}#kv-address-svg .ghost{fill:#f8fafc;stroke:#94a3b8}#kv-address-svg .acc1{fill:#dbeafe;stroke:#2563eb}#kv-address-svg .acc2{fill:#ffedd5;stroke:#ea580c}</style><rect width="900" height="555" fill="white"/>`];
s.push(t(24,32,'同一个编号 ≠ 同一种物理排列','title'),t(24,58,'共享 backing 的局部切片：每格是一层的一页；省略其他块与层。','small'));
s.push(t(24,91,'LBNHC：按层排，再排块；地址向右增加 →','body'));
const layerMajor=[[0,6],[0,7],[0,8],[1,6],[1,7],[1,8]];
const blockMajor=[[0,6],[1,6],[0,7],[1,7],[0,8],[1,8]];
function lane(items,y){items.forEach(([l,b],i)=>{let x=145+i*118;s.push(box(x,y,108,64,b===7?'acc1':'neutral'),t(x+12,y+25,`layer ${l}`),t(x+12,y+49,`block ${b}`));if(i<5)s.push(t(x+109,y+37,'›','small'));});}
lane(layerMajor,106);
s.push(t(24,196,'仅展示 layer 0/1 与 block 6/7/8；省略范围不表示地址相邻。','small'));
s.push(t(24,226,'BLNHC：按块排，再排层；地址向右增加 →','body'));
lane(blockMajor,240);
s.push(t(24,340,`放大：layer 1 / block ${block} / offset ${offset} / head 0；slot = ${slot}`,'body'));
s.push(t(24,365,'逻辑 view：[block, head, state, content]；以下是一个 head 的 content 轴。','small'));
s.push(box(145,386,350,66,'acc1'),box(495,386,350,66,'neutral'),t(166,412,`K：content 0–${D-1}`),t(166,438,`${D} 元素 × ${e} B = ${D*e} B`,'small'),t(516,412,`V：content ${D}–${2*D-1}`),t(516,438,`${D} 元素 × ${e} B = ${D*e} B`,'small'));
s.push(box(160,461,330,29,'acc2'),t(170,482,`K 的 d=${d} → cache[7,0,8,${d}]`,'small'),box(510,461,335,29,'acc2'),t(520,482,`V 的 d=${d} → cache[7,0,8,${D+d}]`,'small'));
s.push(t(24,519,`无 padding、LBNHC、H=${H}、B=${B}：K 地址偏移 ${k} B；V 偏移 ${v} B。`,'small'),t(24,541,'两者只差 D×e；跨层 / 跨 head 的连续性仍要读真实 stride。','small'),'</svg>');
fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,s.join('\n'));console.log(out);
