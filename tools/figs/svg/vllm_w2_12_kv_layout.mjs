/** Conceptual packed-layout illustration for vLLM 12, source baseline 199cb9b.
 * Given group memberships illustrate the 3+3 / 3 / 2 grouping shape tested in
 * test_mixed_page_size_groups_use_spec_compatibility. Page sizes below are
 * explicit teaching inputs, not measured/model-specific byte counts.
 * All widths, padding and capacity labels are derived from those inputs.
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const output=path.join(repo,'wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_w2_12_packed_layout.svg');
const sizes={A:2,B:1,C:4};
const groups=[['A0','A1','A2','B0','B1','B2'],['C0','C2','C4'],['C1','C3']];
const totals=groups.map(g=>g.reduce((n,l)=>n+sizes[l[0]],0));
const stride=Math.max(...totals),memory=60,blocks=Math.floor(memory/stride);
const W=800,H=442,x0=195,unit=45;
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,cls='body',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;
let parts=[`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc"><title id="title">Packed KV cache：不同group解释同一物理block范围</title><desc id="desc">三个group的每block字节数分别为${totals.join('、')} KiB，统一stride取最大值${stride} KiB。group视图互相覆盖，但同一block ID同一时刻只交给一个group。</desc><style>text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#0f172a}.title{font-size:21px;font-weight:600}.body{font-size:17px}.small{font-size:15px;fill:#475569}.neutral{fill:#fff;stroke:#64748b}.ghost{fill:#f8fafc;stroke:#94a3b8;stroke-dasharray:4 3}.acc1{fill:#dbeafe;stroke:#2563eb}.acc2{fill:#ffedd5;stroke:#ea580c}</style><rect width="100%" height="100%" fill="white"/>`];
parts.push(text(24,32,'Packed layout：相同 block ID 的三种 group 视图','title'));
parts.push(text(24,61,'教学输入：A page 2 KiB，B page 1 KiB，C page 4 KiB','small'));
parts.push(text(24,85,'每行是不同解释；不表示同一 ID 可同时交给三个 group。','small'));
for(let i=0;i<groups.length;i++){
 const y=125+i*83;
 parts.push(text(24,y+24,`group ${i}`),text(24,y+48,`${totals[i]} KiB / block`,'small'));
 let pos=0;
 for(const label of groups[i]){
  const w=sizes[label[0]]*unit;
  parts.push(`<rect x="${x0+pos*unit}" y="${y}" width="${w}" height="58" class="${i===0?'acc1':'neutral'}"/>`);
  parts.push(text(x0+pos*unit+w/2,y+25,label,'body','middle'),text(x0+pos*unit+w/2,y+46,`${sizes[label[0]]} KiB`,'small','middle'));
  pos+=sizes[label[0]];
 }
 const pad=stride-pos;
 if(pad){parts.push(`<rect x="${x0+pos*unit}" y="${y}" width="${pad*unit}" height="58" class="acc2"/>`,text(x0+(pos+pad/2)*unit,y+25,'空余','body','middle'),text(x0+(pos+pad/2)*unit,y+46,`${pad} KiB`,'small','middle'));}
}
parts.push(text(24,395,`block stride = max(${totals.join(', ')}) = ${stride} KiB；下一 ID 从此范围后开始`));
parts.push(text(24,424,`KV 可用内存 ${memory} KiB → ${blocks} 个 pool blocks；扣除 null block 后可分配 ${blocks-1} 个`,'small'));
parts.push('</svg>');fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,parts.join('\n'));console.log(output);
