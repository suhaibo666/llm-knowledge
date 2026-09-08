// Source-faithful teaching transform: vLLM@199cb9b9 auto_awq._convert_awq_to_standard_format
// and quant_utils.pack_quantized_values_into_int32. No GPU kernel layout simulation.
// Run: node tools/figs/svg/vllm_w2_21_pack_layout.mjs > wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_w2_21_pack_layout.svg
import { pathToFileURL } from 'node:url';
export const reverseOrder = [0,4,1,5,2,6,3,7];
export const pack = values => values.reduce((word, value, i) => word | BigInt(value) << BigInt(4*i), 0n);
export const unpack = word => Array.from({length:8}, (_,i) => Number(word >> BigInt(4*i) & 15n));
export const hex = word => `0x${word.toString(16).padStart(8,'0')}`;
export function example() {
  const codes = Array.from({length:8}, (_,k) => Array.from({length:8}, (_,n) => k+n));
  const awqSlots = codes.map(row => {
    const raw=Array(8); reverseOrder.forEach((slot,n)=>raw[slot]=row[n]); return raw;
  });
  const awqWords = awqSlots.map(pack);
  const recovered = awqWords.map(word => reverseOrder.map(i=>unpack(word)[i]));
  const standardWords = Array.from({length:8}, (_,n)=>pack(recovered.map(row=>row[n])));
  const dequant = codes.map(row=>(row[0]-8)*0.5);
  const y = dequant[0]+dequant[7];
  const fp8Values = Array.from({length:127}, (_,byte)=> {
    const e=byte>>3, m=byte&7;
    return {byte, value:e===0 ? m*2**-9 : (1+m/8)*2**(e-7)};
  });
  const weight=[0.546875,1.09375,2.1875,224];
  const scale=Math.max(...weight.map(Math.abs))/448;
  const fp8=weight.map(w=>fp8Values.reduce((best,c)=> {
    const a=Math.abs(c.value-w/scale), b=Math.abs(best.value-w/scale);
    return a<b || (a===b && c.byte%2===0) ? c : best;
  }));
  return {codes,awqSlots,awqWords,recovered,standardWords,dequant,y,weight,scale,fp8,
    reference:weight.slice(0,3).reduce((a,b)=>a+b),
    quantized:fp8.slice(0,3).reduce((a,b)=>a+b.value*scale,0)};
}
export function render() {
  const d=example();
  const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const items=[];
  const text=(x,y,s,cls='label',anchor='start')=>items.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const rect=(x,y,w,h,cls='neutral')=>items.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" class="${cls}"/>`);
  const arrow=(x1,y1,x2,y2,cls='main')=>items.push(`<path d="M${x1} ${y1}L${x2} ${y2}" class="arrow ${cls}" marker-end="url(#${cls})"/>`);
  text(32,38,'同一组 4-bit 编码：AWQ 输出轴 pack → 逻辑矩阵 → 标准输入轴 pack','title');
  text(32,72,'教学块 K=N=8；u[k,n]=k+n。每格是编码；改变位置不改 scale / zero-point。','cap');
  text(32,120,'AWQ checkpoint [8,1] int32','heading');
  text(32,148,'沿 N 打包 · 每行一个 word','cap');
  text(415,120,'恢复后的逻辑编码 [K,N] = [8,8]','heading');
  text(415,148,'行是 K，列是 N；蓝色追踪 n=0','cap');
  text(903,120,'位序与数值放大','heading');
  const gx=440,gy=192,cw=45;
  for(let n=0;n<8;n++)text(gx+n*cw+cw/2,180,`n${n}`,'cap','middle');
  for(let k=0;k<8;k++) {
    text(26,gy+k*cw+29,`k${k}`,'cap');
    rect(62,gy+k*cw,255,42,k===0?'acc2':'neutral');
    text(80,gy+k*cw+28,hex(d.awqWords[k]),'mono');
    text(gx-16,gy+k*cw+29,`k${k}`,'cap','end');
    for(let n=0;n<8;n++) {
      rect(gx+n*cw,gy+k*cw,cw-2,cw-2,n===0?'acc1':k===0?'acc2':'neutral');
      text(gx+n*cw+21,gy+k*cw+28,d.codes[k][n],'label','middle');
    }
  }
  arrow(324,210,401,210); text(335,244,'拆位','cap');text(331,266,'再逆序','cap');
  rect(903,174,379,169,'ghost');
  text(921,202,'row 0 · 原始低位 → 高位','cap');
  text(921,233,d.awqSlots[0].join('  '),'mono');
  text(921,264,'按索引 0,4,1,5,2,6,3,7 取回','cap');
  text(921,298,d.recovered[0].join('  '),'mono');
  text(921,326,'恢复逻辑 n=0…7','cap');
  arrow(808,214,887,214,'aux');
  rect(903,365,379,188,'acc1');
  text(921,395,'column 0 · 标准低位 → 高位','cap');
  text(921,429,d.codes.map(row=>row[0]).join('  '),'mono');
  text(921,463,`word = ${hex(d.standardWords[0])}`,'mono');
  text(921,497,'uint4b8：w = 0.5 × (u − 8)','cap');
  text(921,529,`x首尾为1 → y = ${d.y}`,'label');
  text(32,594,'标准 input-packed qweight [1,8] int32：沿 K 把每列八格压成一个 word','heading');
  for(let n=0;n<8;n++) {
    const x=32+n*157;
    text(x+76,632,`n${n}`,'cap','middle');
    rect(x,644,153,48,n===0?'acc1':'neutral');
    text(x+76,675,hex(d.standardWords[n]),'smallmono','middle');
  }
  text(32,729,'AWQ qzeros 另行恢复位序并转置：[G,N/8] → [N/8,G]；不沿 qweight 的 K 轴照搬。','cap');
  text(32,758,'这里只生成标准格式；后续 Marlin / Exllama 等仍各自 repack，不能把本图当作 Kernel tile。','cap');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="790" viewBox="0 0 1320 790" role="img" aria-label="AWQ 与标准输入打包的二维编码布局"><defs><marker id="main" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#2563eb"/></marker><marker id="aux" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#94a3b8"/></marker></defs><style>
  text{font-family:Inter,"Noto Sans CJK SC","PingFang SC",sans-serif;fill:#172033}.title{font-size:25px;font-weight:650}.heading{font-size:20px;font-weight:600}.label{font-size:21px}.cap{font-size:17px;fill:#475569}.mono{font-family:monospace;font-size:23px}.smallmono{font-family:monospace;font-size:18px}.neutral{fill:white;stroke:#cbd5e1}.ghost{fill:#f8fafc;stroke:#cbd5e1}.acc1{fill:#eaf2ff;stroke:#2563eb}.acc2{fill:#fff4e8;stroke:#d97706}.arrow{fill:none}.main{stroke:#2563eb;stroke-width:2.5}.aux{stroke:#94a3b8;stroke-width:1.5}
  </style><rect width="1320" height="790" fill="white"/>${items.join('\n')}</svg>`;
}
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) process.stdout.write(render());
