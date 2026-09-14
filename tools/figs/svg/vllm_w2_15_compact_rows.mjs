// 教学数据；对应 vLLM 199cb9b9 的 InputBatch.condense/swap_states。
// 不运行 vLLM。用整条记录模拟伴随字段同行，并计算正文的 token-major 索引。
import { pathToFileURL } from 'node:url';
export function replay() {
  const A = { id:'A', tokens:Array.from({length:20},(_,i)=>100+i), prompt:20, computed:18, temp:0, lora:0, blocks:[12,13], scheduled:2 };
  const B = { id:'B', tokens:Array.from({length:6},(_,i)=>200+i), prompt:5, computed:5, temp:0.6, lora:7, blocks:[28], scheduled:1 };
  const holes=[A,null,B];
  const compact=holes.slice();
  while (compact.includes(null)) {
    const hole=compact.indexOf(null);
    while (compact.at(-1)===null) compact.pop();
    if (hole>=compact.length) break;
    compact[hole]=compact.pop();
  }
  // 重放 reorder_batch_to_split_decodes_and_prefills（阈值 1）：四区 decode→short→long→prefill，
  // 误置 row 经 src_dest_map 转成 swap_states 调用链；本例 A 为 long_extend、B 为 decode，得到一次 swap_states(1,0)。
  const threshold=1;
  const region=r=>r.computed===0?3:r.scheduled>threshold?2:r.computed<r.prompt?1:0;
  const ordered=compact.slice(), req=ordered.map(region);
  const target=req.slice().sort((a,b)=>a-b);
  const orig=req.flatMap((g,i)=>g!==target[i]?[i]:[]);
  const src=orig.slice().sort((i,j)=>req[i]-req[j]);
  const dest=new Map(src.map((s,k)=>[s,orig[k]]));
  const swaps=[];
  for (const s of src) {
    let d=dest.get(s);
    while (s!==d) {
      swaps.push([s,d]);
      [ordered[s],ordered[d]]=[ordered[d],ordered[s]];
      const next=dest.get(d)??d;
      dest.set(d,d);
      d=next;
    }
  }
  const reqIndices=[], positions=[], tokenIndices=[], ids=[], qsl=[0];
  const stride=32;
  ordered.forEach((r,row)=>{
    for(let offset=0;offset<r.scheduled;offset++) {
      const pos=r.computed+offset;
      reqIndices.push(row);positions.push(pos);tokenIndices.push(row*stride+pos);ids.push(r.tokens[pos]);
    }
    qsl.push(ids.length);
  });
  return {holes,compact,ordered,swaps,stride,reqIndices,positions,tokenIndices,ids,qsl,
    seqLens:ordered.map(r=>r.computed+r.scheduled),
    loraTokens:ordered.flatMap(r=>Array(r.scheduled).fill(r.lora))};
}
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
export function draw() {
  const r=replay(),width=1020,height=790;
  const out=[`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><title>MRV1：空洞压紧与全部请求字段同行</title><desc>删除 X 后 B 从 row 2 移到 row 1，再与 A 交换，最终 B row 0、A row 1。图中块号是表内容，不表示复制 KV 缓存。</desc><style>text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#2A313B;font-size:20px}.neutral{fill:#fff;stroke:#C7CCD3}.ghost{fill:#F7F6F3;stroke:#DDD9D2}.acc1{fill:#EAF1FD;stroke:#2563EB}.acc2{fill:#FCF1E6;stroke:#C3651F}.arrow.main{stroke:#2563EB;stroke-width:2.5;fill:none}.cap{font-size:19px;fill:#6B7280}.heading{font-size:24px;font-weight:600}</style><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#2563EB"/></marker></defs><rect width="1020" height="790" fill="white"/>`];
  const text=(x,y,s,cls='')=>out.push(`<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`);
  text(30,36,'紧凑 row 是整组状态的位置；只改请求 id 会错配','heading');
  const columns=[['row',65],['请求',85],['token 有效前缀',220],['computed',130],['温度',90],['LoRA',90],['block table',250]];
  function table(y,title,rows,mark) {
    text(30,y,title,'heading');
    let x=30;
    columns.forEach(([label,w])=>{out.push(`<rect x="${x}" y="${y+13}" width="${w}" height="36" class="ghost"/>`);text(x+10,y+38,label);x+=w;});
    for(let row=0;row<3;row++) {
      const item=rows[row], yy=y+49+row*37;
      const values=item?[row,item.id,`${item.id}0…${item.id}${item.tokens.length-1}`,item.computed,item.temp,item.lora,'['+item.blocks.join(', ')+']']:[row,'—',row<rows.length?'空洞':'非活跃容量','—','—','—','—'];
      let xx=30;
      values.forEach((v,c)=>{const cls=!item?(row<rows.length?'acc2':'ghost'):(row===mark?'acc1':'neutral');out.push(`<rect x="${xx}" y="${yy}" width="${columns[c][1]}" height="37" class="${cls}"/>`);text(xx+10,yy+26,v);xx+=columns[c][1];});
    }
  }
  table(80,'① 原 row [A, X, B]：移除 X，留下内部空洞',r.holes,-1);
  out.push('<path d="M45 249V276" class="arrow main" marker-end="url(#arrow)"/>');
  text(68,270,'condense：尾部 B 从 2 → 1；有效 token 前缀与伴随字段一起移');
  table(310,'② 压紧完成：[A, B]，row 2 不再活跃',r.compact,1);
  out.push('<path d="M45 479V506" class="arrow main" marker-end="url(#arrow)"/>');
  text(68,500,`reorder：B 是 decode，A 是 long_extend；${r.swaps.map(([i1,i2])=>`swap_states(${i1}, ${i2})`).join('、')}`);
  table(540,'③ 执行顺序：[B, A]，全部列继续对应同一个请求',r.ordered,0);
  text(30,740,'图中仅展开部分字段；generator、mask、prompt embeds、processor 状态也须按同一移动更新。','cap');
  text(30,771,'这里移动 block table 的行，不搬运这些块中的 KV 字节。','cap');
  out.push('</svg>');return out.join('\n');
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href) console.log(draw());
