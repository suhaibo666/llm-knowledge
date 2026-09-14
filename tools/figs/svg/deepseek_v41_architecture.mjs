// Conceptual architecture, redrawn from DeepSeek-V4.1-Flash report Fig.3 and
// config.json @ fb2764a5cf321eaa5070ca8f9e892818f477c16d (2026-09-10).
// Layout contract: input/conditional memory -> exact 40-layer stack -> block zoom.
// Main arrows carry hidden states; blue carries global KV; dashed is auxiliary.
// A-F refer to the page's version-delta table. No performance simulation.
// Run: node tools/figs/svg/deepseek_v41_architecture.mjs
import {mkdirSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const out=fileURLToPath(new URL('../../../wiki/01_theory/01_models/deepseek/assets/deepseek_v41_architecture.svg', import.meta.url));
const parts=[];
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function text(x,y,s,cls='label',anchor='start'){parts.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);}
function box(x,y,w,h,lines,cls='neutral'){parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" class="${cls}"/>`);lines.forEach((s,i)=>text(x+w/2,y+(h-lines.length*23)/2+18+i*23,s,i===0?'label':'small','middle'));}
function path(d,cls='main'){parts.push(`<path d="${d}" class="arrow ${cls}" marker-end="url(#${cls==='kv'?'blue':'tip'})"/>`);}
parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="930" viewBox="0 0 1240 930" role="img" aria-labelledby="title desc"><title id="title">DeepSeek-V4.1-Flash 完整模型骨架与版本改动</title><desc id="desc">图文输入进入20层因果编码器与20层解码器。全局KV由编码器输出投影，单层保留独立query、SWA与MoE。Engram在第1和14层注入，DSpark为独立草稿分支。</desc><style>text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#2A313B}.title{font-size:25px;font-weight:700}.section{font-size:19px;font-weight:700}.label{font-size:17px;font-weight:600}.small{font-size:15px}.cap{font-size:14px;fill:#596575}.neutral{fill:#fff;stroke:#C7CCD3;stroke-width:1.5}.ghost{fill:#F6F7F9;stroke:#DDE1E6;stroke-width:1.3}.acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}.acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}.arrow{fill:none;stroke:#667281;stroke-width:2}.main{stroke-width:2.5}.kv{stroke:#2563EB;stroke-width:2.5}.aux{stroke-dasharray:5 4;stroke-width:1.5}</style><defs><marker id="tip" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#667281"/></marker><marker id="blue" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#2563EB"/></marker></defs><rect width="1240" height="930" fill="white"/>`);
text(28,38,'DeepSeek-V4.1-Flash｜先看完整骨架，再看训推如何执行','title');
text(28,65,'依据报告 Figure 3 重绘；层号从 0 起。A–F 对应正文的前代差异表。','cap');
text(28,106,'输入与条件记忆','section');text(420,106,'B  40 层因果主干 · hidden 5120','section');text(866,106,'单个全局注意力层的内部','section');
box(28,127,300,52,['文本 → Token Embedding']);
box(28,209,300,87,['A  图像 → DeepSeek-ViT','32 层 · hidden 1024','patch 14 · 2D-RoPE']);
box(28,323,300,75,['3×3 pixel-unshuffle → MLP','视觉位置数 ÷9 → hidden 5120']);
box(28,427,300,57,['图文按位置合并','联合输入 N×5120']);
path('M178 296 V323');path('M178 398 V427');path('M28 153 H15 V455 H28');
box(28,559,300,101,['D  Engram 条件记忆','文本 n-gram → 哈希查表','两模块共约 196B · FP8']);
text(340,483,'注入','cap');text(340,505,'L1/L14','cap');
box(402,127,385,288,[],'ghost');text(420,155,'Causal Encoder · L0–19','section');
box(420,178,349,58,['L0–1：SWA + MoE ×2']);
box(420,260,349,106,['L2–19：三组，每组 6 层','Full + 5×Reuse；CSA2 比率 2','Full 所在层：L2 / L8 / L14'],'acc1');
text(420,392,'所有层都保留局部 SWA，窗口 128','small');path('M595 236 V260');
path('M328 455 H384 V206 H420');
box(420,442,349,59,['最终 Encoder Hidden States','蓝线：投影 decoder 全局 KV']);path('M595 415 V442');
box(402,529,385,266,[],'ghost');text(420,558,'Decoder · L20–39','section');
box(420,580,349,62,['L20–23：Full + 3×Reuse','CSA2 比率 1；全局 KV 来源见蓝线']);
box(420,670,349,91,['L24–39：四组，每组 4 层','Reindex + 3×Reuse；比率 1','Reindex：L24 / L28 / L32 / L36'],'acc1');
path('M595 501 V580');path('M595 642 V670');
path('M769 472 H818 V612 H769','kv');
box(420,822,349,58,['归一化 / 输出头 → 文本 token']);path('M595 795 V822');
box(866,128,346,68,['本层 hidden / 4 路残差','C  Single-Pass mHC 输入混合']);
box(866,226,346,63,['本层 Q + 本层 SWA KV','局部窗口 W=128']);
box(866,319,346,77,['所选 global KV + 局部 KV','→ Attention → 残差更新','E  global FP4；SWA FP8']);
box(866,426,346,83,['C  mHC 输入混合 → MoE','384 路由专家选 6 + 1 共享专家','专家中间维 2304']);
box(866,539,346,55,['残差更新 → 下一层']);
path('M1039 196 V226');path('M1039 289 V319');path('M1039 396 V426');path('M1039 509 V539');
box(866,627,346,99,['B  decoder 检索分工','首个 Full：建 KV + 索引 + 候选池','Reindex：池内重选；Reuse：沿用']);
path('M1039 627 V611 H846 V355 H866','aux');
box(866,767,346,107,['F  DSpark 草稿分支','3 层 → 5 个草稿位置 → 目标验证','前代正式权重已含 DSpark','本次明确训练与调度协同']);path('M787 737 H824 V820 H866','aux');
path('M328 609 H366 V325 H420','aux');path('M366 325 V220 H420','aux');
text(28,711,'读图提示','section');
text(28,743,'主干箭头展示完整因果计算关系。','small');
text(28,772,'长 prompt 的 prefill 可跳过大部分','small');
text(28,797,'decoder 计算，但须重建尾部 SWA。','small');
text(28,826,'生成新 token 时仍经过全部 40 层。','small');
text(28,907,'来源：HF deepseek-ai/DeepSeek-V4.1-Flash @ fb2764a5cf32 · Report §2 / Fig.3 / §4.2.1 + config.json；不是生产调度实测图。','cap');
parts.push('</svg>');mkdirSync(fileURLToPath(new URL('.', 'file://'+out)),{recursive:true});writeFileSync(out,parts.join('\n'));console.log(out);
