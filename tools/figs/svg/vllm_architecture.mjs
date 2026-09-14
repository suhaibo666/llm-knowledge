// Static responsibility map for vLLM 02, baseline 199cb9b964822.
// Geometry expresses dependencies, not process placement or request timing.
// Main column: semantics -> engine -> scheduling/execution -> runner -> model.
// A side panel names optional integration points; the bottom names external APIs.
import {writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const parts=[];
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,cls='detail')=>parts.push(`<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`);
const box=(x,y,w,h,title,lines,cls='neutral')=>{
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" class="${cls}"/>`);
  text(x+18,y+28,title,'label');
  lines.forEach((s,i)=>text(x+18,y+53+i*23,s));
};
const arrow=(d,cls='main')=>parts.push(`<path d="${d}" class="arrow ${cls}" marker-end="url(#tip)"/>`);

text(28,34,'vLLM：六个职责模块与依赖边界','title');
text(28,59,'实线表示主依赖；虚线表示侧接或外部 API。框的位置不表示进程数量。');
box(28,82,732,74,'接口与语义',['协议路由 / Renderer / InputProcessor / OutputProcessor']);
box(28,186,732,74,'Engine 运行',['同步与异步 facade / CoreClient / EngineCore 循环与生命周期']);
box(28,292,348,104,'资源调度',['Scheduler / KVCacheManager / 依赖管理','拥有请求进度、预算与逻辑 block'],'acc1');
box(412,292,348,104,'执行组织',['Executor / Worker / rank 协作','拥有派发、通信次序与结果汇集']);
box(28,432,732,74,'设备运行',['Model Runner / 活跃请求状态 / 当步 tensor、metadata 与图执行'],'acc1');
box(28,538,732,74,'模型与算子',['Registry / Loader / Attention / 量化、LoRA、编译接合与 kernel']);
box(28,660,1024,83,'外部底座：框架、运行时与硬件',['PyTorch tensor / stream / graph / 分布式 API；设备运行时、通信库、设备与互联'],'ghost');

arrow('M 394 156 V 186');
arrow('M 394 260 V 276 H 202 V 292');
arrow('M 394 276 H 586 V 292');
arrow('M 586 396 V 432');
arrow('M 394 506 V 538');
arrow('M 394 612 V 660','aux');
text(410,643,'tensor / kernel / 后端 API');

box(804,82,248,314,'侧接能力与接合位置',[
  '观测与故障传播',
  '  Engine / 前端 / Executor',
  'KV 与 encoder 传输',
  '  Scheduler / Runner',
  '在线权重与版本更新',
  '  Engine / Worker / 模型',
  '平台、I/O、endpoint 插件',
  '  各自的构造与生命周期',
],'ghost');
arrow('M 804 223 H 760','aux');
arrow('M 804 360 H 784 V 469 H 760','aux');
box(804,432,248,180,'读图约定',[
  'Scheduler 与 Executor',
  '是 Engine 的两个协作分支。',
  '计划与结果经 Engine 配对；',
  '外部底座也为 Executor',
  '提供通信，为 Runner',
  '提供设备 API。',
],'ghost');
text(28,780,'基线：vllm-project/vllm@199cb9b964822 · 具体合同和源码锚点见正文 §2–4');

const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="802" viewBox="0 0 1080 802" role="img" aria-label="vLLM 六个职责模块、侧接能力与外部依赖">
<style>
text{font-family:Arial,"PingFang SC","Microsoft YaHei",sans-serif;fill:#0f172a}
.title{font-size:23px;font-weight:700}.label{font-size:18px;font-weight:700}.detail{font-size:15px;fill:#475569}
.neutral{fill:#fff;stroke:#64748b;stroke-width:1.3}.ghost{fill:#f8fafc;stroke:#94a3b8;stroke-width:1.1}
.acc1{fill:#dbeafe;stroke:#2563eb;stroke-width:1.8}.arrow{fill:none;stroke:#64748b;stroke-width:1.5}
.main{stroke:#2563eb;stroke-width:2}.aux{stroke-dasharray:5 4;stroke:#94a3b8}
</style>
<defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#64748b"/></marker></defs>
<rect width="1080" height="802" fill="#fff"/>
${parts.join('\n')}
</svg>\n`;
const target=fileURLToPath(new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_architecture.svg',import.meta.url));
writeFileSync(target,svg);
console.log(target);
