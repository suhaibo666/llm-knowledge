import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {model,renderers,root,assetDir,lifecycleContracts,quantizationExample,quantizationExampleTable,fp8NormalExample,fp8NormalTable,denseLayerModel} from '../megatron_precision_graph_figures.mjs';
const pagePath=resolve(root,'wiki/02_engineering/02_train_frameworks/megatron-lm/23_megatron_precision_cudagraph_fusion_analysis.md');
function checkPage(md,d){
 assert.match(md,new RegExp(`一次 GEMM[^\\n]*\\*\\*${d.flops.toLocaleString('en-US')} FLOPs\\*\\*`));
 assert.match(md,new RegExp(`BF16 载荷是 \\*\\*${d.bf16} KiB\\*\\*，名义 FP8 载荷是 \\*\\*${d.fp8} KiB\\*\\*，打包 FP4 载荷是 \\*\\*${d.fp4} KiB\\*\\*`));
 assert.match(md,new RegExp(`一次写回加一次读出是 \\*\\*${d.roundtrip} KiB\\*\\*`));
 for(const [i,suffix] of ['次设备操作提交','次','次图提交'].entries())assert.ok(md.includes(`**${d.submissions[i].length} ${suffix}**`));
 assert.ok(md.includes('`'+d.order.join(' ')+'`'));
 assert.ok(md.includes('`'+d.values.join(',')+'`'));
 assert.ok(md.includes(`峰值为 ${d.peak}，四个微批可复用 ${d.peak} 个槽位`));
 const c=d.costs;
 for(const s of [`**${c.baselineKiB} KiB 降到 ${c.lowKiB} KiB**`,`**${c.intensity.toFixed(2)} FLOP/byte**`,`**${c.payloadRatio.toFixed(2)} 倍**`,`**${c.unfusedKiB} KiB 降到 ${c.fusedKiB} KiB**`,`**${c.f*100}%**`,`**${c.s} 倍**`,`**${c.c*100}%**`,`**${c.speedup.toFixed(2)} 倍**`])assert.ok(md.includes(s),s);
}
test('真实 Markdown 的载荷、FLOPs、提交次数和槽位全部由模型约束',async()=>checkPage(await readFile(pagePath,'utf8'),model()));
test('正文单边修改数值会失败',async()=>{const md=await readFile(pagePath,'utf8');assert.throws(()=>checkPage(md.replace('一次写回加一次读出是 **64 KiB**','一次写回加一次读出是 **65 KiB**'),model()));});

function checkQuantizationPage(md){
 assert.ok(md.replaceAll('\r\n','\n').includes(quantizationExampleTable()),'真实 Markdown 量化表必须与可执行复演一致');
 assert.ok(md.includes('教学整数码本，不模拟 FP8/FP4 或 TE 内核'));
 assert.ok(md.includes('没有定义训练时对量化器求导的算法'));
}
test('同一输入的五条量化路径逐项重建 GEMM，真实 Markdown 复用全部结果',async()=>{
 const d=quantizationExample();assert.equal(d.exact,8.75);
 assert.deepEqual(d.rows.map(r=>r.q),[[0,0,2,3],[0,1,3,3],[2,3,2,3],[1,2,2,3],[1,1,1,1]]);
 const expected=[10,5,59/6,9.75,12];
 d.rows.forEach((r,i)=>{
  const reconstructedDot=r.reconstructed.reduce((s,v,j)=>s+v*d.w[j],0);
  assert.ok(Math.abs(r.output-reconstructedDot)<1e-12);
  assert.ok(Math.abs(r.output-expected[i])<1e-12);
  assert.ok(r.q.every(v=>Number.isInteger(v)&&Math.abs(v)<=r.maxCode));
  assert.deepEqual(r.qw.map(v=>v*r.weightScale),d.w);
 });
 assert.deepEqual(d.rows[2].blockDots,[-3,15]);
 assert.equal(d.rows[1].stats[0],3);assert.equal(d.amax,6);
 checkQuantizationPage(await readFile(pagePath,'utf8'));
});
test('量化正文单边改误差或省略依赖边界会失败',async()=>{
 const md=await readFile(pagePath,'utf8');
 assert.throws(()=>checkQuantizationPage(md.replace('| 9.833 | 1.083 |','| 9.833 | 0.083 |')));
 assert.throws(()=>checkQuantizationPage(md.replace('教学整数码本，不模拟 FP8/FP4 或 TE 内核','实际 FP8 内核')));
});
test('真实 FP8 正规数位串、动态范围和局部间隔经 CPU 枚举核对正文',async()=>{
 const d=fp8NormalExample(),md=(await readFile(pagePath,'utf8')).replaceAll('\r\n','\n');
 assert.deepEqual(d.formats.map(f=>[f.bias,f.spacing,f.max,f.output]),[[7,0.5,448,8.75],[15,1,57344,8.75]]);
 assert.deepEqual(d.formats.map(f=>f.encoded.at(-1).bits),['0 1001 100','0 10001 10']);
 const check=s=>assert.ok(s.includes(fp8NormalTable()),'真实 Markdown FP8 位串或重建值漂移');
 check(md);assert.throws(()=>check(md.replace('0 1001 100','0 1001 101')));
 assert.ok(md.includes('间隔 **0.5**'));assert.ok(md.includes('间隔 **1**'));assert.ok(md.includes('两种格式都得到 **8.75**'));
});
test('完整 MHA dense 层六项乘法及前后向总账读取真实 Markdown',async()=>{
 const d=denseLayerModel(),md=await readFile(pagePath,'utf8');
 assert.equal(d.terms[2],model().flops);
 assert.equal(d.forward,8*d.tokens*d.hidden**2+4*d.tokens*d.hidden*d.ffn+4*d.batch*d.sequence**2*d.hidden);
 assert.deepEqual([d.forward,d.backward,d.total],[33554432,67108864,100663296]);
 for(const v of [...new Set(d.terms),d.forward,d.backward,d.total])assert.ok(md.includes(v.toLocaleString('en-US')),v);
 assert.ok(md.includes('不按因果三角跳过上半区'));assert.ok(md.includes('FLOPs 比例不等于时间比例'));
});
test('机制总账先于源码与特殊路径，三条调用链及上下文组织保留',async()=>{
 const md=await readFile(pagePath,'utf8');
 const headings=['### 2.2 低精度','### 2.3 算子融合','### 2.4 CUDA Graph','### 2.5 组合成完整一层','### 3.1 配置决定构造','### 3.2 精度','### 3.3 融合','### 3.4 Graph','### 4.4 MoE','### 4.5 推理','## 5.','## 6.'];
 const offsets=headings.map(h=>md.indexOf(h));assert.ok(offsets.every((v,i)=>v>=0&&(!i||v>offsets[i-1])));
 for(const s of ['classDiagram','全局层号','is_init=True','delayed FP8 上下文放在整个层遍历之外','FP4 也总是逐层建立 inner context','GeLUFunction.backward','DistributedOptimizer._copy_main_params_to_model_params','GraphableMegatronModule.__call__'])assert.ok(md.includes(s),s);
});
test('反向前不得复用固定槽位，最终存活集合清空',()=>{const d=model();for(const a of d.intervals)for(const b of d.intervals)if(a.id<b.id&&a.slot===b.slot)assert.ok(a.end<=b.start);assert.equal(d.values.at(-1),0);assert.equal(Math.max(...d.values),d.peak);});
test('工作区 SVG 与生成器一致且数字来自模型',async()=>{for(const [name,render]of Object.entries(renderers)){const actual=await readFile(resolve(assetDir,name+'.svg'),'utf8');assert.equal(actual.replace(/\r\n/g,'\n'),render());assert.ok(actual.includes('<title>'));assert.ok(!actual.includes('undefined'));}});
test('页面确实引用全部图，原有 33 个配置字段未丢失',async()=>{const md=await readFile(pagePath,'utf8');for(const name of Object.keys(renderers))assert.ok(md.includes(`](assets/${name}.svg)`));const fields=['fp16','bf16','params_dtype','moe_grad_scale_func','enable_autocast','autocast_dtype','apply_query_key_layer_scaling','attention_softmax_in_fp32','disable_bf16_reduced_precision_matmul','fp8_param','fp8_margin','fp8_interval','fp8_amax_history_len','fp8_amax_compute_algo','fp8_wgrad','fp8_dot_product_attention','fp8_multi_head_attention','tp_only_amax_red','num_layers_at_start_in_bf16','num_layers_at_end_in_bf16','use_kitchen','use_kitchen_attention','kitchen_attention_backend','fp4_recipe','fp4_param','fp4_quantizer_factory','enable_cuda_graph','cuda_graph_use_single_mempool','cuda_graph_retain_backward_graph','cuda_graph_warmup_steps','external_cuda_graph','cuda_graph_dynamic_microbatches','quant_recipe'];assert.equal(fields.length,33);for(const field of fields)assert.ok(md.includes('| `'+field+'` |'),field);});
test('覆盖清单指派的所有正文 owner 字段均在真实页面，包含 auto',async()=>{const md=await readFile(pagePath,'utf8');const yaml=await readFile(resolve(root,'docs/coverage/megatron-lm.yaml'),'utf8');let count=0;for(const block of yaml.split(/\r?\n- name: /).slice(1)){if(/^  owner: 23_megatron_precision_cudagraph_fusion_analysis\s*$/m.test(block)){const name=block.split(/\r?\n/)[0].trim();assert.ok(md.includes('`'+name+'`')||md.includes('.'+name+'`'),name);count++;}}assert.ok(count>=33,`manifest parse found only ${count} fields`);});

test('三条图状态通路与真实正文一致，并保留GeLU近似边界',async()=>{const md=await readFile(pagePath,'utf8');for(const contract of Object.values(lifecycleContracts))assert.ok(md.includes(contract),contract);assert.ok(md.includes('同例')||md.includes('本例 `X[128,128]`'));assert.ok(md.includes('对齐同一种 GeLU 近似'));assert.ok(md.includes('其默认公式使用 erf'));assert.ok(md.includes('固定采用 tanh 近似'));assert.ok(md.includes('当前不存在独立 Rampup calculator'));});
