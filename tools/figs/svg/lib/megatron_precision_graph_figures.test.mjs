import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {model,renderers,root,assetDir,lifecycleContracts} from '../megatron_precision_graph_figures.mjs';
const pagePath=resolve(root,'wiki/02_engineering/02_train_frameworks/megatron-lm/23_megatron_precision_cudagraph_fusion_analysis.md');
function checkPage(md,d){
 assert.match(md,new RegExp(`一次 GEMM[^\\n]*\\*\\*${d.flops.toLocaleString('en-US')} FLOPs\\*\\*`));
 assert.match(md,new RegExp(`BF16 载荷是 \\*\\*${d.bf16} KiB\\*\\*，名义 FP8 载荷是 \\*\\*${d.fp8} KiB\\*\\*，打包 FP4 载荷是 \\*\\*${d.fp4} KiB\\*\\*`));
 assert.match(md,new RegExp(`一次写回加一次读出是 \\*\\*${d.roundtrip} KiB\\*\\*`));
 for(const [i,suffix] of ['次设备操作提交','次','次图提交'].entries())assert.ok(md.includes(`**${d.submissions[i].length} ${suffix}**`));
 assert.ok(md.includes('`'+d.order.join(' ')+'`'));
 assert.ok(md.includes('`'+d.values.join(',')+'`'));
 assert.ok(md.includes(`峰值为 ${d.peak}，四个微批可复用 ${d.peak} 个槽位`));
}
test('真实 Markdown 的载荷、FLOPs、提交次数和槽位全部由模型约束',async()=>checkPage(await readFile(pagePath,'utf8'),model()));
test('正文单边修改数值会失败',async()=>{const md=await readFile(pagePath,'utf8');assert.throws(()=>checkPage(md.replace('一次写回加一次读出是 **64 KiB**','一次写回加一次读出是 **65 KiB**'),model()));});
test('反向前不得复用固定槽位，最终存活集合清空',()=>{const d=model();for(const a of d.intervals)for(const b of d.intervals)if(a.id<b.id&&a.slot===b.slot)assert.ok(a.end<=b.start);assert.equal(d.values.at(-1),0);assert.equal(Math.max(...d.values),d.peak);});
test('工作区 SVG 与生成器一致且数字来自模型',async()=>{for(const [name,render]of Object.entries(renderers)){const actual=await readFile(resolve(assetDir,name+'.svg'),'utf8');assert.equal(actual.replace(/\r\n/g,'\n'),render());assert.ok(actual.includes('<title>'));assert.ok(!actual.includes('undefined'));}});
test('页面确实引用全部图，原有 33 个配置字段未丢失',async()=>{const md=await readFile(pagePath,'utf8');for(const name of Object.keys(renderers))assert.ok(md.includes(`](assets/${name}.svg)`));const fields=['fp16','bf16','params_dtype','moe_grad_scale_func','enable_autocast','autocast_dtype','apply_query_key_layer_scaling','attention_softmax_in_fp32','disable_bf16_reduced_precision_matmul','fp8_param','fp8_margin','fp8_interval','fp8_amax_history_len','fp8_amax_compute_algo','fp8_wgrad','fp8_dot_product_attention','fp8_multi_head_attention','tp_only_amax_red','num_layers_at_start_in_bf16','num_layers_at_end_in_bf16','use_kitchen','use_kitchen_attention','kitchen_attention_backend','fp4_recipe','fp4_param','fp4_quantizer_factory','enable_cuda_graph','cuda_graph_use_single_mempool','cuda_graph_retain_backward_graph','cuda_graph_warmup_steps','external_cuda_graph','cuda_graph_dynamic_microbatches','quant_recipe'];assert.equal(fields.length,33);for(const field of fields)assert.ok(md.includes('| `'+field+'` |'),field);});
test('覆盖清单指派的所有正文 owner 字段均在真实页面，包含 auto',async()=>{const md=await readFile(pagePath,'utf8');const yaml=await readFile(resolve(root,'docs/coverage/megatron-lm.yaml'),'utf8');let count=0;for(const block of yaml.split(/\r?\n- name: /).slice(1)){if(/^  owner: 23_megatron_precision_cudagraph_fusion_analysis\s*$/m.test(block)){const name=block.split(/\r?\n/)[0].trim();assert.ok(md.includes('`'+name+'`')||md.includes('.'+name+'`'),name);count++;}}assert.ok(count>=33,`manifest parse found only ${count} fields`);});

test('三条图状态通路与真实正文一致，并保留GeLU近似边界',async()=>{const md=await readFile(pagePath,'utf8');for(const contract of Object.values(lifecycleContracts))assert.ok(md.includes(contract),contract);assert.ok(md.includes('同例')||md.includes('本例 `X[128,128]`'));assert.ok(md.includes('对齐同一种 GeLU 近似'));assert.ok(md.includes('其默认公式使用 erf'));assert.ok(md.includes('固定采用 tanh 近似'));assert.ok(md.includes('当前不存在独立 Rampup calculator'));});
