# torch_npu 自定义融合 Pass 逐个深挖 — 场景 · 问题 · 优化 · 效果

> **Source baseline**：torch_npu `E:\97-codes\torch_parallel\torch_npu` @ `b3c8a815b`（tag `v2.7.1`，2026-07-15）。除注明外，`file:line` 均指 `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py`（全文 2548 行）。
> **Dimension**：Deep Dive（mechanism-level，逐函数读源）
> 本页是 [[npu_vs_upstream_fusion_passes]] 的**配套深挖页**。母页给「谁有谁无」的对照地图，本页对 torch_npu **26 个自定义 pass + 3 个后端融合机制**逐个回答：**什么代码场景触发（待优化问题）→ 为什么这么优化 → 优化带来什么效果**，每条带已核验 `file:line`。硬件级 why 的全景在 [[npu_inductor_optimization_analysis]]。

---

## 1. 怎么读这一页（含诚实边界）

每个 pass 用四拍讲清：

- **场景**：什么 FX 子图 / 用户代码会触发它（带最小 before 代码）。
- **问题**：这个子图在 NPU 上为什么费——多一次 kernel launch、多一次 GM 往返拷贝、非连续访存、UB 白占、i64 双倍位宽、或本是 no-op 却生成真 kernel。
- **优化**：改写成什么（after 代码 + 改写手法）。
- **效果**：具体收益。

> [!important] 效果口径（务必先读，避免误读）
> 逐函数 grep 确认：**全文除 CATLASS epilogue 外没有任何 `counters[...]` 埋点、也没有 benchmark**（`catlass_scheduling.py:151` 是唯一计数器）。因此本页所有「效果」都是**结构性的**（少几个 kernel / 少一次拷贝 / 转零拷贝 view / 索引降到 int32），**不是实测加速比**——凡涉及数量级的措辞均为机制推断。
> 另外：① 源文件里**没有任何 NPU 硬件注释**（无「达芬奇/vector core/UB/i64」字样），凡「因为硬件 X」的因果均标 **[硬件推断]**，其依据是 [[npu_inductor_optimization_analysis]] 归纳的硬件特性，而非本 pass 文件；② 部分 pass 带**中文 docstring 直述动机**，这类标 **[docstring 原文]**，可信度最高；③ 26 个 pass **全部只处理静态 shape**（`get_node_shape` 遇 SymInt 维返回 None，`get_binary_fold_result.py:37-38`）；④ 生效门控：POST 全部**仅推理**、PRE 除 `fusion_attention_v3_pass` 外也**仅推理**（`ascend_custom_passes/__init__.py:15-23`）。

**「效果」锚定的 6 个 NPU 硬件事实**（母页 §5 / [[npu_inductor_optimization_analysis]] 展开）：① 逐元素/规约算子会各起一个 vector kernel 并把张量经 UB 在 GM 间搬运；② matmul 由 Cube 单元算、结果落 L0C，接逐元素若不融合就得 Cube→GM→Vector 往返；③ UB 是便笺，中间张量越多压力越大；④ vector core 对 i64 支持弱，int32 索引省一半位宽 [硬件推断]；⑤ 达芬奇偏好连续访存；⑥ view/expand/squeeze 是零拷贝元数据，cat/clone/pad/repeat/slice_scatter 是真拷贝。

---

## 2. PRE passes（4 个）

### 2.1 `cat_slice_cat_fold_pass`（`:52-127`，PRE·仅推理）
**场景**：一个 `cat` 的每个输入都是对**同一个父张量**的 `getitem` 切片，而那个父张量本身又是个 `cat`，且切片在同一维上**无缝拼满** `[0,len)`：
```python
c   = torch.cat([a, b], dim=-1)     # cat1
s0  = c[..., 0:Da]; s1 = c[..., Da:Da+Db]
out = torch.cat([s0, s1], dim=-1)   # cat2 —— 与 c 逐字节相同
```
（匹配条件：同父 `:88-95`、同轴 `:80-83`、无缝拼满 `:105-120`）
**问题**：`cat2` 重新物化了一份与 `cat1` 完全一样的张量，中间的切片+再拼接是纯数据搬运；NPU 上每个 cat/slice 都是 GM 上的 DMA/vector 拷贝，白白耗带宽 [硬件推断]。
**优化**：`cat2.replace_all_uses_with(cat1)`，再删 `cat2` 与各切片 getitem（`:121-125`）。
**效果**：消掉 1 个 cat + N 个 slice 拷贝，下游直接读 `cat1`。

### 2.2 `pad_slice_fold`（`:130-186`，PRE·仅推理）
**场景**：`pad` 沿某维扩张后，下游每个切片窗口都完全落在 **pad 前**的原范围内：
```python
xp = F.pad(x, (0, 8))   # x:(...,L) -> xp:(...,L+8)
y  = xp[..., 0:L]       # 切片没碰到 pad 区
```
（条件 `slice_end <= 原 shape[pad_dim]`，`:169-174`）
**问题**：pad 把一个更大的张量写进 GM，但下游切片把 pad 出来的部分整个丢弃——那次扩张写是纯浪费。
**优化**：把每个切片的输入**改绑到 pad 前的源**（`user.args=(input, idx)`，`:182-183`），删掉 pad（`:184`）。
**效果**：删掉 1 个 pad 及其 GM 写，切片直接读未 pad 的源。

### 2.3 `dtype_optimal_pass`（`:806-882`，PRE·仅推理）
**场景**：值域塞得进 int32 的 int64 `arange`，或源 dtype∈{fp32,int32,bool,int16,int8} 的 `.to(int64)`：
```python
idx = torch.arange(0, seq_len, dtype=torch.int64)   # (a) 静态范围
pos = something.to(torch.int64)                      # (b) 源是 int32
```
（int32 范围守卫 `:855`；`.to` 源 dtype 白名单 `:869-870`）
**问题**：**docstring 原文（`:808-809`）**「将不必要的 int64 优化为 int32……减少访存与计算开销」。i64 在 NPU vector core 上弱且索引张量双倍位宽 [硬件推断]。
**优化**：`arange` 把 `dtype` 改 int32（`:856`）；`.to` 目标改 int32（`:875/:877-880`）。
**效果**：索引/iota 张量变 int32，索引带宽减半、避开 i64 vector 运算 [「减半」为推断，「减少访存/计算」为 docstring 原文]。POST 侧还有个同思路的 `fold_iota_arithmetic_pass`（§4.4b）。

### 2.4 `fusion_attention_v3_pass`（`:885-906`，PRE·**训练+推理**）
**场景**：图里任何 `torch.ops.npu.npu_fusion_attention.default` 节点（`:892-894`）。
**问题**：**docstring 原文（`:887-888`）**「以等价的 v3 算子替换原节点……从而启用更高性能的实现」。为什么 v3 更快不在本文件 [推断为手工 kernel 升级]。
**优化**：新建 `npu_fusion_attention_v3.default` 节点，**args/kwargs 原样透传**（`:897-901`）、`meta` 逐字复制（`:902`），`replace_all_uses_with`+erase（`:903-904`）。
**效果**：所有 FA 调用路由到 v3 kernel。
> **校正**：本 baseline **不存在 `npu_fusion_attention_v2`**（全仓 grep 无引用），本 pass 是把**基础版 `npu_fusion_attention.default` → `v3.default`**，不是「v2→v3」。FX 层看是**同签名的 drop-in 换目标**（args/meta 不变，v3 schema 的 `softmax_layout`/`sink` 走默认）。它是**唯一在训练也跑**的 PRE pass（运行器 `ascend_custom_passes/__init__.py:20-23` 在 `is_inference_check()` 门外单独跑它）。

---

## 3. POST — 结构折叠 / 冗余消除（13 个）

共性：全是 `@register_custom_pass(PassType.POST)`、**仅推理**，收尾统一 `eliminate_dead_code`（lint + DCE，`:2541-2547`）。共同「问题」是——一个语义上的 no-op / 恒等运算仍会生成一个真 kernel 或真拷贝。下表每行给 before→after 与效果性质：

| Pass（file:line） | 场景（before → after） | 待优化问题 | 效果 |
|---|---|---|---|
| `fold_four_op_pass`（`:189`） | `x+0 / x-0 / 0-x / x*1 / x/1` → 保留非平凡操作数 | 恒等算术仍起一个 vector kernel | 消掉该二元 kernel；**但** `get_binary_fold_result` 至少插一个 `clone`（`GBF:96-124`），故**非零成本**、余下 clone 可由 `fold_clone` 再消 |
| `fold_cast`（`:237`） | `convert_element_type(x_fp16, fp16)` → `x_fp16` | 同 dtype cast 仍是整张量 DMA 拷贝 | 真零拷贝：`replace_all_uses_with(input)`，删一次全量拷贝 |
| `fold_cat`（`:265`） | 单用户、同轴的嵌套 `cat(cat(x,y),z)` → 一个 `cat` | 内层 cat 物化一份立即被再拼接的中间 buffer | 少 1 个 concat kernel + 中间 buffer/每折 |
| `fold_clone`（`:322`） | `clone(x)`（mem_format 不变、非输出）→ `x` | clone 是整张量 memcpy | 真拷贝消除 |
| `fold_detach`（`:360`） | `detach(x)` → `x`（无条件） | 推理图里 detach 数值 no-op 却占节点 | 删掉 detach（detach 是共享存储视图，替换安全） |
| `fold_expand`（`:379`） | `expand(x,[4,8])`（x 本就 [4,8]）→ `x` | 不真广播的 expand 是元数据 no-op、堵融合 | 删冗余 view 节点 |
| `fold_reduce`（`:416`） | `sum(x,dim=[1])`（该维 size=1）→ `squeeze` 或原样 | 对 size-1 维求和仍起 reduce kernel、白占 UB | 把 reduce 换成 squeeze 视图（零拷贝）或直通 |
| `fold_slice`（`:551`） | 全范围 `slice` → 输入；全覆盖 `slice_scatter` → 替换张量 | slice_scatter 全覆盖仍复制整张量 | 删 no-op slice / 删一次全量拷贝（日志 `FoldSliceLike: Folded`） |
| `fold_squeeze`（`:571`） | `squeeze(unsqueeze(x,1),1)` → `x`；相邻 squeeze 合并 | 相邻/互逆的 squeeze/unsqueeze 是冗余 view | 抵消视图节点，利于融合（要求 prev 单用户） |
| `fold_to_copy`（`:604`） | `_to_copy(x)`（dtype/device/layout/mem_format 全不变、非输出）→ `x` | 无变化的 `_to_copy` 仍整张量拷贝 | 真拷贝消除 |
| `view_fold_pass`（`:668`） | `view(reshape(x,..),..)` → 折成单个 view；恒等 view 删除 | view/reshape/squeeze 链只加节点、堵融合 | 折叠 view 链、删恒等 view |
| `fold_where`（`:715`） | `where(cond,a,a)`（两支相同/都 0/都 1）→ 一支 | where 读三张量（cond+两支）起 select kernel | 消掉 select kernel；**同样**经 `get_binary_fold_result` 余下 broadcast+clone，故部分收益 |
| `fold_redundant_ops`（`:741`） | `squeeze.dim(view(x))`（形状+dtype 回到原样）→ `x` | view→squeeze 往返是两个冗余 view、堵融合 | 删掉这对往返（至多 2 节点） |

> 小结（agent 核实）：真·零拷贝直删的是 `fold_cast/clone/detach/expand/reduce/slice/squeeze/to_copy/view_fold/redundant_ops`（都直接 `replace_all_uses_with(input)`）；`fold_four_op` 与 `fold_where` 因走 `get_binary_fold_result` 会**留一个可后续再消的 clone**，不是保证零成本。

---

## 4. POST — 结构重排 & 真·融合（9 个，重点）

这 9 个是 NPU 真正「有想法」的 pass——把一段业务子图整体改写成更省的形态。

### 4.1 `fold_sink_view`（`:448`，POST·仅推理）
**场景**：一个单用户的 `view/reshape`，其唯一用户是激活（silu/gelu/relu/sigmoid）或可广播的二元 add/sub/mul/div：
```python
v = x.view(B*S, H)
y = torch.relu(v)      # 激活夹在 view 之后
```
**问题**：**docstring 原文（`:449-450`）**「先在原 shape 上执行计算，再做 view，从而便于后续融合且不影响数值结果」——激活作用在 reshape 后的张量上，无法与 `x` 的生产者融合。
**优化**：反序重建——先 `relu(x)` 再 `view`（`:462-477`），`user.replace_all_uses_with(new_act_view)`（`:481-482`）；二元分支带广播安全校验（`:504-508`）。
**效果**：激活/二元落到 view 前的张量上，可与生产者融合；view 退化为纯元数据、**不新增拷贝**。

### 4.2 `cat_to_view_pass`（`:909`，POST·仅推理）
**场景**：`cat` 的每个输入都是**同一父张量**的连续切片、无缝拼满整维：
```python
out = torch.cat([x[:,0:4], x[:,4:8]], dim=1)   # == x（恒等）
out = torch.cat([x[:,3:8], x[:,0:3]], dim=1)   # == 循环左移 3（rotation）
```
**问题**：**docstring 原文（`:911-912`）**「cat 等价于恒等视图或循环位移，从而避免实际数据搬运」——cat 起一个 concat kernel 物化新 buffer，而结果只是父张量本身或其循环移位 [拷贝成本为硬件推断]。
**优化**：恒等→`replace_all_uses_with(parent)`（`:1011`）；循环→建 `aten.roll(parent, shift, dim)`（`shift=-normalised[0][0]`，`:1050-1060`）。切片+cat 被 DCE。
**效果**：恒等是零拷贝；循环用 1 个 `roll` 替掉 N 切片+cat。日志明写 `collapsed cat(...) → identity view` / `→ roll(shift=%d)`（`:1013,:1062`）。

### 4.3 `repeat_to_expand_pass`（`:1112`，POST·仅推理）
**场景**：只做广播（从 size-1 维复制）的 `repeat`，且所有用户都是广播友好算子（mul/add/where/比较/逻辑…，白名单 `:1075-1109`）：
```python
m = mask.repeat(1, H, 1)   # [B,1,S] -> [B,H,S]
y = x * m                  # mul 天然可广播 stride-0 维
```
**问题**：**docstring 原文（`:1114-1115）**「用零拷贝的 expand 替代物理拷贝的 repeat」——`repeat` 物化整份广播结果，而消费者本可对 size-1 维做免费广播 [UB/GM 物化成本为硬件推断]。
**优化**：建 `aten.expand(inp, out_shape)`（`:1166-1168`），`replace_all_uses_with(exp)`（`:1182`）。
**效果**：`expand` 是 stride-0 视图、零拷贝；`repeat` 被 DCE。日志 `rewrote repeat(...) → expand(...) (broadcast-only, %d consumers)`（`:1184`）。

### 4.4 `fold_iota_arithmetic_pass`（`:1394`，POST·仅推理）—— 三个子变换
**(a) 常量 CSE**：去重 arg 相同的 `iota/arange/full`（`:1402-1408`），并**排除被 mutation 引用的 buffer**（`_collect_mutation_buffer_ids`，`:1326-1363`）以免两个原地写共用一块。
**(b) int64→int32 iota 降位**：`prims.iota(int64)` 且值域进 int32、且到「闭合算子」（比较/`_to_copy`/`convert_element_type`）前只经**dtype 透明算子**（view/permute/add/mul…）时，改 int32（`:1431-1433`），refresh 后若没得到 int32 则**回滚**（`:1435-1439`）。
**(c) `cmp(sub(a,b),0)` 折叠**：`sub(a,b) {ge/gt/le/lt/eq/ne} 0` → `cmp(a,b)`（`:1497-1510`）。
**问题**：(a) 重复常量各起一个 fill kernel；(b) i64 索引双倍位宽 [硬件推断，`opt_int64_to_int32` 兄弟 docstring `:808-809` 述「减少访存与计算」]；(c) `sub` 是多起的一个逐元素 kernel + 物化 diff。
**效果**：去重常量 / int32 窄 iota / 省掉 `sub`。日志 `downcast iota[...] int64→int32` `folded ge(sub(a,b),0) → ge(a,b)`（`:1453,:1512`）。

### 4.5 `broadcast_const_mask_compress`（`:1536`，POST·仅推理）
**场景**：把 bool mask 表达成 0/1 数值的常见写法：
```python
sel = torch.where(bool_mask, full(1), full(0))
m   = sel.to(torch.float32)
```
（两支都是常量 `full`、真值不同、非 bool 目标时限定为 {0,1}，`:1567-1579`）
**问题**：**docstring 原文（`:1538-1539）**「用 mask 自身（或 logical_not(mask)）替代该 where+cast，消除显式广播」——物化了两个 `full` 常量张量 + 一次 broadcast select + 一次 cast [UB 物化为硬件推断]。
**优化**：真值为「1」在真支则 `new_cond=cond`，否则插 `logical_not(cond)`（`:1590-1594`）；bool 目标直接 `replace_all_uses_with(new_cond)` 连 cast 都省（`:1605-1607`），否则把 cast 改读 bool mask、让原生 bool→dtype cast 完成 0/1（`:1608-1610`）。
**效果**：消掉 `where` + 两个 `full`（DCE），最好连 cast 也省。日志 `:1614`。

### 4.6 `masked_add_compose_pass`（`:1691`，POST·仅推理）
**场景**：注意力/分支合并里「互补掩码相加」：
```python
out = torch.where(mask, a, 0) + torch.where(~mask, b, 0)
```
（两 `where` 各单用户、两 mask 互为逻辑取反，`:1706-1715`）
**问题**：**docstring 原文（`:1693-1694）**「两个互补的掩码加法等价于一次三目选择，可节省一次加法与一次 where」。
**优化**：合成 `where(mask, a, b)`（`:1729-1733`），`add.replace_all_uses_with(new_where)`（`:1749`）。
**效果**：`2×where + 1×add → 1×where`。日志 `folded where(m,a,0)+where(~m,b,0) → where(m,a,b)`（`:1751`）。

### 4.7 `bool_cast_mul_to_where_pass`（`:1835`，POST·仅推理）
**场景**：施加 bool 掩码的常见写法（mf 与 mul 间可夹 reshape/unsqueeze）：
```python
mf = bool_mask.to(x.dtype)
y  = mf * x
```
（经单用户 view 链回溯到 cast、cast 源为 bool、cast 目标 dtype==另一操作数 dtype，`:1863-1878`）
**问题**：**docstring 原文（`:1837-1838）**「避免显式的 bool→numeric 类型转换与广播乘法，让后端有更多融合机会」——一次整张量 cast kernel + 一次广播 mul，且乘 0/1 本可用 select 代替。
**优化**：建标量零 `full([],0)`，把 view 链**重放到 bool 源**上使其到达 mul 形状（`_replay_view_chain`，`:1801`），合成 `where(new_cond, other, zero)`（`:1912-1915`），`mul.replace_all_uses_with(new_where)`（`:1931`）。
**效果**：`cast + mul → where`，省整张量数值 cast、一次 select 取代 cast+乘。日志 `:1934`。

### 4.8 `sign_diff_hamming_fuse_pass`（`:1969`，POST·仅推理）
**场景**：符号位汉明距离（二值哈希 / sign-LSH 相似度；`relu(sign(·))` 把值映射到 {0 if ≤0, 1 if >0}）：
```python
d = torch.sum(torch.abs(torch.relu(torch.sign(x)) - torch.relu(torch.sign(y))), dim)
```
（`sum←abs←sub←两个 relu(sign)`，各单用户，`:1977-2004`）
**问题**：**docstring 原文（`:1971-1972）**「用 sum(ne(gt(x,0),gt(y,0))) 等价替换，简化算子链并降低中间张量成本」——原链 6 个算子、两个 `relu(sign)` **float** 中间张量 + diff [UB 压力为硬件推断]。
**优化**：建 `gt(x,0)`、`gt(y,0)`、`ne(·,·)`、`sum(ne, dim)`（保留原 dim/keepdim/out_dtype，`:2031-2039`），`sum.replace_all_uses_with(new_sum)`（`:2047`）。
**效果**：6 算子 float 链 → `gt/gt/ne/sum`，中间张量变 **bool**，reduce 直接数不同符号位。日志 `:2049`。

### 4.9 `batch_embedding_fusion_pass`（`:2493`，POST·仅推理）
**场景**：把 EmbeddingBag 式池化写成对同一索引张量分段的 N 次 embed+reduce（同权重、默认参数、各段等长且 `N*L==dim`）：
```python
for i in range(N):
    seg = idx[:, i*L:(i+1)*L]
    r_i = embedding(W, seg).sum(dim)   # embed + reduce/段
```
（分组键=（权重身份, D, 索引形状），`:2509-2526`；同父连续切片全覆盖 `:2157-2213`；同一 reduce 算子 `:2112`）
**问题**：**docstring 原文（`:2495-2496）**「对同组的多次 embedding+reduce 合并为单次 reshape→embedding→reduce，从而显著降低调度与访存开销」——N 个各自的小 gather kernel + N 个对同权重 `W` 的 reduce [每 kernel launch 开销为硬件推断]。
**优化**：把父索引 reshape 成 `(N,L)`，一次 `embedding(W, reshaped)`，一次沿新轴 reduce（`:2407-2424`）；下游或按 `select(i)` 接回、或整体塌进一个 `cat→reshape`（`:2426-2454`）。
**效果**：`N×(slice+emb+reduce) → 1 reshape + 1 emb + 1 reduce + (N select | 1 reshape)`。**日志逐字给出该收益**（`:2464` collapsed / `:2478` 非 collapsed）。

---

## 5. 后端级融合（场景 · 问题 · 效果）

FX pass 之外，真正的重活在后端 scheduler。这三条与上游「都有但机制不同」（对照见母页 §3.6/§4.3）。

### 5.1 CATLASS EVG epilogue（`codegen/catlass/catlass_scheduling.py`）
**场景**：GEMM 后接逐元素尾巴：
```python
y = torch.relu(x @ w + bias)   # mm(CATLASS 模板) + add + relu(Pointwise epilogue)
```
（`_can_fuse_epilogue_impl:234`：消费者须是 `ComputedBuffer(Pointwise)`、同输出尺寸、读模板 buffer、非 reduction/非 mutation，`:267-297`）
**问题**：不融合时 GEMM 结果落 GM、`add/relu` 再读回——Cube→GM→Vector 往返 [硬件推断]。
**优化**：`codegen_template` 用 `CatlassEVGCodegen`（EVG=epilogue visitor graph）把尾巴折进 GEMM kernel（`:141-204`）。**门控**：`catlass_epilogue_fusion_enable AND config.epilogue_fusion`（`:298-303`）+ 模板 `epilogue_fusion_type!=0`；EVG 回退路径需 `type==2` 且**拒 bf16**（`:333-337`）。默认 **`CATLASS_EPILOGUE_FUSION=0` 关**（`config.py:76-78`）。
**效果**：epilogue 在 GEMM kernel 内片上完成，省 GEMM 结果的 GM 写回+读回。**唯一有计数器**：`counters["inductor"]["catlass_epilogue_fusion_counter"] += len(epilogue_nodes)`（`:151`）。硬限制：不支持链式 epilogue（`:256-258`）、不融进已 `FusedSchedulerNode`、不融 reduction、EVG 回退不支持 bf16。

### 5.2 DVM 图级分区融合（`dvm/graph_fusion.py`）
**场景**：一段连通的逐元素/GEMM 子图（算子都在 `GRAPH_FUSION_SUPPORT_OP` 白名单、47 个 active）：
```python
t = torch.rsqrt(x * x + eps); y = (x * t) * w   # mul/add/rsqrt/mul 链 -> 一个 dvm::fused_graph 算子
```
（`DvmOpSupport.is_node_supported:123-128`；白名单 `:42-92`，`expand/reshape` 被注释掉）
**问题**：每个逐元素算子否则各起一个 kernel、彼此间 GM 往返；DVM 把最大连通子图融成**单个**自定义算子 [launch/GM 成本为硬件推断]。
**优化**：作为 post-grad FX pass 安装（`DvmGraphFusionPatch.enable` 把 `config.post_grad_custom_post_pass=dvm_graph_fusion`，`:400-413`）：`CapabilityBasedPartitioner` 提议最大支持子图 → 按数据依赖连通性再切 → 每块 `fuse_as_graphmodule`、做 `decompose_k1_matmul`/`insert_sum_fp32_prepost_cast` 等子图内改写 → 定义动态 `dvm::fused_graph_<in>_<out>` 自定义算子（带 `flexible_layout` tag，`:246-248`）替换整块（`:311-340`）。
**效果**：整段连通区塌成**一次** DVM kernel 调用；`flexible_layout` 让共享生产者保持共享、避免每个消费者各物化定长拷贝（源注释 `:246-247`）。

### 5.3 `NPUTritonScheduling.can_fuse` 重写（`codegen/scheduling.py:579-721`）
**场景**：scheduler 问两个节点能否融合（垂直=水平都走这一个方法，`:720-721`），按 reduce 状态分支。
**问题**：上游 GPU tiling 把迭代范围塌成 1D，**NPU 需要非塌缩多轴范围**（`candidate_tilings` docstring `:726-728`「npu needs non-collapse ranges」）；naive 融合可能产生 NPU kernel 切不动、或两节点不一致的 tiling。
**优化**：在上游 numel/rnumel 检查上叠两道 NPU 门：① **tiling 门**（pw+pw，`:657-679`）——对 node1/node2/合并集各算 `select_tiling`，多于 2 维时要求三者相等否则 `why("tiling mismatch")`；② **`is_compatible` 门**（pw→reduce，`:687-690`）——每个子节点须能被 `NPUIndexTritonKernel._split_iteration_ranges` 无塌缩切到 reduction group，`CantSplit` 即拒（`triton.py:1327-1348`）。
**效果**：仅当 NPU 多轴 tiling 一致/可切时才融合，避免非法或劣化的融合 kernel；每次拒绝经 `WhyNoFuse` 记因（如 `"tiling mismatch"`）。

---

## 6. 效果的诚实边界（复述，防误引）

- **无 benchmark、无计数器**（唯一例外 CATLASS `catlass_epilogue_fusion_counter`，`catlass_scheduling.py:151`）。本页「效果」= 结构性收益（少 N 个 kernel / 少一次拷贝 / 转零拷贝 view / int32 索引），**非实测加速比**。
- **docstring/日志 = 源述动机**（可信度最高，已在各条标「原文」），**硬件因果 = 推断**（源文件无硬件注释）。
- `fold_four_op` / `fold_where` 经 `get_binary_fold_result` 会**留一个 clone**，非零成本。
- **仅静态 shape**（SymInt 即跳过）；**仅推理生效**（`fusion_attention_v3_pass` 例外，训练也跑）。
- 单条 pass 可用 `SHUT_DOWN_FX_PASS_LIST=<name>`（或 `all`）关闭（`register_custom_pass.py:15-35`）。

---

## Related Pages

- [[npu_vs_upstream_fusion_passes]] — 母页：torch_npu vs 上游融合 Pass 全流程对照（谁有谁无谁不同 + `is_gpu` 总开关）
- [[npu_inductor_optimization_analysis]] — 硬件特性 → 优化思想 → 案例（本页「效果」锚定的硬件 why 全景）
- [[npu_inductor_splittiling_backend_analysis]] — 内置 default 路径 what/how（golden_var_list、CATLASS、tiling）
- [[scheduler_analysis]] — Scheduler 融合策略与自定义 Pass（§5.3 can_fuse 的上游基线）
- [[post_grad_passes_guide]] · [[pre_grad_passes_guide]] — 上游 pass 详解（对照上游侧）
