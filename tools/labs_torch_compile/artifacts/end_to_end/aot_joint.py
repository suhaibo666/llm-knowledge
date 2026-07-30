graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=1] = placeholder[target=primals_2]
    %primals_3 : [num_users=1] = placeholder[target=primals_3]
    %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
    %tangents_2 : [num_users=1] = placeholder[target=tangents_2]
    %view : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%primals_3, [-1, 4]), kwargs = {})
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %primals_1), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %primals_2), kwargs = {})
    %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {})
    %view_1 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%clone, [-1]), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, 0.25), kwargs = {})
    %view_2 : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%add_1, [3, 4]), kwargs = {})
    %sin : [num_users=2] = call_function[target=torch.ops.aten.sin.default](args = (%view_2,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sin, [1]), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%tangents_2, 1), kwargs = {})
    %expand : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [3, 4]), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %expand), kwargs = {})
    %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_2,), kwargs = {})
    %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %cos), kwargs = {})
    %new_empty_strided : [num_users=1] = call_function[target=torch.ops.aten.new_empty_strided.default](args = (%mul, [3, 4], [4, 1]), kwargs = {})
    %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%new_empty_strided, %mul), kwargs = {})
    %view_6 : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%copy, [-1]), kwargs = {})
    %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_6,), kwargs = {memory_format: torch.contiguous_format})
    %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%view_6, %clone_1), kwargs = {})
    %view_7 : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%copy_1, [3, 4]), kwargs = {})
    %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%view,), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t, %view_7), kwargs = {})
    %t_1 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%primals_1,), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_7, %t_1), kwargs = {})
    %view_9 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%mm_2, [3, 4]), kwargs = {})
    return [sin, sum_1, mm_1, None, view_9]

# Generated Python



def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    view = torch.ops.aten.view.default(primals_3, [-1, 4]);  primals_3 = None
    mm = torch.ops.aten.mm.default(view, primals_1)
    add = torch.ops.aten.add.Tensor(mm, primals_2);  mm = primals_2 = None
    clone = torch.ops.aten.clone.default(add);  add = None
    view_1 = torch.ops.aten.view.default(clone, [-1]);  clone = None
    add_1 = torch.ops.aten.add.Tensor(view_1, 0.25);  view_1 = None
    view_2 = torch.ops.aten.view.default(add_1, [3, 4]);  add_1 = None
    sin = torch.ops.aten.sin.default(view_2)
    sum_1 = torch.ops.aten.sum.dim_IntList(sin, [1])
    unsqueeze = torch.ops.aten.unsqueeze.default(tangents_2, 1);  tangents_2 = None
    expand = torch.ops.aten.expand.default(unsqueeze, [3, 4]);  unsqueeze = None
    add_2 = torch.ops.aten.add.Tensor(tangents_1, expand);  tangents_1 = expand = None
    cos = torch.ops.aten.cos.default(view_2);  view_2 = None
    mul = torch.ops.aten.mul.Tensor(add_2, cos);  add_2 = cos = None
    new_empty_strided = torch.ops.aten.new_empty_strided.default(mul, [3, 4], [4, 1])
    copy = torch.ops.aten.copy.default(new_empty_strided, mul);  new_empty_strided = mul = None
    view_6 = torch.ops.aten.view.default(copy, [-1]);  copy = None
    clone_1 = torch.ops.aten.clone.default(view_6, memory_format = torch.contiguous_format)
    copy_1 = torch.ops.aten.copy.default(view_6, clone_1);  view_6 = clone_1 = None
    view_7 = torch.ops.aten.view.default(copy_1, [3, 4]);  copy_1 = None
    t = torch.ops.aten.t.default(view);  view = None
    mm_1 = torch.ops.aten.mm.default(t, view_7);  t = None
    t_1 = torch.ops.aten.t.default(primals_1);  primals_1 = None
    mm_2 = torch.ops.aten.mm.default(view_7, t_1);  view_7 = t_1 = None
    view_9 = torch.ops.aten.view.default(mm_2, [3, 4]);  mm_2 = None
    return pytree.tree_unflatten([sin, sum_1, mm_1, None, view_9], self._out_spec)
