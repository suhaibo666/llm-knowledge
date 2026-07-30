graph():
    %primals_1 : [num_users=1] = placeholder[target=primals_1]
    %view : [num_users=1] = placeholder[target=view]
    %view_2 : [num_users=1] = placeholder[target=view_2]
    %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
    %tangents_2 : [num_users=1] = placeholder[target=tangents_2]
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
    return (mm_1, None, view_9)

# Generated Python



def forward(self, primals_1, view, view_2, tangents_1, tangents_2):
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
    return (mm_1, None, view_9)
