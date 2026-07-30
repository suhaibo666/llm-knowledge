graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=1] = placeholder[target=primals_2]
    %primals_3 : [num_users=1] = placeholder[target=primals_3]
    %view : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%primals_3, [-1, 4]), kwargs = {})
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %primals_1), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %primals_2), kwargs = {})
    %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {})
    %view_1 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%clone, [-1]), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, 0.25), kwargs = {})
    %view_2 : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%add_1, [3, 4]), kwargs = {})
    %sin : [num_users=2] = call_function[target=torch.ops.aten.sin.default](args = (%view_2,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sin, [1]), kwargs = {})
    return (sin, sum_1, primals_1, view, view_2)

# Generated Python



def forward(self, primals_1, primals_2, primals_3):
    view = torch.ops.aten.view.default(primals_3, [-1, 4]);  primals_3 = None
    mm = torch.ops.aten.mm.default(view, primals_1)
    add = torch.ops.aten.add.Tensor(mm, primals_2);  mm = primals_2 = None
    clone = torch.ops.aten.clone.default(add);  add = None
    view_1 = torch.ops.aten.view.default(clone, [-1]);  clone = None
    add_1 = torch.ops.aten.add.Tensor(view_1, 0.25);  view_1 = None
    view_2 = torch.ops.aten.view.default(add_1, [3, 4]);  add_1 = None
    sin = torch.ops.aten.sin.default(view_2)
    sum_1 = torch.ops.aten.sum.dim_IntList(sin, [1])
    return (sin, sum_1, primals_1, view, view_2)
