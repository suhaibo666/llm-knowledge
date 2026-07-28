graph():
    %p_weight : [num_users=1] = placeholder[target=p_weight]
    %b_offset : [num_users=1] = placeholder[target=b_offset]
    %x : [num_users=1] = placeholder[target=x]
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%x, [-1, 4]), kwargs = {})
    %matmul : [num_users=1] = call_function[target=torch.ops.aten.matmul.default](args = (%view, %p_weight), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%matmul, %b_offset), kwargs = {})
    %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {})
    %view_1 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%clone, [-1]), kwargs = {})
    %add_ : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%view_1, 0.25), kwargs = {})
    %sin : [num_users=2] = call_function[target=torch.ops.aten.sin.default](args = (%clone,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sin, [1]), kwargs = {})
    return (sin, sum_1)

# Generated Python



def forward(self, p_weight, b_offset, x):
    view = torch.ops.aten.view.default(x, [-1, 4]);  x = None
    matmul = torch.ops.aten.matmul.default(view, p_weight);  view = p_weight = None
    add = torch.ops.aten.add.Tensor(matmul, b_offset);  matmul = b_offset = None
    clone = torch.ops.aten.clone.default(add);  add = None
    view_1 = torch.ops.aten.view.default(clone, [-1])
    add_ = torch.ops.aten.add_.Tensor(view_1, 0.25);  view_1 = add_ = None
    sin = torch.ops.aten.sin.default(clone);  clone = None
    sum_1 = torch.ops.aten.sum.dim_IntList(sin, [1])
    return (sin, sum_1)
