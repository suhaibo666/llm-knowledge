graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %view_copy : [num_users=1] = call_function[target=torch.ops.aten.view_copy.default](args = (%arg0_1, [-1, 4]), kwargs = {})
    %_param_constant0 : [num_users=1] = get_attr[target=_param_constant0]
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_copy, %_param_constant0), kwargs = {})
    %_tensor_constant0 : [num_users=1] = get_attr[target=_tensor_constant0]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %_tensor_constant0), kwargs = {})
    %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {})
    %view_copy_1 : [num_users=1] = call_function[target=torch.ops.aten.view_copy.default](args = (%clone, [-1]), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_copy_1, 0.25), kwargs = {})
    %view_copy_2 : [num_users=2] = call_function[target=torch.ops.aten.view_copy.default](args = (%add_1, [3, 4]), kwargs = {})
    %view_copy_3 : [num_users=0] = call_function[target=torch.ops.aten.view_copy.default](args = (%view_copy_2, [-1]), kwargs = {})
    %sin : [num_users=2] = call_function[target=torch.ops.aten.sin.default](args = (%view_copy_2,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sin, [1]), kwargs = {})
    return {'activation': sin, 'summary': sum_1}

# Generated Python



def forward(self, arg0_1):
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [-1, 4]);  arg0_1 = None
    _param_constant0 = self._param_constant0
    mm = torch.ops.aten.mm.default(view_copy, _param_constant0);  view_copy = _param_constant0 = None
    _tensor_constant0 = self._tensor_constant0
    add = torch.ops.aten.add.Tensor(mm, _tensor_constant0);  mm = _tensor_constant0 = None
    clone = torch.ops.aten.clone.default(add);  add = None
    view_copy_1 = torch.ops.aten.view_copy.default(clone, [-1]);  clone = None
    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 0.25);  view_copy_1 = None
    view_copy_2 = torch.ops.aten.view_copy.default(add_1, [3, 4]);  add_1 = None
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [-1]);  view_copy_3 = None
    sin = torch.ops.aten.sin.default(view_copy_2);  view_copy_2 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(sin, [1])
    return {'activation': sin, 'summary': sum_1}
