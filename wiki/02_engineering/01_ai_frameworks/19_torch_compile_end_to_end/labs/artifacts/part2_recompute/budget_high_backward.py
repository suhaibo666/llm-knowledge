graph():
    %primals_1 : [num_users=1] = placeholder[target=primals_1]
    %t : [num_users=1] = placeholder[target=t]
    %t_1 : [num_users=1] = placeholder[target=t_1]
    %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
    %expand : [num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [8, 8]), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t, %expand), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%expand, %t_1), kwargs = {})
    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%primals_1,), kwargs = {})
    %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sin,), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_2, %neg), kwargs = {})
    return (mul, mm_1)

# Generated Python



def forward(self, primals_1, t, t_1, tangents_1):
    expand = torch.ops.aten.expand.default(tangents_1, [8, 8]);  tangents_1 = None
    mm_1 = torch.ops.aten.mm.default(t, expand);  t = None
    mm_2 = torch.ops.aten.mm.default(expand, t_1);  expand = t_1 = None
    sin = torch.ops.aten.sin.default(primals_1);  primals_1 = None
    neg = torch.ops.aten.neg.default(sin);  sin = None
    mul = torch.ops.aten.mul.Tensor(mm_2, neg);  mm_2 = neg = None
    return (mul, mm_1)
