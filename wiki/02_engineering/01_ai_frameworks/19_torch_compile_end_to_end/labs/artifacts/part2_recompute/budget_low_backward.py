graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=1] = placeholder[target=primals_2]
    %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
    %expand : [num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [8, 8]), kwargs = {})
    %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%primals_1,), kwargs = {})
    %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%cos,), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t, %expand), kwargs = {})
    %t_1 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%primals_2,), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%expand, %t_1), kwargs = {})
    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%primals_1,), kwargs = {})
    %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sin,), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_2, %neg), kwargs = {})
    return (mul, mm_1)

# Generated Python



def forward(self, primals_1, primals_2, tangents_1):
    expand = torch.ops.aten.expand.default(tangents_1, [8, 8]);  tangents_1 = None
    cos = torch.ops.aten.cos.default(primals_1)
    t = torch.ops.aten.t.default(cos);  cos = None
    mm_1 = torch.ops.aten.mm.default(t, expand);  t = None
    t_1 = torch.ops.aten.t.default(primals_2);  primals_2 = None
    mm_2 = torch.ops.aten.mm.default(expand, t_1);  expand = t_1 = None
    sin = torch.ops.aten.sin.default(primals_1);  primals_1 = None
    neg = torch.ops.aten.neg.default(sin);  sin = None
    mul = torch.ops.aten.mul.Tensor(mm_2, neg);  mm_2 = neg = None
    return (mul, mm_1)
