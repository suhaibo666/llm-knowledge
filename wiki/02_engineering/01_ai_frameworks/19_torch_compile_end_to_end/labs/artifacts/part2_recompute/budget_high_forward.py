graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=2] = placeholder[target=primals_2]
    %cos : [num_users=2] = call_function[target=torch.ops.aten.cos.default](args = (%primals_1,), kwargs = {})
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%cos, %primals_2), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mm,), kwargs = {})
    %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%cos,), kwargs = {})
    %t_1 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%primals_2,), kwargs = {})
    return (sum_1, primals_1, t, t_1)

# Generated Python



def forward(self, primals_1, primals_2):
    cos = torch.ops.aten.cos.default(primals_1)
    mm = torch.ops.aten.mm.default(cos, primals_2)
    sum_1 = torch.ops.aten.sum.default(mm);  mm = None
    t = torch.ops.aten.t.default(cos);  cos = None
    t_1 = torch.ops.aten.t.default(primals_2);  primals_2 = None
    return (sum_1, primals_1, t, t_1)
