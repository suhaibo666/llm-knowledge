# module_id=1579104708864
# graph_id=1579105361968
# owner_id=1579110109872

graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=2] = placeholder[target=primals_2]
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%primals_1, %primals_2), kwargs = {})
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_1, [1, 0]), kwargs = {})
    %permute_1 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_2, [1, 0]), kwargs = {})
    return (mm, permute, permute_1)

# Generated Python



def forward(self, primals_1, primals_2):
    mm = torch.ops.aten.mm.default(primals_1, primals_2)
    permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    permute_1 = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    return (mm, permute, permute_1)
