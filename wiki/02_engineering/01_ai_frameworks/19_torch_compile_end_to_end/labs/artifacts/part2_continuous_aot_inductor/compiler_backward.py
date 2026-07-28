# module_id=1579102678592
# graph_id=1579102572368
# owner_id=1579103132640

graph():
    %permute : [num_users=1] = placeholder[target=permute]
    %permute_1 : [num_users=1] = placeholder[target=permute_1]
    %tangents_1 : [num_users=2] = placeholder[target=tangents_1]
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%permute, %tangents_1), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%tangents_1, %permute_1), kwargs = {})
    return (mm_2, mm_1)

# Generated Python



def forward(self, permute, permute_1, tangents_1):
    mm_1 = torch.ops.aten.mm.default(permute, tangents_1);  permute = None
    mm_2 = torch.ops.aten.mm.default(tangents_1, permute_1);  tangents_1 = permute_1 = None
    return (mm_2, mm_1)
