


def forward(self, primals_1, primals_2, primals_3):
    addmm_default = torch.ops.aten.addmm.default(primals_3, primals_1, primals_2);  primals_3 = None
    permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    permute_1 = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    return (addmm_default, permute, permute_1)
