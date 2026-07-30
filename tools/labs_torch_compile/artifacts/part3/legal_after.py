


def forward(self, x, weight, bias):
    addmm = torch.addmm(bias, x, weight);  bias = x = weight = None
    return addmm
