


def forward(self, x, weight, bias):
    matmul = torch.matmul(x, weight);  x = weight = None
    add = torch.add(matmul, bias);  matmul = bias = None
    return add
