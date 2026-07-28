


def forward(self, L_x_ : torch.Tensor, L_weight_ : torch.Tensor, L_bias_ : torch.Tensor):
    l_x_ = L_x_
    l_weight_ = L_weight_
    l_bias_ = L_bias_
    addmm = torch.addmm(l_bias_, l_x_, l_weight_);  l_bias_ = l_x_ = l_weight_ = None
    return (addmm,)
