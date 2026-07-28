graph():
    %x : torch.Tensor [num_users=1] = placeholder[target=x]
    %weight : [num_users=1] = get_attr[target=weight]
    %view : [num_users=1] = call_method[target=view](args = (%x, -1, 4), kwargs = {})
    %matmul : [num_users=1] = call_function[target=operator.matmul](args = (%view, %weight), kwargs = {})
    %offset : [num_users=1] = get_attr[target=offset]
    %add : [num_users=1] = call_function[target=operator.add](args = (%matmul, %offset), kwargs = {})
    %clone : [num_users=2] = call_method[target=clone](args = (%add,), kwargs = {})
    %view_1 : [num_users=1] = call_method[target=view](args = (%clone, -1), kwargs = {})
    %add_ : [num_users=0] = call_method[target=add_](args = (%view_1, 0.25), kwargs = {})
    %sin : [num_users=2] = call_function[target=torch.sin](args = (%clone,), kwargs = {})
    %sum_1 : [num_users=1] = call_method[target=sum](args = (%sin,), kwargs = {dim: 1})
    return {'activation': sin, 'summary': sum_1}

# Generated Python



def forward(self, x : torch.Tensor) -> dict[str,torch.Tensor]:
    weight = self.weight
    view = x.view(-1, 4);  x = None
    matmul = view @ weight;  view = weight = None
    offset = self.offset
    add = matmul + offset;  matmul = offset = None
    clone = add.clone();  add = None
    view_1 = clone.view(-1)
    add_ = view_1.add_(0.25);  view_1 = add_ = None
    sin = torch.sin(clone);  clone = None
    sum_1 = sin.sum(dim = 1)
    return {'activation': sin, 'summary': sum_1}
