graph():
    %l_self_parameters_weight_ : torch.nn.parameter.Parameter [num_users=1] = placeholder[target=L_self_parameters_weight_]
    %l_self_buffers_offset_ : torch.Tensor [num_users=1] = placeholder[target=L_self_buffers_offset_]
    %s77 : torch.SymInt [num_users=0] = placeholder[target=s77]
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %view : [num_users=1] = call_method[target=view](args = (%l_x_, -1, 4), kwargs = {})
    %projected : [num_users=1] = call_function[target=operator.matmul](args = (%view, %l_self_parameters_weight_), kwargs = {})
    %biased : [num_users=1] = call_function[target=operator.add](args = (%projected, %l_self_buffers_offset_), kwargs = {})
    %scratch : [num_users=2] = call_method[target=clone](args = (%biased,), kwargs = {})
    %view_1 : [num_users=1] = call_method[target=view](args = (%scratch, -1), kwargs = {})
    %add_ : [num_users=0] = call_method[target=add_](args = (%view_1, 0.25), kwargs = {})
    %activation : [num_users=2] = call_function[target=torch.sin](args = (%scratch,), kwargs = {})
    %summary : [num_users=1] = call_method[target=sum](args = (%activation,), kwargs = {dim: 1})
    return (activation, summary)

# Generated Python



def forward(self, L_self_parameters_weight_ : torch.nn.parameter.Parameter, L_self_buffers_offset_ : torch.Tensor, s77 : torch.SymInt, L_x_ : torch.Tensor):
    l_self_parameters_weight_ = L_self_parameters_weight_
    l_self_buffers_offset_ = L_self_buffers_offset_
    l_x_ = L_x_
    view = l_x_.view(-1, 4);  l_x_ = None
    projected = view @ l_self_parameters_weight_;  view = l_self_parameters_weight_ = None
    biased = projected + l_self_buffers_offset_;  projected = l_self_buffers_offset_ = None
    scratch = biased.clone();  biased = None
    view_1 = scratch.view(-1)
    add_ = view_1.add_(0.25);  view_1 = add_ = None
    activation = torch.sin(scratch);  scratch = None
    summary = activation.sum(dim = 1)
    return (activation, summary)
