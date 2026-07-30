graph():
    %x : [num_users=1] = placeholder[target=x]
    %predicate : [num_users=1] = placeholder[target=predicate]
    %true_graph_0 : [num_users=1] = get_attr[target=true_graph_0]
    %false_graph_0 : [num_users=1] = get_attr[target=false_graph_0]
    %cond : [num_users=1] = call_function[target=torch.ops.higher_order.cond](args = (%predicate, %true_graph_0, %false_graph_0, (%x,)), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%cond, 0), kwargs = {})
    return (getitem,)

# Generated Python



def forward(self, x, predicate):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(predicate, true_graph_0, false_graph_0, (x,));  predicate = true_graph_0 = false_graph_0 = x = None
    getitem = cond[0];  cond = None
    return (getitem,)
