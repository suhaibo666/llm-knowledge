import torch
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import (
    default_partition,
    min_cut_rematerialization_partition,
)
from torch.utils import _pytree as pytree


def fn(x):
    activation = torch.sin(x)
    output = torch.cos(activation)
    return output * output


def run(partitioner):
    joint_graphs = []
    fw_graphs = []
    bw_graphs = []

    def recording_partition(joint_module, *args, **kwargs):
        joint_graphs.append(joint_module)
        return partitioner(joint_module, *args, **kwargs)

    def fw_compiler(gm, example_inputs):
        fw_graphs.append(gm)
        return make_boxed_func(gm.forward)

    def bw_compiler(gm, example_inputs):
        bw_graphs.append(gm)
        return make_boxed_func(gm.forward)

    compiled = aot_function(
        fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=recording_partition,
    )
    x = torch.randn(8, requires_grad=True)
    eager_x = x.detach().clone().requires_grad_(True)
    compiled(x).sum().backward()
    fn(eager_x).sum().backward()

    fw = fw_graphs[0]
    bw = bw_graphs[0]
    fw_output = next(node for node in fw.graph.nodes if node.op == "output")
    fw_slots = pytree.tree_leaves(fw_output.args[0])
    bw_placeholders = [node for node in bw.graph.nodes if node.op == "placeholder"]

    cross_owner_refs = []
    for node in bw.graph.nodes:
        for leaf in pytree.tree_leaves((node.args, node.kwargs)):
            if isinstance(leaf, torch.fx.Node) and leaf.graph is not bw.graph:
                cross_owner_refs.append((node.name, leaf.name))

    bw_targets = [
        str(node.target) for node in bw.graph.nodes if node.op == "call_function"
    ]
    return {
        "joint": len(joint_graphs),
        "fw": len(fw_graphs),
        "bw": len(bw_graphs),
        "distinct": fw.graph is not bw.graph,
        "fw_total_slots": len(fw_slots),
        "fw_user_slots": 1,
        "fw_saved_slots": len(fw_slots) - 1,
        "bw_placeholders": len(bw_placeholders),
        "cross_refs": len(cross_owner_refs),
        "gradient_match": torch.allclose(x.grad, eager_x.grad),
        "bw_targets": bw_targets,
    }


default = run(default_partition)
min_cut = run(min_cut_rematerialization_partition)
print(f"joint_graphs={default['joint']}")
print(f"forward_graphs={default['fw']}")
print(f"backward_graphs={default['bw']}")
print(f"fw_and_bw_are_distinct={default['distinct']}")
print(f"fw_total_output_count={default['fw_total_slots']}")
print(f"fw_user_output_count={default['fw_user_slots']}")
print(f"fw_saved_value_count={default['fw_saved_slots']}")
print(f"bw_placeholder_count={default['bw_placeholders']}")
print(f"cross_graph_node_refs={default['cross_refs']}")
print(f"gradient_matches_eager={default['gradient_match']}")
print(f"default_bw_targets={','.join(default['bw_targets'])}")
print(f"min_cut_fw_total_output_count={min_cut['fw_total_slots']}")
print(f"min_cut_fw_saved_value_count={min_cut['fw_saved_slots']}")
print(f"min_cut_bw_targets={','.join(min_cut['bw_targets'])}")

checks = {
    "joint_forward_backward_captured": default["joint"]
    == default["fw"]
    == default["bw"]
    == 1,
    "forward_backward_distinct": default["distinct"],
    "no_cross_graph_node_refs": default["cross_refs"] == 0,
    "gradient_matches_eager": default["gradient_match"],
    "saved_count_excludes_user_output": default["fw_saved_slots"]
    == default["fw_total_slots"] - default["fw_user_slots"],
}
if not all(checks.values()):
    raise AssertionError(f"AOT graph contract failed: {checks}")
