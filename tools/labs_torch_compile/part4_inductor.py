import torch
from torch._inductor import config, metrics


def pointwise_reduction(x):
    value = torch.sin(torch.relu(x + 1))
    return value.sum(dim=1)


def compile_and_measure(max_fusion_size):
    torch._dynamo.reset()
    metrics.reset()
    with config.patch({"max_fusion_size": max_fusion_size}):
        compiled = torch.compile(pointwise_reduction, fullgraph=True)
        x = torch.randn(32, 64)
        expected = pointwise_reduction(x)
        actual = compiled(x)
        return (
            torch.allclose(actual, expected),
            metrics.generated_kernel_count,
            metrics.generated_cpp_vec_kernel_count,
        )


try:
    fused_ok, fused_kernels, fused_cpp_vec = compile_and_measure(64)
    split_ok, split_kernels, split_cpp_vec = compile_and_measure(1)
    print("pointwise_inductor_status=compiled")
    print(f"fused_output_matches={fused_ok}")
    print(f"fused_generated_kernels={fused_kernels}")
    print(f"fused_cpp_vec_kernels={fused_cpp_vec}")
    print(f"max_fusion_size_1_output_matches={split_ok}")
    print(f"max_fusion_size_1_generated_kernels={split_kernels}")
    print(f"max_fusion_size_1_cpp_vec_kernels={split_cpp_vec}")
except Exception as exc:
    message = str(exc)
    if "Compiler: cl is not found" not in message:
        raise
    print("pointwise_inductor_status=blocked_missing_msvc_cl")


def matmul_model(x, weight):
    return x @ weight


torch._dynamo.reset()
metrics.reset()
compiled_matmul = torch.compile(matmul_model, fullgraph=True)
x = torch.randn(8, 16)
weight = torch.randn(16, 32)
expected = matmul_model(x, weight)
actual = compiled_matmul(x, weight)
print(f"matmul_output_matches={torch.allclose(actual, expected)}")
print(f"matmul_generated_kernels={metrics.generated_kernel_count}")
print("triton_autotune_tested=False")
