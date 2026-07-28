import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import torch._functorch.config as functorch_config
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import min_cut_rematerialization_partition


SOURCE_BASELINE = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"
HIGH_BUDGET = 1.0
LOW_BUDGET = 0.0


def model(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(torch.cos(x), weight).sum()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


class LogicalSavedTensorPeak:
    def __init__(self) -> None:
        self.active_storage_refcounts: dict[tuple[str, int, int], int] = {}
        self.live_logical_tensor_bytes = 0
        self.peak_logical_tensor_bytes = 0
        self.live_unique_backing_storage_bytes = 0
        self.peak_unique_backing_storage_bytes = 0
        self.peak_unique_storages = 0
        self.pack_count = 0
        self.unpack_count = 0
        self.events: list[dict[str, object]] = []

    def pack(self, tensor: torch.Tensor) -> list[object]:
        detached = tensor.detach()
        storage = detached.untyped_storage()
        storage_nbytes = storage.nbytes()
        storage_key = (
            str(detached.device),
            storage.data_ptr(),
            storage_nbytes,
        )
        logical_tensor_bytes = detached.numel() * detached.element_size()
        storage_refcount = self.active_storage_refcounts.get(storage_key, 0)
        if storage_refcount == 0:
            self.live_unique_backing_storage_bytes += storage_nbytes
        self.active_storage_refcounts[storage_key] = storage_refcount + 1
        self.live_logical_tensor_bytes += logical_tensor_bytes
        self.peak_logical_tensor_bytes = max(
            self.peak_logical_tensor_bytes,
            self.live_logical_tensor_bytes,
        )
        self.peak_unique_backing_storage_bytes = max(
            self.peak_unique_backing_storage_bytes,
            self.live_unique_backing_storage_bytes,
        )
        self.peak_unique_storages = max(
            self.peak_unique_storages,
            len(self.active_storage_refcounts),
        )
        self.pack_count += 1
        packed: list[object] = [
            detached,
            storage_key,
            logical_tensor_bytes,
            storage_nbytes,
            True,
        ]
        self.events.append(
            {
                "event": "pack",
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "logical_tensor_bytes": logical_tensor_bytes,
                "backing_storage_bytes": storage_nbytes,
                "live_logical_tensor_bytes": self.live_logical_tensor_bytes,
                "live_unique_backing_storage_bytes": (
                    self.live_unique_backing_storage_bytes
                ),
                "active_unique_storages": len(
                    self.active_storage_refcounts
                ),
            }
        )
        return packed

    def unpack(self, packed: list[object]) -> torch.Tensor:
        tensor, storage_key, logical_tensor_bytes, storage_nbytes, active = packed
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(storage_key, tuple)
        assert isinstance(logical_tensor_bytes, int)
        assert isinstance(storage_nbytes, int)
        if active:
            self.live_logical_tensor_bytes -= logical_tensor_bytes
            storage_refcount = self.active_storage_refcounts[storage_key] - 1
            if storage_refcount == 0:
                del self.active_storage_refcounts[storage_key]
                self.live_unique_backing_storage_bytes -= storage_nbytes
            else:
                self.active_storage_refcounts[storage_key] = storage_refcount
            packed[4] = False
        self.unpack_count += 1
        self.events.append(
            {
                "event": "unpack",
                "logical_tensor_bytes": logical_tensor_bytes,
                "backing_storage_bytes": storage_nbytes,
                "live_logical_tensor_bytes": self.live_logical_tensor_bytes,
                "live_unique_backing_storage_bytes": (
                    self.live_unique_backing_storage_bytes
                ),
                "active_unique_storages": len(
                    self.active_storage_refcounts
                ),
            }
        )
        return tensor

    def result(self) -> dict[str, object]:
        return {
            "pack_count": self.pack_count,
            "unpack_count": self.unpack_count,
            "peak_logical_tensor_bytes": self.peak_logical_tensor_bytes,
            "peak_unique_backing_storage_bytes": (
                self.peak_unique_backing_storage_bytes
            ),
            "peak_unique_storages": self.peak_unique_storages,
            "final_live_logical_tensor_bytes": self.live_logical_tensor_bytes,
            "final_live_unique_backing_storage_bytes": (
                self.live_unique_backing_storage_bytes
            ),
            "final_active_unique_storages": len(
                self.active_storage_refcounts
            ),
            "events": self.events,
        }


def _measure_budget(
    budget: float,
    base_x: torch.Tensor,
    base_weight: torch.Tensor,
) -> dict[str, object]:
    def eager_compiler(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[object],
    ):
        return make_boxed_func(graph_module.forward)

    with functorch_config.patch(activation_memory_budget=budget):
        compiled = aot_function(
            model,
            fw_compiler=eager_compiler,
            bw_compiler=eager_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        warmup_x = base_x.detach().clone().requires_grad_(True)
        warmup_weight = base_weight.detach().clone().requires_grad_(True)
        compiled(warmup_x, warmup_weight).backward()

        tracker = LogicalSavedTensorPeak()
        x = base_x.detach().clone().requires_grad_(True)
        weight = base_weight.detach().clone().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(
            tracker.pack,
            tracker.unpack,
        ):
            output = compiled(x, weight)
            output.backward()

    assert x.grad is not None
    assert weight.grad is not None
    return {
        "activation_memory_budget": budget,
        **tracker.result(),
        "output": output.detach(),
        "x_grad": x.grad.detach().clone(),
        "weight_grad": weight.grad.detach().clone(),
    }


def _json_measurement(measurement: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in measurement.items()
        if key not in {"output", "x_grad", "weight_grad"}
    }


def _artifact_records(output_dir: Path, names: list[str]) -> list[dict[str, str]]:
    return [
        {
            "path": name,
            "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
        }
        for name in names
    ]


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    base_x = torch.randn(8, 8)
    base_weight = torch.randn(8, 8)
    eager_x = base_x.detach().clone().requires_grad_(True)
    eager_weight = base_weight.detach().clone().requires_grad_(True)
    eager_output = model(eager_x, eager_weight)
    eager_output.backward()

    high = _measure_budget(HIGH_BUDGET, base_x, base_weight)
    low = _measure_budget(LOW_BUDGET, base_x, base_weight)
    gradient_matches = all(
        torch.allclose(actual, expected)
        for measurement in (high, low)
        for actual, expected in (
            (measurement["output"], eager_output.detach()),
            (measurement["x_grad"], eager_x.grad),
            (measurement["weight_grad"], eager_weight.grad),
        )
    )
    physical_allocator_peak_status = (
        "blocked_no_cuda"
        if not torch.cuda.is_available()
        else "not_measured_by_logical_activation_lab"
    )
    checks = {
        "saved_tensors_hooks_executed": (
            high["pack_count"] > 0 and low["pack_count"] > 0
        ),
        "budget_high_logical_tensor_peak_gt_low": (
            high["peak_logical_tensor_bytes"]
            > low["peak_logical_tensor_bytes"]
        ),
        "budget_high_unique_backing_storage_peak_gt_low": (
            high["peak_unique_backing_storage_bytes"]
            > low["peak_unique_backing_storage_bytes"]
        ),
        "budget_high_pack_unpack_balanced": (
            high["pack_count"] == high["unpack_count"]
        ),
        "budget_low_pack_unpack_balanced": (
            low["pack_count"] == low["unpack_count"]
        ),
        "budget_high_logical_tensor_live_bytes_returned_to_zero": (
            high["final_live_logical_tensor_bytes"] == 0
        ),
        "budget_low_logical_tensor_live_bytes_returned_to_zero": (
            low["final_live_logical_tensor_bytes"] == 0
        ),
        "budget_high_unique_backing_storage_live_bytes_returned_to_zero": (
            high["final_live_unique_backing_storage_bytes"] == 0
        ),
        "budget_low_unique_backing_storage_live_bytes_returned_to_zero": (
            low["final_live_unique_backing_storage_bytes"] == 0
        ),
        "gradient_matches": gradient_matches,
        "physical_allocator_peak_status": physical_allocator_peak_status,
        "physical_allocator_peak_measured": False,
        "logical_peak_is_physical_allocator_peak": False,
    }
    assert high["peak_logical_tensor_bytes"] == 768, high
    assert low["peak_logical_tensor_bytes"] == 512, low
    assert high["peak_unique_backing_storage_bytes"] == 768, high
    assert low["peak_unique_backing_storage_bytes"] == 512, low
    assert all(
        value is True
        for key, value in checks.items()
        if key
        not in {
            "physical_allocator_peak_status",
            "physical_allocator_peak_measured",
            "logical_peak_is_physical_allocator_peak",
        }
    ), checks
    assert checks["physical_allocator_peak_status"] == "blocked_no_cuda"
    assert checks["physical_allocator_peak_measured"] is False
    assert checks["logical_peak_is_physical_allocator_peak"] is False

    results = {
        "measurement_kinds": {
            "logical_tensor_bytes": (
                "sum_of_active_saved_tensor_numel_times_element_size"
            ),
            "unique_backing_storage_bytes": (
                "deduplicated_active_untyped_storage_nbytes"
            ),
        },
        "deduplication_rule": (
            "Logical tensor bytes count each active saved value. Unique backing "
            "storage bytes charge each active device/data_ptr/storage_nbytes "
            "storage once at untyped_storage().nbytes()."
        ),
        "lifecycle_rule": (
            "pack adds one active saved-tensor reference; its first unpack removes "
            "that reference. Peak is the maximum live logical charge over hook events."
        ),
        "budget_high": _json_measurement(high),
        "budget_low": _json_measurement(low),
        "physical_allocator_peak_bytes": None,
        "physical_allocator_peak_status": physical_allocator_peak_status,
        "physical_allocator_peak_measured": False,
        "logical_peak_is_physical_allocator_peak": False,
        "allocator_boundary": (
            "This CPU-only Lab does not call CUDA allocator APIs. Logical saved-"
            "tensor bytes, process RSS, and Scheduler estimates are not a physical "
            "CUDA allocator peak."
        ),
        "checks": checks,
    }
    environment = {
        "audit_source_baseline": SOURCE_BASELINE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_available": torch.cuda.is_available(),
        "seed": 0,
    }
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "results.json", results)
    artifact_names = ["environment.json", "results.json"]
    manifest = {
        "entrypoint": Path(__file__).resolve().as_posix(),
        "command": [sys.executable, *sys.argv],
        "model": "sum(mm(cos(x), weight))",
        "activation_memory_budgets": {
            "high": HIGH_BUDGET,
            "low": LOW_BUDGET,
        },
        "artifacts": _artifact_records(output_dir, artifact_names),
        "checks": checks,
    }
    _write_json(output_dir / "manifest.json", manifest)

    for key, value in checks.items():
        print(f"{key}={value}")
    print(
        "budget_high_peak_logical_tensor_bytes="
        f"{high['peak_logical_tensor_bytes']}"
    )
    print(
        "budget_low_peak_logical_tensor_bytes="
        f"{low['peak_logical_tensor_bytes']}"
    )
    print(
        "budget_high_peak_unique_backing_storage_bytes="
        f"{high['peak_unique_backing_storage_bytes']}"
    )
    print(
        "budget_low_peak_unique_backing_storage_bytes="
        f"{low['peak_unique_backing_storage_bytes']}"
    )
    print(f"artifact_dir={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
