from __future__ import annotations

import json
import importlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from demo_harness import (
    CapabilitySnapshot,
    CaseSpec,
    DemoContext,
    execute_case,
    exit_code_for,
    write_run_summary,
)


def fake_context(
    output_dir: Path,
    *,
    cuda_available: bool = False,
) -> DemoContext:
    capabilities = CapabilitySnapshot(
        torch_available=True,
        cuda_available=cuda_available,
        cuda_device_count=1 if cuda_available else 0,
        distributed_available=True,
        triton_available=False,
        native_compiler_available=False,
        linux=False,
        details={"test_snapshot": True},
    )
    return DemoContext(
        volume="T",
        device="cuda" if cuda_available else "cpu",
        output_dir=output_dir,
        seed=0,
        capabilities=capabilities,
    )


class HarnessContractTest(unittest.TestCase):
    def test_missing_cuda_is_blocked_before_case_body_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            called = False

            def body(context: DemoContext) -> dict[str, object]:
                nonlocal called
                called = True
                return {"unexpected": True}

            result = execute_case(
                CaseSpec(
                    case_id="cuda_case",
                    title="CUDA case",
                    pages=("a01",),
                    requirements=("cuda",),
                    run=body,
                ),
                fake_context(Path(temp_dir)),
            )

        self.assertEqual(result.status, "BLOCKED")
        self.assertFalse(called)
        self.assertIn("cuda", result.missing_requirements)

    def test_exception_is_fail_and_is_not_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:

            def broken(context: DemoContext) -> dict[str, object]:
                raise ZeroDivisionError("teaching failure")

            result = execute_case(
                CaseSpec(
                    case_id="broken",
                    title="Broken case",
                    pages=("a01",),
                    requirements=(),
                    run=broken,
                ),
                fake_context(Path(temp_dir)),
            )

        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.error["type"], "ZeroDivisionError")
        self.assertIn("teaching failure", result.error["message"])

    def test_pass_requires_mapping_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_case(
                CaseSpec(
                    case_id="passing",
                    title="Passing case",
                    pages=("a01",),
                    requirements=(),
                    run=lambda context: {"observed": True},
                ),
                fake_context(Path(temp_dir)),
            )

        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.observations, {"observed": True})

    def test_exit_code_prioritizes_fail_over_blocked(self) -> None:
        self.assertEqual(exit_code_for(["PASS"]), 0)
        self.assertEqual(exit_code_for(["PASS", "BLOCKED"]), 3)
        self.assertEqual(exit_code_for(["BLOCKED", "FAIL"]), 2)

    def test_summary_is_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = fake_context(root)
            result = execute_case(
                CaseSpec(
                    case_id="passing",
                    title="Passing case",
                    pages=("a01",),
                    requirements=(),
                    run=lambda current: {"value": 7},
                ),
                context,
            )
            summary_path = write_run_summary(context, [result])
            document = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(document["schema_version"], "torch-compile-volume-demo/v1")
        self.assertEqual(document["volume"], "T")
        self.assertEqual(document["status_counts"], {"PASS": 1})
        self.assertEqual(document["cases"][0]["case_id"], "passing")


class VolumeABContractTest(unittest.TestCase):
    def test_volume_a_registry_and_cpu_mechanisms(self) -> None:
        import demo_a_execution_model as volume_a

        expected = {
            "tensor_storage_layout",
            "dispatcher_autograd",
            "python_frame_bytecode",
            "proxy_fake_tensor",
            "eager_compile_cost",
        }
        self.assertEqual({case.case_id for case in volume_a.CASES}, expected)
        by_id = {case.case_id: case for case in volume_a.CASES}

        with tempfile.TemporaryDirectory() as temp_dir:
            context = fake_context(Path(temp_dir))
            storage = execute_case(by_id["tensor_storage_layout"], context)
            dispatch = execute_case(by_id["dispatcher_autograd"], context)

        self.assertEqual(storage.status, "PASS")
        self.assertTrue(storage.observations["view_shares_storage"])
        self.assertFalse(storage.observations["clone_shares_storage"])
        self.assertEqual(dispatch.status, "PASS")
        self.assertTrue(dispatch.observations["dispatch_observed"])
        self.assertTrue(dispatch.observations["gradient_matches"])

    def test_volume_a_cuda_timing_blocks_without_cuda(self) -> None:
        import demo_a_execution_model as volume_a

        timing = next(
            case for case in volume_a.CASES if case.case_id == "eager_compile_cost"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            context = DemoContext(
                volume="A",
                device="cuda",
                output_dir=Path(temp_dir),
                seed=0,
                capabilities=fake_context(Path(temp_dir)).capabilities,
            )
            result = execute_case(timing, context)

        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("cuda", result.missing_requirements)

    def test_volume_b_registry_and_compile_cache_mechanisms(self) -> None:
        import demo_b_dynamo_capture as volume_b

        expected = {
            "compile_lifecycle",
            "backend_modes_fullgraph",
            "eval_frame_cache",
            "bytecode_state_machine",
            "variable_source_guards",
            "output_graph_side_effects",
            "guards_recompile",
            "graph_break_resume",
            "dynamic_shapes",
            "custom_backend_contract",
        }
        self.assertEqual({case.case_id for case in volume_b.CASES}, expected)
        by_id = {case.case_id: case for case in volume_b.CASES}

        with tempfile.TemporaryDirectory() as temp_dir:
            context = fake_context(Path(temp_dir))
            lifecycle = execute_case(by_id["compile_lifecycle"], context)
            guards = execute_case(by_id["guards_recompile"], context)

        self.assertEqual(lifecycle.status, "PASS")
        self.assertEqual(lifecycle.observations["first_call_compile_count"], 1)
        self.assertEqual(lifecycle.observations["second_call_compile_count"], 1)
        self.assertEqual(guards.status, "PASS")
        self.assertGreaterEqual(guards.observations["final_compile_count"], 2)


class VolumeCContractTest(unittest.TestCase):
    def test_volume_c_registry_covers_all_graph_compiler_pages(self) -> None:
        import demo_c_graph_compiler as volume_c

        expected = {
            "ir_fx",
            "capture_normalize",
            "aot_recompute",
            "pattern_rewrite",
            "inductor_ir",
            "full_bundle",
        }
        self.assertEqual({case.case_id for case in volume_c.CASES}, expected)
        covered_pages = {
            page
            for case in volume_c.CASES
            if case.case_id != "full_bundle"
            for page in case.pages
        }
        self.assertEqual(covered_pages, {f"c{index:02d}" for index in range(1, 22)})

    def test_volume_c_child_runner_preserves_process_evidence(self) -> None:
        import demo_c_graph_compiler as volume_c

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child = root / "child.py"
            child.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "target = Path(sys.argv[2])\n"
                "target.mkdir(parents=True, exist_ok=True)\n"
                "(target / 'evidence.txt').write_text('ok', encoding='utf-8')\n"
                "print('child-observation')\n",
                encoding="utf-8",
            )
            context = fake_context(root / "run")
            observations = volume_c.run_child_scripts(
                context,
                (volume_c.ChildScript(child.name, requires_output_dir=True),),
                script_root=root,
                python_executable=sys.executable,
            )

        self.assertEqual(observations["child_count"], 1)
        self.assertEqual(observations["children"][0]["exit_code"], 0)
        self.assertIn("child-observation", observations["children"][0]["stdout"])
        self.assertEqual(observations["children"][0]["artifacts"], ["evidence.txt"])

    def test_volume_c_child_failure_is_a_case_failure(self) -> None:
        import demo_c_graph_compiler as volume_c

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child = root / "broken.py"
            child.write_text("raise SystemExit(7)\n", encoding="utf-8")
            context = fake_context(root / "run")
            spec = CaseSpec(
                case_id="child_failure",
                title="Child failure",
                pages=("c01",),
                requirements=(),
                run=lambda current: volume_c.run_child_scripts(
                    current,
                    (volume_c.ChildScript(child.name),),
                    script_root=root,
                    python_executable=sys.executable,
                ),
            )
            result = execute_case(spec, context)

        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.error["type"], "ChildProcessError")


class VolumeDEFContractTest(unittest.TestCase):
    def _assert_registry(
        self,
        module: object,
        expected: set[str],
        expected_pages: set[str],
    ) -> dict[str, CaseSpec]:
        cases = getattr(module, "CASES")
        self.assertEqual({case.case_id for case in cases}, expected)
        self.assertEqual(
            {page for case in cases for page in case.pages},
            expected_pages,
        )
        return {case.case_id: case for case in cases}

    def test_volume_d_registry_and_lazy_backward(self) -> None:
        import demo_d_artifact_runtime as volume_d

        by_id = self._assert_registry(
            volume_d,
            {
                "compile_fx_orchestration",
                "aot_wrappers_lazy_backward",
                "async_compile_loading",
                "cache_keys_invalidation",
                "wrapper_memory_reuse",
                "cudagraph_replay",
                "artifact_lifecycle_failure",
            },
            {f"d{index:02d}" for index in range(1, 8)},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_case(
                by_id["aot_wrappers_lazy_backward"],
                fake_context(Path(temp_dir)),
            )
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.observations["fw_compile_after_forward"], 1)
        self.assertEqual(result.observations["bw_compile_after_forward"], 0)
        self.assertEqual(result.observations["bw_compile_after_backward"], 1)
        self.assertTrue(result.observations["gradient_matches"])

    def test_volume_d_cuda_case_blocks_without_execution(self) -> None:
        import demo_d_artifact_runtime as volume_d

        case = next(
            item for item in volume_d.CASES if item.case_id == "cudagraph_replay"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_case(case, fake_context(Path(temp_dir)))
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("cuda", result.missing_requirements)

    def test_volume_e_registry_explain_and_rollout(self) -> None:
        import demo_e_diagnostics as volume_e

        by_id = self._assert_registry(
            volume_e,
            {
                "logs_artifact_map",
                "dynamo_explain",
                "guard_failure",
                "stage_failure_localization",
                "minifier_repro",
                "correctness_validation",
                "cold_warm_steady",
                "fusion_memory_profiler",
                "rollout_fallback",
            },
            {f"e{index:02d}" for index in range(1, 10)},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            context = fake_context(Path(temp_dir))
            explain = execute_case(by_id["dynamo_explain"], context)
            rollout = execute_case(by_id["rollout_fallback"], context)
        self.assertEqual(explain.status, "PASS")
        self.assertGreaterEqual(explain.observations["graph_count"], 1)
        self.assertGreaterEqual(explain.observations["graph_break_count"], 1)
        self.assertEqual(rollout.status, "PASS")
        self.assertEqual(rollout.observations["compiled_successes"], 1)
        self.assertEqual(rollout.observations["eager_fallbacks"], 1)
        self.assertTrue(rollout.observations["fallback_matches_eager"])

    def test_volume_f_registry_checkpoint_and_custom_backend(self) -> None:
        import demo_f_advanced_topics as volume_f

        by_id = self._assert_registry(
            volume_f,
            {
                "compiled_autograd",
                "checkpoint_recompute",
                "ddp_compile",
                "fsdp_dtensor",
                "custom_op_contract",
                "custom_backend",
                "aotinductor_package",
                "inference_freezing_cudagraph",
            },
            {f"f{index:02d}" for index in range(1, 9)},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            context = fake_context(Path(temp_dir))
            checkpoint = execute_case(by_id["checkpoint_recompute"], context)
            backend = execute_case(by_id["custom_backend"], context)
        self.assertEqual(checkpoint.status, "PASS")
        self.assertTrue(checkpoint.observations["gradient_matches"])
        self.assertGreater(
            checkpoint.observations["checkpoint_forward_calls"],
            checkpoint.observations["baseline_forward_calls"],
        )
        self.assertEqual(backend.status, "PASS")
        self.assertEqual(backend.observations["compile_count"], 1)
        self.assertTrue(backend.observations["output_matches"])

    def test_volume_f_ddp_declares_platform_runtime_gate(self) -> None:
        import demo_f_advanced_topics as volume_f

        case = next(
            item for item in volume_f.CASES if item.case_id == "ddp_compile"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_case(case, fake_context(Path(temp_dir)))
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("linux", result.missing_requirements)

    def test_cuda_first_advanced_cases_are_explicitly_blocked(self) -> None:
        import demo_e_diagnostics as volume_e
        import demo_f_advanced_topics as volume_f

        selected = [
            next(case for case in volume_e.CASES if case.case_id == "cold_warm_steady"),
            next(
                case
                for case in volume_f.CASES
                if case.case_id == "inference_freezing_cudagraph"
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            context = fake_context(Path(temp_dir))
            results = [execute_case(case, context) for case in selected]
        self.assertTrue(all(result.status == "BLOCKED" for result in results))
        self.assertTrue(
            all("cuda" in result.missing_requirements for result in results)
        )


# kb-reorg P4 physically relocates course volumes out of
# 19_torch_compile_end_to_end one volume (or, for D, one page) at a time
# (Task 4: E -> 07_debugging; Task 5: B -> 01_dynamo; Task 6: D scatters
# across four directories per spec appendix A instead of moving wholesale).
# Manifest entries still carry only a bare filename, so resolve each entry's
# directory from its "volume" field (and, for D, its "page_id"); volumes/
# pages not yet moved fall back to the legacy course directory.
_D_PAGE_ROOTS = {
    "d01": ("02_compile_stack", "04_inductor"),
    "d02": ("02_compile_stack", "02_aot_autograd"),
    "d03": ("02_compile_stack", "04_inductor"),
    "d04": ("02_compile_stack", "06_compile_cache"),
    "d06": ("03_runtime_graphs", "cuda"),
    "d07": ("02_compile_stack", "07_debugging"),
}
# Volume C physically scatters out of 19_torch_compile_end_to_end in two
# batches (kb-reorg P4 Task 7: 12 FX-data-model/pass pages -> new
# 03_graph_ir_and_passes; Task 8 will move the remaining 9 pages -> 01_dynamo /
# 02_aot_autograd / 04_inductor). Only list pages that have actually moved;
# unlisted c-ids fall back to the legacy course directory below.
_C_PAGE_ROOTS = {
    "c02": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c04": ("02_compile_stack", "01_dynamo"),
    "c09": ("02_compile_stack", "02_aot_autograd"),
    "c10": ("02_compile_stack", "02_aot_autograd"),
    "c17": ("02_compile_stack", "04_inductor"),
    "c18": ("02_compile_stack", "04_inductor"),
    "c19": ("02_compile_stack", "04_inductor"),
    "c20": ("02_compile_stack", "04_inductor"),
    "c21": ("02_compile_stack", "04_inductor"),
    "c03": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c05": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c06": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c07": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c08": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c11": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c12": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c13": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c14": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c15": ("02_compile_stack", "03_graph_ir_and_passes"),
    "c16": ("02_compile_stack", "03_graph_ir_and_passes"),
}
# Volume F scatters out of 19_torch_compile_end_to_end across six
# directories (kb-reorg P4 Task 9, mirroring D/C's scatter pattern rather
# than B/E's wholesale move): f01 -> eager_runtime autograd engine, f02 ->
# aot_autograd, f03/f04 -> distributed_primitives, f05 -> fx_export
# extensibility, f06 -> dispatcher_and_device, f07 -> inductor. f08 already
# moved in Task 6 alongside D06 (see the standalone check below).
_F_PAGE_ROOTS = {
    "f01": ("01_eager_runtime", "05_autograd_engine"),
    "f02": ("02_compile_stack", "02_aot_autograd"),
    "f03": ("04_export_and_distributed", "02_distributed_primitives"),
    "f04": ("04_export_and_distributed", "02_distributed_primitives"),
    "f05": ("04_export_and_distributed", "01_fx_export_extensibility"),
    "f06": ("01_eager_runtime", "02_dispatcher_and_device"),
    "f07": ("02_compile_stack", "04_inductor"),
    "f08": ("03_runtime_graphs", "cuda"),
}


def _page_root(labs_root: Path, volume: str, page_id: str | None = None) -> Path:
    ai_frameworks_root = (
        labs_root.parent.parent / "wiki" / "02_engineering" / "01_ai_frameworks"
    )
    if volume == "E":
        return ai_frameworks_root / "02_compile_stack" / "07_debugging"
    if volume == "B":
        return ai_frameworks_root / "02_compile_stack" / "01_dynamo"
    if volume == "D" and page_id in _D_PAGE_ROOTS:
        return ai_frameworks_root.joinpath(*_D_PAGE_ROOTS[page_id])
    if volume == "C" and page_id in _C_PAGE_ROOTS:
        return ai_frameworks_root.joinpath(*_C_PAGE_ROOTS[page_id])
    if volume == "F" and page_id in _F_PAGE_ROOTS:
        return ai_frameworks_root.joinpath(*_F_PAGE_ROOTS[page_id])
    return ai_frameworks_root / "19_torch_compile_end_to_end"


class DemoManifestContractTest(unittest.TestCase):
    def test_manifest_maps_every_course_page_to_a_real_case(self) -> None:
        labs_root = Path(__file__).resolve().parent
        manifest = json.loads(
            (labs_root / "demo_manifest.json").read_text(encoding="utf-8")
        )
        entries = manifest["pages"]
        expected_ids = {
            *(f"b{index:02d}" for index in range(1, 11)),
            *(f"c{index:02d}" for index in range(1, 22)),
            *({f"d{index:02d}" for index in range(1, 8)} - {"d05"}),
            *(f"e{index:02d}" for index in range(1, 10)),
            *(f"f{index:02d}" for index in range(1, 9)),
        }
        self.assertEqual(manifest["schema_version"], "torch-compile-demo-manifest/v1")
        # Volume A (a01-a05) was dropped from the course + manifest by kb-reorg
        # P4 Task 3 (2026-07-30): the 5 recap pages were deleted, their unique
        # content migrated verbatim into the eager_runtime/compile_stack pages
        # that cover their mechanisms. demo_a_execution_model.py and its two
        # VolumeABContractTest cases above still exercise the CPU-mechanism
        # scripts directly (script-level, not page-level), so they're unaffected.
        # d05 (wrapper_execution_memory_allocation_and_reuse_analysis.md) was
        # dropped from the manifest by kb-reorg P4 Task 8 group D (2026-07-30):
        # the page's unique content was merged verbatim into C19
        # (buffer_liveness_memory_planning_and_reuse_analysis.md) Sec18, which
        # already carries c19's own manifest entry with that filename -- the
        # manifest requires unique page values, so d05 can't point at the same
        # file. The demo_d_artifact_runtime.py wrapper_memory_reuse case still
        # runs; C19 Sec18 links to it directly as a pointer instead of a
        # manifest-checked "## 配套 Demo" section (C pages are exempt from that
        # check below, same as d05's sibling C-volume pages always were).
        self.assertEqual(len(entries), 54)
        self.assertEqual({entry["page_id"] for entry in entries}, expected_ids)
        self.assertEqual(len({entry["page"] for entry in entries}), 54)

        modules: dict[str, object] = {}
        for entry in entries:
            page_path = (
                _page_root(labs_root, entry["volume"], entry.get("page_id"))
                / entry["page"]
            )
            self.assertTrue(page_path.is_file(), page_path)
            module_name = Path(entry["script"]).stem
            module = modules.setdefault(
                module_name, importlib.import_module(module_name)
            )
            case_ids = {case.case_id for case in getattr(module, "CASES")}
            self.assertIn(entry["case"], case_ids)

    def test_non_c_pages_contain_executable_demo_backlinks(self) -> None:
        labs_root = Path(__file__).resolve().parent
        entries = json.loads(
            (labs_root / "demo_manifest.json").read_text(encoding="utf-8")
        )["pages"]
        for entry in entries:
            if entry["volume"] == "C":
                continue
            text = (
                _page_root(labs_root, entry["volume"], entry.get("page_id"))
                / entry["page"]
            ).read_text(encoding="utf-8")
            self.assertIn("## 配套 Demo", text, entry["page"])
            self.assertIn(entry["script"], text, entry["page"])
            self.assertIn(f"--case {entry['case']}", text, entry["page"])
            self.assertLess(
                text.index("## 配套 Demo"),
                text.index("## Related Pages"),
                entry["page"],
            )

    def test_all_volume_entries_support_json_listing_in_subprocesses(self) -> None:
        labs_root = Path(__file__).resolve().parent
        for volume in "abcdef":
            script = next(labs_root.glob(f"demo_{volume}_*.py"))
            completed = subprocess.run(
                [sys.executable, "-B", str(script), "--list", "--json"],
                cwd=labs_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            listing = json.loads(completed.stdout)
            self.assertEqual(listing["volume"], volume.upper())
            self.assertGreater(len(listing["cases"]), 0)

    def test_cli_contract_errors_use_exit_code_four(self) -> None:
        labs_root = Path(__file__).resolve().parent
        script = labs_root / "demo_a_execution_model.py"
        completed = subprocess.run(
            [sys.executable, "-B", str(script), "--case", "not_a_case"],
            cwd=labs_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(completed.returncode, 4)
        self.assertIn("unknown case ids", completed.stderr)


class CourseMarkdownContractTest(unittest.TestCase):
    @staticmethod
    def _course_pages() -> list[Path]:
        ai_frameworks_root = (
            Path(__file__).resolve().parent.parent.parent
            / "wiki"
            / "02_engineering"
            / "01_ai_frameworks"
        )
        course_root = ai_frameworks_root / "19_torch_compile_end_to_end"
        # Volume E physically moved to 07_debugging (kb-reorg P4 Task 4); its
        # markdown quality gates below still apply, just from the new home.
        debugging_root = ai_frameworks_root / "02_compile_stack" / "07_debugging"
        # Volume B physically moved to 01_dynamo (kb-reorg P4 Task 5); same.
        dynamo_root = ai_frameworks_root / "02_compile_stack" / "01_dynamo"
        # 12 of volume C's 21 pages physically moved to the new
        # 03_graph_ir_and_passes directory (kb-reorg P4 Task 7); same gates
        # apply from the new home. The remaining 9 pages stay under
        # course_root until Task 8 scatters them further.
        graph_ir_root = (
            ai_frameworks_root / "02_compile_stack" / "03_graph_ir_and_passes"
        )
        return (
            sorted(course_root.glob("*.md"))
            + sorted(debugging_root.glob("*.md"))
            + sorted(dynamo_root.glob("*.md"))
            + sorted(graph_ir_root.glob("*.md"))
        )

    def test_list_markers_render_as_commonmark_lists(self) -> None:
        malformed: list[str] = []
        for path in self._course_pages():
            fence: str | None = None
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.lstrip()
                marker = stripped[:3]
                if marker in {"```", "~~~"}:
                    if fence is None:
                        fence = marker
                    elif marker == fence:
                        fence = None
                    continue
                if fence is None and re.match(r"^-[^ \t-]", line):
                    malformed.append(f"{path.name}:{line_number}")

        self.assertEqual(malformed, [])

    def test_ordered_list_markers_render_as_commonmark_lists(self) -> None:
        malformed: list[str] = []
        for path in self._course_pages():
            fence: str | None = None
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.lstrip()
                marker = stripped[:3]
                if marker in {"```", "~~~"}:
                    if fence is None:
                        fence = marker
                    elif marker == fence:
                        fence = None
                    continue
                if fence is None and re.match(r"^\s*\d+\.(?![ \t])", line):
                    malformed.append(f"{path.name}:{line_number}")

        self.assertEqual(malformed, [])

    def test_mermaid_pipe_labels_do_not_embed_quotes(self) -> None:
        invalid: list[str] = []
        quoted_pipe_label = re.compile(r"\|[\"'][^|]*[\"']\|")
        quoted_inline_label = re.compile(
            r"(?:--|==|-\.)(?:\s+)[\"'][^\"']+[\"'](?:\s+)(?:-->|==>|\.->)"
        )
        for path in self._course_pages():
            in_mermaid = False
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.strip() == "```mermaid":
                    in_mermaid = True
                    continue
                if in_mermaid and line.strip() == "```":
                    in_mermaid = False
                    continue
                if in_mermaid and (
                    quoted_pipe_label.search(line)
                    or quoted_inline_label.search(line)
                ):
                    invalid.append(f"{path.name}:{line_number}")

        self.assertEqual(invalid, [])

    def test_call_chain_pages_have_source_walkthroughs(self) -> None:
        ai_frameworks_root = (
            Path(__file__).resolve().parent.parent.parent
            / "wiki"
            / "02_engineering"
            / "01_ai_frameworks"
        )
        # Volume B physically moved to 01_dynamo (kb-reorg P4 Task 5); the two
        # b04/b07 call-chain pages below now live there.
        dynamo_root = ai_frameworks_root / "02_compile_stack" / "01_dynamo"
        # Volume D physically moved out of 19_torch_compile_end_to_end one page
        # at a time (kb-reorg P4 Task 6), scattering across four directories
        # (unlike B/E which each moved wholesale into one directory).
        inductor_root = ai_frameworks_root / "02_compile_stack" / "04_inductor"
        aot_autograd_root = ai_frameworks_root / "02_compile_stack" / "02_aot_autograd"
        cuda_root = ai_frameworks_root / "03_runtime_graphs" / "cuda"
        # Volume F physically scatters out of 19_torch_compile_end_to_end
        # (kb-reorg P4 Task 9); f01's call-chain page now lives here.
        autograd_engine_root = (
            ai_frameworks_root / "01_eager_runtime" / "05_autograd_engine"
        )
        target_pages = [
            (dynamo_root, "instruction_translator_and_bytecode_state_machine_analysis.md"),
            (dynamo_root, "guards_cache_lookup_and_recompilation_analysis.md"),
            (inductor_root, "inductor_compile_fx_orchestration_analysis.md"),
            (aot_autograd_root, "aot_runtime_wrappers_and_lazy_backward_compile_analysis.md"),
            (cuda_root, "cudagraph_trees_warmup_record_and_replay_analysis.md"),
            (autograd_engine_root, "compiled_autograd_analysis.md"),
        ]
        locator = re.compile(
            r"(?:torch|test|tests|functorch|aten|c10|tools)/"
            r"[^`\s:]+:(?:L)?\d+(?:-(?:L)?\d+)?"
        )
        missing: list[str] = []
        shallow: list[str] = []
        for root, filename in target_pages:
            lines = (root / filename).read_text(encoding="utf-8").splitlines()
            start = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if line.startswith("## ") and "源码跟读" in line
                ),
                None,
            )
            if start is None:
                missing.append(filename)
                continue
            end = next(
                (
                    index
                    for index in range(start + 1, len(lines))
                    if lines[index].startswith("## ")
                ),
                len(lines),
            )
            source_count = len(locator.findall("\n".join(lines[start:end])))
            if source_count < 5:
                shallow.append(f"{filename}:{source_count}")

        self.assertEqual(missing, [])
        self.assertEqual(shallow, [])

    def test_b07_cache_miss_formula_contains_all_addends(self) -> None:
        # Volume B physically moved to 01_dynamo (kb-reorg P4 Task 5).
        dynamo_root = (
            Path(__file__).resolve().parent.parent.parent
            / "wiki"
            / "02_engineering"
            / "01_ai_frameworks"
            / "02_compile_stack"
            / "01_dynamo"
        )
        page = dynamo_root / "guards_cache_lookup_and_recompilation_analysis.md"
        text = page.read_text(encoding="utf-8")
        self.assertIn(
            "O\\left(\\sum_{i=1}^{C} Q_i\\right)\n"
            "+ T_{\\text{capture}}\n"
            "+ T_{\\text{backend}}",
            text,
        )

    def test_course_source_locators_are_not_region_sized(self) -> None:
        broad: list[str] = []
        locator = re.compile(
            r"(?:torch|test|tests|functorch|aten|c10|tools)/"
            r"[^`\s:]+:(?:L)?(\d+)-(?:L)?(\d+)"
        )
        for path in self._course_pages():
            for match in locator.finditer(path.read_text(encoding="utf-8")):
                start, end = map(int, match.groups())
                if end - start + 1 > 100:
                    broad.append(f"{path.name}:{match.group(0)}")

        self.assertEqual(broad, [])


if __name__ == "__main__":
    unittest.main()
