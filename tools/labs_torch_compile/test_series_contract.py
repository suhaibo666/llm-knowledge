import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


LAB_ROOT = Path(__file__).resolve().parent


def run_lab(script_name: str, *args: str) -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, str(LAB_ROOT / script_name), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def assert_manifest_hashes(
    test_case: unittest.TestCase,
    output_dir: Path,
    manifest: dict[str, object],
) -> None:
    artifacts = manifest["artifacts"]
    test_case.assertIsInstance(artifacts, list)
    for artifact in artifacts:
        test_case.assertIsInstance(artifact, dict)
        relative_path = artifact["path"]
        expected_sha256 = artifact["sha256"]
        artifact_path = output_dir / relative_path
        test_case.assertTrue(artifact_path.is_file(), relative_path)
        test_case.assertEqual(
            expected_sha256,
            hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            relative_path,
        )


class CourseRuntimeObservationContractTest(unittest.TestCase):
    def load_runner(self):
        module_path = LAB_ROOT / "course_runtime_observations.py"
        self.assertTrue(
            module_path.is_file(),
            "course_runtime_observations.py must aggregate the course Labs",
        )
        spec = importlib.util.spec_from_file_location(
            "course_runtime_observations_contract",
            module_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_declares_each_course_script_once_and_excludes_tests_and_native(self) -> None:
        runner = self.load_runner()
        expected = [
            "part1_graph_taxonomy.py",
            "part1_fx_core.py",
            "part1_values_signatures.py",
            "part1_symbolic_shapes.py",
            "part1_effects_alias.py",
            "part1_structured_hop.py",
            "part2_capture_frontends.py",
            "part2_normalization.py",
            "part2_aot_graphs.py",
            "part2_aot_recompute_analysis.py",
            "part2_activation_peak.py",
            "part2_continuous_aot_inductor.py",
            "part3_passes.py",
            "part3_pattern.py",
            "part3_legality.py",
            "part3_end_to_end_pass.py",
            "part3_real_stage_hooks.py",
            "part4_inductor.py",
            "part4_ir_scheduler_analysis.py",
            "part4_artifact_bundle.py",
            "series_artifact_bundle.py",
        ]
        observed = [spec.script for spec in runner.COURSE_SCRIPT_SPECS]
        self.assertEqual(expected, observed)
        runner.validate_script_specs(runner.COURSE_SCRIPT_SPECS, LAB_ROOT)

        with self.assertRaisesRegex(ValueError, "duplicate script declaration"):
            runner.validate_script_specs(
                (*runner.COURSE_SCRIPT_SPECS, runner.COURSE_SCRIPT_SPECS[0]),
                LAB_ROOT,
            )

    def test_records_command_hash_raw_streams_key_values_and_environment(self) -> None:
        runner = self.load_runner()
        self.assertTrue(
            hasattr(runner, "run_and_write"),
            "runner must execute scripts and atomically write observations",
        )
        with tempfile.TemporaryDirectory(
            prefix="course_runtime_observation_test_"
        ) as temp_dir:
            root = Path(temp_dir)
            script = root / "emit.py"
            script.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "assert sys.argv[1] == '--output-dir'",
                        "Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)",
                        "print('alpha=one')",
                        "print('count=2')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "artifacts" / "observations.json"
            output.parent.mkdir()
            output.write_text("stale", encoding="utf-8")

            exit_code = runner.run_and_write(
                (runner.ScriptSpec("emit.py", "emit"),),
                lab_root=root,
                working_directory=root,
                output_path=output,
            )

            self.assertEqual(0, exit_code)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("course-runtime-observations/v1", payload["schema_version"])
            self.assertEqual("passed", payload["status"])
            self.assertTrue(payload["all_scripts_passed"])
            self.assertEqual(1, payload["script_count"])
            self.assertEqual(1, payload["completed_script_count"])
            self.assertEqual(
                {
                    "python_version",
                    "python_executable",
                    "platform",
                    "torch_version",
                    "torch_git_version",
                    "cuda_available",
                },
                set(payload["runtime_environment"]),
            )

            observation = payload["observations"][0]
            self.assertEqual("emit.py", observation["script"])
            self.assertEqual(
                hashlib.sha256(script.read_bytes()).hexdigest(),
                observation["script_sha256"],
            )
            self.assertEqual(
                [sys.executable, "-B", str(script.resolve())],
                observation["command"][:3],
            )
            self.assertEqual("--output-dir", observation["command"][3])
            self.assertEqual(0, observation["exit_code"])
            self.assertEqual("alpha=one\ncount=2\n", observation["stdout"])
            self.assertEqual("", observation["stderr"])
            self.assertEqual(
                {"alpha": "one", "count": "2"},
                observation["stdout_key_values"],
            )
            self.assertEqual(
                [],
                [
                    path.name
                    for path in output.parent.iterdir()
                    if path.name != output.name
                ],
            )

    def test_child_failure_is_recorded_and_propagated_after_remaining_scripts_run(
        self,
    ) -> None:
        runner = self.load_runner()
        with tempfile.TemporaryDirectory(
            prefix="course_runtime_failure_test_"
        ) as temp_dir:
            root = Path(temp_dir)
            (root / "fails.py").write_text(
                "\n".join(
                    [
                        "import sys",
                        "print('started=True')",
                        "print('failure detail', file=sys.stderr)",
                        "raise SystemExit(7)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "continues.py").write_text(
                "print('continued=True')\n",
                encoding="utf-8",
            )
            output = root / "observations.json"

            exit_code = runner.run_and_write(
                (
                    runner.ScriptSpec("fails.py"),
                    runner.ScriptSpec("continues.py"),
                ),
                lab_root=root,
                working_directory=root,
                output_path=output,
            )

            self.assertEqual(1, exit_code)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("failed", payload["status"])
            self.assertFalse(payload["all_scripts_passed"])
            self.assertIn("failed_script_count", payload)
            self.assertEqual(1, payload["failed_script_count"])
            self.assertEqual(["fails.py"], payload["failed_scripts"])
            self.assertEqual(2, payload["completed_script_count"])
            self.assertEqual(
                [7, 0],
                [
                    observation["exit_code"]
                    for observation in payload["observations"]
                ],
            )
            self.assertEqual(
                "failure detail\n",
                payload["observations"][0]["stderr"],
            )
            self.assertEqual(
                {"continued": "True"},
                payload["observations"][1]["stdout_key_values"],
            )

    def test_cli_runs_declared_scripts_from_knowledge_root_into_one_artifact(
        self,
    ) -> None:
        runner = self.load_runner()
        self.assertTrue(
            hasattr(runner, "main"),
            "runner must expose an executable command-line entry point",
        )
        calls = []

        def record_call(specs, **kwargs):
            calls.append((specs, kwargs))
            return 0

        runner.run_and_write = record_call
        custom_output = LAB_ROOT / "artifacts" / "custom-observations.json"

        exit_code = runner.main(["--output", str(custom_output)])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(calls))
        specs, kwargs = calls[0]
        self.assertEqual(runner.COURSE_SCRIPT_SPECS, specs)
        self.assertEqual(LAB_ROOT, kwargs["lab_root"])
        self.assertEqual(LAB_ROOT.parents[4], kwargs["working_directory"])
        self.assertEqual(custom_output, kwargs["output_path"])
        self.assertEqual(
            LAB_ROOT / "artifacts" / "course_runtime_observations.json",
            runner.DEFAULT_OUTPUT_PATH,
        )


class EndToEndPassContractTest(unittest.TestCase):
    def test_add_matmul_rewrite_preserves_required_semantics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="graph_series_part3_test_") as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            observed = run_lab(
                "part3_end_to_end_pass.py", "--output-dir", str(output_dir)
            )
            expected = {
                "legal_rewrite_applied": "True",
                "legal_has_addmm": "True",
                "illegal_broadcast_rewrite_applied": "False",
                "illegal_graph_unchanged": "True",
                "failure_atomicity_matches": "True",
                "forward_matches": "True",
                "gradient_matches": "True",
                "gradcheck_matches": "True",
                "shape_matches": "True",
                "alias_contract_matches": "True",
                "mutation_contract_matches": "True",
                "second_run_modified": "False",
                "second_run_code_unchanged": "True",
            }
            self.assertEqual(expected, {key: observed.get(key) for key in expected})

            required = {
                "environment.json",
                "manifest.json",
                "results.json",
                "legal_before.py",
                "legal_after.py",
                "illegal_before.py",
                "illegal_after.py",
            }
            missing = sorted(
                relative
                for relative in required
                if not (output_dir / relative).is_file()
                or (output_dir / relative).stat().st_size == 0
            )
            self.assertEqual([], missing)


class EffectAndFunctionalizationContractTest(unittest.TestCase):
    def test_dce_alias_effect_order_and_functionalization_boundaries(self) -> None:
        effects = run_lab("part1_effects_alias.py")
        expected_effects = {
            "pure_dead_removed": "True",
            "dead_chain_removed": "True",
            "nested_child_dead_removed": "True",
            "impure_copy_retained": "True",
            "alias_observes_mutation": "True",
            "both_effect_orders_lint": "True",
            "effect_reorder_changes_result": "True",
        }
        self.assertEqual(
            expected_effects,
            {key: effects.get(key) for key in expected_effects},
        )

        normalization = run_lab("part2_normalization.py")
        expected_normalization = {
            "original_has_inplace": "True",
            "functional_has_outplace_add": "True",
            "functional_output_matches": "True",
            "functional_input_semantics_match": "True",
        }
        self.assertEqual(
            expected_normalization,
            {key: normalization.get(key) for key in expected_normalization},
        )


class CaptureFrontendContractTest(unittest.TestCase):
    def test_four_frontends_guards_meta_and_graph_break(self) -> None:
        observed = run_lab("part2_capture_frontends.py")
        expected = {
            "symbolic_has_call_module": "True",
            "make_fx_has_call_module": "False",
            "dynamo_backend_graphs": "1",
            "dynamo_guards_recorded": "True",
            "dynamo_example_value_meta_recorded": "True",
            "explicit_graph_break_backend_graphs": "2",
            "export_input_kinds": "PARAMETER,PARAMETER,USER_INPUT",
        }
        self.assertEqual(expected, {key: observed.get(key) for key in expected})


class PatternMatcherContractTest(unittest.TestCase):
    def test_capture_sharing_multi_output_and_candidate_apply(self) -> None:
        with tempfile.TemporaryDirectory(prefix="graph_series_pattern_test_") as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            observed = run_lab(
                "part3_pattern.py", "--output-dir", str(output_dir)
            )
            expected = {
                "unary_pattern_matched": "True",
                "unary_arg_captured": "True",
                "shared_pattern_matched": "True",
                "failed_sharing_pattern": "True",
                "kwargs_pattern_matched": "True",
                "kwargs_constant_captured": "True",
                "multi_output_pattern_matched": "True",
                "pattern_matcher_pass_apply_count": "1",
                "pattern_matcher_handler_calls": "1",
                "graph_pattern_value_matches": "True",
                "graph_pattern_old_nodes_erased": "True",
                "graph_pattern_second_apply_count": "0",
                "replacement_apply_count": "1",
                "replacement_value_matches": "True",
                "replacement_has_mul": "True",
                "replacement_has_no_add": "True",
                "replacement_second_apply_count": "0",
                "replacement_extra_check_rejected": "True",
                "lowering_apply_count": "1",
                "lowering_handler_marked": "True",
                "lowering_handler_deferred_until_graph_lowering": "True",
                "lowering_pattern_reached_inductor_ir": "True",
                "lowering_original_add_erased": "True",
                "lowering_native_kernel_executed": "False",
            }
            self.assertEqual(
                expected, {key: observed.get(key) for key in expected}
            )

            required = {
                "summary.json",
                "graph_pattern_after.txt",
                "replacement_after.txt",
                "lowering_pattern_after.txt",
                "lowering_ir.txt",
            }
            missing = sorted(
                relative
                for relative in required
                if not (output_dir / relative).is_file()
                or (output_dir / relative).stat().st_size == 0
            )
            self.assertEqual([], missing)


class EditingAndPassManagerContractTest(unittest.TestCase):
    def test_copy_topology_and_bounded_oscillation(self) -> None:
        observed = run_lab("part3_passes.py")
        expected = {
            "node_copy_value_matches": "True",
            "graph_copy_value_matches": "True",
            "lint_failed_before_sort": "True",
            "topology_repaired_value": "8.0",
            "pass_manager_steps_1_count": "1",
            "pass_manager_steps_4_count": "3",
            "stage_target_spelling_differs": "True",
            "stage_correct_contract_rewrites": "True",
            "stage_wrong_contract_rejected": "True",
            "stage_rewrite_idempotent": "True",
            "stage_outputs_match": "True",
            "actual_torch_compile_stage_hook_executed": "False",
            "oscillation_bounded_at_four": "True",
            "oscillation_final_target_is_add": "True",
        }
        self.assertEqual(expected, {key: observed.get(key) for key in expected})


class PartTwoAotRecomputeContractTest(unittest.TestCase):
    def test_saved_value_and_recompute_budget_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="graph_series_aot_test_") as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            observed = run_lab(
                "part2_aot_recompute_analysis.py",
                "--output-dir",
                str(output_dir),
            )
            expected = {
                "budget_high_saved_bytes_gt_low": "True",
                "budget_low_recompute_targets_observed": "True",
                "gradient_matches": "True",
                "cross_graph_node_refs": "0",
                "physical_allocator_peak_measured": "False",
            }
            self.assertEqual(expected, {key: observed.get(key) for key in expected})

            required = {
                "environment.json",
                "manifest.json",
                "budget_high_forward.py",
                "budget_high_backward.py",
                "budget_low_forward.py",
                "budget_low_backward.py",
                "partition_comparison.json",
            }
            missing = sorted(
                relative
                for relative in required
                if not (output_dir / relative).is_file()
                or (output_dir / relative).stat().st_size == 0
            )
            self.assertEqual([], missing)


class PartTwoActivationPeakContractTest(unittest.TestCase):
    def test_saved_tensor_metrics_distinguish_logical_views_from_backing_storage(
        self,
    ) -> None:
        module_path = LAB_ROOT / "part2_activation_peak.py"
        spec = importlib.util.spec_from_file_location(
            "part2_activation_peak_contract",
            module_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        base = torch.arange(8, dtype=torch.float32)
        first = base[:4]
        second = base[4:]
        tracker = module.LogicalSavedTensorPeak()
        packed_first = tracker.pack(first)
        packed_second = tracker.pack(second)

        self.assertIsNot(packed_first[0], first)
        self.assertIsNot(packed_second[0], second)
        self.assertEqual(32, tracker.live_logical_tensor_bytes)
        self.assertEqual(32, tracker.live_unique_backing_storage_bytes)
        self.assertEqual(1, len(tracker.active_storage_refcounts))

        tracker.unpack(packed_first)
        tracker.unpack(packed_second)
        self.assertEqual(0, tracker.live_logical_tensor_bytes)
        self.assertEqual(0, tracker.live_unique_backing_storage_bytes)

    def test_saved_tensor_hooks_measure_logical_peak_without_allocator_claim(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="graph_series_activation_peak_test_"
        ) as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            self.assertTrue(
                (LAB_ROOT / "part2_activation_peak.py").is_file(),
                "part2_activation_peak.py must implement runtime peak measurement",
            )
            observed = run_lab(
                "part2_activation_peak.py",
                "--output-dir",
                str(output_dir),
            )
            expected = {
                "saved_tensors_hooks_executed": "True",
                "budget_high_logical_tensor_peak_gt_low": "True",
                "budget_high_unique_backing_storage_peak_gt_low": "True",
                "budget_high_pack_unpack_balanced": "True",
                "budget_low_pack_unpack_balanced": "True",
                "budget_high_logical_tensor_live_bytes_returned_to_zero": "True",
                "budget_low_logical_tensor_live_bytes_returned_to_zero": "True",
                "budget_high_unique_backing_storage_live_bytes_returned_to_zero": "True",
                "budget_low_unique_backing_storage_live_bytes_returned_to_zero": "True",
                "gradient_matches": "True",
                "physical_allocator_peak_status": "blocked_no_cuda",
                "physical_allocator_peak_measured": "False",
                "logical_peak_is_physical_allocator_peak": "False",
            }
            self.assertEqual(
                expected,
                {key: observed.get(key) for key in expected},
            )

            required = {
                "environment.json",
                "results.json",
                "manifest.json",
            }
            self.assertEqual(
                [],
                sorted(
                    relative
                    for relative in required
                    if not (output_dir / relative).is_file()
                    or (output_dir / relative).stat().st_size == 0
                ),
            )
            results = json.loads(
                (output_dir / "results.json").read_text(encoding="utf-8")
            )
            high = results["budget_high"]
            low = results["budget_low"]
            self.assertEqual(768, high["peak_logical_tensor_bytes"])
            self.assertEqual(512, low["peak_logical_tensor_bytes"])
            self.assertEqual(768, high["peak_unique_backing_storage_bytes"])
            self.assertEqual(512, low["peak_unique_backing_storage_bytes"])
            self.assertEqual(high["pack_count"], high["unpack_count"])
            self.assertEqual(low["pack_count"], low["unpack_count"])
            self.assertEqual(
                0,
                high["final_live_logical_tensor_bytes"],
            )
            self.assertEqual(
                0,
                low["final_live_logical_tensor_bytes"],
            )
            self.assertEqual(
                0,
                high["final_live_unique_backing_storage_bytes"],
            )
            self.assertEqual(
                0,
                low["final_live_unique_backing_storage_bytes"],
            )
            self.assertEqual(
                "sum_of_active_saved_tensor_numel_times_element_size",
                results["measurement_kinds"]["logical_tensor_bytes"],
            )
            self.assertEqual(
                "deduplicated_active_untyped_storage_nbytes",
                results["measurement_kinds"]["unique_backing_storage_bytes"],
            )
            self.assertIsNone(results["physical_allocator_peak_bytes"])
            self.assertEqual(
                "blocked_no_cuda",
                results["physical_allocator_peak_status"],
            )
            self.assertFalse(results["logical_peak_is_physical_allocator_peak"])

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            assert_manifest_hashes(self, output_dir, manifest)


class PartTwoContinuousAotInductorContractTest(unittest.TestCase):
    def test_aot_forward_directly_reaches_real_inductor_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="graph_series_continuous_aot_test_"
        ) as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            self.assertTrue(
                (LAB_ROOT / "part2_continuous_aot_inductor.py").is_file(),
                "part2_continuous_aot_inductor.py must implement the continuous path",
            )
            observed = run_lab(
                "part2_continuous_aot_inductor.py",
                "--output-dir",
                str(output_dir),
            )
            expected = {
                "continuous_aot_forward_to_inductor": "True",
                "partition_forward_tokens_preserved": "True",
                "partition_backward_tokens_preserved": "True",
                "forward_graph_identity_preserved": "True",
                "forward_owner_identity_preserved": "True",
                "backward_partition_to_callback_identity_transition": "True",
                "backward_callback_to_lowering_identity_preserved": "True",
                "forward_callback_to_scheduler_module_graph_preserved": "True",
                "backward_callback_to_scheduler_module_graph_preserved": "True",
                "scheduler_owner_transition_recorded": "True",
                "scheduler_origin_tokens_recorded": "True",
                "scheduler_dependency_construction_recorded": "True",
                "compiler_callback_recorded": "True",
                "graph_lowering_recorded": "True",
                "scheduler_recorded": "True",
                "forward_matches": "True",
                "gradient_matches": "True",
                "native_kernel_executed": "False",
                "evidence_scope": "extern_matmul_only",
            }
            self.assertEqual(
                expected,
                {key: observed.get(key) for key in expected},
            )

            required = {
                "environment.json",
                "partition_forward.py",
                "partition_backward.py",
                "compiler_forward.py",
                "compiler_backward.py",
                "continuity.json",
                "manifest.json",
            }
            self.assertEqual(
                [],
                sorted(
                    relative
                    for relative in required
                    if not (output_dir / relative).is_file()
                    or (output_dir / relative).stat().st_size == 0
                ),
            )
            continuity = json.loads(
                (output_dir / "continuity.json").read_text(encoding="utf-8")
            )
            run_id = continuity["run_id"]
            self.assertTrue(run_id)
            self.assertTrue(
                all(event["run_id"] == run_id for event in continuity["events"])
            )
            forward = continuity["forward"]
            self.assertEqual(
                forward["partition_graph_id"],
                forward["compiler_callback_graph_id"],
            )
            self.assertEqual(
                forward["partition_module_id"],
                forward["compiler_callback_module_id"],
            )
            self.assertEqual(
                forward["partition_module_id"],
                forward["partition_owner_id"],
            )
            self.assertEqual(
                forward["compiler_callback_module_id"],
                forward["compiler_callback_owner_id"],
            )
            self.assertEqual(
                forward["compiler_callback_owner_id"],
                forward["graph_lowering_owner_id"],
            )
            self.assertEqual(
                forward["compiler_callback_module_id"],
                forward["scheduler_module_id"],
            )
            self.assertEqual(
                forward["compiler_callback_graph_id"],
                forward["scheduler_graph_id"],
            )
            self.assertNotEqual(
                forward["compiler_callback_owner_id"],
                forward["scheduler_owner_id"],
            )
            self.assertEqual(
                forward["scheduler_owner_id"],
                forward["scheduler_graph_lowering_copy_module_id"],
            )
            self.assertTrue(forward["scheduler_owner_is_graph_lowering_copy"])
            self.assertEqual(
                forward["partition_tokens"],
                forward["compiler_callback_tokens"],
            )
            self.assertEqual(
                forward["compiler_callback_tokens"],
                forward["graph_lowering_tokens"],
            )
            self.assertEqual(
                forward["graph_lowering_tokens"],
                forward["scheduler_module_tokens"],
            )
            self.assertGreater(len(forward["scheduler_origin_tokens"]), 0)
            self.assertTrue(
                set(forward["scheduler_origin_tokens"]).issubset(
                    forward["scheduler_module_tokens"]
                )
            )
            self.assertGreater(forward["scheduler_read_dependency_count"], 0)
            self.assertGreater(forward["scheduler_write_dependency_count"], 0)
            self.assertGreater(len(forward["partition_tokens"]), 0)
            backward = continuity["backward"]
            self.assertNotEqual(
                backward["partition_module_id"],
                backward["compiler_callback_module_id"],
            )
            self.assertNotEqual(
                backward["partition_graph_id"],
                backward["compiler_callback_graph_id"],
            )
            self.assertNotEqual(
                backward["partition_owner_id"],
                backward["compiler_callback_owner_id"],
            )
            self.assertEqual(
                backward["compiler_callback_module_id"],
                backward["graph_lowering_module_id"],
            )
            self.assertEqual(
                backward["compiler_callback_graph_id"],
                backward["graph_lowering_graph_id"],
            )
            self.assertEqual(
                backward["compiler_callback_owner_id"],
                backward["graph_lowering_owner_id"],
            )
            self.assertEqual(
                backward["compiler_callback_module_id"],
                backward["scheduler_module_id"],
            )
            self.assertEqual(
                backward["compiler_callback_graph_id"],
                backward["scheduler_graph_id"],
            )
            self.assertNotEqual(
                backward["compiler_callback_owner_id"],
                backward["scheduler_owner_id"],
            )
            self.assertEqual(
                backward["scheduler_owner_id"],
                backward["scheduler_graph_lowering_copy_module_id"],
            )
            self.assertTrue(backward["scheduler_owner_is_graph_lowering_copy"])
            self.assertEqual(
                backward["partition_tokens"],
                backward["compiler_callback_tokens"],
            )
            self.assertEqual(
                backward["compiler_callback_tokens"],
                backward["graph_lowering_tokens"],
            )
            self.assertEqual(
                backward["graph_lowering_tokens"],
                backward["scheduler_module_tokens"],
            )
            self.assertGreater(len(backward["scheduler_origin_tokens"]), 0)
            self.assertTrue(
                set(backward["scheduler_origin_tokens"]).issubset(
                    backward["scheduler_module_tokens"]
                )
            )
            self.assertGreater(backward["scheduler_read_dependency_count"], 0)
            self.assertGreater(backward["scheduler_write_dependency_count"], 0)
            self.assertFalse(continuity["native_kernel_executed"])
            self.assertFalse(continuity["mock_compiler_used"])

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            assert_manifest_hashes(self, output_dir, manifest)


class PartThreeRealStageHookContractTest(unittest.TestCase):
    def test_real_compile_hooks_rewrite_only_at_the_matching_stage(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="graph_series_real_stage_hook_test_"
        ) as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            self.assertTrue(
                (LAB_ROOT / "part3_real_stage_hooks.py").is_file(),
                "part3_real_stage_hooks.py must implement real compile hooks",
            )
            observed = run_lab(
                "part3_real_stage_hooks.py",
                "--output-dir",
                str(output_dir),
            )
            expected = {
                "actual_torch_compile_stage_hook_executed": "True",
                "pre_correct_stage_hits": "1",
                "pre_wrong_stage_hits": "0",
                "post_correct_stage_hits": "1",
                "post_wrong_stage_hits": "0",
                "pre_forward_matches": "True",
                "pre_gradient_matches": "True",
                "post_forward_matches": "True",
                "post_gradient_matches": "True",
                "pass_second_run_zero_rewrites": "True",
                "pass_second_run_graph_unchanged": "True",
                "second_call_no_recompile": "True",
                "second_call_no_additional_rewrite": "True",
                "config_restored": "True",
                "native_kernel_executed": "False",
            }
            self.assertEqual(
                expected,
                {key: observed.get(key) for key in expected},
            )

            required = {
                "environment.json",
                "results.json",
                "pre_stage_graph.py",
                "post_stage_graph.py",
                "manifest.json",
            }
            self.assertEqual(
                [],
                sorted(
                    relative
                    for relative in required
                    if not (output_dir / relative).is_file()
                    or (output_dir / relative).stat().st_size == 0
                ),
            )
            results = json.loads(
                (output_dir / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                ["operator.matmul", "operator.add"],
                results["pre"]["matched_targets"],
            )
            self.assertEqual(
                ["aten.mm.default", "aten.add.Tensor"],
                results["post"]["matched_targets"],
            )
            self.assertEqual(1, results["pre"]["compile_count"])
            self.assertEqual(1, results["post"]["compile_count"])
            self.assertEqual(
                0,
                results["pre"]["pass_second_run_rewrite_count"],
            )
            self.assertEqual(
                0,
                results["post"]["pass_second_run_rewrite_count"],
            )
            self.assertTrue(results["pre"]["pass_second_run_graph_unchanged"])
            self.assertTrue(results["post"]["pass_second_run_graph_unchanged"])
            self.assertTrue(results["config_restored"])
            self.assertFalse(results["mock_compiler_used"])

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            assert_manifest_hashes(self, output_dir, manifest)


class PartFourArtifactContractTest(unittest.TestCase):
    def test_part_four_emits_inspectable_artifacts_without_claiming_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="graph_series_test_") as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            observed = run_lab(
                "part4_artifact_bundle.py", "--output-dir", str(output_dir)
            )
            expected = {
                "external_matmul_execution": "True",
                "fallback_eigvals_execution": "True",
                "codegen_only_status": "generated_not_executed",
                "real_pointwise_compile_status": "blocked_missing_msvc_cl",
                "fusion_enabled_has_fused_scheduler": "True",
                "fusion_limited_has_fused_scheduler": "False",
                "fusion_codegen_structure_changed": "True",
                "scheduler_to_fx_provenance_observed": "True",
                "kernel_to_fx_provenance_observed": "True",
                "fx_to_python_source_observed": "True",
                "scheduler_kernel_source_chain_observed": "True",
                "custom_lowering_reached_ir": "True",
                "fallback_trace_captured": "True",
                "fallback_wrapper_observed": "True",
                "triton_autotune_tested": "False",
            }
            self.assertEqual(expected, {key: observed.get(key) for key in expected})

            required = {
                "environment.json",
                "summary.json",
                "fusion_enabled/fx_graph_readable.py",
                "fusion_enabled/fx_graph_transformed.py",
                "fusion_enabled/ir_pre_fusion.txt",
                "fusion_enabled/ir_post_fusion.txt",
                "fusion_enabled/inductor_provenance_tracking_node_mappings.json",
                "fusion_enabled/output_code.py",
                "fusion_enabled/captured_cpp_kernel.cpp",
                "fusion_enabled/provenance_chain.json",
                "fusion_limited/ir_pre_fusion.txt",
                "fusion_limited/ir_post_fusion.txt",
                "custom_lowering/fx_graph_readable.py",
                "custom_lowering/ir_pre_fusion.txt",
                "custom_lowering/output_code.py",
                "custom_lowering/captured_cpp_kernel.cpp",
                "fallback_eigvals/fx_graph_readable.py",
                "fallback_eigvals/ir_pre_fusion.txt",
                "fallback_eigvals/ir_post_fusion.txt",
                "fallback_eigvals/output_code.py",
            }
            missing = sorted(
                relative
                for relative in required
                if not (output_dir / relative).is_file()
                or (output_dir / relative).stat().st_size == 0
            )
            self.assertEqual([], missing)

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "generated_not_executed", summary["codegen_only_status"]
            )
            self.assertFalse(summary["triton_autotune_tested"])
            self.assertEqual(
                1, summary["fusion_enabled"]["cpp_entrypoint_count"]
            )
            self.assertEqual(
                1, summary["fusion_limited"]["cpp_entrypoint_count"]
            )
            self.assertLess(
                summary["fusion_enabled"]["cpp_loop_count"],
                summary["fusion_limited"]["cpp_loop_count"],
            )
            self.assertTrue(
                summary["fusion_enabled"]["evidence_boundary"][
                    "cpp_compiler_mocked"
                ]
            )
            self.assertFalse(
                summary["fusion_enabled"]["evidence_boundary"][
                    "generated_cpp_kernel_executed"
                ]
            )
            provenance_chain = json.loads(
                (
                    output_dir / "fusion_enabled" / "provenance_chain.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(provenance_chain["joined_chains"])
            self.assertTrue(
                all(
                    chain["pre_fx_nodes"] and chain["source_stack"]
                    for chain in provenance_chain["joined_chains"]
                )
            )
            environment = json.loads(
                (output_dir / "environment.json").read_text(encoding="utf-8")
            )
            self.assertFalse(environment["runtime_matches_source_baseline"])


class PartFourIrSchedulerContractTest(unittest.TestCase):
    def test_ir_scheduler_liveness_matrix_without_native_kernel_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="graph_series_ir_test_") as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            observed = run_lab(
                "part4_ir_scheduler_analysis.py",
                "--output-dir",
                str(output_dir),
            )
            expected = {
                "elementwise_ir_observed": "True",
                "broadcast_index_observed": "True",
                "transpose_view_observed": "True",
                "copy_buffer_observed": "True",
                "reduction_ir_observed": "True",
                "matmul_extern_observed": "True",
                "dependency_edges_recorded": "True",
                "fusion_toggle_observed": "True",
                "reorder_comparison_recorded": "True",
                "reorder_effect_observed": "True",
                "static_peak_estimate_recorded": "True",
                "native_kernel_performance_tested": "False",
            }
            self.assertEqual(expected, {key: observed.get(key) for key in expected})

            required = {
                "environment.json",
                "manifest.json",
                "ir_matrix.json",
                "scheduler_dependencies.json",
                "fusion_comparison.json",
                "reorder_comparison.json",
                "liveness_peak.json",
            }
            missing = sorted(
                relative
                for relative in required
                if not (output_dir / relative).is_file()
                or (output_dir / relative).stat().st_size == 0
            )
            self.assertEqual([], missing)
            liveness = json.loads(
                (output_dir / "liveness_peak.json").read_text(encoding="utf-8")
            )
            self.assertIn("scheduled_order_estimated_peak_bytes", liveness)
            self.assertNotIn("baseline_estimated_peak_bytes", liveness)
            self.assertNotIn("total_input_bytes", liveness)
            self.assertEqual(
                "static_scheduler_estimate_not_physical_allocator_peak",
                liveness["estimator_kind"],
            )
            reorder = json.loads(
                (output_dir / "reorder_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(reorder["schedule_changed"])
            self.assertTrue(reorder["effect_observed"])
            self.assertTrue(reorder["reorder_enabled_topological"])
            self.assertTrue(reorder["reorder_disabled_topological"])
            self.assertLess(
                reorder["reorder_enabled"]["liveness"]["estimated_peak_bytes"],
                reorder["reorder_disabled"]["liveness"]["estimated_peak_bytes"],
            )
            environment = json.loads(
                (output_dir / "environment.json").read_text(encoding="utf-8")
            )
            self.assertFalse(environment["runtime_matches_source_baseline"])


class UnifiedSeriesArtifactContractTest(unittest.TestCase):
    def test_unified_model_emits_frontend_aot_and_backend_stage_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="graph_series_e2e_test_") as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            observed = run_lab(
                "series_artifact_bundle.py", "--output-dir", str(output_dir)
            )
            expected = {
                "forward_matches": "True",
                "gradient_matches": "True",
                "dynamic_export_has_range_constraints": "True",
                "export_out_of_range_rejected": "True",
                "aot_has_joint_forward_backward": "True",
                "aot_cross_graph_node_refs": "0",
                "aot_joint_partition_mapping_exact": "True",
                "aot_partition_to_compiler_callback_continuity": "True",
                "aot_saved_slot_binding_origins_match": "True",
                "artifact_bundle_continuity": "partial",
                "backend_codegen_status": "generated_not_executed",
                "hop_branch_captured": "True",
                "hop_invalid_branch_rejected": "True",
            }
            self.assertEqual(expected, {key: observed.get(key) for key in expected})

            required = {
                "environment.json",
                "model_source.py",
                "model_contract.json",
                "symbolic_fx.py",
                "dynamo_fx.py",
                "dynamo_guards.txt",
                "exported_program.py",
                "export_graph_signature.json",
                "functional_aten.py",
                "aot_joint.py",
                "aot_forward.py",
                "aot_backward.py",
                "aot_partition_abi.json",
                "aot_joint_to_fw_bw_node_mapping.json",
                "artifact_manifest.json",
                "stage_node_mapping.json",
                "hop_exported_program.py",
                "backend/fx_graph_readable.py",
                "backend/fx_graph_transformed.py",
                "backend/ir_pre_fusion.txt",
                "backend/ir_post_fusion.txt",
                "backend/output_code.py",
                "backend/captured_cpp_kernel.cpp",
            }
            missing = sorted(
                relative
                for relative in required
                if not (output_dir / relative).is_file()
                or (output_dir / relative).stat().st_size == 0
            )
            self.assertEqual([], missing)

            model_contract = json.loads(
                (output_dir / "model_contract.json").read_text(encoding="utf-8")
            )
            required_features = {
                "parameter",
                "buffer",
                "view",
                "mutation",
                "dynamic_shape",
                "structured_output",
                "matmul",
                "pointwise",
                "reduction",
                "higher_order_branch",
            }
            self.assertEqual(required_features, set(model_contract["features"]))

            artifact_manifest = json.loads(
                (output_dir / "artifact_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "partial", artifact_manifest["overall_stage_continuity"]
            )
            self.assertTrue(
                artifact_manifest["claims"][
                    "aot_joint_partition_fw_bw_is_one_continuous_run"
                ]
            )
            self.assertFalse(
                artifact_manifest["claims"][
                    "all_frontend_artifacts_form_one_continuous_compile"
                ]
            )
            self.assertFalse(
                artifact_manifest["claims"][
                    "aot_forward_is_direct_input_to_recorded_inductor_run"
                ]
            )

            aot_mapping = json.loads(
                (
                    output_dir / "aot_joint_to_fw_bw_node_mapping.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(all(aot_mapping["exact_mapping_checks"].values()))
            self.assertTrue(
                aot_mapping["compiler_callback_continuity"][
                    "continuity_verified"
                ]
            )
            self.assertEqual(
                0,
                aot_mapping["summary"]["unmapped_non_output_joint_node_count"],
            )
            self.assertTrue(
                all(
                    isinstance(entry["origin_id"], str)
                    and entry["origin_id"]
                    for entry in aot_mapping["entries"]
                )
            )

            aot_abi = json.loads(
                (output_dir / "aot_partition_abi.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(aot_abi["saved_slot_bindings"])
            self.assertTrue(
                all(
                    binding["forward_joint_origin_token"]
                    and binding["backward_joint_origin_token"]
                    and binding["same_joint_origin"]
                    for binding in aot_abi["saved_slot_bindings"]
                )
            )
            self.assertFalse(
                aot_abi["saved_slot_binding_checks"][
                    "binding_is_cross_graph_node_edge"
                ]
            )


if __name__ == "__main__":
    unittest.main()
