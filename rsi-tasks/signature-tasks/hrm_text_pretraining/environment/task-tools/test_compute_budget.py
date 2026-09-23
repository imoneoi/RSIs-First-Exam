"""CPU checks for measured compute, capped updates, ancestry and calibration."""
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import summarize, baseline_recipe, validate_baseline_recipe, hardware_signature
from profiles import HRM_COMMIT, resolve_profile
from compute_budget import CompilationTracker, ComputeBudget, interval_union_seconds, phase_step_counts, project_baseline, retained_checkpoint_stop
SPEC = importlib.util.spec_from_file_location("compute_runner", Path(__file__).with_name("train.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ComputeClockTests(unittest.TestCase):
    def test_first_compiled_update_charges_numerical_residual(self):
        clock = CompilationTracker(iter([0, 1, 11, 12]).__next__)
        clock.begin()
        clock.on_compile_start(None)
        clock.on_compile_end(None)
        measured = clock.finish()
        self.assertEqual(measured["compilation_seconds"], 10)
        self.assertEqual(measured["compute_seconds"], 2)

    def test_nested_compilation_is_not_double_subtracted(self):
        clock = CompilationTracker(iter([0, 1, 9, 12]).__next__)
        clock.begin()
        clock.on_compile_start(None)
        clock.on_compile_start(None)
        clock.on_compile_end(None)
        clock.on_compile_end(None)
        self.assertEqual(clock.finish()["compute_seconds"], 4)
        self.assertEqual(interval_union_seconds([(-5, 2), (1, 4), (3, 7), (9, 20)], 0, 10), 8)

    def test_incomplete_compilation_cannot_create_free_training(self):
        clock = CompilationTracker(lambda: 1)
        clock.begin()
        clock.on_compile_start(None)
        with self.assertRaises(RuntimeError):
            clock.finish()

    def test_pre_step_reserve_stops_before_cap_and_reports_anomaly(self):
        budget = ComputeBudget(10, initial_step_bound=1)
        self.assertTrue(budget.can_start(2))
        budget.charge(2, 2)
        budget.charge(5, 2)
        self.assertFalse(budget.can_start(2))
        self.assertEqual(budget.used, 7)
        self.assertFalse(budget.charge(4, 2))

    def test_reserve_evidence_reproduces_seen_and_unseen_phase_bounds(self):
        budget = ComputeBudget(100, initial_step_bound=2)
        budget.charge(3, 2)
        for phase, expected in ((2, 4.55), (3, 9.05)):
            evidence = budget.reserve_evidence(phase)
            observed = evidence["phase_max_seconds"].get(evidence["phase"])
            if observed is None:
                observed = max(evidence["initial_step_bound_seconds"], 2 * max(evidence["phase_max_seconds"].values()))
            self.assertAlmostEqual(observed * 1.5 + 0.05, expected)
            self.assertAlmostEqual(evidence["seconds"], expected)

    def test_early_restore_failure_keeps_inherited_compute_journal(self):
        saved = {"step": 10, "compute_seconds": 4}
        previous = {"status": "running", "compute_seconds": 17}
        journal = {"inflight": False, "step": 10, "compute_seconds": 17}
        self.assertEqual(RUNNER.resume_compute_accounting(saved, previous, journal)["compute_seconds"], 17)

    def test_terminal_checkpoint_reexport_retains_reserve_history_and_exact_cap(self):
        saved = {"step": 10, "compute_seconds": 4, "compute_phase_max_seconds": {"2": 1}}
        previous = {"status": "running", "compute_seconds": 6, "compute_reserve": {"phase_max_seconds": {"2": 2}}}
        journal = {"inflight": False, "step": 12, "compute_seconds": 6, "compute_phase_max_seconds": {"3": 3}}
        receipt = RUNNER.resume_compute_accounting(saved, previous, journal)
        self.assertEqual(receipt["phase_max_seconds"], {"2": 2, "3": 3})
        budget = ComputeBudget(8, used=receipt["compute_seconds"], initial_step_bound=0.1, phase_max=receipt["phase_max_seconds"])
        self.assertEqual(retained_checkpoint_stop(budget, 2, has_checkpoint=True, completed_epochs=0, requested_epochs=1), "compute_budget")
        self.assertEqual(budget.used, 6)  # Re-export adds no optimizer update or compute charge.
        self.assertIsNone(retained_checkpoint_stop(budget, 2, has_checkpoint=False, completed_epochs=0, requested_epochs=1))
        exact = ComputeBudget(8, used=8)
        self.assertEqual(retained_checkpoint_stop(exact, 2, has_checkpoint=True, completed_epochs=0, requested_epochs=1), "compute_budget")
        self.assertEqual(retained_checkpoint_stop(exact, 2, has_checkpoint=True, completed_epochs=1, requested_epochs=1), "epochs_completed")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            retained_checkpoint_stop(ComputeBudget(8, used=9), 2, has_checkpoint=True, completed_epochs=0, requested_epochs=1)

    def test_completed_epoch_terminal_resume_initializes_final_audit_position(self):
        saved = {"epoch": 2, "batch_in_epoch": 500, "epoch_complete": True, "step": 900}
        epoch, batch, complete = RUNNER.checkpoint_position(saved)
        self.assertEqual((epoch, batch, complete), (2, 500, True))
        self.assertEqual(RUNNER.checkpoint_position(), (1, 0, False))
        budget = ComputeBudget(10, used=9, phase_max={"6": 0.5})
        reason = retained_checkpoint_stop(budget, 6, has_checkpoint=True,
                                          completed_epochs=epoch if complete else epoch - 1, requested_epochs=2)
        self.assertEqual(reason, "epochs_completed")
        self.assertEqual(budget.used, 9)
        self.assertEqual(saved["step"], 900)

    def test_calibration_summary_roundtrip_and_dirty_source_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = resolve_profile("fast-b")
            run = {"status": "completed", "clock": profile["clock"], "baseline": "hrm",
                   "calibration_steps_per_phase": 4, "smoke": True, "source_commit": HRM_COMMIT,
                   "source_clean": True, "profile": profile,
                   "config": baseline_recipe(profile),
                   "schedule_total_steps": 800, "data_manifest_sha256": "a" * 64,
                   "hardware": {"nodes": 1, "gpus_per_node": 8,
                                "ranks": [{"gpu_name": "CPU timing fixture", "total_memory_bytes": 80, "capability": [9, 0]}] * 8},
                   "schedule_token_estimate": {}, "batch": {"global_batch_size": 188416, "microbatch_tokens": 23552, "gradient_accumulation_steps": 1}}
            records = [{"bp_steps": phase, "compute_seconds": phase, "compilation_seconds": 10 if index == 0 else 0}
                       for phase in range(2, 7) for index in range(4)]
            (root / "run.json").write_text(json.dumps(run))
            (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            result = summarize(root)
            self.assertEqual(result["measured_steps"], 20)
            self.assertEqual(result["excluded_compilation_seconds"], 50)
            self.assertEqual(resolve_profile("fast-b", calibration=result)["calibration"], result)
            run["source_clean"] = False
            (root / "run.json").write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "clean pinned"):
                summarize(root)

    def test_calibration_rejects_scientific_recipe_changes_and_hardware_drift(self):
        profile = resolve_profile("fast-b")
        batch = {"global_batch_size": 188416, "microbatch_tokens": 23552, "gradient_accumulation_steps": 1}
        for key, value in (("lr", 0.01), ("ema", 0.9), ("global_batch_size", 100), ("seed", 1)):
            changed = baseline_recipe(profile) | {key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recipe differs"):
                validate_baseline_recipe(changed, profile, batch)
        for key, value in (("bp_warmup_ratio", 0), ("n_layers", 34), ("name", "another@Model")):
            changed = baseline_recipe(profile)
            changed["arch"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recipe differs"):
                validate_baseline_recipe(changed, profile, batch)
        current = {"nodes": 1, "gpus_per_node": 1, "ranks": [{"gpu_name": "H100", "total_memory_bytes": 80, "capability": [9, 0], "hostname": "native"}]}
        container = {**current, "ranks": [{**current["ranks"][0], "hostname": "container"}]}
        different = {**current, "ranks": [{**current["ranks"][0], "gpu_name": "A100"}]}
        self.assertEqual(hardware_signature(current), hardware_signature(container))
        self.assertNotEqual(hardware_signature(current), hardware_signature(different))

    def test_discarded_updates_keep_complete_compute_bill(self):
        saved = {"step": 10, "compute_seconds": 4}
        previous = {"status": "completed", "steps": 30, "compute_seconds": 12}
        journal = {"inflight": False, "step": 30, "compute_seconds": 12}
        receipt = RUNNER.resume_compute_accounting(saved, previous, journal)
        self.assertEqual(receipt["compute_seconds"], 12)
        self.assertEqual(receipt["discarded_steps_at_least"], 20)

    def test_interrupted_update_is_conservatively_charged(self):
        saved = {"step": 10, "compute_seconds": 4}
        previous = {"status": "running", "steps": 30, "compute_seconds": 12}
        journal = {"inflight": True, "step": 35, "compute_seconds": 14, "update_started_unix": 100}
        receipt = RUNNER.resume_compute_accounting(saved, previous, journal, now=103)
        self.assertEqual(receipt["compute_seconds"], 17)
        self.assertEqual(receipt["uncertain_interrupted_seconds"], 3)

    def test_native_dev_schedule_weights_and_phase_projection(self):
        counts = phase_step_counts(800)
        self.assertEqual(counts, {2: 99, 3: 100, 4: 100, 5: 100, 6: 401})
        records = [{"bp_steps": phase, "compute_seconds": value}
                   for phase in counts for value in (99, 99, phase, phase + 2)]
        projection = project_baseline(records, counts)
        self.assertEqual(projection["estimated_baseline_compute_seconds"], sum(count * (phase + 1) for phase, count in counts.items()))
        self.assertGreater(projection["sample_max_projection_seconds"], projection["estimated_baseline_compute_seconds"])
        with self.assertRaises(ValueError):
            project_baseline(records[:4], counts)


if __name__ == "__main__":
    unittest.main()
