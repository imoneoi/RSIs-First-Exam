"""CPU regression tests for effective batches, accumulation, tails and resume."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F


SPEC = importlib.util.spec_from_file_location("hrm_group_runner", Path(__file__).with_name("train.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TrainingGroupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(91)
        self.initial = nn.Linear(4, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.initial.weight.copy_(torch.randn(3, 4, generator=generator, dtype=torch.float64))
        self.microbatches = []
        for rank_counts in ((3, 1), (9, 2), (1, 3), (0, 2), (7, 1)):
            ranks = []
            for count in rank_counts:
                inputs = torch.randn(count + 2, 4, generator=generator, dtype=torch.float64)
                labels = torch.randint(3, (count + 2,), generator=generator)
                labels[count:] = -100
                ranks.append((inputs, labels))
            self.microbatches.append(ranks)

    @staticmethod
    def count(microbatch):
        return sum(int((labels != -100).sum()) for _, labels in microbatch)

    def accumulated(self, model, group, update):
        counts = [self.count(microbatch) for microbatch in group]

        def backward(microbatch, weight):
            # Emulate native two-rank FSDP SUM reduction and each head's
            # division by globally averaged valid-token count.
            loss_sum = sum(F.cross_entropy(model(inputs), labels, reduction="sum", ignore_index=-100)
                           for inputs, labels in microbatch)
            (loss_sum / (self.count(microbatch) / 2) * weight).backward()
            return {"loss": (loss_sum.detach(), self.count(microbatch))}

        return RUNNER.run_accumulated_update(group, counts, backward, update)

    def test_reference_batch_defaults_and_explicit_research_batch(self):
        self.assertEqual(RUNNER.resolve_batch("L", 8), {"global_batch_size": 172032,
                         "microbatch_tokens": 21504, "gradient_accumulation_steps": 1,
                         "global_microbatch_tokens": 172032})
        self.assertEqual(RUNNER.resolve_batch("XL", 2)["gradient_accumulation_steps"], 16)
        self.assertEqual(RUNNER.resolve_batch("XL", 2, 12288)["gradient_accumulation_steps"], 8)
        self.assertEqual(RUNNER.resolve_batch("XL", 2, 4096, 4, 32768)["global_batch_size"], 32768)
        with self.assertRaises(ValueError):
            RUNNER.resolve_batch("XL", 2, 6144, 8)
        for invalid in (0, False, -1):
            with self.assertRaises(ValueError):
                RUNNER.resolve_batch("XL", 2, global_batch_size=invalid)

    def test_unequal_response_counts_match_combined_token_objective(self):
        group = self.microbatches[:3]
        accumulated, combined, naive = (copy.deepcopy(self.initial) for _ in range(3))
        updates = []
        self.accumulated(accumulated, group, lambda: updates.append("optimizer"))
        total = sum(self.count(microbatch) for microbatch in group)
        full_loss = sum(F.cross_entropy(combined(inputs), labels, reduction="sum", ignore_index=-100)
                        for microbatch in group for inputs, labels in microbatch)
        (full_loss / (total / 2)).backward()
        torch.testing.assert_close(accumulated.weight.grad, combined.weight.grad, rtol=1e-12, atol=1e-12)
        self.assertEqual(updates, ["optimizer"])
        for microbatch in group:
            loss_sum = sum(F.cross_entropy(naive(inputs), labels, reduction="sum", ignore_index=-100)
                           for inputs, labels in microbatch)
            (loss_sum / (self.count(microbatch) / 2) / len(group)).backward()
        self.assertFalse(torch.allclose(naive.weight.grad, combined.weight.grad, rtol=1e-6, atol=1e-6))

    def test_short_final_group_uses_its_actual_response_total(self):
        groups = list(RUNNER.accumulation_groups(self.microbatches, 3))
        self.assertEqual([(end, len(group), complete) for end, group, complete in groups],
                         [(3, 3, False), (5, 2, True)])
        accumulated, combined = (copy.deepcopy(self.initial) for _ in range(2))
        group = groups[-1][1]
        metrics = self.accumulated(accumulated, group, lambda: None)
        loss_sum = sum(F.cross_entropy(combined(inputs), labels, reduction="sum", ignore_index=-100)
                       for microbatch in group for inputs, labels in microbatch)
        total = sum(self.count(microbatch) for microbatch in group)
        (loss_sum / (total / 2)).backward()
        self.assertEqual(metrics["loss"][1], total)
        torch.testing.assert_close(accumulated.weight.grad, combined.weight.grad, rtol=1e-12, atol=1e-12)

    def test_resume_preserves_optimizer_schedule_and_ema_clocks(self):
        model = copy.deepcopy(self.initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        ema = model.weight.detach().clone()
        steps, history = 0, []
        checkpoint = None
        for end, group, complete in RUNNER.accumulation_groups(self.microbatches, 3):
            steps += 1
            optimizer.param_groups[0]["lr"] = 0.01 * steps / 10
            history.append((steps, optimizer.param_groups[0]["lr"], 2 + steps // 2))
            def update():
                optimizer.step()
                optimizer.zero_grad()
                ema.lerp_(model.weight.detach(), 0.2)
            self.accumulated(model, group, update)
            if not complete:
                checkpoint = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
                              ema.clone(), steps, end)
        saved_model, saved_optimizer, saved_ema, saved_step, saved_microbatch = checkpoint
        resumed = copy.deepcopy(self.initial)
        resumed.load_state_dict(saved_model)
        resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.01, momentum=0.8)
        resumed_optimizer.load_state_dict(saved_optimizer)
        resumed_history = []
        for _, group, complete in RUNNER.accumulation_groups(self.microbatches, 3, saved_microbatch):
            saved_step += 1
            resumed_optimizer.param_groups[0]["lr"] = 0.01 * saved_step / 10
            resumed_history.append((saved_step, resumed_optimizer.param_groups[0]["lr"], 2 + saved_step // 2))
            def update():
                resumed_optimizer.step()
                resumed_optimizer.zero_grad()
                saved_ema.lerp_(resumed.weight.detach(), 0.2)
            self.accumulated(resumed, group, update)
            self.assertTrue(complete)
        self.assertEqual(steps, 2)
        self.assertEqual(resumed_history, history[1:])
        torch.testing.assert_close(resumed.weight, model.weight, rtol=0, atol=0)
        torch.testing.assert_close(saved_ema, ema, rtol=0, atol=0)

    def test_epoch_boundary_and_invalid_resume_positions(self):
        groups = list(RUNNER.accumulation_groups(range(6), 3))
        self.assertEqual(groups, [(3, [0, 1, 2], False), (6, [3, 4, 5], True)])
        self.assertEqual(list(RUNNER.accumulation_groups(range(6), 3, 6)), [])
        with self.assertRaises(ValueError):
            list(RUNNER.accumulation_groups(range(6), 3, 7))

    def test_no_cuda_was_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_carry_restore_preserves_values_rng_and_initialized_devices(self):
        # A meta tensor checks non-CPU device dispatch without allocating a GPU.
        initialized = {"hidden": [torch.empty(2, device="meta"), torch.empty(1)], "step": 0}
        saved = {"hidden": [torch.tensor([3., 4.]), torch.tensor([9.])], "step": 7}
        runtime = {"carry": saved, "cpu_rng": torch.get_rng_state()}
        rng_before = runtime["cpu_rng"].clone()
        restored = RUNNER.restore_carry(runtime["carry"], initialized)
        self.assertEqual(restored["hidden"][0].device.type, "meta")
        self.assertEqual(restored["hidden"][1].device.type, "cpu")
        torch.testing.assert_close(restored["hidden"][1], torch.tensor([9.]))
        self.assertEqual(restored["step"], 7)
        self.assertEqual(saved["hidden"][0].device.type, "cpu")
        torch.testing.assert_close(saved["hidden"][0], torch.tensor([3., 4.]))
        self.assertEqual(runtime["cpu_rng"].device.type, "cpu")
        self.assertTrue(torch.equal(runtime["cpu_rng"], rng_before))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertIsNone(RUNNER.restore_carry(None, None))

    def test_carry_restore_rejects_structure_shape_dtype_and_leaf_type_changes(self):
        initial = {"hidden": torch.empty(2), "step": 0}
        invalid = [
            {"other": torch.ones(2), "step": 1},
            {"hidden": torch.ones(3), "step": 1},
            {"hidden": torch.ones(2, dtype=torch.float64), "step": 1},
            {"hidden": [1., 2.], "step": 1},
            {"hidden": torch.ones(2), "step": 1.},
        ]
        for saved in invalid:
            with self.subTest(saved=saved), self.assertRaisesRegex(ValueError, "Checkpoint carry"):
                RUNNER.restore_carry(saved, initial)

    def test_export_weight_label_matches_actual_ema_buffer_coverage(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # No EMA buffers is the native ema=null behavior.
        self.assertEqual(RUNNER.export_weight_kind(optimizer), "model")
        optimizer.state[model.weight]["param_ema"] = model.weight.detach().clone()
        self.assertEqual(RUNNER.export_weight_kind(optimizer), "mixed EMA/model")
        optimizer.state[model.bias]["param_ema"] = model.bias.detach().clone()
        self.assertEqual(RUNNER.export_weight_kind(optimizer), "EMA")

    def test_selected_epoch_schedule_does_not_use_unselected_epoch_lengths(self):
        manifest = {"validation": {"epochs": [
            {"epoch": 0, "tokens_including_ar_shift": 100},
            {"epoch": 1, "tokens_including_ar_shift": 300},
        ]}}
        selected = RUNNER.selected_schedule_tokens({"total_length": 200}, manifest, 1)
        self.assertEqual(selected["tokens"], 100)
        self.assertEqual(RUNNER.selected_schedule_tokens({"total_length": 200}, manifest, 2)["tokens"], 400)
        with self.assertRaises(ValueError):
            RUNNER.selected_schedule_tokens({"total_length": 200}, manifest, 3)

    def test_architecture_export_freezes_inference_dtype_and_bounds_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, export = root / "checkout", root / "submission"
            (source / "models").mkdir(parents=True)
            (source / "evaluation").mkdir()
            (source / "models" / "custom.py").write_text("class Model: pass\n")
            (source / "evaluation" / "private.py").write_text("SHOULD_NOT_EXPORT = True\n")
            export.mkdir()
            training_config = {"fwd_bwd_dtype": "float32", "arch": {"name": "custom@Model", "head": "lm_head@LMHead"}}
            model = RUNNER.export_architecture(source, export, training_config["arch"], 1024)
            self.assertEqual(model["forward_dtype"], "bfloat16")
            self.assertEqual(set(model["source_files"]), {"models/custom.py"})
            self.assertEqual(json.loads((export / "model.json").read_text()), model)
            self.assertEqual(model["source_files"]["models/custom.py"], RUNNER.file_sha256(export / "source/models/custom.py"))
            with self.assertRaises(ValueError):
                RUNNER.architecture_sources(source, 4)

    def test_export_inside_checkout_excludes_previous_export_subtree(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            export = source / "submission"
            (source / "models").mkdir()
            (source / "models/core.py").write_text("class Model: pass\n")
            (export / "source/models").mkdir(parents=True)
            (export / "source/models/obsolete.py").write_text("THIS_IS_NOT_SOURCE = True\n")
            model = RUNNER.export_architecture(source, export, {"name": "core@Model"}, 1024)
            self.assertEqual(set(model["source_files"]), {"models/core.py"})
            self.assertFalse((export / "source/submission").exists())


if __name__ == "__main__":
    unittest.main()
