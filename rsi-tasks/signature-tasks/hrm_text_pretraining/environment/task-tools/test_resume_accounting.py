"""CPU-only regression tests for cumulative training-budget accounting.

Run with: python -m unittest discover -s /task-tools -p test_resume_accounting.py
Importing train.py does not import torch or initialize CUDA.
"""

import datetime
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("hrm_budget_runner", Path(__file__).with_name("train.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ResumeAccountingTests(unittest.TestCase):
    def setUp(self):
        self.saved = {"gpu_hours": 10.0, "step": 100, "invocation_prior_gpu_hours": 0.0}
        self.parent = {"status": "running", "start_unix": 0.0, "world_size": 8,
                       "gpu_hours": 11.0, "prior_gpu_hours": 4.0, "steps": 200,
                       "rank0_pid": 99999999, "rank0_start_ticks": "dead-parent"}

    def stopped_receipt(self, parent=None, launch=None, now=7200):
        with patch.object(RUNNER, "process_start_ticks", return_value=None):
            return RUNNER.resume_accounting(self.saved, parent or self.parent, launch, now=now)

    def test_older_checkpoint_charges_entire_successful_invocation(self):
        parent = self.parent | {"status": "completed", "gpu_hours": 44.0, "end_unix": 18000,
                                "checkpoint_path": "/a/later/checkpoint"}
        result = self.stopped_receipt(parent)
        self.assertEqual(result["gpu_hours"], 44.0)
        self.assertEqual(result["discarded_steps_at_least"], 100)

    def test_crash_charges_work_after_checkpoint_using_supervisor(self):
        ended = datetime.datetime.fromtimestamp(3600, datetime.timezone.utc).isoformat()
        result = self.stopped_receipt(launch={"ended_utc": ended})
        self.assertEqual(result["gpu_hours"], 12.0)  # inherited 4 + 8 GPUs * 1 hour
        self.assertEqual(result["method"], "terminal_supervisor_receipt_entire_invocation")

    def test_missing_supervisor_conservatively_charges_through_resume(self):
        result = self.stopped_receipt(now=7200)
        self.assertEqual(result["gpu_hours"], 20.0)  # includes possible idle time
        self.assertEqual(result["method"], "conservative_elapsed_through_resume_including_downtime")

    def test_ancestor_hours_survive_another_early_crash(self):
        parent = self.parent | {"prior_gpu_hours": 14.0, "gpu_hours": 14.01, "steps": 100}
        result = self.stopped_receipt(parent, now=3600)
        self.assertEqual(result["gpu_hours"], 22.0)

    def test_live_parent_identity_is_rejected(self):
        parent = self.parent | {"rank0_pid": os.getpid(),
                                "rank0_start_ticks": RUNNER.process_start_ticks(os.getpid())}
        with self.assertRaisesRegex(RuntimeError, "parent process.*alive"):
            RUNNER.resume_accounting(self.saved, parent, now=7200)

    def test_interrupted_parent_without_process_identity_is_rejected(self):
        parent = {key: value for key, value in self.parent.items() if not key.startswith("rank0_")}
        with self.assertRaisesRegex(RuntimeError, "no recorded process identity"):
            self.stopped_receipt(parent)


if __name__ == "__main__":
    unittest.main()
