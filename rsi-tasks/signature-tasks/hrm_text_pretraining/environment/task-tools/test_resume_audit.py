"""Explicit failed-child receipts preserve work before the next checkpoint."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("resume_audit_runner", Path(__file__).with_name("train.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ResumeAuditTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = Path("/attempts/parent/checkpoint_step_00000010")
        self.audit_path = Path("/attempts/failed-child/run.json")
        self.profile = {"clock": "synchronized_training_minus_compile", "track": "fast-b"}
        self.digest = "a" * 64
        self.saved = {"step": 10, "compute_seconds": 4, "profile": self.profile,
                      "data_manifest_sha256": self.digest, "clock": self.profile["clock"],
                      "config": {"arch": {"name": "native"}, "epochs": 1, "lr": 0.001, "run_name": "parent"}}
        self.previous = {"status": "running", "resume_from": str(self.checkpoint), "steps": 13,
                         "compute_seconds": 6, "profile": self.profile, "data_manifest_sha256": self.digest,
                         "clock": self.profile["clock"], "config": self.saved["config"] | {"run_name": "child"}}
        self.launch = {"status": "failed", "ended_utc": "2026-09-23T00:00:00+00:00"}

    def validate(self, previous=None, launch=None):
        RUNNER.validate_resume_audit(self.checkpoint, self.audit_path, self.saved,
                                    previous if previous is not None else self.previous,
                                    launch if launch is not None else self.launch, self.profile, self.digest)

    def test_stopped_child_before_checkpoint_keeps_its_complete_charge(self):
        self.validate()
        journal = {"inflight": False, "step": 14, "compute_seconds": 7.5, "compute_phase_max_seconds": {"2": 0.5}}
        receipt = RUNNER.resume_compute_accounting(self.saved, self.previous, journal, self.launch)
        self.assertEqual(receipt["compute_seconds"], 7.5)
        self.assertEqual(receipt["discarded_steps_at_least"], 4)
        self.assertEqual(receipt["phase_max_seconds"], {"2": 0.5})

    def test_unrelated_unstopped_or_mismatched_child_receipts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.validate(launch={"status": "running"})
        for key, value in (("resume_from", "/other/checkpoint"), ("profile", {"track": "other"}),
                           ("data_manifest_sha256", "b" * 64), ("clock", "wall")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(self.previous | {key: value})
        changed = copy.deepcopy(self.previous)
        changed["config"]["lr"] = 0.5
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.validate(changed)


if __name__ == "__main__":
    unittest.main()
