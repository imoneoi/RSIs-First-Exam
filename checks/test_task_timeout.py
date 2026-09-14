"""Long signature runs must not weaken the ordinary task timeout ceiling."""
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("check-task-timeout.sh")


class TimeoutContract(unittest.TestCase):
    def check_task(self, *, signature=False, seconds="259200", opt_in=True,
                   smoke_seconds="1800", note="Dedicated GPU runner required"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / ("rsi-tasks/signature-tasks/example" if signature else "tasks/example")
            task.mkdir(parents=True)
            body = f"[agent]\ntimeout_sec = {seconds}\n[verifier]\ntimeout_sec = 86400\n"
            if opt_in:
                body += (
                    "[metadata.run_contract]\nlong_horizon = true\n"
                    'ci_mode = "smoke-only"\n'
                    f"ci_smoke_timeout_sec = {smoke_seconds}\n"
                    f'timeout_note = "{note}"\n'
                )
            (task / "task.toml").write_text(body)
            return subprocess.run(["bash", str(SCRIPT), str(task)], text=True,
                                  capture_output=True)

    def test_signature_with_bounded_smoke_contract_passes(self):
        result = self.check_task(signature=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_public_task_cannot_claim_signature_exemption(self):
        result = self.check_task(signature=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeds 18000", result.stdout)

    def test_signature_without_explicit_opt_in_fails(self):
        result = self.check_task(signature=True, opt_in=False)
        self.assertNotEqual(result.returncode, 0)

    def test_unbounded_or_invalid_smoke_contract_fails(self):
        for seconds in ("18001", "0", "-1", "nan", "inf", "true", '"1800"'):
            with self.subTest(seconds=seconds):
                self.assertNotEqual(self.check_task(signature=True, smoke_seconds=seconds).returncode, 0)

    def test_explanation_is_required(self):
        self.assertNotEqual(self.check_task(signature=True, note="").returncode, 0)

    def test_invalid_task_timeouts_remain_rejected(self):
        for seconds in ("0", "-1", "nan", "inf", "true"):
            with self.subTest(seconds=seconds):
                self.assertNotEqual(self.check_task(signature=True, seconds=seconds).returncode, 0)

    def test_ordinary_task_within_ceiling_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary) / "tasks/example"
            task.mkdir(parents=True)
            (task / "task.toml").write_text("[agent]\ntimeout_sec = 18000\n[verifier]\ntimeout_sec = 1200\n")
            result = subprocess.run(["bash", str(SCRIPT), str(task)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
