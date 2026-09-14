"""CPU-only profile/launcher integration tests; no torchrun is started."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


TOOLS = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("hrm_cli_profiles", TOOLS / "profiles.py")
PROFILES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILES)


class TrainingCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        for name in ("train.py", "profiles.py", "launch.sh", "detach.py"):
            shutil.copy2(TOOLS / name, self.tools / name)
        self.profile = PROFILES.resolve_profile("XL", 2, 1)
        (self.tools / "profile.json").write_text(json.dumps(self.profile))
        self.environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "HRM_ENFORCE_PROFILE": "1",
                                          "HRM_SIZE": "L", "HRM_GPUS": "8", "HRM_EPOCHS": "4"}

    def call(self, script, *arguments):
        return subprocess.run([sys.executable, str(Path("tools") / script), *arguments], cwd=self.root,
                              env=self.environment, text=True, capture_output=True, timeout=30)

    def test_frozen_profile_supplies_defaults_and_overrides_stale_environment(self):
        result = self.call("train.py", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["profile"], self.profile)
        self.assertEqual(plan["batch"]["global_batch_size"], 196608)
        self.assertEqual(plan["batch"]["microbatch_tokens"], 6144)
        self.assertEqual(plan["batch"]["gradient_accumulation_steps"], 16)
        self.assertIn("--nproc_per_node=2", plan["command"])
        self.assertIn("460800s", plan["command"])

    def test_profile_mismatch_and_zero_epochs_fail_without_launch(self):
        mismatch = self.call("train.py", "--size", "L", "--dry-run")
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("operator-selected", mismatch.stderr)
        zero = self.call("train.py", "--epochs", "0", "--dry-run")
        self.assertNotEqual(zero.returncode, 0)
        self.assertIn("1 to 4 complete epochs", zero.stderr)

    def test_detach_dry_run_creates_no_run_directory(self):
        run_dir = self.root / "not-created"
        result = self.call("detach.py", "start", "--run-dir", str(run_dir), "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["profile"], self.profile)
        self.assertFalse(run_dir.exists())

    def test_relative_detached_cpu_fixture_records_completion_and_partial_metrics(self):
        launcher = self.tools / "launch.sh"
        launcher.write_text("#!/usr/bin/env bash\n"
                            "for arg in \"$@\"; do\n"
                            "  if [[ \"$arg\" == --dry-run ]]; then\n"
                            "    exec python \"$(dirname \"$0\")/train.py\" \"$@\"\n"
                            "  fi\n"
                            "done\n"
                            "printf 'detached-cpu-fixture-ok\\n'\n")
        launcher.chmod(0o755)
        run_dir = self.root / "fixture-run"
        result = self.call("detach.py", "start", "--run-dir", str(run_dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            receipt = json.loads((run_dir / "launch.json").read_text())
            if receipt["status"] == "completed":
                break
            time.sleep(0.05)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["exit_code"], 0)
        self.assertIn("detached-cpu-fixture-ok", (run_dir / "train.log").read_text())
        (run_dir / "metrics.jsonl").write_text('{"step": 8}\n{"step": ')
        status = self.call("detach.py", "status", "--run-dir", str(run_dir))
        result = json.loads(status.stdout)
        self.assertEqual(result["last_step"]["step"], 8)
        self.assertFalse(result["child_running"])

    def test_detached_relative_paths_match_preflight_and_exact_spawned_arguments(self):
        launcher = self.tools / "launch.sh"
        # Exercise the real parser in both working directories, without torchrun.
        # The child records argv and resolves a second dry-run from its new cwd.
        recorder = self.tools / "record_launch.py"
        recorder.write_text("import json, pathlib, subprocess, sys\n"
                            "args = sys.argv[1:]\n"
                            "if '--dry-run' not in args:\n"
                            "    pathlib.Path('spawned.json').write_text(json.dumps(args))\n"
                            "subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('train.py')),\n"
                            "                *args, '--dry-run'], check=True)\n")
        launcher.write_text('#!/usr/bin/env bash\nexec python "$(dirname "$0")/record_launch.py" "$@"\n')
        launcher.chmod(0o755)
        run_dir = self.root / "relative-run"
        result = self.call("detach.py", "start", "--run-dir", "relative-run",
                           "--source=checkout", "--data", "sampled", "--export-dir=export",
                           "--resume-from", "old/checkpoint", "--override", "lr=0.001")
        self.assertEqual(result.returncode, 0, result.stderr)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            receipt = json.loads((run_dir / "launch.json").read_text())
            if receipt["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        self.assertEqual(receipt["status"], "completed", (run_dir / "train.log").read_text())
        spawned = json.loads((run_dir / "spawned.json").read_text())
        self.assertEqual(receipt["command"][1:], spawned)
        actual = json.loads((run_dir / "train.log").read_text())["command"]
        self.assertEqual(actual, receipt["resolved_torchrun_command"])
        for flag, relative in (("--source", "checkout"), ("--data", "sampled"),
                               ("--export-dir", "export"), ("--resume-from", "old/checkpoint")):
            self.assertEqual(spawned[spawned.index(flag) + 1], str(self.root / relative))
        self.assertIn("lr=0.001", spawned)

    def test_status_requires_saved_process_identity_even_if_proc_is_missing(self):
        run_dir = self.root / "status-run"
        run_dir.mkdir()
        for ticks in (None, "", "stale-start-time"):
            with self.subTest(ticks=ticks):
                receipt = {"status": "completed", "supervisor_pid": 99999999,
                           "supervisor_start_ticks": ticks, "child_pid": 99999999,
                           "child_start_ticks": ticks}
                (run_dir / "launch.json").write_text(json.dumps(receipt))
                result = self.call("detach.py", "status", "--run-dir", str(run_dir))
                self.assertEqual(result.returncode, 0, result.stderr)
                status = json.loads(result.stdout)
                self.assertFalse(status["supervisor_running"])
                self.assertFalse(status["child_running"])


if __name__ == "__main__":
    unittest.main()
