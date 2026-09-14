#!/usr/bin/env python3
"""Start a durable local training supervisor, or inspect its recorded status.

python /task-tools/detach.py start --run-dir /app/output/attempts/full-l \
    --source /app/hrm-text --data /datasets/hrm-text/sampled \
    --export-dir /app/output/submission --verify-checkpoint
python /task-tools/detach.py status --run-dir /app/output/attempts/full-l

The process group survives the requesting terminal. launch.json records the
supervisor/torchrun command and final exit code; train.log contains all output.
The launch.sh watchdog and train.py cumulative budget both remain active.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_start_ticks(pid):
    try:
        # The executable name is parenthesized and can contain spaces.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("start", "status", "_supervise"))
    parser.add_argument("--run-dir", required=True, type=Path)
    args, training_arguments = parser.parse_known_args()
    run_dir = args.run_dir.resolve()
    launch_path = run_dir / "launch.json"

    if args.action == "status":
        if training_arguments:
            parser.error("status accepts only --run-dir")
        launch = json.loads(launch_path.read_text())
        pid = launch.get("supervisor_pid")
        ticks = launch.get("supervisor_start_ticks")
        launch["supervisor_running"] = bool(pid and ticks and process_start_ticks(pid) == ticks)
        child_pid = launch.get("child_pid")
        child_ticks = launch.get("child_start_ticks")
        launch["child_running"] = bool(child_pid and child_ticks and process_start_ticks(child_pid) == child_ticks)
        for filename, key in (("run.json", "training"), ("metrics.jsonl", "last_step")):
            path = run_dir / filename
            if path.is_file():
                if filename.endswith("jsonl"):
                    with path.open("rb") as handle:
                        handle.seek(max(0, path.stat().st_size - 65536))
                        lines = handle.read().splitlines()
                    for line in reversed(lines):
                        try:
                            launch[key] = json.loads(line)
                            break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # The writer may currently be appending the last line.
                            continue
                else:
                    run = json.loads(path.read_text())
                    launch[key] = {name: run.get(name) for name in
                                   ("status", "profile", "batch", "steps", "microbatches_processed", "completed_epochs",
                                    "gpu_hours", "smoke", "checkpoint_path", "weights_sha256")}
        print(json.dumps(launch, indent=2, sort_keys=True))
        return

    if args.action == "start":
        command = [str(Path(__file__).resolve().with_name("launch.sh")), "--run-dir", str(run_dir), *training_arguments]
        # Resolve and validate CPU-only before creating artifacts or detaching.
        plan_result = subprocess.run([*command, "--dry-run"], text=True, capture_output=True)
        if plan_result.returncode:
            sys.stderr.write(plan_result.stderr)
            raise SystemExit(plan_result.returncode)
        plan = json.loads(plan_result.stdout)
        if "--dry-run" in training_arguments:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return
        # Use the validated runner arguments, including absolute data/source/
        # export/checkpoint paths, before changing cwd for the detached process.
        # The recorded command is exactly the argv passed to the supervisor's child.
        runner = str(Path(__file__).resolve().with_name("train.py"))
        resolved = plan["command"]
        command = [command[0], *resolved[resolved.index(runner) + 1:]]
        if run_dir.exists() and any(run_dir.iterdir()):
            parser.error("Refusing to overwrite a nonempty run directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "launch_request.json", {"command": command, "requested_utc": utc_now(),
                                                    "profile": plan["profile"], "batch": plan["batch"],
                                                    "resolved_torchrun_command": plan["command"],
                                                    "cwd": str(run_dir), "log": str(run_dir / "train.log")})
        try:
            with (run_dir / "train.log").open("ab", buffering=0) as log:
                supervisor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_supervise", "--run-dir", str(run_dir)],
                                              stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True, close_fds=True, cwd=run_dir)
        except OSError as error:
            write_json(launch_path, {"status": "failed", "error": f"{type(error).__name__}: {error}", "ended_utc": utc_now()})
            raise
        # Wait only for the supervisor's startup acknowledgement, not for training.
        deadline = time.monotonic() + 10
        while not launch_path.exists() and supervisor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if not launch_path.exists():
            raise RuntimeError(f"Supervisor did not acknowledge startup; inspect {run_dir / 'train.log'}")
        print(launch_path.read_text(), end="")
        return

    request = json.loads((run_dir / "launch_request.json").read_text())
    launch = {**request, "status": "starting", "supervisor_pid": os.getpid(),
              "supervisor_start_ticks": process_start_ticks(os.getpid()), "started_utc": utc_now()}
    write_json(launch_path, launch)
    try:
        child = subprocess.Popen(request["command"], stdin=subprocess.DEVNULL, cwd=run_dir)
    except OSError as error:
        launch.update(status="failed", error=f"{type(error).__name__}: {error}", ended_utc=utc_now())
        write_json(launch_path, launch)
        raise
    launch.update(status="running", child_pid=child.pid, child_start_ticks=process_start_ticks(child.pid))
    write_json(launch_path, launch)

    def terminate(signum, _frame):
        # Forward terminal/lifecycle signals to the torchrun watchdog. torchrun
        # then terminates its own ranks; the supervisor remains to record exit.
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    code = child.wait()
    launch.update(status="completed" if code == 0 else "failed", exit_code=code, ended_utc=utc_now())
    write_json(launch_path, launch)


if __name__ == "__main__":
    main()
