#!/usr/bin/env python3
"""Train HRM-Text research profiles with audited epochs and token-weighted accumulation.

Use torchrun --standalone --nproc_per_node=8 train.py --source /app/hrm-text
--data /datasets/hrm-text/sampled --run-dir /app/output/attempts/baseline.
--max-steps is diagnostic only and never shortens the configured learning/BP schedule.
Training/model/data/optimizer functions are imported from the upstream checkout.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import shutil
import subprocess
import sys
import time


# Snapshot before argument/config loading or GPU initialization. A later edit
# must not relabel an already-running process with a different harness revision.
RUNNER_SOURCE_SHA256 = {name: hashlib.sha256(Path(__file__).resolve().with_name(name).read_bytes()).hexdigest()
                        for name in ("train.py", "compute_budget.py", "profiles.py", "calibrate.py")
                        if Path(__file__).resolve().with_name(name).is_file()}
RUNNER_SNAPSHOT_UNIX = time.time()


def resolve_batch(size, gpus, microbatch_tokens=None, accumulation_steps=None, global_batch_size=None):
    """Resolve token slots per optimizer update without importing torch."""
    effective = {"B": 188416, "L": 147456, "XL": 188416}[size] if global_batch_size is None else global_batch_size
    if type(effective) is not int or effective <= 0:
        raise ValueError("Effective global batch size must be a positive integer")
    if microbatch_tokens is None and accumulation_steps is None:
        microbatch_tokens = {"B": 23552, "L": 18432, "XL": 11776}[size]
    if microbatch_tokens is not None and microbatch_tokens <= 0:
        raise ValueError("--microbatch-tokens must be positive")
    if accumulation_steps is not None and accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if microbatch_tokens is None:
        divisor = gpus * accumulation_steps
        if effective % divisor:
            raise ValueError("Effective batch must be divisible by GPUs times accumulation steps")
        microbatch_tokens = effective // divisor
    if accumulation_steps is None:
        divisor = gpus * microbatch_tokens
        if effective % divisor:
            raise ValueError("Microbatch times GPUs must divide the effective global batch exactly")
        accumulation_steps = effective // divisor
    if microbatch_tokens * gpus * accumulation_steps != effective:
        raise ValueError("microbatch_tokens * gpus * gradient_accumulation_steps must equal global_batch_size")
    return {"microbatch_tokens": microbatch_tokens, "gradient_accumulation_steps": accumulation_steps,
            "global_batch_size": effective, "global_microbatch_tokens": microbatch_tokens * gpus}


def accumulation_groups(iterator, group_size, skip_microbatches=0):
    """Yield (end position, microbatches, epoch exhausted), including a short tail.

    One-item lookahead distinguishes an epoch boundary from a diagnostic stop.
    Checkpoints are taken only after whole optimizer groups, so skipped records
    always describe an already-committed microbatch prefix.
    """
    if group_size < 1 or skip_microbatches < 0:
        raise ValueError("Invalid accumulation group size or resume position")
    iterator = iter(iterator)
    position = 0
    for _ in range(skip_microbatches):
        try:
            next(iterator)
        except StopIteration:
            raise ValueError("Checkpoint microbatch position exceeds this epoch") from None
        position += 1
    sentinel = object()
    pending = next(iterator, sentinel)
    while pending is not sentinel:
        group = [pending]
        position += 1
        while len(group) < group_size:
            item = next(iterator, sentinel)
            if item is sentinel:
                break
            group.append(item)
            position += 1
        pending = next(iterator, sentinel)
        yield position, group, pending is sentinel


def response_normalization_weights(global_response_counts):
    """Weight upstream microbatch means into the complete group's token mean."""
    counts = [int(count) for count in global_response_counts]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Each training microbatch must contain supervised response tokens")
    total = sum(counts)
    return [count / total for count in counts]


def run_accumulated_update(group, global_response_counts, backward_microbatch, optimizer_update):
    """Apply one token-normalized optimizer/EMA update to a possibly short group."""
    weights = response_normalization_weights(global_response_counts)
    if len(weights) != len(group):
        raise ValueError("Response counts and microbatches must have the same length")
    metrics = None
    for microbatch, weight in zip(group, weights):
        current = backward_microbatch(microbatch, weight)
        if metrics is None:
            metrics = current
        else:
            metrics = {key: (metrics[key][0] + value[0], metrics[key][1] + value[1])
                       for key, value in current.items()}
    optimizer_update()
    return metrics


def restore_carry(saved, initialized):
    """Restore carry devices without moving the runtime's CPU RNG state.

    Resume requires the initial carry's pytree structure, tensor shapes/dtypes,
    and non-tensor leaf types. Scalar values may advance during training.
    """
    import torch
    from torch.utils._pytree import tree_flatten, tree_unflatten

    saved_leaves, saved_spec = tree_flatten(saved)
    initial_leaves, initial_spec = tree_flatten(initialized)
    if saved_spec != initial_spec:
        raise ValueError("Checkpoint carry structure differs from initial_carry")
    restored = []
    for index, (value, initial) in enumerate(zip(saved_leaves, initial_leaves)):
        if torch.is_tensor(initial):
            if not torch.is_tensor(value) or value.shape != initial.shape or value.dtype != initial.dtype:
                raise ValueError(f"Checkpoint carry tensor {index} differs in type, shape or dtype from initial_carry")
            restored.append(value.to(device=initial.device))
        else:
            if type(value) is not type(initial):
                raise ValueError(f"Checkpoint carry leaf {index} differs in type from initial_carry")
            restored.append(value)
    return tree_unflatten(restored, initial_spec)


def export_weight_kind(optimizer):
    """Describe the buffers that the native optimizer's swap_ema will export."""
    has_ema = ["param_ema" in optimizer.state.get(parameter, {})
               for group in optimizer.param_groups for parameter in group["params"]]
    if has_ema and all(has_ema):
        return "EMA"
    return "mixed EMA/model" if any(has_ema) else "model"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("fast-b", "standard-l", "standard-xl", "slow-l", "slow-xl"))
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--calibration", type=Path, help="Operator-measured baseline calibration JSON")
    parser.add_argument("--baseline", choices=("hrm", "transformer"), default="hrm")
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--rdzv-endpoint")
    parser.add_argument("--rdzv-id", default="hrm-text")
    parser.add_argument("--microbatch-tokens", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--source", type=Path, default=Path("/app/hrm-text"))
    parser.add_argument("--data", type=Path, default=Path("/datasets/hrm-text/sampled"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--diagnostic-compute-seconds", type=float, help="Smoke-only compute cap for stopping/resume diagnostics")
    parser.add_argument("--max-steps", type=int, default=0, help="Diagnostic optimizer-step limit; never a scored candidate")
    parser.add_argument("--calibration-steps-per-phase", type=int, default=0, help="Diagnostic timing sweep of the baseline backward phases")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved profile/launch command without importing torch")
    parser.add_argument("--launch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-audit", type=Path, help="Stopped child run.json when recovering before that child saved a checkpoint")
    parser.add_argument("--override", action="append", default=[], help="Hydra recipe override, retained in the audit")
    args = parser.parse_args(argv)
    if args.resume_audit and not args.resume_from:
        parser.error("--resume-audit requires --resume-from")
    if args.max_steps < 0 or args.calibration_steps_per_phase < 0:
        parser.error("Diagnostic step counts must be nonnegative")
    if args.calibration_steps_per_phase and (args.calibration_steps_per_phase < 4 or args.resume_from or args.max_steps or args.override or args.baseline != "hrm"):
        parser.error("Baseline calibration needs at least 4 updates per phase and no resume, overrides, transformer control or max-steps")
    from profiles import resolve_profile
    active = json.loads(Path(__file__).with_name("profile.json").read_text()) if os.environ.get("HRM_ENFORCE_PROFILE") == "1" else None
    args.track = args.track or (active["track"] if active else "fast-b")
    args.gpus_per_node = args.gpus_per_node if args.gpus_per_node is not None else (active["gpus_per_node"] if active else 8)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else (active.get("calibration") if active else None)
    try:
        args.profile = resolve_profile(args.track, gpus_per_node=args.gpus_per_node, calibration=calibration)
    except ValueError as error:
        parser.error(str(error))
    if active is not None and args.profile != active:
        parser.error("Requested settings differ from the operator-sealed /task-tools/profile.json")
    if args.diagnostic_compute_seconds is not None:
        if (not args.max_steps or args.calibration_steps_per_phase or not math.isfinite(args.diagnostic_compute_seconds)
                or not 0 < args.diagnostic_compute_seconds <= args.profile["compute_budget_seconds"]):
            parser.error("Diagnostic compute cap requires --max-steps and must be positive and no larger than the profile budget")
    args.compute_limit = args.diagnostic_compute_seconds if args.diagnostic_compute_seconds is not None else args.profile["compute_budget_seconds"]
    args.size, args.gpus, args.epochs, args.nodes = (args.profile[key] for key in ("size", "gpus", "epochs", "nodes"))
    if args.baseline == "transformer" and args.size != "B":
        parser.error("The published transformer control uses the fast B track")
    framework = bool(os.environ.get("RSI_MULTINODE_ROOT"))
    if framework and (args.node_rank is not None or args.master_addr or args.rdzv_endpoint):
        parser.error("The framework torchrun shim owns node rank and rendezvous settings")
    if args.node_rank is not None and not 0 <= args.node_rank < args.nodes:
        parser.error("--node-rank must identify a node in the selected track")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be a valid TCP port")
    if args.nodes > 1 and not framework and not (args.rdzv_endpoint or args.master_addr) and not args.dry_run:
        parser.error("Multi-node launch requires --rdzv-endpoint or --master-addr on every node")
    if args.nodes > 1 and not framework and args.node_rank is None and not args.dry_run:
        parser.error("Multi-node launch requires explicit --node-rank on each node")
    explicit_global = None
    for override in args.override:
        key, _, value = override.lstrip("+").partition("=")
        if key == "global_batch_size":
            try:
                explicit_global = int(value)
            except ValueError:
                parser.error("global_batch_size must be an explicit integer for batch accounting")
    try:
        args.batch = resolve_batch(args.size, args.gpus, args.microbatch_tokens,
                                   args.gradient_accumulation_steps, explicit_global)
    except ValueError as error:
        parser.error(str(error))
    args.microbatch_tokens = args.batch["microbatch_tokens"]
    args.gradient_accumulation_steps = args.batch["gradient_accumulation_steps"]
    if args.run_dir is None:
        if not args.dry_run:
            parser.error("--run-dir is required except with --dry-run")
        args.run_dir = Path("/app/output/attempts") / args.profile["id"]
    args.source, args.data, args.run_dir = args.source.resolve(), args.data.resolve(), args.run_dir.resolve()
    args.export_dir = (args.export_dir or args.run_dir / "submission").resolve()
    if args.calibration:
        args.calibration = args.calibration.resolve()
    return args


def launch_command(args):
    command = ["timeout", "--signal=TERM", "--kill-after=60s", f"{args.profile['attempt_wall_seconds']}s",
               "torchrun", f"--nnodes={args.nodes}", f"--nproc_per_node={args.gpus_per_node}"]
    if not os.environ.get("RSI_MULTINODE_ROOT"):
        if args.nodes == 1:
            command.append("--standalone")
        else:
            command.append(f"--node_rank={args.node_rank if args.node_rank is not None else 0}")
            if args.rdzv_endpoint:
                command += ["--rdzv_backend=c10d", f"--rdzv_endpoint={args.rdzv_endpoint}", f"--rdzv_id={args.rdzv_id}"]
            else:
                command += [f"--master_addr={args.master_addr or 'MASTER_HOST'}", f"--master_port={args.master_port}"]
    command += [str(Path(__file__).resolve()), "--track", args.track, "--gpus-per-node", str(args.gpus_per_node),
                "--baseline", args.baseline, "--microbatch-tokens", str(args.microbatch_tokens),
                "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
                "--source", str(args.source), "--data", str(args.data), "--run-dir", str(args.run_dir), "--export-dir", str(args.export_dir)]
    # Retain native rendezvous intent when detached launch.sh reconstructs argv.
    if not os.environ.get("RSI_MULTINODE_ROOT") and args.nodes > 1:
        command += ["--node-rank", str(args.node_rank if args.node_rank is not None else 0), "--master-port", str(args.master_port)]
        if args.master_addr:
            command += ["--master-addr", args.master_addr]
        if args.rdzv_endpoint:
            command += ["--rdzv-endpoint", args.rdzv_endpoint, "--rdzv-id", args.rdzv_id]
    for option, value in (("--max-steps", args.max_steps), ("--calibration-steps-per-phase", args.calibration_steps_per_phase),
                          ("--calibration", args.calibration), ("--resume-from", args.resume_from), ("--resume-audit", args.resume_audit),
                          ("--diagnostic-compute-seconds", args.diagnostic_compute_seconds)):
        if value:
            command += [option, str(value.resolve() if isinstance(value, Path) else value)]
    if args.verify_checkpoint:
        command.append("--verify-checkpoint")
    for override in args.override:
        command += ["--override", override]
    return command


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkout_identity(source, source_paths):
    """Calibration uses the pinned tracked recipe and no untracked Python code."""
    try:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        clean = subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=False).returncode == 0
        tracked = set(subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"], text=True).split("\0"))
        clean = clean and all(path.relative_to(source).as_posix() in tracked for path in source_paths)
        return revision, clean
    except subprocess.CalledProcessError:
        return None, False


def architecture_sources(source, max_source_bytes=None, exclude_paths=()):
    """Validate the small pure-Python architecture snapshot before allocating GPUs."""
    excluded = {".git", ".venv", "venv", "__pycache__", "evaluation", "tests", "test",
                "checkpoints", "wandb", "data", "outputs", "build", "dist", ".cache"}
    paths, total_bytes = [], 0
    excluded_roots = [path.resolve() for path in exclude_paths]
    for directory, subdirectories, filenames in os.walk(source, followlinks=False):
        if any(Path(directory).resolve().is_relative_to(path) for path in excluded_roots):
            subdirectories[:] = []
            continue
        subdirectories[:] = sorted(name for name in subdirectories if name not in excluded and not name.startswith(".")
                                   and not any((Path(directory) / name).resolve().is_relative_to(path) for path in excluded_roots))
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            if not path.resolve().is_relative_to(source.resolve()):
                raise ValueError(f"Architecture source escapes the checkout: {path}")
            total_bytes += path.stat().st_size
            if max_source_bytes is not None and total_bytes > max_source_bytes:
                raise ValueError(f"Architecture sources exceed the {max_source_bytes}-byte profile limit")
            paths.append(path)
    if not any(path.relative_to(source).as_posix().startswith("models/") for path in paths):
        raise ValueError("Exported architecture contains no models/*.py sources")
    return paths


def export_architecture(source, export_dir, arch, max_source_bytes=None, exclude_paths=(), data_config=None):
    """Snapshot the inference ABI; evaluation precision is always frozen bf16."""
    paths = architecture_sources(source, max_source_bytes, (*exclude_paths, export_dir))
    target = export_dir / "source"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    hashes = {}
    for path in paths:
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[relative.as_posix()] = file_sha256(destination)
    manifest = {"schema_version": 1, "arch": arch, "data_config": data_config or {}, "forward_dtype": "bfloat16",
                "source_files": dict(sorted(hashes.items()))}
    write_json(export_dir / "model.json", manifest)
    return manifest


def selected_schedule_tokens(raw_metadata, manifest, epochs):
    """Estimate optimizer-step scheduling from only the selected epoch prefix.

    Lengths include the upstream autoregressive shift convention. Packing,
    dropped final microbatches and short accumulated tails affect actual steps.
    """
    if manifest is not None:
        lengths = {row["epoch"]: row["tokens_including_ar_shift"]
                   for row in manifest.get("validation", {}).get("epochs", [])}
        if not all(epoch in lengths for epoch in range(epochs)):
            raise ValueError("Data manifest does not describe every selected epoch")
        selected = [int(lengths[epoch]) for epoch in range(epochs)]
        return {"tokens": sum(selected), "basis": "selected_epoch_manifest_raw_lengths",
                "epoch_raw_tokens": selected}
    return {"tokens": int(raw_metadata["total_length"]) * epochs,
            "basis": "metadata_mean_length_fallback", "epoch_raw_tokens": None}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_start_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def checkpoint_position(saved=None):
    """Initialize final-audit position even when a terminal resume runs no loop."""
    if saved is None:
        return 1, 0, False
    epoch, batch = saved["epoch"], saved["batch_in_epoch"]
    complete = saved.get("epoch_complete", saved.get("completed_epochs") == epoch)
    if type(epoch) is not int or epoch < 1 or type(batch) is not int or batch < 0 or type(complete) is not bool:
        raise ValueError("Invalid retained checkpoint epoch position")
    return epoch, batch, complete


def comparable_resume_config(config):
    return {key: value for key, value in config.items()
            if key not in ("checkpoint_path", "project_name", "run_name", "log_interval")}


def validate_resume_audit(checkpoint, audit_path, saved, previous, launch, profile, data_manifest_sha256):
    """Bind an explicit stopped-child receipt to the checkpoint being restored."""
    if not previous:
        raise ValueError("Explicit resume audit is missing")
    checkpoint, audit_path = Path(checkpoint).resolve(), Path(audit_path).resolve()
    recorded_parent = previous.get("resume_from")
    direct_child = bool(recorded_parent and Path(recorded_parent).resolve() == checkpoint)
    if checkpoint.parent != audit_path.parent and not direct_child:
        raise ValueError("Resume audit is not the checkpoint owner or its direct resumed child")
    launch = launch or {}
    supervisor_terminal = launch.get("status") in ("completed", "failed") and bool(launch.get("ended_utc"))
    audit_terminal = (previous.get("status") in ("completed", "budget_exhausted", "failed")
                      and type(previous.get("end_unix")) in (int, float)
                      and math.isfinite(previous["end_unix"]))
    if not (supervisor_terminal or audit_terminal):
        raise ValueError("Explicit resume audit requires a terminal supervisor or run receipt")
    if previous.get("profile") != profile or saved.get("profile") != profile:
        raise ValueError("Explicit resume audit has a different task profile")
    if (not data_manifest_sha256 or previous.get("data_manifest_sha256") != data_manifest_sha256
            or saved.get("data_manifest_sha256") != data_manifest_sha256):
        raise ValueError("Explicit resume audit has a different or missing data manifest")
    if previous.get("clock") != profile["clock"] or saved.get("clock") != profile["clock"]:
        raise ValueError("Explicit resume audit has a different compute clock")
    if (not isinstance(previous.get("config"), dict) or not previous["config"]
            or not isinstance(saved.get("config"), dict) or not saved["config"]
            or comparable_resume_config(previous["config"]) != comparable_resume_config(saved["config"])):
        raise ValueError("Explicit resume audit has a different model, data or optimizer configuration")


def resume_accounting(saved, previous, launch=None, now=None):
    """Charge a whole stopped invocation, even when rolling back an older checkpoint.

    A terminal supervisor receipt is preferred for interrupted runs. Without one,
    elapsed time through resume is charged conservatively, including downtime.
    This deliberately never treats checkpoint-time usage as final invocation usage.
    """
    if previous is None or "start_unix" not in previous:
        raise ValueError("Resume requires the parent run.json for complete invocation accounting")
    launch = launch or {}
    identities = [(previous.get("rank0_pid"), previous.get("rank0_start_ticks"), previous.get("rank0_hostname")),
                  (launch.get("supervisor_pid"), launch.get("supervisor_start_ticks"), launch.get("hostname")),
                  (launch.get("child_pid"), launch.get("child_start_ticks"), launch.get("hostname"))]
    terminal = previous.get("status") in ("completed", "budget_exhausted", "failed") or "ended_utc" in launch
    for pid, ticks, host in identities:
        if host and host != socket.gethostname():
            if not terminal:
                raise RuntimeError("Remote parent requires a terminal supervisor/run receipt before resuming")
            continue
        if pid and ticks and process_start_ticks(pid) == ticks:
            raise RuntimeError(f"Cannot resume while parent process {pid} is alive")
    if previous.get("status") not in ("completed", "budget_exhausted", "failed") and not any(pid and ticks for pid, ticks, _ in identities):
        raise RuntimeError("Cannot establish that the interrupted parent stopped: no recorded process identity")
    inherited = float(previous.get("prior_gpu_hours", saved.get("invocation_prior_gpu_hours", 0.0)))
    lower_bound = max(float(saved["gpu_hours"]), float(previous.get("gpu_hours", 0)))
    if "ended_utc" in launch:
        end = datetime.datetime.fromisoformat(launch["ended_utc"]).timestamp()
        method = "terminal_supervisor_receipt_entire_invocation"
    elif previous.get("status") in ("completed", "budget_exhausted", "failed") and "end_unix" in previous and "gpu_hours" in previous:
        end = float(previous["end_unix"])
        method = "terminal_run_audit_entire_invocation"
    else:
        end = time.time() if now is None else now
        method = "conservative_elapsed_through_resume_including_downtime"
    elapsed = end - float(previous["start_unix"])
    if elapsed < 0:
        raise ValueError("Parent accounting receipt ends before its training start")
    charged = max(lower_bound, inherited + int(previous["world_size"]) * elapsed / 3600)
    if not math.isfinite(charged) or charged < 0:
        raise ValueError("Invalid parent GPU-hour receipt")
    return {"gpu_hours": charged, "method": method,
            "checkpoint_step": saved["step"], "parent_last_recorded_step": previous.get("steps", saved["step"]),
            "discarded_steps_at_least": max(0, previous.get("steps", saved["step"]) - saved["step"]),
            "note": "Discarded/replayed updates remain charged; token counters describe the retained checkpoint trajectory."}


def resume_compute_accounting(saved, previous, journal=None, launch=None, now=None):
    """Retain all measured ancestor work, including updates discarded by rollback.

    An interrupted update lacks a complete synchronized measurement. Charge its
    entire elapsed wall interval conservatively rather than inventing exclusions.
    The journal is written before and after every update, outside the compute
    clock. A successful older-checkpoint rollback retains the parent's full bill.
    """
    if previous is None or "compute_seconds" not in previous or "compute_seconds" not in saved:
        raise ValueError("This compute-clock track cannot resume a legacy wall-only checkpoint")
    journal, launch = journal or {}, launch or {}
    candidates = [saved["compute_seconds"], previous["compute_seconds"], journal.get("compute_seconds", 0)]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in candidates):
        raise ValueError("Invalid ancestor compute receipt")
    charged = max(candidates)
    method = "complete_measured_update_journal"
    uncertain = 0.0
    if journal.get("inflight"):
        end = (datetime.datetime.fromisoformat(launch["ended_utc"]).timestamp() if launch.get("ended_utc")
               else (time.time() if now is None else now))
        uncertain = max(0.0, end - float(journal["update_started_unix"]))
        charged = max(charged, float(journal["compute_seconds"]) + uncertain)
        method = "measured_history_plus_conservative_interrupted_update_wall"
    elif not journal and previous.get("status") not in ("completed", "budget_exhausted", "failed"):
        raise ValueError("Interrupted compute-clock run is missing its per-update journal")
    phase_max = {}
    for receipt in (saved, previous, journal):
        history = receipt.get("compute_phase_max_seconds", receipt.get("compute_reserve", {}).get("phase_max_seconds", {}))
        for phase, seconds in history.items():
            if not isinstance(phase, str) or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid ancestor phase timing history")
            phase_max[phase] = max(phase_max.get(phase, 0), seconds)
    return {"compute_seconds": charged, "phase_max_seconds": phase_max, "method": method, "uncertain_interrupted_seconds": uncertain,
            "checkpoint_step": saved["step"], "parent_last_measured_step": journal.get("step", previous.get("steps", saved["step"])),
            "discarded_steps_at_least": max(0, journal.get("step", previous.get("steps", saved["step"])) - saved["step"])}


def main():
    args = arguments()
    if args.dry_run:
        print(json.dumps({"profile": args.profile, "batch": args.batch, "command": launch_command(args),
                          "default_learning_rate": args.profile["baseline_lr"], "compute_budget_seconds": args.compute_limit,
                          "diagnostic_only": bool(args.max_steps or args.calibration_steps_per_phase)}, indent=2, sort_keys=True))
        return
    if args.launch:
        os.environ["HRM_LAUNCH_STARTED_UNIX"] = str(time.time())
        command = launch_command(args)
        os.execvpe(command[0], command, os.environ)
    raw_metadata = json.loads((args.data / "metadata.json").read_text())
    data_manifest_path = next((args.data / name for name in ("data_manifest.json", "manifest.json", "sampling_manifest.json", "rsi_manifest.json")
                               if (args.data / name).is_file()), None)
    manifest = json.loads(data_manifest_path.read_text()) if data_manifest_path else None
    schedule = selected_schedule_tokens(raw_metadata, manifest, args.epochs)
    source_paths = architecture_sources(args.source, args.profile["max_source_bytes"], (args.run_dir, args.export_dir))
    revision, source_clean = checkout_identity(args.source, source_paths)
    if args.calibration_steps_per_phase:
        from profiles import HRM_COMMIT
        if revision != HRM_COMMIT or not source_clean:
            raise ValueError("Calibration requires the clean pinned HRM checkout and configuration")
    if args.microbatch_tokens < raw_metadata["max_seq_len"] - 1:
        raise ValueError("Microbatch must hold the dataset's longest sequence without truncation")
    for epoch_id in range(args.epochs):
        for name in ("inst_start", "inst_len", "resp_start", "resp_len"):
            if not (args.data / f"epoch_{epoch_id}" / f"{name}.npy").is_file():
                raise FileNotFoundError(f"Selected epoch {epoch_id} is not fully staged")
    os.environ.setdefault("WANDB_MODE", "offline")
    if os.environ["WANDB_MODE"] not in ("offline", "disabled"):
        raise ValueError("This runner requires offline/disabled W&B")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_from and any((args.run_dir / name).exists() for name in ("run.json", "metrics.jsonl")):
        raise FileExistsError("Run directory already contains a run; choose a new --run-dir or explicitly --resume-from")
    resume_receipt = None
    compute_receipt = None
    if args.resume_from:
        previous_audit_path = (args.resume_audit or args.resume_from.resolve().parent / "run.json").resolve()
        previous_run_audit = json.loads(previous_audit_path.read_text()) if previous_audit_path.is_file() else None
        previous_launch_path = previous_audit_path.with_name("launch.json")
        previous_launch = json.loads(previous_launch_path.read_text()) if previous_launch_path.is_file() else None
        saved_progress = json.loads((args.resume_from / "progress.json").read_text())
        # Reject live parents before touching an existing same-directory audit.
        if int(os.environ.get("RANK", 0)) == 0:
            if args.resume_audit:
                validate_resume_audit(args.resume_from, previous_audit_path, saved_progress, previous_run_audit,
                                      previous_launch, args.profile, file_sha256(data_manifest_path) if data_manifest_path else None)
            resume_receipt = resume_accounting(saved_progress, previous_run_audit, previous_launch)
            journal_path = previous_audit_path.with_name("compute_journal.json")
            journal = json.loads(journal_path.read_text()) if journal_path.is_file() else None
            compute_receipt = resume_compute_accounting(saved_progress, previous_run_audit, journal, previous_launch)
    os.environ.setdefault("WANDB_DIR", str(args.run_dir))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.run_dir.parent / "inductor-cache"))
    sys.path.insert(0, str(args.source))

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_optimizer_state_dict,
    )
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    import pretrain as upstream

    from compute_budget import CLOCK, CompilationTracker, ComputeBudget, retained_checkpoint_stop
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != args.gpus:
        raise ValueError(f"Profile requires {args.gpus} GPUs; torchrun initialized {world_size}")
    if int(os.environ.get("LOCAL_WORLD_SIZE", args.gpus_per_node)) != args.gpus_per_node:
        raise ValueError("Local torchrun world size differs from the selected profile")
    # Checkpoint plans are Python/CPU metadata; a separate Gloo group avoids
    # NCCL GPU staging allocations when the reference model nearly fills VRAM.
    checkpoint_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.profile["checkpoint_grace_seconds"]))
    start_time = time.time()
    launch_start_time = float(os.environ.get("HRM_LAUNCH_STARTED_UNIX", start_time))
    receipts = [(resume_receipt, compute_receipt) if rank == 0 else None]
    dist.broadcast_object_list(receipts, src=0)
    resume_receipt, compute_receipt = receipts[0]
    overrides = [
        f"arch/size@arch={args.size}", f"lr={args.profile['baseline_lr']}", f"ema={args.profile['ema']}",
        f"global_batch_size={args.batch['global_batch_size']}",
        f"epochs={args.epochs}", f"data.path={args.data}",
        "+project_name=RSI-HRM-Text", f"+run_name={args.run_dir.name}",
        f"+checkpoint_path={args.run_dir}", "+log_interval=1",
    ] + (["arch/net@arch=transformer", "arch.n_layers=34"] if args.baseline == "transformer" else []) + args.override
    with initialize_config_dir(config_dir=str(args.source / "config"), version_base=None):
        hydra_config = compose(config_name="cfg_pretrain", overrides=overrides)
    config = upstream.PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))
    if Path(config.data.path).resolve() != args.data:
        raise ValueError("Specify data with --data so the manifest audit identifies the consumed dataset")
    if config.resume_from is not None:
        raise ValueError("Use the runner's --resume-from to preserve progress and cumulative resource accounting")
    if config.epochs != args.epochs:
        raise ValueError("Select epochs with --track so the task profile and training schedule agree")
    if config.global_batch_size != args.batch["global_batch_size"]:
        raise ValueError("Resolved global batch disagrees with the audited microbatch/accumulation product")
    if args.calibration_steps_per_phase:
        from calibrate import validate_baseline_recipe
        validate_baseline_recipe(config.model_dump(), args.profile, args.batch)
    torch.random.manual_seed(config.seed + rank)
    # The effective batch determines LR/BP schedules; the physical microbatch
    # determines data packing and activation memory.
    loader, metadata = upstream.create_dataloader(config, args.microbatch_tokens, True, rank, world_size)
    model, carry, optim = upstream.create_model_and_carry(config, metadata, args.microbatch_tokens)
    state = upstream.TrainState(model=model, carry=carry, optim=optim, step=0,
                               total_steps=schedule["tokens"] // config.global_batch_size)
    parameter_count = sum(parameter.numel() for parameter in state.model.parameters())
    torch.cuda.synchronize()

    compiler = CompilationTracker()
    from torch._dynamo.callback import callback_handler
    callback_handler.register_start_callback(compiler.on_compile_start)
    callback_handler.register_end_callback(compiler.on_compile_end)

    @torch.compile(dynamic=False)
    def accumulate_microbatch(train_state, batch, response_weight, **extra):
        train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **extra)
        (loss * response_weight).backward()
        return metrics

    @torch.compile(dynamic=False)
    def accumulated_optimizer_update(train_state):
        train_state.optim.step()
        train_state.optim.zero_grad()

    package_names = ["torch", "flash-attn-3", "numpy", "numba", "hydra-core", "pydantic", "safetensors"]
    package_versions = {name: importlib.metadata.version(name) for name in package_names}
    data_hashes = {"metadata.json": file_sha256(args.data / "metadata.json")}
    data_manifest_sha256 = None
    for name in ("data_manifest.json", "manifest.json", "sampling_manifest.json", "rsi_manifest.json"):
        if (args.data / name).is_file():
            data_hashes[name] = file_sha256(args.data / name)
            if data_manifest_sha256 is None:
                data_manifest_sha256 = data_hashes[name]
    identity = {"manifest": data_manifest_sha256, "metadata": data_hashes["metadata.json"],
                "source": {str(path.relative_to(args.source)): file_sha256(path)
                           for path in architecture_sources(args.source, args.profile["max_source_bytes"], (args.run_dir, args.export_dir))}}
    identities = [None] * world_size
    dist.all_gather_object(identities, identity)
    if any(item != identity for item in identities):
        raise ValueError("All nodes must use identical source and sampled-data manifests")
    if args.profile["calibration"] and (args.profile["calibration"]["data_manifest_sha256"] != data_manifest_sha256
                                         or args.profile["calibration"]["hrm_commit"] != revision):
        raise ValueError("Baseline calibration does not match the current source/data")
    properties = torch.cuda.get_device_properties(local_rank)
    hardware = {"rank": rank, "hostname": socket.gethostname(), "gpu_name": properties.name,
                "total_memory_bytes": properties.total_memory, "capability": [properties.major, properties.minor]}
    hardware_all = [None] * world_size
    dist.all_gather_object(hardware_all, hardware)
    hardware_audit = {"gpu_name": properties.name, "nodes": args.nodes, "gpus_per_node": args.gpus_per_node, "ranks": hardware_all}
    if args.profile["calibration"]:
        from calibrate import hardware_signature
        if hardware_signature(hardware_audit) != hardware_signature(args.profile["calibration"]["hardware"]):
            raise ValueError("Baseline calibration does not match the current GPU hardware/topology")
    audit = {
        "schema_version": 3, "status": "running", "source_commit": revision, "source_clean": source_clean,
        "runner_source_sha256": RUNNER_SOURCE_SHA256, "runner_snapshot_unix": RUNNER_SNAPSHOT_UNIX,
        "profile": args.profile, "batch": args.batch, "baseline": args.baseline,
        "source": str(args.source), "data": str(args.data), "data_manifest_sha256": data_manifest_sha256,
        "data_file_hashes": data_hashes,
        "config": config.model_dump(), "train_metadata": metadata.model_dump(),
        "world_size": world_size, "gpu_name": torch.cuda.get_device_name(),
        "hardware": hardware_audit,
        "rank0_hostname": socket.gethostname(),
        "clock": CLOCK, "compute_seconds": compute_receipt["compute_seconds"] if compute_receipt else 0.0,
        "prior_compute_seconds": compute_receipt["compute_seconds"] if compute_receipt else 0.0,
        "compute_phase_max_seconds": compute_receipt["phase_max_seconds"] if compute_receipt else {},
        "compute_budget_seconds": args.compute_limit, "diagnostic_compute_seconds": args.diagnostic_compute_seconds,
        "compilation_seconds": 0.0, "checkpoint_seconds": 0.0, "checkpoint_metadata_backend": "gloo",
        "clock_note": "All-rank maximum synchronized forward/backward/optimizer elapsed time minus local TorchDynamo compilation callback intervals; includes dispatch and communication, not pure GPU kernel seconds. Data transfer, loader, metrics, checkpoints and export are outside the measured region.",
        "python": platform.python_version(), "packages": package_versions,
        "num_params": parameter_count, "schedule_total_steps": state.total_steps,
        "schedule_token_estimate": schedule,
        "data_policy": {"sampled_epoch_ids": list(range(args.epochs)),
                        "drop_last_batch": True, "partial_accumulation_groups": "train",
                        "schedule_basis": schedule["basis"],
                        "schedule_note": "Raw token estimate; packing and dropped final microbatches affect actual optimizer steps."},
        "max_steps": args.max_steps, "smoke": bool(args.max_steps or args.calibration_steps_per_phase),
        "calibration_steps_per_phase": args.calibration_steps_per_phase,
        "start_unix": start_time, "resume_from": str(args.resume_from) if args.resume_from else None,
        "resume_audit": str(previous_audit_path) if args.resume_from else None,
        "resume_audit_sha256": file_sha256(previous_audit_path) if args.resume_from else None,
        "prior_gpu_hours": resume_receipt["gpu_hours"] if resume_receipt else 0.0,
        "gpu_hours": (resume_receipt["gpu_hours"] if resume_receipt else 0.0) + world_size * (time.time() - start_time) / 3600,
        "rank0_pid": os.getpid() if rank == 0 else None,
        "rank0_start_ticks": process_start_ticks(os.getpid()) if rank == 0 else None,
        "torch_compile_enabled": True, "wandb_mode": os.environ["WANDB_MODE"],
        "provenance_note": "Runner output is an audit aid; evaluation requires independent trusted resource/data accounting.",
    }
    if rank == 0:
        upstream.wandb.init(project=config.project_name, name=config.run_name,
                            config=config.model_dump() | {"train_metadata": metadata.model_dump()},
                            settings=upstream.wandb.Settings(_disable_stats=True))
        upstream.save_code_and_config(config, metadata)
        write_json(args.run_dir / "run.json", audit)
        # Write inherited usage before checkpoint loading, which itself can fail.
        write_json(args.run_dir / "compute_journal.json", {"clock": CLOCK, "inflight": False,
                   "compute_seconds": audit["prior_compute_seconds"], "compute_phase_max_seconds": audit["compute_phase_max_seconds"],
                   "step": saved_progress["step"] if args.resume_from else 0, "measured_unix": time.time()})
        print("RSI_INIT " + json.dumps(audit, sort_keys=True), flush=True)

    def checkpoint(epoch, batch_in_epoch, epoch_complete):
        checkpoint_started = time.perf_counter()
        path = args.run_dir / f"checkpoint_step_{state.step:08d}"
        dcp.save({"model": state.model.state_dict(), "optim": get_optimizer_state_dict(state.model, state.optim)}, checkpoint_id=path, process_group=checkpoint_group)
        torch.save({"cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "carry": state.carry},
                   path / f"rank_{rank}_runtime.pt")
        if rank == 0:
            write_json(path / "progress.json", {"step": state.step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                                                "microbatches_in_epoch": batch_in_epoch,
                                                "microbatches_processed": prior_microbatches + microbatches,
                                                "epoch_complete": epoch_complete,
                                                "completed_epochs": epoch if epoch_complete else epoch - 1,
                                                "next_epoch": epoch + 1 if epoch_complete else epoch,
                                                "next_microbatch": 0 if epoch_complete else batch_in_epoch,
                                                "profile": args.profile, "batch": args.batch,
                                                "schedule_total_steps": state.total_steps,
                                                "compute_seconds": budget.used, "compute_phase_max_seconds": dict(budget.phase_max),
                                                "clock": CLOCK,
                                                "prior_compute_seconds": audit["prior_compute_seconds"],
                                                "gpu_hours": prior_gpu_hours + world_size * (time.time() - start_time) / 3600,
                                                "invocation_prior_gpu_hours": prior_gpu_hours,
                                                "token_slots_processed": prior_token_slots + token_slots,
                                                "packed_tokens_processed": prior_real_tokens + real_tokens,
                                                "smoke": audit["smoke"], "data_manifest_sha256": data_manifest_sha256,
                                                "config": config.model_dump()})
        dist.barrier()
        audit["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
        if rank == 0:
            print("RSI_CHECKPOINT_SAVED " + str(path), flush=True)
        return path

    def restore(path):
        optim_state = get_optimizer_state_dict(state.model, state.optim)
        dcp.load({"model": state.model.state_dict(), "optim": optim_state}, checkpoint_id=path, process_group=checkpoint_group)
        set_optimizer_state_dict(state.model, state.optim, optim_state)
        runtime = torch.load(path / f"rank_{rank}_runtime.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(runtime["cpu_rng"])
        torch.cuda.set_rng_state(runtime["cuda_rng"])
        state.carry = restore_carry(runtime["carry"], carry)
        return json.loads((path / "progress.json").read_text())

    resume_epoch, resume_batch = 1, 0
    prior_gpu_hours, prior_token_slots, prior_real_tokens, prior_microbatches = 0.0, 0, 0, 0
    resume_rng = None
    if args.resume_from:
        saved = restore(args.resume_from.resolve())
        if saved["schedule_total_steps"] != state.total_steps:
            raise ValueError("Resume requires unchanged total training schedule")
        if saved["data_manifest_sha256"] != data_manifest_sha256:
            raise ValueError("Resume requires the identical data manifest")
        if "profile" in saved and saved["profile"] != args.profile:
            raise ValueError("Resume requires the identical task profile")
        if "batch" in saved and saved["batch"] != args.batch:
            raise ValueError("Resume requires identical microbatch and accumulation settings")
        if "batch" not in saved and (args.gradient_accumulation_steps != 1 or previous_run_audit["world_size"] != world_size):
            raise ValueError("Legacy checkpoints require their original non-accumulated world size")
        if comparable_resume_config(saved["config"]) != comparable_resume_config(config.model_dump()):
            raise ValueError("Resume requires unchanged model, data, and optimizer configuration")
        prior_gpu_hours = resume_receipt["gpu_hours"]
        audit["resume_accounting"] = resume_receipt
        prior_token_slots, prior_real_tokens = saved["token_slots_processed"], saved["packed_tokens_processed"]
        prior_microbatches = saved.get("microbatches_processed", saved["step"])
        audit["smoke"] = audit["smoke"] or saved["smoke"]
        state.step = saved["step"]
        completed_boundary = saved.get("epoch_complete", not saved.get("smoke", False))
        resume_epoch = saved.get("next_epoch", saved["epoch"] + int(completed_boundary))
        resume_batch = saved.get("next_microbatch", 0 if completed_boundary else saved["batch_in_epoch"])
        if not completed_boundary and resume_batch % args.gradient_accumulation_steps:
            raise ValueError("Checkpoint is not at a completed optimizer-group boundary")
        resume_rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
        if args.max_steps and args.max_steps <= state.step:
            raise ValueError("--max-steps must exceed the checkpoint step when resuming")
        # Worker-local dataset index is incremented once for every iterator/epoch.
        loader.dataset._epoch = resume_epoch - 1
    prior_compute = compute_receipt["compute_seconds"] if compute_receipt else 0.0
    initial_step_bound = args.profile["baseline_estimate_seconds"] / max(1, state.total_steps) * 3
    budget = ComputeBudget(args.compute_limit, prior_compute, initial_step_bound,
                           phase_max=compute_receipt["phase_max_seconds"] if compute_receipt else None)
    if budget.used > budget.limit:
        raise RuntimeError("The experiment already exceeds its cumulative active-training budget")
    audit["compute_resume_accounting"] = compute_receipt
    # Independent wall watchdog includes compilation and checkpoint/export grace.
    remaining_wall = args.profile["attempt_wall_seconds"] - (time.time() - launch_start_time)
    if remaining_wall <= 0:
        raise RuntimeError("The wall watchdog expired during initialization")
    signal.alarm(max(1, math.floor(remaining_wall)))
    audit["prior_gpu_hours"] = prior_gpu_hours
    initial_step = state.step
    measured_steps = []
    measured_token_slots = []
    token_slots = 0
    real_tokens = 0
    microbatches = 0
    checkpoint_path = args.resume_from.resolve() if args.resume_from else None
    epoch, batch_in_epoch, epoch_complete = checkpoint_position(saved if args.resume_from else None)
    state.step += 1
    phase = state.model.compute_train_extra_args(state).get("bp_steps", 0)
    state.step -= 1
    terminal_stop = retained_checkpoint_stop(budget, phase, has_checkpoint=checkpoint_path is not None,
                                             completed_epochs=epoch if epoch_complete else epoch - 1,
                                             requested_epochs=config.epochs)
    stop_reason = terminal_stop or "epochs_completed"
    compilation_seconds = 0.0
    calibration_phases = list(range(int(getattr(config.arch, "bp_min_steps", 2)), int(getattr(config.arch, "bp_max_steps", 6)) + 1))
    calibration_limit = len(calibration_phases) * args.calibration_steps_per_phase

    def update_running_audit(epoch, batch_in_epoch, epoch_complete=False):
        if rank == 0:
            audit.update({"status": "running", "steps": state.step, "epoch": epoch,
                          "batch_in_epoch": batch_in_epoch, "microbatches_in_epoch": batch_in_epoch,
                          "microbatches_processed": prior_microbatches + microbatches,
                          "completed_epochs": epoch if epoch_complete else epoch - 1,
                          "prior_gpu_hours": prior_gpu_hours,
                          "compute_seconds": budget.used, "compute_phase_max_seconds": dict(budget.phase_max), "compilation_seconds": compilation_seconds,
                          "gpu_hours": prior_gpu_hours + world_size * (time.time() - start_time) / 3600,
                          "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                          "token_slots_processed": prior_token_slots + token_slots,
                          "packed_tokens_processed": prior_real_tokens + real_tokens})
            write_json(args.run_dir / "run.json", audit)

    update_running_audit(epoch, batch_in_epoch, epoch_complete)
    for epoch in (() if terminal_stop else range(resume_epoch, config.epochs + 1)):
        state.model.train()
        batch_in_epoch = resume_batch if epoch == resume_epoch else 0
        epoch_complete = False
        groups = accumulation_groups(loader, args.gradient_accumulation_steps, batch_in_epoch)
        for next_batch, group, next_epoch_complete in groups:
            if resume_rng is not None:
                torch.set_rng_state(resume_rng[0])
                torch.cuda.set_rng_state(resume_rng[1])
                resume_rng = None
            # Compute the next native schedule without committing an update.
            state.step += 1
            lr = upstream.update_lr(config, state)
            extra = state.model.compute_train_extra_args(state)
            state.step -= 1
            if args.calibration_steps_per_phase:
                extra["bp_steps"] = calibration_phases[(state.step - initial_step) // args.calibration_steps_per_phase]
            phase = extra.get("bp_steps", 0)
            remaining_wall = args.profile["attempt_wall_seconds"] - (time.time() - launch_start_time)
            decision = ("compute_budget" if not budget.can_start(phase) else
                        "wall_watchdog_reserve" if remaining_wall < args.profile["checkpoint_grace_seconds"] + budget.step_reserve(phase) else "")
            decisions = [None] * world_size
            dist.all_gather_object(decisions, decision)
            if any(decisions):
                stop_reason = "compute_budget" if "compute_budget" in decisions else "wall_watchdog_reserve"
                break
            # Input transfer and loss-normalization preparation are not compute.
            prepared = []
            counts = []
            for batch, info in group:
                counts.append(int((batch["labels"] != -100).sum()))
                gpu_batch = {key: value.to(device="cuda", non_blocking=True) for key, value in batch.items()}
                prepared.append(gpu_batch | {key: upstream.wrap_tensor(torch.tensor(value, device="cpu")) for key, value in info.items()})
            if args.gradient_accumulation_steps > 1:
                counts_tensor = torch.tensor(counts, device="cuda", dtype=torch.int64)
                dist.all_reduce(counts_tensor)
                counts = counts_tensor.cpu().tolist()
            if rank == 0:
                write_json(args.run_dir / "compute_journal.json", {"clock": CLOCK, "inflight": True,
                           "compute_seconds": budget.used, "compute_phase_max_seconds": dict(budget.phase_max), "step": state.step, "update_started_unix": time.time()})
            dist.barrier()
            torch.cuda.synchronize()
            state.step += 1
            compiler.begin()
            if args.gradient_accumulation_steps > 1:
                metrics = run_accumulated_update(prepared, counts,
                    lambda batch, weight: accumulate_microbatch(state, batch, torch.tensor(weight, device="cuda", dtype=torch.float32), **extra),
                    lambda: accumulated_optimizer_update(state))
            else:
                metrics = upstream.train_batch(state, prepared[0], **extra)
            torch.cuda.synchronize()
            measured = compiler.finish()
            timings = torch.tensor([measured["compute_seconds"], measured["compilation_seconds"], measured["training_region_seconds"]], device="cuda", dtype=torch.float64)
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
            compute_seconds, compile_seconds, region_seconds = timings.cpu().tolist()
            within_budget = budget.charge(compute_seconds, phase)
            compilation_seconds += compile_seconds
            measured_steps.append(compute_seconds)
            update_token_slots = len(group) * args.microbatch_tokens * world_size
            measured_token_slots.append(update_token_slots)
            packed_tokens = torch.tensor(sum(info["total_seqlen"] for _, info in group), device="cuda", dtype=torch.int64)
            dist.all_reduce(packed_tokens)
            packed_tokens = packed_tokens.item()
            token_slots += update_token_slots
            real_tokens += packed_tokens
            microbatches += len(group)
            batch_in_epoch, epoch_complete = next_batch, next_epoch_complete
            if rank == 0:
                write_json(args.run_dir / "compute_journal.json", {"clock": CLOCK, "inflight": False,
                           "compute_seconds": budget.used, "compute_phase_max_seconds": dict(budget.phase_max), "step": state.step, "measured_unix": time.time()})
            metrics = upstream.reduce_metrics(metrics, prefix="train/")
            record = {"step": state.step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                      "microbatches_in_epoch": batch_in_epoch, "microbatches_this_update": len(group),
                      "token_slots_this_update": update_token_slots, "short_accumulation_group": len(group) < args.gradient_accumulation_steps,
                      "epoch_complete": epoch_complete, "step_seconds": compute_seconds,
                      "compute_seconds": compute_seconds, "compute_seconds_total": budget.used,
                      "compilation_seconds": compile_seconds, "training_region_seconds": region_seconds,
                      "step_reserve_seconds": budget.step_reserve(phase),
                      "token_slots_per_second": update_token_slots / max(compute_seconds, 1e-12),
                      "packed_tokens": packed_tokens, "packed_tokens_per_second": packed_tokens / max(compute_seconds, 1e-12),
                      "lr": lr, **extra, **metrics}
            if rank == 0:
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError(f"Nonfinite training metric at step {state.step}: {metrics}")
                with (args.run_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                upstream.wandb.log(record, step=state.step)
                print("RSI_STEP " + json.dumps(record, sort_keys=True), flush=True)
                if state.step % 100 == 0:
                    update_running_audit(epoch, batch_in_epoch, epoch_complete)
            if not within_budget:
                stop_reason = "compute_budget_overrun"
                break
            if args.max_steps and state.step >= args.max_steps or calibration_limit and state.step - initial_step >= calibration_limit:
                stop_reason = "diagnostic_step_limit"
                break
        if state.step == initial_step:
            raise RuntimeError("Budget/watchdog leaves no room for one new update; no candidate was produced")
        checkpoint_path = checkpoint(epoch, batch_in_epoch, epoch_complete)
        update_running_audit(epoch, batch_in_epoch, epoch_complete)
        if stop_reason != "epochs_completed":
            break

    if checkpoint_path is None or not measured_steps and not terminal_stop:
        raise RuntimeError("Run completed no new optimizer steps")

    if args.verify_checkpoint:
        def local_tensors():
            for name, tensor in state.model.state_dict().items():
                yield "model/" + name, tensor.to_local() if hasattr(tensor, "to_local") else tensor
            names = {id(parameter): name for name, parameter in state.model.named_parameters()}
            for parameter, values in state.optim.state.items():
                for name, tensor in values.items():
                    if torch.is_tensor(tensor):
                        yield "optim/" + names[id(parameter)] + "/" + name, tensor.to_local() if hasattr(tensor, "to_local") else tensor
        snapshots = {name: tensor.detach().cpu().clone() for name, tensor in local_tensors()}
        with torch.no_grad():
            for _, tensor in local_tensors():
                tensor.zero_()
        restore(checkpoint_path)
        mismatches = [name for name, tensor in local_tensors() if not torch.equal(tensor.detach().cpu(), snapshots[name])]
        ok = torch.tensor(not mismatches, device="cuda", dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not ok.item():
            raise RuntimeError(f"Checkpoint roundtrip failed on rank {rank}: {mismatches[:10]}")
        audit["checkpoint_roundtrip"] = {"exact_equal_all_ranks": True, "local_tensor_count": len(snapshots),
                                         "method": "Snapshot tensors to CPU, zero model/optimizer tensors, DCP reload, torch.equal every local tensor"}
        if rank == 0:
            print("RSI_CHECKPOINT_VERIFIED " + json.dumps(audit["checkpoint_roundtrip"]), flush=True)
        del snapshots

    # Export EMA where present, otherwise model tensors, with accurate provenance.
    weight_kind = export_weight_kind(state.optim)
    state.optim.swap_ema()
    full_state = get_model_state_dict(state.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        weights_path = args.export_dir / "weights.safetensors"
        save_file({name: tensor.contiguous() for name, tensor in full_state.items()}, str(weights_path),
                  metadata={"weights": weight_kind, "source_commit": revision or "unknown", "step": str(state.step)})
        audit["weights_sha256"] = file_sha256(weights_path)
        model_manifest = export_architecture(args.source, args.export_dir, config.arch.model_dump(),
                                             args.profile["max_source_bytes"], (args.run_dir,), data_config=config.data.model_dump())
        audit["model_manifest_sha256"] = file_sha256(args.export_dir / "model.json")
        audit["export"] = {"path": str(weights_path), "weights": weight_kind, "sha256": audit["weights_sha256"],
                           "tensor_count": len(full_state), "tensor_keys": sorted(full_state),
                           "model_manifest_sha256": audit["model_manifest_sha256"],
                           "source_files": model_manifest["source_files"]}
    state.optim.swap_ema()
    dist.barrier()
    memory_this_rank = {"rank": rank, "allocated_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_bytes": torch.cuda.max_memory_reserved()}
    memory_all_ranks = [None] * world_size
    dist.all_gather_object(memory_all_ranks, memory_this_rank)
    wall_seconds = time.time() - start_time
    if rank == 0:
        # Drop the first two steps (compilation/warmup); retain all raw measurements in metrics.jsonl.
        steady = measured_steps[2:]
        status = ("budget_exhausted" if stop_reason == "compute_budget" else
                  "failed" if stop_reason in ("compute_budget_overrun", "wall_watchdog_reserve") else "completed")
        audit.update({"status": status, "stop_reason": stop_reason, "compute_seconds": budget.used,
                      "compute_seconds_this_run": budget.used - prior_compute,
                      "compute_phase_max_seconds": dict(budget.phase_max),
                      "terminal_checkpoint_reexport": bool(terminal_stop),
                      "compilation_seconds": compilation_seconds,
                      "compute_reserve_seconds": budget.step_reserve(phase), "compute_reserve": budget.reserve_evidence(phase), "clock": CLOCK,
                      "steps": state.step, "new_steps": state.step - initial_step,
                      "completed_epochs": epoch if epoch_complete else epoch - 1,
                      "microbatches_processed": prior_microbatches + microbatches,
                      "checkpoint_path": str(checkpoint_path), "wall_seconds": wall_seconds,
                      "gpu_hours": prior_gpu_hours + world_size * wall_seconds / 3600,
                      "gpu_hours_this_run": world_size * wall_seconds / 3600,
                      "token_slots_processed": prior_token_slots + token_slots,
                      "packed_tokens_processed": prior_real_tokens + real_tokens,
                      "token_slots_processed_this_run": token_slots, "packed_tokens_processed_this_run": real_tokens,
                      "first_step_seconds": measured_steps[0] if measured_steps else None,
                      "steady_step_seconds_mean": sum(steady) / len(steady) if steady else None,
                      "steady_token_slots_per_second": sum(measured_token_slots[2:]) / sum(steady) if steady else None,
                      "peak_gpu_memory_allocated_bytes_rank0": torch.cuda.max_memory_allocated(),
                      "peak_gpu_memory_by_rank": memory_all_ranks,
                      "peak_gpu_memory_allocated_bytes": max(item["allocated_bytes"] for item in memory_all_ranks),
                      "peak_gpu_memory_reserved_bytes": max(item["reserved_bytes"] for item in memory_all_ranks),
                      "end_unix": time.time()})
        write_json(args.run_dir / "run.json", audit)
        write_json(args.export_dir / "run.json", audit)
        print("RSI_COMPLETE " + json.dumps({key: value for key, value in audit.items() if key not in ("export", "config", "train_metadata")}, sort_keys=True), flush=True)
        upstream.wandb.finish()
    callback_handler.remove_start_callback(compiler.on_compile_start)
    callback_handler.remove_end_callback(compiler.on_compile_end)
    dist.destroy_process_group(checkpoint_group)
    dist.destroy_process_group()
    signal.alarm(0)
    if stop_reason in ("compute_budget_overrun", "wall_watchdog_reserve"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
