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
import shutil
import subprocess
import sys
import time


def resolve_batch(size, gpus, microbatch_tokens=None, accumulation_steps=None, global_batch_size=None):
    """Resolve token slots per optimizer update without importing torch."""
    effective = {"L": 172032, "XL": 196608}[size] if global_batch_size is None else global_batch_size
    if type(effective) is not int or effective <= 0:
        raise ValueError("Effective global batch size must be a positive integer")
    if microbatch_tokens is None and accumulation_steps is None:
        microbatch_tokens = 6144 if size == "XL" else (21504 if gpus == 8 else 10752)
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
    parser.add_argument("--size", choices=("L", "XL"))
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--microbatch-tokens", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--source", type=Path, default=Path("/app/hrm-text"))
    parser.add_argument("--data", type=Path, default=Path("/datasets/hrm-text/sampled"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=0, help="Diagnostic optimizer-step limit; 0 completes the selected epochs")
    parser.add_argument("--dry-run", action="store_true", help="Print the profile and launch command without importing torch")
    parser.add_argument("--launch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="A checkpoint_step_* directory written by this runner")
    parser.add_argument("--override", action="append", default=[], help="Explicit Hydra recipe override; recorded in run.json")
    args = parser.parse_args(argv)
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    from profiles import resolve_profile
    active = None
    if os.environ.get("HRM_ENFORCE_PROFILE") == "1":
        active = json.loads(Path(__file__).with_name("profile.json").read_text())
    args.size = args.size if args.size is not None else (active["size"] if active else "L")
    args.gpus = args.gpus if args.gpus is not None else (active["gpus"] if active else (8 if args.size == "L" else 2))
    args.epochs = args.epochs if args.epochs is not None else (active["epochs"] if active else 4)
    try:
        args.profile = resolve_profile(args.size, args.gpus, args.epochs)
    except ValueError as error:
        parser.error(str(error))
    if active is not None:
        if args.profile != active:
            parser.error("Requested profile differs from the operator-selected /task-tools/profile.json")
    explicit_global = None
    for override in args.override:
        key, _, value = override.lstrip("+").partition("=")
        if key == "global_batch_size":
            try:
                explicit_global = int(value)
            except ValueError:
                parser.error("global_batch_size override must be an explicit integer for batch accounting")
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
    args.source = args.source.resolve()
    args.data = args.data.resolve()
    args.run_dir = args.run_dir.resolve()
    args.export_dir = (args.export_dir or args.run_dir / "submission").resolve()
    return args


def launch_command(args):
    command = ["timeout", "--signal=TERM", "--kill-after=60s", f"{args.profile['attempt_wall_seconds']}s",
               "torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={args.gpus}", str(Path(__file__).resolve()),
               "--size", args.size, "--gpus", str(args.gpus), "--epochs", str(args.epochs),
               "--microbatch-tokens", str(args.microbatch_tokens),
               "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
               "--source", str(args.source), "--data", str(args.data), "--run-dir", str(args.run_dir),
               "--export-dir", str(args.export_dir)]
    if args.max_steps:
        command += ["--max-steps", str(args.max_steps)]
    if args.verify_checkpoint:
        command.append("--verify-checkpoint")
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from.resolve())]
    for override in args.override:
        command += ["--override", override]
    return command


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def export_architecture(source, export_dir, arch, max_source_bytes=None, exclude_paths=()):
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
    manifest = {"schema_version": 1, "arch": arch, "forward_dtype": "bfloat16",
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


def resume_accounting(saved, previous, launch=None, now=None):
    """Charge a whole stopped invocation, even when rolling back an older checkpoint.

    A terminal supervisor receipt is preferred for interrupted runs. Without one,
    elapsed time through resume is charged conservatively, including downtime.
    This deliberately never treats checkpoint-time usage as final invocation usage.
    """
    if previous is None or "start_unix" not in previous:
        raise ValueError("Resume requires the parent run.json for complete invocation accounting")
    launch = launch or {}
    identities = [(previous.get("rank0_pid"), previous.get("rank0_start_ticks")),
                  (launch.get("supervisor_pid"), launch.get("supervisor_start_ticks")),
                  (launch.get("child_pid"), launch.get("child_start_ticks"))]
    for pid, ticks in identities:
        if pid and ticks and process_start_ticks(pid) == ticks:
            raise RuntimeError(f"Cannot resume while parent process {pid} is alive")
    if previous.get("status") not in ("completed", "failed") and not any(pid and ticks for pid, ticks in identities):
        raise RuntimeError("Cannot establish that the interrupted parent stopped: no recorded process identity")
    inherited = float(previous.get("prior_gpu_hours", saved.get("invocation_prior_gpu_hours", 0.0)))
    lower_bound = max(float(saved["gpu_hours"]), float(previous.get("gpu_hours", 0)))
    if "ended_utc" in launch:
        end = datetime.datetime.fromisoformat(launch["ended_utc"]).timestamp()
        method = "terminal_supervisor_receipt_entire_invocation"
    elif previous.get("status") in ("completed", "failed") and "end_unix" in previous and "gpu_hours" in previous:
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


def main():
    args = arguments()
    if args.dry_run:
        print(json.dumps({"profile": args.profile, "batch": args.batch, "command": launch_command(args),
                          "default_learning_rate": 2.5e-4 if args.size == "L" else 2.2e-4,
                          "diagnostic_only": bool(args.max_steps)}, indent=2, sort_keys=True))
        return
    if args.launch:
        command = launch_command(args)
        os.execvpe(command[0], command, os.environ)
    raw_metadata = json.loads((args.data / "metadata.json").read_text())
    data_manifest_path = next((args.data / name for name in ("data_manifest.json", "manifest.json", "sampling_manifest.json", "rsi_manifest.json")
                               if (args.data / name).is_file()), None)
    manifest = json.loads(data_manifest_path.read_text()) if data_manifest_path else None
    schedule = selected_schedule_tokens(raw_metadata, manifest, args.epochs)
    architecture_sources(args.source, args.profile["max_source_bytes"], (args.run_dir, args.export_dir))
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
    if args.resume_from:
        previous_audit_path = args.resume_from.resolve().parent / "run.json"
        previous_run_audit = json.loads(previous_audit_path.read_text()) if previous_audit_path.is_file() else None
        previous_launch_path = previous_audit_path.with_name("launch.json")
        previous_launch = json.loads(previous_launch_path.read_text()) if previous_launch_path.is_file() else None
        saved_progress = json.loads((args.resume_from / "progress.json").read_text())
        # Reject live parents before touching an existing same-directory audit.
        resume_receipt = resume_accounting(saved_progress, previous_run_audit, previous_launch)
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

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != args.gpus:
        raise ValueError(f"Profile requires {args.gpus} GPUs; torchrun initialized {world_size}")
    start_time = time.time()
    receipts = [resume_receipt if rank == 0 else None]
    dist.broadcast_object_list(receipts, src=0)
    resume_receipt = receipts[0]
    overrides = [
        f"arch/size@arch={args.size}", f"lr={2.5e-4 if args.size == 'L' else 2.2e-4}",
        f"global_batch_size={args.batch['global_batch_size']}",
        f"epochs={args.epochs}", f"data.path={args.data}",
        "+project_name=RSI-HRM-Text", f"+run_name={args.run_dir.name}",
        f"+checkpoint_path={args.run_dir}", "+log_interval=1",
    ] + args.override
    with initialize_config_dir(config_dir=str(args.source / "config"), version_base=None):
        hydra_config = compose(config_name="cfg_pretrain", overrides=overrides)
    config = upstream.PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))
    if Path(config.data.path).resolve() != args.data:
        raise ValueError("Specify data with --data so the manifest audit identifies the consumed dataset")
    if config.resume_from is not None:
        raise ValueError("Use the runner's --resume-from to preserve progress and cumulative resource accounting")
    if config.epochs != args.epochs:
        raise ValueError("Set epochs with --epochs so the task profile and training schedule agree")
    if config.global_batch_size != args.batch["global_batch_size"]:
        raise ValueError("Resolved global batch disagrees with the audited microbatch/accumulation product")
    torch.random.manual_seed(config.seed + rank)
    # The effective batch determines LR/BP schedules; the physical microbatch
    # determines data packing and activation memory.
    loader, metadata = upstream.create_dataloader(config, args.microbatch_tokens, True, rank, world_size)
    model, carry, optim = upstream.create_model_and_carry(config, metadata, args.microbatch_tokens)
    state = upstream.TrainState(model=model, carry=carry, optim=optim, step=0,
                               total_steps=schedule["tokens"] // config.global_batch_size)
    parameter_count = sum(parameter.numel() for parameter in state.model.parameters())
    torch.cuda.synchronize()

    @torch.compile(dynamic=False)
    def accumulate_microbatch(train_state, batch, response_weight, **extra):
        train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **extra)
        (loss * response_weight).backward()
        return metrics

    @torch.compile(dynamic=False)
    def accumulated_optimizer_update(train_state):
        train_state.optim.step()
        train_state.optim.zero_grad()

    try:
        revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        revision = None
    package_names = ["torch", "flash-attn-3", "numpy", "numba", "hydra-core", "pydantic", "safetensors"]
    package_versions = {name: importlib.metadata.version(name) for name in package_names}
    data_hashes = {"metadata.json": file_sha256(args.data / "metadata.json")}
    data_manifest_sha256 = None
    for name in ("data_manifest.json", "manifest.json", "sampling_manifest.json", "rsi_manifest.json"):
        if (args.data / name).is_file():
            data_hashes[name] = file_sha256(args.data / name)
            if data_manifest_sha256 is None:
                data_manifest_sha256 = data_hashes[name]
    audit = {
        "schema_version": 2, "status": "running", "source_commit": revision,
        "profile": args.profile, "batch": args.batch,
        "source": str(args.source), "data": str(args.data), "data_manifest_sha256": data_manifest_sha256,
        "data_file_hashes": data_hashes,
        "config": config.model_dump(), "train_metadata": metadata.model_dump(),
        "world_size": world_size, "gpu_name": torch.cuda.get_device_name(),
        "python": platform.python_version(), "packages": package_versions,
        "num_params": parameter_count, "schedule_total_steps": state.total_steps,
        "schedule_token_estimate": schedule,
        "data_policy": {"sampled_epoch_ids": list(range(args.epochs)),
                        "drop_last_batch": True, "partial_accumulation_groups": "train",
                        "schedule_basis": schedule["basis"],
                        "schedule_note": "Raw token estimate; packing and dropped final microbatches affect actual optimizer steps."},
        "max_steps": args.max_steps, "smoke": bool(args.max_steps),
        "start_unix": start_time, "resume_from": str(args.resume_from) if args.resume_from else None,
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
        print("RSI_INIT " + json.dumps(audit, sort_keys=True), flush=True)

    def checkpoint(epoch, batch_in_epoch, epoch_complete):
        path = args.run_dir / f"checkpoint_step_{state.step:08d}"
        dcp.save({"model": state.model.state_dict(), "optim": get_optimizer_state_dict(state.model, state.optim)}, checkpoint_id=path)
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
                                                "gpu_hours": prior_gpu_hours + world_size * (time.time() - start_time) / 3600,
                                                "invocation_prior_gpu_hours": prior_gpu_hours,
                                                "token_slots_processed": prior_token_slots + token_slots,
                                                "packed_tokens_processed": prior_real_tokens + real_tokens,
                                                "smoke": audit["smoke"], "data_manifest_sha256": data_manifest_sha256,
                                                "config": config.model_dump()})
        dist.barrier()
        if rank == 0:
            print("RSI_CHECKPOINT_SAVED " + str(path), flush=True)
        return path

    def restore(path):
        optim_state = get_optimizer_state_dict(state.model, state.optim)
        dcp.load({"model": state.model.state_dict(), "optim": optim_state}, checkpoint_id=path)
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
        configuration = lambda cfg: {key: value for key, value in cfg.items()
                                     if key not in ("checkpoint_path", "project_name", "run_name", "log_interval")}
        if configuration(saved["config"]) != configuration(config.model_dump()):
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
        if resume_epoch > config.epochs:
            raise ValueError("Checkpoint already completed every selected epoch")
    remaining_seconds = (args.profile["gpu_hours_cap"] - prior_gpu_hours) * 3600 / world_size - (time.time() - start_time)
    if remaining_seconds <= 0:
        raise RuntimeError("The cumulative profile GPU-hour training budget is exhausted")
    # Every rank terminates at the remaining cumulative task deadline, including resumed runs.
    signal.alarm(max(1, math.floor(remaining_seconds)))
    audit["prior_gpu_hours"] = prior_gpu_hours
    initial_step = state.step
    measured_steps = []
    measured_token_slots = []
    token_slots = 0
    real_tokens = 0
    microbatches = 0
    checkpoint_path = None

    def update_running_audit(epoch, batch_in_epoch, epoch_complete=False):
        if rank == 0:
            audit.update({"status": "running", "steps": state.step, "epoch": epoch,
                          "batch_in_epoch": batch_in_epoch, "microbatches_in_epoch": batch_in_epoch,
                          "microbatches_processed": prior_microbatches + microbatches,
                          "completed_epochs": epoch if epoch_complete else epoch - 1,
                          "prior_gpu_hours": prior_gpu_hours,
                          "gpu_hours": prior_gpu_hours + world_size * (time.time() - start_time) / 3600,
                          "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                          "token_slots_processed": prior_token_slots + token_slots,
                          "packed_tokens_processed": prior_real_tokens + real_tokens})
            write_json(args.run_dir / "run.json", audit)

    update_running_audit(resume_epoch, resume_batch)
    for epoch in range(resume_epoch, config.epochs + 1):
        state.model.train()
        batch_in_epoch = 0
        epoch_complete = False
        previous_step_end = time.perf_counter()
        groups = accumulation_groups(loader, args.gradient_accumulation_steps,
                                     resume_batch if epoch == resume_epoch else 0)
        for batch_in_epoch, group, epoch_complete in groups:
            if resume_rng is not None:
                # A newly constructed DataLoader iterator consumes a CPU base seed.
                # Restore after iterator startup/skipping so the next update sees
                # the checkpoint RNG state, as an uninterrupted run would. The
                # upstream sampled-data loader itself has no stochastic transforms.
                torch.set_rng_state(resume_rng[0])
                torch.cuda.set_rng_state(resume_rng[1])
                resume_rng = None
            state.step += 1
            lr = upstream.update_lr(config, state)
            extra = state.model.compute_train_extra_args(state)
            if args.gradient_accumulation_steps > 1:
                # The head divides each local summed loss by its microbatch's
                # globally averaged valid-token count. Weight those means by
                # count to recover the whole group's identical objective. FSDP
                # keeps its native SUM reduction, including on every backward.
                counts = torch.tensor([int((batch["labels"] != -100).sum()) for batch, _ in group],
                                      device="cuda", dtype=torch.int64)
                dist.all_reduce(counts)
                def backward_microbatch(microbatch, weight):
                    batch, batch_info = microbatch
                    model_batch = batch | {
                        key: upstream.wrap_tensor(torch.tensor(value, device="cpu")) for key, value in batch_info.items()
                    }
                    return accumulate_microbatch(state, model_batch,
                                                 torch.tensor(weight, device="cuda", dtype=torch.float32), **extra)
                metrics = run_accumulated_update(group, counts.cpu().tolist(), backward_microbatch,
                                                 lambda: accumulated_optimizer_update(state))
            else:
                batch, batch_info = group[0]
                model_batch = batch | {
                    key: upstream.wrap_tensor(torch.tensor(value, device="cpu")) for key, value in batch_info.items()
                }
                # Preserve the reference L8 compiled update exactly.
                metrics = upstream.train_batch(state, model_batch, **extra)
            metrics = upstream.reduce_metrics(metrics, prefix="train/")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - previous_step_end
            step_seconds = torch.tensor(elapsed, device="cuda", dtype=torch.float64)
            dist.all_reduce(step_seconds, op=dist.ReduceOp.MAX)
            packed_tokens = torch.tensor(sum(info["total_seqlen"] for _, info in group), device="cuda", dtype=torch.int64)
            dist.all_reduce(packed_tokens)
            elapsed, packed_tokens = step_seconds.item(), packed_tokens.item()
            measured_steps.append(elapsed)
            update_token_slots = len(group) * args.microbatch_tokens * world_size
            measured_token_slots.append(update_token_slots)
            token_slots += update_token_slots
            real_tokens += packed_tokens
            microbatches += len(group)
            record = {"step": state.step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                      "microbatches_in_epoch": batch_in_epoch, "microbatches_this_update": len(group),
                      "token_slots_this_update": update_token_slots, "short_accumulation_group": len(group) < args.gradient_accumulation_steps,
                      "epoch_complete": epoch_complete,
                      "step_seconds": elapsed, "token_slots_per_second": update_token_slots / elapsed,
                      "packed_tokens": packed_tokens, "packed_tokens_per_second": packed_tokens / elapsed,
                      "lr": lr, **extra, **metrics}
            if rank == 0:
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError(f"Nonfinite training metric at step {state.step}: {metrics}")
                with (args.run_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                upstream.wandb.log(record, step=state.step)
                print("RSI_STEP " + json.dumps(record, sort_keys=True), flush=True)
                if state.step % 100 == 0:
                    update_running_audit(epoch, batch_in_epoch)
            previous_step_end = time.perf_counter()
            if args.max_steps and state.step >= args.max_steps:
                break
        if not batch_in_epoch:
            raise RuntimeError(f"No training batches in epoch {epoch}")
        checkpoint_path = checkpoint(epoch, batch_in_epoch, epoch_complete)
        update_running_audit(epoch, batch_in_epoch, epoch_complete)
        if args.max_steps and state.step >= args.max_steps:
            break

    if checkpoint_path is None or not measured_steps:
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
                                             args.profile["max_source_bytes"], (args.run_dir,))
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
        audit.update({"status": "completed", "steps": state.step, "new_steps": state.step - initial_step,
                      "completed_epochs": epoch if epoch_complete else epoch - 1,
                      "microbatches_processed": prior_microbatches + microbatches,
                      "checkpoint_path": str(checkpoint_path), "wall_seconds": wall_seconds,
                      "gpu_hours": prior_gpu_hours + world_size * wall_seconds / 3600,
                      "gpu_hours_this_run": world_size * wall_seconds / 3600,
                      "token_slots_processed": prior_token_slots + token_slots,
                      "packed_tokens_processed": prior_real_tokens + real_tokens,
                      "token_slots_processed_this_run": token_slots, "packed_tokens_processed_this_run": real_tokens,
                      "first_step_seconds": measured_steps[0],
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
    dist.destroy_process_group()
    signal.alarm(0)


if __name__ == "__main__":
    main()
