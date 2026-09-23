#!/usr/bin/env python3
"""Build an operator calibration from a completed baseline phase sweep.

First run launch.sh --track TRACK --calibration-steps-per-phase 12 ... on the
actual deployment topology. This command only summarizes its measured audit and
metrics; it never launches GPU processes or converts a smoke model into a score.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from compute_budget import CLOCK, phase_step_counts, project_baseline
from profiles import HRM_COMMIT, resolve_profile


def baseline_recipe(profile):
    """Scientific configuration of the pinned dev baseline, excluding output paths."""
    layers, hidden, heads = {"B": (12, 1024, 8), "L": (24, 1280, 10), "XL": (32, 1536, 12)}[profile["size"]]
    return {"arch": {"name": "baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel", "head": "lm_head@LMHead",
                     "half_layers": True, "H_cycles": 2, "L_cycles": 3, "H_override": {},
                     "bp_warmup_ratio": 0.5, "bp_min_steps": 2, "bp_max_steps": 6,
                     "n_layers": layers, "hidden_size": hidden, "num_heads": heads, "expansion": 4,
                     "norm_type": "pre", "norm_eps": 1e-6, "rope_theta": 10000.0,
                     "pos_emb_type": "rope", "init_type": "lecun_normal"},
            "data": {"target_only": True}, "global_batch_size": profile["baseline_global_batch"],
            "epochs": profile["epochs"], "lr": profile["baseline_lr"], "ema": profile["ema"],
            "lr_min_ratio": 1.0, "lr_warmup_steps": 2000, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
            "fwd_bwd_dtype": "bfloat16", "resume_from": None, "resume_epoch": None,
            "weights_only_resume_from_ema": False, "seed": 0, "checkpoint_interval": 1}


def validate_baseline_recipe(config, profile, batch):
    dynamic = {"project_name", "run_name", "checkpoint_path", "log_interval"}
    actual = {key: value for key, value in config.items() if key not in dynamic}
    actual["data"] = {key: value for key, value in config.get("data", {}).items() if key != "path"}
    if actual != baseline_recipe(profile) or batch.get("global_batch_size") != profile["baseline_global_batch"]:
        raise ValueError("Calibration recipe differs from the pinned baseline architecture, data objective, optimizer or schedule")
    if batch.get("microbatch_tokens", 0) * profile["gpus"] * batch.get("gradient_accumulation_steps", 0) != profile["baseline_global_batch"]:
        raise ValueError("Calibration physical batches do not reproduce the reference global batch")


def hardware_signature(hardware):
    """Compare device capability/topology while allowing container hostname changes."""
    ranks = hardware.get("ranks", [])
    if len(ranks) != hardware.get("nodes", 0) * hardware.get("gpus_per_node", 0) or not ranks:
        raise ValueError("Calibration is missing the measured device topology")
    devices = sorted((row["gpu_name"], row["total_memory_bytes"], tuple(row["capability"])) for row in ranks)
    return hardware["nodes"], hardware["gpus_per_node"], devices


def summarize(run_dir):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
    if (run.get("status") != "completed" or run.get("clock") != CLOCK or run.get("baseline") != "hrm"
            or not run.get("calibration_steps_per_phase") or not run.get("smoke")
            or run.get("source_commit") != HRM_COMMIT or run.get("source_clean") is not True or run.get("resume_from")):
        raise ValueError("Calibration requires a completed diagnostic sweep of the clean pinned HRM baseline")
    profile = run["profile"]
    validate_baseline_recipe(run["config"], profile, run["batch"])
    hardware_signature(run["hardware"])
    arch = run["config"]["arch"]
    phase_counts = phase_step_counts(run["schedule_total_steps"], arch["bp_min_steps"], arch["bp_max_steps"], arch["bp_warmup_ratio"])
    projection = project_baseline(records, phase_counts)
    calibration = {"schema_version": 1, "track": profile["track"], "nodes": profile["nodes"],
                   "gpus_per_node": profile["gpus_per_node"], "hrm_commit": run["source_commit"],
                   "data_manifest_sha256": run["data_manifest_sha256"], "clock": CLOCK,
                   "estimated_baseline_compute_seconds": projection["estimated_baseline_compute_seconds"],
                   "measured_steps": len(records), "hardware": run["hardware"],
                   "method": "measured_schedule_projection", "projection": projection,
                   "schedule_total_steps": run["schedule_total_steps"],
                   "schedule_token_estimate": run["schedule_token_estimate"],
                   "measured_compute_seconds": sum(record["compute_seconds"] for record in records),
                   "excluded_compilation_seconds": sum(record["compilation_seconds"] for record in records),
                   "metrics_sha256": hashlib.sha256((run_dir / "metrics.jsonl").read_bytes()).hexdigest(),
                   "run_sha256": hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest(),
                   "baseline_config": run["config"], "baseline_batch": run["batch"],
                   "runner_source_sha256": run.get("runner_source_sha256")}
    # Round-trip through the operator profile validator before writing anything.
    resolve_profile(profile["track"], gpus_per_node=profile["gpus_per_node"], calibration=calibration)
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "estimated_baseline_compute_hours": result["estimated_baseline_compute_seconds"] / 3600,
                      "measured_steps": result["measured_steps"], "uncertainty": result["projection"]["uncertainty_note"]}, indent=2))


if __name__ == "__main__":
    main()
