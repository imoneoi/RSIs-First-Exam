#!/usr/bin/env python3
"""Offline-build preparation for HRM-Text's public training corpus.

With no --source-tokenized, download the pinned cleaned HF corpus, tokenize it
using pinned data_io, then sample. Supplying local tokens records a distinct
lineage; it never claims those tokens reproduce the published HF snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from sampled_layout import MAX_EPOCHS, available_epoch_ids


HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "data_sources.json").read_text())
FIELDS = ("inst_start", "inst_len", "resp_start", "resp_len")
TOKENIZATION_RECEIPT = "rsi_tokenization_receipt.json"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path, full: bool = True) -> dict:
    item = {"bytes": path.stat().st_size}
    if full:
        item["sha256"] = sha256(path)
    else:
        # Explicitly a diagnostic fingerprint, not whole-file authentication.
        size = item["bytes"]
        offsets = sorted({0, max(0, size // 2 - 524288), max(0, size - 1048576)})
        with path.open("rb") as stream:
            samples = []
            for offset in offsets:
                stream.seek(offset)
                data = stream.read(1048576)
                samples.append({"offset": offset, "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()})
        item["sampled_blocks_only"] = samples
    return item


def checked_checkout(cache: Path) -> Path:
    root = cache / "data_io"
    if not root.exists():
        run("git", "clone", PINS["data_io_repository"], root)
        run("git", "-C", root, "checkout", "--detach", PINS["data_io_revision"])
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != PINS["data_io_revision"]:
        raise ValueError(f"Unexpected data_io revision in {root}: {revision}")
    changed = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--name-only"], text=True)
    if changed.strip():
        raise ValueError(f"Pinned data_io checkout has tracked changes: {changed}")
    return root


def cleaned_inputs(clean: Path) -> dict[str, Path]:
    """Match pinned Rust scan_inputs names and reject ambiguous/missing inputs."""
    result = {}
    for directory in (clean / "data", clean / "data_clustered"):
        if not directory.is_dir():
            raise ValueError(f"Missing cleaned corpus directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".parquet", ".jsonl"):
                name = path.relative_to(directory).as_posix().replace("/", "__").replace("\\", "__")
                if name in result:
                    raise ValueError(f"Cleaned input names collide in the upstream tokenizer: {name}")
                result[name] = path
    if not result:
        raise ValueError("The cleaned corpus has no tokenizable shards")
    return result


def materialize_cleaned_snapshot(snapshot: Path, output: Path) -> Path:
    """The pinned Rust scanner skips symlinks used by the HF snapshot cache."""
    expected = {path.relative_to(snapshot): path for path in cleaned_inputs(snapshot).values()}
    if output.is_symlink():
        raise ValueError("Cleaned snapshot view must be a real directory")
    if output.exists():
        actual = {p.relative_to(output) for p in output.rglob("*") if p.is_file()}
        if actual != set(expected) or any(p.is_symlink() for p in output.rglob("*")):
            raise ValueError("Cleaned snapshot view is incomplete or mixed; use a fresh --cache directory")
        for relative, original in expected.items():
            path = output / relative
            if not path.samefile(original) and fingerprint(path) != fingerprint(original):
                raise ValueError("Cleaned snapshot view differs from the pinned snapshot; use a fresh --cache directory")
        return output
    output.mkdir(parents=True)
    for name in ("data", "data_clustered"):
        (output / name).mkdir()
    for relative, original in expected.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(str(original.resolve()), str(target))
    return output


def validate_tokenized(source: Path, inputs: dict[str, Path]) -> dict:
    """Rust logs per-shard failures without failing its process; check completion."""
    expected_files = {"tokenizer_info.json"} | {
        f"{name}/{field}" for name in inputs
        for field in ("metadata.json", "tokens.npy", *(f"{key}.npy" for key in FIELDS))}
    actual_files = {p.relative_to(source).as_posix() for p in source.rglob("*")
                    if p.is_file() and p != source / TOKENIZATION_RECEIPT}
    if actual_files != expected_files or {p.name for p in source.iterdir() if p.is_dir()} != set(inputs):
        raise ValueError("Tokenization is incomplete or contains unexpected shards/files; use a fresh --cache directory")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Tokenized cache must contain regular files, not symlinks")
    for name, original in inputs.items():
        completion = json.loads((source / name / "metadata.json").read_text())
        if completion != {"source_mtime": int(original.stat().st_mtime), "source_size": original.stat().st_size}:
            raise ValueError(f"Tokenized shard has stale or incomplete source metadata: {name}")
        tokens = np.load(source / name / "tokens.npy", mmap_mode="r")
        arrays = [np.load(source / name / f"{key}.npy", mmap_mode="r") for key in FIELDS]
        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or any(
                a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or len(a) != len(arrays[0]) for a in arrays):
            raise ValueError(f"Invalid tokenized shard array layout: {name}")
    return {name: fingerprint(source / name) for name in sorted(expected_files)}


def prepare_tokenized(clean: Path, source: Path, tokenizer: Path, upstream: Path) -> dict:
    """Reuse only fully authenticated caches made by this pinned recipe."""
    if source.is_symlink():
        raise ValueError("Tokenized cache must be a real directory, not a symlink")
    inputs = cleaned_inputs(clean)
    recipe = {"data_io_revision": PINS["data_io_revision"],
              "cleaned_dataset": PINS["cleaned_dataset"],
              "cleaned_dataset_revision": PINS["cleaned_dataset_revision"],
              "tokenizer": {p.name: fingerprint(p) for p in sorted(tokenizer.iterdir()) if p.is_file()},
              "inputs": {name: {"path": path.relative_to(clean).as_posix(),
                                "bytes": path.stat().st_size, "mtime": int(path.stat().st_mtime)}
                         for name, path in inputs.items()}}
    receipt_path = source / TOKENIZATION_RECEIPT
    if source.exists() and any(source.iterdir()):
        if not receipt_path.is_file():
            raise ValueError(f"Refusing nonempty tokenization cache without a sealed recipe receipt: {source}; use a fresh --cache directory")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema_version") != 1 or receipt.get("recipe") != recipe:
            raise ValueError("Tokenization cache belongs to a different recipe/input snapshot; use a fresh --cache directory")
        if validate_tokenized(source, inputs) != receipt.get("files"):
            raise ValueError("Tokenized cache content differs from its sealed receipt; use a fresh --cache directory")
        return receipt
    run("cargo", "run", "--release", "--locked", "--bin", "tokenizer", "--",
        clean / "data", clean / "data_clustered", "--tokenizer-path", tokenizer,
        "-o", source, cwd=upstream / "tokenizer")
    receipt = {"schema_version": 1, "recipe": recipe, "files": validate_tokenized(source, inputs)}
    # Publish only after every shard has its completion record and array hashes.
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def validate_sampled(root: Path) -> dict:
    meta = json.loads((root / "metadata.json").read_text())
    if meta.get("max_seq_len") != PINS["context_size_including_ar_shift"]:
        raise ValueError("Expected PrefixLM metadata max_seq_len=4097; encdec output is incompatible")
    tokens = np.load(root / "tokens.npy", mmap_mode="r")
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("tokens.npy must be a one-dimensional integer array")
    stats = []
    epoch_ids = available_epoch_ids(root)
    for epoch in epoch_ids:
        arrays = {key: np.load(root / f"epoch_{epoch}" / f"{key}.npy", mmap_mode="r") for key in FIELDS}
        length = len(arrays["inst_start"])
        if not length or any(a.ndim != 1 or len(a) != length or not np.issubdtype(a.dtype, np.integer)
                             for a in arrays.values()):
            raise ValueError(f"Invalid index arrays for epoch {epoch}")
        total = 0
        for start in range(0, length, 1_000_000):
            a = {k: v[start:start + 1_000_000] for k, v in arrays.items()}
            combined = a["inst_len"] + a["resp_len"]
            if np.any(a["inst_len"] < 1) or np.any(a["resp_len"] < 1) or np.any(combined > 4097):
                raise ValueError(f"Invalid sequence lengths in epoch {epoch}")
            for key in ("inst", "resp"):
                if np.any(a[f"{key}_start"] < 0) or np.any(a[f"{key}_start"] + a[f"{key}_len"] > len(tokens)):
                    raise ValueError(f"Out-of-bounds token indices in epoch {epoch}")
            total += int(combined.sum())
        stats.append({"epoch": epoch, "rows": length, "tokens_including_ar_shift": total})
    if meta["total_length"] != round(sum(s["tokens_including_ar_shift"] for s in stats) / len(epoch_ids)):
        raise ValueError("Sampled metadata total_length does not match epoch indices")
    return {"metadata": meta, "token_array_shape": list(tokens.shape),
            "token_array_dtype": str(tokens.dtype), "epochs": stats,
            "available_epoch_ids": epoch_ids, "available_epoch_count": len(epoch_ids)}


def write_manifest(args: argparse.Namespace, source: Path, tokenizer: Path,
                   prefix: Path, upstream: Path, lineage: str) -> Path:
    stats = validate_sampled(args.output)
    manifest_path = args.manifest or args.output / "data_manifest.json"
    existing = manifest_path if manifest_path.is_file() else args.output / "data_manifest.json"
    if (getattr(args, "existing_sampled", False) and not existing.is_file()
            and lineage == "pinned_hf_cleaned_then_tokenized"):
        raise ValueError("Cannot label existing unmanifested samples canonical; use a fresh output with --compact, or explicit local inputs for a diagnostic artifact")
    if getattr(args, "existing_sampled", False) and existing.is_file():
        # Epoch selection is a run budget, not a change to an existing data
        # artifact. Validate and reuse even legacy four-epoch manifests verbatim.
        previous = json.loads(existing.read_text())
        if lineage == "pinned_hf_cleaned_then_tokenized" and (
                previous.get("canonical_hf_recipe_applied") is not True or previous.get("pins") != PINS):
            raise ValueError("Canonical existing samples require a finalized manifest for the same pinned HF recipe")
        for key, expected in previous.get("validation", {}).items():
            if key in stats and stats[key] != expected:
                raise ValueError(f"Existing data manifest validation differs: {key}")
        sealed_ids = previous.get("available_epoch_ids", [item["epoch"] for item in previous["validation"]["epochs"]])
        if sealed_ids != stats["available_epoch_ids"]:
            raise ValueError("Existing data manifest seals a different set of epochs")
        if previous.get("available_epoch_count", len(sealed_ids)) != len(sealed_ids):
            raise ValueError("Existing data manifest epoch count disagrees with its epoch IDs")
        for label, path in (("prefix_config", prefix), ("sampler", upstream / "sample_tokenized.py")):
            if sha256(path) != previous[label]["sha256"]:
                raise ValueError(f"Existing data manifest {label} differs from supplied input")
        for name, expected in previous["tokenizer"].items():
            if fingerprint(tokenizer / name) != expected:
                raise ValueError(f"Existing data manifest tokenizer differs: {name}")
        for name, expected in previous["sampled_files"].items():
            path = args.output / name
            if args.output.resolve() not in path.resolve().parents or fingerprint(path, "sha256" in expected) != expected:
                raise ValueError(f"Existing sampled artifact differs from its manifest: {name}")
        if manifest_path != existing:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, manifest_path)
        return manifest_path
    source_files = sorted(source.glob("*/*.npy"))
    source_listing = [{"path": str(p.relative_to(source)), "bytes": p.stat().st_size} for p in source_files]
    inventory = args.output.parent / (args.output.name + "_source_inventory.json")
    inventory.write_text(json.dumps(source_listing, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lineage": lineage,
        "canonical_hf_recipe_applied": lineage == "pinned_hf_cleaned_then_tokenized",
        "pins": PINS,
        "runtime": {"python": sys.version, "packages": {package: version(package) for package in
                    ("numpy", "pydantic", "omegaconf", "PyYAML", "tqdm", "huggingface_hub")}},
        "sampling_parameters": {"epochs": stats["available_epoch_count"], "seed": 0, "context_size": 4097, "min_resp_length": 2},
        "available_epoch_ids": stats["available_epoch_ids"],
        "available_epoch_count": stats["available_epoch_count"],
        "source_tokenized_path": str(source.resolve()),
        "source_inventory": {"file": str(inventory), "sha256": sha256(inventory),
                             "task_count": len(list(source.glob("*/tokens.npy"))),
                             "array_bytes": sum(p["bytes"] for p in source_listing),
                             "contents_hashed": False},
        "source_tokenizer_info": json.loads((source / "tokenizer_info.json").read_text()),
        "tokenizer": {p.name: fingerprint(p) for p in sorted(tokenizer.iterdir()) if p.is_file()},
        "prefix_config": {"path": str(prefix.resolve()), **fingerprint(prefix)},
        "sampler": {"path": "sample_tokenized.py", **fingerprint(upstream / "sample_tokenized.py")},
        "sampled_files": {str(p.relative_to(args.output)): fingerprint(p, args.hash_mode == "full" or p.name != "tokens.npy")
                          for p in sorted(args.output.rglob("*")) if p.is_file() and p.suffix in (".json", ".npy")
                          and p.name not in ("manifest.json", "data_manifest.json")
                          and (args.manifest is None or p.resolve() != args.manifest.resolve())},
        "sampled_integrity": "full_sha256" if args.hash_mode == "full" else "full_indices_and_metadata_sampled_token_blocks",
        "validation": stats,
    }
    if lineage == "pinned_hf_cleaned_then_tokenized":
        receipt = source / TOKENIZATION_RECEIPT
        if not receipt.is_file():
            raise ValueError("Canonical sampling requires the validated tokenization cache receipt")
        manifest["tokenization_receipt"] = {"path": str(receipt.resolve()), **fingerprint(receipt)}
    if (source.parent / ".git").exists():
        manifest["local_source_checkout"] = {
            "revision": subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip(),
            "working_tree_status": subprocess.check_output(["git", "-C", str(source.parent), "status", "--short"], text=True).splitlines(),
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/hrm-text-data"))
    parser.add_argument("--source-tokenized", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prefix-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--existing-sampled", action="store_true", help="Validate completed sampling; preserve an existing finalized manifest byte-for-byte")
    parser.add_argument("--epochs", type=int, choices=range(1, MAX_EPOCHS + 1), default=4, help="Epochs to sample for a new artifact (default: 4); existing sampled artifacts retain all available epochs")
    parser.add_argument("--compact", action="store_true", help="Retain only referenced token ranges; keep the original sampler output in a sibling .uncompacted directory")
    parser.add_argument("--hash-mode", choices=("full", "sampled"), default="full")
    parser.add_argument("--stage", type=Path, help="Populate environment/staged for an offline Docker build")
    args = parser.parse_args()
    if args.compact and args.existing_sampled:
        parser.error("For existing sampled data, run compact_sampled.py to a separate output, then manifest that output with --existing-sampled")
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    upstream = checked_checkout(args.cache)
    if args.source_tokenized:
        if not args.tokenizer or not args.prefix_config:
            parser.error("Local tokens require explicit --tokenizer and --prefix-config to preserve lineage")
        source, tokenizer, prefix = args.source_tokenized.resolve(), args.tokenizer.resolve(), args.prefix_config.resolve()
        lineage = "operator_supplied_local_tokenized"
    else:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(PINS["cleaned_dataset"], repo_type="dataset",
                     revision=PINS["cleaned_dataset_revision"], cache_dir=args.cache / "huggingface",
                     allow_patterns=["data/**", "data_clustered/**", "README.md"]))
        clean = materialize_cleaned_snapshot(snapshot, args.cache / f"cleaned-{PINS['cleaned_dataset_revision']}")
        source = args.cache.resolve() / "tokenized"
        tokenizer = upstream.resolve() / "trained_tokenizers" / "bpe"
        prefix = upstream.resolve() / "prefix_config.yaml"
        if sha256(tokenizer / "tokenizer.json") != PINS["tokenizer_sha256"]:
            raise ValueError("Pinned tokenizer SHA-256 mismatch")
        prepare_tokenized(clean, source, tokenizer, upstream)
        lineage = "pinned_hf_cleaned_then_tokenized"
    if not args.existing_sampled:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite sampled output: {args.output}")
        sampler_output = args.output.with_name(args.output.name + ".uncompacted") if args.compact else args.output
        if sampler_output.exists():
            raise FileExistsError(f"Refusing to overwrite sampler output: {sampler_output}")
        estimated = sum(p.stat().st_size for p in source.glob("*/*.npy")) * 1.25
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.output.parent).free < estimated:
            raise ValueError(f"Insufficient output space: reserve at least {estimated / 2**30:.1f} GiB")
        analytics = args.output.parent / (args.output.name + "_analytics.md")
        with analytics.open("w") as report:
            subprocess.run([sys.executable, str(upstream / "sample_tokenized.py"),
                f"tokenized_path={source}", f"prefix_config_path={prefix}", f"output_path={sampler_output}",
                f"epochs={args.epochs}", "seed=0", "context_size=4097", "min_resp_length=2"],
                stdout=report, check=True)
        if args.compact:
            from compact_sampled import compact_sampled
            compact_sampled(sampler_output, args.output)
    manifest = write_manifest(args, source, tokenizer, prefix, upstream, lineage)
    if args.stage:
        prepared_manifest = json.loads(manifest.read_text())
        if prepared_manifest.get("canonical_hf_recipe_applied") is True and prepared_manifest.get("sampled_integrity") != "full_sha256":
            raise ValueError("Canonical release staging requires --hash-mode full; sampled-block hashes are diagnostic only")
        if args.stage.exists():
            raise FileExistsError(f"Refusing to overwrite staging directory: {args.stage}")
        args.stage.mkdir(parents=True)
        shutil.copytree(args.output, args.stage / "sampled", copy_function=link_or_copy)
        shutil.copytree(tokenizer, args.stage / "tokenizer")
        # Replace the hard-linked metadata before rewriting its runtime path.
        meta_path = args.stage / "sampled" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["tokenizer_info"]["tokenizer_path"] = "/datasets/hrm-text/tokenizer"
        meta_path.unlink()
        meta_path.write_text(json.dumps(meta) + "\n")
        staged_manifest = json.loads(manifest.read_text())
        staged_manifest["sampled_files"]["metadata.json"] = fingerprint(meta_path)
        staged_manifest["validation"]["metadata"] = meta
        staged_manifest["staged_runtime_tokenizer"] = "/datasets/hrm-text/tokenizer"
        (args.stage / "data_manifest.json").write_text(json.dumps(staged_manifest, indent=2) + "\n")
        inside_manifest = args.stage / "sampled" / "data_manifest.json"
        if inside_manifest.exists():
            inside_manifest.unlink()
        shutil.copy2(args.stage / "data_manifest.json", inside_manifest)
        inventory = args.output.parent / (args.output.name + "_source_inventory.json")
        if not inventory.is_file():
            inventory = Path(staged_manifest["source_inventory"]["file"])
        if sha256(inventory) != staged_manifest["source_inventory"]["sha256"]:
            raise ValueError("Source inventory differs from the sealed sampled artifact")
        shutil.copy2(inventory, args.stage / "source_inventory.json")
        shutil.copy2(prefix, args.stage / "prefix_config.yaml")
    print(f"Validated {args.output}; manifest: {manifest}")


if __name__ == "__main__":
    main()
