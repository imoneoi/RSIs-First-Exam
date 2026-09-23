# HRM-Text data preparation

Use the frozen four-epoch sampled release for deployment. Every track consumes
the same immutable training files; one, two or four epochs select the matching
prefix of epoch index sets. Authenticate on the staging host if access is
required. Credentials remain there, and Work and Judge run offline.

```bash
python -m pip install -r environment/setup/requirements-data.txt
python environment/setup/download_sampled.py \
  --cache /mnt/hrm-text-hf-cache --output environment/staged
```

## Dataset and task lock

The Hugging Face repository contains 20 training/tokenizer files and
`README.md` (the dataset card), excluding repository-managed metadata:

| Files | Count |
| --- | ---: |
| `sampled/tokens.npy` and `sampled/metadata.json` | 2 |
| `sampled/epoch_0` through `epoch_3`, each with `inst_start.npy`, `inst_len.npy`, `resp_start.npy`, `resp_len.npy` | 16 |
| `tokenizer/tokenizer.json` and `tokenizer/config.json` | 2 |

No `data_manifest.json`, source inventory, preparation scripts or provenance
files are published in the dataset.

The external task file [`sampled_release.json`](sampled_release.json) pins the
final repository revision and embeds `training_manifest`. This minimal allowlist
contains full relative-file sizes and SHA-256 hashes, loader/tokenizer semantics,
token-array shape and dtype, and per-epoch row/token counts. It contains no
operator paths, host identity, source inventory, timestamps or preparation
history. The immutable Hub revision is outside the runtime manifest, avoiding
an identity that depends on the dataset card or repository history.

The downloader authenticates all 20 files, checks array bounds and lengths,
and compares their semantic metadata with the task lock. It then stages regular
files and creates exactly two identical local runtime manifests:

- `environment/staged/data_manifest.json`
- `environment/staged/sampled/data_manifest.json`

These are generated from `training_manifest` with deterministic JSON
serialization (`indent=2`, sorted keys, one trailing newline). They are not
fetched from Hugging Face and do not reproduce removed provenance. The tokenizer
path is the fixed container path `/datasets/hrm-text/tokenizer`.

The output appears only after successful validation. Existing destinations
are never overwritten. Hard links avoid copying large immutable assets where
possible; do not modify linked files in place.

```bash
# Reuse only the pinned snapshot already present in the local cache.
python environment/setup/download_sampled.py --offline \
  --cache /mnt/hrm-text-hf-cache --output /mnt/hrm-text-offline-staged

# Authenticate an existing staging directory without changing it.
python environment/setup/download_sampled.py --verify-only \
  --output environment/staged
```

| Build input | Container path |
| --- | --- |
| `staged/sampled` | `/datasets/hrm-text/sampled` |
| `staged/tokenizer` | `/datasets/hrm-text/tokenizer` |
| `staged/data_manifest.json` | `/datasets/hrm-text/data_manifest.json` |

## Epochs and calibration

A one-epoch track reads only `epoch_0`; two epochs read `epoch_0` and `epoch_1`;
four epochs read all four. Do not regenerate data when selecting a track or
train on unused indexed spans. PrefixLM metadata includes the autoregressive
shift (`max_seq_len=4097`), producing a 4096-token training context.
`metadata.json.total_length` is the mean raw length over available epochs;
`validation.epochs` retains exact per-epoch token totals for schedule projection.

The sanitized runtime manifest has a new identity even though all training
and tokenizer file bytes are unchanged. Existing calibration and resume audits
bind the identity they actually used. Do not replace their recorded hashes
silently. Recalibrate, or use an explicit reviewed migration that proves the
training-file hashes and scheduling semantics are identical.

Optional local sampling tools remain separate from canonical deployment.
Locally prepared artifacts cannot substitute for the pinned release merely by
copying its descriptor or changing a manifest hash.

## Training data and verifier data

Stage benchmark records separately under `tests/staged` using the verifier's
pinned revisions. Judge receives a copy of the sanitized runtime manifest and
its own benchmark/tokenizer reference; candidate Work receives training data
and the frozen tokenizer. Never copy a shared Hugging Face cache into Work,
since it can contain benchmark labels, model weights or credentials.

## Regression checks

```bash
HRM_DATA_IO_TEST_ROOT=/path/to/pinned/data_io \
  python -m unittest discover -s environment/setup -p 'test_*.py' -v
```

The fixtures use tiny arrays and cache-only or mocked Hub access. The strict
repository check runner rejects skipped tests; provide the pinned data_io
checkout. Tests cover file authentication, sanitized manifest generation,
array semantics, incomplete/mixed assets, no-overwrite behavior and offline
reuse without GPU jobs or changes to existing training data.
