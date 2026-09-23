# HRM-Text data preparation

Use the frozen four-epoch sampled release for this task. Downloading it avoids
repeating cleaning, tokenization and stratified sampling, and gives every track
identical input bytes. The initial repository is private in the `sapientinc`
organization; the staging operator needs an authorized Hugging Face login.
Credentials remain on the staging host. Work and Judge images run offline.

```bash
python -m pip install -r environment/setup/requirements-data.txt
python environment/setup/download_sampled.py \
  --cache /mnt/hrm-text-hf-cache \
  --output environment/staged
```

[`sampled_release.json`](sampled_release.json) pins the dataset repository, exact
Hub commit and sampled manifest SHA-256. The downloader verifies every complete
file hash, the tokenizer, all four epoch index sets, their bounds and lengths,
and the release descriptor. It stages regular files, using hard links when
possible, and publishes the output directory only after validation. It refuses
to overwrite an existing destination. The source manifest is preserved exactly;
its tokenizer path already names `/datasets/hrm-text/tokenizer`.

```bash
# Reuse a previously downloaded snapshot without any network access.
python environment/setup/download_sampled.py --offline \
  --cache /mnt/hrm-text-hf-cache --output /mnt/hrm-text-offline-staged

# Authenticate an existing staging directory without changing it.
python environment/setup/download_sampled.py --verify-only \
  --output environment/staged
```

The data manifest records its original `operator_supplied_local_tokenized`
lineage. It does not claim to reproduce the older public cleaned-HF recipe.
Its `canonical_dataset` descriptor and the separately pinned commit/manifest
identify the task release. Keeping the Hub revision outside the dataset's own
manifest avoids a circular reference. No benchmark cache or pretrained weights
belong in this release.

| Build input | Container path |
| --- | --- |
| `staged/sampled` | `/datasets/hrm-text/sampled` |
| `staged/tokenizer` | `/datasets/hrm-text/tokenizer` |
| `staged/data_manifest.json` | `/datasets/hrm-text/data_manifest.json` |

The complete sampled artifact has four immutable epoch index sets. A one-epoch
track uses only `epoch_0`; two epochs use `epoch_0` and `epoch_1`; four epochs use
all four. Selecting a track does not regenerate data or change its manifest.
Later epochs can contain previously unseen documents. A scored attempt must
meet the task's current exposure and computation-time policy; short smoke runs
are infrastructure diagnostics.

## Reproducing sampling from the supplied tokenized corpus

[`data_sources.json`](data_sources.json) pins the public `data_io` dev commit
`3728ebea6b6a7ff554af4e51350e6fd9e4b83fa6`. Its `sample_tokenized.py` and
`prefix_config.yaml` match the local v3 checkout's corresponding files exactly.
The corpus on this machine is read-only input:

```bash
python environment/setup/prepare_data.py \
  --cache /mnt/hrm-text-build-cache \
  --source-tokenized /sg-pretrain/one/data_io/data_tokenized_bpe_chat_code_65k \
  --tokenizer /sg-pretrain/one/data_io/trained_tokenizers/bpe_chat_code \
  --prefix-config /sg-pretrain/one/data_io/prefix_config.yaml \
  --output /mnt/hrm-text-new-sampled \
  --epochs 4
```

Use a new output path. Never replace data used by an existing run. The dev
sampler directly retains only sampled, truncated rows and writes `uint16`
tokens for this 65,536-token vocabulary. The separate `--compact` transformation
is unnecessary for this sampler; it remains available for older artifacts.
Seed 0 and the pinned prefix rules determine the samples. `context_size=4097`
includes the autoregressive shift, giving a training context of 4096.

One-epoch and four-epoch sampling produce identical first-epoch token sequences
in the same order. Their compact token files and offsets can differ because the
four-epoch artifact contains additional selected rows. Reusing the published
four-epoch artifact provides one byte identity across all tracks.

`prepare_data.py` validates arrays and writes full SHA-256 hashes of sampled
files, tokenizer and sampler, plus epoch token/row counts and source array size
inventory. `metadata.json.total_length` is the mean over available epochs;
`validation.epochs` records exact per-epoch totals. Source size inventory alone
does not authenticate all original tokenized bytes: the released sampled files
are fully hashed. `--hash-mode sampled` remains diagnostic only and is not
accepted by the canonical release downloader.

`--existing-sampled` validates a completed sample without resampling and preserves
an existing finalized manifest byte-for-byte. `--epochs` affects new sampling
only. The `encdec_sample_tokenized.py` alternative is incompatible with the
PrefixLM loader's combined-context metadata; use `sample_tokenized.py`.

The older cleaned-HF/tokenization path in `prepare_data.py` is retained for
reproduction work. It downloads the pinned cleaned `data/` and `data_clustered/`
snapshot, requires Rust/Cargo, authenticates completed tokenization caches, and
uses the public BPE tokenizer. That historical corpus/tokenizer recipe is a
different lineage from this task's sampled release and cannot substitute for it.
Its `--stage` option rewrites a copied metadata path and seals the changed
manifest; the canonical sampled downloader requires no such rewrite.

## Training data and verifier data

Prepare benchmark records separately under `tests/staged` using the verifier's
pinned revisions and settings. Frozen benchmark records can be evaluated offline
without a Hugging Face dataset cache. Never copy a shared `HF_HOME` into a Work
image: it can contain benchmark answers, weights or credentials. The training
container receives only the sampled corpus and frozen tokenizer. Enable offline
Hugging Face mode at runtime.

## Data regression checks

```bash
HRM_DATA_IO_TEST_ROOT=/path/to/pinned/data_io \
  python -m unittest discover -s environment/setup -p 'test_*.py' -v
```

Tests use tiny arrays and mock/cache-only Hub access. They cover full manifest
and file authentication, incomplete/mixed assets, copied snapshot regular files,
no-overwrite behavior, offline staging, dev sampler sequence-prefix parity,
compaction and tokenization cache integrity. They do not access GPUs or change
existing training data. Sampler tests skip explicitly if the pinned checkout is
unavailable.
