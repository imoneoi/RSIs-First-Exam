# HRM-Text data preparation

Run these commands on a staging host with network access, Python, Git, Rust/Cargo,
and enough disk space. Training and grading use baked, read-only data and can run
offline. Large datasets and caches are not committed to Git.

The [public data_io README](https://github.com/sapientinc/data_io/tree/c82cd5f66459c5e66709e8430eccf8daecd0ca7d)
links a [cleaned Hugging Face corpus](https://huggingface.co/datasets/sapientinc/HRM-Text-data-io-cleaned-20260515/tree/f033bc0e1a81634385093afc60445a52a7ade64a).
This skips changing raw sources and their cleaning dependencies. It contains
instruction/response text, **not token arrays**: Rust tokenization is still needed.
`data_sources.json` pins the source commits, tokenizer digest, and sampling rules.

```bash
python -m pip install -r environment/setup/requirements-data.txt
python environment/setup/prepare_data.py \
  --cache /mnt/hrm-text-build-cache \
  --output /mnt/hrm-text-sampled \
  --epochs 4 \
  --compact \
  --stage environment/staged
```

The script downloads only `data/` and `data_clustered/`, uses the published BPE
tokenizer, invokes Cargo with the upstream lockfile, and samples one through four
epochs (`--epochs`, default 4) with Philox seed 0 and the pinned upstream prefix
configuration. Sampling one epoch produces the same first epoch as sampling four;
changing the requested count does not change the sampler or mixture rules. Later
epochs can contain previously unselected documents because stratified sampling
continues through each source's permutation. The
sampler's `context_size=4097` includes the one-token autoregressive shift, so the
training context is 4096. The upstream sampler initially retains the complete
corpus in a token mmap; epoch indices select the stratified training mixture.
`--compact` then retains only the union of token ranges referenced by all available
epochs and remaps the start offsets. It preserves every instruction/response
token, sequence length, and row order. This reduces image storage and removes
source tokens never selected for the fixed training mixture. The original sampler
output remains in a sibling `.uncompacted` directory; budget space for both until
release validation is done. Compaction sorts all referenced intervals in host
memory (approximately 64 bytes per instruction/response interval), then copies
retained tokens in bounded blocks. For a completed sample, use a separate output:

```bash
python environment/setup/compact_sampled.py \
  --source /mnt/hrm-text-sampled.uncompacted \
  --output /mnt/hrm-text-compacted
```

For canonical HF releases, use `--compact` in the initial preparation command;
canonical `--existing-sampled` requires an already finalized canonical manifest.
Separate compaction does not establish that provenance by itself. A separate
compacted output can be manifested as a diagnostic using `--existing-sampled`
with explicit local source/tokenizer/prefix arguments. Never
replace a dataset used by an active training run. `compaction.json` records the
source manifest digest, retained token count, and storage transformation. The
source files are never modified.

The script validates all sampled index bounds and lengths and emits a JSON manifest
with epoch row/token counts, full SHA-256 hashes for sampled arrays, tokenizer,
prefix configuration and sampler, plus a size inventory of the source arrays.
`available_epoch_ids` and `available_epoch_count` seal the actual contiguous epoch
directories; `sampling_parameters.epochs` records the actual sampled count. The
legacy `validation.epochs` list remains available, and `pins.epochs=4` retains the
default used by earlier releases. `metadata.json` stores the mean token count over
all available epochs, while the manifest also records each epoch's exact count.
The source inventory does **not** independently authenticate source file contents.
The sampled hashes are the identity to freeze for a published task release.
Pin the staging Python environment to `requirements-data.txt`; preserve the Rust
version/build log and resulting manifest with the release artifacts.

The HF download uses the pinned snapshot cache and a separate view of regular
files, linked or copied from that snapshot: the upstream Rust scanner skips HF
cache symlinks. Reuse checks this view against the exact snapshot inventory.
Tokenization checks the exact cleaned shard inventory and every
completion record and array before publishing `rsi_tokenization_receipt.json`.
This is necessary because the pinned Rust program logs individual shard errors
without returning a failing exit code. Reusing the tokenized cache requires a
matching recipe, source inventory, and full content hashes; reuse does not rerun
the tokenizer. Hashing can read hundreds of GiB. A nonempty cache without this
receipt, an interrupted build, or a changed tokenizer is rejected. Supply a fresh
`--cache` directory to rebuild; existing caches are never refreshed or deleted.

`environment/staged` contains real files (hard links when the filesystem allows
them), suitable for Docker `COPY`:

| Build input | Container path |
| --- | --- |
| `staged/sampled` | `/datasets/hrm-text/sampled` |
| `staged/tokenizer` | `/datasets/hrm-text/tokenizer` |
| `staged/data_manifest.json` | `/datasets/hrm-text/data_manifest.json` |

The staged metadata tokenizer path is rewritten for the container and its manifest
digest is updated. Original sampled files remain unchanged. Prefer sampling to
persistent staging storage for image builds; sampling to `/dev/shm` and copying to
disk requires additional storage. Existing output/staging directories are never
overwritten automatically.

## Existing tokenized data on this test machine

The supplied corpus is at
`/sg-pretrain/one/data_io/data_tokenized_bpe_chat_code_65k`; use its matching
`trained_tokenizers/bpe_chat_code` tokenizer and local `prefix_config.yaml`:

```bash
python environment/setup/prepare_data.py \
  --source-tokenized /sg-pretrain/one/data_io/data_tokenized_bpe_chat_code_65k \
  --tokenizer /sg-pretrain/one/data_io/trained_tokenizers/bpe_chat_code \
  --prefix-config /sg-pretrain/one/data_io/prefix_config.yaml \
  --output /dev/shm/hrm_text_rsi \
  --manifest /mnt/hrm-text-results/data_manifest.json
```

Add `--existing-sampled` to validate a completed sampler run without resampling.
An existing finalized manifest is checked and retained byte-for-byte, including
older four-epoch manifests. `--epochs` controls only new sampling: selecting one
training epoch from a four-epoch artifact does not alter its data, prefix mix, or
manifest identity. Set the selected epoch budget in the training command, which
consumes the first N available epochs. A scored run requires at least one complete
epoch; partial smoke runs remain diagnostic and receive zero reward. Model size
(L or XL) and GPU topology do not change the sampled data layout.

For a newly created manifest, `--hash-mode sampled` is an explicitly weaker diagnostic option: it
hashes every index array and metadata file, but only three blocks of the large
token file. Canonical release staging requires the default full hashing. Reusing an existing
manifest preserves its original integrity mode instead of silently changing its
identity. The complete source remains untouched when staging fewer new epochs or
compacting to a separate output.

This machine's local corpus contains additional chat/code data; its tokenizer and
prefix mix differ from the public pinned recipe. Its manifest labels it
`operator_supplied_local_tokenized`, and a training smoke test on it does not
establish results for the canonical public recipe. The local `sample_tokenized.py`
is identical to the pinned public sampler. The local, untracked
`encdec_sample_tokenized.py` produces separate instruction/response context lengths
and metadata; it is incompatible with public HRM-Text's PrefixLM loader, which
requires `max_seq_len` and a combined context bound. Use `sample_tokenized.py`.

## Public training data versus verifier data

Only the pretraining corpus and matching frozen tokenizer belong in the public
environment. Prepare benchmark snapshots separately using the verifier's pinned
dataset revisions, splits, evaluation code, and evaluation settings. The verifier
staging script serializes prompts and labels under `tests/staged` with a separate
manifest; scoring those records works offline without a Hugging Face dataset cache.
The operator's download cache remains separate. Do not copy a
shared `HF_HOME` into the public image: it may include benchmark answers, model
weights, or credentials. The verifier receives the frozen tokenizer but must not
mount benchmark records or their download cache into the training container.
Enable offline Hugging Face mode at runtime.
No pretrained checkpoint is part of the training data build.

## Data regression checks

```bash
python -m unittest discover -s environment/setup -p 'test_*.py' -v
```

Compaction tests need only NumPy and small temporary arrays. Sampling-prefix tests
use the pinned `data_io` checkout from `/var/tmp/hrm-text-data/data_io`, or from
`HRM_DATA_IO_TEST_ROOT`; they skip explicitly if that checkout is unavailable.
They do not download a corpus, access GPUs, or modify existing training data.
