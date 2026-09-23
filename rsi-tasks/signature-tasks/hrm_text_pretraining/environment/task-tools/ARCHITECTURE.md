# Candidate architecture interface

HRM-Text is the starting model for architecture research. You may change model
structure, tensor names/shapes, recurrence, parameter sharing and supporting
training code. The selected profile limits resources and data; its B/L/XL name
does not require the candidate to retain the baseline architecture.

The task launcher exports four artifacts into the submission directory:

- `weights.safetensors`: complete model state, using EMA when the recipe selects it.
- `model.json`: architecture construction settings, inference dtype, and a SHA-256 inventory of exported Python source.
- `source/`: Python model implementation and its local Python dependencies, preserving paths relative to the training checkout.
- `run.json`: training configuration, selected profile, progress, data/source identity, resource accounting and hashes for weights and the model manifest.

The manifest format is:

```json
{
  "schema_version": 1,
  "arch": {
    "name": "baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel",
    "head": "lm_head@LMHead",
    "hidden_size": 1024
  },
  "data_config": {"seq_len": 4096},
  "forward_dtype": "bfloat16",
  "source_files": {
    "models/lm_head.py": "<sha256>"
  }
}
```

This illustrates the format; a real manifest includes the complete resolved
architecture configuration and complete source inventory. The launcher writes
it automatically. Neither changing the manifest nor matching a digest proves
training provenance: preserve the source patch and full training trajectory.

## Model construction

`arch.name` and `arch.head` use the native `module@Class` convention. Modules
are imported below the exported `models` package. The core constructor accepts
a configuration dictionary; the head constructor accepts `(core, config)`.
The launcher also exports the resolved `data` configuration as `data_config`,
which the dev model constructor needs alongside architecture settings. The
verifier supplies trusted vocabulary, context and tokenizer metadata from
its sealed data reference. Candidate architecture declarations cannot redefine
those data semantics. Submitted weights must strict-load the submitted model,
not an unrelated canonical L model.

To introduce an architecture, put its Python implementation in the writable
checkout, for example `models/my_arch.py`, then select it through the native
configuration:

```bash
/task-tools/launch.sh --run-dir /app/output/attempts/my-architecture \
  --override 'arch.name=my_arch@MyArchitecture' \
  --export-dir /app/output/submission
```

Keep other constructor fields required by your implementation in the resolved
architecture config. Supporting Python dependencies must live in the checkout
or be available in the preinstalled runtime. Do not depend on absolute paths,
external files, runtime package downloads, pickle objects or hidden checkpoints.
Source artifacts are bounded to 16 MB and safetensors to 16 GB. Runtime memory
and the common inference timeout impose additional practical limits.

## Inference model methods

The headed model follows HRM-Text's native token/logit interface:

```python
cache = model.create_cache(
    max_batch_size=batch_size,
    max_seq_len=context_limit,
    dtype=torch.bfloat16,
    device=device,
)
carry = core.initial_carry(batch_size, dtype=torch.bfloat16)
result = model(carry=carry, batch={
    "inputs": inputs,
    "position_ids": position_ids,
    "cache": cache,
    "cache_lengths": cache_lengths,
})
logits = result[-1]  # [batch, sequence, frozen_vocabulary_size]
```

Prefill receives a complete prompt with batch size one. Decode receives the
next input token for each batch slot, with position IDs and cache lengths.
Cache state must be a PyTorch pytree whose tensor leaves can be sliced by batch
slot, as in the native implementation. The worker supplies the core
`initial_carry(batch_size, dtype)` result, which is `None` for the released HRM.
It does not replace that carry with forward outputs, matching native inference.
Keep mutable per-sequence inference state in the cache and make any carry
compatible with both single-slot prefill and batched decode. You may implement
an adapter for a different architecture. Return finite logits in the fixed vocabulary. Do not retain
information across independent benchmark examples outside each sequence's cache.

The original model/head also supply `initial_carry`, `compute_train_extra_args`
and the training forward interface; follow the pinned training implementation
when using the provided launcher. The default accumulation path expects loss
normalized by the mean valid response-token count across ranks and native
metric numerator/denominator pairs. If changing the training objective or head,
preserve that normalization contract or supply a mathematically equivalent
implementation and document it.

Checkpoint resume requires carry to retain its initial pytree structure, tensor
shapes/dtypes and scalar leaf types. Tensor values and devices are restored
against `initial_carry`; RNG state remains on its required device.

## Evaluation ownership

The verifier owns prompt construction, the frozen tokenizer, condition tokens,
context/generation limits, greedy token selection, stop conditions, answer
extraction, benchmark membership and metric computation. Candidate code receives
model inputs and returns logits through a restricted process. It does not
receive labels or scoring files. New tensor shapes and recurrent computation
are accepted through this interface; candidate-produced benchmark answers or
scores are not artifacts.

The worker runs with an unprivileged identity in a private Linux chroot with
restricted system calls. It receives only the necessary runtime, submitted model
and public data metadata. The verifier fails closed if it cannot establish
isolation. The current worker executes model forwards eagerly and transfers
logits to the parent; inference timing therefore describes this evaluator, not
the upstream compiled engine. The final wall watchdog and allocated GPUs are sealed in the selected profile;
evaluation time is reported separately from training computation. Record inference time and generated-token counts with quality.

Changing the model to recognize benchmark identities, substitute stored answers,
read unrelated files, access the parent process or alter the protocol violates
the task. Source and execution review remain part of accepting a scored result.
