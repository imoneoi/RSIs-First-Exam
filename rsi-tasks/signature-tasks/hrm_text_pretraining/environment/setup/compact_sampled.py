#!/usr/bin/env python3
"""Copy only the union of token ranges referenced by all available sampled epochs.

Sequence contents and row order are unchanged; only token storage and start
offsets change. The source is never modified. NumPy sorting needs roughly
64 bytes per referenced instruction/response interval, plus bounded copy buffers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sampled_layout import available_epoch_ids


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            result.update(block)
    return result.hexdigest()


def compact_sampled(source: Path, output: Path, block_tokens: int = 4 << 20) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate, non-nested directories')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if block_tokens < 1:
        raise ValueError('block_tokens must be positive')
    metadata = json.loads((source / 'metadata.json').read_text())
    if metadata.get('max_seq_len') != 4097:
        raise ValueError('Expected PrefixLM sampled data, max_seq_len=4097')
    tokens = np.load(source / 'tokens.npy', mmap_mode='r')
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError('Expected a one-dimensional integer token array')
    epoch_ids = available_epoch_ids(source)
    rows = [len(np.load(source / f'epoch_{epoch}/inst_start.npy', mmap_mode='r')) for epoch in epoch_ids]
    if not all(rows):
        raise ValueError('Every sampled epoch must have at least one row')
    intervals = np.empty((2 * sum(rows), 2), dtype=np.int64)
    cursor = 0
    total_sampled_tokens = 0
    for epoch, count in enumerate(rows):
        inst_lengths = np.load(source / f'epoch_{epoch}/inst_len.npy', mmap_mode='r')
        resp_lengths = np.load(source / f'epoch_{epoch}/resp_len.npy', mmap_mode='r')
        if inst_lengths.shape != (count,) or resp_lengths.shape != (count,):
            raise ValueError(f'Misaligned epoch {epoch} lengths')
        for lo in range(0, count, block_tokens):
            combined = inst_lengths[lo:lo + block_tokens].astype(np.int64) + resp_lengths[lo:lo + block_tokens].astype(np.int64)
            if np.any(combined > 4097):
                raise ValueError(f'Combined sequence exceeds PrefixLM context in epoch {epoch}')
            total_sampled_tokens += int(combined.sum())
        for kind in ('inst', 'resp'):
            starts = np.load(source / f'epoch_{epoch}/{kind}_start.npy', mmap_mode='r')
            lengths = np.load(source / f'epoch_{epoch}/{kind}_len.npy', mmap_mode='r')
            if starts.shape != (count,) or lengths.shape != (count,) or not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer):
                raise ValueError(f'Misaligned epoch {epoch} index arrays')
            for lo in range(0, count, block_tokens):
                hi = min(lo + block_tokens, count)
                s, n = starts[lo:hi].astype(np.int64), lengths[lo:hi].astype(np.int64)
                if np.any(s < 0) or np.any(n < 1) or np.any(n > 4097) or np.any(s > len(tokens) - n):
                    raise ValueError(f'Invalid {kind} ranges in epoch {epoch}')
                intervals[cursor + lo:cursor + hi, 0] = s
                intervals[cursor + lo:cursor + hi, 1] = s + n
            cursor += count
    if metadata.get('total_length') != round(total_sampled_tokens / len(epoch_ids)):
        raise ValueError('Sampled metadata total_length does not match available epochs')
    print(f'Merging {len(intervals):,} referenced ranges', flush=True)
    order = np.argsort(intervals[:, 0], kind='stable')
    intervals = intervals[order]
    del order
    furthest = np.maximum.accumulate(intervals[:, 1])
    boundaries = np.flatnonzero(np.r_[True, intervals[1:, 0] > furthest[:-1]])
    starts = intervals[boundaries, 0].copy()
    ends = furthest[np.r_[boundaries[1:] - 1, len(intervals) - 1]].copy()
    del intervals, furthest, boundaries
    lengths = ends - starts
    offsets = np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)]
    total = int(offsets[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    index_bytes = sum(p.stat().st_size for p in source.glob('epoch_*/*.npy'))
    required = total * tokens.dtype.itemsize + index_bytes + (1 << 20)
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError(f'Compacted output needs {required / 2**30:.2f} GiB free')
    output.mkdir()
    target = np.lib.format.open_memmap(output / 'tokens.npy', mode='w+', dtype=tokens.dtype, shape=(total,))
    # Expand interval shifts only for a bounded output block, avoiding a Python
    # loop per row and avoiding a giant index for every retained token at once.
    for lo in range(0, total, block_tokens):
        hi = min(lo + block_tokens, total)
        first = int(np.searchsorted(offsets, lo, side='right') - 1)
        stop = int(np.searchsorted(offsets, hi, side='left'))
        counts = np.minimum(offsets[first + 1:stop + 1], hi) - np.maximum(offsets[first:stop], lo)
        positions = np.arange(lo, hi, dtype=np.int64)
        positions += np.repeat(starts[first:stop] - offsets[first:stop], counts)
        target[lo:hi] = tokens[positions]
        if lo // block_tokens % 256 == 0:
            print(f'Copying retained tokens: {hi:,}/{total:,}', flush=True)
    target.flush()
    del target
    for epoch, count in enumerate(rows):
        destination = output / f'epoch_{epoch}'
        destination.mkdir()
        for kind in ('inst', 'resp'):
            old = np.load(source / f'epoch_{epoch}/{kind}_start.npy', mmap_mode='r')
            sizes = np.load(source / f'epoch_{epoch}/{kind}_len.npy', mmap_mode='r')
            remapped = np.lib.format.open_memmap(destination / f'{kind}_start.npy', mode='w+', dtype=np.int64, shape=(count,))
            for lo in range(0, count, block_tokens):
                hi = min(lo + block_tokens, count)
                segment = np.searchsorted(starts, old[lo:hi], side='right') - 1
                if np.any(segment < 0) or np.any(old[lo:hi] + sizes[lo:hi] > ends[segment]):
                    raise ValueError('Referenced sequence crosses a removed token range')
                remapped[lo:hi] = offsets[segment] + old[lo:hi] - starts[segment]
            remapped.flush()
            del remapped
            shutil.copy2(source / f'epoch_{epoch}/{kind}_len.npy', destination / f'{kind}_len.npy')
    shutil.copy2(source / 'metadata.json', output / 'metadata.json')
    provenance = {'algorithm': 'sorted_interval_union_v1', 'source_sampled': str(source),
                  'source_metadata_sha256': digest(source / 'metadata.json'),
                  'source_data_manifest_sha256': digest(source / 'data_manifest.json') if (source / 'data_manifest.json').is_file() else None,
                  'source_token_count': len(tokens), 'retained_token_count': total,
                  'merged_intervals': len(starts), 'epoch_rows': rows,
                  'available_epoch_ids': epoch_ids, 'available_epoch_count': len(epoch_ids),
                  'preserves_sequence_tokens_lengths_and_row_order': True}
    (output / 'compaction.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Compacted {len(tokens):,} to {total:,} tokens ({total / len(tokens):.2%}) at {output}', flush=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-tokens', type=int, default=4 << 20)
    args = parser.parse_args()
    compact_sampled(args.source, args.output, args.block_tokens)


if __name__ == '__main__':
    main()
