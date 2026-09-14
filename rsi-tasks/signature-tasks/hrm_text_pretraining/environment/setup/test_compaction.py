"""Sequence-preservation regression tests; no training data or GPU needed."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np

from compact_sampled import compact_sampled, digest
from prepare_data import validate_sampled


class CompactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hrm-compaction-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.tokens = np.arange(300, dtype=np.int32) * 7
        np.save(self.source / 'tokens.npy', self.tokens)
        self.rows = {}
        self.referenced_positions = set()
        totals = []
        for epoch in range(4):
            destination = self.source / f'epoch_{epoch}'
            destination.mkdir()
            # Duplicates, overlaps, gaps, adjacency, a long range, and a
            # one-token response exercise both union and copy boundaries.
            row = {'inst_start': np.array([3, 20, 20, 51, 150]),
                   'inst_len': np.array([3, 2, 2, 4, 12]),
                   'resp_start': np.array([30, 50, 70, 100 + epoch, 162]),
                   'resp_len': np.array([2, 2, 5, 4, 1])}
            permutation = np.random.default_rng(epoch).permutation(5)
            self.rows[epoch] = {key: value[permutation] for key, value in row.items()}
            for key, value in self.rows[epoch].items():
                np.save(destination / f'{key}.npy', value)
            totals.append(int(row['inst_len'].sum() + row['resp_len'].sum()))
            for kind in ('inst', 'resp'):
                for start, length in zip(row[f'{kind}_start'], row[f'{kind}_len']):
                    self.referenced_positions.update(range(start, start + length))
        metadata = {'max_seq_len': 4097, 'total_length': round(sum(totals) / 4),
                    'tokenizer_info': {'vocab_size': 65536}}
        (self.source / 'metadata.json').write_text(json.dumps(metadata))

    def hashes(self, path):
        return {p.relative_to(path).as_posix(): digest(p) for p in path.rglob('*') if p.is_file()}

    def test_preserves_sequences_row_order_and_source_across_block_sizes(self):
        original_hashes = self.hashes(self.source)
        previous = None
        for block_size in (1, 7, 500):
            with self.subTest(block_size=block_size):
                destination = self.root / f'compact_{block_size}'
                compact_sampled(self.source, destination, block_size)
                compact_tokens = np.load(destination / 'tokens.npy')
                np.testing.assert_array_equal(compact_tokens, self.tokens[sorted(self.referenced_positions)])
                self.assertEqual(len(compact_tokens), 37)
                for epoch, row in self.rows.items():
                    for kind in ('inst', 'resp'):
                        starts = np.load(destination / f'epoch_{epoch}/{kind}_start.npy')
                        lengths = np.load(destination / f'epoch_{epoch}/{kind}_len.npy')
                        np.testing.assert_array_equal(lengths, row[f'{kind}_len'])
                        for old, new, length in zip(row[f'{kind}_start'], starts, lengths):
                            np.testing.assert_array_equal(self.tokens[old:old + length], compact_tokens[new:new + length])
                current = self.hashes(destination)
                if previous is not None:
                    self.assertEqual(previous, current)
                previous = current
        self.assertEqual(original_hashes, self.hashes(self.source))

    def test_refuses_overwrite_and_nested_output(self):
        destination = self.root / 'existing'
        destination.mkdir()
        marker = destination / 'keep.txt'
        marker.write_text('preserve me')
        with self.assertRaises(FileExistsError):
            compact_sampled(self.source, destination)
        self.assertEqual(marker.read_text(), 'preserve me')
        for destination in (self.source, self.source / 'nested', self.root):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                compact_sampled(self.source, destination)

    def test_compacts_every_available_epoch_for_one_through_four(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                source = self.root / f'source_{count}'
                shutil.copytree(self.source, source)
                for epoch in range(count, 4):
                    shutil.rmtree(source / f'epoch_{epoch}')
                destination = self.root / f'prefix_{count}'
                provenance = compact_sampled(source, destination, 7)
                self.assertEqual(provenance['available_epoch_ids'], list(range(count)))
                self.assertEqual(provenance['available_epoch_count'], count)
                self.assertEqual(validate_sampled(destination)['available_epoch_count'], count)
                token_array = np.load(destination / 'tokens.npy')
                for epoch in range(count):
                    row = self.rows[epoch]
                    for kind in ('inst', 'resp'):
                        starts = np.load(destination / f'epoch_{epoch}/{kind}_start.npy')
                        for old, new, length in zip(row[f'{kind}_start'], starts, row[f'{kind}_len']):
                            np.testing.assert_array_equal(self.tokens[old:old + length], token_array[new:new + length])

    def test_rejects_noncontiguous_or_excess_epoch_directories(self):
        (self.source / 'epoch_1').rename(self.source / 'epoch_4')
        with self.assertRaises(ValueError):
            compact_sampled(self.source, self.root / 'gap')
        (self.source / 'epoch_4').rename(self.source / 'epoch_1')
        (self.source / 'epoch_4').mkdir()
        with self.assertRaises(ValueError):
            compact_sampled(self.source, self.root / 'excess')

    def test_rejects_invalid_ranges_and_copy_block(self):
        path = self.source / 'epoch_0/resp_start.npy'
        original = np.load(path)
        for bad_value in (-1, len(self.tokens)):
            broken = original.copy()
            broken[0] = bad_value
            np.save(path, broken)
            destination = self.root / f'invalid_{bad_value}'
            with self.subTest(start=bad_value), self.assertRaises(ValueError):
                compact_sampled(self.source, destination)
            self.assertFalse(destination.exists())
        np.save(path, original)
        with self.assertRaises(ValueError):
            compact_sampled(self.source, self.root / 'zero_block', 0)
        lengths = self.source / 'epoch_0/resp_len.npy'
        broken = np.load(lengths)
        broken[0] = 0
        np.save(lengths, broken)
        with self.assertRaises(ValueError):
            compact_sampled(self.source, self.root / 'empty_response')


if __name__ == '__main__':
    unittest.main()
