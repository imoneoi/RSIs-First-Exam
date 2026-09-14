"""Fail closed on partial upstream tokenization and incompatible cached data."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_data


class TokenizedCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='hrm-tokenization-receipt-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clean = self.root / 'cleaned'
        for directory, name in (('data', 'first'), ('data_clustered', 'second')):
            path = self.clean / directory / f'{name}.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text('{"condition":"direct","instruction":"q","response":"a"}\n')
        self.source = self.root / 'tokenized'
        self.tokenizer = self.root / 'tokenizer'
        self.tokenizer.mkdir()
        (self.tokenizer / 'tokenizer.json').write_text('{"test":"first tokenizer"}')
        (self.tokenizer / 'config.json').write_text('{}')
        self.upstream = self.root / 'upstream'
        self.upstream.mkdir()

    def fake_cargo(self, *args, **kwargs):
        self.source.mkdir()
        (self.source / 'tokenizer_info.json').write_text('{"vocab_size":65536}')
        for name, original in prepare_data.cleaned_inputs(self.clean).items():
            path = self.source / name
            path.mkdir()
            for key, array in {'tokens': [1, 2, 3, 4], 'inst_start': [0],
                               'inst_len': [2], 'resp_start': [2], 'resp_len': [2]}.items():
                np.save(path / f'{key}.npy', np.array(array, dtype=np.int64))
            (path / 'metadata.json').write_text(json.dumps({
                'source_mtime': int(original.stat().st_mtime), 'source_size': original.stat().st_size}))

    def prepare(self):
        return prepare_data.prepare_tokenized(self.clean, self.source, self.tokenizer, self.upstream)

    def test_complete_cache_reuse_authenticates_files_without_rerunning_cargo(self):
        with patch.object(prepare_data, 'run', side_effect=self.fake_cargo) as cargo:
            first = self.prepare()
            self.assertEqual(self.prepare(), first)
            self.assertEqual(cargo.call_count, 1)
        self.assertEqual(len(first['files']), 13)

    def test_cargo_success_with_missing_or_partial_shard_is_rejected(self):
        import shutil
        for field in (None, 'metadata.json', 'tokens.npy'):
            with self.subTest(field=field):
                def incomplete(*args, **kwargs):
                    self.fake_cargo()
                    if field is None:
                        shutil.rmtree(self.source / 'second.jsonl')
                    else:
                        (self.source / 'second.jsonl' / field).unlink()
                with patch.object(prepare_data, 'run', side_effect=incomplete):
                    with self.assertRaisesRegex(ValueError, 'incomplete'):
                        self.prepare()
                self.assertFalse((self.source / prepare_data.TOKENIZATION_RECEIPT).exists())
                shutil.rmtree(self.source)

    def test_nonempty_unsealed_cache_is_not_modified(self):
        self.fake_cargo()
        marker = self.source / 'first.jsonl/tokens.npy'
        before = marker.read_bytes()
        with patch.object(prepare_data, 'run') as cargo:
            with self.assertRaisesRegex(ValueError, 'without a sealed recipe receipt'):
                self.prepare()
            cargo.assert_not_called()
        self.assertEqual(marker.read_bytes(), before)

    def test_changed_tokenizer_or_input_recipe_rejected_without_cargo(self):
        with patch.object(prepare_data, 'run', side_effect=self.fake_cargo):
            self.prepare()
        tokenizer_file = self.tokenizer / 'tokenizer.json'
        original = tokenizer_file.read_bytes()
        with patch.object(prepare_data, 'run') as cargo:
            tokenizer_file.write_text('{"test":"different tokenizer"}')
            with self.assertRaisesRegex(ValueError, 'different recipe'):
                self.prepare()
            tokenizer_file.write_bytes(original)
            (self.clean / 'data/first.jsonl').write_text('changed source')
            with self.assertRaisesRegex(ValueError, 'different recipe'):
                self.prepare()
            cargo.assert_not_called()

    def test_same_size_changed_array_rejected(self):
        with patch.object(prepare_data, 'run', side_effect=self.fake_cargo):
            self.prepare()
        array = self.source / 'first.jsonl/tokens.npy'
        np.save(array, np.array([4, 3, 2, 1], dtype=np.int64))
        with patch.object(prepare_data, 'run') as cargo:
            with self.assertRaisesRegex(ValueError, 'content differs'):
                self.prepare()
            cargo.assert_not_called()

    def test_colliding_upstream_names_rejected_before_tokenization(self):
        (self.clean / 'data_clustered/first.jsonl').write_text('{}')
        with patch.object(prepare_data, 'run') as cargo:
            with self.assertRaisesRegex(ValueError, 'collide'):
                self.prepare()
            cargo.assert_not_called()

    def test_hf_snapshot_symlinks_become_regular_files_for_rust_scanner(self):
        blob = self.root / 'blob'
        path = self.clean / 'data/first.jsonl'
        path.rename(blob)
        path.symlink_to(blob)
        output = self.root / 'regular-cleaned'
        prepare_data.materialize_cleaned_snapshot(self.clean, output)
        copied = output / 'data/first.jsonl'
        self.assertTrue(copied.is_file())
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), blob.read_bytes())
        self.assertEqual(prepare_data.materialize_cleaned_snapshot(self.clean, output), output)
        (output / 'data/unrelated.jsonl').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'incomplete or mixed'):
            prepare_data.materialize_cleaned_snapshot(self.clean, output)

    def test_unmanifested_existing_samples_cannot_be_labeled_canonical(self):
        self.fake_cargo()
        sampled = self.root / 'sampled'
        sampled.mkdir()
        import shutil
        shutil.copytree(self.source / 'first.jsonl', sampled / 'epoch_0')
        (sampled / 'epoch_0/tokens.npy').rename(sampled / 'tokens.npy')
        (sampled / 'metadata.json').write_text(json.dumps({
            'max_seq_len': 4097, 'total_length': 4, 'tokenizer_info': {'vocab_size': 65536}}))
        args = argparse.Namespace(output=sampled, manifest=None, existing_sampled=True,
                                  epochs=1, hash_mode='full')
        with self.assertRaisesRegex(ValueError, 'existing unmanifested samples canonical'):
            prepare_data.write_manifest(args, self.source, self.tokenizer,
                                        self.root / 'prefix.yaml', self.upstream,
                                        'pinned_hf_cleaned_then_tokenized')
        self.assertFalse((sampled / 'data_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
