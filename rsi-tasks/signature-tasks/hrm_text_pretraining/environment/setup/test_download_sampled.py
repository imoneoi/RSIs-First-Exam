"""Offline fixtures for immutable sampled release authentication and staging."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import download_sampled as download
from prepare_data import fingerprint, sha256, validate_sampled


class SampledReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hrm-release-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / 'snapshot'
        sampled = self.snapshot / 'sampled'
        sampled.mkdir(parents=True)
        self.lock = {'schema_version': 1, 'repo_id': 'sapientinc/fixture', 'revision': 'a' * 40,
                     'release': 'fixture', 'format': 'hrm-text-prefixlm-v1', 'available_epoch_ids': [0, 1, 2, 3]}
        meta = {'tokenizer_info': {'vocab_size': 65536, 'tokenizer_path': download.RUNTIME_TOKENIZER},
                'max_seq_len': 4097, 'vocab_size': None, 'total_length': 8}
        (sampled / 'metadata.json').write_text(json.dumps(meta))
        np.save(sampled / 'tokens.npy', np.arange(32, dtype=np.uint16))
        for epoch in range(4):
            directory = sampled / f'epoch_{epoch}'
            directory.mkdir()
            for name, values in {'inst_start': [0, 8], 'inst_len': [2, 2],
                                 'resp_start': [2, 10], 'resp_len': [2, 2]}.items():
                np.save(directory / f'{name}.npy', np.array(values, dtype=np.int64))
        tokenizer = self.snapshot / 'tokenizer'
        tokenizer.mkdir()
        (tokenizer / 'tokenizer.json').write_text('{}\n')
        (tokenizer / 'config.json').write_text('{"model_type":"qwen3"}\n')
        aux = ('README.md', 'source_inventory.json', 'prefix_config.yaml', 'provenance/sample_tokenized.py')
        for name in aux:
            path = self.snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture\n')
        self.manifest = {
            'canonical_dataset': {k: self.lock[k] for k in ('repo_id', 'release', 'format', 'available_epoch_ids')},
            'available_epoch_ids': [0, 1, 2, 3], 'available_epoch_count': 4, 'sampled_integrity': 'full_sha256',
            'sampled_files': {p.relative_to(sampled).as_posix(): fingerprint(p) for p in sampled.rglob('*') if p.is_file()},
            'tokenizer': {p.name: fingerprint(p) for p in tokenizer.iterdir()},
            'release_auxiliary_files': {name: fingerprint(self.snapshot / name) for name in aux},
            'validation': validate_sampled(sampled),
        }
        self.seal()

    def seal(self):
        content = json.dumps(self.manifest, indent=2) + '\n'
        (self.snapshot / 'sampled/data_manifest.json').write_text(content)
        (self.snapshot / 'data_manifest.json').write_text(content)
        self.lock['manifest_sha256'] = sha256(self.snapshot / 'data_manifest.json')
        self.lock_path = self.root / 'lock.json'
        self.lock_path.write_text(json.dumps(self.lock))

    def test_valid_cached_snapshot_materializes_regular_files_and_preserves_identity(self):
        blob = self.root / 'token-blob'
        token_path = self.snapshot / 'sampled/tokens.npy'
        token_path.rename(blob)
        token_path.symlink_to(blob)
        output = self.root / 'staged'
        download.materialize_release(self.snapshot, output, self.lock)
        download.validate_release(output, self.lock)
        self.assertFalse(any(p.is_symlink() for p in output.rglob('*')))
        self.assertEqual((output / 'data_manifest.json').read_bytes(), (self.snapshot / 'data_manifest.json').read_bytes())
        self.assertEqual(sha256(blob), self.manifest['sampled_files']['tokens.npy']['sha256'])
        with self.assertRaises(FileExistsError):
            download.materialize_release(self.snapshot, output, self.lock)

    def test_corruption_rejects_before_output_creation(self):
        token_path = self.snapshot / 'sampled/tokens.npy'
        tokens = np.load(token_path, mmap_mode='r+')
        tokens[0] += 1
        tokens.flush()
        output = self.root / 'staged'
        with self.assertRaisesRegex(ValueError, 'full hash'):
            download.materialize_release(self.snapshot, output, self.lock)
        self.assertFalse(output.exists())

    def test_incomplete_or_extra_sampled_files_reject(self):
        path = self.snapshot / 'sampled/epoch_3/resp_len.npy'
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete or mixed'):
            download.validate_release(self.snapshot, self.lock)
        path.write_bytes(original)
        (self.snapshot / 'sampled/extra.npy').write_bytes(b'not-training-data')
        with self.assertRaisesRegex(ValueError, 'Incomplete or mixed'):
            download.validate_release(self.snapshot, self.lock)

    def test_wrong_descriptor_partial_hash_and_traversal_reject_even_when_resealed(self):
        self.manifest['canonical_dataset']['release'] = 'wrong'
        self.seal()
        with self.assertRaisesRegex(ValueError, 'identity'):
            download.validate_release(self.snapshot, self.lock)
        self.manifest['canonical_dataset']['release'] = 'fixture'
        self.manifest['sampled_files']['tokens.npy'].pop('sha256')
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Full SHA-256'):
            download.validate_release(self.snapshot, self.lock)
        self.manifest['sampled_files']['tokens.npy'] = fingerprint(self.snapshot / 'sampled/tokens.npy')
        self.manifest['release_auxiliary_files']['../outside'] = fingerprint(self.snapshot / 'README.md')
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Unsafe release file path'):
            download.validate_release(self.snapshot, self.lock)

    def test_manifest_lock_and_metadata_disagreement_reject(self):
        lock = self.lock | {'manifest_sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'operator-pinned'):
            download.validate_release(self.snapshot, lock)
        self.manifest['validation']['metadata']['total_length'] += 1
        self.seal()
        with self.assertRaisesRegex(ValueError, 'array semantics'):
            download.validate_release(self.snapshot, self.lock)

    def test_offline_cli_uses_only_pinned_cache_and_excludes_cache_files(self):
        (self.snapshot / '.cache').mkdir()
        (self.snapshot / '.cache/credential-placeholder').write_text('must not stage')
        calls = []
        def cached_snapshot(**kwargs):
            calls.append(kwargs)
            return str(self.snapshot)
        output = self.root / 'offline-staged'
        argv = ['download_sampled.py', '--output', str(output), '--lock', str(self.lock_path), '--offline']
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(snapshot_download=cached_snapshot)}), contextlib.redirect_stdout(io.StringIO()):
            download.main()
        self.assertTrue(calls[0]['local_files_only'])
        self.assertEqual(calls[0]['revision'], self.lock['revision'])
        self.assertEqual(calls[0]['repo_id'], self.lock['repo_id'])
        self.assertFalse((output / '.cache').exists())
        download.validate_release(output, self.lock)

    def test_unpinned_or_non_four_epoch_lock_is_rejected(self):
        for update in ({'schema_version': 0}, {'revision': 'main'}, {'manifest_sha256': None}, {'available_epoch_ids': [0]}):
            with self.subTest(update=update):
                self.lock_path.write_text(json.dumps(self.lock | update))
                with self.assertRaisesRegex(ValueError, 'finalized four-epoch'):
                    download.load_lock(self.lock_path)


if __name__ == '__main__':
    unittest.main()
