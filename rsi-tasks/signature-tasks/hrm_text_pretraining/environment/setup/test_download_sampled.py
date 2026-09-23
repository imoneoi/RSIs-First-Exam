"""CPU fixtures for clean HF assets and deterministic local integrity manifests."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
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
        info = {'vocab_size': 65536, 'tokenizer_path': download.RUNTIME_TOKENIZER,
                'boa': '<|box_start|>', 'boq': '<|im_start|>', 'eoa': '<|box_end|>', 'eoq': '<|im_end|>',
                'condition_mapping': {'cot': '<|object_ref_end|>', 'direct': '<|object_ref_start|>',
                                      'noisy': '<|quad_start|>', 'synth': '<|quad_end|>'}}
        meta = {'tokenizer_info': info, 'max_seq_len': 4097, 'vocab_size': None, 'total_length': 8}
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
        (self.snapshot / 'README.md').write_text('Generic training dataset.\n')
        (self.snapshot / '.gitattributes').write_text('*.npy filter=lfs diff=lfs merge=lfs -text\n')
        self.manifest = {
            'schema_version': 1,
            'canonical_dataset': {k: self.lock[k] for k in download.DESCRIPTOR_KEYS},
            'available_epoch_ids': [0, 1, 2, 3], 'available_epoch_count': 4, 'sampled_integrity': 'full_sha256',
            'sampled_files': {p.relative_to(sampled).as_posix(): fingerprint(p) for p in sampled.rglob('*') if p.is_file()},
            'tokenizer': {p.name: fingerprint(p) for p in tokenizer.iterdir()},
            'validation': validate_sampled(sampled),
        }
        self.lock_path = self.root / 'lock.json'
        self.seal()

    def seal(self):
        self.lock['training_manifest'] = copy.deepcopy(self.manifest)
        self.lock['manifest_sha256'] = hashlib.sha256(download.manifest_bytes(self.manifest)).hexdigest()
        self.lock_path.write_text(json.dumps(self.lock))

    def test_raw_snapshot_materializes_only_regular_training_files_and_local_manifests(self):
        blob = self.root / 'token-blob'
        token_path = self.snapshot / 'sampled/tokens.npy'
        token_path.rename(blob)
        token_path.symlink_to(blob)
        output = self.root / 'staged'
        download.materialize_release(self.snapshot, output, self.lock)
        download.validate_release(output, self.lock, require_manifests=True)
        self.assertFalse(any(p.is_symlink() for p in output.rglob('*')))
        self.assertFalse((self.snapshot / 'data_manifest.json').exists())
        expected = set(download.release_inventory(self.manifest)) | set(download.MANIFEST_PATHS)
        self.assertEqual({p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}, expected)
        self.assertEqual(len(expected), 22)
        for name in download.MANIFEST_PATHS:
            self.assertEqual((output / name).read_bytes(), download.manifest_bytes(self.manifest))
        self.assertEqual(sha256(blob), self.manifest['sampled_files']['tokens.npy']['sha256'])
        with self.assertRaises(FileExistsError):
            download.materialize_release(self.snapshot, output, self.lock)

    def test_generated_manifests_are_deterministic_and_safe_to_restage(self):
        first, second = self.root / 'first', self.root / 'second'
        download.materialize_release(self.snapshot, first, self.lock)
        reordered = copy.deepcopy(self.lock)
        reordered['training_manifest'] = dict(reversed(list(reordered['training_manifest'].items())))
        download.materialize_release(first, second, reordered)
        for name in download.MANIFEST_PATHS:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertEqual(sha256(second / name), self.lock['manifest_sha256'])

    def test_corrupted_arrays_and_tokenizer_reject_before_output_creation(self):
        for name in ('sampled/tokens.npy', 'sampled/epoch_0/inst_start.npy', 'tokenizer/tokenizer.json'):
            with self.subTest(name=name):
                path = self.snapshot / name
                original = path.read_bytes()
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                with self.assertRaisesRegex(ValueError, 'full hash'):
                    download.materialize_release(self.snapshot, self.root / 'staged', self.lock)
                self.assertFalse((self.root / 'staged').exists())
                path.write_bytes(original)

    def test_resealed_invalid_array_semantics_reject(self):
        path = self.snapshot / 'sampled/epoch_0/resp_start.npy'
        np.save(path, np.array([999, 10], dtype=np.int64))
        self.manifest['sampled_files']['epoch_0/resp_start.npy'] = fingerprint(path)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Out-of-bounds'):
            download.validate_release(self.snapshot, self.lock)

    def test_missing_extra_and_nonregular_training_files_reject(self):
        path = self.snapshot / 'sampled/epoch_3/resp_len.npy'
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete or mixed'):
            download.validate_release(self.snapshot, self.lock)
        path.write_bytes(original)
        for name in ('sampled/extra.npy', 'tokenizer/private.json'):
            with self.subTest(name=name):
                extra = self.snapshot / name
                extra.write_text('must not stage')
                with self.assertRaisesRegex(ValueError, 'Incomplete or mixed'):
                    download.validate_release(self.snapshot, self.lock)
                extra.unlink()
        extra = self.snapshot / 'sampled/empty-provenance'
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, 'Incomplete or mixed'):
            download.validate_release(self.snapshot, self.lock)
        extra.rmdir()
        extra = self.snapshot / 'tokenizer/dangling'
        extra.symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'regular'):
            download.validate_release(self.snapshot, self.lock, cache_symlinks=True)

    def test_cache_file_links_allowed_but_directory_links_reject(self):
        directory = self.snapshot / 'sampled/epoch_0'
        destination = self.root / 'epoch-blob'
        directory.rename(destination)
        directory.symlink_to(destination, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'regular'):
            download.validate_release(self.snapshot, self.lock, cache_symlinks=True)

    def test_unsafe_embedded_fields_reject_even_when_resealed(self):
        original = copy.deepcopy(self.manifest)
        paths = [(), ('canonical_dataset',), ('sampled_files', 'tokens.npy'), ('tokenizer', 'config.json'),
                 ('validation',), ('validation', 'metadata'), ('validation', 'metadata', 'tokenizer_info'),
                 ('validation', 'metadata', 'tokenizer_info', 'condition_mapping'), ('validation', 'epochs', 0)]
        for path in paths:
            with self.subTest(path=path):
                self.manifest = copy.deepcopy(original)
                node = self.manifest
                for key in path:
                    node = node[key]
                node['private_source_path'] = '/operator/private/corpus'
                self.seal()
                with self.assertRaisesRegex(ValueError, 'fields'):
                    download.load_lock(self.lock_path)
        self.manifest = original
        self.manifest['validation']['metadata']['tokenizer_info']['tokenizer_path'] = '/operator/private/tokenizer'
        self.seal()
        with self.assertRaisesRegex(ValueError, 'container path'):
            download.load_lock(self.lock_path)

    def test_missing_partial_and_unexpected_fingerprints_reject(self):
        original = copy.deepcopy(self.manifest)
        cases = [('sampled_files', 'tokens.npy', {'bytes': 1, 'sampled_sha256': 'a' * 64}),
                 ('sampled_files', 'tokens.npy', {'bytes': True, 'sha256': 'a' * 64}),
                 ('sampled_files', '../outside', {'bytes': 1, 'sha256': 'a' * 64}),
                 ('tokenizer', 'private.json', {'bytes': 1, 'sha256': 'a' * 64})]
        for section, name, value in cases:
            with self.subTest(name=name, value=value):
                self.manifest = copy.deepcopy(original)
                self.manifest[section][name] = value
                self.seal()
                with self.assertRaises(ValueError):
                    download.validate_release(self.snapshot, self.lock)

    def test_wrong_descriptor_manifest_digest_and_semantics_reject(self):
        with self.assertRaisesRegex(ValueError, 'operator-pinned'):
            download.validate_release(self.snapshot, self.lock | {'manifest_sha256': '0' * 64})
        self.manifest['canonical_dataset']['release'] = 'wrong'
        self.seal()
        with self.assertRaisesRegex(ValueError, 'identity'):
            download.validate_release(self.snapshot, self.lock)
        self.manifest['canonical_dataset']['release'] = 'fixture'
        self.manifest['validation']['metadata']['total_length'] += 1
        self.seal()
        with self.assertRaisesRegex(ValueError, 'array semantics'):
            download.validate_release(self.snapshot, self.lock)

    def test_old_root_provenance_is_rejected(self):
        (self.snapshot / 'source_inventory.json').write_text('old private provenance')
        with self.assertRaisesRegex(ValueError, 'non-training'):
            download.materialize_release(self.snapshot, self.root / 'staged', self.lock)

    def test_missing_mismatching_or_noncanonical_runtime_manifests_reject(self):
        staged = self.root / 'staged'
        download.materialize_release(self.snapshot, staged, self.lock)
        path = staged / 'data_manifest.json'
        expected = path.read_bytes()
        for changed in (b'{}\n', json.dumps(self.manifest).encode()):
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'safe embedded'):
                download.validate_release(staged, self.lock)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'required together'):
            download.validate_release(staged, self.lock)
        path.write_bytes(expected)
        download.validate_release(staged, self.lock)
        with self.assertRaisesRegex(ValueError, 'required together'):
            download.validate_release(self.snapshot, self.lock, require_manifests=True)

    def test_copy_or_manifest_write_failure_does_not_publish_partial_staging(self):
        def corrupt_copy(source, destination):
            shutil.copyfile(source, destination)
            with open(destination, 'ab') as stream:
                stream.write(b'corrupted copy')
        output = self.root / 'staged'
        with patch.object(download, 'link_or_copy', corrupt_copy), self.assertRaisesRegex(ValueError, 'Staged copy differs'):
            download.materialize_release(self.snapshot, output, self.lock)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('staged.staging-*')))
        original = Path.write_bytes
        def fail_second_manifest(path, data):
            if path.name == 'data_manifest.json' and path.parent.name == 'sampled':
                raise OSError('injected manifest publication failure')
            return original(path, data)
        with patch.object(Path, 'write_bytes', fail_second_manifest), self.assertRaisesRegex(OSError, 'injected'):
            download.materialize_release(self.snapshot, output, self.lock)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('staged.staging-*')))

    def test_offline_cli_fetches_only_pinned_training_files_and_excludes_cache_card(self):
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
        self.assertEqual(set(calls[0]['allow_patterns']), set(download.release_inventory(self.manifest)))
        for name in ('.cache', 'README.md', '.gitattributes', 'source_inventory.json', 'provenance'):
            self.assertFalse((output / name).exists())
        download.validate_release(output, self.lock, require_manifests=True)
        verify = ['download_sampled.py', '--output', str(output), '--lock', str(self.lock_path), '--verify-only']
        with patch.object(sys, 'argv', verify), patch.dict(sys.modules, {'huggingface_hub': None}), contextlib.redirect_stdout(io.StringIO()):
            download.main()
        self.assertEqual(len(calls), 1)

    def test_boolean_descriptor_epoch_ids_reject(self):
        self.manifest['canonical_dataset']['available_epoch_ids'] = [False, True, 2, 3]
        self.seal()
        with self.assertRaisesRegex(ValueError, 'integer IDs'):
            download.load_lock(self.lock_path)

    def test_private_top_level_lock_fields_reject(self):
        self.lock['operator_context'] = {'source_path': '/operator/private/corpus'}
        self.lock_path.write_text(json.dumps(self.lock))
        with self.assertRaisesRegex(ValueError, 'release lock fields'):
            download.load_lock(self.lock_path)

    def test_unpinned_non_four_epoch_or_missing_embedded_lock_rejects(self):
        for update in ({'schema_version': 0}, {'revision': 'main'}, {'manifest_sha256': None},
                       {'available_epoch_ids': [False, True, 2, 3]}, {'training_manifest': None}):
            with self.subTest(update=update):
                self.lock_path.write_text(json.dumps(self.lock | update))
                with self.assertRaises(ValueError):
                    download.load_lock(self.lock_path)


if __name__ == '__main__':
    unittest.main()
