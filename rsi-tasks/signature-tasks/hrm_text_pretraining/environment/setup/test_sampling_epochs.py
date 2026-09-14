"""Check epoch-prefix identity using the pinned upstream sampler on tiny data.

Set HRM_DATA_IO_TEST_ROOT if the pinned checkout is outside the setup cache.
This test never downloads data or changes an existing corpus.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from prepare_data import FIELDS, PINS, sha256, validate_sampled, write_manifest


class SamplingEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('HRM_DATA_IO_TEST_ROOT', '/var/tmp/hrm-text-data/data_io')).resolve()
        if not (cls.upstream / 'sample_tokenized.py').is_file():
            raise unittest.SkipTest('Run setup first or set HRM_DATA_IO_TEST_ROOT to the pinned data_io checkout')
        git = ['git', '-c', f'safe.directory={cls.upstream}', '-C', str(cls.upstream)]
        revision = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
        if revision != PINS['data_io_revision']:
            raise unittest.SkipTest('HRM_DATA_IO_TEST_ROOT is not the pinned data_io revision')
        pinned_sampler = subprocess.check_output([*git, 'show',
                                                  f"{PINS['data_io_revision']}:sample_tokenized.py"])
        if hashlib.sha256(pinned_sampler).hexdigest() != sha256(cls.upstream / 'sample_tokenized.py'):
            raise AssertionError('The pinned data_io sampler has local modifications')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hrm-epoch-prefix-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        task = self.source / 'tiny.jsonl'
        task.mkdir(parents=True)
        self.tokenizer = self.root / 'tokenizer'
        self.tokenizer.mkdir()
        (self.tokenizer / 'tokenizer.json').write_text('{}\n')
        (self.tokenizer / 'config.json').write_text('{"model_type":"qwen3"}\n')
        (self.source / 'tokenizer_info.json').write_text(json.dumps({'vocab_size': 65536, 'tokenizer_path': str(self.tokenizer)}))
        inst_len = np.full(12, 2, dtype=np.int64)
        resp_len = 2 + np.arange(12, dtype=np.int64) % 4
        inst_start = np.r_[0, np.cumsum(inst_len + resp_len)[:-1]]
        for name, array in {'inst_start': inst_start, 'inst_len': inst_len,
                            'resp_start': inst_start + inst_len, 'resp_len': resp_len}.items():
            np.save(task / f'{name}.npy', array)
        np.save(task / 'tokens.npy', np.arange(int((inst_len + resp_len).sum()), dtype=np.int32))
        self.prefix = self.root / 'prefix.yaml'
        self.prefix.write_text('- prefix: tiny\n  max_per_file: 2\n')

    def sample(self, epochs):
        output = self.root / f'sampled_{epochs}'
        subprocess.run([sys.executable, str(self.upstream / 'sample_tokenized.py'),
                        f'tokenized_path={self.source}', f'prefix_config_path={self.prefix}',
                        f'output_path={output}', f'epochs={epochs}', 'seed=0', 'context_size=4097'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output

    def test_one_epoch_is_exact_prefix_of_four_and_later_epochs_add_documents(self):
        one, four = self.sample(1), self.sample(4)
        for name in ('tokens.npy', *(f'epoch_0/{field}.npy' for field in FIELDS)):
            self.assertEqual(sha256(one / name), sha256(four / name))
        for root, count in ((one, 1), (four, 4)):
            stats = validate_sampled(root)
            self.assertEqual(stats['available_epoch_ids'], list(range(count)))
            self.assertEqual(stats['available_epoch_count'], count)
            self.assertEqual(stats['metadata']['total_length'],
                             round(sum(row['tokens_including_ar_shift'] for row in stats['epochs']) / count))
        first_rows = set(np.load(four / 'epoch_0/inst_start.npy'))
        later_rows = set(np.load(four / 'epoch_1/inst_start.npy'))
        self.assertTrue(later_rows - first_rows)

    def test_existing_legacy_four_epoch_manifest_identity_survives_one_epoch_budget(self):
        output = self.sample(4)
        args = argparse.Namespace(output=output, manifest=None, hash_mode='full', existing_sampled=False, epochs=4)
        path = write_manifest(args, self.source, self.tokenizer, self.prefix, self.upstream, 'operator_supplied_local_tokenized')
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest['available_epoch_count'], 4)
        self.assertEqual(manifest['sampling_parameters']['epochs'], 4)
        # Existing releases infer availability from validation.epochs, so no
        # rewrite is required merely to support the new selected run budget.
        for target in (manifest, manifest['validation']):
            target.pop('available_epoch_ids')
            target.pop('available_epoch_count')
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        before = path.read_bytes()
        args.existing_sampled, args.epochs = True, 1
        self.assertEqual(write_manifest(args, self.source, self.tokenizer, self.prefix, self.upstream, 'operator_supplied_local_tokenized'), path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(validate_sampled(output)['available_epoch_count'], 4)
        changed = np.load(output / 'tokens.npy', mmap_mode='r+')
        changed[0] += 1
        changed.flush()
        with self.assertRaises(ValueError):
            write_manifest(args, self.source, self.tokenizer, self.prefix, self.upstream, 'operator_supplied_local_tokenized')
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
