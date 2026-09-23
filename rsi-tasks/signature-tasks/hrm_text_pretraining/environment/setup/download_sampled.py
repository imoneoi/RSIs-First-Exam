#!/usr/bin/env python3
"""Authenticate a clean HF training snapshot and create safe local manifests.

The operator's HF client handles credentials. Only the twenty training/tokenizer
files and two deterministic runtime manifests enter the offline Work image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from prepare_data import FIELDS, fingerprint, link_or_copy, validate_sampled

HERE = Path(__file__).resolve().parent
LOCK = HERE / 'sampled_release.json'
RUNTIME_TOKENIZER = '/datasets/hrm-text/tokenizer'
EPOCHS = [0, 1, 2, 3]
DESCRIPTOR_KEYS = {'repo_id', 'release', 'format', 'available_epoch_ids'}
LOCK_KEYS = {'schema_version', 'repo_id', 'revision', 'manifest_sha256', 'release',
             'format', 'available_epoch_ids', 'training_manifest'}
MANIFEST_KEYS = {'schema_version', 'canonical_dataset', 'available_epoch_ids',
                 'available_epoch_count', 'sampled_integrity', 'sampled_files', 'tokenizer', 'validation'}
MANIFEST_PATHS = ('data_manifest.json', 'sampled/data_manifest.json')


def exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'Unexpected or missing {label} fields')


def positive_integer(value):
    return type(value) is int and value > 0


def four_epochs(ids, count=4):
    return isinstance(ids, list) and all(type(i) is int for i in ids) and ids == EPOCHS and type(count) is int and count == 4


def manifest_bytes(manifest):
    return (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def safe_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError('Release file paths must be strings')
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name or not path.parts
            or any(part in ('', '.', '..') or part.startswith('.') for part in path.parts)
            or '\\' in name):
        raise ValueError(f'Unsafe release file path: {name}')
    return name


def validate_stats(stats):
    """Whitelist the native validator's numeric and tokenizer semantics only."""
    exact_keys(stats, {'metadata', 'token_array_shape', 'token_array_dtype', 'epochs',
                       'available_epoch_ids', 'available_epoch_count'}, 'validation')
    if not four_epochs(stats['available_epoch_ids'], stats['available_epoch_count']):
        raise ValueError('Validation must describe all four sampled epochs')
    shape = stats['token_array_shape']
    if (not isinstance(shape, list) or len(shape) != 1 or not positive_integer(shape[0])
            or stats['token_array_dtype'] not in ('uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64')):
        raise ValueError('Invalid token-array validation shape or dtype')
    epochs = stats['epochs']
    if not isinstance(epochs, list) or len(epochs) != 4:
        raise ValueError('Missing native epoch validation records')
    for epoch, row in enumerate(epochs):
        exact_keys(row, {'epoch', 'rows', 'tokens_including_ar_shift'}, 'epoch validation')
        if (type(row['epoch']) is not int or row['epoch'] != epoch
                or not positive_integer(row['rows']) or not positive_integer(row['tokens_including_ar_shift'])):
            raise ValueError('Invalid native epoch validation counters')
    metadata = stats['metadata']
    exact_keys(metadata, {'max_seq_len', 'tokenizer_info', 'total_length', 'vocab_size'}, 'metadata')
    if (type(metadata['max_seq_len']) is not int or metadata['max_seq_len'] != 4097
            or metadata['vocab_size'] is not None or not positive_integer(metadata['total_length'])):
        raise ValueError('Invalid native PrefixLM metadata')
    info = metadata['tokenizer_info']
    exact_keys(info, {'boa', 'boq', 'eoa', 'eoq', 'condition_mapping', 'tokenizer_path', 'vocab_size'}, 'tokenizer metadata')
    if (info['tokenizer_path'] != RUNTIME_TOKENIZER or type(info['vocab_size']) is not int
            or info['vocab_size'] != 65536):
        raise ValueError('Tokenizer metadata requires the frozen vocabulary and container path')
    exact_keys(info['condition_mapping'], {'cot', 'direct', 'noisy', 'synth'}, 'condition mapping')
    special_tokens = [info[name] for name in ('boa', 'boq', 'eoa', 'eoq')] + list(info['condition_mapping'].values())
    if any(not isinstance(token, str) or not re.fullmatch(r'<\|[A-Za-z0-9_]{1,64}\|>', token) for token in special_tokens):
        raise ValueError('Invalid tokenizer special-token metadata')


def release_inventory(manifest: dict) -> dict:
    exact_keys(manifest, MANIFEST_KEYS, 'training manifest')
    if (type(manifest['schema_version']) is not int or manifest['schema_version'] != 1
            or not four_epochs(manifest['available_epoch_ids'], manifest['available_epoch_count'])
            or manifest['sampled_integrity'] != 'full_sha256'):
        raise ValueError('The safe manifest must fully authenticate all four sampled epochs')
    exact_keys(manifest['canonical_dataset'], DESCRIPTOR_KEYS, 'canonical dataset')
    if not four_epochs(manifest['canonical_dataset']['available_epoch_ids']):
        raise ValueError('Canonical dataset epoch IDs must be the four integer IDs')
    expected = {'tokens.npy', 'metadata.json'} | {
        f'epoch_{epoch}/{field}.npy' for epoch in EPOCHS for field in FIELDS}
    exact_keys(manifest['sampled_files'], expected, 'sampled file inventory')
    exact_keys(manifest['tokenizer'], {'tokenizer.json', 'config.json'}, 'tokenizer file inventory')
    inventory = {f'sampled/{safe_name(name)}': item for name, item in manifest['sampled_files'].items()}
    inventory.update({f'tokenizer/{safe_name(name)}': item for name, item in manifest['tokenizer'].items()})
    for name, item in inventory.items():
        exact_keys(item, {'bytes', 'sha256'}, f'file fingerprint for {name}')
        if (not positive_integer(item['bytes']) or not isinstance(item['sha256'], str)
                or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
            raise ValueError(f'Full SHA-256 and size required for every release file: {name}')
    validate_stats(manifest['validation'])
    return inventory


def validate_lock(lock: dict) -> tuple[dict, dict]:
    exact_keys(lock, LOCK_KEYS, 'release lock')
    if (not isinstance(lock, dict) or type(lock.get('schema_version')) is not int or lock['schema_version'] != 1
            or not isinstance(lock.get('repo_id'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*', lock['repo_id'])
            or not re.fullmatch(r'[0-9a-f]{40}', str(lock.get('revision', '')))
            or not re.fullmatch(r'[0-9a-f]{64}', str(lock.get('manifest_sha256', '')))
            or lock.get('format') != 'hrm-text-prefixlm-v1' or not four_epochs(lock.get('available_epoch_ids'))
            or not isinstance(lock.get('release'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', lock['release'])):
        raise ValueError('A finalized four-epoch repository/revision/manifest release lock is required')
    manifest = lock.get('training_manifest')
    inventory = release_inventory(manifest)
    if manifest['canonical_dataset'] != {key: lock[key] for key in DESCRIPTOR_KEYS}:
        raise ValueError('Embedded manifest identity differs from the release lock')
    if hashlib.sha256(manifest_bytes(manifest)).hexdigest() != lock['manifest_sha256']:
        raise ValueError('Embedded manifest differs from the operator-pinned hash')
    return manifest, inventory


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text())
    validate_lock(lock)
    return lock


def validate_release(root: Path, lock: dict, *, cache_symlinks: bool = False,
                     require_manifests: bool = False) -> tuple[dict, dict]:
    """Accept clean HF snapshots or staged trees with both exact local manifests."""
    root = Path(root)
    manifest, inventory = validate_lock(lock)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Release root must be a regular directory')
    allowed_root = {'sampled', 'tokenizer', 'README.md', '.gitattributes', '.cache', 'data_manifest.json'}
    if any(path.name not in allowed_root for path in root.iterdir()):
        raise ValueError('Unexpected non-training release files')
    present = [(root / name).exists() or (root / name).is_symlink() for name in MANIFEST_PATHS]
    if any(present) != all(present) or require_manifests and not all(present):
        raise ValueError('Both generated runtime manifests are required together')
    staged = all(present)
    for subtree in ('sampled', 'tokenizer'):
        directory = root / subtree
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f'Release subtree must be a regular directory: {subtree}')
        expected = {name for name in inventory if name.startswith(subtree + '/')}
        if staged and subtree == 'sampled':
            expected.add('sampled/data_manifest.json')
        expected_dirs = {str(PurePosixPath(name).parent) for name in expected} - {subtree}
        actual = set()
        for path in directory.rglob('*'):
            name = path.relative_to(root).as_posix()
            if path.is_dir() and not path.is_symlink():
                if name not in expected_dirs:
                    raise ValueError(f'Incomplete or mixed {subtree} directory')
            elif path.is_file() and (cache_symlinks or not path.is_symlink()):
                actual.add(name)
            else:
                raise ValueError(f'Release files must be present and regular: {name}')
        if actual != expected:
            raise ValueError(f'Incomplete or mixed {subtree} directory')
    if staged:
        for name in MANIFEST_PATHS:
            path = root / name
            if not path.is_file() or path.is_symlink() or path.read_bytes() != manifest_bytes(manifest):
                raise ValueError('Generated runtime manifests differ from the safe embedded manifest')
    for name, expected in inventory.items():
        if fingerprint(root / name) != expected:
            raise ValueError(f'Release file content differs from its full hash: {name}')
    if validate_sampled(root / 'sampled') != manifest['validation']:
        raise ValueError('Release array semantics differ from the sealed validation record')
    return manifest, inventory


def materialize_release(snapshot: Path, output: Path, lock: dict) -> dict:
    """Authenticate cached bytes, then atomically publish training-only assets."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite release output: {output}')
    manifest, inventory = validate_release(snapshot, lock, cache_symlinks=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        for name, expected in inventory.items():
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(str((snapshot / name).resolve()), str(destination))
            if not destination.samefile(snapshot / name) and fingerprint(destination) != expected:
                raise ValueError(f'Staged copy differs from the authenticated snapshot: {name}')
        for name in MANIFEST_PATHS:
            (temporary / name).write_bytes(manifest_bytes(manifest))
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New staged directory containing sampled/ and tokenizer/')
    parser.add_argument('--lock', type=Path, default=LOCK)
    parser.add_argument('--cache', type=Path, help='HF cache location; defaults beside output')
    parser.add_argument('--offline', action='store_true', help='Use only the already cached pinned snapshot')
    parser.add_argument('--verify-only', action='store_true', help='Authenticate staged files and both generated manifests without mutation')
    args = parser.parse_args()
    lock = load_lock(args.lock)
    output = args.output.absolute()
    if args.verify_only:
        validate_release(output, lock, require_manifests=True)
    else:
        if output.exists() or output.is_symlink():
            parser.error('Output already exists; use --verify-only or choose a new path')
        from huggingface_hub import snapshot_download
        cache = args.cache or output.parent / '.hf-sampled-cache'
        snapshot = Path(snapshot_download(repo_id=lock['repo_id'], repo_type='dataset', revision=lock['revision'],
                                         cache_dir=cache, local_files_only=args.offline,
                                         allow_patterns=sorted(release_inventory(lock['training_manifest']))))
        materialize_release(snapshot, output, lock)
    print(json.dumps({'output': str(output), 'repo_id': lock['repo_id'], 'revision': lock['revision'],
                      'manifest_sha256': lock['manifest_sha256'], 'available_epoch_ids': lock['available_epoch_ids']}, indent=2))


if __name__ == '__main__':
    main()
