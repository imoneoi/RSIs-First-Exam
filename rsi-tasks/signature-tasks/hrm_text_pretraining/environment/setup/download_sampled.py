#!/usr/bin/env python3
"""Fetch the immutable sampled release before building an offline Work image.

The private Hugging Face credential is read by the operator's normal HF client;
credentials and cache metadata are never copied into the staged task assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from prepare_data import FIELDS, fingerprint, link_or_copy, sha256, validate_sampled

HERE = Path(__file__).resolve().parent
LOCK = HERE / 'sampled_release.json'
RUNTIME_TOKENIZER = '/datasets/hrm-text/tokenizer'


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text())
    if (lock.get('schema_version') != 1 or not isinstance(lock.get('repo_id'), str) or '/' not in lock['repo_id']
            or not re.fullmatch(r'[0-9a-f]{40}', str(lock.get('revision', '')))
            or not re.fullmatch(r'[0-9a-f]{64}', str(lock.get('manifest_sha256', '')))
            or lock.get('format') != 'hrm-text-prefixlm-v1'
            or lock.get('available_epoch_ids') != [0, 1, 2, 3]
            or not isinstance(lock.get('release'), str)):
        raise ValueError('A finalized four-epoch repository/revision/manifest release lock is required')
    return lock


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not isinstance(name, str) or path.is_absolute() or path.as_posix() != name
            or any(part in ('', '.', '..') or part.startswith('.') for part in path.parts)
            or '\\' in name or not path.parts):
        raise ValueError(f'Unsafe release file path: {name}')
    return name


def release_inventory(manifest: dict) -> dict:
    epochs = manifest.get('available_epoch_ids')
    if epochs != [0, 1, 2, 3] or manifest.get('available_epoch_count') != 4:
        raise ValueError('The canonical release must contain all four sampled epochs')
    expected = {'tokens.npy', 'metadata.json'} | {
        f'epoch_{epoch}/{field}.npy' for epoch in epochs for field in FIELDS}
    sampled = manifest.get('sampled_files', {})
    if set(sampled) != expected:
        raise ValueError('Incomplete or unexpected sampled array inventory')
    tokenizer = manifest.get('tokenizer', {})
    if 'tokenizer.json' not in tokenizer or 'config.json' not in tokenizer:
        raise ValueError('The release must include the complete frozen tokenizer')
    auxiliary = manifest.get('release_auxiliary_files', {})
    if not {'README.md', 'source_inventory.json', 'prefix_config.yaml', 'provenance/sample_tokenized.py'} <= set(auxiliary):
        raise ValueError('Missing sampled release provenance files')
    inventory = {f'sampled/{safe_name(name)}': item for name, item in sampled.items()}
    inventory.update({f'tokenizer/{safe_name(name)}': item for name, item in tokenizer.items()})
    for name, item in auxiliary.items():
        safe_name(name)
        if name in inventory or name in ('data_manifest.json', 'sampled/data_manifest.json') or name.startswith(('sampled/', 'tokenizer/')):
            raise ValueError('Auxiliary release file collides with a data asset')
        inventory[name] = item
    for name, item in inventory.items():
        if (not isinstance(item, dict) or type(item.get('bytes')) is not int or item['bytes'] < 0
                or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))):
            raise ValueError(f'Full SHA-256 and size required for every release file: {name}')
    return inventory


def validate_release(root: Path, lock: dict, *, cache_symlinks: bool = False) -> tuple[dict, dict]:
    manifest_path = root / 'sampled/data_manifest.json'
    if sha256(manifest_path) != lock['manifest_sha256']:
        raise ValueError('Sampled manifest differs from the operator-pinned release')
    if (root / 'data_manifest.json').read_bytes() != manifest_path.read_bytes():
        raise ValueError('Root and sampled data manifests must be identical')
    manifest = json.loads(manifest_path.read_text())
    descriptor = {key: lock[key] for key in ('repo_id', 'release', 'format', 'available_epoch_ids')}
    if manifest.get('canonical_dataset') != descriptor or manifest.get('sampled_integrity') != 'full_sha256':
        raise ValueError('Sampled release identity or integrity mode differs from the pinned contract')
    inventory = release_inventory(manifest)
    for subtree in ('sampled', 'tokenizer'):
        actual = {p.relative_to(root).as_posix() for p in (root / subtree).rglob('*') if p.is_file()}
        expected = {name for name in inventory if name.startswith(subtree + '/')}
        if subtree == 'sampled':
            expected.add('sampled/data_manifest.json')
        if actual != expected:
            raise ValueError(f'Incomplete or mixed {subtree} directory')
    for name in (*inventory, 'data_manifest.json', 'sampled/data_manifest.json'):
        path = root / name
        if not path.is_file() or (not cache_symlinks and any(p.is_symlink() for p in (path, *path.parents) if p != root.parent)):
            raise ValueError(f'Release files must be present and regular: {name}')
    for name, expected in inventory.items():
        if fingerprint(root / name) != expected:
            raise ValueError(f'Release file content differs from its full hash: {name}')
    stats = validate_sampled(root / 'sampled')
    if stats != manifest.get('validation'):
        raise ValueError('Release array semantics differ from the sealed validation record')
    if stats['metadata']['tokenizer_info'].get('tokenizer_path') != RUNTIME_TOKENIZER:
        raise ValueError('Canonical metadata must use the frozen container tokenizer path')
    return manifest, inventory


def materialize_release(snapshot: Path, output: Path, lock: dict) -> dict:
    """Authenticate cached bytes, then publish a regular-file staging tree atomically."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite release output: {output}')
    manifest, inventory = validate_release(snapshot, lock, cache_symlinks=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        for name in (*inventory, 'data_manifest.json', 'sampled/data_manifest.json'):
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(str((snapshot / name).resolve()), str(destination))
        # Hash the copy too when hard-linking was unavailable (e.g. cross-device).
        for name, expected in inventory.items():
            path = temporary / name
            if not path.samefile(snapshot / name) and fingerprint(path) != expected:
                raise ValueError(f'Staged copy differs from the authenticated snapshot: {name}')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New environment/staged directory containing sampled/ and tokenizer/')
    parser.add_argument('--lock', type=Path, default=LOCK)
    parser.add_argument('--cache', type=Path, help='HF cache location; defaults beside output so hard links avoid a second data copy')
    parser.add_argument('--offline', action='store_true', help='Use only the already cached pinned snapshot')
    parser.add_argument('--verify-only', action='store_true', help='Authenticate an existing output without download or mutation')
    args = parser.parse_args()
    lock = load_lock(args.lock)
    output = args.output.absolute()
    if args.verify_only:
        validate_release(output, lock)
    else:
        if output.exists() or output.is_symlink():
            parser.error('Output already exists; use --verify-only or choose a new path')
        from huggingface_hub import snapshot_download
        cache = args.cache or output.parent / '.hf-sampled-cache'
        snapshot = Path(snapshot_download(repo_id=lock['repo_id'], repo_type='dataset', revision=lock['revision'],
                                         cache_dir=cache, local_files_only=args.offline))
        materialize_release(snapshot, output, lock)
    print(json.dumps({'output': str(output), 'repo_id': lock['repo_id'], 'revision': lock['revision'],
                      'manifest_sha256': lock['manifest_sha256'], 'available_epoch_ids': lock['available_epoch_ids']}, indent=2))


if __name__ == '__main__':
    main()
