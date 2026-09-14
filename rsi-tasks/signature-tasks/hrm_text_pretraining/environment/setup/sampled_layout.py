"""Shared layout checks for immutable, consecutively sampled epoch artifacts."""
from pathlib import Path

MAX_EPOCHS = 4


def available_epoch_ids(root: Path) -> list[int]:
    paths = sorted(root.glob('epoch_*'))
    expected = [f'epoch_{epoch}' for epoch in range(len(paths))]
    if not 1 <= len(paths) <= MAX_EPOCHS or [p.name for p in paths] != expected:
        raise ValueError('Expected contiguous epoch_0 through epoch_N-1, with 1 <= N <= 4')
    if any(not p.is_dir() or p.is_symlink() for p in paths):
        raise ValueError('Sampled epochs must be real directories')
    return list(range(len(paths)))
