"""Two-rank CPU regression for the dedicated DCP metadata process group."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch


def checkpoint_worker(rank, root):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    metadata_group = dist.new_group(backend="gloo")
    mesh = init_device_mesh("cpu", (2,))
    weight = distribute_tensor(torch.arange(16, dtype=torch.float32), mesh, [Shard(0)])
    moment = weight.clone() * 2
    state = {"model": {"weight": weight}, "optim": {"moment": moment}}
    dcp.save(state, checkpoint_id=Path(root) / "checkpoint", process_group=metadata_group)
    weight.zero_()
    moment.zero_()
    dcp.load(state, checkpoint_id=Path(root) / "checkpoint", process_group=metadata_group)
    expected = torch.arange(rank * 8, (rank + 1) * 8, dtype=torch.float32)
    torch.testing.assert_close(weight.to_local(), expected, rtol=0, atol=0)
    torch.testing.assert_close(moment.to_local(), expected * 2, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
    dist.destroy_process_group(metadata_group)
    dist.destroy_process_group()


class CheckpointMetadataTests(unittest.TestCase):
    def test_sharded_model_and_optimizer_roundtrip_with_separate_cpu_group(self):
        import torch.multiprocessing as multiprocessing
        # DeviceMesh may auto-create accelerator backends even for CPU tensors.
        # Hide devices before process spawn, before either child imports torch.
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            multiprocessing.spawn(checkpoint_worker, args=(root,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
