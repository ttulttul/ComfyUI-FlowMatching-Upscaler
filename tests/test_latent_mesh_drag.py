import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_mesh_drag import LatentMeshDrag, mesh_drag_warp  # noqa: E402


def test_mesh_drag_warp_noop_when_disabled():
    tensor = torch.randn(2, 4, 16, 16)

    out_points_zero = mesh_drag_warp(tensor, points=0, drag_min=0.0, drag_max=4.0, seed=123)
    assert torch.equal(out_points_zero, tensor)

    out_range_zero = mesh_drag_warp(tensor, points=8, drag_min=0.0, drag_max=0.0, seed=123)
    assert torch.equal(out_range_zero, tensor)


def test_mesh_drag_warp_deterministic_for_seed():
    tensor = torch.randn(1, 4, 32, 32)
    out1 = mesh_drag_warp(tensor, points=12, drag_min=1.0, drag_max=4.0, seed=999)
    out2 = mesh_drag_warp(tensor, points=12, drag_min=1.0, drag_max=4.0, seed=999)
    assert torch.allclose(out1, out2)


def test_mesh_drag_warp_changes_tensor_for_nonzero_drag():
    base = torch.arange(0, 32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    tensor = base.repeat(1, 4, 1, 1)

    out = mesh_drag_warp(tensor, points=8, drag_min=2.0, drag_max=6.0, seed=42)
    assert not torch.allclose(out, tensor)


def test_node_warps_noise_mask_in_lockstep():
    base = torch.linspace(0.0, 1.0, 16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    samples = base.repeat(1, 4, 1, 1)
    noise_mask = base.squeeze(1)
    latent = {"samples": samples, "noise_mask": noise_mask}

    node = LatentMeshDrag()
    (out_latent,) = node.drag(latent, seed=123, points=10, drag_min=1.0, drag_max=3.0)

    assert isinstance(out_latent, dict)
    assert out_latent["samples"].shape == samples.shape
    assert out_latent["noise_mask"].shape == noise_mask.shape
    assert torch.allclose(out_latent["noise_mask"], out_latent["samples"][:, 0])


def test_mesh_drag_warp_repeats_per_batch_across_extra_dims():
    base_frame = torch.randn(1, 2, 1, 16, 16).repeat(1, 1, 3, 1, 1)
    out = mesh_drag_warp(base_frame, points=6, drag_min=0.5, drag_max=2.0, seed=7)
    assert torch.allclose(out[:, :, 0], out[:, :, 1])
    assert torch.allclose(out[:, :, 1], out[:, :, 2])

