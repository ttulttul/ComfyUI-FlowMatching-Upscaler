import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_mesh_drag import (  # noqa: E402
    ImageMeshDrag,
    LatentMeshDrag,
    mesh_drag_warp,
    mesh_drag_warp_image,
)


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


def test_mesh_drag_warp_image_noop_when_disabled():
    image = torch.rand(1, 32, 32, 3)
    out_points_zero = mesh_drag_warp_image(image, points=0, drag_min=0.0, drag_max=32.0, seed=1)
    assert torch.equal(out_points_zero, image)

    out_range_zero = mesh_drag_warp_image(image, points=8, drag_min=0.0, drag_max=0.0, seed=1)
    assert torch.equal(out_range_zero, image)


def test_mesh_drag_warp_image_deterministic_for_seed():
    image = torch.linspace(0.0, 1.0, 32 * 32, dtype=torch.float32).reshape(1, 32, 32, 1).repeat(1, 1, 1, 3)
    out1 = mesh_drag_warp_image(image, points=12, drag_min=4.0, drag_max=32.0, seed=999)
    out2 = mesh_drag_warp_image(image, points=12, drag_min=4.0, drag_max=32.0, seed=999)
    assert torch.allclose(out1, out2)


def test_image_mesh_drag_node_preserves_shape_and_changes_content():
    image = torch.linspace(0.0, 1.0, 64 * 64, dtype=torch.float32).reshape(1, 64, 64, 1).repeat(1, 1, 1, 3)
    node = ImageMeshDrag()
    (out_image,) = node.drag(image, seed=123, points=10, drag_min=8.0, drag_max=48.0)
    assert out_image.shape == image.shape
    assert not torch.allclose(out_image, image)
