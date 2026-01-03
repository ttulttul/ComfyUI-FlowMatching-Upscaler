import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_upscale_advanced import LatentUpscaleAdvanced  # noqa: E402


def _mean_cov(samples: torch.Tensor, *, epsilon: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    data = samples.detach().float()
    batch, channels = data.shape[0], data.shape[1]
    flat = data.reshape(batch, channels, -1)
    mu = flat.mean(dim=-1)
    centered = flat - mu.unsqueeze(-1)
    denom = max(int(centered.shape[-1]) - 1, 1)
    cov = centered @ centered.transpose(-1, -2)
    cov = cov / float(denom)
    if epsilon > 0.0:
        eye = torch.eye(channels, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + eye * float(epsilon)
    return mu, cov


def test_scale_by_one_returns_input_unchanged():
    node = LatentUpscaleAdvanced()
    latent = {"samples": torch.randn(1, 4, 8, 8)}

    (out,) = node.upscale(
        samples=latent,
        scale_by=1.0,
        upscale_method="bicubic",
        crop="disabled",
        covariance_mode="whiten",
        moment_match=True,
        per_batch_stats=True,
        stats_sample_pixels=0,
        stats_seed=0,
        shrinkage=0.0,
        epsilon=0.0,
        stats_latent=None,
    )

    assert out is latent


def test_covariance_mode_none_resizes_samples_and_noise_mask():
    import comfy.utils  # imported via stubs in conftest

    comfy.utils._common_upscale_calls.clear()

    node = LatentUpscaleAdvanced()
    latent = {
        "samples": torch.ones(1, 4, 4, 4),
        "noise_mask": torch.zeros(1, 1, 4, 4),
    }

    (out,) = node.upscale(
        samples=latent,
        scale_by=2.0,
        upscale_method="nearest-exact",
        crop="disabled",
        covariance_mode="none",
        moment_match=False,
        per_batch_stats=True,
        stats_sample_pixels=0,
        stats_seed=0,
        shrinkage=0.0,
        epsilon=0.0,
        stats_latent=None,
    )

    assert tuple(out["samples"].shape) == (1, 4, 8, 8)
    assert tuple(out["noise_mask"].shape) == (1, 1, 8, 8)
    assert len(comfy.utils._common_upscale_calls) == 2
    assert comfy.utils._common_upscale_calls[0]["method"] == "nearest-exact"


def test_whiten_upscale_runs_without_nan():
    node = LatentUpscaleAdvanced()
    latent = {"samples": torch.randn(1, 4, 8, 8)}

    (out,) = node.upscale(
        samples=latent,
        scale_by=2.0,
        upscale_method="bicubic",
        crop="disabled",
        covariance_mode="whiten",
        moment_match=False,
        per_batch_stats=True,
        stats_sample_pixels=0,
        stats_seed=0,
        shrinkage=0.05,
        epsilon=1e-6,
        stats_latent=None,
    )

    assert tuple(out["samples"].shape) == (1, 4, 16, 16)
    assert torch.isfinite(out["samples"]).all()


def test_moment_match_reduces_covariance_error(monkeypatch):
    import comfy.utils

    def bilinear_upscale(samples: torch.Tensor, width: int, height: int, upscale_method: str, crop: str):
        tensor = samples
        squeeze_channel = False
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
            squeeze_channel = True

        reshape_higher = False
        if tensor.ndim > 4:
            reshape_higher = True
            reshape_base = tensor.shape
            tensor = tensor.reshape(reshape_base[0], reshape_base[1], -1, reshape_base[-2], reshape_base[-1])
            tensor = tensor.movedim(2, 1).reshape(-1, reshape_base[1], reshape_base[-2], reshape_base[-1])

        tensor = torch.nn.functional.interpolate(tensor.float(), size=(height, width), mode="bilinear", align_corners=False)

        if reshape_higher:
            tensor = tensor.reshape(reshape_base[0], -1, reshape_base[1], height, width)
            tensor = tensor.movedim(2, 1).reshape(reshape_base[:-2] + (height, width))
        elif squeeze_channel:
            tensor = tensor.squeeze(1)

        return tensor.to(dtype=samples.dtype)

    monkeypatch.setattr(comfy.utils, "common_upscale", bilinear_upscale)

    torch.manual_seed(0)
    base = torch.randn(1, 1, 8, 8)
    latent_tensor = torch.cat([base, 2.0 * base, -base, base + 0.25 * torch.randn_like(base)], dim=1)
    latent = {"samples": latent_tensor}

    node = LatentUpscaleAdvanced()

    (out_no_match,) = node.upscale(
        samples=latent,
        scale_by=2.0,
        upscale_method="bilinear",
        crop="disabled",
        covariance_mode="whiten",
        moment_match=False,
        per_batch_stats=True,
        stats_sample_pixels=0,
        stats_seed=0,
        shrinkage=0.0,
        epsilon=1e-8,
        stats_latent=None,
    )

    (out_match,) = node.upscale(
        samples=latent,
        scale_by=2.0,
        upscale_method="bilinear",
        crop="disabled",
        covariance_mode="whiten",
        moment_match=True,
        per_batch_stats=True,
        stats_sample_pixels=0,
        stats_seed=0,
        shrinkage=0.0,
        epsilon=1e-8,
        stats_latent=None,
    )

    _, target_cov = _mean_cov(latent_tensor, epsilon=1e-8)
    _, cov_no_match = _mean_cov(out_no_match["samples"], epsilon=1e-8)
    _, cov_match = _mean_cov(out_match["samples"], epsilon=1e-8)

    err_no_match = torch.linalg.norm(cov_no_match - target_cov).item()
    err_match = torch.linalg.norm(cov_match - target_cov).item()

    assert err_match < err_no_match
    assert err_match < 1e-3

