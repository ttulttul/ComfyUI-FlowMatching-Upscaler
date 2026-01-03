import logging
import math
from typing import Dict, Optional, Tuple

import torch

import comfy.utils

logger = logging.getLogger(__name__)


def _ensure_int(value: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value)))


def _align(value: int, multiple: int = 2) -> int:
    return max(multiple, int(math.ceil(value / multiple) * multiple))


def _log_channel_stats(label: str, tensor: torch.Tensor, *, limit: int = 16) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[1] <= 0:
        logger.debug("%s channel stats unavailable.", label)
        return

    with torch.no_grad():
        reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != 1)
        means = tensor.detach().float().mean(dim=reduce_dims).cpu().tolist()
        stds = tensor.detach().float().std(dim=reduce_dims, unbiased=False).cpu().tolist()

    entries = []
    for idx, (mean_val, std_val) in enumerate(zip(means, stds)):
        if idx >= limit:
            break
        entries.append(f"c{idx}: {mean_val:.4f}±{std_val:.4f}")
    if len(means) > limit:
        entries.append("...")

    logger.debug("%s channel stats (%d ch): %s", label, len(means), "; ".join(entries))


def _maybe_fallback_lanczos(method: str, channels: int) -> str:
    if method != "lanczos":
        return method
    if channels in (1, 3, 4):
        return method
    logger.warning(
        "Lanczos upscaling is unsupported for %d-channel latents. Falling back to bicubic.",
        channels,
    )
    return "bicubic"


def _common_upscale(samples: torch.Tensor, *, width: int, height: int, method: str, crop: str) -> torch.Tensor:
    if width == samples.shape[-1] and height == samples.shape[-2]:
        return samples

    upscale_method = _maybe_fallback_lanczos(method, channels=int(samples.shape[1]) if samples.ndim >= 2 else 0)
    with torch.no_grad():
        return comfy.utils.common_upscale(samples, width, height, upscale_method, crop)


def _resize_noise_mask(
    latent_dict: dict,
    *,
    width: int,
    height: int,
    method: str,
    crop: str,
) -> None:
    if "noise_mask" not in latent_dict:
        return
    mask = latent_dict.get("noise_mask")
    if not isinstance(mask, torch.Tensor):
        return
    latent_dict["noise_mask"] = _common_upscale(mask, width=width, height=height, method=method, crop=crop)


def _compute_mean_cov(
    samples: torch.Tensor,
    *,
    per_batch: bool,
    shrinkage: float,
    epsilon: float,
    sample_pixels: int,
    sample_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim < 4:
        raise ValueError("Expected latent samples with at least 4 dimensions (B,C,H,W).")
    if samples.shape[1] <= 0:
        raise ValueError("Latent tensor has zero channels.")

    shrinkage = float(max(0.0, min(1.0, shrinkage)))
    epsilon = float(max(0.0, epsilon))

    with torch.no_grad():
        data = samples.detach().float()
        batch = int(data.shape[0])
        channels = int(data.shape[1])
        flat = data.reshape(batch, channels, -1)

        pixels = int(flat.shape[-1])
        if sample_pixels > 0 and sample_pixels < pixels:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(sample_seed) & 0xFFFFFFFFFFFFFFFF)
            idx = torch.randint(0, pixels, (int(sample_pixels),), generator=generator, device="cpu")
            idx = idx.to(flat.device)
            flat = flat.index_select(-1, idx)
            pixels = int(flat.shape[-1])

        if per_batch:
            mu = flat.mean(dim=-1)
            centered = flat - mu.unsqueeze(-1)
            denom = max(pixels - 1, 1)
            cov = centered @ centered.transpose(-1, -2)
            cov = cov / float(denom)
        else:
            merged = flat.permute(1, 0, 2).reshape(channels, -1)
            mu = merged.mean(dim=-1, keepdim=True).transpose(0, 1)
            centered = merged - mu.transpose(0, 1)
            denom = max(centered.shape[-1] - 1, 1)
            cov = centered @ centered.transpose(-1, -2)
            cov = (cov / float(denom)).unsqueeze(0)
            mu = mu

        if shrinkage > 0.0:
            diag = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
            cov = cov * (1.0 - shrinkage) + diag * shrinkage

        if epsilon > 0.0:
            eye = torch.eye(channels, device=cov.device, dtype=cov.dtype)
            cov = cov + eye.unsqueeze(0) * epsilon

        return mu, cov


def _eigh_whiten_params(
    cov: torch.Tensor,
    *,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (eigvecs, sqrt_eigs, inv_sqrt_eigs) for symmetric covariance matrices.
    """
    if cov.ndim != 3 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError("Expected covariance with shape (B,C,C).")

    with torch.no_grad():
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(float(max(epsilon, 1e-12)))
        sqrt_eigs = torch.sqrt(eigvals)
        inv_sqrt_eigs = 1.0 / sqrt_eigs
        return eigvecs, sqrt_eigs, inv_sqrt_eigs


def _apply_whiten(
    samples: torch.Tensor,
    *,
    mean: torch.Tensor,
    eigvecs: torch.Tensor,
    inv_sqrt_eigs: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        batch = int(samples.shape[0])
        channels = int(samples.shape[1])
        flat = samples.detach().float().reshape(batch, channels, -1).transpose(1, 2)  # (B,N,C)
        centered = flat - mean[:, None, :]
        coeffs = centered @ eigvecs
        whitened = coeffs * inv_sqrt_eigs[:, None, :]
        out = whitened.transpose(1, 2).reshape(samples.shape)
        return out.to(dtype=samples.dtype)


def _apply_recolor(
    whitened: torch.Tensor,
    *,
    mean: torch.Tensor,
    eigvecs: torch.Tensor,
    sqrt_eigs: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        batch = int(whitened.shape[0])
        channels = int(whitened.shape[1])
        flat = whitened.detach().float().reshape(batch, channels, -1).transpose(1, 2)  # (B,N,C)
        colored = (flat * sqrt_eigs[:, None, :]) @ eigvecs.transpose(-1, -2)
        out = colored + mean[:, None, :]
        out = out.transpose(1, 2).reshape(whitened.shape)
        return out.to(dtype=whitened.dtype)


def _moment_match(
    samples: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    target_eigvecs: torch.Tensor,
    target_sqrt_eigs: torch.Tensor,
    shrinkage: float,
    epsilon: float,
) -> torch.Tensor:
    with torch.no_grad():
        current_mean, current_cov = _compute_mean_cov(
            samples,
            per_batch=True,
            shrinkage=shrinkage,
            epsilon=epsilon,
            sample_pixels=0,
            sample_seed=0,
        )
        current_eigvecs, current_sqrt, current_inv_sqrt = _eigh_whiten_params(current_cov, epsilon=epsilon)

        batch = int(samples.shape[0])
        channels = int(samples.shape[1])
        flat = samples.detach().float().reshape(batch, channels, -1).transpose(1, 2)  # (B,N,C)
        centered = flat - current_mean[:, None, :]
        coords = centered @ current_eigvecs
        whitened = coords * current_inv_sqrt[:, None, :]
        recolored = (whitened * target_sqrt_eigs[:, None, :]) @ target_eigvecs.transpose(-1, -2)
        matched = recolored + target_mean[:, None, :]
        out = matched.transpose(1, 2).reshape(samples.shape)
        return out.to(dtype=samples.dtype)


class LatentUpscaleAdvanced:
    """
    Covariance-aware latent upscaler inspired by `upscaling.md`.

    Implements:
      - Whiten → upscale → recolor (global PCA/eigen whitening).
      - Optional moment-preserving correction to re-match mean/covariance after scaling.
    """

    CATEGORY = "latent/upscaling"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    _UPSCALE_METHODS: Tuple[str, ...] = (
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
        "bislerp",
    )
    _COVARIANCE_MODES: Tuple[str, ...] = ("none", "whiten")
    _CROP_METHODS: Tuple[str, ...] = ("disabled", "center")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Input latent to upscale."}),
                "scale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.01,
                    "max": 8.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Spatial scale factor applied to the latent grid (e.g., 2.0 = 2× latent resolution).",
                }),
                "upscale_method": (cls._UPSCALE_METHODS, {
                    "default": "bicubic",
                    "tooltip": "Resampling kernel for spatial upscaling (applied to whitened latents when enabled).",
                }),
                "crop": (cls._CROP_METHODS, {
                    "default": "disabled",
                    "tooltip": "Cropping behavior when target aspect ratio differs (rare when using scale_by).",
                }),
                "covariance_mode": (cls._COVARIANCE_MODES, {
                    "default": "whiten",
                    "tooltip": "Enable covariance-aware whitening (whiten→upscale→recolor).",
                }),
                "moment_match": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After upscaling, apply an affine correction so the upscaled latent matches the source mean/covariance.",
                }),
                "per_batch_stats": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Compute stats per batch element (recommended). Disable to share one covariance estimate across the batch.",
                }),
                "stats_sample_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16_777_216,
                    "step": 1024,
                    "tooltip": "0 = use all pixels for covariance estimation; otherwise randomly sample this many pixel vectors.",
                }),
                "stats_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed used when stats_sample_pixels > 0.",
                }),
                "shrinkage": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Shrink covariance toward its diagonal (stabilizes whitening when covariance is noisy).",
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-5,
                    "min": 0.0,
                    "max": 1e-1,
                    "step": 1e-5,
                    "round": 1e-8,
                    "tooltip": "Diagonal jitter added to covariance (prevents non-positive-definite matrices).",
                }),
            },
            "optional": {
                "stats_latent": ("LATENT", {"tooltip": "Optional latent used to estimate mean/covariance (defaults to input)."}),
            },
        }

    def upscale(
        self,
        samples,
        scale_by: float,
        upscale_method: str,
        crop: str,
        covariance_mode: str,
        moment_match: bool,
        per_batch_stats: bool,
        stats_sample_pixels: int,
        stats_seed: int,
        shrinkage: float,
        epsilon: float,
        stats_latent: Optional[dict] = None,
    ):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("LatentUpscaleAdvanced expects a LATENT dictionary input.")

        source_tensor = samples["samples"]
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError("LATENT['samples'] must be a torch.Tensor.")
        if source_tensor.ndim < 4:
            raise ValueError("LATENT['samples'] must have shape (B,C,H,W) (or higher with extra dims).")

        if scale_by <= 0:
            raise ValueError("scale_by must be > 0.")

        height = _ensure_int(source_tensor.shape[-2] * float(scale_by))
        width = _ensure_int(source_tensor.shape[-1] * float(scale_by))
        height = _align(height, 2)
        width = _align(width, 2)

        if height == source_tensor.shape[-2] and width == source_tensor.shape[-1]:
            logger.debug("LatentUpscaleAdvanced: scale_by produced identical resolution; returning input unchanged.")
            return (samples,)

        logger.info(
            "LatentUpscaleAdvanced: upscale %s -> (%d,%d) using %s (covariance_mode=%s, moment_match=%s).",
            tuple(source_tensor.shape),
            height,
            width,
            upscale_method,
            covariance_mode,
            moment_match,
        )

        out = samples.copy()

        if covariance_mode == "none":
            out["samples"] = _common_upscale(source_tensor, width=width, height=height, method=upscale_method, crop=crop)
            _resize_noise_mask(out, width=width, height=height, method=upscale_method, crop=crop)
            return (out,)

        stats_payload = stats_latent if stats_latent is not None else samples
        if not isinstance(stats_payload, dict) or "samples" not in stats_payload:
            raise ValueError("stats_latent must be a LATENT dictionary when provided.")

        stats_tensor = stats_payload["samples"]
        if not isinstance(stats_tensor, torch.Tensor):
            raise ValueError("stats_latent['samples'] must be a torch.Tensor.")
        if stats_tensor.ndim < 4:
            raise ValueError("stats_latent['samples'] must have shape (B,C,H,W) (or higher with extra dims).")
        if int(stats_tensor.shape[1]) != int(source_tensor.shape[1]):
            raise ValueError("stats_latent must have the same channel count as the input latent.")
        if per_batch_stats and int(stats_tensor.shape[0]) != int(source_tensor.shape[0]):
            raise ValueError(
                "stats_latent batch size must match the input batch when per_batch_stats is enabled."
            )

        _log_channel_stats("latent.stats_input", stats_tensor)
        mu, cov = _compute_mean_cov(
            stats_tensor,
            per_batch=bool(per_batch_stats),
            shrinkage=float(shrinkage),
            epsilon=float(epsilon),
            sample_pixels=int(stats_sample_pixels),
            sample_seed=int(stats_seed),
        )

        eigvecs, sqrt_eigs, inv_sqrt_eigs = _eigh_whiten_params(cov, epsilon=float(epsilon))

        if not per_batch_stats:
            mu = mu.expand(source_tensor.shape[0], -1)
            eigvecs = eigvecs.expand(source_tensor.shape[0], -1, -1)
            sqrt_eigs = sqrt_eigs.expand(source_tensor.shape[0], -1)
            inv_sqrt_eigs = inv_sqrt_eigs.expand(source_tensor.shape[0], -1)

        whitened = _apply_whiten(source_tensor, mean=mu, eigvecs=eigvecs, inv_sqrt_eigs=inv_sqrt_eigs)
        _log_channel_stats("latent.whitened", whitened)

        whitened_up = _common_upscale(whitened, width=width, height=height, method=upscale_method, crop=crop)
        recolored = _apply_recolor(whitened_up, mean=mu, eigvecs=eigvecs, sqrt_eigs=sqrt_eigs)

        if moment_match:
            recolored = _moment_match(
                recolored,
                target_mean=mu,
                target_eigvecs=eigvecs,
                target_sqrt_eigs=sqrt_eigs,
                shrinkage=float(shrinkage),
                epsilon=float(epsilon),
            )

        _log_channel_stats("latent.upscaled", recolored)

        out["samples"] = recolored
        _resize_noise_mask(out, width=width, height=height, method=upscale_method, crop=crop)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "LatentUpscaleAdvanced": LatentUpscaleAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentUpscaleAdvanced": "Latent Upscale Advanced",
}
