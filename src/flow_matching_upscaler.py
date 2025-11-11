import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

import comfy.samplers
import comfy.utils

from nodes import common_ksampler

logger = logging.getLogger(__name__)

_SEED_STRIDE = 0x9E3779B97F4A7C15  # 64-bit golden ratio constant


@dataclass(frozen=True)
class StageConfig:
    """Resolved configuration for a single progressive upscale stage."""

    scale_factor: float
    noise_level: float
    skip_blend: float
    steps: int
    seed: int
    denoise: Optional[float] = None
    is_cleanup: bool = False


def _interp_schedule(
    start: float,
    end: float,
    count: int,
) -> List[float]:
    """Create an interpolated schedule between start and end inclusive."""
    if count <= 0:
        return []
    if count == 1:
        return [start]
    delta = (end - start) / (count - 1)
    return [start + i * delta for i in range(count)]


def _exp_schedule(
    start: float,
    end: float,
    count: int,
) -> List[float]:
    """Create an exponentially decaying schedule between start and end inclusive."""
    if count <= 0:
        return []
    if count == 1:
        return [start]

    eps = 1e-6
    start_eff = start if start > 0 else eps
    end_eff = end if end > 0 else eps
    ratio = math.pow(end_eff / start_eff, 1.0 / (count - 1))

    values = [start_eff * (ratio ** i) for i in range(count)]
    values[0] = start
    values[-1] = end
    return values


def _parse_schedule(
    override: str,
    count: int,
    start: float,
    end: float,
    *,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    curve: str = "linear",
) -> List[float]:
    """
    Parse a comma-separated override schedule. When empty, interpolate between start and end.
    """
    if override.strip():
        values = [float(part.strip()) for part in override.split(",") if part.strip()]
        if len(values) == 1:
            values = values * count
        if len(values) != count:
            raise ValueError(
                f"Expected {count} schedule values but received {len(values)} ({values})."
            )
    else:
        if curve == "exponential":
            values = _exp_schedule(start, end, count)
        else:
            values = _interp_schedule(start, end, count)

    clamped = [max(clamp_min, min(clamp_max, value)) for value in values]
    return clamped


def _resolve_stage_configs(
    *,
    total_scale: float,
    stages: int,
    steps_per_stage: int,
    seed: int,
    noise_schedule: Sequence[float],
    skip_schedule: Sequence[float],
    include_cleanup: bool,
    cleanup_noise: float,
    cleanup_denoise: float,
) -> List[StageConfig]:
    """Create the per-stage configuration, including seeds and scale factors."""
    if stages <= 0:
        raise ValueError("stages must be a positive integer.")

    scale_per_stage = math.pow(total_scale, 1.0 / stages)
    configs: List[StageConfig] = []
    next_seed = seed & 0xFFFFFFFFFFFFFFFF

    for idx in range(stages):
        configs.append(
            StageConfig(
                scale_factor=scale_per_stage,
                noise_level=noise_schedule[idx],
                skip_blend=skip_schedule[idx],
                steps=steps_per_stage,
                seed=next_seed,
            )
        )
        next_seed = (next_seed + _SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF

    if include_cleanup:
        configs.append(
            StageConfig(
                scale_factor=1.0,
                noise_level=cleanup_noise,
                skip_blend=0.0,
                steps=steps_per_stage,
                seed=next_seed,
                denoise=cleanup_denoise,
                is_cleanup=True,
            )
        )

    return configs


def _ensure_int(value: float, minimum: int = 1) -> int:
    """Safe conversion from float to int with lower bound."""
    return max(minimum, int(round(value)))


class FlowMatchingProgressiveUpscaler:
    """
    Progressive upscaler tailored for flow-matching models.

    Each stage performs:
      1. Latent upscaling.
      2. Flow-style re-noising (linear mix with fresh noise).
      3. Denoising via the configured sampler.
      4. Skip residual blending to preserve coarse composition.
      5. Optional dilated sampling refinement.
    """

    CATEGORY = "latent/upscaling"
    FUNCTION = "progressive_upscale"
    RETURN_TYPES = ("LATENT",)

    _UPSCALE_METHODS: Tuple[str, ...] = (
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
        "bislerp",
    )

    _SAMPLERS = tuple(comfy.samplers.KSampler.SAMPLERS)
    _SCHEDULERS = tuple(comfy.samplers.KSampler.SCHEDULERS)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Flow-matching diffusion model to drive refinement."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for CFG."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for CFG."}),
                "latent": ("LATENT", {"tooltip": "Low resolution latent to progressively upscale."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Base seed controlling the re-noising at each stage.",
                }),
                "steps_per_stage": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "tooltip": "Denoising steps executed at every progressive stage.",
                }),
                "cfg": ("FLOAT", {
                    "default": 4.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Classifier Free Guidance strength per stage.",
                }),
                "sampler_name": (cls._SAMPLERS, {"tooltip": "Sampler backend leveraged during refinement."}),
                "scheduler": (cls._SCHEDULERS, {"tooltip": "Noise schedule applied during denoising."}),
                "total_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 16.0,
                    "step": 0.25,
                    "tooltip": "Overall scale factor from the input latent to the final output.",
                }),
                "stages": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "tooltip": "Number of progressive stages to reach the total scale.",
                }),
                "renoise_start": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Noise ratio applied at the first stage.",
                }),
                "renoise_end": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Noise ratio applied at the last stage (interpolated in-between).",
                }),
                "skip_blend_start": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend weight for the upsampled latent at the first stage.",
                }),
                "skip_blend_end": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend weight for the upsampled latent at the last stage.",
                }),
                "upscale_method": (cls._UPSCALE_METHODS, {
                    "default": "bicubic",
                    "tooltip": "Algorithm used when resizing the latents between stages.",
                }),
            },
            "optional": {
                "noise_schedule_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated override for per-stage noise ratios. Empty uses interpolated start/end.",
                }),
                "skip_schedule_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated override for per-stage skip blend weights.",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Denoising strength supplied to the sampler per stage.",
                }),
                "enable_dilated_sampling": (["disable", "enable"], {
                    "default": "enable",
                    "tooltip": "Optionally run a dilated refinement pass for global coherence.",
                }),
                "dilated_downscale": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.25,
                    "tooltip": "Factor used when downscaling for the dilated pass (>=1.0).",
                }),
                "dilated_blend": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend weight of the dilated refinement result.",
                }),
                "cleanup_stage": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Run an extra non-scaling clean-up denoise pass at the end.",
                }),
                "cleanup_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Noise ratio for the optional clean-up stage (set to 0 to disable re-noising).",
                }),
                "cleanup_denoise": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Denoising strength used during the clean-up stage.",
                }),
            },
        }

    @staticmethod
    def _progressive_upscale_latent(
        latent: torch.Tensor,
        scale_factor: float,
        method: str,
    ) -> torch.Tensor:
        """Resize latent tensor using ComfyUI's shared helpers."""
        height = _ensure_int(latent.shape[-2] * scale_factor)
        width = _ensure_int(latent.shape[-1] * scale_factor)
        if height == latent.shape[-2] and width == latent.shape[-1]:
            return latent

        upscale_method = method

        # Lanczos uses PIL internally and only supports 1/3/4 channel tensors.
        if method == "lanczos" and latent.ndim >= 4:
            channels = latent.shape[1]
            if channels not in (1, 3, 4):
                logger.warning(
                    "Lanczos upscaling is unsupported for %d-channel latents. Falling back to bicubic.",
                    channels,
                )
                upscale_method = "bicubic"

        upscaled = comfy.utils.common_upscale(
            latent,
            width,
            height,
            upscale_method,
            crop="disabled",
        )
        return upscaled

    @staticmethod
    def _apply_flow_renoise(
        latent: torch.Tensor,
        noise_level: float,
        seed: int,
    ) -> torch.Tensor:
        """Re-noise latent following flow matching's linear interpolation."""
        if noise_level <= 0.0:
            return latent

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device="cpu", dtype=latent.dtype)

        if latent.device != noise.device:
            noise = noise.to(latent.device)

        noise_level = max(0.0, min(1.0, float(noise_level)))
        blended = latent * (1.0 - noise_level) + noise * noise_level
        return blended

    @staticmethod
    def _run_sampler(
        model,
        positive,
        negative,
        latent_template: dict,
        sampler_name: str,
        scheduler: str,
        cfg: float,
        steps: int,
        seed: int,
        denoise: float,
    ) -> dict:
        """Execute a sampling pass using ComfyUI's built-in sampler bridge."""
        refined, = common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_template,
            denoise=denoise,
        )
        return refined

    @staticmethod
    def _dilated_refinement(
        *,
        model,
        positive,
        negative,
        sampler_name: str,
        scheduler: str,
        cfg: float,
        steps: int,
        seed: int,
        denoise: float,
        base_latent: dict,
        downscale_factor: float,
        blend: float,
    ) -> torch.Tensor:
        """
        Run a coarse-grained refinement by downscaling, sampling, then upscaling back.
        """
        blend = max(0.0, min(1.0, blend))
        if blend == 0.0:
            return base_latent["samples"]

        samples = base_latent["samples"]
        if downscale_factor <= 1.0:
            return samples

        down_height = _ensure_int(samples.shape[-2] / downscale_factor)
        down_width = _ensure_int(samples.shape[-1] / downscale_factor)

        if down_height < 4 or down_width < 4:
            logger.debug("Dilated refinement skipped due to excessive downscale (target too small).")
            return samples

        downsampled = comfy.utils.common_upscale(
            samples,
            down_width,
            down_height,
            "area",
            crop="disabled",
        )

        latent_copy = base_latent.copy()
        latent_copy["samples"] = downsampled

        logger.debug(
            "Running dilated refinement at %sx%s (factor %.2f).",
            down_width,
            down_height,
            downscale_factor,
        )

        refined_down = FlowMatchingProgressiveUpscaler._run_sampler(
            model=model,
            positive=positive,
            negative=negative,
            latent_template=latent_copy,
            sampler_name=sampler_name,
            scheduler=scheduler,
            cfg=cfg,
            steps=max(1, steps // 2),
            seed=seed + 10_000,
            denoise=denoise,
        )

        upsampled = comfy.utils.common_upscale(
            refined_down["samples"],
            samples.shape[-1],
            samples.shape[-2],
            "bilinear",
            crop="disabled",
        )

        merged = samples * (1.0 - blend) + upsampled * blend
        return merged

    def progressive_upscale(
        self,
        model,
        positive,
        negative,
        latent,
        seed,
        steps_per_stage,
        cfg,
        sampler_name,
        scheduler,
        total_scale,
        stages,
        renoise_start,
        renoise_end,
        skip_blend_start,
        skip_blend_end,
        upscale_method,
        noise_schedule_override="",
        skip_schedule_override="",
        denoise=1.0,
        enable_dilated_sampling="enable",
        dilated_downscale=2.0,
        dilated_blend=0.25,
        cleanup_stage="disable",
        cleanup_noise=0.0,
        cleanup_denoise=0.4,
    ):
        if total_scale <= 0:
            raise ValueError("total_scale must be greater than zero.")
        if stages <= 0:
            raise ValueError("stages must be at least 1.")

        logger.info(
            "Starting progressive upscale: total_scale=%.2f, stages=%d, steps_per_stage=%d",
            total_scale,
            stages,
            steps_per_stage,
        )

        noise_schedule = _parse_schedule(
            noise_schedule_override,
            stages,
            renoise_start,
            renoise_end,
            clamp_min=0.0,
            clamp_max=1.0,
        )
        skip_schedule = _parse_schedule(
            skip_schedule_override,
            stages,
            skip_blend_start,
            skip_blend_end,
            clamp_min=0.0,
            clamp_max=1.0,
            curve="exponential",
        )

        stage_configs = _resolve_stage_configs(
            total_scale=total_scale,
            stages=stages,
            steps_per_stage=steps_per_stage,
            seed=seed,
            noise_schedule=noise_schedule,
            skip_schedule=skip_schedule,
            include_cleanup=cleanup_stage == "enable",
            cleanup_noise=max(0.0, min(1.0, cleanup_noise)),
            cleanup_denoise=max(0.0, min(1.0, cleanup_denoise)),
        )

        current_latent_dict = latent.copy()
        current_latent = current_latent_dict["samples"]

        main_stage_count = stages
        total_stage_count = len(stage_configs)

        for stage_index, stage in enumerate(stage_configs):
            if stage.is_cleanup:
                logger.info(
                    "Cleanup stage -> noise=%.3f denoise=%.3f steps=%d",
                    stage.noise_level,
                    stage.denoise if stage.denoise is not None else denoise,
                    stage.steps,
                )
            else:
                logger.info(
                    "Stage %d/%d (total %d) -> scale=%.3f noise=%.3f skip=%.3f steps=%d",
                    stage_index + 1,
                    main_stage_count,
                    total_stage_count,
                    stage.scale_factor,
                    stage.noise_level,
                    stage.skip_blend,
                    stage.steps,
                )

            upscaled = self._progressive_upscale_latent(
                current_latent,
                stage.scale_factor,
                method=upscale_method,
            )

            skip_reference = upscaled.clone()
            re_noised = self._apply_flow_renoise(
                upscaled,
                stage.noise_level,
                stage.seed,
            )

            latent_payload = current_latent_dict.copy()
            latent_payload["samples"] = re_noised

            refined_dict = self._run_sampler(
                model=model,
                positive=positive,
                negative=negative,
                latent_template=latent_payload,
                sampler_name=sampler_name,
                scheduler=scheduler,
                cfg=cfg,
                steps=stage.steps,
                seed=stage.seed,
                denoise=stage.denoise if stage.denoise is not None else denoise,
            )

            refined_samples = refined_dict["samples"]
            if refined_samples.shape != skip_reference.shape:
                refined_samples = comfy.utils.common_upscale(
                    refined_samples,
                    skip_reference.shape[-1],
                    skip_reference.shape[-2],
                    "bilinear",
                    crop="disabled",
                )

            blended = skip_reference * stage.skip_blend + refined_samples * (1.0 - stage.skip_blend)

            if enable_dilated_sampling == "enable" and not stage.is_cleanup:
                blended_dict = refined_dict.copy()
                blended_dict["samples"] = blended
                dilated = self._dilated_refinement(
                    model=model,
                    positive=positive,
                    negative=negative,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    cfg=cfg,
                    steps=stage.steps,
                    seed=stage.seed,
                    denoise=denoise,
                    base_latent=blended_dict,
                    downscale_factor=max(1.0, float(dilated_downscale)),
                    blend=dilated_blend,
                )
                blended = blended * (1.0 - dilated_blend) + dilated * dilated_blend

            current_latent = blended
            current_latent_dict["samples"] = current_latent

        logger.info("Progressive upscale complete. Final latent shape: %s", tuple(current_latent.shape))
        return (current_latent_dict,)


NODE_CLASS_MAPPINGS = {
    "FlowMatchingProgressiveUpscaler": FlowMatchingProgressiveUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowMatchingProgressiveUpscaler": "Flow Matching Progressive Upscaler",
}
