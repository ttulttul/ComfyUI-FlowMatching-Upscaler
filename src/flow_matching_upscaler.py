import gc
import logging
import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Callable, Any

import torch
import numpy as np

import comfy.samplers
import comfy.utils
import comfy.model_management as model_management

from nodes import common_ksampler

logger = logging.getLogger(__name__)

_SEED_STRIDE = 0x9E3779B97F4A7C15  # 64-bit golden ratio constant
_STREAMING_ATTENTION_BUDGET_MB = 256.0

_BACKGROUND_COLOR = np.array([0.08, 0.08, 0.08], dtype=np.float32)
_GRIDLINE_COLOR = np.array([0.25, 0.25, 0.25], dtype=np.float32)
_MEAN_BAR_COLOR = np.array([0.2, 0.65, 0.95], dtype=np.float32)
_STD_BAR_COLOR = np.array([0.95, 0.6, 0.2], dtype=np.float32)
_TEXT_COLOR = np.array([0.85, 0.85, 0.85], dtype=np.float32)

_FONT_BITMAPS = {
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    "3": [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "5": [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
    "6": [
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    "7": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    "9": [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100",
    ],
    "-": [
        "00000",
        "00000",
        "00100",
        "00100",
        "00100",
        "00000",
        "00000",
    ],
    ".": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00100",
        "00100",
    ],
    " ": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10101",
        "10001",
        "10001",
        "10001",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "S": [
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "D": [
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ],
}
_FONT_HEIGHT = len(next(iter(_FONT_BITMAPS.values())))
_FONT_WIDTH = len(next(iter(_FONT_BITMAPS.values()))[0])


def _measure_text(text: str) -> int:
    """Compute pixel width of text using the fixed bitmap font."""
    if not text:
        return 0
    width = 0
    for char in text:
        glyph = _FONT_BITMAPS.get(char.upper(), _FONT_BITMAPS[" "])
        width += len(glyph[0]) + 1
    return max(0, width - 1)


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int, color: np.ndarray) -> None:
    """Draw text onto the canvas in-place using the bitmap font."""
    cursor = x
    for char in text:
        glyph = _FONT_BITMAPS.get(char.upper(), _FONT_BITMAPS[" "])
        glyph_height = len(glyph)
        glyph_width = len(glyph[0])
        for gy in range(glyph_height):
            cy = y + gy
            if cy < 0 or cy >= canvas.shape[0]:
                continue
            row = glyph[gy]
            for gx in range(glyph_width):
                if row[gx] != "1":
                    continue
                cx = cursor + gx
                if cx < 0 or cx >= canvas.shape[1]:
                    continue
                canvas[cy, cx, :] = color
        cursor += glyph_width + 1


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


def progressive_upscale_latent(
    latent: torch.Tensor,
    scale_factor: float,
    method: str,
) -> torch.Tensor:
    """Resize latent tensor using ComfyUI's shared helpers."""

    # If no scaling is specified, just return the latent unmodified.
    if scale_factor == 1.0:
        return latent

    height = _ensure_int(latent.shape[-2] * scale_factor)
    width = _ensure_int(latent.shape[-1] * scale_factor)
    def _align(value: int, multiple: int = 2) -> int:
        return max(multiple, int(math.ceil(value / multiple) * multiple))

    width = _align(width, 2)
    height = _align(height, 2)

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


def _resize_noise_mask(
    latent_dict: dict,
    *,
    scale_factor: float,
    method: str,
    context: str,
) -> Optional[torch.Tensor]:
    """
    Ensure the latent dictionary's noise mask matches the scaled latent resolution.
    """
    if "noise_mask" not in latent_dict:
        return None

    mask = latent_dict["noise_mask"]
    if not isinstance(mask, torch.Tensor):
        logger.debug("%s: noise mask present but not a tensor; skipping resize.", context)
        return None

    original_shape = tuple(mask.shape)
    upscaled_mask = progressive_upscale_latent(mask, scale_factor, method=method)
    latent_dict["noise_mask"] = upscaled_mask

    resized_shape = tuple(upscaled_mask.shape)
    if resized_shape != original_shape:
        logger.debug(
            "%s: resized noise mask from %s to %s using %s scaling.",
            context,
            original_shape,
            resized_shape,
            method,
        )
    else:
        logger.debug(
            "%s: noise mask already at target resolution %s (scale %.3f).",
            context,
            resized_shape,
            scale_factor,
        )
    return upscaled_mask


def apply_flow_renoise(
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


def _prepare_latent_for_sampler(latent_dict: dict) -> Tuple[dict, callable]:
    samples = latent_dict["samples"]
    if samples.ndim == 5:
        b, c, t, h, w = samples.shape
        payload = latent_dict.copy()
        payload["samples"] = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        if "noise_mask" in payload:
            mask = payload["noise_mask"]
            if isinstance(mask, torch.Tensor) and mask.ndim == 5:
                mb, mc, mt, mh, mw = mask.shape
                payload["noise_mask"] = mask.permute(0, 2, 1, 3, 4).reshape(mb * mt, mc, mh, mw)

        def restore(tensor: torch.Tensor) -> torch.Tensor:
            restored = tensor.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            return restored

        return payload, restore

    return latent_dict, lambda tensor: tensor


def _reduce_channel_statistics(tensor: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute per-channel mean and standard deviation across all non-channel dimensions.
    Returns None when statistics cannot be computed (e.g., invalid tensor shape).
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.ndim < 2:
        return None

    with torch.no_grad():
        data = tensor.detach()
        if data.shape[1] == 0:
            return None

        reduce_dims = tuple(dim for dim in range(data.ndim) if dim != 1)
        channel_means = data.float().mean(dim=reduce_dims)
        channel_stds = data.float().std(dim=reduce_dims, unbiased=False)
    return channel_means, channel_stds


def _log_channel_stats(label: str, tensor: torch.Tensor, *, limit: int = 16) -> None:
    """
    Emit per-channel mean and standard deviation diagnostics for the provided tensor.
    Limited to the first `limit` channels to keep log volume manageable.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    stats = _reduce_channel_statistics(tensor)
    if stats is None:
        logger.debug("%s channel stats unavailable (unsupported tensor).", label)
        return
    channel_means, channel_stds = stats

    mean_vals = channel_means.detach().cpu().tolist()
    std_vals = channel_stds.detach().cpu().tolist()

    entries = []
    for idx, (mean_val, std_val) in enumerate(zip(mean_vals, std_vals)):
        if idx >= limit:
            break
        entries.append(f"c{idx}: {mean_val:.4f}±{std_val:.4f}")
    if len(mean_vals) > limit:
        entries.append("...")

    logger.debug(
        "%s channel stats (%d ch): %s",
        label,
        len(mean_vals),
        "; ".join(entries),
    )


def _render_channel_stats_image(
    channel_means: torch.Tensor,
    channel_stds: torch.Tensor,
    *,
    limit: int = 16,
    height: int = 256,
) -> Tuple[torch.Tensor, dict]:
    """
    Render a simple bar chart visualizing per-channel mean (top) and std (bottom).

    Returns:
        image (torch.Tensor): Float tensor with shape (1, H, W, 3) in [0, 1].
        layout (dict): Metadata describing bar placement and section bounds.
    """
    if channel_means.numel() == 0 or channel_stds.numel() == 0:
        raise ValueError("Channel statistics must be non-empty to render.")

    display_channels = max(1, min(int(limit), channel_means.numel()))
    means_np = channel_means.detach().cpu().float().numpy()
    stds_np = channel_stds.detach().cpu().float().numpy()

    display_means = means_np[:display_channels]
    display_stds = np.clip(stds_np[:display_channels], a_min=0.0, a_max=None)

    margin_left = 52
    margin_right = 18
    margin_y = 16
    label_margin_bottom = _FONT_HEIGHT + 10
    bar_spacing = 6
    bar_width = 16
    section_count = 2
    min_section_height = 28

    if height < 72:
        height = 72

    available = height - label_margin_bottom - margin_y * (section_count + 1)
    if available < section_count * min_section_height:
        section_height = min_section_height
    else:
        section_height = max(min_section_height, available // section_count)

    actual_height = margin_y * (section_count + 1) + section_height * section_count + label_margin_bottom

    width = margin_left + margin_right + display_channels * bar_width + max(0, (display_channels - 1) * bar_spacing)
    canvas = np.broadcast_to(_BACKGROUND_COLOR, (actual_height, width, 3)).copy()

    # draw outer border
    canvas[0, :, :] = _GRIDLINE_COLOR
    canvas[-1, :, :] = _GRIDLINE_COLOR
    canvas[:, 0, :] = _GRIDLINE_COLOR
    canvas[:, -1, :] = _GRIDLINE_COLOR

    top_start = margin_y
    bottom_start = margin_y * 2 + section_height
    x_start = margin_left
    x_end = width - margin_right
    channel_label_y = bottom_start + section_height + 4

    # Baselines for sections
    canvas[top_start - 1, x_start:x_end, :] = _GRIDLINE_COLOR
    canvas[top_start + section_height - 1, x_start:x_end, :] = _GRIDLINE_COLOR
    canvas[bottom_start - 1, x_start:x_end, :] = _GRIDLINE_COLOR
    canvas[bottom_start + section_height - 1, x_start:x_end, :] = _GRIDLINE_COLOR

    mean_min = float(np.min(display_means))
    mean_max = float(np.max(display_means))
    if math.isclose(mean_min, mean_max):
        mean_min -= 0.5
        mean_max += 0.5
    mean_range = max(mean_max - mean_min, 1e-6)

    def _clamp_unit(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _section_value_to_y(norm_value: float, section_start: int) -> int:
        clamped = _clamp_unit(norm_value)
        return section_start + section_height - 1 - int(round(clamped * (section_height - 1)))

    def _draw_y_tick(y_pos: int):
        tick_x0 = max(1, margin_left - 8)
        tick_x1 = max(tick_x0 + 1, margin_left - 2)
        if 0 <= y_pos < canvas.shape[0]:
            canvas[y_pos, tick_x0:tick_x1, :] = _GRIDLINE_COLOR

    def _clamp_text_y(y_pos: int) -> int:
        max_y = canvas.shape[0] - _FONT_HEIGHT - 1
        return max(1, min(max_y, y_pos))

    mean_max_y = _section_value_to_y(1.0, top_start)
    mean_min_y = _section_value_to_y(0.0, top_start)

    if mean_min <= 0.0 <= mean_max:
        zero_norm = (0.0 - mean_min) / mean_range
        zero_y = _section_value_to_y(zero_norm, top_start)
        canvas[zero_y, x_start:x_end, :] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    else:
        zero_y = None

    std_max = float(np.max(display_stds))
    if math.isclose(std_max, 0.0):
        std_max = 1.0

    bar_layout = []
    for idx in range(display_channels):
        x0 = x_start + idx * (bar_width + bar_spacing)
        x1 = x0 + bar_width
        x1 = min(x1, x_end)

        # mean bar (top section)
        mean_val = float(display_means[idx])
        mean_norm = np.clip((mean_val - mean_min) / mean_range, 0.0, 1.0)
        mean_pixels = max(1, int(round(mean_norm * (section_height - 2))))
        mean_bottom = top_start + section_height - 2
        mean_top = mean_bottom - mean_pixels
        mean_top = max(top_start, mean_top)
        canvas[mean_top:mean_bottom, x0:x1, :] = _MEAN_BAR_COLOR

        # std bar (bottom section)
        std_val = float(display_stds[idx])
        std_norm = 0.0 if std_max <= 0.0 else np.clip(std_val / std_max, 0.0, 1.0)
        std_pixels = 0 if std_max <= 0.0 else max(1, int(round(std_norm * (section_height - 2))))
        std_bottom = bottom_start + section_height - 2
        std_top = std_bottom - std_pixels
        std_top = max(bottom_start, std_top)
        if std_pixels > 0:
            canvas[std_top:std_bottom, x0:x1, :] = _STD_BAR_COLOR

        bar_layout.append(
            {
                "channel": idx,
                "x_range": (x0, x1),
            }
        )

        x_center = (x0 + x1 - 1) // 2

        tick_y_start = bottom_start + section_height - 1
        tick_y_end = min(canvas.shape[0], tick_y_start + 6)
        xa = max(0, x_center - 1)
        xb = min(canvas.shape[1], x_center + 2)
        canvas[tick_y_start:tick_y_end, xa:xb, :] = _GRIDLINE_COLOR

        channel_text = str(idx)
        text_width = _measure_text(channel_text)
        text_x = x_center - text_width // 2
        text_y = channel_label_y
        _draw_text(canvas, channel_text, text_x, text_y, _TEXT_COLOR)

    if channel_means.numel() > display_channels:
        overflow_indicator_width = min(bar_width, x_end - (x_end - bar_width))
        canvas[top_start - 1:top_start + 1, x_end - overflow_indicator_width:x_end, :] = np.array(
            [0.8, 0.2, 0.2], dtype=np.float32
        )
        canvas[
            bottom_start + section_height - 1:bottom_start + section_height + 1,
            x_end - overflow_indicator_width:x_end,
            :
        ] = np.array([0.8, 0.2, 0.2], dtype=np.float32)

    _draw_y_tick(mean_max_y)
    _draw_y_tick(mean_min_y)
    mean_max_label = f"{mean_max:.2f}"
    mean_min_label = f"{mean_min:.2f}"
    _draw_text(canvas, mean_max_label, 4, _clamp_text_y(mean_max_y - _FONT_HEIGHT // 2), _TEXT_COLOR)
    _draw_text(canvas, mean_min_label, 4, _clamp_text_y(mean_min_y - _FONT_HEIGHT // 2), _TEXT_COLOR)

    std_max_y = _section_value_to_y(1.0, bottom_start)
    std_min_y = _section_value_to_y(0.0, bottom_start)
    _draw_y_tick(std_max_y)
    _draw_y_tick(std_min_y)
    std_max_label = f"{std_max:.2f}"
    std_min_label = f"{0.0:.2f}"
    _draw_text(canvas, std_max_label, 4, _clamp_text_y(std_max_y - _FONT_HEIGHT // 2), _TEXT_COLOR)
    _draw_text(canvas, std_min_label, 4, _clamp_text_y(std_min_y - _FONT_HEIGHT // 2), _TEXT_COLOR)

    _draw_text(
        canvas,
        "MEAN",
        4,
        _clamp_text_y(top_start + section_height // 2 - _FONT_HEIGHT // 2),
        _MEAN_BAR_COLOR,
    )
    _draw_text(
        canvas,
        "STD",
        4,
        _clamp_text_y(bottom_start + section_height // 2 - _FONT_HEIGHT // 2),
        _STD_BAR_COLOR,
    )

    image = torch.from_numpy(canvas.astype(np.float32)).unsqueeze(0)
    layout = {
        "bar_ranges": bar_layout,
        "mean_section": (top_start, section_height),
        "std_section": (bottom_start, section_height),
        "width": width,
        "height": actual_height,
        "channel_label_y": channel_label_y,
    }
    return image, layout


@contextmanager
def _temporary_low_vram_mode(enabled: bool):
    """
    Temporarily force ComfyUI into LOW_VRAM mode so layers stream between CPU/GPU.
    """
    if not enabled:
        yield
        return

    original_state = model_management.vram_state
    try:
        model_management.vram_state = model_management.VRAMState.LOW_VRAM
        yield
    finally:
        model_management.vram_state = original_state


@contextmanager
def _throttled_attention_budget(max_free_mb: float):
    """
    Cap the reported available VRAM so attention kernels fall back to smaller chunks.
    """
    budget_bytes = max(0, int(max_free_mb * 1024 * 1024))
    if budget_bytes <= 0:
        yield
        return

    original_get_free_memory = model_management.get_free_memory

    def _capped_get_free_memory(dev=None, torch_free_too: bool = False):
        result = original_get_free_memory(dev, torch_free_too)
        if torch_free_too:
            total_free, torch_free = result
            capped_total = min(total_free, budget_bytes)
            capped_torch = min(torch_free, capped_total)
            return capped_total, capped_torch
        return min(result, budget_bytes)

    model_management.get_free_memory = _capped_get_free_memory
    try:
        yield
    finally:
        model_management.get_free_memory = original_get_free_memory


def _execute_with_memory_controls(
    operation: Callable[[], Any],
    *,
    attention_budget_mb: float,
    enable_low_vram: bool,
) -> Any:
    with ExitStack() as stack:
        stack.enter_context(_temporary_low_vram_mode(enable_low_vram))
        if attention_budget_mb > 0.0:
            stack.enter_context(_throttled_attention_budget(attention_budget_mb))
        return operation()


def _is_oom_exception(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, model_management.OOM_EXCEPTION) or (
        isinstance(exc, RuntimeError) and "out of memory" in message
    )


def run_sampler(
    *,
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


def _run_with_oom_retry(operation: Callable[[], Any], *, description: str):
    """
    Execute `operation` and retry once after clearing caches if an OOM occurs.
    """
    attempts = 0
    while True:
        try:
            return operation()
        except (model_management.OOM_EXCEPTION, RuntimeError) as exc:
            message = str(exc).lower()
            is_runtime_oom = isinstance(exc, RuntimeError) and "out of memory" in message
            if not isinstance(exc, model_management.OOM_EXCEPTION) and not is_runtime_oom:
                raise

            attempts += 1
            if attempts > 1:
                logger.error("Out of memory during %s (no recovery possible).", description, exc_info=True)
                raise

            logger.warning("Out of memory during %s; clearing caches and retrying once.", description, exc_info=True)
            gc.collect()
            model_management.soft_empty_cache(True)


def dilated_refinement(
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
    _log_channel_stats("dilated_refinement/input", samples)
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
    _log_channel_stats("dilated_refinement/downsampled", downsampled)

    sampler_payload, restore_fn = _prepare_latent_for_sampler(latent_copy)

    refined_down = run_sampler(
        model=model,
        positive=positive,
        negative=negative,
        latent_template=sampler_payload,
        sampler_name=sampler_name,
        scheduler=scheduler,
        cfg=cfg,
        steps=max(1, steps // 2),
        seed=seed + 10_000,
        denoise=denoise,
    )

    restored_samples = restore_fn(refined_down["samples"])
    if restored_samples is not refined_down["samples"]:
        refined_down = refined_down.copy()
        refined_down["samples"] = restored_samples

    _log_channel_stats("dilated_refinement/refined", refined_down["samples"])

    upsampled = comfy.utils.common_upscale(
        refined_down["samples"],
        samples.shape[-1],
        samples.shape[-2],
        "bilinear",
        crop="disabled",
    )

    _log_channel_stats("dilated_refinement/upsampled", upsampled)

    merged = samples * (1.0 - blend) + upsampled * blend
    _log_channel_stats("dilated_refinement/merged", merged)
    return merged


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
    RETURN_TYPES = ("LATENT", "INT", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "next_seed", "model", "positive", "negative")

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
            stage_label = (
                "cleanup stage"
                if stage.is_cleanup
                else f"progressive stage {min(stage_index + 1, main_stage_count)}/{main_stage_count}"
            )
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

            _log_channel_stats(f"{stage_label} input", current_latent)

            upscaled = progressive_upscale_latent(
                current_latent,
                stage.scale_factor,
                method=upscale_method,
            )

            _resize_noise_mask(
                current_latent_dict,
                scale_factor=stage.scale_factor,
                method=upscale_method,
                context=f"{stage_label} mask",
            )

            skip_reference = upscaled
            _log_channel_stats(f"{stage_label} skip_reference", skip_reference)
            re_noised = apply_flow_renoise(
                upscaled,
                stage.noise_level,
                stage.seed,
            )
            _log_channel_stats(f"{stage_label} renoised", re_noised)

            # We don't need the upscaled tensor anymore; free up the VRAM
            del upscaled

            latent_payload = current_latent_dict.copy()
            latent_payload["samples"] = re_noised

            sampler_payload, restore_fn = _prepare_latent_for_sampler(latent_payload)

            refined_dict = _run_with_oom_retry(
                lambda: run_sampler(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent_template=sampler_payload,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    cfg=cfg,
                    steps=stage.steps,
                    seed=stage.seed,
                    denoise=stage.denoise if stage.denoise is not None else denoise,
                ),
                description=f"{stage_label} sampling",
            )

            restored_samples = restore_fn(refined_dict["samples"])
            if restored_samples is not refined_dict["samples"]:
                refined_dict = refined_dict.copy()
                refined_dict["samples"] = restored_samples
            refined_samples = refined_dict["samples"]
            if refined_samples.shape != skip_reference.shape:
                refined_samples = comfy.utils.common_upscale(
                    refined_samples,
                    skip_reference.shape[-1],
                    skip_reference.shape[-2],
                    "bilinear",
                    crop="disabled",
                )

            _log_channel_stats(f"{stage_label} refined", refined_samples)
            blended = skip_reference * stage.skip_blend + refined_samples * (1.0 - stage.skip_blend)
            _log_channel_stats(f"{stage_label} blended", blended)

            if enable_dilated_sampling == "enable" and not stage.is_cleanup:
                blended_dict = refined_dict.copy()
                blended_dict["samples"] = blended
                _log_channel_stats(f"{stage_label} dilated/base", blended)
                dilated = _run_with_oom_retry(
                    lambda: dilated_refinement(
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
                    ),
                    description=f"{stage_label} dilated refinement",
                )
                _log_channel_stats(f"{stage_label} dilated/result", dilated)
                blended = blended * (1.0 - dilated_blend) + dilated * dilated_blend
                _log_channel_stats(f"{stage_label} dilated/merged", blended)

            current_latent = blended
            current_latent_dict["samples"] = current_latent

        logger.info("Progressive upscale complete. Final latent shape: %s", tuple(current_latent.shape))
        final_seed = stage_configs[-1].seed if stage_configs else seed
        next_seed = (final_seed + _SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF
        return (current_latent_dict, next_seed, model, positive, negative)


class FlowMatchingStage:
    CATEGORY = "latent/upscaling"
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT", "LATENT", "INT", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "presampler_latent", "next_seed", "model", "positive", "negative")

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
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "steps": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                }),
                "cfg": ("FLOAT", {
                    "default": 4.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "round": 0.01,
                }),
                "sampler_name": (cls._SAMPLERS, {}),
                "scheduler": (cls._SCHEDULERS, {}),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.05,
                }),
                "noise_ratio": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "skip_blend": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                }),
                "upscale_method": (cls._UPSCALE_METHODS, {
                    "default": "bicubic",
                }),
            },
            "optional": {
                "enable_dilated_sampling": (["disable", "enable"], {
                    "default": "disable",
                }),
                "reduce_memory_use": (["disable", "enable"], {
                    "default": "enable",
                }),
                "dilated_downscale": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.25,
                }),
                "dilated_blend": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
        }

    def execute(
        self,
        model,
        positive,
        negative,
        latent,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        scale_factor,
        noise_ratio,
        skip_blend,
        denoise,
        upscale_method,
        enable_dilated_sampling="disable",
        reduce_memory_use="enable",
        dilated_downscale=2.0,
        dilated_blend=0.25,
    ):
        skip_blend = max(0.0, min(1.0, skip_blend))
        reduce_memory_flag = reduce_memory_use == "enable"
        streaming_enabled = False

        while True:
            try:
                return self._execute_single_pass(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent=latent,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    scale_factor=scale_factor,
                    noise_ratio=noise_ratio,
                    skip_blend=skip_blend,
                    denoise=denoise,
                    upscale_method=upscale_method,
                    enable_dilated_sampling=enable_dilated_sampling,
                    reduce_memory_flag=reduce_memory_flag,
                    dilated_downscale=dilated_downscale,
                    dilated_blend=dilated_blend,
                    streaming_enabled=streaming_enabled,
                )
            except BaseException as exc:
                if not _is_oom_exception(exc) or streaming_enabled:
                    raise
                logger.warning(
                    "Out of memory during stage sampling; enabling low-VRAM streaming fallback.",
                    exc_info=True,
                )
                streaming_enabled = True
                reduce_memory_flag = True

    def _execute_single_pass(
        self,
        *,
        model,
        positive,
        negative,
        latent,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        scale_factor,
        noise_ratio,
        skip_blend,
        denoise,
        upscale_method,
        enable_dilated_sampling,
        reduce_memory_flag: bool,
        dilated_downscale,
        dilated_blend,
        streaming_enabled: bool,
    ):
        current_latent_dict = latent.copy()
        current_latent = current_latent_dict["samples"]

        upscaled = progressive_upscale_latent(
            current_latent,
            scale_factor,
            method=upscale_method,
        )

        _resize_noise_mask(
            current_latent_dict,
            scale_factor=scale_factor,
            method=upscale_method,
            context="FlowMatchingStage/mask",
        )

        if reduce_memory_flag:
            skip_reference = upscaled
        else:
            skip_reference = upscaled.clone()

        re_noised = apply_flow_renoise(
            upscaled,
            noise_ratio,
            seed,
        )

        # We don't need this tensor anymore; free the VRAM before we sample
        del upscaled

        latent_payload = current_latent_dict.copy()
        latent_payload["samples"] = re_noised

        sampler_payload, restore_fn = _prepare_latent_for_sampler(latent_payload)

        attention_budget = _STREAMING_ATTENTION_BUDGET_MB if streaming_enabled else 0.0
        low_vram = streaming_enabled
        description = "stage sampling (streaming fallback)" if streaming_enabled else "stage sampling"

        if streaming_enabled:
            logger.info(
                "Stage fallback enabled (attention budget %.1f MB, low_vram=%s).",
                attention_budget,
                low_vram,
            )

        refined_dict = _run_with_oom_retry(
            lambda: _execute_with_memory_controls(
                lambda: run_sampler(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent_template=sampler_payload,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    cfg=cfg,
                    steps=steps,
                    seed=seed,
                    denoise=denoise,
                ),
                attention_budget_mb=attention_budget,
                enable_low_vram=low_vram,
            ),
            description=description,
        )

        restored_samples = restore_fn(refined_dict["samples"])
        if restored_samples is not refined_dict["samples"]:
            refined_dict = refined_dict.copy()
            refined_dict["samples"] = restored_samples
        refined_samples = refined_dict["samples"]
        if refined_samples.shape != skip_reference.shape:
            refined_samples = comfy.utils.common_upscale(
                refined_samples,
                skip_reference.shape[-1],
                skip_reference.shape[-2],
                "bilinear",
                crop="disabled",
            )

        blended = skip_reference * skip_blend + refined_samples * (1.0 - skip_blend)

        if enable_dilated_sampling == "enable":
            blended_dict = refined_dict.copy()
            blended_dict["samples"] = blended
            dilated = _run_with_oom_retry(
                lambda: _execute_with_memory_controls(
                    lambda: dilated_refinement(
                        model=model,
                        positive=positive,
                        negative=negative,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        cfg=cfg,
                        steps=steps,
                        seed=seed,
                        denoise=denoise,
                        base_latent=blended_dict,
                        downscale_factor=max(1.0, float(dilated_downscale)),
                        blend=dilated_blend,
                    ),
                    attention_budget_mb=attention_budget,
                    enable_low_vram=low_vram,
                ),
                description="stage dilated refinement" + (" (streaming fallback)" if streaming_enabled else ""),
            )
            blended = blended * (1.0 - dilated_blend) + dilated * dilated_blend

            # Delete the sampler output dict if no longer needed before the next
            # assignment to help garbage collection.
            del refined_dict

        out = current_latent_dict.copy()
        out["samples"] = blended
        next_seed = (seed + _SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF
        presampler_latent = latent_payload.copy()
        presampler_latent["samples"] = re_noised
        return (out, presampler_latent, next_seed, model, positive, negative)


class LatentChannelStatsPreview:
    CATEGORY = "latent/debug"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "channel_limit": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                }),
                "height": ("INT", {
                    "default": 256,
                    "min": 72,
                    "max": 1024,
                    "step": 4,
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Mark as dynamic so previews refresh reliably.
        return float("nan")

    def render(self, latent, channel_limit=16, height=256):
        stats = _reduce_channel_statistics(latent["samples"])
        if stats is None:
            raise ValueError("Latent tensor must expose a channel dimension to compute statistics.")

        means, stds = stats
        image, _ = _render_channel_stats_image(
            means,
            stds,
            limit=max(1, int(channel_limit)),
            height=max(72, int(height)),
        )
        return (image,)


NODE_CLASS_MAPPINGS = {
    "FlowMatchingProgressiveUpscaler": FlowMatchingProgressiveUpscaler,
    "FlowMatchingStage": FlowMatchingStage,
    "LatentChannelStatsPreview": LatentChannelStatsPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowMatchingProgressiveUpscaler": "Flow Matching Progressive Upscaler",
    "FlowMatchingStage": "Flow Matching Stage",
    "LatentChannelStatsPreview": "Latent Channel Stats Preview",
}
