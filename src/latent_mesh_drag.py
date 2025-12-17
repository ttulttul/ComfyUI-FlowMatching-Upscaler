from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SEED_STRIDE = 0x9E3779B97F4A7C15  # 64-bit golden ratio constant
_DEFAULT_VERTEX_SPACING = 16  # Latent pixels between control vertices.
_DEFAULT_IMAGE_VERTEX_SPACING = _DEFAULT_VERTEX_SPACING * 8  # Approx SD-style VAE scale factor.
_DISPLACEMENT_INTERPOLATION_MODES = ("bilinear", "bicubic", "bspline", "nearest")
_SAMPLING_INTERPOLATION_MODES = ("bilinear", "bicubic", "nearest")


@dataclass(frozen=True)
class _Flattened2D:
    tensor: torch.Tensor
    restore: Callable[[torch.Tensor], torch.Tensor]
    batch_size: int
    extra_dim: int


def _flatten_to_nchw(tensor: torch.Tensor) -> _Flattened2D:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    original_shape = tuple(tensor.shape)
    squeeze_channel = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
        squeeze_channel = True

    if tensor.ndim < 4:
        raise ValueError(f"Expected tensor with at least 3 dims (B,H,W) or 4 dims (B,C,H,W), got {original_shape}")

    batch_size = int(tensor.shape[0])
    extra_dim = 1
    reshape_higher = tensor.ndim > 4
    if reshape_higher:
        reshape_base = tuple(tensor.shape)
        extra_dim = int(math.prod(reshape_base[2:-2])) if len(reshape_base) > 4 else 1
        tensor = tensor.reshape(reshape_base[0], reshape_base[1], -1, reshape_base[-2], reshape_base[-1])
        tensor = tensor.movedim(2, 1).reshape(-1, reshape_base[1], reshape_base[-2], reshape_base[-1])

        def restore(out: torch.Tensor) -> torch.Tensor:
            restored = out.reshape(
                original_shape[0],
                -1,
                original_shape[1],
                original_shape[-2],
                original_shape[-1],
            )
            restored = restored.movedim(2, 1).reshape(original_shape)
            return restored

        return _Flattened2D(tensor=tensor, restore=restore, batch_size=batch_size, extra_dim=extra_dim)

    def restore(out: torch.Tensor) -> torch.Tensor:
        if squeeze_channel:
            return out.squeeze(1)
        return out

    return _Flattened2D(tensor=tensor, restore=restore, batch_size=batch_size, extra_dim=extra_dim)


def _compute_vertex_grid_size(height: int, width: int, points: int, *, spacing: int = _DEFAULT_VERTEX_SPACING) -> Tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid spatial dimensions {height}x{width}.")
    if spacing <= 0:
        raise ValueError(f"Vertex spacing must be positive, got {spacing}.")

    base_h = max(4, int(math.ceil((height - 1) / spacing)) + 1)
    base_w = max(4, int(math.ceil((width - 1) / spacing)) + 1)

    grid_h = base_h
    grid_w = base_w
    if points <= 0:
        return grid_h, grid_w

    while grid_h * grid_w < points:
        if grid_h <= grid_w:
            grid_h += 1
        else:
            grid_w += 1

    return grid_h, grid_w


def _seed_for_batch(seed: int, batch_index: int) -> int:
    return (int(seed) + int(batch_index) * _SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF


def _make_control_displacement(
    *,
    grid_h: int,
    grid_w: int,
    points: int,
    drag_min: float,
    drag_max: float,
    seed: int,
) -> torch.Tensor:
    disp = torch.zeros((2, grid_h, grid_w), dtype=torch.float32)
    if points <= 0 or drag_max <= 0.0:
        return disp

    total_vertices = grid_h * grid_w
    if points > total_vertices:
        raise ValueError(f"Requested {points} mesh points but only {total_vertices} vertices available.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    chosen = torch.randperm(total_vertices, generator=generator)[:points]
    ys = (chosen // grid_w).to(torch.long)
    xs = (chosen % grid_w).to(torch.long)

    magnitudes = torch.empty((points,), dtype=torch.float32).uniform_(float(drag_min), float(drag_max), generator=generator)
    angles = torch.empty((points,), dtype=torch.float32).uniform_(0.0, float(2.0 * math.pi), generator=generator)
    dx = magnitudes * torch.cos(angles)
    dy = magnitudes * torch.sin(angles)

    disp[0, ys, xs] = dx
    disp[1, ys, xs] = dy
    return disp


def _base_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.unsqueeze(0)

def _apply_bspline_smoothing(displacement: torch.Tensor, *, passes: int = 1) -> torch.Tensor:
    if passes <= 0:
        return displacement
    if displacement.ndim != 4:
        raise ValueError(f"Expected displacement tensor with shape (B,2,H,W), got {tuple(displacement.shape)}")

    channels = int(displacement.shape[1])
    if channels != 2:
        raise ValueError(f"Expected displacement to have 2 channels (dx,dy), got {channels}")

    device = displacement.device
    dtype = displacement.dtype

    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype) / 16.0
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    weight = kernel_2d.expand(channels, 1, 5, 5).contiguous()

    smoothed = displacement
    for _ in range(int(passes)):
        smoothed = F.conv2d(smoothed, weight, padding=2, groups=channels)
    return smoothed


def _upsample_displacement(
    control: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
    spline_passes: int,
) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode not in _DISPLACEMENT_INTERPOLATION_MODES:
        raise ValueError(f"Unsupported displacement_interpolation '{mode}'. Supported: {_DISPLACEMENT_INTERPOLATION_MODES}")

    if mode == "bspline":
        base = F.interpolate(
            control,
            size=(height, width),
            mode="bicubic",
            align_corners=True,
        )
        return _apply_bspline_smoothing(base, passes=int(spline_passes))

    align_corners = True if mode in ("bilinear", "bicubic") else None
    return F.interpolate(
        control,
        size=(height, width),
        mode=mode,
        align_corners=align_corners,
    )


def mesh_drag_warp(
    tensor: torch.Tensor,
    *,
    points: int,
    drag_min: float,
    drag_max: float,
    seed: int,
    padding_mode: str = "border",
    vertex_spacing: int = _DEFAULT_VERTEX_SPACING,
    displacement_interpolation: str = "bicubic",
    spline_passes: int = 2,
    sampling_interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Warp a latent-like tensor by perturbing random vertices on a coarse mesh and interpolating
    the displacement across the image.

    Drag amounts are specified in pixels of the input tensor's spatial dimensions.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    if points < 0:
        raise ValueError(f"points must be >= 0, got {points}")
    if drag_min < 0.0 or drag_max < 0.0:
        raise ValueError("drag_min and drag_max must be >= 0.0")
    if drag_min > drag_max:
        raise ValueError(f"drag_min ({drag_min}) must be <= drag_max ({drag_max})")

    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    if height <= 1 or width <= 1:
        logger.debug("Mesh drag warp skipped for degenerate tensor size %sx%s.", height, width)
        return tensor
    if points == 0 or drag_max <= 0.0:
        return tensor
    sampling_interpolation = str(sampling_interpolation).strip().lower()
    if sampling_interpolation not in _SAMPLING_INTERPOLATION_MODES:
        raise ValueError(f"Unsupported sampling_interpolation '{sampling_interpolation}'. Supported: {_SAMPLING_INTERPOLATION_MODES}")

    flattened = _flatten_to_nchw(tensor)
    nchw = flattened.tensor
    device = nchw.device

    grid_h, grid_w = _compute_vertex_grid_size(height, width, points, spacing=int(vertex_spacing))
    control = torch.stack(
        [
            _make_control_displacement(
                grid_h=grid_h,
                grid_w=grid_w,
                points=points,
                drag_min=drag_min,
                drag_max=drag_max,
                seed=_seed_for_batch(seed, batch_index),
            )
            for batch_index in range(flattened.batch_size)
        ],
        dim=0,
    ).to(device=device)

    with torch.no_grad():
        disp = _upsample_displacement(
            control,
            height=height,
            width=width,
            mode=displacement_interpolation,
            spline_passes=int(spline_passes),
        )
        if drag_max > 0.0:
            disp = disp.clamp(min=-float(drag_max), max=float(drag_max))

        if flattened.extra_dim != 1:
            disp = disp.repeat_interleave(flattened.extra_dim, dim=0)

        if disp.shape[0] != nchw.shape[0]:
            raise RuntimeError(
                f"Internal shape mismatch: got displacement batch {disp.shape[0]} but tensor batch {nchw.shape[0]}"
            )

        disp_norm = disp.permute(0, 2, 3, 1).to(dtype=torch.float32)
        disp_norm[..., 0] *= 2.0 / float(width - 1)
        disp_norm[..., 1] *= 2.0 / float(height - 1)

        grid = _base_grid(height, width, device=device) - disp_norm
        warped = F.grid_sample(
            nchw,
            grid,
            mode=sampling_interpolation,
            padding_mode=padding_mode,
            align_corners=True,
        )
        return flattened.restore(warped)


def mesh_drag_warp_image(
    image: torch.Tensor,
    *,
    points: int,
    drag_min: float,
    drag_max: float,
    seed: int,
    padding_mode: str = "border",
    vertex_spacing: int = _DEFAULT_IMAGE_VERTEX_SPACING,
    displacement_interpolation: str = "bicubic",
    spline_passes: int = 2,
    sampling_interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Warp an image tensor using the same mesh drag logic.

    Accepts ComfyUI's `IMAGE` format (B,H,W,C) as well as NCHW inputs.
    Drag amounts are in image pixels.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        warped = mesh_drag_warp_image(
            image.unsqueeze(0),
            points=points,
            drag_min=drag_min,
            drag_max=drag_max,
            seed=seed,
            padding_mode=padding_mode,
            vertex_spacing=vertex_spacing,
            displacement_interpolation=displacement_interpolation,
            spline_passes=spline_passes,
            sampling_interpolation=sampling_interpolation,
        )
        return warped.squeeze(0)

    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with shape (B,H,W,C) or (B,C,H,W), got {tuple(image.shape)}")

    if image.shape[-1] in (1, 3, 4):
        nchw = image.permute(0, 3, 1, 2)
        warped_nchw = mesh_drag_warp(
            nchw,
            points=points,
            drag_min=drag_min,
            drag_max=drag_max,
            seed=seed,
            padding_mode=padding_mode,
            vertex_spacing=vertex_spacing,
            displacement_interpolation=displacement_interpolation,
            spline_passes=spline_passes,
            sampling_interpolation=sampling_interpolation,
        )
        return warped_nchw.permute(0, 2, 3, 1)

    if image.shape[1] in (1, 3, 4):
        return mesh_drag_warp(
            image,
            points=points,
            drag_min=drag_min,
            drag_max=drag_max,
            seed=seed,
            padding_mode=padding_mode,
            vertex_spacing=vertex_spacing,
            displacement_interpolation=displacement_interpolation,
            spline_passes=spline_passes,
            sampling_interpolation=sampling_interpolation,
        )

    raise ValueError(
        "Unable to infer channel axis for image tensor. Expected channels-last (B,H,W,C) "
        "or channels-first (B,C,H,W) with C in {1,3,4}, got shape "
        f"{tuple(image.shape)}."
    )


class LatentMeshDrag:
    """
    Applies a cloth-like mesh warp to a latent tensor by randomly dragging a subset of control vertices.
    """

    CATEGORY = "latent/perturb"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "drag"

    _DISPLACEMENT_INTERPOLATION_OPTIONS: Tuple[str, ...] = _DISPLACEMENT_INTERPOLATION_MODES
    _SAMPLING_INTERPOLATION_OPTIONS: Tuple[str, ...] = _SAMPLING_INTERPOLATION_MODES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to warp spatially using a random mesh drag."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed controlling which mesh points are dragged and by how much.",
                }),
                "points": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 2048,
                    "tooltip": "Number of mesh vertices to randomly drag.",
                }),
                "drag_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 128.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Minimum drag distance (latent pixels).",
                }),
                "drag_max": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 128.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Maximum drag distance (latent pixels).",
                }),
            },
            "optional": {
                "displacement_interpolation": (cls._DISPLACEMENT_INTERPOLATION_OPTIONS, {
                    "default": "bicubic",
                    "tooltip": "How to interpolate the sparse mesh drags into a full displacement field. "
                               "Use 'bspline' for smoother, more organic warps.",
                }),
                "spline_passes": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Only used when displacement_interpolation = 'bspline'. More passes = smoother warp.",
                }),
                "sampling_interpolation": (cls._SAMPLING_INTERPOLATION_OPTIONS, {
                    "default": "bilinear",
                    "tooltip": "How to sample the source tensor when applying the warp (bilinear/bicubic/nearest).",
                }),
            },
        }

    def drag(
        self,
        latent,
        seed: int,
        points: int,
        drag_min: float,
        drag_max: float,
        displacement_interpolation: str = "bicubic",
        spline_passes: int = 2,
        sampling_interpolation: str = "bilinear",
    ):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("LATENT input must be a dictionary containing a 'samples' tensor.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError(f"LATENT['samples'] must be a torch.Tensor, got {type(samples)}.")

        logger.debug(
            "Applying LatentMeshDrag: samples=%s points=%d drag=[%.3f, %.3f] seed=%d disp=%s spline=%d sample=%s",
            tuple(samples.shape),
            points,
            drag_min,
            drag_max,
            seed,
            displacement_interpolation,
            spline_passes,
            sampling_interpolation,
        )

        warped = mesh_drag_warp(
            samples,
            points=int(points),
            drag_min=float(drag_min),
            drag_max=float(drag_max),
            seed=int(seed),
            displacement_interpolation=str(displacement_interpolation),
            spline_passes=int(spline_passes),
            sampling_interpolation=str(sampling_interpolation),
        )

        output = latent.copy()
        output["samples"] = warped

        noise_mask = output.get("noise_mask")
        if isinstance(noise_mask, torch.Tensor):
            output["noise_mask"] = mesh_drag_warp(
                noise_mask,
                points=int(points),
                drag_min=float(drag_min),
                drag_max=float(drag_max),
                seed=int(seed),
                displacement_interpolation=str(displacement_interpolation),
                spline_passes=int(spline_passes),
                sampling_interpolation=str(sampling_interpolation),
            )
        return (output,)


class ImageMeshDrag:
    """
    Applies a cloth-like mesh warp to an image tensor by randomly dragging a subset of control vertices.

    This node operates directly in image space (pixel units) and accepts ComfyUI's BHWC `IMAGE` tensors.
    """

    CATEGORY = "image/perturb"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "drag"

    _DISPLACEMENT_INTERPOLATION_OPTIONS: Tuple[str, ...] = _DISPLACEMENT_INTERPOLATION_MODES
    _SAMPLING_INTERPOLATION_OPTIONS: Tuple[str, ...] = _SAMPLING_INTERPOLATION_MODES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to warp spatially using a random mesh drag."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed controlling which mesh points are dragged and by how much.",
                }),
                "points": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Number of mesh vertices to randomly drag.",
                }),
                "drag_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 1.0,
                    "round": 0.01,
                    "tooltip": "Minimum drag distance (image pixels).",
                }),
                "drag_max": ("FLOAT", {
                    "default": 32.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 1.0,
                    "round": 0.01,
                    "tooltip": "Maximum drag distance (image pixels).",
                }),
            },
            "optional": {
                "displacement_interpolation": (cls._DISPLACEMENT_INTERPOLATION_OPTIONS, {
                    "default": "bicubic",
                    "tooltip": "How to interpolate the sparse mesh drags into a full displacement field. "
                               "Use 'bspline' for smoother, more organic warps.",
                }),
                "spline_passes": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Only used when displacement_interpolation = 'bspline'. More passes = smoother warp.",
                }),
                "sampling_interpolation": (cls._SAMPLING_INTERPOLATION_OPTIONS, {
                    "default": "bilinear",
                    "tooltip": "How to sample the source image when applying the warp (bilinear/bicubic/nearest).",
                }),
            },
        }

    def drag(
        self,
        image,
        seed: int,
        points: int,
        drag_min: float,
        drag_max: float,
        displacement_interpolation: str = "bicubic",
        spline_passes: int = 2,
        sampling_interpolation: str = "bilinear",
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"IMAGE input must be a torch.Tensor, got {type(image)}.")

        logger.debug(
            "Applying ImageMeshDrag: image=%s points=%d drag=[%.3f, %.3f] seed=%d disp=%s spline=%d sample=%s",
            tuple(image.shape),
            points,
            drag_min,
            drag_max,
            seed,
            displacement_interpolation,
            spline_passes,
            sampling_interpolation,
        )

        warped = mesh_drag_warp_image(
            image,
            points=int(points),
            drag_min=float(drag_min),
            drag_max=float(drag_max),
            seed=int(seed),
            displacement_interpolation=str(displacement_interpolation),
            spline_passes=int(spline_passes),
            sampling_interpolation=str(sampling_interpolation),
        )
        return (warped,)


NODE_CLASS_MAPPINGS = {
    "LatentMeshDrag": LatentMeshDrag,
    "ImageMeshDrag": ImageMeshDrag,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentMeshDrag": "Latent Mesh Drag",
    "ImageMeshDrag": "Image Mesh Drag",
}
