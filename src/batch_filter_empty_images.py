import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class BatchFilterEmptyImages:
    """Filter an IMAGE batch, removing entries that are entirely zero (within epsilon)."""

    CATEGORY = "utils/image"
    FUNCTION = "filter"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image batch (BHWC)."}),
                "epsilon": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.0001,
                    "round": 0.000001,
                    "tooltip": "Treat pixels with |value| <= epsilon as zero.",
                }),
            },
        }

    def filter(self, images, epsilon: float):
        if not isinstance(images, torch.Tensor):
            raise ValueError("BatchFilterEmptyImages expected images to be a torch.Tensor (ComfyUI IMAGE).")

        if images.ndim == 3:
            images = images.unsqueeze(0)

        if images.ndim != 4:
            raise ValueError(
                "BatchFilterEmptyImages expects images with shape (B,H,W,C); "
                f"got ndim={images.ndim}, shape={tuple(images.shape)}."
            )

        epsilon = float(epsilon)
        if epsilon < 0.0:
            raise ValueError(f"BatchFilterEmptyImages epsilon must be >= 0; got {epsilon}.")

        batch = int(images.shape[0])
        if batch == 0:
            return (images,)

        with torch.no_grad():
            max_abs = images.detach().abs().amax(dim=(1, 2, 3))
            keep_mask = max_abs > epsilon
            kept = int(keep_mask.sum().item())
            removed = batch - kept
            out = images[keep_mask]

        logger.debug(
            "BatchFilterEmptyImages: kept %d/%d images (removed=%d, epsilon=%.6g).",
            kept,
            batch,
            removed,
            epsilon,
        )
        return (out,)


NODE_CLASS_MAPPINGS = {
    "BatchFilterEmptyImages": BatchFilterEmptyImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchFilterEmptyImages": "Batch Filter Empty Images",
}
