import logging
from typing import Dict, Tuple

try:
    from .qwen_spatial import apply_dype_to_qwen_image  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for direct execution
    from qwen_spatial import apply_dype_to_qwen_image

logger = logging.getLogger(__name__)


class DyPEQwenImage:
    """
    Applies DyPE-style spatial extrapolation to a Qwen Image diffusion model.
    """

    CATEGORY = "model_patches/unet"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    _METHOD_OPTIONS: Tuple[str, ...] = ("yarn", "ntk", "base")
    _EDITING_MODE_OPTIONS: Tuple[str, ...] = ("adaptive", "timestep_aware", "resolution_aware", "minimal", "full")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Qwen Image model to patch with DyPE."}),
                "width": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 16384,
                    "step": 8,
                    "tooltip": "Target output width in pixels.",
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 16384,
                    "step": 8,
                    "tooltip": "Target output height in pixels.",
                }),
                "auto_detect": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically derive patch size and base resolution from the model when possible.",
                }),
                "base_width": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 16384,
                    "step": 8,
                    "tooltip": "Training width used by the base Qwen model when auto detection fails.",
                }),
                "base_height": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 16384,
                    "step": 8,
                    "tooltip": "Training height used by the base Qwen model when auto detection fails.",
                }),
                "method": (cls._METHOD_OPTIONS, {
                    "default": "yarn",
                    "tooltip": "Spatial RoPE extrapolation strategy.",
                }),
                "enable_dype": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Dynamic Position Extrapolation scaling.",
                }),
                "dype_exponent": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Controls how strongly DyPE ramps across sampling timesteps.",
                }),
                "base_shift": ("FLOAT", {
                    "default": 1.15,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Baseline shift applied to the flow-matching noise schedule.",
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.35,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Maximum shift applied when operating at the target resolution.",
                }),
                "editing_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Scale DyPE while editing images (1.0 = full strength, 0.0 = disable DyPE scaling in edits).",
                }),
                "editing_mode": (cls._EDITING_MODE_OPTIONS, {
                    "default": "adaptive",
                    "tooltip": "Strategy for tapering DyPE during edits.",
                }),
            },
        }

    def apply(
        self,
        model,
        width: int,
        height: int,
        auto_detect: bool,
        base_width: int,
        base_height: int,
        method: str,
        enable_dype: bool,
        dype_exponent: float,
        base_shift: float,
        max_shift: float,
        editing_strength: float,
        editing_mode: str,
    ):
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            raise ValueError("DyPE for Qwen Image expects a diffusion model input.")

        logger.info(
            "DyPE_QwenImage: requested patch (width=%d, height=%d, method=%s, "
            "enable_dype=%s, dype_exponent=%s, base_shift=%s, max_shift=%s, "
            "auto_detect=%s, editing_mode=%s, editing_strength=%.3f).",
            width,
            height,
            method,
            enable_dype,
            dype_exponent,
            base_shift,
            max_shift,
            auto_detect,
            editing_mode,
            editing_strength,
        )

        patched_model = apply_dype_to_qwen_image(
            model=model,
            width=int(width),
            height=int(height),
            method=str(method),
            enable_dype=bool(enable_dype),
            dype_exponent=float(dype_exponent),
            base_width=int(base_width),
            base_height=int(base_height),
            base_shift=float(base_shift),
            max_shift=float(max_shift),
            auto_detect=bool(auto_detect),
            editing_strength=float(editing_strength),
            editing_mode=str(editing_mode),
        )
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "DyPEQwenImage": DyPEQwenImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DyPEQwenImage": "DyPE for Qwen Image",
}
