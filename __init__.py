from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Any, Dict


def _load_module_from_path(module_name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load module '{module_name}' from '{path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __package__:
    from .src.dype_qwen_image import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as DYPE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as DYPE_DISPLAY_NAME_MAPPINGS,
    )
    from .src.flow_matching_upscaler import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as FM_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as FM_DISPLAY_NAME_MAPPINGS,
    )
    from .src.latent_upscale_advanced import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as LUA_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as LUA_DISPLAY_NAME_MAPPINGS,
    )
else:  # pragma: no cover - direct execution fallback
    _ROOT_DIR = pathlib.Path(__file__).resolve().parent
    _load_module_from_path("rope", _ROOT_DIR / "src" / "rope.py")
    _load_module_from_path("qwen_spatial", _ROOT_DIR / "src" / "qwen_spatial.py")
    flow_module = _load_module_from_path("flow_matching_upscaler", _ROOT_DIR / "src" / "flow_matching_upscaler.py")
    dype_module = _load_module_from_path("dype_qwen_image", _ROOT_DIR / "src" / "dype_qwen_image.py")
    latent_module = _load_module_from_path("latent_upscale_advanced", _ROOT_DIR / "src" / "latent_upscale_advanced.py")

    FM_NODE_CLASS_MAPPINGS = getattr(flow_module, "NODE_CLASS_MAPPINGS")
    FM_DISPLAY_NAME_MAPPINGS = getattr(flow_module, "NODE_DISPLAY_NAME_MAPPINGS")
    DYPE_NODE_CLASS_MAPPINGS = getattr(dype_module, "NODE_CLASS_MAPPINGS")
    DYPE_DISPLAY_NAME_MAPPINGS = getattr(dype_module, "NODE_DISPLAY_NAME_MAPPINGS")
    LUA_NODE_CLASS_MAPPINGS = getattr(latent_module, "NODE_CLASS_MAPPINGS")
    LUA_DISPLAY_NAME_MAPPINGS = getattr(latent_module, "NODE_DISPLAY_NAME_MAPPINGS")

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    **FM_NODE_CLASS_MAPPINGS,
    **DYPE_NODE_CLASS_MAPPINGS,
    **LUA_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    **FM_DISPLAY_NAME_MAPPINGS,
    **DYPE_DISPLAY_NAME_MAPPINGS,
    **LUA_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
