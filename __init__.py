try:
    from .src.flow_matching_upscaler import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:  # pragma: no cover - fallback for non-package execution
    from src.flow_matching_upscaler import (  # noqa: F401
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
