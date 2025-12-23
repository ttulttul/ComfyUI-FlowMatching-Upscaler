# Changelog

## 2.0.0 - 2025-12-23
- BREAKING: Moved mesh-drag and latent debug nodes into the standalone `Skoogeer-Noise` pack.

## 1.0.7 - 2025-12-17
- Added selectable mesh-drag interpolation options, including a B-spline-smoothed displacement mode for more organic warps.

## 1.0.6 - 2025-12-17
- Added Image Mesh Drag node to apply the same cloth-like warp directly to IMAGE tensors.

## 1.0.5 - 2025-12-17
- Added Latent Mesh Drag node for cloth-like latent-space spatial warps.
- Added deterministic mesh-warp tests (including noise-mask lockstep coverage).

## 1.0.2 - 2025-11-14
- Added DyPE for Qwen Image node with dynamic import fallbacks for standalone usage.
- Ported DyPE rotary embedding utilities and spatial patcher to this repository.
- Expanded test suite with DyPE coverage and COMFY stubs; refreshed README and learnings.

## 1.0.1 - 2025-11-13
- Introduced automatic low-VRAM streaming fallback and standalone streaming stage node.
- Hardened progressive upscaler with seed perturbation, tiling safeguards, and retry logic.
- Added pytest helper scripts, comfy API stubs, and documentation on memory handling.

## 1.0.0 - 2025-11-12
- Initial release of progressive flow-matching upscaler and modular stage nodes.
- Added cleanup stage controls, exponential skip scheduling, and comprehensive README guidance.
- Published repository metadata for Comfy Registry distribution.

## Initial import
- Initial import.
