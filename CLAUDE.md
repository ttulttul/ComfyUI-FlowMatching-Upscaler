# CLAUDE.md - AI Assistant Guide

## Project Overview

ComfyUI-FlowMatching-Upscaler is a ComfyUI custom node package providing:
1. **Flow Matching Progressive Upscaler** - Progressive latent upscaling for flow-matching models (like Qwen Image)
2. **DyPE for Qwen Image** - Dynamic Position Extrapolation to enable high-resolution generation beyond training resolution

The nodes are registered under **Flow Matching** and **DyPE** categories in ComfyUI.

## Repository Structure

```
ComfyUI-FlowMatching-Upscaler/
├── __init__.py              # ComfyUI node registration, exports NODE_CLASS_MAPPINGS
├── src/
│   ├── __init__.py          # Package namespace
│   ├── flow_matching_upscaler.py  # Main upscaler nodes (FlowMatchingProgressiveUpscaler, FlowMatchingStage, LatentChannelStatsPreview)
│   ├── dype_qwen_image.py   # DyPE node wrapper (DyPEQwenImage)
│   ├── qwen_spatial.py      # DyPE spatial embedding implementation (QwenSpatialPosEmbed)
│   └── rope.py              # Rotary Position Embedding utilities
├── tests/
│   ├── conftest.py          # Pytest fixtures and ComfyUI stub setup
│   ├── test_flow_matching_upscaler.py  # Upscaler unit tests
│   └── test_dype_qwen_image.py         # DyPE unit tests
├── web/
│   └── preview.js           # Frontend extension for live preview support
├── docs/
│   └── LEARNINGS.md         # Development insights and decisions
├── examples/
│   └── Method-Comparison.json  # Example ComfyUI workflow
├── pyproject.toml           # Package metadata (version, dependencies)
├── requirements.txt         # Python dependencies
├── test.sh                  # Test runner script
└── .github/workflows/publish.yml  # CI for Comfy Registry publishing
```

## Key Architecture Patterns

### Node Registration
- Nodes export `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` dictionaries
- Root `__init__.py` aggregates mappings from `src/flow_matching_upscaler.py` and `src/dype_qwen_image.py`
- Supports both package import (via `__package__`) and direct execution fallback

### ComfyUI Integration
- Uses `nodes.common_ksampler` for sampling operations
- Uses `comfy.utils.common_upscale` for latent resizing
- Uses `comfy.model_management` for VRAM management
- Uses `comfy.model_patcher.ModelPatcher` for model patching

### Memory Management
- Automatic LOW_VRAM fallback on OOM errors
- Attention budget throttling via `_STREAMING_ATTENTION_BUDGET_MB`
- Manual tensor cleanup with `del` statements after use
- OOM retry logic with cache clearing (`gc.collect()`, `soft_empty_cache`)

## Development Workflow

### Running Tests

```bash
# Using the test script (sets up venv and ComfyUI stubs)
./test.sh

# Direct pytest (requires comfy stubs in path)
pytest

# Run specific test file
pytest tests/test_flow_matching_upscaler.py -v
```

### Test Architecture
- Tests use **stub modules** in `conftest.py` to mock ComfyUI dependencies
- Stubs provide: `comfy.samplers`, `comfy.utils`, `comfy.model_management`, `comfy.model_patcher`, `comfy.model_sampling`, `nodes`
- Tests mock `common_ksampler` to capture sampler calls without real model execution

### Code Style
- Type hints used throughout (Python 3.9+ style)
- Logging via `logging.getLogger(__name__)`
- DEBUG level logging for per-channel statistics diagnostics
- Dataclasses for configuration (`StageConfig`, `_GridConfig`, `_ModelGeometry`)

## Key Implementation Details

### Flow Matching Progressive Upscaler (`flow_matching_upscaler.py`)

**Core Functions:**
- `progressive_upscale_latent()` - Resize latent using ComfyUI's upscaler
- `apply_flow_renoise()` - Flow-style linear interpolation between latent and noise
- `run_sampler()` - Bridge to ComfyUI's `common_ksampler`
- `dilated_refinement()` - Low-pass refinement via downscale-sample-upscale

**Key Classes:**
- `FlowMatchingProgressiveUpscaler` - Main multi-stage upscaler node
- `FlowMatchingStage` - Single-stage node for caching benefits
- `LatentChannelStatsPreview` - Debug visualization node

**Important Constants:**
- `_SEED_STRIDE = 0x9E3779B97F4A7C15` - Golden ratio for seed perturbation
- `_STREAMING_ATTENTION_BUDGET_MB = 256.0` - Memory budget for fallback mode
- `_DILATED_BLEND_METHODS` - Tuple of available blending methods

**Dilated Blending Methods:**

The `dilated_blend_method` parameter controls how the dilated (low-pass) result is blended with the original latent:

- `frequency` (default, recommended) - FFT-based blending that cleanly separates frequency bands. Takes low frequencies from the dilated result and high frequencies from the original. Best for avoiding grid artifacts.
- `laplacian` - Multi-scale Laplacian pyramid blending. Coarse levels get more dilated influence, fine levels get less. Good for seamless compositing.
- `gaussian` - Gaussian-smoothed difference blending. Applies blur to the difference before adding, smoothing out grid artifacts.
- `linear` - Simple `torch.lerp` interpolation. Original behavior, may cause grid artifacts at high blend ratios.

**Key Blending Functions:**
- `_frequency_blend()` - FFT-based frequency domain blending
- `_laplacian_pyramid_blend()` - Multi-scale pyramid blending
- `_gaussian_weighted_blend()` - Gaussian-smoothed blending
- `_apply_dilated_blend()` - Dispatcher for all blend methods

### DyPE Implementation (`qwen_spatial.py`, `dype_qwen_image.py`)

**Core Components:**
- `QwenSpatialPosEmbed` - Custom positional embedder with YaRN/NTK extrapolation
- `apply_dype_to_qwen_image()` - Patches model with DyPE embedder
- `_DyPEModelSampling` - Wrapper for sigma adjustments (non-mutating)

**Methods:** `yarn`, `ntk`, `base`

**Editing Modes:** `adaptive`, `timestep_aware`, `resolution_aware`, `minimal`, `full`

### rope.py
- `get_1d_rotary_pos_embed()` - Core RoPE with YaRN/NTK/DyPE support
- `find_correction_range()` - YaRN frequency band correction
- `linear_ramp_mask()` - Smooth interpolation between frequency treatments

## Testing Conventions

### Mocking ComfyUI
```python
def fake_common_ksampler(**kwargs):
    latent_payload = kwargs["latent"]
    result = latent_payload.copy()
    result["samples"] = torch.zeros_like(latent_payload["samples"])
    return (result,)

with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
    # Test code
```

### Testing OOM Fallback
- Simulate OOM by raising `RuntimeError("CUDA out of memory")`
- Verify streaming mode activation via `_execute_with_memory_controls` parameters

### Testing Seed Determinism
- Capture seeds passed to `common_ksampler`
- Verify golden ratio stride: `(seed + N * _SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF`

## Publishing

The package publishes to Comfy Registry via GitHub Actions:
- Trigger: push to `main`/`master` with `pyproject.toml` changes
- Action: `Comfy-Org/publish-node-action@v1`
- Version defined in `pyproject.toml`

## Common Tasks

### Adding a New Node
1. Create class in appropriate `src/*.py` file
2. Define `CATEGORY`, `FUNCTION`, `RETURN_TYPES`, `RETURN_NAMES`
3. Implement `INPUT_TYPES` classmethod
4. Add to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
5. Add tests with ComfyUI stubs

### Modifying Upscale Logic
- Core upscaling: `progressive_upscale_latent()` in `flow_matching_upscaler.py`
- Re-noising: `apply_flow_renoise()` - uses `torch.lerp(latent, noise, noise_level)`
- Schedule parsing: `_parse_schedule()` supports linear and exponential curves

### Adding VRAM Optimizations
- Context managers: `_temporary_low_vram_mode()`, `_throttled_attention_budget()`
- Wrapped execution: `_execute_with_memory_controls()`
- Always include `del` for intermediate tensors

## Debugging Tips

- Enable DEBUG logging to see per-channel statistics
- Use `LatentChannelStatsPreview` node to visualize latent statistics
- Check `comfy.utils._common_upscale_calls` in tests for upscale call history
- OOM issues: Look for `_run_with_oom_retry` and streaming fallback activation

## Dependencies

Runtime (from `requirements.txt`):
- `numpy>=2.0`
- `torch>=2.1`
- `einops>=0.6`
- `aiohttp>=3.9`

Development:
- `pytest` (installed by `test.sh`)

## Important Notes for AI Assistants

1. **Always read existing code** before suggesting modifications
2. **Test stubs are required** - tests cannot run against real ComfyUI
3. **Memory management matters** - delete tensors when no longer needed
4. **Seed determinism** - use the golden ratio stride for reproducibility
5. **Mask synchronization** - always resize `noise_mask` alongside latents
6. **Non-mutating patches** - use wrappers like `_DyPEModelSampling` to avoid modifying shared state
7. **Keep imports conditional** - support both package and direct execution modes
