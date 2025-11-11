# ComfyUI Flow Matching Upscaler

## Overview

This repository ships a progressive upscaling node designed for flow-matching
models such as Qwen Image. The node implements the approach outlined in
`docs/Approach.pdf`: it incrementally doubles resolution, re-noises the latent
in a flow-consistent manner, denoises with the selected sampler, and blends
skip residuals to preserve composition. An optional dilated refinement pass
provides additional global coherence.

## Installation

1. Clone this repository inside the `custom_nodes/` directory of your ComfyUI
   installation.
2. Launch ComfyUI; the node will be registered as
   **Flow Matching Progressive Upscaler** under *latent/upscaling*.

## Node parameters

| Parameter | Description |
|-----------|-------------|
| `total_scale` | Overall scale factor from the input latent to the final output. |
| `stages` | Number of progressive stages (each stage scales by `total_scale^(1/stages)`). |
| `renoise_start / end` | Flow-style noise ratios at the first/last stage. Intermediate values are interpolated unless overridden. |
| `skip_blend_start / end` | Skip residual blend weights at the first/last stage (defaults 0.20→0.05 to keep the sampler output dominant). |
| `noise_schedule_override` | Optional comma-separated per-stage noise values. |
| `skip_schedule_override` | Optional comma-separated per-stage skip blend weights. |
| `enable_dilated_sampling` | Enables a downscale → sample → upscale refinement for better global structure. |

All other sampler-specific arguments (`sampler_name`, `scheduler`, `cfg`,
`steps_per_stage`, `denoise`) are passed through to ComfyUI's sampling
infrastructure.

## Development

### Running tests

Use the standard library test runner:

```bash
python -m unittest discover -s tests
```

### Logging

The node uses Python's `logging` subsystem (`logging.getLogger(__name__)`) so it
respects ComfyUI's runtime logging configuration.
