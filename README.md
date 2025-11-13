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
| `skip_blend_start / end` | Skip residual blend weights at the first/last stage (defaults 0.80→0.05; interpolated exponentially across stages). |
| `noise_schedule_override` | Optional comma-separated per-stage noise values. |
| `skip_schedule_override` | Optional comma-separated per-stage skip blend weights. |
| `enable_dilated_sampling` | Enables a downscale → sample → upscale refinement for better global structure. |
| `cleanup_stage` | Adds a final non-scaling clean-up pass with a fresh seed. |
| `cleanup_noise` | Flow-style noise ratio for the clean-up pass (0 keeps the latent untouched before denoising). |
| `cleanup_denoise` | Denoising strength used during the clean-up pass. |

The node outputs the refined latent, the next seed (base seed plus stage stride),
the latent that entered the sampler (`presampler_latent`), and pass-through
references for the model and conditioning so you can chain additional nodes if
needed. A thumbnail preview is rendered directly on the node while sampling,
matching the native KSampler experience.

### Modular nodes

For workflows that benefit from ComfyUI’s caching, the repo also exposes a
`FlowMatchingStage` node. Chain multiple stage nodes and feed the output latent
from one stage into the next; only the stages whose inputs change will be
recomputed.

`FlowMatchingStage` accepts the same sampler/scheduler/CFG inputs as the
composite node, plus per-stage controls:

- `scale_factor`, `noise_ratio`, `skip_blend`, and `denoise` determine how the
  stage behaves.
- Optional dilated refinement mimics the all-in-one node’s global touch-up step.
- The seed input lets you deterministically perturb or randomise individual
  stages.
- Outputs include the updated seed, the latent fed into the sampler (`presampler_latent`), and passthrough model/conditioning handles to simplify chaining.
- Stage nodes also display the live thumbnail preview via the shared frontend
  extension.

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
