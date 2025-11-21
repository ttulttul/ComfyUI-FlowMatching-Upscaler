<p align="center">
  <img src="images/Icon.png" alt="ComfyUI Flow Matching Upscaler Icon" width="120">
</p>

# ComfyUI Flow Matching Upscaler

## Overview

This repository ships a progressive upscaling node designed for flow-matching
models such as Qwen Image. The node implements the approach outlined in
`docs/Approach.pdf`: it incrementally doubles resolution, re-noises the latent
in a flow-consistent manner, denoises with the selected sampler, and blends
skip residuals to preserve composition. An optional dilated refinement pass
provides additional global coherence.

Alongside the upscaler, the pack now includes **DyPE for Qwen Image**: a model
patch node that extends Qwen Image’s spatial rotary embeddings and flow-matching
noise schedule so the diffusion model stays coherent far beyond its native
training resolution. Drop it after your model loader to auto-detect geometry,
apply Dynamic Position Extrapolation (DyPE), and keep edits stable with
configurable tapering strategies.

### Memory Use

The progressive upscaler still samples the full latent at once, so extremely
large resolutions can exhaust VRAM. When the sampler reports an out-of-memory
error, the node automatically retries with a streaming fallback: it enables
LOW_VRAM mode and throttles the VRAM budget reported to ComfyUI’s attention
kernels so the UNet iterates over smaller chunks while preserving the full
latent and conditioning. Expect a slower pass, but no need to rewire the graph.

### Note: Upscaling is Optional

You don't necessarily have to use this node pack to **upscale** images.  The
techniques used by the node involve blending a prior latent at an optionally
different scale and then re-sampling on top of it to produce a new latent that
may posess more or different levels of detail -- effectively giving the model a
second chance to generate but with some hints as to the structure that you're
wanting it to generate.

Try setting the `scale_factor` or `total_scale` to 1.0 to play around with
the node's noising, blending, and re-sampling approach without necessarily
also upscaling the latents with each step.

## Installation

1. Clone this repository inside the `custom_nodes/` directory of your ComfyUI
   installation.
2. Launch ComfyUI; the node will be registered as
   **Flow Matching Progressive Upscaler** under *latent/upscaling*.

## Node parameters

### DyPE for Qwen Image

**Required inputs**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `model` | MODEL | – | Qwen Image diffusion model (expects a UNet with `pe_embedder`). |
| `width` | INT `16 → 16384` | `1024` | Target render width in pixels (match your latent). |
| `height` | INT `16 → 16384` | `1024` | Target render height in pixels. |
| `auto_detect` | BOOLEAN | `True` | Try to infer patch size, base resolution, and VAE scale from the model. |
| `base_width` | INT `16 → 16384` | `1024` | Training width to fall back to when auto detection fails. |
| `base_height` | INT `16 → 16384` | `1024` | Training height fallback used with manual geometry. |
| `method` | enum (`yarn`, `ntk`, `base`) | `yarn` | Spatial RoPE extrapolation strategy (YARN recommended for large jumps). |
| `enable_dype` | BOOLEAN | `True` | Toggle Dynamic Position Extrapolation scaling of the embeddings. |
| `dype_exponent` | FLOAT `0.0 → 4.0` | `2.0` | Controls how aggressively DyPE ramps across the sampling trajectory. |
| `base_shift` | FLOAT `0.0 → 10.0` | `1.15` | Baseline shift applied to the flow-matching noise schedule. |
| `max_shift` | FLOAT `0.0 → 10.0` | `1.35` | Maximum shift reached when running at the target resolution. |
| `editing_strength` | FLOAT `0.0 → 1.0` | `1.0` | Scales DyPE while editing (lower values preserve originals). |
| `editing_mode` | enum (`adaptive`, `timestep_aware`, `resolution_aware`, `minimal`, `full`) | `adaptive` | Strategy for tapering DyPE when image conditioning is present. |

**Outputs**

- `model` (`MODEL`): Cloned and patched Qwen Image model with DyPE-enabled positional embeddings.

The node installs a sampler wrapper and exposes log messages whenever geometry is auto-detected or sampler fallbacks trigger. For editing workflows, keep `auto_detect` enabled so the patch reflects the model’s native context length before applying the DyPE scaling.

#### How DyPE Works

<p align="left">
  <img src="images/DyPE-Extrapolation-Process.jpeg" alt="How DyPE works">
</p>

Think of the internal "coordinate system" of models like FLUX or Qwen as a
printed map that covers exactly 1024x1024 pixels. When you try to generate a 4K
image, you are effectively asking the model to drive off the edge of that map.
Without help, the model gets confused and starts hallucinating—often simply
repeating the same subject or texture over and over (the "cloning" effect)
because it ran out of unique coordinate numbers to assign to the new space.
This is where RoPE (Rotary Positional Embeddings) usually comes in; standard
tricks basically stretch that small map to fit the large area, but this often
distorts the image, making it look washed out or structureless.

DyPE solves this by treating the generation process as a journey rather than a
static snapshot. It takes advantage of how diffusion works: early steps
determine the broad shapes, while later steps fill in the fine details. Instead
of stretching the coordinate map permanently, DyPE dynamically adjusts it
during the sampling steps. It keeps the coordinates close to the model’s native
training size at the beginning (ensuring your composition stays solid) and
slowly expands the grid as the KSampler progresses. This allows the model to
"see" the massive 4K canvas clearly right when it needs to paint the
high-frequency details, giving you a coherent high-res image without the
repeating artifacts or the blurry look of standard upscaling.

### Flow Matching Progressive Upscaler

**Required inputs**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `model` | MODEL | – | Flow-matching diffusion model used for denoising each stage. |
| `positive` | CONDITIONING | – | Positive conditioning fed to CFG sampling. |
| `negative` | CONDITIONING | – | Negative conditioning paired with `positive`. |
| `latent` | LATENT | – | Base latent to upscale. Must match the model’s channel layout. |
| `seed` | INT `0 → 2^64-1` | `0` | Base seed used to derive deterministic per-stage seeds. |
| `steps_per_stage` | INT `1 → 256` | `16` | Number of sampler steps executed at every stage (including clean-up). |
| `cfg` | FLOAT `0.0 → 20.0` | `4.5` | Classifier Free Guidance strength passed to the sampler. |
| `sampler_name` | choice | first sampler in ComfyUI list | Sampler backend (e.g., `euler`, `dpmpp_2m`). |
| `scheduler` | choice | first scheduler in ComfyUI list | Scheduler curve applied by the sampler. |
| `total_scale` | FLOAT `≥ 1.0` | `4.0` | Overall scale factor: final size = base size × `total_scale`. |
| `stages` | INT `1 → 5` | `2` | Number of progressive stages to hit `total_scale`; each stage uses `total_scale^(1/stages)`. |
| `renoise_start` | FLOAT `0.0 → 1.0` | `0.35` | Flow-style noise ratio applied at the first stage. |
| `renoise_end` | FLOAT `0.0 → 1.0` | `0.15` | Noise ratio at the final stage; intermediate values interpolate. |
| `skip_blend_start` | FLOAT `0.0 → 1.0` | `0.8` | Mix weight of the upscaled latent before denoising in stage 1 (`1-weight` comes from the refined latent). |
| `skip_blend_end` | FLOAT `0.0 → 1.0` | `0.05` | Final-stage skip weight, exponentially interpolated unless overridden. |
| `upscale_method` | choice | `bicubic` | Resampling algorithm used between stages (falls back if unsupported, e.g., Lanczos with >4 channels). |

**Optional controls**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `noise_schedule_override` | STRING | `""` | Comma-separated per-stage noise ratios. Provide exactly `stages` values (or one value to repeat). Overrides `renoise_start/end`. |
| `skip_schedule_override` | STRING | `""` | Comma-separated skip weights per stage. Provide `stages` values (or one to repeat). Overrides `skip_blend_start/end`. |
| `denoise` | FLOAT `0.0 → 1.0` | `1.0` | Global denoise strength. Each stage uses this unless a cleanup override is configured. |
| `enable_dilated_sampling` | enum | `"enable"` | Adds a dilated refinement pass (downscale → sample → blend) for non-cleanup stages. Disable to save compute. |
| `dilated_downscale` | FLOAT `1.0 → 4.0` | `2.0` | Factor used when shrinking the latent for the dilated pass. Higher = more global look, longer runtime. |
| `dilated_blend` | FLOAT `0.0 → 1.0` | `0.25` | Contribution of the dilated refinement back into the main latent. |
| `cleanup_stage` | enum | `"disable"` | When enabled, appends an extra non-scaling denoise stage to polish the output. |
| `cleanup_noise` | FLOAT `0.0 → 1.0` | `0.0` | Noise ratio applied during the cleanup stage. Leave at `0` to only denoise existing detail. |
| `cleanup_denoise` | FLOAT `0.0 → 1.0` | `0.4` | Denoise strength for the cleanup pass. Ignored when `cleanup_stage` is disabled. |

#### How Flow Matching Upscaling Works

<p align="left">
  <img src="images/Flow-Matching-Upscale-Process.jpeg" alt="How Flow Matching Upscaling works">
</p>

In traditional diffusion models (like SD1.5 or SDXL), generating an image is
often described as "removing noise" layer by layer. However, **Flow Matching**
models (like Qwen) work a bit differently: think of them as calculating a
direct, straight-line path (a "flow") from a chaotic state (noise) to an
ordered state (your image). When you use Flow Matching Upscaler nodes, you aren't
just denoising; you are traveling along a specific trajectory. The first stage
generates your base image effectively by following this path at a lower
resolution, establishing the composition and major colors quickly and
efficiently.

The magic happens in the second stage. Instead of just stretching the image and
adding random noise on top (like a standard "img2img" workflow), the Flow
Matching nodes physically resize your latents and then mathematically "rewind"
the clock. As shown in the center of the diagram, we step back along the
timeline—for example, rewinding from the finished state ($t=0.0$) back to a
mid-point ($t=0.6$). This places the model back onto the flow trajectory but at
a much higher resolution. Because the model is resuming an existing journey
rather than starting a new one, it doesn't hallucinate new objects or change
your composition; it simply "flows" forward again, filling in the missing
high-frequency details that naturally belong on that path.

##### A note on "Dilated Sampling"

Dilated sampling adds a coarse refinement lap immediately after the main
sampler completes each stage. The node:

1. Downscales the freshly denoised latent by `dilated_downscale` using an area kernel, which acts like a low-pass filter.
2. Runs a short sampler pass in that reduced space with a seed offset so it explores slightly different noise.
3. Upscales the new latent back to stage resolution and mixes it into the main result with weight `dilated_blend`.

Because high resolutions exaggerate fine-grained hallucinations (e.g., extra
pores or glittering fabric), that low-pass/upsample cycle gently suppresses
high-frequency artifacts while keeping large shapes intact. Higher downscale
factors and blend weights increase the smoothing but also lengthen runtime and
can wash out intentional texture, so start with the defaults (`downscale =
2.0`, `blend = 0.25`) and adjust toward `0.15–0.35` when you only need a light
cleanup.

**Outputs**

- `latent` (`LATENT`): The refined latent after the final stage.
- `next_seed` (`INT`): Base seed advanced by one golden-ratio stride for easy chaining.
- `model`, `positive`, `negative`: Straight-through handles so you can wire downstream nodes without re-routing.

A thumbnail preview is rendered on the node while sampling, matching the native KSampler UX.

## Example workflow

![Flow Matching Upscaler example workflow](<examples/Flow Matching Upscaler.png>)

### Modular nodes

For workflows that benefit from ComfyUI’s caching, the repo ships the
`FlowMatchingStage` node alongside the composite progressive upscaler. Chain
multiple stage nodes and feed the output latent from one stage into the next; only
the stages whose inputs change will be
recomputed.

#### LatentChannelStatsPreview

This lightweight debug node converts a `LATENT` input into a batched `IMAGE`
tensor whose top half plots per-channel means and bottom half plots standard
deviations. The preview honors the logging helper above (channels are reduced
across batch/temporal dimensions) so it is useful for spotting channels that
dominate the energy budget before experimenting with per-channel blends.
Each bar carries the channel index below it, while the y-axis shows min/max
values so you can gauge scale quickly; the panel labels make it clear that the
blue section corresponds to means and the orange section to standard deviations.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `latent` | LATENT | – | Source latent whose channel statistics are visualized. |
| `channel_limit` | INT `1 → 64` | `16` | Number of channels to show; an overflow marker appears when additional channels are truncated. |
| `height` | INT `72 → 1024` | `256` | Output image height in pixels; width scales with channel count. |

The resulting tensor uses a dark background, blue bars for means, and orange
bars for standard deviations so you can drop the node into any preview chain
without additional styling.

#### FlowMatchingStage

**Required inputs**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `model` | MODEL | – | Flow-matching diffusion model shared across the pipeline. |
| `positive` | CONDITIONING | – | Positive conditioning for this stage. |
| `negative` | CONDITIONING | – | Negative conditioning for this stage. |
| `latent` | LATENT | – | Latent entering the stage (typically previous stage’s output). |
| `seed` | INT `0 → 2^64-1` | `0` | Seed used for both re-noising and sampler invocation. |
| `steps` | INT `1 → 256` | `16` | Sampler steps for this stage only. |
| `cfg` | FLOAT `0.0 → 20.0` | `4.5` | Guidance strength passed to the sampler. |
| `sampler_name` | choice | first sampler in ComfyUI list | Back-end sampler. |
| `scheduler` | choice | first scheduler in ComfyUI list | Scheduler curve per stage. |
| `scale_factor` | FLOAT `0.1 → 8.0` | `1.0` | Upscale factor applied before sampling (keep at `1.0` for pure refinement). |
| `noise_ratio` | FLOAT `0.0 → 1.0` | `0.0` | Flow-style noise amount mixed in before the sampler runs. |
| `skip_blend` | FLOAT `0.0 → 1.0` | `0.5` | Blend factor between the pre-sampler latent and the denoised result. |
| `denoise` | FLOAT `0.0 → 1.0` | `1.0` | Denoise strength fed to the sampler (lower for gentle edits). |
| `upscale_method` | choice | `bicubic` | Resampler used by the latent resize. |

**Optional controls**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `enable_dilated_sampling` | enum | `"disable"` | Enables the dilated refinement pass for this specific stage. |
| `reduce_memory_use` | enum | `"enable"` | Reuses the upscaled latent instead of cloning it ahead of skip blending to save VRAM. |
| `dilated_downscale` | FLOAT `1.0 → 4.0` | `2.0` | Downscale factor when dilated sampling runs. |
| `dilated_blend` | FLOAT `0.0 → 1.0` | `0.25` | Blend weight used when reintegrating the dilated result. |

**Outputs**

- `latent` (`LATENT`): Stage output after blending and optional dilation.
- `presampler_latent` (`LATENT`): Latent that entered the sampler after noise injection (useful for debugging).
- `next_seed` (`INT`): `seed + stride` helper value for chaining stages deterministically.
- `model`, `positive`, `negative`: Pass-through handles for downstream nodes.

Stage nodes display the same inline thumbnail preview as the composite node.

When chaining stages manually, remember to forward the `next_seed` output into
the subsequent stage’s `seed` input to maintain deterministic noise progression.
If you keep `reduce_memory_use` enabled, downstream custom logic that inspects
the upscaled latent while this node runs may observe in-place updates because
the tensor is shared instead of cloned.
If a stage hits an out-of-memory condition, it retries automatically with the
same streaming fallback as the progressive node: LOW_VRAM mode is temporarily
enabled and attention kernels see a capped free-memory budget so they chunk the
computation while preserving global conditioning.

## More about DyPE

The `method` parameter determine how the model handles "coordinates" that fall outside
its native training resolution (e.g., when asking a 1024px model to generate a
4096px image).

### 1. YaRN (Yet another RoPE extension)
**The Recommended Default**

YaRN is the most sophisticated method and the core foundation of the DyPE
technique. Unlike simple stretching, YaRN divides the model's internal
frequencies into different bands. It "stretches" the low-frequency signals
(which define big shapes and global structure) to fit the larger canvas, but it
leaves high-frequency signals (which define fine texture and noise) largely
alone.

*   **How it works in DyPE:** When `enable_dype` is active, the node implements
    "DY-YaRN." It dynamically ramps the interpolation parameters over time.
    Early in the generation, it aggressively adjusts the coordinates to secure
    the global composition. As the image clarifies, it eases off, allowing the
    model to render crisp, high-frequency details without blurring.
*   **Best for:** Ultra-high resolutions (4K+), maintaining sharp textures, and
    preventing the "mushy" look often associated with model stretching.

### 2. NTK (Neural Tangent Kernel)
**The "Smoother" Alternative**

NTK Scaling is a broader, more uniform approach to coordinate stretching.
Instead of treating low and high frequencies differently like YaRN, NTK
effectively changes the "rotation speed" of the positional embeddings across
the board. It tells the model to pretend that spatial changes happen more
slowly, allowing the 1024px "map" to cover a 4096px area.

*   **How it works in DyPE:** The node applies a scalar multiplier to the
    rotation frequencies (`theta`). When `enable_dype` is on, this multiplier
    scales up and down based on the timestep.
*   **Best for:** Situations where YaRN looks "over-sharpened" or introduces
    gridded artifacts. It is also the default recommendation for the **Qwen
    CLIP** (text encoder) node, as language models tend to respond better to
    the uniform scaling of NTK than the complex banding of YaRN.

### 3. Base
**The Control Group / Vanilla Behavior**

This setting effectively turns off the spatial extrapolation mathematics. It
forces the model to use its original, unmodified positional embeddings.

*   **How it works:** If you generate at 1024x1024, this is identical to the
    original model. If you generate at 4K, the model will run out of unique
    coordinates. This usually results in the model repeating the same image
    patches over and over (tiling artifacts) or completely losing coherence.
*   **Best for:** Comparisons (to see "before and after"), or generating at the
    model's native resolution. In the Qwen Image node, you might use this if
    you only want to use the **Noise Schedule Shift** features of DyPE without
    altering the spatial embeddings.

### Summary Table

| Method | Fidelity | Sharpness | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **YaRN** | High | High | **Default for Images.** Best for 2K/4K generation and preserving texture. |
| **NTK** | Medium | Medium | Use for images if YaRN creates jagged artifacts. |
| **Base** | Native | N/A | Testing, native resolution, or when only using noise-schedule shifts. |

## Development

### Running tests

Use `pytest`; the suite installs lightweight `comfy` and `nodes` stubs so it can
run outside of a live ComfyUI process:

```bash
pytest
```

### Logging

The node uses Python's `logging` subsystem (`logging.getLogger(__name__)`) so it
respects ComfyUI's runtime logging configuration.
Enabling DEBUG level now emits per-channel mean and standard deviation
diagnostics around skip/dilated blending steps, which is handy when probing how
the 16-channel Qwen latent space behaves through the pipeline.
