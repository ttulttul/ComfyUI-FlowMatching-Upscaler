# Learnings

- Implementing progressive flow-matching upscaling inside ComfyUI is easiest when
  delegating denoising to `nodes.common_ksampler`, which already supports flow
  models via `ModelType.FLOW`.
- Flow-style re-noising works well with ComfyUI latents by generating CPU noise
  seeds through `torch.Generator(device="cpu")`, keeping parity with core noise
  preparation utilities.
- A lightweight dilated refinement pass can be approximated by running a
  downsampled sampling pass and blending it back, yielding a simple but effective
  global coherence improvement without reproducing DemoFusion's full dilation logic.
- Providing an optional clean-up stage (no scaling, zero skip blend) helps remove
  interpolation artifacts after progressive upscaling while keeping control over
  added noise and denoising strength.
- An exponential skip schedule (high early weight tapering toward zero) keeps
  structure stable in early passes while letting later stages and optional clean-up
  add detail without reintroducing latent artifacts.
- Exposing a per-stage node mirrors the progressive pipeline while allowing
  ComfyUI’s existing cache to short-circuit unchanged stages, speeding up
  iteration on late-stage parameters.
- Shipping a lightweight frontend extension lets custom nodes hook into ComfyUI’s
  live preview events so users retain the familiar inline thumbnail experience.
- Progressive stages currently invoke `common_ksampler` on the full latent; to
  make ultra-high resolutions practical we need an automatic fallback (e.g.
  tiling or attention chunking) when `comfy.model_management` reports
  insufficient VRAM.
- Throttling the reported free VRAM (and temporarily forcing LOW_VRAM streaming)
  nudges ComfyUI’s attention kernels into smaller chunks; wiring this in as an
  automatic fallback keeps the full-scene conditioning intact while stretching
  sampling time to fit tight memory budgets.
- When running beyond a model's native training resolution, enabling the dilated
  refinement pass reliably suppresses the over-sharp "hallucinated pore" detail
  by re-sampling a low-pass-filtered latent before blending it back.
- When running tests outside ComfyUI we must hydrate comprehensive stubs for
  `comfy.utils` and `comfy.model_management` (including VRAM controls and
  samplers) so the node module imports successfully without ComfyUI’s runtime.
- Channel-wise diagnostics surfaced via DEBUG logging helped confirm that the
  skip and dilated blends maintain stable statistics across the 16 latent
  channels, providing a foundation for future per-channel mixing experiments.
- Visualizing the per-channel means/stds directly inside ComfyUI via a
  `LATENT`→`IMAGE` helper node makes it easier to spot channels worth gating,
  and the rendering math is lightweight enough to run in pure NumPy without
  pulling matplotlib into the dependency tree. Adding inline axis ticks and
  channel indices removed the guesswork when correlating bars with latent
  channels.
- Loading DyPE’s Qwen Image patch locally (outside of ComfyUI’s package loader)
  required explicit import fallbacks and dynamic module loading so both pytest
  stubs and the live runtime share a single implementation without relying on
  the ambient `src` namespace.
- Progressive stages must resize any accompanying `noise_mask` in lockstep with
  the latent—otherwise sampler inputs drift after the first upscale. Keeping
  the mask synchronized per stage (and covering it with targeted tests) guards
  inpaint workflows against subtle misalignment bugs.
- Pytest needs a richer `ModelPatcher` stub (diffusion model, embedder, and sampler)
  so DyPE helpers can exercise their patches without tripping ValueErrors; mirroring
  the live attributes in the shared test fixture keeps individual test modules simple.
- Collapsing dilated refinement onto the FFT-based frequency blend removed the
  method selector surface area—our tests now focus on validating that helper and the
  multi-frame lerp fallback instead of juggling several legacy modes.
- Comfy's runtime enforces that `model_sampling` children are `nn.Module` instances,
  so our DyPE wrapper now subclasses `torch.nn.Module` to satisfy `ModelPatcher` and
  GPU loading without mutating the wrapped sampler.
