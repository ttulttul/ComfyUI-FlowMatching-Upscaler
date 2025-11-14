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
- Throttling the reported free VRAM (and optionally forcing LOW_VRAM streaming)
  nudges ComfyUI’s attention kernels into smaller chunks, letting the stage keep
  global conditioning intact while stretching sampling time to fit tight memory
  budgets.
