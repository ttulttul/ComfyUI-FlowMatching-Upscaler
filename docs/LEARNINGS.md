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
