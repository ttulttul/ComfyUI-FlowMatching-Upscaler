import sys
import types
import torch

class _DummyProgressBar:
    def __init__(self, total):
        self.total = total
        self.updated = 0

    def update(self, amount):
        self.updated += amount


def _ensure_comfy_stubs():
    if "comfy" in sys.modules:
        return

    comfy_module = types.ModuleType("comfy")

    model_management_module = types.ModuleType("comfy.model_management")
    sampling_module = types.ModuleType("comfy.model_sampling")
    model_patcher_module = types.ModuleType("comfy.model_patcher")

    class _VRAMState:
        NORMAL = "normal"
        LOW_VRAM = "low_vram"

    model_management_module.VRAMState = _VRAMState
    model_management_module.vram_state = _VRAMState.NORMAL
    model_management_module.OOM_EXCEPTION = RuntimeError

    model_management_module.get_torch_device = lambda: torch.device("cpu")
    model_management_module.unet_dtype = lambda: torch.float32
    model_management_module.load_models_gpu = lambda models: None
    model_management_module.soft_empty_cache = lambda *_args, **_kwargs: None

    def _dummy_get_free_memory(dev=None, torch_free_too=False):
        total = 2 * 1024 * 1024 * 1024  # 2 GB in bytes
        if torch_free_too:
            return total, total
        return total

    model_management_module.get_free_memory = _dummy_get_free_memory

    utils_module = types.ModuleType("comfy.utils")
    utils_module.ProgressBar = _DummyProgressBar

    upscale_call_log = []

    def common_upscale(samples, width, height, upscale_method, crop):
        upscale_call_log.append(
            {
                "method": upscale_method,
                "shape": tuple(samples.shape),
                "target": (height, width),
            }
        )
        tensor = samples
        orig_shape = samples.shape

        squeeze_channel = False
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
            squeeze_channel = True

        reshape_higher = False
        if tensor.ndim > 4:
            reshape_higher = True
            reshape_base = tensor.shape
            tensor = tensor.reshape(reshape_base[0], reshape_base[1], -1, reshape_base[-2], reshape_base[-1])
            tensor = tensor.movedim(2, 1).reshape(-1, reshape_base[1], reshape_base[-2], reshape_base[-1])

        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(height, width),
            mode="nearest",
        )

        if reshape_higher:
            tensor = tensor.reshape(orig_shape[0], -1, orig_shape[1], height, width)
            tensor = tensor.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))
        elif squeeze_channel:
            tensor = tensor.squeeze(1)

        return tensor

    utils_module.common_upscale = common_upscale
    utils_module._common_upscale_calls = upscale_call_log

    samplers_module = types.ModuleType("comfy.samplers")

    class _DummyKSampler:
        KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral",
                "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2",
                "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
                "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde",
                "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde",
                "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
                "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
                "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "res_multistep",
                "res_multistep_cfg_pp", "res_multistep_ancestral",
                "res_multistep_ancestral_cfg_pp", "gradient_estimation",
                "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3",
                "sa_solver", "sa_solver_pece"]
        SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]
        SAMPLERS = tuple(SAMPLER_NAMES)
        SCHEDULERS = ("normal", "karras", "exponential", "simple")

        def __init__(self, model, steps, device=None):
            self.model = model
            self.steps = steps
            self.device = device or torch.device("cpu")
            self.sigmas = torch.linspace(1.0, 0.0, steps)

        def sample(self, *args, **kwargs):
            return torch.zeros((1, 16, 1, 8, 8), device=self.device)

    samplers_module.KSampler = _DummyKSampler

    class _ModelSamplingFlux:
        pass

    def _flux_time_shift(*_args, **_kwargs):
        return 0.0

    sampling_module.ModelSamplingFlux = _ModelSamplingFlux
    sampling_module.flux_time_shift = _flux_time_shift

    class _StubEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = 10_000.0
            self.axes_dim = [4, 4, 4]
            self.patch_size = 2

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            batch = ids.shape[0]
            tokens = ids.shape[1]
            embedding_dim = sum(self.axes_dim) // 2
            return torch.zeros(batch, 1, tokens, embedding_dim, 2, 2)

    class _StubDiffusionModel:
        def __init__(self):
            self.pe_embedder = _StubEmbedder()
            self.patch_size = 2
            self.vae_scale_factor = 8
            self.sample_size = (16, 16)
            self.config = types.SimpleNamespace(
                sample_size=self.sample_size,
                patch_size=self.patch_size,
                vae_scale_factor=self.vae_scale_factor,
            )

    class _ModelSamplingFlux:
        def __init__(self):
            self._dype_patched = False
            self.sigma_max = torch.tensor(1.0)

        def sigma(self, timestep: float):
            return timestep

    class _StubModelWrapper:
        def __init__(self):
            self.diffusion_model = _StubDiffusionModel()
            self.model_sampling = _ModelSamplingFlux()

    class _ModelPatcherStub:
        def __init__(self):
            self.model = _StubModelWrapper()
            self._object_patches: dict[str, object] = {}
            self._wrapper = None

        def clone(self):
            cloned = _ModelPatcherStub()
            cloned.model = self.model
            return cloned

        def add_object_patch(self, path: str, obj) -> None:
            self._object_patches[path] = obj
            if path == "diffusion_model.pe_embedder":
                self.model.diffusion_model.pe_embedder = obj
            elif path == "model_sampling":
                self.model.model_sampling = obj

        def set_model_unet_function_wrapper(self, wrapper) -> None:
            self._wrapper = wrapper

    model_patcher_module.ModelPatcher = _ModelPatcherStub

    comfy_module.model_management = model_management_module
    comfy_module.utils = utils_module
    comfy_module.samplers = samplers_module
    comfy_module.model_sampling = sampling_module
    comfy_module.model_patcher = model_patcher_module

    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_management_module
    sys.modules["comfy.utils"] = utils_module
    sys.modules["comfy.samplers"] = samplers_module
    sys.modules["comfy.model_sampling"] = sampling_module
    sys.modules["comfy.model_patcher"] = model_patcher_module


_ensure_comfy_stubs()
