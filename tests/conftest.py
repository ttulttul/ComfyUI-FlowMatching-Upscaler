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

    comfy_module.model_management = model_management_module
    comfy_module.utils = utils_module
    comfy_module.samplers = samplers_module

    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_management_module
    sys.modules["comfy.utils"] = utils_module
    sys.modules["comfy.samplers"] = samplers_module


_ensure_comfy_stubs()
