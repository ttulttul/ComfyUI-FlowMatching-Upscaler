import logging
import pathlib
import sys
import types

import pytest
import torch


def _ensure_comfy_stubs() -> None:
    comfy_module = sys.modules.get("comfy")
    if comfy_module is None:
        comfy_module = types.ModuleType("comfy")
        comfy_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["comfy"] = comfy_module

    if not hasattr(comfy_module, "samplers"):
        samplers_module = types.ModuleType("comfy.samplers")

        class _SamplerStub:
            SAMPLERS = ("test_sampler",)
            SCHEDULERS = ("test_scheduler",)

        samplers_module.KSampler = _SamplerStub  # type: ignore[attr-defined]
        comfy_module.samplers = samplers_module  # type: ignore[attr-defined]
        sys.modules["comfy.samplers"] = samplers_module

    if not hasattr(comfy_module, "utils"):
        utils_module = types.ModuleType("comfy.utils")
        comfy_module.utils = utils_module  # type: ignore[attr-defined]
        sys.modules["comfy.utils"] = utils_module

    model_patcher_module = sys.modules.get("comfy.model_patcher")
    model_sampling_module = sys.modules.get("comfy.model_sampling")
    if model_patcher_module is None or model_sampling_module is None:
        model_patcher_module = types.ModuleType("comfy.model_patcher")

        class _StubEmbedder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = 10_000.0
                self.axes_dim = [4, 4, 4]
                self.patch_size = 2

            def forward(self, ids: torch.Tensor) -> torch.Tensor:
                batch = ids.shape[0]
                tokens = ids.shape[1]
                return torch.zeros(batch, 1, tokens, 6, 2, 2)

        class _StubDiffusionModel:
            def __init__(self):
                self.pe_embedder = _StubEmbedder()
                self.patch_size = 2
                self.vae_scale_factor = 8
                self.sample_size = (16, 16)
                self.config = types.SimpleNamespace(
                    sample_size=(16, 16),
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

        class _StubModelPatcher:
            def __init__(self):
                self.model = _StubModelWrapper()
                self._wrapper = None

            def clone(self):
                cloned = _StubModelPatcher()
                cloned.model = self.model
                return cloned

            def add_object_patch(self, path: str, obj) -> None:
                if path == "diffusion_model.pe_embedder":
                    self.model.diffusion_model.pe_embedder = obj

            def set_model_unet_function_wrapper(self, wrapper) -> None:
                self._wrapper = wrapper

        model_sampling_module = types.ModuleType("comfy.model_sampling")

        def flux_time_shift(mu: float, sigma: float, timestep):
            return timestep * max(mu, 0.0)

        model_sampling_module.ModelSamplingFlux = _ModelSamplingFlux  # type: ignore[attr-defined]
        model_sampling_module.flux_time_shift = flux_time_shift  # type: ignore[attr-defined]

        model_patcher_module.ModelPatcher = _StubModelPatcher  # type: ignore[attr-defined]

        comfy_module.model_patcher = model_patcher_module  # type: ignore[attr-defined]
        comfy_module.model_sampling = model_sampling_module  # type: ignore[attr-defined]
        sys.modules["comfy.model_patcher"] = model_patcher_module
        sys.modules["comfy.model_sampling"] = model_sampling_module

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ensure_comfy_stubs()

import src.qwen_spatial as qwen_spatial  # noqa: E402
from src.dype_qwen_image import DyPEQwenImage  # noqa: E402


class _ListHandler(logging.Handler):
    def __init__(self, bucket: list[str], level: int = logging.INFO):
        super().__init__(level=level)
        self.bucket = bucket

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            message = record.msg  # type: ignore[assignment]
        self.bucket.append(str(message))


def _make_model_patcher():
    from comfy.model_patcher import ModelPatcher  # type: ignore

    return ModelPatcher()


class _RecordingEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = 10_000.0
        self.axes_dim = [4, 4, 4]
        self.calls = 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        batch = ids.shape[0]
        tokens = ids.shape[1]
        embedding_dim = sum(self.axes_dim) // 2
        return torch.zeros(batch, 1, tokens, embedding_dim, 2, 2)


def _make_token_ids(expanded: bool) -> torch.Tensor:
    text_tokens = torch.tensor(
        [
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    if expanded:
        image_tokens = torch.tensor(
            [
                [0.0, -2.0, -2.0],
                [0.0, -2.0, 1.0],
                [0.0, 1.0, -2.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
    else:
        image_tokens = torch.tensor(
            [
                [0.0, -1.0, -1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    return torch.cat([text_tokens, image_tokens], dim=0).unsqueeze(0)


def test_qwen_spatial_embedder_expands_when_needed():
    backing = _RecordingEmbedder()
    embedder = qwen_spatial.QwenSpatialPosEmbed(
        theta=10_000.0,
        axes_dim=backing.axes_dim,
        patch_size=2,
        vae_scale_factor=8,
        method="ntk",
        enable_dype=False,
        dype_exponent=1.0,
        base_resolution=(32, 32),
        target_resolution=(64, 64),
        backing_embedder=backing,
        editing_strength=0.0,
        editing_mode="full",
    )

    ids = _make_token_ids(expanded=True)
    output = embedder(ids)
    assert output.shape[0] == 1
    assert output.shape[1] == 1
    assert output.shape[-2:] == (2, 2)
    tokens = ids.shape[1]
    embedding_dim = sum(backing.axes_dim) // 2
    assert output.shape == (1, 1, tokens, embedding_dim, 2, 2)
    assert backing.calls == 0


def test_apply_dype_to_qwen_image_installs_embedder_and_wrapper():
    patcher = _make_model_patcher()

    patched = qwen_spatial.apply_dype_to_qwen_image(
        model=patcher,
        width=2048,
        height=2048,
        method="ntk",
        enable_dype=True,
        dype_exponent=1.5,
        base_width=1024,
        base_height=1024,
        base_shift=1.15,
        max_shift=1.35,
        auto_detect=False,
        editing_strength=1.0,
        editing_mode="full",
    )

    embedder = patched.model.diffusion_model.pe_embedder
    assert isinstance(embedder, qwen_spatial.QwenSpatialPosEmbed)
    assert hasattr(patched, "_wrapper") and patched._wrapper is not None
    assert patched.model.model_sampling._dype_patched is True

    args_dict = {
        "input": torch.randn(1),
        "timestep": torch.tensor([0.5]),
        "c": {},
    }

    def _dummy_model_fn(x, timestep, **kwargs):
        return x, timestep, kwargs

    result = patched._wrapper(_dummy_model_fn, args_dict)
    assert isinstance(result, tuple) and result[1] is args_dict["timestep"]
    assert embedder.current_editing is False


def test_apply_dype_to_qwen_image_auto_detects_geometry():
    patcher = _make_model_patcher()
    recorded_messages: list[str] = []
    handler = _ListHandler(recorded_messages)
    qwen_spatial.logger.addHandler(handler)
    previous_level = qwen_spatial.logger.level
    qwen_spatial.logger.setLevel(logging.INFO)

    try:
        patched = qwen_spatial.apply_dype_to_qwen_image(
            model=patcher,
            width=2048,
            height=2048,
            method="yarn",
            enable_dype=True,
            dype_exponent=2.0,
            base_width=512,
            base_height=512,
            base_shift=1.0,
            max_shift=1.2,
            auto_detect=True,
            editing_strength=0.5,
            editing_mode="adaptive",
        )

        embedder = patched.model.diffusion_model.pe_embedder
        assert embedder.grid.base_axes[0] > 0
        assert any("detected geometry" in message for message in recorded_messages)
    finally:
        qwen_spatial.logger.removeHandler(handler)
        qwen_spatial.logger.setLevel(previous_level)


def test_apply_dype_to_qwen_image_patches_fallback_sampler():
    patcher = _make_model_patcher()

    class _FallbackSampler:
        def __init__(self):
            self.sigma_max = 2.0
            self._dype_patched = False

        def sigma(self, timestep):
            return timestep * 2.0

    fallback_sampler = _FallbackSampler()
    patcher.model.model_sampling = fallback_sampler

    patched = qwen_spatial.apply_dype_to_qwen_image(
        model=patcher,
        width=2048,
        height=2048,
        method="ntk",
        enable_dype=True,
        dype_exponent=2.0,
        base_width=1024,
        base_height=1024,
        base_shift=1.0,
        max_shift=1.3,
        auto_detect=False,
        editing_strength=1.0,
        editing_mode="full",
    )

    assert fallback_sampler._dype_patched is True
    adjusted_sigma = fallback_sampler.sigma(0.5)
    assert adjusted_sigma > 0.5
    assert isinstance(patched.model.diffusion_model.pe_embedder, qwen_spatial.QwenSpatialPosEmbed)


def test_dype_qwen_image_node_executes_and_logs(caplog: pytest.LogCaptureFixture):
    patcher = _make_model_patcher()
    node = DyPEQwenImage()
    caplog.set_level(logging.INFO, logger="src.dype_qwen_image")
    caplog.set_level(logging.INFO, logger="src.qwen_spatial")

    result = node.apply(
        model=patcher,
        width=1024,
        height=2048,
        auto_detect=True,
        base_width=1024,
        base_height=1024,
        method="yarn",
        enable_dype=True,
        dype_exponent=2.0,
        base_shift=1.15,
        max_shift=1.35,
        editing_strength=0.8,
        editing_mode="adaptive",
    )

    assert isinstance(result, tuple) and hasattr(result[0], "model")
    node_messages = [record.message for record in caplog.records if record.name == "src.dype_qwen_image"]
    assert any("DyPE_QwenImage: requested patch" in message for message in node_messages)
