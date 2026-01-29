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

    model_patcher_module = sys.modules.get("comfy.model_patcher")
    if model_patcher_module is None:
        model_patcher_module = types.ModuleType("comfy.model_patcher")

        class _DummyModelPatcher:
            pass

        model_patcher_module.ModelPatcher = _DummyModelPatcher  # type: ignore[attr-defined]
        comfy_module.model_patcher = model_patcher_module  # type: ignore[attr-defined]
        sys.modules["comfy.model_patcher"] = model_patcher_module

    model_sampling_module = sys.modules.get("comfy.model_sampling")
    if model_sampling_module is None:
        model_sampling_module = types.ModuleType("comfy.model_sampling")
        comfy_module.model_sampling = model_sampling_module  # type: ignore[attr-defined]
        sys.modules["comfy.model_sampling"] = model_sampling_module

    if not hasattr(model_sampling_module, "flux_time_shift"):
        def flux_time_shift(mu: float, sigma: float, timestep):
            return timestep * max(mu, 0.0)

        model_sampling_module.flux_time_shift = flux_time_shift  # type: ignore[attr-defined]

    if not hasattr(model_sampling_module, "ModelSamplingFlux"):
        class _ModelSamplingFlux:
            def __init__(self):
                self.sigma_max = torch.tensor(1.0)

            def sigma(self, timestep):
                return timestep

        model_sampling_module.ModelSamplingFlux = _ModelSamplingFlux  # type: ignore[attr-defined]


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ensure_comfy_stubs()

import src.qwen_spatial as qwen_spatial  # noqa: E402
from src.dype_flux2 import DyPEFlux2  # noqa: E402


class _FluxEmbedder(torch.nn.Module):
    def __init__(self, axes_dim=None):
        super().__init__()
        self.theta = 2000.0
        self.axes_dim = axes_dim or [32, 32, 32, 32]
        self.patch_size = 1

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        batch = ids.shape[0]
        tokens = ids.shape[1]
        embedding_dim = sum(self.axes_dim) // 2
        return torch.zeros(batch, 1, tokens, embedding_dim, 2, 2)


class _FluxDiffusionModel:
    def __init__(self, embedder=None):
        self.pe_embedder = embedder or _FluxEmbedder()
        self.patch_size = 1
        self.vae_scale_factor = 8


class _FluxModelWrapper:
    def __init__(self, embedder=None):
        from comfy import model_sampling  # type: ignore

        self.diffusion_model = _FluxDiffusionModel(embedder=embedder)
        self.model_sampling = model_sampling.ModelSamplingFlux()


class _FluxModelPatcher:
    def __init__(self, embedder=None):
        self.model = _FluxModelWrapper(embedder=embedder)
        self._wrapper = None
        self._object_patches = {}

    def clone(self):
        cloned = _FluxModelPatcher()
        cloned.model = self.model
        return cloned

    def add_object_patch(self, path: str, obj) -> None:
        self._object_patches[path] = obj
        if path == "diffusion_model.pe_embedder":
            self.model.diffusion_model.pe_embedder = obj

    def set_model_unet_function_wrapper(self, wrapper) -> None:
        self._wrapper = wrapper


def _make_flux_ids() -> torch.Tensor:
    text_tokens = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    image_tokens = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return torch.cat([text_tokens, image_tokens], dim=0).unsqueeze(0)


def test_flux2_spatial_embedder_accepts_four_axes():
    backing = _FluxEmbedder()
    embedder = qwen_spatial.QwenSpatialPosEmbed(
        theta=backing.theta,
        axes_dim=backing.axes_dim,
        patch_size=1,
        vae_scale_factor=8,
        method="ntk",
        enable_dype=True,
        dype_exponent=1.0,
        base_resolution=(1024, 1024),
        target_resolution=(2048, 2048),
        backing_embedder=backing,
        editing_strength=0.0,
        editing_mode="full",
    )

    ids = _make_flux_ids()
    output = embedder(ids)
    tokens = ids.shape[1]
    embedding_dim = sum(backing.axes_dim) // 2
    assert output.shape == (1, 1, tokens, embedding_dim, 2, 2)


def test_apply_dype_to_flux2_installs_embedder_and_wrapper():
    patcher = _FluxModelPatcher()

    patched = qwen_spatial.apply_dype_to_flux2(
        model=patcher,
        width=2048,
        height=2048,
        method="yarn",
        enable_dype=True,
        dype_exponent=2.0,
        base_width=1024,
        base_height=1024,
        base_shift=2.02,
        max_shift=2.35,
        auto_detect=False,
        editing_strength=1.0,
        editing_mode="full",
    )

    embedder = patched.model.diffusion_model.pe_embedder
    assert isinstance(embedder, qwen_spatial.QwenSpatialPosEmbed)
    assert hasattr(patched, "_wrapper") and patched._wrapper is not None
    assert "model_sampling" in patched._object_patches
    wrapper = patched._object_patches["model_sampling"]
    assert isinstance(wrapper, qwen_spatial._DyPEModelSampling)


def test_apply_dype_to_flux2_rejects_non_flux_embedder():
    patcher = _FluxModelPatcher(embedder=_FluxEmbedder(axes_dim=[16, 56, 56]))

    with pytest.raises(ValueError, match="expected 4 axes"):
        qwen_spatial.apply_dype_to_flux2(
            model=patcher,
            width=1024,
            height=1024,
            method="yarn",
            enable_dype=True,
            dype_exponent=2.0,
            base_width=1024,
            base_height=1024,
            base_shift=2.02,
            max_shift=2.35,
            auto_detect=False,
            editing_strength=1.0,
            editing_mode="full",
        )


def test_dype_flux2_node_executes_and_logs(caplog: pytest.LogCaptureFixture):
    patcher = _FluxModelPatcher()
    node = DyPEFlux2()
    caplog.set_level(logging.INFO, logger="src.dype_flux2")

    result = node.apply(
        model=patcher,
        width=1024,
        height=1024,
        auto_detect=False,
        base_width=1024,
        base_height=1024,
        method="yarn",
        enable_dype=True,
        dype_exponent=2.0,
        base_shift=2.02,
        max_shift=2.35,
        editing_strength=1.0,
        editing_mode="adaptive",
    )

    assert isinstance(result, tuple) and hasattr(result[0], "model")
    node_messages = [record.message for record in caplog.records if record.name == "src.dype_flux2"]
    assert any("DyPE_Flux2: requested patch" in message for message in node_messages)
