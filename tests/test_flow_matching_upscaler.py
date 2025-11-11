import sys
import types
import unittest
from unittest import mock

import torch


def _install_test_stubs():
    """Install lightweight stubs for comfy and nodes before importing the module under test."""
    if "comfy" not in sys.modules:
        comfy_module = types.ModuleType("comfy")
        samplers_module = types.ModuleType("comfy.samplers")
        utils_module = types.ModuleType("comfy.utils")

        class _SamplerStub:
            SAMPLERS = ("test_sampler",)
            SCHEDULERS = ("test_scheduler",)

        samplers_module.KSampler = _SamplerStub

        def common_upscale(samples, width, height, upscale_method, crop):
            orig_shape = samples.shape
            tensor = samples

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

        comfy_module.samplers = samplers_module
        comfy_module.utils = utils_module

        sys.modules["comfy"] = comfy_module
        sys.modules["comfy.samplers"] = samplers_module
        sys.modules["comfy.utils"] = utils_module

    if "nodes" not in sys.modules:
        nodes_module = types.ModuleType("nodes")

        def _placeholder_common_ksampler(**_kwargs):
            raise RuntimeError("common_ksampler stub should be patched during tests.")

        nodes_module.common_ksampler = _placeholder_common_ksampler
        sys.modules["nodes"] = nodes_module


_install_test_stubs()

import src.flow_matching_upscaler as fm_upscaler  # noqa: E402
from src.flow_matching_upscaler import FlowMatchingProgressiveUpscaler  # noqa: E402


class FlowMatchingUpscalerTests(unittest.TestCase):
    def setUp(self):
        self.node = FlowMatchingProgressiveUpscaler()
        self.base_latent = {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float32)}
        self.default_sampler = FlowMatchingProgressiveUpscaler._SAMPLERS[0]
        self.default_scheduler = FlowMatchingProgressiveUpscaler._SCHEDULERS[0]

    def test_progressive_upscale_increases_size(self):
        captured_shapes = []

        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            captured_shapes.append(tuple(latent_payload["samples"].shape[-2:]))
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            output_latent, = self.node.progressive_upscale(
                model=object(),
                positive=[],
                negative=[],
                latent=self.base_latent,
                seed=123,
                steps_per_stage=1,
                cfg=1.0,
                sampler_name=self.default_sampler,
                scheduler=self.default_scheduler,
                total_scale=4.0,
                stages=2,
                renoise_start=0.0,
                renoise_end=0.0,
                skip_blend_start=0.0,
                skip_blend_end=0.0,
                upscale_method="nearest-exact",
                noise_schedule_override="0.0,0.0",
                skip_schedule_override="0.0,0.0",
                enable_dilated_sampling="disable",
            )

        self.assertEqual(captured_shapes, [(16, 16), (32, 32)])
        self.assertEqual(tuple(output_latent["samples"].shape[-2:]), (32, 32))

    def test_skip_blend_override_controls_result(self):
        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.full_like(latent_payload["samples"], 2.0)
            return (result,)

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            output_latent, = self.node.progressive_upscale(
                model=object(),
                positive=[],
                negative=[],
                latent=self.base_latent,
                seed=7,
                steps_per_stage=2,
                cfg=1.0,
                sampler_name=self.default_sampler,
                scheduler=self.default_scheduler,
                total_scale=4.0,
                stages=2,
                renoise_start=0.0,
                renoise_end=0.0,
                skip_blend_start=1.0,
                skip_blend_end=0.0,
                upscale_method="nearest-exact",
                noise_schedule_override="0.0,0.0",
                skip_schedule_override="1.0,0.0",
                enable_dilated_sampling="disable",
            )

        expected = torch.full_like(output_latent["samples"], 2.0)
        self.assertTrue(torch.allclose(output_latent["samples"], expected))

    def test_dilated_refinement_handles_extra_dimensions(self):
        latent = {"samples": torch.ones((1, 4, 2, 8, 8), dtype=torch.float32)}

        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            output_latent, = self.node.progressive_upscale(
                model=object(),
                positive=[],
                negative=[],
                latent=latent,
                seed=5,
                steps_per_stage=1,
                cfg=1.0,
                sampler_name=self.default_sampler,
                scheduler=self.default_scheduler,
                total_scale=2.0,
                stages=1,
                renoise_start=0.0,
                renoise_end=0.0,
                skip_blend_start=0.5,
                skip_blend_end=0.5,
                upscale_method="nearest-exact",
                noise_schedule_override="0.0",
                skip_schedule_override="0.5",
                enable_dilated_sampling="enable",
                dilated_downscale=2.0,
                dilated_blend=0.5,
            )

        self.assertEqual(tuple(output_latent["samples"].shape), (1, 4, 2, 16, 16))


if __name__ == "__main__":
    unittest.main()
