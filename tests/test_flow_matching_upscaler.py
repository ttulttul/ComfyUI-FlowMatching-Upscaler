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

        call_log = []

        def common_upscale(samples, width, height, upscale_method, crop):
            call_log.append(
                {
                    "method": upscale_method,
                    "shape": tuple(samples.shape),
                    "target": (height, width),
                }
            )
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
        utils_module._common_upscale_calls = call_log

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
from src.flow_matching_upscaler import FlowMatchingTiledStage  # noqa: E402


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

        model_obj = object()
        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            output_latent, next_seed, out_model, out_positive, out_negative = self.node.progressive_upscale(
                model=model_obj,
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
        mask64 = 0xFFFFFFFFFFFFFFFF
        expected_next_seed = (123 + 2 * fm_upscaler._SEED_STRIDE) & mask64
        self.assertEqual(next_seed, expected_next_seed)
        self.assertIs(out_model, model_obj)
        self.assertEqual(out_positive, [])
        self.assertEqual(out_negative, [])

    def test_skip_blend_override_controls_result(self):
        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.full_like(latent_payload["samples"], 2.0)
            return (result,)

        model_obj = object()
        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            output_latent, next_seed, out_model, out_positive, out_negative = self.node.progressive_upscale(
                model=model_obj,
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
        mask64 = 0xFFFFFFFFFFFFFFFF
        expected_next_seed = (7 + 2 * fm_upscaler._SEED_STRIDE) & mask64
        self.assertEqual(next_seed, expected_next_seed)
        self.assertIs(out_model, model_obj)
        self.assertEqual(out_positive, [])
        self.assertEqual(out_negative, [])

    def test_dilated_refinement_handles_extra_dimensions(self):
        latent = {"samples": torch.ones((1, 4, 2, 8, 8), dtype=torch.float32)}

        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        model_obj = object()
        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            output_latent, next_seed, out_model, out_positive, out_negative = self.node.progressive_upscale(
                model=model_obj,
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
        mask64 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(next_seed, (5 + fm_upscaler._SEED_STRIDE) & mask64)
        self.assertIs(out_model, model_obj)
        self.assertEqual(out_positive, [])
        self.assertEqual(out_negative, [])

    def test_lanczos_falls_back_for_high_channel_latent(self):
        latent = {"samples": torch.ones((1, 16, 8, 8), dtype=torch.float32)}

        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        utils_module = fm_upscaler.comfy.utils
        utils_module._common_upscale_calls.clear()

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            self.node.progressive_upscale(
                model=object(),
                positive=[],
                negative=[],
                latent=latent,
                seed=11,
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
                upscale_method="lanczos",
                noise_schedule_override="0.0",
                skip_schedule_override="0.5",
                enable_dilated_sampling="disable",
            )

        first_call = utils_module._common_upscale_calls[0]
        self.assertEqual(first_call["method"], "bicubic")

    def test_stage_seeds_are_perturbed(self):
        latent = {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float32)}
        recorded_seeds = []

        def fake_common_ksampler(**kwargs):
            recorded_seeds.append(kwargs["seed"])
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            self.node.progressive_upscale(
                model=object(),
                positive=[],
                negative=[],
                latent=latent,
                seed=1234,
                steps_per_stage=1,
                cfg=1.0,
                sampler_name=self.default_sampler,
                scheduler=self.default_scheduler,
                total_scale=4.0,
                stages=3,
                renoise_start=0.0,
                renoise_end=0.0,
                skip_blend_start=0.5,
                skip_blend_end=0.5,
                upscale_method="nearest-exact",
                noise_schedule_override="0.0,0.0,0.0",
                skip_schedule_override="0.5,0.5,0.5",
                enable_dilated_sampling="disable",
            )

        mask64 = 0xFFFFFFFFFFFFFFFF
        expected = [
            1234,
            (1234 + fm_upscaler._SEED_STRIDE) & mask64,
            (1234 + 2 * fm_upscaler._SEED_STRIDE) & mask64,
        ]
        self.assertEqual(recorded_seeds, expected)

    def test_cleanup_stage_uses_custom_denoise_and_noise(self):
        latent = {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float32)}
        recorded_seeds = []
        recorded_denoise = []
        noise_records = []

        def fake_common_ksampler(**kwargs):
            recorded_seeds.append(kwargs["seed"])
            recorded_denoise.append(kwargs["denoise"])
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        def fake_renoise(latent_tensor, noise_level, seed):
            noise_records.append(noise_level)
            return latent_tensor

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            with mock.patch.object(
                fm_upscaler,
                "apply_flow_renoise",
                side_effect=fake_renoise,
            ):
                self.node.progressive_upscale(
                    model=object(),
                    positive=[],
                    negative=[],
                    latent=latent,
                    seed=4321,
                    steps_per_stage=2,
                    cfg=1.0,
                    sampler_name=self.default_sampler,
                    scheduler=self.default_scheduler,
                    total_scale=2.0,
                    stages=1,
                    renoise_start=0.05,
                    renoise_end=0.05,
                    skip_blend_start=0.2,
                    skip_blend_end=0.05,
                    upscale_method="nearest-exact",
                    noise_schedule_override="0.05",
                    skip_schedule_override="0.2",
                    enable_dilated_sampling="disable",
                    cleanup_stage="enable",
                    cleanup_noise=0.15,
                    cleanup_denoise=0.35,
                    denoise=0.6,
                )

        mask64 = 0xFFFFFFFFFFFFFFFF
        expected_seeds = [
            4321,
            (4321 + fm_upscaler._SEED_STRIDE) & mask64,
        ]
        self.assertEqual(recorded_seeds, expected_seeds)
        self.assertEqual(recorded_denoise, [0.6, 0.35])
        self.assertEqual(noise_records, [0.05, 0.15])

    def test_stage_node_rescales_latent(self):
        stage_node = fm_upscaler.FlowMatchingStage()
        latent = {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float32)}

        def fake_common_ksampler(**kwargs):
            template = kwargs["latent"]
            result = template.copy()
            result["samples"] = torch.full_like(template["samples"], 2.0)
            return (result,)

        model_obj = object()
        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            with mock.patch.object(fm_upscaler, "apply_flow_renoise", side_effect=lambda x, *_: x):
                output_latent, presampler_latent, next_seed, out_model, out_positive, out_negative = stage_node.execute(
                    model=model_obj,
                    positive=[],
                    negative=[],
                    latent=latent,
                    seed=99,
                    steps=2,
                    cfg=3.0,
                    sampler_name=fm_upscaler.FlowMatchingStage._SAMPLERS[0],
                    scheduler=fm_upscaler.FlowMatchingStage._SCHEDULERS[0],
                    scale_factor=2.0,
                    noise_ratio=0.0,
                    skip_blend=0.25,
                    denoise=0.5,
                    upscale_method="nearest-exact",
                    enable_dilated_sampling="disable",
                )

        self.assertEqual(tuple(output_latent["samples"].shape[-2:]), (16, 16))
        expected_value = 0.25 * 1.0 + 0.75 * 2.0
        self.assertTrue(torch.allclose(output_latent["samples"], torch.full_like(output_latent["samples"], expected_value)))
        mask64 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(next_seed, (99 + fm_upscaler._SEED_STRIDE) & mask64)
        self.assertIs(out_model, model_obj)
        self.assertEqual(out_positive, [])
        self.assertEqual(out_negative, [])
        self.assertEqual(tuple(presampler_latent["samples"].shape[-2:]), (16, 16))


class FlowMatchingTiledStageTests(unittest.TestCase):
    def setUp(self):
        self.node = FlowMatchingTiledStage()
        self.default_sampler = FlowMatchingProgressiveUpscaler._SAMPLERS[0]
        self.default_scheduler = FlowMatchingProgressiveUpscaler._SCHEDULERS[0]

    def test_half_tile_size_splits_landscape_in_two(self):
        base_latent = {"samples": torch.ones((1, 4, 8, 16), dtype=torch.float32)}
        call_shapes = []

        def fake_run_sampler(**kwargs):
            latent_payload = kwargs["latent_template"]
            tile_samples = latent_payload["samples"]
            call_shapes.append(tuple(tile_samples.shape[-2:]))
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(tile_samples)
            return result

        with mock.patch.object(fm_upscaler, "run_sampler", new=fake_run_sampler):
            output, presampler, next_seed, *_ = self.node.execute(
                model=object(),
                positive=[],
                negative=[],
                latent=base_latent,
                seed=21,
                steps=3,
                cfg=2.0,
                sampler_name=self.default_sampler,
                scheduler=self.default_scheduler,
                scale_factor=1.0,
                noise_ratio=0.0,
                skip_blend=0.0,
                denoise=1.0,
                upscale_method="nearest-exact",
                tile_size=0.5,
                enable_dilated_sampling="disable",
                reduce_memory_use="enable",
                dilated_downscale=2.0,
                dilated_blend=0.25,
            )

        self.assertEqual(call_shapes, [(8, 8), (8, 8)])
        self.assertEqual(tuple(output["samples"].shape[-2:]), (8, 16))
        self.assertEqual(tuple(presampler["samples"].shape[-2:]), (8, 16))
        mask64 = 0xFFFFFFFFFFFFFFFF
        expected_seed = (21 + 2 * fm_upscaler._SEED_STRIDE) & mask64
        self.assertEqual(next_seed, expected_seed)

    def test_quadrant_split_for_quarter_tile_size(self):
        base_latent = {"samples": torch.ones((1, 4, 16, 8), dtype=torch.float32)}
        observed_tiles = []

        def fake_run_sampler(**kwargs):
            latent_payload = kwargs["latent_template"]
            samples = latent_payload["samples"]
            observed_tiles.append(tuple(samples.shape[-2:]))
            result = latent_payload.copy()
            result["samples"] = torch.full_like(samples, 5.0)
            return result

        with mock.patch.object(fm_upscaler, "run_sampler", new=fake_run_sampler):
            output, presampler, next_seed, *_ = self.node.execute(
                model=object(),
                positive=[],
                negative=[],
                latent=base_latent,
                seed=99,
                steps=1,
                cfg=1.0,
                sampler_name=self.default_sampler,
                scheduler=self.default_scheduler,
                scale_factor=1.0,
                noise_ratio=0.0,
                skip_blend=0.5,
                denoise=1.0,
                upscale_method="nearest-exact",
                tile_size=0.25,
                enable_dilated_sampling="disable",
                reduce_memory_use="enable",
                dilated_downscale=2.0,
                dilated_blend=0.25,
            )

        self.assertEqual(observed_tiles, [(8, 4), (8, 4), (8, 4), (8, 4)])
        self.assertTrue(torch.allclose(output["samples"], torch.full_like(output["samples"], 2.5)))
        self.assertEqual(tuple(presampler["samples"].shape[-2:]), (16, 8))
        mask64 = 0xFFFFFFFFFFFFFFFF
        expected_seed = (99 + 4 * fm_upscaler._SEED_STRIDE) & mask64
        self.assertEqual(next_seed, expected_seed)


if __name__ == "__main__":
    unittest.main()
