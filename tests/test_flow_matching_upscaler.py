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

    def test_stage_falls_back_to_streaming_on_oom(self):
        stage_node = fm_upscaler.FlowMatchingStage()
        latent = {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float32)}

        call_count = {"value": 0}
        recorded_controls = []

        def fake_execute_controls(operation, *, attention_budget_mb, enable_low_vram):
            recorded_controls.append((attention_budget_mb, enable_low_vram))
            return operation()

        def fake_run_sampler(**kwargs):
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise RuntimeError("CUDA out of memory")
            payload = kwargs["latent_template"]
            result = payload.copy()
            result["samples"] = torch.zeros_like(payload["samples"])
            return result

        with mock.patch.object(fm_upscaler, "_execute_with_memory_controls", side_effect=fake_execute_controls):
            with mock.patch.object(fm_upscaler, "run_sampler", side_effect=fake_run_sampler):
                with mock.patch.object(fm_upscaler, "apply_flow_renoise", side_effect=lambda x, *_: x):
                    output_latent, presampler_latent, next_seed, *_ = stage_node.execute(
                        model=object(),
                        positive=[],
                        negative=[],
                        latent=latent,
                        seed=123,
                        steps=2,
                        cfg=1.0,
                        sampler_name=fm_upscaler.FlowMatchingStage._SAMPLERS[0],
                        scheduler=fm_upscaler.FlowMatchingStage._SCHEDULERS[0],
                        scale_factor=1.0,
                        noise_ratio=0.0,
                        skip_blend=0.1,
                        denoise=1.0,
                        upscale_method="nearest-exact",
                        enable_dilated_sampling="disable",
                        reduce_memory_use="disable",
                    )

        self.assertEqual(call_count["value"], 3)
        self.assertEqual(recorded_controls[0], (0.0, False))
        self.assertEqual(recorded_controls[-1], (fm_upscaler._STREAMING_ATTENTION_BUDGET_MB, True))
        mask64 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(next_seed, (123 + fm_upscaler._SEED_STRIDE) & mask64)
        self.assertEqual(tuple(output_latent["samples"].shape[-2:]), (8, 8))
        self.assertEqual(tuple(presampler_latent["samples"].shape[-2:]), (8, 8))

    def test_channel_stats_logging_emits_debug_diagnostics(self):
        tensor = torch.arange(0, 64, dtype=torch.float32).reshape(1, 4, 4, 4)

        with self.assertLogs(fm_upscaler.logger, level="DEBUG") as captured:
            fm_upscaler._log_channel_stats("diagnostic-test", tensor)

        self.assertTrue(
            any("diagnostic-test channel stats" in message for message in captured.output),
            msg=f"Expected diagnostic log entry, got: {captured.output}",
        )

    def test_render_channel_stats_image_monotonic(self):
        means = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        stds = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

        image, layout = fm_upscaler._render_channel_stats_image(means, stds, limit=3, height=128)

        self.assertEqual(tuple(image.shape[-1:]), (3,))
        self.assertEqual(image.shape[0], 1)
        self.assertTrue(torch.all(image >= 0.0))
        self.assertTrue(torch.all(image <= 1.0))

        mean_color = torch.from_numpy(fm_upscaler._MEAN_BAR_COLOR)
        std_color = torch.from_numpy(fm_upscaler._STD_BAR_COLOR)

        mean_start, mean_height = layout["mean_section"]
        std_start, std_height = layout["std_section"]

        mean_heights = []
        std_heights = []
        for bar in layout["bar_ranges"]:
            x0, x1 = bar["x_range"]
            x_center = (x0 + x1 - 1) // 2

            mean_column = image[0, mean_start:mean_start + mean_height, x_center, :]
            std_column = image[0, std_start:std_start + std_height, x_center, :]

            mean_mask = torch.linalg.norm(mean_column - mean_color, dim=-1) < 1e-3
            std_mask = torch.linalg.norm(std_column - std_color, dim=-1) < 1e-3

            mean_heights.append(int(mean_mask.sum().item()))
            std_heights.append(int(std_mask.sum().item()))

        self.assertGreater(mean_heights[1], mean_heights[0])
        self.assertGreater(mean_heights[2], mean_heights[1])
        self.assertGreater(std_heights[1], std_heights[0])
        self.assertGreater(std_heights[2], std_heights[1])

    def test_latent_channel_stats_preview_outputs_image(self):
        samples = torch.stack(
            [
                torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32).reshape(1, 4, 4),
                torch.linspace(0.0, 0.5, steps=16, dtype=torch.float32).reshape(1, 4, 4),
                torch.linspace(-0.5, 0.5, steps=16, dtype=torch.float32).reshape(1, 4, 4),
                torch.linspace(0.25, 0.75, steps=16, dtype=torch.float32).reshape(1, 4, 4),
            ],
            dim=1,
        )
        latent = {"samples": samples}

        preview_node = fm_upscaler.LatentChannelStatsPreview()
        image_tensor, = preview_node.render(latent=latent, channel_limit=4, height=128)

        self.assertEqual(image_tensor.shape[0], 1)
        self.assertGreater(image_tensor.shape[1], 0)
        self.assertGreater(image_tensor.shape[2], 0)
        self.assertEqual(image_tensor.shape[3], 3)
        self.assertTrue(torch.all(image_tensor >= 0.0))
        self.assertTrue(torch.all(image_tensor <= 1.0))
