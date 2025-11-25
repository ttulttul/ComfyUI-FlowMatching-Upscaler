import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_test_stubs():
    """Install lightweight stubs for comfy and nodes before importing the module under test."""
    if "comfy" not in sys.modules:
        comfy_module = types.ModuleType("comfy")
        samplers_module = types.ModuleType("comfy.samplers")
        utils_module = types.ModuleType("comfy.utils")
        sampling_module = types.ModuleType("comfy.model_sampling")
        model_patcher_module = types.ModuleType("comfy.model_patcher")

        class _SamplerStub:
            SAMPLERS = ("test_sampler",)
            SCHEDULERS = ("test_scheduler",)

        samplers_module.KSampler = _SamplerStub

        class _ModelSamplingFlux:
            pass

        def _flux_time_shift(*_args, **_kwargs):
            return 0.0

        sampling_module.ModelSamplingFlux = _ModelSamplingFlux
        sampling_module.flux_time_shift = _flux_time_shift

        class _ModelPatcherStub:
            def __init__(self):
                self.model = types.SimpleNamespace()

            def clone(self):
                return self

        model_patcher_module.ModelPatcher = _ModelPatcherStub

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
        comfy_module.model_sampling = sampling_module
        comfy_module.model_patcher = model_patcher_module

        sys.modules["comfy"] = comfy_module
        sys.modules["comfy.samplers"] = samplers_module
        sys.modules["comfy.utils"] = utils_module
        sys.modules["comfy.model_sampling"] = sampling_module
        sys.modules["comfy.model_patcher"] = model_patcher_module

    if "nodes" not in sys.modules:
        nodes_module = types.ModuleType("nodes")

        def _placeholder_common_ksampler(**_kwargs):
            raise RuntimeError("common_ksampler stub should be patched during tests.")

        nodes_module.common_ksampler = _placeholder_common_ksampler
        sys.modules["nodes"] = nodes_module


_install_test_stubs()

import src.flow_matching_upscaler as fm_upscaler  # noqa: E402
from src.flow_matching_upscaler import FlowMatchingProgressiveUpscaler  # noqa: E402


class TorchLerpEquivalenceTests(unittest.TestCase):
    """Tests to verify torch.lerp produces equivalent results to manual blending."""

    def test_lerp_equivalent_to_manual_blend(self):
        """Verify torch.lerp(a, b, weight) == a + weight * (b - a) == b * weight + a * (1 - weight)."""
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        weight = 0.3

        # torch.lerp(a, b, weight) = a + weight * (b - a)
        lerp_result = torch.lerp(a, b, weight)

        # Manual: b * weight + a * (1 - weight)
        manual_result = b * weight + a * (1 - weight)

        self.assertTrue(torch.allclose(lerp_result, manual_result, atol=1e-6))

    def test_apply_flow_renoise_uses_lerp(self):
        """Verify apply_flow_renoise produces consistent results with lerp."""
        latent = torch.ones(1, 4, 8, 8)
        noise_level = 0.5
        seed = 42

        # Call twice with same seed should produce same result
        result1 = fm_upscaler.apply_flow_renoise(latent, noise_level, seed)
        result2 = fm_upscaler.apply_flow_renoise(latent, noise_level, seed)

        self.assertTrue(torch.allclose(result1, result2))
        # With 50% noise, result should be between pure latent and pure noise
        self.assertFalse(torch.allclose(result1, latent))

    def test_apply_flow_renoise_zero_noise(self):
        """Verify zero noise returns original latent unchanged."""
        latent = torch.randn(1, 4, 8, 8)
        result = fm_upscaler.apply_flow_renoise(latent, 0.0, 123)
        self.assertTrue(torch.equal(result, latent))

    def test_apply_flow_renoise_full_noise(self):
        """Verify full noise level returns pure noise."""
        latent = torch.ones(1, 4, 8, 8)
        result = fm_upscaler.apply_flow_renoise(latent, 1.0, 456)
        # Result should be different from input (pure noise)
        self.assertFalse(torch.allclose(result, latent))


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

    def test_progressive_upscale_rescales_noise_mask(self):
        latent = {
            "samples": torch.ones((1, 4, 8, 8), dtype=torch.float32),
            "noise_mask": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        }
        mask_shapes = []

        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            self.assertIn("noise_mask", latent_payload)
            mask_shapes.append(tuple(latent_payload["noise_mask"].shape[-2:]))
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            with mock.patch.object(fm_upscaler, "apply_flow_renoise", side_effect=lambda tensor, *_: tensor):
                output_latent, *_ = self.node.progressive_upscale(
                    model=object(),
                    positive=[],
                    negative=[],
                    latent=latent,
                    seed=0,
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

        self.assertEqual(mask_shapes, [(16, 16), (32, 32)])
        self.assertIn("noise_mask", output_latent)
        self.assertEqual(tuple(output_latent["noise_mask"].shape[-2:]), (32, 32))

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
        latent = {
            "samples": torch.ones((1, 4, 8, 8), dtype=torch.float32),
            "noise_mask": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        }
        recorded_mask_shapes = []

        def fake_common_ksampler(**kwargs):
            template = kwargs["latent"]
            self.assertIn("noise_mask", template)
            recorded_mask_shapes.append(tuple(template["noise_mask"].shape[-2:]))
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
        self.assertIn("noise_mask", output_latent)
        self.assertEqual(tuple(output_latent["noise_mask"].shape[-2:]), (16, 16))
        self.assertEqual(recorded_mask_shapes, [(16, 16)])
        self.assertIn("noise_mask", presampler_latent)
        self.assertEqual(tuple(presampler_latent["noise_mask"].shape[-2:]), (16, 16))
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
        background = torch.from_numpy(fm_upscaler._BACKGROUND_COLOR)

        mean_start, mean_height = layout["mean_section"]
        std_start, std_height = layout["std_section"]
        channel_label_y = layout["channel_label_y"]

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

            label_slice = image[
                0,
                channel_label_y:channel_label_y + fm_upscaler._FONT_HEIGHT,
                max(0, x_center - 2):min(image.shape[2], x_center + 3),
                :,
            ]
            background_diff = torch.abs(label_slice - background.view(1, 1, -1)).max()
            self.assertGreater(background_diff.item(), 1e-3)

        self.assertGreater(mean_heights[1], mean_heights[0])
        self.assertGreater(mean_heights[2], mean_heights[1])
        self.assertGreater(std_heights[1], std_heights[0])
        self.assertGreater(std_heights[2], std_heights[1])

        mean_label_region = image[0, mean_start:mean_start + mean_height, 4:4 + fm_upscaler._FONT_WIDTH, :]
        std_label_region = image[0, std_start:std_start + std_height, 4:4 + fm_upscaler._FONT_WIDTH, :]
        self.assertGreater(torch.abs(mean_label_region - background.view(1, 1, -1)).max().item(), 1e-3)
        self.assertGreater(torch.abs(std_label_region - background.view(1, 1, -1)).max().item(), 1e-3)

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


class DilatedBlendMethodTests(unittest.TestCase):
    """Tests for the dilated sampling blend methods."""

    def setUp(self):
        # Create test tensors
        self.original = torch.randn(1, 4, 32, 32, dtype=torch.float32)
        self.dilated = torch.randn(1, 4, 32, 32, dtype=torch.float32)
        self.downscale_factor = 2.0

    def test_linear_blend_is_torch_lerp(self):
        """Verify linear blend matches torch.lerp."""
        blend = 0.5
        result = fm_upscaler._apply_dilated_blend(
            original=self.original,
            dilated=self.dilated,
            blend=blend,
            method="linear",
            downscale_factor=self.downscale_factor,
        )
        expected = torch.lerp(self.original, self.dilated, blend)
        self.assertTrue(torch.allclose(result, expected))

    def test_frequency_blend_preserves_shape(self):
        """Verify frequency blend returns correct shape."""
        result = fm_upscaler._frequency_blend(
            self.original, self.dilated, blend=0.5, downscale_factor=self.downscale_factor
        )
        self.assertEqual(result.shape, self.original.shape)
        self.assertEqual(result.dtype, self.original.dtype)

    def test_frequency_blend_zero_returns_original(self):
        """Verify frequency blend with blend=0 returns original."""
        result = fm_upscaler._frequency_blend(
            self.original, self.dilated, blend=0.0, downscale_factor=self.downscale_factor
        )
        self.assertTrue(torch.equal(result, self.original))

    def test_frequency_blend_one_returns_dilated(self):
        """Verify frequency blend with blend=1 returns dilated."""
        result = fm_upscaler._frequency_blend(
            self.original, self.dilated, blend=1.0, downscale_factor=self.downscale_factor
        )
        self.assertTrue(torch.equal(result, self.dilated))

    def test_laplacian_blend_preserves_shape(self):
        """Verify Laplacian pyramid blend returns correct shape."""
        result = fm_upscaler._laplacian_pyramid_blend(
            self.original, self.dilated, blend=0.5
        )
        self.assertEqual(result.shape, self.original.shape)

    def test_laplacian_blend_zero_returns_original(self):
        """Verify Laplacian blend with blend=0 returns original."""
        result = fm_upscaler._laplacian_pyramid_blend(
            self.original, self.dilated, blend=0.0
        )
        self.assertTrue(torch.equal(result, self.original))

    def test_laplacian_blend_one_returns_dilated(self):
        """Verify Laplacian blend with blend=1 returns dilated."""
        result = fm_upscaler._laplacian_pyramid_blend(
            self.original, self.dilated, blend=1.0
        )
        self.assertTrue(torch.equal(result, self.dilated))

    def test_laplacian_blend_small_image_fallback(self):
        """Verify Laplacian blend falls back to linear for small images."""
        small_orig = torch.randn(1, 4, 4, 4, dtype=torch.float32)
        small_dilated = torch.randn(1, 4, 4, 4, dtype=torch.float32)
        result = fm_upscaler._laplacian_pyramid_blend(
            small_orig, small_dilated, blend=0.5
        )
        self.assertEqual(result.shape, small_orig.shape)

    def test_gaussian_blend_preserves_shape(self):
        """Verify Gaussian-weighted blend returns correct shape."""
        result = fm_upscaler._gaussian_weighted_blend(
            self.original, self.dilated, blend=0.5, blur_sigma=2.0
        )
        self.assertEqual(result.shape, self.original.shape)

    def test_gaussian_blend_zero_returns_original(self):
        """Verify Gaussian blend with blend=0 returns original."""
        result = fm_upscaler._gaussian_weighted_blend(
            self.original, self.dilated, blend=0.0, blur_sigma=2.0
        )
        self.assertTrue(torch.equal(result, self.original))

    def test_gaussian_blend_smooths_difference(self):
        """Verify Gaussian blend produces smoothed result."""
        # Use a pattern that would show grid artifacts without smoothing
        grid_dilated = self.original.clone()
        grid_dilated[:, :, ::2, ::2] += 1.0  # Grid pattern

        result = fm_upscaler._gaussian_weighted_blend(
            self.original, grid_dilated, blend=0.5, blur_sigma=2.0
        )
        # Result should be different from both inputs
        self.assertFalse(torch.equal(result, self.original))
        self.assertFalse(torch.equal(result, grid_dilated))

    def test_apply_dilated_blend_unknown_method_fallback(self):
        """Verify unknown method falls back to linear."""
        result = fm_upscaler._apply_dilated_blend(
            original=self.original,
            dilated=self.dilated,
            blend=0.5,
            method="unknown_method",
            downscale_factor=self.downscale_factor,
        )
        expected = torch.lerp(self.original, self.dilated, 0.5)
        self.assertTrue(torch.allclose(result, expected))

    def test_all_blend_methods_produce_valid_output(self):
        """Verify all blend methods produce valid (non-NaN, non-Inf) output."""
        for method in fm_upscaler._DILATED_BLEND_METHODS:
            with self.subTest(method=method):
                result = fm_upscaler._apply_dilated_blend(
                    original=self.original,
                    dilated=self.dilated,
                    blend=0.5,
                    method=method,
                    downscale_factor=self.downscale_factor,
                )
                self.assertFalse(torch.any(torch.isnan(result)), f"{method} produced NaN")
                self.assertFalse(torch.any(torch.isinf(result)), f"{method} produced Inf")

    def test_dilated_refinement_accepts_blend_method(self):
        """Verify dilated_refinement can be called with blend_method parameter."""
        latent = {"samples": torch.ones((1, 4, 16, 16), dtype=torch.float32)}

        def fake_common_ksampler(**kwargs):
            latent_payload = kwargs["latent"]
            result = latent_payload.copy()
            result["samples"] = torch.zeros_like(latent_payload["samples"])
            return (result,)

        with mock.patch.object(fm_upscaler, "common_ksampler", new=fake_common_ksampler):
            for method in fm_upscaler._DILATED_BLEND_METHODS:
                with self.subTest(method=method):
                    result = fm_upscaler.dilated_refinement(
                        model=object(),
                        positive=[],
                        negative=[],
                        sampler_name="test_sampler",
                        scheduler="test_scheduler",
                        cfg=1.0,
                        steps=2,
                        seed=42,
                        denoise=1.0,
                        base_latent=latent.copy(),
                        downscale_factor=2.0,
                        blend=0.5,
                        blend_method=method,
                    )
                    self.assertEqual(result.shape, latent["samples"].shape)
