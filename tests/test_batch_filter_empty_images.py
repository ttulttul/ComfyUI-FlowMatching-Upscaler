import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_filter_empty_images import BatchFilterEmptyImages  # noqa: E402


def test_filters_exact_zero_images():
    node = BatchFilterEmptyImages()
    images = torch.zeros((3, 2, 2, 3), dtype=torch.float32)
    images[1, 0, 0, 0] = 0.5

    (out,) = node.filter(images=images, epsilon=0.0)

    assert tuple(out.shape) == (1, 2, 2, 3)
    assert torch.equal(out[0], images[1])


def test_filters_near_zero_using_epsilon():
    node = BatchFilterEmptyImages()
    images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
    images[0].fill_(5e-5)
    images[1].fill_(-2e-4)

    (out,) = node.filter(images=images, epsilon=1e-4)

    assert tuple(out.shape) == (1, 2, 2, 3)
    assert torch.allclose(out[0], images[1])


def test_accepts_3d_image_tensor_and_returns_batched_output():
    node = BatchFilterEmptyImages()
    image = torch.zeros((2, 2, 3), dtype=torch.float32)

    (out,) = node.filter(images=image, epsilon=0.0)

    assert out.ndim == 4
    assert tuple(out.shape) == (0, 2, 2, 3)


def test_rejects_non_tensor_inputs():
    node = BatchFilterEmptyImages()

    with pytest.raises(ValueError, match="torch\\.Tensor"):
        node.filter(images=[0], epsilon=0.0)
