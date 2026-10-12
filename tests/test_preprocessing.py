"""Tests for image preprocessing pipeline."""

import numpy as np
import torch
from PIL import Image

from src.config import ModelSettings
from src.preprocessing import load_image_as_numpy, preprocess_chest_xray


def test_load_image_as_numpy_grayscale(valid_png_bytes):
    arr = load_image_as_numpy(valid_png_bytes)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (256, 256)
    assert arr.ndim == 2


def test_load_image_as_numpy_rgb_conversion(valid_rgb_jpeg_bytes):
    # RGB must convert to 2D single channel luminance
    arr = load_image_as_numpy(valid_rgb_jpeg_bytes)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (300, 300)
    assert arr.ndim == 2


def test_preprocess_chest_xray_tensor_properties(valid_png_bytes):
    settings = ModelSettings(
        input_resolution=(224, 224),
        raw_max_pixel_val=255.0,
        device="cpu",
    )
    tensor = preprocess_chest_xray(valid_png_bytes, settings)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 224, 224)
    assert tensor.dtype == torch.float32
    # Check that intensity values have been scaled into [-1024, 1024] range
    assert tensor.min() >= -1025.0
    assert tensor.max() <= 1025.0


def test_preprocess_from_pil_image():
    pil_img = Image.new("L", (512, 512), color=100)
    settings = ModelSettings(input_resolution=(224, 224), device="cpu")
    tensor = preprocess_chest_xray(pil_img, settings)
    assert tensor.shape == (1, 1, 224, 224)
