"""Pytest configuration and shared test fixtures for Medical Vision AI."""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.config import AppConfig, ValidationSettings, ModelSettings, InferenceSettings


@pytest.fixture
def default_config() -> AppConfig:
    """Return default application configuration."""
    return AppConfig(
        model=ModelSettings(
            weights="densenet121-res224-all",
            device="cpu",
        ),
        validation=ValidationSettings(
            max_file_size_mb=10,
            allowed_extensions=[".png", ".jpg", ".jpeg"],
            min_dimension=64,
            max_dimension=4096,
        ),
        inference=InferenceSettings(
            operating_point_threshold=0.5,
            top_k_findings=5,
        ),
    )


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate in-memory valid 256x256 8-bit grayscale PNG image."""
    arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_rgb_jpeg_bytes() -> bytes:
    """Generate in-memory valid 300x300 RGB JPEG image."""
    arr = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """Generate corrupted non-image bytes."""
    return b"\x89PNG\r\n\x1a\nCORRUPTED_GARBAGE_PAYLOAD_NOT_A_VALID_IMAGE"


@pytest.fixture
def small_image_bytes() -> bytes:
    """Generate image below minimum dimension threshold (32x32)."""
    arr = np.ones((32, 32), dtype=np.uint8) * 128
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def real_sample_image_path() -> Path:
    """Path to the real NIH sample image if present."""
    p = Path(__file__).resolve().parent.parent / "data" / "sample_images" / "nih_cardiomegaly_sample.png"
    return p
