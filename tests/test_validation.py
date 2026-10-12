"""Tests for image upload validation."""

import pytest
from src.config import ValidationSettings
from src.validation import validate_image_bytes


def test_validate_valid_png(valid_png_bytes):
    settings = ValidationSettings()
    res = validate_image_bytes(valid_png_bytes, "chest_xray.png", settings)
    assert res.is_valid is True
    assert res.error_message is None
    assert res.image_metadata is not None
    assert res.image_metadata.filename == "chest_xray.png"
    assert res.image_metadata.format == "PNG"
    assert res.image_metadata.width == 256
    assert res.image_metadata.height == 256
    assert res.image_metadata.size_bytes == len(valid_png_bytes)


def test_validate_valid_jpeg(valid_rgb_jpeg_bytes):
    settings = ValidationSettings()
    res = validate_image_bytes(valid_rgb_jpeg_bytes, "patient_scan.jpg", settings)
    assert res.is_valid is True
    assert res.error_message is None
    assert res.image_metadata is not None
    assert res.image_metadata.format == "JPEG"
    assert res.image_metadata.width == 300
    assert res.image_metadata.height == 300


def test_validate_invalid_extension(valid_png_bytes):
    settings = ValidationSettings()
    res = validate_image_bytes(valid_png_bytes, "report.pdf", settings)
    assert res.is_valid is False
    assert "Unsupported file extension" in res.error_message


def test_validate_corrupted_image(corrupted_image_bytes):
    settings = ValidationSettings()
    res = validate_image_bytes(corrupted_image_bytes, "corrupt.png", settings)
    assert res.is_valid is False
    assert "Corrupted or invalid image file" in res.error_message


def test_validate_empty_bytes():
    settings = ValidationSettings()
    res = validate_image_bytes(b"", "empty.png", settings)
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()


def test_validate_image_too_small(small_image_bytes):
    settings = ValidationSettings(min_dimension=64)
    res = validate_image_bytes(small_image_bytes, "tiny.png", settings)
    assert res.is_valid is False
    assert "smaller than minimum supported resolution" in res.error_message


def test_validate_file_size_exceeded(valid_png_bytes):
    # Set limit to 0 MB to trigger size failure
    settings = ValidationSettings(max_file_size_mb=0)
    res = validate_image_bytes(valid_png_bytes, "large.png", settings)
    assert res.is_valid is False
    assert "exceeds maximum allowed limit" in res.error_message
