"""Image upload validation and integrity checking for Medical Vision AI."""

import io
from pathlib import Path
from typing import Union
from PIL import Image, UnidentifiedImageError

from src.config import ValidationSettings
from src.schemas import ImageMetadata, ValidationResult


def validate_image_bytes(
    data: bytes,
    filename: str,
    settings: ValidationSettings,
) -> ValidationResult:
    """Validate raw image bytes against format, size, dimension, and integrity limits.

    Args:
        data: Raw uploaded file bytes.
        filename: Original file name.
        settings: ValidationSettings containing allowed constraints.

    Returns:
        ValidationResult indicating success or specific failure reason.
    """
    # 1. Non-empty check
    if not data or len(data) == 0:
        return ValidationResult(
            is_valid=False,
            error_message="Uploaded file is empty (0 bytes).",
        )

    # 2. File size check
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"File size ({len(data) / (1024 * 1024):.2f} MB) exceeds maximum "
                f"allowed limit of {settings.max_file_size_mb} MB."
            ),
        )

    # 3. File extension check
    suffix = Path(filename).suffix.lower()
    if suffix not in [ext.lower() for ext in settings.allowed_extensions]:
        allowed_str = ", ".join(settings.allowed_extensions)
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Unsupported file extension '{suffix}'. Allowed formats are: {allowed_str}."
            ),
        )

    # 4. Image integrity & decoding check
    try:
        # First verify structural integrity
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()

        # Re-open to read raster pixels and metadata
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            img_format = img.format or "UNKNOWN"
            channels = len(img.getbands())

    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Corrupted or invalid image file. Could not decode: {str(e)}",
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Image decoding failure: {str(e)}",
        )

    # 5. Dimension checks
    if width < settings.min_dimension or height < settings.min_dimension:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Image dimensions ({width}x{height}) are smaller than minimum "
                f"supported resolution ({settings.min_dimension}x{settings.min_dimension})."
            ),
        )

    if width > settings.max_dimension or height > settings.max_dimension:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Image dimensions ({width}x{height}) exceed maximum allowed "
                f"resolution ({settings.max_dimension}x{settings.max_dimension})."
            ),
        )

    # 6. Format check against PIL format
    supported_pil_formats = {"PNG", "JPEG"}
    if img_format.upper() not in supported_pil_formats:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Decoded image format '{img_format}' is not supported. "
                "Only PNG and JPEG formats are supported."
            ),
        )

    metadata = ImageMetadata(
        filename=filename,
        format=img_format,
        width=width,
        height=height,
        channels=channels,
        size_bytes=len(data),
    )

    return ValidationResult(
        is_valid=True,
        error_message=None,
        image_metadata=metadata,
    )
