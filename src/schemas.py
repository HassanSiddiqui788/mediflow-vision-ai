"""Pydantic schemas and data models for Medical Vision AI."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    """Metadata describing the uploaded medical image."""

    filename: str = Field(description="Original file name of the uploaded image")
    format: str = Field(description="Image format (PNG, JPEG, etc.)")
    width: int = Field(ge=1, description="Image width in pixels")
    height: int = Field(ge=1, description="Image height in pixels")
    channels: int = Field(ge=1, description="Number of color channels in source")
    size_bytes: int = Field(ge=0, description="Image file size in bytes")


class ValidationResult(BaseModel):
    """Result of image input validation."""

    is_valid: bool = Field(description="Whether the image passed all validation checks")
    error_message: Optional[str] = Field(
        default=None, description="Human-readable error description if invalid"
    )
    image_metadata: Optional[ImageMetadata] = Field(
        default=None, description="Extracted metadata if valid"
    )


class PathologyPrediction(BaseModel):
    """Individual pathology classification result."""

    pathology: str = Field(description="Name of the chest pathology")
    calibrated_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Threshold-calibrated operating point score (0.5 is decision boundary)",
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Standard multi-label sigmoid probability (uncalibrated)",
    )
    raw_logit: float = Field(
        description="Raw output from final linear classifier layer before sigmoid"
    )
    operating_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Pretrained operating point threshold for this pathology",
    )
    is_detected: bool = Field(
        description="True if calibrated_score >= 0.5 (above pathology operating threshold)"
    )


class InferenceResult(BaseModel):
    """Structured inference execution result for a single image."""

    analysis_id: str = Field(description="Unique UUID for this analysis execution")
    timestamp: str = Field(description="ISO 8601 timestamp of analysis execution")
    model_name: str = Field(description="Human-readable model name")
    model_weights: str = Field(description="Identifier of verified pretrained weights")
    device: str = Field(description="Execution device (cpu or cuda)")
    inference_time_ms: float = Field(
        ge=0.0, description="Measured model inference duration in milliseconds"
    )
    predictions: List[PathologyPrediction] = Field(
        description="All evaluated pathologies with full metrics"
    )
    top_findings: List[PathologyPrediction] = Field(
        description="Top pathologies ranked by calibrated score"
    )
    image_metadata: ImageMetadata = Field(
        description="Validated metadata of processed image"
    )
