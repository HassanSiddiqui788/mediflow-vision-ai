"""Configuration loader and settings manager for Medical Vision AI."""

import os
from pathlib import Path
from typing import List, Tuple
import yaml
from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    """Settings defining the computer vision model and weights."""

    name: str = "TorchXRayVision DenseNet-121"
    architecture: str = "densenet121"
    weights: str = "densenet121-res224-all"
    provider: str = "torchxrayvision"
    library_version: str = "1.5.5"
    license: str = "Apache-2.0"
    citation: str = "Cohen et al., TorchXRayVision (2022)"
    input_resolution: Tuple[int, int] = (224, 224)
    input_channels: int = 1
    normalization_range: Tuple[float, float] = (-1024.0, 1024.0)
    raw_max_pixel_val: float = 255.0
    device: str = "cpu"
    apply_sigmoid: bool = False


class ValidationSettings(BaseModel):
    """Limits and parameters for input image validation."""

    max_file_size_mb: int = 10
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".png", ".jpg", ".jpeg"]
    )
    min_dimension: int = 64
    max_dimension: int = 4096


class InferenceSettings(BaseModel):
    """Inference thresholding and presentation options."""

    operating_point_threshold: float = 0.5
    top_k_findings: int = 5


class AppConfig(BaseModel):
    """Root configuration model combining all subsystems."""

    model: ModelSettings = Field(default_factory=ModelSettings)
    pathologies: List[str] = Field(default_factory=list)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = "configs/model_config.yaml") -> AppConfig:
    """Load configuration from YAML file and apply environment variable overrides.

    Args:
        config_path: Path relative to project root or absolute path.

    Returns:
        Validated AppConfig instance.
    """
    root = get_project_root()
    path = Path(config_path) if Path(config_path).is_absolute() else root / config_path

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    else:
        raw_data = {}

    config = AppConfig(**raw_data)

    # Environment variable overrides
    env_weights = os.getenv("MODEL_WEIGHTS")
    if env_weights:
        config.model.weights = env_weights

    env_device = os.getenv("MODEL_DEVICE")
    if env_device:
        config.model.device = env_device

    env_max_size = os.getenv("MAX_UPLOAD_SIZE_MB")
    if env_max_size and env_max_size.isdigit():
        config.validation.max_file_size_mb = int(env_max_size)

    return config
