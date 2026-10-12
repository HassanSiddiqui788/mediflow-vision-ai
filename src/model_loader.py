"""Verified model loader with caching and evaluation-mode configuration."""

import logging
from typing import Dict, List, Optional
import torch
import torchxrayvision as xrv

from src.config import ModelSettings

logger = logging.getLogger(__name__)

# In-memory model cache to prevent reloading weights on every interaction
_LOADED_MODELS: Dict[str, xrv.models.DenseNet] = {}


class ModelLoadError(RuntimeError):
    """Raised when loading model architecture or verified weights fails."""


def load_chest_xray_model(
    settings: ModelSettings,
    force_reload: bool = False,
) -> xrv.models.DenseNet:
    """Load, verify, and cache the medically pretrained TorchXRayVision DenseNet-121 model.

    The model is:
    - Initialized with official weights pretrained on combined chest X-ray datasets.
    - Set to evaluation mode (`model.eval()`).
    - Configured with gradients disabled on all parameters.
    - Cached in memory so weights are not re-read from disk on subsequent inferences.

    Args:
        settings: ModelSettings specifying weights name and execution device.
        force_reload: If True, bypass cache and reload from weights source.

    Returns:
        Configured and verified xrv.models.DenseNet instance.

    Raises:
        ModelLoadError: If weights loading fails or architecture verification fails.
    """
    weights_name = settings.weights
    device_str = settings.device if torch.cuda.is_available() and settings.device == "cuda" else "cpu"
    cache_key = f"{weights_name}:{device_str}"

    if not force_reload and cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key]

    try:
        logger.info("Loading medically pretrained model with weights: %s", weights_name)
        # Instantiate DenseNet with verified weights
        # apply_sigmoid=False ensures we can extract both raw logits and probabilities
        model = xrv.models.DenseNet(weights=weights_name, apply_sigmoid=False)

        # Set evaluation mode
        model.eval()

        # Disable gradients across all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Move to target device
        device = torch.device(device_str)
        model = model.to(device)

        # Verification checks
        if not hasattr(model, "pathologies") or not model.pathologies:
            raise ModelLoadError(
                f"Model loaded with weights '{weights_name}' lacks supported 'pathologies' list."
            )

        if not hasattr(model, "op_threshs") or model.op_threshs is None:
            logger.warning(
                "Model does not contain pre-computed operating thresholds; calibrated scoring will fall back."
            )

        _LOADED_MODELS[cache_key] = model
        logger.info(
            "Model successfully loaded and verified. Supported classes: %d",
            len(model.pathologies),
        )
        return model

    except Exception as e:
        error_msg = (
            f"Failed to load chest X-ray classification model '{weights_name}': {str(e)}. "
            "Please check internet connectivity for initial weight download or ensure weights "
            "exist in '~/.torchxrayvision/models_data/'."
        )
        logger.error(error_msg, exc_info=True)
        raise ModelLoadError(error_msg) from e


def get_supported_pathologies(model: xrv.models.DenseNet) -> List[str]:
    """Return the list of validated pathology labels supported by the model."""
    return list(model.pathologies)
