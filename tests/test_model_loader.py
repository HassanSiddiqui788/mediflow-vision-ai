"""Tests for verified model loading and caching."""

import pytest
from src.config import ModelSettings
from src.model_loader import load_chest_xray_model, ModelLoadError, get_supported_pathologies


def test_load_real_model():
    settings = ModelSettings(weights="densenet121-res224-all", device="cpu")
    model = load_chest_xray_model(settings)

    assert model is not None
    assert model.training is False  # Must be in eval mode
    assert len(model.pathologies) == 18

    # Ensure gradients are disabled
    for param in model.parameters():
        assert param.requires_grad is False


def test_model_caching():
    settings = ModelSettings(weights="densenet121-res224-all", device="cpu")
    model1 = load_chest_xray_model(settings)
    model2 = load_chest_xray_model(settings)
    assert model1 is model2  # Must be cached in memory


def test_supported_pathologies_list():
    settings = ModelSettings(weights="densenet121-res224-all", device="cpu")
    model = load_chest_xray_model(settings)
    pathologies = get_supported_pathologies(model)
    assert "Cardiomegaly" in pathologies
    assert "Pneumonia" in pathologies
    assert "Atelectasis" in pathologies
    assert len(pathologies) == 18


def test_invalid_weights_failure():
    settings = ModelSettings(weights="nonexistent-invalid-weights-xyz", device="cpu")
    with pytest.raises(ModelLoadError):
        load_chest_xray_model(settings, force_reload=True)
