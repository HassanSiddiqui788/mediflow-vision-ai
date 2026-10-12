"""Tests for inference execution, metric calculation, and error handling."""

import pytest
import torch

from src.config import AppConfig
from src.inference import run_inference, InferenceExecutionError
from src.model_loader import load_chest_xray_model
from src.schemas import ImageMetadata


@pytest.fixture
def loaded_model(default_config):
    return load_chest_xray_model(default_config.model)


@pytest.fixture
def sample_metadata():
    return ImageMetadata(
        filename="test_scan.png",
        format="PNG",
        width=256,
        height=256,
        channels=1,
        size_bytes=10240,
    )


def test_run_inference_structure_and_types(loaded_model, sample_metadata, default_config):
    # Dummy valid tensor in [-1024, 1024] range
    tensor = torch.zeros(1, 1, 224, 224, dtype=torch.float32)

    result = run_inference(tensor, loaded_model, sample_metadata, default_config)

    assert result.analysis_id.startswith("ana_")
    assert result.timestamp is not None
    assert result.inference_time_ms > 0.0
    assert result.model_name == default_config.model.name
    assert result.model_weights == default_config.model.weights
    assert len(result.predictions) == 18
    assert len(result.top_findings) == default_config.inference.top_k_findings

    # Check predictions are sorted descending by calibrated score
    scores = [p.calibrated_score for p in result.predictions]
    assert scores == sorted(scores, reverse=True)

    # Check individual prediction fields
    for p in result.predictions:
        assert 0.0 <= p.calibrated_score <= 1.0
        assert 0.0 <= p.probability <= 1.0
        assert isinstance(p.raw_logit, float)
        assert 0.0 <= p.operating_threshold <= 1.0
        assert p.is_detected == (p.calibrated_score >= default_config.inference.operating_point_threshold)


def test_run_inference_invalid_tensor_shape(loaded_model, sample_metadata, default_config):
    # Pass 3D tensor instead of 4D
    bad_tensor = torch.zeros(1, 224, 224)
    with pytest.raises(InferenceExecutionError):
        run_inference(bad_tensor, loaded_model, sample_metadata, default_config)
