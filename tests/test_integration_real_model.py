"""Integration tests executing real model inference end-to-end on real chest X-ray images."""

from pathlib import Path
import pytest

from src.config import load_config
from src.inference import run_inference
from src.model_loader import load_chest_xray_model
from src.preprocessing import preprocess_chest_xray
from src.validation import validate_image_bytes


def test_real_model_inference_pipeline_end_to_end(real_sample_image_path):
    """End-to-end integration test executing real inference on a genuine NIH chest X-ray image."""
    if not real_sample_image_path.exists():
        pytest.skip(f"Test sample image not found at: {real_sample_image_path}")

    with open(real_sample_image_path, "rb") as f:
        image_bytes = f.read()

    config = load_config()

    # 1. Validation phase
    val_result = validate_image_bytes(
        image_bytes,
        real_sample_image_path.name,
        config.validation,
    )
    assert val_result.is_valid is True
    assert val_result.image_metadata is not None
    assert val_result.image_metadata.width == 512
    assert val_result.image_metadata.height == 512

    # 2. Preprocessing phase
    tensor = preprocess_chest_xray(image_bytes, config.model)
    assert tensor.shape == (1, 1, 224, 224)

    # 3. Model loading
    model = load_chest_xray_model(config.model)
    assert model is not None
    assert model.training is False

    # 4. Real inference execution
    result = run_inference(tensor, model, val_result.image_metadata, config)

    # 5. Verification of real predictions
    assert result.analysis_id.startswith("ana_")
    assert result.inference_time_ms > 0.0
    assert len(result.predictions) == 18

    # The sample image is patient #1 from NIH ChestX-ray14 with Cardiomegaly
    top_pathology = result.top_findings[0].pathology
    assert top_pathology == "Cardiomegaly"
    assert result.top_findings[0].calibrated_score >= 0.5
    assert result.top_findings[0].is_detected is True

    # Ensure all predictions are finite and non-NaN
    for p in result.predictions:
        assert not (p.calibrated_score != p.calibrated_score)  # Check not NaN
        assert not (p.probability != p.probability)
        assert not (p.raw_logit != p.raw_logit)
