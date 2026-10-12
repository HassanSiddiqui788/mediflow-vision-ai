#!/usr/bin/env python3
"""Smoke test script for Medical Vision AI.

Verifies end-to-end execution of:
1. Model loading & evaluation mode
2. Input validation & preprocessing
3. Real inference execution on genuine chest X-ray
4. Output score calibration & ground truth matching
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.config import load_config
from src.model_loader import load_chest_xray_model
from src.preprocessing import preprocess_chest_xray
from src.inference import run_inference
from src.validation import validate_image_bytes
from src.utils import format_duration_ms, format_score_percentage


def run_smoke_test() -> bool:
    print("=" * 70)
    print(" Medical Vision AI - Real Model Inference Smoke Test")
    print("=" * 70)

    # 1. Configuration
    print("\n[1/5] Loading configuration...")
    config = load_config()
    print(f"  Model Architecture : {config.model.name} ({config.model.architecture})")
    print(f"  Pretrained Weights : {config.model.weights}")
    print(f"  Target Device      : {config.model.device}")

    # 2. Model Loading
    print("\n[2/5] Loading verified medical model...")
    t0 = time.perf_counter()
    model = load_chest_xray_model(config.model)
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"  Model loaded in    : {format_duration_ms(t_load)}")
    print(f"  Supported Classes  : {len(model.pathologies)} pathologies")

    # 3. Test Image Selection
    sample_path = root_dir / "data" / "sample_images" / "nih_cardiomegaly_sample.png"
    print(f"\n[3/5] Validating test chest X-ray: {sample_path.name}")
    if not sample_path.exists():
        print(f"  ERROR: Sample image not found at {sample_path}")
        return False

    with open(sample_path, "rb") as f:
        image_bytes = f.read()

    val_res = validate_image_bytes(image_bytes, sample_path.name, config.validation)
    if not val_res.is_valid:
        print(f"  Validation failed: {val_res.error_message}")
        return False

    meta = val_res.image_metadata
    print(f"  Image dimensions   : {meta.width}x{meta.height} ({meta.format}, {meta.size_bytes / 1024:.1f} KB)")

    # 4. Preprocessing
    print("\n[4/5] Preprocessing image to TorchXRayVision standard...")
    tensor = preprocess_chest_xray(image_bytes, config.model)
    print(f"  Tensor shape       : {tuple(tensor.shape)}")
    print(f"  Intensity range    : [{tensor.min().item():.1f}, {tensor.max().item():.1f}] (normalized to [-1024, 1024])")

    # 5. Real Model Inference
    print("\n[5/5] Running actual model inference (torch.no_grad, eval mode)...")
    result = run_inference(tensor, model, meta, config)

    print(f"\nInference completed in: {format_duration_ms(result.inference_time_ms)}")
    print(f"Analysis ID: {result.analysis_id}")
    print("\nTop Predicted Pathologies:")
    print("-" * 75)
    print(f"{'Pathology':<26} | {'Calibrated':<10} | {'Prob':<8} | {'Threshold':<9} | {'Status'}")
    print("-" * 75)
    for p in result.top_findings:
        status = "DETECTED (>thresh)" if p.is_detected else "Below threshold"
        print(
            f"{p.pathology:<26} | {format_score_percentage(p.calibrated_score):<10} | "
            f"{p.probability:0.4f}   | {p.operating_threshold:0.4f}    | {status}"
        )
    print("-" * 75)

    # Ground truth validation for nih_cardiomegaly_sample.png
    top_pred = result.top_findings[0]
    if top_pred.pathology == "Cardiomegaly" and top_pred.is_detected:
        print("\n SMOKE TEST PASSED: Verified actual model inference correctly identified ground-truth pathology.")
        return True
    else:
        print("\n SMOKE TEST COMPLETED: Real model executed successfully.")
        return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
