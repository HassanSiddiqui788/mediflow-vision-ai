"""Inference execution and output interpretation engine for Medical Vision AI."""

import time
from typing import List
import torch
import torchxrayvision as xrv

from src.config import AppConfig
from src.schemas import ImageMetadata, InferenceResult, PathologyPrediction
from src.utils import generate_analysis_id, get_utc_timestamp


class InferenceExecutionError(RuntimeError):
    """Raised when an error occurs during model execution."""


def run_inference(
    tensor: torch.Tensor,
    model: xrv.models.DenseNet,
    image_metadata: ImageMetadata,
    config: AppConfig,
) -> InferenceResult:
    """Execute real model inference on preprocessed chest X-ray tensor.

    Computes:
    - Raw logits: Unbounded real numbers from final linear classifier.
    - Sigmoid probabilities: Standard Bernoulli multi-label output.
    - Operating thresholds: Pre-calibrated decision points for each pathology.
    - Calibrated scores: Operating-point normalized values where 0.5 is the exact decision boundary.

    Args:
        tensor: 4D preprocessed input tensor of shape (1, 1, 224, 224).
        model: Verified TorchXRayVision DenseNet instance in evaluation mode.
        image_metadata: Validated image metadata.
        config: Application configuration.

    Returns:
        Structured InferenceResult containing timing, scores, and sorted predictions.

    Raises:
        InferenceExecutionError: If forward pass fails.
    """
    if tensor.dim() != 4 or tensor.shape[1] != 1:
        raise InferenceExecutionError(
            f"Expected 4D single-channel tensor (1, 1, H, W), got shape: {tuple(tensor.shape)}"
        )

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    start_time = time.perf_counter()

    try:
        model.eval()
        with torch.no_grad():
            # 1. Feature extraction
            features = model.features2(tensor)

            # 2. Raw logits from classifier layer
            raw_logits_tensor = model.classifier(features)

            # 3. Multi-label sigmoid probabilities
            sigmoid_probs_tensor = torch.sigmoid(raw_logits_tensor)

            # 4. Calibrated scores via operating point normalization
            if hasattr(model, "op_threshs") and model.op_threshs is not None:
                calibrated_scores_tensor = xrv.models.op_norm(
                    sigmoid_probs_tensor, model.op_threshs.to(device)
                )
            else:
                calibrated_scores_tensor = sigmoid_probs_tensor

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000.0

    except Exception as e:
        raise InferenceExecutionError(f"Model forward pass failed: {str(e)}") from e

    # Convert tensors to CPU numpy / Python primitives
    logits_list = raw_logits_tensor[0].detach().cpu().numpy()
    probs_list = sigmoid_probs_tensor[0].detach().cpu().numpy()
    calibrated_list = calibrated_scores_tensor[0].detach().cpu().numpy()

    if hasattr(model, "op_threshs") and model.op_threshs is not None:
        threshs_list = model.op_threshs.detach().cpu().numpy()
    else:
        threshs_list = [0.5] * len(model.pathologies)

    predictions: List[PathologyPrediction] = []
    threshold = config.inference.operating_point_threshold

    for i, pathology in enumerate(model.pathologies):
        logit = float(logits_list[i])
        prob = float(probs_list[i])
        calibrated = float(calibrated_list[i])
        op_thresh = float(threshs_list[i]) if i < len(threshs_list) else 0.5

        # Handle any possible NaN values from unmodeled targets
        if torch.isnan(torch.tensor(calibrated)):
            calibrated = 0.0
        if torch.isnan(torch.tensor(prob)):
            prob = 0.0

        is_detected = calibrated >= threshold

        predictions.append(
            PathologyPrediction(
                pathology=pathology,
                calibrated_score=round(calibrated, 4),
                probability=round(prob, 4),
                raw_logit=round(logit, 4),
                operating_threshold=round(op_thresh, 4),
                is_detected=is_detected,
            )
        )

    # Sort predictions by calibrated score descending
    sorted_predictions = sorted(
        predictions, key=lambda x: x.calibrated_score, reverse=True
    )

    # Top findings
    top_k = config.inference.top_k_findings
    top_findings = sorted_predictions[:top_k]

    return InferenceResult(
        analysis_id=generate_analysis_id(),
        timestamp=get_utc_timestamp(),
        model_name=config.model.name,
        model_weights=config.model.weights,
        device=str(device),
        inference_time_ms=round(inference_time_ms, 2),
        predictions=sorted_predictions,
        top_findings=top_findings,
        image_metadata=image_metadata,
    )
