"""Image preprocessing pipeline matching TorchXRayVision specifications."""

import io
from typing import Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchxrayvision as xrv

from src.config import ModelSettings


def load_image_as_numpy(image_source: Union[bytes, Image.Image]) -> np.ndarray:
    """Load image from bytes or PIL Image and convert to 2D float32 numpy array.

    Args:
        image_source: Image raw bytes or PIL Image object.

    Returns:
        2D float32 numpy array representing grayscale pixel intensities.
    """
    if isinstance(image_source, bytes):
        pil_img = Image.open(io.BytesIO(image_source))
    else:
        pil_img = image_source

    # Convert to standard grayscale (Luminance)
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    arr = np.array(pil_img, dtype=np.float32)
    return arr


def preprocess_chest_xray(
    image_source: Union[bytes, Image.Image],
    settings: ModelSettings,
) -> torch.Tensor:
    """Preprocess a chest X-ray image according to verified TorchXRayVision standards.

    Pipeline:
    1. Decode image into 2D float32 grayscale array.
    2. Normalize intensity range from [0, 255] to [-1024, 1024] using xrv.datasets.normalize.
    3. Construct 4D tensor with shape (1, 1, H, W).
    4. Spatially resize to target resolution (224, 224) using bilinear interpolation with antialiasing.
    5. Cast to float32 and transfer to target device (cpu/cuda).

    Args:
        image_source: Raw uploaded bytes or PIL Image instance.
        settings: ModelSettings containing resolution, target range, and device.

    Returns:
        Preprocessed torch.Tensor with shape (1, 1, 224, 224) on target device.
    """
    # 1. Load as 2D grayscale numpy array
    img_np = load_image_as_numpy(image_source)

    # 2. TorchXRayVision normalization to [-1024, 1024]
    # xrv models were trained on Hounsfield-like normalized scales
    normalized_np = xrv.datasets.normalize(img_np, maxval=settings.raw_max_pixel_val)

    # 3. Create 4D tensor (Batch=1, Channel=1, H, W)
    tensor = torch.from_numpy(normalized_np)[None, None, ...].float()

    # 4. Resize to target resolution (e.g. 224x224)
    target_h, target_w = settings.input_resolution
    resize_transform = T.Resize((target_h, target_w), antialias=True)
    resized_tensor = resize_transform(tensor)

    # 5. Move to specified device
    device = torch.device(settings.device if torch.cuda.is_available() and settings.device == "cuda" else "cpu")
    final_tensor = resized_tensor.to(device)

    return final_tensor
