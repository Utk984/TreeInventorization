import logging
import time
from typing import Tuple, Union, List, Optional
from dataclasses import dataclass
import torch.nn as nn

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

ImageLike = Union[str, Image.Image, np.ndarray]

def verify_mask(image: ImageLike, model) -> Tuple[bool, float]:
    """
    Classify mask usability using a trained model.

    Parameters
    ----------
    image : str | PIL.Image.Image | np.ndarray
        - File path, PIL image, or numpy array (H×W or H×W×3). If float array in [0,1],
          it will be scaled to [0,255] before conversion to PIL.
    model : InferenceBundle
        The bundle returned by `load_for_inference(...)`.
        Must be for variant='resnet_overlay' and backbone='resnet50' or 'resnet18' or 'efficientnet_b0'.
        Uses `model.model` (torch module), `model.device`, `model.image_size`,
        `model.threshold`, and optional `model.norm_mean`/`model.norm_std`.

    Returns
    -------
    (usable: bool, prob: float)
        usable: True if prob >= model.threshold
        prob:   sigmoid(logit) ∈ [0, 1]
    """
    start_time = time.time()
    try:
        # Canonicalize image → PIL RGB
        if isinstance(image, str):
            pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                # grayscale → RGB
                if arr.dtype != np.uint8:
                    arr = (arr * (255.0 if arr.max() <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, mode="L").convert("RGB")
            elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                if arr.dtype != np.uint8:
                    arr = (arr * (255.0 if arr.max() <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
                if arr.shape[-1] == 4:  # RGBA → RGB
                    arr = Image.fromarray(arr, mode="RGBA").convert("RGB")
                    pil = arr
                else:
                    pil = Image.fromarray(arr, mode="RGB")
            else:
                raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        logger.info(f"🖼️  Starting usability inference | size={pil.size} | variant={getattr(model, 'variant', '?')} | backbone={getattr(model, 'backbone', '?')}")

        # Build transforms consistent with training
        from torchvision import transforms as T
        tfs = [T.Resize((model.image_size, model.image_size)), T.ToTensor()]
        if getattr(model, "norm_mean", None) is not None and getattr(model, "norm_std", None) is not None:
            tfs.append(T.Normalize(mean=model.norm_mean, std=model.norm_std))
        tf = T.Compose(tfs)

        x = tf(pil).unsqueeze(0).to(model.device)

        model.model.eval()
        with torch.no_grad():
            logits = model.model(x)
            prob = torch.sigmoid(logits).item()

        thr = float(getattr(model, "threshold", 0.5))
        usable = prob >= thr
        dt = time.time() - start_time
        logger.info(f"✅ Usability inference done in {dt:.3f}s | prob={prob:.3f} | thr={thr:.2f} | usable={usable}")
        return usable, float(prob)

    except Exception as e:
        logger.error(f"❌ Error during mask verification: {e}")
        raise
