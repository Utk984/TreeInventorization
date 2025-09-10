import logging
import time
import numpy as np

# Configure logger for tree detection
logger = logging.getLogger(__name__)

def detect_trunks(view, model, device, model_version="V3"):
    """
    Detect trunks only in view with logging for TreeModelV3.
    Filters out tree detections and returns only trunk detections.
    
    Args:
        view: Input image array
        model: Loaded YOLO model (TreeModelV3)
        device: Device to run inference on
        model_version: String identifier for logging ("V3")
    
    Returns:
        ultralytics Results object with trunk-only detections
    """
    logger.debug(f"🌳 Starting trunk detection for {model_version} - view shape: {view.shape}")
    
    try:
        # Run inference with same parameters as original segment.py
        results = model.predict(view, verbose=False, imgsz=(1024,1024), device=device, conf=0.01)
        
        # Filter for trunk-only detections (class 1)
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Get trunk detections only (class 1)
                trunk_indices = (result.boxes.cls == 1).cpu().numpy()
                if np.any(trunk_indices):
                    # Filter boxes, masks, and confidences for trunk only
                    result.boxes = result.boxes[trunk_indices]
                    if hasattr(result, 'masks') and result.masks is not None:
                        result.masks = result.masks[trunk_indices]
                    
                    trunk_count = np.sum(trunk_indices)
                    logger.debug(f"📊 Found {trunk_count} trunk detections (filtered from {len(trunk_indices)} total)")
                else:
                    logger.debug(f"📊 No trunk detections found")
                    # Return empty results
                    results = []
            else:
                logger.debug(f"📊 No bounding boxes found")
                results = []
        else:
            logger.debug(f"📊 No results returned")
            results = []
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error during trunk detection ({model_version}): {str(e)}")
        raise


def detect_trees(view, model, device):
    """Detect trees in view with logging."""
    logger.debug(f"🌳 Starting tree detection for view shape: {view.shape}")
    
    try:
        results = model.predict(view, verbose=False, imgsz=(1024,1024), device=device)
        
        # Log detection statistics
        total_detections = 0
        for result in results:
            if result.masks is not None:
                total_detections += len(result.masks)
        
        logger.debug(f"📊 Found {total_detections} tree detections")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error during tree detection: {str(e)}")
        raise