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
    logger.info(f"🌳 Starting trunk detection for {model_version} - view shape: {view.shape}")
    
    try:
        # Run inference with same confidence threshold as evaluate_yolo.py
        logger.info(f"🔍 Running YOLO prediction with conf=0.25, imgsz=(1024,1024), device={device}")
        results = model.predict(view, verbose=False, imgsz=(640,640), device=device, conf=0.1)
        
        logger.info(f"📊 Raw YOLO results: {len(results) if results else 0} result(s)")
        
        # Filter for trunk-only detections using logic from evaluate_yolo.py
        if results and len(results) > 0:
            result = results[0]
            logger.info(f"📊 Processing first result - has boxes: {hasattr(result, 'boxes') and result.boxes is not None}")
            logger.info(f"📊 Processing first result - has masks: {hasattr(result, 'masks') and result.masks is not None}")
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Get class predictions
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                logger.info(f"📊 Total detections: {len(classes)}")
                if len(classes) > 0:
                    logger.info(f"📊 Classes found: {np.unique(classes)}")
                    logger.info(f"📊 Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
                    logger.info(f"📊 All confidences: {confidences}")
                else:
                    logger.warning(f"⚠️ NO DETECTIONS FOUND BY YOLO!")
                
                # TreeModelV3 uses class 1 for trunks, stage2 uses class 0 (matching evaluate_yolo.py)
                if 'TreeModelV3' in model_version or 'TreeModelV3' in str(model):
                    trunk_indices = (classes == 1)
                    logger.info(f"📊 Using TreeModelV3 logic - looking for class 1 (trunks)")
                else:
                    trunk_indices = (classes == 0)
                    logger.info(f"📊 Using stage2 logic - looking for class 0 (trunks)")
                
                logger.info(f"📊 Trunk indices: {trunk_indices}")
                logger.info(f"📊 Number of trunk detections: {np.sum(trunk_indices)}")
                
                if np.any(trunk_indices):
                    # Filter boxes, masks, and confidences for trunk only
                    trunk_confidences = confidences[trunk_indices]
                    logger.info(f"📊 Trunk confidences: {trunk_confidences}")
                    
                    result.boxes = result.boxes[trunk_indices]
                    if hasattr(result, 'masks') and result.masks is not None:
                        result.masks = result.masks[trunk_indices]
                        logger.info(f"📊 Filtered masks: {len(result.masks)}")
                    
                    trunk_count = np.sum(trunk_indices)
                    logger.info(f"✅ Found {trunk_count} trunk detections (filtered from {len(trunk_indices)} total)")
                else:
                    logger.warning(f"⚠️ No trunk detections found - all classes: {classes}")
                    # Return empty results
                    results = []
            else:
                logger.warning(f"⚠️ No bounding boxes found in result")
                results = []
        else:
            logger.warning(f"⚠️ No results returned from YOLO")
            results = []
        
        logger.info(f"📊 Final results: {len(results)} result(s)")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error during trunk detection ({model_version}): {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
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