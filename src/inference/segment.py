import logging
import time

# Configure logger for tree detection
logger = logging.getLogger(__name__)

def detect_trees(view, model, device):
    """Detect trees in view with logging."""
    logger.debug(f"🌳 Starting tree detection for view shape: {view.shape}")
    start_time = time.time()
    
    try:
        results = model.predict(view, verbose=False, imgsz=(1024,1024), device=device)
        
        inference_time = time.time() - start_time
        
        # Log detection statistics
        total_detections = 0
        for result in results:
            if result.masks is not None:
                total_detections += len(result.masks)
        
        logger.debug(f"✅ Tree detection completed in {inference_time:.3f}s")
        logger.debug(f"📊 Found {total_detections} tree detections")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error during tree detection: {str(e)}")
        raise