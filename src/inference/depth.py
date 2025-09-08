import logging
import time

# Configure logger for depth inference
logger = logging.getLogger(__name__)

def estimate_depth(image, model):
    """Estimate depth from image with logging."""
    logger.debug(f"🔍 Starting depth estimation for image shape: {image.shape}")
    start_time = time.time()
    
    try:
        depth = model.infer_image(image)
        
        inference_time = time.time() - start_time
        logger.debug(f"✅ Depth estimation completed in {inference_time:.3f}s")
        logger.debug(f"📊 Depth map shape: {depth.shape}, min: {depth.min():.2f}, max: {depth.max():.2f}")
        
        return depth
        
    except Exception as e:
        logger.error(f"❌ Error during depth estimation: {str(e)}")
        raise
