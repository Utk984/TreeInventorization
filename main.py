import logging
import time
from src.pipeline.pano_parallel import process_panoramas_parallel
from config import Config
from ultralytics import YOLO
import asyncio
from src.utils.system_resources import calculate_optimal_concurrency

logger = logging.getLogger(__name__)

def load_models(config: Config):
    """Load tree segmentation and calibration models with logging."""
    logger.info("Starting model loading process")
    
    try:
        # Load tree segmentation model from config
        logger.info(f"Loading tree segmentation model from: {config.TREE_MODEL_PATH}")
        tree_model = YOLO(config.TREE_MODEL_PATH)
        logger.info("✅ Tree segmentation model loaded successfully")
        return tree_model
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        raise

def main():
    """Main pipeline execution with comprehensive logging."""
    pipeline_start_time = time.time()
    logger.info("=" * 60)
    logger.info("🌳 Tree Detection Pipeline Started")
    logger.info("=" * 60)
    
    try:
        config = Config()
        
        # Load models
        tree_model = load_models(config)
        
        # Calculate optimal concurrency based on system resources
        try:
            optimal_concurrent = calculate_optimal_concurrency()
            logger.info(f"🔧 Auto-calculated optimal concurrency: {optimal_concurrent}")
        except Exception as e:
            logger.error(f"💥 Failed to calculate optimal concurrency: {str(e)}")
            optimal_concurrent = config.MAX_CONCURRENT
            logger.info(f"🔧 Using user-specified concurrency: {optimal_concurrent}")
        
        # Run parallel pipeline
        logger.info("🔄 Starting parallel panorama processing pipeline")
        asyncio.run(process_panoramas_parallel(config, tree_model, max_concurrent=optimal_concurrent))
        
        total_time = time.time() - pipeline_start_time
        logger.info("=" * 60)
        logger.info(f"🎉 Pipeline completed successfully in {total_time:.2f} seconds")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main()