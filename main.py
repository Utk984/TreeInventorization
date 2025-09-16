import logging
import time
from cli import parse_args, build_config
from src.pipeline.pano_parallel import process_panoramas_parallel
from models.CalibrateDepth.model import DepthCalibrator
from config import Config
import torch
from ultralytics import YOLO
import asyncio

logger = logging.getLogger(__name__)

def load_models(config: Config):
    """Load tree segmentation and calibration models with logging."""
    logger.info("Starting model loading process")
    
    try:
        # Load tree segmentation model
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
        # Parse arguments and build configuration
        logger.info("Parsing command line arguments")
        args = parse_args()
        logger.info(f"Arguments parsed: {vars(args)}")
        
        logger.info("Building configuration")
        config = build_config(args)
        logger.info(f"Configuration built successfully")
        logger.info(f"Input CSV: {config.PANORAMA_CSV}")
        logger.info(f"Output CSV: {config.OUTPUT_CSV}")
        logger.info(f"View directory: {config.VIEW_DIR}")
        logger.info(f"Full directory: {config.FULL_DIR}")
        
        # Load models
        tree_model = load_models(config)
        
        # Run parallel pipeline
        logger.info("🔄 Starting parallel panorama processing pipeline")
        asyncio.run(process_panoramas_parallel(config, tree_model, max_concurrent=3))
        
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