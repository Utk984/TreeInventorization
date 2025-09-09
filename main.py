import logging
import time
from cli import parse_args, build_config
from src.pipeline.pano_async import process_panoramas
from models.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2
from models.CalibrateDepth.model import DepthCalibrator
from models.MaskQuality.load import load_for_inference
from config import Config
import torch
from ultralytics import YOLO
import asyncio

logger = logging.getLogger(__name__)

def load_models(config: Config):
    """Load depth estimation and tree segmentation models with logging."""
    logger.info("Starting model loading process")
    
    try:
        # Load depth model
        logger.info(f"Loading depth model from: {config.DEPTH_MODEL_PATH}")
        depth_model = DepthAnythingV2(**{**config.DEPTH_MODEL_CONFIGS["vitl"], "max_depth": 80})
        depth_model.load_state_dict(torch.load(config.DEPTH_MODEL_PATH, map_location="cpu"))
        depth_model.to(config.DEVICE).eval()
        logger.info("✅ Depth model loaded successfully")

        # Load tree segmentation model
        logger.info(f"Loading tree segmentation model from: {config.TREE_MODEL_PATH}")
        tree_model = YOLO(config.TREE_MODEL_PATH)
        logger.info("✅ Tree segmentation model loaded successfully")

        logger.info(f"Loading depth calibration model from: {config.DEPTH_CALIBRATION_MODEL_PATH}")
        depth_calibrator = DepthCalibrator(config.DEPTH_CALIBRATION_MODEL_PATH)
        logger.info("✅ Depth calibration model loaded successfully")

        # Load mask model
        logger.info(f"Loading mask model from: {config.MASK_MODEL_PATH}")
        mask_model = load_for_inference(config.MASK_MODEL_PATH)
        logger.info("✅ Mask model loaded successfully")
        
        return depth_model, tree_model, depth_calibrator, mask_model
        
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
        depth_model, tree_model, depth_calibrator, mask_model = load_models(config)
        
        # Run pipeline
        logger.info("🔄 Starting panorama processing pipeline")
        asyncio.run(process_panoramas(config, depth_model, tree_model, depth_calibrator, mask_model))
        
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