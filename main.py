from src.cli import parse_args, build_config
from src.pipeline.pano_async import process_panoramas
from models.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2
from src.config import Config
import torch
from ultralytics import YOLO
import asyncio

def load_models(config: Config):
    # Load depth model
    depth_model = DepthAnythingV2(**{**config.MODEL_CONFIGS["vitl"], "max_depth": 80})
    depth_model.load_state_dict(torch.load(config.DEPTH_MODEL_PATH, map_location="cpu"))
    depth_model.to(config.DEVICE).eval()

    # Load tree segmentation model
    tree_model = YOLO(config.TREE_MODEL_PATH)

    return depth_model, tree_model

def main():
    args = parse_args()
    config = build_config(args)
    depth_model, tree_model = load_models(config)
    asyncio.run(process_panoramas(config, depth_model, tree_model))

if __name__ == "__main__":
    main()