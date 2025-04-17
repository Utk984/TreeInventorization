import os
import torch
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()

        # Base directories
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        # Data directories
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.VIEW_DIR = os.path.join(self.DATA_DIR, "views")
        self.FULL_DIR = os.path.join(self.DATA_DIR, "full")
        self.LOG_DIR = os.path.join(self.DATA_DIR, "logs")

        # Ensure output folders exist
        os.makedirs(self.VIEW_DIR, exist_ok=True)
        os.makedirs(self.FULL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        # Panorama CSV input
        self.PANORAMA_CSV = os.path.join(self.ROOT_DIR, "delhi_streets.csv")

        # Output CSV
        self.OUTPUT_CSV = os.path.join(self.ROOT_DIR, "tree_data.csv")

        # Model config
        self.TREE_MODEL_PATH = os.path.join(
            self.ROOT_DIR, "models", "TreeModel", "weights", "best.pt"
        )

        self.DEPTH_MODEL_PATH = os.path.join(
            self.ROOT_DIR, "models", "DepthAnything", "checkpoints",
            "depth_anything_v2_metric_vkitti_vitl.pth"
        )

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MODEL_CONFIGS = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        # Other settings
        self.FOV = 90
        self.BATCH_SIZE = 10

        # Logging
        self.LOG_FILE = os.path.join(self.LOG_DIR, "pipeline.log")

        # Optional: Cloud/DB/API keys
        #self.CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
        #self.CLOUD_URL = os.getenv("CLOUD_URL")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.DB_URL = os.getenv("DB_URL")
