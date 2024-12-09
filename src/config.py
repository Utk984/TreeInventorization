"""
src/config.py
Configuration file for the urban tree inventory pipeline.
"""

import os

from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()

        # Data paths
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "../data")
        self.INPUT_DIR = os.path.join(self.DATA_DIR, "input")
        self.OUTPUT_DIR = os.path.join(self.DATA_DIR, "images/tree")
        self.LOG_FILE = os.path.join(self.DATA_DIR, "logs", "pipeline.log")

        # Panorama configuration
        self.PANORAMA_CSV = os.path.join(self.INPUT_DIR, "chandigarh_panoramas.csv")
        self.BATCH_SIZE = 50  # Number of panoramas to process in one batch

        # Cloud storage
        self.CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
        self.CLOUD_URL = os.getenv("CLOUD_URL")

        # API keys
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        # Database configuration
        self.DB_URL = os.getenv("DB_URL")
