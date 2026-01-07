import os
from pydantic import BaseModel

class Config(BaseModel):
    # Camera Settings
    RTSP_URL: str = "rtsp://admin:123456..@192.168.75.192:554/Preview_01_main"  # Default to local file for testing
    SAMPLE_RATE: float = 1.0  # Frames Per Second

    # Detection Settings
    SIMILARITY_THRESHOLD: float = 0.85
    HISTORY_WINDOW_SIZE: int = 60  # Short-term memory size (e.g., 60 seconds)

    # Storage Settings
    DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "daily_log"

    # Model Settings
    # Using ViT-B-16-SigLIP2 as requested
    MODEL_NAME: str = "ViT-B-16-SigLIP2"
    PRETRAINED_WEIGHTS: str = "webli"
    
    # ROI Settings
    MOTION_THRESHOLD: float = 25.0  # Pixel difference threshold
    MIN_CONTOUR_AREA: int = 500     # Minimum area to be considered motion
    MOTION_BLUR_SIZE: int = 21      # Gaussian Blur size (must be odd)
    ZERO_SHOT_LABELS: list[str] = ["a person", "a dog", "shadow", "light change"]
    
    # YOLO Settings
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    YOLO_CLASSES: list[str] = ["person", "car", "bicycle", "dog", "cat"] # Default selection

    # System Settings
    DEVICE: str = "cpu" # Will be updated dynamically if cuda/mps available

settings = Config()

# Update device based on availability
try:
    import torch
    if torch.cuda.is_available():
        settings.DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        settings.DEVICE = "mps"
except ImportError:
    pass
