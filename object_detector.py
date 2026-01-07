from ultralytics import YOLO
from loguru import logger
import numpy as np
from config import settings
import torch

class ObjectDetector:
    def __init__(self):
        logger.info(f"Loading YOLO model: {settings.YOLO_MODEL_NAME}")
        self.model = YOLO(settings.YOLO_MODEL_NAME)
        # Check if we can move to device
        # Ultralytics usually handles device automatically or via .to()
        if settings.DEVICE in ["cuda", "mps"]:
             self.model.to(settings.DEVICE)
        
        # Mapping COCO class names to IDs
        self.class_names = self.model.names
        self.name_to_id = {v: k for k, v in self.class_names.items()}
        logger.info("YOLO model loaded.")

    def detect(self, frame: np.ndarray, target_classes: list[str]) -> list[tuple[int, int, int, int]]:
        """
        Detects objects in the frame.
        Returns a list of bounding boxes (x, y, w, h).
        """
        if not target_classes:
            return []

        # Filter IDs based on target_classes (strings)
        target_ids = [self.name_to_id[c] for c in target_classes if c in self.name_to_id]
        
        if not target_ids:
            return []

        results = self.model.predict(frame, classes=target_ids, verbose=False, device=settings.DEVICE)
        
        boxes = []
        for r in results:
            for box in r.boxes:
                # box.xywh is center_x, center_y, width, height
                # box.xyxy is x1, y1, x2, y2
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                boxes.append((int(x1), int(y1), int(w), int(h)))
        
        return boxes
