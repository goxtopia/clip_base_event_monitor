import time
import signal
from loguru import logger
import numpy as np
import cv2

from config import settings
from stream_loader import StreamLoader
from vector_engine import VectorEngine
from memory_store import MemoryStore
from detector import AnomalyDetector
from motion_detector import MotionDetector

class SentinelSystem:
    def __init__(self):
        self.running = False
        
        # Initialize components
        self.stream_loader = StreamLoader(
            source=settings.RTSP_URL,
            sample_rate=settings.SAMPLE_RATE
        )
        self.vector_engine = VectorEngine()
        self.memory_store = MemoryStore()
        self.detector = AnomalyDetector(self.memory_store)
        self.motion_detector = MotionDetector()
        
        # Zero-shot setup
        self.current_labels = settings.ZERO_SHOT_LABELS
        self.text_vectors = None
        self._update_text_vectors()

    def _update_text_vectors(self):
        logger.info(f"Encoding text labels: {self.current_labels}")
        self.text_vectors = self.vector_engine.encode_text(self.current_labels)

    def update_labels(self, new_labels: list[str]):
        if new_labels != self.current_labels:
            self.current_labels = new_labels
            self._update_text_vectors()

    def start(self):
        if not self.running:
            logger.info("Starting CLIP-Sentinel System...")
            self.stream_loader.start()
            self.running = True

    def stop(self):
        if self.running:
            logger.info("Stopping CLIP-Sentinel System...")
            self.stream_loader.stop()
            self.running = False

    def process_step(self):
        """
        Performs a single detection step using Hybrid Motion + CLIP logic.
        """
        if not self.running:
            return None

        # 1. Get Frame
        frame = self.stream_loader.get_frame()
        if frame is None:
            return None

        timestamp = time.time()
        
        # 2. Motion Detection
        has_motion, bbox = self.motion_detector.detect(frame)
        
        # Logic: 
        # If motion -> Crop -> Encode -> Classify -> Detect Anomaly (maybe?)
        # If no motion -> Encode Full -> Detect Anomaly (Standard)
        
        # For simplicity and robustness in this demo:
        # We always encode something.
        # If motion: we encode the crop AND classify it.
        # If no motion: we encode the full frame.
        
        # But wait, standard anomaly detection relies on comparing vector history.
        # Mixing "Full Frame Vectors" and "Crop Vectors" in the same memory store (ChromaDB) 
        # is dangerous because they represent different semantic scales.
        # The prompt says: "If no motion: Full frame vector comparison (low freq)."
        # This implies we should separate the streams or usage.
        
        # Ideally, we have two memory stores or just use the same one but with metadata?
        # Or maybe we rely on the fact that crop vectors will look VERY different from full frame vectors,
        # so they naturally won't match well?
        
        # Let's refine the logic for the DEMO:
        # We primarily want to detect objects moving.
        
        detection_mode = "Full"
        roi_img = frame
        
        if has_motion and bbox is not None:
            detection_mode = "ROI"
            x, y, w, h = bbox
            # Ensure crop is within bounds
            h_img, w_img = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            roi_img = frame[y:y+h, x:x+w]
            # Safety check for empty crop
            if roi_img.size == 0:
                roi_img = frame
                detection_mode = "Full (Fallback)"
                bbox = None # Treat as no motion
        
        # 3. Encode
        try:
            vector = self.vector_engine.encode_image(roi_img)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None

        # 4. Zero-shot Classification (Always check, or only on ROI?)
        # Useful even for full frame (e.g. "light change")
        label = "Unknown"
        conf = 0.0
        if self.text_vectors is not None:
            label, conf = self.vector_engine.predict_zero_shot(vector, self.text_vectors, self.current_labels)
        
        # 5. Anomaly Detection
        # We pass the vector to the detector. 
        # Note: If we switch between ROI and Full often, the "Short-term mean" might get noisy.
        # But "Smart Crop" implies we focus on the interesting part.
        
        result = self.detector.detect(vector, timestamp)
        
        # Hybrid Logic Adjustment:
        # If classifier says "shadow" or "light change" with high confidence, override anomaly?
        if result["is_anomaly"]:
             if label in ["shadow", "light change"] and conf > 0.5:
                 result["is_anomaly"] = False
                 result["reason"] = f"Ignored ({label}: {conf:.2f})"
             else:
                 # Enhance reason with classification
                 result["reason"] += f" [{label}: {conf:.2f}]"
        
        if result["is_anomaly"]:
            logger.warning(f"ALERT: {result['reason']}")
        else:
            logger.info(f"Status: {result['reason']}")

        # 6. Update Memory
        # Only update memory if it's NOT a trivial event like shadow?
        # Or always update to learn?
        # If we update with ROI vectors, we pollute the history if we switch back to full frame.
        # This is the tricky part of "Hybrid".
        # For this demo, let's update. The vector space is high dimensional.
        self.memory_store.add_record(vector, timestamp, result["is_anomaly"])

        return {
            "frame": frame,
            "timestamp": timestamp,
            "is_anomaly": result["is_anomaly"],
            "reason": result["reason"],
            "sim_short": result["sim_short"],
            "sim_long": result["sim_long"],
            "bbox": bbox, # For UI visualization
            "label": label,
            "confidence": conf,
            "mode": detection_mode
        }
