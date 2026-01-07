import time
from collections import deque
from loguru import logger

from config import settings
from stream_loader import StreamLoader
from detector import AnomalyDetector
from memory_store import MemoryStore
from motion_detector import MotionDetector
from videomae_engine import VideoMAEEngine


class VideoMAESentinelSystem:
    def __init__(self):
        self.running = False
        self.stream_loader = StreamLoader(
            source=settings.RTSP_URL,
            sample_rate=settings.VIDEOMAE_SAMPLE_RATE
        )
        self.motion_detector = MotionDetector()
        self.video_engine = VideoMAEEngine()
        self.memory_store = MemoryStore()
        self.detector = AnomalyDetector(self.memory_store)
        self.clip_size = settings.VIDEOMAE_CLIP_SIZE
        self.frame_buffer = deque(maxlen=self.clip_size)

    def start(self):
        if not self.running:
            logger.info("Starting VideoMAE Sentinel System...")
            self.stream_loader.start()
            self.running = True

    def stop(self):
        if self.running:
            logger.info("Stopping VideoMAE Sentinel System...")
            self.stream_loader.stop()
            self.running = False

    def process_step(self):
        if not self.running:
            return None

        frame = self.stream_loader.get_frame()
        if frame is None:
            return None

        timestamp = time.time()
        has_motion, _ = self.motion_detector.detect(frame)

        self.frame_buffer.append(frame)
        clip_ready = len(self.frame_buffer) == self.clip_size

        if not clip_ready:
            return {
                "frame": frame,
                "timestamp": timestamp,
                "has_motion": has_motion,
                "clip_ready": False,
                "status": "Warming up",
                "sim_short": None,
                "sim_long": None,
                "is_anomaly": False
            }

        if not has_motion:
            return {
                "frame": frame,
                "timestamp": timestamp,
                "has_motion": False,
                "clip_ready": True,
                "status": "No motion",
                "sim_short": None,
                "sim_long": None,
                "is_anomaly": False
            }

        try:
            vector = self.video_engine.encode_clip(list(self.frame_buffer))
        except Exception as e:
            logger.error(f"VideoMAE encoding failed: {e}")
            return None

        result = self.detector.detect(vector, timestamp)
        self.memory_store.add_record(vector, timestamp, result["is_anomaly"])

        return {
            "frame": frame,
            "timestamp": timestamp,
            "has_motion": True,
            "clip_ready": True,
            "status": result["reason"],
            "sim_short": result["sim_short"],
            "sim_long": result["sim_long"],
            "is_anomaly": result["is_anomaly"]
        }
