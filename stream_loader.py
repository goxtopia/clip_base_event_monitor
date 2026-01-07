import cv2
import threading
import time
from loguru import logger
from typing import Optional
import numpy as np

class StreamLoader:
    def __init__(self, source: str, sample_rate: float = 1.0):
        self.source = source
        self.sample_rate = sample_rate
        self.interval = 1.0 / sample_rate
        self.running = False
        self.thread = None
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.last_sample_time = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info(f"StreamLoader started for {self.source}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("StreamLoader stopped")

    def _update_loop(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame or end of stream")
                # If it's a file, loop it for testing purposes? Or just stop?
                # For RTSP we might want to reconnect. For now, stop.
                # If testing with file, let's reset to beginning to simulate continuous stream
                if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                     continue
                else:
                    break

            # Drop frames logic: we only care about the latest frame available
            # However, we only need to update `latest_frame` if enough time has passed
            # actually, the independent thread should constantly read to clear buffer,
            # but only store the frame if we need it? 
            # Or better: Constantly read to clear RTSP buffer, and update latest_frame always.
            # The consumer determines sampling rate.
            
            with self.lock:
                self.latest_frame = frame
            
            # Small sleep to prevent busy loop if reading is too fast (e.g. file)
            # For RTSP, read() blocks until next frame usually.
            time.sleep(0.005) 

        cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame if the sampling interval has passed since last sample.
        Returns None if no new frame or not enough time passed.
        """
        current_time = time.time()
        if current_time - self.last_sample_time < self.interval:
            return None
        
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        
        self.last_sample_time = current_time
        return frame
