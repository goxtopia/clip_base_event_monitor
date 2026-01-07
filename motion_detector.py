import cv2
import numpy as np
from config import settings

class MotionDetector:
    def __init__(self):
        # Using MOG2 for background subtraction
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        """
        Detects motion in the frame.
        Returns: (has_motion, bounding_box)
        bounding_box: (x, y, w, h) of the largest moving area.
        """
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Remove noise (shadows are gray in MOG2, we can filter them or keep them)
        # Threshold to remove shadows (usually 127) if detectShadows=True
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological opening to remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_box = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > settings.MIN_CONTOUR_AREA:
                if area > max_area:
                    max_area = area
                    best_box = cv2.boundingRect(contour)
        
        if best_box is not None:
            return True, best_box
        
        return False, None
