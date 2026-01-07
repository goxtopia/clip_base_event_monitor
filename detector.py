import numpy as np
from loguru import logger
from memory_store import MemoryStore
from config import settings

class AnomalyDetector:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        # Vectors are already normalized by VectorEngine
        return float(np.dot(v1, v2))

    def detect(self, current_vector: list[float], timestamp: float) -> dict:
        """
        Detects anomaly based on short-term and long-term history.
        Returns: dict with keys 'is_anomaly', 'reason', 'sim_short', 'sim_long'
        """
        curr_vec_np = np.array(current_vector)
        
        # 1. Short-term Check
        mean_short_term = self.memory_store.get_short_term_mean()
        
        sim_short = 1.0
        if mean_short_term is not None:
            sim_short = self.cosine_similarity(curr_vec_np, mean_short_term)
        
        # 2. Long-term Check (Double Validation)
        sim_long = self.memory_store.query_long_term_history(current_vector, timestamp)
        
        logger.debug(f"Sim Short: {sim_short:.4f}, Sim Long: {sim_long:.4f}")

        is_anomaly = False
        reason = "Normal"

        if sim_short >= settings.SIMILARITY_THRESHOLD:
             reason = "Normal (Short-term stable)"
        elif sim_long < settings.SIMILARITY_THRESHOLD:
            # Both short-term (change happened) AND long-term (unusual for this time) are low
            is_anomaly = True
            reason = f"Anomaly Detected! (Short: {sim_short:.2f}, Long: {sim_long:.2f})"
        else:
            # Short-term change, but matches historical pattern
            reason = f"Normal (Historical Match: {sim_long:.2f})"

        return {
            "is_anomaly": is_anomaly,
            "reason": reason,
            "sim_short": sim_short,
            "sim_long": sim_long
        }
