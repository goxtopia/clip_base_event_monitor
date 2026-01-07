import chromadb
from collections import deque
import numpy as np
from loguru import logger
from datetime import datetime
from config import settings
import uuid

class MemoryStore:
    def __init__(self):
        logger.info(f"Initializing ChromaDB at {settings.DB_PATH}")
        self.client = chromadb.PersistentClient(path=settings.DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # In-memory short-term buffer
        self.short_term_buffer = deque(maxlen=settings.HISTORY_WINDOW_SIZE)

    def add_record(self, vector: list[float], timestamp: float, is_anomaly: bool):
        """
        Add a record to both short-term memory and long-term storage (ChromaDB).
        """
        # Add to short-term buffer
        self.short_term_buffer.append(vector)

        # Add to ChromaDB
        dt = datetime.fromtimestamp(timestamp)
        metadata = {
            "timestamp": timestamp,
            "day_of_week": dt.weekday(), # 0=Monday, 6=Sunday
            "hour": dt.hour,
            "is_anomaly": is_anomaly
        }
        
        try:
            self.collection.add(
                embeddings=[vector],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            logger.error(f"Failed to add record to ChromaDB: {e}")

    def get_short_term_mean(self) -> np.ndarray:
        """
        Calculate the mean vector of the short-term buffer.
        """
        if not self.short_term_buffer:
            return None
        
        vectors = np.array(self.short_term_buffer)
        return np.mean(vectors, axis=0)

    def get_short_term_vectors(self) -> np.ndarray:
        """
        Return short-term vectors as a numpy array.
        """
        if not self.short_term_buffer:
            return None
        return np.array(self.short_term_buffer)

    def query_long_term_history(self, vector: list[float], timestamp: float, n_results: int = 3) -> float:
        """
        Query long-term history for similar vectors from yesterday or last week at the same hour.
        Returns the average similarity score.
        """
        dt = datetime.fromtimestamp(timestamp)
        current_hour = dt.hour
        current_day_of_week = dt.weekday()

        # Target days: Yesterday and Last Week
        # Yesterday: (current - 1) % 7
        # Last Week: same day index (current) -- wait, prompt says "Same day last week" usually implies 7 days ago, so same day_of_week.
        # Prompt says: "day_of_week equals (today - 1) or (today - 7)"
        # Note: (today - 7) is same as today in terms of weekday index (0-6).
        # But if we strictly follow prompt logic of checking 'day_of_week' metadata:
        # If today is Monday (0), yesterday is Sunday (6).
        # If today is Monday (0), last week Monday is (0).
        
        target_days = [
            (current_day_of_week - 1) % 7, # Yesterday
            current_day_of_week            # Last Week (Same weekday)
        ]

        # Construct filter
        # ChromaDB 'where' clause supports basic operators.
        # We need (day_of_week IN target_days) AND (hour == current_hour)
        # ChromaDB where format: {"$and": [{...}, {...}]}
        # "day_of_week" IN is usually supported via multiple ORs or $in operator if available.
        # Let's check ChromaDB docs or assume standard operators.
        # ChromaDB supports $in since recent versions. If not, use $or.
        
        where_filter = {
            "$and": [
                {"hour": {"$eq": current_hour}},
                {"day_of_week": {"$in": target_days}}
            ]
        }

        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=n_results,
                where=where_filter
            )
            
            # results['distances'] contains cosine distance (if default space is cosine)
            # wait, ChromaDB default is usually L2 or Cosine? 
            # We should probably check collection creation or assume default.
            # Default is usually L2 (Squared L2).
            # To get cosine similarity from distance depends on metric.
            # If we initialized collection without metadata, it uses defaults.
            # We should specify cosine space if we want cosine similarity.
            
            # Let's re-initialize collection with cosine space in __init__ if possible,
            # or handle distance conversion.
            # If default is L2, and vectors are normalized, L2 distance relates to cosine similarity:
            # L2^2 = 2 - 2*cos_sim => cos_sim = 1 - L2^2 / 2
            
            # However, prompt asks for Cosine Similarity.
            # Let's check if we can specify metadata={"hnsw:space": "cosine"}
            
            distances = results['distances'][0]
            if not distances:
                return 1.0 # No history, assume normal (high similarity)
                
            # If space is 'cosine', distance = 1 - similarity.
            # So similarity = 1 - distance.
            
            # I will ensure collection is created with cosine space.
            avg_distance = np.mean(distances)
            return 1.0 - avg_distance

        except Exception as e:
            logger.warning(f"Query failed (possibly empty DB): {e}")
            return 1.0 # Fail open (assume normal)
