import torch
import open_clip
from PIL import Image
import numpy as np
from loguru import logger
import cv2
from config import settings

class VectorEngine:
    def __init__(self):
        logger.info(f"Loading model: {settings.MODEL_NAME} ({settings.PRETRAINED_WEIGHTS}) on {settings.DEVICE}")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                settings.MODEL_NAME, 
                pretrained=settings.PRETRAINED_WEIGHTS,
                device=settings.DEVICE
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(settings.MODEL_NAME)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def encode_image(self, frame: np.ndarray) -> list[float]:
        """
        Encodes an OpenCV image (numpy array) into a normalized feature vector.
        """
        # Convert BGR (OpenCV) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Preprocess
        image_input = self.preprocess(pil_image).unsqueeze(0).to(settings.DEVICE)

        # Inference
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Convert to list
        return image_features.cpu().numpy()[0].tolist()

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encodes a list of text strings into normalized feature vectors.
        Returns a Tensor on the configured device.
        """
        try:
            tokens = self.tokenizer(texts).to(settings.DEVICE)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return None

    def predict_zero_shot(self, image_vector: list[float], text_vectors: torch.Tensor, labels: list[str]) -> tuple[str, float]:
        """
        Classifies the image vector against text vectors.
        Returns: (best_label, confidence_score)
        """
        if text_vectors is None:
            return "Unknown", 0.0

        image_features = torch.tensor(image_vector, device=settings.DEVICE).unsqueeze(0)
        
        # Cosine similarity
        with torch.no_grad():
            similarity = (100.0 * image_features @ text_vectors.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            
        return labels[indices[0]], values[0].item()
