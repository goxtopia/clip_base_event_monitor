import cv2
import torch
from PIL import Image
from loguru import logger
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
from config import settings


class VideoMAEEngine:
    def __init__(self):
        logger.info(f"Loading VideoMAE model: {settings.VIDEOMAE_MODEL_NAME} on {settings.DEVICE}")
        self.config = AutoConfig.from_pretrained(
            settings.VIDEOMAE_MODEL_NAME,
            trust_remote_code=True,
        )
        self.processor = VideoMAEImageProcessor.from_pretrained(settings.VIDEOMAE_MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            settings.VIDEOMAE_MODEL_NAME,
            config=self.config,
            trust_remote_code=True,
        ).to(settings.DEVICE)
        self.model.eval()

    def encode_clip(self, frames: list) -> list[float]:
        """
        Encodes a list of frames (BGR numpy arrays) into a normalized feature vector.
        """
        if not frames:
            return []

        rgb_frames = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames
        ]
        inputs = self.processor(rgb_frames, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)
        inputs = {key: value.to(settings.DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            if isinstance(outputs, torch.Tensor):
                clip_features = outputs
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                clip_features = outputs.pooler_output
            else:
                clip_features = outputs.last_hidden_state.mean(dim=1)
            clip_features = torch.nn.functional.normalize(clip_features, dim=-1)

        return clip_features.cpu().numpy()[0].tolist()
