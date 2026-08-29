"""Predictor: single source of truth for generating a caption from an image.

Used by:
    - src/evaluation/evaluate.py (batch, over the test set)
    - app/api.py, app/streamlit_app.py (single uploaded image)
Never duplicate this logic in the app layer -- always go through this class,
so training-time preprocessing and inference-time preprocessing can't drift.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.preprocessing import get_image_transform
from src.data.vocabulary import Vocabulary
from src.features.extractor import ResNet50FeatureExtractor, ResNet50SpatialFeatureExtractor
from src.models.caption_model import CaptionModel

FEATURE_EXTRACTOR_REGISTRY = {
    "resnet50": ResNet50FeatureExtractor,
    "resnet50_spatial": ResNet50SpatialFeatureExtractor,
}

class Predictor:
    def __init__(self, checkpoint_path: str | Path, vocab_path: str | Path, device: str = "cpu"):
        self.device = device
        self.vocab = Vocabulary.load(vocab_path)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = checkpoint["config"]
        vocab_size = checkpoint["vocab_size"]

        self.model = CaptionModel.from_config(self.config, vocab_size=vocab_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        encoder_type = self.config["encoder"]["type"]
        self.feature_extractor = FEATURE_EXTRACTOR_REGISTRY[encoder_type](device=device)
        self.transform = get_image_transform(train=False)
        self.max_len = self.config["vocab"]["max_len"]
        self.decoding = self.config.get("inference", {}).get("decoding", "greedy")
        self.beam_width = self.config.get("inference", {}).get("beam_width", 3)

    def predict(self, image: str | Path | Image.Image, decoding: str | None = None) -> str:
        """Generate a caption for a single image (path or already-loaded PIL Image).

        `decoding` overrides the config's default ("greedy" or "beam") for this call.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0)
        image_feature = self.feature_extractor.extract(image_tensor)
        image_feature = image_feature.to(self.device)

        generated_ids = self.model.generate(
            image_feature,
            start_idx=self.vocab.start_idx,
            end_idx=self.vocab.end_idx,
            max_len=self.max_len,
            decoding=decoding or self.decoding,
            beam_width=self.beam_width,
        )
        words = self.vocab.decode(generated_ids, strip_special=True)
        return " ".join(words)