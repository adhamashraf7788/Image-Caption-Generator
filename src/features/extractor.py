"""Frozen CNN feature extractor.

Used by:
    - scripts/extract_features.py (batch, offline, caches to disk)
    - src/inference/predict.py (single image, online, at request time)
Both call the SAME class so train and inference features are computed identically.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50FeatureExtractor:
    """Wraps a frozen, pretrained ResNet50 with its classification head removed."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        # drop the final fc layer -> output is the (2048,) pooled feature
        self.model = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """image_tensor: (batch, 3, 224, 224) -> returns (batch, 2048)."""
        image_tensor = image_tensor.to(self.device)
        features = self.model(image_tensor)  # (batch, 2048, 1, 1)
        return features.squeeze(-1).squeeze(-1).cpu()  # (batch, 2048)