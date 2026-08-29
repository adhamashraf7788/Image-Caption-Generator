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
    """Wraps a frozen, pretrained ResNet50 with its classification head removed.

    Produces a single pooled (2048,) feature vector per image -- used by the
    baseline/regularized DecoderLSTM, which relies on one global image vector.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        self.model = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """image_tensor: (batch, 3, 224, 224) -> returns (batch, 2048)."""
        image_tensor = image_tensor.to(self.device)
        features = self.model(image_tensor)
        return features.squeeze(-1).squeeze(-1).cpu()


class ResNet50SpatialFeatureExtractor:
    """Wraps a frozen, pretrained ResNet50 with BOTH the classification head
    AND the final average-pooling layer removed.

    Produces a spatial GRID of features (7x7 regions, 2048-d each) instead of
    one pooled global vector -- required for attention-based decoders, which
    need to attend to different image regions at each generation step.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        self.model = nn.Sequential(*list(resnet.children())[:-2]).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """image_tensor: (batch, 3, 224, 224) -> returns (batch, 49, 2048).

        49 = 7x7 spatial grid positions, each a 2048-d feature vector.
        """
        image_tensor = image_tensor.to(self.device)
        features = self.model(image_tensor)
        batch, channels, h, w = features.shape
        features = features.view(batch, channels, h * w)
        features = features.permute(0, 2, 1)
        return features.cpu()
