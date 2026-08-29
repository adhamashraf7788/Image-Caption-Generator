
"""Encoder implementations.

Note: ResNet50 itself is NOT run here during training -- features are

precomputed/cached by scripts/extract_features.py (see src/features/extractor.py).

This module only holds the small trainable projection layer that maps cached

CNN features (e.g. 2048-d) down to the shared embedding space (e.g. 256-d).

"""

from __future__ import annotations

import torch

import torch.nn as nn

from src.models.base import BaseEncoder

class ResNet50Encoder(BaseEncoder):

    """Linear projection of cached, frozen ResNet50 features -> embed_dim."""

    def __init__(self, feature_dim: int = 2048, embed_dim: int = 256, freeze: bool = True, **kwargs):

        super().__init__()

        self.freeze = freeze

        self.linear = nn.Linear(feature_dim, embed_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:

        x = self.linear(image_features)

        x = self.relu(x)

        x = self.dropout(x)

        return x

class ResNet50AttentionEncoder(BaseEncoder):

    """Projects a SPATIAL GRID of cached, frozen ResNet50 features -> embed_dim.

    Unlike ResNet50Encoder (which projects one pooled global vector), this

    projects each of the 49 spatial grid positions independently, preserving

    per-region information for an attention-based decoder to attend over.

    """

    def __init__(self, feature_dim: int = 2048, embed_dim: int = 256, freeze: bool = True, **kwargs):

        super().__init__()

        self.freeze = freeze

        self.linear = nn.Linear(feature_dim, embed_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:

        # image_features: (batch, num_pixels, feature_dim) -- num_pixels=49 for a 7x7 grid

        x = self.linear(image_features)

        x = self.relu(x)

        x = self.dropout(x)

        return x

