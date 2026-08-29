"""Abstract base classes.

Any encoder/decoder you add later (attention, transformer, EfficientNet...)
MUST implement these methods with these exact signatures. That's what lets
Trainer, evaluate.py, and predict.py stay architecture-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Takes precomputed CNN feature vectors and projects them to embed_dim."""

    @abstractmethod
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """image_features: (batch, feature_dim) -> returns (batch, embed_dim)."""
        raise NotImplementedError


class BaseDecoder(nn.Module, ABC):
    """Consumes image embedding + token sequence, predicts next-word logits."""

    @abstractmethod
    def forward(self, image_embed: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        """
        image_embed: (batch, embed_dim)
        input_seq:   (batch, seq_len) token indices, teacher-forced input
        returns:     (batch, seq_len, vocab_size) logits aligned with target_seq
        """
        raise NotImplementedError

    #for greedy search
    @abstractmethod
    def generate(
        self,
        image_embed: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int,
    ) -> list[int]:
        """Autoregressive greedy generation for a SINGLE image (batch=1).

        Returns a list of generated token indices, excluding <start> and <end>.
        """
        raise NotImplementedError
    
    #for beam search
    def generate_beam(
        self,
        image_embed: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int,
        beam_width: int = 3,
    ) -> list[int]:
        """Beam search generation for a SINGLE image (batch=1).

        Optional: not marked @abstractmethod so existing decoders that only
        implement greedy generate() remain valid; override in subclasses
        that support beam search (see DecoderLSTM).
        """
        raise NotImplementedError("This decoder does not implement beam search.")