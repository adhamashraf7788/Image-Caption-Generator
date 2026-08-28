"""CaptionModel: wires encoder + decoder together.

This class never needs to change when you swap architectures -- it only
relies on the BaseEncoder/BaseDecoder contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseDecoder, BaseEncoder
from src.models.registry import build_decoder, build_encoder


class CaptionModel(nn.Module):
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_config(cls, config: dict, vocab_size: int) -> "CaptionModel":
        encoder = build_encoder(config)
        decoder = build_decoder(config, vocab_size)
        return cls(encoder, decoder)

    def forward(self, image_features: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        image_embed = self.encoder(image_features)
        return self.decoder.forward(image_embed, input_seq)

    @torch.no_grad()
    def generate(
        self,
        image_features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int,
    ) -> list[int]:
        self.eval()
        image_embed = self.encoder(image_features)
        return self.decoder.generate(image_embed, start_idx, end_idx, max_len)