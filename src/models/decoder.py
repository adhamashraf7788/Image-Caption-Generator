"""Decoder implementations."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseDecoder


class DecoderLSTM(BaseDecoder):
    """Image-as-first-token LSTM decoder (Show and Tell style baseline)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_embed: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        word_embeds = self.embedding(input_seq)
        image_step = image_embed.unsqueeze(1)

        lstm_input = torch.cat([image_step, word_embeds], dim=1)
        lstm_out, _ = self.lstm(lstm_input)
        logits = self.fc(lstm_out)

        return logits[:, 1:, :]

    @torch.no_grad()
    def generate(
        self,
        image_embed: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int,
    ) -> list[int]:
        assert image_embed.size(0) == 1, "generate() supports batch_size=1"
        device = image_embed.device

        image_step = image_embed.unsqueeze(1)
        _, hidden = self.lstm(image_step)

        current_token = torch.tensor([[start_idx]], device=device)
        generated: list[int] = []

        for _ in range(max_len):
            word_embed = self.embedding(current_token)
            lstm_out, hidden = self.lstm(word_embed, hidden)
            logits = self.fc(lstm_out.squeeze(1))
            predicted_idx = int(logits.argmax(dim=-1).item())

            if predicted_idx == end_idx:
                break

            generated.append(predicted_idx)
            current_token = torch.tensor([[predicted_idx]], device=device)

        return generated