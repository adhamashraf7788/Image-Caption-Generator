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
        # NOTE: nn.LSTM's own `dropout` arg only applies BETWEEN stacked layers,
        # so it has no effect with num_layers=1 (our baseline). This separate
        # output_dropout applies to the LSTM's output before the final
        # projection, giving a real regularization effect even for a
        # single-layer LSTM.
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_embed: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        # image_embed: (batch, embed_dim)
        # input_seq:   (batch, seq_len) token indices (teacher-forced)
        word_embeds = self.embedding(input_seq)  # (batch, seq_len, embed_dim)
        image_step = image_embed.unsqueeze(1)  # (batch, 1, embed_dim)

        lstm_input = torch.cat([image_step, word_embeds], dim=1)  # (batch, seq_len+1, embed_dim)
        lstm_out, _ = self.lstm(lstm_input)  # (batch, seq_len+1, hidden_dim)
        lstm_out = self.output_dropout(lstm_out)
        logits = self.fc(lstm_out)  # (batch, seq_len+1, vocab_size)

        # drop the output at the image step (index 0) -- it has no word target
        return logits[:, 1:, :]  # (batch, seq_len, vocab_size)

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

    @torch.no_grad()
    def generate_beam(
        self,
        image_embed: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int,
        beam_width: int = 3,
    ) -> list[int]:
        """Beam search decoding for a SINGLE image (batch=1).

        Keeps the top `beam_width` candidate sequences at each step instead of
        committing to a single best word (as generate()/greedy does), which
        can recover from a locally-suboptimal early word choice. No retraining
        required -- this only changes inference-time decoding.
        """
        assert image_embed.size(0) == 1, "generate_beam() supports batch_size=1"
        device = image_embed.device

        image_step = image_embed.unsqueeze(1)  # (1, 1, embed_dim)
        _, init_hidden = self.lstm(image_step)

        # each beam: (token_ids, cumulative_log_prob, hidden_state, finished)
        beams = [([start_idx], 0.0, init_hidden, False)]

        for _ in range(max_len):
            all_candidates = []
            for tokens, log_prob, hidden, finished in beams:
                if finished:
                    all_candidates.append((tokens, log_prob, hidden, finished))
                    continue

                last_token = torch.tensor([[tokens[-1]]], device=device)
                word_embed = self.embedding(last_token)
                lstm_out, new_hidden = self.lstm(word_embed, hidden)
                logits = self.fc(lstm_out.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_indices = log_probs.topk(beam_width)
                for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_tokens = tokens + [idx]
                    new_finished = idx == end_idx
                    all_candidates.append((new_tokens, log_prob + lp, new_hidden, new_finished))

            all_candidates.sort(key=lambda c: c[1], reverse=True)
            beams = all_candidates[:beam_width]

            if all(finished for _, _, _, finished in beams):
                break

        best_tokens, _, _, _ = max(beams, key=lambda c: c[1])
        result = best_tokens[1:]
        if result and result[-1] == end_idx:
            result = result[:-1]
        return result

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau-style) attention over spatial image regions.

    At each decoder step, computes a weighted combination of the encoder's
    spatial feature grid, where the weights depend on the current decoder
    hidden state -- this is what lets the decoder "look at" different image
    regions for different words, instead of relying on one static vector.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(embed_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderAttentionLSTM(BaseDecoder):
    """LSTM decoder with Bahdanau-style attention over spatial image features.

    Unlike DecoderLSTM (which injects one static image vector as the first
    LSTM step), this decoder recomputes a fresh, weighted combination of
    image regions at EVERY generation step, using an LSTMCell in an explicit
    loop (required since each step's attention depends on that step's hidden
    state -- unlike DecoderLSTM, the whole sequence can't be processed in one
    nn.LSTM call).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(embed_dim, hidden_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + embed_dim, hidden_dim)
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)
        self.f_beta = nn.Linear(hidden_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden_state(self, encoder_out: torch.Tensor):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, image_embed: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = image_embed.size(0)
        seq_len = input_seq.size(1)
        vocab_size = self.fc.out_features
        device = image_embed.device

        embeddings = self.embedding(input_seq)
        h, c = self._init_hidden_state(image_embed)

        outputs = torch.zeros(batch_size, seq_len, vocab_size, device=device)

        for t in range(seq_len):
            context, _ = self.attention(image_embed, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            preds = self.fc(self.dropout(h))
            outputs[:, t, :] = preds

        return outputs

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

        h, c = self._init_hidden_state(image_embed)
        current_token = torch.tensor([start_idx], device=device)
        generated: list[int] = []

        for _ in range(max_len):
            word_embed = self.embedding(current_token)
            context, _ = self.attention(image_embed, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([word_embed, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.fc(h)
            predicted_idx = int(logits.argmax(dim=-1).item())

            if predicted_idx == end_idx:
                break

            generated.append(predicted_idx)
            current_token = torch.tensor([predicted_idx], device=device)

        return generated
    
    
    @torch.no_grad()
    def generate_beam(
        self,
        image_embed: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int,
        beam_width: int = 3,
    ) -> list[int]:
        """Beam search decoding for a SINGLE image (batch=1).

        Same beam-search principle as DecoderLSTM.generate_beam(), adapted for
        the LSTMCell + per-step attention loop: each beam carries its own
        (h, c) hidden state, since attention (and therefore the next word
        distribution) depends on each beam's own hidden state, not a shared one.
        """
        assert image_embed.size(0) == 1, "generate_beam() supports batch_size=1"
        device = image_embed.device

        init_h, init_c = self._init_hidden_state(image_embed)

        # each beam: (token_ids, cumulative_log_prob, h, c, finished)
        beams = [([start_idx], 0.0, init_h, init_c, False)]

        for _ in range(max_len):
            all_candidates = []
            for tokens, log_prob, h, c, finished in beams:
                if finished:
                    all_candidates.append((tokens, log_prob, h, c, finished))
                    continue

                last_token = torch.tensor([tokens[-1]], device=device)
                word_embed = self.embedding(last_token)
                context, _ = self.attention(image_embed, h)
                gate = self.sigmoid(self.f_beta(h))
                context = gate * context

                lstm_input = torch.cat([word_embed, context], dim=1)
                new_h, new_c = self.lstm_cell(lstm_input, (h, c))
                logits = self.fc(new_h)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_indices = log_probs.topk(beam_width)
                for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_tokens = tokens + [idx]
                    new_finished = idx == end_idx
                    all_candidates.append((new_tokens, log_prob + lp, new_h, new_c, new_finished))

            all_candidates.sort(key=lambda cand: cand[1], reverse=True)
            beams = all_candidates[:beam_width]

            if all(finished for _, _, _, _, finished in beams):
                break

        best_tokens, _, _, _, _ = max(beams, key=lambda cand: cand[1])
        result = best_tokens[1:]
        if result and result[-1] == end_idx:
            result = result[:-1]
        return result
