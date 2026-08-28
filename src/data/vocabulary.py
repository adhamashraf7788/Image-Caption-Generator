"""Vocabulary: word <-> index mapping built from training captions only.

Special tokens are reserved at fixed indices by convention:
    <pad> = 0   (padding_idx for nn.Embedding, ignore_index for loss)
    <start> = 1
    <end> = 2
    <unk> = 3
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.preprocessing import clean_caption

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


class Vocabulary:
    def __init__(self, word2idx: dict[str, int], min_freq: int = 5):
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}
        self.min_freq = min_freq

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def start_idx(self) -> int:
        return self.word2idx[START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.word2idx[END_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @classmethod
    def build(cls, captions: list[str], min_freq: int = 5) -> "Vocabulary":
        """Build vocabulary from a list of raw caption strings (training split only)."""
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(clean_caption(caption))

        word2idx: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        next_idx = len(SPECIAL_TOKENS)
        for word, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if count >= min_freq:
                word2idx[word] = next_idx
                next_idx += 1

        return cls(word2idx, min_freq=min_freq)

    def numericalize(self, tokens: list[str]) -> list[int]:
        """Convert a token list (already including <start>/<end>) to indices."""
        return [self.word2idx.get(tok, self.unk_idx) for tok in tokens]

    def decode(self, indices: list[int], strip_special: bool = True) -> list[str]:
        """Convert indices back to words. Optionally drop special tokens."""
        words = [self.idx2word.get(idx, UNK_TOKEN) for idx in indices]
        if strip_special:
            words = [w for w in words if w not in SPECIAL_TOKENS]
        return words

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"word2idx": self.word2idx, "min_freq": self.min_freq}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, "r") as f:
            payload = json.load(f)
        return cls(payload["word2idx"], min_freq=payload.get("min_freq", 5))