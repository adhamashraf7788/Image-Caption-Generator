"""PyTorch Dataset for the caption model.

Each item = one (image, caption) pair, NOT one image. Since each image has 5
captions, an image appears up to 5 times across items in a split -- this is
correct and expected (more training signal per image, standard practice).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import add_special_tokens, clean_caption
from src.data.vocabulary import Vocabulary


class CaptionDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        features_path: str | Path,
        vocab: Vocabulary,
        max_len: int = 35,
    ):
        self.df = pd.read_csv(csv_path)
        self.features: dict[str, torch.Tensor] = torch.load(features_path, weights_only=True)
        self.vocab = vocab
        self.max_len = max_len

        missing = set(self.df["image"].unique()) - set(self.features.keys())
        if missing:
            raise ValueError(
                f"{len(missing)} images in {csv_path} have no cached feature "
                f"(run scripts/extract_features.py first). Example: {next(iter(missing))}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_filename = row["image"]
        raw_caption = row["caption"]

        image_feature = self.features[image_filename]  # (feature_dim,)

        tokens = add_special_tokens(clean_caption(raw_caption))
        ids = self.vocab.numericalize(tokens)

        # truncate (rare) or pad to fixed length
        ids = ids[: self.max_len]
        pad_len = self.max_len - len(ids)
        ids = ids + [self.vocab.pad_idx] * pad_len

        full_seq = torch.tensor(ids, dtype=torch.long)  # (max_len,)
        input_seq = full_seq[:-1]  # everything except last token
        target_seq = full_seq[1:]  # everything except first token (<start>)

        return {
            "image_feature": image_feature,
            "input_seq": input_seq,
            "target_seq": target_seq,
            "image_filename": image_filename,
            "raw_caption": raw_caption,
        }