"""Build vocabulary from the TRAIN split only (never val/test -- avoids leakage).

Usage:
    python scripts/build_vocab.py --config configs/base_resnet_lstm.yaml

Reads:
    <paths.processed_dir>/train.csv

Writes:
    <paths.processed_dir>/vocab.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.vocabulary import Vocabulary
from src.utils.config import load_config


def build_vocab(config: dict) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")

    vocab = Vocabulary.build(
        captions=train_df["caption"].tolist(),
        min_freq=config["vocab"]["min_freq"],
    )

    out_path = processed_dir / "vocab.json"
    vocab.save(out_path)

    print(f"Vocabulary size (incl. special tokens): {len(vocab)}")
    print(f"min_freq={config['vocab']['min_freq']}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    build_vocab(config)