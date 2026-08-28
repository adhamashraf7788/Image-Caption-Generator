"""Split dataset by unique image into train/val/test (no leakage across splits).

Usage:
    python scripts/split_dataset.py --config configs/base_resnet_lstm.yaml

Reads:
    <paths.captions_file>   e.g. data/raw/captions.txt   (columns: image, caption)

Writes:
    <paths.processed_dir>/train.csv
    <paths.processed_dir>/val.csv
    <paths.processed_dir>/test.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from src.utils.config import load_config


def split_dataset(config: dict) -> None:
    captions_file = Path(config["paths"]["captions_file"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(captions_file)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "image" in df.columns and "caption" in df.columns, (
        f"Expected columns 'image' and 'caption' in {captions_file}, got {list(df.columns)}"
    )

    unique_images = sorted(df["image"].unique().tolist())
    rng = random.Random(config["seed"])
    rng.shuffle(unique_images)

    n = len(unique_images)
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_images = set(unique_images[:n_train])
    val_images = set(unique_images[n_train : n_train + n_val])
    test_images = set(unique_images[n_train + n_val :])

    train_df = df[df["image"].isin(train_images)].reset_index(drop=True)
    val_df = df[df["image"].isin(val_images)].reset_index(drop=True)
    test_df = df[df["image"].isin(test_images)].reset_index(drop=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(f"Total unique images: {n}")
    print(f"Train: {len(train_images)} images / {len(train_df)} captions")
    print(f"Val:   {len(val_images)} images / {len(val_df)} captions")
    print(f"Test:  {len(test_images)} images / {len(test_df)} captions")
    print(f"Saved to {processed_dir}/{{train,val,test}}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    split_dataset(config)