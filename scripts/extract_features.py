"""Extract and cache frozen CNN features for every unique image.

Run ONCE before training. Training reads these cached features instead of
re-running the CNN every epoch (the CNN is frozen, so its output for a given
image never changes -- recomputing it every epoch would be pure waste).

Usage:
    python scripts/extract_features.py --config configs/base_resnet_lstm.yaml

Reads:
    <paths.processed_dir>/{train,val,test}.csv
    <paths.raw_images_dir>/<image filename>

Writes:
    <paths.features_dir>/<encoder_type>_features.pt   (dict: filename -> (feature_dim,) tensor)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.preprocessing import get_image_transform, load_image
from src.features.extractor import ResNet50FeatureExtractor, ResNet50SpatialFeatureExtractor
from src.utils.config import load_config

EXTRACTOR_REGISTRY = {
    "resnet50": ResNet50FeatureExtractor,
    "resnet50_spatial": ResNet50SpatialFeatureExtractor,
}

class _ImageOnlyDataset(Dataset):
    """Loads raw images for feature extraction (no captions needed here)."""

    def __init__(self, image_filenames: list[str], images_dir: Path):
        self.image_filenames = image_filenames
        self.images_dir = images_dir
        self.transform = get_image_transform(train=False)

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        filename = self.image_filenames[idx]
        image = load_image(str(self.images_dir / filename))
        return filename, self.transform(image)


def extract_features(config: dict) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    images_dir = Path(config["paths"]["raw_images_dir"])
    features_dir = Path(config["paths"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    encoder_type = config["encoder"]["type"]
    if encoder_type not in EXTRACTOR_REGISTRY:
        raise ValueError(f"No feature extractor registered for encoder type '{encoder_type}'")

    all_images: set[str] = set()
    for split in ["train", "val", "test"]:
        df = pd.read_csv(processed_dir / f"{split}.csv")
        all_images.update(df["image"].unique().tolist())
    all_images = sorted(all_images)

    device = "cuda" if torch.cuda.is_available() and config["training"]["device"] != "cpu" else "cpu"
    print(f"Extracting features for {len(all_images)} images on device={device}")

    extractor = EXTRACTOR_REGISTRY[encoder_type](device=device)
    dataset = _ImageOnlyDataset(all_images, images_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    features: dict[str, torch.Tensor] = {}
    for filenames, image_tensors in tqdm(loader, desc="Extracting features"):
        batch_features = extractor.extract(image_tensors)
        for filename, feat in zip(filenames, batch_features):
            features[filename] = feat

    out_path = features_dir / f"{encoder_type}_features.pt"
    torch.save(features, out_path)
    print(f"Saved {len(features)} feature vectors to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    extract_features(config)