"""Training entrypoint.

Usage:
    python -m src.training.train --config configs/base_resnet_lstm.yaml
    python -m src.training.train --config configs/base_resnet_lstm.yaml --resume
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CaptionDataset
from src.data.vocabulary import Vocabulary
from src.models.caption_model import CaptionModel
from src.training.trainer import Trainer
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main(config: dict, resume: bool) -> None:
    set_seed(config["seed"])
    device = resolve_device(config["training"]["device"])
    print(f"Using device: {device}")

    processed_dir = Path(config["paths"]["processed_dir"])
    features_dir = Path(config["paths"]["features_dir"])
    encoder_type = config["encoder"]["type"]
    features_path = features_dir / f"{encoder_type}_features.pt"

    vocab = Vocabulary.load(processed_dir / "vocab.json")
    print(f"Vocab size: {len(vocab)}")

    max_len = config["vocab"]["max_len"]
    train_dataset = CaptionDataset(processed_dir / "train.csv", features_path, vocab, max_len)
    val_dataset = CaptionDataset(processed_dir / "val.csv", features_path, vocab, max_len)
    print(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")

    train_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    model = CaptionModel.from_config(config, vocab_size=len(vocab))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        vocab_size=len(vocab),
        device=device,
    )

    resume_path = trainer.run_dir / "last_checkpoint.pt" if resume else None
    trainer.fit(resume_from=resume_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config, resume=args.resume)