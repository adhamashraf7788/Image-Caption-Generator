"""Trainer: train/val loop, LR scheduling, early stopping, checkpointing."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.caption_model import CaptionModel
from src.utils.config import save_config


class Trainer:
    def __init__(
        self,
        model: CaptionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        vocab_size: int,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config["training"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg["lr"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=train_cfg["lr_scheduler_factor"],
            patience=train_cfg["lr_scheduler_patience"],
        )
        self.vocab_size = vocab_size
        self.early_stop_patience = train_cfg["early_stop_patience"]
        self.epochs = train_cfg["epochs"]

        self.run_dir = Path(config["paths"]["models_dir"]) / config["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "training_log.csv"

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(mode=train)
        total_loss = 0.0
        n_batches = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                image_features = batch["image_feature"].to(self.device)
                input_seq = batch["input_seq"].to(self.device)
                target_seq = batch["target_seq"].to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(image_features, input_seq)
                loss = self.criterion(
                    outputs.reshape(-1, self.vocab_size),
                    target_seq.reshape(-1),
                )

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "vocab_size": self.vocab_size,
                "config": {k: v for k, v in self.config.items() if not k.startswith("_")},
            },
            path,
        )

    def load_checkpoint(self, path: Path, resume_training: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if resume_training:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint["val_loss"]

    def fit(self, resume_from: Path | None = None) -> None:
        if resume_from is not None and resume_from.exists():
            print(f"Resuming from checkpoint: {resume_from}")
            self.load_checkpoint(resume_from, resume_training=True)

        save_config(self.config, self.run_dir / "config.yaml")

        write_header = not self.log_path.exists()
        log_file = open(self.log_path, "a", newline="")
        log_writer = csv.writer(log_file)
        if write_header:
            log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.2e}"
            )
            log_writer.writerow([epoch + 1, train_loss, val_loss, current_lr])
            log_file.flush()

            self.save_checkpoint(self.run_dir / "last_checkpoint.pt", epoch, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(self.run_dir / "best_model.pt", epoch, val_loss)
                print(f"  -> new best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        log_file.close()