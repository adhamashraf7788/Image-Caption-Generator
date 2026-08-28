"""Shared preprocessing functions.

These are pure functions used identically at training, validation, test,
and inference time. Never duplicate this logic elsewhere (e.g. inside the
API layer) -- always import from here, so train/inference never drift.
"""
from __future__ import annotations

import re
import string

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def get_image_transform(train: bool = False) -> transforms.Compose:
    """Return the torchvision transform pipeline for ResNet-style encoders.

    Same resize/crop/normalize is used for train, val, test, and inference.
    `train=True` is accepted for future augmentation experiments, but the
    baseline intentionally does not augment (see README: augmenting would
    invalidate the cached, frozen-CNN feature strategy).
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(path: str) -> Image.Image:
    """Load an image from disk and force RGB (some Flickr8k jpgs are grayscale/CMYK)."""
    return Image.open(path).convert("RGB")


def clean_caption(caption: str) -> list[str]:
    """Normalize + tokenize a raw caption string.

    Steps: lowercase -> strip punctuation -> collapse whitespace -> split.
    This is intentionally simple (whitespace tokenization) -- sufficient for
    Flickr8k's short descriptive sentences. Swap in a real tokenizer later
    if experimenting with subword/transformer decoders.
    """
    caption = caption.lower().strip()
    caption = caption.translate(_PUNCT_TABLE)
    caption = _WHITESPACE_RE.sub(" ", caption).strip()
    if not caption:
        return []
    return caption.split(" ")


def add_special_tokens(tokens: list[str]) -> list[str]:
    """Wrap a token list with <start> / <end>."""
    return ["<start>", *tokens, "<end>"]