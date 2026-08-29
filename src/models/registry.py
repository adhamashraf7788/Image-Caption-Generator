"""Registry: maps config-file type strings to model classes.

To add a new architecture later:
    1. Write the class in encoder.py or decoder.py, subclassing BaseEncoder/BaseDecoder.
    2. Add one line to the relevant registry dict below.
    3. Add a new config YAML with `encoder.type` / `decoder.type` set to your new key.
No other file needs to change.
"""
from __future__ import annotations

from src.models.base import BaseDecoder, BaseEncoder
from src.models.decoder import DecoderAttentionLSTM, DecoderLSTM
from src.models.encoder import ResNet50AttentionEncoder, ResNet50Encoder

ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "resnet50": ResNet50Encoder,
    "resnet50_spatial": ResNet50AttentionEncoder,
}

DECODER_REGISTRY: dict[str, type[BaseDecoder]] = {
    "lstm": DecoderLSTM,
    "attention_lstm": DecoderAttentionLSTM,
}


def build_encoder(config: dict) -> BaseEncoder:
    enc_config = dict(config["encoder"])
    enc_type = enc_config.pop("type")
    if enc_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type '{enc_type}'. Available: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[enc_type](**enc_config)


def build_decoder(config: dict, vocab_size: int) -> BaseDecoder:
    dec_config = dict(config["decoder"])
    dec_type = dec_config.pop("type")
    if dec_type not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder type '{dec_type}'. Available: {list(DECODER_REGISTRY)}")
    return DECODER_REGISTRY[dec_type](vocab_size=vocab_size, **dec_config)
