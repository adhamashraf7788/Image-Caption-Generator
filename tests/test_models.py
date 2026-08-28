import torch

from src.models.caption_model import CaptionModel
from src.utils.config import load_config

def _base_config():
    config = load_config("configs/base_resnet_lstm.yaml")
    # keep it tiny/fast for tests -- encoder embed_dim must match decoder embed_dim
    config["encoder"]["embed_dim"] = 16
    config["decoder"]["embed_dim"] = 16
    config["decoder"]["hidden_dim"] = 32
    return config


def test_forward_output_shape():
    config = _base_config()
    vocab_size = 50
    model = CaptionModel.from_config(config, vocab_size=vocab_size)

    batch, seq_len = 4, 7
    image_features = torch.randn(batch, 2048)
    input_seq = torch.randint(0, vocab_size, (batch, seq_len))

    out = model(image_features, input_seq)
    assert out.shape == (batch, seq_len, vocab_size)


def test_generate_returns_list_of_ints_within_max_len():
    config = _base_config()
    vocab_size = 50
    model = CaptionModel.from_config(config, vocab_size=vocab_size)

    image_feature = torch.randn(1, 2048)
    generated = model.generate(image_feature, start_idx=1, end_idx=2, max_len=10)

    assert isinstance(generated, list)
    assert len(generated) <= 10
    assert all(isinstance(x, int) for x in generated)


def test_generate_stops_at_end_token():
    """With end_idx forced to be the argmax at step 1 (untrainable check),
    at minimum generate() must never exceed max_len."""
    config = _base_config()
    vocab_size = 50
    model = CaptionModel.from_config(config, vocab_size=vocab_size)

    image_feature = torch.randn(1, 2048)
    generated = model.generate(image_feature, start_idx=1, end_idx=2, max_len=5)
    assert len(generated) <= 5