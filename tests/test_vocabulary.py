from src.data.vocabulary import Vocabulary


def test_special_tokens_have_fixed_indices():
    vocab = Vocabulary.build(["a dog runs"], min_freq=1)
    assert vocab.pad_idx == 0
    assert vocab.start_idx == 1
    assert vocab.end_idx == 2
    assert vocab.unk_idx == 3


def test_min_freq_filters_rare_words():
    captions = ["a dog runs", "a dog runs", "a rare word appears"]
    vocab = Vocabulary.build(captions, min_freq=2)
    assert "dog" in vocab.word2idx  # appears twice, kept
    assert "rare" not in vocab.word2idx  # appears once, dropped


def test_numericalize_unknown_word_maps_to_unk():
    vocab = Vocabulary.build(["a dog runs", "a dog runs"], min_freq=1)
    ids = vocab.numericalize(["a", "totally", "unseen", "word"])
    assert ids[1] == vocab.unk_idx


def test_save_and_load_roundtrip(tmp_path):
    vocab = Vocabulary.build(["a dog runs", "a cat sits"], min_freq=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = Vocabulary.load(path)
    assert loaded.word2idx == vocab.word2idx
    assert len(loaded) == len(vocab)


def test_decode_strips_special_tokens_by_default():
    vocab = Vocabulary.build(["a dog runs"], min_freq=1)
    ids = [vocab.start_idx, vocab.word2idx["a"], vocab.word2idx["dog"], vocab.end_idx]
    decoded = vocab.decode(ids)
    assert decoded == ["a", "dog"]