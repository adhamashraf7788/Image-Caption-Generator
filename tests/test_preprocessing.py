from src.data.preprocessing import add_special_tokens, clean_caption


def test_clean_caption_lowercases_and_strips_punctuation():
    result = clean_caption("A Dog Runs, in THE grass!!")
    assert result == ["a", "dog", "runs", "in", "the", "grass"]


def test_clean_caption_collapses_whitespace():
    result = clean_caption("a   dog    runs")
    assert result == ["a", "dog", "runs"]


def test_clean_caption_empty_string():
    assert clean_caption("") == []
    assert clean_caption("   ") == []


def test_add_special_tokens_wraps_correctly():
    tokens = add_special_tokens(["a", "dog", "runs"])
    assert tokens == ["<start>", "a", "dog", "runs", "<end>"]