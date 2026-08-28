from src.evaluation.metrics import compute_all_metrics


def test_perfect_match_gives_high_scores():
    hypotheses = ["a dog runs in the grass"]
    references = [["a dog runs in the grass"]]

    scores = compute_all_metrics(hypotheses, references)
    assert scores["bleu_1"] > 0.9
    assert scores["rouge_l"] > 0.9


def test_completely_different_caption_gives_low_bleu4():
    hypotheses = ["a completely unrelated sentence about nothing"]
    references = [["a dog runs in the grass"]]

    scores = compute_all_metrics(hypotheses, references)
    assert scores["bleu_4"] < 0.1


def test_multi_reference_matches_best_reference():
    hypotheses = ["a dog runs in the park"]
    references = [
        [
            "a cat sleeps on the mat",
            "a dog runs in the park",  # exact match should dominate
            "a bird flies in the sky",
        ]
    ]
    scores = compute_all_metrics(hypotheses, references)
    assert scores["rouge_l"] > 0.9