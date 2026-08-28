"""BLEU / ROUGE / METEOR computation.

All metrics compare ONE generated caption against the MULTIPLE (5) human
reference captions for that image -- never against just one reference,
since there's no single "correct" caption.
"""
from __future__ import annotations

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def compute_bleu(hypotheses: list[list[str]], references: list[list[list[str]]]) -> dict[str, float]:
    """Corpus-level BLEU-1..4.

    hypotheses: list of tokenized generated captions, one per image
    references: list of lists of tokenized reference captions, one outer list per image
    """
    smoothing = SmoothingFunction().method1
    scores = {}
    for n in (1, 2, 3, 4):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
        scores[f"bleu_{n}"] = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoothing)
    return scores


def compute_rouge_l(hypotheses: list[str], references: list[list[str]]) -> float:
    """Average ROUGE-L (F-measure), taken against the best-matching reference per image.

    hypotheses: list of generated caption strings (not tokenized)
    references: list of lists of reference caption strings, one outer list per image
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = 0.0
    for hyp, refs in zip(hypotheses, references):
        best = max(scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
        total += best
    return total / max(len(hypotheses), 1)


def compute_meteor(hypotheses: list[list[str]], references: list[list[list[str]]]) -> float:
    """Average METEOR score across all images.

    hypotheses: list of tokenized generated captions
    references: list of lists of tokenized reference captions
    """
    total = 0.0
    for hyp, refs in zip(hypotheses, references):
        total += meteor_score(refs, hyp)
    return total / max(len(hypotheses), 1)


def compute_all_metrics(
    hypotheses_str: list[str],
    references_str: list[list[str]],
) -> dict[str, float]:
    """Compute BLEU-1..4, ROUGE-L, METEOR from plain (untokenized) strings."""
    hyp_tokens = [h.split() for h in hypotheses_str]
    ref_tokens = [[r.split() for r in refs] for refs in references_str]

    results = compute_bleu(hyp_tokens, ref_tokens)
    results["rouge_l"] = compute_rouge_l(hypotheses_str, references_str)
    results["meteor"] = compute_meteor(hyp_tokens, ref_tokens)
    return results