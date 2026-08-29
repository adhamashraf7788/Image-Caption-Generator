"""Evaluate a trained model on the test set: BLEU/ROUGE/METEOR + qualitative examples.
Usage:
    python -m src.evaluation.evaluate --config configs/base_resnet_lstm.yaml --decoding greedy
    python -m src.evaluation.evaluate --config configs/base_resnet_lstm.yaml --decoding beam --beam-width 3
Reads:
    <paths.processed_dir>/test.csv
    <paths.models_dir>/<run_name>/best_model.pt
    <paths.processed_dir>/vocab.json
Writes (filenames include the decoding mode so greedy/beam results don't overwrite each other):
    <paths.models_dir>/<run_name>/eval_metrics_<decoding>.json
    <paths.models_dir>/<run_name>/eval_qualitative_<decoding>.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.evaluation.metrics import compute_all_metrics
from src.inference.predict import Predictor
from src.utils.config import load_config
def evaluate(
    config: dict,
    device: str = "cuda",
    decoding: str = "greedy",
    beam_width: int = 3,
    n_qualitative: int = 12,
) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    run_dir = Path(config["paths"]["models_dir"]) / config["run_name"]
    predictor = Predictor(
        checkpoint_path=run_dir / "best_model.pt",
        vocab_path=processed_dir / "vocab.json",
        device=device,
    )
    predictor.decoding = decoding
    predictor.beam_width = beam_width
    test_df = pd.read_csv(processed_dir / "test.csv")
    grouped = test_df.groupby("image")["caption"].apply(list).reset_index()
    print(f"Evaluating on {len(grouped)} unique test images (decoding={decoding}, beam_width={beam_width})")
    images_dir = Path(config["paths"]["raw_images_dir"])
    hypotheses: list[str] = []
    references: list[list[str]] = []
    per_image_results = []
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Generating captions"):
        image_filename = row["image"]
        refs = row["caption"]
        caption = predictor.predict(images_dir / image_filename)
        hypotheses.append(caption)
        references.append(refs)
        per_image_results.append(
            {"image": image_filename, "generated": caption, "references": refs}
        )
    metrics = compute_all_metrics(hypotheses, references)
    print(f"\n=== Evaluation Metrics (decoding={decoding}) ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    metrics_path = run_dir / f"eval_metrics_{decoding}.json"
    with open(metrics_path, "w") as f:
        json.dump({"decoding": decoding, "beam_width": beam_width, **metrics}, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    qual_path = run_dir / f"eval_qualitative_{decoding}.md"
    with open(qual_path, "w") as f:
        f.write(f"# Qualitative Evaluation -- {config['run_name']} ({decoding} decoding)\n\n")
        f.write("| Image | Generated | References |\n|---|---|---|\n")
        for item in per_image_results[:n_qualitative]:
            refs_joined = "<br>".join(item["references"])
            f.write(f"| {item['image']} | {item['generated']} | {refs_joined} |\n")
    print(f"Saved qualitative examples to {qual_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoding", default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam-width", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config, device=args.device, decoding=args.decoding, beam_width=args.beam_width) 