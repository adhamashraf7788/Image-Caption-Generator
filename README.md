# Image Caption Generator

An end-to-end, production-oriented image captioning system: given a photo, the model generates a natural-language description of it. Built on the **Flickr8k dataset**, combining a **frozen pretrained ResNet50** (Computer Vision / transfer learning) with an **LSTM decoder** (NLP / sequence generation).

This project intentionally goes beyond a single notebook — it's organized into a modular, testable Python package with a config-driven, plug-and-play architecture (swap encoders/decoders via one config file), a full training/evaluation pipeline, unit tests, three deployment interfaces (FastAPI, Streamlit, Gradio on Hugging Face Spaces), and a Dockerfile.

- **Live demo (Hugging Face Space):** https://huggingface.co/spaces/AdhamAshraf/image_caption_generator
- **Trained model weights:** https://huggingface.co/AdhamAshraf/image-caption-generator
- **Source code:** https://github.com/adhamashraf7788/Image-Caption-Generator

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Preprocessing](#preprocessing)
- [Training Process](#training-process)
- [Evaluation: Metrics and Results](#evaluation-metrics-and-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How to Use the Application / API](#how-to-use-the-application--api)
- [Demo Video](#demo-video)
- [Docker](#docker)
- [What Was Tried / Design Decisions](#what-was-tried--design-decisions)
- [Known Limitations & Next Steps](#known-limitations--next-steps)

---

## Project Overview

Given an input image, the system:
1. Extracts a visual feature vector using a **frozen, pretrained ResNet50** (transfer learning — no CNN training from scratch).
2. Projects that feature vector into a shared embedding space.
3. Feeds it into an **LSTM decoder**, which generates a caption word-by-word (greedy decoding).

The project satisfies both ML and software-engineering goals: a working CV+NLP pipeline, plus modular code, config-driven experiments, unit tests, checkpointing, quantitative + qualitative evaluation, and three separate deployable interfaces.

## Dataset

**Flickr8k** — 8,091 images (this repackaged Kaggle release; the "official" release is 8,000), each paired with **5 human-written reference captions** (40,455 captions total).

**Split strategy:** captions are split by **unique image**, not by caption, to avoid data leakage (an image's 5 captions must never be split across train/val/test). Using a fixed seed (42):

| Split | Images | Caption pairs |
|---|---|---|
| Train | 6,472 (80%) | 32,360 |
| Val   | 809 (10%)   | 4,045 |
| Test  | 810 (10%)   | 4,050 |

> Note: this is a custom seeded 80/10/10 split (the raw download used here ships without the "official" `trainImages.txt`/`devImages.txt`/`testImages.txt` files some Flickr8k releases include). Results here are therefore not directly comparable to papers using the official split, though the split methodology (by image, not caption) is the same.

## Architecture

**Baseline: ResNet50 (frozen) → Linear projection → LSTM decoder** ("Show and Tell" style — image fed as the first input token to the LSTM).

```
Image → preprocess (resize/crop/normalize)
      → ResNet50 (frozen, ImageNet-pretrained, classification head removed)
      → 2048-d feature vector
      → Linear(2048 → 256) + ReLU + Dropout
      → 256-d "image embedding"
      → fed as the FIRST input step to the LSTM

Caption → normalize/tokenize → numericalize (vocab) → pad
        → word embeddings (256-d) → fed as subsequent LSTM input steps

LSTM (hidden_dim=512, 1 layer) → Linear(512 → vocab_size) → word logits at each step
```

| Component | Choice |
|---|---|
| Encoder | ResNet50, ImageNet-pretrained, **frozen** (no fine-tuning) |
| Feature caching | Features precomputed once and cached to disk (`.pt`) — the encoder never re-runs during training |
| Embedding dim | 256 |
| Decoder | 1-layer LSTM, hidden_dim = 512 |
| Vocabulary size | 2,662 (incl. `<pad>`, `<start>`, `<end>`, `<unk>`) |
| Decoding (inference) | Greedy (argmax at each step) |

The codebase uses a **registry pattern** (`src/models/registry.py`) so new encoders/decoders (e.g. attention, transformer) can be added by writing one class + one registry entry + one new YAML config — no other code changes. This baseline is `configs/base_resnet_lstm.yaml`.

## Preprocessing

**Images:**
- Resize shorter side to 256px, center-crop to 224×224 (ResNet50's expected input size)
- Convert to tensor, normalize with ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
- Identical transform used at train, val, test, and inference time (`src/data/preprocessing.py`)

**Captions:**
- Lowercase, strip punctuation, collapse whitespace, whitespace-tokenize
- Wrap with `<start>` / `<end>`
- Vocabulary built **from the training split only** (avoids leakage), keeping words with frequency ≥ 5; rarer words map to `<unk>`
- Numericalized and padded/truncated to a fixed max length (35 tokens)

**Feature caching:** since the CNN encoder is frozen, its output for a given image never changes — so every image is run through ResNet50 **once**, and the resulting 2048-d vectors are cached to disk (`data/features/resnet50_features.pt`). Training reads from this cache instead of recomputing CNN features every epoch (8,091 images extracted in ~25 seconds on an RTX 3060).

## Training Process

- **Loss:** Cross-entropy, ignoring `<pad>` positions
- **Optimizer:** Adam, initial lr = 1e-3
- **LR scheduling:** `ReduceLROnPlateau` (factor 0.5, patience 2 epochs on val loss)
- **Early stopping:** patience 5 epochs (no val loss improvement)
- **Checkpointing:** best model (lowest val loss) and last-epoch model both saved every epoch, enabling `--resume`
- **Teacher forcing:** ground-truth previous word fed at each decoder step during training (not the model's own prediction), allowing the whole sequence to be processed in one parallel forward pass per batch

**Actual training run** (RTX 3060, batch size 32):

| Epoch | Train Loss | Val Loss | LR |
|---|---|---|---|
| 1 | 3.5452 | 3.0476 | 1e-3 |
| 2 | 2.8399 | 2.8323 | 1e-3 |
| 3 | 2.5703 | 2.7623 | 1e-3 |
| **4** | **2.3746** | **2.7350** ⭐ best | 1e-3 |
| 5 | 2.2038 | 2.7508 | 1e-3 |
| 6 | 2.0496 | 2.7792 | 1e-3 |
| 7 | 1.9066 | 2.8252 | 5e-4 |
| 8 | 1.6822 | 2.8361 | 5e-4 |
| 9 | 1.5876 | 2.8757 | 5e-4 |

Early stopping triggered after epoch 9 (5 epochs with no improvement over epoch 4). The LR scheduler correctly halved the learning rate at epoch 7 after 2 stagnant epochs, though this wasn't enough to reverse the overfitting trend. **The checkpoint used for inference/evaluation is from epoch 4** (lowest val loss), not epoch 9 — this is what early stopping is designed to do.

Train loss continuing to drop (3.55 → 1.59) while val loss climbs after epoch 4 is a textbook overfitting signature for a first baseline with no dropout tuning, no augmentation, and a fairly small training set relative to vocabulary/sequence complexity — expected for an initial baseline, and a natural target for future optimization (not pursued in this iteration; see [What Was Tried](#what-was-tried--design-decisions)).

## Evaluation: Metrics and Results

Evaluated on the held-out **test set** (810 unique images, 4,050 reference captions), using the epoch-4 checkpoint, greedy decoding, and each generated caption scored against all 5 references per image.

| Metric | Score |
|---|---|
| BLEU-1 | 0.5127 |
| BLEU-2 | 0.3265 |
| BLEU-3 | 0.2008 |
| BLEU-4 | 0.1221 |
| ROUGE-L | 0.4177 |
| METEOR | 0.3266 |

BLEU-4 (the most commonly reported single figure for image captioning) of **0.12** is a working, honest baseline — somewhat below well-tuned literature baselines for this architecture family on Flickr8k (which often reach ~0.15–0.20 with more epochs, dropout tuning, and/or beam search), consistent with this being an intentionally minimal first baseline.

### Qualitative examples (Input Image → Generated Caption → References)

| Image | Generated | References |
|---|---|---|
| 1003163366_44323f5815.jpg | a man in a white shirt and jeans is standing in front of a brick building | A man lays on a bench while his dog sits by him. / A man lays on the bench to which a white dog is also tied. / a man sleeping on a bench outside with a white and black dog sitting next to him. / A shirtless man lies on a park bench with his dog. / man laying on bench holding leash of dog sitting on ground |
| 1007129816_e794419615.jpg | a man in a black shirt is holding a camera | A man in an orange hat starring at something. / A man wears an orange hat and glasses. / A man with gauges and glasses is wearing a Blitz hat. / A man with glasses is wearing a beer can crocheted hat. / The man with pierced ears is wearing glasses and an orange hat. |
| 1019077836_6fc9b15408.jpg | a brown dog is jumping over a fence | A brown dog chases the water from a sprinkler on a lawn. / a brown dog plays with the hose. / A brown dog running on a lawn near a garden hose / A dog is playing with a hose. / Large brown dog running away from the sprinkler in the grass. |
| 1082379191_ec1e53f996.jpg | a man in a white shirt is jumping into a pool | A lady and a man with no shirt sit on a dock. / A man and a woman are sitting on a dock together. / A man and a woman sitting on a dock. / A man and woman sitting on a deck next to a lake. / A shirtless man and a woman sitting on a dock. |
| 109260216_85b0be5378.jpg | a man climbs up a rock face | A person climbing down a sheer rock cliff using a rope / A person climbs a tall, flat mountain while holding onto a safety rope. / A person is climbing a rock while holding onto a white rope. / A person rappels down a steep incline. / A person wearing a red vest climbs up a steep rock. |

Full 12-example report: `models/base_resnet_lstm/eval_qualitative.md`.

**Observed pattern:** the model reliably produces fluent, grammatically well-formed captions matching Flickr8k's typical sentence structure ("a [subject] in a [color] [item] is [verb]-ing..."), but frequently doesn't ground the caption in the image's actual specific content — e.g. captioning a man asleep on a bench with a dog as "a man in a white shirt... in front of a brick building." This suggests the model is leaning on **language-model priors** (common caption openings) more than the 256-d image embedding signal, likely a consequence of early stopping at epoch 4 combined with a single, static global image vector. This is a known weak point of this exact architecture family and is the direct motivation for trying an **attention-based decoder** as a follow-up comparison (not yet implemented in this iteration).

## Project Structure

```
image-caption-generator/
├── configs/                  # YAML experiment configs (architecture + hyperparameters)
├── data/
│   ├── raw/                  # images/ + captions.txt (not committed — see Installation)
│   ├── processed/            # train/val/test.csv, vocab.json
│   └── features/             # cached ResNet50 feature vectors (.pt)
├── src/
│   ├── data/                 # preprocessing, vocabulary, PyTorch Dataset
│   ├── features/             # frozen CNN feature extractor
│   ├── models/                # base classes, encoder, decoder, registry, CaptionModel
│   ├── training/              # Trainer class + train.py entrypoint
│   ├── evaluation/             # BLEU/ROUGE/METEOR metrics + evaluate.py
│   ├── inference/              # Predictor class (single source of truth for generation)
│   └── utils/                  # config loader
├── scripts/                  # split_dataset.py, build_vocab.py, extract_features.py, push_to_hub.py
├── app/                       # FastAPI (api.py) + Streamlit (streamlit_app.py)
├── tests/                     # unit tests (pytest)
├── models/<run_name>/         # checkpoints, config.yaml, training_log.csv, eval results
├── Dockerfile
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/adhamashraf7788/Image-Caption-Generator.git
cd Image-Caption-Generator

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**Dataset:** download Flickr8k (e.g. the Kaggle "flickr8k" dataset by adityajn105) and place it as:
```
data/raw/images/*.jpg
data/raw/captions.txt      # columns: image, caption
```

## How to Run

Run once, in order, to prepare data and train:

```bash
python -m scripts.split_dataset --config configs/base_resnet_lstm.yaml
python -m scripts.build_vocab --config configs/base_resnet_lstm.yaml
python -m scripts.extract_features --config configs/base_resnet_lstm.yaml
python -m src.training.train --config configs/base_resnet_lstm.yaml
```

Evaluate on the test set:
```bash
python -m src.evaluation.evaluate --config configs/base_resnet_lstm.yaml
```

Run the test suite:
```bash
python -m pytest tests/ -v
```

> Note: run all `scripts/` and `src/` entrypoints as modules (`python -m ...`), not as bare file paths — this ensures the `src` package resolves correctly regardless of working directory.

## How to Use the Application / API

**Option 1 — Hugging Face Space (no setup required):**
https://huggingface.co/spaces/AdhamAshraf/image_caption_generator
Upload an image in the browser, get a caption back.

**Option 2 — FastAPI (local):**
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs`, use the interactive Swagger UI to `POST /predict` with an image file. Returns:
```json
{ "caption": "a man and a woman are sitting on a dock by a lake" }
```

**Option 3 — Streamlit (local):**
```bash
streamlit run app/streamlit_app.py
```
Opens a browser UI at `http://localhost:8501` — upload an image, caption appears below it.

All three interfaces share the exact same `Predictor` class (`src/inference/predict.py`), so behavior is identical across all of them.

## Demo Video

`docs/demo.mp4` — screen recording showing the same image being captioned through all three interfaces (Hugging Face Space, FastAPI Swagger UI, Streamlit).

<!-- Once recorded and committed to docs/demo.mp4, GitHub will render it inline below: -->
<!-- ![demo](docs/demo.mp4) -->

## Docker

```bash
docker build -t image-caption-generator .
docker run -p 8000:8000 image-caption-generator
```
Then open `http://localhost:8000/docs` — same FastAPI interface, running fully containerized.

## What Was Tried / Design Decisions

- **Frozen ResNet50 + feature caching**, rather than fine-tuning the CNN or running it live every epoch — deliberate speed/simplicity trade-off appropriate for an 8k-image dataset and a first baseline.
- **Image-as-first-LSTM-token** fusion strategy (vs. using the image to initialize LSTM hidden state) — simpler to implement/debug, standard "Show and Tell" baseline choice.
- **Greedy decoding** for this baseline, rather than beam search — beam search is a planned inference-time-only comparison (no retraining needed) for a later iteration.
- **Config + registry architecture** (`src/models/registry.py`) so alternative encoders (e.g. EfficientNet, InceptionV3) or decoders (attention-based LSTM, transformer) can be added without modifying training/evaluation/inference code — built specifically to support architecture comparisons, though only the ResNet50+LSTM baseline was trained in this iteration.
- **Split-before-vocab, image-level split** — vocabulary is built only from the training split and images (not individual captions) are the split unit, both specifically to prevent data leakage.
- **min_freq=5 vocabulary threshold** — reduces vocabulary from Flickr8k's full raw vocabulary down to 2,662 words, trading a small amount of `<unk>` coverage for a more learnable output space.
- Deliberately **did not pursue further optimization** in this iteration (e.g. dropout tuning, more epochs with regularization, data augmentation, unfreezing ResNet layers) in order to get a complete, working, evaluated, and deployed end-to-end system first — per the project's own stated priority on software engineering practice and full pipeline completion over squeezing out maximum accuracy from a single architecture.

## Known Limitations & Next Steps

- Baseline overfits after ~4 epochs; val loss does not improve further despite LR scheduling — candidates: dropout tuning, more training data via augmentation, or simply more regularization.
- Generated captions are fluent but frequently not well-grounded in image-specific detail (see qualitative examples) — primary motivation for trying an **attention-based decoder** next, which lets the model attend to different image regions per generated word instead of relying on one static global vector.
- Only one architecture (ResNet50 + LSTM) has been trained and evaluated so far; the registry-based design supports adding EfficientNet/InceptionV3 encoders and attention/transformer decoders as directly comparable experiments (new config file, no other code changes).
- Beam search decoding (vs. current greedy) is implemented as inference-time-only future work.

### Failure case: out-of-distribution images

Testing with an image type essentially absent from Flickr8k (a close-up bird perched on a hand) produced a grammatically broken output:

| Image | Generated |
|---|---|
| close-up photo of a chickadee perched on a hand, snowy background | `a bird is its wings in a` |

This is a different, more severe failure mode than the "fluent but ungrounded" captions seen on in-distribution test images (see [Evaluation](#evaluation-metrics-and-results)) — the output isn't just inaccurate, it isn't a valid sentence. Two likely contributing causes:

1. **Distribution shift** — Flickr8k is overwhelmingly people/dogs/outdoor-scene photography; close-up bird macro shots are essentially unrepresented in training, so the model is extrapolating far outside anything it learned.
2. **Greedy decoding degeneracy** — with a weak, early-stopped decoder (epoch 4) facing an unfamiliar image embedding, argmax word selection at each step can commit early to a low-confidence path with no ability to recover, producing an incoherent sequence. **Beam search** (retaining multiple candidate sequences instead of committing greedily) is expected to reduce this failure mode without any retraining, since it only changes inference-time decoding — this is a concrete, low-cost candidate for the "next steps" beam search work noted above.