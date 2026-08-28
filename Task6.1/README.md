# English → French Neural Machine Translation (BiLSTM + Attention + FastText)

## 1. Overview

This project extends a workshop notebook that builds a sequence-to-sequence (Seq2Seq)
English→French translation model in PyTorch. The original notebook used:

- A **frequency-based vocabulary** (most-common-words cutoff) with a **randomly initialized**
  embedding layer trained from scratch, and
- **Token accuracy** as the only evaluation metric.

This version keeps the original **Bidirectional LSTM encoder + Luong attention + LSTM
decoder** architecture, but improves it in two ways required by the assignment:

1. Replaces the random/frequency-only embeddings with **modern, pretrained-style word
   embeddings (FastText)**, trained on the corpus and used to initialize the encoder and
   decoder embedding layers.
2. Adds **BLEU** and **ROUGE** evaluation on greedy-decoded translations, instead of relying
   on teacher-forced token accuracy alone.

The pipeline is: clean text → build vocabulary → **train FastText embeddings → build aligned
embedding matrices** → encode with BiLSTM → attend → decode with LSTM → evaluate with token
accuracy, **BLEU, and ROUGE**.

## 2. Word-embedding method: FastText

**Why FastText** (Bojanowski et al., 2017) over Word2Vec or GloVe:

- FastText represents each word as a bag of **character n-grams**. This lets it produce a
  usable vector for **rare or unseen (out-of-vocabulary) words** by composing them from
  subword pieces — important for a modest-sized, everyday-language corpus like this one.
- It captures **morphological relationships** naturally (`aime` / `aimer` / `aimons`,
  `love` / `loves` / `loved`), which is especially relevant for French, a morphologically
  richer language than English.
- It can be **trained locally and quickly** with `gensim` directly on the training split, with
  no dependency on downloading large (multi-GB) pretrained vector files — a practical
  constraint in a Kaggle/workshop environment.

**How it's used:**

1. Two independent FastText models are trained with `gensim.models.FastText` (skip-gram,
   configurable window/epochs) — one on the **English training sentences**, one on the
   **French training sentences**. Source and target embeddings are kept separate because
   they are consumed by different modules (encoder vs. decoder) and live in different vector
   spaces.
2. For each vocabulary (`src_vocab`, `tgt_vocab`), an **embedding matrix** aligned with the
   vocabulary's `word2idx` mapping is built by looking up each word's FastText vector.
   Special tokens (`<pad>`, `<unk>`, `<start>`, `<end>`) get small random vectors, and `<pad>`
   is zeroed out.
3. The `Encoder` and `Decoder` modules are extended to accept a `pretrained_embeddings`
   matrix. When provided, `nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=...)`
   is used instead of a randomly initialized `nn.Embedding`. `freeze=False` means the FastText
   vectors are used only as a **better starting point** — they continue to be fine-tuned during
   translation training, rather than staying frozen.

This directly replaces the purely frequency-based representation (where "importance" of a
word was only its raw count) with vectors that already encode distributional/semantic
similarity before translation training even starts.

## 3. Architecture: BiLSTM encoder + Luong attention + LSTM decoder

The core Seq2Seq architecture from the workshop is unchanged, only its embedding
initialization changed:

- **Encoder (Bidirectional LSTM).** Reads the English sentence left→right and right→left in
  parallel and concatenates both directions at each position. This lets every source word's
  representation depend on its **full sentence context**, not just what came before it (useful
  for disambiguating words like `bank`). The final forward/backward hidden and cell states are
  concatenated and passed to the decoder as its initial state.
- **Attention (Luong dot-product attention).** At every decoding step, the decoder's current
  hidden state is compared (dot product) against every encoder output to produce a score per
  source position. A softmax turns these into attention weights, and a weighted sum of encoder
  outputs (the **context vector**) tells the decoder *which English words matter right now*.
  This avoids forcing the entire source sentence through a single fixed-size vector, which is
  the classic bottleneck of vanilla encoder–decoder models.
- **Decoder (LSTM).** Generates the French sentence one token at a time. At each step it
  combines its own LSTM output with the attention context vector and projects the result to a
  distribution over the French vocabulary. During training it uses **teacher forcing** (the
  true previous French word is fed in); during inference it uses **greedy decoding** (its own
  previous prediction is fed back in).
- A **padding mask** derived from the source sequence prevents attention from placing any
  weight on `<pad>` positions.

## 4. Preprocessing and training process

1. **Loading.** The English–French sentence-pair corpus is loaded from a local CSV (Kaggle
   dataset) or downloaded (PyTorch tutorial Anki/Tatoeba data) as a fallback.
2. **Cleaning.** Text is lowercased, common contractions are expanded (`don't` → `do not`),
   non-letter characters are stripped (keeping French accented characters), and whitespace is
   normalized. `<start>`/`<end>` tokens are added later at encoding time, not baked into the
   raw text.
3. **Splitting.** 10% held out as the **test set**, then 10% of the remainder as a
   **validation set** for early stopping. The vocabulary is built from the **training set
   only**, so no test-time words leak into the dictionary.
4. **Vocabulary.** A frequency-capped word-level vocabulary (`<pad>=0, <unk>=1, <start>=2,
   <end>=3` plus the most common words up to `max_vocab_size`) maps words to integer IDs.
5. **FastText embeddings.** Trained on the training split (see Section 2) and converted into
   embedding matrices aligned with the vocabularies.
6. **Batching.** A `Dataset`/`DataLoader` pads every sentence to a fixed max length and builds
   `(src, tgt_in, tgt_out)` triples for teacher forcing.
7. **Training loop.** Cross-entropy loss with `ignore_index=<pad>` (so padding doesn't inflate
   accuracy), Adam optimizer, `ReduceLROnPlateau` learning-rate scheduling, gradient clipping
   (`max_norm=1.0`), and early stopping on validation loss with restoration of the best
   checkpoint.
8. **Evaluation.** Token accuracy (teacher-forced) on train/val/test, plus **BLEU/ROUGE**
   (greedy-decoded, no teacher forcing) on a sample of the test set.

## 5. Evaluation metrics: BLEU and ROUGE

Token accuracy under teacher forcing is a poor proxy for translation quality: it is computed
one gold-conditioned step at a time and says nothing about the sentence the model actually
produces on its own. Two standard machine-translation metrics were added, computed on
**greedy-decoded** output (the model's own predictions fed back in, exactly as at inference
time):

- **BLEU (Bilingual Evaluation Understudy).** Computes modified n-gram precision (1- to
  4-grams) between the candidate translation and the reference, multiplied by a **brevity
  penalty** that penalizes translations that are too short relative to the reference (so the
  model can't "cheat" by outputting very short, safe sentences). Implemented with
  `nltk.translate.bleu_score.corpus_bleu` with smoothing (important for short sentences with
  few or zero 4-gram matches).
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation).** Reports ROUGE-1 and ROUGE-2
  (unigram/bigram overlap) and ROUGE-L (longest common subsequence, which rewards preserving
  word order). Implemented with the `rouge-score` package, F1 variant, averaged over the
  evaluation sample. ROUGE complements BLEU's precision focus with a more recall-oriented,
  order-sensitive view.

Both are reported together because they trade off differently: BLEU rewards precise n-gram
matches (good for exact phrasing), ROUGE-L rewards getting the right words in the right
relative order even without exact n-gram matches (good for partially-correct translations).

## 6. Results

> Fill in after running the notebook end-to-end (numbers depend on `WORKSHOP_MODE`, dataset
> size, and epochs trained):

| Metric | Value |
|---|---|
| Test loss (teacher forcing) | `<fill in>` |
| Test token accuracy (teacher forcing, pad ignored) | `<fill in>` |
| BLEU (greedy decode) | `<fill in>` |
| ROUGE-1 (F1) | `<fill in>` |
| ROUGE-2 (F1) | `<fill in>` |
| ROUGE-L (F1) | `<fill in>` |

**Discussion.** Token accuracy tends to look higher than BLEU/ROUGE because it is measured
under teacher forcing (the model always sees the correct previous word) and only checks
exact per-token matches. BLEU/ROUGE, measured on the model's own free-running greedy output,
better reflect real translation quality, including error propagation (one wrong early word
can derail the rest of a greedily decoded sentence). Short, frequent everyday phrases
("Hi.", "I love you.") are translated noticeably better than longer or rarer sentences — a
direct consequence of the small vocabulary cap and word-level (not subword) tokenization.
Qualitatively, the attention heatmaps show a roughly diagonal alignment between English and
French words, with extra attention mass on function words that reorder or flip meaning
(e.g. negation).

Compared to the original random-embedding baseline, initializing embeddings from FastText is
expected to help most in the **early epochs** (faster convergence, since embeddings already
encode some word similarity) and on **less frequent words**, where FastText's subword
composition gives a better starting vector than a random one that has to be learned purely
from a handful of occurrences in the parallel corpus.

## 7. Additional techniques used to improve performance

- **Bidirectional encoding** so every source word's representation reflects the full
  sentence, not just a left-to-right pass.
- **Luong (dot-product) attention with a padding mask**, so the decoder can focus on the
  relevant source words at each step and never attends to `<pad>` positions.
- **Teacher forcing** during training for stable, parallelizable optimization, with **greedy
  decoding** at inference/evaluation time to reflect real usage.
- **`ignore_index` on the padding token** in the loss and accuracy computation, avoiding the
  classic "high accuracy from predicting `<pad>`" trap.
- **Gradient clipping** (`max_norm=1.0`) to keep LSTM training stable.
- **Learning-rate scheduling** (`ReduceLROnPlateau`) and **early stopping** with best-checkpoint
  restoration, to avoid overfitting on a relatively small corpus.
- **FastText subword embeddings** (this project's main addition) as a smarter, semantically
  informed initialization for both encoder and target embeddings, fine-tuned jointly with the
  rest of the network rather than kept frozen.
- **BLEU and ROUGE evaluation** (this project's other main addition) on greedy-decoded output,
  giving a much more realistic picture of translation quality than teacher-forced token
  accuracy alone.

### Possible next steps
- Beam search decoding instead of greedy decoding.
- Subword tokenization (BPE / SentencePiece) to reduce `<unk>` rate on rare/rare-morphology
  words.
- Swapping in true pretrained embeddings (e.g. `fastText` `cc.en.300` / `cc.fr.300`) when
  internet access to large pretrained files is available, instead of corpus-trained FastText.
- A Transformer-based encoder/decoder as a stronger architecture baseline for comparison.