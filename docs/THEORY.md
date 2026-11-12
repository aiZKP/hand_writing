# Handwritten Word Recognition — Study Guide

This guide explains **why** every part of the project exists, with the math you
know from the lecture and pointers into the code. Read it next to the source.
The [concept index](#11-lecture-concept--code-index) at the end maps lecture
topics to exact files and functions.

Contents
1. [The problem](#1-the-problem)
2. [Data: EMNIST and synthetic words](#2-data)
3. [Preprocessing and train/serve skew](#3-preprocessing-and-trainserve-skew)
4. [Model A — CharCNN + segmentation](#4-model-a--charcnn--segmentation)
5. [Model B — CRNN](#5-model-b--crnn)
6. [CTC — Connectionist Temporal Classification](#6-ctc--connectionist-temporal-classification)
7. [Training engineering on a CPU](#7-training-engineering-on-a-cpu)
8. [Evaluation metrics](#8-evaluation-metrics)
9. [Results and discussion](#9-results-and-discussion)
10. [Domain gap, fine-tuning, and exercises](#10-domain-gap-fine-tuning-and-exercises)
11. [Lecture concept → code index](#11-lecture-concept--code-index)
12. [Deployment: the models in the browser](#12-deployment-the-models-in-the-browser)

---

## 1. The problem

Input: an image of a handwritten word. Output: a string over the alphabet
`0-9 A-Z a-z` (62 symbols, `hwr/config.py: CHARSET`).

Recognizing a **word** is harder than recognizing a **character**:

| | Single character | Word |
|---|---|---|
| Output | one of K = 62 classes | a sequence of unknown length L |
| Task type | classification | sequence labeling |
| Main difficulty | similar-looking classes | *where* does each character start and end? (alignment / segmentation) |

There are two classic strategies, and this project builds both:

* **Model A — segment, then classify.** Cut the word into character images
  with image processing, then classify each one with a CNN. This uses only
  the "standard" CNN classification from the lecture. Its weak point is the
  segmentation step, which has no learning in it.
* **Model B — segmentation-free (CRNN + CTC).** A network reads the whole
  word image left-to-right and outputs a probability distribution per time
  step. The CTC loss lets us train it **without** ever telling it where each
  letter is.

---

## 2. Data

### 2.1 EMNIST ByClass (`hwr/emnist.py`)
* Real handwriting from NIST Special Database 19, converted to the MNIST
  format: 28×28 grayscale, white ink on black, 62 classes.
* 697,932 training and 116,323 test images.
* **Quirk 1 — transposed storage.** The raw IDX files store each image
  column-major. Without `images.transpose(0, 2, 1)` every letter is
  mirrored and rotated. We verified this visually (`reports/emnist_samples.png`).
* **Quirk 2 — class imbalance.** `'1'` has 38,374 training images while `'j'`
  has 1,896 (20×). Plain accuracy is dominated by frequent classes, so we
  also report **balanced accuracy** (mean recall per class).
* **Quirk 3 — an accuracy ceiling.** EMNIST scales every character to fill
  the 28×28 box. Size information is lost, so `c/C, o/O, s/S, u/U, v/V, w/W,
  x/X, z/Z, k/K, p/P, y/Y` and `0/O/o, 1/l/I` become (nearly) the same
  picture. Published models reach only ~88 % on ByClass for this reason.

The IDX format itself is tiny: 4 magic bytes `00 00 <dtype> <ndim>`, then
`ndim` big-endian uint32 sizes, then raw data (`parse_idx`). The downloaded
zip is untrusted input, so we read only the named members instead of calling
`extractall()`, which avoids "zip-slip" path traversal.

### 2.2 Synthetic words (`hwr/synth.py`)
Model B needs word images with word labels, which EMNIST does not have. So
we **compose** words from EMNIST characters on the fly. Every training step
sees new words, so the model practically never sees the same image twice.
That is a strong regularizer.

**Text mixture** (`TextSampler`):

| Source | Share | Example | Why |
|---|---|---|---|
| Dictionary word (lower / Capitalized / UPPER) | 40 % | `learning`, `Deep`, `CNN` | realistic letter combinations |
| Random string over 62 symbols | 35 % | `xQ7bZ` | stops the BiLSTM from relying on an English "language model" |
| Number | 12 % | `2026` | digits |
| Alphanumeric mix | 13 % | `Room42`, `3D` | realistic mixes |

**Writing-band geometry** (`config.char_extent`). The dashboard canvas and the
synthetic words share the same guide lines:

```
0.12 ─────────────  ascender / cap line   A-Z, 0-9, b d f h k l
0.24                (top of i, j, t)
0.40 - - - - - - -  x-height line         a c e m n o r s u v w x z
0.70 ─────────────  baseline
0.92 ─────────────  descender line        g j p q y
```

Lowercase `o` is therefore *small and low*, while `O` and `0` are tall.
Because the CRNN sees the **whole band** (not a tight crop), it can use letter
size to resolve the case ambiguity that limits Model A.

**Realism tricks:**
* **One pen per word.** EMNIST stroke width scales with the glyph. When we
  shrink an uppercase-sized glyph to x-height, its stroke becomes too thin.
  `estimate_stroke_width` (≈ 2·area / perimeter) measures it, and a 3×3 or
  5×5 max-filter (dilation) or min-filter (erosion) brings every glyph to the
  same target width as the dashboard brush.
* **Spacing** is random, including **negative gaps**. Letters then touch or
  overlap, which is exactly the case where segmentation fails (41–43 % of
  validation/test words).
* **Global variation:** size ×0.85–1.12, vertical shift, slant (shear ±0.25),
  occasional blur.

**No data leakage.** Training words use only EMNIST *train* characters.
Validation words use the first half of EMNIST *test*, and test words the
second half. Validation and test are fixed `.npz` files, so every evaluation
uses identical images.

---

## 3. Preprocessing and train/serve skew

*Train/serve skew* is the classic production ML bug: the model is trained on
data prepared one way and served data prepared another way. Here every path
(synthetic training words, dashboard drawings, saved user samples) goes
through the **same** functions in `hwr/preprocess.py`:

`band_to_model_input` (Model B input):
1. Ink = the canvas **alpha channel** (the guide lines live on a different
   canvas layer and are never sent).
2. Keep the **full band height**, which preserves case information.
3. Crop horizontally to the ink plus a margin of 0.06 × band height.
4. Resize to height 48 keeping the aspect ratio (Lanczos, which
   anti-aliases).
5. Values in [0, 1], ink = 1.

`normalize_char_emnist_style` (Model A input) copies EMNIST's own pipeline:
1. Crop to the bounding box.
2. Center in a square, keeping the aspect ratio.
3. Fit the content into the central 24×24 of a 28×28 image.

The CharCNN stores its normalization constants (EMNIST mean 0.1736, std
0.3317) **inside the model** as buffers, so inference code cannot forget them.

---

## 4. Model A — CharCNN + segmentation

### 4.1 Architecture (`hwr/models/char_cnn.py`)

| Layer | Output shape | Parameters |
|---|---|---|
| input | 1×28×28 | |
| Conv 3×3, 1→32, BN, ReLU | 32×28×28 | 1·32·9 = 288 (+64 BN) |
| MaxPool 2×2 | 32×14×14 | 0 |
| Conv 3×3, 32→64, BN, ReLU | 64×14×14 | 32·64·9 = 18,432 (+128) |
| Conv 3×3, 64→64, BN, ReLU | 64×14×14 | 36,864 (+128) |
| MaxPool 2×2 | 64×7×7 | 0 |
| Conv 3×3, 64→128, BN, ReLU | 128×7×7 | 73,728 (+256) |
| MaxPool 2×2 | 128×3×3 | 0 |
| Flatten, Dropout 0.5, Linear 1152→256, ReLU | 256 | 295,168 |
| Dropout 0.5, Linear 256→62 | 62 logits | 15,934 |
| **Total** | | **440,990** |

The convolutions use `bias=False` because the following BatchNorm has its own
shift β, so a bias would be redundant.

**Convolution output size:** `out = floor((in + 2·pad − kernel) / stride) + 1`.
With `kernel = 3, pad = 1, stride = 1`, the spatial size is kept and only
pooling halves it.

**Receptive field** (how many input pixels influence one output unit). The
first conv sees 3×3 pixels. Every later 3×3 conv adds 2·(product of earlier
strides), and every 2×2 pool adds 1·(product of earlier strides). After the
last conv one unit sees 18×18 pixels: most of a 28×28 character.

**Lecture concepts in this model:**
* **Weight sharing / local connectivity.** A 3×3 conv has 9·C_in·C_out
  weights regardless of image size, compared with a dense layer's
  784·hidden.
* **ReLU** `max(0, z)` has gradient 1 for positive inputs, so it does not
  saturate the way sigmoid/tanh do (vanishing gradients).
* **Batch Normalization**: `ŷ = γ · (x − μ_B) / sqrt(σ²_B + ε) + β`.
  - It uses batch statistics in training and running averages in `eval()`
    mode, which is why `model.eval()` matters.
  - It smooths the loss landscape and allows higher learning rates.
* **Max pooling** gives small translation invariance and downsampling.
* **Dropout** zeroes units with probability p in training (scaling the rest
  by 1/(1−p)). This acts like an ensemble of sub-networks and fights
  co-adaptation.

### 4.2 Loss and optimization (`hwr/train_char_cnn.py`)
* **Softmax + cross-entropy:** `p_k = exp(z_k) / Σ_j exp(z_j)` and
  `L = −log p_y`.
  - PyTorch's `CrossEntropyLoss` takes raw logits and fuses log-softmax for
    numerical stability, so the model has no softmax layer.
  - The gradient w.r.t. the logits is beautifully simple: `∂L/∂z = p − onehot(y)`.
* **Label smoothing** (ε = 0.05) replaces the one-hot target with
  `(1−ε)·onehot + ε/K`. It discourages over-confident logits, which helps
  here because EMNIST labels for o/O/0 are genuinely ambiguous.
* **AdamW** is Adam (per-parameter step sizes from running estimates of the
  first and second gradient moments) plus *decoupled* weight decay
  `θ ← θ − lr·λ·θ`. In plain Adam, L2 regularization would be distorted by
  the adaptive step size.
* **OneCycle learning-rate schedule:**
  - It warms up linearly for 15 % of the steps, from max_lr/25 to max_lr =
    3e-3, then cosine-anneals to almost 0.
  - The warm-up avoids early instability while BatchNorm statistics and
    Adam's moments are still unreliable. The annealing lets the model settle
    into a minimum.
* **Augmentation as regularization** (`hwr/augment.py`), done on the whole
  batch in torch:
  - **Random affine** (rotation ±10°, scale 0.85–1.1, shear ±0.2,
    translation ±8 %):
    - `F.affine_grid` builds, for each output pixel, the input position to
      sample from (θ maps output → input).
    - `F.grid_sample` interpolates bilinearly.
  - **Stroke-width jitter:** dilation = `maxpool3×3(x)`, erosion =
    `−maxpool3×3(−x)`. These are grey-scale morphology expressed as pooling.

### 4.3 Segmentation (`hwr/segment.py`) — no learning
1. Binarize the image with a threshold.
2. Label **8-connected components** (`scipy.ndimage.label`). Two ink pixels
   belong to the same component if they touch, diagonals included.
3. Drop components smaller than 0.0004 · band_height² pixels (noise).
4. **Merge** components whose x-ranges overlap by more than 50 % of the
   narrower one. This handles the dot of `i`/`j` and letters drawn with
   separate strokes (`E`, `t`).
5. Sort left → right. Each segment is normalized EMNIST-style and
   classified.

**Failure modes, which motivate Model B:**
* Touching letters form one component, so two letters become one
  prediction and the word gets shorter.
* A letter broken into pieces that don't overlap in x is split into two.
* Size is lost by the 28×28 normalization, so case is a guess.

---

## 5. Model B — CRNN

Shi, Bai & Yao (2016), *An End-to-End Trainable Neural Network for
Image-based Sequence Recognition*. Code: `hwr/models/crnn.py`.

### 5.1 Shape trace (input height 48, width W; T = W/4)

| Stage | Output shape | Note |
|---|---|---|
| input | B×1×48×W | full writing band |
| Conv 1→32, BN, ReLU; MaxPool (2,2) | B×32×24×W/2 | |
| Conv 32→64, BN, ReLU; MaxPool (2,2) | B×64×12×W/4 | |
| Conv 64→128, BN, ReLU; MaxPool **(2,1)** | B×128×6×W/4 | pool **height only** |
| Conv 128→160, BN, ReLU; MaxPool **(2,1)** | B×160×3×W/4 | |
| Dropout2d(0.1) | | drops whole channels |
| **map-to-sequence**: permute + reshape | B×T×480 | each column = one time step |
| Linear 480→256, ReLU | B×T×256 | |
| BiLSTM, 2 layers, hidden 128 | B×T×256 | 128 forward + 128 backward |
| Dropout, Linear 256→63 | B×T×63 | 62 characters + blank |
| log_softmax, transpose | **T×B×63** | layout `nn.CTCLoss` expects |

1,207,391 parameters.

**Why pool only the height in later layers?** We need many time steps
horizontally. CTC needs `T ≥ L + (number of repeated letters)`, and each
character should get a few columns. Vertically we compress everything into
3 rows of features.

**Receptive field of one column:** 46 px vertically, nearly the whole 48 px
band, so the network can judge whether a letter is tall (upper case) or short
(lower case). It is 26 px horizontally, about one letter. Wider context comes
from the LSTM.

### 5.2 Recurrent part: LSTM
A plain RNN `h_t = tanh(W·x_t + U·h_{t−1})` multiplies by `U` at every step
during backpropagation through time. Its gradients vanish or explode
exponentially in T. The **LSTM** adds a cell state with *additive* updates
controlled by gates:

```
i_t = σ(W_i x_t + U_i h_{t−1} + b_i)        input gate
f_t = σ(W_f x_t + U_f h_{t−1} + b_f)        forget gate
o_t = σ(W_o x_t + U_o h_{t−1} + b_o)        output gate
g_t = tanh(W_g x_t + U_g h_{t−1} + b_g)     candidate
c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t             additive path -> gradients survive
h_t = o_t ⊙ tanh(c_t)
```

**Bidirectional.** One LSTM reads left→right and another right→left, and
their outputs are concatenated. Each time step knows both its left and right
context. This helps with touching letters and with case: "is this letter
taller than its neighbours?"

**Gradient clipping** (`clip_grad_norm_(…, 5)`) rescales the whole gradient
if its norm exceeds 5. It is the standard remedy for *exploding* gradients in
RNNs.

---

## 6. CTC — Connectionist Temporal Classification

Graves et al. (2006). Code: `nn.CTCLoss` for training and `hwr/ctc.py` for
decoding.

### 6.1 The alignment problem
We know the label `"Hello"` (L = 5), but the network produces T = 36
distributions, one per column. Which columns belong to which letter? We
don't know, and we don't want to label it by hand.

### 6.2 Blank and the collapse function B
CTC adds a special **blank** symbol `-` (index 0). A *path* π picks one symbol
per frame. The collapse function **B**:
1. merges consecutive repeats;
2. removes blanks.

```
B("--HH-e-ll-l-oo--") = "Hello"
B("Heelo")            = "Helo"      (repeats merge)
B("Hel-lo")           = "Hello"     (a blank separates the two l's)
```

The blank therefore does two jobs: it means "no character here", and it
separates real double letters.

### 6.3 The loss
Assuming the frames are conditionally independent given the image:

```
P(π | x)  = Π_t  y_t(π_t)                      y_t = softmax output at frame t
P(l | x)  = Σ_{π : B(π) = l}  P(π | x)          sum over ALL valid alignments
L_CTC     = −log P(l | x)
```

There are exponentially many paths, but the sum is computed with
**dynamic programming** (the forward algorithm, like an HMM):

1. Extend the label with blanks: `l' = - H - e - l - l - o -` (length 2L+1).
2. Let `α_t(s)` = total probability of all path prefixes up to frame t that
   end at position s of `l'`.
3. Recursion:

```
α_t(s) = ( α_{t−1}(s) + α_{t−1}(s−1) ) · y_t(l'_s)                   if l'_s = blank or l'_s = l'_{s−2}
α_t(s) = ( α_{t−1}(s) + α_{t−1}(s−1) + α_{t−1}(s−2) ) · y_t(l'_s)    otherwise
```

4. The answer is `P(l|x) = α_T(|l'|) + α_T(|l'|−1)`.
5. Cost is O(T·L) instead of O(Cᵀ). The backward variables β give the
   gradients.

**Constraint:** `T ≥ L + (number of adjacent repeats)`, because every repeat
needs a blank between its letters. With T = W/4 and letters 15–30 px wide,
this holds easily. `zero_infinity=True` protects against the rare impossible
sample.

`nn.CTCLoss` needs `log_probs (T,B,C)`, the concatenated `targets`,
`input_lengths` (the true T of each sample, padding excluded) and
`target_lengths`.

### 6.4 A worked example (`tests/test_metrics_ctc.py`)
Take 2 frames, each with P(blank) = 0.6 and P(a) = 0.4.

| Path | Collapses to | Probability |
|---|---|---|
| `--` | "" | 0.6·0.6 = 0.36 |
| `-a` | "a" | 0.6·0.4 = 0.24 |
| `a-` | "a" | 0.4·0.6 = 0.24 |
| `aa` | "a" | 0.4·0.4 = 0.16 |

So P("") = 0.36 but P("a") = 0.24 + 0.24 + 0.16 = **0.64**.

### 6.5 Decoding
* **Greedy (best path):** take the argmax per frame, then apply B.
  - It is fast, but it finds the most likely **path**, not the most likely
    **text**.
  - In the example above, greedy outputs `""` (path `--`) although `"a"` is
    more likely.
* **Prefix beam search** (`prefix_beam_search`, Hannun et al. 2014):
  - It keeps the `beam` most likely text **prefixes** and sums the
    probability of all paths leading to each.
  - Each prefix tracks two numbers: P(…ends in blank) and P(…ends in its
    last letter). Both are needed, because a repeated letter only creates a
    new character after a blank.
  - It works in log space with log-sum-exp for numerical stability.
  - The dashboard shows the top 5 prefixes and their probabilities.

---

## 7. Training engineering on a CPU

Everything here is trained on 6 CPU threads (no GPU), so efficiency mattered.
These are the measured effects:

| Change | Training speed (CRNN) | Why it helps |
|---|---|---|
| Baseline (5 convs, random batches, packed LSTM) | 39 words/s | |
| Smaller CNN (4 convs, ~105M MACs/word) | | fewer FLOPs, still covers the full band height |
| **Width bucketing** (`BucketedSynthBatches`) | 65 words/s | sorting a pool of 8 batches by width removes ~50 % padding work |
| No LSTM packing in training, `channels_last` layout, 1 data worker | **≈ 83 words/s** | packing costs ~20 % on CPU; NHWC layout vectorizes better in oneDNN; fewer processes competing for cores |

**Bucketing** is worth understanding:
* A batch tensor must be rectangular, so every image is padded to the widest
  one.
* With random batches the widest word is ~2× the average, so half of the
  convolution FLOPs are spent on zeros.
* We generate a pool, sort it by width, cut it into batches, then shuffle the
  *order* of the batches.

**Packing vs. not packing:**
* `pack_padded_sequence` makes the LSTM stop at each sequence's true end.
  This is exact, so we use it for evaluation.
* With bucketing the padding is only a few pixels, so in training we skip
  packing.
* The CTC loss still ignores padded frames, because it receives the true
  `input_lengths`.

**Data generation in worker processes:**
* Synthetic words are generated by a `DataLoader` worker in parallel with
  the training step.
* Each worker seeds its RNG with `(seed, epoch, worker_id)`, so the data is
  reproducible and different in every epoch and worker.

---

## 8. Evaluation metrics (`hwr/metrics.py`)

* **Accuracy / top-3 / balanced accuracy / confusion matrix** for the
  CharCNN.
* **Levenshtein (edit) distance** is the minimum number of insertions,
  deletions and substitutions, computed by dynamic programming:

  ```
  D[i][j] = min(D[i−1][j] + 1,  D[i][j−1] + 1,  D[i−1][j−1] + [a_i ≠ b_j])
  ```

  For example, `levenshtein("kitten", "sitting") = 3`.
* **CER (character error rate)** = Σ edit distance / Σ reference length. This
  is the standard metric for text recognition.
* **Word accuracy** is the fraction of exactly correct words. **Word accuracy
  ignoring case** shows how much of the error is only upper/lower confusion.

---

## 9. Results and discussion

All numbers come from `python -m hwr.evaluate` (`reports/evaluation.json`,
`reports/comparison.md`) and from the training logs. The hardware was 6 CPU
threads, with no GPU.

### 9.1 Model A alone — single EMNIST characters (116,323 test images)

| Top-1 | Top-3 | Balanced accuracy | Top-1 ignoring case |
|---|---|---|---|
| **86.0 %** | 99.1 % | 71.0 % | 90.2 % |

Most frequent confusions (`reports/char_cnn_confusion.png`):

| true | predicted | count | share of the true class |
|---|---|---|---|
| `O` | `0` | 2577 | 62.0 % |
| `l` | `1` | 2175 | 85.8 % |
| `I` | `1` | 1388 | 67.8 % |
| `m` | `M` | 461 | 99.4 % |
| `u` | `U` | 449 | 93.2 % |
| `c` | `C` | 424 | 98.1 % |
| `0` | `O` | 409 | 7.1 % |
| `v` | `V` | 408 | 87.2 % |
| `s` | `S` | 382 | 87.4 % |
| `f` | `F` | 377 | 94.2 % |

**How to read this:**
* **The network knows the shape.** Top-3 is 99 %, so the right answer is
  almost always among its first three guesses. The remaining top-1 error is
  mostly *irreducible ambiguity*.
  - Without size information, `O` and `0`, `l`/`I`/`1`, and the case pairs
    are often literally the same picture.
  - Ignoring case alone lifts accuracy from 86.0 % to 90.2 %.
* **Why balanced accuracy is much lower than accuracy.** For an ambiguous
  image the best guess is the class with the larger prior P(class). Uppercase
  `M, U, C, S` and the digits `0`/`1` are far more frequent in EMNIST than
  `m, u, c, s, o, l`, so the minority class almost always loses (`m → M` in
  99 % of cases).
  - This is the Bayes decision under the training distribution, not a bug.
  - It only shows up in per-class recall, which is why plain accuracy alone
    is misleading on imbalanced data.
* **The confusion matrix shows a "second diagonal".** Lowercase rows are
  predicted as their uppercase twins.
* **Model selection on validation data.** Validation accuracy was 84.2 % after
  epoch 1, 85.4 % after epoch 2 and 85.3 % after epoch 3. The checkpoint from
  **epoch 2** was kept, which is exactly why we select on validation data
  instead of taking the last epoch.

### 9.2 Model B — CRNN + CTC

Validation CER per epoch (%): 100.0 → 20.5 → 12.7 → 10.2 → 9.9 → 8.6 → 8.1 → 6.8 → 6.6 → 6.4 → 6.3 → 6.2

* **Epoch 1: the blank plateau.** After 500 steps the model output *only
  blanks* (CER 100 %), although the loss was already falling.
  - Predicting blank everywhere is the easiest first solution for CTC.
  - The model leaves this plateau once it learns where characters are, and
    then CER drops very fast (20.5 % after epoch 2).
  - If you ever see CER stuck at 100 % for many epochs, check the learning
    rate and input lengths.
* **No overfitting gap.** Each epoch uses 32,000 *new* synthetic words
  (384,000 in total, each seen once), so training and validation curves stay
  together (`reports/crnn_curves.png`). The improvement in the last epochs
  comes from the decaying OneCycle learning rate.
* **Test set** (3,000 words of unseen EMNIST test handwriting):
  - CER **6.0 %**;
  - word accuracy **70.1 %** with beam search (69.8 % greedy);
  - 74.5 % ignoring case.

  Beam search helps only a little here because the per-frame distributions
  are already peaked. It matters more when the model is uncertain.

### 9.3 The key experiment — A vs. B on the same 3000 test words
41.2 % of the words contain touching letters.

| Model | Word accuracy | Word acc. ignoring case | CER | Touching letters | Separated letters |
|---|---|---|---|---|---|
| A: segmentation + CharCNN | **16.2 %** | 25.5 % | 49.1 % | 4.8 % | 24.3 % |
| B: CRNN + CTC, greedy | **69.8 %** | 74.2 % | 6.1 % | 65.2 % | 73.1 % |
| B: CRNN + CTC, beam search | **70.1 %** | 74.5 % | 6.0 % | 65.6 % | 73.2 % |

Model A found the right **number** of characters in 55.1 % of
the words: only 13.9 % when letters touch, and
83.9 % when they are separated.

**Why A is so much worse, even on separated letters.** The errors **compound**:
* For a 6-letter word, 86 % per-character accuracy gives about 0.86⁶ ≈ 40 %
  word accuracy, even with perfect segmentation.
* Segmentation itself is wrong for 16 % of separated words and 86 % of
  touching words.
* Every segment is normalized to 28×28, so the size that distinguishes `o`
  from `O` is thrown away.

**Why B works.** One network is trained end-to-end on the final objective.
CTC removes the need for segmentation, the BiLSTM supplies context, and the
full-height input keeps the size cue. The remaining errors are mostly genuine
EMNIST look-alikes (`reports/comparison_examples.png`), e.g. a `Y` that looks
like `4`, or `l` vs `I`.

### 9.4 Domain-gap check — words drawn like on the canvas
Training data is scanned EMNIST handwriting, while the dashboard produces
smooth constant-width strokes. To measure the gap honestly:
* 40 words (20 upright, 20 slanted) were drawn with a hand-made stroke font
  at the dashboard's brush width.
* They were sent through the real `/api/predict` endpoint.
* The same words were also drawn with real mouse events in headless Chrome,
  with identical results.

| | Exact words |
|---|---|
| A: segmentation + CharCNN | 12 / 40 (30 %) |
| B: CRNN + CTC | **34 / 40 (85 %)** |

**Model A's errors are almost all case/look-alike errors:** `Hell0`, `R00M42`,
`d0g`, `LiMe`, `iCe`, `0Pen`, `Pen` for `pen`.

**Model B made zero case errors.** It reads `Room42`, `dog`, `open` and `ice`
correctly by comparing letter heights with the guide lines. Its six mistakes
are letter-shape confusions, mostly the stroke font's `a` read as `o`/`ai`:
* `cat → cait`, `tap → top`, `pen → pan`, `dream → drooim`;
* `41 → 47`, `Deep → Deop`.

This is the kind of gap that fine-tuning on your own samples closes.

### 9.5 Fine-tuning sanity check
A deliberately tiny run tested the fine-tuning path: 6 saved samples and 15
steps.
* The 6 personal samples went to **100 %** correct.
* Synthetic validation CER rose from 6.2 % to 7.2 %. That is *catastrophic
  forgetting* in miniature.

The default fine-tuning settings are a small learning rate (2e-4), 3 epochs,
and only 30 % personal samples per batch mixed with fresh synthetic words. The
script prints both numbers every epoch, so you can see the trade-off.

### 9.6 Summary

| | Model A | Model B |
|---|---|---|
| Parameters | 0.44 M | 1.21 M |
| Training data | 678k real characters | 384k synthetic words (from real characters) |
| Training time (6 CPU threads) | ~30 min (3 epochs) | ~80 min (12 epochs) |
| Single-character accuracy | 86.0 % (top-3 99.1 %) | — |
| Word accuracy (synthetic test) | 16.2 % | **70.1 %** |
| Word accuracy (canvas-like strokes) | 30 % | **85 %** |
| Inference time per word (CPU) | ~25 ms | ~13 ms (+ beam search) |

---

## 10. Domain gap, fine-tuning, and exercises

**Domain gap.** Training data is EMNIST, scanned pen strokes. The dashboard
gives mouse or touch strokes that are smoother, with a constant width and
sometimes different letter shapes. Some accuracy loss on your own drawings
is expected. The project offers the standard remedy, **fine-tuning (transfer
learning)**:

1. Draw words on the dashboard, correct the text, and press *Save sample*
   (saved to `data/user_samples/`).
2. Run `python -m hwr.train_crnn --finetune`. It starts from the trained
   weights, uses a small learning rate (2e-4) and only 3 short epochs, and
   **mixes** your samples (30 % of each batch) with fresh synthetic words.
3. Press *reload models* on the dashboard.

Why mix? Training only on 30 personal words would overwrite what was learned
from 400k synthetic words. This is **catastrophic forgetting**. Validation
CER on the synthetic set is printed every epoch, so you can watch for it.

**Exercises** (good for a report):
1. **Case by height for Model A.** After classifying a segment as `o`/`O`,
   decide the case from the segment's height relative to the guide lines.
   How much does Model A's word accuracy improve?
2. **Projection-profile splitting.** Split very wide components at the column
   with the least ink. Does it fix touching letters?
3. **Lexicon decoding.** Restrict beam search to dictionary words (prefix
   tree). What happens on random strings like `xQ7b`?
4. **Ablations.**
   - Remove the BiLSTM: does the CNN alone with CTC still work?
   - Make the LSTM unidirectional.
   - Train with dictionary words only, then test on random strings.
5. **Compare with a Transformer** decoder (TrOCR-style attention instead of
   CTC).

---

## 11. Lecture concept → code index

Click a link to jump to the exact line (VS Code / GitHub).

| Lecture concept | Where in the code | Note |
|---|---|---|
| Convolution + BatchNorm + ReLU block | [hwr/models/char_cnn.py:29](../hwr/models/char_cnn.py#L29) | bias=False because BN has β |
| Max-pooling, dropout, fully-connected classifier | [hwr/models/char_cnn.py:54](../hwr/models/char_cnn.py#L54) |  |
| Input normalization stored in the model (buffers) | [hwr/models/char_cnn.py:43](../hwr/models/char_cnn.py#L43) | no train/serve skew |
| Forward / backward / optimizer step (training loop) | [hwr/train_char_cnn.py:10](../hwr/train_char_cnn.py#L10) |  |
| Cross-entropy + label smoothing | [hwr/train_char_cnn.py:80](../hwr/train_char_cnn.py#L80) |  |
| AdamW (decoupled weight decay) | [hwr/train_char_cnn.py:81](../hwr/train_char_cnn.py#L81) |  |
| OneCycle learning-rate schedule | [hwr/train_char_cnn.py:84](../hwr/train_char_cnn.py#L84) | warm-up + cosine annealing |
| Train / validation split | [hwr/train_char_cnn.py:65](../hwr/train_char_cnn.py#L65) | hold-out 20k images |
| model.eval() + inference_mode for evaluation | [hwr/train_char_cnn.py:39](../hwr/train_char_cnn.py#L39) | BN uses running stats, dropout off |
| Keep the best checkpoint (model selection on validation) | [hwr/train_char_cnn.py:125](../hwr/train_char_cnn.py#L125) |  |
| Random affine augmentation (affine_grid / grid_sample) | [hwr/augment.py:16](../hwr/augment.py#L16) |  |
| Morphological augmentation via max-pooling | [hwr/augment.py:46](../hwr/augment.py#L46) | dilation / erosion |
| IDX binary format parsing | [hwr/emnist.py:80](../hwr/emnist.py#L80) |  |
| EMNIST transpose fix | [hwr/emnist.py:124](../hwr/emnist.py#L124) |  |
| Synthetic data generation (word composer) | [hwr/synth.py:139](../hwr/synth.py#L139) | guide lines, stroke width, spacing |
| Text mixture against a language-model bias | [hwr/synth.py:93](../hwr/synth.py#L93) |  |
| Shared preprocessing (train/serve skew) | [hwr/preprocess.py:73](../hwr/preprocess.py#L73) |  |
| EMNIST-style character normalization | [hwr/preprocess.py:111](../hwr/preprocess.py#L111) |  |
| Connected components segmentation | [hwr/segment.py:54](../hwr/segment.py#L54) | 8-connectivity |
| CRNN: map-to-sequence (feature columns = time steps) | [hwr/models/crnn.py:79](../hwr/models/crnn.py#L79) |  |
| Bidirectional LSTM | [hwr/models/crnn.py:63](../hwr/models/crnn.py#L63) |  |
| pack_padded_sequence (variable-length sequences) | [hwr/models/crnn.py:86](../hwr/models/crnn.py#L86) |  |
| log-softmax output in (T, B, C) layout | [hwr/models/crnn.py:92](../hwr/models/crnn.py#L92) |  |
| CTC loss | [hwr/train_crnn.py:119](../hwr/train_crnn.py#L119) |  |
| Gradient-norm clipping | [hwr/train_crnn.py:148](../hwr/train_crnn.py#L148) | exploding gradients |
| channels_last memory format | [hwr/train_crnn.py:118](../hwr/train_crnn.py#L118) | CPU speed |
| Fine-tuning + catastrophic forgetting | [hwr/train_crnn.py:99](../hwr/train_crnn.py#L99) |  |
| Width bucketing in an IterableDataset | [hwr/datasets.py:97](../hwr/datasets.py#L97) |  |
| Per-worker reproducible random streams | [hwr/datasets.py:92](../hwr/datasets.py#L92) |  |
| Padding + collate for CTC | [hwr/datasets.py:34](../hwr/datasets.py#L34) |  |
| CTC greedy (best-path) decoding | [hwr/ctc.py:33](../hwr/ctc.py#L33) |  |
| CTC prefix beam search | [hwr/ctc.py:62](../hwr/ctc.py#L62) |  |
| log-sum-exp trick | [hwr/ctc.py:54](../hwr/ctc.py#L54) | numerical stability |
| Levenshtein distance (dynamic programming) | [hwr/metrics.py:22](../hwr/metrics.py#L22) |  |
| CER / word accuracy | [hwr/metrics.py:34](../hwr/metrics.py#L34) |  |
| Balanced accuracy, top-k | [hwr/metrics.py:50](../hwr/metrics.py#L50) |  |
| Confusion matrix | [hwr/metrics.py:71](../hwr/metrics.py#L71) |  |
| Comparison experiment A vs B | [hwr/evaluate.py:102](../hwr/evaluate.py#L102) | touching vs separated |
| Safe checkpoint loading (weights_only) | [hwr/utils.py:25](../hwr/utils.py#L25) |  |
| Serving the model behind an HTTP API | [app/server.py:74](../app/server.py#L74) |  |
| ONNX export + numerical check against PyTorch | [hwr/export_web.py:62](../hwr/export_web.py#L62) | deployment |
| Same preprocessing in the browser (Pillow-exact resize) | [app/static/inference.js:88](../app/static/inference.js#L88) | no train/serve skew on the web |

---

## 12. Deployment: the models in the browser

Live demo: **https://aizkp.github.io/hand_writing/**

GitHub Pages serves only static files, so there is no Python and no PyTorch
on the server side. Deployment therefore moves *inference* into the visitor's
browser (`python -m hwr.export_web`):

1. **ONNX export.**
   - `torch.onnx.export` traces each network into an ONNX graph: Conv,
     BatchNormalization, MaxPool, LSTM, Gemm, LogSoftmax, …
   - The CRNN's input width is declared a **dynamic axis**, so words of any
     length work.
   - BatchNorm in eval mode is just an affine transform, and dropout
     disappears, which is why the model must be in `eval()` before export.
2. **Verify the export.** ONNX Runtime and PyTorch are compared on random
   inputs of several widths: max difference < 1e-5.
3. **Port everything that is not a neural network to JavaScript**
   (`app/static/inference.js`): canvas → ink, `band_to_model_input`,
   segmentation, EMNIST normalization, greedy and prefix-beam CTC decoding.
   - Resizing images is the dangerous part. Browsers' built-in canvas scaling
     uses a *different* filter than Pillow, which would create a
     train/serve skew in the deployed model.
   - So we ported Pillow's LANCZOS resampler bit-exactly: the same filter
     coefficients, the same 22-bit fixed-point arithmetic, horizontal pass
     first.
4. **End-to-end check.** In headless Chrome, 40 words were drawn with real
   mouse events. Each canvas was recognized in the browser *and* sent to the
   Python server. All 40 gave identical model-input pixels, CTC paths,
   segmentation boxes and texts.

The model files are small: 4.8 MB for the CRNN and 1.8 MB for the CharCNN.
ONNX Runtime Web runs them with WebAssembly in ~20–30 ms per word, so the
static site is fast and private: drawings never leave the browser.
