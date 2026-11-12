# Handwritten Word Recognition (A–Z, a–z, 0–9)

Draw a word on a web dashboard and two deep-learning models read it. Everything is
built with PyTorch and trained on a CPU.

▶ **Live demo: https://aizkp.github.io/hand_writing/**. The models run entirely in your browser
(ONNX Runtime Web), so nothing you draw is uploaded.

| | Model A | Model B |
|---|---|---|
| Idea | cut the word into letters, classify each | read the whole word at once |
| Network | CharCNN (VGG-style CNN, 0.44 M params) | CRNN = CNN → BiLSTM → CTC (1.2 M params) |
| Trained on | EMNIST ByClass characters | synthetic words built from EMNIST characters |
| Loss | cross-entropy | CTC |

📘 **Study guide with all the theory: [docs/THEORY.md](docs/THEORY.md)**

### Results (CPU only, details in [THEORY.md §9](docs/THEORY.md#9-results-and-discussion))

| | Model A: seg + CharCNN | Model B: CRNN + CTC |
|---|---|---|
| Single characters (EMNIST test, 116k) | 86.0 % (top-3 99.1 %) | — |
| Words, synthetic test set (3,000) | 16.2 % | **70.1 %** (CER 6.0 %) |
| … with touching letters | 4.8 % | **65.6 %** |
| Words drawn like on the canvas (40) | 30 % | **85 %** |

Model A's errors are mostly upper/lower-case and look-alike confusions (`Hell0`, `R00M42`) plus
touching letters it can't separate. Model B reads the whole word and uses each letter's height
relative to the guide lines, so it makes almost no case errors.

---

## 1. Setup

```bash
cd ~/Documents/lecture/DL

# Normal route (needs the python3-venv package):
python3 -m venv .venv
# On this machine python3-venv/pip were missing, so the venv was bootstrapped with:
#   python3 -m venv --without-pip .venv
#   curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && .venv/bin/python /tmp/get-pip.py

.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build (~200 MB)
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate          # the commands below assume the venv is active
```

## 2. Run everything

| Step | Command | Time on 6 CPU threads |
|---|---|---|
| Download EMNIST (536 MB), build word lists and fixed val/test sets | `python -m hwr.prepare_data` | ~2 min |
| Train Model B (CRNN + CTC) | `python -m hwr.train_crnn` | ~80 min |
| Train Model A (CharCNN) | `python -m hwr.train_char_cnn` | ~10 min per epoch |
| Evaluate and compare both | `python -m hwr.evaluate` | ~3 min |
| Start the dashboard | `python -m app.server` → open http://127.0.0.1:8000 | |
| Unit tests | `python -m pytest` | ~15 s |
| Build the static website (ONNX export) | `python -m hwr.export_web` → `build/site/` | ~10 s |

Every script accepts `--help`. Quick smoke test: `python -m hwr.train_crnn --max-steps 30`.

Outputs:
* `checkpoints/char_cnn_best.pt`, `checkpoints/crnn_best.pt`: the best weights by validation
  metric. They contain only tensors and numbers and are loaded with `torch.load(weights_only=True)`.
* `reports/`:
  * training logs (`*_log.csv`, `*_train.log`) and curves (`*_curves.png`);
  * the confusion matrix (`char_cnn_confusion.png`);
  * the comparison table (`comparison.md`) and example images.

## 3. Using the dashboard

* **Write between the guide lines.**
  * Lowercase letters (`a c e m n o r s u v w x z`) go from the dashed line to the baseline.
  * Capitals, digits and `b d f h k l` reach the top line.
  * `g j p q y` hang below the baseline.

  Model B uses these sizes to tell `o` from `O`.
* Shortcuts: <kbd>Enter</kbd> recognize, <kbd>Esc</kbd> clear, <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo.
  Recognition also runs automatically 0.6 s after you lift the pen.
* Panels:
  * **Model B** shows the beam-search result, per-letter confidence colours and the top-5
    alternatives.
  * **Model A** draws the segmentation boxes on your drawing and shows each 28×28 crop with its
    top-3 classes.
  * **What the CRNN sees** shows the exact 48-px input and the CTC best path for each time step
    (`·` = blank).
* **Correct & save** stores a drawing with its true text in `data/user_samples/`. After about 30
  words, run `python -m hwr.train_crnn --finetune` (a few minutes) and click *reload models*.
  The model then adapts to *your* handwriting.

The server listens on `127.0.0.1` only, so it isn't reachable from the internet.

## 4. Deployment (GitHub Pages)

GitHub Pages only serves static files, so the Python server can't run there. `python -m hwr.export_web` builds the
site instead:

1. It exports both networks to **ONNX** and checks that ONNX Runtime reproduces PyTorch's outputs (difference < 1e-5).
2. It copies the dashboard together with [`app/static/inference.js`](app/static/inference.js). That file is a
   JavaScript port of the preprocessing (including a bit-exact port of Pillow's Lanczos resize), the segmentation
   and the CTC decoders.
3. It writes `config.json` (canvas geometry, model metrics) and the report images.

On 40 test drawings the in-browser version produced **identical** results to the Python server: same model-input
pixels, CTC paths, boxes and texts. The site in `build/site/` is published on the `gh-pages` branch. The export
needs `pip install onnx onnxscript onnxruntime`.

## 5. Project layout

```
hwr/                   the Python package
  config.py            alphabet, writing-band geometry, sizes, paths (single source of truth)
  emnist.py            download + IDX parsing + transpose fix
  synth.py             synthetic word generator (guide-line geometry, stroke width, spacing)
  preprocess.py        shared preprocessing for training AND the dashboard
  augment.py           batch augmentation (affine_grid / grid_sample, morphology)
  segment.py           connected-component segmentation (Model A)
  models/char_cnn.py   Model A network
  models/crnn.py       Model B network
  ctc.py               greedy + prefix beam search decoding
  metrics.py           Levenshtein, CER, accuracy, confusion matrix
  datasets.py          width-bucketed synthetic stream, fixed val/test sets, user samples
  train_char_cnn.py    training script A
  train_crnn.py        training script B (+ --finetune)
  evaluate.py          comparison of A and B on the same test words
  inference.py         Recognizer used by the dashboard
  export_web.py        ONNX export + static site build (GitHub Pages)
app/server.py          FastAPI backend
app/static/            dashboard (HTML/CSS/JS); inference.js = in-browser version of the pipeline
tests/                 pytest unit tests
docs/THEORY.md         study guide
```
