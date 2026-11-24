# FER2013 Facial Emotion Recognition

PyTorch project for classifying grayscale facial-expression images into seven
classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Dataset

The project uses the Kaggle
[Challenges in Representation Learning: Facial Expression Recognition](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)
dataset. Each FER2013 image is 48 × 48 pixels. The labelled data is divided
into Training, PublicTest (validation), and PrivateTest (held-out test) splits.

Place the source files under `data/`:

- `icml_face_data.csv` — labelled FER2013 data, with `emotion`, `pixels`, and
  `Usage` columns.
- `test.csv` — Kaggle's unlabelled images, with a `pixels` column.

The notebooks create compressed processed splits in `data/processed/`. All
files under `data/` are excluded from Git because of their size.

## Setup

Use Python 3.10 or later. From this project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

In VS Code, select the `.venv` interpreter and run the notebooks in order:

1. `src/notebooks/01 - Data Processing.ipynb` — parse FER2013, inspect class
   distribution and save normalized-data statistics.
2. `src/notebooks/02 - Benchmark Epoch.ipynb` — estimate CPU training time for
   the four CNN configurations.
3. `src/notebooks/03 - Train.ipynb` — run a short stratified smoke test by
   default. Set `CFG["smoke"] = False` and adjust `CFG["epochs"]` to train on
   the complete Training and PublicTest splits. A class-balanced sampler
   compensates for FER2013's uneven class distribution.
4. `src/notebooks/04 - Evaluation.ipynb` — select the newest best checkpoint,
   report metrics on the held-out PrivateTest split, save a confusion matrix
   and classification report, and optionally export Kaggle predictions.

## Models and outputs

`src/models.py` defines the `small`, `medium`, `medium_01`, and `large` CNNs.
Checkpoints and per-epoch CSV histories are written to `exps/checkpoints/` and
`exps/history/`; timestamped application logs are written to `logs/`.
Generated checkpoints and logs are ignored by Git.

The evaluation notebook writes `exps/final_metrics.json`,
`exps/classification_report.csv`, `exps/confusion_matrix.png`, and
`exps/submission.csv`. The test metric is only meaningful for the checkpoint
and data split recorded with it; a smoke-test or reduced-data run is not a
final model result.

## Project files

- `src/` — PyTorch model, data, training, evaluation utilities, and notebooks.
- `data/` — source and cached FER2013 data (not tracked by Git).
- `exps/` — experiment histories, plots, metrics, and optional predictions.
- `docs/` — Vietnamese project report and presentation.
- `guide/` — instructor-provided reference materials.
