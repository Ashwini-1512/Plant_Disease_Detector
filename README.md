# Plant Disease Detector

EfficientNetB0 transfer-learning model for plant disease detection across 38 PlantVillage classes, with Grad-CAM explainability and a deployed Flask web app.

## Overview

This project classifies a photo of a plant leaf into one of 38 crop/disease categories from the PlantVillage taxonomy (e.g. Tomato — Early Blight, Apple — Cedar Apple Rust, Corn — Healthy), and explains *why* it made that prediction using Grad-CAM, an explainability technique that highlights which regions of the leaf image the model actually looked at.

- **Model** — EfficientNetB0, pretrained on ImageNet, fine-tuned in two stages: first with the base frozen (training only the new classification head), then with the last 100 layers unfrozen and fine-tuned end-to-end at a low learning rate.
- **Data** — a cleaned, balanced manifest (`plantvillage_cleaned_balanced.csv`) drives which classes and images are used, with the real PlantVillage photos downloaded automatically from Kaggle and materialised locally to match the manifest.
- **Explainability** — Grad-CAM heatmaps are generated for the predicted class and overlaid on the original image, shown on the website as "Evidence Gathered by the Model."
- **Deployment** — a self-contained Flask web app that a user can upload a leaf photo to and get an instant prediction, confidence score, Grad-CAM overlay, treatment advice, and a running history of recent checks. An About tab lists every crop and disease the model can detect.
- **Evaluation** — includes a held-out test set evaluation and a System Usability Scale (SUS) usability evaluation section.

## Repository contents

| File | Description |
|---|---|
| `plantdiseasedetector.py` | Full end-to-end pipeline: dependency install, manifest cleaning, real-photo download and materialisation, augmentation, two-stage EfficientNetB0 training, evaluation, Grad-CAM, model export, and the Flask deployment. Written to run in Google Colab. |
| `plantvillage_cleaned_balanced.csv` | Cleaned, balanced manifest of image classes used to drive training and downloading. |
| 'kaggle.json' | Create a Kaggle container to store the sample images from kaggle. |

## Tech stack

Python 3 · TensorFlow / Keras (EfficientNetB0) · Flask · pandas · NumPy · Kaggle API (for the PlantVillage image download)

## Running it

1. Open a new Google Colab notebook and select a GPU runtime (Runtime → Change runtime type → GPU). Training on CPU is possible but very slow.
2. Paste the contents of `plantdiseasedetector.py` into a single cell and run it.
3. When prompted, upload `plantvillage_cleaned_balanced.csv`and after that upload 'kaggle.json'.
4. The pipeline downloads the matching real PlantVillage photos from Kaggle automatically, trains EfficientNetB0 in two stages, evaluates on a held-out test set, and generates Grad-CAM visualisations.
5. The pipeline then writes and launches the Flask website automatically.

### Fast mode vs. full run

The script includes two speed/accuracy controls near the top:

```python
FAST_MODE = True     # True = quick smoke-test run. Set False for a real accuracy run.
SAMPLE_FRAC = 0.25    # Fraction of the manifest to use (1.0 = all of it — recommended for real results).
```

`FAST_MODE=True` runs a short 6+4 epoch smoke test on 25% of the data to confirm the pipeline works end-to-end. For a genuine accuracy run, set `FAST_MODE = False` and `SAMPLE_FRAC = 1.0` (20 + 15 epochs on the full dataset).

## Web app features

- Upload a leaf photo and get an instant prediction with a confidence score.
- Grad-CAM heatmap overlay showing exactly which part of the leaf the model based its decision on.
- Treatment/advice text specific to the predicted disease.
- A running history of recent checks.
- An About tab listing every crop and disease the model is able to detect.
