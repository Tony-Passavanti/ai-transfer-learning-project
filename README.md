# Hammer Subtype Classifier — Transfer Learning Demo

A transfer learning project that fine-tunes ResNet50 to distinguish
between four types of hammers: **ball-peen hammer**, **claw hammer**,
**rubber mallet**, and **sledge hammer**.

The Streamlit demo lets you pick a sample image and see side-by-side
predictions from two models:

- **Baseline** — stock ImageNet-pretrained ResNet50 with its original
  1000-class output head, completely untrained on hammer images. It can
  only predict generic ImageNet categories and cannot distinguish
  between hammer subtypes.
- **Fine-tuned** — `layer3`, `layer4`, and classifier unfrozen and
  trained on hammer images with strong augmentation. Much better at
  subtype discrimination.

## Quick Start

### 1. Create and activate a virtual environment

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv; .venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the demo

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

## Full Pipeline (training from scratch)

```bash
python src/split_dataset.py      # raw_data/ → data/{train,val,test}/
python src/train.py              # trains baseline + fine-tuned models
python src/evaluate.py           # computes metrics and confusion matrices
streamlit run app/streamlit_app.py
```

## Project Structure

```
app/streamlit_app.py        # demo UI
media/                      # curated images for the demo gallery
src/model.py                # ResNet50 builder
src/train.py                # training pipeline
src/evaluate.py             # evaluation + metrics
src/data.py                 # DataLoader factory
src/split_dataset.py        # dataset splitter
src/utils.py                # seeds, transforms, device helpers
models/                     # saved .pt weight files
runs/                       # metrics JSON + confusion matrix .npy
```
