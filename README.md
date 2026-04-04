# Hammer Subtype Classifier — Transfer Learning Demo

A transfer learning project that fine-tunes ResNet18 to distinguish
between four types of hammers: **ball-peen hammer**, **claw hammer**,
**rubber mallet**, and **sledge hammer**.

The Streamlit demo lets you pick a sample image and see side-by-side
predictions from two models:

- **Baseline** — frozen ImageNet backbone, only the classifier head is
  trained. Struggles to tell similar hammer types apart.
- **Fine-tuned** — backbone `layer4` unfrozen and trained further on
  hammer images. Much better at subtype discrimination.

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
src/model.py                # ResNet18 builder
src/train.py                # training pipeline
src/evaluate.py             # evaluation + metrics
src/data.py                 # DataLoader factory
src/split_dataset.py        # dataset splitter
src/utils.py                # seeds, transforms, device helpers
models/                     # saved .pt weight files
runs/                       # metrics JSON + confusion matrix .npy
```
