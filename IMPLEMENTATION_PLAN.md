# Implementation Plan

## Overview

Fine-tune a pretrained ResNet18 to classify four hammer subtypes.
Compare a frozen-backbone baseline against a fine-tuned model in an
interactive Streamlit demo. The demo lets a user pick from sample images
and see side-by-side predictions from both models.

---

## Step 1 — Project Scaffolding

**Files:** `requirements.txt`

- Confirm dependencies: `torch`, `torchvision`, `streamlit`,
  `scikit-learn`, `matplotlib`, `numpy`, `pillow`
- Create empty directories: `models/`, `runs/`, `data/`, `raw_data/`,
  `app/`, `src/`
- Verify imports work: `python -c "import torch, torchvision, streamlit, sklearn, matplotlib, numpy, PIL"`

---

## Step 2 — Shared Utilities

**Files:** `src/utils.py`

Provide reusable helpers used by every other module:

- `set_seeds(seed=42)` — sets `random`, `numpy`, `torch` seeds
- `get_device()` — returns `cuda` if available, else `cpu`
- `get_train_transform()` / `get_eval_transform()` — returns the
  standard image transform pipelines (ImageNet normalization)
- `CLASS_NAMES` — canonical sorted list of the 4 class labels

---

## Step 3 — Model Builder

**Files:** `src/model.py`

- `build_resnet18(num_classes, freeze_backbone=True)`:
  - Loads `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`
  - Replaces `fc` layer: `Linear(512, num_classes)`
  - If `freeze_backbone=True`, freezes everything except `fc`
- `unfreeze_for_finetuning(model)`:
  - Unfreezes `layer4` and `fc`, keeps everything else frozen
  - Used when transitioning from baseline to fine-tuning

**Validation:** instantiate both configurations and print trainable
parameter counts.

---

## Step 4 — Streamlit UI (Placeholder Mode)

**Files:** `app/streamlit_app.py`

Build the full UI shell. In this step, use the stock ImageNet-pretrained
ResNet18 (1000 classes) so the app is functional before any custom
training. This means predictions will show ImageNet labels like "hammer"
— that's fine as a placeholder.

### Layout

1. **Title and explanation** — brief description of what transfer
   learning is and what this demo shows.
2. **Sample image gallery** — display thumbnail grid from `media/`
   (13 images already present: bp1-3, ch1-4, rm1-3, sh1-3 covering
   all four hammer types). User clicks to select one.
3. **Side-by-side results** — two columns:
   - Left: "Baseline Model (Frozen Backbone)"
   - Right: "Fine-Tuned Model (Hammer-Specialized)"
   - Each column shows: predicted class, confidence %, top-3 list
### Inference logic

- Baseline is always the stock ImageNet-pretrained ResNet18 (1000 classes).
- Load fine-tuned weights from `models/finetuned.pt` if available.
- If the fine-tuned weight file is missing, fall back to stock ImageNet
  ResNet18 and display a banner.
- Apply eval transform → unsqueeze → forward pass → softmax → top-3.

**Validation:** `streamlit run app/streamlit_app.py` launches without
errors and renders the page with placeholder images.

---

## Step 5 — Data Loader

**Files:** `src/data.py`

- `get_dataloaders(data_dir, batch_size=32)`:
  - Uses `torchvision.datasets.ImageFolder` for `train/`, `val/`,
    `test/` subdirectories
  - Applies training transforms to train, eval transforms to val/test
  - Returns three `DataLoader` objects
- Print dataset sizes and class-to-index mapping when run as
  `__main__`.
- Friendly error if `data_dir` does not exist.

---

## Step 6 — Dataset Split Script

**Files:** `src/split_dataset.py`

- Reads from `raw_data/<class>/` (supports jpg, jpeg, png, webp)
- Splits 70/15/15 into `data/{train,val,test}/<class>/`
- Copies files (does not move)
- Deterministic with seed 42
- Prints per-class per-split counts
- Writes `runs/class_names.json`
- CLI: `python src/split_dataset.py [--raw_dir raw_data] [--out_dir data] [--seed 42]`

---

## Step 7 — Training Script

**Files:** `src/train.py`

This script will be copied to the school GPU server and run there.
It must be self-contained (importing only from `src/` siblings).

### Phase A — Baseline

- `build_resnet18(num_classes=4, freeze_backbone=True)`
- Train for `--baseline_epochs` (default 10) with lr=1e-3, Adam
- Track train/val loss and accuracy per epoch
- Save best-val-accuracy checkpoint to `models/baseline.pt`
- Save training curves to `runs/baseline_metrics.json`

### Phase B — Fine-Tuning

- Load baseline weights
- Call `unfreeze_for_finetuning(model)` (unfreezes `layer4` + `fc`)
- Train for `--finetune_epochs` (default 10) with lr=1e-4, Adam
- Save best-val-accuracy checkpoint to `models/finetuned.pt`
- Save training curves to `runs/finetuned_metrics.json`

### CLI

```
python src/train.py \
  --data_dir data \
  --batch_size 32 \
  --baseline_epochs 10 \
  --finetune_epochs 10
```

### Logging

Print: device, dataset sizes, learning rate, epoch progress with
train/val loss and accuracy.

---

## Step 8 — Collect Dataset and Train (Human Steps)

### 8a — Collect Images

Take or source ~50-100 photos per class of:
- Claw hammer
- Ball-peen hammer
- Sledge hammer
- Rubber mallet

Vary angles, lighting, and backgrounds. Place into:

```
raw_data/
  ball_peen_hammer/
  claw_hammer/
  rubber_mallet/
  sledge_hammer/
```

### 8b — Split Dataset

```bash
python src/split_dataset.py
```

### 8c — Train on Remote GPU Server

Copy these to the remote machine:
- `src/train.py`
- `src/model.py`
- `src/utils.py`
- `src/data.py`
- `data/` (the split dataset)
- `requirements.txt`

Run:
```bash
pip install -r requirements.txt
python src/train.py --data_dir data
```

### 8d — Bring Back Artifacts

Copy this file back to your local machine:

```
models/finetuned.pt
```

This is the model `state_dict` (learned weights). The UI loads it
to run fine-tuned inference.

---

## Step 9 — Final Assembly

- Sample images already in `media/` — no collection needed
- Confirm the UI loads real weights and shows hammer-subtype predictions
- Test the full user flow end-to-end
- Record the screen-capture demo with voiceover

---

## Step 10 (Optional) — Deploy Online

Options for free hosting:

| Platform | How |
|----------|-----|
| **Streamlit Community Cloud** | Connect GitHub repo, point to `app/streamlit_app.py`. Free. Easiest. |
| **HuggingFace Spaces** | Create a Space of type "Streamlit", push the repo. Free. |
| **Render** | Add a `render.yaml` or use their dashboard. Free tier available. |

For any of these, the model weight file (`models/finetuned.pt`) must be
committed to the repo or uploaded as an artifact, since the hosting
platform needs access to it.
