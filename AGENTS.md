# AGENTS.md

This file defines how Codex should operate in this repository.

## Project Goal

Build a transfer learning image classification project comparing:
- Baseline ResNet18 (frozen backbone)
- Fine-tuned ResNet18 (unfrozen `layer4` + `fc`)

Includes:
- training/evaluation pipeline
- metrics + confusion matrices
- Streamlit side-by-side inference UI

Primary reference: `IMPLEMENTATION_PLAN.md`.

## Working Rules


You are implementing the project described in IMPLEMENTATION_PLAN.md in this repo on a hosted school VM.

Hard rules:
1) Follow IMPLEMENTATION_PLAN.md exactly (file structure, model strategy, outputs, metrics, UI requirements).
2) The code must run end-to-end on CPU; if CUDA is available, it may use it automatically.
3) Implement one TASK at a time. Keep changes small and reviewable.
4) After each TASK, run a validation command and show the output.
5) Make scripts robust with helpful errors if data directories are missing.
6) Save artifacts exactly:
   - models/baseline.pt, models/finetuned.pt
   - runs/baseline_metrics.json, runs/finetuned_metrics.json
   - runs/baseline_confusion_matrix.npy, runs/finetuned_confusion_matrix.npy
   - runs/class_names.json
7) Include reproducibility: set random seeds (torch/numpy/random).
8) Use torchvision ResNet18 pretrained. Baseline: freeze backbone + train fc. Fine-tuned: unfreeze layer4+fc.
9) Streamlit UI must do:
   - one image upload
   - side-by-side predictions (top1+confidence+top3)
   - confusion matrices for both + toggle raw vs normalized
10) Prefer clear CLI args, logging (device, dataset sizes, epochs, LR), and no interactive prompts.

Start with TASK 1 and proceed sequentially.
Before writing code, list the exact files to be created/modified for the TASK.
After implementing, run the validation command and fix any failures before moving on.

## Expected Structure

Create and maintain:

- `requirements.txt`
- `README.md`
- `src/data.py`
- `src/model.py`
- `src/train.py`
- `src/evaluate.py`
- `src/split_dataset.py`
- `src/utils.py`
- `app/streamlit_app.py`
- runtime dirs: `data/`, `models/`, `runs/`

## Runtime Commands

Install deps:

```bash
pip install -r requirements.txt
```

Split data:

```bash
python src/split_dataset.py
```

Train:

```bash
python src/train.py
```

Evaluate:

```bash
python src/evaluate.py
```

Run UI:

```bash
streamlit run app/streamlit_app.py
```

## Implementation Defaults

- Seed: `42` for `random`, `numpy`, and `torch`.
- Model: `torchvision.models.resnet18` with ImageNet pretrained weights.
- Metrics: accuracy, precision, recall, F1, confusion matrix.
- Save artifacts to:
  - `models/baseline.pt`
  - `models/finetuned.pt`
  - `runs/baseline_metrics.json`
  - `runs/finetuned_metrics.json`
  - `runs/baseline_confusion_matrix.npy`
  - `runs/finetuned_confusion_matrix.npy`

## Code Quality Checklist

Before finishing changes:

- Ensure scripts run from repo root with relative paths.
- Verify imports resolve without package hacks.
- Verify output directories are created when missing.
- Run a basic syntax check for touched Python files.
- Keep logs concise and actionable.

## Collaboration Style

- State what you are about to change before editing.
- After changes, summarize exactly what was created/modified.
- If a command cannot run due to missing deps/data, report it and continue with what can be completed.

