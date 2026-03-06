# PROJECT_RUNBOOK.md

Transfer Learning Image Classifier Project Runbook

This document provides a step‑by‑step checklist to execute the full
project on a hosted VM. It includes:

• dataset collection steps\
• VM environment setup\
• Codex workflow prompts\
• training and evaluation commands\
• Streamlit UI launch\
• artifacts to capture for the assignment report

------------------------------------------------------------------------

# 1. Project Overview

Goal: Compare two transfer learning strategies using ResNet18.

Models:

Baseline • pretrained ResNet18 • backbone frozen • train classifier only

Fine‑tuned • start from baseline weights • unfreeze layer4 + fc •
continue training

Dataset:

5 tool classes hammer screwdriver power_drill saw shovel

Target dataset size: 50--100 images per class

------------------------------------------------------------------------

# 2. Folder Structure

ai-transfer-learning-project/

IMPLEMENTATION_PLAN.md\
CODEX_WORKFLOW_PROMPTS.md\
PROJECT_RUNBOOK.md

data/ train/ val/ test/

models/

runs/

src/ data.py model.py train.py evaluate.py split_dataset.py

app/ streamlit_app.py

requirements.txt README.md

------------------------------------------------------------------------

# 3. Dataset Collection (Human Task)

Go to a hardware store (example: Lowe's).

Capture photos of:

hammer\
screwdriver\
power drill\
saw\
shovel

Tips:

• vary angles • vary lighting • vary backgrounds • avoid duplicates •
include partially visible tools • avoid blurry images

Recommended per class:

50--100 images

Place images into:

raw_data/

raw_data/ hammer/ screwdriver/ power_drill/ saw/ shovel/

------------------------------------------------------------------------

# 4. Upload Dataset to VM

From your local computer:

scp -r raw_data USER@VM:/path/to/ai-transfer-learning-project/

Example:

scp -r raw_data student@vm.school.edu:\~/ai-transfer-learning-project/

------------------------------------------------------------------------

# 5. VM Environment Setup

SSH into VM:

ssh USER@VM

Navigate to project:

cd ai-transfer-learning-project

Create virtual environment:

python3 -m venv .venv

Activate:

source .venv/bin/activate

Upgrade pip:

python -m pip install --upgrade pip wheel setuptools

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

# 6. Codex Implementation Workflow

Open Codex CLI in the project directory.

Paste prompts from:

CODEX_WORKFLOW_PROMPTS.md

Execute tasks sequentially:

TASK 1 Create requirements and README

TASK 2 Create data loaders

TASK 3 Create dataset splitting script

TASK 4 Create model builder

TASK 5 Create training script

TASK 6 Create evaluation script

TASK 7 Create Streamlit UI

------------------------------------------------------------------------

# 7. Split Dataset

After Codex creates the script:

python src/split_dataset.py --raw_dir raw_data --out_dir data --seed 42

Verify output:

data/ train/ val/ test/

Each class folder should exist.

------------------------------------------------------------------------

# 8. Train Models

Run training:

python src/train.py\
--data_dir data\
--batch_size 32\
--baseline_epochs 10\
--finetune_epochs 10

If CPU training is slow:

python src/train.py\
--data_dir data\
--batch_size 32\
--baseline_epochs 6\
--finetune_epochs 6

Training artifacts:

models/ baseline.pt finetuned.pt

runs/ baseline_metrics.json finetuned_metrics.json

------------------------------------------------------------------------

# 9. Evaluate Models

Run evaluation:

python src/evaluate.py\
--data_dir data\
--baseline_ckpt models/baseline.pt\
--finetuned_ckpt models/finetuned.pt

Artifacts generated:

runs/ baseline_confusion_matrix.npy finetuned_confusion_matrix.npy

Metrics printed:

accuracy\
precision\
recall\
F1 score

------------------------------------------------------------------------

# 10. Run Streamlit UI

Launch UI:

streamlit run app/streamlit_app.py --server.port 8501 --server.address
0.0.0.0

If remote VM:

ssh -L 8501:localhost:8501 USER@VM

Then open browser:

http://localhost:8501

UI Features:

Upload tool image

View predictions side‑by‑side:

Baseline model\
Fine‑tuned model

Display:

predicted class\
confidence score\
top‑3 predictions

Below predictions:

confusion matrices for both models

Toggle:

raw counts\
normalized

------------------------------------------------------------------------

# 11. Validation Checklist

Confirm the following artifacts exist:

models/baseline.pt\
models/finetuned.pt

runs/baseline_metrics.json\
runs/finetuned_metrics.json

runs/baseline_confusion_matrix.npy\
runs/finetuned_confusion_matrix.npy

runs/class_names.json

Streamlit UI loads without errors.

------------------------------------------------------------------------

# 12. Screenshots for Assignment Submission

Capture screenshots of:

1.  Dataset folder structure

2.  Training console output

3.  Evaluation metrics

4.  Confusion matrices

5.  Streamlit UI with uploaded image

6.  Side‑by‑side predictions

------------------------------------------------------------------------

# 13. Suggested Report Sections

Include in assignment write‑up:

Introduction\
Dataset description\
Model architecture\
Training strategy\
Evaluation metrics\
Confusion matrix comparison\
Discussion of improvements from fine‑tuning

Key question:

Did fine‑tuning improve classification accuracy?

Explain why.

------------------------------------------------------------------------

# 14. Common Troubleshooting

Issue: Torch cannot find GPU

Solution:

Training automatically falls back to CPU.

Issue: Dataset not detected

Check:

data/train/class folders exist

Issue: Streamlit cannot load models

Verify:

models/baseline.pt exists

------------------------------------------------------------------------

# 15. Final Deliverables

Code repository

Trained models

Evaluation metrics

Confusion matrices

Working Streamlit demo

Assignment report
