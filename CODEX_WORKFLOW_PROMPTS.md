# Codex CLI Workflow Prompts for Transfer Learning Project

This file contains the prompts to paste into **Codex CLI** while
implementing the project described in `IMPLEMENTATION_PLAN.md`.

Work through the prompts sequentially. Each section corresponds to one
implementation task.

------------------------------------------------------------------------

# Session Initialization Prompt

Paste this once at the start of your Codex session.

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

------------------------------------------------------------------------

# TASK 1 --- Requirements and README (COMPLETE)

    Implement TASK 1 from IMPLEMENTATION_PLAN.md.

    Create requirements.txt and a minimal README.md with setup and run commands for a hosted VM.

    Then run:

    pip install -r requirements.txt

    python -c "import torch, torchvision, streamlit, sklearn, matplotlib, numpy, PIL"

    Fix any import errors.

------------------------------------------------------------------------

# TASK 2 --- Data Loader

    Implement TASK 2.

    Create src/data.py with get_dataloaders(data_dir, batch_size) and transforms per the plan.

    Add a small CLI (argparse) so I can run:

    python src/data.py --data_dir data --batch_size 4

    If data/ doesn't exist, print a friendly error and exit non-zero.

------------------------------------------------------------------------

# TASK 3 --- Dataset Split Script

    Implement TASK 3.

    Create src/split_dataset.py that reads raw_data/ and creates data/train,val,test with 70/15/15.

    Requirements:
    - deterministic with seed 42
    - supports jpg, jpeg, png, webp
    - copies files (not moves)
    - prints counts per class per split
    - writes runs/class_names.json mapping index->class based on sorted folder names

    Add argparse so I can run:

    python src/split_dataset.py --raw_dir raw_data --out_dir data --seed 42

    Then run:

    python src/split_dataset.py --help

------------------------------------------------------------------------

# TASK 4 --- Model Builder

    Implement TASK 4.

    Create src/model.py with:
    - build_resnet18(num_classes, freeze_backbone)
    - a helper to unfreeze layer4 + fc for finetuning

    Then run:

    python -c "from src.model import build_resnet18; m=build_resnet18(5, True); print('trainable params', sum(p.requires_grad for p in m.parameters()))"

------------------------------------------------------------------------

# TASK 5 --- Training Script

    Implement TASK 5.

    Create src/train.py that:

    - trains baseline (frozen backbone) and saves models/baseline.pt
    - fine-tunes from baseline by unfreezing layer4+fc and saves models/finetuned.pt
    - saves runs/baseline_metrics.json and runs/finetuned_metrics.json including curves + hyperparameters
    - uses validation set for best checkpoint selection
    - auto-detects device (cuda if available else cpu)

    Add argparse so I can run:

    python src/train.py --data_dir data --batch_size 32 --baseline_epochs 10 --finetune_epochs 10

    Then run:

    python src/train.py --help

------------------------------------------------------------------------

# TASK 6 --- Evaluation Script

    Implement TASK 6.

    Create src/evaluate.py that:

    - loads both model checkpoints
    - evaluates on the test dataset
    - computes accuracy, precision, recall, macro F1
    - generates confusion matrices
    - saves metrics to runs/*.json
    - saves confusion matrices to runs/*.npy
    - prints a summary table

    Add argparse so I can run:

    python src/evaluate.py --data_dir data --baseline_ckpt models/baseline.pt --finetuned_ckpt models/finetuned.pt

    Then run:

    python src/evaluate.py --help

------------------------------------------------------------------------

# TASK 7 --- Streamlit UI

    Implement TASK 7.

    Create app/streamlit_app.py that:

    - loads models and runs/class_names.json
    - allows one image upload
    - runs inference on both models
    - displays results side-by-side:
        predicted class
        confidence
        top-3 predictions

    Add confusion matrices below the predictions:

    - baseline confusion matrix
    - fine-tuned confusion matrix

    Include a toggle:

    Raw counts
    Normalized

    Use matplotlib for rendering.

    Then run:

    python -m py_compile app/streamlit_app.py

------------------------------------------------------------------------

# Debug Prompt (Use Whenever Something Fails)

    The previous command failed. Here is stdout/stderr:

    <PASTE ERROR>

    Please:
    1) Identify the root cause precisely.
    2) Apply the minimal fix.
    3) Re-run the same command and confirm success.
    Do not proceed until the command succeeds.
