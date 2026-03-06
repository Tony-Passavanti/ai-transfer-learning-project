# Transfer Learning Image Classifier with Baseline vs Fine-Tuned Comparison

## Project Overview

This project trains and evaluates two transfer learning strategies using
a pretrained ResNet18 on a custom dataset of tool images. The goal is to
compare the performance of a **baseline model with a frozen backbone**
against a **fine-tuned model with partially unfrozen layers**.

A Streamlit UI allows a user to upload an image and see predictions from
both models **side-by-side**, along with confusion matrices computed on
a held-out test set.

This project demonstrates:

-   Transfer learning
-   Experimental design (ablation study)
-   Evaluation metrics
-   Model comparison
-   Interactive inference UI

------------------------------------------------------------------------

# Dataset

## Classes

hammer screwdriver power_drill saw shovel

Total classes: **5**

Expected dataset size:

-   \~50--100 images per class
-   Total dataset size: \~250--500 images

------------------------------------------------------------------------

# Dataset Directory Structure

The dataset must be arranged in the following structure before training
begins:

data/ train/ hammer/ screwdriver/ power_drill/ saw/ shovel/

val/ hammer/ screwdriver/ power_drill/ saw/ shovel/

test/ hammer/ screwdriver/ power_drill/ saw/ shovel/

Split ratio:

70% train\
15% validation\
15% test

------------------------------------------------------------------------

# Human Tasks (Outside Codex)

These steps must be completed manually before running training.

## Task H1 --- Collect Dataset

Go to a hardware store (example: Lowe's) and take photos of tools.

Capture images for each class:

-   hammer
-   screwdriver
-   power drill
-   saw
-   shovel

Guidelines:

-   vary angles
-   vary lighting
-   vary background
-   avoid near duplicates

Goal:

50--100 images per class

------------------------------------------------------------------------

## Task H2 --- Organize Dataset

Place images into folders by class:

raw_data/ hammer/ screwdriver/ power_drill/ saw/ shovel/

------------------------------------------------------------------------

## Task H3 --- Run Dataset Split Script

Codex will create a script:

src/split_dataset.py

You will run:

python src/split_dataset.py

This script will:

-   split images into train / val / test
-   copy them into `data/` directory structure

------------------------------------------------------------------------

# Project Folder Structure

ai-transfer-learning-project/

IMPLEMENTATION_PLAN.md

data/ train/ val/ test/

models/

runs/

src/ data.py model.py train.py evaluate.py split_dataset.py utils.py

app/ streamlit_app.py

requirements.txt README.md

------------------------------------------------------------------------

# Model Architecture

Backbone:

ResNet18 (pretrained on ImageNet)

Framework:

PyTorch + TorchVision

Classifier head:

Linear(512 → 5)

------------------------------------------------------------------------

# Training Strategy

Two models will be trained.

## Model A --- Baseline

Backbone layers:

Frozen

Only train:

final fully connected layer

------------------------------------------------------------------------

## Model B --- Fine-Tuned

Start from baseline weights.

Unfreeze:

layer4\
fc

Train with smaller learning rate.

------------------------------------------------------------------------

# Hyperparameters

Recommended defaults:

batch_size = 32\
baseline_epochs = 10\
finetune_epochs = 10\
learning_rate_baseline = 1e-3\
learning_rate_finetune = 1e-4

Optimizer:

Adam

Loss:

CrossEntropyLoss

------------------------------------------------------------------------

# Data Transforms

Training transforms:

Resize(256)\
CenterCrop(224)\
RandomHorizontalFlip\
RandomRotation(10)\
ToTensor\
Normalize(mean, std from ImageNet)

Validation / test transforms:

Resize(256)\
CenterCrop(224)\
ToTensor\
Normalize(mean, std from ImageNet)

------------------------------------------------------------------------

# Metrics

Evaluate both models on the **test dataset**.

Metrics to compute:

accuracy\
precision\
recall\
F1 score\
confusion matrix

Artifacts saved to:

runs/ baseline_metrics.json finetuned_metrics.json
baseline_confusion_matrix.npy finetuned_confusion_matrix.npy

------------------------------------------------------------------------

# Codex Implementation Tasks

Codex CLI should implement the following tasks sequentially.

------------------------------------------------------------------------

# TASK 1 --- Create requirements.txt

Dependencies:

torch\
torchvision\
streamlit\
scikit-learn\
matplotlib\
numpy\
pillow

------------------------------------------------------------------------

# TASK 2 --- Implement Dataset Loader

File:

src/data.py

Functions:

def get_dataloaders(data_dir: str, batch_size: int): \"\"\" Returns
train, val, and test DataLoaders. Uses torchvision.datasets.ImageFolder.
\"\"\"

------------------------------------------------------------------------

# TASK 3 --- Implement Dataset Split Script

File:

src/split_dataset.py

Responsibilities:

-   read from `raw_data/`
-   split into train/val/test
-   copy images into `data/`

Split ratios:

70 / 15 / 15

------------------------------------------------------------------------

# TASK 4 --- Implement Model Builder

File:

src/model.py

Function:

def build_resnet18(num_classes: int, freeze_backbone: bool):

Responsibilities:

-   load pretrained ResNet18
-   replace final FC layer
-   freeze backbone if requested

------------------------------------------------------------------------

# TASK 5 --- Implement Training Script

File:

src/train.py

Responsibilities:

Train two models:

Baseline: freeze_backbone = True

Train classifier only.

Save model:

models/baseline.pt

Fine-tuned:

Load baseline weights.

Unfreeze:

layer4\
fc

Train additional epochs.

Save:

models/finetuned.pt

------------------------------------------------------------------------

# TASK 6 --- Implement Evaluation Script

File:

src/evaluate.py

Responsibilities:

-   load model
-   run inference on test dataset
-   compute metrics
-   generate confusion matrix

Save results:

runs/*.json\
runs/*.npy

------------------------------------------------------------------------

# TASK 7 --- Build Streamlit UI

File:

app/streamlit_app.py

UI features:

## Image Upload

User uploads one image.

## Side-by-Side Prediction

| Baseline Model \| Fine-Tuned Model \|

Each panel shows:

Predicted class\
Confidence score\
Top-3 predictions

## Confusion Matrix Section

Two panels:

Baseline Confusion Matrix\
Fine-Tuned Confusion Matrix

Add UI toggle:

Raw counts / Normalized

Use matplotlib to render matrices.

------------------------------------------------------------------------

# Inference Pipeline

When image uploaded:

1.  apply test transform\
2.  run inference on both models\
3.  compute softmax probabilities\
4.  display results

------------------------------------------------------------------------

# Reproducibility

Set random seeds:

torch.manual_seed(42)\
numpy.random.seed(42)\
random.seed(42)

------------------------------------------------------------------------

# Running the Project

Step 1 --- Install dependencies

pip install -r requirements.txt

Step 2 --- Split dataset

python src/split_dataset.py

Step 3 --- Train models

python src/train.py

Step 4 --- Evaluate models

python src/evaluate.py

Step 5 --- Run UI

streamlit run app/streamlit_app.py

------------------------------------------------------------------------

# Expected Result

When uploading an image, the UI displays:

Baseline prediction\
Fine-tuned prediction\
Confidence scores

Below that:

Confusion matrices for both models

Allowing visual comparison of model performance.
