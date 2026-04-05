"""Shared helpers: seeds, device, transforms, class names."""

import random

import numpy as np
import torch
from torchvision import transforms

# Canonical class labels in alphabetical order.
# Every module that maps indices ↔ names must use this list.
CLASS_NAMES = ["ball_peen_hammer", "claw_hammer", "rubber_mallet", "sledge_hammer"]

# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return cuda device if available, otherwise cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_transform() -> transforms.Compose:
    """Return the image transform pipeline for training data.

    Uses aggressive augmentation to help the model generalise from a
    small dataset of visually similar hammer types.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """Return the image transform pipeline for validation, test, and inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
