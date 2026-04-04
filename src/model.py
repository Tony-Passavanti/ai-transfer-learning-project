"""ResNet18 model builder for baseline and fine-tuned configurations."""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Load a pretrained ResNet18 and replace the classifier head.

    Args:
        num_classes: Number of output classes (4 for hammer subtypes).
        freeze_backbone: If True, freeze all layers except the final fc layer.
            Used for baseline training.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def unfreeze_for_finetuning(model: nn.Module) -> nn.Module:
    """Unfreeze layer4 and fc for fine-tuning, keep everything else frozen.

    Call this after loading baseline weights to prepare for the
    fine-tuning phase.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Unfreeze fc
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
