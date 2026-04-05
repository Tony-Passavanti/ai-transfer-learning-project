"""ResNet50 model builder for baseline and fine-tuned configurations."""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_resnet50(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Load a pretrained ResNet50 and replace the classifier head.

    Args:
        num_classes: Number of output classes (4 for hammer subtypes).
        freeze_backbone: If True, freeze all layers except the final fc layer.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer (2048 -> num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def unfreeze_for_finetuning(model: nn.Module) -> nn.Module:
    """Unfreeze layer3, layer4, and fc for fine-tuning.

    Unfreezing deeper layers lets the model adapt features for
    distinguishing visually similar hammer subtypes.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model
