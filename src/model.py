
# ==========================================
# NEUROSCAN AI - MODEL ARCHITECTURE
# ==========================================

import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

NUM_CLASSES = 4


def build_resnet18(pretrained=True):

    # Step 1: Load ImageNet pretrained weights
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained else None
    )

    model = resnet18(weights=weights)

    # Step 2: Get the existing classifier input size
    in_features = model.fc.in_features

    # Step 3: Replace classifier with 4 outputs
    model.fc = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model




# ==========================================
# EFFICIENTNET-B0 MODEL
# ==========================================

def build_efficientnet_b0(pretrained=True):

    import torch.nn as nn

    from torchvision.models import (
        efficientnet_b0,
        EfficientNet_B0_Weights
    )

    # Load ImageNet-pretrained weights
    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = efficientnet_b0(
        weights=weights
    )

    # Get input features of classification layer
    in_features = model.classifier[1].in_features

    # Replace ImageNet classifier with four classes
    model.classifier[1] = nn.Linear(
        in_features,
        4
    )

    return model




# ==========================================
# DENSENET121 MODEL
# ==========================================

def build_densenet121(pretrained=True):

    import torch.nn as nn

    from torchvision.models import (
        densenet121,
        DenseNet121_Weights
    )

    weights = (
        DenseNet121_Weights.DEFAULT
        if pretrained
        else None
    )

    model = densenet121(weights=weights)

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        4
    )

    return model
