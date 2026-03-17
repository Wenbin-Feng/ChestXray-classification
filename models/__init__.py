"""Base model interface and model factory."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Type

from config import config


class BaseClassifier(nn.Module, ABC):
    """Base class for all disease classifiers."""
    
    def __init__(self, num_classes: int = 3, img_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
    @abstractmethod
    def get_backbone_output_dim(self) -> int:
        """Return the output dimension of the backbone."""
        pass
    
    def _build_classifier(self, input_dim: int) -> nn.Sequential:
        """Build the classification head."""
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )


def create_model(model_type: str = None, **kwargs) -> BaseClassifier:
    """Factory function to create models."""
    model_type = model_type or config.model.type
    
    if model_type == "vit":
        from .vit_classifier import ViTClassifier
        return ViTClassifier(
            num_classes=kwargs.get("num_classes", config.model.num_classes),
            img_size=kwargs.get("img_size", config.model.img_size),
            pretrained=kwargs.get("pretrained", config.model.pretrained)
        )
    elif model_type == "mamba":
        from .mamba_classifier import MambaClassifier
        return MambaClassifier(
            num_classes=kwargs.get("num_classes", config.model.num_classes),
            img_size=kwargs.get("img_size", config.model.img_size),
            pretrained=kwargs.get("pretrained", config.model.pretrained)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
