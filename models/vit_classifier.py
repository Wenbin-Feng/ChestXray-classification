"""ViT-based disease classifier."""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from config import config
from . import BaseClassifier


class ViTClassifier(BaseClassifier):
    """Vision Transformer based classifier for chest X-ray classification."""
    
    def __init__(self, num_classes: int = 3, img_size: int = 224, pretrained: bool = True):
        super().__init__(num_classes, img_size)
        
        # Load pretrained ViT from HuggingFace
        if pretrained:
            # Try to load local pretrained model, fallback to default
            try:
                self.vit = ViTForImageClassification.from_pretrained(
                    '/home/ubuntu/DETR/vit/vit_hg'
                )
            except (OSError, ValueError):
                # Fallback to default pretrained ViT
                self.vit = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            # Freeze backbone
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            self.vit = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        
        # Build custom classifier head
        self.classifier = self._build_classifier(768)
        
    def get_backbone_output_dim(self) -> int:
        return 768
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Get features from ViT backbone
        x = self.vit.vit(x).last_hidden_state
        # Extract CLS token
        cls_token = x[:, 0]
        # Pass through classifier
        return self.classifier(cls_token)
