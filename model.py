from Vit import VisionTransformer
import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights
from transformers import ViTForImageClassification


class DieaseClassifier(nn.Module):
    def __init__(self, num_classes=2, img_size=224, pretrained=True):
        super().__init__()
        # 使用timm库的VisionTransformer并加载预训练权重
        self.vit = ViTForImageClassification.from_pretrained('/home/ubuntu/DETR/vit/vit_hg')
        for param in self.vit.parameters():
            param.requires_grad = False
        # 增强的分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.vit.vit(x).last_hidden_state
        cls_token = x[:, 0]
        return self.classifier(cls_token)
    

# class CatDogClassifier(nn.Module):
#     def __init__(self, num_classes=2, img_size=224, pretrained=True):
#         super().__init__()
#         # 使用timm库的VisionTransformer并加载预训练权重
#         self.vit = VisionTransformer(
#             img_size=img_size,
#             patch_size=16,
#             in_chans=3,
#             embed_dim=768,
#             depth=12,
#             num_heads=12
#         )
        
#         # 增强的分类头
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(768),
#             nn.Linear(768, 512),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
        
#     def forward(self, x):
#         x = self.vit(x)  # 输出包含CLS token的特征
#         cls_token = x[:, 0]  # 使用CLS token
#         return self.classifier(cls_token)
    
