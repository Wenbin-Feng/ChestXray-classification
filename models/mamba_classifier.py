"""Vision Mamba (Vim) based disease classifier.

A simplified Vision Mamba implementation using selective state space models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import config
from . import BaseClassifier


class SelectiveScanFn(torch.autograd.Function):
    """Simplified selective scan operation for state space models."""
    
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None):
        """Forward pass of selective scan.
        
        Args:
            u: Input (B, L, N)
            delta: Time step (B, L, N)
            A: State matrix (N,)
            B: Input-dependent transition (B, L, N)
            C: Output matrix (B, L, N)
            D: Skip connection (N,)
            delta_bias: Bias for delta
        """
        # Simplified forward pass
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias)
        
        batch, seq_len, dim = u.shape
        
        # Discretization
        if delta_bias is not None:
            delta = delta + delta_bias
        delta = F.softplus(delta)
        
        # Simplified state update (forward Euler)
        x = torch.zeros(batch, dim, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seq_len):
            # State update: x = exp(delta * A) * x + delta * B * u
            dA = torch.exp(delta[:, i] * A)  # (B, N)
            dB = delta[:, i] * B[:, i]  # (B, N)
            
            x = dA * x + dB * u[:, i]  # (B, N)
            y = C[:, i] * x  # (B, N)
            
            if D is not None:
                y = y + D * u[:, i]
            
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, N)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # Simplified backward - just return None for now
        # In production, implement proper backward pass
        return grad_output, None, None, None, None, None, None


class MambaBlock(nn.Module):
    """Single Mamba block with selective SSM."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Delta (time step) projection
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # B and C projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        batch, seq_len, dim = x.shape
        
        # Residual connection
        residual = x
        x = self.norm(x)
        
        # Input projection and split
        x_and_gate = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_ssm, gate = x_and_gate.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Apply activation to gate
        gate = F.silu(gate)
        
        # Convolution
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)
        
        # SSM parameters
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Project to get B and C
        x_proj_out = self.x_proj(x_conv)  # (B, L, 2 * d_state)
        B, C = x_proj_out.chunk(2, dim=-1)  # Each: (B, L, d_state)
        
        # Compute delta
        delta = self.dt_proj(x_conv)  # (B, L, d_inner)
        
        # Selective scan
        # Simplified: use matrix multiplication approximation
        y = self.selective_scan(x_conv, delta, A, B, C)
        
        # Gating
        y = y * gate
        
        # Output projection
        y = self.out_proj(y)
        
        # Residual
        return y + residual
    
    def selective_scan(self, u, delta, A, B, C):
        """Simplified selective scan without CUDA kernel."""
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        # Discretize
        delta = F.softplus(delta)
        
        # Simplified parallel scan approximation
        # In practice, you'd use the CUDA kernel from mamba_ssm
        
        # Use cumulative sum approximation
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # State evolution (simplified)
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seq_len):
            x = dA[:, i] * x + dB[:, i] * u[:, i].unsqueeze(-1)
            y = (C[:, i].unsqueeze(1) * x).sum(dim=-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        # Add skip connection
        y = y + self.D * u
        
        return y


class VisionMambaBackbone(nn.Module):
    """Vision Mamba backbone similar to ViT architecture."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        d_model: int = 768,
        n_layer: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        
        # Dropout
        self.pos_drop = nn.Dropout(0.1)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, L, D) where L = num_patches + 1 (CLS token)
        """
        batch = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, d_model)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


class MambaClassifier(BaseClassifier):
    """Vision Mamba based classifier for chest X-ray classification."""
    
    def __init__(self, num_classes: int = 3, img_size: int = 224, pretrained: bool = True):
        super().__init__(num_classes, img_size)
        
        # Vision Mamba backbone
        self.backbone = VisionMambaBackbone(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            d_model=config.model.mamba_d_model,
            n_layer=config.model.mamba_n_layer,
            d_state=config.model.mamba_d_state,
            d_conv=config.model.mamba_d_conv,
            expand=config.model.mamba_expand
        )
        
        # Classification head
        self.classifier = self._build_classifier(config.model.mamba_d_model)
        
        if pretrained:
            # Note: Vision Mamba pretrained weights not available in this simplified version
            # Would load from checkpoint in production
            pass
    
    def get_backbone_output_dim(self) -> int:
        return config.model.mamba_d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_classes)
        """
        # Get features from backbone
        x = self.backbone(x)  # (B, L, D)
        # Extract CLS token
        cls_token = x[:, 0]  # (B, D)
        # Pass through classifier
        return self.classifier(cls_token)
