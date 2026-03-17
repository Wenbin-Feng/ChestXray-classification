"""Grad-CAM visualization for model interpretability."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple


class GradCAM:
    """Grad-CAM implementation for ViT and Mamba models."""
    
    def __init__(self, model, target_layer: str = None):
        """
        Args:
            model: The model to visualize
            target_layer: Name of the target layer for CAM (auto-detected if None)
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Hook handlers
        self.forward_hook = None
        self.backward_hook = None
        
        # Auto-detect target layer if not specified
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
    
    def _find_target_layer(self) -> str:
        """Automatically find the best layer for CAM."""
        # For ViT: look for the last transformer block
        # For Mamba: look for the last mamba block
        for name, module in self.model.named_modules():
            if 'blocks' in name or 'backbone' in name:
                if 'norm' in name or isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
                    continue
                last_layer = name
        return last_layer if 'last_layer' in locals() else None
    
    def _get_layer(self, layer_name: str):
        """Get layer by name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _save_gradient(self, grad):
        """Hook to save gradients."""
        self.gradients = grad
    
    def _register_hooks(self, layer):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(self._save_gradient)
        
        handle = layer.register_forward_hook(forward_hook)
        return handle
    
    @torch.enable_grad()
    def generate_cam(self, image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Register hooks
        layer = self._get_layer(self.target_layer)
        if layer is None:
            raise ValueError(f"Target layer {self.target_layer} not found")
        
        handle = self._register_hooks(layer)
        
        # Forward pass
        image = image.requires_grad_(True)
        output = self.model(image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # (B, L, D) or (B, C, H, W)
        activations = self.activations[0]  # Same shape
        
        # Handle different architectures
        if len(gradients.shape) == 3:  # Transformer/Mamba (B, L, D)
            # For ViT/Mamba: activations are (B, num_patches+1, dim)
            # We need to exclude CLS token and reshape
            gradients = gradients[:, 1:, :]  # Remove CLS token
            activations = activations[:, 1:, :]
            
            # Weighted sum of gradients
            weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, D)
            cam = (weights * activations).sum(dim=-1)  # (B, L)
            
            # Reshape to spatial grid
            batch_size, num_patches = cam.shape
            grid_size = int(np.sqrt(num_patches))
            cam = cam.reshape(batch_size, grid_size, grid_size)
            
        elif len(gradients.shape) == 4:  # CNN (B, C, H, W)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1)
        
        # Normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to image size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(image.shape[2], image.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam.squeeze().cpu().numpy()
        
        # Clean up hooks
        handle.remove()
        
        return cam
    
    def visualize(
        self,
        image: torch.Tensor,
        original_image: Image.Image,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        colormap: str = 'jet',
        save_path: Optional[str] = None,
        class_names: Optional[list] = None
    ) -> plt.Figure:
        """
        Create full visualization with original image, heatmap, and overlay.
        
        Args:
            image: Preprocessed image tensor
            original_image: Original PIL image
            target_class: Target class for CAM
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            save_path: Optional path to save figure
            class_names: List of class names
            
        Returns:
            Matplotlib figure
        """
        # Generate CAM
        cam = self.generate_cam(image, target_class)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(cam)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        original_np = np.array(original_image.resize((image.shape[3], image.shape[2])))
        cam_resized = np.uint8(255 * cam)
        cam_color = cmap(cam_resized)
        cam_color = (cam_color[:, :, :3] * 255).astype(np.uint8)
        
        overlay = original_np * (1 - alpha) + cam_color * alpha
        overlay = overlay.astype(np.uint8)
        
        axes[2].imshow(overlay)
        
        # Title with prediction info
        if class_names:
            pred_name = class_names[pred_class]
            title = f'Overlay\nPredicted: {pred_name} ({confidence:.2%})'
        else:
            title = f'Overlay\nClass {pred_class} ({confidence:.2%})'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return fig


def visualize_prediction(
    model,
    image_path: str,
    transform,
    save_path: Optional[str] = None,
    class_names: Optional[list] = None
) -> plt.Figure:
    """
    Convenience function for one-shot visualization.
    
    Usage:
        from utils.visualization import visualize_prediction
        
        fig = visualize_prediction(
            model, 
            '/path/to/image.jpg',
            test_transform,
            save_path='output.png',
            class_names=['BACTERIA', 'NORMAL', 'VIRUS']
        )
    """
    from utils.data import test_transform
    
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0)
    
    # Create GradCAM
    grad_cam = GradCAM(model)
    
    # Generate visualization
    fig = grad_cam.visualize(
        image_tensor,
        original_image,
        save_path=save_path,
        class_names=class_names
    )
    
    return fig
