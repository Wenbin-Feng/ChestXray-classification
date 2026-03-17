"""Configuration management using Pydantic Settings."""
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Model architecture configuration."""
    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    type: Literal["vit", "mamba"] = Field(
        default="vit",
        description="Model architecture: vit or mamba"
    )
    num_classes: int = Field(default=3, description="Number of output classes")
    img_size: int = Field(default=224, description="Input image size")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    
    # ViT specific
    vit_patch_size: int = Field(default=16, description="ViT patch size")
    vit_embed_dim: int = Field(default=768, description="ViT embedding dimension")
    vit_depth: int = Field(default=12, description="ViT transformer depth")
    vit_num_heads: int = Field(default=12, description="ViT number of attention heads")
    
    # Mamba specific  
    mamba_d_model: int = Field(default=768, description="Mamba hidden dimension")
    mamba_n_layer: int = Field(default=12, description="Mamba number of layers")
    mamba_d_state: int = Field(default=16, description="Mamba state dimension")
    mamba_d_conv: int = Field(default=4, description="Mamba convolution kernel size")
    mamba_expand: int = Field(default=2, description="Mamba expansion factor")


class TrainingConfig(BaseSettings):
    """Training configuration."""
    model_config = SettingsConfigDict(
        env_prefix="TRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    batch_size: int = Field(default=32, description="Training batch size")
    num_epochs: int = Field(default=10, description="Number of training epochs")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    weight_decay: float = Field(default=1e-4, description="Weight decay")
    num_workers: int = Field(default=4, description="DataLoader workers")
    shuffle: bool = Field(default=True, description="Shuffle training data")
    save_dir: str = Field(default="./checkpoints", description="Model checkpoint directory")
    

class DataConfig(BaseSettings):
    """Data paths configuration."""
    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    train_dir: str = Field(
        default="/home/ubuntu/DETR/5.06组会/ChexRay/train/img",
        description="Training data directory"
    )
    test_dir: str = Field(
        default="/home/ubuntu/DETR/5.06组会/ChexRay/test/img",
        description="Test data directory"
    )
    val_dir: Optional[str] = Field(
        default=None,
        description="Validation data directory (optional)"
    )
    

class DeviceConfig(BaseSettings):
    """Device configuration."""
    model_config = SettingsConfigDict(
        env_prefix="DEVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    type: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device type: cuda, cpu, or auto"
    )
    
    @property
    def device(self) -> str:
        """Get the actual device string."""
        import torch
        if self.type == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.type


class AppConfig(BaseSettings):
    """Main application configuration aggregating all sub-configs."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Nested configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    
    # App settings
    seed: int = Field(default=42, description="Random seed")
    debug: bool = Field(default=False, description="Debug mode")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set random seeds
        import random
        import numpy as np
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


# Global config instance
config = AppConfig()
