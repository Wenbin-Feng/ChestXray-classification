#!/usr/bin/env python3
"""Training script for chest X-ray classification.

Usage:
    # Train with default settings (ViT)
    python scripts/train.py
    
    # Train with Mamba
    MODEL_TYPE=mamba python scripts/train.py
    
    # Train with custom config
    MODEL_TYPE=vit TRAIN_NUM_EPOCHS=20 python scripts/train.py
"""
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/ChestXray-classification')

from config import config
from models import create_model
from utils import create_dataloaders, Trainer


def main():
    """Main training function."""
    # Print configuration
    print("=" * 60)
    print("Chest X-ray Classification - Training")
    print("=" * 60)
    print(f"Model: {config.model.type}")
    print(f"Device: {config.device.device}")
    print(f"Data dir: {config.data.train_dir}")
    print("=" * 60)
    print()
    
    # Create model
    model = create_model()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create dataloaders
    train_loader, test_loader, val_loader = create_dataloaders()
    
    # Print dataset info
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Class distribution: {train_loader.dataset.get_class_distribution()}")
    print()
    
    # Create trainer
    trainer = Trainer(model)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    print("=" * 60)
    print("Final Test Set Evaluation")
    print("=" * 60)
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    
    print()
    print("Training complete!")


if __name__ == "__main__":
    main()
