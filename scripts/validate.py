#!/usr/bin/env python3
"""Validation script for chest X-ray classification.

Usage:
    # Validate default model
    python scripts/validate.py
    
    # Validate specific checkpoint
    python scripts/validate.py --checkpoint checkpoints/best_vit.pth
    
    # Validate Mamba model
    MODEL_TYPE=mamba python scripts/validate.py
"""
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/ChestXray-classification')

from config import config
from models import create_model
from utils import create_dataloaders, Trainer


def main():
    """Main validation function."""
    print("=" * 60)
    print("Chest X-ray Classification - Validation")
    print("=" * 60)
    print(f"Model: {config.model.type}")
    print(f"Device: {config.device.device}")
    print("=" * 60)
    print()
    
    # Create model
    model = create_model()
    
    # Create trainer (for loading checkpoint)
    trainer = Trainer(model)
    
    # Try to load best checkpoint
    checkpoint_name = f"best_{config.model.type}.pth"
    try:
        trainer.load_checkpoint(checkpoint_name)
    except FileNotFoundError:
        print(f"Warning: Checkpoint {checkpoint_name} not found")
        print("Using randomly initialized model")
        print()
    
    # Create dataloaders
    _, test_loader, _ = create_dataloaders()
    
    # Evaluate
    print(f"Evaluating on {len(test_loader.dataset)} test samples...")
    print()
    
    metrics = trainer.evaluate(test_loader)
    
    print("=" * 60)
    print("Test Set Results")
    print("=" * 60)
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
