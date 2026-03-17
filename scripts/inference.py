#!/usr/bin/env python3
"""Inference script for chest X-ray classification.

Usage:
    # Predict single image
    python scripts/inference.py --image /path/to/image.jpg
    
    # Predict with specific checkpoint
    python scripts/inference.py --image /path/to/image.jpg --checkpoint checkpoints/best_vit.pth
    
    # Predict with Mamba
    MODEL_TYPE=mamba python scripts/inference.py --image /path/to/image.jpg
"""
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/ChestXray-classification')

import argparse
from PIL import Image

from config import config
from models import create_model
from utils import test_transform, Predictor


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Chest X-ray Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chest X-ray Classification - Inference")
    print("=" * 60)
    print(f"Model: {config.model.type}")
    print(f"Device: {config.device.device}")
    print("=" * 60)
    print()
    
    # Load image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    # Preprocess
    image_tensor = test_transform(image)
    
    # Create model
    model = create_model()
    
    # Load checkpoint if provided
    if args.checkpoint:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location=config.device.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Create predictor
    predictor = Predictor(model)
    
    # Predict
    print("Predicting...")
    result = predictor.predict(image_tensor)
    
    print()
    print("=" * 60)
    print("Prediction Result")
    print("=" * 60)
    print(f"Predicted Class: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print()
    print("Class Probabilities:")
    for label, prob in result['probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"  {label:12s}: {prob:.4f} {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
