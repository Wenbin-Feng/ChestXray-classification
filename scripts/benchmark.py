#!/usr/bin/env python3
"""Benchmark and compare ViT vs Vision Mamba models.

Usage:
    # Run full benchmark (train both models and compare)
    python scripts/benchmark.py
    
    # Quick benchmark (fewer epochs)
    python scripts/benchmark.py --epochs 3 --quick
    
    # Compare existing checkpoints
    python scripts/benchmark.py --compare-only
    
    # Export results to CSV and plots
    python scripts/benchmark.py --export
"""
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/ChestXray-classification')

import os
import argparse
import time
import json
from typing import Dict, List
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import pandas as pd

from config import config, AppConfig
from models import create_model
from utils import create_dataloaders, Trainer


class Benchmark:
    """Benchmark runner for model comparison."""
    
    def __init__(self, epochs: int = 5, quick: bool = False):
        self.epochs = epochs
        self.quick = quick
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_training(self, model_type: str) -> Dict:
        """Train a model and return metrics."""
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}\n")
        
        # Override config for this run
        os.environ["MODEL_TYPE"] = model_type
        os.environ["TRAIN_NUM_EPOCHS"] = str(self.epochs)
        if self.quick:
            os.environ["TRAIN_BATCH_SIZE"] = "64"  # Larger batch for speed
        
        # Reload config
        from config import config as fresh_config
        
        # Create model
        model = create_model(model_type)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count FLOPs (approximate)
        flops = self._estimate_flops(model, fresh_config.model.img_size)
        
        # Create dataloaders
        train_loader, test_loader, val_loader = create_dataloaders()
        
        # Train
        trainer = Trainer(model)
        start_time = time.time()
        history = trainer.train(train_loader, val_loader)
        train_time = time.time() - start_time
        
        # Test evaluation
        test_metrics = trainer.evaluate(test_loader)
        
        # Inference speed test
        inference_time = self._benchmark_inference(model, fresh_config.model.img_size)
        
        return {
            "model_type": model_type,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": flops,
            "train_time": train_time,
            "inference_time": inference_time,
            "final_train_loss": history["train_loss"][-1],
            "best_val_metrics": max(history["val_metrics"], key=lambda x: x["f1"]),
            "test_metrics": test_metrics,
            "history": history
        }
    
    def _estimate_flops(self, model, img_size: int) -> int:
        """Estimate FLOPs for the model."""
        try:
            from thop import profile
            dummy_input = torch.randn(1, 3, img_size, img_size)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            return int(flops)
        except ImportError:
            # Rough estimation
            total_params = sum(p.numel() for p in model.parameters())
            # Assume 2 FLOPs per param per forward pass, and forward+backward for training
            return total_params * 2
    
    def _benchmark_inference(self, model, img_size: int, num_runs: int = 100) -> float:
        """Benchmark inference speed."""
        device = config.device.device
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
                if device == "cuda":
                    torch.cuda.synchronize()
        
        total_time = time.time() - start
        return (total_time / num_runs) * 1000  # ms per image
    
    def run(self) -> Dict:
        """Run full benchmark."""
        print("=" * 60)
        print("CHEST X-RAY CLASSIFICATION BENCHMARK")
        print("=" * 60)
        print(f"Epochs: {self.epochs}")
        print(f"Quick mode: {self.quick}")
        print(f"Device: {config.device.device}")
        print("=" * 60)
        
        # Train both models
        for model_type in ["vit", "mamba"]:
            try:
                self.results[model_type] = self.run_training(model_type)
            except Exception as e:
                print(f"Error training {model_type}: {e}")
                self.results[model_type] = {"error": str(e)}
        
        return self.results
    
    def compare_checkpoints(self) -> Dict:
        """Compare existing checkpoints without retraining."""
        print("=" * 60)
        print("COMPARING EXISTING CHECKPOINTS")
        print("=" * 60)
        
        for model_type in ["vit", "mamba"]:
            checkpoint_path = f"checkpoints/best_{model_type}.pth"
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                continue
            
            print(f"\nLoading {model_type}...")
            
            os.environ["MODEL_TYPE"] = model_type
            from config import config as fresh_config
            
            model = create_model(model_type)
            trainer = Trainer(model)
            trainer.load_checkpoint(f"best_{model_type}.pth")
            
            _, test_loader, _ = create_dataloaders()
            test_metrics = trainer.evaluate(test_loader)
            
            inference_time = self._benchmark_inference(model, fresh_config.model.img_size)
            
            self.results[model_type] = {
                "model_type": model_type,
                "test_metrics": test_metrics,
                "inference_time": inference_time
            }
        
        return self.results
    
    def generate_report(self, export: bool = False):
        """Generate comparison report."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        # Create comparison table
        data = []
        for model_type, result in self.results.items():
            if "error" in result:
                continue
            
            row = {
                "Model": model_type.upper(),
                "Params (M)": f"{result['total_params'] / 1e6:.2f}",
                "FLOPs (G)": f"{result.get('flops', 0) / 1e9:.2f}",
                "Train Time (min)": f"{result['train_time'] / 60:.1f}",
                "Inference (ms)": f"{result['inference_time']:.2f}",
                "Accuracy": f"{result['test_metrics']['accuracy']:.4f}",
                "Precision": f"{result['test_metrics']['precision']:.4f}",
                "Recall": f"{result['test_metrics']['recall']:.4f}",
                "F1 Score": f"{result['test_metrics']['f1']:.4f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        
        # Export
        if export:
            os.makedirs("benchmark_results", exist_ok=True)
            
            # CSV
            csv_path = f"benchmark_results/benchmark_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nSaved CSV: {csv_path}")
            
            # JSON
            json_path = f"benchmark_results/benchmark_{self.timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"Saved JSON: {json_path}")
            
            # Plots
            self._generate_plots()
    
    def _generate_plots(self):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training curves
        for model_type, result in self.results.items():
            if "error" in result or "history" not in result:
                continue
            
            history = result["history"]
            epochs = range(1, len(history["train_loss"]) + 1)
            
            # Loss curve
            axes[0, 0].plot(epochs, history["train_loss"], label=f"{model_type.upper()}", marker='o')
            
            # F1 curve
            f1_scores = [m["f1"] for m in history["val_metrics"]]
            axes[0, 1].plot(epochs, f1_scores, label=f"{model_type.upper()}", marker='o')
        
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Training Loss")
        axes[0, 0].set_title("Training Loss Curves")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].set_title("Validation F1 Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bar charts for final metrics
        models = list(self.results.keys())
        accuracies = [self.results[m]["test_metrics"]["accuracy"] for m in models if "error" not in self.results[m]]
        inference_times = [self.results[m]["inference_time"] for m in models if "error" not in self.results[m]]
        
        axes[1, 0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Test Accuracy Comparison")
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].bar(models, inference_times, color=['#1f77b4', '#ff7f0e'])
        axes[1, 1].set_ylabel("Time (ms)")
        axes[1, 1].set_title("Inference Time per Image")
        
        plt.tight_layout()
        
        plot_path = f"benchmark_results/benchmark_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plots: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ViT vs Vision Mamba")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer epochs, larger batch)")
    parser.add_argument("--compare-only", action="store_true", help="Compare existing checkpoints only")
    parser.add_argument("--export", action="store_true", help="Export results to CSV and plots")
    args = parser.parse_args()
    
    benchmark = Benchmark(epochs=args.epochs, quick=args.quick)
    
    if args.compare_only:
        benchmark.compare_checkpoints()
    else:
        benchmark.run()
    
    benchmark.generate_report(export=args.export)


if __name__ == "__main__":
    main()
