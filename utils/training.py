"""Training and evaluation utilities."""
import os
from typing import Dict, List
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import config


class Trainer:
    """Training manager for chest X-ray classification."""
    
    def __init__(self, model, device=None):
        """
        Args:
            model: PyTorch model
            device: Device to train on
        """
        self.model = model
        self.device = device or config.device.device
        self.model.to(self.device)
        
        # Create save directory
        os.makedirs(config.train.save_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_metrics": []
        }
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if config.debug:
                print(f"  Batch {num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation/test set.
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Predictions
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate metrics
        metrics = {
            "loss": total_loss / num_batches,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(
                all_labels, all_preds, average='macro', zero_division=0
            ),
            "recall": recall_score(
                all_labels, all_preds, average='macro', zero_division=0
            ),
            "f1": f1_score(
                all_labels, all_preds, average='macro', zero_division=0
            )
        }
        
        return metrics
    
    def train(self, train_loader, val_loader) -> Dict:
        """Full training loop.
        
        Returns:
            Training history
        """
        print(f"Training on device: {self.device}")
        print(f"Model type: {config.model.type}")
        print(f"Epochs: {config.train.num_epochs}")
        print(f"Batch size: {config.train.batch_size}")
        print(f"Learning rate: {config.train.learning_rate}")
        print("-" * 50)
        
        best_f1 = 0.0
        
        for epoch in range(config.train.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history["val_metrics"].append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"Epoch [{epoch+1}/{config.train.num_epochs}] - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  F1 Score: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.save_checkpoint(f"best_{config.model.type}.pth")
                print(f"  ✓ Saved best model (F1: {best_f1:.4f})")
            
            print()
        
        # Save final model
        self.save_checkpoint(f"final_{config.model.type}.pth")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(config.train.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": config.model.dict(),
            "history": self.history
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(config.train.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_metrics": []})
        print(f"Loaded checkpoint from {path}")


class Predictor:
    """Inference manager for chest X-ray classification."""
    
    # Class labels
    LABELS = {0: "BACTERIA", 1: "NORMAL", 2: "VIRUS"}
    
    def __init__(self, model, device=None):
        """
        Args:
            model: PyTorch model
            device: Device to run inference on
        """
        self.model = model
        self.device = device or config.device.device
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """Predict single image.
        
        Args:
            image_tensor: Preprocessed image tensor (C, H, W) or (1, C, H, W)
            
        Returns:
            Dictionary with prediction and probabilities
        """
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        output = self.model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        
        return {
            "label": self.LABELS[prediction],
            "class_idx": prediction,
            "confidence": probabilities[0, prediction].item(),
            "probabilities": {
                label: prob.item()
                for label, prob in zip(self.LABELS.values(), probabilities[0])
            }
        }
    
    @torch.no_grad()
    def predict_batch(self, dataloader) -> List[Dict]:
        """Predict batch of images.
        
        Args:
            dataloader: DataLoader with images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for images, _ in dataloader:
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(len(predictions)):
                pred = predictions[i].item()
                results.append({
                    "label": self.LABELS[pred],
                    "class_idx": pred,
                    "confidence": probabilities[i, pred].item(),
                    "probabilities": {
                        label: prob.item()
                        for label, prob in zip(
                            self.LABELS.values(), probabilities[i]
                        )
                    }
                })
        
        return results
