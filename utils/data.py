"""Data loading and preprocessing utilities."""
import os
import random
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import config


# Disease label mapping
DISEASE_LABELS = {
    "BACTERIA": 0,
    "NORMAL": 1,
    "VIRUS": 2
}


def get_files(path: str) -> List[str]:
    """Recursively get all files in directory."""
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, filename))
    return files


def get_data(images_name: List[str], label: str) -> List[Tuple[str, int]]:
    """Filter images by label and return (path, label) tuples."""
    to_return = []
    for img_name in images_name:
        if label in os.path.basename(img_name).upper():
            to_return.append((img_name, DISEASE_LABELS[label]))
    return to_return


# Training data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.model.img_size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test/validation transform (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((config.model.img_size, config.model.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ChestDataset(Dataset):
    """Chest X-ray dataset for pneumonia classification."""
    
    def __init__(self, root_dir: str, transform=None, split: str = "train"):
        """
        Args:
            root_dir: Root directory containing images
            transform: Optional transform to apply
            split: Dataset split ("train" or "test")
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Get all images
        self.images = get_files(root_dir)
        
        # Build dataset
        self.data = []
        for disease in DISEASE_LABELS.keys():
            self.data += get_data(self.images, disease)
        
        # Shuffle for training
        if split == "train":
            random.shuffle(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get distribution of classes in dataset."""
        dist = {label: 0 for label in DISEASE_LABELS.keys()}
        for _, label in self.data:
            for name, idx in DISEASE_LABELS.items():
                if idx == label:
                    dist[name] += 1
        return dist


def create_dataloaders():
    """Create train and test dataloaders from config."""
    from torch.utils.data import DataLoader
    
    # Training dataset
    train_dataset = ChestDataset(
        root_dir=config.data.train_dir,
        transform=train_transform,
        split="train"
    )
    
    # Test dataset
    test_dataset = ChestDataset(
        root_dir=config.data.test_dir,
        transform=test_transform,
        split="test"
    )
    
    # Validation dataset (if provided, else use test)
    if config.data.val_dir:
        val_dataset = ChestDataset(
            root_dir=config.data.val_dir,
            transform=test_transform,
            split="val"
        )
    else:
        val_dataset = test_dataset
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, val_loader
