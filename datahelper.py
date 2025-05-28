from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random
from PIL import Image
from torchvision import transforms
dieases = {
    "BACTERIA":0,
    "NORMAL":1,
    "VIRUS":2
}
def get_files(path):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def get_data(images_name, label):
    to_return = []
    for img_name in images_name:
        if label in img_name:
            to_return.append((img_name, dieases[label]))
    return to_return

class Chest_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = get_files(root_dir)
        self.data = []
        for diease in dieases.keys():
            self.data+= get_data(self.images, diease)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        return image, label
    
transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.ToTensor()
    ])

test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.ToTensor()
    ])


# dataset = Chest_Dataset(root_dir='/home/ubuntu/DETR/5.06组会/ChexRay/train/img', transform=transform)
# #print(dataset[86])  # Example to check if the dataset is working
# cat_dog_loader = DataLoader(dataset, batch_size=32, shuffle=True)
