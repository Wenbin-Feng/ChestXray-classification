from model import DieaseClassifier
import torch
from datahelper import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chest_dataset = Chest_Dataset(root_dir='/home/ubuntu/DETR/5.06组会/ChexRay/test/img', transform=test_transform)
Chestloader = torch.utils.data.DataLoader(chest_dataset, batch_size=32, shuffle=False)

model = DieaseClassifier(num_classes=3, img_size=224).to(device)  
model.load_state_dict(torch.load('Chest.pth', map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in Chestloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算四个分类指标
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f'Accuracy: {acc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')