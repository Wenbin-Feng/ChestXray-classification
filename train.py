from model import DieaseClassifier
import torch
import torch.optim as optim
from datahelper import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = DieaseClassifier(num_classes=3, img_size=224).to(device)  

chest_dataset = Chest_Dataset(root_dir='/home/ubuntu/DETR/5.06组会/ChexRay/train/img', transform=transform)
Chestloader = torch.utils.data.DataLoader(chest_dataset, batch_size=32, shuffle=True)
num_epochs = 10
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
print('Training...')

for epoch in range(num_epochs):
    model.train()
    for images, labels in Chestloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
#保存
torch.save(model.state_dict(), 'Chest.pth')

