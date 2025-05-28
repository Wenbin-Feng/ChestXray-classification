from Vit import VisionTransformer
import torch
from model import CatDogClassifier
from PIL import Image
from datahelper import transform,test_transform
d = {0:"cat", 1:"dog"}  
print('cuda' if torch.cuda.is_available() else 'cpu')
model = CatDogClassifier(num_classes=2, img_size=224)
model.load_state_dict(torch.load('cat_dog_model.pth'))
model.eval()
img = Image.open("/home/ubuntu/DETR/5.06组会/cat_dog/cat/cat.0.jpg").convert('RGB')
img = test_transform(img).unsqueeze(0)  # 添加batch维度

img = img.to('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    print(output)
    print(f'Predicted class: {d[predicted.item()]}')