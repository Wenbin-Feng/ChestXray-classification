import torch
from PIL import Image
from model import DieaseClassifier
from datahelper import test_transform

# 类别映射，根据你的训练集类别顺序调整
idx2label = {
    0:"BACTERIA",
    1:"NORMAL",
    2:"VIRUS"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = DieaseClassifier(num_classes=3, img_size=224).to(device)
model.load_state_dict(torch.load('Chest.pth', map_location=device))
model.eval()

# 预测函数
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
    return idx2label[pred]

if __name__ == "__main__":
    img_path = "/home/ubuntu/DETR/5.06组会/ChexRay/test/img/VIRUS-7638941-0001.jpeg"  # 替换为你的图片路径
    pred_label = predict_image(img_path)
    print(f"预测类别: {pred_label}")