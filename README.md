# 胸部X光图像分类系统

基于深度学习（ViT / Vision Mamba）的肺炎自动分类系统，支持环境变量配置和规范化项目结构。

## 新特性

- ✅ **双架构支持**: Vision Transformer (ViT) 和 Vision Mamba
- ✅ **Pydantic Settings**: 环境变量注入配置
- ✅ **规范化结构**: 模块化设计，易于扩展
- ✅ **统一训练流程**: 自动验证、指标追踪、模型保存

## 项目结构

```
.
├── config/                 # 配置管理
│   └── __init__.py        # Pydantic Settings
├── models/                 # 模型定义
│   ├── __init__.py        # 模型工厂
│   ├── vit_classifier.py  # ViT分类器
│   └── mamba_classifier.py # Vision Mamba分类器
├── utils/                  # 工具函数
│   ├── data.py            # 数据集和数据加载
│   └── training.py        # 训练和推理工具
├── scripts/                # 可执行脚本
│   ├── train.py           # 训练脚本
│   ├── validate.py        # 验证脚本
│   └── inference.py       # 推理脚本
├── .env.example           # 环境变量示例
├── datahelper.py          # 旧版数据工具（兼容）
├── model.py               # 旧版模型（兼容）
├── train.py               # 旧版训练脚本（兼容）
├── val.py                 # 旧版验证脚本（兼容）
├── inference.py           # 旧版推理脚本（兼容）
├── Vit.py                 # 原始ViT实现
└── README.md              # 本文档
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，修改数据路径等配置
```

### 2. 使用新架构训练

```bash
# 使用 ViT 训练（默认）
python scripts/train.py

# 使用 Vision Mamba 训练
MODEL_TYPE=mamba python scripts/train.py

# 自定义训练参数
MODEL_TYPE=vit TRAIN_NUM_EPOCHS=20 TRAIN_BATCH_SIZE=16 python scripts/train.py
```

### 3. 验证模型

```bash
# 验证默认模型
python scripts/validate.py

# 验证 Mamba 模型
MODEL_TYPE=mamba python scripts/validate.py
```

### 4. 推理单张图片

```bash
python scripts/inference.py --image /path/to/image.jpg
```

## 配置说明

所有配置通过环境变量管理，支持 `.env` 文件:

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MODEL_TYPE` | 模型类型: `vit` 或 `mamba` | `vit` |
| `MODEL_NUM_CLASSES` | 分类类别数 | `3` |
| `MODEL_IMG_SIZE` | 输入图像尺寸 | `224` |
| `TRAIN_NUM_EPOCHS` | 训练轮数 | `10` |
| `TRAIN_BATCH_SIZE` | 批次大小 | `32` |
| `TRAIN_LEARNING_RATE` | 学习率 | `3e-4` |
| `DATA_TRAIN_DIR` | 训练数据目录 | - |
| `DATA_TEST_DIR` | 测试数据目录 | - |
| `DEVICE_TYPE` | 设备: `cuda`, `cpu`, `auto` | `auto` |

## 模型对比

| 特性 | ViT | Vision Mamba |
|------|-----|--------------|
| 架构基础 | Transformer | State Space Model |
| 计算复杂度 | O(n²) | O(n) |
| 长序列效率 | 较低 | 更高 |
| 实现依赖 | transformers | 纯 PyTorch |
| 预训练权重 | 可用 | 需从头训练 |

## 依赖安装

```bash
# 核心依赖
pip install torch torchvision
pip install transformers
pip install pydantic-settings
pip install scikit-learn
pip install pillow

# 可选: 如果使用原始 mamba_ssm
# pip install mamba-ssm
```

## 旧版兼容

保留了原始文件以确保向后兼容:
- `model.py` - 原始模型定义
- `train.py` - 原始训练脚本
- `val.py` - 原始验证脚本
- `datahelper.py` - 原始数据工具

## 技术细节

### Vision Mamba 实现

基于选择性状态空间模型 (Selective State Space Model) 的视觉骨干网络:

- **Patch Embedding**: 将图像分割为 16x16 patches
- **Mamba Blocks**: 使用选择性 SSM 替代自注意力
- **线性复杂度**: 序列长度呈线性增长，适合高分辨率图像

### Pydantic Settings

类型安全的配置管理，自动从环境变量加载:

```python
from config import config

# 访问配置
print(config.model.type)       # "vit" 或 "mamba"
print(config.device.device)    # "cuda" 或 "cpu"
print(config.train.batch_size) # 32
```

## 数据集

使用 [ZhangLabData: Chest X-Ray](https://datasetninja.com/zhang-lab-data-chest-xray) 数据集:

- 总计 5,856 张图像
- 类别: NORMAL, BACTERIA, VIRUS
- 训练集/测试集已划分

## 引用

```bibtex
@misc{ visualization-tools-for-zhang-lab-data-chest-xray-dataset,
  title = { Visualization Tools for ZhangLabData: Chest X-Ray Dataset },
  type = { Computer Vision Tools },
  author = { Dataset Ninja },
  howpublished = { \url{ https://datasetninja.com/zhang-lab-data-chest-xray } },
  url = { https://datasetninja.com/zhang-lab-data-chest-xray },
  journal = { Dataset Ninja },
  publisher = { Dataset Ninja },
  year = { 2025 },
  month = { may },
  note = { visited on 2025-05-28 },
}
```
