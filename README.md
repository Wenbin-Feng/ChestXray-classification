# 大型OCT与胸部X光图像标注数据集——肺炎分类项目

本项目基于 ZhangLabData: Chest X-Ray 数据集，旨在实现对胸部X光图像的肺炎自动分类。项目包含数据预处理、模型训练、推理与结果可视化等完整流程，适用于医学影像智能诊断相关研究与应用。

## 数据集简介
- The authors of the ZhanLabData: Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images addressed challenges related to reliability and interpretability in the implementation of clinical-decision support algorithms for medical imaging. The Chest XRay part has a total of 5,856 patients contributed to the dataset, with 4,273 images characterized as depicting PNEUMONIA_BACTERIA and PNEUMONIA_VIRUS (rest - NORMAL images). They established a diagnostic tool based on a deep-learning framework specifically designed for the screening of patients with common treatable blinding retinal diseases.
- **数据集来源**：[ZhangLabData: Chest X-Ray](https://datasetninja.com/zhang-lab-data-chest-xray)
- **数据类型**：胸部X光图像，带有标注
- **任务类型**：分类（如正常/肺炎）

## 项目功能

- 数据预处理与增强
- 基于ViT等主流模型的肺炎分类模型训练
- 支持模型评估与推理
- 结果可视化

## 项目结构

## 项目结构

```text
.
├── datahelper.py
├── model.py
├── test.py
├── train.py
├── val.py
├── inference.py
├── Vit.py
└── ChexRay/
    ├── train/
    │   └── img/
    │       ├── BACTERIA-7422-0001.jpeg
    │       ├── BACTERIA-7422-0002.jpeg
    │       ├── VIRUS-1234-0001.jpeg
    │       ├── NORMAL-5678-0001.jpeg
    │       └── ...（更多图片）
    └── test/
        └── img/
            ├── BACTERIA-8000-0001.jpeg
            ├── VIRUS-2000-0001.jpeg
            ├── NORMAL-9000-0001.jpeg
            └── ...（更多图片）
```

## 快速开始

1. 克隆本仓库并安装依赖
2. 下载并解压数据集到指定目录
3. 运行训练脚本进行模型训练
4. 使用测试脚本进行推理和评估

## 依赖环境

请参考 `requirements.txt` 安装所需依赖：

```bash
pip install -r requirements.txt
```
开始训练前请修改对应数据集路径

```bash
python train.py
```
测试训练效果：
```bash
python val.py
```
## 数据集引用
如在您的研究中使用该数据集，请引用如下：
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
