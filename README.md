# 基于YOLOv8的多场景目标检测项目

## 项目简介
本项目是基于YOLOv8的目标检测课程实践项目，针对多场景（拥挤、稀疏、低光照）的行人目标检测需求，实现了数据预处理、低光照增强、模型推理与可视化交互的完整流程。

## 项目结构
```
merged_dataset/
├── configs/ # 配置文件目录
│ ├── dataset.yaml # 通用数据集配置
│ ├── dataset_crowded.yaml # 拥挤场景数据集配置
│ └── dataset_sparse.yaml # 稀疏场景数据集配置
├── code/
│ ├── analysis/ # 数据分析与低光照增强模块
│ │ ├── classify_crowded.py
│ │ ├── filter_low_light.py
│ │ ├── filter_normal_light.py
│ │ ├── model.py # 低光照增强核心模型
│ │ ├── predict.py # 低光照增强推理脚本
│ │ └── vis_crowded_vs_sparse.py
│ └── preprocess/ # 数据预处理脚本
│ ├── merge_datasets.py
│ ├── yuchuli.py
│ ├── yuchuli_crowded.py
│ └── yuchuli_sparse.py
├── images/ # 图像数据目录
├── .gitignore
├── demo_gui_modified.py # 可视化交互界面
├── README.md # 项目说明文档
└── requirements.txt # 依赖清单
```

> 说明：因文件体积限制，训练日志及大型权重文件等未上传至仓库，可按本文档说明自行准备。

## 运行环境
- Python 版本：3.8 ~ 3.11（推荐3.10）
- 操作系统：Windows 10/11 或 Linux
- 依赖管理：通过 `requirements.txt` 一键安装

## 环境安装
1.  克隆项目到本地
```bash
git clone https://github.com/zhujiaqi2450958/merged_dataset.git
cd merged_dataset

# 安装所有依赖
pip install -r requirements.txt

### 使用说明
1.  **数据预处理**
    执行通用数据预处理脚本，对图像和标注进行清洗、格式转换：
    ```bash
    python code/preprocess/yuchuli.py

2.  **低光照图像增强**
    对低光照场景的图像进行亮度和细节增强：
    ```bash
    python code/analysis/predict.py

3.  **启动可视化 GUI 界面**
    运行交互界面，加载模型并对图像进行目标检测：
    ```bash
    python demo_gui_modified.py

## 注意事项
1.  运行前请检查 `configs/` 目录下的 `.yaml` 配置文件，确保里面的数据集路径和你本地实际路径一致；
2.  低光照增强脚本依赖 PyTorch，使用 GPU 环境可以大幅提升处理速度；
3.  所有大型数据文件（如数据集、训练日志）已被 `.gitignore` 屏蔽，不会提交到远程仓库；
4.  若运行报错，优先检查 Python 版本和依赖是否安装完整。

---

## 项目总结
本项目完整实现了基于 YOLOv8 的多场景行人目标检测流程，包含数据预处理、低光照增强、模型推理与可视化交互等模块，可作为课程实践和二次开发的基础。