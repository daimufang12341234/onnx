# 多模型目标检测demo
    目前可处理的焊缝为：
        A型焊缝
        C型焊缝
        D型焊缝
    （注意：甲板片焊缝检测可能效果不佳，数据集还未完成标注）
    数据集数量：
        A型焊缝：1000张
        C型焊缝：848张
        D型焊缝：1044张

## 环境要求
- Python版本：3.7 ~ 3.10
- CUDA支持（可选）：使用GPU加速时需要

## 目录结构
V1.0/
├── pt/
│ ├── A.onnx
│ ├── C.onnx
│ └── D.onnx
├── test/ # 存放待检测的图片
├── multi_model_detector.py
├── README.md
└── requirements.txt（环境依赖）

## 使用方法
1. 安装环境依赖
2. 将待检测的图片放入test文件夹（确保检测的图片在项目目录下的test文件夹中）
3. 运行multi_model_detector.py
4. 检测结果CSV文件会保存在test文件夹下的detected_best文件夹下

