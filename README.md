# 多模型目标检测demo


## 环境要求
- Python版本：3.7 ~ 3.10
- CUDA支持（可选）：使用GPU加速时需要

## 目录结构
V1.0/
|
├─pt/                    # 模型文件目录
|  ├─A.onnx             # A型焊缝检测模型
|  ├─C.onnx             # C型焊缝检测模型
|  └─D.onnx             # D型焊缝检测模型
|
├─test/                 # 存放待检测的图片
|
├─multi_model_detector.py # 主程序文件
├─README.md              # 说明文档
└─requirements.txt       # 环境依赖

## 使用方法
1. 安装环境依赖
2. 将待检测的图片放入test文件夹（确保检测的图片在项目目录下的test文件夹中）
3. 运行multi_model_detector.py
4. 检测结果CSV文件会保存在test文件夹下的detected_best文件夹下

