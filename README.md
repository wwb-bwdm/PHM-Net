# PHM-Net

## 适配运行版本
- Python：3.8.20
- PyTorch：1.11.0+cu113
- 操作系统：Windows
## 项目介绍
本仓库为本论文对应算法开源代码，便于快速复现论文实验结果。

## 环境依赖
主要依赖如下，完整依赖请查看 `requirements.txt`
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- opencv-python
## 安装教程
1. 克隆项目源码
```bash
git clone https://github.com/wwb-bwdm/PHM-Net.git
cd PHM-Net
```
2.安装环境
```bash
pip install -r requirements.txt
```
## 运行指令
```bash
python test.py
```
## 项目目录
```bash
├── train.py        # 训练脚本
├── test.py         # 测试脚本
├── requirements.txt
├── config/         # 参数配置
├── datasets/       # 数据集
└── weights/        # 模型权重
```
## 常见问题

## 相关链接
