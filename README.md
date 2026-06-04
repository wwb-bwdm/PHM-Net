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
PHM-Net/
├── dataset/
│   ├── lux/                 # 光照扰动数据集
│   ├── Pixels/              # 不同分辨率数据集
│   ├── Temperature/         # 温度变化数据集
│   └── Velocity/            # 不同运动速度数据集
├── test/                    
│   ├── mask_out.py          # 掩码生成
│   ├── Stripe_Displacement.py # 掩码计算位移
│   └── test.py              # 完整测试
├── .gitignore
├── LICENSE                  # 开源协议
├── README.md                # 项目说明文档
└── requirements.txt         # 项目环境依赖清单
```
## 常见问题

## 相关链接
