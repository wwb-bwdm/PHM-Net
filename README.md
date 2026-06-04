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
### Q1：位移计算环节报错、匹配失败
原始原图命名格式统一为：组号_序号，示例：1_1.png、1_2.png、3_1.png、3_2.png；同组仅匹配序号 1（基准图）、序号 2（位移图）进行差值计算；
模型推理生成掩码时代码会自动在文件名后添加_pred，最终掩码名称为1_1_pred.png、1_2_pred.png，无需手动修改文件名添加_pred；
全部待测算图片放在同一个文件夹，程序会自动按组批量配对。
### Q2：预训练权重加载失败
权重路径不能包含中文、空格、特殊字符；
Windows路径建议使用原始字符串r"路径"，避免反斜杠转义报错；
核对权重文件名与代码配置路径保持一致，缺失权重请重新下载。
### Q3：CUDA显存溢出报错`out of memory`
在配置中将运行设备改为device="cpu"使用 CPU 推理，或调低输入图片尺寸input_size。
### Q5：OpenCV 读取图片为空
图片存储路径、文件名可能使用中文与特殊符号。

