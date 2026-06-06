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
## 运行
```
当前版本暂未开发命令行参数启动模式，请使用 PyCharm / VSCode 打开项目文件夹，通过 IDE 直接运行对应脚本；后续迭代版本将补充命令行启动配置。
```
## 项目目录
PHM-Net/
├── dataset/
│   ├── Lux/                 # 光照扰动数据集：high强光 / low弱光
│   │   ├── high/
│   │   └── low/
│   ├── Pixels/              # 不同分辨率数据集：400×275 / 800×550
│   │   ├── 400×275/
│   │   └── 800×550/
│   ├── Temperature/         # 不同环境温度数据集：17℃ / 23℃ / 26℃
│   │   ├── 17/
│   │   ├── 23/
│   │   └── 26/
│   └── Velocity/            # 不同速度数据集：30 / 50 / 80
│       ├── 30/
│       ├── 50/
│       └── 80/
├── test/                    # 基线测试集
├── .gitignore               # git忽略配置
├── LICENSE                  # 项目开源协议
├── README.md                # 项目说明、环境、训练测试文档
├── requirements.txt         # Python环境依赖清单
├── mask_out.py          # 掩码生成
├── Stripe_Displacement.py # 掩码计算
└── test.py              # 模型完整推理测试
## 常见问题
### Q1：位移计算环节报错、匹配失败
答：图像命名统一格式为``组号_序号.png``，例：1_1.png、1_2.png；同组规则：xx_1为基准参考图，xx_2为待测位移图，程序自动两两配对计算位移差值。后续我们会上传图片匹配更鲁棒的代码。
### Q2：预训练权重加载失败
答：权重存储路径禁止包含中文、空格、全角符号；
Windows 系统读取权重建议使用原始字符串r"D:\xxx\xxx.pth"规避反斜杠转义报错；
核对代码内配置路径与权重实际存放路径、文件名完全一致；权重文件损坏 / 缺失需重新下载。
### Q3：CUDA显存溢出报错`out of memory`
答：可以在配置中将运行设备改为device="cpu"使用 CPU 推理，或调低输入图片尺寸input_size。注意。调整输入尺寸可能会影响到模型性能。
### Q5：OpenCV 读取图片为空
答：图片存储路径、文件名可能使用了中文与特殊符号，请全程使用纯英文+数字命名。

