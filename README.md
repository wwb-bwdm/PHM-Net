# PHM-Net

## Compatible Runtime Version
- Python：3.8.20
- PyTorch：1.11.0+cu113
- Operating System：Windows
## Project Introduction
This repository contains the open-source code for the algorithm proposed in this paper, facilitating the rapid reproduction of experimental results.

## Dependencies
The main dependencies are listed below. For the complete list, please refer to `requirements.txt`
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- opencv-python
## Installation
1. Clone the repository
```bash
git clone https://github.com/wwb-bwdm/PHM-Net.git
cd PHM-Net
```
2.Install dependencies
```bash
pip install -r requirements.txt
```
## Usage
The current version does not support command-line arguments. Please open the project folder in PyCharm / VSCode and run the corresponding scripts directly via the IDE. Command-line support will be added in future updates.
## Project Structure
```
PHM-Net/
├── PHM-Net_Baseline_Test/                # Baseline test set
├── PHM-Net_Generalization_Test/
│   ├── Intensities/                      # Illumination disturbance dataset
│   │   ├── High_Intensity/
│   │   └── Low_Intensity/
│   ├── Resolutions/                      # Different resolution dataset
│   │   ├── 400_275/
│   │   └── 800_550/
│   ├── Temperatures/                    # Different ambient temperature dataset (℃)
│   │   ├── 17_Degrees_Celsius/
│   │   ├── 23_Degrees_Celsius/
│   │   └── 26_Degrees_Celsius/
│   └── Speeds/                          # Different motion speed dataset
│       ├── 30_Micrometers_Per_Second/
│       ├── 50_Micrometers_Per_Second/
│       └── 80_Micrometers_Per_Second/
├── PHM-Net_Weights/                      # Model checkpoint storage
│   └── PHM_best.pth                      # Best trained weight file
├── .gitignore                            # Git ignore configuration
├── LICENSE                               # Open-source license
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependency list
├── MaskOut.py                            # Mask generation
├── Stripe_Displacement.py                # Stripe displacement calculation
└── Test.py                               # Full model inference script
```
## Frequently Asked Questions
### Q1: Errors or matching failures during displacement calculation
A: Image files must follow the naming format: `groupID_serial.png`, e.g., `1_1.png`, `1_2.png`.
- `xx_1`: Reference image (fixed)
- `xx_2`: Target image (to be measured)
The program automatically pairs images in the same group. A more robust matching version will be released later.
### Q2: Failed to load pre-trained weights
A:
- Weight paths must not contain Chinese characters, spaces, or full-width symbols.
- On Windows, use raw string r"D:\xxx\xxx.pth" to avoid escape errors.
- Check that the path in the code matches the actual weight file location.
- Re-download the weights if the file is corrupted or missing.
### Q3: CUDA Out-of-Memory (OOM) error
A:
- Switch to device="cpu" for CPU inference.
- Reduce the input_size of images.
Note: Resizing images may affect model performance.
### Q3: OpenCV cannot read images (empty result)
A:
- Image paths and filenames must use only English letters and numbers.
- Avoid Chinese characters and special symbols.
