# ASCA-YOLO: Adaptive Sparse and Context-Aware YOLO for Forest Wildfire Detection

A lightweight, high-robustness UAV remote sensing model for early forest wildfire detection, deeply optimized based on YOLO26. This implementation targets the pain points of UAV-based wildfire detection, achieving an optimal balance between ultra-lightweight deployment and high-precision detection for real-time edge computing on UAVs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.10+](https://img.shields.io/badge/PyTorch-2.10+-red.svg)](https://pytorch.org/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🔥 Project Overview
This repository is the official PyTorch implementation of the paper **ASCA-YOLO: Adaptive Sparse and Context-Aware YOLO Algorithm for Forest Wildfire Detection**.

### Core Challenges Solved
UAV-based forest wildfire detection faces two critical practical limitations:
1. **High miss rate of tiny targets**: Ultra-small fire spots and thin smoke plumes are easily lost in deep downsampling under limited computing power of UAV edge devices.
2. **Poor anti-interference and adaptability**: Complex forest backgrounds (sunset, bright rock, forest mist) cause false fire alarms, and traditional models fail to adapt to the non-rigid, dynamic spread of wildfires/smoke, leading to localization divergence.

### Key Contributions
We propose three innovative modules and integrate them into YOLO26 to build the ASCA-YOLO model, which achieves:
- **Ultra-lightweight**: Only 1.87M parameters and 4.2G FLOPs
- **High precision**: 91.9% mAP50 and 61.5% mAP50-95 on the forest wildfire dataset
- **Strong robustness**: Effectively suppress false alarms and adapt to non-rigid wildfire features
- **Real-time inference**: Perfectly fit for UAV edge computing devices with limited computing resources

## ✨ Core Innovative Modules

### 1. Forest Wildfire Adaptive Multi-Scale Convolution (FWAMSConv)
- Replaces the standard Conv module of YOLO26 with a **parallel multi-scale depthwise separable convolution** architecture (3×3/5×5/7×7).
- Compresses channels via 1×1 convolution to reduce computation, while capturing both local tiny fire features and global smoke spread context.
- Solves the problem of small target semantic loss in deep downsampling under limited computing power.

### 2. Forest Wildfire Sparse Context Saliency Attention (FWSCSAttention)
- Establishes global statistical modeling of feature maps to distinguish wildfire targets (**abnormal deviation**) from complex backgrounds (**statistical normal**).
- Achieves background noise separation **without introducing extra learnable parameters**.
- Precisely suppresses "false fire" false alarms caused by high-similarity forest background interference.

### 3. Forest Wildfire Adaptive Sparse-Aware IoU Loss (FWASIoU)
- Extends traditional IoU with three dynamic geometric constraints: **center stability constraint**, **enclosure consistency constraint**, **shape adaptive constraint**.
- Adapts to the non-rigid and dynamic spread characteristics of wildfires and smoke.
- Overcomes localization divergence and regression inaccuracy caused by irregular target shape distortion.

## 📊 Performance Results

### Core Metrics vs. Baseline YOLO26
All experiments are conducted on a unified forest wildfire dataset (7,414 images, 2 categories: `fire`/`smoke`).

| Metric | YOLO26 | ASCA-YOLO | Change |
| :--- | :--- | :--- | :--- |
| Params (M) | 2.375 | 1.870 | ↓ 21.3% |
| FLOPs (G) | 5.2 | 4.2 | ↓ 19.2% |
| Precision (P) | 0.849 | 0.892 | ↑ 4.3% |
| Recall (R) | 0.798 | 0.842 | ↑ 4.4% |
| mAP50 | 0.873 | 0.919 | ↑ 4.6% |
| mAP50-95 | 0.544 | 0.615 | ↑ 7.1% |

### Superiority Over Mainstream Models
ASCA-YOLO outperforms state-of-the-art detection models (YOLOv5/v8/v10/v11/v12, SSD, Faster R-CNN, RT-DETR) on **mAP50** and **mAP50-95** while maintaining the **lowest computation and parameter scale**, making it the best choice for UAV edge deployment.

## 🛠️ Environment Setup

### Hardware Requirements
- CPU: Intel Core i5+/AMD Ryzen 5+ (or higher)
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060/4060/5060 recommended, for CUDA acceleration)
- Memory: 16GB+ RAM (32GB recommended)

### Software Dependencies
- OS: Windows 10/11, Linux (Ubuntu 20.04+/CentOS 7+), macOS 12+
- Python: 3.10 or higher
- PyTorch: 2.10 or higher
- CUDA: 12.8 or higher (for GPU acceleration)
- Other basic dependencies: `opencv-python`, `numpy`, `pillow`, `matplotlib`, `tqdm`

### Quick Installation

1. Clone the repository
   ```bash
   git clone [https://github.com/Weaston-create/forest-wildfire-asca-yolo.git](https://github.com/Weaston-create/forest-wildfire-asca-yolo.git)
   cd forest-wildfire-asca-yolo
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   # Conda
   conda create -n asca-yolo python=3.10
   conda activate asca-yolo
   
   # Or venv
   python -m venv asca-yolo-env
   
   # Windows
   asca-yolo-env\Scripts\activate
   
   # Linux/macOS
   source asca-yolo-env/bin/activate
   ```

3. Install dependencies
   ```bash
   # Install PyTorch (with CUDA) first
   pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### 1. Dataset Preparation

#### Dataset Introduction
The model is trained on a combined dataset of:
- Roboflow: `forest-fire-detection-qkpap_dataset`
- Public UAV remote sensing dataset: `M4SFWD`

Total: 7,414 labeled images (640×640), split into **training (70%) / validation (20%) / test (10%)** sets, with two categories: `fire` and `smoke`.

#### Dataset Format
- Adopt **YOLO annotation format** (`.txt` files, one file per image)
- Dataset directory structure:

  ```text
  dataset/
  ├── train/
  │   ├── images/  # Training images (*.jpg/*.png)
  │   └── labels/  # Training annotations (*.txt)
  ├── val/
  │   ├── images/  # Validation images
  │   └── labels/  # Validation annotations
  └── test/
      ├── images/  # Test images
      └── labels/  # Test annotations
  ```

#### Configuration File
Modify the dataset path and category information in `data/wildfire.yaml`:

  ```yaml
  train: path/to/dataset/train/images
  val: path/to/dataset/val/images
  test: path/to/dataset/test/images

  nc: 2  # Number of classes
  names: ['fire', 'smoke']  # Class names
  ```

### 2. Model Training

#### Train from Scratch
```bash
python train.py \
  --data data/wildfire.yaml \
  --epochs 100 \
  --batch-size 4 \
  --img 640 \
  --weights weights/yolov26n.pt \
  --device 0  # 0 for single GPU, cpu for CPU
```

#### Train with Custom Settings
Key training parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (adjust according to GPU VRAM, default: 4)
- `--img`: Input image resolution (default: 640, fixed for this model)
- `--weights`: Pretrained weight path (YOLO26n baseline)
- `--device`: Training device (`0`, `0,1` for multi-GPU, `cpu`)
- `--lr0`: Initial learning rate (default: 0.002)

#### Training Output
All training results are saved in `runs/train/`:
- `weights/`: Best and last trained weights (`asca-yolo-best.pt`, `asca-yolo-last.pt`)
- `metrics.csv`: Training metrics (loss, P, R, mAP) for each epoch
- `plots/`: Visualization of training curves (loss, mAP)

### 3. Model Inference

#### Inference on Single Image
```bash
python detect.py \
  --weights runs/train/exp/weights/asca-yolo-best.pt \
  --source test_images/fire_sample.jpg \
  --img 640 \
  --device 0 \
  --save-txt  # Save detection results as txt (optional) \
  --save-conf  # Save confidence scores (optional)
```

#### Inference on Video/UAV Stream
```bash
# Video file
python detect.py --weights weights/asca-yolo-best.pt --source test_videos/fire_video.mp4 --img 640

# UAV real-time stream (RTSP/HTTP)
python detect.py --weights weights/asca-yolo-best.pt --source rtsp://xxx.xxx.xxx.xxx:554/stream --img 640

# Webcam (for testing)
python detect.py --weights weights/asca-yolo-best.pt --source 0 --img 640
```

#### Inference Output
Detection results are saved in `runs/detect/`:
- `images/`: Visualized detection results (with bounding boxes)
- `labels/`: Detection annotations (YOLO format, if `--save-txt` is enabled)

### 4. Model Validation
Evaluate the model performance on the test set with standard metrics (P, R, mAP50, mAP50-95):

```bash
python val.py \
  --data data/wildfire.yaml \
  --weights runs/train/exp/weights/asca-yolo-best.pt \
  --img 640 \
  --device 0 \
  --iou-thres 0.5  # IoU threshold for mAP50
```
Validation results (metrics and confusion matrix) are saved in `runs/val/`.

## 📁 Project Structure

```text
forest-wildfire-asca-yolo/
├── data/                  # Dataset configuration files
│   └── wildfire.yaml      # Forest wildfire dataset config
├── models/                # Core model code
│   ├── asca_yolo.py       # Full ASCA-YOLO architecture
│   ├── modules/           # Innovative modules implementation
│   │   ├── fwamsconv.py   # FWAMSConv module
│   │   └── fwscs_attention.py # FWSCSAttention module
│   └── loss/              # Loss function implementation
│       └── fwas_iou.py    # FWASIoU loss function
├── runs/                  # Auto-generated output (train/detect/val)
├── test_images/           # Sample test images
├── test_videos/           # Sample test videos
├── weights/               # Pretrained weights (YOLO26n baseline)
├── train.py               # Model training script
├── detect.py              # Inference & prediction script
├── val.py                 # Model validation script
├── requirements.txt       # Dependencies list
├── LICENSE                # MIT License file
└── README.md              # Project documentation
```

## 📚 Citation
If this project or the related paper contributes to your research, please cite our work:

```bibtex
@article{ASCA-YOLO2026,
  title={ASCA-YOLO: Adaptive Sparse and Context-Aware YOLO Algorithm for Forest Wildfire Detection},
  author={Hao, Yua and Wang, Kangning},
  year={2026},
}
```

## 📄 License
This project is released under the **MIT License** - see the [LICENSE](LICENSE) file for details.  
**Usage Restriction**: For academic research only, commercial use is prohibited without explicit permission from the authors.

## 🙏 Acknowledgements
1. **Ultralytics Team**: For the open-source YOLO series models, especially the YOLO26 baseline.
2. **Roboflow & M4SFWD**: For providing the public forest wildfire datasets used in this research.
3. **PyTorch Community**: For the open-source deep learning framework and related tools.

## 📧 Contact
If you have any questions, issues or suggestions about the project, please contact us via:
- Email: weaston116@163.com
- GitHub Issues: [https://github.com/Weaston-create/forest-wildfire-asca-yolo/issues](https://github.com/Weaston-create/forest-wildfire-asca-yolo/issues)
