# Sereact - 3D Object Detection with RGB-PointCloud Fusion

A PyTorch implementation of 3D object detection using a modified 3DETR (3D Detection Transformer) architecture with RGB-PointCloud fusion capabilities. This project combines point cloud data with RGB images for improved 3D bounding box detection.

## 🚀 Features

- **3DETR-based Architecture**: Built on the proven 3D Detection Transformer framework
- **RGB-PointCloud Fusion**: Combines RGB image features with point cloud data for enhanced detection
- **Multi-component Loss Function**: Includes GIoU, box corners, size, regularization, and angle losses
- **Angle Prediction**: 3D bounding box orientation estimation with classification and regression losses
- **Comprehensive Evaluation**: IoU metrics at multiple thresholds (0.25, 0.5) and mean IoU
- **Data Augmentation**: Point cloud augmentations including random cuboid cropping and flipping
- **Distributed Training**: Support for multi-GPU training with PyTorch distributed
- **Type Safety**: Comprehensive type annotations throughout the codebase
- **Extensive Testing**: Unit and integration tests with 90%+ coverage
- **Wandb Integration**: Experiment tracking and visualization

## 📋 Requirements & Environment Setup

### System Requirements

- **Python**: 3.7 - 3.10 (Recommended: 3.8)
- **CUDA**: 11.0+ (for GPU acceleration)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space

### 🐍 Python Environment Setup

#### Option 1: Conda Environment (Recommended)

```bash
# Create new conda environment with Python 3.8
conda create -n sereact python=3.8 -y

# Activate the environment
conda activate sereact

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Navigate to project directory
cd /path/to/sereact

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option 2: Virtual Environment (venv)

```bash
# Create virtual environment
python3.8 -m venv sereact_env

# Activate environment (Linux/Mac)
source sereact_env/bin/activate

# Activate environment (Windows)
# sereact_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 📦 Core Dependencies

The project uses a minimal set of essential packages (see `requirements.txt`):

```txt
# Core Deep Learning
torch>=1.8.0,<2.5.0
torchvision>=0.9.0,<0.20.0

# Scientific Computing
numpy>=1.21.0,<2.0.0
scipy>=1.7.0,<2.0.0

# Computer Vision & 3D Processing
Pillow>=8.0.0
opencv-python>=4.5.0
open3d>=0.15.0

# Configuration & Utilities
yacs>=0.1.8
PyYAML>=5.4.0
timm>=0.4.12,<1.0.0

# Experiment Tracking
wandb>=0.12.0

# Data Processing
imageio>=2.9.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
tqdm>=4.60.0
```

### 🔧 CUDA Extensions Setup

The project includes custom CUDA extensions for PointNet++ operations. These will be compiled automatically on first run:

```bash
# Ensure CUDA toolkit is installed and accessible
nvcc --version

# The extensions will compile automatically when first imported
# Look for compilation messages during first training run
```

### ✅ Verify Installation

```bash
# Test PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test core imports
python -c "
import torch
import torchvision
import numpy as np
import open3d as o3d
import wandb
print('✅ All core dependencies imported successfully!')
"

# Test model import (this will compile CUDA extensions)
python -c "
from models.detr3d.model_3ddetr import build_3ddetr_model
print('✅ Model imports successful!')
"
```

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd sereact

# Create and activate conda environment
conda create -n sereact python=3.8 -y
conda activate sereact

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset in the following structure:
```
dataset/
├── object_1/
│   ├── bbox3d.npy          # 3D bounding box annotations
│   ├── mask.npy            # Segmentation mask
│   ├── pc.npy              # Point cloud data
│   └── rgb.png             # RGB image
├── object_2/
└── ...
```

### 3. Configure Training

Edit `config/base_train.yaml`:
```yaml
data:
  data_path: "/path/to/your/dataset"
  batch_size: 1

model:
  pretrained_weights_path: "/path/to/pretrained/weights"
```

### 4. Start Training

```bash
# Single GPU training
python main.py --cfg config/base_train.yaml --data-path /path/to/dataset

# Multi-GPU distributed training
bash train.sh
```

### 5. Monitor Training

```bash
# View logs
tail -f logs/training.log

# Monitor with Wandb (if enabled)
# Check your Wandb dashboard for real-time metrics
```

## 🏗️ Architecture

### Model Components

1. **Pre-Encoder**
   - Point cloud preprocessing and downsampling
   - Farthest-Point-Sampling to 2048 points
   - Projects points to 256-dimensional feature space

2. **RGB Fusion Module**
   - ResNet18-based RGB feature extraction
   - 3D-to-2D point projection using camera intrinsics
   - Feature fusion between RGB and point cloud features

3. **Transformer Encoder**
   - Multi-head attention mechanism
   - Processes fused features for object detection

4. **Transformer Decoder**
   - Query-based object detection
   - Generates 3D bounding box predictions

5. **Prediction Heads**
   - **Center Head**: 3D bounding box center prediction
   - **Size Head**: 3D bounding box dimensions
   - **Angle Classification Head**: Orientation bin classification (12 bins)
   - **Angle Regression Head**: Fine-grained angle residual prediction
   - **Optional Segmentation Head**: For mask supervision

## 📊 Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── object_1/
│   ├── bbox3d.npy          # 3D bounding box coordinates (K, 8, 3)
│   ├── mask.npy            # Segmentation masks
│   ├── pc.npy              # Point cloud data (N, 3)
│   └── rgb.jpg             # RGB image
├── object_2/
├── object_3/
...
```

## 🚀 Quick Start

### Training

1. **Configure training parameters** in `config/base_train.yaml`:
```yaml
input_folder_path: "/path/to/your/dataset"
batch_size: 2
max_epochs: 100
```

2. **Run training**:
```bash
# Single GPU
python main.py --cfg config/base_train.yaml --data-path /path/to/dataset

# Multi-GPU (distributed)
bash train.sh
```

### Evaluation

```bash
python main.py --cfg config/base_train.yaml --data-path /path/to/dataset --eval
```

## 📈 Loss Function

The model uses a comprehensive multi-component loss function:

- **GIoU Loss**: Generalized Intersection over Union for box positioning
- **Box Corners Loss**: L1 distance for corner accuracy
- **Size Loss**: L1 distance for size prediction
- **Size Regularization**: Penalty for oversized predictions
- **Angle Classification Loss**: Cross-entropy loss for orientation bin prediction
- **Angle Regression Loss**: Smooth L1 loss for fine-grained angle residuals

Total loss: `λ₁ × L_giou + λ₂ × L_corners + λ₃ × L_size + λ₄ × L_reg + λ₅ × L_angle_cls + λ₆ × L_angle_reg`

## 📊 Evaluation Metrics

- **IoU@0.25**: Loose threshold for rough localization
- **IoU@0.50**: Strict threshold for precise detection
- **Mean IoU**: Overall prediction quality measure

## 🔧 Configuration

Key configuration options in `config/base_train.yaml`:

```yaml
model:
  encoder:
    dim: 256
    nheads: 4
    num_layers: 3
  decoder:
    dim: 256
    nhead: 4
    num_layers: 3
  num_queries: 256
  num_angular_bins: 12
  pretrained_weights_path: "/path/to/pretrained/weights"

loss:
  weights:
    giou: 5.0
    box_corners: 1.0
    size: 1.0
    size_reg: 0.1
    angle_cls: 0.1
    angle_reg: 0.1
```

## 📁 Project Structure

```
sereact/
├── config/                 # Configuration files
├── dataloader/            # Dataset loading and augmentation
├── losses/                # Loss function implementations
├── models/                # Model architectures
│   └── detr3d/           # 3DETR model components
├── tests/                 # Comprehensive test suite
├── utils/                 # Utility functions
├── main.py               # Training/evaluation script
├── train.sh              # Distributed training script
├── run_tests.py          # Test runner script
├── pytest.ini           # Test configuration
└── requirements.txt      # Dependencies
```

## 🎯 Pre-trained Models

The project supports loading pre-trained weights from the original 3DETR model trained on ScanNet dataset. Configure the path in your YAML file:

```yaml
model:
  pretrained_weights_path: "/path/to/scannet_ep1080.pth"
```

## 🧪 Testing

The project includes a comprehensive test suite with 90%+ code coverage:

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --type unit          # Unit tests only
python run_tests.py --type integration   # Integration tests only
python run_tests.py --type coverage     # With coverage report

# Run specific test file
python run_tests.py --file test_losses.py
```

## 📝 Example Results

| Epoch | Val/IoU@0.25 | Val/IoU@0.50 | Val/Mean IoU | Angle Accuracy |
|-------|--------------|--------------|--------------|----------------|
|   10  |    0.348     |    0.064     |    0.213     |     0.156      |
|   30  |    0.392     |    0.061     |    0.227     |     0.184      |

## 🔧 Troubleshooting

### Common Environment Issues

#### CUDA Extension Compilation Errors
```bash
# Error: "Microsoft Visual C++ 14.0 is required" (Windows)
# Solution: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Error: "nvcc not found"
# Solution: Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### Memory Issues
```bash
# Error: "CUDA out of memory"
# Solution: Reduce batch size in config
data:
  batch_size: 1  # Reduce from default

# Error: "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"
# Solution: Clear GPU cache and restart
python -c "import torch; torch.cuda.empty_cache()"
```

#### Import Errors
```bash
# Error: "ModuleNotFoundError: No module named 'timm'"
# Solution: Install missing packages
pip install timm

# Error: "ImportError: cannot import name '_ext_src'"
# Solution: CUDA extensions need compilation
# This happens automatically on first run - be patient!
```

#### Performance Issues
```bash
# Slow training on CPU
# Solution: Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Environment Verification Script

Create `verify_env.py`:
```python
#!/usr/bin/env python3
"""Verify Sereact environment setup."""

def verify_environment():
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")

        import torchvision
        print(f"✅ Torchvision {torchvision.__version__}")

        import numpy as np
        print(f"✅ NumPy {np.__version__}")

        import open3d as o3d
        print(f"✅ Open3D {o3d.__version__}")

        from models.detr3d.model_3ddetr import build_3ddetr_model
        print("✅ Model imports successful")

        print("\n🎉 Environment verification complete!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing dependencies")

if __name__ == "__main__":
    verify_environment()
```

Run verification:
```bash
python verify_env.py
```

## 🔧 Development

### Code Quality
- **Type Safety**: Comprehensive type annotations with mypy compatibility
- **Testing**: 90%+ test coverage with unit and integration tests
- **Code Style**: Follows PEP 8 with Ruff linting
- **Documentation**: Comprehensive docstrings and README

### Key Features Added
- ✅ **Angle Loss Integration**: Complete 3D orientation prediction
- ✅ **Type Annotations**: Full type safety throughout codebase
- ✅ **Comprehensive Testing**: Unit and integration test suite
- ✅ **DRY Principles**: Eliminated code duplication
- ✅ **Clean Code**: Removed commented code and debug statements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. **Run tests**: `python run_tests.py`
5. **Check types**: `mypy .` (if mypy is installed)
6. Add tests for new functionality
7. Submit a pull request

## 📄 License

This project is based on the original 3DETR implementation and follows similar licensing terms.

## 📊 Recent Updates

### v2.0 - Enhanced 3D Detection
- **Angle Loss Integration**: Added comprehensive 3D orientation prediction
- **Type Safety**: Complete type annotation coverage
- **Testing Suite**: 90%+ test coverage with unit and integration tests
- **Code Quality**: Eliminated code duplication and improved maintainability
- **Performance**: Optimized distributed training with `find_unused_parameters=False`

### Key Improvements
- 🎯 **Better Orientation**: Angle classification + regression for precise 3D box orientation
- 🔒 **Type Safety**: Full mypy compatibility with comprehensive type hints
- 🧪 **Robust Testing**: Extensive test suite covering all major components
- 🧹 **Clean Code**: Removed commented code, fixed imports, improved structure
- 📈 **Performance**: Enhanced training stability and convergence

## 🙏 Acknowledgments

- Original [3DETR](https://github.com/facebookresearch/3detr) implementation by Facebook Research
- Swin Transformer components (Microsoft)
- PyTorch and torchvision teams

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
