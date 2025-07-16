# Sereact 3D Object Detection

## 🧠 3D Object Detection with 3DETR and RGB-PointCloud Fusion

A PyTorch-based implementation of 3D object detection using **3DETR (3D Detection Transformer)** with RGB–PointCloud fusion. This project combines 3D point cloud data and 2D RGB images to improve 3D bounding box prediction and orientation estimation accuracy.

📄 For detailed explanation and architecture overview, see [`Submission Report`](SUBMISSION_REPORT.md).


## 🚀 Features

- **3DETR Architecture**: Transformer-based 3D object detection
- **RGB-PointCloud Fusion**: Multi-modal input processing
- **3D Bounding Box Detection**: Accurate position and orientation estimation
- **Distributed Training**: Multi-GPU support with PyTorch distributed
- **CUDA Extensions**: Optimized PointNet++ operations
- **Comprehensive Loss Functions**: GIoU, angle, size, and corner losses

### 📈 Experimental Results 

TODO : Analysis of visual results

<p align="center">
  <img src="box_distributions.png" alt="Box Distributions" width="51%"/>
  <img src="point_cloud_visualisation.png" alt="3D Point Cloud" width="47%"/>
</p>

<p align="center">
  <img src="validation_miou.png" alt="Validation mIoU" width="45%" />
  <img src="Train_loss.png" alt="Train Loss" width="45%" />
</p>


| Setting              |  IoU@0.25  |  Mean IoU  |
|----------------------|------------|------------|
| Without Augmentation |   0.7950   | **0.4432** |
| With Augmentation    |   0.8250   | **0.4653** |

## 📋 Requirements

### System Requirements
- **Python**: 3.7.16 (exact version required)
- **PyTorch**: 1.8.0 with CUDA support
- **CUDA**: 10.2 or 11.1 (compatible with PyTorch 1.8.0)
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **RAM**: 12GB+ system memory
- **Storage**: 8GB+ free space

### CUDA Toolkit
Ensure CUDA toolkit is installed and accessible:
```bash
nvcc --version  # Should show CUDA 10.2 or 11.1
```

## 🛠️ Installation

### 1. Create Python Environment

```bash
# Create conda environment with Python 3.7.16
conda create -n sereact python=3.7.16 -y
conda activate sereact

# Install PyTorch 1.8.0 with CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Build CUDA Extensions

The project includes custom CUDA extensions for PointNet++ operations that need to be compiled:

```bash
# Navigate to CUDA extensions directory
cd models/detr3d/_ext_src

# Build CUDA extensions
python3 setup.py build_ext --inplace

# Verify compilation
python -c "import _ext_src; print('CUDA extensions compiled successfully!')"

# Return to project root
cd ../../..
```

### 4. Verify Installation

```bash
# Test all imports
python -c "
import torch
import torchvision
import numpy as np
from models.detr3d.model_3ddetr import build_3ddetr_model
from models.detr3d._ext_src import _ext_src
print('✅ All components imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### 5. TensorRT Installation (Optional - For Model Optimization)

TensorRT converts trained models to optimized inference engines. **Manual download required from NVIDIA.**

#### Quick Installation Steps

1. **Download from NVIDIA** (free account required):
   - Visit: https://developer.nvidia.com/tensorrt
   - Download: `TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.1.cudnn8.2.tar.gz`

2. **Extract and Install**:
```bash
mkdir -p ~/tensorrt && cd ~/tensorrt
tar -xzf TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.1.cudnn8.2.tar.gz
cd TensorRT-8.2.1.8/python
pip install tensorrt-8.2.1.8-cp37-none-linux_x86_64.whl
```

3. **Fix Library Paths** (fixes common import errors):
```bash
# Add to ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/tensorrt/TensorRT-8.2.1.8/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

4. **Test Installation**:
```bash
python -c "import tensorrt as trt; print(f'TensorRT {trt.__version__} ready!')"
```

#### Common Errors & Quick Fixes

**Error**: `ImportError: libcudnn.so.8: cannot open shared object file`
**Fix**: `export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH`

**Error**: `ImportError: libnvinfer.so.8: cannot open shared object file`
**Fix**: `export LD_LIBRARY_PATH=~/tensorrt/TensorRT-8.2.1.8/lib:$LD_LIBRARY_PATH`


#### Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.7.16 | Exact version required |
| PyTorch | 1.8.0 | With CUDA 11.1 support |
| TensorRT | 8.2.1.8 | Compatible with CUDA 11.1 |
| CUDA | 11.1+ | Runtime and toolkit |
| cuDNN | 8.x | Required by TensorRT |

## 📁 Dataset Structure

Organize your dataset as follows:
```
dataset/
├── object_1/
│   ├── bbox3d.npy          # 3D bounding box annotations
│   ├── mask.npy            # Segmentation mask
│   ├── pc.npy              # Point cloud data (N, 3 or N, 6)
│   └── rgb.png             # RGB image
├── object_2/
│   ├── bbox3d.npy
│   ├── mask.npy
│   ├── pc.npy
│   └── rgb.png
└── ...
```

## ⚙️ Configuration

Edit `config/base_train.yaml` to configure training:

```yaml
data:
  dataset: 'Sereact_dataset'
  data_path: '/path/to/your/dataset'
  batch_size: 1

model:
  encoder:
    type: 'vanilla'
    dim: 256
    nheads: 4
    num_layers: 3
  decoder:
    num_queries: 256
    num_layers: 6

train:
  max_epoch: 100
  base_lr: 0.0001
```

## 🚀 Training

### Single GPU Training
```bash
python main.py \
  --cfg config/base_train.yaml \
  --data-path /path/to/dataset \
  --batch-size 1 \
  --local_rank 0
```

### Multi-GPU Distributed Training
```bash
python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port 12346 \
  main.py \
  --cfg config/base_train.yaml \
  --data-path /path/to/dataset \
  --batch-size 1
```

### Training Script
Create `train.sh`:
```bash
#!/bin/bash
export WANDB_MODE="disabled"
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
--nproc_per_node 2 \
--master_port 12346 \
main.py \
--cfg config/base_train.yaml \
--data-path /path/to/dataset \
--batch-size 1
```

## 🧪 Evaluation

```bash
python main.py \
  --cfg config/base_train.yaml \
  --data-path /path/to/dataset \
  --eval \
  --pretrained /path/to/checkpoint.pth
```

## 🏗️ Architecture

### Model Components
1. **Pre-Encoder**: Point cloud preprocessing and downsampling
2. **Encoder**: Transformer encoder with multi-head attention
3. **Decoder**: Transformer decoder for object queries
4. **Prediction Heads**: Box regression, classification, and angle prediction

### [Loss Functions](losses/LOSS_FUNCTIONS_GUIDE.md)
- **GIoU Loss**: Generalized IoU for 3D bounding boxes
- **Box Corner Loss**: L1 loss for corner accuracy
- **Size Loss**: L1 loss for size prediction
- **Angle Loss**: Classification + regression for orientation


### Training Time
- **Single GPU**: ~2 minute 24 seconds per epoch
- **Multi-GPU**: ~49 seconds per epoch

## 🔗 Dependencies

See `requirements.txt` for complete list of dependencies optimized for Python 3.7.16 and PyTorch 1.8.0.

## 📝 Notes

- This implementation is optimized for Python 3.7.16 and PyTorch 1.8.0
- CUDA extensions require compilation before first use
- Uses legacy `torch.distributed.launch` for distributed training
- Compatible with CUDA 10.2 and 11.1

## 🤝 Contributing

1. Ensure Python 3.7.16 and PyTorch 1.8.0 compatibility
2. Compile CUDA extensions after any changes
3. Test with both single and multi-GPU setups
4. Verify all imports work correctly
