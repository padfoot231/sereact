# Sereact 3D Object Detection

A PyTorch implementation of 3D object detection using 3DETR (3D Detection Transformer) with RGB-PointCloud fusion. This project combines point cloud data with RGB images for enhanced 3D bounding box detection and orientation estimation.

## 🚀 Features

- **3DETR Architecture**: Transformer-based 3D object detection
- **RGB-PointCloud Fusion**: Multi-modal input processing
- **3D Bounding Box Detection**: Accurate position and orientation estimation
- **Distributed Training**: Multi-GPU support with PyTorch distributed
- **CUDA Extensions**: Optimized PointNet++ operations
- **Comprehensive Loss Functions**: GIoU, angle, size, and corner losses

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

### 5. TensorRT Installation (For Model Conversion)

TensorRT is required for converting trained models to optimized inference engines. Follow these steps for Python 3.7 and PyTorch 1.8.0 compatibility:

#### Download and Install TensorRT

```bash
# Download TensorRT 8.2.1.8 from NVIDIA Developer website
# Navigate to TensorRT Python directory
cd TensorRT-8.2.1.8/python

# Install the wheel for Python 3.7
pip install tensorrt-8.2.1.8-cp37-none-linux_x86_64.whl
```

#### Common Installation Issues & Fixes

**Issue 1: `ImportError: libcudnn.so.8: cannot open shared object file`**

This occurs when TensorRT can't find cuDNN libraries. Fix by adding PyTorch's CUDA libraries:

```bash
# Temporary fix (current session)
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Permanent fix (add to ~/.bashrc)
echo 'export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Issue 2: `ImportError: libnvinfer.so.8: cannot open shared object file`**

Add TensorRT libraries to your library path:

```bash
# Replace /path/to/ with your actual TensorRT installation path
export LD_LIBRARY_PATH=/path/to/TensorRT-8.2.1.8/lib:$LD_LIBRARY_PATH

# Make permanent
echo 'export LD_LIBRARY_PATH=/path/to/TensorRT-8.2.1.8/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

#### Verify TensorRT Installation

```bash
# Test TensorRT import and functionality
python -c "
import tensorrt as trt
print(f'TensorRT version: {trt.__version__}')

# Test basic functionality
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print('✅ TensorRT installation successful!')
"
```

#### Environment Setup Script

Create a setup script for easy environment configuration:

```bash
# Create setup_tensorrt.sh
cat > setup_tensorrt.sh << 'EOF'
#!/bin/bash
# TensorRT Environment Setup

# Set paths (update these to match your installation)
export CUDA_HOME=/usr/local/cuda
export TENSORRT_HOME=/path/to/TensorRT-8.2.1.8

# Update library paths
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Update Python path
export PYTHONPATH=$TENSORRT_HOME/python:$PYTHONPATH

echo "✅ TensorRT environment configured!"
EOF

# Make executable and run
chmod +x setup_tensorrt.sh
source setup_tensorrt.sh
```

#### Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.7.16 | Exact version required |
| PyTorch | 1.8.0 | With CUDA 11.1 support |
| TensorRT | 8.2.1.8 | Compatible with CUDA 11.1 |
| CUDA | 11.1+ | Runtime and toolkit |
| cuDNN | 8.x | Required by TensorRT |

#### Additional Troubleshooting

**Issue 3: `CUDA driver version is insufficient for CUDA runtime version`**
```bash
# Check CUDA driver version
nvidia-smi

# Check CUDA runtime version
nvcc --version

# Update NVIDIA drivers if needed
sudo apt update && sudo apt install nvidia-driver-470
```

**Issue 4: `ModuleNotFoundError: No module named 'tensorrt'`**
```bash
# Ensure correct Python environment is activated
conda activate sereact

# Reinstall TensorRT wheel
pip uninstall tensorrt
pip install tensorrt-8.2.1.8-cp37-none-linux_x86_64.whl
```

**Issue 5: Memory issues during conversion**
```bash
# Reduce batch size in conversion script
# Monitor GPU memory usage
nvidia-smi -l 1

# Clear GPU cache if needed
python -c "import torch; torch.cuda.empty_cache()"
```

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

### Loss Functions
- **GIoU Loss**: Generalized IoU for 3D bounding boxes
- **Box Corner Loss**: L1 loss for corner accuracy
- **Size Loss**: L1 loss for size prediction
- **Angle Loss**: Classification + regression for orientation

## 🔧 Troubleshooting

### CUDA Extensions Issues
```bash
# If compilation fails, check CUDA installation
nvcc --version
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Clean and rebuild
cd models/detr3d/_ext_src
rm -rf build/
python3 setup.py build_ext --inplace
```

### Memory Issues
```bash
# Reduce batch size
data:
  batch_size: 1

# Enable memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Import Errors
```bash
# Ensure CUDA extensions are compiled
cd models/detr3d/_ext_src
python3 setup.py build_ext --inplace

# Test import
python -c "import _ext_src"
```

## 📊 Performance

### Expected Results
- **IoU@0.25**: ~0.35-0.40
- **IoU@0.50**: ~0.06-0.08
- **Mean IoU**: ~0.20-0.25

### Training Time
- **Single GPU**: ~2-3 hours per epoch
- **Multi-GPU**: ~1-1.5 hours per epoch

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

## 📄 License

[Add your license information here]
