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
   - Download: `TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz`

2. **Extract and Install**:
```bash
tar -xvf TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz 
cd TensorRT-8.2.1.8
pip install python/tensorrt-8.2.1.8-cp37-none-linux_x86_64.whl
```

3. **Fix Library Paths** (fixes common import errors):
```bash
# libnvinfer.so.8 not found 
find . -name "libnvinfer.so.8"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home-local/akath.nobkp/sereact/TensorRT-8.2.1.8/lib

#libcudnn.so.8 not found
find /usr -name "libcudnn.so.8" 2>/dev/null
# /usr/lib/python3/dist-packages/torch/lib enter path to your library found from above command
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/tensorflow:/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH
#
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

