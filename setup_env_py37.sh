#!/bin/bash
# =============================================================================
# Sereact 3D Object Detection - Python 3.7.16 Environment Setup
# =============================================================================
# This script sets up the environment with Python 3.7.16 and PyTorch 1.8.0
# for maximum compatibility with older systems and CUDA versions.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="sereact_py37"
PYTHON_VERSION="3.7.16"
PYTORCH_VERSION="1.8.0"
TORCHVISION_VERSION="0.9.0"
CUDA_VERSION="10.2"  # Compatible with PyTorch 1.8.0

echo -e "${BLUE}🚀 Sereact Environment Setup (Python 3.7.16 + PyTorch 1.8.0)${NC}"
echo "=================================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        print_status "Using existing environment."
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

# Create new conda environment with Python 3.7.16
print_status "Creating conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
print_status "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    print_error "Failed to activate environment"
    exit 1
fi

print_status "Environment activated: $CONDA_DEFAULT_ENV"

# Install PyTorch 1.8.0 with CUDA 10.2
print_status "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}..."
conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA_VERSION} -c pytorch -y

# Verify PyTorch installation
print_status "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install other dependencies
print_status "Installing other dependencies from requirements_py37.txt..."
pip install -r requirements_py37.txt

# Create legacy training script for PyTorch 1.8.0
print_status "Creating legacy training script for PyTorch 1.8.0..."
cat > train_py37.sh << 'EOF'
#!/bin/bash
# Legacy training script for Python 3.7.16 + PyTorch 1.8.0
export WANDB_MODE="disabled"

# cuDNN Error Fixes
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# Memory and Performance Settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "🚀 Starting Sereact Training (Python 3.7.16 + PyTorch 1.8.0)"
echo "Using legacy torch.distributed.launch..."

# Use legacy torch.distributed.launch for PyTorch 1.8.0
python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12346 \
main.py \
--cfg config/base_train.yaml \
--output /home-local2/akath.extra.nobkp/sereact \
--data-path /home-local2/akath.extra.nobkp/dl_challenge \
--batch-size 1 \
--local_rank 0
# --pretrained /home-local2/akath.extra.nobkp/scannet_ep1080.pth

echo "✅ Training completed!"
EOF

chmod +x train_py37.sh

# Create verification script for Python 3.7.16
print_status "Creating verification script..."
cat > verify_env_py37.py << 'EOF'
#!/usr/bin/env python3
"""Verify Sereact environment for Python 3.7.16 + PyTorch 1.8.0"""

import sys
import platform

def verify_python_version():
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 7:
        print("✅ Python 3.7 confirmed")
        return True
    else:
        print(f"❌ Expected Python 3.7, got {version.major}.{version.minor}")
        return False

def verify_pytorch():
    try:
        import torch
        print(f"🔥 PyTorch version: {torch.__version__}")
        
        if torch.__version__.startswith('1.8'):
            print("✅ PyTorch 1.8.x confirmed")
        else:
            print(f"⚠️  Expected PyTorch 1.8.x, got {torch.__version__}")
        
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def verify_packages():
    packages = [
        'numpy', 'scipy', 'PIL', 'cv2', 'open3d', 
        'yacs', 'yaml', 'timm', 'sklearn', 'wandb',
        'imageio', 'matplotlib', 'tqdm', 'termcolor', 'shapely'
    ]
    
    print("\n📦 Package verification:")
    all_ok = True
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {pkg}: {version}")
        except ImportError:
            print(f"❌ {pkg}: Not installed")
            all_ok = False
    
    return all_ok

def main():
    print("🔍 Sereact Environment Verification (Python 3.7.16 + PyTorch 1.8.0)")
    print("=" * 70)
    
    python_ok = verify_python_version()
    pytorch_ok = verify_pytorch()
    packages_ok = verify_packages()
    
    print("\n" + "=" * 70)
    if python_ok and pytorch_ok and packages_ok:
        print("🎉 Environment verification successful!")
        print("Ready for training with Python 3.7.16 + PyTorch 1.8.0")
    else:
        print("❌ Environment verification failed")
        print("Please check missing dependencies")

if __name__ == "__main__":
    main()
EOF

# Verify installation
print_status "Verifying installation..."
python verify_env_py37.py

# Create activation script
print_status "Creating activation script..."
cat > activate_sereact_py37.sh << EOF
#!/bin/bash
# Sereact Python 3.7.16 Environment Activation Script
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
echo "🎯 Sereact Python 3.7.16 environment activated!"
echo "Python: \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Use: bash train_py37.sh to start training"
EOF

chmod +x activate_sereact_py37.sh

# Final instructions
echo
echo "=================================================================="
print_status "Python 3.7.16 + PyTorch 1.8.0 environment setup complete!"
echo
echo "📋 Next steps:"
echo "1. Activate environment: conda activate ${ENV_NAME}"
echo "   Or use: source activate_sereact_py37.sh"
echo
echo "2. Verify setup: python verify_env_py37.py"
echo
echo "3. Configure dataset path in config/base_train.yaml"
echo
echo "4. Start training: bash train_py37.sh"
echo
print_warning "Note: This setup uses PyTorch 1.8.0 with legacy distributed training"
print_warning "CUDA extensions will compile automatically on first training run"
echo
print_status "Happy training with Python 3.7.16! 🚀"
