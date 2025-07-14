#!/bin/bash
# =============================================================================
# Sereact 3D Object Detection - Environment Setup Script
# =============================================================================
# This script automatically sets up the Python environment for Sereact
# 3D object detection system with all required dependencies.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="sereact"
PYTHON_VERSION="3.8"
CUDA_VERSION="11.8"

echo -e "${BLUE}🚀 Sereact Environment Setup${NC}"
echo "=================================="

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

# Create new conda environment
print_status "Creating conda environment: ${ENV_NAME}"
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

# Install PyTorch with CUDA
print_status "Installing PyTorch with CUDA ${CUDA_VERSION}..."
conda install pytorch torchvision pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install other dependencies
print_status "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
print_status "Verifying installation..."
python verify_env.py

# Create activation script
print_status "Creating activation script..."
cat > activate_sereact.sh << EOF
#!/bin/bash
# Sereact Environment Activation Script
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
echo "🎯 Sereact environment activated!"
echo "Python: \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x activate_sereact.sh

# Final instructions
echo
echo "=================================="
print_status "Environment setup complete!"
echo
echo "📋 Next steps:"
echo "1. Activate environment: conda activate ${ENV_NAME}"
echo "   Or use: source activate_sereact.sh"
echo
echo "2. Verify setup: python verify_env.py"
echo
echo "3. Configure dataset path in config/base_train.yaml"
echo
echo "4. Start training: bash train.sh"
echo
print_warning "Note: CUDA extensions will compile automatically on first training run"
echo "This may take a few minutes - be patient!"
echo
print_status "Happy training! 🚀"
