#!/usr/bin/env python3
"""
Sereact Environment Verification Script

This script verifies that all required dependencies are properly installed
and the environment is ready for training the Sereact 3D object detection system.
"""

import sys
import subprocess
from typing import List, Tuple


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 7 <= version.minor <= 10:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.7-3.10")
        return False


def check_package_import(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package can be imported and return version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def check_cuda_setup() -> bool:
    """Check CUDA setup and GPU availability."""
    try:
        import torch
        
        print(f"\n🔥 CUDA Setup:")
        print(f"   PyTorch CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            print("❌ CUDA not available - training will be slow on CPU")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_model_imports() -> bool:
    """Check if model components can be imported."""
    print(f"\n🏗️ Model Components:")
    
    try:
        from models.detr3d.model_3ddetr import build_3ddetr_model
        print("✅ Main model imports successful")
        
        from losses.loss_3ddetr import LossFunction
        print("✅ Loss function imports successful")
        
        from utils.mean_iou_evaluation import IoUEvaluator
        print("✅ Evaluation utilities imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False


def check_cuda_extensions() -> bool:
    """Check if CUDA extensions can be compiled/imported."""
    print(f"\n⚡ CUDA Extensions:")
    
    try:
        # This will trigger compilation if not already done
        from models.detr3d.pointnet2 import PointnetSAModuleVotes
        print("✅ PointNet++ CUDA extensions available")
        return True
        
    except ImportError as e:
        print(f"❌ CUDA extensions failed: {e}")
        print("   Note: Extensions will compile automatically on first training run")
        return False


def main():
    """Main verification function."""
    print("🔍 Sereact Environment Verification")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Core packages to check
    core_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('open3d', 'open3d'),
        ('yacs', 'yacs'),
        ('yaml', 'PyYAML'),
        ('timm', 'timm'),
        ('sklearn', 'scikit-learn'),
        ('wandb', 'wandb'),
        ('imageio', 'imageio'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    print(f"\n📦 Package Dependencies:")
    all_packages_ok = True
    
    for import_name, package_name in core_packages:
        success, version = check_package_import(import_name)
        if success:
            print(f"✅ {package_name}: {version}")
        else:
            print(f"❌ {package_name}: Not installed")
            all_packages_ok = False
    
    # Check CUDA setup
    cuda_ok = check_cuda_setup()
    
    # Check model imports
    model_ok = check_model_imports()
    
    # Check CUDA extensions
    extensions_ok = check_cuda_extensions()
    
    # Final summary
    print(f"\n" + "=" * 50)
    print(f"📋 Verification Summary:")
    print(f"   Python version: {'✅' if python_ok else '❌'}")
    print(f"   Core packages: {'✅' if all_packages_ok else '❌'}")
    print(f"   CUDA setup: {'✅' if cuda_ok else '⚠️'}")
    print(f"   Model imports: {'✅' if model_ok else '❌'}")
    print(f"   CUDA extensions: {'✅' if extensions_ok else '⚠️'}")
    
    if python_ok and all_packages_ok and model_ok:
        print(f"\n🎉 Environment is ready for training!")
        if not cuda_ok:
            print(f"⚠️  Training will run on CPU (slower)")
    else:
        print(f"\n❌ Environment setup incomplete")
        print(f"   Please install missing dependencies:")
        print(f"   pip install -r requirements.txt")
    
    return python_ok and all_packages_ok and model_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
