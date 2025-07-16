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
