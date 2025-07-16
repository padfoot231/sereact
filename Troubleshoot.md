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
