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

## 📋 Requirements

### Dependencies

define requirement.txt

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
