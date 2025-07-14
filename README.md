# Sereact - 3D Object Detection with RGB-PointCloud Fusion

A PyTorch implementation of 3D object detection using a modified 3DETR (3D Detection Transformer) architecture with RGB-PointCloud fusion capabilities. This project combines point cloud data with RGB images for improved 3D bounding box detection.

## 🚀 Features

- **3DETR-based Architecture**: Built on the proven 3D Detection Transformer framework
- **RGB-PointCloud Fusion**: Combines RGB image features with point cloud data for enhanced detection
- **Multi-component Loss Function**: Includes GIoU, box corners, size, and regularization losses
- **Comprehensive Evaluation**: IoU metrics at multiple thresholds (0.25, 0.5) and mean IoU
- **Data Augmentation**: Point cloud augmentations including random cuboid cropping and flipping
- **Distributed Training**: Support for multi-GPU training with PyTorch distributed
- **Wandb Integration**: Experiment tracking and visualization

## 📋 Requirements

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.4
- torchvision >= 0.13
- open3d == 0.18.0
- numpy == 1.26.4
- wandb == 0.19.4
- matplotlib >= 3.5
- scipy == 1.15.1

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
   - 3D bounding box regression
   - Optional segmentation head for mask supervision

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

The model uses a multi-component loss function:

- **GIoU Loss**: Generalized Intersection over Union for box positioning
- **Box Corners Loss**: L1 distance for corner accuracy
- **Size Loss**: L1 distance for size prediction
- **Size Regularization**: Penalty for oversized predictions

Total loss: `λ₁ × L_giou + λ₂ × L_corners + λ₃ × L_size + λ₄ × L_reg`

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
  pretrained_weights_path: "/path/to/pretrained/weights"
```

## 📁 Project Structure

```
sereact/
├── config/                 # Configuration files
├── dataloader/            # Dataset loading and augmentation
├── losses/                # Loss function implementations
├── models/                # Model architectures
│   └── detr3d/           # 3DETR model components
├── utils/                 # Utility functions
├── main.py               # Training/evaluation script
├── train.sh              # Distributed training script
└── requirements.txt      # Dependencies
```

## 🎯 Pre-trained Models

The project supports loading pre-trained weights from the original 3DETR model trained on ScanNet dataset. Configure the path in your YAML file:

```yaml
model:
  pretrained_weights_path: "/path/to/scannet_ep1080.pth"
```

## 📝 Example Results

| Epoch | Val/IoU@0.25 | Val/IoU@0.50 | Val/Mean IoU |
|-------|--------------|--------------|--------------|
|   10  |    0.348     |    0.064     |    0.213     |
|   30  |    0.392     |    0.061     |    0.227     |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is based on the original 3DETR implementation and follows similar licensing terms.

## 🙏 Acknowledgments

- Original [3DETR](https://github.com/facebookresearch/3detr) implementation by Facebook Research
- Swin Transformer components (Microsoft)
- PyTorch and torchvision teams

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
