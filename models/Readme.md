## 2. Architecture Design

### 2.1 Pipeline Diagram

```
Input Pipeline:
Point Cloud (N, 3)    RGB Image (3, H, W)
      |                      |
      v                      v
 Pre-Encoder           Image Encoder (NEW)
(2048 points)       (ResNet-based)
      |                      |
      \_________  Fusion Layer  __________/
                   |
                   v
         Transformer Encoder (3 layers, 256D)
                   |
                   v
         Transformer Decoder (6 layers, 256D)
                   |
                   v
            Prediction Heads:
            - Box Corners (8x3)
            - Classification
            - Angle (cls + reg)
```

### 2.2 Architectural Innovations

#### a. **Image Encoder Integration**

A ResNet18-based encoder processes RGB images and projects visual context into the feature space.

#### b. **RGB-PointCloud Fusion**

Cross-modal feature fusion layer projects global RGB features into point cloud space and concatenates them with point features, followed by a learnable MLP fusion layer.

#### c. **Transformer Backbone**

A lightweight transformer encoder-decoder enables effective global reasoning over 3D features with only 4.49M parameters.

---