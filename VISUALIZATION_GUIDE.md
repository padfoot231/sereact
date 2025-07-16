# 🎯 Visualization Guide for 3D Object Detection

## 📊 **Available Visualization Features**

### **1. Box Distribution Analysis (Automatic)**
- **What**: Statistical analysis of predicted vs ground truth bounding boxes
- **When**: Automatically generated during evaluation (`--eval` flag)
- **Output**: `box_distributions.png` + wandb logging

### **2. Point Cloud Visualization (Optional)**
- **What**: 3D visualization of point clouds with bounding boxes saved as images
- **When**: Only when `--visualize-point-cloud` flag is used during evaluation
- **Output**: High-resolution PNG images (1920x1080)

## 🚀 **How to Use Visualizations**

### **Option 1: Box Distribution Analysis Only**
```bash
# Quick evaluation with box distribution plots
./eval_box_distributions_only.sh
```
**Features:**
- ✅ Box volume distribution comparison
- ✅ Dimension analysis (X, Y, Z)
- ✅ Aspect ratio analysis
- ✅ Center position distribution
- ✅ Automatic plot generation
- ✅ No interactive windows (runs unattended)

### **Option 2: Full Visualization (Box + Point Cloud)**
```bash
# Complete evaluation with interactive point cloud visualization
./eval_with_visualization.sh
```
**Features:**
- ✅ All box distribution features
- ✅ 3D point cloud images saved to disk
- ✅ RGB-colored point clouds
- ✅ Predicted boxes (red) vs GT boxes (green)
- ✅ Up to 3 sample visualizations per evaluation
- ✅ High-resolution outputs (1920x1080)
- ✅ Works on headless servers

### **Option 3: Manual Command**
```bash
# Custom evaluation with specific settings
python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12349 main.py \
--eval \
--visualize-point-cloud \  # Add this flag for point cloud visualization
--cfg config/enhanced_loss_training.yaml \
--output /path/to/output \
--data-path /path/to/data \
--resume /path/to/checkpoint.pth \
--batch-size 1
```

## 📈 **Box Distribution Analysis Details**

### **Generated Plots:**
1. **Volume Distribution**
   - Histogram comparing predicted vs GT box volumes
   - Helps identify small volume under-prediction

2. **Dimension Distribution**
   - X, Y, Z dimension histograms
   - Shows dimensional bias in predictions

3. **Aspect Ratio Analysis**
   - Normalized dimension ratios
   - Reveals shape diversity issues

4. **Center Position Distribution**
   - Spatial distribution of box centers
   - Indicates localization accuracy

### **Key Insights to Look For:**
- **Small Volume Under-prediction**: Predicted volume histogram shifted left
- **Shape Diversity Issues**: Narrow aspect ratio distributions
- **Dimensional Bias**: Uneven X/Y/Z dimension distributions
- **Localization Errors**: Misaligned center position distributions

## 🎮 **Interactive Point Cloud Visualization**

### **Controls:**
- **Mouse Drag**: Rotate view
- **Mouse Wheel**: Zoom in/out
- **Right Click + Drag**: Pan view
- **Close Window**: Proceed to next visualization

### **Color Coding:**
- **Point Cloud**: RGB-colored based on camera image
- **Red Boxes**: Predicted bounding boxes
- **Green Boxes**: Ground truth bounding boxes
- **Coordinate Frame**: Reference axes (if visible)

### **What to Look For:**
- **Box Alignment**: How well predicted boxes align with objects
- **Size Accuracy**: Whether predicted boxes match object sizes
- **Shape Matching**: Whether box shapes match object geometry
- **Localization**: Whether boxes are centered on objects

## ⚙️ **Configuration Options**

### **In Config File:**
```yaml
# Enable/disable visualizations
eval_mode: true                    # Required for any visualization
visualize_point_cloud: false       # Set to true for point cloud viz

# Visualization settings (optional)
print_freq: 5                      # Logging frequency
save_freq: 5                       # Checkpoint saving frequency
```

### **Command Line Flags:**
```bash
--eval                             # Enable evaluation mode (required)
--visualize-point-cloud           # Enable point cloud visualization
--batch-size 1                    # Recommended for visualization
```

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **"No display" Error**
   ```bash
   # Fixed! Point cloud visualization now works on headless servers
   # Images are saved directly without requiring a display
   # No X11 forwarding needed
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size for visualization
   --batch-size 1
   ```

3. **No Boxes to Visualize**
   ```
   # Check if model is making predictions
   # Verify checkpoint path is correct
   # Check data path and format
   ```

4. **Images Not Being Generated**
   ```bash
   # Check if Open3D is properly installed
   pip install open3d
   # Verify write permissions in current directory
   ls -la *.png
   ```

## 📊 **Output Files**

### **Automatic Outputs:**
- `box_distributions.png` - Box analysis plots
- `point_cloud_sample_*.png` - 3D point cloud visualizations
- Console logs with visualization progress
- Wandb logs (if enabled)

### **Point Cloud Image Outputs:**
- High-resolution PNG files (1920x1080)
- Multiple camera angles and views
- RGB-colored point clouds with bounding boxes
- Coordinate frame for spatial reference

## 💡 **Best Practices**

### **For Analysis:**
1. **Start with box distributions** to identify issues
2. **Use point cloud visualization** to understand spatial context
3. **Compare before/after** training improvements
4. **Focus on problematic samples** for debugging

### **For Performance:**
1. **Use batch_size=1** for point cloud visualization
2. **Limit to few samples** (max 3 per evaluation)
3. **Run on machine with display** for interactive features
4. **Use headless mode** for automated analysis

## 🎯 **Integration with Training**

### **During Training:**
- Box distributions automatically generated during validation
- No point cloud visualization (would interrupt training)
- Metrics logged to wandb for monitoring

### **During Evaluation:**
- Full visualization capabilities available
- Interactive exploration of model performance
- Detailed analysis of prediction quality

The visualization system is designed to help you understand and improve your 3D object detection model's performance! 🚀
