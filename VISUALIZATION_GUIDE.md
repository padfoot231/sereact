# 🎯 Visualization Guide for 3D Object Detection

## 📊 **Available Visualization Features**

### **1. Box Distribution Analysis (Automatic)**
- **What**: Statistical analysis of predicted vs ground truth bounding boxes
- **When**: Automatically generated during evaluation (`--eval` flag)
- **Output**: `box_distributions.png` + wandb logging
- **Features**: Volume, dimension, aspect ratio, and center position analysis

### **2. Point Cloud Video Visualization (Enhanced)**
- **What**: 3D rotating video visualization of point clouds with bounding boxes
- **When**: Only when `--visualize-point-cloud` flag is used during evaluation
- **Output**: HD MP4 video files with 360° rotation
- **Features**: RGB-colored point clouds, red (predicted) and green (GT) boxes

### **3. Interactive Point Cloud Visualization (Legacy)**
- **What**: Static PNG images of 3D point clouds with bounding boxes
- **When**: Available as fallback option
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

### **Option 2: Full Visualization Suite with 3D Videos**
```bash
# Complete evaluation with 3D video visualizations
./eval_with_visualization.sh
```
**Features:**
- ✅ All box distribution analysis features
- ✅ 3D point cloud video generation (MP4 format)
- ✅ 360° rotating camera views (8-second duration)
- ✅ HD quality output (1280x720, 24fps)
- ✅ RGB-colored point clouds with height-based gradients
- ✅ Red wireframe boxes (predicted) vs Green wireframe boxes (GT)
- ✅ Professional dark theme with legend
- ✅ Up to 5 sample videos per evaluation
- ✅ Headless compatible (no display system required)
- ✅ Perfect for presentations and analysis

### **Option 3: Manual Command**
```bash
# Custom evaluation with specific settings
python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12349 main.py \
--eval \
--visualize-point-cloud \  # Add this flag for 3D video generation
--cfg config/enhanced_loss_training.yaml \
--output /path/to/output \
--data-path /path/to/data \
--resume /path/to/checkpoint.pth \
--batch-size 1
```

### **Option 4: Test Video Generation**
```bash
# Test the video generation system
python test_point_cloud_video.py
```
**Features:**
- ✅ Generates sample 3D point cloud videos
- ✅ Tests video generation functionality
- ✅ Creates both quick preview and HD quality videos
- ✅ Validates all dependencies and settings

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
   # Fixed! Video generation works on headless servers
   # MP4 videos are saved directly without requiring a display
   # No X11 forwarding needed
   ```

2. **Video Generation Fails**
   ```bash
   # Check dependencies
   pip install matplotlib opencv-python
   # Test video generation
   python test_point_cloud_video.py
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size for visualization
   --batch-size 1
   # Point clouds are automatically subsampled to 10k points
   ```

4. **No Boxes to Visualize**
   ```
   # Check if model is making predictions
   # Verify checkpoint path is correct
   # Check data path and format
   ```

5. **Videos Not Being Generated**
   ```bash
   # Check if matplotlib backend is set correctly
   # Verify write permissions in current directory
   ls -la *.mp4
   # Check available disk space for video files
   ```

## 📊 **Output Files**

### **Automatic Outputs:**
- `box_distributions.png` - Statistical box analysis plots
- `point_cloud_sample_*.mp4` - 3D point cloud video visualizations
- Console logs with visualization progress
- Wandb logs (if enabled)

### **Point Cloud Video Outputs:**
- HD MP4 video files (1280x720, 24fps)
- 8-second duration with 360° rotation
- RGB-colored point clouds with height-based gradients
- Red wireframe boxes (predicted) vs Green wireframe boxes (GT)
- Professional dark theme with legend and labels
- File sizes typically 2-5 MB per video

## 🎬 **Point Cloud Video Features**

### **Video Specifications:**
- **Format**: MP4 with H.264 encoding
- **Resolution**: 1280x720 (HD) - customizable
- **Frame Rate**: 24 fps for smooth motion
- **Duration**: 8 seconds (full 360° rotation)
- **Camera Motion**: Smooth orbital rotation around scene center

### **Visual Elements:**
- **Point Clouds**: RGB-colored with height-based gradients
  - Red increases with height
  - Blue decreases with height
  - Green constant for balance
- **Predicted Boxes**: Red wireframe bounding boxes
- **Ground Truth Boxes**: Green wireframe bounding boxes
- **Background**: Professional dark theme
- **Legend**: Clear identification of all elements

### **Technical Features:**
- **Automatic Subsampling**: Limits to 10k points for performance
- **Smart Camera Positioning**: Optimal viewing distance and angle
- **Headless Rendering**: Works without display system
- **Memory Efficient**: Handles large point clouds safely
- **Quality Options**: Configurable resolution and frame rate

### **Benefits Over Static Images:**
- **Complete Spatial Understanding**: 360° view reveals all relationships
- **Depth Perception**: True 3D structure clearly visible
- **Error Detection**: Misaligned boxes easily spotted
- **Professional Presentation**: Impressive videos for reports/demos
- **Analysis Efficiency**: Single video shows all angles

## 💡 **Best Practices**

### **For Analysis:**
1. **Start with box distributions** to identify systematic issues
2. **Use point cloud videos** to understand 3D spatial relationships
3. **Compare before/after** training improvements visually
4. **Focus on failure cases** revealed by video analysis
5. **Share videos** for team discussions and presentations

### **For Performance:**
1. **Use batch_size=1** for video generation
2. **Limit to 5 samples** per evaluation run
3. **Generate quick previews** first (low resolution, short duration)
4. **Create HD videos** for important samples and final results
5. **Monitor disk space** as videos can accumulate quickly

### **For Video Quality:**
1. **Quick Preview**: 640x480, 15fps, 3s (fast generation)
2. **Standard Quality**: 1280x720, 24fps, 8s (default)
3. **High Quality**: 1920x1080, 30fps, 10s (presentations)
4. **Debug Mode**: 480x360, 12fps, 2s (rapid iteration)

## 🎯 **Integration with Training**

### **During Training:**
- Box distributions automatically generated during validation
- No video generation (would slow training significantly)
- Metrics logged to wandb for monitoring
- Quick statistical analysis only

### **During Evaluation:**
- Full visualization capabilities available
- 3D video generation for detailed analysis
- Comprehensive model performance assessment
- Professional presentation materials

## 🚀 **Advanced Usage**

### **Custom Video Settings:**
```python
# In your evaluation script
save_point_cloud_video(
    point_cloud=your_data,
    rgb_tensor=rgb_image,
    predicted_boxes=pred_boxes,
    gt_boxes=gt_boxes,
    save_path="custom_analysis.mp4",
    duration=10.0,           # 10 second video
    fps=30,                  # High frame rate
    resolution=(1920, 1080)  # Full HD
)
```

### **Batch Video Generation:**
```bash
# Generate videos for multiple checkpoints
for checkpoint in ckpt_*.pth; do
    python -m torch.distributed.launch \
    --nproc_per_node 1 main.py \
    --eval --visualize-point-cloud \
    --resume $checkpoint \
    --output "analysis_$(basename $checkpoint .pth)"
done
```

### **Performance Monitoring:**
- **Video Generation Time**: ~30-60 seconds per video
- **Memory Usage**: ~2-4GB during generation
- **Disk Space**: ~3-5MB per video file
- **CPU Usage**: High during rendering phase

The enhanced visualization system provides comprehensive 3D analysis tools to understand and improve your model's performance! 🎯
