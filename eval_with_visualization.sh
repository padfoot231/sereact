#!/bin/bash

# Enhanced Evaluation Script with Point Cloud Visualization
# Includes both box distribution plots and interactive point cloud visualization

echo "🎯 Starting Enhanced Evaluation with Visualizations"
echo "=================================================="

# Environment setup
export WANDB_MODE="disabled"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0,1,2

# Memory and Performance Settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDNN_DETERMINISTIC=1

echo "📊 Visualization Features:"
echo "  ✅ Box distribution plots (automatic)"
echo "  ✅ 3D point cloud plots using matplotlib"
echo "  ✅ Predicted vs Ground Truth comparison"
echo "  ✅ RGB point cloud coloring"
echo "  ✅ Truly headless operation (no display required)"
echo ""

echo "💾 Note: All visualizations saved as image files"
echo "   Uses matplotlib backend - works on any server"
echo ""

# Run evaluation with enhanced visualizations
python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12349 main.py \
--eval \
--visualize-point-cloud \
--cfg config/enhanced_loss_training.yaml \
--output /home-local2/akath.extra.nobkp/sereact_enhanced \
--data-path /home-local2/akath.extra.nobkp/dl_challenge \
--resume /home-local2/akath.extra.nobkp/sereact/3DDETR/no_augmentation/ckpt_best.pth \
--batch-size 1

echo ""
echo "🎉 Enhanced evaluation completed!"
echo "📊 Generated visualizations:"
echo "  - box_distributions.png (box analysis plots)"
echo "  - point_cloud_sample_*.png (3D point cloud visualizations)"
echo "  - All images logged to wandb (if enabled)"
echo ""
echo "💡 Visualization details:"
echo "  - Red wireframes = Predicted bounding boxes"
echo "  - Green wireframes = Ground Truth bounding boxes"
echo "  - RGB-colored point clouds (subsampled for clarity)"
echo "  - 3D matplotlib plots with proper legends"
echo "  - Fallback text summaries if visualization fails"
