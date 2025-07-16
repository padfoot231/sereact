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
