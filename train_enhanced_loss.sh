#!/bin/bash

# Enhanced Loss Training Script
# Focuses on improved loss functions for better box prediction

echo "🎯 Starting Enhanced Loss Training"
echo "=================================="

# Environment setup
# export WANDB_MODE="disabled"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0,1,2

# Memory and Performance Settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDNN_DETERMINISTIC=1

# Run training with enhanced loss configuration
python -m torch.distributed.launch \
--nproc_per_node 3 \
--master_port 12348 main.py \
--cfg config/enhanced_loss_training.yaml \
--output /home-local2/akath.extra.nobkp/sereact_enhanced \
--data-path /home-local2/akath.extra.nobkp/dl_challenge \
--tag "enhanced_loss_data_aug" \
--batch-size 2