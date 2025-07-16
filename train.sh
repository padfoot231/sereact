#!/bin/bash
# export WANDB_MODE="disabled"

# cuDNN Error Fixes
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0,1,2

# Memory and Performance Settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDNN_DETERMINISTIC=1

# Debugging (uncomment if needed)
# export NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


python -m torch.distributed.launch \
--nproc_per_node 3 \
--master_port 12346  main.py \
--cfg config/base_train.yaml \
--output /home-local2/akath.extra.nobkp/sereact_dataaug \
--data-path /home-local2/akath.extra.nobkp/dl_challenge \
--tag "augmentation" \
--batch-size 2
# --pretrained /home-local2/akath.extra.nobkp/scannet_ep1080.pth
