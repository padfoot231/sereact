#!/bin/bash 
export WANDB_MODE="disabled"
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12346  main.py \
--cfg config/base_train.yaml \
--output /home-local2/akath.extra.nobkp/sereact \
--data-path /home-local2/akath.extra.nobkp/dl_challenge \
--batch-size 2 
# --pretrained /home-local2/akath.extra.nobkp/scannet_ep1080.pth
