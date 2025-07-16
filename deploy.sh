#!/bin/bash

python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12347  deploy.py \
--cfg config/enhanced_loss_training.yaml \
--output /home-local2/akath.extra.nobkp/sereact_enhanced \
--resume /home-local2/akath.extra.nobkp/sereact_enhanced/3DDETR/enhanced_loss_data_aug/ckpt_best.pth \
--tag "enhanced_loss_data_aug" \
--export True \
