#!/bin/bash

python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12347  deploy.py \
--cfg config/base_train.yaml \
--output /home-local2/akath.extra.nobkp/sereact \
--resume /home-local2/akath.extra.nobkp/sereact/3DDETR/no_augmentation/ckpt_best.pth \
--export True \
