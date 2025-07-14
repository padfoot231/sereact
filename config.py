# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

from __future__ import annotations

import os
import argparse
from typing import Any
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.base = ['']  

_C.tag = '123'

_C.AMP_ENABLE = True 
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.data = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.data.batch_size = 32
# Path to dataset, could be overwritten by command line argument
_C.data.data_path = ''
# Dataset name
_C.data.dataset = 'Sereact_dataset'

_C.data.zip_mode = False
# Cache Data in Memory, could be overwritten by command line argument
_C.data.cache_mode = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.data.pin_memory = True
# Number of data loading threads
_C.data.num_workers = 3
# Transform
_C.data.transform = None
#debug
_C.data.debug = False
#augment
_C.data.augment = False

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.model = CN()

# Model type
# _C.MODEL.TYPE = 'swin'
# # Model name
_C.model.name = '3DDETR.yaml'

#deeplab model
# _C.MODEL.CONVNAME = 'deeplabv3plus_resnet101'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.model.pretrained = None
# '/home-local2/akath.extra.nobkp/swin_tiny_patch4_window7_224.pth'
# '/home-local2/akath.extra.nobkp/swin_tiny_patch4_window7_224.pth'
# 
# Checkpoint to resume, could be overwritten by command line argument
_C.model.resume = ''
# Number of classes, overwritten in data preparation

##############
_C.model.position_embedding = 'fourier' 
_C.model.mlp_dropout = 0.3
_C.model.num_queries = 256
_C.model.num_angular_bins = 12
_C.model.pretrained_weights_path = "/home/s.bhat/Coding/Pre_trained_Weights/3detr/scannet_ep1080.pth"
_C.model.export_model = False #not accounted for
_C.model.training = True
_C.model.unit_test = False

#######encoder 
_C.model.encoder = CN()
_C.model.encoder.dim = 256
_C.model.encoder.nheads = 4
_C.model.encoder.ffn_dim = 128
_C.model.encoder.dropout = 0.1
_C.model.encoder.activation = 'relu'
_C.model.encoder.num_layers = 3
_C.model.encoder.type = 'vanilla'  #not in the model
_C.model.encoder.preencoder_npoints = 2048
_C.model.encoder.use_color = False
#######decoder 
_C.model.decoder = CN()
_C.model.decoder.dim = 256
_C.model.decoder.nhead = 4
_C.model.decoder.ffn_dim = 256
_C.model.decoder.dropout = 0.1
_C.model.decoder.num_layers = 3
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.train = CN()
_C.train.base_lr = 5e-4
_C.train.lr_scheduler = 'cosine'
_C.train.max_epoch = 500 #_C.TRAIN.EPOCHS = 500
_C.train.start_epoch = 0
_C.train.clip_grad = 5.0 #_C.TRAIN.CLIP_GRAD = 5.0

_C.train.accumulation_steps = 1 #_C.TRAIN.ACCUMULATION_STEPS = 1

_C.train.warm_lr_epochs = 20
_C.train.warm_lr = 0.03
_C.train.final_lr = 1e-6

_C.train.filter_biases_wd = True
_C.train.weight_decay = 0.01

_C.train.unit_test_epoch = 100




# _C.TRAIN.WARMUP_LR = 5e-7``
# _C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm

# Auto resume from latest checkpoint
_C.train.auto_resume = False
# Gradient accumulation steps
# could be overwritten by command line argument
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.train.use_checkpoint = False


_C.loss = CN()
_C.loss.matcher_costs = CN()

_C.loss.matcher_costs.giou = 5.0
_C.loss.matcher_costs.cost_box_corners = 1.0
_C.loss.matcher_costs.l1 = 2.0

_C.loss.weights = CN()
_C.loss.weights.giou = 1.0
_C.loss.weights.box_corners = 1.0
_C.loss.weights.size = 1.0
_C.loss.weights.size_reg = 1.0

# # Whether to use center crop when testing
# _C.TEST.CROP = True
# # Whether to use SequentialSampler as validation sampler
# _C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.amp_opt_level = ''
# Path to output folder, overwritten by command line argument
_C.output = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.save_freq = 1
# Frequency to logging info
_C.print_freq = 10
# Fixed random seed
_C.seed = 0
# Perform evaluation only, overwritten by command line argument
_C.eval_mode = False
# Test throughput only, overwritten by command line argument
_C.unit_test = False
# local rank for DistributedDataParallel, given by command line argument
_C.local_rank = 0


def _update_config_from_file(config: CN, cfg_file: str) -> None:
    """Update config from YAML file.

    Args:
        config: Configuration node to update
        cfg_file: Path to YAML configuration file
    """
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config: CN, args: argparse.Namespace) -> None:
    """Update configuration with command line arguments.

    Args:
        config: Configuration node to update
        args: Parsed command line arguments
    """
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # lst = ['MODEL.SWIN.RADIUS_CUTS', 16, 'MODEL.SWIN.AZIMUTH_CUTS', 64]
    # config.merge_from_list(lst)

    # merge from specific arguments
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.data_path:
        config.data.data_path = args.data_path
    # if args.zip:
    #     config.data.ZIP_MODE = True
    # if args.cache_mode:
    #     config.data.CACHE_MODE = args.cache_mode
    if args.pretrained:
        config.model.pretrained = args.pretrained
    if args.resume:
        config.model.resume = args.resume
    if args.accumulation_steps:
        config.train.accumulation_steps = args.accumulation_steps
    if args.use_checkpoint:
        config.train.use_checkpoint = True
    if args.base_lr:
        config.train.base_lr = 0.05
    if args.amp_opt_level:
        config.amp_opt_level = args.amp_opt_level
    if args.output:
        config.output = args.output
    if args.tag:
        config.tag = args.tag
    if args.eval:
        config.eval_mode = True
    if args.unit_test:
        config.unit_test = True

    # set local rank for distributed training
    config.local_rank = args.local_rank

    # output folder
    config.output = os.path.join(config.output, config.model.name, config.tag)

    config.freeze()


def get_config(args: argparse.Namespace) -> CN:
    """Get a yacs CfgNode object with default values.

    Args:
        args: Parsed command line arguments

    Returns:
        CN: Configuration node with updated values
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
