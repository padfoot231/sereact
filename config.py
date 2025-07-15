"""
Configuration file for Sereact 3D Object Detection

This file contains all configuration parameters for the 3DETR-based 3D object detection system
with RGB-PointCloud fusion for 3D bounding box detection and orientation estimation.

Original 3DETR implementation:
Copyright (c) 2021 Microsoft
Licensed under The MIT License [see LICENSE for details]
Written by Ze Liu
"""

from __future__ import annotations

import os
import argparse
from typing import Any
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# =============================================================================
# GENERAL SETTINGS
# =============================================================================
_C.base = ['']                    # Base configuration files to inherit from
_C.tag = 'no_augmenation'                    # Experiment tag for identification
_C.AMP_ENABLE = True             # Enable Automatic Mixed Precision training
# =============================================================================
# DATA CONFIGURATION
# =============================================================================
_C.data = CN()

# Dataset Configuration
_C.data.dataset = 'Sereact_dataset'                                     # Dataset name
_C.data.data_path = ''                                                  # Path to dataset root directory
_C.data.batch_size = 32                                                 # Batch size per GPU

# Data Loading Configuration
_C.data.num_workers = 3                                                 # Number of data loading threads
_C.data.pin_memory = True                                               # Pin memory for faster GPU transfer
_C.data.zip_mode = False                                                # Use zipped dataset format
_C.data.cache_mode = 'part'                                            # Cache strategy: 'part', 'full', 'no'

# Data Processing Configuration
_C.data.transform = None                                                # Data transformation pipeline
_C.data.debug = False                                                   # Enable debug mode for data loading
_C.data.augment = False                                                 # Enable data augmentation

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
_C.model = CN()

# Core Model Settings
_C.model.name = '3DDETR'                                                # Model name for 3D detection
_C.model.training = True                                                # Enable training mode
_C.model.unit_test = False                                              # Run unit testing mode
_C.model.export_model = False                                           # Export model to low precision formats

# 3DETR Architecture Parameters
_C.model.position_embedding = 'fourier'                                 # Position embedding type: 'fourier', 'sine'
_C.model.mlp_dropout = 0.3                                              # MLP head dropout rate
_C.model.num_queries = 256                                              # Number of object queries for detection
_C.model.num_angular_bins = 12                                          # Angle discretization bins (30° each)

# Model Weights and Checkpoints
_C.model.pretrained = None                                              # Path to pretrained weights
_C.model.resume = ''                                                    # Path to checkpoint for resuming training
_C.model.pretrained_weights_path = "/home/s.bhat/Coding/Pre_trained_Weights/3detr/scannet_ep1080.pth"

# Transformer Encoder Configuration
_C.model.encoder = CN()
_C.model.encoder.dim = 256                                              # Encoder hidden dimension
_C.model.encoder.nheads = 4                                             # Number of attention heads
_C.model.encoder.ffn_dim = 128                                          # Feed-forward network dimension
_C.model.encoder.dropout = 0.1                                          # Dropout rate
_C.model.encoder.activation = 'relu'                                    # Activation function
_C.model.encoder.num_layers = 3                                         # Number of encoder layers
_C.model.encoder.type = 'vanilla'                                       # Encoder type (used in YAML configs)
_C.model.encoder.preencoder_npoints = 2048                             # Number of points after pre-processing
_C.model.encoder.use_color = False                                      # Use RGB color information

# Transformer Decoder Configuration
_C.model.decoder = CN()
_C.model.decoder.dim = 256                                              # Decoder hidden dimension
_C.model.decoder.nhead = 4                                              # Number of attention heads
_C.model.decoder.ffn_dim = 256                                          # Feed-forward network dimension
_C.model.decoder.dropout = 0.1                                          # Dropout rate
_C.model.decoder.num_layers = 3                                         # Number of decoder layers
# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
_C.train = CN()

# Basic Training Parameters
_C.train.max_epoch = 500                                                # Maximum number of training epochs
_C.train.start_epoch = 0                                                # Starting epoch (for resuming)
_C.train.unit_test_epoch = 100                                          # Epoch interval for unit testing

# Learning Rate Configuration
_C.train.base_lr = 5e-4                                                 # Base learning rate
_C.train.lr_scheduler = 'cosine'                                        # Learning rate scheduler type
_C.train.warm_lr_epochs = 20                                            # Warmup epochs
_C.train.warm_lr = 0.03                                                 # Warmup learning rate
_C.train.final_lr = 1e-6                                                # Final learning rate

# Optimization Parameters
_C.train.weight_decay = 0.01                                            # Weight decay for regularization
_C.train.filter_biases_wd = True                                        # Filter biases from weight decay
_C.train.clip_grad = 5.0                                                # Gradient clipping threshold
_C.train.accumulation_steps = 1                                         # Gradient accumulation steps
# Memory Optimization
_C.train.use_checkpoint = False                                         # Use gradient checkpointing to save memory

# =============================================================================
# LOSS FUNCTION CONFIGURATION
# =============================================================================
_C.loss = CN()

# Hungarian Matcher Cost Weights (for assignment during training)
_C.loss.matcher_costs = CN()
_C.loss.matcher_costs.giou = 5.0                                        # GIoU cost weight for matching
_C.loss.matcher_costs.cost_box_corners = 1.0                           # Box corner cost weight for matching
_C.loss.matcher_costs.l1 = 2.0                                         # L1 distance cost weight for matching

# Loss Component Weights (for final loss computation)
_C.loss.weights = CN()
_C.loss.weights.giou = 1.0                                              # GIoU loss weight
_C.loss.weights.box_corners = 1.0                                       # Box corner loss weight
_C.loss.weights.size = 1.0                                              # Size prediction loss weight
_C.loss.weights.size_reg = 1.0                                          # Size regularization loss weight
_C.loss.weights.angle_cls = 0.1                                         # Angle classification loss weight
_C.loss.weights.angle_reg = 0.1                                         # Angle regression loss weight

# =============================================================================
# SYSTEM AND EXPERIMENT CONFIGURATION
# =============================================================================

# Experiment Management
_C.output = ''                                                          # Output directory path
_C.seed = 0                                                             # Random seed for reproducibility
_C.eval_mode = False                                                    # Run in evaluation-only mode
_C.print_freq = 10                                                      # Frequency of logging during training

# Training System Configuration
_C.amp_opt_level = ''                                                   # Mixed precision level ('O0', 'O1', 'O2')
_C.local_rank = 0                                                       # Local rank for distributed training
_C.save_freq = 1                                                        # Checkpoint saving frequency (epochs)

# Testing and Debugging
_C.unit_test = False                                                    # Enable unit testing mode


# =============================================================================
# CONFIGURATION UTILITY FUNCTIONS
# =============================================================================

def _update_config_from_file(config: CN, cfg_file: str) -> None:
    """Update config from YAML file with hierarchical inheritance support.

    Args:
        config: Configuration node to update
        cfg_file: Path to YAML configuration file
    """
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Handle hierarchical config inheritance
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config: CN, args: argparse.Namespace) -> None:
    """Update configuration with command line arguments and YAML files.

    Args:
        config: Configuration node to update
        args: Parsed command line arguments containing config file path and overrides
    """
    # Load configuration from YAML file
    _update_config_from_file(config, args.cfg)

    config.defrost()

    # Apply command line option overrides
    if args.opts:
        config.merge_from_list(args.opts)

    # Apply specific command line argument overrides
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.data_path:
        config.data.data_path = args.data_path
    if args.augment:
        config.data.augment = args.augment
    if args.pretrained:
        config.model.pretrained = args.pretrained
    if args.export: 
        config.model.export_model = args.export
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

    # Set distributed training configuration
    config.local_rank = args.local_rank

    # Configure output directory structure
    config.output = os.path.join(config.output, config.model.name, config.tag)

    config.freeze()


def get_config(args: argparse.Namespace) -> CN:
    """Get a complete configuration object with default values and command line overrides.

    Args:
        args: Parsed command line arguments containing config file and parameter overrides

    Returns:
        CN: Fully configured configuration node ready for training/evaluation
    """
    # Return a clone to preserve default values for other instances
    config = _C.clone()
    update_config(config, args)

    return config
