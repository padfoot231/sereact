# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim
import torch

def build_optimizer(config, model: torch.nn.Module) -> torch.optim.AdamW:
    """
    Build an AdamW optimizer with optional weight decay filtering for biases and parameters with shape length 1.

    Args:
        cfg_opt (DictConfig): A configuration object containing optimizer parameters.
        model (torch.nn.Module): The model containing parameters to optimize.

    Returns:
        torch.optim.AdamW: An AdamW optimizer configured with the specified parameters and weight decay settings.
    """
    # Initialize lists to hold parameters with and without weight decay
    params_with_decay = []
    params_without_decay = []

    # Iterate over model parameters
    for name, param in model.named_parameters():
        # Skip parameters that do not require gradients
        if param.requires_grad is False:
            continue
        # Filter out biases and parameters with shape length 1 if specified
        if config.train.filter_biases_wd and (
            len(param.shape) == 1 or name.endswith('bias') or 'norm' in name
        ):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    # Create parameter groups with appropriate weight decay settings
    if config.train.filter_biases_wd:
        param_groups = [
            {'params': params_without_decay, 'weight_decay': 0.0},
            {'params': params_with_decay, 'weight_decay': config.train.weight_decay},
        ]
    else:
        param_groups = [
            {'params': params_with_decay, 'weight_decay': config.train.weight_decay},
        ]

    # Build the AdamW optimizer with the specified parameter groups and learning rate
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.base_lr)

    return optimizer

