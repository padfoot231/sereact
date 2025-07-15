from __future__ import annotations

import os
import itertools
import random
from typing import Any, Dict, Optional, Tuple, Union
import logging
# from matplotlib.pyplot import title
import imageio.v2 as imageio
import torch
import cv2
import numpy as np
import torch.distributed as dist
import scipy.optimize
# from pyinstrument import Profiler
from PIL import Image
import torch.nn as nn
# profiler = Profiler(interval=0.0001)
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

# import SimpleITK as sitk
# from medpy import metric
from scipy.ndimage import zoom
import torch

transform = T.ToPILImage()

def save_checkpoint(
    config: Any,
    epoch: int,
    model: torch.nn.Module,
    max_miou: float,
    miou: float,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    loss_scaler: Any,
    logger: logging.Logger
) -> None:
    """Save model checkpoint if current IoU is better than max IoU.

    Args:
        config: Configuration object
        epoch: Current epoch number
        model: PyTorch model to save
        max_miou: Maximum IoU achieved so far
        miou: Current IoU
        optimizer: Optimizer state to save
        lr_scheduler: Learning rate scheduler state to save
        loss_scaler: Loss scaler state to save
        logger: Logger for output messages
    """
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': max_miou,
        'scaler': loss_scaler.state_dict(),
        'epoch': epoch,
        'config': config
    }

    if miou > max_miou:
        save_path = os.path.join(config.output, f'ckpt_best.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")


def load_checkpoint(
    config: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    loss_scaler: Any,
    logger: logging.Logger
) -> float:
    """Load checkpoint for resuming training.

    Args:
        config: Configuration object
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into
        lr_scheduler: Learning rate scheduler to load state into
        loss_scaler: Loss scaler to load state into
        logger: Logger for output messages

    Returns:
        float: Maximum IoU from the loaded checkpoint
    """
    logger.info(f"==============> Resuming from {config.model.resume}......")

    checkpoint = torch.load(config.model.resume, map_location='cpu')

    # Load model state
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(f"Model loading result: {msg}")

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Load scheduler state
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # Load scaler state
    loss_scaler.load_state_dict(checkpoint['scaler'])

    # Get max accuracy (IoU)
    max_miou = checkpoint.get('max_accuracy', 0.0)

    logger.info(f"=> loaded checkpoint '{config.model.resume}' (epoch {checkpoint['epoch']}, max_miou: {max_miou})")

    del checkpoint
    torch.cuda.empty_cache()

    return max_miou


def load_pretrained(config: Any, model: torch.nn.Module, logger: logging.Logger) -> None:
    """Load pretrained weights into model.

    Args:
        config: Configuration object containing pretrained weights path
        model: PyTorch model to load weights into
        logger: Logger for output messages
    """
    logger.info(f"==============> Loading weight {config.model.pretrained} for fine-tuning......")
    checkpoint = torch.load(config.model.pretrained, map_location='cpu')
    state_dict = checkpoint['model']

    msg = model.load_state_dict(state_dict, strict=False)
    # logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.model.pretrained}'")

    del checkpoint
    torch.cuda.empty_cache()
    


def get_grad_norm(parameters: Union[torch.Tensor, list], norm_type: float = 2) -> float:
    """Calculate gradient norm for given parameters.

    Args:
        parameters: Model parameters or list of parameters
        norm_type: Type of norm to calculate (default: 2)

    Returns:
        float: Calculated gradient norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes in distributed training.

    Args:
        tensor: Tensor to reduce

    Returns:
        torch.Tensor: Reduced tensor
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters: Union[torch.Tensor, list], norm_type: float = 2.0) -> torch.Tensor:
    """Calculate gradient norm for AMP scaler.

    Args:
        parameters: Model parameters or list of parameters
        norm_type: Type of norm to calculate (default: 2.0)

    Returns:
        torch.Tensor: Calculated gradient norm tensor
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm



class NativeScalerWithGradNormCount:
    """Native PyTorch AMP scaler with gradient norm counting."""

    state_dict_key = "amp_scaler"

    def __init__(self) -> None:
        """Initialize the scaler."""
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[float] = None,
        parameters: Optional[list] = None,
        create_graph: bool = False,
        update_grad: bool = True
    ) -> Optional[torch.Tensor]:
        """Scale loss and perform backward pass with optional gradient clipping.

        Args:
            loss: Loss tensor to scale and backward
            optimizer: Optimizer to update
            clip_grad: Gradient clipping value (optional)
            parameters: Model parameters for gradient clipping
            create_graph: Whether to create computation graph for backward
            update_grad: Whether to update gradients

        Returns:
            Optional[torch.Tensor]: Gradient norm if computed, None otherwise
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state dict."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scaler state dict."""
        self._scaler.load_state_dict(state_dict)

