"""
Sereact 3D Object Detection Deployement Script

Main training pipeline for 3DETR-based 3D object detection with RGB-PointCloud fusion.
Supports distributed training, multi-component losses, and comprehensive evaluation metrics.
"""

from __future__ import annotations

# Standard library imports
import os
import json
import random
import argparse
from typing import Any

# Third-party imports
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# Initialize cuDNN for stability
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # Set to False for deterministic behavior
torch.backends.cudnn.deterministic = True

# Local imports - Core components
from config import get_config
from logger import create_logger

# Local imports - Model and training
from models.detr3d.model_3ddetr import build_3ddetr_model
from utils.low_precision_conversion import convert_model_to_low_precision


def parse_option():
    """Parse command line arguments for model deployment."""
    parser = argparse.ArgumentParser('3DETR model deployment script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--base_lr', type=float , help="base learning rate")

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, _ = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config: Any) -> None:
    """Main deployment function that converts model to optimized formats."""
    # Build model for deployment (no data loaders needed)
    model = build_3ddetr_model(config)

    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

   
    # breakpoint()
    logger.info('Start of conversion to low precision formats')
    try:
        convert_model_to_low_precision(config, model_without_ddp, torch.device('cuda'))
        logger.info('Model conversion successful')
    except ImportError:
        logger.warning('Low precision conversion module not available. Skipping model export.')
    except Exception as e:
        logger.error(f'Model conversion failed: {e}')
    return


if __name__ == '__main__':
    args, config = parse_option()

    # No batch size or learning rate adjustments needed for deployment

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.output, exist_ok=True)
    logger = create_logger(output_dir=config.output, dist_rank=dist.get_rank(), name=f"{config.model.name}")

    if dist.get_rank() == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
