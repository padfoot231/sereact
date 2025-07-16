"""
Sereact 3D Object Detection Training Script

Main training pipeline for 3DETR-based 3D object detection with RGB-PointCloud fusion.
Supports distributed training, multi-component losses, and comprehensive evaluation metrics.
"""

from __future__ import annotations

# Standard library imports
import os
import time
import json
import random
import datetime
import argparse
from typing import Dict, Any, Tuple

# Third-party imports
import numpy as np
import wandb
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from timm.utils import AverageMeter

# Initialize cuDNN for stability
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # Set to False for deterministic behavior
torch.backends.cudnn.deterministic = True

# Local imports - Core components
from config import get_config
from dataloader import build_loader
from logger import create_logger
from optimizer import build_optimizer

# Local imports - Model and training
from models.detr3d.model_3ddetr import build_3ddetr_model
from losses.loss_3ddetr import LossFunction
from utils.mean_iou_evaluation import IoUEvaluator
from utils.model_utils import (
    load_pretrained,
    save_checkpoint,
    load_checkpoint,
    NativeScalerWithGradNormCount
)

# Initialize Wandb for experiment tracking
wandb.init(project="sereact project", entity='padfoot')
def parse_option() -> Tuple[argparse.Namespace, Any]:
    """Parse command line arguments and return args and config."""
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
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
    """Main training function that orchestrates model training, evaluation, and checkpointing."""
    # Initialize GPU and cuDNN for stability
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Warm up cuDNN with a small operation
        try:
            dummy_input = torch.randn(1, 3, 32, 32).cuda()
            dummy_output = torch.nn.functional.relu(dummy_input)
            del dummy_input, dummy_output
            torch.cuda.empty_cache()
            logger.info("cuDNN initialized successfully")
        except Exception as e:
            logger.warning(f"cuDNN initialization warning: {e}")
    _, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    model = build_3ddetr_model(config)

    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=False, broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.train.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.max_epoch)
    elif config.train.lr_scheduler == 'cosine_warmup':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.train.warm_lr_epochs, T_mult=2
        )
    else:
        raise ValueError(f'Invalid learning rate scheduler: {config.train.lr_scheduler}')

    loss_module = LossFunction(config)
    iou_evaluator = IoUEvaluator()

    max_miou = 0.0

    if config.model.resume:
        max_miou = load_checkpoint(config, model_without_ddp, optimizer, scheduler, loss_scaler, logger)
        miou, _ = validate(config, loss_module, 0, iou_evaluator, data_loader_val, model)
        logger.info(f"Mean iou of the network on the {len(dataset_val)} test images: {miou:.4f}")
        if config.eval_mode:
            return
        
    if config.model.pretrained and (not config.model.resume):
        load_pretrained(config, model_without_ddp, logger)
        miou, _ = validate(config, loss_module, 0, iou_evaluator, data_loader_val, model)
        logger.info(f"Mean iou of the network on the {len(dataset_val)} test images: {miou:.4f}")

        # Model export to low precision formats
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.train.start_epoch, config.train.max_epoch):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, loss_module, iou_evaluator, data_loader_train, optimizer, epoch,scheduler,
                        loss_scaler)

        miou, loss = validate(loss_module, epoch, iou_evaluator, data_loader_val, model)
        
        # if dist.get_rank() == 0 and (epoch % config.save_freq == 0 or epoch == (config.train.max_epoch - 1)):
        save_checkpoint(config, epoch, model_without_ddp, max_miou, miou, optimizer, scheduler, loss_scaler,
                        logger)

        logger.info(f"Mean IOU of the network on the {len(dataset_val)} test images: {miou:.4f}%")
        max_miou = max(max_miou, miou)
        logger.info(f'Max miou: {max_miou:.4f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(
    config: Any,
    model: torch.nn.Module,
    loss_module: LossFunction,
    iou_evaluator: IoUEvaluator,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lr_scheduler: Any,
    loss_scaler: NativeScalerWithGradNormCount
) -> Dict[str, float]:
    """Execute one training epoch with forward/backward pass and metrics tracking."""
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    start = time.time()
    end = time.time()

    for batch_idx, batch in enumerate(data_loader):
        # Move input data to GPU
        inputs = batch['pcd_tensor'].cuda()
        inputs_rgb = batch['rgb_tensor'].cuda()
        gt_bboxes = batch['bbox3d_tensor'].cuda()
        pcd_dims_min = batch['point_cloud_dims_min'].cuda()
        pcd_dims_max = batch['point_cloud_dims_max'].cuda()
        torch.autograd.set_detect_anomaly(True)

        try:
            # Forward pass
            outputs = model(
                inputs,
                inputs_rgb,
                point_cloud_dims_min=pcd_dims_min,
                point_cloud_dims_max=pcd_dims_max,
            )
        except RuntimeError as e:
            if "cuDNN error" in str(e):
                logger.error(f"cuDNN error encountered: {e}")
                logger.info("Clearing GPU cache and retrying...")
                torch.cuda.empty_cache()
                # Retry once
                outputs = model(
                    inputs,
                    inputs_rgb,
                    point_cloud_dims_min=pcd_dims_min,
                    point_cloud_dims_max=pcd_dims_max,
                )
            else:
                raise e

        # Compute losses
        pred_boxes = outputs['outputs']
        pred_boxes_aux = outputs['auxiliary_outputs']
        loss, _, assignments = loss_module(pred_boxes, gt_bboxes)
        loss_aux = 0
        for aux in pred_boxes_aux:
            loss_aux_cls, _, _ = loss_module(aux, gt_bboxes)
            loss_aux += loss_aux_cls
        total_loss = loss + 0.01 * loss_aux
        # breakpoint()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.train.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(batch_idx + 1) % config.train.accumulation_steps == 0)

        if (batch_idx + 1) % config.train.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step()
        loss_scale_value = loss_scaler.state_dict()["scale"]
        loss_meter.update(total_loss.item())
        
        predicted_bboxes_matched, gt_bboxes_matched = (
                get_predicted_and_gt_boxes_from_assignments(
                    pred_boxes=pred_boxes, assignments=assignments, gt_bbox=gt_bboxes
                )
            )

        iou_evaluator.update(predicted_bboxes_matched, gt_bboxes_matched)
        torch.cuda.synchronize()
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
       
        if batch_idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - batch_idx)
            metric = iou_evaluator.compute_metrics()['mean_iou']
            logger.info(
                f'Train: [{epoch}/{config.train.max_epoch}][{batch_idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'miou {metric:.4f}\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
        wandb.log(
                {
                    'Iteration': batch_idx,
                    'train_loss': loss_meter.avg,
                    'train_miou': metric,
                }
            )
    _ = iou_evaluator.compute_metrics()

            
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(
    loss_module: LossFunction,
    epoch: int,
    iou_evaluator: IoUEvaluator,
    data_loader: DataLoader,
    model: torch.nn.Module
) -> Tuple[float, float]:
    """Validate model on validation dataset and return loss and IoU metrics."""
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    iou_evaluator.reset()

    logger.info('Starting validation...')

    for _, batch_data in enumerate(data_loader):
        inputs = batch_data['pcd_tensor'].cuda()
        gt_bboxes = batch_data['bbox3d_tensor'].cuda()
        inputs_rgb = batch_data['rgb_tensor'].cuda()
        pcd_dims_min = batch_data['point_cloud_dims_min'].cuda()
        pcd_dims_max = batch_data['point_cloud_dims_max'].cuda()

        outputs = model(
            inputs,
            inputs_rgb,
            point_cloud_dims_min=pcd_dims_min,
            point_cloud_dims_max=pcd_dims_max,
        )

        # Get predictions
        pred_boxes = outputs['outputs']

        # Compute loss (if criterion is provided)
        # if criterion:
        loss, _, assignments = loss_module(outputs=pred_boxes, targets=gt_bboxes)

        loss_meter.update(loss.item())
        if iou_evaluator:
            predicted_bboxes_matched, gt_bboxes_matched = (
                get_predicted_and_gt_boxes_from_assignments(
                    pred_boxes=pred_boxes, assignments=assignments, gt_bbox=gt_bboxes
                )
            )
        iou_evaluator.update(predicted_bboxes_matched, gt_bboxes_matched)

        batch_time.update(time.time() - end)
        end = time.time()

    metrics = iou_evaluator.compute_metrics()['mean_iou']
    wandb.log(
        {
            'epoch': epoch,
            'val_loss': loss_meter.avg,
            'val_miou': metrics,
        }
    )
    return metrics, loss_meter.avg


def get_predicted_and_gt_boxes_from_assignments(
    pred_boxes: dict,
    assignments: dict,
    gt_bbox: torch.Tensor
) -> tuple:
    """
    Extracts matched predicted and ground truth bounding boxes across the entire batch.

    Args:
        pred_boxes (dict): Dict containing predicted boxes with key 'box_corners' or similar.
            Shape: [B, N_pred, 8, 3]
        assignments (dict): Dict with 'assignments' as a list of assignment tuples per batch.
            assignments['assignments'][b] = [pred_indices_tensor, gt_indices_tensor] or []
        gt_bbox (Tensor): Ground truth boxes [B, N_gt, 8, 3]

    Returns:
        Tuple of:
            - predicted_bboxes_matched: List of [num_matched_i, 24] per batch
            - gt_bboxes_matched: List of [num_matched_i, 24] per batch
    """
    import numpy as np

    B = gt_bbox.shape[0]
    pred_key = next((k for k in ['box_corners', 'pred_boxes', 'boxes'] if k in pred_boxes), None)
    if pred_key is None:
        raise ValueError("No valid prediction key found in pred_boxes")

    predicted_batched = []
    gt_batched = []

    for b in range(B):
        # Check if there are any assignments for this batch element
        if (b >= len(assignments['assignments']) or
            not assignments['assignments'][b] or
            len(assignments['assignments'][b]) == 0):
            # No assignments for this batch element
            predicted_batched.append(np.empty((0, 24)))
            gt_batched.append(np.empty((0, 24)))
            continue

        # Extract assignment indices
        assignment_pair = assignments['assignments'][b]
        if len(assignment_pair) != 2:
            # Invalid assignment format
            predicted_batched.append(np.empty((0, 24)))
            gt_batched.append(np.empty((0, 24)))
            continue

        pred_idx, gt_idx = assignment_pair[0], assignment_pair[1]

        pred_tensor = pred_boxes[pred_key][b]  # [N_pred, 8, 3]
        gt_tensor = gt_bbox[b]                 # [N_gt, 8, 3]

        # Validate tensor shapes
        if pred_tensor.shape[0] == 0 or gt_tensor.shape[0] == 0:
            predicted_batched.append(np.empty((0, 24)))
            gt_batched.append(np.empty((0, 24)))
            continue

        # Convert to CPU and validate indices
        pred_idx_cpu = pred_idx.cpu()
        gt_idx_cpu = gt_idx.cpu()

        # Filter valid indices (within bounds)
        max_pred = pred_tensor.shape[0] - 1
        max_gt = gt_tensor.shape[0] - 1

        valid_pred_mask = (pred_idx_cpu >= 0) & (pred_idx_cpu <= max_pred)
        valid_gt_mask = (gt_idx_cpu >= 0) & (gt_idx_cpu <= max_gt)
        valid_mask = valid_pred_mask & valid_gt_mask

        if valid_mask.sum() > 0:
            pred_idx_valid = pred_idx_cpu[valid_mask]
            gt_idx_valid = gt_idx_cpu[valid_mask]

            # Extract matched boxes and reshape to [num_matches, 24]
            matched_preds = pred_tensor[pred_idx_valid].reshape(-1, 24)  # [num, 24]
            matched_gts = gt_tensor[gt_idx_valid].reshape(-1, 24)

            predicted_batched.append(matched_preds.detach().cpu().numpy())
            gt_batched.append(matched_gts.cpu().numpy())
        else:
            # No valid matches
            predicted_batched.append(np.empty((0, 24)))
            gt_batched.append(np.empty((0, 24)))

    return predicted_batched, gt_batched


if __name__ == '__main__':
    args, config = parse_option()

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    if config.amp_opt_level:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

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
