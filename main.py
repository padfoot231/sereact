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
import numpy.typing as npt
from matplotlib import pyplot as plt

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

# Suppress DDP reducer bucket warnings for cleaner logs
import logging
logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)

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
from utils.visualize_point_cloud import (
    visualize_gui_pointcloud_with_bounding_boxes,
    visualize_bounding_boxes
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
    parser.add_argument('--visualize-point-cloud', action='store_true', help='Enable point cloud visualization during evaluation')
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
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.local_rank],
        find_unused_parameters=False,    # Fix for "Reducer buckets rebuilt" warning
        broadcast_buffers=False
    )
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
        miou, _ = validate(loss_module, 0, iou_evaluator, data_loader_val, model, config)
        logger.info(f"Mean iou of the network on the {len(dataset_val)} test images: {miou:.4f}")
        if config.eval_mode:
            return
        
    if config.model.pretrained and (not config.model.resume):
        load_pretrained(config, model_without_ddp, logger)
        miou, _ = validate(loss_module, 0, iou_evaluator, data_loader_val, model, config)
        logger.info(f"Mean iou of the network on the {len(dataset_val)} test images: {miou:.4f}")

        # Model export to low precision formats
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.train.start_epoch, config.train.max_epoch):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, loss_module, iou_evaluator, data_loader_train, optimizer, epoch,scheduler,
                        loss_scaler)

        miou, loss = validate(loss_module, epoch, iou_evaluator, data_loader_val, model, config)
        
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
            train_metrics = iou_evaluator.compute_metrics()
            metric = train_metrics['mean_iou']
            iou_25_accuracy = train_metrics['threshold_accuracy'].get(0.25, 0.0)
            logger.info(
                f'Train: [{epoch}/{config.train.max_epoch}][{batch_idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'miou {metric:.4f}\t'
                f'IoU@0.25 {iou_25_accuracy:.4f} ({iou_25_accuracy*100:.1f}%)\t'
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
    model: torch.nn.Module,
    config: Any = None
) -> Tuple[float, float]:
    """Validate model on validation dataset and return loss and IoU metrics."""
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    iou_evaluator.reset()

    # For visualization during evaluation only
    all_predicted_boxes = []
    all_gt_boxes = []

    # For point cloud visualization during evaluation only
    visualization_data = []
    max_visualizations = 5# Limit number of visualizations

    logger.info('Starting validation...')

    for batch_idx, batch_data in enumerate(data_loader):
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

            # Collect boxes for visualization during evaluation only
            if config and config.eval_mode:
                for pred_batch, gt_batch in zip(predicted_bboxes_matched, gt_bboxes_matched):
                    if pred_batch.size > 0 and gt_batch.size > 0:
                        all_predicted_boxes.append(pred_batch)
                        all_gt_boxes.append(gt_batch)

                # Collect point cloud data for visualization if enabled
                if config.visualize_point_cloud and len(visualization_data) < max_visualizations:
                    # Store first sample from batch for visualization
                    sample_data = {
                        'point_cloud': inputs[0].detach().cpu().numpy(),  # First sample
                        'rgb_image': inputs_rgb[0].detach().cpu(),        # First RGB sample
                        'predicted_boxes': predicted_bboxes_matched[0] if len(predicted_bboxes_matched) > 0 else np.array([]),
                        'gt_boxes': gt_bboxes_matched[0] if len(gt_bboxes_matched) > 0 else np.array([]),
                        'batch_idx': batch_idx
                    }
                    visualization_data.append(sample_data)

        iou_evaluator.update(predicted_bboxes_matched, gt_bboxes_matched)

        batch_time.update(time.time() - end)
        end = time.time()

    # Get comprehensive metrics including threshold accuracies
    full_metrics = iou_evaluator.compute_metrics()
    mean_iou = full_metrics['mean_iou']
    threshold_accuracies = full_metrics['threshold_accuracy']

    # Log detailed metrics
    logger.info(f"Validation Results:")
    logger.info(f"  Mean IoU: {mean_iou:.4f}")
    breakpoint()
    for threshold, accuracy in threshold_accuracies.items():
        logger.info(f"  IoU@{threshold}: {accuracy:.4f} ({accuracy*100:.2f}% correct predictions)")

    # Log to wandb with detailed metrics
    wandb_metrics = {
        'epoch': epoch,
        'val_loss': loss_meter.avg,
        'val_miou': mean_iou,
    }

    wandb.log(wandb_metrics)

    # Generate box distribution visualization during evaluation only
    if config and config.eval_mode and all_predicted_boxes and all_gt_boxes:
        logger.info('Generating bounding box distribution visualization...')
        try:
            # Concatenate all collected boxes
            all_pred = np.concatenate(all_predicted_boxes, axis=0)
            all_gt = np.concatenate(all_gt_boxes, axis=0)

            # Reshape from (N, 24) to (N, 8, 3) if needed
            if all_pred.shape[1] == 24:
                all_pred = all_pred.reshape(-1, 8, 3)
            if all_gt.shape[1] == 24:
                all_gt = all_gt.reshape(-1, 8, 3)

            # Call visualization function
            visualize_box_distributions(all_pred, all_gt)
            logger.info('Box distribution visualization saved successfully!')

        except Exception as e:
            logger.warning(f'Failed to generate box distribution visualization: {e}')

    # Generate point cloud visualizations during evaluation only
    if config and config.eval_mode and config.visualize_point_cloud and visualization_data:
        logger.info('Generating point cloud visualizations...')
        try:
            for i, vis_data in enumerate(visualization_data):
                logger.info(f'Visualizing sample {i+1}/{len(visualization_data)} from batch {vis_data["batch_idx"]}')

                # Reshape point cloud if needed
                point_cloud = vis_data['point_cloud']
                if len(point_cloud.shape) == 3:
                    point_cloud = point_cloud.transpose(1, 2, 0).reshape(-1, 3)

                # Get bounding boxes
                pred_boxes = vis_data['predicted_boxes']
                gt_boxes = vis_data['gt_boxes']

                # Only visualize if we have both predicted and GT boxes
                if pred_boxes.size > 0 and gt_boxes.size > 0:
                    # Reshape boxes if needed (from (N, 24) to (N, 8, 3))
                    if len(pred_boxes.shape) == 2 and pred_boxes.shape[1] == 24:
                        pred_boxes = pred_boxes.reshape(-1, 8, 3)
                    if len(gt_boxes.shape) == 2 and gt_boxes.shape[1] == 24:
                        gt_boxes = gt_boxes.reshape(-1, 8, 3)

                    # Save point cloud visualization as image
                    save_path = f"point_cloud_sample_{i+1}_batch_{vis_data['batch_idx']}.png"
                    save_point_cloud_visualization(
                        point_cloud=point_cloud,
                        rgb_tensor=vis_data['rgb_image'],
                        predicted_boxes=pred_boxes,
                        gt_boxes=gt_boxes,
                        save_path=save_path
                    )
                else:
                    logger.info(f'Skipping visualization for sample {i+1} - no valid boxes')

            logger.info('Point cloud visualizations completed!')

        except Exception as e:
            logger.warning(f'Failed to generate point cloud visualizations: {e}')

    return mean_iou, loss_meter.avg


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

def save_point_cloud_visualization(
        point_cloud: npt.NDArray,
        rgb_tensor: torch.Tensor,
        predicted_boxes: npt.NDArray,
        gt_boxes: npt.NDArray,
        save_path: str = "point_cloud_visualization.png"
    ) -> None:
        """
        Save point cloud visualization with bounding boxes as image file using matplotlib.
        Truly headless - no display system required.

        Args:
            point_cloud: (N, 3) point cloud coordinates
            rgb_tensor: (3, H, W) RGB tensor for coloring
            predicted_boxes: (M, 8, 3) predicted bounding box corners
            gt_boxes: (K, 8, 3) ground truth bounding box corners
            save_path: Path to save the visualization image
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt

            # Create figure with 3D subplot
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

            # Plot point cloud
            if point_cloud.size > 0:
                # Subsample points for better visualization performance
                max_points = 5000
                if point_cloud.shape[0] > max_points:
                    indices = np.random.choice(point_cloud.shape[0], max_points, replace=False)
                    pc_vis = point_cloud[indices]
                else:
                    pc_vis = point_cloud

                # Get RGB colors if available
                if rgb_tensor is not None:
                    rgb_points = rgb_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    rgb_points = rgb_points.reshape(-1, 3)
                    if rgb_points.shape[0] == point_cloud.shape[0]:
                        if point_cloud.shape[0] > max_points:
                            colors = rgb_points[indices]
                        else:
                            colors = rgb_points
                        ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
                                 c=colors, s=1, alpha=0.6)
                    else:
                        ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
                                 c='lightblue', s=1, alpha=0.6)
                else:
                    ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
                             c='lightblue', s=1, alpha=0.6)

            # Function to draw a 3D bounding box
            def draw_bbox(corners, color):
                if corners.size == 0:
                    return

                # Define the 12 edges of a cube
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
                ]

                for edge in edges:
                    points = corners[edge]
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                             color=color, linewidth=2, alpha=0.8)

            # Draw predicted bounding boxes (red)
            for box_corners in predicted_boxes:
                if box_corners.size > 0:
                    draw_bbox(box_corners, 'red')

            # Draw ground truth bounding boxes (green)
            for box_corners in gt_boxes:
                if box_corners.size > 0:
                    draw_bbox(box_corners, 'green')

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Point Cloud with Bounding Boxes\n(Red: Predicted, Green: Ground Truth)')

            # Set equal aspect ratio
            if point_cloud.size > 0:
                max_range = np.array([
                    point_cloud[:, 0].max() - point_cloud[:, 0].min(),
                    point_cloud[:, 1].max() - point_cloud[:, 1].min(),
                    point_cloud[:, 2].max() - point_cloud[:, 2].min()
                ]).max() / 2.0

                mid_x = (point_cloud[:, 0].max() + point_cloud[:, 0].min()) * 0.5
                mid_y = (point_cloud[:, 1].max() + point_cloud[:, 1].min()) * 0.5
                mid_z = (point_cloud[:, 2].max() + point_cloud[:, 2].min()) * 0.5

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label='Predicted Boxes'),
                Line2D([0], [0], color='green', lw=2, label='Ground Truth Boxes'),
                Line2D([0], [0], marker='o', color='lightblue', lw=0,
                       markersize=5, label='Point Cloud')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            # Save the figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Point cloud visualization saved to: {save_path}")

            # Log to wandb if available
            try:
                wandb.log({f'point_cloud_viz_{save_path.split("/")[-1]}': wandb.Image(save_path)})
            except:
                pass  # Ignore wandb errors

        except Exception as e:
            logger.warning(f"Failed to save point cloud visualization: {e}")
            # Create a simple text summary as fallback
            try:
                summary = f"""
Point Cloud Visualization Summary
================================
Point Cloud: {point_cloud.shape[0]} points
Predicted Boxes: {len(predicted_boxes)}
Ground Truth Boxes: {len(gt_boxes)}

Point Cloud Range:
  X: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]
  Y: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]
  Z: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]

Predicted Box Centers:
{[f"  Box {i}: [{box.mean(axis=0)[0]:.3f}, {box.mean(axis=0)[1]:.3f}, {box.mean(axis=0)[2]:.3f}]" for i, box in enumerate(predicted_boxes) if box.size > 0]}

Ground Truth Box Centers:
{[f"  Box {i}: [{box.mean(axis=0)[0]:.3f}, {box.mean(axis=0)[1]:.3f}, {box.mean(axis=0)[2]:.3f}]" for i, box in enumerate(gt_boxes) if box.size > 0]}
"""
                with open(save_path.replace('.png', '_summary.txt'), 'w') as f:
                    f.write(summary)
                logger.info(f"Saved visualization summary to: {save_path.replace('.png', '_summary.txt')}")
            except:
                pass


def visualize_box_distributions(
        predicted_bboxes: npt.NDArray, gt_bboxes: npt.NDArray
    ) -> None:
        """Visualize the distribution of predicted vs ground truth box sizes.

        Args:
            predicted_bboxes: shape (num_boxes, 8, 3) - predicted box corners
            gt_bboxes: shape (num_boxes, 8, 3) - ground truth box corners
        """

        # Compute box dimensions
        def get_box_dims(boxes: npt.NDArray) -> tuple:
            # Reshape to (num_boxes, 8, 3)
            boxes = boxes.reshape(-1, 8, 3)
            # Get min and max coordinates
            mins = boxes.min(axis=1)  # (num_boxes, 3)
            maxs = boxes.max(axis=1)  # (num_boxes, 3)
            # Compute lengths along each dimension
            dims = maxs - mins  # (num_boxes, 3)
            # Compute volumes
            volumes = dims.prod(axis=1)  # (num_boxes,)
            return dims, volumes

        pred_dims, pred_volumes = get_box_dims(predicted_bboxes)
        gt_dims, gt_volumes = get_box_dims(gt_bboxes)

        # Create figure with multiple subplots
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot volume distributions
        ax1.hist(pred_volumes, bins=20, alpha=0.5, label='Predicted', color='red')
        ax1.hist(gt_volumes, bins=20, alpha=0.5, label='Ground Truth', color='blue')
        ax1.set_title('Box Volumes Distribution')
        ax1.set_xlabel('Volume')
        ax1.set_ylabel('Count')
        ax1.legend()

        # Plot dimension distributions
        dimensions = ['X', 'Y', 'Z']
        for i, dim in enumerate(dimensions):
            ax2.hist(pred_dims[:, i], bins=20, alpha=0.5, label=f'Pred {dim}', color=f'C{i}')
            ax2.hist(
                gt_dims[:, i], bins=20, alpha=0.5, label=f'GT {dim}', linestyle='--', color=f'C{i}'
            )
        ax2.set_title('Box Dimensions Distribution')
        ax2.set_xlabel('Length')
        ax2.set_ylabel('Count')
        ax2.legend()

        # Plot dimension ratios
        pred_ratios = pred_dims / pred_dims.max(axis=1, keepdims=True)
        gt_ratios = gt_dims / gt_dims.max(axis=1, keepdims=True)

        for i, dim in enumerate(dimensions):
            ax3.hist(pred_ratios[:, i], bins=20, alpha=0.5, label=f'Pred {dim}', color=f'C{i}')
            ax3.hist(
                gt_ratios[:, i],
                bins=20,
                alpha=0.5,
                label=f'GT {dim}',
                linestyle='--',
                color=f'C{i}',
            )
        ax3.set_title('Box Dimension Ratios')
        ax3.set_xlabel('Ratio to Largest Dimension')
        ax3.set_ylabel('Count')
        ax3.legend()

        # Plot centers
        pred_centers = predicted_bboxes.reshape(-1, 8, 3).mean(axis=1)
        gt_centers = gt_bboxes.reshape(-1, 8, 3).mean(axis=1)

        for i, dim in enumerate(dimensions):
            ax4.hist(pred_centers[:, i], bins=20, alpha=0.5, label=f'Pred {dim}', color=f'C{i}')
            ax4.hist(
                gt_centers[:, i],
                bins=20,
                alpha=0.5,
                label=f'GT {dim}',
                linestyle='--',
                color=f'C{i}',
            )
        ax4.set_title('Box Center Positions')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Count')
        ax4.legend()

        plt.tight_layout()

        # Save the plot
        plt.savefig('box_distributions.png')
        plt.close()

        # Log to wandb
        wandb.log({'box_distributions': wandb.Image('box_distributions.png')})



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
