"""
Adaptive Loss Weight Scheduler
Dynamically adjusts loss weights based on training progress and box prediction quality.
"""

import torch
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AdaptiveLossScheduler:
    """
    Scheduler that adapts loss weights based on validation metrics and training progress.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_frequency: int = 10,
        small_box_threshold: float = 0.1,
        aspect_ratio_threshold: float = 0.3,
        volume_error_threshold: float = 0.5
    ):
        """
        Args:
            initial_weights: Initial loss weights
            adaptation_frequency: How often to adapt weights (in epochs)
            small_box_threshold: Volume threshold for "small" boxes
            aspect_ratio_threshold: Threshold for aspect ratio errors
            volume_error_threshold: Threshold for volume prediction errors
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.adaptation_frequency = adaptation_frequency
        self.small_box_threshold = small_box_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.volume_error_threshold = volume_error_threshold
        
        # Tracking metrics
        self.metrics_history = []
        self.last_adaptation_epoch = 0
    
    def compute_box_quality_metrics(
        self,
        predicted_boxes: List[np.ndarray],
        gt_boxes: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute quality metrics for box predictions.
        
        Args:
            predicted_boxes: List of predicted box arrays (N, 8, 3)
            gt_boxes: List of ground truth box arrays (N, 8, 3)
            
        Returns:
            Dictionary of quality metrics
        """
        if not predicted_boxes or not gt_boxes:
            return {
                'small_box_error': 0.0,
                'aspect_ratio_error': 0.0,
                'volume_error': 0.0,
                'center_error': 0.0
            }
        
        # Concatenate all boxes
        all_pred = np.concatenate(predicted_boxes, axis=0)
        all_gt = np.concatenate(gt_boxes, axis=0)
        
        if all_pred.shape[1] == 24:
            all_pred = all_pred.reshape(-1, 8, 3)
        if all_gt.shape[1] == 24:
            all_gt = all_gt.reshape(-1, 8, 3)
        
        # Compute box properties
        def get_box_properties(boxes):
            mins = boxes.min(axis=1)  # (N, 3)
            maxs = boxes.max(axis=1)  # (N, 3)
            dims = maxs - mins  # (N, 3)
            volumes = dims.prod(axis=1)  # (N,)
            centers = (mins + maxs) / 2  # (N, 3)
            
            # Aspect ratios
            max_dims = dims.max(axis=1, keepdims=True)
            aspect_ratios = dims / (max_dims + 1e-6)
            
            return dims, volumes, centers, aspect_ratios
        
        pred_dims, pred_volumes, pred_centers, pred_aspects = get_box_properties(all_pred)
        gt_dims, gt_volumes, gt_centers, gt_aspects = get_box_properties(all_gt)
        
        # Identify small boxes
        small_box_mask = gt_volumes < self.small_box_threshold
        
        # Compute metrics
        metrics = {}
        
        # Small box volume error
        if small_box_mask.sum() > 0:
            small_vol_error = np.abs(
                np.log(pred_volumes[small_box_mask] + 1e-6) - 
                np.log(gt_volumes[small_box_mask] + 1e-6)
            ).mean()
            metrics['small_box_error'] = small_vol_error
        else:
            metrics['small_box_error'] = 0.0
        
        # Overall volume error
        vol_error = np.abs(
            np.log(pred_volumes + 1e-6) - np.log(gt_volumes + 1e-6)
        ).mean()
        metrics['volume_error'] = vol_error
        
        # Aspect ratio error
        aspect_error = np.abs(pred_aspects - gt_aspects).mean()
        metrics['aspect_ratio_error'] = aspect_error
        
        # Center error
        center_error = np.linalg.norm(pred_centers - gt_centers, axis=1).mean()
        metrics['center_error'] = center_error
        
        return metrics
    
    def should_adapt(self, epoch: int) -> bool:
        """Check if weights should be adapted at this epoch."""
        return (epoch - self.last_adaptation_epoch) >= self.adaptation_frequency
    
    def adapt_weights(
        self,
        epoch: int,
        predicted_boxes: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        current_loss_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adapt loss weights based on current performance.
        
        Args:
            epoch: Current epoch
            predicted_boxes: Predicted boxes from validation
            gt_boxes: Ground truth boxes from validation
            current_loss_dict: Current loss values
            
        Returns:
            Updated loss weights
        """
        if not self.should_adapt(epoch):
            return self.current_weights
        
        # Compute quality metrics
        metrics = self.compute_box_quality_metrics(predicted_boxes, gt_boxes)
        self.metrics_history.append(metrics)
        
        logger.info(f"Epoch {epoch} - Box Quality Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Adapt weights based on metrics
        new_weights = self.current_weights.copy()
        
        # Increase aspect ratio loss if aspect ratio error is high
        if metrics['aspect_ratio_error'] > self.aspect_ratio_threshold:
            new_weights['loss_aspect_ratio_weight'] = min(
                new_weights['loss_aspect_ratio_weight'] * 1.2, 2.0
            )
            logger.info(f"Increased aspect ratio loss weight to {new_weights['loss_aspect_ratio_weight']:.3f}")
        
        # Increase volume-aware loss if small box error is high
        if metrics['small_box_error'] > self.volume_error_threshold:
            new_weights['loss_volume_aware_weight'] = min(
                new_weights['loss_volume_aware_weight'] * 1.3, 1.5
            )
            logger.info(f"Increased volume-aware loss weight to {new_weights['loss_volume_aware_weight']:.3f}")
        
        # Increase size regularization if overall volume error is high
        if metrics['volume_error'] > self.volume_error_threshold:
            new_weights['loss_size_reg_weight'] = min(
                new_weights['loss_size_reg_weight'] * 1.1, 2.0
            )
            logger.info(f"Increased size regularization weight to {new_weights['loss_size_reg_weight']:.3f}")
        
        # Decrease weights if performance is good
        if len(self.metrics_history) >= 3:
            recent_metrics = self.metrics_history[-3:]
            
            # Check if aspect ratio error is consistently low
            if all(m['aspect_ratio_error'] < self.aspect_ratio_threshold * 0.7 for m in recent_metrics):
                new_weights['loss_aspect_ratio_weight'] = max(
                    new_weights['loss_aspect_ratio_weight'] * 0.9, 0.1
                )
                logger.info(f"Decreased aspect ratio loss weight to {new_weights['loss_aspect_ratio_weight']:.3f}")
            
            # Check if small box error is consistently low
            if all(m['small_box_error'] < self.volume_error_threshold * 0.7 for m in recent_metrics):
                new_weights['loss_volume_aware_weight'] = max(
                    new_weights['loss_volume_aware_weight'] * 0.9, 0.1
                )
                logger.info(f"Decreased volume-aware loss weight to {new_weights['loss_volume_aware_weight']:.3f}")
        
        self.current_weights = new_weights
        self.last_adaptation_epoch = epoch
        
        return new_weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.current_weights.copy()
    
    def reset_to_initial(self):
        """Reset weights to initial values."""
        self.current_weights = self.initial_weights.copy()
        self.metrics_history = []
        self.last_adaptation_epoch = 0
        logger.info("Reset loss weights to initial values")


def create_adaptive_scheduler(config: Any) -> AdaptiveLossScheduler:
    """
    Create adaptive loss scheduler from config.
    
    Args:
        config: Configuration object
        
    Returns:
        AdaptiveLossScheduler instance
    """
    initial_weights = {
        'loss_giou_weight': config.loss.weights.giou,
        'loss_box_corners_weight': config.loss.weights.box_corners,
        'loss_size_weight': config.loss.weights.size,
        'loss_size_reg_weight': config.loss.weights.size_reg,
        'loss_angle_cls_weight': config.loss.weights.angle_cls,
        'loss_angle_reg_weight': config.loss.weights.angle_reg,
        'loss_aspect_ratio_weight': getattr(config.loss.weights, 'aspect_ratio', 0.5),
        'loss_volume_aware_weight': getattr(config.loss.weights, 'volume_aware', 0.3),
    }
    
    return AdaptiveLossScheduler(
        initial_weights=initial_weights,
        adaptation_frequency=getattr(config.train, 'loss_adaptation_frequency', 10),
        small_box_threshold=getattr(config.loss, 'small_box_threshold', 0.1),
        aspect_ratio_threshold=getattr(config.loss, 'aspect_ratio_threshold', 0.3),
        volume_error_threshold=getattr(config.loss, 'volume_error_threshold', 0.5)
    )
