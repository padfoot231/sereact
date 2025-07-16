"""
Box-Aware Augmentations for Improving Small Box Detection
Addresses issues with under-prediction of small volumes and shape diversity.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
import random


class BoxAwareAugmentation:
    """
    Augmentation strategies specifically designed to improve box prediction quality.
    """
    
    def __init__(
        self,
        small_box_boost_prob: float = 0.3,
        aspect_ratio_aug_prob: float = 0.4,
        volume_scaling_prob: float = 0.2,
        min_box_size: float = 0.1,
        max_scale_factor: float = 1.5
    ):
        """
        Args:
            small_box_boost_prob: Probability of applying small box boosting
            aspect_ratio_aug_prob: Probability of aspect ratio augmentation
            volume_scaling_prob: Probability of volume scaling augmentation
            min_box_size: Minimum box size threshold for "small" boxes
            max_scale_factor: Maximum scaling factor for augmentations
        """
        self.small_box_boost_prob = small_box_boost_prob
        self.aspect_ratio_aug_prob = aspect_ratio_aug_prob
        self.volume_scaling_prob = volume_scaling_prob
        self.min_box_size = min_box_size
        self.max_scale_factor = max_scale_factor
    
    def get_box_properties(self, bbox_corners: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract box properties from corner coordinates.
        
        Args:
            bbox_corners: (N, 8, 3) box corners
            
        Returns:
            Dictionary with box properties
        """
        # Get min/max coordinates
        mins = bbox_corners.min(dim=1)[0]  # (N, 3)
        maxs = bbox_corners.max(dim=1)[0]  # (N, 3)
        
        # Compute dimensions and properties
        dims = maxs - mins  # (N, 3)
        volumes = dims.prod(dim=1)  # (N,)
        centers = (mins + maxs) / 2  # (N, 3)
        
        # Aspect ratios (normalized by largest dimension)
        max_dims = dims.max(dim=1, keepdim=True)[0]
        aspect_ratios = dims / (max_dims + 1e-6)  # (N, 3)
        
        return {
            'dims': dims,
            'volumes': volumes,
            'centers': centers,
            'aspect_ratios': aspect_ratios,
            'mins': mins,
            'maxs': maxs
        }
    
    def small_box_boost(
        self, 
        point_cloud: torch.Tensor, 
        bbox_corners: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Boost representation of small boxes by duplicating/emphasizing their points.
        
        Args:
            point_cloud: (N_points, 3) point cloud
            bbox_corners: (N_boxes, 8, 3) bounding box corners
            
        Returns:
            Augmented point cloud and bounding boxes
        """
        if random.random() > self.small_box_boost_prob:
            return point_cloud, bbox_corners
        
        box_props = self.get_box_properties(bbox_corners)
        volumes = box_props['volumes']
        
        # Identify small boxes
        small_box_mask = volumes < self.min_box_size
        
        if not small_box_mask.any():
            return point_cloud, bbox_corners
        
        # For each small box, duplicate points within it
        augmented_points = [point_cloud]
        
        for i, is_small in enumerate(small_box_mask):
            if is_small:
                box_min = box_props['mins'][i]
                box_max = box_props['maxs'][i]
                
                # Find points inside this box
                inside_mask = (
                    (point_cloud >= box_min).all(dim=1) & 
                    (point_cloud <= box_max).all(dim=1)
                )
                
                if inside_mask.sum() > 0:
                    # Duplicate points with small noise
                    inside_points = point_cloud[inside_mask]
                    noise = torch.randn_like(inside_points) * 0.01
                    augmented_points.append(inside_points + noise)
        
        # Concatenate all points
        if len(augmented_points) > 1:
            point_cloud = torch.cat(augmented_points, dim=0)
        
        return point_cloud, bbox_corners
    
    def aspect_ratio_augmentation(
        self, 
        bbox_corners: torch.Tensor
    ) -> torch.Tensor:
        """
        Augment bounding boxes to increase aspect ratio diversity.
        
        Args:
            bbox_corners: (N_boxes, 8, 3) bounding box corners
            
        Returns:
            Augmented bounding box corners
        """
        if random.random() > self.aspect_ratio_aug_prob:
            return bbox_corners
        
        box_props = self.get_box_properties(bbox_corners)
        centers = box_props['centers']
        dims = box_props['dims']
        
        # Apply random scaling to each dimension independently
        scale_factors = torch.rand(dims.shape) * (self.max_scale_factor - 0.8) + 0.8
        new_dims = dims * scale_factors
        
        # Reconstruct corners from center and new dimensions
        half_dims = new_dims / 2
        
        # Create new corners
        new_corners = torch.zeros_like(bbox_corners)
        corner_offsets = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=bbox_corners.dtype, device=bbox_corners.device)
        
        for i in range(len(bbox_corners)):
            for j, offset in enumerate(corner_offsets):
                new_corners[i, j] = centers[i] + offset * half_dims[i]
        
        return new_corners
    
    def volume_scaling_augmentation(
        self,
        point_cloud: torch.Tensor,
        bbox_corners: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply volume-aware scaling to improve small volume prediction.
        
        Args:
            point_cloud: (N_points, 3) point cloud
            bbox_corners: (N_boxes, 8, 3) bounding box corners
            
        Returns:
            Augmented point cloud and bounding boxes
        """
        if random.random() > self.volume_scaling_prob:
            return point_cloud, bbox_corners
        
        # Apply global scaling with bias towards smaller scales
        scale_factor = random.uniform(0.8, 1.2)
        
        # Scale point cloud
        point_cloud = point_cloud * scale_factor
        
        # Scale bounding boxes
        bbox_corners = bbox_corners * scale_factor
        
        return point_cloud, bbox_corners
    
    def __call__(
        self, 
        point_cloud: torch.Tensor, 
        bbox_corners: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all augmentations.
        
        Args:
            point_cloud: (N_points, 3) point cloud
            bbox_corners: (N_boxes, 8, 3) bounding box corners
            
        Returns:
            Augmented point cloud and bounding boxes
        """
        # Apply augmentations in sequence
        point_cloud, bbox_corners = self.small_box_boost(point_cloud, bbox_corners)
        bbox_corners = self.aspect_ratio_augmentation(bbox_corners)
        point_cloud, bbox_corners = self.volume_scaling_augmentation(point_cloud, bbox_corners)
        
        return point_cloud, bbox_corners


def create_box_aware_augmentation(config: Any) -> BoxAwareAugmentation:
    """
    Factory function to create box-aware augmentation from config.
    
    Args:
        config: Configuration object
        
    Returns:
        BoxAwareAugmentation instance
    """
    return BoxAwareAugmentation(
        small_box_boost_prob=getattr(config.data, 'small_box_boost_prob', 0.3),
        aspect_ratio_aug_prob=getattr(config.data, 'aspect_ratio_aug_prob', 0.4),
        volume_scaling_prob=getattr(config.data, 'volume_scaling_prob', 0.2),
        min_box_size=getattr(config.data, 'min_box_size', 0.1),
        max_scale_factor=getattr(config.data, 'max_scale_factor', 1.5)
    )
