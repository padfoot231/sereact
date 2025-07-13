import numpy as np
from typing import Optional, Tuple, List
import torch



def check_aspect_torch(crop_range: torch.Tensor, aspect_min: float) -> bool:
    xy_aspect = torch.min(crop_range[:2]) / torch.max(crop_range[:2])
    xz_aspect = torch.min(crop_range[[0, 2]]) / torch.max(crop_range[[0, 2]])
    yz_aspect = torch.min(crop_range[1:]) / torch.max(crop_range[1:])
    return (xy_aspect >= aspect_min) or (xz_aspect >= aspect_min) or (yz_aspect >= aspect_min)


class RandomCuboid:
    def __init__(
        self,
        min_points: int,
        aspect: float = 0.8,
        min_crop: float = 0.5,
        max_crop: float = 1.0,
        box_filter_policy: str = 'center',
    ) -> None:
        self.aspect = aspect
        self.min_crop = min_crop
        self.max_crop = max_crop
        self.min_points = min_points
        self.box_filter_policy = box_filter_policy

    def __call__(
        self,
        point_cloud: torch.Tensor,        # shape: (N, 3)
        target_boxes: torch.Tensor,       # shape: (B, 8, 3)
        per_point_labels: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:

        assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, f"Expected shape (N, 3), got {point_cloud.shape}"
        assert target_boxes.ndim == 3 and target_boxes.shape[2] == 3, f"Expected shape (B, 8, 3), got {target_boxes.shape}"

        range_xyz = torch.max(point_cloud, dim=0).values - torch.min(point_cloud, dim=0).values

        for _ in range(100):
            # Generate random crop range
            crop_range = torch.FloatTensor(3).uniform_(self.min_crop, self.max_crop)
            if not check_aspect_torch(crop_range, self.aspect):
                continue

            # Random center
            idx = torch.randint(0, point_cloud.shape[0], (1,))
            sample_center = point_cloud[idx].squeeze(0)

            new_range = range_xyz * crop_range / 2.0
            min_xyz = sample_center - new_range
            max_xyz = sample_center + new_range

            mask_upper = (point_cloud <= max_xyz).all(dim=1)
            mask_lower = (point_cloud >= min_xyz).all(dim=1)
            valid_points = mask_upper & mask_lower

            if valid_points.sum() < self.min_points:
                continue

            new_point_cloud = point_cloud[valid_points]

            # Box filtering
            if self.box_filter_policy == 'center':
                new_boxes = target_boxes
                if target_boxes.numel() > 0:
                    box_centers = target_boxes.mean(dim=1)  # shape: (B, 3)
                    keep_boxes = (
                        (box_centers >= min_xyz.unsqueeze(0)) &
                        (box_centers <= max_xyz.unsqueeze(0))
                    ).all(dim=1)

                    if keep_boxes.sum() == 0:
                        continue

                    new_boxes = target_boxes[keep_boxes]

                new_per_point_labels = (
                    [lbl[valid_points] for lbl in per_point_labels] if per_point_labels else None
                )

                return new_point_cloud, new_boxes, new_per_point_labels

        return point_cloud, target_boxes, per_point_labels


class SereactAugmentation:
    def __init__(self, use_random_cuboid=True):
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid = RandomCuboid(min_points=30000, aspect=0.75, min_crop=0.5, max_crop=1.0)


    def __call__(
            self, pcd: np.ndarray, bbox_3d: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Applies augmentations to the point cloud and bounding boxes.

            Args:
                pcd (np.ndarray): Input point cloud data (N, 3).
                bbox_3d (np.ndarray): Input bounding box data (K, 8, 3).

            Returns:
                tuple: Augmented point cloud and bounding boxes.
            """

            # First augmentation: Flip along X-axis
            # breakpoint()
            if np.random.rand() > 0.5:
            # print(pcd.shape)
            # if True:
            # breakpoint()
            # if True:
                pcd[:, 0] = -pcd[:, 0]
                bbox_3d[:, :, 0] = -bbox_3d[:, :, 0]

            # Second augmentation: Rotate around Z-axis
            if np.random.rand() > 0.5:
            # if True:
            # if True:
                # Rotate -30° to +30°
                angle = (np.random.rand() * np.pi / 3) - np.pi / 6
                rotation_matrix = torch.tensor([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle),  np.cos(angle), 0],
                    [0,              0,             1]
                ], dtype=torch.float32, device=pcd.device)  # keep on same device as input

                # Apply rotation
                pcd = pcd @ rotation_matrix.T  # shape: (N, 3)

                # Rotate bbox
                bbox_3d = bbox_3d.reshape(-1, 3) @ rotation_matrix.T
                bbox_3d = bbox_3d.reshape(-1, 8, 3)
            # Third augmentation: Scale the point cloud
            scale_factor = np.random.uniform(0.8, 1.2)
            pcd = pcd * scale_factor
            bbox_3d = bbox_3d * scale_factor

            # Add Gaussian jitter (mean=0, std=0.01)
            jitter = torch.normal(
                mean=0.0,
                std=0.01,
                size=pcd.shape,
                dtype=pcd.dtype,
                device=pcd.device
            )
            pcd = pcd + jitter
            # breakpoint()
            # Fifth augmentation: Use the RandomCuboid augmentation from DepthContrast
            if np.random.rand() > 0.5:
            # if True:
                pcd, bbox_3d, _ = self.random_cuboid(pcd, bbox_3d)

            return pcd, bbox_3d

