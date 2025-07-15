"""
Python file that contains some miscellaneous functions that for augmentation and model training.
"""

from typing import List, Union
import torch.nn.functional as F
import numpy as np
import torch


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize the random seed for a worker process.

    This function is used to ensure that each worker process has a different random seed,
    which is important for data loading in parallel.

    Args:
        worker_id (int): The ID of the worker process.
    """
    # Set the random seed for numpy based on the current random state and worker ID
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz


# Link: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/bba1f6156371fbabf02bf4c47062dfde21a32b46/log/classification/pointnet2_ssg_wo_normals/pointnet2_utils.py#L63
# Discussion: https://github.com/rusty1s/pytorch_cluster/issues/102#issuecomment-834017428
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # Initialize centroids
    distance = torch.ones(B, N).to(device) * 1e10  # Initialize distances to a large value
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(
        device
    )  # Randomly select the first point
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # Batch indices for indexing

    for i in range(npoint):
        centroids[:, i] = farthest  # Assign the farthest point as a centroid
        centroid = xyz[batch_indices, farthest, :].view(
            B, 1, 3
        )  # Get the coordinates of the farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)  # Compute squared distances from the centroid
        mask = dist < distance  # Find points closer than the current farthest distance
        distance[mask] = dist[mask]  # Update distances
        farthest = torch.max(distance, -1)[1]  # Select the next farthest point

    return centroids

def pad_rgb_to_size(img, target_h=600, target_w=600):
    """
    Pads a 3D tensor [C, H, W] to [C, target_h, target_w] using zero-padding.
    """
    _, h, w = img.shape
    pad_h = target_h - h
    pad_w = target_w - w

    # Padding format: (left, right, top, bottom)
    pad = [0, pad_w, 0, pad_h]
    return F.pad(img, pad, mode='constant', value=0)

def collate_variable_3d_batch_nomask(batch):
    """
    Collate function with RGB image padding to [3, 600, 600].
    """
    pcds = [item['pcd_tensor'] for item in batch]
    bboxes = [item['bbox3d_tensor'] for item in batch]
    pcd_min = [torch.from_numpy(item['point_cloud_dims_min']) for item in batch]
    pcd_max = [torch.from_numpy(item['point_cloud_dims_max']) for item in batch]
    rgbs = [pad_rgb_to_size(item['rgb_tensor']) for item in batch]

    max_points = max(p.shape[0] for p in pcds)
    padded_pcds = []
    for pcd in pcds:
        pad_len = max_points - pcd.shape[0]
        padded = torch.cat([pcd, torch.zeros(pad_len, 3, dtype=pcd.dtype, device=pcd.device)], dim=0)
        padded_pcds.append(padded)

    max_boxes = max(b.shape[0] for b in bboxes)
    padded_bboxes = []
    for bbox in bboxes:
        pad_len = max_boxes - bbox.shape[0]
        padded = torch.cat([bbox, torch.zeros(pad_len, 8, 3, dtype=bbox.dtype, device=bbox.device)], dim=0)
        padded_bboxes.append(padded)

    return {
        'pcd_tensor': torch.stack(padded_pcds),           # [B, max_points, 3]
        'bbox3d_tensor': torch.stack(padded_bboxes),      # [B, max_boxes, 8, 3]
        'point_cloud_dims_min': torch.stack(pcd_min),     # [B, 3]
        'point_cloud_dims_max': torch.stack(pcd_max),     # [B, 3]
        'rgb_tensor': torch.stack(rgbs),                  # [B, 3, 600, 600]
    }

def move_to_device(data: Union[dict, torch.Tensor], device: torch.device) -> Union[dict, torch.Tensor]:
    """
    Move the data to the specified device.

    Args:
        data (dict | torch.Tensor): A dictionary containing the data to be moved to the device.
        device (torch.device): The device to move the data to.

    Returns:
        dict | torch.Tensor: A dictionary containing the data moved to the device.
    """
    if isinstance(data, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
    

