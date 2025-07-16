"""
The file conatins the loss function used to match the 3D bounding box.
This file is also inspired from the official implementation of 3DDETR.
Link: https://facebookresearch.github.io/3detr
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

# Some util functions that needs to be imported as well
from scipy.optimize import linear_sum_assignment

from utils.bounding_box_operations import generalized_box3d_iou


def huber_loss(error: float, delta: float = 1.0) -> torch.Tensor:
    """
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss


def is_distributed() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    dim_squeeze = False
    if tensor.ndim == 0:
        tensor = tensor[None, ...]
        dim_squeeze = True
    torch.distributed.all_reduce(tensor)
    if dim_squeeze:
        tensor = tensor.squeeze(0)
    return tensor


def all_reduce_average(tensor: torch.Tensor) -> torch.Tensor:
    val = all_reduce_sum(tensor)
    return val / get_world_size()


class MatcherLoss(nn.Module):
    def __init__(self, cost_giou: float, cost_box_corners: float, cost_l1: float) -> None:
        super().__init__()
        self.cost_giou = cost_giou
        self.cost_box_corners = cost_box_corners
        self.cost_l1 = cost_l1

    @torch.no_grad()
    def forward(self, outputs: dict, targets: torch.Tensor, epoch: int) -> dict:
        # Get the batch size
        batch_size = outputs['box_corners'].shape[0]
        # Get the number of queries
        num_queries = outputs['box_corners'].shape[1]

        # Objectness cost (batch, num_queries, 1). Negative log prob of objectness
        # objectness_matching = -outputs["objectness_prob"].unsqueeze(-1)

        # Center cost (batch, num_queries, ngt). Distance b/w predicted and ground truth
        # center_matching = outputs['center_dist'].detach()

        # Compute the L1 distance between predicted and ground truth box corners
        box_corners_dist = torch.cdist(
            outputs['box_corners'].view(batch_size, num_queries, -1),
            targets.view(batch_size, -1, 24),
            p=1,
        )
        # Normalize the costs between [0,1] range for stable matching
        # box_corners_dist = box_corners_dist / (box_corners_dist.max() + 1e-6)

        # GIoU cost (batch, num_queries, ngt). Negative GIoU score. Normalize this as well.
        giou_matching = -outputs['gious'].detach()
        # giou_matching = giou_matching / (giou_matching.max() + 1e-6)

        # Progressive matching scheme
        if epoch < 10:
            self.cost_giou = 1.0
        else:
            self.cost_giou = 5.0

        # Calculate the final cost
        final_cost = self.cost_box_corners * box_corners_dist + self.cost_giou * giou_matching

        # Detach from GPU to CPU
        final_cost = final_cost.detach().cpu().numpy()

        # Append the assignments
        assignments = []

        # Auxiliary variables useful for batched loss computation
        per_prop_gt_inds = torch.zeros(
            [batch_size, num_queries], dtype=torch.int64, device=outputs['box_corners'].device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, num_queries], dtype=torch.float32, device=outputs['box_corners'].device
        )

        # Here, we perform the Hungarian matching
        for b in range(batch_size):
            if targets.shape[1] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : targets.shape[1]])
                assign = [
                    torch.from_numpy(x).long().to(device=outputs['box_corners'].device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
                assignments.append(assign)
            else:
                assignments.append([])

        return {
            'assignments': assignments,
            'per_prop_gt_inds': per_prop_gt_inds,
            'proposal_matched_mask': proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, matcher_loss: MatcherLoss, loss_weight_dict: dict) -> None:
        super().__init__()
        self.matcher_loss = matcher_loss
        self.loss_weight_dict = loss_weight_dict

        # Losses
        self.loss_functions = {
            'loss_box_corners': self.loss_box_corners,
            'loss_giou': self.loss_giou,
            'loss_size': self.loss_size,
            'loss_size_reg': self.loss_size_regularization,
            'loss_angle': self.loss_angle,
            'loss_aspect_ratio': self.loss_aspect_ratio,
            'loss_volume_aware': self.loss_volume_aware,
        }

    def loss_angle(
        self,
        outputs: dict,
        gt_bbox_corners: torch.Tensor,
        assignments: dict
    ) -> dict:
        """Compute angle loss for 3D bounding box orientation.

        Args:
            outputs: Model outputs containing angle predictions
            gt_bbox_corners: Ground truth bounding box corners (B, N_gt, 8, 3)
            assignments: Assignment dictionary from matcher

        Returns:
            dict: Dictionary containing angle loss
        """
        # Check if angle predictions are available in outputs
        if "angle_logits" not in outputs or "angle_residual" not in outputs:
            # Return zero loss if angle predictions are not available
            device = gt_bbox_corners.device
            return {
                "loss_angle_cls": torch.tensor(0.0, device=device),
                "loss_angle_reg": torch.tensor(0.0, device=device)
            }

        # Extract angle logits and residuals from the outputs
        angle_logits = outputs["angle_logits"]  # (B, N_q, num_angle_bins)
        angle_residual = outputs["angle_residual"]  # (B, N_q, num_angle_bins)

        batch_size, num_queries = angle_logits.shape[:2]
        num_gt = gt_bbox_corners.shape[1]

        if num_gt > 0:
            # Compute ground truth angles from bounding box corners
            # Extract the front face center and back face center to determine orientation
            gt_front_center = gt_bbox_corners[:, :, :4, :].mean(dim=2)  # (B, N_gt, 3)
            gt_back_center = gt_bbox_corners[:, :, 4:, :].mean(dim=2)   # (B, N_gt, 3)

            # Compute the direction vector (front to back)
            gt_direction = gt_back_center - gt_front_center  # (B, N_gt, 3)

            # Compute the yaw angle (rotation around Z-axis)
            gt_angles = torch.atan2(gt_direction[:, :, 1], gt_direction[:, :, 0])  # (B, N_gt)

            # Discretize angles into bins (e.g., 24 bins for 15-degree intervals)
            num_angle_bins = angle_logits.shape[-1]
            angle_bin_size = 2 * np.pi / num_angle_bins

            # Convert angles to bin indices
            gt_angle_bins = ((gt_angles + np.pi) / angle_bin_size).long()  # (B, N_gt)
            gt_angle_bins = torch.clamp(gt_angle_bins, 0, num_angle_bins - 1)

            # Compute residuals within bins
            gt_angle_residuals = (gt_angles + np.pi) % angle_bin_size - angle_bin_size / 2  # (B, N_gt)
            gt_angle_residuals = gt_angle_residuals / angle_bin_size  # Normalize to [-0.5, 0.5]

            # Get matched ground truth angles
            matched_gt_angle_bins = torch.gather(
                gt_angle_bins, 1, assignments["per_prop_gt_inds"]
            )  # (B, N_q)

            matched_gt_angle_residuals = torch.gather(
                gt_angle_residuals, 1, assignments["per_prop_gt_inds"]
            )  # (B, N_q)

            # Compute classification loss
            angle_cls_loss = F.cross_entropy(
                angle_logits.view(-1, num_angle_bins),
                matched_gt_angle_bins.view(-1),
                reduction="none"
            ).view(batch_size, num_queries)

            # Apply matching mask and normalize
            angle_cls_loss = (angle_cls_loss * assignments["proposal_matched_mask"]).sum()
            angle_cls_loss = angle_cls_loss / max(assignments["proposal_matched_mask"].sum(), 1)

            # Compute regression loss for the predicted bin
            # Extract residuals for the ground truth bins
            gt_bin_one_hot = F.one_hot(matched_gt_angle_bins, num_angle_bins).float()  # (B, N_q, num_bins)
            predicted_residuals = (angle_residual * gt_bin_one_hot).sum(dim=-1)  # (B, N_q)

            # Compute smooth L1 loss for residuals
            angle_reg_loss = F.smooth_l1_loss(
                predicted_residuals, matched_gt_angle_residuals, reduction="none"
            )

            # Apply matching mask and normalize
            angle_reg_loss = (angle_reg_loss * assignments["proposal_matched_mask"]).sum()
            angle_reg_loss = angle_reg_loss / max(assignments["proposal_matched_mask"].sum(), 1)

        else:
            # No ground truth boxes
            device = angle_logits.device
            angle_cls_loss = torch.tensor(0.0, device=device)
            angle_reg_loss = torch.tensor(0.0, device=device)

        return {
            "loss_angle_cls": angle_cls_loss,
            "loss_angle_reg": angle_reg_loss
        }



    def loss_size(self, outputs: dict, gt_bbox_corners: torch.Tensor, assignments: dict) -> dict:
        # Get the ground truth bbox sizes
        # Because the ground truth does not have size, we calculate the size from the corners
        # Need to verify if the corners are normalized
        gt_box_sizes = gt_bbox_corners.max(dim=2)[0] - gt_bbox_corners.min(dim=2)[0]
        # get the predicted bbox sizes
        pred_box_sizes = outputs['size_normalized']

        # Check if there are any ground truth boxes
        if gt_bbox_corners.shape[1] > 0:
            # Gather the matched ground truth sizes using assignments
            matched_gt_sizes = torch.gather(
                gt_box_sizes, 1, assignments['per_prop_gt_inds'].unsqueeze(-1).expand(-1, -1, 3)
            )
            # Get the loss
            size_loss = F.l1_loss(
                torch.log(pred_box_sizes + 1e-6),
                torch.log(matched_gt_sizes + 1e-6),
                reduction='none',
            ).sum(dim=-1)

            # Zero-out non-matched proposals
            size_loss *= assignments['proposal_matched_mask']
            size_loss = size_loss.sum()

            # Normalize by total number of GT boxes across the batch (batch-compatible)
            total_gt_boxes = gt_bbox_corners.shape[0] * gt_bbox_corners.shape[1]
            if total_gt_boxes > 0:
                size_loss /= total_gt_boxes

        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()

        # Return the Size loss
        return {'loss_size': size_loss}

    def loss_giou(self, outputs: dict, targets: torch.Tensor, assignments: dict) -> dict:
        # Get the GIoU distances
        gious_dist = 1 - outputs['gious']

        # Select appropriate GIoUs by using proposal to ground truth matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments['per_prop_gt_inds'].unsqueeze(-1)
        ).squeeze(-1)
        # Zero-out non-matched proposals
        giou_loss = giou_loss * assignments['proposal_matched_mask']
        giou_loss = giou_loss.sum()

        # Normalize by total number of GT boxes across the batch (batch-compatible)
        total_gt_boxes = targets.shape[0] * targets.shape[1]
        if total_gt_boxes > 0:
            giou_loss /= total_gt_boxes

        # Return the GIoU loss
        return {'loss_giou': giou_loss}

    def loss_size_regularization(
        self, outputs: dict, gt_bbox_corners: torch.Tensor, assignments: dict
    ) -> dict:
        """Penalize boxes that are too large compared to the GT boxes."""
        # Get the predicted box corners (B, N_q, 8, 3)
        predicted_box_corners = outputs['box_corners']

        # Compute the box dimensions
        # (B. N_q, 3)
        pred_dims = predicted_box_corners.max(dim=2)[0] - predicted_box_corners.min(dim=2)[0]
        # (B, N_q, 3)
        gt_dims = gt_bbox_corners.max(dim=2)[0] - gt_bbox_corners.min(dim=2)[0]

        # Get the matched ground truth dimensions
        matched_gt_dims = torch.gather(
            gt_dims, 1, assignments['per_prop_gt_inds'].unsqueeze(-1).expand(-1, -1, 3)
        )

        # FIXED: Scale mismatch resolved - now GT and predictions use same normalization

        # Create a safe version of matched_gt_dims that won't cause division issues
        safe_matched_gt_dims = torch.clamp(matched_gt_dims, min=1e-3)

        # Compute the size ratio penalty (penalize when pred > ground_truth)
        size_ratio = pred_dims / safe_matched_gt_dims

        # Only consider ratios for matched proposals by masking
        matched_mask_3d = assignments['proposal_matched_mask'].unsqueeze(-1).expand(-1, -1, 3)
        size_ratio = size_ratio * matched_mask_3d  # Zero out unmatched proposals

        # Cap the size ratio to prevent extreme penalties (max 5x larger is reasonable)
        size_ratio = torch.clamp(size_ratio, max=5.0)

        # Penalize boxes that are > 20% larger than the ground truth boxes
        size_penalty = F.relu(size_ratio - 1.2)

        # Sum penalty across dimensions, already masked to matched proposals only
        size_reg_loss = size_penalty.sum(dim=-1).sum()

        # Normalize by number of matched proposals
        num_matched = assignments['proposal_matched_mask'].sum()
        if num_matched > 0:
            size_reg_loss /= num_matched

        # Directly penalize large bounding boxes
        # size_reg_loss += torch.mean(pred_dims ** 2) * 0.1

        return {'loss_size_reg': size_reg_loss}

    def loss_box_corners(
        self, outputs: dict, gt_bbox_corners: torch.Tensor, assignments: dict
    ) -> dict:
        # Get the predicted box corners (B, N_q, 8, 3)
        predicted_box_corners = outputs['box_corners']

        # Check if there are any ground truth boxes
        if gt_bbox_corners.shape[1] > 0:
            # Gather the matched ground truth box corners based on assignments
            matched_gt_box_corners = torch.gather(
                gt_bbox_corners,
                1,
                assignments['per_prop_gt_inds'].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 3),
            )

            # Computer L1 loss between predicted and ground truth box corners
            # Also sum over the 8 corners and 3 coordinates
            box_corners_loss = F.l1_loss(
                predicted_box_corners, matched_gt_box_corners, reduction='none'
            ).sum(dim=(-1, -2))

            # Zero-out non-matched proposals
            box_corners_loss = box_corners_loss * assignments['proposal_matched_mask']
            box_corners_loss = box_corners_loss.sum()

            # Normalize by total number of GT boxes across the batch (batch-compatible)
            total_gt_boxes = gt_bbox_corners.shape[0] * gt_bbox_corners.shape[1]
            if total_gt_boxes > 0:
                box_corners_loss /= total_gt_boxes
        else:
            # Because there are no ground truth boxes, set the loss to zero
            box_corners_loss = torch.zeros(1, device=predicted_box_corners.device).squeeze()

        return {'loss_box_corners': box_corners_loss}

    def loss_aspect_ratio(
        self, outputs: dict, gt_bbox_corners: torch.Tensor, assignments: dict
    ) -> dict:
        """
        Aspect ratio aware loss to encourage better shape diversity.
        Penalizes predictions that don't match GT aspect ratios.
        """
        predicted_box_corners = outputs['box_corners']  # (B, N_q, 8, 3)

        if assignments['proposal_matched_mask'].sum() > 0:
            # Get matched GT corners
            matched_gt_box_corners = torch.gather(
                gt_bbox_corners,
                1,
                assignments['per_prop_gt_inds'].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 3),
            )

            # Compute box dimensions from corners
            def get_box_dimensions(corners):
                # corners: (B, N_q, 8, 3)
                mins = corners.min(dim=2)[0]  # (B, N_q, 3)
                maxs = corners.max(dim=2)[0]  # (B, N_q, 3)
                dims = maxs - mins  # (B, N_q, 3) - [length, width, height]
                return dims

            pred_dims = get_box_dimensions(predicted_box_corners)
            gt_dims = get_box_dimensions(matched_gt_box_corners)

            # Compute aspect ratios (normalize by largest dimension)
            pred_dims_norm = pred_dims / (pred_dims.max(dim=-1, keepdim=True)[0] + 1e-6)
            gt_dims_norm = gt_dims / (gt_dims.max(dim=-1, keepdim=True)[0] + 1e-6)

            # L1 loss on normalized aspect ratios
            aspect_ratio_loss = F.l1_loss(
                pred_dims_norm, gt_dims_norm, reduction='none'
            ).sum(dim=-1)

            # Apply mask and normalize
            aspect_ratio_loss *= assignments['proposal_matched_mask']
            aspect_ratio_loss = aspect_ratio_loss.sum()

            # Normalize by number of matched proposals
            num_matched = assignments['proposal_matched_mask'].sum()
            if num_matched > 0:
                aspect_ratio_loss /= num_matched
        else:
            aspect_ratio_loss = torch.zeros(1, device=predicted_box_corners.device).squeeze()

        return {'loss_aspect_ratio': aspect_ratio_loss}

    def loss_volume_aware(
        self, outputs: dict, gt_bbox_corners: torch.Tensor, assignments: dict
    ) -> dict:
        """
        Volume-aware loss that gives higher weight to small boxes.
        Addresses the under-prediction of small volumes.
        """
        predicted_box_corners = outputs['box_corners']  # (B, N_q, 8, 3)

        if assignments['proposal_matched_mask'].sum() > 0:
            # Get matched GT corners
            matched_gt_box_corners = torch.gather(
                gt_bbox_corners,
                1,
                assignments['per_prop_gt_inds'].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 3),
            )

            # Compute volumes
            def get_box_volume(corners):
                mins = corners.min(dim=2)[0]  # (B, N_q, 3)
                maxs = corners.max(dim=2)[0]  # (B, N_q, 3)
                dims = maxs - mins  # (B, N_q, 3)
                volume = dims.prod(dim=-1)  # (B, N_q)
                return volume, dims

            pred_volume, pred_dims = get_box_volume(predicted_box_corners)
            gt_volume, gt_dims = get_box_volume(matched_gt_box_corners)

            # Volume-weighted L1 loss (higher weight for smaller boxes)
            volume_weights = 1.0 / (gt_volume + 1e-3)  # Inverse volume weighting
            volume_weights = torch.clamp(volume_weights, max=10.0)  # Cap the weights

            # L1 loss on log volumes (more stable for small volumes)
            volume_loss = F.l1_loss(
                torch.log(pred_volume + 1e-6),
                torch.log(gt_volume + 1e-6),
                reduction='none'
            )

            # Apply volume weighting and mask
            volume_loss = volume_loss * volume_weights * assignments['proposal_matched_mask']
            volume_loss = volume_loss.sum()

            # Normalize by weighted number of matched proposals
            weighted_matched = (volume_weights * assignments['proposal_matched_mask']).sum()
            if weighted_matched > 0:
                volume_loss /= weighted_matched
        else:
            volume_loss = torch.zeros(1, device=predicted_box_corners.device).squeeze()

        return {'loss_volume_aware': volume_loss}

    def single_output_forward(
        self, outputs: dict, targets: torch.Tensor, epoch: int
    ) -> Tuple[torch.Tensor, dict, dict]:
        # Compute the Generalized Intersection over Union (GIoU) between predicted and ground truth boxes
        # NOTE: Here we have assumed that the boxes are not rotated.
        #       We also set needs_grad to False.
        # breakpoint()
        gious = generalized_box3d_iou(
            outputs['box_corners'],
            targets,
            nums_k2=torch.tensor([targets.shape[0], targets.shape[1]], device=outputs['box_corners'].device),
            rotated_boxes=False,
            needs_grad=True,  # (self.loss_weight_dict["loss_giou_weight"] > 0)
        )
        # Store the GIoU in the outputs dictionary
        outputs['gious'] = gious

        """
        # Compute the L1 distance between predicted and ground truth box centers
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        # Store the center distances in the outputs dictionary
        outputs["center_dist"] = center_diƒst
        """

        # Perform the matching between predictions and ground truth boxes
        # targets is already a tensor
        assignments = self.matcher_loss(outputs, targets, epoch)

        # Initialize a dictionary to store the losses
        losses = {}

        # Iterate over each loss function
        for k in self.loss_functions:
            loss_wt_key = k + '_weight'
            # Check if the loss weight is greater than 0 or if the loss weight key is not in the dictionary
            if (
                loss_wt_key in self.loss_weight_dict and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # Compute the current loss
                curr_loss = self.loss_functions[k](outputs, targets, assignments)



                # Update the losses dictionary with the current loss
                losses.update(curr_loss)
        # Initialize the final loss to 0
        final_loss = 0

        # Iterate over each loss weight in the dictionary
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                # Multiply the loss by its weight
                losses[k.replace('_weight', '')] *= self.loss_weight_dict[k]
                # Add the weighted loss to the final loss
                final_loss += losses[k.replace('_weight', '')]

        # Return the final loss and the individual losses
        return final_loss, losses, assignments

    def forward(
        self, outputs: dict, targets: torch.Tensor, epoch: Optional[int] = 0
    ) -> Tuple[torch.Tensor, dict, dict]:
        # Because the outputs has the batch dimension, add that in targets
        if targets.dim() == 3:
            targets = targets.unsqueeze(0)

        # Calculate the number of ground truth boxes present in the batch
        # nactual_gt = targets.shape[1]

        # Calculate the total number of boxes across all replicas and clamp it to a minimum of 1
        # num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()

        # Update the targets dictionary with the number of actual ground truth boxes
        # targets["nactual_gt"] = nactual_gt
        # targets["num_boxes"] = num_boxes

        # Update the targets dictionary with the number of boxes on this worker for distributed training
        # targets["num_boxes_replica"] = nactual_gt.sum().item()

        # Compute the loss and loss dictionary for the main outputs
        # breakpoint()
        loss, loss_dict, assingments = self.single_output_forward(outputs, targets, epoch)

        # If there are auxiliary outputs, compute the loss for each of them
        """
        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                # Compute the intermediate loss and loss dictionary for the auxiliary outputs
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )

                # Accumulate the intermediate loss to the total loss
                loss += interm_loss

                # Update the loss dictionary with the intermediate losses
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        """
        # Return the total loss and the loss dictionary
        return loss, loss_dict, assingments


class LossFunction(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # Define the matcher loss
        matcher_loss = MatcherLoss(
            cost_giou=config.loss.matcher_costs.giou,
            cost_box_corners=config.loss.matcher_costs.cost_box_corners,
            cost_l1=config.loss.matcher_costs.l1,
        )
        # Define the loss weight dictionary
        loss_weight_dict = {
            'loss_giou_weight': config.loss.weights.giou,
            'loss_box_corners_weight': config.loss.weights.box_corners,
            'loss_size_weight': config.loss.weights.size,
            'loss_size_reg_weight': config.loss.weights.size_reg,
            'loss_angle_cls_weight': config.loss.weights.angle_cls,
            'loss_angle_reg_weight': config.loss.weights.angle_reg,
            'loss_aspect_ratio_weight': getattr(config.loss.weights, 'aspect_ratio', 0.5),
            'loss_volume_aware_weight': getattr(config.loss.weights, 'volume_aware', 0.3),
        }
        # Define the criterion
        self.criterion = SetCriterion(matcher_loss=matcher_loss, loss_weight_dict=loss_weight_dict)

    def forward(
        self, outputs: dict, targets: torch.Tensor, epoch: Optional[int] = 0
    ) -> Tuple[torch.Tensor, dict, dict]:
        return self.criterion(outputs, targets, epoch)
