"""
The model is inspired by: https://facebookresearch.github.io/3detr
Although this is not a straight copy paste of the code, most of the code is inspired by the above link
There are some noteworthy changes that have been made to the code.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple, Dict

from models.detr3d.helpers import ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT
from models.detr3d.pointnet2 import PointnetSAModuleVotes
from models.detr3d.transformer_detr import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torchvision.models import resnet18
from utils.bounding_box_operations import flip_axis_to_camera_tensor, get_3d_box_batch_tensor
from utils.miscellaneous import farthest_point_sample, scale_points, shift_scale_points


class GenericMLP(nn.Module):
    """
    A generic Multi-Layer Perceptron (MLP) class that can be configured with various options.
    The class can be used to create both linear and convolutional MLPs for comparison of results.

    Args:
        input_dim (int): The dimension of the input features.
        hidden_dims (list of int): A list containing the dimensions of the hidden layers.
        output_dim (int): The dimension of the output layer.
        norm_fn_name (str, optional): The name of the normalization function to use. Default is None.
        activation (str, optional): The activation function to use. Default is "relu".
        use_conv (bool, optional): Whether to use convolutional layers instead of linear layers. Default is False.
        dropout (float or list of float, optional): Dropout probability for hidden layers. Default is None.
        hidden_use_bias (bool, optional): Whether to use bias in hidden layers. Default is False.
        output_use_bias (bool, optional): Whether to use bias in the output layer. Default is True.
        output_use_activation (bool, optional): Whether to apply activation function on the output. Default is False.
        output_use_norm (bool, optional): Whether to apply normalization on the output. Default is False.
        weight_init_name (str, optional): The name of the weight initialization method. Default is None.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        norm_fn_name: Optional[str] = None,
        activation: str = 'relu',
        use_conv: bool = False,
        dropout: Optional[Union[float, List[float]]] = None,
        hidden_use_bias: bool = False,
        output_use_bias: bool = True,
        output_use_activation: bool = False,
        output_use_norm: bool = False,
        weight_init_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Select activation function
        activation = ACTIVATION_DICT[activation]
        norm = None

        # Select normalization function
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == 'ln' and use_conv:
            # Use GroupNorm as a substitute for LayerNorm in Conv1d
            def group_norm(x: torch.Tensor) -> nn.GroupNorm:
                return nn.GroupNorm(1, x)

            norm = group_norm

        # Ensure dropout is a list if specified
        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        # Initialize layers
        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)

            # Add normalization if specified
            if norm:
                layers.append(norm(x))

            # Add activation
            layers.append(activation())

            # Add dropout if specified
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx], inplace=False))
            prev_dim = x

        # Add final layer
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        # Add optional output normalization
        if output_use_norm:
            layers.append(norm(output_dim))

        # Add optional output activation
        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        # Apply weight initialization if specified
        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name: str) -> None:
        """
        Apply weight initialization to the layers.

        Args:
            weight_init_name (str): The name of the weight initialization method.
        """
        func = WEIGHT_INIT_DICT[weight_init_name]
        for _, param in self.named_parameters():
            if param.dim() > 1:  # Skip batch normalization/layer normalization parameters
                func(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output of the MLP.
        """
        return self.layers(x)


class PositionEmbeddingCoordsSine(nn.Module):
    """
    Position Embedding using Sine or Fourier transformations.

    Args:
        temperature (int, optional): Temperature parameter for scaling. Default is 10000.
        normalize (bool, optional): Whether to normalize the input coordinates. Default is False.
        scale (float, optional): Scaling factor. Must be set if `normalize` is True. Default is None.
        pos_type (str, optional): Type of positional encoding, either "sine" or "fourier". Default is "fourier".
        d_pos (int, optional): Dimensionality of the position encoding. Required for "fourier".
        d_in (int, optional): Input dimensionality. Default is 3.
        gauss_scale (float, optional): Scale for Gaussian random matrix used in Fourier embeddings. Default is 1.0.
    """

    def __init__(
        self,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = None,
        pos_type: str = 'fourier',
        d_pos: int = None,
        d_in: int = 3,
        gauss_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ['sine', 'fourier'], "pos_type must be 'sine' or 'fourier'"
        self.pos_type = pos_type
        self.scale = scale

        if pos_type == 'fourier':
            assert d_pos is not None and d_pos % 2 == 0, (
                "d_pos must be provided and even for 'fourier'"
            )
            B = torch.empty((d_in, d_pos // 2)).normal_() * gauss_scale
            self.register_buffer('gauss_B', B)
            self.d_pos = d_pos

    def get_sine_embeddings(
        self, xyz: torch.Tensor, num_channels: int, input_range: tuple
    ) -> torch.Tensor:
        """
        Compute sine positional embeddings.

        Args:
            xyz (Tensor): Input coordinates of shape (batch_size, num_points, d_in).
            num_channels (int): Number of output channels.
            input_range (tuple): Range for normalization if applicable.

        Returns:
            Tensor: Sine positional embeddings of shape (batch_size, num_channels, num_points).
        """
        xyz = xyz.clone()

        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        rems = num_channels - (ndim * xyz.shape[2])

        final_embeds = []
        for d in range(xyz.shape[2]):
            cdim = ndim + (2 if rems > 0 else 0)
            rems -= 2 if rems > 0 else 0

            dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            raw_pos = xyz[:, :, d] * self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)

        return torch.cat(final_embeds, dim=2).permute(0, 2, 1)

    def get_fourier_embeddings(
        self, xyz: torch.Tensor, num_channels: int = None, input_range: tuple = None
    ) -> torch.Tensor:
        """
        Compute Fourier positional embeddings.

        Args:
            xyz (Tensor): Input coordinates of shape (batch_size, num_points, d_in).
            num_channels (int, optional): Number of output channels. Default is None.
            input_range (tuple, optional): Range for normalization if applicable. Default is None.

        Returns:
            Tensor: Fourier positional embeddings of shape (batch_size, num_channels, num_points).
        """
        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        d_out = num_channels // 2
        assert d_out <= self.gauss_B.shape[1]
        assert xyz.shape[-1] == self.gauss_B.shape[0]

        xyz = xyz.clone()
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz_proj = torch.mm(xyz.view(-1, xyz.shape[-1]), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        return torch.cat((xyz_proj.sin(), xyz_proj.cos()), dim=2).permute(0, 2, 1)

    def forward(
        self, xyz: torch.Tensor, num_channels: int = None, input_range: tuple = None
    ) -> torch.Tensor:
        """
        Forward method to compute positional embeddings.

        Args:
            xyz (Tensor): Input coordinates of shape (batch_size, num_points, d_in).
            num_channels (int, optional): Number of output channels. Default is None.
            input_range (tuple, optional): Range for normalization if applicable. Default is None.

        Returns:
            Tensor: Positional embeddings of shape (batch_size, num_channels, num_points).
        """
        if self.pos_type == 'sine':
            return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == 'fourier':
            return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f'Unknown pos_type: {self.pos_type}')

    def extra_repr(self) -> str:
        """
        Extra representation string for printing module info.

        Returns:
            str: Extra information about the module.
        """
        return f'type={self.pos_type}, scale={self.scale}, normalize={self.normalize}, ' + (
            f'gaussB_shape={self.gauss_B.shape}, gaussB_sum={self.gauss_B.sum().item()}'
            if hasattr(self, 'gauss_B')
            else ''
        )


class BoxProcessor:
    """
    Convert the output of 3D DETR MLP heads into Bounding boxes.
    """

    def __init__(self) -> None:
        pass

    def compute_predicted_center(
        self,
        center_offset: torch.Tensor,
        query_xyz: torch.Tensor,
        point_cloud_dims: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the predicted center of the bounding box.

        Args:
            center_offset (Tensor): Offset of the predicted center from the query points.
            query_xyz (Tensor): Query points coordinates.
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.

        Returns:
            Tuple[Tensor, Tensor]: Normalized and unnormalized predicted center coordinates.
        """
        center_unnormalized = center_offset + query_xyz
        # Normalize the center by shift scaling
        center_normalized = shift_scale_points(center_unnormalized, point_cloud_dims)

        return center_normalized, center_unnormalized

    def compute_predicted_size(
        self, size_unnormalized: torch.Tensor, point_cloud_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the predicted size of the bounding box.

        Args:
            size_unnormalized (Tensor): Unnormalized size of the bounding box.
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.

        Returns:
            Tensor: Normalized size of the bounding box.
        """
        # Normalize the size by shift scaling
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        # Clamp the size to be within the scene scale
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_unnormalized, scene_scale)

        return size_unnormalized

    def compute_predicted_angle(
        self, angle_logits: torch.Tensor, angle_residuals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the predicted angle of the bounding box.

        Args:
            angle_logits (Tensor): Logits for the angle classification.
            angle_residuals (Tensor): Residuals for the angle regression.

        Returns:
            Tensor: Predicted angle of the bounding box.
        """
        if angle_logits.shape[-1] == 1:
            # Special case for datasets with no rotation angle
            angle = angle_logits * 0 + angle_residuals * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = (
                2 * np.pi / 12
            )  # Paper mentions that the angle is quantized into 12 bins
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residuals.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi

        return angle

    def box_parameterization_to_corners(
        self,
        unnormalized_box_center: torch.Tensor,
        unnormalized_box_size: torch.Tensor,
        box_angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert box parameterization to corner coordinates.

        Args:
            unnormalized_box_center (Tensor): Unnormalized center coordinates of the bounding box.
            unnormalized_box_size (Tensor): Unnormalized size of the bounding box.
            box_angle (Tensor): Angle of the bounding box.

        Returns:
            Tensor: Corner coordinates of the bounding box.
        """
        # Adjust the centers to the appropriate coordinate system
        box_center_upright = flip_axis_to_camera_tensor(unnormalized_box_center)

        # Generate the 3D bounding box corners
        box_corners = get_3d_box_batch_tensor(unnormalized_box_size, box_angle, box_center_upright)

        return box_corners


class Model3DDETR(nn.Module):
    """
    The main 3D DETR model as described by the paper:
    http://arxiv.org/abs/2109.08141

    Important points of the model:
    1) Pre-Encoder module is responsible for taking in raw point cloud data and converting to N'X D feature map
    2) Encoder architecture containing Multihead attention
    3) Decoder architecture is based on transformer architecture
    """

    def __init__(
        self,
        pre_encoder: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_dim: int = 256,
        decoder_dim: int = 256,
        position_embedding: str = 'Fourier',
        mlp_dropout: float = 0.3,
        num_queries: int = 256,
        num_angular_bins: int = 1,
        rgb_backbone_output_dim=64,
        intrinsics=None,
        image_size=(578, 646),
        use_rgb_fusion=True,
        use_mask_supervision=True,
        num_classes=10,
    ) -> None:
        # Calling parent constructor and initializing the variables
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder

        # If inductive bias is needed we can then add masking checks to introduce inductive bias
        # Create a member for encoder to decoder projection
        self.encoder_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[encoder_dim],  # if inductive bias is needed, we need to add a hidden layer
            output_dim=decoder_dim,
            norm_fn_name='bn1d',  # Was 'bn1d'
            activation='relu',
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        # Member for positional embedding
        self.positional_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )

        # Member for query projection
        self.query_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        # Image fusion 

        self.use_rgb_fusion = use_rgb_fusion
        self.use_mask_supervision = use_mask_supervision
        self.image_size = image_size
        self.intrinsics = intrinsics if intrinsics is not None else torch.tensor([
            [575.0, 0.0, 323.0],
            [0.0, 575.0, 289.0],
            [0.0, 0.0, 1.0]
        ])

        if use_rgb_fusion:
            self.rgb_backbone = nn.Sequential(
                *list(resnet18(pretrained=True).children())[:6]
            )  # Output: (B, 64, H/8, W/8)
            # self.rgb_proj = nn.Conv1d(rgb_backbone_output_dim + self.pre_encoder.mlp[-1].out_channels,
            #                           self.pre_encoder.mlp[-1].out_channels, 1)
            feat_dim = encoder_dim  # or config.model.encoder.dim if you're passing it
            self.rgb_proj = nn.Conv1d(
                rgb_backbone_output_dim*2,
                3,
                kernel_size=1
)


        # if use_mask_supervision:
        #     self.segmentation_head = nn.Sequential(
        #         nn.Conv1d(encoder_dim, 128, kernel_size=1),
        #         nn.ReLU(),
        #         nn.Conv1d(128, num_classes, kernel_size=1)
        #     )



        # Member for decoder
        self.decoder = decoder

        # Need to build MLP heads
        self.build_mlp_heads(decoder_dim, mlp_dropout, num_angular_bins)

        # Member for number of queries
        self.num_queries = num_queries

        # Member for converting MLP output to bounding boxes
        self.box_processor = BoxProcessor()

    def project_points(self, points_3d, intrinsics, image_size):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x = points_3d[:, :, 0]
        y = points_3d[:, :, 1]
        z = points_3d[:, :, 2].clamp(min=1e-5)

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        h, w = image_size
        u = u.clamp(0, w - 1)
        v = v.clamp(0, h - 1)

        return torch.stack([u, v], dim=-1)  # (B, N, 2)

    def sample_rgb_features(self, rgb_feat, uv_coords):
        _, _, H, W = rgb_feat.shape
        uv_norm = uv_coords.clone()
        uv_norm[..., 0] = (uv_coords[..., 0] / (W - 1)) * 2 - 1
        uv_norm[..., 1] = (uv_coords[..., 1] / (H - 1)) * 2 - 1
        grid = uv_norm.unsqueeze(2)
        sampled = F.grid_sample(rgb_feat, grid, mode='bilinear', align_corners=True)
        return sampled.squeeze(3)

    def fuse_rgb_with_points(self, pc_xyz, pc_features, rgb_image):
        rgb_feat = self.rgb_backbone(rgb_image)
        uv_coords = self.project_points(pc_xyz, self.intrinsics.to(pc_xyz.device), self.image_size)
        sampled_rgb = self.sample_rgb_features(rgb_feat, uv_coords)
        if pc_features is not None:
            fused = torch.cat([pc_features, sampled_rgb], dim=1)
        else:
            fused = sampled_rgb
        return self.rgb_proj(fused)

    def build_mlp_heads(self, decoder_dim: int, mlp_dropout: float, num_angular_bins: int) -> None:
        """Builds the MLP heads for the 3D bounding box detection model.

        Args:
            decoder_dim (int): Dimension of the decoder output.
            mlp_dropout (float): Dropout rate for the MLP layers.
        Returns:
            None
        """
        # Define an MLP function
        mlp_func = partial(
            GenericMLP,
            norm_fn_name='bn1d',
            activation='relu',
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # The bounding box head is a 3D bounding box
        # MLP head for the various 3D bounding box parameters
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)

        # The paper mentions quantizing the angles into 12 bins
        # Angle classification head: Which bin does the angle belong to
        angle_cls_head = mlp_func(output_dim=num_angular_bins)
        # Angle regression head: Finetune the residual within the classification to give continuous value
        angle_reg_head = mlp_func(output_dim=num_angular_bins)

        # Aggregate the individual heads
        self.mlp_heads = nn.ModuleDict(
            [
                ('center_head', center_head),
                ('size_head', size_head),
                ('angle_cls_head', angle_cls_head),
                ('angle_residual_head', angle_reg_head),
            ]
        )

    def get_query_embedding(
        self, encoder_xyz: torch.Tensor, point_cloud_dims: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate query embeddings by sampling points from the encoder output and applying positional encoding.

        Args:
            encoder_xyz (Tensor): Encoder output coordinates of shape (batch_size, num_points, 3).
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.

        Returns:
            Tuple[Tensor, Tensor]: Query coordinates and query embeddings.
        """
        # The farthest point sample function is inspired from the PointNet++ paper
        # The input is the encoder_xyz and we need to sample the num_queries sample points
        query_indices = farthest_point_sample(encoder_xyz, self.num_queries)
        # Convert to long
        query_indices = query_indices.long()

        # Convert to xyz
        """
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_indices) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)
        """
        query_xyz = torch.gather(
            encoder_xyz, 1, query_indices.unsqueeze(-1).expand(-1, -1, encoder_xyz.size(-1))
        )

        # Positional embedding for the query points using default Fourier transform
        positional_embedding = self.positional_embedding(query_xyz, input_range=point_cloud_dims)
        # Query embeddings
        query_embeddings = self.query_projection(positional_embedding)

        return query_xyz, query_embeddings

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Break up the point cloud into coordinates and features.
            pc (torch.Tensor): A point cloud tensor of shape (N, M, C) where N is the batch size,
                       M is the number of points, and C is the number of channels. The first
                       three channels are expected to be the XYZ coordinates. Additional
                       channels may include colors, normals, etc.
        Returns:
            tuple: A tuple containing:
            - xyz (torch.Tensor): A tensor of shape (N, M, 3) containing the XYZ coordinates.
            - features (torch.Tensor or None): A tensor of shape (N, C-3, M) containing the additional
                               features, or None if there are no additional features.
        """

        # Point cloud may contain colours and/or normals, so we need to break them up
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    # def run_encoder(
    #     self, point_clouds: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Runs the encoder on the given point clouds.
    #         point_clouds (torch.Tensor): The input point clouds with shape (batch_size, num_points, num_features).

    #     Returns:
    #         tuple: A tuple containing:
    #         - encoder_xyz (torch.Tensor): The encoded xyz coordinates with shape (batch_size, num_points, 3).
    #         - encoder_features (torch.Tensor): The encoded features with shape (num_points, batch_size, num_features).
    #         - encoder_indices (torch.Tensor): The indices of the points used in the encoder with shape (batch_size, num_points).
    #     """
    #     # break-up point cloud into xyz and features
    #     xyz, features = self._break_up_pc(point_clouds)

    #     # Pass the point cloud through the pre-encoder
    #     pre_encoder_xyz, pre_encoder_features, pre_encoder_indices = self.pre_encoder(xyz, features)

    #     # Dimensions are:
    #     # xyz: (batch_size, num_points, 3)
    #     # features: (batch_size, num_features, num_points)
    #     # indices: (batch_size, num_points)

    #     # Multihead attention in encoder expects num_points x batch x num_features
    #     pre_encoder_features = pre_encoder_features.permute(2, 0, 1)

    #     # XYZ points are in (batch, num_points, 3) order
    #     encoder_xyz, encoder_features, encoder_indices = self.encoder(
    #         pre_encoder_features, xyz=pre_encoder_xyz
    #     )

    #     # Checks
    #     if encoder_indices is None:
    #         # Encoder does not perform dowmsampling
    #         encoder_indices = pre_encoder_indices
    #     else:
    #         # Use gather to ensure that it works for both FPS and random sampling
    #         encoder_indices = torch.gather(
    #             pre_encoder_indices, 1, encoder_indices.type(torch.int64)
    #         )

    #     return encoder_xyz, encoder_features, encoder_indices

    def get_box_prediction(
        self,
        query_xyz: torch.Tensor,
        point_cloud_dims: List[torch.Tensor],
        box_features: torch.Tensor,
    ) -> dict:
        """
        Generate box predictions from the decoder output features.

        Args:
            query_xyz (Tensor): Query coordinates of shape (batch_size, num_queries, 3).
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.
            box_features (Tensor): Decoder output features of shape (num_layers, num_queries, batch, channel).

        Returns:
            dict: A dictionary containing the final and auxiliary outputs with keys:
                - "outputs": Output from the last layer of the decoder.
                - "auxiliary_outputs": Outputs from the intermediate layers of the decoder.
        """

        # Dimensions are:
        # query_xyz: (batch_size, num_queries, 3)
        # point_cloud_dims: [min, max] coordinates
        # box_features: (num_layers, num_queries, batch, channel)
        # Change box_features to ((num_layers x batch), channel, num_queries)
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch_size, channel, num_queries = box_features.shape
        box_features = box_features.reshape(num_layers * batch_size, channel, num_queries)

        # Bounding box prediction parameters
        center_offset = self.mlp_heads['center_head'](box_features).sigmoid().transpose(1, 2) - 0.5
        size_normalized = self.mlp_heads['size_head'](box_features).sigmoid().transpose(1, 2)
        angle_logits = self.mlp_heads['angle_cls_head'](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads['angle_residual_head'](box_features).transpose(
            1, 2
        )

        # Reshape the outputs to (num_layers, batch, num_queries, num_output)
        center_offset = center_offset.reshape(num_layers, batch_size, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch_size, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch_size, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch_size, num_queries, -1
        )

        # Get the angle residual values
        angle_residual = angle_residual_normalized * (np.pi / angle_residual_normalized.shape[-1])

        # Placeholder for outputs
        outputs = []

        for i in range(num_layers):
            # The box processor class converts the MLP outputs to bounding boxes
            (center_normalized, center_unnormalized) = self.box_processor.compute_predicted_center(
                center_offset[i], query_xyz, point_cloud_dims
            )
            # Get the angle
            angle_contiguous = self.box_processor.compute_predicted_angle(
                angle_logits[i], angle_residual[i]
            )
            # Get size unnormlized
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[i], point_cloud_dims
            )
            # Get the corners of the bounding box
            box_corners = self.box_processor.box_parameterization_to_corners(
                center_unnormalized, size_unnormalized, angle_contiguous
            )

            box_prediction = {
                'center_normalized': center_normalized.contiguous(),
                'center_unnormalized': center_unnormalized,
                'size_normalized': size_normalized[i],
                'angle_logits': angle_logits[i],
                'angle_residual': angle_residual[i],
                'angle_residual_normalized': angle_residual_normalized[i],
                'angle_contiguous': angle_contiguous,
                'box_corners': box_corners,
            }

            outputs.append(box_prediction)

        # Intermediate decoder layer outputs are only used during training
        auxiliary_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            'outputs': outputs,  # output from the last layer of the decoder
            'auxiliary_outputs': auxiliary_outputs,  # Output from the intermediate layers of the decoder
        }

    def pad_point_clouds(self, point_clouds: List[torch.Tensor], max_points: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad point clouds to same size and create attention masks.

        Args:
            point_clouds: List of point cloud tensors with different sizes
            max_points: Maximum number of points (if None, use max in batch)

        Returns:
            Tuple of (padded_point_clouds, attention_masks)
        """
        if max_points is None:
            max_points = max(pc.shape[0] for pc in point_clouds)

        batch_size = len(point_clouds)
        feature_dim = point_clouds[0].shape[1]

        # Create padded tensor
        padded_pcs = torch.zeros(batch_size, max_points, feature_dim,
                                device=point_clouds[0].device, dtype=point_clouds[0].dtype)

        # Create attention masks (True for valid points, False for padding)
        attention_masks = torch.zeros(batch_size, max_points, dtype=torch.bool,
                                     device=point_clouds[0].device)

        for i, pc in enumerate(point_clouds):
            num_points = pc.shape[0]
            padded_pcs[i, :num_points] = pc
            attention_masks[i, :num_points] = True

        return padded_pcs, attention_masks

    # def forward(self, inputs_point_cloud, input_image, point_cloud_dims_min, point_cloud_dims_max, encoder_only=False):
    #     """Main forward pass - can switch between batched and loop-based processing."""
    #     # For now, keep the original loop-based approach for compatibility
    #     # TODO: Switch to forward_batched once pre_encoder supports masking

    #     batch_predictions = []

    #     for inputs_pcd, input_rgb, pcd_dim_min, pcd_dim_max in zip(inputs_point_cloud, input_image, point_cloud_dims_min, point_cloud_dims_max):
    #         rgb = input_rgb
    #         pc_tensor = inputs_pcd

    #         xyz, feats = self._break_up_pc(pc_tensor.unsqueeze(0))

    #         if self.use_rgb_fusion:
    #             feats = self.fuse_rgb_with_points(xyz, feats, rgb.unsqueeze(0))

    #         xyz, features, indices = self.pre_encoder(xyz, feats)
    #         features = features.permute(2, 0, 1)

    #         encoder_xyz, encoder_features, _ = self.encoder(features, xyz=xyz)
    #         encoder_features = self.encoder_decoder_projection(
    #             encoder_features.permute(1, 2, 0)
    #         ).permute(2, 0, 1)


    def forward(self, inputs_point_cloud, input_image, point_cloud_dims_min, point_cloud_dims_max, encoder_only=False):

        point_clouds = inputs_point_cloud

        xyz, feats = self._break_up_pc(point_clouds)

        if self.use_rgb_fusion:
            feats = self.fuse_rgb_with_points(xyz, feats, input_image)

        xyz, features, _ = self.pre_encoder(xyz, feats)
        features = features.permute(2, 0, 1)

        enc_xyz, enc_features, _ = self.encoder(features, xyz=xyz)

        enc_features = self.encoder_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            point_cloud_dims_min,
            point_cloud_dims_max,
        ]
        query_xyz, query_embed = self.get_query_embedding(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.positional_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]
        box_predictions = self.get_box_prediction(
            query_xyz, point_cloud_dims, box_features
        )
        return box_predictions

# Function to build pre_encoder
def build_preencoder(config) -> PointnetSAModuleVotes:
    """
    Builds the preencoder configuration for the model.

    Args:
        cfg_model: An argument parser object that contains the following attributes:
            - use_color (int): A flag indicating whether to use color information (1 for True, 0 for False).
            - encoder_dim (int): The dimension of the encoder output.

    Returns:
        PointnetSAModuleVotes: The preencoder module.
    """
    mlp_dimensions = [3 * int(config.model.encoder.use_color), 64, 128, config.model.encoder.dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=config.model.encoder.preencoder_npoints,
        mlp=mlp_dimensions,
        normalize_xyz=True,
    )
    return preencoder


# Function to build encoder
def build_encoder(config) -> TransformerEncoder:
    """
    Builds and returns an encoder based on the specified arguments.
    Args:
        cfg_model: An object containing the following attributes:
            - encoder_type (str): The type of encoder to build. Can be "Vanilla" or "masked".
            - encoder_dim (int): The dimension of the encoder model.
            - encoder_nheads (int): The number of heads in the multi-head attention mechanism.
            - encoder_ffn_dim (int): The dimension of the feedforward network.
            - encoder_dropout (float): The dropout rate.
            - encoder_activation (str): The activation function to use.
            - encoder_num_layers (int): The number of layers in the encoder.
    Returns:
        TransformerEncoder: An instance of TransformerEncoder based on the specified encoder type.
    Raises:
        ValueError: If the encoder_type is not recognized.
    """
    encoder_layer = TransformerEncoderLayer(
        d_model=config.model.encoder.dim,
        nhead=config.model.encoder.nheads,
        dim_feedforward=config.model.encoder.ffn_dim,
        dropout=config.model.encoder.dropout,
        activation=config.model.encoder.activation, #check for string
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=config.model.encoder.num_layers
    )

    return encoder


# Function to build the decoder
def build_decoder(config) -> TransformerDecoder:
    """
    Builds a Transformer decoder using the provided arguments.
    Args:
        cfg_model: An object containing the following attributes:
            - decoder_dim (int): The dimension of the decoder model.
            - decoder_nhead (int): The number of heads in the multihead attention mechanism.
            - decoder_ffn_dim (int): The dimension of the feedforward network.
            - decoder_dropout (float): The dropout rate for the decoder.
            - decoder_num_layers (int): The number of layers in the decoder.
    Returns:
        TransformerDecoder: A Transformer decoder instance configured with the specified parameters.
    """
    decoder_layer = TransformerDecoderLayer(
        d_model=config.model.decoder.dim,
        nhead=config.model.decoder.nhead,
        dim_feedforward=config.model.decoder.ffn_dim,
        dropout=config.model.decoder.dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=config.model.decoder.num_layers,
        return_intermediate=True,
    )

    return decoder


# Function to build the 3D DETR model
def build_3ddetr_model(config) -> Model3DDETR:
    """
    Build the 3DDETR model and its output processor.
    Args:
        cfg_model (Namespace): The arguments containing model hyperparameters and configurations.
            - encoder_dim (int): Dimension of the encoder.
            - decoder_dim (int): Dimension of the decoder.
            - position_embedding (str): Type of position embedding to use.
            - mlp_dropout (float): Dropout rate for the MLP layers.
            - num_queries (int): Number of queries for the decoder.
    Returns:
            - model (Model3DDETR): The constructed 3DDETR model.
    """
    pre_encoder = build_preencoder(config)
    encoder = build_encoder(config)
    decoder = build_decoder(config)
    model = Model3DDETR(
        pre_encoder=pre_encoder,
        encoder=encoder,
        decoder=decoder,
        encoder_dim=config.model.encoder.dim,
        decoder_dim=config.model.decoder.dim,
        position_embedding=config.model.position_embedding, #check for string
        mlp_dropout=config.model.mlp_dropout,
        num_queries=config.model.num_queries,
        num_angular_bins=config.model.num_angular_bins,
    )
    # Not sure if we need the output_processor, so simply comment this out now
    # output_processor = BoxProcessor(config=config)

    return model
