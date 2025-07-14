import os
from typing import Dict, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class SereactDataloader(Dataset):
    """Sereact specific dataloader class with data augmentation for 3D bounding box localization.

    Args:
        Dataset (_type_): Torch datasetclass. Override abstract methods
    """

    def __init__(
        self,
        source_path: str,
        task: str,
        transform: callable = None,
        debug: bool = False,
    ) -> None:
        """Constructor method

        Args:
            source_path (str): Path to where the data is stored
            transform (_type_, optional): Transforms to applied to sample. Defaults to None.
            debug (bool, optional): If True, enables debug visualization. Defaults to False.
            augment (bool, optional): If True, applies data augmentation. Defaults to False.
        """
        self.source_path = source_path
        self.transform = transform
        self.folderpath = os.listdir(source_path)
        if task == 'train':
            self.folderpath = self.folderpath[40:]
        elif task == 'test':
            self.folderpath = self.folderpath[:40]
        self.debug = debug

    def __len__(self) -> int:
        """Returns the total number of data points available."""
        return len(self.folderpath)


    def normalize_pointcloud(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize a point cloud to zero mean and unit sphere.

        Args:
            points (torch.Tensor): Input point cloud of shape (N, 3)

        Returns:
            Tuple:
                - normalized_points: (N, 3) tensor
                - centroid: (3,) tensor
                - scale: scalar tensor
        """
        # print(points.shape, type(points))
        assert points.ndim == 2 and points.shape[1] == 3, f"Expected shape (N, 3), got {points.shape}"

        # Compute centroid
        centroid = points.mean(dim=0)  # shape (3,)

        # Center the point cloud
        points_centered = points - centroid  # shape (N, 3)

        # Compute max distance from origin (L2 norm)
        scale = torch.norm(points_centered, dim=1).max()  # scalar

        # Normalize to unit sphere
        normalized_points = points_centered / scale

        return normalized_points, centroid, scale

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Abstract __getitem__() method that needs to be overriden.

        Args:
            index (int): Index to get the item

        Returns:
            Dict[str, torch.Tensor]: Dictionary holding the appropriate information for that index.
        """
        subfolder = self.folderpath[index]
        subfolder_path = os.path.join(self.source_path, subfolder)

        # Load data from subdirectory. It has 4 items
        # 1) Color image named "rgb.jpg"
        # 2) 3D bounding box representation named "bbox3d.npy"
        # 3) Mask information named "mask.npy"
        # 4) Point cloud information "pc.npy"

        # 1) Load RGB image
        # breakpoint()
        image = np.array(Image.open(os.path.join(subfolder_path, 'rgb.jpg')).convert('RGB')) #(533, 692, 3)
        # Convert this to a torch tensor for further usage
        rgb_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # Normalize the image
        if rgb_tensor.max() > 1:
            rgb_tensor = rgb_tensor / 255.0
        # 2) 3D bounding box
        bbox_3d = np.load(os.path.join(subfolder_path, 'bbox3d.npy'))
        # Have a tensor form of the bounding box
        bbox_3d_tensor = torch.tensor(bbox_3d, dtype=torch.float32) #torch.Size([9, 8, 3])

        # 3) Mask information
        mask = np.load(os.path.join(subfolder_path, 'mask.npy'))

        # 4) Point cloud
        pcd = np.load(os.path.join(subfolder_path, 'pc.npy'))
        # Have a tensor form of the point cloud
        
        pcd_tensor = torch.from_numpy(pcd.reshape(3, -1).T).float() #torch.Size([368836, 3])
        # breakpoint()
        # Check if asked to apply augmentation.
        if self.transform:
            pcd_tensor, bbox_3d_tensor = self.transform(pcd_tensor, bbox_3d_tensor)
        # Normalize point cloud
        
        pcd_normalized, centroid, max_dist = self.normalize_pointcloud(pcd_tensor) #(368836, 3)
        # breakpoint()
        # Convert to tensor (already in correct shape)
        pcd_tensor = pcd_normalized.float()

        # Normalize bbox coordinates (shape: (K, 8, 3))
        # bbox_3d_normalized = (bbox_3d - centroid.view(1, 1, 3)) / scale
        bbox_3d_normalized = (bbox_3d_tensor - centroid) / max_dist  # shape: [9, 8, 3]

        # If not already a tensor, convert numpy to torch:
        # bbox_3d_tensor = torch.from_numpy(bbox_3d_normalized).float()
        # But here it's already a tensor:
        bbox_3d_tensor = bbox_3d_normalized.float()

        # Store in data dict
        if self.debug:
            data_dict = {
                'rgb': image,
                'bbox3d': bbox_3d,
                'mask': mask,
                'pcd': pcd,
                'pcd_tensor': pcd_tensor,
                'bbox3d_tensor': bbox_3d_tensor,
                'normalization_params': {
                    'centroid': centroid.float(),
                    'scale': max_dist.float(),
                },
            }
        else:
            # print("ass", mask)
            data_dict = {
                'rgb_tensor': rgb_tensor,
                'pcd_tensor': pcd_tensor,
                'mask': mask,
                'bbox3d_tensor': bbox_3d_tensor,
                'point_cloud_dims_min': np.array(pcd_normalized).reshape(-1, 3).min(axis=0)[:3],
                'point_cloud_dims_max': np.array(pcd_normalized).reshape(-1, 3).max(axis=0)[:3],
                'normalization_params': {
                    'centroid': centroid.float(),
                    'scale': max_dist.float(),
                },
            }
        return data_dict

    

if __name__=='__main__':

    # transform = SereactAugmentation()
    dataset = SereactDataloader('/home-local2/akath.extra.nobkp/dl_challenge', task = 'train', transform=None, debug = False)
    data = dataset[0]
    breakpoint()
    print("ass")