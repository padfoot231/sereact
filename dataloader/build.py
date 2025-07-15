# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from __future__ import annotations

import os
from re import L
from typing import Any, Tuple
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from .samplers import SubsetRandomSampler
from utils.miscellaneous import collate_variable_3d_batch_nomask, worker_init_fn
from .sereact_data_loader import SereactDataloader
from .augmentations import SereactAugmentation


def build_loader(config: Any) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """Build data loaders for training and validation.

    Args:
        config: Configuration object containing data parameters

    Returns:
        Tuple[Dataset, Dataset, DataLoader, DataLoader]: Train dataset, test dataset, train loader, test loader
    """
    config.defrost()
    dataset_train = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build train dataset")

    dataset_test= build_dataset(is_train=False, config=config)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    # if config.data.zip_mode and config.data.cache_mode == 'part':
    #     indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
    #     sampler_train = SubsetRandomSampler(indices)
    # else:
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    # if config.TEST.SEQUENTIAL:
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_test)
    # else:
    sampler_val = torch.utils.data.distributed.DistributedSampler(
        dataset_test, shuffle=False
    )


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_variable_3d_batch_nomask,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        worker_init_fn=worker_init_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_val,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_variable_3d_batch_nomask,
    )
    # setup mixup / cutmix
    # mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_test, data_loader_train, data_loader_test


def build_dataset(is_train: bool, config: Any) -> Dataset:
    """Build dataset for training or testing.

    Args:
        is_train: Whether to build training dataset (True) or test dataset (False)
        config: Configuration object containing dataset parameters

    Returns:
        Dataset: The constructed dataset

    Raises:
        NotImplementedError: If dataset type is not supported
    """
    if config.data.dataset == 'Sereact_dataset':
        if config.data.augment:
            transform = SereactAugmentation() if is_train else None
        else:
            transform = None
                    
        task = 'train' if is_train else 'test'
        dataset = SereactDataloader(
            source_path=config.data.data_path,
            task=task,
            transform=transform,
        )
    else:
        raise NotImplementedError("Dataset Not Available")
    return dataset