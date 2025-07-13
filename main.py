# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import wandb
import pickle as pkl
from PIL import Image


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as T
from losses.loss_3ddetr import LossFunction
from models.detr3d.model_3ddetr import build_3ddetr_model
# from utils.low_precision_conversion import convert_model_to_low_precision
from utils.mean_iou_evaluation import IoUEvaluator
from torch import optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt 
from torchvision import transforms
import seaborn as sns
from optimizer import build_optimizer
from utils.mean_iou_evaluation import IoUEvaluator

from timm.utils import accuracy, AverageMeter

from config import get_config
from dataloader import build_loader

from logger import create_logger
from utils_help import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

transform = T.ToPILImage()
# res_bi = transforms.Resize(size=(640, 768), interpolation=Image.BILINEAR)
# res_n = transforms.Resize(size=(640, 768), interpolation=Image.NEAREST)

T = transforms.ToTensor()
pil = transforms.ToPILImage()

data_dic = {}

wandb.init(project="sereact project", entity='padfoot')

def parse_option():
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
    parser.add_argument('--unit_test', action='store_true', help='Test throughput only')
    parser.add_argument('--base_lr', type=float , help="base learning rate")

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    base_lr = config.train.base_lr
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    
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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=True, broadcast_buffers=False)
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
    # breakpoint()

    max_miou = 0.0

    if config.train.auto_resume:
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            if config.model.resume:
                logger.warning(f"auto-resume changing resume file from {config.model.resume} to {resume_file}")
            config.defrost()
            config.model.resume = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.output}, ignoring auto resume')


    if config.model.resume:
        max_miou = load_checkpoint(config, model_without_ddp, optimizer, scheduler, loss_scaler, logger)
        miou, loss = validate(config, loss_module, epoch, iou_evaluator, data_loader_val, model)
        logger.info(f"Mean iou of the network on the {len(dataset_val)} test images: {miou:.4f}")
        if config.eval_mode:
            return

    if config.model.training:
        logger.info("Start training")
        start_time = time.time()
        for epoch in range(config.train.start_epoch, config.train.max_epoch):
            data_loader_train.sampler.set_epoch(epoch)

            train_one_epoch(config, model, loss_module, iou_evaluator, data_loader_train, optimizer, epoch,scheduler,
                            loss_scaler)

            miou, loss = validate(config, loss_module, epoch, iou_evaluator, data_loader_val, model)
            
            # if dist.get_rank() == 0 and (epoch % config.save_freq == 0 or epoch == (config.train.max_epoch - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_miou, miou, optimizer, scheduler, loss_scaler,
                            logger)

            logger.info(f"Mean IOU of the network on the {len(dataset_val)} test images: {miou:.4f}%")
            max_miou = max(max_miou, miou)
            logger.info(f'Max miou: {max_miou:.4f}%')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

    if config.model.unit_test:
        logger.info("Start training on one batch")
        start_time = time.time()
        for epoch in range(config.train.start_epoch, config.train.unit_test_epoch):
            data_loader_train.sampler.set_epoch(epoch)

            count = train_one_epoch(config, model, loss_module, iou_evaluator, data_loader_train, optimizer, epoch,scheduler,
                            loss_scaler)
            
            logger.info(f'Unit test count: {count:.4f}%')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, loss_module, iou_evaluator, data_loader, optimizer, epoch, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    miou_meter = AverageMeter()
    iter_num = 0
    max_iterations = config.train.max_epoch*num_steps
    start = time.time()
    end = time.time()
    # breakpoint()
    for batch_idx, batch in enumerate(data_loader):
        # breakpoint()
        # Move input data to the specified device
        inputs = [obj.cuda() for obj in batch['pcd_tensor']]
        gt_bboxes = [obj.cuda() for obj in batch['bbox3d_tensor']]
        pcd_dims_min = [obj.cuda() for obj in batch['point_cloud_dims_min']]
        pcd_dims_max = [obj.cuda() for obj in batch['point_cloud_dims_max']]
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        
        outputs = model(
            inputs,
            point_cloud_dims_min=pcd_dims_min,
            point_cloud_dims_max=pcd_dims_max,
        )
        # breakpoint()
        output = outputs[0]
        gt_bbox = gt_bboxes[0]
        # breakpoint()
        # Get predictions
        pred_boxes = output['outputs']
        pred_boxes_aux = output['auxiliary_outputs']
        # Compute loss
        loss, loss_dict, assignments = loss_module(pred_boxes, gt_bbox)
        loss_aux = 0
        for aux in pred_boxes_aux:
            loss_aux_cls, loss_dict_aux, assignments_aux = loss_module(aux, gt_bbox)
            loss_aux += loss_aux_cls
        total_loss = loss + 0.2*loss_aux
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.train.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(batch_idx + 1) % config.train.accumulation_steps == 0)
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"⚠️ Unused parameter: {name}")

        # breakpoint()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"⚠️ Unused parameter: {name}")
        if (batch_idx + 1) % config.train.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step()
        loss_scale_value = loss_scaler.state_dict()["scale"]
        loss_meter.update(total_loss.item())
        

        predicted_bboxes_matched, gt_bboxes_matched = (
                get_predicted_and_gt_boxes_from_assignments(
                    pred_boxes=pred_boxes, assignments=assignments, gt_bbox=gt_bbox
                )
            )

        # Update IoU evaluator
        # breakpoint()
        iou_evaluator.update(predicted_bboxes_matched, gt_bboxes_matched)

        # breakpoint()
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
        # breakpoint()
    metrics = iou_evaluator.compute_metrics()

            
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, loss_module, epoch, iou_evaluator, data_loader, model):
    
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    # metric_perclass = MulticlassJaccardIndex(num_classes=config.MODEL.NUM_CLASSES, average=None).cuda()
    # metric = MultilabelJaccardIndex(num_classes=config.MODEL.NUM_CLASSES).cuda()
    # evaluator.reset()
    end = time.time()
    iou_evaluator.reset()
    # metrics = {'val_iou': {}, 'val_loss': 0.0}
    total_loss = 0.0
    num_batches = len(data_loader)
    time_per_batch = []

    logger.info('Starting validation...')
    for _, batch_data in enumerate(data_loader):
        start_time = time.time()

        # Move input data to the specified device
        inputs = [obj.cuda() for obj in batch_data['pcd_tensor']]
        gt_bboxes = [obj.cuda() for obj in batch_data['bbox3d_tensor']]
        pcd_dims_min = [obj.cuda() for obj in batch_data['point_cloud_dims_min']]
        pcd_dims_max = [obj.cuda() for obj in batch_data['point_cloud_dims_max']]

        # Forward pass
        # inputs = {"point_clouds": batch_data["pcd"]}
        outputs = model(
            inputs,
            point_cloud_dims_min=pcd_dims_min,
            point_cloud_dims_max=pcd_dims_max,
        )

        # Unpack the output from the list
        output = outputs[0]
        gt_bbox = gt_bboxes[0]

        # Get predictions
        pred_boxes = output['outputs']

        # Compute loss (if criterion is provided)
        # if criterion:
        loss, loss_dict, assignments = loss_module(outputs=pred_boxes, targets=gt_bbox)

        loss_meter.update(loss.item())
        if iou_evaluator:
            predicted_bboxes_matched, gt_bboxes_matched = (
                get_predicted_and_gt_boxes_from_assignments(
                    pred_boxes=pred_boxes, assignments=assignments, gt_bbox=gt_bbox
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

def test_overfit_on_single_sample(config, model, loss_module, iou_evaluator, data_loader, optimizer, epoch, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    miou_meter = AverageMeter()
    iter_num = 0
    max_iterations = config.train.max_epoch*num_steps
    start = time.time()
    end = time.time()
    for batch_idx, single_batch in enumerate(data_loader):

       # Move input data to the specified device
        inputs = [single_batch['pcd_tensor'][0].cuda()]
        gt_bboxes = [single_batch['bbox3d_tensor'][0].cuda()]
        pcd_dims_min = [single_batch['point_cloud_dims_min'][0].cuda()]
        pcd_dims_max = [single_batch['point_cloud_dims_max'][0].cuda()]
        
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        outputs = model(
            inputs,
            point_cloud_dims_min=pcd_dims_min,
            point_cloud_dims_max=pcd_dims_max,
        )
        
        output = outputs[0]
        gt_bbox = gt_bboxes[0]
        # breakpoint()

        # Get predictions
        pred_boxes = output['outputs']

        # Compute loss
        loss, loss_dict, assignments = loss_module(pred_boxes, gt_bbox)
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(batch_idx + 1) % config.train.accumulation_steps == 0)
        if (batch_idx + 1) % config.train.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step()

        loss_scale_value = loss_scaler.state_dict()["scale"]

        

        predicted_bboxes_matched, gt_bboxes_matched = (
                get_predicted_and_gt_boxes_from_assignments(
                    pred_boxes=pred_boxes, assignments=assignments, gt_bbox=gt_bbox
                )
            )

        # Update IoU evaluator
        iou_evaluator.update(predicted_bboxes_matched, gt_bboxes_matched)

        metrics = iou_evaluator.compute_metrics()

        if metrics['mean_iou'] > 0.25:
            print(f'Mean IoU value above 0.25: {metrics["mean_iou"]}')
            # self.visualize_box_distributions(predicted_bboxes_matched, gt_bboxes_matched)
            count += 1

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
            logger.info(
                f'Train: [{epoch}/{config.train.max_epoch}][{batch_idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return count


def get_predicted_and_gt_boxes_from_assignments(
        pred_boxes: dict, assignments: dict, gt_bbox: torch.Tensor
    ) -> tuple:
        """
        Extracts matched predicted and ground truth bounding boxes based on assignments.

        Args:
            pred_boxes (dict): Dictionary containing predicted bounding boxes.
            assignments (dict): Dictionary containing assignment indices.
            gt_bbox (torch.Tensor): Ground truth bounding boxes.

        Returns:
            tuple: Matched predicted and ground truth bounding boxes.
        """
        # Get the predicted and gt indices
        matched_predicted_indices = assignments['assignments'][0][0]
        matched_gt_indices = assignments['assignments'][0][1]

        # Move to CPU
        matched_predicted_indices = matched_predicted_indices.cpu().detach().numpy()
        matched_gt_indices = matched_gt_indices.cpu().detach().numpy()

        # Index-slicing to get matched boxes
        predicted_bboxes_matched = pred_boxes['box_corners'][0, matched_predicted_indices]
        gt_bboxes_matched = gt_bbox[matched_gt_indices]

        # Move to CPU
        predicted_bboxes_matched = predicted_bboxes_matched.cpu().detach().numpy()
        gt_bboxes_matched = gt_bboxes_matched.cpu().detach().numpy()

        return predicted_bboxes_matched, gt_bboxes_matched


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
    # linear scale the learning rate according to total batch size, may not be optimal
    # if args.batch_size != 24 and args.batch_size % 6 == 0:
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0 ## change it based on the scheduler performance
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = config.TRAIN.BASE_LR * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    
    # config.defrost()
    # config.TRAIN.BASE_LR = config.TRAIN.BASE_LR
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.freeze()

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
