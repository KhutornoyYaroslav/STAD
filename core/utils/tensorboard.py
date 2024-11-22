import torch
import cv2 as cv
import numpy as np
from numbers import Number
from typing import List, Tuple, Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from core.utils.ops import xywh2xyxy
from core.data.transforms.transforms import BaseTransform


@torch.no_grad()
def add_metrics(summary_writer: SummaryWriter,
                scalars: Dict[str, Number],
                samples: Dict[str, List],
                global_step: int,
                is_train: bool = False):
    if summary_writer is None:
        return

    prefix = 'train' if is_train else 'val'

    # scalars
    for key, val in scalars.items():
        summary_writer.add_scalar(f'{prefix}/{key}', val, global_step=global_step)

    # samples
    for key in samples.keys():
        if len(samples[key]):
            images = [s[1] for s in samples[key]]
            image_grid = torch.concatenate(images, dim=1)
            summary_writer.add_image(f"{prefix}/{key}", image_grid, global_step=global_step, dataformats='CHW')

    summary_writer.flush()


@torch.no_grad()
def draw_objects(image: torch.Tensor, # (C, H, W)
                 box: torch.Tensor, # (num_objects, 4)
                 cls: torch.Tensor, # (num_objects, num_classes)
                 conf_thresh: float) -> torch.Tensor: # (3, H, W)
    assert len(box) == len(cls)

    # filter by cls score
    idxs = torch.arange(box.shape[0], dtype=torch.long, device=box.device)
    cls_mask = cls.amax(-1).gt_(conf_thresh).bool()
    idxs = idxs[cls_mask]

    # draw
    box_to_draw = torch.index_select(box, 0, idxs)

    if len(box_to_draw):
        result = draw_bounding_boxes(image, xywh2xyxy(box_to_draw), colors=(0, 255, 0))
    else:
        result = image.clone()

    return result.to(image.device)


@torch.no_grad()
def create_tensorboard_sample_collage(image: torch.Tensor, # (C, H, W), [0 - 255]
                                      preds: torch.Tensor, # (num_preds, 4 + num_classes)
                                      targets: torch.Tensor, # (max_targets, 4 + num_classes)
                                      conf_thresh: float) -> torch.Tensor:
    assert preds.shape[-1] == targets.shape[-1]

    # to color
    if image.shape[0] == 1:
        image = image.expand((3, -1, -1))
    image = image.type(torch.uint8)

    # prepare box, cls
    num_classes = targets.shape[-1] - 4
    preds_box, preds_cls = preds.split((4, num_classes), -1)        # (B, num_preds, 4), (B, num_preds, num_classes)
    targets_box, targets_cls = targets.split((4, num_classes), -1)  # (B, max_targets, 4), (B, max_targets, num_classes)
    imgsz = torch.tensor(image.shape[1:], device=targets_box.device, dtype=targets_box.dtype) # (H, W)
    targets_box = targets_box.mul_(imgsz[[1, 0, 1, 0]])  

    # draw objects
    preds_image = draw_objects(image, preds_box, preds_cls, conf_thresh)
    targets_image = draw_objects(image, targets_box, targets_cls, conf_thresh)

    return torch.cat([image, targets_image, preds_image], dim=-1)


@torch.no_grad()
def select_samples(limit: int,
                   accumulator: List[Tuple[float, torch.Tensor]],
                   image: torch.Tensor, # (B, C, H, W)
                   targets: torch.Tensor, # (B, max_targets, 4 + num_classes)
                   preds: torch.Tensor, # (B, num_preds, 4 + num_classes)
                   metric: torch.Tensor, # (B)
                   conf_thresh: float,
                   min_metric_better: bool,
                   image_transforms: Optional[BaseTransform] = None) -> None:
    # select only nonzero samples
    num_classes = targets.shape[-1] - 4
    targets_box, _ = targets.split((4, num_classes), -1) # (B, max_targets, 4)
    batch_idxs = torch.arange(targets_box.shape[0], dtype=torch.long, device=targets_box.device)
    batch_idxs = batch_idxs[targets_box.sum((1, 2)).gt_(0.0).bool()]

    if not torch.numel(batch_idxs):
        return

    # find best metric
    metric_per_sample = torch.index_select(metric, 0, batch_idxs)
    if min_metric_better:
        choosen_idx = torch.argmin(metric_per_sample).item()
    else:
        choosen_idx = torch.argmax(metric_per_sample).item()
    choosen_metric = metric_per_sample[choosen_idx].item()

    # check if better than existing
    need_save = True
    id_to_remove = None
    if len(accumulator) >= limit:
        if min_metric_better:
            id_to_remove = max(range(len(accumulator)), key=lambda x : accumulator[x][0])
            if choosen_metric > accumulator[id_to_remove][0]:
                need_save = False
        else:
            id_to_remove = min(range(len(accumulator)), key=lambda x : accumulator[x][0])
            if choosen_metric < accumulator[id_to_remove][0]:
                need_save = False

    if need_save:
        # prepare tensorboard collage
        best_image = torch.index_select(image, 0, batch_idxs)[choosen_idx]
        best_preds = torch.index_select(preds, 0, batch_idxs)[choosen_idx]
        best_targets = torch.index_select(targets, 0, batch_idxs)[choosen_idx]

        if image_transforms:
            best_image = image_transforms({"img": best_image})["img"]

        collage = create_tensorboard_sample_collage(best_image,
                                                    best_preds,
                                                    best_targets,
                                                    conf_thresh)
        # add to best collages
        if id_to_remove != None:
            accumulator.pop(id_to_remove)
        accumulator.append((choosen_metric, collage))
