import torch
from torch import nn
from tqdm import tqdm
from typing import Dict, Any
from core.config import CfgNode
from torch.utils.data import DataLoader
from core.engine.loss import DetectionLoss
from core.utils.tensorboard import select_samples
from torchmetrics.detection import MeanAveragePrecision
from core.utils.ops import non_max_suppression, xyxy2xywh
from core.data.transforms.transforms import (
    Denormalize,
    ToNumpy,
    ToTensor,
    Compose
)


@torch.no_grad()
def do_validation(cfg: CfgNode,
                  model: nn.Module,
                  data_loader: DataLoader,
                  device: torch.device) -> Dict[str, Any]:
    # create metrics
    strides = model.get_strides()
    num_classes=model.get_num_classes()
    det_loss = DetectionLoss(num_classes=num_classes,
                             strides=strides,
                             dfl_bins=model.get_dfl_num_bins(),
                             loss_box_k=cfg.LOSS.BOX_WEIGHT,
                             loss_dfl_k=cfg.LOSS.DFL_WEIGHT,
                             loss_cls_k=cfg.LOSS.CLS_WEIGHT,
                             device=device,
                             tal_topk=cfg.LOSS.TAL_TOPK)
    map_metric = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')
    # MeanAveragePrecision.warn_on_many_detections=False
    # TODO: extended_summary=True to get recall, precision

    # tensorboard image transforms
    tb_img_transforms = [
        ToNumpy(),
        Denormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE),
        ToTensor()
    ]
    tb_img_transforms = Compose(tb_img_transforms)

    # create stats
    stats = {
        'loss_sum': 0,
        'loss_box_sum': 0,
        'loss_dfl_sum': 0,
        'loss_cls_sum': 0,
        'map_50': 0,
        'best_samples': [],
        'worst_samples': [],
        'iterations': 0
    }

    # gather stats
    for data_entry in tqdm(data_loader):
        # get data
        images = data_entry["img"].to(device)       # (B, T, C, H, W)
        bboxes = data_entry["bbox"].to(device)      # (B, T, max_targets, 4)
        classes = data_entry["cls"].to(device)      # (B, T, max_targets, num_classes)

        # reshape
        bboxes = bboxes.flatten(0, 1)               # (B*T, max_targets, 4)
        classes = classes.flatten(0, 1)             # (B*T, max_targets, num_classes)
        targets = torch.cat([bboxes, classes], -1)  # (B*T, max_targets, 4 + num_classes)

        # forward model
        output_y, output_x = model(images)          # 3 x (B*T, C, Hi, Wi)

        # TODO: multi_label = True ?
        outputs = non_max_suppression(output_y, conf_thres=0.005, iou_thres=0.45, nc=num_classes, in_place=False) # (num_boxes, 4 + score + label), xyxy

        # calculate loss
        losses = det_loss(output_x, targets)
        loss = losses[0] / cfg.SOLVER.GRAD_ACCUM_ITERS
        loss_box, loss_cls, loss_dfl = losses[1]

        # calculate mAP
        imgsz = torch.tensor(output_x[0].shape[2:], device=device, dtype=bboxes.dtype) * strides[0] # image size (h, w)

        frame_preds, frame_targets, frame_metric = [], [], []
        for frame_idx in range(bboxes.shape[0]):
            gt_bboxes = bboxes[frame_idx]                       # (max_targets, 4)
            gt_classes = classes[frame_idx]                     # (max_targets, num_classes)

            # remove padding
            nonzero_boxes = gt_bboxes.sum(-1).gt(0.0)
            nonzero_boxes_idxs = torch.where(nonzero_boxes == True)
            gt_bboxes = gt_bboxes[nonzero_boxes_idxs]               # (num_boxes, 4)
            gt_classes = gt_classes[nonzero_boxes_idxs]             # (num_boxes, num_classes)

            # classes to one-hot encode
            gt_labels_idxs = gt_classes.argmax(-1, keepdim=True)    # (num_boxes, num_classes), int64
            gt_labels = gt_labels_idxs.squeeze(-1)                  # (num_boxes), int64
            frame_targets.append(dict(
                boxes=gt_bboxes.mul(imgsz[[1, 0, 1, 0]]),           # scale boxes to image size
                labels=gt_labels
            ))

            # prepare predictions (after NMS)
            preds_bboxes = xyxy2xywh(outputs[frame_idx][:, :4])
            preds_scores = outputs[frame_idx][:, 4]
            preds_labels = outputs[frame_idx][:, 5].to(torch.long)
            frame_preds.append(dict(
                boxes=preds_bboxes,
                labels=preds_labels,
                scores=preds_scores
            ))

            # calculate mAP metric per frame
            frame_map_metric = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')
            frame_map_metric.update([frame_preds[-1]], [frame_targets[-1]])
            frame_metric.append(frame_map_metric.compute())

        # calculate total mAP metric
        map_metric.update(frame_preds, frame_targets)

        # select best and worst samples
        images = images.flatten(0, 1) # (B*T, C, H, W)

        if len(frame_metric):
            frame_metric = torch.stack([m['map_50'] for m in frame_metric], 0).to(device)

            select_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                           accumulator=stats['best_samples'],
                           images=images.detach(),
                           targets=targets.detach(),
                           preds=output_y.permute(0, 2, 1).detach(),
                           metric=frame_metric,
                           conf_thresh=cfg.TENSORBOARD.CONF_THRESH,
                           min_metric_better=False,
                           image_transforms=tb_img_transforms)

            select_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                           accumulator=stats['worst_samples'],
                           images=images.detach(),
                           targets=targets.detach(),
                           preds=output_y.permute(0, 2, 1).detach(),
                           metric=frame_metric,
                           conf_thresh=cfg.TENSORBOARD.CONF_THRESH,
                           min_metric_better=True,
                           image_transforms=tb_img_transforms)

        # update stats
        stats['loss_sum'] += loss.item()
        stats['loss_box_sum'] += loss_box.item()
        stats['loss_dfl_sum'] += loss_dfl.item()
        stats['loss_cls_sum'] += loss_cls.item()
        stats['iterations'] += 1

    map_results = map_metric.compute()
    stats['map_50'] = map_results['map_50']
    stats['map_75'] = map_results['map_75']

    return stats
