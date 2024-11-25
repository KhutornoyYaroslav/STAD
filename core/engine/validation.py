import torch
import logging
from torch import nn
from tqdm import tqdm
from typing import Dict, Any
from core.config import CfgNode
from torch.utils.data import DataLoader
from core.engine.loss import DetectionLoss
from core.utils.tensorboard import select_samples
from torchmetrics.detection import MeanAveragePrecision
from core.utils.ops import non_max_suppression, xyxy2xywh
from core.data.transforms.transforms import Denormalize, ToNumpy, ToTensor, Compose


@torch.no_grad()
def do_validation(cfg: CfgNode,
                  model: nn.Module,
                  data_loader: DataLoader,
                  device: torch.device) -> Dict[str, Any]:
    logger = logging.getLogger("CORE")

    # create metrics
    strides = model.get_strides()
    num_classes=model.get_num_classes()
    det_loss = DetectionLoss(num_classes=num_classes,
                             strides=strides,
                             dfl_bins=model.get_dfl_num_bins(),
                             loss_box_k=cfg.SOLVER.LOSS_BOX_WEIGHT,
                             loss_dfl_k=cfg.SOLVER.LOSS_DFL_WEIGHT,
                             loss_cls_k=cfg.SOLVER.LOSS_CLS_WEIGHT,
                             device=device,
                             tal_topk=cfg.SOLVER.TAL_TOPK)
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
        images = data_entry["img"].to(device)                   # (B, T, C, H, W)
        bboxes = data_entry["bbox"].to(device)                  # (B, T, max_targets, 4)
        classes = data_entry["cls"].to(device)                  # (B, T, max_targets, num_classes)

        cur_image = images[:, -1]                               # (B, C, H, W)
        cur_bboxes = bboxes[:, -1]                              # (B, max_targets, 4)
        cur_classes = classes[:, -1]                            # (B, max_targets, num_classes)
        cur_targets = torch.cat([cur_bboxes, cur_classes], -1)  # (B, max_targets, 4 + num_classes)

        clip = images.permute(0, 2, 1, 3, 4)                    # (B, T, C, H, W) -> (B, C, T, H, W)

        # forward model
        output_y, output_x = model(clip, cur_image)             # 3 x (B, C, Hi, Wi)
        # output_y = output_y.permute(0, 2, 1)                  # (B, num_anchors, 4 + num_classes)

        # TODO: multiclass to multi targets
        # TODO: multi_label=True ?
        outputs = non_max_suppression(output_y, conf_thres=0.25, iou_thres=0.45, nc=num_classes, in_place=False) # (num_boxes, 4 + score + label)

        # calculate loss
        losses = det_loss(output_x, cur_targets)
        loss = losses[0]
        loss_box, loss_cls, loss_dfl = losses[1]

        # calculate mAP
        imgsz = torch.tensor(output_x[0].shape[2:], device=device, dtype=cur_bboxes.dtype) * strides[0] # image size (h, w)

        preds, targets, batch_metric = [], [], []
        for batch_idx in range(cur_image.shape[0]):
            gt_bboxes = cur_bboxes[batch_idx] # (max_targets, 4)
            gt_classes = cur_classes[batch_idx] # (max_targets, num_classes)

            # remove padding
            nonzero_boxes = gt_bboxes.sum(-1).gt_(0.0)
            nonzero_boxes_idxs = torch.nonzero(nonzero_boxes, as_tuple=True)[0]
            gt_bboxes = gt_bboxes[nonzero_boxes_idxs] # (num_boxes, 4)
            gt_classes = gt_classes[nonzero_boxes_idxs] # (num_boxes, num_classes)

            # classes to one-hot encode
            gt_labels_idxs = gt_classes.argmax(-1, keepdim=True) # (num_boxes, num_classes), int64
            gt_labels = gt_labels_idxs.squeeze(-1) # (num_boxes), int64
            targets.append(dict(
                boxes=gt_bboxes.mul_(imgsz[[1, 0, 1, 0]]), # scale boxes to image size
                labels=gt_labels
            ))

            # prepare predictions (after NMS)
            preds_bboxes = xyxy2xywh(outputs[batch_idx][:, :4])
            preds_scores = outputs[batch_idx][:, 4]
            preds_labels = outputs[batch_idx][:, 5].to(torch.long)
            preds.append(dict(
                boxes=preds_bboxes,
                labels=preds_labels,
                scores=preds_scores
            ))

            # calculate mAP metric per image
            batch_map_metric = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')
            batch_map_metric.update([preds[-1]], [targets[-1]])
            batch_metric.append(batch_map_metric.compute())

        # calculate total mAP metric
        map_metric.update(preds, targets)

        # select best and worst samples
        if len(batch_metric):
            batch_metric = torch.stack([m['map_75'] for m in batch_metric], 0).to(device)

            select_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                           accumulator=stats['best_samples'],
                           image=cur_image.detach(),
                           targets=cur_targets.detach(),
                           preds=output_y.permute(0, 2, 1).detach(),
                           metric=batch_metric,
                           conf_thresh=cfg.TENSORBOARD.CONF_THRESH,
                           min_metric_better=False,
                           image_transforms=tb_img_transforms)

            select_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                           accumulator=stats['worst_samples'],
                           image=cur_image.detach(),
                           targets=cur_targets.detach(),
                           preds=output_y.permute(0, 2, 1).detach(),
                           metric=batch_metric,
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
