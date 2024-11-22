import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from core.utils.ops import xywh2xyxy
from core.utils.metrics import bbox_iou
from core.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors


class DFLoss(nn.Module):
    def __init__(self, bins: int = 16) -> None:
        super().__init__()
        self.bins = bins

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.bins - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, dfl_bins: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(dfl_bins) if dfl_bins > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # dfl
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.bins - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.bins), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class DetectionLoss:
    def __init__(self,
                 num_classes: int,
                 strides: List[float],
                 dfl_bins: int,
                 loss_box_k: float,
                 loss_dfl_k: float,
                 loss_cls_k: float,
                 device: torch.device,
                 tal_topk: int = 10,
                 ):
        self.loss_box_k = loss_box_k
        self.loss_dfl_k = loss_dfl_k
        self.loss_cls_k = loss_cls_k

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = strides
        self.nc = num_classes
        self.no = num_classes + dfl_bins * 4
        self.dfl_bins = dfl_bins
        self.device = device
        self.use_dfl = dfl_bins > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.dfl_bins).to(device)
        self.proj = torch.arange(self.dfl_bins, dtype=torch.float, device=device)

    # def preprocess(self, targets, batch_size, scale_tensor):
    #     """Preprocesses the target counts and matches with the input batch size to output a tensor."""
    #     nl, ne = targets.shape # (n_targets, 1 + 4 + num_classes)
    #     if nl == 0:
    #         out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
    #     else:
    #         i = targets[:, 0]  # images indexes
    #         _, counts = i.unique(return_counts=True)
    #         counts = counts.to(dtype=torch.int32)
    #         out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
    #         for j in range(batch_size):
    #             matches = i == j
    #             n = matches.sum()
    #             if n:
    #                 out[j, :n] = targets[matches, 1:] # (bs, num_max_boxes, 4 + num_classes)
    #         out[..., :4] = xywh2xyxy(out[..., :4].mul_(scale_tensor))
    #     return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, targets):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.dfl_bins * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        # (n_targets, 1 + 4 + num_classes) = (n_targets, 1) + (n_targets, 4) + (n_targets, num_classes)
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["bboxes"], batch["cls"]), 1)
        # (bs, n_max_boxes, cls + 4), scale_tensor = (w, h, w, h)
        # targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        gt_bboxes, gt_scores = targets.split((4, self.nc), 2)  # xywh, num_classes
        gt_bboxes = xywh2xyxy(gt_bboxes) # xyxy
        gt_bboxes = gt_bboxes.mul_(imgsz[[1, 0, 1, 0]]) # scaled to image size
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_scores,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.loss_box_k  # box gain
        loss[1] *= self.loss_cls_k  # cls gain
        loss[2] *= self.loss_dfl_k  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
