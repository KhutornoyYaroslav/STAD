import tqdm
import torch
import argparse
import numpy # TODO: to np
import numpy as np
from torch import nn
from core.config import CfgNode, cfg
from core.data import make_data_loader
from core.utils.checkpoint import CheckPointer
from core.modeling.model.stad import build_stad
from core.utils.ops import non_max_suppression, xywh2xyxy
# from core.utils.metrics import bbox_iou 


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            #x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            #ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
            ap[ci, j] = numpy.trapz(m_pre, m_rec)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def box_iou(box1, box2): # TODO: change to core.utils.metrics.bbox_iou
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)


@torch.no_grad()
def eval_model(cfg: CfgNode):
    # create device
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_stad(cfg)
    model.to(device)
    model.eval()

    # load weights
    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)

    # checkpointer = CheckPointer(model)
    # checkpointer.load("outputs/train_ucf_2/model_037950.pth")

   

    # create data loader
    data_loader = make_data_loader(cfg, is_train=False)
    if data_loader is None:
        print(f"Failed to create dataset loader.")
        return None
    
    # configurate metric
    iou_v = torch.tensor([0.5]).cuda()
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []

    # eval
    iteration = 0
    pbar = tqdm.tqdm(data_loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    for data_entry in pbar:
        iteration += 1
        # get data
        images = data_entry["img"].to(device)                   # (B, T, C, H, W)
        bboxes = data_entry["bbox"].to(device)                  # (B, T, max_targets, 4)
        classes = data_entry["cls"].to(device)                  # (B, T, max_targets, num_classes)

        cur_image = images[:, -1]                               # (B, C, H, W)
        cur_bboxes = bboxes[:, -1]                               # (B, max_targets, 4)
        cur_classes = classes[:, -1]                             # (B, max_targets, num_classes)
        # cur_targets = torch.cat([cur_bboxes, cur_classes], -1)  # (B, max_targets, 4 + num_classes)
        clip = images.permute(0, 2, 1, 3, 4)                    # (B, T, C, H, W) -> (B, C, T, H, W)

        # infer model
        output_y, _ = model(clip, cur_image)

        # post process outputs
        outputs = non_max_suppression(output_y, conf_thres=0.005, iou_thres=0.5, in_place=False) # list. batch_size * (num_boxes, 4 + score + label), xyxy

        # calc metrics
        imgsz = torch.tensor(images.shape[-2:]).to(device) # (H, W)

        for batch_idx, output in enumerate(outputs):
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)

            # get ground truths
            gt_bboxes = cur_bboxes[batch_idx]                           # (max_targets, 4)
            gt_classes = cur_classes[batch_idx]                         # (max_targets, num_classes)

            # remove padding
            nonzero_boxes = gt_bboxes.sum(-1).gt(0.0)
            nonzero_boxes_idxs = torch.where(nonzero_boxes == True)
            gt_bboxes = gt_bboxes[nonzero_boxes_idxs]                   # (num_boxes, 4)
            gt_classes = gt_classes[nonzero_boxes_idxs]                 # (num_boxes, num_classes)

            # classes to one-hot encode
            gt_labels_idxs = gt_classes.argmax(-1, keepdim=True)        # (num_boxes, num_classes), int64
            gt_labels = gt_labels_idxs #.squeeze(-1)                      # (num_boxes, 1), int64

            # scacle bboxes to pixels
            gt_bboxes = gt_bboxes.mul(imgsz[[1, 0, 1, 0]])
            gt_bboxes = xywh2xyxy(gt_bboxes)

            # no detections
            if output.shape[0] == 0:
                if gt_bboxes.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()

            # evaluate
            if gt_bboxes.shape[0]:
                tbox = gt_bboxes.clone()

                correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((gt_labels, tbox), 1)
                iou = box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]

                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)

            metrics.append((correct, output[:, 4], output[:, 5], gt_labels[:, 0]))

        # Compute metrics
        if iteration % 100 == 0:
            metrics__ = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
            if len(metrics__) and metrics__[0].any():
                tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics__)

            # Print results
            print('%10.3g' * 3 % (m_pre, m_rec, mean_ap), flush=True)

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap), flush=True)

    # Return results
    # model.float()  # for training
    #return map50, mean_ap
    print(map50, flush=True)
    print(flush=True)
    print("=================================================================", flush=True)
    print(flush=True)
    print(mean_ap, flush=True)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection Model Evaluation')
    parser.add_argument("-c", "--config-file", dest="config_file", required=False, type=str,
                        default="configs/cfg.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # create config
    cfg.merge_from_file(args.config_file)
    cfg.SOLVER.BATCH_SIZE = 4 # TODO:
    cfg.freeze()

    # do eval
    eval_model(cfg)

if __name__ == '__main__':
    main()
