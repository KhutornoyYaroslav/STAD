import math
import torch
import logging
import torch.nn as nn
from typing import List
from core.config import CfgNode
from core.modeling.backbone2d.yolov8 import Conv
from core.utils.model_zoo import load_state_dict
from core.utils.tal import make_anchors, dist2bbox


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, bins: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(bins, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(bins, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, bins, 1, 1)) # TODO: as buffer ?
        self.bins = bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assert self.conv.requires_grad == False # TODO: how to proparly check it ?
        b, _, a = x.shape
        x = x.view(b, 4, self.bins, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


# TODO: !!!!!!!!!!!!!!!!!!!!!!!!
# DFL модуль не должен обновлять веса ?? Это надо контролировать, чтобы случайно не поставить requires_grad_(True)

class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    # dynamic = False  # force grid reconstruction
    # export = False  # export mode
    # shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.dfl_bins = 16
        self.no = nc + self.dfl_bins * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.dfl_bins * 4)), max(ch[0], min(self.nc, 100))  # channels

        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3),
                                               Conv(c2, c2, 3),
                                               nn.Conv2d(c2, 4 * self.dfl_bins, 1)) for x in ch)

        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3),
                                               Conv(c3, c3, 3),
                                               nn.Conv2d(c3, self.nc, 1)) for x in ch)

        self.dfl = DFL(self.dfl_bins) if self.dfl_bins > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # if self.training:  # Training path
        #     return x
        y = self._inference(x)
        # return y if self.export else (y, x)
        return y, x # TODO: return only y

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # if self.dynamic or self.shape != shape:
        #     self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        #     self.shape = shape
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        # if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
        #     box = x_cat[:, : self.dfl_bins * 4]
        #     cls = x_cat[:, self.dfl_bins * 4 :]
        # else:
        box, cls = x_cat.split((self.dfl_bins * 4, self.nc), 1)

        # if self.export and self.format in {"tflite", "edgetpu"}:
        #     # Precompute normalization factor to increase numerical stability
        #     # See https://github.com/ultralytics/ultralytics/issues/7371
        #     grid_h = shape[2]
        #     grid_w = shape[3]
        #     grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
        #     norm = self.strides / (self.stride[0] * grid_size)
        #     dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        # else:
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            # TODO: why 640 as magic number! what to do if (H != W)?
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2) # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


def initialize_weights(model: nn.Module):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def build_yolov8(cfg: CfgNode,
                 channels: List[int],
                 strides: List[float]) -> nn.Module:
    logger = logging.getLogger('CORE')
    cfg_head = cfg.MODEL.HEAD

    model = Detect(nc=cfg_head.NUM_CLASSES, ch=channels)

    # strides must be provided before initing bias
    model.stride = torch.tensor(strides)

    # bias must be inited once before training
    # if pretrained weights are provided,
    # bias parameters will be overrided
    model.bias_init()

    # common parameters must be inited always
    initialize_weights(model)

    if cfg_head.PRETRAINED_WEIGHTS:
        # parse state dict
        state_dict = load_state_dict(cfg_head.PRETRAINED_WEIGHTS)
        if "model" in state_dict:
            state_dict = state_dict["model"]
            if isinstance(state_dict, nn.Module):
                state_dict = state_dict.state_dict()

        # try extract head parameters
        yolo_prefix = "model.22."
        yolo_head_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith(yolo_prefix):
                yolo_head_state_dict[name[len(yolo_prefix):]] = param
        if len(yolo_head_state_dict):
            state_dict = yolo_head_state_dict

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info("Head pretrained weights loaded: {0} missing, {1} unexpected".
                    format(len(missing_keys), len(unexpected_keys)))
        assert not len(missing_keys)

    if cfg_head.FREEZE:
        model.requires_grad_(False)

    return model
