import torch
from torch import nn
from typing import List
from core.config import CfgNode
from core.modeling.head import build_head
from core.modeling.backbone2d import build_backbone2d


class YOLOv8(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        args:
            input (torch.Tensor) with shape (b, t, c, h, w)

        returns:
            3 x (b*t, c, hi, wi)
        """
        input = input.flatten(0, 1) # (b*t, c, h, w)

        fs = self.backbone(input)
        y = self.head(list(fs))

        return y
    
    def get_num_classes(self) -> int:
        return self.head.nc
    
    def get_strides(self) -> List[float]:
        return self.head.stride
    
    def get_dfl_num_bins(self) -> int:
        return self.head.dfl_bins


def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


def build_yolov8(cfg: CfgNode) -> nn.Module:
    # build backbone
    backbone = build_backbone2d(cfg)

    # evaluate channels and strides
    channels = []
    strides = []
    x = torch.zeros(size=(1, 3, 256, 256), dtype=torch.float32) # (b, c, h, w)
    features = backbone(x)
    for f in features:
        channels.append(f.shape[1])
        strides.append(x.shape[-2] / f.shape[-2])

    # build head
    head = build_head(cfg, channels=channels, strides=strides)

    # build model
    model = YOLOv8(backbone, head)
    initialize_weights(model)

    return model
