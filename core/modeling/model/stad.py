import torch
from torch import nn
from core.config import CfgNode
from core.modeling.head import build_head
from core.modeling.backbone2d import build_backbone2d


class STAD(nn.Module):
    def __init__(self,
                 backbone2d: nn.Module,
                 head: nn.Module):
        super().__init__()
        self.backbone2d = backbone2d
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone2d(x)
        y = self.head(f)
        return y


def build_stad(cfg: CfgNode) -> nn.Module:
    # build backbone
    backbone2d = build_backbone2d(cfg)

    # evaluate channels and strides
    channels = []
    strides = []
    x = torch.zeros(size=(1, 3, 256, 256), dtype=torch.float32)
    features = backbone2d(x)
    for f in features:
        channels.append(f.shape[1])
        strides.append(x.shape[-2] / f.shape[-2])

    # build head
    head = build_head(cfg, channels=channels, strides=strides)

    return STAD(backbone2d, head)
