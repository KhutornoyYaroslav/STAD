import torch
from torch import nn
from typing import List
from core.config import CfgNode
from core.modeling.head import build_head
from core.modeling.backbone2d import build_backbone2d
from core.modeling.temporal import build_tempfusion


class YOLOv8RNN(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module,
                 temporal_fusion: nn.Module):
        super(YOLOv8RNN, self).__init__()
        self.backbone = backbone
        self.head = head
        self.temporal_fusion = temporal_fusion

    def prepare_inference(self, img_w: int, img_h: int):
        batch_sizes = []
        for s in self.get_strides():
            batch_sizes.append(int(img_w * img_h / s**2))
        self.temporal_fusion.prepare_inference(batch_sizes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        args:
            input (torch.Tensor) with shape (b, t, c, h, w)

        returns:
            3 x (b*t, c, hi, wi)
        """
        B, T, C, H, W = input.shape

        # get 2d features
        input = input.view(-1, C, H, W)     # from (b, t, c, h, w) to (b*t, c, h, w)
        f2d = self.backbone(input)

        # from (b*t, ci, hi, wi) to (b, t, ci, hi, wi)
        for i in range(len(f2d)):  
            f2d[i] = f2d[i].view(B, T, *f2d[i].shape[1:])

        # temporal fusion
        f3d = self.temporal_fusion(f2d)     # (b, t, ci, hi, wi)

        # from (b, t, c, h, w) to (b*t, c, h, w)
        for i in range(len(f3d)):  
            f3d[i] = f3d[i].view(-1, *f3d[i].shape[2:])

        # class head
        y = self.head(f3d)

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


def build_yolov8rnn(cfg: CfgNode) -> nn.Module:
    # build 2d backbone
    backbone2d = build_backbone2d(cfg)

    # evaluate channels and strides
    channels2d = []
    strides = []
    x = torch.zeros(size=(1, 3, 256, 256), dtype=torch.float32) # (b, c, h, w)
    f2d = backbone2d(x)
    for f in f2d:
        channels2d.append(f.shape[1])
        strides.append(x.shape[-2] / f.shape[-2])

    # build temporal fusion module
    temporal_fusion = build_tempfusion(cfg, channels2d, channels2d)

    # build head
    head = build_head(cfg, channels=channels2d, strides=strides)

    # build model
    model = YOLOv8RNN(backbone2d, head, temporal_fusion)
    initialize_weights(model)

    return model
