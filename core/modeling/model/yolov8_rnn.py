import torch
from torch import nn
from typing import List
from core.config import CfgNode
from core.modeling.head import build_head
from core.modeling.backbone2d import build_backbone2d
from core.modeling.temporal import TemporalFusion


class YOLOv8RNN(nn.Module):
    def __init__(self,
                 backbone2d: nn.Module,
                 head: nn.Module,
                 temporal_fusion: nn.Module):
        super().__init__()
        self.backbone2d = backbone2d
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
        input = input.view(-1, C, H, W)                          # from (b, t, c, h, w) to (b*t, c, h, w)
        f2d = self.backbone2d(input)

        # reshape
        f2d_reshaped = []
        for f in f2d:
            _, F_C, F_H, F_W = f.shape
            f2d_reshaped.append(f.view(B, T, F_C, F_H, F_W))     # from (b*t, ci, hi, wi) to (b, t, ci, hi, wi)

        # temporal fusion
        temporal_f2ds = self.temporal_fusion(f2d_reshaped)          # (b, t, ci, hi, wi)

        # class head
        temporal_f2ds_reshaped = []
        for f in temporal_f2ds:
            B, T, F_C, F_H, F_W = f.shape
            f = f.view(-1, F_C, F_H, F_W)                        # from (b, t, c, h, w) to (b*t, c, h, w)
            temporal_f2ds_reshaped.append(f)
        y = self.head(temporal_f2ds_reshaped)

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
    temporal_fusion = TemporalFusion(
        in_channels=channels2d,
        hidden_channels=channels2d,
        stateful_training=cfg.MODEL.TEMPORAL_FUSION.STATEFUL_TRAINING,
        learnable_init_state=cfg.MODEL.TEMPORAL_FUSION.LEARNABLE_INIT_STATE)

    # build head
    head = build_head(cfg, channels=channels2d, strides=strides)

    # build model
    model = YOLOv8RNN(backbone2d, head, temporal_fusion)
    initialize_weights(model)

    return model
