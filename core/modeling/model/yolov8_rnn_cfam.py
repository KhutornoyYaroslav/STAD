import torch
from torch import nn
from typing import List
from core.config import CfgNode
from core.modeling.head import build_head
from core.modeling.backbone2d import build_backbone2d
from core.modeling.temporal import TemporalFusion
from core.modeling.attention import CFAMFusion2


class YOLOv8RNNCFAM(nn.Module):
    def __init__(self,
                 backbone2d: nn.Module,
                 head: nn.Module,
                 temporal_fusion: nn.Module,
                 cfam_fusion: nn.Module):
        super(YOLOv8RNNCFAM, self).__init__()
        self.backbone2d = backbone2d
        self.head = head
        self.temporal_fusion = temporal_fusion
        self.cfam_fusion = cfam_fusion

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
        f2d = self.backbone2d(input)

        # from (b*t, ci, hi, wi) to (b, t, ci, hi, wi)
        for i in range(len(f2d)):  
            f2d[i] = f2d[i].view(B, T, *f2d[i].shape[1:])

        # temporal fusion
        f3d = self.temporal_fusion(f2d)   # (b, t, ci, hi, wi)

        # from (b, t, ci, hi, wi) to  (b*t, ci, hi, wi)
        assert len(f2d) == len(f3d)
        for i in range(len(f2d)):
            f2d[i] = f2d[i].view(-1, *f2d[i].shape[2:])
            f3d[i] = f3d[i].view(-1, *f3d[i].shape[2:])

        # cfam fusion
        f = self.cfam_fusion(f2d, f3d)

        # class head
        y = self.head(f)

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


def build_yolov8rnncfam(cfg: CfgNode) -> nn.Module:
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
        learnable_init_state=cfg.MODEL.TEMPORAL_FUSION.LEARNABLE_INIT_STATE,
        residual_type=cfg.MODEL.TEMPORAL_FUSION.RESIDUAL_TYPE)

    # build cfam fusion
    channels3d = channels2d
    interchannels = cfg.MODEL.FEATURE_FUSION.INTER_CHANNELS
    feature_fusion = CFAMFusion2(channels2d, channels3d, interchannels)

    # build head
    head_in_channels = len(channels2d) * [interchannels]
    head = build_head(cfg, channels=head_in_channels, strides=strides)

    # build model
    model = YOLOv8RNNCFAM(
        backbone2d,
        head,
        temporal_fusion,
        feature_fusion)

    initialize_weights(model)

    return model
