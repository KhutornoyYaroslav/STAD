import torch
from torch import nn
from typing import List
from core.config import CfgNode
from core.modeling.head import build_head
from core.modeling.backbone2d import build_backbone2d
from core.modeling.backbone3d import build_backbone3d
from core.modeling.attention.cfam import CFAMFusion


class STAD(nn.Module):
    def __init__(self,
                 backbone2d: nn.Module,
                 backbone3d: nn.Module,
                 feature_fusion: nn.Module,
                 head: nn.Module):
        super().__init__()
        self.backbone2d = backbone2d
        self.backbone3d = backbone3d
        self.feature_fusion = feature_fusion
        self.head = head

    def forward(self,
                clip: torch.Tensor,
                keyframe: torch.Tensor) -> torch.Tensor:
        """
        args:
            clip (torch.Tensor) with shape (b, c, t, h, w)
            keyframe (torch.Tensor) with shape (b, c, h, w)
        """
        # get 2d features
        f2d = self.backbone2d(keyframe)

        # get 3d features
        f3d = self.backbone3d(clip)
        f3d = f3d.squeeze(2)

        # fuse 2d-3d features
        f = self.feature_fusion(f2d, f3d)

        # class head
        # y = self.head(f)
        y = self.head(list(f))
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


def build_stad(cfg: CfgNode) -> nn.Module:
    # build 2d backbone
    backbone2d = build_backbone2d(cfg)

    # evaluate channels and strides
    channels2d = []
    strides = []
    x = torch.zeros(size=(1, 3, 256, 256), dtype=torch.float32) # (b, c, h, w)
    features2d = backbone2d(x)
    for f in features2d:
        channels2d.append(f.shape[1])
        strides.append(x.shape[-2] / f.shape[-2])

    # build 3d backbone
    backbone3d = build_backbone3d(cfg)

    # evaluate channels
    x = torch.zeros(size=(1, 3, 5, 256, 256), dtype=torch.float32) # (b, c, t, h, w)
    features3d = backbone3d(x)
    channels3d = features3d.shape[1]

    # build channel fusion module
    interchannels = cfg.MODEL.FEATURE_FUSION.INTER_CHANNELS
    feature_fusion = CFAMFusion(channels2d, channels3d, interchannels, 'coupled')

    # build head
    head_in_channels = len(channels2d) * [interchannels]
    head = build_head(cfg, channels=head_in_channels, strides=strides)

    model = STAD(backbone2d, backbone3d, feature_fusion, head)
    initialize_weights(model)

    return model
