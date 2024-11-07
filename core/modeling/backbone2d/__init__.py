from torch import nn
from core.config import CfgNode
from core.modeling.backbone2d.yolov8 import build_yolov8


def build_backbone2d(cfg: CfgNode, *args, **kwargs) -> nn.Module:
    arch = cfg.MODEL.BACKBONE2D.ARCHITECTURE

    if arch.startswith("yolov8"):
        backbone2d = build_yolov8(cfg)
    else:
        raise ValueError(f"Backbone2d architecture '{arch}' not found")
    
    return backbone2d
