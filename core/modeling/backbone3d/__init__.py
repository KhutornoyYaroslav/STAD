from torch import nn
from core.config import CfgNode
from core.modeling.backbone3d.mobilenetv2 import build_mobilenetv2
from core.modeling.backbone3d.i3d import build_i3d


def build_backbone3d(cfg: CfgNode, *args, **kwargs) -> nn.Module:
    arch = cfg.MODEL.BACKBONE3D.ARCHITECTURE

    if arch == 'mobilenetv2':
        backbone3d = build_mobilenetv2(cfg)
    elif arch == 'i3d':
        backbone3d = build_i3d(cfg)
    else:
        raise ValueError(f"Backbone3d architecture '{arch}' not found")
    
    return backbone3d
