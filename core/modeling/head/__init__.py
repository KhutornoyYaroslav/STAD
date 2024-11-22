from torch import nn
from core.config import CfgNode
from core.modeling.head.yolov8 import build_yolov8


def build_head(cfg: CfgNode, *args, **kwargs) -> nn.Module:
    arch = cfg.MODEL.HEAD.ARCHITECTURE

    if arch == "yolov8":
        if "channels" not in kwargs or "strides" not in kwargs:
            raise ValueError(f"Head 'yolov8' requires kwargs 'channels', 'strides'")

        head = build_yolov8(cfg, kwargs["channels"], kwargs["strides"])
    else:
        raise ValueError(f"Head architecture '{arch}' not found")

    return head
