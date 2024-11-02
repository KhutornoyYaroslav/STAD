from torch import nn
from core.config import CfgNode
from core.utils.model_zoo import load_state_dict
from core.modeling.backbone2d.yolov8 import build_yolov8


def build_backbone2d(cfg: CfgNode) -> nn.Module:
    # create model
    if cfg.ARCHITECTURE.startswith("yolov8"):
        backbone2d = build_yolov8(cfg)
    else:
        raise ValueError(f"Backbone2d architecture '{cfg.ARCHITECTURE}' not found")
    
    # init by weights
    if cfg.PRETRAINED_WEIGHTS:
        state_dict = load_state_dict(cfg.PRETRAINED_WEIGHTS)
        if "model" in state_dict:
            state_dict = state_dict["model"]
            if isinstance(state_dict, nn.Module):
                state_dict = state_dict.state_dict()

        missing_keys, unexpected_keys = backbone2d.load_state_dict(state_dict, strict=False)
        print("Backbone2d pretrained weights loaded: {0} missing, {1} unexpected".
              format(len(missing_keys), len(unexpected_keys)))
        
    # freeze weights
    if cfg.FREEZE:
        backbone2d.requires_grad_(False)

    return backbone2d
