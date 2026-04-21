import logging
from torch import nn
from core.config import CfgNode
from .model.yolov8 import build_yolov8
from .model.yolov8_rnn import build_yolov8rnn


def build_model(cfg: CfgNode) -> nn.Module:
    logger = logging.getLogger('CORE')

    # build model
    arch = str(cfg.MODEL.ARCHITECTURE).lower()
    if arch == 'yolov8':
        model = build_yolov8(cfg)
    elif arch == 'yolov8rnn':
        model = build_yolov8rnn(cfg)
    else:
        raise NotImplementedError(f"Model architecture '{arch}' hasn't been implemented yet")

    # model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built. Parameters in total: {total_params}")

    return model
