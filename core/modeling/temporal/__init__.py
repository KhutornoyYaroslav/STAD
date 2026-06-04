from torch import nn
from typing import Sequence
from core.config import CfgNode
from .temporal_fusion import TemporalFusion


def build_tempfusion(
        cfg: CfgNode,
        input_channels: Sequence[int],
        hidden_channels: Sequence[int]) -> nn.Module:
    
    cfg_tf = cfg.MODEL.TEMPORAL_FUSION

    model = TemporalFusion(
        in_channels=input_channels,
        hidden_channels=hidden_channels,
        stateful_training=cfg_tf.STATEFUL_TRAINING,
        learnable_init_state=cfg_tf.LEARNABLE_INIT_STATE,
        residual_type=cfg_tf.RESIDUAL_TYPE)
    
    if cfg_tf.FREEZE:
        model.requires_grad_(False)

    return model