import math
import torch
import torch.nn as nn
import logging
from typing import Sequence
from core.config import CfgNode
from core.utils.model_zoo import load_state_dict


def conv_bn(ch_in: int, ch_out: int, stride: Sequence[int]):
    return nn.Sequential(
        nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(ch_out),
        nn.SiLU(inplace=True)
    )


def conv_1x1x1_bn(ch_in: int, ch_out: int):
    return nn.Sequential(
        nn.Conv3d(ch_in, ch_out, 1, 1, 0, bias=False),
        nn.BatchNorm3d(ch_out),
        nn.SiLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, 
                 ch_in: int,
                 ch_out: int,
                 stride: Sequence[int],
                 expand_ratio: float):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == (1,1,1) and ch_in == ch_out
        ch_hidden = round(ch_in * expand_ratio)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(ch_hidden, ch_hidden, 3, stride, 1, groups=ch_hidden, bias=False),
                nn.BatchNorm3d(ch_hidden),
                nn.SiLU(inplace=True),
                # pw-linear
                nn.Conv3d(ch_hidden, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm3d(ch_out),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(ch_in, ch_hidden, 1, 1, 0, bias=False),
                nn.BatchNorm3d(ch_hidden),
                nn.SiLU(inplace=True),
                # dw
                nn.Conv3d(ch_hidden, ch_hidden, 3, stride, 1, groups=ch_hidden, bias=False),
                nn.BatchNorm3d(ch_hidden),
                nn.SiLU(inplace=True),
                # pw-linear
                nn.Conv3d(ch_hidden, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm3d(ch_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, ch_mult: float = 1.0):
        super(MobileNetV2, self).__init__()
        ch_in = 32
        ch_last = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, (1, 1, 1)],
            [6,  24, 2, (2, 2, 2)],
            [6,  32, 3, (2, 2, 2)],
            [6,  64, 4, (2, 2, 2)],
            [6,  96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        # building first layer
        ch_in = int(ch_in * ch_mult)
        self.ch_last = int(ch_last * ch_mult) if ch_mult > 1.0 else ch_last
        self.features = [conv_bn(3, ch_in, (1, 2, 2))]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            ch_out = int(c * ch_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(InvertedResidual(ch_in, ch_out, stride, expand_ratio=t))
                ch_in = ch_out

        # building last several layers
        self.features.append(conv_1x1x1_bn(ch_in, self.ch_last))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AvgPool3d((2, 1, 1), stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if x.size(2) == 2:
            x = self.avgpool(x)
        return x


def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def build_mobilenetv2(cfg: CfgNode) -> nn.Module:
    logger = logging.getLogger('CORE')
    cfg_bb = cfg.MODEL.BACKBONE3D

    model = MobileNetV2(ch_mult=cfg_bb.CHANNEL_MULT)
    initialize_weights(model)

    if cfg_bb.PRETRAINED_WEIGHTS:
        # parse state dict
        state_dict = load_state_dict(cfg_bb.PRETRAINED_WEIGHTS)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # prepare names
        prefix = "module."
        prepared_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith(prefix):
                prepared_state_dict[name[len(prefix):]] = param
        if len(prepared_state_dict):
            state_dict = prepared_state_dict

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info("Backbone3d pretrained weights loaded: {0} missing, {1} unexpected".
                    format(len(missing_keys), len(unexpected_keys)))
        assert not len(missing_keys)

    if cfg_bb.FREEZE:
        model.requires_grad_(False)

    return model
