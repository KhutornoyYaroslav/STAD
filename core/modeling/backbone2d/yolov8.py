import torch
import logging
from torch import nn
from typing import Optional, Tuple, List
from core.config import CfgNode
from core.utils.model_zoo import load_state_dict


def calc_padding_size(kernel: int,
                      padding: Optional[int] = None,
                      dilation: int = 1):
    if dilation > 1:
        kernel = dilation * (kernel - 1) + 1
    if padding is None:
        padding = kernel // 2
    return padding


class Concat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 kernel: int = 1,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=ch_in,
                              out_channels=ch_out,
                              kernel_size=kernel,
                              stride=stride,
                              padding=calc_padding_size(kernel, padding, dilation),
                              dilation=dilation,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 shortcut: bool = True,
                 groups: int = 1,
                 kernels: Tuple[int, int] = (3, 3),
                 expansion: float = 0.5):
        super().__init__()
        ch_hidden = int(ch_out * expansion)
        self.cv1 = Conv(ch_in, ch_hidden, kernels[0], 1)
        self.cv2 = Conv(ch_hidden, ch_out, kernels[1], 1, groups=groups)
        self.add = shortcut and ch_in == ch_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return (x + y) if self.add else y


class C2f(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 bnecks: int = 1,
                 shortcut: bool = False,
                 groups: int = 1,
                 expansion: float = 0.5):
        super().__init__()
        self.ch_hidden = int(ch_out * expansion)
        self.cv1 = Conv(ch_in, 2 * self.ch_hidden, 1, 1)
        self.cv2 = Conv((2 + bnecks) * self.ch_hidden, ch_out, 1)
        self.m = nn.ModuleList(Bottleneck(self.ch_hidden, self.ch_hidden, shortcut, groups, expansion=1.0) for _ in range(bnecks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.ch_hidden, self.ch_hidden), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel: int = 5):
        super().__init__()
        ch_hidden = ch_in // 2
        self.cv1 = Conv(ch_in, ch_hidden, 1, 1)
        self.cv2 = Conv(ch_hidden * 4, ch_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class YOLOv8(nn.Module):
    def __init__(self, channels: List[int], c2f_bnecks: List[int]):
        super().__init__()
        modules = [
            Conv(channels[0], channels[1], 3, 2),                               # p1
            Conv(channels[1], channels[2], 3, 2),
            C2f(channels[2], channels[2], c2f_bnecks[0], True),                 # p2
            Conv(channels[2], channels[3], 3, 2),
            C2f(channels[3], channels[3], c2f_bnecks[1], True),                 # p3
            Conv(channels[3], channels[4], 3, 2),
            C2f(channels[4], channels[4], c2f_bnecks[2], True),                 # p4
            Conv(channels[4], channels[5], 3, 2),
            C2f(channels[5], channels[5], c2f_bnecks[3], True),
            SPPF(channels[5], channels[5], 5),                                  # p5
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(),
            C2f(channels[4] + channels[5], channels[4], c2f_bnecks[0], False),  # h1
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(),
            C2f(channels[3] + channels[4], channels[3], c2f_bnecks[0], False),  # h2
            Conv(channels[3], channels[3], 3, 2),
            Concat(),
            C2f(channels[3] + channels[4], channels[4], c2f_bnecks[0], False),  # h4
            Conv(channels[4], channels[4], 3, 2),
            Concat(),
            C2f(channels[4] + channels[5], channels[5], c2f_bnecks[0], False)   # h6
        ]
        self.model = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p1 = self.model[0](x)
        p2 = self.model[2](self.model[1](p1))
        p3 = self.model[4](self.model[3](p2))
        p4 = self.model[6](self.model[5](p3))
        p5 = self.model[9](self.model[8](self.model[7](p4)))

        h1 = self.model[12](self.model[11]([self.model[10](p5), p4]))
        h2 = self.model[15](self.model[14]([self.model[13](h1), p3]))
        h4 = self.model[18](self.model[17]([self.model[16](h2), h1]))
        h6 = self.model[21](self.model[20]([self.model[19](h4), p5]))

        return [h2, h4, h6]


def build_yolov8(cfg: CfgNode) -> nn.Module:
    logger = logging.getLogger('CORE')
    cfg_bb = cfg.MODEL.BACKBONE2D

    version = cfg_bb.ARCHITECTURE[len("yolov8"):]
    if version == 'n':
        c2f_bnecks = [1, 2, 2, 1]
        channels = [3, 16, 32, 64, 128, 256]
    elif version == 's':
        c2f_bnecks = [1, 2, 2, 1]
        channels = [3, 32, 64, 128, 256, 512]
    elif version == 'm':
        c2f_bnecks = [2, 4, 4, 2]
        channels = [3, 48, 96, 192, 384, 576]
    elif version == 'l':
        c2f_bnecks = [3, 6, 6, 3]
        channels = [3, 64, 128, 256, 512, 512]
    elif version == 'x':
        c2f_bnecks = [3, 6, 6, 3]
        channels = [3, 80, 160, 320, 640, 640]
    else:
        raise ValueError(f"yolov8 with version '{version}' not found")

    model = YOLOv8(channels, c2f_bnecks)

    if cfg_bb.PRETRAINED_WEIGHTS:
        state_dict = load_state_dict(cfg_bb.PRETRAINED_WEIGHTS)
        if "model" in state_dict:
            state_dict = state_dict["model"]
            if isinstance(state_dict, nn.Module):
                state_dict = state_dict.state_dict()

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info("Backbone2d pretrained weights loaded: {0} missing, {1} unexpected".
                    format(len(missing_keys), len(unexpected_keys)))
        assert not len(missing_keys)

    if cfg_bb.FREEZE:
        model.requires_grad_(False)

    return model
