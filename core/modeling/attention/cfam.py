import torch
from torch import nn
from typing import Sequence


class CAM(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            inputs :
                x : input feature maps( B X C X H X W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CFAMBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super(CFAMBlock, self).__init__()
        inter_channels = ch_out
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(ch_in, inter_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.SiLU())
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.SiLU())
        self.sc = CAM(inter_channels)
        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.SiLU())
        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, ch_out, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.sc(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)
        return output


class CFAMFusion(nn.Module):
    def __init__(self,
                 ch_2d: Sequence[int],
                 ch_3d: int,
                 interchannels: int,
                 mode='decoupled'):
        super().__init__()
        assert mode in ['coupled', 'decoupled'], "wrong mode in CFAMFusion"
        self.mode = mode

        if mode == 'coupled':
            layers = []
            for channels2D in ch_2d:
                layers.append(CFAMBlock(channels2D + ch_3d, interchannels))
            self.fusion = nn.ModuleList(layers)
        elif mode == 'decoupled':
            box = []
            cls = []
            for channels2D in ch_2d:
                box.append(CFAMBlock(channels2D[0] + ch_3d, interchannels))
                cls.append(CFAMBlock(channels2D[1] + ch_3d, interchannels))
            self.box = nn.ModuleList(box)
            self.cls = nn.ModuleList(cls)

    def forward(self, ft_2D, ft_3D):
        _, C_3D, H_3D, W_3D = ft_3D.shape

        fts = []
        if self.mode == 'coupled':
            for idx, ft2D in enumerate(ft_2D):
                _, C_2D, H_2D, W_2D = ft2D.shape
                assert H_2D/H_3D == W_2D/W_3D, "can't upscale"

                upsampling = nn.Upsample(scale_factor=H_2D/H_3D)
                ft_3D_t = upsampling(ft_3D)
                ft = torch.cat((ft2D, ft_3D_t), dim = 1)
                fts.append(self.fusion[idx](ft))
        elif self.mode == 'decoupled':
            for idx, ft2D in enumerate(ft_2D):
                _, C_2D, H_2D, W_2D = ft2D[0].shape
                assert H_2D/H_3D == W_2D/W_3D, "can't upscale"

                upsampling = nn.Upsample(scale_factor=H_2D/H_3D)
                ft_3D_t = upsampling(ft_3D)
                ft_box = torch.cat((ft2D[0], ft_3D_t), dim = 1)
                ft_cls = torch.cat((ft2D[1], ft_3D_t), dim = 1)
                fts.append([self.box[idx](ft_box), self.cls[idx](ft_cls)])

        return fts
