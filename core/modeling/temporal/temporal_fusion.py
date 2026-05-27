import torch
from torch import nn
from .gru_block import GRUBlock
from typing import Sequence, List


class TemporalFusion(nn.Module):
    valid_residual_types = {
        'none',
        'concat'
    }

    def __init__(
            self,
            in_channels: Sequence[int],
            hidden_channels: Sequence[int],
            stateful_training: bool = False,
            learnable_init_state: bool = False,
            residual_type: str = 'none'):
        super(TemporalFusion, self).__init__()
        # check args
        if len(in_channels) != len(hidden_channels):
            raise ValueError("'in_channels' length must be equal 'hidden_channels' length")
        
        self.residual_type = residual_type.lower()
        if self.residual_type not in self.valid_residual_types:
            raise ValueError(f"Residual type must be one of {self.valid_residual_types}")

        # create blocks
        self.in_convs = nn.ModuleList()
        self.rnn_blocks = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        for in_ch, hi_ch in zip(in_channels, hidden_channels):
            # in conv
            self.in_convs.append(nn.Conv2d(in_ch, in_ch, kernel_size=1))
            block = GRUBlock(
                in_ch,
                hi_ch,
                stateful_training,
                learnable_init_state
                )
            # rnn block
            self.rnn_blocks.append(block)
            # out conv
            if self.residual_type == 'concat':
                out = nn.Conv2d(2 * in_ch, in_ch, kernel_size=1)
            else:
                out = nn.Identity()
            self.out_convs.append(out)

    def prepare_inference(self, batch_sizes: Sequence[int]):
        assert len(batch_sizes) == len(self.rnn_blocks)
        for i, bs in enumerate(batch_sizes):
            self.rnn_blocks[i].reset_state(bs)

    def forward(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        args:
            inputs
                sequence of torch.Tensor with shapes (b, t, ci, hi, wi)

        returns:
            outputs
                sequence of torch.Tensor with shapes (b, t, ci, hi, wi)
        """
        assert len(inputs) == len(self.rnn_blocks)

        outputs = []
        for id, input in enumerate(inputs):
            B, T, C, H, W = input.shape

            # in conv
            x = input.view(-1, C, H, W)         # from (b, t, c, h, w) to (b*t, c, h, w)
            x = self.in_convs[id](x)

            # rnn block
            x = x.view(B, T, C, H, W)           # from (b*t, c, h, w) to (b, t, c, h, w)
            x = x.view(B, T, C, -1)             # from (b, t, c, h, w) to (b, t, c, h*w)
            x = x.permute(0, 3, 1, 2)           # from (b, t, fc, fh*fw) to (b, fh*fw, t, fc)
            x = x.reshape(-1, T, C)             # from (b, h*w, t, c) to (b*h*w, t, c)
            x = self.rnn_blocks[id](x)

            # out conv
            x = x.view(B, -1, T, C)             # from (b*h*w, t, c) to (b, h*w, t, c)
            x = x.permute(0, 2, 3, 1)           # from (b, h*w, t, c) to (b, t, c, h*w)
            x = x.view(B, T, C, H, W)           # from (b, t, c, h*w) to (b, t, c, h, w)

            if self.residual_type == 'concat':
                x = torch.concat([input, x], 2)
                x = x.view(-1, 2*C, H, W)       # from (b, t, 2*c, h, w) to (b*t, 2*c, h, w)
            else:
                x = x.reshape(-1, C, H, W)         # from (b, t, c, h, w) to (b*t, c, h, w)

            x = self.out_convs[id](x)
            x = x.view(B, T, C, H, W)    # from (b*t, c, h, w) to (b, t, c, h, w)

            outputs.append(x)

        return outputs
