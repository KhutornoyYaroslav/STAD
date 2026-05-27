import torch
import torch.nn as nn


class GRUBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            stateful_training: bool = False,
            learnable_init_state: bool = False):
        super(GRUBlock, self).__init__()
        self.stateful_training = stateful_training
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        self.learnable_init_state = learnable_init_state
        if learnable_init_state:
            self.init_hidden_state = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        else:
            self.register_buffer('init_hidden_state', torch.zeros(hidden_size))

        self.register_buffer("state", torch.empty(0), persistent=False)
        self.to_out = nn.Linear(hidden_size, input_size) if hidden_size != input_size else nn.Identity()

    def reset_state(self, batch_size: int):
        # print("GRU state reset to: ", batch_size)
        self.state = self.init_hidden_state.repeat(1, batch_size, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, C = input.shape

        if self.state.numel() == 0 or (self.training and self.learnable_init_state):
            self.reset_state(B)

        gru_out, h = self.gru(input, self.state)
        if not self.training or self.stateful_training:
            self.state = h.detach()

        return self.to_out(gru_out)