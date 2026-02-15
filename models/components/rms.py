import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        rms = (hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (hidden_states * rms).to(input_type) * self.weight
