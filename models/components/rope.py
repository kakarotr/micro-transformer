import torch
import torch.nn as nn


class Rope(nn.Module):
    def __init__(self, *, base: int, max_position_embeddings: int, head_dim: int):
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim
        cos, sin = self._precompute_freqs_cis()

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _precompute_freqs_cis(self):
        inv_freq = 1 / self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        freqs = torch.outer(torch.arange(self.max_position_embeddings).float(), inv_freq)
        return torch.cos(freqs), torch.sin(freqs)


class DefaultRope(Rope):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input: torch.Tensor, position_ids: torch.Tensor):
        input_type = input.dtype
        input = input.to(torch.float32)

        cos: torch.Tensor = self.cos_cached[position_ids].unsqueeze(1)
        sin: torch.Tensor = self.sin_cached[position_ids].unsqueeze(1)

        input_even = input[..., ::2]
        input_odd = input[..., 1::2]

        input_rot_even = input_even * cos - input_odd * sin
        input_rot_odd = input_even * sin + input_odd * cos

        input_rot = torch.empty_like(input)
        input_rot[..., ::2] = input_rot_even
        input_rot[..., 1::2] = input_rot_odd

        return input_rot.to(input_type)
