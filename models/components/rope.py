import torch
import torch.nn as nn


class Rope(nn.Module):
    def __init__(self, *, base: int, max_position_embeddings: int, head_dim: int):
        super().__init__()
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim
        cos, sin = self._precompute_freqs_cis()

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _precompute_freqs_cis(self):
        inv_freq = 1 / self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        freqs = torch.outer(torch.arange(self.max_position_embeddings).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class DefaultRope(Rope):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input: torch.Tensor, position_ids: torch.Tensor):
        input = input.to(torch.float32)

        cos: torch.Tensor = self.cos_cached[position_ids].unsqueeze(1)
        sin: torch.Tensor = self.sin_cached[position_ids].unsqueeze(1)

        out = (input * cos) + (self._rotate_half(input) * sin)
        return out.type_as(input)

    def _rotate_half(self, x: torch.Tensor):
        x_1 = x[..., : x.shape[-1] // 2]
        x_2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x_2, x_1), dim=-1)
