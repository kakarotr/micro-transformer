import torch
import torch.nn as nn


class Rope(nn.Module):
    def __init__(self, max_position_embeddings: int, head_dim: int, base: int):
        super().__init__()
        self.base = base
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        cos, sin = self._precompute_freqs_cis()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        input_type = x.dtype
        x.to(torch.float32)

        cos: torch.Tensor = self.cos_cached[position_ids].unsqueeze(1)
        sin: torch.Tensor = self.sin_cached[position_ids].unsqueeze(1)

        out = (x * cos) + (self._rotate_half(x) * sin)
        return out.to(input_type)

    def _precompute_freqs_cis(self):
        inv_freq = 1 / self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        t = torch.arange(self.max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _rotate_half(self, x: torch.Tensor):
        x_1 = x[..., : x.shape[-1] // 2]
        x_2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x_2, x_1), dim=-1)
