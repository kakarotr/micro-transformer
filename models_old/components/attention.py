import math

import torch
import torch.nn as nn

from models.utilities.positional_encoding.rope import Rope


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        dropout: float,
        rope: Rope,
    ):
        super().__init__()
        self.rope = rope
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.qkv_dim = self.head_dim * self.num_attention_heads + self.head_dim * self.num_key_value_heads * 2
        self.qkv_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, mask: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        qkv: torch.Tensor = self.qkv_proj(x)
        q_dim = self.hidden_size
        kv_dim = self.num_key_value_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q: torch.Tensor = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k: torch.Tensor = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v: torch.Tensor = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q: torch.Tensor = self.rope(q, position_ids)
        k: torch.Tensor = self.rope(k, position_ids)

        n_rep = self.num_attention_heads // self.num_key_value_heads
        if n_rep > 1:
            k = self._repeat_kv(k, batch_size, seq_len, n_rep)
            v = self._repeat_kv(v, batch_size, seq_len, n_rep)

        attn_score = q.matmul(k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_score = attn_score + mask
        atten_weight = torch.softmax(attn_score, dim=-1)
        atten_weight = self.dropout(atten_weight)
        result = atten_weight.matmul(v).transpose(1, 2)
        result = result.reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(result)

    def _repeat_kv(self, x: torch.Tensor, batch_size: int, seq_len: int, n_rep: int):
        x = x.unsqueeze(2)
        x = x.expand(batch_size, self.num_key_value_heads, n_rep, seq_len, self.head_dim)
        x = x.reshape(batch_size, self.num_attention_heads, seq_len, self.head_dim)

        return x
