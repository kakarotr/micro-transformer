import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.rope import Rope


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        *,
        rope: Rope,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_prob = dropout_prob
        self.n_rep = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads * 2, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = hidden_states.size()

        q: torch.Tensor = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        kv: torch.Tensor = self.kv_proj(hidden_states)
        kv = kv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim, 2).transpose(1, 2)
        k, v = kv.unbind(dim=-1)

        # 传统写法
        # q: torch.Tensor = self.rope(q, position_ids)
        # k: torch.Tensor = self.rope(k, position_ids)
        # if self.n_rep > 1:
        #     # 兼容MQA或GQA
        #     # q shape is [batch_size, num_heads, seq_len, head_dim]
        #     # k/v shape is [batch_size, num_kv_heads, seq_len, head_dim]
        #     q = q.unsqueeze(2).reshape(batch_size, self.num_key_value_heads, self.n_rep, seq_len, self.head_dim)
        #     k = k.unsqueeze(2)
        #     v = v.unsqueeze(2)
        # atten_scores = q.matmul(v.transpose(1, 2)).div(math.sqrt(self.head_dim))
        # if mask:
        #     atten_scores = atten_scores + mask
        # atten_probs: torch.Tensor = self.dropout(F.softmax(atten_scores, dim=-1))
        # context_vectors = atten_probs.matmul(v)
        # context_vectors = context_vectors.flatten(1, 2)
        # context_vectors = context_vectors.reshape(batch_size, seq_len, self.hidden_size)
        # return self.o_proj(context_vectors)

        # Pytorch优化之后的写法, 支持Flash Attention
        q: torch.Tensor = self.rope(q, position_ids)
        k: torch.Tensor = self.rope(k, position_ids)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, 1)
            v = v.repeat_interleave(self.n_rep, 1)

        context_vectors = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=self.dropout_prob if self.training > 0 else 0,
            is_causal=mask is None,
        )
        context_vectors = context_vectors.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context_vectors)
