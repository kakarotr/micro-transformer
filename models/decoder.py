import torch
import torch.nn as nn

from models.components.attention import MultiHeadAttention
from models.components.mlp import MLP
from models.components.rms import RMSNorm
from models.components.rope import DefaultRope, Rope
from models.config import TransformerConfig
from models.utilities.mask import create_causal_mask


class DecoderLayer(nn.Module):
    def __init__(
        self,
        rope: Rope,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_eps: float,
        dropout_prob: float,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            rope=rope,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_prob=dropout_prob,
        )
        self.mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout_prob=dropout_prob)
        self.atten_norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)
        self.mlp_norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, mask: torch.Tensor | None = None):
        residual = hidden_states
        output = self.attention(self.atten_norm(hidden_states), position_ids, mask)
        hidden_states = residual + output

        residual = hidden_states
        output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = residual + output

        return hidden_states


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_eps: float,
        dropout_prob: float,
        rope_base: int,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.rope = DefaultRope(
            base=rope_base, max_position_embeddings=max_position_embeddings, head_dim=hidden_size // num_attention_heads
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    rope=self.rope,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    rms_eps=rms_eps,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )
        self.embedd_dropout = nn.Dropout(dropout_prob)
        self.norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)

    def forward(self, input_ids: torch.Tensor):
        _, seq_len = input_ids.size()
        device = input_ids.device

        hidden_states = self.embedd_dropout(self.embeddings(input_ids))
        positions_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = create_causal_mask(seq_len=seq_len).to(device=device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions_ids, causal_mask)

        return self.norm(hidden_states)


class CausalLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            rms_eps=config.rms_eps,
            dropout_prob=config.dropout_prob,
            rope_base=config.rope_base,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.decoder(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
