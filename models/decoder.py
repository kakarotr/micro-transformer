import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.attention import MultiHeadAttention
from models.components.mlp import SwiGLUMLP
from models.components.rms import RMSNorm
from models.components.rope import DefaultRope, Rope
from models.config import TransformerConfig
from models.utilities.mask import create_causal_mask, create_padding_mask


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
        self.mlp = SwiGLUMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.attn_norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)
        self.mlp_norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.mlp_dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if use_cache or past_key_value is not None:
            raise NotImplementedError("KV cache path is reserved but not implemented yet.")
        residual = hidden_states
        hidden_states = self.attention(self.attn_norm(hidden_states), position_ids, attn_mask)
        hidden_states = residual + self.attn_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = residual + self.mlp_dropout(hidden_states)

        return hidden_states, None


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
        pad_token_id: int,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
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
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if use_cache or past_key_values is not None:
            raise NotImplementedError("Decoder KV cache path is not implemented yet.")

        _, seq_len = input_ids.size()
        if seq_len > self.rope.max_position_embeddings:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_position_embeddings ({self.rope.max_position_embeddings})"
            )

        device = input_ids.device

        hidden_states = self.embedd_dropout(self.embeddings(input_ids))
        positions_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = create_causal_mask(seq_len=seq_len, device=input_ids.device)
        padding_mask = create_padding_mask(input_ids=input_ids, pad_token_id=self.pad_token_id)
        attn_mask = causal_mask + padding_mask

        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, positions_ids, attn_mask, past_key_values, use_cache)

        return self.norm(hidden_states)


class CausalLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
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
            pad_token_id=config.pad_token_id,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        base_std = 0.02
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if name.endswith("o_proj") or name.endswith("down_proj"):
                    scaled_std = base_std / math.sqrt(2 * self.num_layers)
                    module.weight.data.normal_(mean=0.0, std=scaled_std)
                else:
                    module.weight.data.normal_(mean=0.0, std=base_std)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        hidden_states = self.decoder(input_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
        return loss, logits


# 0.68B
config = TransformerConfig(
    vocab_size=32768,
    max_position_embeddings=4096,
    hidden_size=1024,
    num_layers=48,
    num_attention_heads=16,
    num_key_value_heads=16,
    dropout_prob=0.0,
    intermediate_size=2816,
    rms_eps=1e-6,
    rope_base=10000,
    pad_token_id=0,
)
print(config.compute_model_size())
