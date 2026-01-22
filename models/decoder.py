import torch
import torch.nn as nn

from config import ModelConfig
from models.components.attention import MultiHeadAttention
from models.components.mlp import MLP
from models.components.rms import RMSNorm
from models.utilities.mask.causal_mask import create_causal_mask
from models.utilities.positional_encoding.rope import Rope


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, rope: Rope) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            rope=rope,
            dropout=config.atten_dropout_prob,
        )
        self.mlp = MLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, dropout=config.mlp_dropout_prob
        )
        self.atten_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_eps)
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_eps)
        self.dropout = nn.Dropout(config.layer_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, position_ids: torch.Tensor, mask: torch.Tensor):
        residual = hidden_state
        atten_output = self.attention(self.atten_norm(hidden_state), position_ids, mask)
        hidden_state = residual + self.dropout(atten_output)

        residual = hidden_state
        mlp_output = self.mlp(self.mlp_norm(hidden_state))
        return residual + self.dropout(mlp_output)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        rope = Rope(
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            base=config.rope_base,
        )
        self.layers = nn.ModuleList([DecoderLayer(config=config, rope=rope) for _ in range(config.num_hidden_layers)])
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.emb_dropout_prob)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_eps)

    def forward(self, input_ids: torch.Tensor):
        _, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        mask = create_causal_mask(seq_len=seq_len).to(input_ids.device)

        hidden_state = self.dropout(self.embeddings(input_ids))
        for layer in self.layers:
            hidden_state = layer(hidden_state, position_ids, mask)
        return self.norm(hidden_state)


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.decoder = Decoder(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.std = 0.02
        self.apply(self._init_weight)

        for name, module in self.named_modules():
            if name.endswith("o_proj") or name.endswith("down_proj"):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(
                        module.weight,
                        std=self.std * torch.rsqrt(torch.tensor(2 * config.num_hidden_layers)).float().item(),
                    )

    def forward(self, input_ids: torch.Tensor):
        last_hidden_state = self.decoder(input_ids)
        logits = self.lm_head(last_hidden_state)
        return logits

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.std)
