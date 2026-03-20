import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import TransformerConfig
from models.decoder import Decoder


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
        self.pad_token_id = config.pad_token_id
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        base_std = 0.02
        residual_std = base_std / math.sqrt(2 * self.num_layers)

        nn.init.normal_(self.decoder.embeddings.weight, mean=0.0, std=base_std)
        for layer in self.decoder.layers:
            nn.init.normal_(layer.attention.q_proj.weight, mean=0.0, std=base_std)
            nn.init.normal_(layer.attention.kv_proj.weight, mean=0.0, std=base_std)
            nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=residual_std)

            nn.init.normal_(layer.mlp.gate_up_proj.weight, mean=0.0, std=base_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)

        nn.init.normal_(self.lm_head.weight, mean=0.0, std=base_std)
        self.decoder.embeddings.weight[self.pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        if attention_mask is None:
            attention_mask = input_ids != self.pad_token_id

        hidden_states = self.decoder(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            assert attention_mask is not None
            shift_labels = shift_labels.masked_fill(attention_mask[:, 1:] == 0, -100)

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss, logits
