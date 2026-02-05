from pydantic import BaseModel


class ModelConfig(BaseModel):
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    rope_base: int
    rms_eps: float
    atten_dropout_prob: float
    mlp_dropout_prob: float
    layer_dropout_prob: float
    emb_dropout_prob: float

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads
