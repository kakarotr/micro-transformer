from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class TransformerConfig(BaseModel):
    vocab_size: Annotated[int, Field(description="词表大小")]
    max_position_embeddings: Annotated[int, Field(description="输入序列最大Token长度")]
    hidden_size: Annotated[int, Field(description="Token维度")]
    num_layers: Annotated[int, Field(description="隐藏层数量")]
    num_attention_heads: Annotated[int, Field(description="多头注意力头数")]
    num_key_value_heads: Annotated[int, Field(description="KV组数")]
    dropout_prob: Annotated[float, Field(description="Dropout强度")]
    intermediate_size: Annotated[int, Field(description="多层感知机升维维度")]
    rms_eps: Annotated[float, Field(description="RMS指数")]
    rope_base: Annotated[int, Field(description="ROPE旋转基数")]

    @model_validator(mode="after")
    def validate_model_params(self) -> "TransformerConfig":
        if self.num_attention_heads == 0:
            raise ValueError("num_attention_heads cannot be zero")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                "num_key_value_heads ({self.num_key_value_heads}) to support GQA/MQA."
            )
        return self
