import torch


def create_causal_mask(seq_len: int):
    """构建因果掩码"""
    x = torch.zeros((seq_len, seq_len))
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    return x.masked_fill(mask, float("-inf"))


print(create_causal_mask(8))
