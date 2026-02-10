import torch


def create_causal_mask(seq_len: int):
    x = torch.zeros([seq_len, seq_len])
    mask = torch.ones([seq_len, seq_len], dtype=torch.bool)
    mask = torch.tril(mask, diagonal=0)
    return x.masked_fill(~mask, float("-inf"))
