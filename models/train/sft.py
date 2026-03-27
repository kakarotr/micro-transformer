import math

import torch
import torch.distributed as dist


class TokenLRSchedule:
    def __init__(
        self,
        total_tokens: int,
        warmup_tokens: int,
        *,
        start_factor: float = 0.1,
        min_lr_ratio: float = 0.0,
    ):
        self.total_tokens = total_tokens
        self.warmup_tokens = warmup_tokens
        self.start_factor = start_factor
        self.min_lr_ratio = min_lr_ratio
        self.consumed_tokens = 0

    def set_consumed_tokens(self, consumed_tokens: int):
        self.consumed_tokens = max(0, min(consumed_tokens, self.total_tokens))

    def __call__(self, _: int):
        consumed_tokens = self.consumed_tokens
        total_tokens = self.total_tokens
        warmup_tokens = min(max(0, self.warmup_tokens), total_tokens)

        if warmup_tokens > 0 and consumed_tokens < warmup_tokens:
            # 处于预热阶段
            progress = consumed_tokens / warmup_tokens
            return self.start_factor + (1.0 - self.start_factor) * progress

        if total_tokens == warmup_tokens:
            return self.min_lr_ratio

        progress = (consumed_tokens - warmup_tokens) / max(1, total_tokens - warmup_tokens)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


def all_reduce_sum_int(value: int, device: torch.device, is_distributed: bool):
    """进行分布式求和"""
    if not is_distributed:
        return value
    tensor_value = torch.tensor(value, device=device, dtype=torch.long)
    dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
    return int(tensor_value.item())
    return int(tensor_value.item())
