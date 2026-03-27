import torch
import torch.nn.functional as F


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid_mask = shift_labels.ne(ignore_index)
    if not valid_mask.any():
        # 当出现labels全是ignore_index时使用下面的方式返回来保留计算图, 避免backward出现问题
        return shift_logits.sum() * 0.0

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss
