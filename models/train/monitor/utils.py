import math

import torch


def format_seconds(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or math.isnan(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_float(value: float | None, ndigits: int = 4) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{value:.{ndigits}f}"


def format_sci(value: float | None) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{value:.2e}"


def format_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def format_percent(value: float | None) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def get_grad_norm(parameters, norm_type: float = 2.0) -> float | None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return None

    if norm_type == math.inf:
        return max(g.detach().abs().max().item() for g in grads)

    norms = [torch.norm(g.detach(), p=norm_type) for g in grads]
    total = torch.norm(torch.stack(norms), p=norm_type)
    return float(total.item())


def has_nan_or_inf(parameters) -> tuple[bool, bool]:
    has_nan = False
    has_inf = False

    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        if torch.isnan(grad).any():
            has_nan = True
        if torch.isinf(grad).any():
            has_inf = True
        if has_nan or has_inf:
            break

    return has_nan, has_inf
