from rich.console import Group
from rich.progress import Progress
from rich.table import Table


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_status_table(
    *,
    epoch: int,
    step: int,
    max_steps: int,
    train_loss: float | None,
    eval_loss: float | None,
    eval_ppl: float | None,
    lr: float,
    grad_norm: float | None,
    tokens_seen: int,
    tokens_per_sec: float,
    eta_seconds: float,
) -> Table:
    table = Table(title="Training Metrics", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    rows = [
        ("epoch", str(epoch)),
        ("step", f"{step}/{max_steps}"),
        ("train_loss", "-" if train_loss is None else f"{train_loss:.4f}"),
        ("eval_loss", "-" if eval_loss is None else f"{eval_loss:.4f}"),
        ("eval_ppl", "-" if eval_ppl is None else f"{eval_ppl:.4f}"),
        ("lr", f"{lr:.6e}"),
        ("grad_norm", "-" if grad_norm is None else f"{grad_norm:.4f}"),
        ("tokens_seen", f"{tokens_seen:,}"),
        ("tok/s", f"{tokens_per_sec:,.0f}"),
        ("eta", format_seconds(eta_seconds)),
    ]

    for metric, value in rows:
        table.add_row(metric, value)

    return table


def build_dashboard(progress: Progress, task_id: int, table: Table):
    return Group(progress, table)
