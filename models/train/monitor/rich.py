import time
from collections import deque
from typing import Any

import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from models.train.monitor.monitor_dataclass import MonitorSnapshot
from models.train.monitor.utils import (
    format_bytes,
    format_float,
    format_int,
    format_percent,
    format_sci,
    format_seconds,
    has_nan_or_inf,
)


class RollingMean:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def mean(self) -> float | None:
        if not self.values:
            return None
        return sum(self.values) / len(self.values)


class UpdateAccumulator:
    def __init__(self, gradient_accumulation_steps: int):
        if gradient_accumulation_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.reset()

    def reset(self) -> None:
        self.micro_steps = 0

        self.loss_raw_sum = 0.0
        self.last_loss_raw: float | None = None

        self.valid_tokens_sum = 0
        self.total_tokens_sum = 0
        self.samples_sum = 0

        self.last_batch_size: int | None = None
        self.last_seq_len: int | None = None
        self.last_valid_tokens: int | None = None
        self.last_total_tokens: int | None = None
        self.last_padding_ratio: float | None = None

        self.data_time_sum_s = 0.0
        self.fwdbwd_time_sum_s = 0.0
        self.micro_step_time_sum_s = 0.0
        self.last_micro_tokens_per_s: float | None = None

    @property
    def accum_progress(self) -> str:
        return f"{self.micro_steps}/{self.gradient_accumulation_steps}"

    def add_micro_step(
        self,
        *,
        loss_raw: float,
        batch: dict[str, torch.Tensor],
        data_time_s: float | None = None,
        fwdbwd_time_s: float | None = None,
        micro_step_time_s: float | None = None,
    ) -> None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        batch_size, seq_len = input_ids.shape
        valid_tokens = int(attention_mask.sum().item())
        total_tokens = int(attention_mask.numel())
        padding_ratio = 1.0 - (valid_tokens / total_tokens) if total_tokens > 0 else None

        self.micro_steps += 1
        self.loss_raw_sum += float(loss_raw)
        self.last_loss_raw = float(loss_raw)

        self.valid_tokens_sum += valid_tokens
        self.total_tokens_sum += total_tokens
        self.samples_sum += batch_size

        self.last_batch_size = batch_size
        self.last_seq_len = seq_len
        self.last_valid_tokens = valid_tokens
        self.last_total_tokens = total_tokens
        self.last_padding_ratio = padding_ratio

        if data_time_s is not None:
            self.data_time_sum_s += data_time_s
        if fwdbwd_time_s is not None:
            self.fwdbwd_time_sum_s += fwdbwd_time_s
        if micro_step_time_s is not None:
            self.micro_step_time_sum_s += micro_step_time_s
            if micro_step_time_s > 0:
                self.last_micro_tokens_per_s = valid_tokens / micro_step_time_s

    def is_ready_to_step(self) -> bool:
        return self.micro_steps >= self.gradient_accumulation_steps

    def summary(
        self,
        *,
        optim_time_s: float | None = None,
        grad_norm_pre_clip: float | None = None,
        grad_norm_post_clip: float | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, Any]:
        micro_steps = self.micro_steps

        loss_update = self.loss_raw_sum / micro_steps if micro_steps > 0 else None
        avg_valid_len = self.valid_tokens_sum / self.samples_sum if self.samples_sum > 0 else None
        update_padding_ratio = (
            1.0 - (self.valid_tokens_sum / self.total_tokens_sum) if self.total_tokens_sum > 0 else None
        )

        update_step_time_s = self.micro_step_time_sum_s + (optim_time_s or 0.0)
        update_tokens_per_s = (
            self.valid_tokens_sum / update_step_time_s if update_step_time_s > 0 and self.valid_tokens_sum > 0 else None
        )

        data_time_ms_avg = (self.data_time_sum_s / micro_steps) * 1000 if micro_steps > 0 else None
        fwdbwd_time_ms_avg = (self.fwdbwd_time_sum_s / micro_steps) * 1000 if micro_steps > 0 else None
        optim_time_ms = (optim_time_s * 1000) if optim_time_s is not None else None
        update_step_time_ms = update_step_time_s * 1000 if update_step_time_s > 0 else None

        was_clipped = False
        clip_ratio = None
        if grad_norm_pre_clip is not None and max_grad_norm is not None and max_grad_norm > 0:
            was_clipped = grad_norm_pre_clip > max_grad_norm
            clip_ratio = grad_norm_pre_clip / max_grad_norm

        return {
            "micro_steps": micro_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "accum_progress": self.accum_progress,
            "loss_micro_raw": self.last_loss_raw,
            "loss_update": loss_update,
            "micro_batch_size": self.last_batch_size,
            "effective_batch_size": self.samples_sum,
            "seq_len": self.last_seq_len,
            "micro_valid_tokens": self.last_valid_tokens,
            "micro_total_tokens": self.last_total_tokens,
            "micro_padding_ratio": self.last_padding_ratio,
            "update_valid_tokens": self.valid_tokens_sum,
            "update_total_tokens": self.total_tokens_sum,
            "effective_batch_tokens": self.valid_tokens_sum,
            "avg_valid_len": avg_valid_len,
            "update_padding_ratio": update_padding_ratio,
            "micro_tokens_per_s": self.last_micro_tokens_per_s,
            "update_tokens_per_s": update_tokens_per_s,
            "data_time_ms_avg": data_time_ms_avg,
            "fwdbwd_time_ms_avg": fwdbwd_time_ms_avg,
            "optim_time_ms": optim_time_ms,
            "update_step_time_ms": update_step_time_ms,
            "grad_norm_pre_clip": grad_norm_pre_clip,
            "grad_norm_post_clip": grad_norm_post_clip,
            "max_grad_norm": max_grad_norm,
            "was_clipped": was_clipped,
            "clip_ratio": clip_ratio,
        }


class MetricTracker:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.start_time = time.perf_counter()

        self.loss20 = RollingMean(20)
        self.loss100 = RollingMean(100)
        self.tokps100 = RollingMean(100)
        self.stepms100 = RollingMean(100)

        self.trained_tokens = 0
        self.trained_samples = 0

        self.last_eval_loss: float | None = None
        self.best_eval_loss: float | None = None

    def update_eval_loss(self, eval_loss: float) -> None:
        self.last_eval_loss = float(eval_loss)
        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            self.best_eval_loss = float(eval_loss)

    def build_snapshot(
        self,
        *,
        epoch: int,
        micro_step: int,
        optimizer_step: int,
        lr: float | None,
        summary: dict[str, Any],
        model: torch.nn.Module,
        committed: bool,
        mode: str = "train",
        max_optimizer_steps: int | None = None,
    ) -> MonitorSnapshot:
        elapsed_s = time.perf_counter() - self.start_time

        # 正式 optimizer step 完成后才更新长期统计
        if committed:
            if summary["loss_update"] is not None:
                self.loss20.update(summary["loss_update"])
                self.loss100.update(summary["loss_update"])
            if summary["update_tokens_per_s"] is not None:
                self.tokps100.update(summary["update_tokens_per_s"])
            if summary["update_step_time_ms"] is not None:
                self.stepms100.update(summary["update_step_time_ms"])

            self.trained_tokens += int(summary["effective_batch_tokens"] or 0)
            self.trained_samples += int(summary["effective_batch_size"] or 0)

        # pending snapshot 展示时，把当前累计窗口的 token 也加进去显示
        display_trained_tokens = self.trained_tokens
        display_trained_samples = self.trained_samples
        if not committed:
            display_trained_tokens += int(summary["effective_batch_tokens"] or 0)
            display_trained_samples += int(summary["effective_batch_size"] or 0)

        eta_s = None
        if max_optimizer_steps is not None and optimizer_step > 0:
            avg_step_s = None
            if self.stepms100.mean is not None:
                avg_step_s = self.stepms100.mean / 1000.0
            elif summary["update_step_time_ms"] is not None:
                avg_step_s = summary["update_step_time_ms"] / 1000.0

            if avg_step_s is not None:
                eta_s = max(max_optimizer_steps - optimizer_step, 0) * avg_step_s

        snapshot = MonitorSnapshot(
            run_name=self.run_name,
            mode=mode,
            epoch=epoch,
            micro_step=micro_step,
            optimizer_step=optimizer_step,
            max_optimizer_steps=max_optimizer_steps,
            gradient_accumulation_steps=int(summary["grad_accum_steps"]),
            accum_progress=str(summary["accum_progress"]),
            elapsed_s=elapsed_s,
            eta_s=eta_s,
            loss_update=summary["loss_update"],
            loss_micro_raw=summary["loss_micro_raw"],
            loss_avg_20=self.loss20.mean,
            loss_avg_100=self.loss100.mean,
            lr=lr,
            last_eval_loss=self.last_eval_loss,
            best_eval_loss=self.best_eval_loss,
            update_tokens_per_s=summary["update_tokens_per_s"],
            micro_tokens_per_s=summary["micro_tokens_per_s"],
            update_step_time_ms=summary["update_step_time_ms"],
            data_time_ms_avg=summary["data_time_ms_avg"],
            fwdbwd_time_ms_avg=summary["fwdbwd_time_ms_avg"],
            optim_time_ms=summary["optim_time_ms"],
            trained_tokens=display_trained_tokens,
            trained_samples=display_trained_samples,
            micro_batch_size=summary["micro_batch_size"],
            effective_batch_size=summary["effective_batch_size"],
            seq_len=summary["seq_len"],
            update_valid_tokens=summary["update_valid_tokens"],
            update_total_tokens=summary["update_total_tokens"],
            effective_batch_tokens=summary["effective_batch_tokens"],
            avg_valid_len=summary["avg_valid_len"],
            update_padding_ratio=summary["update_padding_ratio"],
            grad_norm_pre_clip=summary["grad_norm_pre_clip"],
            grad_norm_post_clip=summary["grad_norm_post_clip"],
            max_grad_norm=summary["max_grad_norm"],
            was_clipped=bool(summary["was_clipped"]),
            clip_ratio=summary["clip_ratio"],
        )

        snapshot.has_nan, snapshot.has_inf = has_nan_or_inf(model.parameters())

        if torch.cuda.is_available():
            snapshot.gpu_mem_alloc = torch.cuda.memory_allocated()
            snapshot.gpu_mem_reserved = torch.cuda.memory_reserved()
            snapshot.gpu_peak_mem = torch.cuda.max_memory_allocated()

        snapshot.warnings = []

        if snapshot.update_padding_ratio is not None and snapshot.update_padding_ratio > 0.10:
            snapshot.warnings.append(f"padding 高: {snapshot.update_padding_ratio * 100:.1f}%")
        if snapshot.was_clipped:
            snapshot.warnings.append("发生梯度裁剪")
        if snapshot.has_nan:
            snapshot.warnings.append("检测到 NaN 梯度")
        if snapshot.has_inf:
            snapshot.warnings.append("检测到 Inf 梯度")

        if snapshot.has_nan or snapshot.has_inf:
            snapshot.status_text = "DANGER | 数值异常，建议立即停止检查"
        elif snapshot.warnings:
            snapshot.status_text = "WARN | " + " | ".join(snapshot.warnings)
        else:
            phase = "update committed" if committed else f"accumulating {snapshot.accum_progress}"
            snapshot.status_text = f"{mode.upper()} | {phase} | training healthy"

        return snapshot


class RichMonitor:
    def __init__(self):
        self.console = Console()

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="train"),
            Layout(name="data"),
        )
        layout["right"].split_column(
            Layout(name="throughput"),
            Layout(name="stability"),
        )
        return layout

    def live(self, initial_snapshot: MonitorSnapshot, refresh_per_second: int = 4) -> Live:
        return Live(
            self.render(initial_snapshot),
            console=self.console,
            refresh_per_second=refresh_per_second,
            vertical_overflow="visible",
        )

    def render(self, s: MonitorSnapshot) -> Layout:
        layout = self.make_layout()
        layout["header"].update(self._render_header(s))
        layout["train"].update(self._render_train_panel(s))
        layout["throughput"].update(self._render_throughput_panel(s))
        layout["data"].update(self._render_data_panel(s))
        layout["stability"].update(self._render_stability_panel(s))
        layout["footer"].update(self._render_footer(s))
        return layout

    def _kv_table(self, title: str) -> Table:
        table = Table(title=title, expand=True, box=None, show_header=False, padding=(0, 1))
        table.add_column("key", style="cyan", no_wrap=True)
        table.add_column("value", justify="right")
        return table

    def _render_header(self, s: MonitorSnapshot) -> Panel:
        opt_text = f"{s.optimizer_step}"
        if s.max_optimizer_steps is not None:
            opt_text += f"/{s.max_optimizer_steps}"

        text = Text()
        text.append(f"Run: {s.run_name}  ", style="bold cyan")
        text.append(f"Mode: {s.mode}  ", style="bold magenta")
        text.append(f"Epoch: {s.epoch}  ", style="bold green")
        text.append(f"OptStep: {opt_text}  ", style="bold yellow")
        text.append(f"MicroStep: {s.micro_step}  ", style="bold white")
        text.append(f"Accum: {s.accum_progress}  ", style="bold bright_blue")
        text.append(f"Elapsed: {format_seconds(s.elapsed_s)}  ", style="white")
        text.append(f"ETA: {format_seconds(s.eta_s)}", style="white")
        return Panel(text, border_style="bright_blue")

    def _render_train_panel(self, s: MonitorSnapshot) -> Panel:
        table = self._kv_table("Train Metrics")
        table.add_row("loss(update)", format_float(s.loss_update))
        table.add_row("loss(micro)", format_float(s.loss_micro_raw))
        table.add_row("avg(20)", format_float(s.loss_avg_20))
        table.add_row("avg(100)", format_float(s.loss_avg_100))
        table.add_row("lr", format_sci(s.lr))
        table.add_row("eval(last)", format_float(s.last_eval_loss))
        table.add_row("eval(best)", format_float(s.best_eval_loss))
        return Panel(table, border_style="green")

    def _render_throughput_panel(self, s: MonitorSnapshot) -> Panel:
        table = self._kv_table("Throughput")
        table.add_row(
            "tok/s(update)",
            format_int(int(s.update_tokens_per_s)) if s.update_tokens_per_s is not None else "n/a",
        )
        table.add_row(
            "tok/s(micro)",
            format_int(int(s.micro_tokens_per_s)) if s.micro_tokens_per_s is not None else "n/a",
        )
        table.add_row(
            "step(update)",
            f"{format_float(s.update_step_time_ms, 1)} ms" if s.update_step_time_ms is not None else "n/a",
        )
        table.add_row(
            "data(avg)",
            f"{format_float(s.data_time_ms_avg, 1)} ms" if s.data_time_ms_avg is not None else "n/a",
        )
        table.add_row(
            "fwd+bwd(avg)",
            f"{format_float(s.fwdbwd_time_ms_avg, 1)} ms" if s.fwdbwd_time_ms_avg is not None else "n/a",
        )
        table.add_row(
            "optim",
            f"{format_float(s.optim_time_ms, 1)} ms" if s.optim_time_ms is not None else "n/a",
        )
        table.add_row("trained tok", format_int(s.trained_tokens))
        table.add_row("trained samp", format_int(s.trained_samples))
        return Panel(table, border_style="yellow")

    def _render_data_panel(self, s: MonitorSnapshot) -> Panel:
        table = self._kv_table("Data Quality")
        valid_text = "n/a"
        if s.update_valid_tokens is not None and s.update_total_tokens is not None:
            valid_text = f"{format_int(s.update_valid_tokens)} / {format_int(s.update_total_tokens)}"

        table.add_row("micro batch", format_int(s.micro_batch_size))
        table.add_row("eff batch", format_int(s.effective_batch_size))
        table.add_row("seq len", format_int(s.seq_len))
        table.add_row("valid tok(update)", valid_text)
        table.add_row("eff batch tok", format_int(s.effective_batch_tokens))
        table.add_row("avg valid len", format_float(s.avg_valid_len, 1))
        table.add_row("pad ratio(update)", format_percent(s.update_padding_ratio))
        return Panel(table, border_style="cyan")

    def _render_stability_panel(self, s: MonitorSnapshot) -> Panel:
        danger = s.has_nan or s.has_inf
        table = self._kv_table("Stability & Resource")
        table.add_row("grad pre-clip", format_float(s.grad_norm_pre_clip, 3))
        table.add_row("grad post-clip", format_float(s.grad_norm_post_clip, 3))
        table.add_row("max grad norm", format_float(s.max_grad_norm, 3))
        table.add_row("was clipped", "yes" if s.was_clipped else "no")
        table.add_row("clip ratio", format_float(s.clip_ratio, 3))
        table.add_row("nan / inf", f"{'yes' if s.has_nan else 'no'} / {'yes' if s.has_inf else 'no'}")
        table.add_row("gpu alloc", format_bytes(s.gpu_mem_alloc))
        table.add_row("gpu reserved", format_bytes(s.gpu_mem_reserved))
        table.add_row("gpu peak", format_bytes(s.gpu_peak_mem))
        return Panel(table, border_style="red" if danger else "magenta")

    def _render_footer(self, s: MonitorSnapshot) -> Panel:
        style = "bold red" if (s.has_nan or s.has_inf) else ("bold yellow" if s.warnings else "bold green")
        return Panel(Text(s.status_text, style=style), border_style="bright_black")
