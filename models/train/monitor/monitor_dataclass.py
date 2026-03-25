from dataclasses import dataclass, field


@dataclass
class MonitorSnapshot:
    run_name: str = "pretrain"
    mode: str = "train"

    epoch: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    max_optimizer_steps: int | None = None

    gradient_accumulation_steps: int = 1
    accum_progress: str = "0/1"

    elapsed_s: float = 0.0
    eta_s: float | None = None

    loss_update: float | None = None
    loss_micro_raw: float | None = None
    loss_avg_20: float | None = None
    loss_avg_100: float | None = None
    lr: float | None = None
    last_eval_loss: float | None = None
    best_eval_loss: float | None = None

    update_tokens_per_s: float | None = None
    micro_tokens_per_s: float | None = None
    update_step_time_ms: float | None = None
    data_time_ms_avg: float | None = None
    fwdbwd_time_ms_avg: float | None = None
    optim_time_ms: float | None = None
    trained_tokens: int = 0
    trained_samples: int = 0

    micro_batch_size: int | None = None
    effective_batch_size: int | None = None
    seq_len: int | None = None
    update_valid_tokens: int | None = None
    update_total_tokens: int | None = None
    effective_batch_tokens: int | None = None
    avg_valid_len: float | None = None
    update_padding_ratio: float | None = None

    grad_norm_pre_clip: float | None = None
    grad_norm_post_clip: float | None = None
    max_grad_norm: float | None = None
    was_clipped: bool = False
    clip_ratio: float | None = None
    has_nan: bool = False
    has_inf: bool = False

    gpu_mem_alloc: int | None = None
    gpu_mem_reserved: int | None = None
    gpu_peak_mem: int | None = None

    status_text: str = "starting..."
    warnings: list[str] = field(default_factory=list)
