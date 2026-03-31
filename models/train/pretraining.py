import json
import math
import os
import time
from collections import deque
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.optim as optim
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.causal_lm import CausalLanguageModel
from models.config import TransformerConfig
from models.train.collator import Collator
from models.train.dataset import PretrainingDataset, load_pretraining_splits
from models.train.loss import compute_loss, eval_compute_loss
from models.train.monitor import build_dashboard, build_status_table


class PretrainingTrainer:
    def __init__(
        self,
        *,
        metadata_path: str,
        model_config_path: str,
        model_path: str,
        manifest_path: str,
        lr: float = 5e-4,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.03,
        warmup_start_factor: float = 0.1,
        max_grad_norm: float | None = None,
        eval_steps_ratio: float = 0.1,
        logging_steps: int = 100,
        output_path: str = "weight",
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_path = output_path
        self._load_config(metadata_path, model_config_path)
        self._init_distributed()
        self._init_tokenizer_and_model(model_path)
        self._load_dataset(manifest_path)
        self._load_dataloader()
        self.is_main_process = (not self.is_distributed) or dist.get_rank() == 0
        self.lr = lr
        self.total_tokens: int = self.metadata["num_pretain_tokens"]
        self.token_per_update = (
            self.per_device_train_batch_size
            * self.world_size
            * self.config.max_position_embeddings
            * self.gradient_accumulation_steps
        )
        self.warmup_start_factor = warmup_start_factor
        self.warmup_ratio = warmup_ratio
        self.max_steps = math.ceil(self.total_tokens / self.token_per_update)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, fused=True)
        self.max_grad_norm = max_grad_norm
        self.eval_interval = math.ceil(eval_steps_ratio * self.max_steps)
        self._init_scheduler()
        self.logging_steps = logging_steps
        self._init_monitor()
        self.monitor_data = {
            "last_eval_loss": 0.0,
            "last_eval_ppl": 0.0,
            "last_grad_norm": 0.0,
            "recent_train_losses": deque(maxlen=100),
            "start_time": time.perf_counter(),
        }

    def train(self):
        micro_steps = 0
        optimizer_steps = 0
        epoch = 0
        self.monitor_data["start_time"] = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss_sum = 0.0
        accum_token_count = 0

        with self.live_context as live:
            while optimizer_steps < self.max_steps:
                self.train_dataset.set_epoch(epoch)
                for batch in self.train_dataloader:
                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                    is_update_step = ((micro_steps + 1) % self.gradient_accumulation_steps) == 0
                    sync_context = self.model.no_sync() if self.is_distributed and not is_update_step else nullcontext()

                    with sync_context:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = self.model(input_ids, attention_mask)
                            raw_loss = compute_loss(logits, labels)

                        valid_token_count = labels[:, 1:].ne(-100).sum().item()
                        accum_loss_sum += raw_loss.detach().item() * valid_token_count
                        accum_token_count += valid_token_count

                        loss = raw_loss / self.gradient_accumulation_steps
                        loss.backward()

                        micro_steps += 1

                        if micro_steps % self.gradient_accumulation_steps == 0:
                            train_loss = accum_loss_sum / max(accum_token_count, 1)
                            self.monitor_data["recent_train_losses"].append(train_loss)
                            accum_loss_sum = 0.0
                            accum_token_count = 0

                            grad_norm = clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.max_grad_norm if self.max_grad_norm else float("inf"),
                            )
                            self.monitor_data["last_grad_norm"] = grad_norm.item()

                            self.optimizer.step()
                            self.scheduler.step()
                            optimizer_steps += 1

                            if optimizer_steps % self.eval_interval == 0 or optimizer_steps == self.max_steps:
                                if self.is_main_process:
                                    assert self.progress is not None and self.progress_task_id is not None
                                    self.progress.update(
                                        self.progress_task_id, description="evaluating", completed=optimizer_steps
                                    )

                                self.monitor_data["last_eval_loss"] = self._evaluate()
                                if self.is_main_process:
                                    try:
                                        self.monitor_data["last_eval_ppl"] = math.exp(
                                            self.monitor_data["last_eval_loss"]
                                        )
                                    except OverflowError:
                                        self.monitor_data["last_eval_ppl"] = float("inf")

                            if self.is_main_process and (
                                optimizer_steps == 1
                                or optimizer_steps % self.logging_steps == 0
                                or optimizer_steps % self.eval_interval == 0
                                or optimizer_steps == self.max_steps
                            ):
                                self._refresh_monitor_data(live, epoch, optimizer_steps)

                            self.optimizer.zero_grad(set_to_none=True)

                            if self.is_main_process:
                                assert self.progress is not None and self.progress_task_id is not None
                                self.progress.update(
                                    self.progress_task_id, description="training", completed=optimizer_steps
                                )
                        if optimizer_steps >= self.max_steps:
                            break
                epoch += 1
        if self.is_main_process:
            state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
            save_file(state_dict, f"{self.output_path}/model.safetensors")

        if self.is_distributed:
            dist.destroy_process_group()

    def _refresh_monitor_data(
        self,
        live,
        epoch: int,
        optimizer_steps: int,
    ):
        assert self.progress is not None and self.progress_task_id is not None

        elapsed = time.perf_counter() - self.monitor_data["start_time"]
        tokens_seen = optimizer_steps * self.token_per_update
        tokens_per_sec = tokens_seen / (max(elapsed, 1e-6))
        remaining_steps = self.max_steps - optimizer_steps
        eta_seconds = remaining_steps * (elapsed / max(optimizer_steps, 1))
        train_loss = sum(self.monitor_data["recent_train_losses"]) / len(self.monitor_data["recent_train_losses"])

        self.progress.update(self.progress_task_id, completed=optimizer_steps)
        live.update(
            build_dashboard(
                self.progress,
                self.progress_task_id,
                table=build_status_table(
                    epoch=epoch,
                    step=optimizer_steps,
                    max_steps=self.max_steps,
                    train_loss=train_loss,
                    eval_loss=self.monitor_data["last_eval_loss"],
                    eval_ppl=self.monitor_data["last_eval_ppl"],
                    lr=self.optimizer.param_groups[0]["lr"],
                    grad_norm=self.monitor_data["last_grad_norm"],
                    tokens_seen=tokens_seen,
                    tokens_per_sec=tokens_per_sec,
                    eta_seconds=eta_seconds,
                ),
            )
        )

    def _load_config(self, metadata_path: str, model_config_path: str):
        """读取配置（数据集、模型）"""
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(model_config_path, mode="r", encoding="utf-8") as f:
            self.config = TransformerConfig.model_validate_json(f.read())

    def _init_distributed(self):
        """判断是否多卡环境并决定是否进行初始化"""
        self.is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if self.is_distributed:
            backend = "nccl" if dist.is_nccl_available() else "gloo"
            dist.init_process_group(backend=backend)
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda")

    def _init_tokenizer_and_model(self, model_path: str):
        """初始化分词器、模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = CausalLanguageModel(config=self.config).to(self.device)
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

    def _load_dataset(self, manifest_path: str):
        """加载训练集、验证集"""
        train_files, eval_files, _ = load_pretraining_splits(manifest_path)
        self.train_dataset = PretrainingDataset(
            files=train_files,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_position_embeddings,
            shuffle=True,
            drop_last=True,
        )
        self.eval_dataset = PretrainingDataset(
            files=eval_files,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_position_embeddings,
            shuffle=False,
            drop_last=False,
        )

    def _load_dataloader(self):
        """构建DataLoader"""
        collator = Collator(pad_token_id=self.tokenizer.pad_token_id, max_seq_len=self.config.max_position_embeddings)  # type: ignore
        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=collator,
            batch_size=self.per_device_train_batch_size,
            pin_memory=True,
            drop_last=True,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            collate_fn=collator,
            batch_size=self.per_device_eval_batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

    def _init_scheduler(self):
        """LR 调度器"""
        warmup_steps = int(self.max_steps * self.warmup_ratio)
        if warmup_steps > 0:
            cosine_steps = self.max_steps - warmup_steps
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(self.optimizer, start_factor=self.warmup_start_factor, total_iters=warmup_steps),
                    CosineAnnealingLR(self.optimizer, T_max=cosine_steps),
                ],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()

        loss_sum = torch.zeros(1, device=self.device)
        token_count = torch.zeros(1, device=self.device, dtype=torch.long)

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids, attention_mask)
                loss, valid_token_count = eval_compute_loss(logits, labels)

            loss_sum += loss
            token_count += valid_token_count

        if self.is_distributed:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

        mean_loss = (loss_sum / token_count.clamp_min(1)).item()
        self.model.train()
        return mean_loss

    def _init_monitor(self):
        self.progress = None
        self.progress_task_id = None
        console = Console()
        if self.is_main_process:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                BarColumn(),
                TextColumn("{task.completed:>8.0f} / {task.total:.0f}"),
            )
            self.progress_task_id = self.progress.add_task("Training", total=self.max_steps, completed=0)

        self.live_context = (
            Live(
                build_dashboard(progress=self.progress, task_id=self.progress_task_id, table=Table()),
                console=console,
                refresh_per_second=4,
                transient=False,
            )
            if self.is_main_process and self.progress is not None and self.progress_task_id is not None
            else nullcontext()
        )


if __name__ == "__main__":
    trainer = PretrainingTrainer(
        metadata_path="models/train/config_files/metadata.json",
        model_config_path="models/train/config_files/0.6B.json",
        model_path="weight",
        manifest_path="models/train/manifest",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
    )
    trainer.train()
