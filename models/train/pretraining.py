import json
import math
import os
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime

import click
import torch
import torch.distributed as dist
import torch.optim as optim
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer

from models.causal_lm import CausalLanguageModel
from models.config import TransformerConfig
from models.train.collator import Collator
from models.train.dataset import PretrainingDataset, load_pretraining_splits
from models.train.loss import compute_loss, eval_compute_loss


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
        tensorboard_log_dir: str | None = None,
        tensorboard_flush_secs: int = 30,
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self._load_config(metadata_path, model_config_path)
        self._init_distributed()
        self.is_main_process = (not self.is_distributed) or dist.get_rank() == 0

        self._init_tokenizer_and_model(model_path)
        self._load_dataset(manifest_path)
        self._load_dataloader()

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
        self.eval_interval = max(1, math.ceil(eval_steps_ratio * self.max_steps))
        self.logging_steps = logging_steps

        self._init_scheduler()
        self._init_tensorboard(
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_flush_secs=tensorboard_flush_secs,
        )

        self.monitor_data = {
            "last_train_loss": 0.0,
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
                        self.monitor_data["last_train_loss"] = train_loss
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
                                print(f"[Eval] start at optimizer step {optimizer_steps}/{self.max_steps}")

                            self.monitor_data["last_eval_loss"] = self._evaluate()
                            try:
                                self.monitor_data["last_eval_ppl"] = math.exp(self.monitor_data["last_eval_loss"])
                            except OverflowError:
                                self.monitor_data["last_eval_ppl"] = float("inf")

                            if self.is_main_process:
                                self._log_eval_metrics(epoch=epoch, optimizer_steps=optimizer_steps)

                        if self.is_main_process and (
                            optimizer_steps == 1
                            or optimizer_steps % self.logging_steps == 0
                            or optimizer_steps % self.eval_interval == 0
                            or optimizer_steps == self.max_steps
                        ):
                            self._log_train_metrics(epoch=epoch, optimizer_steps=optimizer_steps)

                        self.optimizer.zero_grad(set_to_none=True)

                if optimizer_steps >= self.max_steps:
                    break

            epoch += 1

        if self.is_main_process:
            state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
            save_file(state_dict, f"{self.output_path}/model.safetensors")
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

        if self.is_distributed:
            dist.destroy_process_group()

    def _log_train_metrics(self, epoch: int, optimizer_steps: int):
        if self.writer is None:
            return

        elapsed = time.perf_counter() - self.monitor_data["start_time"]
        tokens_seen = optimizer_steps * self.token_per_update
        tokens_per_sec = tokens_seen / max(elapsed, 1e-6)
        eta_seconds = (self.max_steps - optimizer_steps) * (elapsed / max(optimizer_steps, 1))
        avg_train_loss = sum(self.monitor_data["recent_train_losses"]) / max(
            len(self.monitor_data["recent_train_losses"]), 1
        )

        self.writer.add_scalar("train/loss", self.monitor_data["last_train_loss"], optimizer_steps)
        self.writer.add_scalar("train/loss_avg_100", avg_train_loss, optimizer_steps)
        self.writer.add_scalar("train/grad_norm", self.monitor_data["last_grad_norm"], optimizer_steps)
        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], optimizer_steps)
        self.writer.add_scalar("train/tokens_seen", tokens_seen, optimizer_steps)
        self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, optimizer_steps)
        self.writer.add_scalar("train/eta_seconds", eta_seconds, optimizer_steps)
        self.writer.add_scalar("train/epoch", epoch, optimizer_steps)
        self.writer.flush()

    def _log_eval_metrics(self, epoch: int, optimizer_steps: int):
        if self.writer is None:
            return

        self.writer.add_scalar("eval/loss", self.monitor_data["last_eval_loss"], optimizer_steps)
        self.writer.add_scalar("eval/ppl", self.monitor_data["last_eval_ppl"], optimizer_steps)
        self.writer.add_scalar("eval/epoch", epoch, optimizer_steps)
        self.writer.flush()

    def _load_config(self, metadata_path: str, model_config_path: str):
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(model_config_path, mode="r", encoding="utf-8") as f:
            self.config = TransformerConfig.model_validate_json(f.read())

    def _init_distributed(self):
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
        collator = Collator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_seq_len=self.config.max_position_embeddings,
        )
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

    def _init_tensorboard(
        self,
        *,
        tensorboard_log_dir: str | None,
        tensorboard_flush_secs: int,
    ):
        self.writer: SummaryWriter | None = None
        self.tensorboard_log_dir = tensorboard_log_dir
        if not self.is_main_process:
            return

        if self.tensorboard_log_dir is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self.tensorboard_log_dir = os.path.join(self.output_path, "tensorboard", run_name)

        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=self.tensorboard_log_dir,
            flush_secs=tensorboard_flush_secs,
        )

        self.writer.add_text("config/model", self.config.model_dump_json(indent=2), 0)
        self.writer.add_text("config/metadata", json.dumps(self.metadata, ensure_ascii=False, indent=2), 0)
        self.writer.add_text(
            "config/train_args",
            json.dumps(
                {
                    "lr": self.lr,
                    "per_device_train_batch_size": self.per_device_train_batch_size,
                    "per_device_eval_batch_size": self.per_device_eval_batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "warmup_ratio": self.warmup_ratio,
                    "warmup_start_factor": self.warmup_start_factor,
                    "max_grad_norm": self.max_grad_norm,
                    "eval_steps_ratio": self.eval_interval / self.max_steps,
                    "logging_steps": self.logging_steps,
                    "world_size": self.world_size,
                    "token_per_update": self.token_per_update,
                    "max_steps": self.max_steps,
                },
                ensure_ascii=False,
                indent=2,
            ),
            0,
        )
        self.writer.flush()
        print(f"[TensorBoard] log_dir: {self.tensorboard_log_dir}")

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


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option("--per_device_train_batch_size", default=16, type=int)
@click.option("--per_device_eval_batch_size", default=8, type=int)
@click.option("--gradient_accumulation_steps", default=1, type=int)
@click.option("--warmup_ratio", default=0.03, type=float)
@click.option("--warmup_start_factor", default=0.1, type=float)
@click.option("--max_grad_norm", default=None, type=float)
@click.option("--eval_steps_ratio", default=0.1, type=float)
@click.option("--logging_steps", default=100, type=int)
@click.option("--output_path", default="weight", type=str)
@click.pass_context
def train(
    ctx,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    warmup_ratio,
    warmup_start_factor,
    max_grad_norm,
    eval_steps_ratio,
    logging_steps,
    output_path,
):
    trainer = PretrainingTrainer(
        metadata_path="models/train/config_files/metadata.json",
        model_config_path="models/train/config_files/1B.json",
        model_path="weight",
        manifest_path="models/train/manifest",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        warmup_start_factor=warmup_start_factor,
        max_grad_norm=max_grad_norm,
        eval_steps_ratio=eval_steps_ratio,
        logging_steps=logging_steps,
        output_path=output_path,
        tensorboard_log_dir="/workspace",
    )
    trainer.train()


if __name__ == "__main__":
    train()
