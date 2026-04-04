import json
import math
import os
import time
from argparse import ArgumentParser
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import click
import torch
import torch.distributed as dist
import torch.optim as optim
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer

from models.causal_lm import CausalLanguageModel
from models.config import TransformerConfig
from models.train.dataset import PackedTokenDataset
from models.train.loss import compute_loss, eval_compute_loss
from models.train.main import TrainingArguments


class PretrainingTrainer:
    def __init__(
        self,
        *,
        model_path: str,
        arguments: TrainingArguments,
        data_dir: str,
        output_path: str,
        log_dir: str = "/workspace",
        log_flush_secs: int = 30,
    ):
        self.arguments = arguments
        self.output_path = output_path

        self.is_distributed, self.world_size, self.local_rank, self.device = self._init_distributed()
        self.is_main_process = (not self.is_distributed) or dist.get_rank() == 0
        self.train_dataset, self.eval_dataset = self._load_dataset(data_dir)
        self.train_dataloader, self.eval_dataloader = self._load_dataloader()
        self.config, self.tokenizer, self.model = self._get_tokenizer_and_model(model_path)
        self.token_per_update = (
            self.arguments.per_device_train_batch_size
            * self.world_size
            * self.config.max_position_embeddings
            * self.arguments.gradient_accumulation_steps
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.arguments.learning_rate, fused=True)
        self.scheduler = self._init_scheduler()

        self._init_tensorboard(log_dir, log_flush_secs)
        self.monitor_data = {
            "last_train_loss": 0.0,
            "last_eval_loss": 0.0,
            "last_eval_ppl": 0.0,
            "last_grad_norm": 0.0,
            "recent_train_losses": deque(maxlen=100),
            "start_time": time.perf_counter(),
        }

    def __call__(self):
        micro_steps = 0
        optimizer_steps = 0
        epoch = 0
        self.monitor_data["start_time"] = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss_sum = 0.0
        accum_token_count = 0

        while optimizer_steps < self.arguments.max_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for batch in self.train_dataloader:
                input_ids = batch.to(device=self.device, non_blocking=True)
                labels = input_ids

                is_update_step = ((micro_steps + 1) % self.arguments.gradient_accumulation_steps) == 0
                sync_context = self.model.no_sync() if self.is_distributed and not is_update_step else nullcontext()

                with sync_context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hidden_states = self.model(input_ids)
                        lm_head_weight = (
                            self.model.lm_head.weight if not self.is_distributed else self.model.module.lm_head.weight
                        )
                        raw_loss = compute_loss(hidden_states, lm_head_weight, labels)

                    valid_token_count = labels[:, 1:].ne(-100).sum().item()
                    accum_loss_sum += raw_loss.detach().item() * valid_token_count
                    accum_token_count += valid_token_count

                    loss = raw_loss / self.arguments.gradient_accumulation_steps
                    loss.backward()
                    micro_steps += 1

                    if micro_steps % self.arguments.gradient_accumulation_steps == 0:
                        train_loss = accum_loss_sum / max(accum_token_count, 1)
                        self.monitor_data["last_train_loss"] = train_loss
                        self.monitor_data["recent_train_losses"].append(train_loss)
                        accum_loss_sum = 0.0
                        accum_token_count = 0

                        grad_norm = clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.arguments.max_grad_norm if self.arguments.max_grad_norm else float("inf"),
                        )
                        self.monitor_data["last_grad_norm"] = grad_norm.item()

                        self.optimizer.step()
                        self.scheduler.step()
                        optimizer_steps += 1

                        if (
                            optimizer_steps % self.arguments.eval_steps == 0
                            or optimizer_steps == self.arguments.max_steps
                        ):
                            if self.is_main_process:
                                print(f"[Eval] start at optimizer step {optimizer_steps}/{self.arguments.max_steps}")

                            self.monitor_data["last_eval_loss"] = self._evaluate()
                            try:
                                self.monitor_data["last_eval_ppl"] = math.exp(self.monitor_data["last_eval_loss"])
                            except OverflowError:
                                self.monitor_data["last_eval_ppl"] = float("inf")

                            if self.is_main_process:
                                self._log_eval_metrics(epoch=epoch, optimizer_steps=optimizer_steps)

                        if self.is_main_process and (
                            optimizer_steps == 1
                            or optimizer_steps % self.arguments.logging_steps == 0
                            or optimizer_steps % self.arguments.eval_steps == 0
                            or optimizer_steps == self.arguments.max_steps
                        ):
                            self._log_train_metrics(epoch=epoch, optimizer_steps=optimizer_steps)

                        if optimizer_steps % self.arguments.save_steps == 0:
                            self._save_checkpoint(optimizer_steps)

                        self.optimizer.zero_grad(set_to_none=True)

                if optimizer_steps >= self.arguments.max_steps:
                    break

            epoch += 1

        self._save_checkpoint(optimizer_steps=None)

        if self.is_distributed:
            dist.destroy_process_group()

    def _save_checkpoint(self, optimizer_steps: int | None):
        if self.is_main_process:
            checkpoint_path = (
                Path(f"{self.output_path}/checkpoint-{optimizer_steps}/model.safetensors")
                if optimizer_steps is not None
                else Path(f"{self.output_path}/model.safetensors")
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
            save_file(state_dict, checkpoint_path)

            if optimizer_steps is None and self.writer is not None:
                self.writer.flush()
                self.writer.close()

    def _init_distributed(self):
        is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if is_distributed:
            backend = "nccl" if dist.is_nccl_available() else "gloo"
            dist.init_process_group(backend=backend)
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            world_size = 1
            local_rank = 0
            device = torch.device("cuda")

        return is_distributed, world_size, local_rank, device

    def _load_dataset(self, data_dir):
        train_dataset = PackedTokenDataset(Path(f"{data_dir}/train"))
        eval_dataset = PackedTokenDataset(Path(f"{data_dir}/eval"))

        return train_dataset, eval_dataset

    def _load_dataloader(self):
        train_sampler = None
        eval_sampler = None
        self.train_sampler = None

        if self.is_distributed:
            train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=True, drop_last=True)
            eval_sampler = DistributedSampler(dataset=self.eval_dataset, shuffle=False, drop_last=False)
            self.train_sampler = train_sampler

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.arguments.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            pin_memory=True,
            drop_last=True,
        )
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.arguments.per_device_eval_batch_size,
            sampler=eval_sampler,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        return train_dataloader, eval_dataloader

    def _get_tokenizer_and_model(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(f"{model_path}/config.json", mode="r", encoding="utf-8") as f:
            config = TransformerConfig.model_validate_json(f.read())
        model = CausalLanguageModel(config=config).to(self.device)
        if self.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        model = torch.compile(model, mode="default")

        return config, tokenizer, model

    def _init_scheduler(self):
        if self.arguments.warmup_steps > 0:
            cosine_steps = self.arguments.max_steps - self.arguments.warmup_steps
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(
                        self.optimizer,
                        start_factor=self.arguments.warmup_start_factor,
                        total_iters=self.arguments.warmup_steps,
                    ),
                    CosineAnnealingLR(self.optimizer, T_max=cosine_steps),
                ],
                milestones=[self.arguments.warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.arguments.max_steps)
        return scheduler

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()

        loss_sum = torch.zeros(1, device=self.device)
        token_count = torch.zeros(1, device=self.device, dtype=torch.long)

        for batch in self.eval_dataloader:
            input_ids = batch.to(device=self.device, non_blocking=True)
            labels = input_ids

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden_states = self.model(input_ids)
                lm_head_weight = (
                    self.model.lm_head.weight if not self.is_distributed else self.model.module.lm_head.weight
                )
                loss, valid_token_count = eval_compute_loss(hidden_states, lm_head_weight, labels)

            loss_sum += loss
            token_count += valid_token_count

        if self.is_distributed:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

        mean_loss = (loss_sum / token_count.clamp_min(1)).item()
        self.model.train()
        return mean_loss

    def _init_tensorboard(
        self,
        log_dir: str,
        flush_secs: int,
    ):
        self.writer: SummaryWriter | None = None
        if not self.is_main_process:
            return

        if log_dir is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            log_dir = os.path.join(self.output_path, "tensorboard", run_name)  # type: ignore

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

        self.writer.add_text(
            "config/train_args",
            json.dumps(
                {
                    "lr": self.arguments.learning_rate,
                    "per_device_train_batch_size": self.arguments.per_device_train_batch_size,
                    "per_device_eval_batch_size": self.arguments.per_device_eval_batch_size,
                    "gradient_accumulation_steps": self.arguments.gradient_accumulation_steps,
                    "warmup_ratio": self.arguments.warmup_steps_ratio,
                    "warmup_start_factor": self.arguments.warmup_start_factor,
                    "max_grad_norm": self.arguments.max_grad_norm,
                    "eval_steps_ratio": self.arguments.eval_steps_ratio,
                    "logging_steps": self.arguments.logging_steps,
                    "world_size": self.world_size,
                    "token_per_update": self.token_per_update,
                    "max_steps": self.arguments.max_steps,
                },
                ensure_ascii=False,
                indent=2,
            ),
            0,
        )
        self.writer.flush()
        print(f"[TensorBoard] log_dir: {log_dir}")

    def _log_train_metrics(self, epoch: int, optimizer_steps: int):
        if self.writer is None:
            return

        elapsed = time.perf_counter() - self.monitor_data["start_time"]
        tokens_seen = optimizer_steps * self.token_per_update
        tokens_per_sec = tokens_seen / max(elapsed, 1e-6)
        eta_seconds = (self.arguments.max_steps - optimizer_steps) * (elapsed / max(optimizer_steps, 1))
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


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="训练脚本参数")

    for name, field in TrainingArguments.model_fields.items():
        arg_name = f"--{name}"
        arg_type = field.annotation
        default = field.default
        help_text = field.description or ""

        parser.add_argument(
            arg_name,
            type=arg_type,  # type: ignore
            default=default,
            help=f"{help_text} (default: {default})",
        )

    return parser


def parse_args() -> TrainingArguments:
    parser = build_parser()
    namespace = parser.parse_args()
    return TrainingArguments(**vars(namespace))


if __name__ == "__main__":
    arguments = parse_args()
    trainer = PretrainingTrainer(
        model_path="weight",
        arguments=arguments,
        data_dir="data/pretraining",
        output_path="weight",
        log_dir="logs",
    )
