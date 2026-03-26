import time
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.causal_lm import CausalLanguageModel
from models.config import TransformerConfig
from models.train.collator import PretrainingCollator
from models.train.dataset import FileDataset, load_pretraining_splits
from models.train.monitor.monitor_dataclass import MonitorSnapshot
from models.train.monitor.rich import MetricTracker, RichMonitor, UpdateAccumulator
from models.train.monitor.utils import get_grad_norm


def train(
    model: nn.Module,
    train_dataset: FileDataset,
    valid_dataset: FileDataset,
    *,
    output_dir: str,
    collattor: PretrainingCollator,
    per_device_batch: int,
    num_epochs: int,
    num_samples: int,
    lr: float = 5e-4,
    dataset_num_workers: int = 1,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
    device: torch.device | None = None,
):
    monitor = RichMonitor()
    tracker = MetricTracker(run_name="pretrain")
    accum = UpdateAccumulator(gradient_accumulation_steps=gradient_accumulation_steps)
    initial = MonitorSnapshot(
        run_name="pretrain",
        status_text="Initializing...",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    max_steps = ceil(num_samples / (per_device_batch * 1 * gradient_accumulation_steps))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_batch,
        num_workers=dataset_num_workers,
        collate_fn=collattor,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=True, fused=True)

    optimizer_step = 0
    micro_step = 0

    with monitor.live(initial, refresh_per_second=4) as live:
        for epoch in range(num_epochs):
            model.train()
            train_dataset.set_epoch(epoch)

            for batch in train_dataloader:
                t0 = time.perf_counter()
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                data_time_s = time.perf_counter() - t0

                t1 = time.perf_counter()
                loss, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss_raw = float(loss.item())

                (loss / gradient_accumulation_steps).backward()
                fwdbwd_time_s = time.perf_counter() - t1

                fwdbwd_time_s = time.perf_counter() - t1
                micro_step_time_s = data_time_s + fwdbwd_time_s

                accum.add_micro_step(
                    loss_raw=loss_raw,
                    batch=batch,
                    data_time_s=data_time_s,
                    fwdbwd_time_s=fwdbwd_time_s,
                    micro_step_time_s=micro_step_time_s,
                )

                current_lr = optimizer.param_groups[0]["lr"]

                pending_summary = accum.summary(
                    optim_time_s=None,
                    grad_norm_pre_clip=None,
                    grad_norm_post_clip=None,
                    max_grad_norm=max_grad_norm,
                )

                pending_snapshot = tracker.build_snapshot(
                    epoch=epoch,
                    micro_step=micro_step,
                    optimizer_step=optimizer_step,
                    lr=current_lr,
                    summary=pending_summary,
                    model=model,
                    committed=False,
                    mode="train",
                    max_optimizer_steps=max_steps,
                )
                live.update(monitor.render(pending_snapshot))

                if not accum.is_ready_to_step():
                    continue

                grad_norm_pre_clip = get_grad_norm(model.parameters())
                grad_norm_post_clip = grad_norm_pre_clip
                if max_grad_norm is not None:
                    grad_norm_pre_clip = float(
                        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
                    )
                    grad_norm_post_clip = get_grad_norm(model.parameters())

                t2 = time.perf_counter()
                optimizer.step()

                # [TODO] Scheduler

                optimizer.zero_grad(set_to_none=True)
                optim_time_s = time.perf_counter() - t2

                optimizer_step += 1
                current_lr = optimizer.param_groups[0]["lr"]

                committed_summary = accum.summary(
                    optim_time_s=optim_time_s,
                    grad_norm_pre_clip=grad_norm_pre_clip,
                    grad_norm_post_clip=grad_norm_post_clip,
                    max_grad_norm=max_grad_norm,
                )

                committed_snapshot = tracker.build_snapshot(
                    epoch=epoch,
                    micro_step=micro_step,
                    optimizer_step=optimizer_step,
                    lr=current_lr,
                    summary=committed_summary,
                    model=model,
                    committed=True,
                    mode="train",
                    max_optimizer_steps=max_steps,
                )
                live.update(monitor.render(committed_snapshot))

                accum.reset()

                if optimizer_step >= max_steps:
                    break
            if optimizer_step >= max_steps:
                break


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("weight")

    config = TransformerConfig(
        vocab_size=32768,
        max_position_embeddings=4096,
        hidden_size=1024,
        num_layers=48,
        num_attention_heads=16,
        num_key_value_heads=16,
        dropout_prob=0.0,
        intermediate_size=2816,
        rms_eps=1e-6,
        rope_base=10000,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(config.model_dump_json(indent=2))

    # model = CausalLanguageModel(config=config)

    # train_files, valid_files, _ = load_pretraining_splits(split_dir="models/train")
    # train_dataset = PretrainingDataset(
    #     train_files,
    #     tokenizer=tokenizer,
    #     max_seq_len=config.max_position_embeddings,
    #     shuffle=True,
    #     drop_last=True,
    # )
    # valid_dataset = PretrainingDataset(
    #     valid_files,
    #     tokenizer=tokenizer,
    #     max_seq_len=config.max_position_embeddings,
    #     shuffle=False,
    #     drop_last=False,
    # )

    # collattor = PretrainingCollator(
    #     pad_token_id=tokenizer.pad_token_id,
    #     max_seq_len=config.max_position_embeddings,
    # )

    # train(
    #     model,
    #     train_dataset,
    #     valid_dataset,
    #     per_device_batch=16,
    #     num_epochs=3,
    #     dataset_num_workers=2,
    #     collattor=collattor,
    #     output_dir="weight",
    #     num_samples=1000,
    # )
