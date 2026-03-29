import json
import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.optim as optim
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from models.causal_lm import CausalLanguageModel
from models.config import TransformerConfig
from models.train.collator import Collator
from models.train.dataset import PretrainingDataset, load_pretraining_splits
from models.train.loss import compute_loss


def load_config():
    """读取配置（数据集、模型）"""
    with open("models/train/config_files/metadata.json", mode="r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open("models/train/config_files/0.6B.json", mode="r", encoding="utf-8") as f:
        config = TransformerConfig.model_validate_json(f.read())

    return metadata, config


def init_distributed():
    """判断是否多卡环境并决定是否进行初始化"""
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


def init_tokenizer_and_model(local_rank: int):
    """初始化分词器、模型"""
    tokenizer = AutoTokenizer.from_pretrained("weight")
    model = CausalLanguageModel(config=config).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    return tokenizer, model


def load_dataset(tokenizer: PreTrainedTokenizerFast):
    """加载训练集、验证集"""
    train_files, eval_files, _ = load_pretraining_splits("models/train/manifest")
    train_dataset = PretrainingDataset(
        files=train_files,
        tokenizer=tokenizer,
        max_seq_len=config.max_position_embeddings,
        shuffle=True,
        drop_last=True,
    )
    eval_dataset = PretrainingDataset(
        files=eval_files,
        tokenizer=tokenizer,
        max_seq_len=config.max_position_embeddings,
        shuffle=False,
        drop_last=False,
    )
    return train_dataset, eval_dataset


def load_dataloader(tokenizer: PreTrainedTokenizerFast):
    """构建DataLoader"""
    collator = Collator(pad_token_id=tokenizer.pad_token_id, max_seq_len=config.max_position_embeddings)  # type: ignore
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collator,
        batch_size=per_device_train_batch_size,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_dataloader, eval_dataloader


def get_scheduler(max_steps: int):
    """LR 调度器"""
    warmup_steps = int(max_steps * warmup_ratio)
    if warmup_steps > 0:
        cosine_steps = max_steps - warmup_steps
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=cosine_steps),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)
    return scheduler


# 训练相关超参数
per_device_train_batch_size = 16
gradient_accumulation_steps = 1
base_lr = 5e-4
max_grad_norm: float | None = None
warmup_ratio = 0.03
start_factor = 0.1

# 验证相关参数
per_device_eval_batch_size = 16
eval_steps_ratio = 0.1


if __name__ == "__main__":
    metadata, config = load_config()

    is_distributed, world_size, local_rank, device = init_distributed()

    if (not torch.cuda.is_available()) or (not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Current device does not support bfloat16")

    tokenizer, model = init_tokenizer_and_model(local_rank=local_rank)

    train_dataset, eval_dataset = load_dataset(tokenizer)

    train_dataloader, eval_dataloader = load_dataloader(tokenizer)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, fused=True)

    total_tokens = metadata["num_pretain_tokens"]
    token_per_update = (
        per_device_train_batch_size * world_size * config.max_position_embeddings * gradient_accumulation_steps
    )
    max_steps = math.ceil(total_tokens / token_per_update)
    scheduler = get_scheduler(max_steps=max_steps)

    micro_steps = 0
    optimizer_steps = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)

    while optimizer_steps < max_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            is_update_step = ((micro_steps + 1) % gradient_accumulation_steps) == 0
            sync_context = model.no_sync() if is_distributed and not is_update_step else nullcontext()  # type: ignore

            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask)
                    loss = compute_loss(logits, labels)

                loss = loss / gradient_accumulation_steps
                loss.backward()

                micro_steps += 1

            if micro_steps % gradient_accumulation_steps == 0:
                if max_grad_norm is not None:
                    clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer_steps += 1

                optimizer.zero_grad(set_to_none=True)
            if optimizer_steps >= max_steps:
                break
        epoch += 1

    is_main_process = (not is_distributed) or dist.get_rank() == 0
    if is_main_process:
        state_dict = model.module.state_dict() if is_distributed else model.state_dict()  # type:ignore
        save_file(state_dict, "weight/model.safetensors")

    if is_distributed:
        dist.destroy_process_group()
