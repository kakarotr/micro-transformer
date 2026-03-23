import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments

from models.causal_lm import CausalLanguageModel
from models.config import TransformerConfig
from models.train.collator import PretrainingCollator
from models.train.dataset import PretrainingDataset, create_pretraining_splits

TrainingArguments()


def train(
    model: nn.Module,
    train_dataset: PretrainingDataset,
    valid_dataset: PretrainingDataset,
    *,
    output_dir: str,
    collattor: PretrainingCollator,
    per_device_batch: int,
    epochs: int,
    lr: float = 5e-4,
    dataset_num_workers: int = 1,
    gradient_accumulation_steps: int = 1,
):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_batch,
        num_workers=dataset_num_workers,
        collate_fn=collattor,
        pin_memory=True,
        persistent_workers=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=True, fused=True)
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        for input_ids, attention_mask, labels in train_dataloader:
            loss, _ = model(input_ids, labels, attention_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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

    model = CausalLanguageModel(config=config)

    train_files, valid_files, _ = create_pretraining_splits(
        data_dir="data/common",
        split_dir="models/train",
        glob_pattern="*.parquet",
    )
    train_dataset = PretrainingDataset(train_files, tokenizer=tokenizer, max_seq_len=4096)
    valid_dataset = PretrainingDataset(valid_files, tokenizer=tokenizer, max_seq_len=4096)

    collattor = PretrainingCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=config.max_position_embeddings,
    )

    train(
        model,
        train_dataset,
        valid_dataset,
        per_device_batch=16,
        epochs=3,
        dataset_num_workers=2,
        collattor=collattor,
        output_dir="",
    )
