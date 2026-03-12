import operator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from models.config import TransformerConfig
from models.decoder import CausalLanguageModel
from models.train.pretraining_dataset import CommonPretrainingDataset


def train(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    epochs: int,
    lr: float = 1e-3,
    dataset_num_workers: int = 1,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=dataset_num_workers)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=True, fused=True)
    for index in enumerate(range(epochs), start=1):
        for input_ids, labels in dataloader:
            loss, _ = model(input_ids, labels)
            loss: torch.Tensor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
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
    )
    model = CausalLanguageModel(config=config)
    dataset = CommonPretrainingDataset(tokenizer=AutoTokenizer.from_pretrained("weight"), max_seq_len=4096)
    train(model, dataset, batch_size=16, epochs=3, dataset_num_workers=2)
