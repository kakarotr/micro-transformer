import json

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.config import TransformerConfig
from models.train.dataset import PretrainingDataset, load_pretraining_splits

if __name__ == "__main__":
    with open("models/train/metadata.json", mode="w", encoding="utf-8") as f:
        metadata = json.load(f)
    with open("models/model_configs/0.6B.json", mode="r", encoding="utf-8") as f:
        config = TransformerConfig.model_validate_json(f.read())

    tokenizer = AutoTokenizer.from_pretrained("weight")

    train_num_samples = metadata["train_num_pretrain_samples"]
    train_files, eval_files, _ = load_pretraining_splits("models/train/manifest")
    train_dataset = PretrainingDataset(
        files=train_files,
        tokenizer=tokenizer,
        max_seq_len=config.max_position_embeddings,
    )
    eval_dataset = PretrainingDataset(
        files=eval_files,
        tokenizer=tokenizer,
        max_seq_len=config.max_position_embeddings,
    )
