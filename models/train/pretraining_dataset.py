import random
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class CommonPretrainingDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_seq_len: int):
        self.files = [file for file in Path("data/common/4_5").rglob("*.parquet")]
        random.shuffle(self.files)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = int(1e9)
        self.max_seq_len = max_seq_len

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info:
            files_to_process = self.files[worker_info.id :: worker_info.num_workers]
        else:
            files_to_process = self.files

        text_buffer = []
        for file_path in files_to_process:
            df = pl.read_parquet(file_path)
            for text in df["text"]:
                token_ids = self.tokenizer.encode(text)
                token_ids.append(self.tokenizer.eos_token_id)  # type: ignore
                text_buffer.extend(token_ids)

                while len(text_buffer) >= self.max_seq_len:
                    chunk = text_buffer[: self.max_seq_len]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    yield (input_ids, input_ids.clone())
                    text_buffer = text_buffer[self.max_seq_len :]


if __name__ == "__main__":
    dataset = CommonPretrainingDataset(tokenizer=AutoTokenizer.from_pretrained("weight"), max_seq_len=128)
    data_loader = DataLoader(dataset=dataset, batch_size=8, num_workers=1)

    for batch in data_loader:
        print(batch)
        break
