import random
from multiprocessing import Value
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerFast


class PretrainingDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int,
        *,
        text_column: str = "text",
        parquet_batch_size: int = 2048,
        tokenizer_batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        glob_pattern: str = "*.parquet",
        compact_threshold: int = 1 << 16,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.rglob(glob_pattern))

        if not self.files:
            raise FileNotFoundError(f"No parquet files found under: {self.data_dir}")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id: int = self.tokenizer.eos_token_id  # type: ignore
        self.pad_token_id: int = self.tokenizer.pad_token_id  # type: ignore
        self.text_column = text_column
        self.parquet_batch_size = parquet_batch_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.compact_threshold = max(8 * self.max_seq_len, compact_threshold)
        self.tokenizer.model_max_length = int(1e9)
        self._shared_epoch = Value("i", 0)

    def __iter__(self):
        files_to_process = self._get_sharded_files()

        token_buffer: list[int] = []
        buffer_start = 0

        for file_path in files_to_process:
            for texts in self._iter_text_batches(file_path):
                encoded = self.tokenizer(
                    texts,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )

                for token_ids in encoded["input_ids"]:  # type: ignore
                    if not token_ids:
                        continue

                    token_buffer.extend(token_ids)
                    token_buffer.append(self.eos_token_id)

                    while len(token_buffer) - buffer_start >= self.max_seq_len:
                        chunk = token_buffer[buffer_start : buffer_start + self.max_seq_len]
                        yield self._build_sample(token_ids=chunk, valid_length=self.max_seq_len)
                        buffer_start += self.max_seq_len

                    if buffer_start >= self.compact_threshold and buffer_start >= len(token_buffer) // 2:
                        token_buffer = token_buffer[buffer_start:]
                        buffer_start = 0

        remaining = len(token_buffer) - buffer_start
        if remaining > 0 and not self.drop_last:
            chunk = token_buffer[buffer_start:]
            yield self._build_sample(token_ids=chunk, valid_length=remaining)

    def _build_sample(self, token_ids: list[int], valid_length: int) -> dict[str, torch.Tensor | int]:
        return {"input_ids": torch.tensor(token_ids, dtype=torch.long), "valid_length": valid_length}

    def set_epoch(self, epoch: int):
        self._shared_epoch.value = int(epoch)

    def _get_sharded_files(self) -> list[Path]:
        files = list(self.files)

        if self.shuffle:
            epoch = self._shared_epoch.value
            rng = random.Random(self.seed + epoch)
            rng.shuffle(files)

        rank, world_size = self._get_dist_info()
        if world_size > 1:
            files = files[rank::world_size]

        worker_info = get_worker_info()
        if worker_info:
            files = files[worker_info.id :: worker_info.num_workers]
        return files

    @staticmethod
    def _get_dist_info() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _iter_text_batches(self, file_path: Path) -> Iterator[list[str]]:
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(
            batch_size=self.parquet_batch_size, columns=[self.text_column], use_threads=True
        ):
            values = batch.column(0).to_pylist()
            texts = [x for x in values if isinstance(x, str) and x.strip()]
            if not texts:
                continue

            yield from self._split_texts(texts)

    def _split_texts(self, texts: list[str]) -> Iterator[list[str]]:
        for start in range(0, len(texts), self.tokenizer_batch_size):
            yield texts[start : start + self.tokenizer_batch_size]
