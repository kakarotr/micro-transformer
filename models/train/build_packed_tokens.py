# build_packed_tokens.py
from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_split_files(split_dir: str | Path) -> tuple[list[Path], list[Path]]:
    split_dir = Path(split_dir)

    def _read(path: Path) -> list[Path]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [Path(line.strip()) for line in lines if line.strip()]

    train_files = _read(split_dir / "train.txt")
    eval_files = _read(split_dir / "eval.txt")

    if not train_files:
        raise ValueError(f"train split is empty or missing under: {split_dir}")
    if not eval_files:
        raise ValueError(f"eval split is empty or missing under: {split_dir}")

    return train_files, eval_files


def choose_storage_dtype(vocab_size: int) -> np.dtype:
    # token id 最大值是 vocab_size - 1
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size too large for uint32: {vocab_size}")


def iter_text_batches(
    files: Sequence[Path],
    *,
    text_column: str,
    parquet_batch_size: int,
    tokenizer_batch_size: int,
) -> Iterator[list[str]]:
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(
            batch_size=parquet_batch_size,
            columns=[text_column],
            use_threads=True,
        ):
            values = batch.column(0).to_pylist()
            texts = [x for x in values if isinstance(x, str) and x.strip()]
            if not texts:
                continue

            for start in range(0, len(texts), tokenizer_batch_size):
                yield texts[start : start + tokenizer_batch_size]


class PackedShardWriter:
    def __init__(
        self,
        *,
        out_dir: Path,
        split: str,
        seq_len: int,
        dtype: np.dtype,
        eos_token_id: int,
        shard_size_mb: int,
        source_file_count: int,
    ) -> None:
        self.out_dir = out_dir
        self.split = split
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.eos_token_id = eos_token_id
        self.source_file_count = source_file_count

        if self.dtype == np.uint16:
            self.typecode = "H"
        elif self.dtype == np.uint32:
            self.typecode = "I"
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        bytes_per_token = self.dtype.itemsize
        approx_tokens = (shard_size_mb * 1024 * 1024) // bytes_per_token
        # 强制对齐到完整 block
        self.shard_size_tokens = max(self.seq_len, (approx_tokens // self.seq_len) * self.seq_len)
        if self.shard_size_tokens == 0:
            self.shard_size_tokens = self.seq_len

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._buffer = array(self.typecode)
        self._shard_idx = 0

        self.total_written_tokens = 0
        self.total_written_blocks = 0
        self.total_shards = 0

    def add_block(self, block: list[int]) -> None:
        if len(block) != self.seq_len:
            raise ValueError(f"Block length must equal seq_len={self.seq_len}, got {len(block)}")

        if len(self._buffer) > 0 and len(self._buffer) + len(block) > self.shard_size_tokens:
            self.flush()

        self._buffer.extend(block)
        self.total_written_tokens += len(block)
        self.total_written_blocks += 1

        if len(self._buffer) >= self.shard_size_tokens:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        arr = np.asarray(self._buffer, dtype=self.dtype)
        num_tokens = int(arr.size)
        num_blocks = num_tokens // self.seq_len

        bin_path = self.out_dir / f"{self.split}_{self._shard_idx:05d}.bin"
        json_path = self.out_dir / f"{self.split}_{self._shard_idx:05d}.json"

        arr.tofile(bin_path)

        meta = {
            "split": self.split,
            "dtype": self.dtype.name,
            "seq_len": self.seq_len,
            "num_tokens": num_tokens,
            "num_blocks": num_blocks,
            "eos_token_id": self.eos_token_id,
            "source_file_count": self.source_file_count,
            "bin_file": bin_path.name,
        }
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self._buffer = array(self.typecode)
        self._shard_idx += 1
        self.total_shards += 1

    def close(self) -> None:
        self.flush()


def process_split(
    *,
    split: str,
    files: Sequence[Path],
    tokenizer: PreTrainedTokenizerFast,
    output_root: Path,
    seq_len: int,
    text_column: str,
    parquet_batch_size: int,
    tokenizer_batch_size: int,
    shard_size_mb: int,
    compact_threshold: int,
) -> dict:
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None, cannot append eos automatically")

    store_dtype = choose_storage_dtype(len(tokenizer))
    split_out_dir = output_root / split

    writer = PackedShardWriter(
        out_dir=split_out_dir,
        split=split,
        seq_len=seq_len,
        dtype=store_dtype,
        eos_token_id=int(tokenizer.eos_token_id),
        shard_size_mb=shard_size_mb,
        source_file_count=len(files),
    )

    token_buffer: list[int] = []
    buffer_start = 0

    total_texts = 0
    total_raw_tokens = 0  # 包含 eos，未丢尾巴前
    dropped_tail_tokens = 0

    for texts in iter_text_batches(
        files,
        text_column=text_column,
        parquet_batch_size=parquet_batch_size,
        tokenizer_batch_size=tokenizer_batch_size,
    ):
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for token_ids in encoded["input_ids"]:  # type: ignore[index]
            if not token_ids:
                continue

            total_texts += 1
            total_raw_tokens += len(token_ids) + 1

            token_buffer.extend(token_ids)
            token_buffer.append(int(tokenizer.eos_token_id))

            while len(token_buffer) - buffer_start >= seq_len:
                block = token_buffer[buffer_start : buffer_start + seq_len]
                writer.add_block(block)
                buffer_start += seq_len

            if buffer_start >= compact_threshold and buffer_start >= len(token_buffer) // 2:
                token_buffer = token_buffer[buffer_start:]
                buffer_start = 0

    dropped_tail_tokens = len(token_buffer) - buffer_start
    writer.close()

    summary = {
        "split": split,
        "dtype": store_dtype.name,
        "seq_len": seq_len,
        "source_file_count": len(files),
        "total_texts": total_texts,
        "total_raw_tokens": total_raw_tokens,
        "written_tokens": writer.total_written_tokens,
        "written_blocks": writer.total_written_blocks,
        "dropped_tail_tokens": dropped_tail_tokens,
        "num_shards": writer.total_shards,
        "eos_token_id": int(tokenizer.eos_token_id),
    }

    summary_path = split_out_dir / f"{split}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build packed token shards (.bin + .json) from parquet splits.")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer path or HF tokenizer directory")
    parser.add_argument("--split-dir", type=str, required=True, help="Directory containing train.txt and eval.txt")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--seq-len", type=int, required=True, help="Fixed packed block length")
    parser.add_argument("--text-column", type=str, default="text", help="Parquet text column name")
    parser.add_argument("--parquet-batch-size", type=int, default=2048, help="Rows per parquet read batch")
    parser.add_argument("--tokenizer-batch-size", type=int, default=1024, help="Texts per tokenizer batch")
    parser.add_argument("--shard-size-mb", type=int, default=512, help="Approx shard size in MB")
    parser.add_argument(
        "--compact-threshold",
        type=int,
        default=1 << 20,
        help="Compact token buffer when consumed prefix exceeds this threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, eval_files = load_split_files(split_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    train_summary = process_split(
        split="train",
        files=train_files,
        tokenizer=tokenizer,
        output_root=output_dir,
        seq_len=args.seq_len,
        text_column=args.text_column,
        parquet_batch_size=args.parquet_batch_size,
        tokenizer_batch_size=args.tokenizer_batch_size,
        shard_size_mb=args.shard_size_mb,
        compact_threshold=max(args.compact_threshold, 8 * args.seq_len),
    )

    eval_summary = process_split(
        split="eval",
        files=eval_files,
        tokenizer=tokenizer,
        output_root=output_dir,
        seq_len=args.seq_len,
        text_column=args.text_column,
        parquet_batch_size=args.parquet_batch_size,
        tokenizer_batch_size=args.tokenizer_batch_size,
        shard_size_mb=args.shard_size_mb,
        compact_threshold=max(args.compact_threshold, 8 * args.seq_len),
    )

    global_summary = {
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": len(tokenizer),
        "storage_dtype": choose_storage_dtype(len(tokenizer)).name,
        "seq_len": args.seq_len,
        "train": train_summary,
        "eval": eval_summary,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(global_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
