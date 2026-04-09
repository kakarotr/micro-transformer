import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def create_pretraining_splits(
    data_dir: str | Path,
    split_dir: str | Path,
    *,
    glob_pattern: str = "*.parquet",
    valid_ratio: float = 0.02,
    test_ratio: float = 0.0,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[list[Path], list[Path], list[Path]]:
    data_dir = Path(data_dir)
    split_dir = Path(split_dir)

    train_path = split_dir / "train.txt"
    valid_path = split_dir / "eval.txt"
    test_path = split_dir / "test.txt"

    if not overwrite and (train_path.exists() or valid_path.exists() or test_path.exists()):
        raise FileExistsError(f"Split files already exist under {split_dir}, set overwrite=True to regenerate")

    all_files = sorted(data_dir.rglob(glob_pattern))
    if not all_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    if not (0.0 <= valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in [0, 1), got {valid_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if valid_ratio + test_ratio >= 1.0:
        raise ValueError("valid_ratio + test_ratio must be < 1.0")

    rng = random.Random(seed)
    files = list(all_files)
    rng.shuffle(files)

    n_total = len(files)
    n_test = int(n_total * test_ratio)
    n_valid = int(n_total * valid_ratio)

    if test_ratio > 0 and n_test == 0:
        n_test = 1
    if valid_ratio > 0 and n_valid == 0:
        n_valid = 1

    test_files = files[:n_test]
    valid_files = files[n_test : n_test + n_valid]
    train_files = files[n_test + n_valid :]

    if not train_files:
        raise ValueError("train split is empty after splitting")

    split_dir.mkdir(parents=True, exist_ok=True)

    train_path.write_text("\n".join(str(p) for p in train_files), encoding="utf-8")
    valid_path.write_text("\n".join(str(p) for p in valid_files), encoding="utf-8")
    test_path.write_text("\n".join(str(p) for p in test_files), encoding="utf-8")

    return train_files, valid_files, test_files


def load_pretraining_splits(
    split_dir: str | Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    split_dir = Path(split_dir)

    def _read_split_file(path: Path) -> list[Path]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [Path(line.strip()) for line in lines if line.strip()]

    train_files = _read_split_file(split_dir / "train.txt")
    valid_files = _read_split_file(split_dir / "eval.txt")
    test_files = _read_split_file(split_dir / "test.txt")

    if not train_files:
        raise ValueError(f"train split is empty or missing under: {split_dir}")

    return train_files, valid_files, test_files


class PackedTokenDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        path: str | Path,
        *,
        dtype: str | None = None,
        seq_len: int | None = None,
    ):
        super().__init__()
        self.data_dir = Path(path)

        meta_files = sorted(self.data_dir.glob("*.json"))
        meta_files = [p for p in meta_files if not p.name.endswith("_summary.json")]

        self.shards: list[dict] = []
        self.cum_blocks: list[int] = []
        total_blocks = 0

        expected_dtype: np.dtype | None = np.dtype(dtype) if dtype is not None else None
        expected_seq_len = seq_len

        for meta_path in meta_files:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            bin_file = meta.get("bin_file")
            bin_path = self.data_dir / bin_file

            shard_dtype = np.dtype(meta["dtype"])
            shard_seq_len = int(meta["seq_len"])
            num_tokens = int(meta["num_tokens"])
            num_blocks = int(meta["num_blocks"])

            if num_tokens != num_blocks * shard_seq_len:
                raise ValueError(
                    f"Invalid shard meta: {meta_path}, "
                    f"num_tokens({num_tokens}) != num_blocks({num_blocks}) * seq_len({shard_seq_len})"
                )

            if expected_dtype is None:
                expected_dtype = shard_dtype
            elif shard_dtype != expected_dtype:
                raise ValueError(f"Inconsistent dtype across shards: got {shard_dtype}, expected {expected_dtype}")

            if expected_seq_len is None:
                expected_seq_len = shard_seq_len
            elif shard_seq_len != expected_seq_len:
                raise ValueError(
                    f"Inconsistent seq_len across shards: got {shard_seq_len}, expected {expected_seq_len}"
                )

            self.shards.append(
                {
                    "meta_path": meta_path,
                    "bin_path": bin_path,
                    "dtype": shard_dtype,
                    "seq_len": shard_seq_len,
                    "num_tokens": num_tokens,
                    "num_blocks": num_blocks,
                }
            )
            total_blocks += num_blocks
            self.cum_blocks.append(total_blocks)

        if expected_dtype is None or expected_seq_len is None:
            raise ValueError("Failed to infer dataset dtype / seq_len")

        self.dtype = expected_dtype
        self.seq_len = expected_seq_len
        self.total_blocks = total_blocks
        self._memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.total_blocks

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total_blocks

        if index < 0 or index >= self.total_blocks:
            raise IndexError(f"index out of range: {index}")

        shard_idx = bisect.bisect_right(self.cum_blocks, index)
        block_start = 0 if shard_idx == 0 else self.cum_blocks[shard_idx - 1]
        local_block_idx = index - block_start

        mm = self._get_memmap(shard_idx)
        token_start = local_block_idx * self.seq_len
        token_end = token_start + self.seq_len

        block = np.asarray(mm[token_start:token_end], dtype=np.int64)
        return torch.from_numpy(block)

    def _get_memmap(self, shard_idx: int):
        mm = self._memmaps.get(shard_idx)
        if mm is None:
            shard = self.shards[shard_idx]
            mm = np.memmap(
                shard["bin_path"],
                mode="r",
                dtype=shard["dtype"],
                shape=(shard["num_tokens"],),
            )
            self._memmaps[shard_idx] = mm
        return mm
