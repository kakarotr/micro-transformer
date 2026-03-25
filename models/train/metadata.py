from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def count_text(fp: str) -> int:
    try:
        pf = pq.ParquetFile(fp)
        if "text" not in pf.schema_arrow.names:
            return 0
        total = 0
        for batch in pf.iter_batches(columns=["text"], batch_size=65536):
            arr = batch.column(0)
            total += len(arr) - arr.null_count
        return total
    except Exception:
        return 0


def count_text_rows(data_dir: str, workers: int = 8) -> int:
    files_dir = []
    with open(data_dir, mode="r", encoding="utf-8") as f:
        files_dir = [line.strip("\n") for line in f.readlines()]
    files = [Path(file) for file in files_dir]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(count_text, map(str, files)))


tokenizer = None


def init_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weight")


def count_token(fp: str) -> int:
    global tokenizer
    try:
        pf = pq.ParquetFile(fp)
        if "text" not in pf.schema_arrow.names:
            return 0

        total_tokens = 0

        for batch in pf.iter_batches(columns=["text"], batch_size=2048):
            texts = [t for t in batch.column(0).to_pylist() if isinstance(t, str)]
            if not texts:
                continue

            for i in range(0, len(texts), 1024):
                sub_texts = texts[i : i + 1024]
                assert tokenizer is not None
                enc = tokenizer(
                    sub_texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=False,
                )
                total_tokens += sum(len(ids) for ids in enc["input_ids"])

        return total_tokens

    except Exception as e:
        print(f"ERROR {fp}: {e}")
        return 0


def count_pretrain_token(data_dir: str, workers: int = 8) -> int:
    files = [str(p) for p in Path(data_dir).rglob("*.parquet")]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_tokenizer) as ex:
        return sum(ex.map(count_token, files))


if __name__ == "__main__":
    train_total = count_text_rows("models/train/manifest/train.txt")
    eval_total = count_text_rows("models/train/manifest/eval.txt", workers=1)
    print(train_total, eval_total)
