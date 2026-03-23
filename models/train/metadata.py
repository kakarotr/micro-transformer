from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer, PreTrainedTokenizerFast


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
    files = list(Path(data_dir).rglob("*.parquet"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(count_text, map(str, files)))


tokenizer = AutoTokenizer.from_pretrained("weight")


def count_token(fp: str) -> int:
    try:
        pf = pq.ParquetFile(fp)
        if "text" not in pf.schema_arrow.names:
            return 0

        total_tokens = 0

        for batch in pf.iter_batches(columns=["text"], batch_size=2048):
            arr = batch.column(0)

            texts = [x.as_py() for x in arr if x is not None]
            if not texts:
                continue

            for i in range(0, len(texts), 1024):
                sub_texts = texts[i : i + 1024]
                enc = tokenizer(
                    sub_texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=False,
                )
                total_tokens += sum(len(ids) for ids in enc["input_ids"])  # type: ignore

        return total_tokens

    except Exception as e:
        print(f"ERROR {fp}: {e}")
        return 0


def count_pretrain_token(data_dir: str, workers: int = 8):
    from pathlib import Path

    import pyarrow.parquet as pq

    files = list(Path(data_dir).rglob("*.parquet"))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(count_token, map(str, files)))


if __name__ == "__main__":
    total = count_pretrain_token(data_dir="data/common", workers=16)
    print(total)
