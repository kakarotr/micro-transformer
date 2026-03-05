import concurrent
import concurrent.futures
import random
from pathlib import Path

import tokenizers
from datasets import load_dataset
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import AddedToken, PreTrainedTokenizerFast

from tokenizer.jieba_tokenizer import get_jieba_pre_tokenizer

MAGIC_SEP = chr(31)

common_pre_tokenizer = None


def init_worker():
    global common_pre_tokenizer
    common_pre_tokenizer, _ = get_jieba_pre_tokenizer()


def worker_process_common_texts(texts_batch):
    global common_pre_tokenizer
    processed = []
    for text in texts_batch:
        processed.append(MAGIC_SEP.join(common_pre_tokenizer.lcut(text)))  # type: ignore
    return processed


def get_training_corpus(batch_size: int = 5000):
    batch = []
    _, knowledge_pre_tokenizer = get_jieba_pre_tokenizer()

    # 处理核心知识, 每个文件加20次等于向上采样20次
    print("开始注入战国语料 (20倍上采样)...")
    knowledge_dir = Path("data/knowledge")
    files = [x for x in knowledge_dir.rglob("*.md") if x.is_file()]
    for _ in range(20):
        for file in files:
            with open(file, mode="r", encoding="utf-8") as f:
                content = f.read()
                if not content:
                    continue

                paragraphs = [p + "\n\n" for p in content.split("\n\n") if p.strip()]
                for p in paragraphs:
                    processed_text = MAGIC_SEP.join(knowledge_pre_tokenizer.lcut(p))
                    batch.append(processed_text)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
    if batch:
        yield batch
        batch = []

    # 处理通用语料
    print("\n开始处理通用语料")
    MAX_GENERAL_BYTES = 5 * 1024 * 1024 * 1024
    ACCEPTANCE_RATE = 0.07
    current_general_bytes = 0

    dataset = load_dataset("parquet", data_files="data/common/4_5/*.parquet", split="train", streaming=True)
    texts_buffer = []
    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=150, initializer=init_worker) as executor:
        for row in dataset:
            if current_general_bytes > MAX_GENERAL_BYTES:
                print("\n[成功] 通用数据采样完成，停止读取。")
                break

            if random.random() > ACCEPTANCE_RATE:
                continue

            text: str = row.get("text", "")  # type: ignore
            if not text:
                continue

            text_bytes = len(text.encode("utf-8"))
            current_general_bytes += text_bytes
            texts_buffer.append(text)

            print(
                f"\r硬盘读取与任务分发进度: {(current_general_bytes / MAX_GENERAL_BYTES) * 100:.2f}%",
                end="",
                flush=True,
            )

            if len(texts_buffer) == batch_size:
                futures.append(executor.submit(worker_process_common_texts, texts_buffer))
                texts_buffer = []

        if texts_buffer:
            futures.append(executor.submit(worker_process_common_texts, texts_buffer))

        for f in concurrent.futures.as_completed(futures):
            yield f.result()

    print("\n所有语料提供完毕")


special_tokens_dict = {
    "pad": "<|endoftext|>",
    "eos": "<|im_end|>",
    "system": "<|im_system|>",
    "user": "<|im_user|>",
    "assistant": "<|im_assistant|>",
}


def train():
    vocab_size = 32768
    add_tokens = ["\n\n", "  ", "   ", "    "]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(MAGIC_SEP), behavior="removed"),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(add_tokens),
        special_tokens=list(special_tokens_dict.values()),
        min_frequency=5,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.add_special_tokens([tokenizers.AddedToken(t, special=True) for t in special_tokens_dict.values()])

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=AddedToken(special_tokens_dict["pad"], lstrip=False, rstrip=False, normalized=False, special=True),
        eos_token=AddedToken(special_tokens_dict["eos"], lstrip=False, rstrip=False, normalized=False, special=True),
        bos_token=None,
        additional_special_tokens=[
            AddedToken(special_tokens_dict["user"], lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken(special_tokens_dict["assistant"], lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken(special_tokens_dict["system"], lstrip=False, rstrip=False, normalized=False, special=True),
        ],
        clean_up_tokenization_spaces=False,
        model_max_length=4096,
    )
    fast_tokenizer.add_bos_token = False
    fast_tokenizer.add_eos_token = False
    fast_tokenizer.add_tokens(add_tokens)
    fast_tokenizer._tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore

    current_size = len(fast_tokenizer)
    if current_size < vocab_size:
        pad_count = vocab_size - current_size
        dummy_tokens = [f"<|dummy_{i}|>" for i in range(pad_count)]
        fast_tokenizer.add_tokens(dummy_tokens)

    output_dir = Path("weight")
    if not output_dir.exists():
        output_dir.mkdir()
    fast_tokenizer.save_pretrained(output_dir)
