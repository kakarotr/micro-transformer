import random
from pathlib import Path
from typing import Literal

import jieba
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


def get_training_corpus(batch_size: int = 1000):
    # 处理核心知识, 每个文件加20次等于向上采样20次
    knowledge_dir = Path("data/knowledge")
    files = [x for x in knowledge_dir.rglob("*.md") if x.is_file()]
    knowledge_texts = []
    for file in files:
        with open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                content = line.strip()
                if content:
                    for _ in range(20):
                        knowledge_texts.append(content)

    # 处理通用语料
    MAX_GENERAL_BYTES = 5 * 1024 * 1024 * 1024
    ACCEPTANCE_RATE = 0.07
    current_general_bytes = 0

    dataset = load_dataset("parquet", data_files="data/common/4_5/*.parquet", split="train", streaming=True)
    common_texts = []
    for row in dataset:
        if random.random() > ACCEPTANCE_RATE:
            continue
        text: str = row.get("text", "").strip()  # type: ignore

        if not text:
            continue

        text_bytes = len(text.encode("utf-8"))
        current_general_bytes += text_bytes
        common_texts.append(text)

        if current_general_bytes > MAX_GENERAL_BYTES:
            print("\n[成功]通用数据采样完成，停止读取。")
            break

    texts = [*knowledge_texts, *common_texts]
    random.shuffle(texts)
    print("\n数据打乱完毕，准备开始训练")

    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
    print("\n数据提供完毕")


special_tokens_dict = {
    "pad": "<|endoftext|>",
    "eos": "<|im_end|>",
    "system": "<|im_system|>",
    "user": "<|im_user|>",
    "assistant": "<|im_assistant|>",
}


def train(domain: Literal["common", "knowledge"] = "knowledge"):
    jieba.initialize()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=10000 if domain == "knowledge" else 26000,
        special_tokens=list(special_tokens_dict.values()),
        min_frequency=5,
        show_progress=True,
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
    fast_tokenizer._tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore

    output_dir = Path("tokenizer/knowledge" if domain == "knowledge" else "tokenizer")
    if not output_dir.exists():
        output_dir.mkdir()
    fast_tokenizer.save_pretrained(output_dir)


train(domain="knowledge")
