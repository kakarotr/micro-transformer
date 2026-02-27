import random
from pathlib import Path

import tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import AddedToken, PreTrainedTokenizerFast

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens_dict = {
    "pad": "<|endoftext|>",
    "eos": "<|im_end|>",
    "system": "<|im_system|>",
    "user": "<|im_user|>",
    "assistant": "<|im_assistant|>",
}

trainer = trainers.BpeTrainer(
    vocab_size=10000,
    special_tokens=list(special_tokens_dict.values()),
    min_frequency=5,
    show_progress=True,
)


def get_training_corpus(batch_size: int = 1000):
    batch = []

    knowledge_dir = Path("data/knowledge")
    files = [x for x in knowledge_dir.rglob("*.md") if x.is_file()]

    knowledge_texts = []

    for file in files:
        with open(file, mode="r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                knowledge_texts.append(content)

    for _ in range(1):
        for text in knowledge_texts:
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []

    # MAX_GENERAL_BYTES = 3 * 1024 * 1024 * 1024
    # current_general_bytes = 0
    # ACCEPTANCE_RATE = 0.04

    # general_dataset = load_dataset("parquet", data_dir="data/common", split="train", streaming=True)

    # for row in general_dataset:
    #     if random.random() > ACCEPTANCE_RATE:
    #         continue
    #     text: str = row.get("text", "").strip()  # type: ignore
    #     if not text:
    #         continue
    #     text_bytes = len(text.encode("utf-8"))
    #     current_general_bytes += text_bytes

    #     if current_general_bytes > MAX_GENERAL_BYTES:
    #         print("\n[成功]提取3GB，停止读取。")
    #         break

    #     batch.append(text)
    #     if len(batch) == batch_size:
    #         yield batch
    #         batch = []

    if batch:
        yield batch


tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
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

output_dir = Path("tokenizer/knowledge")
fast_tokenizer.save_pretrained(output_dir)
tokenizer.save(str(output_dir / "tokenizer.json"))
tokenizer.model.save(str(output_dir))
