import random
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import AddedToken, PreTrainedTokenizerFast

tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens_dict = {
    "pad": AddedToken("<|endoftext|>", lstrip=False, rstrip=False, normalized=False, special=True),
    "user": AddedToken("<|im_user|>", lstrip=False, rstrip=False, normalized=False, special=True),
    "assistant": AddedToken("<|im_assistant|>", lstrip=False, rstrip=False, normalized=False, special=True),
    "eos": AddedToken("<|im_end|>", lstrip=False, rstrip=False, normalized=False, special=True),
}

trainer = trainers.BpeTrainer(
    vocab_size=32768,
    special_tokens=[token.content for token in special_tokens_dict.values()],
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

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token=special_tokens_dict["pad"],
    eos_token=special_tokens_dict["eos"],
    bos_token=None,
    additional_special_tokens=[
        special_tokens_dict["user"],
        special_tokens_dict["assistant"],
    ],
    clean_up_tokenization_spaces=False,
    model_max_length=4096,
)
fast_tokenizer.add_bos_token = False
fast_tokenizer.add_eos_token = False
fast_tokenizer.save_pretrained("weight/b2")
