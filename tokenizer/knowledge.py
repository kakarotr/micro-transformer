from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizerFast


def get_training_corpus(batch_size: int = 1000):
    batch = []
    knowledge_dir = Path("data/knowledge")
    files = [x for x in knowledge_dir.rglob("*.md") if x.is_file()]
    for file in files:
        with open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                content = line.strip()
                if content:
                    batch.append(content)
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


def train():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Punctuation(),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=8000,
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
    fast_tokenizer._tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore

    output_dir = Path("tokenizer/knowledge")
    if not output_dir.exists():
        output_dir.mkdir()
    fast_tokenizer.save_pretrained(output_dir)


def output_keys():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("tokenizer/knowledge")
    with open("tokenizer/keys.txt", mode="w", encoding="utf-8") as f:
        for key in tokenizer.get_vocab().keys():
            char: str = tokenizer.convert_tokens_to_string([key])
            f.write(char)
            f.write("\n")


with open("tokenizer/jieba_words.txt", mode="r", encoding="utf-8") as f:
    words = []
    for line in f.readlines():
        words.append(line.strip())


def print_keys():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("tokenizer/knowledge")
    tokens = [tokenizer.convert_tokens_to_string([key]) for key in tokenizer.get_vocab().keys()]
    suffixs = [
        "的",
        "了",
        "着",
        "过",
        "地",
        "得",
        "于",
        "与",
        "和",
        "中",
        "上",
        "下",
        "内",
        "外",
        "前",
        "后",
        "时",
        "际",
    ]
    action_suffixs = ["是", "有", "说", "击", "杀", "攻", "守", "战", "灭", "败", "退"]
    city_suffixs = ["城", "府", "馆"]
    # "鸟取", 鸟取城
    # prefixs = ["在", "于", "从", "向", "对", "与", "和", "同", "被", "将", "由", "为"]
    for item in city_suffixs:
        print(f"{'-' * 5}{item}{'-' * 5}")
        for key in tokenizer.get_vocab().keys():
            char: str = tokenizer.convert_tokens_to_string([key])
            if char.endswith(item):
                prefix = char[:-1]
                if prefix not in words:
                    print(f'"{prefix}",', char)
        print(f"{'-' * 5}{item}{'-' * 5}")


def distinst_words():
    s = set()
    o = []
    with open("tokenizer/jieba_words.txt", mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            if line not in s:
                o.append(line)
                s.add(line)
    with open("tokenizer/jieba_words2.txt", mode="w+", encoding="utf-8") as f:
        for item in o:
            f.write(item.strip("\n"))
            f.write("\n")


# train()
# output_keys()
# print_keys()
distinst_words()
