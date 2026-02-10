from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=50000,
    special_tokens=["<|endofsentence|>", "<|user|>", "<|assistant|>"],
    min_frequency=2,
    show_progress=True,
)
