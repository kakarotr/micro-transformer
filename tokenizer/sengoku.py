from pathlib import Path

import sentencepiece as spm


def get_training_corpus():
    knowledge_dir = Path("data/knowledge")
    files = [x for x in knowledge_dir.rglob("*.md") if x.is_file()]

    for file in files:
        with open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text


spm.SentencePieceTrainer.Train(
    sentence_iterator=get_training_corpus(),
    model_prefix="weight/sengoku/sengoku_sp",
    vocab_size=5000,
    character_coverage=0.9995,
    model_type="bpe",
    max_sentence_length=20480,
)
