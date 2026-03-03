import sys

import jieba

with open("tokenizer/jieba_words.txt", mode="r", encoding="utf-8") as f:
    words = []
    for line in f.readlines():
        words.append(line.strip())

arg = sys.argv[1]
jieba.initialize()
for word in words:
    jieba.add_word(word=word)
print(jieba.lcut(arg))
