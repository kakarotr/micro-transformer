import jieba

sengoku_suffixes = ["军", "家", "氏", "势", "党", "众", "城", "国", "守", "殿"]
danger_particles = [
    # 方位
    "中",
    "内",
    "外",
    "上",
    "下",
    "里",
    "前",
    "后",
    # 介/连词
    "在",
    "与",
    "和",
    "同",
    "对",
    "向",
    "由",
    "从",
    # 助词
    "的",
    "了",
    "着",
    "过",
    "地",
    "得",
    # 高危副/动词
    "必",
    "将",
    "多",
    "也",
    "都",
    "却",
    "就",
]


def get_jieba_pre_tokenizer():
    knowledge_pre_tokenizer = jieba.Tokenizer()
    knowledge_pre_tokenizer.initialize()
    with open("tokenizer/jieba/jieba_add_words.txt", mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            knowledge_pre_tokenizer.add_word(line.strip(), freq=20000)
    for suffix in sengoku_suffixes:
        for particle in danger_particles:
            knowledge_pre_tokenizer.suggest_freq((suffix, particle), True)

    common_pre_tokenizer = jieba.Tokenizer()
    common_pre_tokenizer.initialize()

    return common_pre_tokenizer, knowledge_pre_tokenizer


_, tokenizer = get_jieba_pre_tokenizer()
print(tokenizer.lcut("德川家康与武田信玄"))
