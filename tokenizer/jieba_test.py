import jieba

for item in jieba.lcut("这里是一段包含\n换行符、\t制表符，以及    多个连续全角和半角空格的文本。"):
    print(item)
