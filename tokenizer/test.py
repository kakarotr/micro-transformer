import json

from transformers import AutoTokenizer, PreTrainedTokenizerFast

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("weight")

common_test_texts = [
    "今天的天气非常晴朗，我和朋友们打算去公园散步，顺便在附近的餐厅吃午饭。",
    "人工智能是计算机科学的一个重要分支，近年来在自然语言处理领域取得了显著的进展。",
    "地球是太阳系八大行星之一，距离太阳约一点五亿公里。",
    "今天天气非常不错，适合和朋友们一起去郊区散心。",
    "人工智能技术正在快速改变我们的生活方式和工作环境。",
    "这家餐厅的招牌菜是红烧肉，味道虽然很正宗，但价格稍微有点贵。",
    "在这个快速发展的时代，保持持续学习是提升竞争力的有效途径。",
    "他做事总是脚踏实地，从来不好高骛远，最终取得了令人瞩目的成就。",
    "毫无疑问，这件事情的来龙去脉还需要进一步调查，才能彻底水落石出。",
    "无论遇到什么样的艰难险阻，我们都必须坚持到底，绝不轻言放弃。",
    "根据气象部门的最新预报，明天下午本市大部分地区将迎来强降雨天气。",
    "截至今年年底，该项目的总投资额度已经正式突破了一百五十万元。",
    "这件商品的原价是￥２９９９，现在的促销折扣是七五折，非常划算。",
    "「你真的确定要这么做吗？」他皱着眉头反问道，——这似乎并不是一个完美的解决方案。",
    "会议时间定在二零二六年二月二十八日下午三点半，请大家准时参加。",
    "看看這些繁體字會不會被正確處理：我們一起去圖書館看書。",
    "测试几个生僻词汇：龘、𪚥、䨻、齉。看看是输出未知错误，还是被平滑切分。",
    "啊啊啊啊啊啊啊啊啊啊，连续重复的单字会不会被合并成一个超长的标记？",
    "这里是一段包含\n换行符、\t制表符，以及    多个连续全角和半角空格的文本。",
]

knowledge_test_text = [
    "天正十年，织田信长在京都本能寺遭到家臣明智光秀的背叛，最终被迫自尽，史称本能寺之变。",
    "丰臣秀吉原名木下藤吉郎，出身贫寒，却凭借出色的才干统一了日本，并在后来就任关白一职。",
    "关原合战是战国时代末期发生的一场决定性战役，德川家康率领的东军最终击败了石田三成带领的西军，奠定了江户幕府的基础。",
    "信长将兵出桶狭间，乘雨骤击，大破今川义元，威震天下。",
    "天正十年（1582年）六月二日，明智光秀发动本能寺之变，织田信长被迫自杀，享年四十九岁。",
    "丰臣秀吉在山崎之战中击败了叛将，随后通过贱岳之战和长久手之战巩固了天下霸权。",
    "德川家康，幼名竹千代，最终在关原合战中取得胜利，建立江户幕府，开启了长达两百多年的和平时代。",
    "武田信玄的‘风林火山’旗帜，源自《孙子兵法·军争篇》；而上杉谦信则被誉为‘越后之龙’，两人在川中岛爆发了五次激烈的合战。",
    "战国大名为了争夺上洛的资格，在各个令制国之间展开了错综复杂的联姻与背叛。",
]

for item in knowledge_test_text:
    encode_input = tokenizer(item)
    # print(tokenizer.convert_ids_to_tokens(encode_input["input_ids"]))  # type: ignore
    print([tokenizer.decode([token_id]) for token_id in encode_input["input_ids"]])  # type: ignore

# with open("tokenizer/keys.txt", mode="w", encoding="utf-8") as f:
#     for key in tokenizer.get_vocab().keys():
#         char: str = tokenizer.convert_tokens_to_string([key])
#         f.write(char)
#         # if len(char) > 3 and (char.endswith("的") or char.endswith("于")):
#         #     print(char, tokenizer.encode(char))
#         f.write("\n")
