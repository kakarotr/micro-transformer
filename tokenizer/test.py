from transformers import AutoTokenizer, PreTrainedTokenizerFast

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("tokenizer/common")

for item in [
    "今天的天气非常晴朗，我和朋友们打算去公园散步，顺便在附近的餐厅吃午饭。",
    "人工智能是计算机科学的一个重要分支，近年来在自然语言处理领域取得了显著的进展。",
    "地球是太阳系八大行星之一，距离太阳约一点五亿公里。",
    "天正十年，织田信长在京都本能寺遭到家臣明智光秀的背叛，最终被迫自尽，史称本能寺之变。",
    "丰臣秀吉原名木下藤吉郎，出身贫寒，却凭借出色的才干统一了日本，并在后来就任关白一职。",
    "关原合战是战国时代末期发生的一场决定性战役，德川家康率领的东军最终击败了石田三成带领的西军，奠定了江户幕府的基础。",
    "信长将兵出桶狭间，乘雨骤击，大破今川义元，威震天下。",
]:
    encode_input = tokenizer(item)
    print([tokenizer.decode([token_id]) for token_id in encode_input["input_ids"]])  # type: ignore
