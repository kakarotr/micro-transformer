import jieba
from transformers import AutoTokenizer, PreTrainedTokenizerFast

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("weight")
# 后世学者对明智光秀发送本能寺之变的原因一直众说纷纭
# 庆长八年(1603)，德川家康在江户建立幕府，史称江户幕府。
# 武田信玄在三方原之战中击败了德川家康

# encode_input = tokenizer("")
# for token_id in encode_input["input_ids"]:  # type: ignore
#     readable_chunk = tokenizer.decode([token_id])
#     print(f"ID: {token_id}  ->  {readable_chunk}")

for item in [
    "第一次国府台之战",
    "第一次国府台合战",
    "桶狭间之战",
    "桶狭间合战",
    "姊川之战",
    "姊川合战",
    "长筱之战",
    "长筱合战",
    "关原之战",
    "关原合战",
    "桶狭间",
    "姊川",
    "长筱",
    "关原",
]:
    print(f"{item} - {tokenizer.encode(item)}")
