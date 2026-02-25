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

print(tokenizer.encode("关原"))
