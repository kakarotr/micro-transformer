import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# --- 1. 初始化设置 ---
path = Path("data/knowledge")
output_file = Path("grouped_output.txt")
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("tokenizer/knowledge")
city_suffixs = ["城", "府", "馆"]

# --- 2. 获取并处理 Token ---
city_tokens_set = set()
for key in tokenizer.get_vocab().keys():
    char: str = tokenizer.convert_tokens_to_string([key])
    if char:
        for item in city_suffixs:
            if char.endswith(item):
                city_tokens_set.add(char)
                break

sorted_city_tokens = sorted(list(city_tokens_set), key=len, reverse=True)
escaped_tokens = [re.escape(token) for token in sorted_city_tokens]
pattern = re.compile(rf"{'|'.join(escaped_tokens)}")

# --- 3. 匹配并分离“展示文本”与“分析文本” ---
grouped_results = defaultdict(dict)

for file in path.rglob("*.md"):
    with open(file, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            for match in pattern.finditer(line):
                found_char = match.group()
                char_index = match.start()
                end_index = match.end()

                # 【展示文本】：按你原本的要求，截取5个字符，用于最终输出查看上下文
                display_text = line[max(0, char_index - 5) : end_index].strip()

                # 【分析文本】：往回看15个字符，专门用于提取纯净的核心实体名
                analysis_window = line[max(0, char_index - 15) : char_index]

                # 核心魔法：使用 \W+ 以任何非文字字符（如标点、括号、空格、#等）为界进行切分
                # 取切分后的最后一部分，这就是紧贴着关键词的“最纯净的前缀文字”
                parts = re.split(r"\W+", analysis_window)
                contiguous_word = parts[-1] if parts else ""

                # 拼接成最终用于去重判定的干净文本（例如提取出纯净的 "大光寺城" 或 "臼井城"）
                clean_text = contiguous_word + found_char

                if clean_text:
                    if clean_text not in grouped_results[found_char]:
                        grouped_results[found_char][clean_text] = display_text
                    else:
                        # 如果同一个核心词出现多次，保留展示文本最短/最干净的那一条
                        if len(display_text) < len(grouped_results[found_char][clean_text]):
                            grouped_results[found_char][clean_text] = display_text

# --- 4. 后缀包含去重并输出 ---
with open(output_file, mode="w", encoding="utf-8") as out_f:
    for keyword in sorted(grouped_results.keys()):
        # 按照“纯净核心词”的长度从小到大排序
        clean_candidates = sorted(grouped_results[keyword].keys(), key=len)

        final_keys = []
        for cand in clean_candidates:
            # 判断：如果当前的词，是以任何已保存的更短的词结尾的，就抛弃！
            # 举例：因为有干净的 "大光寺城" 打底，"攻克大光寺城" 就会被过滤掉
            if not any(cand.endswith(existing) for existing in final_keys):
                final_keys.append(cand)

        # 将保留下来的核心词，还原为对应的5字符展示文本
        contexts = [grouped_results[keyword][k] for k in final_keys]

        out_f.write(f"========== 【{keyword}】 (共 {len(contexts)} 条终极去重记录) ==========\n")

        for context in sorted(contexts):
            out_f.write(f"{context}\n")

        out_f.write("\n")

print(f"处理完成！已彻底隔离语法与标点干扰，结果保存在: {output_file.absolute()}")
