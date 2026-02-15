import re
import time

import pyperclip
from opencc import OpenCC

opencc = OpenCC("tw2sp")


def cn_to_ar(text):
    # 定义中文数字到阿拉伯数字的映射
    cn_map = {
        "〇": "0",
        "零": "0",
        "。": "0",
        "0": "0",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }

    def replace_func(match):
        cn_str = match.group(0)
        return "".join(cn_map[char] for char in cn_str) + "年"

    # 使用正则进行全局替换
    pattern = r"[〇零。0一二三四五六七八九]{4}"
    return re.sub(pattern, replace_func, text)


def process_text(text: str):
    text = text.replace(" ", "").replace("^", "").replace("(", "（").replace(")", "）").strip()
    text = cn_to_ar(text)
    return opencc.convert(text)


def main():
    last_text = pyperclip.paste()
    while True:
        time.sleep(1)
        current_text = pyperclip.paste()
        if current_text != last_text:
            new_text = process_text(current_text)
            if new_text != current_text:
                pyperclip.copy(new_text)
                last_text = new_text
            else:
                last_text = current_text


main()
