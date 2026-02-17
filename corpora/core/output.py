import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import TypeAdapter
from sympy import sec

from corpora.core.wiki.entities import WikiPage, WikiSection
from corpora.utils.db import get_cursor

load_dotenv()


def remove_pinyin_brackets(text):
    # 正则解释：
    # [\[【]  -> 匹配左括号（兼容半角 '[' 和 全角 '【'）
    # .*?     -> 非贪婪匹配中间的任意内容
    # [\]】]  -> 匹配右括号（兼容半角 ']' 和 全角 '】'）
    pattern = r"[\[【].*?[\]】]"

    # 将匹配到的内容替换为空字符串
    text = re.sub(pattern, "", text)
    text = text.replace(" ", "")
    return text


def baidu_clean_japanese_in_parens(text):
    # 1. 查找第一个全角括号及其内部内容
    # pattern解释: 匹配以 '（' 开头，'）' 结尾，中间非贪婪匹配的内容
    match = re.search(r"[（\(](.*?)[）\)]", text)

    if not match:
        return text  # 如果没找到括号，原样返回

    full_part = match.group(0)  # 捕获整个括号部分，例如： （日语：xxx，1549年...）
    inner_content = match.group(1)  # 捕获括号内的内容

    # 2. 数据清洗
    # A. 去除 "日语：" 或 "日语:" 标签 (针对例子1)
    inner_content = re.sub(r"(日语|日文名)[：:]", "", inner_content)

    # B. 去除假名 (平假名 \u3040-\u309F, 片假名 \u30A0-\u30FF, 长音符 \u30fc)
    inner_content = re.sub(r"[\u3040-\u309F\u30A0-\u30FF\u30fc]+", "", inner_content)

    # C. 去除残留的开头标点 (去除假名后，可能会剩下 "，1549..." 这种情况)
    # lstrip 用于去除字符串左侧指定的字符（全角逗号、半角逗号、空格）
    inner_content = inner_content.lstrip("，, 　；;、")

    # 3. 结果重组
    if not inner_content:
        # 如果清洗后里面什么都不剩 (针对例子3)，将原文本中的整个括号部分删除
        return text.replace(full_part, "", 1)
    else:
        # 如果还有内容 (针对例子1和2)，保留括号，填入清洗后的内容
        return text.replace(full_part, f"（{inner_content}）", 1)


def douyin_clean_japanese_in_parens(text):
    # 1. 定位括号内容
    match = re.search(r"[（\(](.*?)[）\)]", text)
    if not match:
        return text

    full_part = match.group(0)  # （假名：...，罗马字：...，1538年...）
    inner_content = match.group(1)

    # 2. 清洗逻辑
    # 步骤 A: 移除带标签的元数据
    # 正则解释：
    # (假名|罗马字|日语) -> 匹配任意一个关键词
    # [：:]             -> 匹配全角或半角冒号
    # .*?               -> 非贪婪匹配后续内容
    # ([，,] | $)       -> 直到遇到逗号或者字符串结束
    inner_content = re.sub(
        r"(日语假名|罗马拼音|平假名|片假名|假名|罗马字|日语|日文|英文|拼音)[：:].*?([，,；]|$)", "", inner_content
    )

    # 步骤 B: 移除残留的纯假名 (为了兼容没有写"假名："标签的情况)
    inner_content = re.sub(r"[\u3040-\u309F\u30A0-\u30FF\u30fc]+", "", inner_content)

    # 步骤 C: 清理最左侧残留的符号 (逗号、空格)
    inner_content = inner_content.lstrip("，, 　；;、")

    # 3. 重组
    if not inner_content:
        return text.replace(full_part, "", 1)
    else:
        return text.replace(full_part, f"（{inner_content}）", 1)


def clear():
    with get_cursor() as cursor:
        cursor.execute(
            "select id, sections, source from pedia_core_corpus where sections is not null and lang ='zh' and source in ('baidu', 'douyin')"
        )
        rows = cursor.fetchall()
        adapter = TypeAdapter(list[WikiSection])
        for id, sections, source in rows:
            sections = adapter.validate_python(sections)
            if len(sections) > 1 and sections[0].title == "summary" and len(sections[0].blocks) > 0:
                summary_text: str = sections[0].blocks[0].content  # type: ignore
                if source == "baidu":
                    summary_text = baidu_clean_japanese_in_parens(summary_text)
                elif source == "douyin":
                    summary_text = douyin_clean_japanese_in_parens(summary_text)
                sections[0].blocks[0].content = summary_text
            if source == "douyin":
                for section in sections:
                    for block in section.blocks:
                        if isinstance(block.content, str):
                            block.content = remove_pinyin_brackets(block.content)
                        elif isinstance(block.content, list):
                            block.content = [remove_pinyin_brackets(item) for item in block.content]
            cursor.execute(
                "update pedia_core_corpus set sections = %s where id = %s", (adapter.dump_json(sections).decode(), id)
            )


def output_article():
    with get_cursor() as cursor:
        cursor.execute("select title, content from article_core_corpus")
        rows = cursor.fetchall()
        for title, content in rows:
            with open(f"/Users/linyongjin/Desktop/output/article/{title}.md", mode="w", encoding="utf-8") as f:
                f.write(content)


def output_peida():
    with get_cursor() as cursor:
        cursor.execute(
            "select title, sections, source from pedia_core_corpus where sections is not null and lang ='zh'"
        )
        rows = cursor.fetchall()
        adapter = TypeAdapter(list[WikiSection])
        for title, sections, source in rows:
            directory = Path(f"/Users/linyongjin/Desktop/output/pedia/{source}")
            if not directory.exists():
                directory.mkdir()
            sections = adapter.validate_python(sections)
            with open(f"/Users/linyongjin/Desktop/output/pedia/{source}/{title}.md", mode="w", encoding="utf-8") as f:
                f.write(
                    WikiPage(
                        title=title, category_name="", lang="zh", sections=adapter.validate_python(sections)
                    ).merge_sections()
                )


# output_peida()
def baidu_test():
    print(baidu_clean_japanese_in_parens("安国寺惠琼（あんこくじえけい，1539年1月3日—1600年11月6日）"))
    print(baidu_clean_japanese_in_parens("安藤守就（あんどうもりなり，1503年－1582年）"))
    print(baidu_clean_japanese_in_parens("柴田胜家（日语：しばたかついえ；1522年－1583年6月14日）"))
    print(baidu_clean_japanese_in_parens("真田信尹（日文名：さなだのぶただ，1547年－1632年6月21日）"))


def douyin_test():
    print(douyin_clean_japanese_in_parens("大友宗麟（日语：おおとも そうりん；1530年1月31日—1587年6月11日）"))
    print(
        douyin_clean_japanese_in_parens(
            "安国寺惠琼（日语假名：あんこくじえけい，罗马拼音：Ankokuji Ekei，1539年1月3日—1600年11月6日）"
        )
    )
    print(douyin_clean_japanese_in_parens("伊达晴宗（だて はるむね；1519年—1578年1月12日）"))
    print(
        douyin_clean_japanese_in_parens(
            "上杉谦信（日文：上杉 謙信，假名：うえすぎ けんしん，生卒：1530年2月18日-1578年4月19日）"
        )
    )
    print(douyin_clean_japanese_in_parens("松田宪秀（生年不详-1590年）"))
    print(
        douyin_clean_japanese_in_parens(
            "大友义统（日语：大友 义统／おおとも よしむね Ōtomo Yoshimune，1558年—1610年9月2日）"
        )
    )
    print(
        douyin_clean_japanese_in_parens("大谷吉继（おおたによしつぐ Otani Yoshitsugu，1559年1月12日—1600年10月21日）")
    )
    print(
        douyin_clean_japanese_in_parens(
            "黑田长政（日文：くろだ ながまさ，英文：Kuroda Nagamasa;；1568年12月21日—1622年8月29日）"
        )
    )
    print(remove_pinyin_brackets("由于反织田势力在近畿[jī]与织田家作战失败后逃"))


# clear()
output_article()
