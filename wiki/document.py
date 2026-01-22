import re
from dataclasses import dataclass

import pandas as pd
import requests
import wikitextparser as wtp

from wiki.data import ignore_sections, replace_links
from wiki.utils import to_simplified


@dataclass
class WikiSection:
    title: str
    content: str


@dataclass
class WikiPage:
    name: str
    summary: str
    sections: list[WikiSection]
    full_content: str


def get_parsed_content(title: str):
    response = requests.get(
        url="https://zh.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "exintro": True,
            "explaintext": True,
        },
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        },
    )

    pages = response.json()["query"]["pages"]
    page_id = next(iter(pages))
    content = pages[page_id]["revisions"][0]["slots"]["main"]["*"]
    return wtp.parse(content)


def clean_cell_text(cell_content):
    if not cell_content:
        return ""

    # 自动去除 [[链接]]、{{模板}} 和 '''加粗'''
    text = wtp.parse(str(cell_content)).plain_text()

    # 处理单元格内的换行符
    text = text.replace("\n", "<br>").replace("\r", "")

    return text.strip()


def template_handler(template: wtp._template.Template):
    if template.name == "zy":
        if len(template.arguments) > 0:
            return template.arguments[0].value
    elif template.name in ["bd"]:
        if len(template.arguments) >= 4:
            args = template.arguments
            return f"{args[0].value}{args[1].value}-{args[2].value}{args[3].value}"


def table_handler(table: wtp._table.Table):
    raw_data = table.data(span=True)
    if raw_data and len(raw_data) > 0:
        cleaned_data = [[clean_cell_text(cell) for cell in row] for row in raw_data]
        if len(cleaned_data) > 1:
            header = cleaned_data[0]
            body = cleaned_data[1:]
            df = pd.DataFrame(body, columns=header)
            return df.to_markdown(index=False)


def replace_tag(section: wtp._section.Section):
    for tag in section.get_tags()[::-1]:
        if tag.name == "ref":
            tag.string = ""
        elif tag.name == "":
            pass


def replace_link(section: wtp._section.Section):
    for link in section.wikilinks[::-1]:
        target = link.target.strip().lower()
        if target.startswith(replace_links):
            link.string = ""


def convert_list(text: str):
    # (?m)^       : 多行模式，匹配行首
    # ([\*\#]+)   : 捕获组1 - 匹配任意数量的 * 或 #
    # \s* : 允许中间有或没有空格（兼容原本就有空格的情况）
    # (.*)$       : 捕获组2 - 匹配这一行剩下的所有内容
    def replace_line(match):
        markers = match.group(1)
        content = match.group(2)
        level = len(markers)
        indent = "  " * (level - 1)

        # 决定使用无序列表 (*) 还是有序列表 (1.)
        if "#" in markers:
            # 只处理一级
            symbol = "1."
        else:
            symbol = "*"

        return f"{indent}{symbol} {content}"

    pattern = r"(?m)^([\*\#]+)\s*(.*)$"
    return re.sub(pattern, replace_line, text)


contents = []


def extract(title: str):
    contents.append(f"# {title}")

    parsed_content = get_parsed_content(title=title)

    # 提取摘要
    summary = parsed_content.sections[0]
    summary_content = summary.plain_text(replace_templates=template_handler).strip()
    summary_content = re.sub(r"（\s*[，,；;、]\s*", "（", summary_content)
    summary_content = to_simplified(summary_content)

    contents.append(f"## 摘要\n{summary_content}")

    # 提取正文内容
    for section in parsed_content.sections[1:]:
        if section.title:
            title = section.title.strip()
            if title not in ignore_sections:
                level = section.level
                raw_content = re.sub(r"^=+\s*.*?\s*=+[\r\n]*", "", section.plain_text())

                if raw_content.startswith("="):
                    # 只有标题没有内容
                    # 页面上表示为标题下面紧接子标题
                    contents.append(f"{'#' * level} {to_simplified(title)}")
                else:
                    replace_tag(section=section)
                    replace_link(section=section)
                    content = wtp.parse(section.string).plain_text(
                        replace_templates=template_handler, replace_tables=table_handler, replace_tags=False
                    )
                    content = convert_list(content)
                    content = re.sub(r"^=+\s*.*?\s*=+[\r\n]*", "", content)
                    contents.append(f"{level * '#'} {to_simplified(title)}\n{to_simplified(content.strip())}")


extract(title="織田信長")

with open("a.md", mode="w", encoding="utf-8") as f:
    f.write("\n\n".join(contents))
