import re
from dataclasses import dataclass

import pandas as pd
import requests
import wikitextparser as wtp

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


def extract(title: str):
    parsed_content = get_parsed_content(title=title)

    # 提取摘要
    summary = parsed_content.sections[0]
    summary_content = summary.plain_text(replace_templates=template_handler).strip()
    summary_content = re.sub(r"（\s*[，,；;、]\s*", "（", summary_content)
    summary_content = to_simplified(summary_content)

    # 提取正文内容
    for section in parsed_content.sections[1:]:
        if section.title:
            title = section.title.strip()
            # level = section.level
            # print(f"{' ' * level}{level}-{title}")
            if title == "年表":
                for ref in section.get_tags("ref"):
                    ref.string = ""
                content = re.sub(
                    r"^=+\s*.*?\s*=+[\r\n]*",
                    "",
                    wtp.parse(section.string).plain_text(
                        replace_templates=template_handler, replace_tables=table_handler, replace_tags=False
                    ),
                )
                print(content)


extract(title="織田信長")
