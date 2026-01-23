import re
from dataclasses import dataclass

import pandas as pd
import requests
import wikitextparser as wtp

from wiki.data import ignore_sections, replace_links
from wiki.utils import get_db_conn, to_simplified


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


def template_handler(template: wtp._template.Template):
    template_name = template.name
    if template_name in ["zy", "link-ja", "Quotation"]:
        if len(template.arguments) > 0:
            return wtp.parse(template.arguments[0].value).plain_text()
    elif template_name == "bd":
        if len(template.arguments) >= 4:
            args = template.arguments
            return f"{args[0].value}{args[1].value}-{args[2].value}{args[3].value}"
    elif template_name in ["columns-list", "col-begin", "col-end", "div col"]:
        if len(template.arguments) > 0:
            return wtp.parse(template.arguments[-1].value).plain_text()
    elif template_name == "tsl":
        if len(template.arguments) > 1:
            return template.arguments[1].value


def clean_cell_text(cell_content):
    """
    清洗单元格文本：去除 Wiki 属性、解析链接、去除换行和特殊符号
    """
    if not cell_content:
        return ""
    content_str = str(cell_content)
    content_str = re.sub(r"(?i)^[a-z\s]+=[^|]+\|\s*", "", content_str)
    text = wtp.parse(content_str).plain_text()
    text = text.replace("\n", "<br>").replace("\r", "")
    text = re.sub(r"\s*[▲△○●◎]\s*", "", text)
    text = re.sub(r"(?i)^[a-z\s]+=[^|]+\|\s*", "", text)
    return text.strip()


def table_handler(table: wtp._table.Table):
    """
    处理 Wiki 表格：补全缺失列、清洗数据、自动移除空列（如被清洗掉的图片列）
    """
    raw_data = table.data(span=True)
    if not raw_data or len(raw_data) == 0:
        return ""
    has_header_symbol = "!" in table.string[:100]
    first_cell_content = str(raw_data[0][0]) if raw_data[0] else ""
    is_list_inside = first_cell_content.strip().startswith("*")
    if not has_header_symbol or is_list_inside:
        merged_text = []
        for row in raw_data:
            for cell in row:
                if cell and str(cell).strip():
                    cell_parsed = wtp.parse(str(cell)).plain_text()
                    merged_text.append(cell_parsed.strip())
        return "\n".join(merged_text)
    else:
        cleaned_data = [[clean_cell_text(cell) for cell in row] for row in raw_data]
        if len(cleaned_data) < 2:
            return ""
        max_cols = max(len(row) for row in cleaned_data)
        normalized_data = [row + [""] * (max_cols - len(row)) for row in cleaned_data]
        df = pd.DataFrame(normalized_data)
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        df = df.replace(r"^\s*$", float("nan"), regex=True)
        df = df.dropna(axis=1, how="all")
        df = df.fillna("")
        return df.to_markdown(index=False)


def replace_tag(text: wtp.WikiText):
    for tag in text.get_tags()[::-1]:
        if tag.name == "ref":
            tag.string = ""


def replace_link(text: wtp.WikiText):
    for link in text.wikilinks[::-1]:
        target = link.target.strip().lower()
        if target.startswith(replace_links):
            link.string = ""


def convert_definition_term(text: str):
    """
    处理维基百科的定义列表语法 (;术语)
    转换为 Markdown 的加粗文本 (**术语**)
    """
    return re.sub(r"(?m)^;\s*(.*)$", r"**\1**", text)


def convert_list(text: str):
    counters = {}

    def replace_line(match):
        # group(1) 是行首的空白（如果有）
        # group(2) 是标记符 (如 *, #, ::, *:)
        # group(3) 是正文内容
        markers = match.group(2)
        content = match.group(3)
        level = len(markers)

        keys_to_del = [k for k in counters if k > level]
        for k in keys_to_del:
            del counters[k]
        last_marker = markers[-1]
        if last_marker == ":":
            symbol = "> " * level
            indent = ""
        elif last_marker == "#":
            current_count = counters.get(level, 0) + 1
            counters[level] = current_count
            symbol = f"{current_count}. "
            indent = "  " * (level - 1)
        else:
            counters[level] = 0
            symbol = "* "
            indent = "  " * (level - 1)

        return f"{indent}{symbol}{content}"

    # (?m)^       : 多行模式行首
    # (\s*)       : Group 1 - 允许行首有空格 (容错)
    # ([\*\#\:]+) : Group 2 - 捕获标记符
    # \s* : 忽略标记符后的空格
    # (.*)$       : Group 3 - 捕获正文
    pattern = r"(?m)^(\s*)([\*\#\:]+)\s*(.*)$"

    return re.sub(pattern, replace_line, text)


def get_pure_own_content(section: wtp._section.Section):
    content = section.string
    # 查询所有标题的位置
    # matches[0]: 当前章节的标题
    # matches[1]: 第一个子章节的标题
    matches = list(re.finditer(r"(?m)^=+\s*.*?\s*=+[\r\n]*", content))

    if len(matches) > 1:
        # 有子章节, 截取当前标题到第一个子标题的内容
        sub_section_start = matches[1].start()
        own_content_with_header = content[:sub_section_start]
    else:
        # 没有子章节
        own_content_with_header = content

    # 去除标题
    return re.sub(r"(?m)^=+\s*.*?\s*=+[\r\n]*", "", own_content_with_header, count=1).strip()


def replace_by_pattern(text: str):
    new_text = re.sub(r"（\s*[，,；;、]\s*", "（", text)
    new_text = re.sub(r"（\s*）|\(\s*\)", "", new_text)
    new_text = re.sub(r"※\s*", "", new_text)
    new_text = re.sub(r"（\s*）|\(\s*\)", "", new_text)
    new_text = re.sub(r"-{\s*(.*?)\s*}-", r"\1", new_text)
    new_text = re.sub(r"^=+\s*.*?\s*=+[\r\n]*", "", new_text)
    return new_text


def extract(title: str):
    contents = []
    wiki_page = WikiPage(name=title, summary="", sections=[], full_content="")

    contents.append(f"{to_simplified(title)}\n\n")

    parsed_content = get_parsed_content(title=title)

    # 提取摘要
    summary = parsed_content.sections[0]
    summary_content = summary.plain_text(replace_templates=template_handler).strip()
    summary_content = replace_by_pattern(summary_content)
    summary_content = to_simplified(summary_content)
    wiki_page.summary = summary_content

    contents.append(f"{summary_content}\n\n")

    # 提取正文内容
    for section in parsed_content.sections[1:]:
        if section.title:
            title = wtp.parse(section.title.strip()).plain_text()
            if title not in ignore_sections:
                raw_pure_text = get_pure_own_content(section=section)
                level = section.level
                if not raw_pure_text:
                    contents.append(f"{to_simplified(title)}\n\n")
                else:
                    pure_text = wtp.parse(raw_pure_text)
                    replace_tag(text=pure_text)
                    replace_link(text=pure_text)
                    content = pure_text.plain_text(
                        replace_templates=template_handler, replace_tables=table_handler, replace_tags=False
                    )
                    content = replace_by_pattern(content)
                    content = convert_list(content)
                    content = convert_definition_term(content)
                    contents.append(f"{to_simplified(title)}\n")
                    contents.append(f"{to_simplified(content.strip())}\n\n")
    wiki_page.full_content = "".join(contents)
    return wiki_page


def fetch_page_content():
    conn = get_db_conn()
    cursor = conn.execute("select name_tc from wiki_page where status = 0")
    rows = cursor.fetchall()
    for row in rows:
        title = row[0]
        extract(title=title)
