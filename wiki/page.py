import json
import re
from dataclasses import asdict, dataclass

import requests
import wikitextparser as wtp
from rich.progress import track

from wiki.data import fuzzy_sections, ignore_sections, replace_links
from wiki.utils import get_db_conn


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
        url="https://ja.wikipedia.org/w/api.php",
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
            return f"{wtp.parse(args[0].value).plain_text()}{wtp.parse(args[1].value).plain_text()}-{wtp.parse(args[2].value).plain_text()}{wtp.parse(args[3].value).plain_text()}"
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
    text = text.replace("\n", "").replace("\r", "").replace("&nbsp;", "").replace("<br>", "")
    text = re.sub(r"\s*[▲△○●◎]\s*", "", text)
    text = re.sub(r"(?i)^[a-z\s]+=[^|]+\|\s*", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def table_handler(table: wtp._table.Table):
    """
    扁平化Wiki表格
    """
    raw_data = table.data(span=True)
    header = [clean_cell_text(item) for item in raw_data[0]]
    rows = raw_data[1:]
    full_content = []
    for row in rows:
        if row:
            contents = []
            for idx, item in enumerate(row):
                text = clean_cell_text(item)
                if text:
                    contents.append(f"{header[idx]}{text}，")
            full_content.append(("".join(contents))[:-1] + "。")
    return "\n".join(full_content)


def replace_tag(text: wtp.WikiText):
    for tag in text.get_tags()[::-1]:
        if tag.name == "ref":
            tag.string = ""
        else:
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
        content = match.group(3).strip()
        level = len(markers)

        # 当层级变浅时，重置更深层级的计数器
        keys_to_del = [k for k in counters if k > level]
        for k in keys_to_del:
            del counters[k]

        last_marker = markers[-1]

        if last_marker == ":":
            symbol = ""
            indent = "  " * level
        elif last_marker == "#":
            current_count = counters.get(level, 0) + 1
            counters[level] = current_count
            symbol = f"{current_count}. "
            indent = "  " * (level - 1)
        else:
            counters[level] = 0
            symbol = "- "
            indent = "  " * (level - 1)

        return f"{indent}{symbol}{content}"

    # Regex 说明:
    # (?m)^      : 多行模式行首
    # (\s*)      : Group 1 - 允许行首有空格 (容错)
    # ([\*\#\:]+): Group 2 - 捕获标记符
    # \s* : 忽略标记符后的空格
    # (.*)$      : Group 3 - 捕获正文
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
    # new_text = re.sub(r"\s+", "", new_text)
    return new_text


def clean_raw_text(raw_text: str):
    text = wtp.parse(raw_text)
    replace_tag(text=text)
    replace_link(text=text)
    content = text.plain_text(
        replace_templates=template_handler,
        replace_tables=table_handler,
        replace_wikilinks=True,
        replace_tags=False,
    )
    content = replace_by_pattern(content)
    content = convert_list(content)
    content = convert_definition_term(content)
    return content.strip()


def extract(title: str):
    contents = []
    wiki_page = WikiPage(name=title, summary="", sections=[], full_content="")

    contents.append(f"# {title}\n\n")

    parsed_content = get_parsed_content(title=title)

    # 提取摘要
    summary = parsed_content.sections[0]
    replace_link(text=summary)
    summary_content = summary.plain_text(replace_templates=template_handler, replace_wikilinks=True).strip()
    summary_content = replace_by_pattern(text=summary_content)
    wiki_page.summary = summary_content

    contents.append(f"{summary_content}{'\n\n' if len(parsed_content.sections) > 1 else ''}")

    current_ignore_level = None
    # 提取正文内容
    for _, section in enumerate(parsed_content.sections[1:], start=2):
        level = section.level

        if current_ignore_level is not None:
            if level > current_ignore_level:
                continue
            else:
                current_ignore_level = None

        if not section.title:
            continue

        title = wtp.parse(section.title.strip()).plain_text()
        if title in ignore_sections:
            current_ignore_level = level
            continue

        raw_pure_text = get_pure_own_content(section=section)
        content = clean_raw_text(raw_text=raw_pure_text)
        if content:
            contents.append(f"{'#' * level} {title}\n")
            contents.append(f"{content} \n\n")
        else:
            contents.append(f"{'#' * level} {title}\n\n")
        section = WikiSection(title=title, content=content)
        wiki_page.sections.append(section)

    wiki_page.full_content = "".join(contents).rstrip("\n")
    return wiki_page


def fetch_page_content():
    conn = get_db_conn()
    cursor = conn.execute("select name_tc from wiki_page where status = 0")
    rows = cursor.fetchall()
    for row in track(rows):
        title = row[0]
        page = extract(title=title)
        conn.execute(
            "insert into wiki_page_content (name, summary, sections, full_content) values (?, ?, ?, ?)",
            (
                page.name,
                page.summary,
                json.dumps([asdict(i) for i in page.sections], ensure_ascii=False),
                page.full_content,
            ),
        )
        conn.execute("update wiki_page set status = 1 where name_tc = ?", (title,))
        conn.commit()

    conn.close()
