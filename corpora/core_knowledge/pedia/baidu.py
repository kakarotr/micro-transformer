import random
import re
import time

import bs4
import pandas as pd
from bs4 import BeautifulSoup
from DrissionPage import ChromiumOptions, ChromiumPage

from corpora.core_knowledge.wiki.entities import (
    BlockType,
    SectionBlock,
    WikiPage,
    WikiSection,
)

ignore_sections = [
    "人际关系",
    "后世纪念",
    "艺术形象",
    "影视形象",
    "相关作品",
    "亲族家室",
    "艺术形象",
    "人物争议",
    "主要作品",
    "历史评价",
    "人物履历",
    "人物家族",
    "亲属成员",
    "墓址寺院",
    "影视形象",
]


def find_title(tag: bs4.Tag):
    """
    查询当前元素的标题

    以当前元素为锚点, 向上查询有指定class的元素
    """
    for prev_sibling in tag.find_previous_siblings():
        if prev_sibling.attrs and prev_sibling.attrs.get("data-tag") == "header":
            return prev_sibling.attrs.get("data-name") or ""
    return ""


def find_title_level(tag: bs4.Tag):
    for prev_sibling in tag.find_previous_siblings():
        if prev_sibling.attrs and prev_sibling.attrs.get("data-tag") == "header":
            return int(prev_sibling.attrs.get("data-level") or 0)  # type: ignore
    return 0


def add_block(
    doc: bs4.Tag,
    page: WikiPage,
    current_title: str,
    block_type: BlockType,
    content: str | list[str],
    list_title: str | None = None,
):
    """
    添加Block
    """
    if current_title == find_title(tag=doc):
        # 当前添加的Block归属的条目已经添加过
        # 判断当前Block和最后一个Block的类型是否一致
        # 一致则在上一个Block的内容后面最后进行追加当前Block的内容
        last_section = page.sections[-1]
        if last_section.blocks and last_section.blocks[-1].type == block_type:
            block_content: str | list[str] = last_section.blocks[-1].content
            if block_content:
                if isinstance(content, str):
                    if block_type == "text":
                        last_section.blocks[-1].content = f"{block_content}\n\n{content}"
                    else:
                        last_section.blocks.append(
                            SectionBlock(type=block_type, content=content, list_title=list_title)
                        )

                else:
                    last_section.blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))
            else:
                last_section.blocks[-1] = SectionBlock(type=block_type, content=content, list_title=list_title)
        else:
            last_section.blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))
    else:
        page.sections[-1].blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))


def parse_baidu(page_title: str, content: str):
    doc = BeautifulSoup(content, "html.parser")
    filter_tag(doc=doc)
    filter_baidu_tag(doc=doc)
    with open(f"preview/{page_title}.html", mode="w", encoding="utf-8") as f:
        f.write(doc.prettify())
    pedia_page = WikiPage(title=page_title, category_name="", lang="zh", sections=[])

    # 摘要
    summary_tag = doc.find("div", class_="J-summary")
    if summary_tag:
        pedia_page.sections.append(WikiSection(title="summary", level=2, blocks=[]))
        for segment in summary_tag.find_all("div"):
            content = segment.get_text(strip=True)
            content = re.sub(r"\s+", "", content)
            pedia_page.sections[-1].blocks.append(SectionBlock(type="text", content=content, lang="zh"))

    current_ignore_level = None
    current_level = 0
    current_title = ""

    main_doc = doc.find("div", class_="J-lemma-content")
    if main_doc:
        for paragraph in main_doc.find_all(recursive=False):
            tag = paragraph.attrs.get("data-tag")
            if tag == "header":
                level = int(paragraph.attrs.get("data-level")) + 1  # type: ignore
                title: str = paragraph.attrs.get("data-name")  # type: ignore

                if current_ignore_level is not None:
                    if level > current_ignore_level:
                        continue
                    else:
                        current_ignore_level = None

                current_level = level
                current_title = title

                if title in ignore_sections:
                    current_ignore_level = level
                    continue
                pedia_page.sections.append(WikiSection(title=title, level=level, blocks=[]))
            elif paragraph.name == "ul" and len(paragraph.find_all("li", recursive=False)) == 1:
                text = paragraph.get_text(strip=True)
                level = find_title_level(tag=paragraph) + 2

                current_title = text
                current_level = level
                pedia_page.sections.append(WikiSection(title=current_title, level=current_level, blocks=[]))
            else:
                if current_ignore_level is not None:
                    if current_level >= current_ignore_level:
                        continue
                if paragraph.name == "div":
                    content = paragraph.get_text(strip=True)
                    content = re.sub(r"\s+", "", content)
                    add_block(
                        doc=main_doc,
                        page=pedia_page,
                        current_title=current_title,
                        block_type="text",
                        content=content,
                    )
                elif paragraph.name == "ul":
                    texts = [li.get_text(strip=True) for li in paragraph.find_all("li", recursive=False)]
                    add_block(
                        doc=paragraph, page=pedia_page, current_title=current_title, block_type="ulist", content=texts
                    )
                elif paragraph.name == "ol":
                    texts = [li.get_text(strip=True) for li in paragraph.find_all("li", recursive=False)]
                    add_block(
                        doc=paragraph, page=pedia_page, current_title=current_title, block_type="olist", content=texts
                    )
                elif paragraph.attrs and paragraph.attrs.get("data-module-type") == "table":
                    table = paragraph.find("table")
                    if table and table.find("tbody"):
                        add_block(
                            doc=paragraph,
                            page=pedia_page,
                            current_title=current_title,
                            block_type="table",
                            content=re.sub(r">\s+<", "><", str(table.find("tbody"))).replace("\n", ""),
                        )
        with open(f"preview/{page_title}.md", mode="w", encoding="utf-8") as f:
            f.write(pedia_page.merge_sections())


def filter_tag(doc: bs4.BeautifulSoup):
    for a in doc.find_all("a"):
        a.unwrap()


def is_title(doc):
    if doc and doc.attrs:
        return doc.attrs.get("data-tag") == "header"
    return False


def filter_baidu_tag(doc: bs4.BeautifulSoup):
    for span in doc.find_all("span"):
        need_remove = False
        if len(span.find_all("sup", attrs={"data-tag": "ref"})):
            need_remove = True
        if need_remove:
            span.decompose()
    main_doc = doc.find(class_="J-lemma-content")

    if main_doc:
        for elem in main_doc.find_all(recursive=False):
            if elem.get("class") and "J-pgc-content" in elem.get("class"):  # type: ignore
                elem.decompose()
            elif "主词条" in elem.get_text(strip=True):
                elem.decompose()
            elif elem.attrs and elem.attrs.get("data-module-type") == "video":
                if is_title(elem.find_previous_sibling()) and is_title(elem.find_next_sibling()):
                    elem.find_previous_sibling().decompose()  # type: ignore
                elem.decompose()
        for elem in main_doc.find_all(class_="J-lemma-content-single-image"):
            elem.decompose()

        for span in main_doc.find_all("span"):
            if span.get("class") and any(["bold" in item for item in span.get("class")]):  # type: ignore
                if span.parent and span.parent.parent and span.parent.parent.name != "li":
                    span.replace_with(f"**{span.get_text(strip=True)}**")

        # for ul in main_doc.find_all("ul", recursive=False):
        #     if len(ul.find_all(recursive=False)) > 1:
        #         if is_title(ul.find_previous_sibling()) and is_title(ul.find_next_sibling()):
        #             ul.find_previous_sibling().decompose()  # type: ignore
        #         ul.decompose()

        # for ul in main_doc.find_all("ol", recursive=False):
        #     if len(ul.find_all(recursive=False)) > 1:
        #         if is_title(ul.find_previous_sibling()) and is_title(ul.find_next_sibling()):
        #             ul.find_previous_sibling().decompose()  # type: ignore
        #         ul.decompose()


def test():
    co = ChromiumOptions()
    co.set_user_agent(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    )

    title = "织田信长"
    page = ChromiumPage(addr_or_opts=co)
    page.get(f"https://baike.baidu.com/item/{title}")
    page.wait.ele_displayed(".lemma-summary", timeout=5)
    parse_baidu(page_title=title, content=page.html)

    page.quit()


def get_url():
    co = ChromiumOptions()
    co.set_argument("--disable-blink-features=AutomationControlled")
    co.set_user_agent(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    )
    page = ChromiumPage(addr_or_opts=co)

    df = pd.read_csv("pedia.csv")
    df["baidu_url"] = df["baidu_url"].astype(object)
    try:
        for index, row in df.iterrows():
            title = row["title"]
            idx = row["index"]
            if idx > 107:
                page.get(f"https://baike.baidu.com/item/{title}")
                if page.wait.ele_displayed(".J-lemma-content"):
                    df.at[index, "baidu_url"] = page.url  # type: ignore
                else:
                    print(f"{title}没有相关条目")
                time.sleep(random.uniform(2, 5))
    finally:
        df.to_csv("a.csv", index=False, encoding="utf-8")
        page.quit()


get_url()
