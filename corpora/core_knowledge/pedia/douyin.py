import random
import re
import time

import bs4
import pandas as pd
from bs4 import BeautifulSoup
from DrissionPage import ChromiumOptions, ChromiumPage
from pydantic import TypeAdapter

from corpora.core_knowledge.wiki.entities import (
    BlockType,
    SectionBlock,
    WikiPage,
    WikiSection,
)
from utils.db import get_cursor, get_db_conn

ignore_titles = ["人物关系", "注释", "参考资料", "条目合集", "陵寝墓地", "系谱", "主要作品", "相关作品"]
fuzzy_titles = ["形象", "守护", "国司", "家族", "纪念", "作品", "艺术", "文艺", "游戏"]


def get_url():
    co = ChromiumOptions()
    co.set_user_agent(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    )
    page = ChromiumPage(addr_or_opts=co)

    df = pd.read_csv("pedia_core_corpus.csv")
    df["url"] = df["url"].astype(object)
    try:
        for index, row in df.iterrows():
            title = row["title"]
            idx = row["index"]
            if idx > 1298:
                page.get(f"https://www.baike.com/search?keyword={title}")
                page.wait.ele_displayed("#list")
                soup = BeautifulSoup(page.html, "html.parser")

                em = soup.find("em")
                if em and em.get_text(strip=True) == title:
                    df.at[index, "url"] = em.parent.attrs.get("href")  # type: ignore
                else:
                    print(f"{title}没有相关条目")
    finally:
        df.to_csv("a.csv", index=False)
        page.quit()


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


def find_title(tag: bs4.Tag):
    """
    查询当前元素的标题

    以当前元素为锚点, 向上查询有指定class的元素
    """
    for prev_sibling in tag.find_previous_siblings():
        if prev_sibling.attrs and prev_sibling.attrs.get("data-tag") == "header":
            return prev_sibling.attrs.get("data-name") or ""
    return ""


def is_title(tag: bs4.Tag):
    if tag.attrs and tag.attrs.get("class"):
        return any(["baike-render-wrapper" in class_item for class_item in tag["class"]])
    return False


def parse_douyin(page_title: str, content: str):
    doc = BeautifulSoup(content, "html.parser")
    filter_tag(doc=doc)

    page = WikiPage(
        title=page_title, category_name="", lang="zh", sections=[WikiSection(title="summary", level=2, blocks=[])]
    )

    content_doc = doc.find(class_="BAIKE_RENDER_INSTANCE")
    if content_doc:
        # 摘要
        anchor = content_doc.find(id="INFOBOX_CONTAINER_ID")
        if anchor:
            for next in anchor.find_next_siblings():
                if "baike-render-paragraph" in next["class"]:
                    page.sections[-1].blocks.append(SectionBlock(type="text", content=next.get_text(strip=True)))
                    next.decompose()
                else:
                    break

        current_ignore_level = None
        current_level = 0
        current_title = ""

        for paragraph in content_doc.find_all("div", recursive=False):
            if is_title(tag=paragraph):
                level = int(paragraph.find(recursive=False).name[-1]) + 1  # type: ignore
                title = paragraph.get_text(strip=True)

                if current_ignore_level is not None:
                    if level > current_ignore_level:
                        continue
                    else:
                        current_ignore_level = None

                current_level = level
                current_title = title

                if title in ignore_titles or any([fuzzy in title for fuzzy in fuzzy_titles]):
                    current_ignore_level = level
                    continue
                page.sections.append(WikiSection(title=title, level=level, blocks=[]))
            else:
                if current_ignore_level is not None:
                    if current_level >= current_ignore_level:
                        continue
                if paragraph.attrs and paragraph.attrs.get("class") and "baike-render-paragraph" in paragraph["class"]:
                    content = paragraph.get_text(strip=True)
                    add_block(doc=paragraph, page=page, current_title=current_title, block_type="text", content=content)
                elif paragraph.attrs and paragraph.attrs.get("class") and "bk-table-wrapper-node" in paragraph["class"]:
                    tbody = paragraph.find("tbody")
                    if tbody:
                        add_block(
                            doc=paragraph,
                            page=page,
                            current_title=current_title,
                            block_type="table",
                            content=re.sub(r">\s+<", "><", str(tbody)).replace("\n", ""),
                        )
        return page


def filter_tag(doc: bs4.BeautifulSoup):
    for a in doc.find_all("a"):
        a.unwrap()
    for reference in doc.find_all(class_="baike-render-reference"):
        reference.decompose()
    for image in doc.find_all("div", class_="baike-render-image"):
        image.decompose()


def fetch_page():
    with get_cursor(autocommit=True) as cursor:
        co = ChromiumOptions()
        co.set_argument("--disable-blink-features=AutomationControlled")
        co.set_user_agent(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        )
        page = ChromiumPage(addr_or_opts=co)
        adapter = TypeAdapter(list[WikiSection])

        cursor.execute("select id, title, url from pedia_core_corpus where source = 'douyin' and raw_sections is null")
        rows = cursor.fetchall()
        try:
            for id, title, url in rows:
                page.get(url)
                pedia = parse_douyin(page_title=title, content=page.html)
                if pedia:
                    cursor.execute(
                        "update pedia_core_corpus set raw_sections = %s where id = %s",
                        (adapter.dump_json(pedia.sections).decode(), id),
                    )
                time.sleep(random.uniform(2, 5))
        finally:
            page.quit()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    fetch_page()
