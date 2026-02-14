import random
import time
from typing import no_type_check

import bs4
from bs4 import BeautifulSoup
from DrissionPage import ChromiumOptions, ChromiumPage

from corpora.core.wiki.entities import SectionBlock, WikiPage, WikiSection
from corpora.utils.page import add_block

co = ChromiumOptions()
co.set_user_agent(
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)
page = ChromiumPage(addr_or_opts=co)


@no_type_check
def parse_1(url: str):
    def find_title(tag: bs4.Tag):
        for element in tag.find_previous_siblings():
            if element.name == "h1":
                return element.get_text(strip=True)
        return ""

    page.get(url)
    soup = BeautifulSoup(page.html, "html.parser")

    if soup.find("h1", class_="title"):
        title = soup.find("h1", class_="title").get_text(strip=True)
    else:
        title = soup.find("span", class_="opus-module-title__text").get_text(strip=True)
    article = WikiPage(title=title, category_name="", lang="", sections=[])

    content_doc = soup.find(id="read-article-holder")
    if not content_doc:
        content_doc = soup.find("div", class_="opus-module-content")

    # 摘要
    first_child = content_doc.find()
    if first_child.name == "p":
        marker = content_doc.find("h1")
        if marker:
            for item in marker.find_previous_siblings()[::-1]:
                if item.name == "p":
                    block = SectionBlock(type="text", content=item.get_text(strip=True))
                    if len(article.sections) > 1:
                        article.sections[-1].blocks.append(block)
                    else:
                        article.sections.append(WikiSection(title="summary", level=2, blocks=[block]))
                    item.decompose()

    # 正文
    current_title = ""
    if len(article.sections) == 0:
        article.sections.append(WikiSection(title="summary", level=2, blocks=[]))
    for element in content_doc.find_all(recursive=False):
        if element.name == "h1":
            title = element.get_text(strip=True)
            current_title = title

            article.sections.append(WikiSection(title=title, level=2, blocks=[]))
        elif element.name == "p":
            content = element.get_text(strip=True)
            add_block(
                doc=content_doc,
                page=article,
                current_title=current_title,
                block_type="text",
                find_title=find_title,
                content=content,
            )

    return article


@no_type_check
def parse_2(url: str):
    def find_title(tag: bs4.Tag):
        for element in tag.find_previous_siblings():
            prev = element.find_previous_sibling()
            next = element.find_next_sibling()
            if next and next.get_text(strip=True) == "" and prev and prev and prev.get_text(strip=True) == "":
                return element.get_text(strip=True)
        return ""

    page.get(url)
    soup = BeautifulSoup(page.html, "html.parser")

    if soup.find("h1", class_="title"):
        title = soup.find("h1", class_="title").get_text(strip=True)
    else:
        title = soup.find("span", class_="opus-module-title__text").get_text(strip=True)
    article = WikiPage(title=title, category_name="", lang="", sections=[])

    content_doc = soup.find(id="read-article-holder")
    if not content_doc:
        content_doc = soup.find("div", class_="opus-module-content")
    current_title = ""
    for idx, element in enumerate(content_doc.find_all("p", recursive=False)):
        if idx == 0:
            title = element.get_text(strip=True)
            article.sections.append(WikiSection(title=title, level=2, blocks=[]))
        else:
            prev = element.find_previous_sibling()
            next = element.find_next_sibling()
            if next and next.get_text(strip=True) == "" and prev and prev and prev.get_text(strip=True) == "":
                title = element.get_text(strip=True)
                article.sections.append(WikiSection(title=title, level=2, blocks=[]))
            else:
                content = element.get_text(strip=True)
                add_block(
                    doc=content_doc,
                    page=article,
                    current_title=current_title,
                    block_type="text",
                    find_title=find_title,
                    content=content,
                )

    return article


def get_url_1():
    page.get("https://www.bilibili.com/read/readlist/rl663102")
    soup = BeautifulSoup(page.html, "html.parser")
    urls = []
    for element in soup.find_all("div", class_="list-content-item"):
        urls.append(f'"https://www.bilibili.com/read/cv{element.attrs.get("data-id")}"')
    print("[" + ",".join(urls) + "]")


urls = [
    "https://www.bilibili.com/read/cv21228404",
    "https://www.bilibili.com/read/cv21228463",
    "https://www.bilibili.com/read/cv21228510",
    "https://www.bilibili.com/read/cv21228578",
    "https://www.bilibili.com/read/cv21228622",
    "https://www.bilibili.com/read/cv21242049",
    "https://www.bilibili.com/read/cv21242147",
    "https://www.bilibili.com/read/cv21242270",
    "https://www.bilibili.com/read/cv21242524",
    "https://www.bilibili.com/read/cv21242681",
    "https://www.bilibili.com/read/cv21288106",
    "https://www.bilibili.com/read/cv21288167",
    "https://www.bilibili.com/read/cv21288210",
    "https://www.bilibili.com/read/cv21288257",
    "https://www.bilibili.com/read/cv21314442",
    "https://www.bilibili.com/read/cv21314509",
    "https://www.bilibili.com/read/cv21314543",
    "https://www.bilibili.com/read/cv21314593",
    "https://www.bilibili.com/read/cv21342530",
    "https://www.bilibili.com/read/cv21342555",
    "https://www.bilibili.com/read/cv21342638",
    "https://www.bilibili.com/read/cv21342680",
    "https://www.bilibili.com/read/cv21342705",
    "https://www.bilibili.com/read/cv21668925",
    "https://www.bilibili.com/read/cv23348669",
]
for idx, url in enumerate(urls):
    article = parse_1(url=url)
    with open(f"preview/article/core/bilibili/{article.title}.md", mode="w", encoding="utf-8") as f:
        f.write(article.merge_sections())
    time.sleep(random.uniform(2, 5))

# get_url_1()
