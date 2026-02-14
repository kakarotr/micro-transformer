import random
import time
from typing import no_type_check

from bs4 import BeautifulSoup
from DrissionPage import ChromiumOptions, ChromiumPage

from corpora.core.wiki.entities import SectionBlock, WikiPage, WikiSection


def start():
    co = ChromiumOptions()
    co.set_user_agent(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    )
    page = ChromiumPage(addr_or_opts=co)
    page.get("http://www.zhanguos.com/zhanguoquwen/")

    soup = BeautifulSoup(page.html, "html.parser")

    pages = []
    page_list = soup.find("ul", class_="pagelist")
    if page_list:
        for li in page_list.find_all("li"):
            if not li.attrs.get("class"):
                a = li.find("a")
                if a:
                    try:
                        int(a.get_text(strip=True))
                        pages.append(f"{page.url[0:-1]}/{a.attrs.get('href')}")
                    except:
                        pass
    pages.insert(0, page.url)
    results: list[WikiPage] = get_page(page=page, pages=pages)
    for result in results:
        with open(f"preview/article/core/zhanguos/{result.title}.md", mode="w", encoding="utf-8") as f:
            f.write(result.merge_sections())


@no_type_check
def get_page(page: ChromiumPage, pages: list[str]):
    results = []
    for url in pages:
        page.get(url)
        soup = BeautifulSoup(page.html, "html.parser")
        ul = soup.find("ul", class_="e2")
        if ul:
            for li in ul.find_all("li"):
                a = li.find("a", class_="title")
                if a:
                    title = a.get_text(strip=True)
                    href = a.attrs.get("href")
                    target_url = f"http://www.zhanguos.com{href}" if href.startswith("/") else href
                    page.get(target_url)
                    result = parse(title=title, content=page.html)
                    results.append(result)
    return results


def parse(title: str, content: str):
    page = WikiPage(title=title, category_name="", lang="zh", sections=[])
    doc = BeautifulSoup(content, "html.parser")
    page.sections.append(WikiSection(title="summary", level=2, blocks=[]))
    for p in doc.find("td").find_all("p"):  # type: ignore
        page.sections[-1].blocks.append(SectionBlock(type="text", content=p.get_text(strip=True)))
    time.sleep(random.uniform(1, 2))
    return page


start()
