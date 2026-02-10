import bs4
from bs4 import BeautifulSoup
from DrissionPage import ChromiumOptions, ChromiumPage

from corpora.core.wiki.entities import WikiPage, WikiSection
from corpora.utils.page import add_block

co = ChromiumOptions()
co.set_user_agent(
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)
page = ChromiumPage(addr_or_opts=co)
page.get("https://zhuanlan.zhihu.com/p/827795244")

soup = BeautifulSoup(page.html, "html.parser")
content_doc = soup.find("span", id="content")


def find_title(tag: bs4.Tag):
    for prev_sibling in tag.find_previous_siblings():
        if prev_sibling.name in ["h2", "h3", "h4"]:
            return prev_sibling.get_text(strip=True)
    return ""


if content_doc:
    article_page = WikiPage(title="阶级社会萌芽-弥生时代", category_name="", lang="zh", sections=[])
    for figure in content_doc.find_all("figure"):
        figure.decompose()
    for a in content_doc.find_all("a"):
        a.unwrap()
    for b in content_doc.find_all("b"):
        b.replace_with(f"**{b.get_text()}**")

    current_title = ""
    for element in content_doc.find("div").find_all():  # type: ignore
        if element.name in ["h2", "h3", "h4"]:
            level = int(element.name[-1])
            title = element.get_text(strip=True)

            article_page.sections.append(WikiSection(title=title, level=level, blocks=[]))
        elif element.name == "p":
            content = element.get_text(strip=True)
            add_block(
                doc=content_doc,
                page=article_page,
                current_title=current_title,
                block_type="text",
                find_title=find_title,
                content=content,
            )

    with open(f"preview/article/background/{article_page.title}.md", mode="w", encoding="utf-8") as f:
        f.write(article_page.merge_sections())

page.quit()
