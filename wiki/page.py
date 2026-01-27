import os
import re
import statistics
from collections import defaultdict

import bs4
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from utils.db import get_db_conn
from wiki.data import fuzzy_sections, ignore_sections
from wiki.entities import BlockType, SectionBlock, WikiPage, WikiSection


class WikiPageParser:
    def __init__(self) -> None:
        self.max_list_mean_char = 10
        self.conn = get_db_conn()
        self.llm_client = OpenAI(base_url=os.environ["DEEPSEEK_URL"], api_key=os.environ["DEEPSEEK_KEY"])

    def parse(self, title: str, lang: str = "zh"):
        content_doc = self._get_html_doc(title=title, lang=lang)
        if not content_doc:
            return None
        wiki_page = WikiPage(title=title, category_name="", lang=lang, sections=[], full_content="")

        # 处理摘要
        marker = content_doc.select_one("div.mw-heading")
        if marker:
            contents = []
            for p_tag in marker.find_previous_siblings("p"):
                contents.insert(0, p_tag.get_text(strip=True))
                p_tag.decompose()
            wiki_page.sections.append(
                WikiSection(title="summary", level=2, blocks=[SectionBlock(type="text", content="\n\n".join(contents))])
            )

        current_ignore_level = None
        current_level = 0
        current_title = ""

        for child in content_doc.find_all(recursive=False):
            if child.has_attr("class") and "mw-heading" in child["class"]:
                level, title = self._handle_title(title_tag=child, classes=child["class"])  # type: ignore
                if current_ignore_level is not None:
                    if level > current_ignore_level:
                        continue
                    else:
                        current_ignore_level = None

                current_level = level
                current_title = title

                if title in ignore_sections or any(fuzzy in title for fuzzy in fuzzy_sections):
                    current_ignore_level = level
                    continue
                wiki_page.sections.append(WikiSection(title=title, level=level, blocks=[]))
            else:
                if current_ignore_level is not None:
                    if current_level >= current_ignore_level:
                        continue
                # 处理段落
                if child.name == "p":
                    content = child.get_text(strip=True)
                    self._add_block(
                        doc=child, page=wiki_page, current_title=current_title, block_type="text", content=content
                    )

                # 处理表格或者列表
                elif child.name == "table":
                    if child.has_attr("class") and "multicol" in child["class"]:
                        list_title, items = self._handle_list(doc=child)
                        self._add_list_to_block(
                            doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                        )
                    else:
                        pass

                # 处理列表
                elif child.name == "ul":
                    list_title, items = self._handle_list(doc=child)
                    self._add_list_to_block(
                        doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                    )

                # 处理列表
                elif child.name == "div":
                    list = child.find("table", class_="multicol", recursive=False)
                    list_title, items = self._handle_list(list)
                    self._add_list_to_block(
                        doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                    )

                    list = child.find("ul", recursive=False)
                    list_title, items = self._handle_list(list)
                    self._add_list_to_block(
                        doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                    )

                # 处理描述列表
                elif child.name == "dl":
                    if (
                        len(child.find_all(recursive=False)) > 1
                        and child.find_all("dt", recursive=False)
                        and child.find("dd", recursive=False)
                    ):
                        list_titles, items = self._handle_dl(doc=child)
                        for idx, title in enumerate(list_titles):
                            item = items[idx]
                            self._add_list_to_block(
                                doc=child,
                                page=wiki_page,
                                current_title=current_title,
                                list_title=list_title,
                                items=item,
                            )
        return wiki_page

    def _get_html_doc(self, title: str, lang: str):
        response = requests.get(
            url=f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text",
                "disableeditsection": 1,
            },
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
            },
        )
        data = response.json()
        if "parse" in data:
            html_content = data["parse"]["text"]["*"]
            soup = BeautifulSoup(html_content, "html.parser")
            content_doc = soup.find()
            if content_doc:
                self._clean_tag(doc=content_doc)
                return content_doc
        return None

    def _clean_tag(self, doc: bs4.Tag):
        for a_tag in doc.find_all("a"):
            a_tag.unwrap()

        for tag in doc.find_all(["style", "meta", "figure", "sup", "blockquote"]):
            tag.decompose()

        for small_tag in doc.find_all("small"):
            if small_tag.parent and small_tag.parent.name == "p":
                small_tag.parent.decompose()

        for block_tag in doc.find_all("b"):
            block_tag.replace_with(f"**{block_tag.get_text()}**")

        for title_tag in doc.find_all("div", class_="mw-heading"):
            title = title_tag.get_text(strip=True)
            title_tag.clear()
            title_tag.string = title

        for tag in doc.find_all("div", class_=["thumb", "rellink", "hatnote", "side-box"]):
            tag.decompose()

        for tag in doc.find_all(class_="gallery"):
            tag.decompose()

    def _handle_title(self, title_tag: bs4.Tag, classes: list[str]):
        level = int(classes[-1][-1])
        return level, title_tag.string or ""

    def _find_title(self, tag: bs4.Tag):
        for prev_sibling in tag.find_previous_siblings():
            if prev_sibling.has_attr("class") and "mw-heading" in prev_sibling["class"]:
                return prev_sibling.string or ""
        return ""

    def _compute_list_mean_char(self, doc: bs4.Tag, is_table: bool = False):
        total = 0
        count = 0
        for li in doc.find_all("li"):
            count += 1
            text = re.sub(r"（[^）]+）|\([^)]+\)", "", li.get_text(strip=True))
            total += len(text)
        return total // count if count != 0 else 0

    def _handle_list(self, doc: bs4.Tag | None):
        items = []
        list_title = ""
        if doc:
            mean_char = self._compute_list_mean_char(doc=doc, is_table=True)
            prev_tag = doc.find_previous_sibling()
            if prev_tag and prev_tag.name == "dl":
                list_title = prev_tag.get_text(strip=True)

            if mean_char < self.max_list_mean_char:
                items = [li.get_text(strip=True) for li in doc.find_all("li")]
                items.append("llm invoke")
            else:
                items = [li.get_text(strip=True) for li in doc.find_all("li")]
        return list_title, items

    def _add_block(self, doc: bs4.Tag, page: WikiPage, current_title: str, block_type: BlockType, content: str):
        if current_title == self._find_title(tag=doc):
            last_section = page.sections[-1]
            if last_section.blocks and last_section.blocks[-1].type == block_type:
                block_content = last_section.blocks[-1].content
                last_section.blocks[-1].content = f"{block_content}\n\n{content}"
            else:
                last_section.blocks.append(SectionBlock(type=block_type, content=content))
        else:
            page.sections[-1].blocks.append(SectionBlock(type=block_type, content=content))

    def _handle_dl(self, doc: bs4.Tag):
        list_titles: list[str] = []
        values: dict[str, list[str]] = defaultdict(list)
        current_title = ""
        for tag in doc.find_all(["dt", "dd"], recursive=False):
            text = tag.get_text(strip=True)
            if tag.name == "dt":
                list_titles.append(text)
                current_title = text
            if tag.name == "dd":
                values[current_title].append(text)

        items: list[list[str]] = []
        remove_titles = []
        for title in list_titles:
            value = values[title]
            if value:
                mean_char = statistics.mean(list(map(lambda x: len(re.sub(r"（[^）]+）|\([^)]+\)", "", x)), value)))
                if mean_char < self.max_list_mean_char:
                    value.append("llm invoke")
                    items.append(value)
                else:
                    items.append(value)
            else:
                remove_titles.append(title)
        for remove_title in remove_titles:
            list_titles.remove(remove_title)
        return list_titles, items

    def _add_list_to_block(self, doc: bs4.Tag, page: WikiPage, current_title: str, list_title: str, items: list[str]):
        if list_title:
            self._add_block(
                doc=doc,
                page=page,
                current_title=current_title,
                block_type="list-title",
                content=list_title,
            )
        self._add_block(
            doc=doc,
            page=page,
            current_title=current_title,
            block_type="list-item",
            content="\n".join(items),
        )
