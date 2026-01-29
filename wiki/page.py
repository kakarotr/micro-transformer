import itertools
import json
import os
import re
import statistics
from collections import defaultdict

import bs4
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from utils.db import get_db_conn
from utils.prompt import wiki_list_prompt, wiki_table_prompt
from utils.schemas import WikiListSchema, WikiTableSchema
from wiki.data import fuzzy_sections, ignore_sections
from wiki.entities import BlockType, SectionBlock, WikiPage, WikiSection


class WikiPageParser:
    def __init__(self) -> None:
        self.max_list_mean_char = 10
        self.conn = get_db_conn()
        self.llm_name = os.environ["LLM_NAME"]
        self.llm_client = OpenAI(base_url=os.environ["LLM_URL"], api_key=os.environ["LLM_KEY"])

    def parse(self, title: str, lang: str = "zh"):
        content_doc = self._get_html_doc(title=title, lang=lang)
        if not content_doc:
            return None
        with open(f"preview/{title}.html", mode="w", encoding="utf-8") as f:
            f.write(content_doc.prettify())
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
                level, title = self._convert_title(title_tag=child, classes=child["class"])  # type: ignore
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

                # 如果下一个元素是标题, 并且level和当前元素一样
                # 说明当前标题没有正文, 同时也不是下一个标题的父标题
                # 则不进行记录
                next_element = child.find_next_sibling()
                if next_element and self._is_title(doc=next_element):
                    next_level, _ = self._convert_title(title_tag=next_element, classes=next_element["class"])  # type: ignore
                    if level != next_level:
                        wiki_page.sections.append(WikiSection(title=title, level=level, blocks=[]))
                else:
                    wiki_page.sections.append(WikiSection(title=title, level=level, blocks=[]))
            else:
                if current_ignore_level is not None:
                    if current_level >= current_ignore_level:
                        continue
                # 处理段落
                if child.name == "p":
                    content = child.get_text(strip=True)
                    if "旧暦" in content and "日付" in content:
                        continue
                    self._add_block(
                        doc=child, page=wiki_page, current_title=current_title, block_type="text", content=content
                    )

                # 处理表格或者列表
                elif child.name == "table":
                    if child.has_attr("class") and "multicol" in child["class"]:
                        list_title, items = self._convert_list(doc=child)
                        self._add_list_to_block(
                            doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                        )
                        continue
                    if len(child.find_all("tr")) == 1:
                        # 只有一个tr按列表处理
                        list_title, items = self._convert_list(doc=child)
                        self._add_list_to_block(
                            doc=child,
                            page=wiki_page,
                            current_title=current_title,
                            list_title=list_title,
                            items=items,
                        )
                        continue
                    if child.has_attr("class") and "wikitable" in child["class"]:
                        tbody = [tr.get_text(separator="||", strip=True) for tr in child.find_all("tr")]
                        # 数据表格
                        self._add_block(
                            doc=child,
                            page=wiki_page,
                            current_title=current_title,
                            block_type="table",
                            content=tbody,
                        )

                # 处理列表
                elif child.name == "ul":
                    list_title, items = self._convert_list(doc=child)
                    self._add_list_to_block(
                        doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                    )

                # 处理有序列表
                elif child.name == "ol":
                    list_title, items = self._convert_list(doc=child)
                    self._add_list_to_block(
                        doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                    )

                # 处理列表
                elif child.name == "div":
                    list = child.find("table", class_="multicol", recursive=False)
                    list_title, items = self._convert_list(list)
                    self._add_list_to_block(
                        doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                    )

                    list = child.find("ul", recursive=False)
                    list_title, items = self._convert_list(list)
                    self._add_list_to_block(
                        doc=child, page=wiki_page, current_title=current_title, list_title=list_title, items=items
                    )

                # 处理描述列表
                elif child.name == "dl":
                    if len(child.find_all(recursive=False)) > 1:
                        if child.find_all("dt", recursive=False) and child.find_all("dd", recursive=False):
                            list_titles, items = self._convert_standard_dl(doc=child)
                            for idx, list_title in enumerate(list_titles):
                                item = items[idx]
                                self._add_list_to_block(
                                    doc=child,
                                    page=wiki_page,
                                    current_title=current_title,
                                    list_title=list_title,
                                    items=item,
                                )
                            continue
                        if len(child.find_all("dd")) > 1:
                            list_title, items = self._convert_two_dd_list(doc=child)
                            self._add_list_to_block(
                                doc=child,
                                page=wiki_page,
                                current_title=current_title,
                                list_title=list_title,
                                items=items,
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
                self._clean_tag(root=soup, doc=content_doc)
                self._special_process(root=soup, doc=content_doc, title=title)
                return content_doc
        return None

    def _clean_tag(self, root: bs4.BeautifulSoup, doc: bs4.Tag):
        """清洗标签"""
        # 去掉a标签的格式, 保留内容
        for a_tag in doc.find_all("a"):
            a_tag.unwrap()

        # 直接删除的标签
        for tag in doc.find_all(["link", "style", "meta", "figure", "sup", "blockquote"]):
            tag.decompose()

        # 删除small, 特殊处理<p><small></small></p>
        for small_tag in doc.find_all("small"):
            if small_tag.parent and small_tag.parent.name == "p":
                small_tag.parent.decompose()

        # 修改加粗标签为Markdown加粗格式
        for block_tag in doc.find_all("b"):
            block_tag.replace_with(f"**{block_tag.get_text()}**")

        # 去掉标题元素内多余的元素, 只保留标题
        for title_tag in doc.find_all("div", class_="mw-heading"):
            title = title_tag.get_text(strip=True)
            title_tag.clear()
            title_tag.string = title

        # 删除指定class的div元素
        for tag in doc.find_all("div", class_=["thumb", "rellink", "hatnote", "side-box", "NavFrame"]):
            tag.decompose()

        # 删除图片画廊
        for tag in doc.find_all(class_="gallery"):
            tag.decompose()

        # 删除没有class但是有style的的div
        for div in doc.find_all("div", recursive=False):
            if not div.has_attr("class") and div.has_attr("style") and len(div.find_all("ul", recursive=False)) == 0:
                div.decompose()

        # 删除无效的表格
        for table in doc.find_all("table", class_=re.compile(r"^box-")):
            table.decompose()

        # 合并连续的dl
        children = [child for child in doc.find_all(recursive=False)]
        for key, group in itertools.groupby(children, key=lambda x: x.name):
            if key == "dl":
                consecutive_dls = list(group)
                if len(consecutive_dls) > 3:
                    merged_dl = root.new_tag("dl", attrs={"class": "merged-dl"})
                    for dl in consecutive_dls:
                        if dl.find_all("ul"):
                            # 标题项
                            dds = dl.find_all("dd", recursive=False)
                            if dds:
                                # 标题
                                new_dt = root.new_tag("dt", string=dds[0].get_text(strip=True))
                                merged_dl.append(new_dt)
                            if len(dds) > 1:
                                # 内容
                                for dd in dds[1:]:
                                    for ul in dd.find_all("ul"):
                                        ul.decompose()
                                    new_dd = root.new_tag("dd", string=dd.get_text(strip=True))
                                    merged_dl.append(new_dd)
                            continue
                        if dl.find_all("ol"):
                            ol = dl.select_one("ol")
                            if ol:
                                for li in ol.find_all("li"):
                                    new_dd = root.new_tag("dd", string=li.get_text(strip=True))
                                    merged_dl.append(new_dd)
                                continue
                        for dd in dl.find_all("dd"):
                            # 列表项
                            for ul in dd.find_all("ul"):
                                ul.decompose()
                            new_dd = root.new_tag("dd", string=dd.get_text(strip=True))
                            merged_dl.append(new_dd)
                    consecutive_dls[0].insert_before(merged_dl)

                    for dl in consecutive_dls:
                        dl.decompose()

        # 处理只有一个子元素的dl
        for dl in doc.find_all("dl", recursive=False):
            if len(dl.find_all(recursive=False)) == 1:
                dl_child_tag = dl.find().name  # type: ignore
                if dl_child_tag == "dd":
                    # 部分页面段落在<dl><dd></dd></dl>内
                    text = dl.get_text(strip=True)
                    dl.name = "p"
                    dl.clear()
                    dl.string = text

    def _convert_title(self, title_tag: bs4.Tag, classes: list[str]):
        """
        处理标题

        返回标题等级和标题内容
        """
        level = int(classes[-1][-1])
        title = title_tag.get_text(strip=True)
        title = re.sub(r"^[\(\（]?(?:[0-9]+|[IVXLCDMivxlcdm]+)[\)\）\.、\s\-]*", "", title)
        return level, title

    def _find_title(self, tag: bs4.Tag):
        """
        查询当前元素的标题

        以当前元素为锚点, 向上查询有指定class的元素
        """
        for prev_sibling in tag.find_previous_siblings():
            if prev_sibling.has_attr("class") and "mw-heading" in prev_sibling["class"]:
                return prev_sibling.string or ""
        return ""

    def _compute_list_mean_char(self, texts: list[str], is_table: bool = False):
        """
        计算ListItem字符的平均长度

        超过了指定长度就是用LLM对列表进行语义化重写
        """
        total = 0
        count = 0
        for text in texts:
            count += 1
            # 去除"()"的内容, 不计入字符数量
            text = re.sub(r"（[^）]+）|\([^)]+\)", "", text)
            # 对内容进行分割, 避免特殊的分隔符影响长度计算
            total += statistics.mean(list(map(lambda x: len(x), re.split(r"[、,-]", text))))
        return total // count if count != 0 else 0

    def _convert_list(self, doc: bs4.Tag | None):
        """
        转换List

        大部分ul元素没有list title, 可能部分Section有使用dl作为title
        """
        items = []
        list_title = ""
        if doc:
            texts = [li.get_text(strip=True) for li in doc.find_all("li")]
            mean_char = self._compute_list_mean_char(texts=texts, is_table=True)
            prev_tag = doc.find_previous_sibling()
            # 提取可能的title
            if prev_tag:
                if prev_tag.name == "dl":
                    list_title = prev_tag.get_text(strip=True)
            elif doc.parent:
                prev = doc.parent.find_previous_sibling()
                if prev and prev.name == "dl":
                    list_title = prev.get_text(strip=True)

            if mean_char < self.max_list_mean_char:
                # 小于指定长度使用LLM改写
                items = [li.get_text(strip=True) for li in doc.find_all("li")]
                items.append("llm invoke")
            else:
                # 大于指定长度直接作为训练语料
                items = [li.get_text(strip=True) for li in doc.find_all("li")]
        return list_title, items

    def _add_block(
        self,
        doc: bs4.Tag,
        page: WikiPage,
        current_title: str,
        block_type: BlockType,
        content: str | list,
        list_title: str | None = None,
    ):
        """
        添加Block
        """
        if current_title == self._find_title(tag=doc):
            # 当前添加的Block归属的条目已经添加过
            # 判断当前Block和最后一个Block的类型是否一致
            # 一致则在上一个Block的内容后面最后进行追加当前Block的内容
            last_section = page.sections[-1]
            if last_section.blocks and last_section.blocks[-1].type == block_type:
                block_content: str | list[str] = last_section.blocks[-1].content
                if block_content:
                    if isinstance(content, str):
                        last_section.blocks[-1].content = f"{block_content}\n\n{content}"
                    else:
                        last_section.blocks.append(
                            SectionBlock(type=block_type, content=content, list_title=list_title)
                        )
                else:
                    last_section.blocks[-1] = SectionBlock(type=block_type, content=content, list_title=list_title)
            else:
                last_section.blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))
        else:
            page.sections[-1].blocks.append(SectionBlock(type=block_type, content=content, list_title=list_title))

    def _convert_standard_dl(self, doc: bs4.Tag):
        """
        处理dl元素

        仅处理下面的结构, 其中dt作为title
        <dl>
          <dt></dt>
          <dd></dd>
          <dt></dt>
          <dd></dd>
        </dl>
        """
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
                mean_char = self._compute_list_mean_char(texts=value)
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

    def _convert_two_dd_list(self, doc: bs4.Tag):
        """
        处理多dd的列表, 第一个dd是按标题处理

        结构:

        """
        title = ""
        items = []
        for idx, dd in enumerate(doc.find_all("dd", recursive=False)):
            text = dd.get_text(strip=True)
            if idx == 0:
                title = text
                continue
            items.append(text)
        mean_char = self._compute_list_mean_char(texts=items)
        if mean_char < self.max_list_mean_char:
            items.append("llm invoke")
        return title, items

    def _add_list_to_block(
        self,
        doc: bs4.Tag,
        page: WikiPage,
        current_title: str,
        list_title: str,
        items: list[str],
        is_order: bool = False,
    ):
        self._add_block(
            doc=doc,
            page=page,
            current_title=current_title,
            block_type="olist" if is_order else "ulist",
            content=items,
            list_title=list_title,
        )

    def _is_title(self, doc: bs4.Tag):
        return doc.has_attr("class") and "mw-heading" in doc["class"]

    def _special_process(self, root: bs4.BeautifulSoup, doc: bs4.Tag, title: str):
        """针对指定页面的特殊处理"""
        if title == "本能寺の変":
            for dd in doc.find_all("dd"):
                if dd.get_text(strip=True) == "光秀・秀吉・家康の三者が共謀して信長を暗殺したという説の総称。":
                    dd.insert_after(root.new_tag("dt", string="土岐明智家滅亡阻止説"))
            for div in doc.find_all("div"):
                if div.get_text(strip=True) == "討死、自害した人物":
                    new_ul = root.new_tag("ul")
                    nexts = div.find_next_siblings(limit=5)
                    for item in nexts[1:]:
                        for li in item.find_all("li"):
                            new_ul.append(li)
                        item.decompose()
                    nexts[0].insert_after(new_ul)
            for caption in doc.find_all("caption"):
                if caption.get_text(strip=True).startswith("本能寺の変の真相をめぐる諸説"):
                    if caption.parent and caption.parent.name == "table":
                        next = caption.parent.find_next_sibling()
                        next.decompose()  # type: ignore
                        caption.parent.decompose()

        if title == "方広寺鐘銘事件":
            for p in doc.find_all("p", recursive=False):
                if (
                    p.get_text(strip=True)
                    == "通説では、崇伝が鐘銘事件へ深く関与したとされるが、当時の一次史料からはそれは窺えない。ただし、崇伝も取り調べには加わっており、東福寺住持は清韓の救援を崇伝へ依頼したが断られている。"
                ):
                    p.decompose()

        if title == "織田信長":
            for ul in doc.find_all("ul", recursive=False):
                lis = ul.find_all("li")
                if len(lis) == 2:
                    text = lis[1].get_text(strip=True)
                    if text == "側室":
                        dl = root.new_tag("dl", string=text)
                        ul.insert_after(dl)
                        lis[1].decompose()

    def _rewrite_list(
        self,
        page_title: str,
        section_title: str,
        content: str,
        list_title: str | None = None,
    ):
        response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {
                    "role": "system",
                    "content": wiki_list_prompt.format(
                        json_schema=json.dumps(WikiListSchema.model_json_schema(), ensure_ascii=False),
                        page_title=page_title,
                        section_title=section_title,
                        list_title=list_title,
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        assert result is not None
        return WikiListSchema.model_validate_json(result).items

    def _rewrite_table(
        self,
        page_title: str,
        section_title: str,
        content: str,
    ):
        response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {
                    "role": "system",
                    "content": wiki_table_prompt.format(
                        json_schema=json.dumps(WikiTableSchema.model_json_schema(), ensure_ascii=False),
                        page_title=page_title,
                        section_title=section_title,
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        assert result is not None
        return WikiTableSchema.model_validate_json(result).lines
