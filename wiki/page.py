import os
import re

import requests
import wikitextparser as wtp
from openai import OpenAI

from utils.db import get_db_conn
from wiki.data import fuzzy_sections, ignore_sections, replace_links
from wiki.entities import SectionBlock, WikiPage, WikiSection


class WikiPageParser:
    def __init__(self):
        self.conn = get_db_conn()
        self.llm_client = OpenAI(base_url=os.environ["DEEPSEEK_URL"], api_key=os.environ["DEEPSEEK_KEY"])

    def parse(self, title: str, lang: str = "zh"):
        sections: list[WikiSection] = []
        parsed_content = self._get_raw_content(title=title, lang=lang)
        # 清洗摘要
        summary = self._clean_text(text=parsed_content.sections[0].string)
        sections.append(WikiSection(title="summary", level=0, blocks=[SectionBlock(type="text", content=summary)]))

        current_ignore_level = None

        # 清洗正文
        for section in parsed_content.sections[1:]:
            level = section.level

            if current_ignore_level is not None:
                if level > current_ignore_level:
                    continue
                else:
                    current_ignore_level = None

            if not section.title:
                continue

            section_title = wtp.parse(section.title.strip()).plain_text()
            section_title = re.sub(r"^[\(\[\{\<\"\'「（【《“](.*?)[\)\]\}\>\"\'」）】》”]$", r"\1", section_title)
            section_title = re.sub(r"^[A-Za-z0-9]+\.\s*", "", section_title)
            if section_title in ignore_sections or any(fuzzy in (section_title) for fuzzy in fuzzy_sections):
                current_ignore_level = level
                continue
            if section_title == "一門衆":
                print(section.string)
            wiki_section = WikiSection(title=section_title, level=level, blocks=self._clean_section(section=section))
            sections.append(wiki_section)
        return WikiPage(title=title, category_name="", lang=lang, sections=sections, full_content="")

    def _clean_summary(self, summary: wtp.WikiText):
        self._clean_link(text=summary)
        summary_content = summary.plain_text(replace_templates=self._template_handler, replace_wikilinks=True).strip()
        return self._clean_text(text=summary_content)

    def _get_raw_content(self, title: str, lang: str):
        response = requests.get(
            url=f"https://{lang}.wikipedia.org/w/api.php",
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

    def _clean_link(self, text: wtp.WikiText):
        for link in text.wikilinks[::-1]:
            target = link.target.strip().lower()
            if target.startswith(replace_links):
                link.string = ""

    def _clean_tag(self, text: wtp.WikiText):
        for tag in text.get_tags()[::-1]:
            if tag.name == "ref":
                tag.string = ""
            else:
                tag.string = ""

    def _clean_table(self, table: wtp._table.Table):
        return "table placeholder"

    def _clean_by_pattern(self, text: str):
        new_text = re.sub(r"（\s*[，,；;、]\s*", "（", text)
        new_text = re.sub(r"\s+([（\(])", r"\1", new_text)
        new_text = re.sub(r"（\s*）|\(\s*\)", "", new_text)
        new_text = re.sub(r"※\s*", "", new_text)
        new_text = re.sub(r"（\s*）|\(\s*\)", "", new_text)
        new_text = re.sub(r"-{\s*(.*?)\s*}-", r"\1", new_text)
        new_text = re.sub(r"^=+\s*.*?\s*=+[\r\n]*", "", new_text)
        # new_text = re.sub(r"\s+", "", new_text)
        return new_text

    def _clean_section(self, section: wtp._section.Section):
        content = section.string

        # 查询所有标题的位置, matches[0]: 当前章节的标题, matches[1]: 第一个子章节的标题
        matches = list(re.finditer(r"(?m)^=+\s*.*?\s*=+[\r\n]*", content))
        if len(matches) > 1:
            # 有子章节, 截取当前标题到第一个子标题的内容
            sub_section_start = matches[1].start()
            own_content_with_header = content[:sub_section_start]
        else:
            # 没有子章节
            own_content_with_header = content

        # 去除标题
        content = re.sub(r"(?m)^=+\s*.*?\s*=+[\r\n]*", "", own_content_with_header, count=1).strip()
        # 分割正文与表格
        if content:
            blocks: list[SectionBlock] = []
            parsed_content = wtp.parse(content)
            tables = parsed_content.tables
            last_pos = 0
            for table in tables:
                start, end = table.span
                content_chunk = parsed_content.string[last_pos:start]
                if content_chunk.strip():
                    cleaned_text = self._clean_text(text=content_chunk)
                    if cleaned_text:
                        blocks.append(SectionBlock(type="text", content=cleaned_text))
                blocks.append(SectionBlock(type="table", content=self._clean_table(table=table)))
                last_pos = end
            remaining_text = parsed_content.string[last_pos:]
            if remaining_text.strip():
                cleaned_text = self._clean_text(text=remaining_text)
                if cleaned_text:
                    blocks.append(SectionBlock(type="text", content=cleaned_text))
            return blocks
        else:
            cleaned_content = self._clean_text(text=content)
            return [SectionBlock(type="text", content=cleaned_content if cleaned_content else None)]

    def _clean_text(self, text: str):
        raw_content = wtp.parse(text)
        self._clean_tag(text=raw_content)
        self._clean_link(text=raw_content)
        content = raw_content.plain_text(
            replace_templates=self._template_handler,
            replace_wikilinks=True,
            replace_tags=False,
        )
        content = self._clean_by_pattern(text=content)
        content = self._convert_list(text=content)
        self.get_list_blocks_indices(content)
        content = self._convert_definition_term(text=content)
        content = "\n".join([line.rstrip() for line in content.split("\n")])
        return content.strip().replace("\n\t\n", "\n\n").replace("\n\n\n", "\n\n").replace("{{col|", "")

    def _template_handler(self, template: wtp._template.Template):
        template_name = template.name
        if template_name in ["zy", "link-ja", "col"]:
            print("sss", template.arguments)
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

    def _convert_list(self, text: str):
        counters = {}

        def replace_line(match):
            # group(1) 是行首的空白（如果有）
            # group(2) 是标记符 (如 *, #, ::, *:)
            # group(3) 是正文内容
            markers = match.group(2)
            content = match.group(3).strip()
            content = re.sub(r"\s+[-–—]+\s+", "，", content)
            level = len(markers)
            if level > 2:
                return ""

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
            # elif last_marker == ";":
            #     counters[level] = 0
            #     symbol = ""
            #     content = f"**{content}**"
            #     indent = "  " * (level - 1)
            else:
                counters[level] = 0
                symbol = "- "
                calc_level = level
                if last_marker == ";" and level > 1:
                    calc_level = level - 1
                indent = "  " * (calc_level - 1)

            return f"{indent}{symbol}{content}"

        pattern = r"(?m)^(\s*)([\*\#\:\;]+)\s*(.*)$"

        return re.sub(pattern, replace_line, text)

    def _convert_definition_term(self, text: str):
        """
        处理维基百科的定义列表语法 (;术语)
        转换为 Markdown 的加粗文本 (**术语**)
        """
        return re.sub(r"(?m)^;\s*(.*)$", r"**\1**", text)

    def get_list_blocks_indices(self, text):
        # 你的正则表达式
        list_line_pattern = re.compile(r"(?m)^(\s*)([\*\#\:\;]+)\s*(.*)$")

        # 1. 找到所有匹配的“行”
        # finditer 返回的是迭代器，包含每一次匹配的 start() 和 end()
        matches = list(list_line_pattern.finditer(text))

        if not matches:
            return []

        blocks = []

        # 初始化第一个块
        current_block_start = matches[0].start()
        current_block_end = matches[0].end()

        # 2. 遍历合并相邻的行
        for i in range(1, len(matches)):
            prev_match = matches[i - 1]
            curr_match = matches[i]

            # 获取上一行结束 到 下一行开始 之间的内容
            gap = text[prev_match.end() : curr_match.start()]

            # 判定：如果中间只是换行符（\n 或 \r\n），说明它们是连续的列表
            if gap.strip() == "":
                # 延长当前块的结束位置
                current_block_end = curr_match.end()
            else:
                # 否则，说明中间断开了（有空行或普通文本），保存当前块，开始新块
                blocks.append((current_block_start, current_block_end))
                current_block_start = curr_match.start()
                current_block_end = curr_match.end()

        # 不要忘记加入最后一个块
        blocks.append((current_block_start, current_block_end))

        return blocks
