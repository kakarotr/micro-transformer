import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import TypeAdapter

from corpora.core_knowledge.wiki.utils import get_chunks
from utils.db import get_db_conn
from utils.prompt import wiki_rewrite_prompt
from utils.schemas import WikiListSchema

from .entities import SectionBlock, WikiPage, WikiSection
from .page import WikiPageParser

ignore_sections = [
    "他",
    "首塚",
    "菩提寺",
    "脚注",
    "肖像",
    "研究",
    "系譜",
    "系図",
    "家族",
    "出典",
    "その他",
    "関連書籍",
    "諸将の去就",
    "テレビ番組",
    "人物像",
    "見しかよの物かたり",
    "墓所・霊廟・寺社",
    "関連事項",
    "関連項目",
    "外部リンク",
    "豊臣秀吉を題材とする作品",
    "テレビ番組",
    "史跡等",
    "関連史料",
    "関連作品",
    "展覧会",
    "脚注",
    "祭祀",
    "祭礼",
    "一族縁者",
    "容姿",
    "年忌供養",
    "趣味・趣向",
    "親族",
    "登場作品",
    "子孫",
    "編纂物",
    "備考",
    "関連人物",
    "登場作品",
    "注釈",
    "補註",
    "和歌",
    "小説",
    "屋敷",
    "岩堰用水路",
    "演者",
    "俗説",
    "登場するテレビドラマ",
    "の資料",
    "創作物",
    "書籍",
    "テレビドラマ",
]
fuzzy_sections = [
    "墓所",
    "文献",
    "参考",
    "出典",
    "支城",
    "碑",
    "霊廟",
    "遺品",
    "肖像",
    "銅像",
    "題材とした作品",
    "演じた人物",
    "する作品",
    "した関連作品",
    "いた作品",
    "偏諱",
    "支配した主な城",
    "主題とした作品",
    "関連作品",
    "家紋",
]
special_title = {"宇都宮成綱": ["家臣", "人物・逸話"], "徳川家康": ["人物・逸話"]}


def fetch_page(n_threads: int = 10):
    chunks = get_chunks(sql="select id, title, lang from wiki_pages where raw_sections is null", n_threads=n_threads)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(process_fetch, chunks)


def process_fetch(chunk):
    conn = get_db_conn()
    conn.autocommit = True
    cursor = conn.cursor()
    parser = WikiPageParser(ignore_sections=ignore_sections, fuzzy_sections=fuzzy_sections, special_title=special_title)
    for item in chunk:
        try:
            page = parser.parse(page_title=item[1], lang=item[2])
            if page:
                cursor.execute(
                    "update wiki_pages set raw_sections = %s where id = %s",
                    (TypeAdapter(list[WikiSection]).dump_json(page.sections).decode(), item[0]),
                )
                print(f"{item[1]}处理完成")
            time.sleep(2)
        except:
            error_stack = traceback.format_exc()
            print(f"{item[1]}错误: err: {error_stack}")
    cursor.close()
    conn.close()


def llm_rewrite(n_threads: int = 10):
    chunks = get_chunks(
        sql="select id, title, raw_sections from wiki_pages where sections is null",
        n_threads=n_threads,
    )

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(process_rewrite, chunks)


def process_rewrite(chunk):
    conn = get_db_conn()
    conn.autocommit = True
    cursor = conn.cursor()
    client = OpenAI(base_url=os.environ["LLM_URL"], api_key=os.environ["LLM_KEY"])
    model_name = os.environ["LLM_NAME"]
    for item in chunk:
        try:
            sections = TypeAdapter(list[WikiSection]).validate_python(item[2])
            new_sections: list[WikiSection] = []
            for section in sections:
                new_section = WikiSection(title=section.title, level=section.level, blocks=[])
                for block in section.blocks:
                    if isinstance(block.content, str) and block.content.startswith("<"):
                        soup = BeautifulSoup(block.content, "html.parser")
                        if bool(soup.find()):
                            response = response = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": wiki_rewrite_prompt.format(
                                            json_schema=json.dumps(
                                                WikiListSchema.model_json_schema(), ensure_ascii=False
                                            ),
                                            page_title=item[1],
                                            section_title=section.title,
                                            list_title=block.list_title,
                                        ),
                                    },
                                    {"role": "user", "content": block.content},
                                ],
                                response_format={"type": "json_object"},
                                temperature=1.2,
                            )
                            result = response.choices[0].message.content
                            assert result is not None
                            result = WikiListSchema.model_validate_json(result)
                            for list_item in result.items:
                                new_section.blocks.append(
                                    SectionBlock(type="text", list_title=block.list_title, content=list_item, lang="zh")
                                )
                            if response.usage:
                                input_tokens = response.usage.prompt_tokens
                                output_tokens = response.usage.completion_tokens
                                print(
                                    f"{item[1]} - {section.title} - {block.list_title} input: {input_tokens} output: {output_tokens}"
                                )
                        else:
                            new_section.blocks.append(block)
                    else:
                        new_section.blocks.append(block)
                new_sections.append(new_section)
            cursor.execute(
                "update wiki_pages set sections = %s where id = %s",
                (TypeAdapter(list[WikiSection]).dump_json(new_sections).decode(), item[0]),
            )
            with open(f"preview/{item[1]}.md", mode="w", encoding="utf-8") as f:
                page = WikiPage(title=item[1], category_name="", lang="ja", sections=new_sections)
                f.write(page.merge_sections())
        except:
            traceback.print_exc()

    cursor.close()
    conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    llm_rewrite(n_threads=50)
