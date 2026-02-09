import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import wikipediaapi
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from opencc import OpenCC
from pydantic import TypeAdapter

from corpora.core_knowledge.wiki.entities import SectionBlock, WikiPage, WikiSection
from corpora.core_knowledge.wiki.main import llm_rewrite, wiki_rewrite_prompt
from corpora.core_knowledge.wiki.page import WikiPageParser
from corpora.core_knowledge.wiki.utils import get_chunks, get_wiki
from utils.db import get_db_conn
from utils.schemas import WikiListSchema

load_dotenv()

ignore_sections = ["家庭", "辭世之句", "參見", "徵引", "遺品", "關聯作品"]
fuzzy_sections = [
    "登場",
    "家族",
    "註釋",
    "註解",
    "資料",
    "參考資料",
    "系譜",
    "文獻",
    "連結",
    "墓",
    "偏諱",
    "銅像",
    "項目",
    "腳註",
    "參考",
    "參看",
    "作品",
    "相關條目",
    "附註",
    "與力",
    "出處",
    "相關",
]


def fetch_page_by_category():
    conn = get_db_conn()
    cursor = conn.cursor()

    inserted_pages = []
    cursor.execute("select id, name from wiki_categories where lang = 'zh'")
    rows = cursor.fetchall()

    for id, category_name in rows:
        wiki = get_wiki(lang="zh")
        category = wiki.page(f"Category:{category_name}")
        for member in category.categorymembers.values():
            if member.namespace == wikipediaapi.Namespace.MAIN:
                if member.title not in inserted_pages:
                    cursor.execute(
                        "insert into pedia_core_corpus (title, source) values (%s, 'zh_wiki')", (member.title,)
                    )
        cursor.execute("update wiki_categories set status = 1 where id = %s", (id,))
    conn.commit()

    cursor.close()
    conn.close()


def fetch_page():
    chunks = get_chunks(
        sql="select id, title from pedia_core_corpus where raw_sections is null and source = 'wiki_zh'", n_threads=10
    )

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process, chunks)


def process(chunk):
    conn = get_db_conn()
    cursor = conn.cursor()

    opencc = OpenCC("tw2sp")
    ignore_sections.extend([opencc.convert(item) for item in ignore_sections])
    fuzzy_sections.extend([opencc.convert(item) for item in fuzzy_sections])
    parser = WikiPageParser(
        ignore_sections=ignore_sections,
        fuzzy_sections=fuzzy_sections,
    )
    adapter = TypeAdapter(list[WikiSection])

    for id, title in chunk:
        try:
            page = parser.parse(page_title=title)
            if page:
                cursor.execute(
                    "update pedia_core_corpus set raw_sections = %s where id = %s",
                    (opencc.convert(adapter.dump_json(page.sections).decode()), id),
                )
                conn.commit()
                print(f"{title}处理完成")
                time.sleep(2)
        except:
            error_stack = traceback.format_exc()
            print(f"{title}错误: err: {error_stack}")

    cursor.close()
    conn.close()


def convert_title():
    opencc = OpenCC("tw2sp")
    conn = get_db_conn()
    cursor = conn.cursor()

    cursor.execute("select id, title from pedia_core_corpus where title_sp is null")
    rows = cursor.fetchall()

    for id, title in rows:
        cursor.execute("update pedia_core_corpus set title_sp = %s where id = %s", (opencc.convert(title), id))

    conn.commit()
    cursor.close()
    conn.close()


def gen_file():
    conn = get_db_conn()
    cursor = conn.cursor()

    cursor.execute("select title_sp, sections from pedia_core_corpus where source = 'wiki_zh' and sections is not null")
    rows = cursor.fetchall()
    adapter = TypeAdapter(list[WikiSection])
    for title, sections in rows:
        content = adapter.validate_python(sections)
        page = WikiPage(title=title, category_name="", lang="zh", sections=content)
        with open(f"preview/wiki_zh/{title}.md", mode="w", encoding="utf-8") as f:
            f.write(page.merge_sections())

    cursor.close()
    conn.close()


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
                "update pedia_core_corpus set sections = %s where id = %s",
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
    chunks = get_chunks(
        sql="select id, title_sp, raw_sections from pedia_core_corpus where sections is null",
        n_threads=10,
    )
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_rewrite, chunks)

