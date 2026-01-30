import json
import math
import os
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, thread

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import TypeAdapter
from sympy import content

from utils.db import get_db_conn
from utils.prompt import wiki_list_prompt, wiki_table_prompt
from utils.schemas import WikiListSchema, WikiTableSchema
from wiki.entities import SectionBlock, WikiSection
from wiki.page import WikiPageParser


def get_chunks(sql: str, n_threads: int):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    total = len(rows)
    chunk_size = math.ceil(total / n_threads)

    chunks = [rows[i : i + chunk_size] for i in range(0, total, chunk_size)]

    return chunks


def fetch_page(n_threads: int = 10):
    chunks = get_chunks(sql="select id, title, lang from wiki_pages where raw_sections is null", n_threads=n_threads)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(process_fetch, chunks)


def process_fetch(chunk):
    conn = get_db_conn()
    conn.autocommit = True
    cursor = conn.cursor()
    parser = WikiPageParser()
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
        sql="select id, title, raw_sections from wiki_pages where sections is null and title = '豊臣秀吉' limit 1",
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
                    if block.type == "table":
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {
                                    "role": "system",
                                    "content": wiki_table_prompt.format(
                                        json_schema=json.dumps(WikiTableSchema.model_json_schema(), ensure_ascii=False),
                                        page_title=item[1],
                                        section_title=section.title,
                                    ),
                                },
                                {"role": "user", "content": block.content},
                            ],
                            response_format={"type": "json_object"},
                        )
                        result = response.choices[0].message.content
                        assert result is not None
                        result = WikiTableSchema.model_validate_json(result)
                        btype = "olist" if result.is_ordered else "ulist"
                        if result.mode == "Flat" and isinstance(result.data, list):
                            new_section.blocks.append(
                                SectionBlock(type=btype, list_title="", content=result.data, lang="zh")
                            )
                        elif result.mode == "Grouped" and isinstance(result.data, dict):
                            for title, items in result.data.items():
                                new_section.blocks.append(
                                    SectionBlock(type=btype, list_title=title, content=items, lang="zh")
                                )

                    elif (block.type == "olist" or block.type == "ulist") and isinstance(block.content, str):
                        if block.content.startswith("<"):
                            soup = BeautifulSoup(block.content, "html.parser")
                            if bool(soup.find()):
                                response = client.chat.completions.create(
                                    model=model_name,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": wiki_list_prompt.format(
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
                                )
                                result = response.choices[0].message.content
                                assert result is not None
                                result = WikiListSchema.model_validate_json(result)
                                new_section.blocks.append(
                                    SectionBlock(
                                        type=block.type, list_title=block.list_title, content=result.items, lang="zh"
                                    )
                                )
                    else:
                        new_section.blocks.append(block)
                new_sections.append(new_section)
            cursor.execute(
                "update wiki_pages set sections = %s where id = %s",
                (TypeAdapter(list[WikiSection]).dump_json(new_sections).decode(), item[0]),
            )
            print(TypeAdapter(list[WikiSection]).dump_json(new_sections).decode())

        except:
            traceback.print_exc()

    cursor.close()
    conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    llm_rewrite(n_threads=1)
