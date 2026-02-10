import csv
import json
import traceback
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from pydantic import TypeAdapter

from corpora.core.wiki.entities import SectionBlock, WikiSection
from corpora.core.wiki.utils import get_chunks
from corpora.utils.client import get_deepseek_client
from corpora.utils.db import get_cursor, get_db_conn
from corpora.utils.prompt import wiki_rewrite_prompt
from corpora.utils.schemas import WikiListSchema


def save_pedia_info():
    conn = get_db_conn()
    cursor = conn.cursor()

    with open("pedia.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            title = row[2]
            douyin_url = row[3]
            baidu_url = row[4]
            if douyin_url:
                cursor.execute(
                    "insert into pedia_core_corpus (title, title_sp, source, url) values (%s, %s, 'douyin', %s)",
                    (title, title, f"https://www.baike.com{douyin_url}"),
                )
            if baidu_url:
                cursor.execute(
                    "insert into pedia_core_corpus (title, title_sp, source, url) values (%s, %s, 'baidu', %s)",
                    (title, title, baidu_url),
                )

    conn.commit()
    cursor.close()
    conn.close()


def rewrite(n_threads: int):
    chunks = get_chunks(
        sql="select id, title, raw_sections from pedia_core_corpus where source in ('douyin', 'baidu') and sections is null",
        n_threads=n_threads,
    )
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(process_rewrite, chunks)


def process_rewrite(chunk):
    with get_cursor(autocommit=True) as cursor:
        model_name, client = get_deepseek_client()
        for id, title, raw_sections in chunk:
            try:
                sections = TypeAdapter(list[WikiSection]).validate_python(raw_sections)
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
                                                page_title=title,
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
                                        SectionBlock(
                                            type="text", list_title=block.list_title, content=list_item, lang="zh"
                                        )
                                    )
                                if response.usage:
                                    input_tokens = response.usage.prompt_tokens
                                    output_tokens = response.usage.completion_tokens
                                    print(
                                        f"{title} - {section.title} - {block.list_title} input: {input_tokens} output: {output_tokens}"
                                    )
                            else:
                                new_section.blocks.append(block)
                        else:
                            new_section.blocks.append(block)
                    new_sections.append(new_section)
                cursor.execute(
                    "update pedia_core_corpus set sections = %s where id = %s",
                    (TypeAdapter(list[WikiSection]).dump_json(new_sections).decode(), id),
                )
            except:
                traceback.print_exc()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    rewrite(n_threads=50)
