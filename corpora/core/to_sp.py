import re

import opencc
from dotenv import load_dotenv
from pydantic import TypeAdapter

from corpora.core.wiki.book import BookSection
from corpora.core.wiki.entities import WikiPage, WikiSection
from corpora.utils.db import get_cursor

load_dotenv()

jp_cc = opencc.OpenCC("jp2t")
zh_cc = opencc.OpenCC("t2s")


def convert(text: str, only_zh: bool = False) -> str:
    # PROTECTED_PATTERN = re.compile(r"(「.*?」|\(.*?\)|\（.*?\）)")
    # result = []
    # parts = PROTECTED_PATTERN.split(text)
    # for i, part in enumerate(parts):
    #     if i % 2 == 0:
    #         if part:
    #             if only_zh:
    #                 result.append(zh_cc.convert(part))
    #             else:
    #                 result.append(zh_cc.convert(jp_cc.convert(part)))
    #     else:
    #         result.append(part)
    # return "".join(result)
    if only_zh:
        return zh_cc.convert(text)
    else:
        return zh_cc.convert(jp_cc.convert(text))


def to_sp():

    with get_cursor(autocommit=True) as cursor:
        cursor.execute("select id, sections from pedia_core_corpus where is_to_sp = false")
        rows = cursor.fetchall()
        print(len(rows))

        adapter = TypeAdapter(list[WikiSection])
        for id, sections_dict in rows:
            sections = adapter.validate_python(sections_dict)
            need_update = True
            for section in sections:
                title = section.title
                title = convert(text=title).replace("长筿", "长篠")
                section.title = title

                for block in section.blocks:
                    content = block.content
                    if isinstance(content, str):
                        block.content = convert(text=content).replace("长筿", "长篠")
                    elif isinstance(content, list):
                        list_title = block.list_title
                        if list_title:
                            block.list_title = convert(text=list_title).replace("长筿", "长篠")
                        block.content = [convert(text=item).replace("长筿", "长篠") for item in content]

            if need_update:
                cursor.execute(
                    "update pedia_core_corpus set sections = %s, is_to_sp = true where id = %s",
                    (adapter.dump_json(sections).decode(), id),
                )


def book_to_sp():
    with get_cursor() as cursor:
        cursor.execute(
            "select id, title, content from book_core_corpus where title in ('日本外史', '太平记', '日本战国‧织丰时代史（上）', '日本战国‧织丰时代史（中）', '日本战国‧织丰时代史（下）')"
        )
        rows = cursor.fetchall()

        for id, title, content_dict in rows:
            if title in ("日本外史", "太平记"):
                content = BookSection.model_validate(content_dict)
                content.name = convert(text=content.name).replace("长筿", "长篠")
                for paragraph in content.paragraphs:
                    if paragraph.title:
                        paragraph.title = convert(text=paragraph.title).replace("长筿", "长篠")
                    paragraph.content = convert(text=paragraph.content).replace("长筿", "长篠")
                cursor.execute(
                    "update book_core_corpus set content = %s where id = %s", (content.model_dump_json(), id)
                )
            else:
                book = WikiPage.model_validate(content_dict)
                for section in book.sections:
                    section.title = convert(text=section.title, only_zh=True).replace("长筿", "长篠")
                    for block in section.blocks:
                        content = block.content
                        if isinstance(content, str):
                            block.content = convert(text=content, only_zh=True).replace("长筿", "长篠")
                        elif isinstance(content, list):
                            list_title = block.list_title
                            if list_title:
                                block.list_title = convert(text=list_title, only_zh=True).replace("长筿", "长篠")
                            block.content = [
                                convert(text=item, only_zh=True).replace("长筿", "长篠") for item in content
                            ]
                cursor.execute("update book_core_corpus set content = %s where id = %s", (book.model_dump_json(), id))


# text = "关白,一揆,惣村"
# print(convert(text, only_zh=False))

to_sp()
book_to_sp()
