import sqlite3

import wikipediaapi
import zhconv


def get_db_conn():
    return sqlite3.connect("database.db")


def to_simplified(text: str):
    return zhconv.convert(text, "zh-cn")


def to_traditional(text: str):
    return zhconv.convert(text, "zh-tw")


def get_wiki(lang: str):
    return wikipediaapi.Wikipedia(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        language=lang,
    )
