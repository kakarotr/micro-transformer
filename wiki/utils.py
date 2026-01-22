import sqlite3

import zhconv


def get_db_conn():
    return sqlite3.connect("database.db")


def to_simplified(text: str):
    return zhconv.convert(text, "zh-cn")


def to_traditional(text: str):
    return zhconv.convert(text, "zh-tw")
